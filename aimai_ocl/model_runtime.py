"""OpenAI-only model runtime helpers.

This module intentionally keeps model configuration minimal:

- one model id from ``RunConfig.model``
- one credential source from ``OPENAI_API_KEY``
- one fixed model client class from AgenticPay (`OpenAILLM`)

No provider routing or dynamic class loading is used.

中文翻译：
本模块只保留 OpenAI 路径，目的是降低运行时复杂度。模型来源、
鉴权方式、以及客户端类型都固定，避免 provider 分流带来的配置漂移。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from typing import Any

from aimai_ocl.experiment_config import RunConfig

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"


@dataclass(frozen=True, slots=True)
class ModelRuntimeConfig:
    """Resolved OpenAI runtime config for one run.

    Input fields:
        model:
            OpenAI model id used by AgenticPay `OpenAILLM`.
        api_key_env:
            Environment variable name used to resolve API key.

    Output:
        Immutable runtime config for ``build_model_client``.

    中文翻译：
    这是单次运行的最小模型配置对象，只承载模型名和 API Key 的环境变量名。
    """

    model: str
    api_key_env: str = OPENAI_API_KEY_ENV


def _sanitize_text_payload(value: Any) -> str:
    """Convert arbitrary text-like input into a JSON-safe UTF-8 string.

    中文翻译：
    把任意文本输入清洗成可安全写入 JSON 请求体的 UTF-8 字符串，去掉空字节、
    非法代理项和不必要的控制字符。
    """
    text = value if isinstance(value, str) else str(value)
    text = text.replace("\x00", "")
    text = "".join(
        character
        if character in "\n\r\t" or ord(character) >= 0x20
        else " "
        for character in text
    )
    return text.encode("utf-8", errors="replace").decode("utf-8")


def _sanitize_json_value(value: Any) -> Any:
    """Recursively sanitize JSON-bound request values.

    中文翻译：
    递归清洗请求字段；字符串做 UTF-8 安全化，非有限浮点数直接丢弃。
    """
    if value is None:
        return None
    if isinstance(value, str):
        return _sanitize_text_payload(value)
    if isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(value)
    if isinstance(value, list | tuple):
        sanitized_items = []
        for item in value:
            sanitized_item = _sanitize_json_value(item)
            if sanitized_item is not None:
                sanitized_items.append(sanitized_item)
        return sanitized_items
    if isinstance(value, dict):
        sanitized_dict: dict[str, Any] = {}
        for key, item in value.items():
            sanitized_item = _sanitize_json_value(item)
            if sanitized_item is not None:
                sanitized_dict[str(key)] = sanitized_item
        return sanitized_dict
    return _sanitize_text_payload(value)


def _is_json_safe(value: Any) -> bool:
    """Return whether a value can be serialized as strict JSON.

    中文翻译：
    检查值是否可被严格 JSON 序列化，避免把 NaN/对象之类直接送进请求体。
    """
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


def _prefers_max_completion_tokens(model: str) -> bool:
    """Return whether a model expects ``max_completion_tokens``.

    中文翻译：判断模型是否更适合使用新版 max_completion_tokens 参数。
    """
    normalized = model.lower()
    return normalized.startswith(("gpt-5", "o1", "o3", "o4"))


class OpenAIChatLLM:
    """Minimal OpenAI chat-completions adapter used by this project.

    中文翻译：
    项目内部最小 OpenAI 文本模型封装。只保留 chat-completions 路径，并在发送
    前对请求体做 JSON 安全清洗。
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
    ) -> None:
        try:
            import openai  # noqa: PLC0415
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
            raise _missing_dependency_error(exc) from exc
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def _build_request(
        self,
        *,
        prompt: str,
        temperature: float | None,
        max_tokens: int | None,
        kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        """Build a sanitized OpenAI request payload.

        中文翻译：
        生成清洗后的请求体，并返回被丢弃的字段名用于调试。
        """
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": _sanitize_text_payload(prompt),
                },
            ],
        }
        dropped_fields: list[str] = []

        if temperature is not None:
            temperature_value = float(temperature)
            if math.isfinite(temperature_value):
                request["temperature"] = temperature_value
            else:
                dropped_fields.append("temperature")
        if max_tokens is not None:
            token_key = (
                "max_completion_tokens"
                if _prefers_max_completion_tokens(self.model)
                else "max_tokens"
            )
            request[token_key] = int(max_tokens)

        for key, value in kwargs.items():
            sanitized_value = _sanitize_json_value(value)
            if sanitized_value is None:
                dropped_fields.append(key)
                continue
            if not _is_json_safe(sanitized_value):
                dropped_fields.append(key)
                continue
            request[key] = sanitized_value
        return request, dropped_fields

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate text via OpenAI with a sanitized request payload.

        中文翻译：
        走 OpenAI chat-completions，并在请求体疑似损坏时自动做一次最小重试。
        """
        request, dropped_fields = self._build_request(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            kwargs=kwargs,
        )
        prompt_chars = len(request["messages"][0]["content"])
        try:
            response = self.client.chat.completions.create(**request)
        except Exception as exc:
            error_text = str(exc)
            if (
                "Unsupported parameter" in error_text
                and "'max_tokens'" in error_text
                and "max_tokens" in request
            ):
                retry_request = dict(request)
                retry_request["max_completion_tokens"] = retry_request.pop("max_tokens")
                try:
                    response = self.client.chat.completions.create(**retry_request)
                except Exception as retry_exc:
                    raise RuntimeError(
                        "OpenAI API error: "
                        f"{retry_exc} [prompt_chars={prompt_chars}, "
                        f"dropped_fields={dropped_fields}]"
                    ) from retry_exc
            elif (
                "Unsupported parameter" in error_text
                and "'max_completion_tokens'" in error_text
                and "max_completion_tokens" in request
            ):
                retry_request = dict(request)
                retry_request["max_tokens"] = retry_request.pop("max_completion_tokens")
                try:
                    response = self.client.chat.completions.create(**retry_request)
                except Exception as retry_exc:
                    raise RuntimeError(
                        "OpenAI API error: "
                        f"{retry_exc} [prompt_chars={prompt_chars}, "
                        f"dropped_fields={dropped_fields}]"
                    ) from retry_exc
            elif (
                (
                    "Unsupported parameter" in error_text
                    or "Unsupported value" in error_text
                )
                and "temperature" in error_text
                and "temperature" in request
            ):
                retry_request = dict(request)
                retry_request.pop("temperature", None)
                try:
                    response = self.client.chat.completions.create(**retry_request)
                except Exception as retry_exc:
                    raise RuntimeError(
                        "OpenAI API error: "
                        f"{retry_exc} [prompt_chars={prompt_chars}, "
                        f"dropped_fields={dropped_fields + ['temperature']}]"
                    ) from retry_exc
            elif "parse the JSON body" in error_text:
                fallback_request = {
                    "model": self.model,
                    "messages": request["messages"],
                }
                if "max_tokens" in request:
                    fallback_request["max_tokens"] = request["max_tokens"]
                if "max_completion_tokens" in request:
                    fallback_request["max_completion_tokens"] = request[
                        "max_completion_tokens"
                    ]
                try:
                    response = self.client.chat.completions.create(**fallback_request)
                except Exception as retry_exc:
                    raise RuntimeError(
                        "OpenAI API error: "
                        f"{retry_exc} [prompt_chars={prompt_chars}, "
                        f"dropped_fields={dropped_fields}]"
                    ) from retry_exc
            else:
                raise RuntimeError(
                    "OpenAI API error: "
                    f"{exc} [prompt_chars={prompt_chars}, "
                    f"dropped_fields={dropped_fields}]"
                ) from exc
        content = response.choices[0].message.content
        return (content or "").strip()

    def __repr__(self) -> str:
        """Return string representation of the runtime model."""
        return f"OpenAIChatLLM(model={self.model})"


def _missing_dependency_error(exc: ModuleNotFoundError) -> RuntimeError:
    """Build a precise runtime dependency error from an import failure.

    中文翻译：
    把底层 ModuleNotFoundError 转成更明确的安装提示，避免所有情况都误报为
    “缺少 AgenticPay”。
    """
    missing_name = exc.name or str(exc)
    if missing_name == "agenticpay":
        install_hint = (
            'pip install "git+https://github.com/SafeRL-Lab/AgenticPay.git" '
            "&& pip install -e ."
        )
    else:
        install_hint = f"pip install {missing_name}"
    return RuntimeError(
        "missing dependency: "
        f"{exc}. Install with: {install_hint}"
    )


def resolve_model_runtime_config(run_config: RunConfig) -> ModelRuntimeConfig:
    """Resolve runtime config from run-level settings.

    Input:
        run_config:
            Shared run configuration.

    Output:
        ``ModelRuntimeConfig`` bound to one model id and OPENAI key env.

    中文翻译：
    从运行配置里提取模型名，并绑定统一的 OPENAI_API_KEY 读取规则。
    """
    return ModelRuntimeConfig(model=run_config.model, api_key_env=OPENAI_API_KEY_ENV)


def build_model_client(runtime_config: ModelRuntimeConfig) -> Any:
    """Instantiate the project's OpenAI text client for current run.

    Input:
        runtime_config:
            Resolved runtime config containing model id and key env name.

    Output:
        One instantiated OpenAI text client.

    Raises:
        RuntimeError:
            If OpenAI client dependency is unavailable or API key is missing.

    中文翻译：
    按固定构造参数创建项目内的 OpenAI 文本客户端；如果依赖缺失或密钥未配置，
    立即失败并给出明确安装提示。
    """
    api_key = os.getenv(runtime_config.api_key_env)
    if not api_key:
        raise RuntimeError(f"{runtime_config.api_key_env} is not set.")
    return OpenAIChatLLM(model=runtime_config.model, api_key=api_key)


def build_agenticpay_agents(
    *,
    run_config: RunConfig,
) -> tuple[Any, Any, ModelRuntimeConfig]:
    """Build AgenticPay buyer/seller agents from OpenAI runtime config.

    Input:
        run_config:
            Full run config with model id and pricing bounds.

    Output:
        Tuple ``(buyer_agent, seller_agent, runtime_config)``.

    Raises:
        RuntimeError:
            If AgenticPay modules are unavailable or model client creation fails.

    中文翻译：
    基于同一模型配置构造 buyer/seller 两个 AgenticPay agent。
    该路径故意不再暴露 seller 实现分支，以避免实验语义混淆。
    """
    try:
        from agenticpay.agents.buyer_agent import BuyerAgent  # noqa: PLC0415
        from agenticpay.agents.seller_agent import SellerAgent  # noqa: PLC0415
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency guard
        raise _missing_dependency_error(exc) from exc

    runtime_config = resolve_model_runtime_config(run_config)
    buyer_model = build_model_client(runtime_config)
    seller_model = build_model_client(runtime_config)
    buyer = BuyerAgent(model=buyer_model, buyer_max_price=run_config.buyer_max_price)
    # Single-seller path is the only supported runtime contract.
    # Multi-agent collaboration is represented by OCL control composition,
    # not by swapping the environment's seller class.
    seller = SellerAgent(
        model=seller_model,
        seller_min_price=run_config.seller_min_price,
    )
    return buyer, seller, runtime_config
