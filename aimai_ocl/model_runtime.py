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
    """Instantiate AgenticPay ``OpenAILLM`` for current run.

    Input:
        runtime_config:
            Resolved runtime config containing model id and key env name.

    Output:
        One instantiated `OpenAILLM` client.

    Raises:
        RuntimeError:
            If AgenticPay `OpenAILLM` is unavailable or API key is missing.

    中文翻译：
    按固定构造参数创建 OpenAILLM；如果依赖缺失或密钥未配置，立即失败并给出
    明确安装提示。
    """
    try:
        from agenticpay.models.openai_llm import OpenAILLM  # noqa: PLC0415
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError(
            "missing dependency: "
            f"{exc}. Install with: "
            "pip install \"git+https://github.com/SafeRL-Lab/AgenticPay.git\" "
            "&& pip install -e ."
        ) from exc

    api_key = os.getenv(runtime_config.api_key_env)
    if not api_key:
        raise RuntimeError(f"{runtime_config.api_key_env} is not set.")
    return OpenAILLM(model=runtime_config.model, api_key=api_key)


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
        raise RuntimeError(
            "missing dependency: "
            f"{exc}. Install with: "
            "pip install \"git+https://github.com/SafeRL-Lab/AgenticPay.git\" "
            "&& pip install -e ."
        ) from exc

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
