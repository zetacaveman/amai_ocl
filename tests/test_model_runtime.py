from __future__ import annotations

from types import SimpleNamespace
import sys

import pytest

from aimai_ocl.model_runtime import OpenAIChatLLM, _sanitize_text_payload


class _DummyCompletions:
    def __init__(
        self,
        fail_first: bool = False,
        fail_unsupported_max_tokens: bool = False,
        fail_unsupported_temperature: bool = False,
    ) -> None:
        self.fail_first = fail_first
        self.fail_unsupported_max_tokens = fail_unsupported_max_tokens
        self.fail_unsupported_temperature = fail_unsupported_temperature
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.fail_unsupported_temperature and len(self.calls) == 1:
            raise RuntimeError(
                "Error code: 400 - {'error': {'message': "
                "\"Unsupported value: 'temperature' is not supported with "
                "this model.\"}}"
            )
        if self.fail_unsupported_max_tokens and len(self.calls) == 1:
            raise RuntimeError(
                "Error code: 400 - {'error': {'message': "
                "\"Unsupported parameter: 'max_tokens' is not supported with "
                "this model. Use 'max_completion_tokens' instead.\"}}"
            )
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError(
                "Error code: 400 - {'error': {'message': "
                "\"We could not parse the JSON body of your request.\"}}"
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=" ok "),
                ),
            ],
        )


class _DummyOpenAIClient:
    last_completions: _DummyCompletions | None = None

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        fail_first: bool = False,
        fail_unsupported_max_tokens: bool = False,
        fail_unsupported_temperature: bool = False,
    ) -> None:
        del api_key, base_url
        completions = _DummyCompletions(
            fail_first=fail_first,
            fail_unsupported_max_tokens=fail_unsupported_max_tokens,
            fail_unsupported_temperature=fail_unsupported_temperature,
        )
        _DummyOpenAIClient.last_completions = completions
        self.chat = SimpleNamespace(completions=completions)


def _install_fake_openai(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_first: bool = False,
    fail_unsupported_max_tokens: bool = False,
    fail_unsupported_temperature: bool = False,
) -> None:
    def _factory(*, api_key: str, base_url: str | None = None) -> _DummyOpenAIClient:
        return _DummyOpenAIClient(
            api_key=api_key,
            base_url=base_url,
            fail_first=fail_first,
            fail_unsupported_max_tokens=fail_unsupported_max_tokens,
            fail_unsupported_temperature=fail_unsupported_temperature,
        )

    fake_module = SimpleNamespace(OpenAI=_factory)
    monkeypatch.setitem(sys.modules, "openai", fake_module)


def test_sanitize_text_payload_removes_invalid_json_characters() -> None:
    raw = f"hello\x00world{chr(0xD800)}\x01done"
    sanitized = _sanitize_text_payload(raw)

    assert "\x00" not in sanitized
    assert chr(0xD800) not in sanitized
    assert "\x01" not in sanitized
    assert "hello" in sanitized
    assert "world" in sanitized
    assert "done" in sanitized


def test_openai_chat_llm_builds_json_safe_request(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch)
    llm = OpenAIChatLLM(model="gpt-4o-mini", api_key="test-key")

    response = llm.generate(
        f"offer\x00price{chr(0xD800)}",
        temperature=float("nan"),
        max_tokens=5,
        metadata={"safe": "x\x00", "bad": float("inf")},
    )

    assert response == "ok"
    calls = _DummyOpenAIClient.last_completions.calls
    assert len(calls) == 1
    request = calls[0]
    assert "temperature" not in request
    assert request["max_tokens"] == 5
    assert request["messages"][0]["content"] == "offerprice?"
    assert request["metadata"] == {"safe": "x"}


def test_openai_chat_llm_uses_max_completion_tokens_for_gpt5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_openai(monkeypatch)
    llm = OpenAIChatLLM(model="gpt-5.4", api_key="test-key")

    response = llm.generate("plain prompt", temperature=0.0, max_tokens=9)

    assert response == "ok"
    calls = _DummyOpenAIClient.last_completions.calls
    assert len(calls) == 1
    assert calls[0]["max_completion_tokens"] == 9
    assert "max_tokens" not in calls[0]


def test_openai_chat_llm_retries_unsupported_max_tokens_with_completion_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_openai(monkeypatch, fail_unsupported_max_tokens=True)
    llm = OpenAIChatLLM(model="custom-new-model", api_key="test-key")

    response = llm.generate("plain prompt", temperature=0.0, max_tokens=11)

    assert response == "ok"
    calls = _DummyOpenAIClient.last_completions.calls
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == 11
    assert calls[1]["max_completion_tokens"] == 11
    assert "max_tokens" not in calls[1]


def test_openai_chat_llm_retries_unsupported_temperature_without_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_openai(monkeypatch, fail_unsupported_temperature=True)
    llm = OpenAIChatLLM(model="gpt-5.4", api_key="test-key")

    response = llm.generate("plain prompt", temperature=0.0, max_tokens=13)

    assert response == "ok"
    calls = _DummyOpenAIClient.last_completions.calls
    assert len(calls) == 2
    assert "temperature" in calls[0]
    assert "temperature" not in calls[1]
    assert calls[1]["max_completion_tokens"] == 13


def test_openai_chat_llm_retries_after_invalid_json_body(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openai(monkeypatch, fail_first=True)
    llm = OpenAIChatLLM(model="gpt-4o-mini", api_key="test-key")

    response = llm.generate("plain prompt", temperature=0.0, max_tokens=7)

    assert response == "ok"
    calls = _DummyOpenAIClient.last_completions.calls
    assert len(calls) == 2
    assert "temperature" in calls[0]
    assert "temperature" not in calls[1]
    assert calls[1]["max_tokens"] == 7
