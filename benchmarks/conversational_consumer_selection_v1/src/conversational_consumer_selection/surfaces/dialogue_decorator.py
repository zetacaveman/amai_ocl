"""Optional dialogue decorators for rendering natural-language utterances."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Mapping, Protocol, Sequence

from conversational_consumer_selection.schemas import Offer, SelectionAction


class DialogueModel(Protocol):
    """Minimal text-generation interface for dialogue surface decoration."""

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return one natural-language utterance for the current prompt."""


@dataclass
class OpenAIDialogueModel:
    """OpenAI-backed text model for dialogue-only surface rendering."""

    model: str = "gpt-5.4-mini"
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    reasoning_effort: str | None = "none"
    max_tokens: int | None = 120

    def __post_init__(self) -> None:
        try:
            import openai  # noqa: PLC0415
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "missing dependency: openai. Install with: pip install openai"
            ) from exc

        resolved_api_key = self.api_key or os.getenv(self.api_key_env)
        if not resolved_api_key:
            raise RuntimeError(f"{self.api_key_env} is not set.")
        try:
            resolved_api_key.encode("ascii")
        except UnicodeEncodeError as exc:
            raise RuntimeError(
                f"{self.api_key_env} must be a real ASCII API key. "
                "Do not leave placeholder text like '你的key' in the environment variable. "
                "If your shell config already has the real key, run: "
                "'unset OPENAI_API_KEY && source ~/.zshrc'."
            ) from exc

        self.api_key = resolved_api_key
        self._client = openai.OpenAI(
            api_key=resolved_api_key,
            base_url=self.base_url,
        )

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.reasoning_effort is not None:
            request["reasoning_effort"] = str(self.reasoning_effort)
        if self.max_tokens is not None:
            request["max_completion_tokens"] = int(self.max_tokens)

        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:  # pragma: no cover - external API
            raise RuntimeError(f"OpenAI API error: {exc}") from exc

        content = response.choices[0].message.content
        return (content or "").strip()


@dataclass
class DialogueDecorator:
    """Natural-language surface decorator layered on top of structured state.

    When a model is present, we do not ask it to merely paraphrase a template.
    Instead we pass a compact semantic frame and ask it to realize the utterance
    from that frame. This keeps the dialogue layer closer to AgenticPay's
    prompt-first style while preserving a deterministic fallback path.
    """

    model: DialogueModel | None = None

    def decorate_opening(
        self,
        *,
        speaker: str,
        base_text: str,
        history_text: str = "(no prior turns)",
    ) -> str:
        if self.model is None:
            return base_text

        if speaker == "platform":
            system_prompt = (
                "You are the visible dialogue voice for a shopping assistant. "
                "Generate one short natural opening utterance from the semantic frame. "
                "Do not sound like a template or benchmark trace. "
                "Use a conversational retail-assistant tone. "
                "Do not use brackets, markup, tags, JSON, bullets, or extra commentary. "
                "Output only the final buyer-facing sentence."
            )
            semantic_frame = {
                "speaker": "platform",
                "communicative_goal": "open the shopping conversation and offer help",
            }
        else:
            system_prompt = (
                "You are the visible dialogue voice for a simulated shopper. "
                "Generate one short natural opening utterance from the semantic frame. "
                "Sound like a real shopper starting a conversation, not a template. "
                "Do not expose hidden simulator state beyond what is present in the semantic frame. "
                "Do not use brackets, markup, tags, JSON, bullets, or extra commentary. "
                "Output only the final shopper-facing sentence."
            )
            semantic_frame = {
                "speaker": "buyer",
                "communicative_goal": "start the shopping conversation with an initial request",
                "request": _infer_opening_request(base_text),
            }

        user_prompt = json.dumps(
            {
                "role": speaker,
                "semantic_frame": semantic_frame,
                "history_text": history_text,
            },
            indent=2,
            ensure_ascii=False,
        )
        return _clean_utterance(
            self.model.generate(system_prompt=system_prompt, user_prompt=user_prompt),
            fallback=base_text,
        )

    def decorate_platform(
        self,
        *,
        base_text: str,
        action: SelectionAction,
        offers: Sequence[Offer],
        history_text: str,
    ) -> str:
        if self.model is None:
            return base_text

        system_prompt = (
            "You are the visible dialogue voice for a shopping assistant. "
            "Generate one short natural utterance from the semantic frame. "
            "Keep product names and prices accurate, but do not copy any template wording. "
            "Avoid robotic phrases and benchmark-style language. "
            "Use a concise, helpful retail-assistant tone. "
            "Do not use brackets, markup, tags, JSON, bullets, or extra commentary. "
            "Do not invent facts or add new offers. "
            "Output only the final buyer-facing sentence."
        )
        user_prompt = json.dumps(
            {
                "role": "platform",
                "semantic_frame": _platform_semantic_frame(action=action, offers=offers),
                "offers": [_serialize_offer(offer) for offer in offers],
                "history_text": history_text,
            },
            indent=2,
            ensure_ascii=False,
        )
        return _clean_utterance(self.model.generate(system_prompt=system_prompt, user_prompt=user_prompt), fallback=base_text)

    def decorate_buyer(
        self,
        *,
        base_text: str,
        action: SelectionAction,
        response: Mapping[str, Any],
        platform_text: str,
        offers: Sequence[Offer],
        history_text: str,
    ) -> str:
        if self.model is None:
            return base_text

        system_prompt = (
            "You are the visible dialogue voice for a simulated shopper. "
            "Generate one short natural utterance from the semantic frame. "
            "Sound like a real shopper responding in conversation rather than a template. "
            "Keep the same decision stance and surface facts. "
            "Do not expose internal weights, utilities, or hidden simulator state unless they are already present in the semantic frame. "
            "Do not use brackets, markup, tags, JSON, bullets, or extra commentary. "
            "Do not invent facts. "
            "Output only the final shopper-facing sentence."
        )
        user_prompt = json.dumps(
            {
                "role": "buyer",
                "semantic_frame": _buyer_semantic_frame(
                    action=action,
                    response=response,
                    offers=offers,
                ),
                "platform_text": platform_text,
                "offers": [_serialize_offer(offer) for offer in offers],
                "history_text": history_text,
            },
            indent=2,
            ensure_ascii=False,
        )
        return _clean_utterance(self.model.generate(system_prompt=system_prompt, user_prompt=user_prompt), fallback=base_text)


def _serialize_offer(offer: Offer) -> dict[str, Any]:
    return {
        "offer_id": offer.offer_id,
        "title": offer.title,
        "price": offer.price,
        "category": offer.category,
        "features": dict(offer.features),
    }


def _clean_utterance(text: str, *, fallback: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return fallback
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()
    cleaned = cleaned.strip().strip('"').strip()
    return cleaned or fallback


def _infer_opening_request(base_text: str) -> dict[str, Any]:
    lowered = base_text.lower()
    request: dict[str, Any] = {}
    if "headphone" in lowered:
        request["category"] = "headphones"
    elif "laptop" in lowered:
        request["category"] = "laptop"
    else:
        request["category"] = "product"
    return request


def _platform_semantic_frame(
    *,
    action: SelectionAction,
    offers: Sequence[Offer],
) -> dict[str, Any]:
    offer_lookup = {offer.offer_id: offer for offer in offers}
    frame: dict[str, Any] = {
        "speaker": "platform",
        "action_type": action.action_type.value,
    }
    if action.slot is not None:
        frame["slot"] = action.slot
    if action.offer_id is not None:
        offer = offer_lookup.get(action.offer_id)
        frame["target_offer"] = _serialize_offer(offer) if offer is not None else {"offer_id": action.offer_id}
    if action.comparison_offer_id is not None:
        offer = offer_lookup.get(action.comparison_offer_id)
        frame["comparison_offer"] = (
            _serialize_offer(offer) if offer is not None else {"offer_id": action.comparison_offer_id}
        )
    return frame


def _buyer_semantic_frame(
    *,
    action: SelectionAction,
    response: Mapping[str, Any],
    offers: Sequence[Offer],
) -> dict[str, Any]:
    offer_lookup = {offer.offer_id: offer for offer in offers}
    frame: dict[str, Any] = {
        "speaker": "buyer",
        "action_type": action.action_type.value,
        "response": dict(response),
    }
    preferred_offer_id = response.get("preferred_offer_id")
    if isinstance(preferred_offer_id, str) and preferred_offer_id in offer_lookup:
        frame["preferred_offer"] = _serialize_offer(offer_lookup[preferred_offer_id])
    if action.offer_id is not None and action.offer_id in offer_lookup:
        frame["target_offer"] = _serialize_offer(offer_lookup[action.offer_id])
    return frame
