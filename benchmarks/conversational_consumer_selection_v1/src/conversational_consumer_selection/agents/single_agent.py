"""LLM-backed platform agent for benchmark-side decision making.

This module turns the public `Observation` contract into one structured
`SelectionAction`. The agent may keep richer internal beliefs, but only the
action leaves the module and reaches the environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Mapping, Protocol

from conversational_consumer_selection.parsers import (
    parse_action_payload,
    validate_action_against_observation,
)
from conversational_consumer_selection.schemas import (
    ActionType,
    CLARIFICATION_BUDGET_MAX,
    CLARIFICATION_MUST_HAVE_PREFIX,
    CLARIFICATION_PREFERENCE_PREFIX,
    clarification_slot_is_revealed,
    HistoryEntry,
    normalize_clarification_slot,
    Observation,
    Offer,
    SelectionAction,
)

_OBSERVATION_JSON_MARKER = "Observation JSON:"


class PlatformAgentModel(Protocol):
    """Minimal text-generation interface for the platform-side agent."""

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return one raw text completion for the current prompt."""


@dataclass
class OpenAIPlatformAgentModel:
    """OpenAI-backed text model for the platform-side agent."""

    model: str = "gpt-5.4-mini"
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    reasoning_effort: str | None = "none"
    max_tokens: int | None = 200
    response_format_json: bool = True

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
        if self.response_format_json:
            request["response_format"] = {"type": "json_object"}

        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:
            if self.response_format_json:
                fallback_request = dict(request)
                fallback_request.pop("response_format", None)
                response = self._client.chat.completions.create(**fallback_request)
            else:
                raise RuntimeError(f"OpenAI API error: {exc}") from exc

        content = response.choices[0].message.content
        return (content or "").strip()


@dataclass(frozen=True)
class SingleAgentDecisionTrace:
    """One platform-side decision trace for debugging and demos."""

    context: Mapping[str, Any]
    system_prompt: str
    user_prompt: str
    raw_output: str
    action: SelectionAction
    working_belief: Mapping[str, Any] | None = None
    used_fallback: bool = False
    error: str | None = None


def build_platform_agent_context(
    observation: Observation,
    *,
    include_user_utterance_history: bool = False,
) -> dict[str, Any]:
    """Serialize the observation contract for one platform decision step."""

    context = {
        "level": observation.level.value,
        "revealed_context": {
            "category": observation.revealed_context["category"],
            "budget_max": observation.revealed_context["budget_max"],
            "must_have": dict(observation.revealed_context["must_have"]),
            "preference_weights": dict(observation.revealed_context["preference_weights"]),
        },
        "available_clarification_slots": list(observation.available_clarification_slots),
        "offers": [_serialize_offer(offer) for offer in observation.offers],
        "turn_index": observation.turn_index,
        "max_turns": observation.max_turns,
        "remaining_turns": observation.remaining_turns,
        "history": [_serialize_history_entry(entry) for entry in observation.history],
        "committed_offer_id": observation.committed_offer_id,
        "escalated": observation.escalated,
        "terminated": observation.terminated,
    }
    if include_user_utterance_history and observation.user_utterance_history:
        context["user_utterance_history"] = list(observation.user_utterance_history)
    return context


def build_platform_agent_system_prompt() -> str:
    """Return the fixed system prompt for the platform-side agent."""

    return (
        "You are the platform-side single agent in a conversational consumer selection "
        "environment. Choose exactly one next structured action from the current revealed "
        "context, candidate offers, and structured interaction history. "
        "In partial-intent settings, user_utterance_history is additional user-side natural "
        "language evidence; use it together with the structured contract, not instead of it. "
        "Return exactly one JSON object and no extra text. "
        "Valid action_type values are: ask_clarification, compare_options, "
        "recommend_option, commit_selection, escalate. "
        "For ask_clarification provide slot using the benchmark namespace. "
        "Supported clarification slots follow a product-agnostic schema such as "
        "budget.max, must_have.<constraint>, and preference.<attribute>. "
        "As a minimal platform-side strategy, prefer asking only clarifications that are "
        "decision-relevant. Ask budget.max when it is unknown. Ask must_have.<constraint> "
        "only when candidate offers differ on that constraint. Ask preference.<attribute> "
        "when candidate offers differ on that attribute and the preference is still unknown. "
        "Avoid asking redundant or non-discriminative clarification questions. "
        "For compare_options provide offer_id and comparison_offer_id. "
        "For recommend_option and commit_selection provide offer_id. "
        "Never use commit_selection before you have already recommended that same offer "
        "in a previous turn. "
        "For escalate do not provide slot or offer ids. "
        "You may optionally return an internal planning object with this shape: "
        '{"working_belief": {...}, "next_action": {...}}. '
        "If you use that form, next_action must contain the executable action. "
        "working_belief is only an internal guess about the user's consumer model and "
        "will not be executed by the environment."
    )


def build_platform_agent_user_prompt(context: Mapping[str, Any]) -> str:
    """Render the user prompt that carries the structured observation context."""

    return (
        "Choose the next action for the platform-side decision module.\n"
        "Return only JSON.\n"
        f"{_OBSERVATION_JSON_MARKER}\n"
        f"{json.dumps(context, indent=2, sort_keys=True, ensure_ascii=False)}"
    )


def build_single_agent_context(observation: Observation) -> dict[str, Any]:
    """Backward-compatible alias for the platform agent context builder."""

    return build_platform_agent_context(observation)


def build_single_agent_system_prompt() -> str:
    """Backward-compatible alias for the platform agent system prompt."""

    return build_platform_agent_system_prompt()


def build_single_agent_user_prompt(context: Mapping[str, Any]) -> str:
    """Backward-compatible alias for the platform agent user prompt."""

    return build_platform_agent_user_prompt(context)


@dataclass
class LLMPlatformAgent:
    """Policy wrapper that turns a text model into a structured platform action."""

    model: PlatformAgentModel
    retries: int = 1
    include_user_utterance_history: bool = False

    def act(self, observation: Observation) -> SelectionAction:
        return self.decide(observation).action

    def decide(self, observation: Observation) -> SingleAgentDecisionTrace:
        # The prompt carries only the benchmark contract. If a model wants to
        # maintain a richer guess about the user, it can return working_belief
        # as a non-executable trace alongside the actual action.
        context = build_platform_agent_context(
            observation,
            include_user_utterance_history=self.include_user_utterance_history,
        )
        system_prompt = build_platform_agent_system_prompt()
        base_user_prompt = build_platform_agent_user_prompt(context)
        user_prompt = base_user_prompt
        last_error: str | None = None
        raw_output = ""

        for _ in range(self.retries + 1):
            raw_output = self.model.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            try:
                payload = _load_json_object(raw_output)
                working_belief = _extract_working_belief(payload)
                action = _parse_action_payload_object(payload)
                validate_action_against_observation(action, observation)
                return SingleAgentDecisionTrace(
                    context=context,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    raw_output=raw_output,
                    action=action,
                    working_belief=working_belief,
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                user_prompt = (
                    f"{base_user_prompt}\n\n"
                    "The previous action was invalid for the current observation.\n"
                    f"Validation error: {last_error}\n"
                    "Return one corrected executable JSON action only."
                )

        fallback = SelectionAction.escalate("invalid_model_output")
        return SingleAgentDecisionTrace(
            context=context,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_output=raw_output,
            action=fallback,
            used_fallback=True,
            error=last_error,
        )


@dataclass
class DemoPlatformAgentModel:
    """Deterministic demo backend that behaves like a platform-side model."""

    clarify_missing_preferences: bool = True
    price_penalty: float = 0.01

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        del system_prompt
        context = _extract_context_from_user_prompt(user_prompt)
        revealed_preferences = context["revealed_context"]["preference_weights"]
        history = context["history"]

        if self.clarify_missing_preferences:
            for slot in self._clarification_priority(context):
                if not clarification_slot_is_revealed(slot, context["revealed_context"]):
                    return json.dumps(
                        {
                            "action_type": ActionType.ASK_CLARIFICATION.value,
                            "slot": slot,
                            "explanation": f"recover missing decision-relevant signal: {slot}",
                        }
                    )

        ranked_offers = self._rank_feasible_offers(context)
        if not ranked_offers:
            return json.dumps(
                {
                    "action_type": ActionType.ESCALATE.value,
                    "explanation": "no feasible offer under explicit constraints",
                }
            )

        if len(ranked_offers) >= 2 and not self._action_already_taken(history, ActionType.COMPARE_OPTIONS):
            return json.dumps(
                {
                    "action_type": ActionType.COMPARE_OPTIONS.value,
                    "offer_id": ranked_offers[0]["offer_id"],
                    "comparison_offer_id": ranked_offers[1]["offer_id"],
                    "explanation": "compare the top two feasible offers",
                }
            )

        if not self._action_already_taken(history, ActionType.RECOMMEND_OPTION):
            return json.dumps(
                {
                    "action_type": ActionType.RECOMMEND_OPTION.value,
                    "offer_id": ranked_offers[0]["offer_id"],
                    "explanation": "recommend the current best feasible offer",
                }
            )

        return json.dumps(
            {
                "action_type": ActionType.COMMIT_SELECTION.value,
                "offer_id": ranked_offers[0]["offer_id"],
                "explanation": "commit the current best feasible offer",
            }
        )

    def _rank_feasible_offers(self, context: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        feasible_offers = [
            offer for offer in context["offers"] if self._is_explicitly_feasible(context, offer)
        ]
        return sorted(
            feasible_offers,
            key=lambda offer: self._score_offer(context, offer),
            reverse=True,
        )

    def _score_offer(self, context: Mapping[str, Any], offer: Mapping[str, Any]) -> float:
        score = 0.0
        revealed_preferences = context["revealed_context"]["preference_weights"]
        attribute_values = offer.get("attribute_values", {})
        for slot, weight in revealed_preferences.items():
            score += float(weight) * float(attribute_values.get(slot, 0.0))
        score -= self.price_penalty * float(offer["price"])
        return score

    def _is_explicitly_feasible(self, context: Mapping[str, Any], offer: Mapping[str, Any]) -> bool:
        revealed_context = context["revealed_context"]
        if offer["category"] != revealed_context["category"]:
            return False
        budget_max = revealed_context["budget_max"]
        if budget_max is not None and float(offer["price"]) > float(budget_max):
            return False

        features = offer.get("features", {})
        attribute_values = offer.get("attribute_values", {})
        for slot, expected in revealed_context["must_have"].items():
            actual = features.get(slot)
            if isinstance(expected, bool):
                if expected is False:
                    continue
                if bool(actual) is not expected:
                    return False
            elif isinstance(expected, (int, float)):
                candidate = features.get(slot, attribute_values.get(slot))
                if not isinstance(candidate, (int, float)) or float(candidate) < float(expected):
                    return False
            elif actual != expected:
                return False
        return True

    def _clarification_priority(self, context: Mapping[str, Any]) -> list[str]:
        slots: list[str] = []
        offers = list(context["offers"])

        if CLARIFICATION_BUDGET_MAX in context["available_clarification_slots"]:
            slots.append(CLARIFICATION_BUDGET_MAX)

        must_have_slots = [
            slot
            for slot in context["available_clarification_slots"]
            if slot.startswith(CLARIFICATION_MUST_HAVE_PREFIX)
        ]
        for slot in must_have_slots:
            constraint = slot[len(CLARIFICATION_MUST_HAVE_PREFIX) :]
            values = {offer.get("features", {}).get(constraint) for offer in offers}
            if len(values) > 1:
                slots.append(slot)

        preference_slots = [
            slot
            for slot in context["available_clarification_slots"]
            if slot.startswith(CLARIFICATION_PREFERENCE_PREFIX)
        ]
        for slot in preference_slots:
            attribute = slot[len(CLARIFICATION_PREFERENCE_PREFIX) :]
            values = {
                float(offer.get("attribute_values", {}).get(attribute, 0.0)) for offer in offers
            }
            if len(values) > 1:
                slots.append(slot)
        return slots

    @staticmethod
    def _action_already_taken(
        history: list[Mapping[str, Any]],
        action_type: ActionType,
    ) -> bool:
        for entry in history:
            action = entry.get("action", {})
            if action.get("action_type") == action_type.value:
                return True
        return False


def _extract_context_from_user_prompt(user_prompt: str) -> dict[str, Any]:
    if _OBSERVATION_JSON_MARKER not in user_prompt:
        raise ValueError("user prompt missing observation marker")
    _, payload = user_prompt.split(_OBSERVATION_JSON_MARKER, maxsplit=1)
    return json.loads(payload.strip())


def _serialize_offer(offer: Offer) -> dict[str, Any]:
    return {
        "offer_id": offer.offer_id,
        "category": offer.category,
        "price": offer.price,
        "title": offer.title,
        "features": dict(offer.features),
        "attribute_values": dict(offer.attribute_values),
    }


def _serialize_history_entry(entry: HistoryEntry) -> dict[str, Any]:
    action = entry.action
    return {
        "turn_index": entry.turn_index,
        "action": {
            "action_type": action.action_type.value,
            "slot": normalize_clarification_slot(action.slot) if action.slot else None,
            "offer_id": action.offer_id,
            "comparison_offer_id": action.comparison_offer_id,
            "explanation": action.explanation,
        },
        "response": dict(entry.response),
    }


def _parse_action(raw_output: str) -> SelectionAction:
    payload = _load_json_object(raw_output)
    return _parse_action_payload_object(payload)


def _parse_action_payload_object(payload: dict[str, Any]) -> SelectionAction:
    if "next_action" in payload:
        next_action = payload["next_action"]
        if not isinstance(next_action, dict):
            raise ValueError("next_action must be a JSON object")
        return parse_action_payload(next_action)
    return parse_action_payload(payload)


def _extract_working_belief(payload: dict[str, Any]) -> Mapping[str, Any] | None:
    working_belief = payload.get("working_belief")
    if working_belief is None:
        return None
    if not isinstance(working_belief, dict):
        raise ValueError("working_belief must be a JSON object")
    return dict(working_belief)


def _load_json_object(raw_output: str) -> dict[str, Any]:
    text = raw_output.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("model output must be a JSON object")
    return payload


# Backward-compatible aliases for older imports.
SingleAgentModel = PlatformAgentModel
OpenAISingleAgentModel = OpenAIPlatformAgentModel
PromptBasedSingleAgent = LLMPlatformAgent
DemoSingleAgentModel = DemoPlatformAgentModel
