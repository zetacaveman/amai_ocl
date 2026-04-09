"""Buyer-side dialogue surface renderers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from conversational_consumer_selection.schemas import (
    CLARIFICATION_BUDGET_MAX,
    CLARIFICATION_MUST_HAVE_PREFIX,
    CLARIFICATION_PREFERENCE_PREFIX,
    normalize_clarification_slot,
    Offer,
    SelectionAction,
)


def render_buyer_response(
    action: SelectionAction,
    response: Mapping[str, Any],
    *,
    offers: Sequence[Offer] | None = None,
    include_signal_tag: bool = False,
    annotate_entities: bool = False,
) -> str:
    """Render one buyer-side utterance from the structured simulator response."""

    del include_signal_tag

    if response.get("status") == "escalated":
        return "Okay, please hand this off for escalation."

    offer_lookup = {offer.offer_id: offer for offer in offers or ()}
    action_type = action.action_type.value

    if action_type == "ask_clarification":
        if response.get("status") == "clarified":
            return _render_clarification_reveal(
                slot=str(response["slot"]),
                value=response["value"],
                annotate_entities=annotate_entities,
            )
        return "I am not sure how to answer that directly."

    if action_type == "compare_options":
        if response.get("status") == "compared":
            preferred_offer_id = response.get("preferred_offer_id")
            if preferred_offer_id is None:
                return "Those two seem pretty similar to me."
            return (
                "If I had to choose, I would lean toward "
                f"{_offer_reference(offer_lookup.get(str(preferred_offer_id)), str(preferred_offer_id), annotate_entities=annotate_entities)}."
            )
        return "I cannot really compare those two because at least one already looks off for me."

    if action_type == "recommend_option":
        status = response.get("status")
        if status == "accept":
            return "That sounds like the strongest fit so far."
        if status == "hesitate":
            return "That could work, but I would want a bit more confidence before deciding."
        if response.get("outside_option_preferred"):
            return "At that point I would rather not buy anything."
        return _render_recommendation_rejection(reason=str(response.get("reason", "reject")))

    if action_type == "commit_selection":
        if response.get("accepted"):
            return "Yes, go ahead and finalize that option."
        if response.get("executed_violation"):
            return _render_commit_violation(reason=str(response.get("reason", "violation")))
        if response.get("outside_option_preferred"):
            return "I would rather not buy anything than commit to that option."
        return "I am not ready to finalize that option."

    return "I need a different next step."


def _render_clarification_reveal(slot: str, value: Any, *, annotate_entities: bool) -> str:
    normalized_slot = normalize_clarification_slot(slot)
    if normalized_slot == CLARIFICATION_BUDGET_MAX:
        return f"My budget ceiling is about ${float(value):.0f}."

    if normalized_slot.startswith(CLARIFICATION_MUST_HAVE_PREFIX):
        key = normalized_slot[len(CLARIFICATION_MUST_HAVE_PREFIX) :]
        readable_key = _slot_reference(key, annotate_entities=annotate_entities)
        if bool(value):
            return f"Yes, {readable_key} is something I definitely need."
        return f"No, {readable_key} is not a strict requirement for me."

    slot_name = normalized_slot
    if normalized_slot.startswith(CLARIFICATION_PREFERENCE_PREFIX):
        slot_name = normalized_slot[len(CLARIFICATION_PREFERENCE_PREFIX) :]
    slot_anchor = _slot_reference(slot_name, annotate_entities=annotate_entities)
    importance = float(value)
    if slot_name == "comfort":
        if importance >= 1.0:
            return f"{slot_anchor} matters a lot to me, especially if I will wear them for a while."
        if importance >= 0.7:
            return f"{slot_anchor} is pretty important to me."
        return f"{slot_anchor} matters some, but it is not my main concern."

    if slot_name == "portability":
        if importance >= 1.0:
            return f"{slot_anchor} is a big deal for me. I want something easy to carry around."
        if importance >= 0.7:
            return f"{slot_anchor} matters quite a bit to me."
        return f"{slot_anchor} is nice to have, but it is not my top priority."

    if slot_name == "battery":
        if importance >= 1.0:
            return (
                f"{slot_anchor} matters a lot to me. I do not want to charge them constantly."
            )
        if importance >= 0.7:
            return f"{slot_anchor} is pretty important to me."
        return f"{slot_anchor} matters some, but I can be flexible on it."

    return f"{slot_anchor} is {_importance_descriptor(importance)} for me."


def _render_recommendation_rejection(reason: str) -> str:
    if reason == "dominated_offer":
        return "I do not think that is the best option for me."
    if reason == "budget_exceeded":
        return "That seems too expensive for me."
    if reason == "must_have_missing":
        return "That misses something I explicitly need."
    if reason == "category_mismatch":
        return "That is not really the kind of product I am looking for."
    return "I do not think that recommendation is right for me."


def _render_commit_violation(reason: str) -> str:
    if reason == "budget_exceeded":
        return "I cannot finalize that because it goes over my budget."
    if reason == "must_have_missing":
        return "I cannot finalize that because it misses one of my must-haves."
    if reason == "category_mismatch":
        return "I cannot finalize that because it is not the right product category."
    if reason == "unknown_offer":
        return "I cannot finalize that because I do not recognize that option."
    return "I cannot finalize that because it conflicts with what I asked for."


def _importance_descriptor(value: float) -> str:
    if value >= 1.0:
        return "extremely important"
    if value >= 0.7:
        return "pretty important"
    if value >= 0.4:
        return "somewhat important"
    return "nice to have"


def _slot_reference(slot: str, *, annotate_entities: bool) -> str:
    if annotate_entities:
        return _slot_anchor(slot)
    return slot.replace("_", " ")


def _slot_anchor(slot: str) -> str:
    return f"[[SLOT:{slot}]]"


def _offer_reference(
    offer: Offer | None,
    fallback_offer_id: str,
    *,
    annotate_entities: bool,
) -> str:
    if annotate_entities:
        return _offer_anchor(offer, fallback_offer_id)
    if offer is None:
        return fallback_offer_id
    if offer.title:
        return offer.title
    return offer.offer_id


def _offer_anchor(offer: Offer | None, fallback_offer_id: str) -> str:
    if offer is None:
        return f"[[PRODUCT:{fallback_offer_id}]]"
    if offer.title:
        return f"[[PRODUCT:{offer.offer_id}|{offer.title}]]"
    return f"[[PRODUCT:{offer.offer_id}]]"
