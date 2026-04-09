"""Utilities for parsing and validating structured platform actions."""

from __future__ import annotations

from typing import Any

from conversational_consumer_selection.schemas import (
    ActionType,
    clarification_slot_is_revealed,
    normalize_clarification_slot,
    Observation,
    SelectionAction,
)


def parse_action_payload(payload: dict[str, Any]) -> SelectionAction:
    """Convert one raw JSON action payload into a structured SelectionAction."""

    if not isinstance(payload, dict):
        raise ValueError("action payload must be a JSON object")

    action_type = ActionType(payload["action_type"])
    explanation = str(payload.get("explanation", ""))

    if action_type is ActionType.ASK_CLARIFICATION:
        return SelectionAction.ask_clarification(str(payload["slot"]), explanation=explanation)
    if action_type is ActionType.COMPARE_OPTIONS:
        return SelectionAction.compare_options(
            str(payload["offer_id"]),
            str(payload["comparison_offer_id"]),
            explanation=explanation,
        )
    if action_type is ActionType.RECOMMEND_OPTION:
        return SelectionAction.recommend_option(str(payload["offer_id"]), explanation=explanation)
    if action_type is ActionType.COMMIT_SELECTION:
        return SelectionAction.commit_selection(str(payload["offer_id"]), explanation=explanation)
    return SelectionAction.escalate(explanation=explanation)


def validate_action_against_observation(action: SelectionAction, observation: Observation) -> None:
    """Ensure the chosen action is executable in the current observation."""

    offer_ids = {offer.offer_id for offer in observation.offers}

    if action.action_type is ActionType.ASK_CLARIFICATION:
        normalized_slot = normalize_clarification_slot(action.slot or "")
        if normalized_slot not in observation.available_clarification_slots:
            raise ValueError(f"unknown clarification slot: {action.slot}")
        if clarification_slot_is_revealed(normalized_slot, observation.revealed_context):
            raise ValueError(f"slot already known: {normalized_slot}")
        return

    if action.action_type is ActionType.COMPARE_OPTIONS:
        if action.offer_id not in offer_ids or action.comparison_offer_id not in offer_ids:
            raise ValueError("compare_options references unknown offer id")
        return

    if action.action_type in {ActionType.RECOMMEND_OPTION, ActionType.COMMIT_SELECTION}:
        if action.offer_id not in offer_ids:
            raise ValueError(f"unknown offer id: {action.offer_id}")
        if action.action_type is ActionType.COMMIT_SELECTION:
            if not _has_prior_recommendation_for_offer(observation, action.offer_id):
                raise ValueError(
                    "commit_selection requires a prior recommend_option for the same offer"
                )


def _has_prior_recommendation_for_offer(observation: Observation, offer_id: str | None) -> bool:
    if offer_id is None:
        return False
    for entry in observation.history:
        if (
            entry.action.action_type is ActionType.RECOMMEND_OPTION
            and entry.action.offer_id == offer_id
        ):
            return True
    return False
