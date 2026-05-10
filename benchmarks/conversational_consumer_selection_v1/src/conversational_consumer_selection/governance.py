"""Lightweight governance adapter for the structured V0 selection demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from conversational_consumer_selection.policies import Policy
from conversational_consumer_selection.schemas import (
    ActionType,
    CLARIFICATION_BUDGET_MAX,
    CLARIFICATION_MUST_HAVE_PREFIX,
    CLARIFICATION_PREFERENCE_PREFIX,
    clarification_slot_is_revealed,
    Observation,
    Offer,
    SelectionAction,
)


@dataclass(frozen=True)
class GovernanceDecision:
    """One governance decision made between a raw policy and the environment."""

    raw_action: SelectionAction
    governed_action: SelectionAction
    intervention: str
    reason: str


class GovernedPolicy(Protocol):
    """Policy protocol for wrappers that expose the most recent decision trace."""

    last_decision: GovernanceDecision | None

    def act(self, observation: Observation) -> SelectionAction:
        """Return the governed action."""


@dataclass
class SelectionGovernedPolicy:
    """Thin V0 action-governance wrapper for the selection benchmark.

    This is intentionally benchmark-local. It demonstrates the OCL idea
    (gate/rewrite/escalate before execution) without claiming to be the full
    `ocl_full` controller used in AgenticPay experiments.
    """

    raw_policy: Policy
    require_budget_before_selection: bool = True
    require_relevant_must_have_before_selection: bool = True
    price_penalty: float = 0.01
    last_decision: GovernanceDecision | None = None

    def act(self, observation: Observation) -> SelectionAction:
        raw_action = self.raw_policy.act(observation)
        governed_action, intervention, reason = self._govern_action(raw_action, observation)
        self.last_decision = GovernanceDecision(
            raw_action=raw_action,
            governed_action=governed_action,
            intervention=intervention,
            reason=reason,
        )
        return governed_action

    def _govern_action(
        self,
        action: SelectionAction,
        observation: Observation,
    ) -> tuple[SelectionAction, str, str]:
        offer_ids = {offer.offer_id for offer in observation.offers}

        if action.action_type is ActionType.ASK_CLARIFICATION:
            if (
                action.slot not in observation.available_clarification_slots
                or clarification_slot_is_revealed(action.slot or "", observation.revealed_context)
            ):
                fallback = self._first_missing_clarification(observation)
                if fallback is not None:
                    return (
                        SelectionAction.ask_clarification(
                            fallback,
                            explanation="governance rewrite: ask an unrevealed valid slot",
                        ),
                        "rewrite",
                        "invalid_or_redundant_clarification",
                    )
                return (
                    SelectionAction.escalate("governance escalation: no valid clarification left"),
                    "escalate",
                    "invalid_clarification_without_fallback",
                )
            return action, "pass", "valid_clarification"

        if action.action_type is ActionType.COMPARE_OPTIONS:
            if action.offer_id in offer_ids and action.comparison_offer_id in offer_ids:
                return action, "pass", "valid_comparison"
            return (
                SelectionAction.escalate("governance escalation: compare references unknown offer"),
                "escalate",
                "unknown_offer_in_comparison",
            )

        if action.action_type not in {ActionType.RECOMMEND_OPTION, ActionType.COMMIT_SELECTION}:
            return action, "pass", "non_selection_action"

        if action.offer_id not in offer_ids:
            return (
                SelectionAction.escalate("governance escalation: selection references unknown offer"),
                "escalate",
                "unknown_offer_in_selection",
            )

        missing_slot = self._required_missing_slot(observation)
        if missing_slot is not None:
            return (
                SelectionAction.ask_clarification(
                    missing_slot,
                    explanation="governance rewrite: recover required constraint evidence",
                ),
                "rewrite",
                f"missing_required_evidence:{missing_slot}",
            )

        target_offer = self._lookup_offer(observation, action.offer_id)
        if target_offer is None:
            return (
                SelectionAction.escalate("governance escalation: target offer unavailable"),
                "escalate",
                "missing_target_offer",
            )

        if not self._is_known_feasible(observation, target_offer):
            fallback_offer = self._best_known_feasible_offer(observation)
            if fallback_offer is None:
                return (
                    SelectionAction.escalate("governance escalation: no known feasible offer"),
                    "escalate",
                    "no_known_feasible_offer",
                )
            if self._has_prior_recommendation(observation, fallback_offer.offer_id):
                return (
                    SelectionAction.commit_selection(
                        fallback_offer.offer_id,
                        explanation="governance rewrite: commit prior recommended feasible offer",
                    ),
                    "rewrite",
                    "target_offer_violates_known_constraints",
                )
            return (
                SelectionAction.recommend_option(
                    fallback_offer.offer_id,
                    explanation="governance rewrite: recommend feasible fallback offer",
                ),
                "rewrite",
                "target_offer_violates_known_constraints",
            )

        if action.action_type is ActionType.COMMIT_SELECTION and not self._has_prior_recommendation(
            observation,
            action.offer_id,
        ):
            return (
                SelectionAction.recommend_option(
                    action.offer_id or "",
                    explanation="governance rewrite: recommend before commit",
                ),
                "rewrite",
                "commit_requires_prior_recommendation",
            )

        return action, "pass", "selection_action_allowed"

    def _required_missing_slot(self, observation: Observation) -> str | None:
        if self.require_budget_before_selection:
            if (
                CLARIFICATION_BUDGET_MAX in observation.available_clarification_slots
                and not clarification_slot_is_revealed(
                    CLARIFICATION_BUDGET_MAX,
                    observation.revealed_context,
                )
            ):
                return CLARIFICATION_BUDGET_MAX

        if self.require_relevant_must_have_before_selection:
            for slot in observation.available_clarification_slots:
                if not slot.startswith(CLARIFICATION_MUST_HAVE_PREFIX):
                    continue
                if clarification_slot_is_revealed(slot, observation.revealed_context):
                    continue
                feature_name = slot[len(CLARIFICATION_MUST_HAVE_PREFIX) :]
                values = {offer.features.get(feature_name) for offer in observation.offers}
                if len(values) > 1:
                    return slot
        return None

    def _first_missing_clarification(self, observation: Observation) -> str | None:
        preferred_prefixes = (
            CLARIFICATION_BUDGET_MAX,
            CLARIFICATION_MUST_HAVE_PREFIX,
            CLARIFICATION_PREFERENCE_PREFIX,
        )
        for prefix in preferred_prefixes:
            for slot in observation.available_clarification_slots:
                if prefix != CLARIFICATION_BUDGET_MAX and not slot.startswith(prefix):
                    continue
                if prefix == CLARIFICATION_BUDGET_MAX and slot != CLARIFICATION_BUDGET_MAX:
                    continue
                if not clarification_slot_is_revealed(slot, observation.revealed_context):
                    return slot
        return None

    def _best_known_feasible_offer(self, observation: Observation) -> Offer | None:
        feasible_offers = [
            offer for offer in observation.offers if self._is_known_feasible(observation, offer)
        ]
        if not feasible_offers:
            return None
        return max(feasible_offers, key=lambda offer: self._score_offer(observation, offer))

    def _score_offer(self, observation: Observation, offer: Offer) -> float:
        score = 0.0
        for slot, weight in observation.revealed_context["preference_weights"].items():
            score += float(weight) * offer.attribute_values.get(slot, 0.0)
        score -= self.price_penalty * offer.price
        return score

    def _is_known_feasible(self, observation: Observation, offer: Offer) -> bool:
        if offer.category != observation.revealed_context["category"]:
            return False
        budget_max = observation.revealed_context["budget_max"]
        if budget_max is not None and offer.price > float(budget_max):
            return False
        for slot, expected in observation.revealed_context["must_have"].items():
            actual = offer.features.get(slot)
            if isinstance(expected, bool):
                if expected is False:
                    continue
                if bool(actual) is not expected:
                    return False
            elif isinstance(expected, (int, float)):
                candidate = offer.features.get(slot, offer.attribute_values.get(slot))
                if not isinstance(candidate, (int, float)) or float(candidate) < float(expected):
                    return False
            elif actual != expected:
                return False
        return True

    @staticmethod
    def _lookup_offer(observation: Observation, offer_id: str | None) -> Offer | None:
        if offer_id is None:
            return None
        for offer in observation.offers:
            if offer.offer_id == offer_id:
                return offer
        return None

    @staticmethod
    def _has_prior_recommendation(observation: Observation, offer_id: str | None) -> bool:
        if offer_id is None:
            return False
        for entry in observation.history:
            if (
                entry.action.action_type is ActionType.RECOMMEND_OPTION
                and entry.action.offer_id == offer_id
            ):
                return True
        return False
