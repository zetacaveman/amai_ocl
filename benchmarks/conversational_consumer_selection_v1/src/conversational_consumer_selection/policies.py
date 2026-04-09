"""Model-agnostic policy contract and simple standalone baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from conversational_consumer_selection.schemas import (
    clarification_slot_is_revealed,
    Observation,
    Offer,
    SelectionAction,
)


class Policy(Protocol):
    """Minimal model-agnostic policy protocol used by the runner.

    Any decision backend can implement this contract:

    - rule-based state machine
    - recommender / ranker
    - classifier-based controller
    - RL policy
    - prompt-based LLM
    - governed LLM stack
    """

    def act(self, observation: Observation) -> SelectionAction:
        """Return the next structured action."""


@dataclass
class GreedySelectionPolicy:
    """Clarify missing preference slots, then commit the current best feasible offer."""

    clarify_missing_preferences: bool = True
    price_penalty: float = 0.01

    def act(self, observation: Observation) -> SelectionAction:
        if self.clarify_missing_preferences:
            for slot in observation.available_clarification_slots:
                if not clarification_slot_is_revealed(slot, observation.revealed_context):
                    return SelectionAction.ask_clarification(slot)

        best_offer = self._best_feasible_offer(observation)
        if best_offer is None:
            return SelectionAction.escalate("no feasible offer under explicit constraints")
        return SelectionAction.commit_selection(best_offer.offer_id)

    def _best_feasible_offer(self, observation: Observation) -> Offer | None:
        feasible_offers = [
            offer for offer in observation.offers if self._is_explicitly_feasible(observation, offer)
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

    def _is_explicitly_feasible(self, observation: Observation, offer: Offer) -> bool:
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
