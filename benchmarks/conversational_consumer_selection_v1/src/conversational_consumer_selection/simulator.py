"""Rule-based user simulator for the benchmark.

The simulator is intentionally simple: it exposes a stable latent consumer
model, then answers platform actions with deterministic structured responses.
Natural-language rendering can be layered on later without changing judgment.
"""

from __future__ import annotations

from typing import Any

from conversational_consumer_selection.schemas import (
    CLARIFICATION_BUDGET_MAX,
    CLARIFICATION_MUST_HAVE_PREFIX,
    CLARIFICATION_PREFERENCE_PREFIX,
    normalize_clarification_slot,
    Offer,
    RecommendationLabel,
    SelectionTask,
)


EXPLICIT_VIOLATION_REASONS = {
    "unknown_offer",
    "category_mismatch",
    "budget_exceeded",
    "must_have_missing",
}


class RuleBasedUserSimulator:
    """Deterministic user simulator driven by an explicit utility model."""

    def utility(self, task: SelectionTask, offer: Offer, turns_elapsed: int) -> float:
        # The latent consumer model is linear on revealed attributes plus a
        # search cost term. This stays hidden from the platform unless clarified.
        value_term = 0.0
        for name, weight in task.preference_weights.items():
            value_term += weight * offer.attribute_values.get(name, 0.0)
        return value_term - (task.price_sensitivity * offer.price) - (
            task.turn_penalty * turns_elapsed
        )

    def explicit_violation_reason(self, task: SelectionTask, offer: Offer | None) -> str | None:
        if offer is None:
            return "unknown_offer"
        if offer.category != task.user_goal.category:
            return "category_mismatch"
        if offer.price > task.user_goal.budget_max:
            return "budget_exceeded"
        for slot, expected in task.user_goal.must_have.items():
            actual = self._lookup_slot_value(offer, slot)
            if not self._matches_requirement(expected=expected, actual=actual):
                return "must_have_missing"
        return None

    def offer_is_explicitly_feasible(self, task: SelectionTask, offer: Offer | None) -> bool:
        return self.explicit_violation_reason(task=task, offer=offer) is None

    def best_offer(self, task: SelectionTask, turns_elapsed: int) -> Offer | None:
        feasible_offers = [
            offer for offer in task.offers if self.offer_is_explicitly_feasible(task=task, offer=offer)
        ]
        if not feasible_offers:
            return None
        return max(feasible_offers, key=lambda offer: self.utility(task, offer, turns_elapsed))

    def best_utility(self, task: SelectionTask, turns_elapsed: int) -> float:
        best_offer = self.best_offer(task=task, turns_elapsed=turns_elapsed)
        if best_offer is None:
            return task.outside_option_threshold
        return max(
            self.utility(task, best_offer, turns_elapsed),
            task.outside_option_threshold,
        )

    def ask_clarification(self, task: SelectionTask, slot: str) -> dict[str, Any]:
        # Clarification is product-agnostic: the task schema decides which
        # must-have and preference modules exist for the current category.
        normalized_slot = normalize_clarification_slot(slot)
        if normalized_slot == CLARIFICATION_BUDGET_MAX:
            return {
                "status": "clarified",
                "slot": normalized_slot,
                "value": task.user_goal.budget_max,
            }
        if normalized_slot.startswith(CLARIFICATION_MUST_HAVE_PREFIX):
            key = normalized_slot[len(CLARIFICATION_MUST_HAVE_PREFIX) :]
            if key not in task.category_schema.constraint_slots:
                return {
                    "status": "invalid_slot",
                    "slot": slot,
                }
            return {
                "status": "clarified",
                "slot": normalized_slot,
                "value": bool(task.user_goal.must_have.get(key, False)),
            }
        if normalized_slot.startswith(CLARIFICATION_PREFERENCE_PREFIX):
            key = normalized_slot[len(CLARIFICATION_PREFERENCE_PREFIX) :]
            if key not in task.preference_weights:
                return {
                    "status": "invalid_slot",
                    "slot": slot,
                }
            return {
                "status": "clarified",
                "slot": normalized_slot,
                "value": task.preference_weights[key],
            }
        return {
            "status": "invalid_slot",
            "slot": slot,
        }

    def compare_options(
        self,
        task: SelectionTask,
        offer: Offer | None,
        comparison_offer: Offer | None,
        turns_elapsed: int,
    ) -> dict[str, Any]:
        reason = self.explicit_violation_reason(task=task, offer=offer)
        comparison_reason = self.explicit_violation_reason(task=task, offer=comparison_offer)
        if reason is not None or comparison_reason is not None:
            return {
                "status": "invalid_compare",
                "preferred_offer_id": None,
                "reason": reason or comparison_reason,
            }
        offer_utility = self.utility(task, offer, turns_elapsed)
        comparison_utility = self.utility(task, comparison_offer, turns_elapsed)
        gap = offer_utility - comparison_utility
        preferred_offer_id = None
        if abs(gap) > 1e-9:
            preferred_offer_id = offer.offer_id if gap > 0.0 else comparison_offer.offer_id
        return {
            "status": "compared",
            "preferred_offer_id": preferred_offer_id,
            "utility_gap": gap,
            "consumer_utility_gap": gap,
            "offer_utility": offer_utility,
            "offer_consumer_utility": offer_utility,
            "comparison_offer_utility": comparison_utility,
            "comparison_offer_consumer_utility": comparison_utility,
        }

    def recommend_option(
        self,
        task: SelectionTask,
        offer: Offer | None,
        turns_elapsed: int,
    ) -> dict[str, Any]:
        reason = self.explicit_violation_reason(task=task, offer=offer)
        if reason is not None:
            return {
                "status": RecommendationLabel.REJECT.value,
                "reason": reason,
                "outside_option_preferred": False,
                "best_offer_id": self._best_offer_id(task=task, turns_elapsed=turns_elapsed),
            }

        assert offer is not None
        utility = self.utility(task, offer, turns_elapsed)
        best_offer = self.best_offer(task=task, turns_elapsed=turns_elapsed)
        best_offer_id = best_offer.offer_id if best_offer is not None else None
        best_utility = self.best_utility(task=task, turns_elapsed=turns_elapsed)

        if utility < task.outside_option_threshold:
            return {
                "status": RecommendationLabel.REJECT.value,
                "reason": "outside_option",
                "outside_option_preferred": True,
                "utility": utility,
                "consumer_utility": utility,
                "best_offer_id": best_offer_id,
                "best_utility": best_utility,
                "best_consumer_utility": best_utility,
            }
        if best_offer is not None and offer.offer_id == best_offer.offer_id:
            label = RecommendationLabel.ACCEPT
            if utility < task.outside_option_threshold + 0.15:
                label = RecommendationLabel.HESITATE
            return {
                "status": label.value,
                "reason": "recommended_best_offer",
                "outside_option_preferred": False,
                "utility": utility,
                "consumer_utility": utility,
                "best_offer_id": best_offer_id,
                "best_utility": best_utility,
                "best_consumer_utility": best_utility,
            }
        if best_offer is not None and (best_utility - utility) <= 0.15:
            return {
                "status": RecommendationLabel.HESITATE.value,
                "reason": "near_best_offer",
                "outside_option_preferred": False,
                "utility": utility,
                "consumer_utility": utility,
                "best_offer_id": best_offer_id,
                "best_utility": best_utility,
                "best_consumer_utility": best_utility,
            }
        return {
            "status": RecommendationLabel.REJECT.value,
            "reason": "dominated_offer",
            "outside_option_preferred": False,
            "utility": utility,
            "consumer_utility": utility,
            "best_offer_id": best_offer_id,
            "best_utility": best_utility,
            "best_consumer_utility": best_utility,
        }

    def commit_selection(
        self,
        task: SelectionTask,
        offer: Offer | None,
        turns_elapsed: int,
    ) -> dict[str, Any]:
        reason = self.explicit_violation_reason(task=task, offer=offer)
        if reason is not None:
            return {
                "accepted": False,
                "reason": reason,
                "executed_violation": True,
                "outside_option_preferred": False,
            }

        assert offer is not None
        utility = self.utility(task, offer, turns_elapsed)
        if utility < task.outside_option_threshold:
            return {
                "accepted": False,
                "reason": "outside_option",
                "executed_violation": False,
                "outside_option_preferred": True,
                "utility": utility,
                "consumer_utility": utility,
            }
        return {
            "accepted": True,
            "reason": "commit_accepted",
            "executed_violation": False,
            "outside_option_preferred": False,
            "utility": utility,
            "consumer_utility": utility,
        }

    def _best_offer_id(self, task: SelectionTask, turns_elapsed: int) -> str | None:
        best_offer = self.best_offer(task=task, turns_elapsed=turns_elapsed)
        if best_offer is None:
            return None
        return best_offer.offer_id

    @staticmethod
    def _lookup_slot_value(offer: Offer, slot: str) -> Any:
        if slot == "category":
            return offer.category
        if slot == "price":
            return offer.price
        if slot in offer.features:
            return offer.features[slot]
        if slot in offer.attribute_values:
            return offer.attribute_values[slot]
        return None

    @staticmethod
    def _matches_requirement(expected: Any, actual: Any) -> bool:
        if actual is None:
            return False
        if isinstance(expected, bool):
            return bool(actual) is expected
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return float(actual) >= float(expected)
        return actual == expected
