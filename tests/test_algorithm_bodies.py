"""Tests for concrete role/gate/attribution algorithm bodies."""

from __future__ import annotations

import unittest

from aimai_ocl.attribution import compute_shapley, fallback_policy
from aimai_ocl.control import apply_control
from aimai_ocl.coordinator import Coordinator
from aimai_ocl.schemas import ActionIntent, ActionRole, ConstraintSeverity, RawAction


class RoleStateMachineTests(unittest.TestCase):
    """Coverage for state-machine role decomposition behavior."""

    def setUp(self) -> None:
        self.coordinator = Coordinator(mode="state_machine")

    def test_state_machine_routes_question_to_expert(self) -> None:
        """Input: recommendation-like buyer question.

        Expected output: recommendation phase with expert decision role.
        """
        plan = self.coordinator.plan_turn(
            round_id=1,
            buyer_text="Which material is better?",
            seller_actor_id="seller",
            max_rounds=10,
        )
        self.assertEqual(ActionRole.EXPERT, plan.decision_role)
        self.assertEqual("recommendation", plan.metadata["phase"])

    def test_state_machine_routes_deadline_to_platform(self) -> None:
        """Input: near-deadline round.

        Expected output: closing phase with platform decision role.
        """
        plan = self.coordinator.plan_turn(
            round_id=9,
            buyer_text="let's continue",
            seller_actor_id="seller",
            max_rounds=10,
        )
        self.assertEqual(ActionRole.PLATFORM, plan.decision_role)
        self.assertEqual("closing", plan.metadata["phase"])

    def test_state_machine_routes_dispute_to_platform_escalation(self) -> None:
        """Input: dispute keyword in buyer text.

        Expected output: escalation phase with platform decision role.
        """
        plan = self.coordinator.plan_turn(
            round_id=3,
            buyer_text="This is unacceptable, I want a manager",
            seller_actor_id="seller",
            max_rounds=10,
        )
        self.assertEqual(ActionRole.PLATFORM, plan.decision_role)
        self.assertEqual("escalation", plan.metadata["phase"])


class ControlRiskGateTests(unittest.TestCase):
    """Coverage for control pipeline risk gating behavior."""

    def test_low_risk_action_is_approved(self) -> None:
        """Input: benign info request.

        Expected output: executable action is approved.
        """
        raw = RawAction(
            actor_id="seller",
            actor_role=ActionRole.SELLER,
            intent=ActionIntent.REQUEST_INFO,
            utterance="Here are the product details.",
        )
        result = apply_control(raw, state={"buyer_max_price": 120.0, "seller_min_price": 90.0})
        self.assertTrue(result.executable.approved)

    def test_high_risk_action_blocks_and_escalates(self) -> None:
        """Input: payment-sensitive text with risky intent.

        Expected output: executable action is blocked and marked for escalation.
        """
        raw = RawAction(
            actor_id="seller",
            actor_role=ActionRole.SELLER,
            intent=ActionIntent.TOOL_CALL,
            utterance="Send me your card account and CVV now.",
        )
        result = apply_control(raw, state={"buyer_max_price": 120.0, "seller_min_price": 90.0})
        self.assertFalse(result.executable.approved)
        self.assertTrue(result.executable.requires_escalation)


class AttributionTests(unittest.TestCase):
    """Coverage for Shapley attribution computation."""

    def test_fallback_is_role_specific_and_deterministic(self) -> None:
        """Input: same role/state twice.

        Expected output: identical deterministic text and role-specific prefix.
        """
        state = {"buyer_max_price": 120.0, "seller_min_price": 90.0, "product_name": "Jacket"}
        a1 = fallback_policy("expert", state)
        a2 = fallback_policy("expert", state)
        self.assertEqual(a1, a2)
        self.assertTrue(a1.startswith("[expert]"))

    def test_shapley_handles_sparse_subset_values(self) -> None:
        """Input: sparse coalition values (not full power set).

        Expected output:
        - returns phi/w for all controlled roles
        - weights sum to <= 1 (allowing zeroed negatives)
        """
        sparse_values = {
            frozenset(): 0.0,
            frozenset({"seller"}): 1.0,
            frozenset({"platform"}): 0.5,
            frozenset({"expert"}): 0.2,
            frozenset({"seller", "platform", "expert"}): 2.0,
        }
        result = compute_shapley(sparse_values)
        self.assertEqual({"platform", "seller", "expert"}, set(result["phi"]))
        self.assertEqual({"platform", "seller", "expert"}, set(result["w"]))
        self.assertLessEqual(sum(result["w"].values()), 1.0 + 1e-9)


if __name__ == "__main__":
    unittest.main()
