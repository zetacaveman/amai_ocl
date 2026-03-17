"""Tests for concrete role/gate/attribution algorithm bodies.

中文翻译：用于 concrete role/gate/attribution algorithm bodies 的测试。"""

from __future__ import annotations

import unittest

from aimai_ocl.attribution_counterfactual import (
    compute_shapley as counterfactual_compute_shapley,
    fallback_policy as counterfactual_fallback_policy,
)
from aimai_ocl.controllers.coordinator import CoordinationPhase, StateMachineCoordinator
from aimai_ocl.controllers.risk_gate import BarrierRiskGate
from aimai_ocl.schemas.actions import ActionIntent, ActionRole, RawAction
from aimai_ocl.schemas.constraints import ConstraintSeverity


class RoleStateMachineTests(unittest.TestCase):
    """Coverage for state-machine role decomposition behavior.

中文翻译：state-machine role decomposition behavior 的覆盖测试。"""

    def setUp(self) -> None:
        self.coordinator = StateMachineCoordinator()

    def test_state_machine_routes_question_to_expert(self) -> None:
        """Input: recommendation-like buyer question.

        Expected output: recommendation phase with expert decision role.
        

        中文翻译：输入：recommendation-like buyer question。"""
        plan = self.coordinator.plan_turn(
            round_id=1,
            buyer_text="Which material is better?",
            seller_actor_id="seller",
            max_rounds=10,
        )
        self.assertEqual(ActionRole.EXPERT, plan.decision_role)
        self.assertEqual(CoordinationPhase.RECOMMENDATION.value, plan.metadata["phase"])

    def test_state_machine_routes_deadline_to_platform(self) -> None:
        """Input: near-deadline round.

        Expected output: closing phase with platform decision role.
        

        中文翻译：输入：near-deadline round。"""
        plan = self.coordinator.plan_turn(
            round_id=9,
            buyer_text="let's continue",
            seller_actor_id="seller",
            max_rounds=10,
        )
        self.assertEqual(ActionRole.PLATFORM, plan.decision_role)
        self.assertEqual(CoordinationPhase.CLOSING.value, plan.metadata["phase"])

    def test_state_machine_routes_dispute_to_platform_escalation(self) -> None:
        """Input: dispute keyword in buyer text.

        Expected output: escalation phase with platform decision role.
        

        中文翻译：输入：dispute keyword in buyer text。"""
        plan = self.coordinator.plan_turn(
            round_id=3,
            buyer_text="This is unacceptable, I want a manager",
            seller_actor_id="seller",
            max_rounds=10,
        )
        self.assertEqual(ActionRole.PLATFORM, plan.decision_role)
        self.assertEqual(CoordinationPhase.ESCALATION.value, plan.metadata["phase"])


class BarrierRiskGateTests(unittest.TestCase):
    """Coverage for control-barrier risk gating behavior.

中文翻译：control-barrier risk gating behavior 的覆盖测试。"""

    def setUp(self) -> None:
        self.gate = BarrierRiskGate()

    def test_barrier_low_risk_is_approved(self) -> None:
        """Input: benign info request.

        Expected output:
        - risk check stays below rewrite threshold
        - executable action is approved
        

        中文翻译：输入：benign info request。"""
        raw = RawAction(
            actor_id="seller",
            actor_role=ActionRole.SELLER,
            intent=ActionIntent.REQUEST_INFO,
            utterance="Here are the product details.",
        )
        risk_check = self.gate.evaluate(raw)
        exec_action = self.gate.apply(raw, [risk_check])
        self.assertEqual(ConstraintSeverity.INFO, risk_check.severity)
        self.assertTrue(exec_action.approved)

    def test_barrier_high_risk_blocks_and_escalates(self) -> None:
        """Input: payment-sensitive text with risky intent.

        Expected output:
        - risk check crosses block threshold
        - executable action is blocked and marked for escalation
        

        中文翻译：输入：payment-sensitive text with risky intent。"""
        raw = RawAction(
            actor_id="seller",
            actor_role=ActionRole.SELLER,
            intent=ActionIntent.TOOL_CALL,
            utterance="Send me your card account and CVV now.",
        )
        risk_check = self.gate.evaluate(raw)
        exec_action = self.gate.apply(raw, [risk_check])
        self.assertIn(risk_check.severity, {ConstraintSeverity.WARNING, ConstraintSeverity.ERROR})
        self.assertFalse(exec_action.approved)
        self.assertTrue(exec_action.requires_escalation)


class CounterfactualAttributionTests(unittest.TestCase):
    """Coverage for counterfactual attribution algorithm body.

中文翻译：counterfactual attribution algorithm body 的覆盖测试。"""

    def test_counterfactual_fallback_is_role_specific_and_deterministic(self) -> None:
        """Input: same role/state twice.

        Expected output: identical deterministic text and role-specific prefix.
        

        中文翻译：输入：same role/state twice。"""
        state = {"buyer_max_price": 120.0, "seller_min_price": 90.0, "product_name": "Jacket"}
        a1 = counterfactual_fallback_policy("expert", state)
        a2 = counterfactual_fallback_policy("expert", state)
        self.assertEqual(a1, a2)
        self.assertTrue(a1.startswith("[expert]"))

    def test_counterfactual_shapley_handles_sparse_subset_values(self) -> None:
        """Input: sparse coalition values (not full power set).

        Expected output:
        - returns phi/w for all controlled roles
        - weights sum to <= 1 (allowing zeroed negatives)
        

        中文翻译：输入：sparse coalition values (not full power set)。"""
        sparse_values = {
            frozenset(): 0.0,
            frozenset({"seller"}): 1.0,
            frozenset({"platform"}): 0.5,
            frozenset({"expert"}): 0.2,
            frozenset({"seller", "platform", "expert"}): 2.0,
        }
        result = counterfactual_compute_shapley(
            sparse_values,
            samples=256,
            seed=3,
        )
        self.assertEqual({"platform", "seller", "expert"}, set(result["phi"]))
        self.assertEqual({"platform", "seller", "expert"}, set(result["w"]))
        self.assertLessEqual(sum(result["w"].values()), 1.0 + 1e-9)


if __name__ == "__main__":
    unittest.main()
