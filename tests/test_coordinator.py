"""Unit tests for minimal OCL coordinator role assignment.

中文翻译：用于 minimal OCL coordinator role assignment 的单元测试。"""

from __future__ import annotations

import unittest

from aimai_ocl.controllers.coordinator import Coordinator, SellerOnlyCoordinator
from aimai_ocl.schemas.actions import ActionRole
from aimai_ocl.schemas.audit import AuditEventType


class CoordinatorTests(unittest.TestCase):
    """Coverage for deterministic turn-role planning rules.

中文翻译：deterministic turn-role planning rules 的覆盖测试。"""

    def setUp(self) -> None:
        """Input: none.

        Output: fresh coordinator instance per test.
        

        中文翻译：输入：none。"""
        self.coordinator = Coordinator()

    def test_default_turn_is_seller_led(self) -> None:
        """Input: plain buyer text away from deadline.

        Expected output:
        - decision role is seller
        - execution role is seller
        - escalation role is platform
        

        中文翻译：输入：plain buyer text away from deadline。"""
        plan = self.coordinator.plan_turn(
            round_id=2,
            buyer_text="ok let's continue",
            seller_actor_id="seller",
            max_rounds=10,
        )
        self.assertEqual(ActionRole.SELLER, plan.decision_role)
        self.assertEqual(ActionRole.SELLER, plan.execution_role)
        self.assertEqual(ActionRole.PLATFORM, plan.escalation_role)

    def test_question_routes_decision_to_expert(self) -> None:
        """Input: buyer asks question that needs clarification.

        Expected output: decision role is expert (A_e).
        

        中文翻译：输入：buyer asks question that needs clarification。"""
        plan = self.coordinator.plan_turn(
            round_id=1,
            buyer_text="Which material is better?",
            seller_actor_id="seller",
            max_rounds=10,
        )
        self.assertEqual(ActionRole.EXPERT, plan.decision_role)

    def test_deadline_routes_decision_to_platform(self) -> None:
        """Input: round near max_rounds boundary.

        Expected output: decision role is platform (A_p).
        

        中文翻译：输入：round near max_rounds boundary。"""
        plan = self.coordinator.plan_turn(
            round_id=9,
            buyer_text="still thinking",
            seller_actor_id="seller",
            max_rounds=10,
        )
        self.assertEqual(ActionRole.PLATFORM, plan.decision_role)

    def test_build_audit_event_contains_role_assignments(self) -> None:
        """Input: one coordination plan.

        Expected output:
        - event type is coordination_planned
        - metadata includes decision/execution/escalation roles
        

        中文翻译：输入：one coordination plan。"""
        plan = self.coordinator.plan_turn(
            round_id=0,
            buyer_text="please compare options",
            seller_actor_id="seller",
            max_rounds=10,
        )
        event = self.coordinator.build_audit_event(plan)
        self.assertEqual(AuditEventType.COORDINATION_PLANNED, event.event_type)
        self.assertEqual(plan.decision_role.value, event.metadata["decision_role"])
        self.assertEqual(plan.execution_role.value, event.metadata["execution_role"])
        self.assertEqual(plan.escalation_role.value, event.metadata["escalation_role"])

    def test_seller_only_coordinator_forces_seller_decision(self) -> None:
        """Input: buyer question and deadline round.

        Expected output:
        - decision role still stays seller in seller-only ablation policy
        

        中文翻译：输入：buyer question and deadline round。"""
        coordinator = SellerOnlyCoordinator()
        plan = coordinator.plan_turn(
            round_id=9,
            buyer_text="Which one do you recommend?",
            seller_actor_id="seller",
            max_rounds=10,
        )
        self.assertEqual(ActionRole.SELLER, plan.decision_role)


if __name__ == "__main__":
    unittest.main()
