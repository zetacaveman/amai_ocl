"""Unit tests for escalation/replan policy manager.

中文翻译：用于 escalation/replan policy manager 的单元测试。"""

from __future__ import annotations

import unittest

from aimai_ocl.controllers.escalation_manager import EscalationManager
from aimai_ocl.schemas.actions import ActionIntent, ActionRole, RawAction
from aimai_ocl.schemas.audit import AuditEventType
from aimai_ocl.schemas.constraints import ViolationType


class EscalationManagerTests(unittest.TestCase):
    """Coverage for step-7 escalation and deterministic replan logic.

中文翻译：step-7 escalation and deterministic replan logic 的覆盖测试。"""

    def setUp(self) -> None:
        """Input: none.

        Output: fresh default escalation manager.
        

        中文翻译：输入：none。"""
        self.manager = EscalationManager()
        self.raw = RawAction(
            actor_id="seller",
            actor_role=ActionRole.SELLER,
            intent=ActionIntent.NEGOTIATE_PRICE,
            utterance="I can do $130.",
            proposed_price=130.0,
        )

    def test_approved_low_risk_action_executes_directly(self) -> None:
        """Input: approved action without escalation flags.

        Expected output:
        - strategy is direct_execute
        - final text is passthrough utterance
        - no escalation audit events
        

        中文翻译：输入：approved action without escalation flags。"""
        outcome = self.manager.resolve(
            round_id=1,
            actor_id="seller",
            raw_action=self.raw,
            approved=True,
            requires_confirmation=False,
            requires_escalation=False,
            violations=[],
            state={"buyer_max_price": 120.0, "seller_min_price": 90.0},
        )
        self.assertEqual("direct_execute", outcome.strategy)
        self.assertEqual("I can do $130.", outcome.final_text)
        self.assertEqual(0, len(outcome.audit_events))

    def test_blocked_price_violation_produces_replan(self) -> None:
        """Input: blocked budget violation with feasible price interval.

        Expected output:
        - strategy is replan_and_retry
        - replan text is emitted
        - escalation + replan events are both recorded
        

        中文翻译：输入：blocked budget violation with feasible price interval。"""
        outcome = self.manager.resolve(
            round_id=2,
            actor_id="seller",
            raw_action=self.raw,
            approved=False,
            requires_confirmation=False,
            requires_escalation=False,
            violations=[ViolationType.BUDGET_EXCEEDED.value],
            state={"buyer_max_price": 120.0, "seller_min_price": 90.0},
            allow_replan=True,
        )
        self.assertEqual("replan_and_retry", outcome.strategy)
        self.assertEqual("I can revise to $120.00.", outcome.replan_text)
        event_types = [event.event_type for event in outcome.audit_events]
        self.assertIn(AuditEventType.ESCALATION_TRIGGERED, event_types)
        self.assertIn(AuditEventType.REPLAN_APPLIED, event_types)

    def test_infeasible_bounds_fall_back_to_human_handoff(self) -> None:
        """Input: blocked floor/budget conflict with infeasible interval.

        Expected output:
        - strategy is human_handoff
        - no replan text
        - escalation event contains strategy metadata
        

        中文翻译：输入：blocked floor/budget conflict with infeasible interval。"""
        outcome = self.manager.resolve(
            round_id=3,
            actor_id="seller",
            raw_action=self.raw,
            approved=False,
            requires_confirmation=False,
            requires_escalation=False,
            violations=[ViolationType.BUDGET_EXCEEDED.value],
            state={"buyer_max_price": 80.0, "seller_min_price": 90.0},
            allow_replan=True,
        )
        self.assertEqual("human_handoff", outcome.strategy)
        self.assertIsNone(outcome.replan_text)
        self.assertTrue(outcome.requires_human_handoff)
        self.assertEqual(1, len(outcome.audit_events))
        self.assertEqual(
            "human_handoff",
            outcome.audit_events[0].metadata.get("strategy"),
        )


if __name__ == "__main__":
    unittest.main()
