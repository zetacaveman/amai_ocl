"""Unit tests for escalation/replan resolution."""

from __future__ import annotations

import unittest

from aimai_ocl.control import resolve_escalation
from aimai_ocl.schemas import (
    ActionIntent,
    ActionRole,
    AuditEventType,
    ControlDecision,
    ExecutableAction,
    RawAction,
)


class EscalationTests(unittest.TestCase):
    """Coverage for escalation and deterministic replan logic."""

    def setUp(self) -> None:
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
        - final text is passthrough utterance
        - no escalation audit events
        """
        executable = ExecutableAction(
            actor_id="seller",
            actor_role=ActionRole.SELLER,
            approved=True,
            decision=ControlDecision.APPROVE,
            final_text="I can do $130.",
            intent=ActionIntent.NEGOTIATE_PRICE,
            final_price=130.0,
        )
        final_text, events = resolve_escalation(
            raw=self.raw,
            executable=executable,
            state={"buyer_max_price": 120.0, "seller_min_price": 90.0},
        )
        self.assertEqual("I can do $130.", final_text)
        self.assertEqual(0, len(events))

    def test_blocked_price_violation_produces_replan(self) -> None:
        """Input: blocked budget violation with feasible price interval.

        Expected output:
        - replanned text with clamped price
        - escalation + replan events are both recorded
        """
        executable = ExecutableAction(
            actor_id="seller",
            actor_role=ActionRole.SELLER,
            approved=False,
            decision=ControlDecision.BLOCK,
            final_text="",
            intent=ActionIntent.NEGOTIATE_PRICE,
            requires_escalation=True,
        )
        final_text, events = resolve_escalation(
            raw=self.raw,
            executable=executable,
            state={"buyer_max_price": 120.0, "seller_min_price": 90.0},
            enable_replan=True,
        )
        self.assertEqual("I can revise to $120.00.", final_text)
        event_types = [event.event_type for event in events]
        self.assertIn(AuditEventType.ESCALATION_TRIGGERED, event_types)
        self.assertIn(AuditEventType.REPLAN_APPLIED, event_types)

    def test_infeasible_bounds_fall_back_to_human_handoff(self) -> None:
        """Input: blocked floor/budget conflict with infeasible interval.

        Expected output:
        - no final text (human handoff)
        - escalation event emitted
        """
        executable = ExecutableAction(
            actor_id="seller",
            actor_role=ActionRole.SELLER,
            approved=False,
            decision=ControlDecision.BLOCK,
            final_text="",
            intent=ActionIntent.NEGOTIATE_PRICE,
            requires_escalation=True,
        )
        final_text, events = resolve_escalation(
            raw=self.raw,
            executable=executable,
            state={"buyer_max_price": 80.0, "seller_min_price": 90.0},
            enable_replan=True,
        )
        self.assertIsNone(final_text)
        self.assertEqual(1, len(events))
        self.assertEqual(AuditEventType.ESCALATION_TRIGGERED, events[0].event_type)


if __name__ == "__main__":
    unittest.main()
