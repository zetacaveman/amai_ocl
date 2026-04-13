"""Unit tests for deterministic control pipeline constraint checks."""

from __future__ import annotations

import unittest

from aimai_ocl.control import check_price_bounds, check_price_format, check_privacy
from aimai_ocl.schemas import (
    ActionIntent,
    ActionRole,
    ConstraintCheck,
    ConstraintSeverity,
    RawAction,
    ViolationType,
)


def _index_by_id(checks: list[ConstraintCheck]) -> dict[str, ConstraintCheck]:
    """Index checks by ``constraint_id`` for concise assertions."""
    return {check.constraint_id: check for check in checks}


class ConstraintCheckTests(unittest.TestCase):
    """Rule-level coverage for price format, privacy, and price bounds."""

    def test_price_format_missing_price_fails(self) -> None:
        """Input: negotiate intent without numeric price.

        Expected output: ``price_format`` fails with ``FORMAT_INVALID``.
        """
        raw = RawAction(
            actor_id="seller",
            actor_role=ActionRole.SELLER,
            intent=ActionIntent.NEGOTIATE_PRICE,
            utterance="Let's discuss first.",
            proposed_price=None,
        )
        check = check_price_format(raw)
        self.assertFalse(check.passed)
        self.assertEqual(check.severity, ConstraintSeverity.ERROR)
        self.assertEqual(check.violation_type, ViolationType.FORMAT_INVALID)

    def test_price_format_valid_price_passes(self) -> None:
        """Input: negotiate intent with valid positive price.

        Expected output: ``price_format`` check passes.
        """
        raw = RawAction(
            actor_id="seller",
            actor_role=ActionRole.SELLER,
            intent=ActionIntent.NEGOTIATE_PRICE,
            utterance="I can do $100.",
            proposed_price=100.0,
        )
        check = check_price_format(raw)
        self.assertTrue(check.passed)

    def test_privacy_keyword_detected(self) -> None:
        """Input: utterance contains explicit sensitive payment keywords.

        Expected output: ``privacy_policy`` fails as CRITICAL violation.
        """
        raw = RawAction(
            actor_id="seller",
            actor_role=ActionRole.SELLER,
            intent=ActionIntent.REQUEST_INFO,
            utterance="Please share your credit card and cvv.",
        )
        check = check_privacy(raw)
        self.assertFalse(check.passed)
        self.assertEqual(check.severity, ConstraintSeverity.CRITICAL)
        self.assertEqual(check.violation_type, ViolationType.POLICY_PRIVACY)

    def test_privacy_long_payment_number_detected(self) -> None:
        """Input: utterance has long digit run plus payment context words.

        Expected output: ``privacy_policy`` fails with ``POLICY_PRIVACY``.
        """
        raw = RawAction(
            actor_id="seller",
            actor_role=ActionRole.SELLER,
            intent=ActionIntent.REQUEST_INFO,
            utterance="send bank card 4242 4242 4242 4242 for payment",
        )
        check = check_privacy(raw)
        self.assertFalse(check.passed)
        self.assertEqual(check.violation_type, ViolationType.POLICY_PRIVACY)

    def test_budget_cap_breach_detected(self) -> None:
        """Input: seller proposed price above buyer max bound.

        Expected output: ``budget_cap`` fails with ``BUDGET_EXCEEDED``.
        """
        raw = RawAction(
            actor_id="seller",
            actor_role=ActionRole.SELLER,
            intent=ActionIntent.NEGOTIATE_PRICE,
            utterance="I can do $130.",
            proposed_price=130.0,
        )
        checks = check_price_bounds(raw, buyer_max=120.0, seller_min=90.0)
        by_id = _index_by_id(checks)
        cap = by_id["budget_cap"]
        self.assertFalse(cap.passed)
        self.assertEqual(cap.violation_type, ViolationType.BUDGET_EXCEEDED)

    def test_seller_floor_breach_detected(self) -> None:
        """Input: seller proposed price below seller floor bound.

        Expected output: ``seller_floor`` fails with ``SELLER_FLOOR_BREACH``.
        """
        raw = RawAction(
            actor_id="seller",
            actor_role=ActionRole.SELLER,
            intent=ActionIntent.NEGOTIATE_PRICE,
            utterance="I can do $80.",
            proposed_price=80.0,
        )
        checks = check_price_bounds(raw, buyer_max=120.0, seller_min=90.0)
        by_id = _index_by_id(checks)
        floor = by_id["seller_floor"]
        self.assertFalse(floor.passed)
        self.assertEqual(floor.violation_type, ViolationType.SELLER_FLOOR_BREACH)

    def test_price_bounds_pass_for_valid_offer(self) -> None:
        """Input: seller proposed price inside [seller_min, buyer_max].

        Expected output: both ``budget_cap`` and ``seller_floor`` pass.
        """
        raw = RawAction(
            actor_id="seller",
            actor_role=ActionRole.SELLER,
            intent=ActionIntent.NEGOTIATE_PRICE,
            utterance="I can do $100.",
            proposed_price=100.0,
        )
        checks = check_price_bounds(raw, buyer_max=120.0, seller_min=90.0)
        by_id = _index_by_id(checks)
        self.assertTrue(by_id["budget_cap"].passed)
        self.assertTrue(by_id["seller_floor"].passed)


if __name__ == "__main__":
    unittest.main()
