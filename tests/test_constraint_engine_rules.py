"""Unit tests for deterministic ConstraintEngine v1 rules.

中文翻译：用于 deterministic ConstraintEngine v1 rules 的单元测试。"""

from __future__ import annotations

import unittest

from aimai_ocl.controllers.constraint_engine import ConstraintEngine
from aimai_ocl.schemas.actions import ActionIntent, ActionRole, RawAction
from aimai_ocl.schemas.constraints import ConstraintCheck, ConstraintSeverity, ViolationType


def _index_by_id(checks: list[ConstraintCheck]) -> dict[str, ConstraintCheck]:
    """Index checks by ``constraint_id`` for concise assertions.

    Input:
        checks: Ordered list returned by ``ConstraintEngine.evaluate``.

    Output:
        Dict keyed by ``constraint_id`` to simplify test expectations.
    

    中文翻译：Index checks by ``constraint_id`` for concise assertions。"""
    return {check.constraint_id: check for check in checks}


class ConstraintEngineRuleTests(unittest.TestCase):
    """Rule-level coverage for price format, privacy, and price bounds.

中文翻译：Rule-level coverage for price format, privacy, and price bounds。"""

    def setUp(self) -> None:
        """Prepare a reusable engine and baseline numeric bounds.

        Output:
            ``self.engine`` and ``self.base_state`` for each test case.
        

        中文翻译：Prepare a reusable engine and baseline numeric bounds。"""
        self.engine = ConstraintEngine()
        self.base_state = {"buyer_max_price": 120.0, "seller_min_price": 90.0}

    def test_price_format_missing_price_fails(self) -> None:
        """Input: negotiate intent without numeric price.

        Expected output: ``price_format`` fails with ``FORMAT_INVALID``.
        

        中文翻译：输入：negotiate intent without numeric price。"""
        checks = self.engine.evaluate(
            RawAction(
                actor_id="seller",
                actor_role=ActionRole.SELLER,
                intent=ActionIntent.NEGOTIATE_PRICE,
                utterance="Let's discuss first.",
                proposed_price=None,
            ),
            state=self.base_state,
        )
        by_id = _index_by_id(checks)
        fmt = by_id["price_format"]
        self.assertFalse(fmt.passed)
        self.assertEqual(fmt.severity, ConstraintSeverity.ERROR)
        self.assertEqual(fmt.violation_type, ViolationType.FORMAT_INVALID)

    def test_price_format_valid_price_passes(self) -> None:
        """Input: negotiate intent with valid positive price.

        Expected output: ``price_format`` check passes.
        

        中文翻译：输入：negotiate intent with valid positive price。"""
        checks = self.engine.evaluate(
            RawAction(
                actor_id="seller",
                actor_role=ActionRole.SELLER,
                intent=ActionIntent.NEGOTIATE_PRICE,
                utterance="I can do $100.",
                proposed_price=100.0,
            ),
            state=self.base_state,
        )
        by_id = _index_by_id(checks)
        self.assertTrue(by_id["price_format"].passed)

    def test_privacy_keyword_detected(self) -> None:
        """Input: utterance contains explicit sensitive payment keywords.

        Expected output: ``privacy_policy`` fails as CRITICAL violation.
        

        中文翻译：输入：utterance contains explicit sensitive payment keywords。"""
        checks = self.engine.evaluate(
            RawAction(
                actor_id="seller",
                actor_role=ActionRole.SELLER,
                intent=ActionIntent.REQUEST_INFO,
                utterance="Please share your credit card and cvv.",
            ),
            state=self.base_state,
        )
        by_id = _index_by_id(checks)
        privacy = by_id["privacy_policy"]
        self.assertFalse(privacy.passed)
        self.assertEqual(privacy.severity, ConstraintSeverity.CRITICAL)
        self.assertEqual(privacy.violation_type, ViolationType.POLICY_PRIVACY)

    def test_privacy_long_payment_number_detected(self) -> None:
        """Input: utterance has long digit run plus payment context words.

        Expected output: ``privacy_policy`` fails with ``POLICY_PRIVACY``.
        

        中文翻译：输入：utterance has long digit run plus payment context words。"""
        checks = self.engine.evaluate(
            RawAction(
                actor_id="seller",
                actor_role=ActionRole.SELLER,
                intent=ActionIntent.REQUEST_INFO,
                utterance="send bank card 4242 4242 4242 4242 for payment",
            ),
            state=self.base_state,
        )
        by_id = _index_by_id(checks)
        privacy = by_id["privacy_policy"]
        self.assertFalse(privacy.passed)
        self.assertEqual(privacy.violation_type, ViolationType.POLICY_PRIVACY)

    def test_budget_cap_breach_detected(self) -> None:
        """Input: seller proposed price above buyer max bound.

        Expected output: ``budget_cap`` fails with ``BUDGET_EXCEEDED``.
        

        中文翻译：输入：seller proposed price above buyer max bound。"""
        checks = self.engine.evaluate(
            RawAction(
                actor_id="seller",
                actor_role=ActionRole.SELLER,
                intent=ActionIntent.NEGOTIATE_PRICE,
                utterance="I can do $130.",
                proposed_price=130.0,
            ),
            state=self.base_state,
        )
        by_id = _index_by_id(checks)
        cap = by_id["budget_cap"]
        self.assertFalse(cap.passed)
        self.assertEqual(cap.violation_type, ViolationType.BUDGET_EXCEEDED)

    def test_seller_floor_breach_detected(self) -> None:
        """Input: seller proposed price below seller floor bound.

        Expected output: ``seller_floor`` fails with ``SELLER_FLOOR_BREACH``.
        

        中文翻译：输入：seller proposed price below seller floor bound。"""
        checks = self.engine.evaluate(
            RawAction(
                actor_id="seller",
                actor_role=ActionRole.SELLER,
                intent=ActionIntent.NEGOTIATE_PRICE,
                utterance="I can do $80.",
                proposed_price=80.0,
            ),
            state=self.base_state,
        )
        by_id = _index_by_id(checks)
        floor = by_id["seller_floor"]
        self.assertFalse(floor.passed)
        self.assertEqual(floor.violation_type, ViolationType.SELLER_FLOOR_BREACH)

    def test_price_bounds_pass_for_valid_offer(self) -> None:
        """Input: seller proposed price inside [seller_min, buyer_max].

        Expected output: both ``budget_cap`` and ``seller_floor`` pass.
        

        中文翻译：输入：seller proposed price inside [seller_min, buyer_max]。"""
        checks = self.engine.evaluate(
            RawAction(
                actor_id="seller",
                actor_role=ActionRole.SELLER,
                intent=ActionIntent.NEGOTIATE_PRICE,
                utterance="I can do $100.",
                proposed_price=100.0,
            ),
            state=self.base_state,
        )
        by_id = _index_by_id(checks)
        self.assertTrue(by_id["budget_cap"].passed)
        self.assertTrue(by_id["seller_floor"].passed)


if __name__ == "__main__":
    unittest.main()
