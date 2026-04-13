"""Tests for OCL control pipeline decisions and violation taxonomy."""

from __future__ import annotations

import unittest

from aimai_ocl.adapters import enforce_single_product
from aimai_ocl.control import apply_control, check_role
from aimai_ocl.experiment import ExperimentConfig, RunConfig, resolve_arm
from aimai_ocl.schemas import (
    ActionIntent,
    ActionRole,
    ControlDecision,
    RawAction,
    ViolationType,
)


class ControlContractTests(unittest.TestCase):
    """Contract-level checks for control decision/violation schema."""

    def test_role_violation_sets_taxonomy(self) -> None:
        """Input: buyer emits disallowed intent.

        Expected output: role-policy check fails with ROLE_PERMISSION.
        """
        check = check_role(
            RawAction(
                actor_id="buyer",
                actor_role=ActionRole.BUYER,
                intent=ActionIntent.EXPLAIN_POLICY,
                utterance="I can override policy",
            )
        )
        self.assertFalse(check.passed)
        self.assertEqual(check.violation_type, ViolationType.ROLE_PERMISSION)

    def test_high_risk_intent_maps_to_rewrite_decision(self) -> None:
        """Input: high-risk TOOL_CALL intent with passing checks.

        Expected output: executable action approved with decision=REWRITE.
        """
        raw = RawAction(
            actor_id="platform",
            actor_role=ActionRole.PLATFORM,
            intent=ActionIntent.TOOL_CALL,
            utterance="Call tool",
        )
        result = apply_control(raw, state={})
        self.assertTrue(result.executable.approved)
        self.assertEqual(result.executable.decision, ControlDecision.REWRITE)

    def test_hard_failure_maps_to_block_decision(self) -> None:
        """Input: role check produces ERROR hard failure.

        Expected output: executable action blocked with decision=BLOCK.
        """
        raw = RawAction(
            actor_id="buyer",
            actor_role=ActionRole.BUYER,
            intent=ActionIntent.EXPLAIN_POLICY,
            utterance="text",
        )
        result = apply_control(raw)
        self.assertFalse(result.executable.approved)
        self.assertEqual(result.executable.decision, ControlDecision.BLOCK)

    def test_control_blocks_budget_exceeded(self) -> None:
        """Input: seller price above buyer max in controller state.

        Expected output: controller blocks and records BUDGET_EXCEEDED.
        """
        result = apply_control(
            RawAction(
                actor_id="seller",
                actor_role=ActionRole.SELLER,
                intent=ActionIntent.NEGOTIATE_PRICE,
                utterance="I can do $130",
                proposed_price=130.0,
            ),
            state={"buyer_max_price": 120.0, "seller_min_price": 90.0},
        )
        self.assertFalse(result.executable.approved)
        self.assertEqual(result.executable.decision, ControlDecision.BLOCK)
        violations = [
            c.violation_type.value for c in result.checks
            if c.violation_type is not None and not c.passed
        ]
        self.assertIn(ViolationType.BUDGET_EXCEEDED.value, violations)

    def test_control_blocks_seller_floor_breach(self) -> None:
        """Input: seller price below seller floor in controller state.

        Expected output: controller blocks and records SELLER_FLOOR_BREACH.
        """
        result = apply_control(
            RawAction(
                actor_id="seller",
                actor_role=ActionRole.SELLER,
                intent=ActionIntent.NEGOTIATE_PRICE,
                utterance="I can do $80",
                proposed_price=80.0,
            ),
            state={"buyer_max_price": 120.0, "seller_min_price": 90.0},
        )
        self.assertFalse(result.executable.approved)
        self.assertEqual(result.executable.decision, ControlDecision.BLOCK)
        violations = [
            c.violation_type.value for c in result.checks
            if c.violation_type is not None and not c.passed
        ]
        self.assertIn(ViolationType.SELLER_FLOOR_BREACH.value, violations)

    def test_control_blocks_missing_required_price(self) -> None:
        """Input: negotiate intent without structured numeric price.

        Expected output: controller blocks with FORMAT_INVALID violation.
        """
        result = apply_control(
            RawAction(
                actor_id="seller",
                actor_role=ActionRole.SELLER,
                intent=ActionIntent.NEGOTIATE_PRICE,
                utterance="Let us discuss options first.",
                proposed_price=None,
            ),
            state={"buyer_max_price": 120.0, "seller_min_price": 90.0},
        )
        self.assertFalse(result.executable.approved)
        self.assertEqual(result.executable.decision, ControlDecision.BLOCK)
        violations = [
            c.violation_type.value for c in result.checks
            if c.violation_type is not None and not c.passed
        ]
        self.assertIn(ViolationType.FORMAT_INVALID.value, violations)

    def test_config_digest_stable_for_equal_configs(self) -> None:
        """Input: two logically identical ExperimentConfig objects.

        Expected output: digest strings are identical.
        """
        run = RunConfig(seed=7)
        arm = resolve_arm("single")
        c1 = ExperimentConfig(run=run, arm=arm).digest()
        c2 = ExperimentConfig(run=run, arm=arm).digest()
        self.assertEqual(c1, c2)

    def test_flat_multi_arm_removed(self) -> None:
        """Input: lookup for removed flat-multi arm key.

        Expected output: registry rejects the removed arm name.
        """
        with self.assertRaises(ValueError):
            resolve_arm("flat_multi")

    def test_single_product_validation_rejects_multiple_products(self) -> None:
        """Input: reset payload with two products.

        Expected output: validation raises ValueError in single-product mode.
        """
        with self.assertRaises(ValueError):
            enforce_single_product(
                {
                    "user_requirement": "need one",
                    "products": [
                        {"name": "A", "price": 1.0},
                        {"name": "B", "price": 2.0},
                    ],
                }
            )

    def test_single_product_validation_accepts_one_product_list(self) -> None:
        """Input: reset payload with one-item products list.

        Expected output: normalized payload contains ``product_info`` only.
        """
        normalized = enforce_single_product(
            {
                "user_requirement": "need one",
                "products": [{"name": "Only", "price": 3.0}],
            }
        )
        self.assertNotIn("products", normalized)
        self.assertEqual("Only", normalized["product_info"]["name"])


if __name__ == "__main__":
    unittest.main()
