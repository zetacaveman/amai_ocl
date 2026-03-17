"""Tests for OCL decision contract and violation taxonomy.

中文翻译：用于 OCL decision contract and violation taxonomy 的测试。"""

from __future__ import annotations

import unittest

from aimai_ocl.controllers.ocl_controller import OCLController
from aimai_ocl.controllers.risk_gate import RiskGate
from aimai_ocl.controllers.role_policy import RolePolicy
from aimai_ocl.experiment_config import ExperimentConfig, resolve_arm, RunConfig
from aimai_ocl.runners.scenario_validation import enforce_single_product_scenario
from aimai_ocl.schemas.actions import (
    ActionIntent,
    ActionRole,
    ControlDecision,
    RawAction,
)
from aimai_ocl.schemas.constraints import ConstraintCheck, ConstraintSeverity, ViolationType


class ControlContractTests(unittest.TestCase):
    """Contract-level checks for step-2 decision/violation schema.

中文翻译：step-2 decision/violation schema 的契约级检查。"""

    def test_role_violation_sets_taxonomy(self) -> None:
        """Input: buyer emits disallowed intent.

        Expected output: role-policy check fails with ROLE_PERMISSION.
        

        中文翻译：输入：buyer emits disallowed intent。"""
        policy = RolePolicy()
        check = policy.evaluate(
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
        """Input: high-risk TOOL_CALL intent with otherwise passing checks.

        Expected output: executable action remains approved but decision=REWRITE.
        

        中文翻译：输入：high-risk TOOL_CALL intent with otherwise passing checks。"""
        gate = RiskGate()
        raw = RawAction(
            actor_id="seller",
            actor_role=ActionRole.SELLER,
            intent=ActionIntent.TOOL_CALL,
            utterance="Call tool",
        )
        checks = [
            ConstraintCheck(
                constraint_id="role_policy",
                passed=True,
                severity=ConstraintSeverity.INFO,
            ),
            gate.evaluate(raw),
        ]
        exec_action = gate.apply(raw, checks)
        self.assertTrue(exec_action.approved)
        self.assertEqual(exec_action.decision, ControlDecision.REWRITE)

    def test_hard_failure_maps_to_block_decision(self) -> None:
        """Input: at least one ERROR hard failure in prior checks.

        Expected output: executable action blocked with decision=BLOCK.
        

        中文翻译：输入：at least one ERROR hard failure in prior checks。"""
        gate = RiskGate()
        raw = RawAction(
            actor_id="seller",
            actor_role=ActionRole.SELLER,
            intent=ActionIntent.OTHER,
            utterance="text",
        )
        checks = [
            ConstraintCheck(
                constraint_id="role_policy",
                passed=False,
                severity=ConstraintSeverity.ERROR,
                violation_type=ViolationType.ROLE_PERMISSION,
            ),
        ]
        exec_action = gate.apply(raw, checks)
        self.assertFalse(exec_action.approved)
        self.assertEqual(exec_action.decision, ControlDecision.BLOCK)

    def test_constraint_engine_blocks_budget_exceeded(self) -> None:
        """Input: seller price above buyer max in controller state.

        Expected output: controller blocks and records BUDGET_EXCEEDED.
        

        中文翻译：输入：seller price above buyer max in controller state。"""
        controller = OCLController()
        result = controller.apply(
            RawAction(
                actor_id="seller",
                actor_role=ActionRole.SELLER,
                intent=ActionIntent.NEGOTIATE_PRICE,
                utterance="I can do $130",
                proposed_price=130.0,
            ),
            state={"buyer_max_price": 120.0, "seller_min_price": 90.0},
        )
        self.assertFalse(result.executable_action.approved)
        self.assertEqual(result.executable_action.decision, ControlDecision.BLOCK)
        self.assertIn(ViolationType.BUDGET_EXCEEDED.value, result.metadata["violations"])

    def test_constraint_engine_blocks_seller_floor_breach(self) -> None:
        """Input: seller price below seller floor in controller state.

        Expected output: controller blocks and records SELLER_FLOOR_BREACH.
        

        中文翻译：输入：seller price below seller floor in controller state。"""
        controller = OCLController()
        result = controller.apply(
            RawAction(
                actor_id="seller",
                actor_role=ActionRole.SELLER,
                intent=ActionIntent.NEGOTIATE_PRICE,
                utterance="I can do $80",
                proposed_price=80.0,
            ),
            state={"buyer_max_price": 120.0, "seller_min_price": 90.0},
        )
        self.assertFalse(result.executable_action.approved)
        self.assertEqual(result.executable_action.decision, ControlDecision.BLOCK)
        self.assertIn(
            ViolationType.SELLER_FLOOR_BREACH.value,
            result.metadata["violations"],
        )

    def test_constraint_engine_blocks_missing_required_price(self) -> None:
        """Input: negotiate intent without structured numeric price.

        Expected output: controller blocks with FORMAT_INVALID violation.
        

        中文翻译：输入：negotiate intent without structured numeric price。"""
        controller = OCLController()
        result = controller.apply(
            RawAction(
                actor_id="seller",
                actor_role=ActionRole.SELLER,
                intent=ActionIntent.NEGOTIATE_PRICE,
                utterance="Let us discuss options first.",
                proposed_price=None,
            ),
            state={"buyer_max_price": 120.0, "seller_min_price": 90.0},
        )
        self.assertFalse(result.executable_action.approved)
        self.assertEqual(result.executable_action.decision, ControlDecision.BLOCK)
        self.assertIn(ViolationType.FORMAT_INVALID.value, result.metadata["violations"])

    def test_config_digest_stable_for_equal_configs(self) -> None:
        """Input: two logically identical ExperimentConfig objects.

        Expected output: digest strings are identical.
        

        中文翻译：输入：two logically identical ExperimentConfig objects。"""
        run = RunConfig(seed=7)
        arm = resolve_arm("single")
        c1 = ExperimentConfig(run=run, arm=arm).digest()
        c2 = ExperimentConfig(run=run, arm=arm).digest()
        self.assertEqual(c1, c2)

    def test_flat_multi_arm_removed(self) -> None:
        """Input: lookup for removed flat-multi arm key.

        Expected output: registry rejects the removed arm name.
        

        中文翻译：输入：lookup for removed flat-multi arm key。"""
        with self.assertRaises(ValueError):
            resolve_arm("flat_multi")

    def test_single_product_validation_rejects_multiple_products(self) -> None:
        """Input: reset payload with two products.

        Expected output: validation raises ValueError in single-product mode.
        

        中文翻译：输入：reset payload with two products。"""
        with self.assertRaises(ValueError):
            enforce_single_product_scenario(
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
        

        中文翻译：输入：reset payload with one-item products list。"""
        normalized = enforce_single_product_scenario(
            {
                "user_requirement": "need one",
                "products": [{"name": "Only", "price": 3.0}],
            }
        )
        self.assertNotIn("products", normalized)
        self.assertEqual("Only", normalized["product_info"]["name"])


if __name__ == "__main__":
    unittest.main()
