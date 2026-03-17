"""Tests for step-8 Shapley attribution interface module.

中文翻译：用于 step-8 Shapley attribution interface module 的测试。"""

from __future__ import annotations

from itertools import combinations
import unittest
from typing import Any

import aimai_ocl.adapters.agenticpay_env as env_mod
from aimai_ocl.attribution_shapley import (
    CONTROLLED_ROLES,
    ValueFunctionConfig,
    compute_shapley,
    compute_V,
    fallback_policy,
    run_episode,
)
from aimai_ocl.schemas.audit import AuditEvent, AuditEventType, EpisodeTrace
from aimai_ocl.schemas.constraints import ConstraintCheck, ConstraintSeverity, ViolationType


class _DummyEnv:
    """One-step env stub for run_episode interface test.

中文翻译：One-step env stub for run_episode interface test。"""

    def __init__(self) -> None:
        self.last_buyer_action: str | None = None
        self.last_seller_action: str | None = None
        self.round = 0

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        self.round = 0
        self.last_buyer_action = None
        self.last_seller_action = None
        return {"current_round": 0, "conversation_history": []}, {}

    def step(
        self,
        buyer_action: str | None = None,
        seller_action: str | None = None,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.last_buyer_action = buyer_action
        self.last_seller_action = seller_action
        self.round += 1
        return (
            {"current_round": self.round, "conversation_history": []},
            0.0,
            True,
            False,
            {
                "round": self.round,
                "status": "agreed",
                "termination_reason": "agreed",
                "agreed_price": 100.0,
                "buyer_price": 100.0,
                "seller_price": 100.0,
                "buyer_reward": 1.0,
                "seller_reward": 1.0,
                "global_score": 1.0,
                "buyer_score": 1.0,
                "seller_score": 1.0,
            },
        )

    def close(self) -> None:
        return None


class _BuyerAgent:
    def respond(
        self,
        conversation_history: list[dict[str, Any]],
        current_state: dict[str, Any],
    ) -> str:
        return "buyer offer $100"


class _SellerAgent:
    name = "seller"

    def respond(
        self,
        conversation_history: list[dict[str, Any]],
        current_state: dict[str, Any],
    ) -> str:
        return "seller offer $130"


class AttributionInterfaceTests(unittest.TestCase):
    """Contract tests for run_episode / compute_V / fallback / shapley.

中文翻译：run_episode / compute_V / fallback / shapley 的契约测试。"""

    def test_fallback_policy_is_deterministic(self) -> None:
        """Input: same role/state twice.

        Expected output: exactly same action text both times.
        

        中文翻译：输入：same role/state twice。"""
        state = {"buyer_max_price": 120.0, "seller_min_price": 90.0}
        a1 = fallback_policy("seller", state)
        a2 = fallback_policy("seller", state)
        self.assertEqual(a1, a2)
        self.assertEqual("[seller] deterministic offer $105.00", a1)

    def test_compute_V_from_trace(self) -> None:
        """Input: synthetic trace with reward/round/violations/escalation.

        Expected output: scalar value matches configured weighted formula.
        

        中文翻译：输入：synthetic trace with reward/round/violations/escalation。"""
        trace = EpisodeTrace(episode_id="ep-v", env_id="Task1_basic_price_negotiation-v0")
        trace.final_status = "agreed"
        trace.final_metrics = {"seller_reward": 10.0, "round": 4}
        trace.add_event(
            AuditEvent(
                event_type=AuditEventType.CONSTRAINT_EVALUATED,
                actor_id="seller",
                constraint_checks=[
                    ConstraintCheck(
                        constraint_id="budget_cap",
                        passed=False,
                        severity=ConstraintSeverity.ERROR,
                        violation_type=ViolationType.BUDGET_EXCEEDED,
                    )
                ],
            )
        )
        trace.add_event(
            AuditEvent(
                event_type=AuditEventType.ESCALATION_TRIGGERED,
                actor_id="seller",
            )
        )
        value = compute_V(trace, config=ValueFunctionConfig())
        self.assertAlmostEqual(12.4, value)

    def test_compute_shapley_exact_for_additive_game(self) -> None:
        """Input: additive coalition game over controlled roles.

        Expected output: exact shapley equals per-role additive contributions.
        

        中文翻译：输入：additive coalition game over controlled roles。"""
        contrib = {"platform": 2.0, "seller": 1.0, "expert": 3.0}
        values: dict[frozenset[str], float] = {}
        role_list = list(CONTROLLED_ROLES)
        for size in range(len(role_list) + 1):
            for subset in combinations(role_list, size):
                values[frozenset(subset)] = float(sum(contrib[r] for r in subset))

        result = compute_shapley(values)
        self.assertAlmostEqual(2.0, result["phi"]["platform"])
        self.assertAlmostEqual(1.0, result["phi"]["seller"])
        self.assertAlmostEqual(3.0, result["phi"]["expert"])
        self.assertAlmostEqual(2.0 / 6.0, result["w"]["platform"])
        self.assertAlmostEqual(1.0 / 6.0, result["w"]["seller"])
        self.assertAlmostEqual(3.0 / 6.0, result["w"]["expert"])

    def test_run_episode_supports_role_mask_and_returns_trace(self) -> None:
        """Input: role mask without seller role.

        Expected output:
        - seller action falls back to deterministic policy
        - returned object is a populated EpisodeTrace
        

        中文翻译：输入：role mask without seller role。"""
        created_envs: list[_DummyEnv] = []

        def _make_env(env_id: str, **kwargs: Any) -> _DummyEnv:
            env = _DummyEnv()
            created_envs.append(env)
            return env

        original_make_env = env_mod.make_env
        env_mod.make_env = _make_env
        try:
            trace = run_episode(
                role_mask=set(),
                seed=7,
                env_id="Task1_basic_price_negotiation-v0",
                buyer_agent=_BuyerAgent(),
                seller_agent=_SellerAgent(),
                env_kwargs={
                    "max_rounds": 10,
                    "initial_seller_price": 180.0,
                    "buyer_max_price": 120.0,
                    "seller_min_price": 90.0,
                },
                reset_kwargs={
                    "user_requirement": "demo",
                    "product_info": {"name": "x", "price": 100},
                    "user_profile": "demo",
                },
            )
        finally:
            env_mod.make_env = original_make_env

        self.assertEqual(1, len(created_envs))
        self.assertEqual("[seller] deterministic offer $105.00", created_envs[0].last_seller_action)
        self.assertIsInstance(trace, EpisodeTrace)
        self.assertEqual([], trace.metadata.get("role_mask"))

    def test_compute_shapley_raises_when_subset_values_incomplete(self) -> None:
        """Input: coalition values missing required subsets.

        Expected output: exact Shapley computation raises ValueError.
        

        中文翻译：输入：coalition values missing required subsets。"""
        with self.assertRaises(ValueError):
            compute_shapley(
                {
                    frozenset(): 0.0,
                    frozenset({"seller"}): 1.0,
                }
            )


if __name__ == "__main__":
    unittest.main()
