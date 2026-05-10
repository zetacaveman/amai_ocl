"""Integration test for OCL constraint blocking in episode execution.

中文翻译：Integration test for OCL constraint blocking in episode execution。"""

from __future__ import annotations

import unittest
from typing import Any

import aimai_ocl.adapters.agenticpay_env as env_mod
from aimai_ocl.runners.ocl_episode import run_ocl_negotiation_episode
from aimai_ocl.schemas.audit import AuditEventType
from aimai_ocl.schemas.constraints import ViolationType


class _InspectEnv:
    """Minimal env that records last stepped actions for assertions.

中文翻译：Minimal env that records last stepped actions for assertions。"""

    def __init__(self) -> None:
        self.round = 0
        self.last_buyer_action: str | None = None
        self.last_seller_action: str | None = None

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Input: optional reset kwargs from runner.

        Output: one-step observation/info tuple with empty history.
        

        中文翻译：输入：optional reset kwargs from runner。"""
        self.round = 0
        self.last_buyer_action = None
        self.last_seller_action = None
        return {"current_round": 0, "conversation_history": []}, {}

    def step(
        self,
        buyer_action: str | None = None,
        seller_action: str | None = None,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Input: buyer/seller action texts from runner.

        Output: terminal one-step transition plus deterministic ``info`` payload.
        

        中文翻译：输入：buyer/seller action texts from runner。"""
        self.last_buyer_action = buyer_action
        self.last_seller_action = seller_action
        self.round += 1
        obs = {"current_round": self.round, "conversation_history": []}
        info = {
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
        }
        return obs, 0.0, True, False, info

    def close(self) -> None:
        return None


class _ConstAgent:
    """Deterministic stub agent returning one fixed utterance.

中文翻译：Deterministic stub agent returning one fixed utterance。"""

    def __init__(self, name: str, utterance: str) -> None:
        self.name = name
        self.utterance = utterance

    def respond(
        self,
        conversation_history: list[dict[str, Any]],
        current_state: dict[str, Any],
    ) -> str:
        """Input: conversation/state from runner loop.

        Output: preconfigured utterance string.
        

        中文翻译：输入：conversation/state from runner loop。"""
        return self.utterance


class OCLConstraintIntegrationTests(unittest.TestCase):
    """Ensure hard constraints affect runner execution and audit outputs.

中文翻译：确保 hard constraints affect runner execution and audit outputs。"""

    def test_seller_floor_violation_triggers_replan_and_records_trace(self) -> None:
        """Input: seller proposes price below seller floor (80 < 90).

        Expected output:
        - escalation path triggers deterministic replan once
        - replanned seller action is executed at seller floor
        - trace includes seller-side constraint evaluation + floor violation
        - final_info/trace final status are still populated by runner flow
        

        中文翻译：输入：seller proposes price below seller floor (80 < 90)。"""
        created_envs: list[_InspectEnv] = []

        def _make_env(env_id: str, **kwargs: Any) -> _InspectEnv:
            env = _InspectEnv()
            created_envs.append(env)
            return env

        original_make_env = env_mod.make_env
        env_mod.make_env = _make_env
        try:
            trace, final_info = run_ocl_negotiation_episode(
                env_id="Task1_basic_price_negotiation-v0",
                buyer_agent=_ConstAgent("buyer", "offer $100"),
                seller_agent=_ConstAgent("seller", "final offer $80"),
                env_kwargs={
                    "buyer_max_price": 120.0,
                    "seller_min_price": 90.0,
                    "max_rounds": 10,
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
        env = created_envs[0]
        self.assertEqual("offer $100", env.last_buyer_action)
        self.assertEqual("I can revise to $90.00.", env.last_seller_action)

        self.assertEqual("agreed", final_info["status"])
        self.assertEqual("agreed", trace.final_status)

        evaluated_events = [
            event
            for event in trace.events
            if event.event_type == AuditEventType.CONSTRAINT_EVALUATED
            and event.actor_id == "seller"
        ]
        self.assertGreater(len(evaluated_events), 0)
        violations = {
            check.violation_type.value
            for event in evaluated_events
            for check in event.constraint_checks
            if check.violation_type is not None
        }
        self.assertIn(ViolationType.SELLER_FLOOR_BREACH.value, violations)

        escalation_events = [
            event
            for event in trace.events
            if event.event_type == AuditEventType.ESCALATION_TRIGGERED
            and event.actor_id == "seller"
        ]
        self.assertGreater(len(escalation_events), 0)
        replan_events = [
            event
            for event in trace.events
            if event.event_type == AuditEventType.REPLAN_APPLIED
            and event.actor_id == "seller"
        ]
        self.assertGreater(len(replan_events), 0)

    def test_buyer_cap_is_not_used_by_seller_side_ocl(self) -> None:
        """Input: seller proposes above buyer cap but above seller floor.

        Expected output:
        - seller action is not clamped to buyer max
        - budget-cap violation is not emitted by seller-side OCL runner
        - no deterministic replan occurs from buyer-private information

        中文翻译：输入：seller proposes above buyer cap but above seller floor。"""
        created_envs: list[_InspectEnv] = []

        def _make_env(env_id: str, **kwargs: Any) -> _InspectEnv:
            env = _InspectEnv()
            created_envs.append(env)
            return env

        original_make_env = env_mod.make_env
        env_mod.make_env = _make_env
        try:
            trace, _final_info = run_ocl_negotiation_episode(
                env_id="Task1_basic_price_negotiation-v0",
                buyer_agent=_ConstAgent("buyer", "offer $100"),
                seller_agent=_ConstAgent("seller", "final offer $130"),
                env_kwargs={
                    "buyer_max_price": 120.0,
                    "seller_min_price": 90.0,
                    "max_rounds": 10,
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
        env = created_envs[0]
        self.assertEqual("final offer $130", env.last_seller_action)

        evaluated_events = [
            event
            for event in trace.events
            if event.event_type == AuditEventType.CONSTRAINT_EVALUATED
            and event.actor_id == "seller"
        ]
        violations = {
            check.violation_type.value
            for event in evaluated_events
            for check in event.constraint_checks
            if check.violation_type is not None
        }
        self.assertNotIn(ViolationType.BUDGET_EXCEEDED.value, violations)
        self.assertNotIn(ViolationType.SELLER_FLOOR_BREACH.value, violations)

        replan_events = [
            event
            for event in trace.events
            if event.event_type == AuditEventType.REPLAN_APPLIED
            and event.actor_id == "seller"
        ]
        self.assertEqual(0, len(replan_events))

    def test_buyer_cap_is_used_by_trusted_ocl(self) -> None:
        """Input: trusted OCL sees buyer cap and seller proposes above it.

        Expected output:
        - budget-cap violation is emitted
        - deterministic replan clamps the action to buyer_max_price
        - trusted OCL executes the replanned seller offer

        中文翻译：输入：trusted OCL 可见 buyer cap 且 seller 报价超出 cap。"""
        created_envs: list[_InspectEnv] = []

        def _make_env(env_id: str, **kwargs: Any) -> _InspectEnv:
            env = _InspectEnv()
            created_envs.append(env)
            return env

        original_make_env = env_mod.make_env
        env_mod.make_env = _make_env
        try:
            trace, _final_info = run_ocl_negotiation_episode(
                env_id="Task1_basic_price_negotiation-v0",
                buyer_agent=_ConstAgent("buyer", "offer $100"),
                seller_agent=_ConstAgent("seller", "final offer $130"),
                env_kwargs={
                    "buyer_max_price": 120.0,
                    "seller_min_price": 90.0,
                    "max_rounds": 10,
                },
                reset_kwargs={
                    "user_requirement": "demo",
                    "product_info": {"name": "x", "price": 100},
                    "user_profile": "demo",
                },
                control_information_scope="trusted_constraints",
            )
        finally:
            env_mod.make_env = original_make_env

        self.assertEqual(1, len(created_envs))
        env = created_envs[0]
        self.assertEqual("I can revise to $120.00.", env.last_seller_action)

        evaluated_events = [
            event
            for event in trace.events
            if event.event_type == AuditEventType.CONSTRAINT_EVALUATED
            and event.actor_id == "seller"
        ]
        violations = {
            check.violation_type.value
            for event in evaluated_events
            for check in event.constraint_checks
            if check.violation_type is not None
        }
        self.assertIn(ViolationType.BUDGET_EXCEEDED.value, violations)
        replan_events = [
            event
            for event in trace.events
            if event.event_type == AuditEventType.REPLAN_APPLIED
            and event.actor_id == "seller"
        ]
        self.assertGreater(len(replan_events), 0)


if __name__ == "__main__":
    unittest.main()
