"""Regression tests for seller-centric baseline path isolation.

中文翻译：Regression tests for seller-centric baseline path isolation。"""

from __future__ import annotations

import unittest
from typing import Any

import aimai_ocl.adapters.agenticpay_env as env_mod
from aimai_ocl.runners.single_episode import run_single_negotiation_episode
from aimai_ocl.schemas.audit import AuditEventType


class _DummyEnv:
    """One-step dummy env for baseline runner isolation tests.

中文翻译：One-step dummy env for baseline runner isolation tests。"""

    def __init__(self) -> None:
        """Output: initializes round counter.

中文翻译：输出：initializes round counter。"""
        self.round = 0

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Input: optional reset kwargs.

        Output: deterministic initial observation/info.
        

        中文翻译：输入：optional reset kwargs。"""
        self.round = 0
        return {"current_round": 0, "conversation_history": []}, {}

    def step(
        self,
        buyer_action: str | None = None,
        seller_action: str | None = None,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Input: buyer/seller action strings from runner.

        Output: terminal transition tuple with fixed metrics.
        

        中文翻译：输入：buyer/seller action strings from runner。"""
        self.round += 1
        terminated = self.round >= 1
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
        return obs, 0.0, terminated, False, info

    def close(self) -> None:
        """Output: no-op close for adapter compatibility.

中文翻译：输出：no-op close for adapter compatibility。"""
        return None


class _DummyAgent:
    """Agent stub returning a deterministic utterance.

中文翻译：Agent stub returning a deterministic utterance。"""

    def __init__(self, name: str) -> None:
        """Input: stable actor name for trace assertions.

中文翻译：输入：stable actor name for trace assertions。"""
        self.name = name

    def respond(
        self,
        conversation_history: list[dict[str, Any]],
        current_state: dict[str, Any],
    ) -> str:
        """Input: current history/state from runner.

        Output: fixed negotiation string.
        

        中文翻译：输入：current history/state from runner。"""
        return "offer $100"


class BuyerSellerIsolationSingleRunnerTests(unittest.TestCase):
    """Ensure buyer is pass-through and seller is the measured control target.

中文翻译：确保 buyer is pass-through and seller is the measured control target。"""

    def test_single_runner_buyer_passthrough_seller_traced(self) -> None:
        """Input: one-round baseline episode with buyer/seller stubs.

        Expected output:
        - buyer has no raw/exec action events
        - seller has raw/exec action events
        

        中文翻译：输入：one-round baseline episode with buyer/seller stubs。"""
        original_make_env = env_mod.make_env
        env_mod.make_env = lambda env_id, **kwargs: _DummyEnv()
        try:
            trace, _info = run_single_negotiation_episode(
                env_id="Task1_basic_price_negotiation-v0",
                buyer_agent=_DummyAgent("buyer"),
                seller_agent=_DummyAgent("seller"),
                reset_kwargs={
                    "user_requirement": "demo",
                    "product_info": {"name": "x", "price": 100},
                    "user_profile": "demo",
                },
            )
        finally:
            env_mod.make_env = original_make_env

        buyer_action_events = [
            event
            for event in trace.events
            if event.actor_id == "buyer"
            and event.event_type
            in {
                AuditEventType.RAW_ACTION_RECEIVED,
                AuditEventType.ACTION_EXECUTED,
            }
        ]
        seller_action_events = [
            event
            for event in trace.events
            if event.actor_id == "seller"
            and event.event_type
            in {
                AuditEventType.RAW_ACTION_RECEIVED,
                AuditEventType.ACTION_EXECUTED,
            }
        ]
        self.assertEqual(0, len(buyer_action_events))
        self.assertGreater(len(seller_action_events), 0)


if __name__ == "__main__":
    unittest.main()
