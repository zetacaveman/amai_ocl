"""Tests for the standalone conversational consumer selection benchmark."""

from __future__ import annotations

import os
from types import SimpleNamespace
import tempfile
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from conversational_consumer_selection import (
    BenchmarkLevel,
    BestOfferSelectionEnv,
    CLARIFICATION_BUDGET_MAX,
    CLARIFICATION_MUST_HAVE_PREFIX,
    CLARIFICATION_PREFERENCE_PREFIX,
    DemoSingleAgentModel,
    LLMPlatformAgent,
    GreedySelectionPolicy,
    OpenAISingleAgentModel,
    Offer,
    PromptBasedSingleAgent,
    RuleBasedUserSimulator,
    SelectionAction,
    SelectionTask,
    UserGoal,
    build_episode_record,
    make_default_task,
    make_v0_demo_task,
    make_v1_direct_intent_task,
    make_v1_partial_intent_task,
    make_v2_hidden_intent_task,
    render_buyer_response,
    render_history_transcript,
    run_benchmark,
    summarize_records,
)


def make_two_offer_task(
    *,
    level: BenchmarkLevel = BenchmarkLevel.DIRECT_INTENT,
    outside_option_threshold: float = 0.20,
    budget_max: float = 100.0,
) -> SelectionTask:
    return SelectionTask(
        task_id=f"task_{level.value}",
        level=level,
        user_goal=UserGoal(
            category="laptop",
            budget_max=budget_max,
            must_have={"ssd": True},
        ),
        offers=(
            Offer(
                offer_id="offer_a",
                category="laptop",
                price=90.0,
                features={"ssd": True},
                attribute_values={"battery": 0.9, "portability": 0.6},
            ),
            Offer(
                offer_id="offer_b",
                category="laptop",
                price=120.0,
                features={"ssd": True},
                attribute_values={"battery": 1.0, "portability": 0.9},
            ),
        ),
        preference_weights={"battery": 1.0, "portability": 0.5},
        price_sensitivity=0.01,
        outside_option_threshold=outside_option_threshold,
        turn_penalty=0.05,
        hidden_preference_slots=()
        if level in {BenchmarkLevel.ORACLE_DEBUG, BenchmarkLevel.DIRECT_INTENT}
        else ("portability",),
        max_turns=4,
    )


class RuleBasedUserSimulatorTests(unittest.TestCase):
    def test_utility_matches_linear_model(self) -> None:
        task = make_two_offer_task()
        simulator = RuleBasedUserSimulator()
        offer = task.offers[0]

        utility = simulator.utility(task, offer, turns_elapsed=2)

        expected = (1.0 * 0.9) + (0.5 * 0.6) - (0.01 * 90.0) - (0.05 * 2)
        self.assertAlmostEqual(expected, utility)

    def test_initial_public_goal_respects_reveal_ratio(self) -> None:
        task = SelectionTask(
            task_id="opening_ratio_task",
            level=BenchmarkLevel.ORACLE_DEBUG,
            user_goal=UserGoal(
                category="headphones",
                budget_max=100.0,
                must_have={"wireless": True, "noise_cancellation": True},
            ),
            offers=(
                Offer(
                    offer_id="offer_a",
                    category="headphones",
                    price=80.0,
                    features={"wireless": True, "noise_cancellation": True},
                    attribute_values={"comfort": 0.7},
                ),
                Offer(
                    offer_id="offer_b",
                    category="headphones",
                    price=90.0,
                    features={"wireless": True, "noise_cancellation": True},
                    attribute_values={"comfort": 0.8},
                ),
            ),
            preference_weights={"comfort": 1.0},
            price_sensitivity=0.01,
            outside_option_threshold=0.1,
            initial_intent_reveal_ratio=1.0 / 3.0,
        )

        public_goal = task.initial_public_goal

        self.assertEqual("headphones", public_goal["category"])
        self.assertIn("budget_max", public_goal)
        self.assertEqual({}, public_goal["must_have"])

    def test_initial_user_request_defaults_to_category_only(self) -> None:
        task = make_default_task(level=BenchmarkLevel.PARTIAL_INTENT)

        self.assertEqual({"category": "headphones"}, task.initial_user_request)

    def test_initial_user_request_can_be_overridden(self) -> None:
        task = SelectionTask(
            task_id="custom_request_task",
            level=BenchmarkLevel.PARTIAL_INTENT,
            user_goal=UserGoal(
                category="headphones",
                budget_max=100.0,
                must_have={"wireless": True},
            ),
            offers=(
                Offer(
                    offer_id="offer_a",
                    category="headphones",
                    price=80.0,
                    features={"wireless": True},
                    attribute_values={"comfort": 0.7},
                ),
                Offer(
                    offer_id="offer_b",
                    category="headphones",
                    price=95.0,
                    features={"wireless": True},
                    attribute_values={"comfort": 0.8},
                ),
            ),
            preference_weights={"comfort": 1.0, "battery": 0.5},
            price_sensitivity=0.01,
            outside_option_threshold=0.1,
            hidden_preference_slots=("comfort",),
            initial_request_payload={
                "category": "headphones",
                "budget_max": 100.0,
            },
        )

        self.assertEqual(
            {"category": "headphones", "budget_max": 100.0},
            task.initial_user_request,
        )

    def test_v0_demo_task_starts_from_category_only(self) -> None:
        task = make_v0_demo_task()

        self.assertEqual({"category": "headphones"}, task.initial_user_request)
        self.assertIsNone(task.initial_revealed_context["budget_max"])
        self.assertEqual({}, task.initial_revealed_context["must_have"])
        self.assertEqual({}, task.initial_revealed_context["preference_weights"])

    def test_v1_direct_intent_task_reveals_full_structured_intent(self) -> None:
        task = make_v1_direct_intent_task()

        self.assertEqual({"category": "headphones"}, task.initial_user_request)
        self.assertEqual(100.0, task.initial_revealed_context["budget_max"])
        self.assertEqual(dict(task.user_goal.must_have), task.initial_revealed_context["must_have"])
        self.assertEqual(dict(task.preference_weights), task.initial_revealed_context["preference_weights"])

    def test_v1_partial_intent_task_reveals_budget_but_not_preferences(self) -> None:
        task = make_v1_partial_intent_task()

        self.assertEqual({"category": "headphones"}, task.initial_user_request)
        self.assertEqual(100.0, task.initial_revealed_context["budget_max"])
        self.assertEqual({"noise_cancellation": True}, task.initial_revealed_context["must_have"])
        self.assertEqual({}, task.initial_revealed_context["preference_weights"])

    def test_v2_hidden_intent_task_keeps_opening_partial_but_structured_reset_minimal(self) -> None:
        task = make_v2_hidden_intent_task()

        self.assertEqual({"category": "headphones"}, task.initial_user_request)
        self.assertIsNone(task.initial_revealed_context["budget_max"])
        self.assertEqual({}, task.initial_revealed_context["must_have"])
        self.assertEqual({}, task.initial_revealed_context["preference_weights"])

    def test_simulator_supports_product_agnostic_clarification_slots(self) -> None:
        task = make_two_offer_task(level=BenchmarkLevel.PARTIAL_INTENT)
        simulator = RuleBasedUserSimulator()

        budget_response = simulator.ask_clarification(task, CLARIFICATION_BUDGET_MAX)
        must_have_response = simulator.ask_clarification(
            task, f"{CLARIFICATION_MUST_HAVE_PREFIX}ssd"
        )
        preference_response = simulator.ask_clarification(
            task, f"{CLARIFICATION_PREFERENCE_PREFIX}portability"
        )

        self.assertEqual("clarified", budget_response["status"])
        self.assertEqual(100.0, budget_response["value"])
        self.assertEqual("clarified", must_have_response["status"])
        self.assertTrue(must_have_response["value"])
        self.assertEqual("clarified", preference_response["status"])
        self.assertEqual(0.5, preference_response["value"])

    def test_simulator_can_clarify_non_required_constraint_as_false(self) -> None:
        task = make_default_task(level=BenchmarkLevel.PARTIAL_INTENT)
        simulator = RuleBasedUserSimulator()

        response = simulator.ask_clarification(task, f"{CLARIFICATION_MUST_HAVE_PREFIX}foldable")

        self.assertEqual("clarified", response["status"])
        self.assertFalse(response["value"])


class BenchmarkEnvironmentTests(unittest.TestCase):
    def test_direct_intent_has_no_user_dialogue_context(self) -> None:
        env = BestOfferSelectionEnv()
        observation, _ = env.reset(task=make_two_offer_task(level=BenchmarkLevel.DIRECT_INTENT))

        self.assertEqual(BenchmarkLevel.DIRECT_INTENT, observation.level)
        self.assertEqual((), observation.user_utterance_history)

    def test_clarification_recovers_hidden_slot(self) -> None:
        env = BestOfferSelectionEnv()
        observation, _ = env.reset(task=make_two_offer_task(level=BenchmarkLevel.PARTIAL_INTENT))
        self.assertIn(
            f"{CLARIFICATION_PREFERENCE_PREFIX}portability",
            observation.available_clarification_slots,
        )
        self.assertNotIn("portability", observation.revealed_context["preference_weights"])
        self.assertEqual(BenchmarkLevel.PARTIAL_INTENT, observation.level)
        self.assertEqual(1, len(observation.user_utterance_history))

        observation, _, terminated, truncated, info = env.step(
            SelectionAction.ask_clarification(f"{CLARIFICATION_PREFERENCE_PREFIX}portability")
        )

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(1, info["clarification_count"])
        self.assertEqual(0.5, observation.revealed_context["preference_weights"]["portability"])
        self.assertEqual(2, len(observation.user_utterance_history))

    def test_invalid_commit_records_executed_violation(self) -> None:
        env = BestOfferSelectionEnv()
        env.reset(task=make_two_offer_task())

        _, _, terminated, truncated, info = env.step(
            SelectionAction.commit_selection("offer_b")
        )

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["commit_success"])
        self.assertTrue(info["has_executed_violation"])
        self.assertTrue(info["has_unrecovered_violation"])
        self.assertEqual("commit_rejected_violation", info["termination_reason"])

    def test_outside_option_rejects_feasible_commit(self) -> None:
        env = BestOfferSelectionEnv()
        env.reset(task=make_two_offer_task(outside_option_threshold=0.40))

        _, _, terminated, truncated, info = env.step(
            SelectionAction.commit_selection("offer_a")
        )

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["commit_success"])
        self.assertFalse(info["has_executed_violation"])
        self.assertFalse(info["has_unrecovered_violation"])
        self.assertEqual("outside_option", info["termination_reason"])
        self.assertTrue(info["last_response"]["outside_option_preferred"])

    def test_all_action_types_advance_state(self) -> None:
        env = BestOfferSelectionEnv()
        env.reset(task=make_default_task(level=BenchmarkLevel.PARTIAL_INTENT))

        observation, _, terminated, _, _ = env.step(
            SelectionAction.ask_clarification(f"{CLARIFICATION_PREFERENCE_PREFIX}comfort")
        )
        self.assertEqual(1, observation.turn_index)
        self.assertFalse(terminated)

        observation, _, terminated, _, _ = env.step(
            SelectionAction.compare_options("offer_budget", "offer_travel")
        )
        self.assertEqual(2, observation.turn_index)
        self.assertFalse(terminated)

        observation, _, terminated, _, _ = env.step(
            SelectionAction.recommend_option("offer_budget")
        )
        self.assertEqual(3, observation.turn_index)
        self.assertFalse(terminated)

        observation, _, terminated, truncated, info = env.step(SelectionAction.escalate())
        self.assertEqual(4, observation.turn_index)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["escalated"])
        self.assertEqual("escalated", info["termination_reason"])

    def test_reset_and_step_loop_terminates_on_valid_commit(self) -> None:
        env = BestOfferSelectionEnv()
        env.reset(task=make_default_task())

        observation, _, terminated, truncated, info = env.step(
            SelectionAction.commit_selection("offer_budget")
        )

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["commit_success"])
        self.assertEqual("offer_budget", observation.committed_offer_id)
        self.assertEqual("commit_accepted", info["termination_reason"])

    def test_timeout_after_transient_violation_marks_unrecovered_violation(self) -> None:
        env = BestOfferSelectionEnv()
        task = make_two_offer_task(level=BenchmarkLevel.PARTIAL_INTENT)
        env.reset(task=task)

        env.step(SelectionAction.ask_clarification("unknown_slot"))
        env.step(SelectionAction.compare_options("offer_a", "offer_a"[:-1] + "c"))
        env.step(SelectionAction.recommend_option("offer_a"))
        _, _, terminated, truncated, info = env.step(SelectionAction.recommend_option("offer_a"))

        self.assertTrue(terminated)
        self.assertTrue(truncated)
        self.assertTrue(info["has_transient_violation"])
        self.assertTrue(info["has_unrecovered_violation"])
        self.assertEqual("timeout", info["termination_reason"])

class MetricsTests(unittest.TestCase):
    def test_summarize_records_groups_by_arm_and_setting(self) -> None:
        records = [
            build_episode_record(
                {
                    "task_id": "ep1",
                    "level": "direct_intent",
                    "commit_success": True,
                    "consumer_utility": 0.8,
                    "consumer_regret": 0.0,
                    "controller_payoff": 0.8,
                    "rounds": 2,
                    "clarification_count": 0,
                    "has_transient_violation": False,
                    "has_executed_violation": False,
                    "has_unrecovered_violation": False,
                    "escalated": False,
                    "termination_reason": "commit_accepted",
                },
                arm="single",
                setting="direct_intent",
            ),
            build_episode_record(
                {
                    "task_id": "ep2",
                    "level": "direct_intent",
                    "commit_success": False,
                    "consumer_utility": 0.4,
                    "consumer_regret": 0.3,
                    "controller_payoff": -0.25,
                    "rounds": 4,
                    "clarification_count": 1,
                    "has_transient_violation": True,
                    "has_executed_violation": False,
                    "has_unrecovered_violation": True,
                    "escalated": False,
                    "termination_reason": "timeout",
                },
                arm="single",
                setting="direct_intent",
            ),
        ]

        summary = summarize_records(records)

        self.assertEqual(1, len(summary))
        row = summary[0]
        self.assertEqual("single", row["arm"])
        self.assertEqual("direct_intent", row["setting"])
        self.assertEqual(2, row["episodes"])
        self.assertAlmostEqual(0.5, row["commit_success_rate"])
        self.assertAlmostEqual(0.6, row["avg_consumer_utility"])
        self.assertAlmostEqual(0.15, row["avg_consumer_regret"])
        self.assertAlmostEqual((0.8 - 0.25) / 2.0, row["avg_controller_payoff"])
        self.assertAlmostEqual(3.0, row["avg_rounds"])
        self.assertAlmostEqual(0.5, row["avg_clarification_count"])
        self.assertAlmostEqual(0.5, row["transient_violation_rate"])
        self.assertAlmostEqual(0.0, row["executed_violation_rate"])
        self.assertAlmostEqual(0.5, row["unrecovered_violation_rate"])

    def test_run_benchmark_writes_summary_files(self) -> None:
        tasks = [
            make_default_task(level=BenchmarkLevel.DIRECT_INTENT),
            make_default_task(level=BenchmarkLevel.PARTIAL_INTENT),
        ]
        policies = {
            "single": GreedySelectionPolicy(clarify_missing_preferences=False),
            "clarify_then_commit": GreedySelectionPolicy(clarify_missing_preferences=True),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            records, summaries = run_benchmark(tasks, policies, output_dir=tmp_dir)
            output_path = Path(tmp_dir)

            self.assertEqual(4, len(records))
            self.assertEqual(4, len(summaries))
            self.assertTrue((output_path / "records.json").exists())
            self.assertTrue((output_path / "summary.json").exists())
            self.assertTrue((output_path / "summary.csv").exists())


class PromptSingleAgentTests(unittest.TestCase):
    def test_platform_agent_context_omits_user_dialogue_by_default(self) -> None:
        env = BestOfferSelectionEnv()
        observation, _ = env.reset(task=make_default_task(level=BenchmarkLevel.PARTIAL_INTENT))

        agent = LLMPlatformAgent(model=DemoSingleAgentModel())
        trace = agent.decide(observation)

        self.assertNotIn("user_utterance_history", trace.context)

    def test_prompt_single_agent_returns_valid_action(self) -> None:
        env = BestOfferSelectionEnv()
        observation, _ = env.reset(task=make_default_task(level=BenchmarkLevel.PARTIAL_INTENT))

        agent = PromptBasedSingleAgent(model=DemoSingleAgentModel())
        trace = agent.decide(observation)

        self.assertEqual("ask_clarification", trace.action.action_type.value)
        self.assertIn(trace.action.slot, observation.available_clarification_slots)
        self.assertFalse(trace.used_fallback)

    def test_demo_platform_agent_avoids_non_discriminative_constraint_question(self) -> None:
        env = BestOfferSelectionEnv()
        observation, _ = env.reset(task=make_default_task(level=BenchmarkLevel.PARTIAL_INTENT))

        agent = PromptBasedSingleAgent(model=DemoSingleAgentModel())
        trace = agent.decide(observation)

        self.assertNotEqual(f"{CLARIFICATION_MUST_HAVE_PREFIX}foldable", trace.action.slot)

    def test_prompt_single_agent_falls_back_to_escalate_on_invalid_output(self) -> None:
        class InvalidModel:
            def generate(self, *, system_prompt: str, user_prompt: str) -> str:
                del system_prompt, user_prompt
                return '{"action_type": "commit_selection", "offer_id": "unknown_offer"}'

        env = BestOfferSelectionEnv()
        observation, _ = env.reset(task=make_default_task(level=BenchmarkLevel.DIRECT_INTENT))

        agent = PromptBasedSingleAgent(model=InvalidModel(), retries=0)
        trace = agent.decide(observation)

        self.assertEqual("escalate", trace.action.action_type.value)
        self.assertTrue(trace.used_fallback)
        self.assertIn("unknown offer id", trace.error or "")

    def test_commit_requires_prior_recommendation(self) -> None:
        env = BestOfferSelectionEnv()
        observation, _ = env.reset(task=make_v1_direct_intent_task())

        agent = PromptBasedSingleAgent(
            model=type(
                "CommitFirstModel",
                (),
                {
                    "generate": lambda self, *, system_prompt, user_prompt: (
                        '{"action_type":"commit_selection","offer_id":"offer_budget"}'
                    )
                },
            )(),
            retries=0,
        )
        trace = agent.decide(observation)

        self.assertEqual("escalate", trace.action.action_type.value)
        self.assertIn("prior recommend_option", trace.error or "")

    def test_prompt_single_agent_accepts_working_belief_envelope(self) -> None:
        class EnvelopeModel:
            def generate(self, *, system_prompt: str, user_prompt: str) -> str:
                del system_prompt, user_prompt
                return (
                    '{"working_belief": {"budget_max_guess": 100, "comfort_guess": "high"}, '
                    '"next_action": {"action_type": "ask_clarification", "slot": "comfort"}}'
                )

        env = BestOfferSelectionEnv()
        observation, _ = env.reset(task=make_default_task(level=BenchmarkLevel.PARTIAL_INTENT))

        agent = PromptBasedSingleAgent(model=EnvelopeModel(), retries=0)
        trace = agent.decide(observation)

        self.assertEqual("ask_clarification", trace.action.action_type.value)
        self.assertEqual("comfort", trace.action.slot)
        assert trace.working_belief is not None
        self.assertEqual(100, trace.working_belief["budget_max_guess"])
        self.assertEqual("high", trace.working_belief["comfort_guess"])

    def test_platform_agent_can_include_user_dialogue_when_enabled(self) -> None:
        env = BestOfferSelectionEnv()
        observation, _ = env.reset(task=make_default_task(level=BenchmarkLevel.PARTIAL_INTENT))

        agent = LLMPlatformAgent(
            model=DemoSingleAgentModel(),
            include_user_utterance_history=True,
        )
        trace = agent.decide(observation)

        self.assertIn("user_utterance_history", trace.context)
        self.assertEqual(1, len(trace.context["user_utterance_history"]))

    def test_openai_single_agent_model_uses_chat_completions(self) -> None:
        class DummyCompletions:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def create(self, **kwargs: object) -> object:
                self.calls.append(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content='{"action_type":"escalate","explanation":"test"}'
                            )
                        )
                    ]
                )

        dummy_completions = DummyCompletions()

        def openai_factory(*, api_key: str, base_url: str | None = None) -> object:
            self.assertEqual("test-key", api_key)
            self.assertIsNone(base_url)
            return SimpleNamespace(chat=SimpleNamespace(completions=dummy_completions))

        fake_module = SimpleNamespace(OpenAI=openai_factory)
        with patch.dict(sys.modules, {"openai": fake_module}):
            model = OpenAISingleAgentModel(api_key="test-key", model="gpt-5.4-mini")
            raw = model.generate(system_prompt="sys", user_prompt="user")

        self.assertEqual('{"action_type":"escalate","explanation":"test"}', raw)
        self.assertEqual(1, len(dummy_completions.calls))
        request = dummy_completions.calls[0]
        self.assertEqual("gpt-5.4-mini", request["model"])
        self.assertEqual("none", request["reasoning_effort"])
        self.assertNotIn("temperature", request)
        self.assertEqual(200, request["max_completion_tokens"])
        self.assertEqual({"type": "json_object"}, request["response_format"])
        self.assertEqual("developer", request["messages"][0]["role"])

    def test_openai_single_agent_model_reads_api_key_from_env(self) -> None:
        class DummyCompletions:
            def create(self, **kwargs: object) -> object:
                del kwargs
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))]
                )

        captured: dict[str, object] = {}

        def openai_factory(*, api_key: str, base_url: str | None = None) -> object:
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            return SimpleNamespace(chat=SimpleNamespace(completions=DummyCompletions()))

        fake_module = SimpleNamespace(OpenAI=openai_factory)
        with patch.dict(sys.modules, {"openai": fake_module}):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=False):
                model = OpenAISingleAgentModel(model="gpt-5.4-mini")
                model.generate(system_prompt="sys", user_prompt="user")

        self.assertEqual("env-key", captured["api_key"])

    def test_openai_single_agent_model_rejects_non_ascii_api_key(self) -> None:
        fake_module = SimpleNamespace(OpenAI=lambda **kwargs: kwargs)
        with patch.dict(sys.modules, {"openai": fake_module}):
            with self.assertRaisesRegex(RuntimeError, "real ASCII API key"):
                OpenAISingleAgentModel(api_key="你的key")


class DialogueSurfaceTests(unittest.TestCase):
    def test_render_history_transcript_and_buyer_surface(self) -> None:
        env = BestOfferSelectionEnv()
        observation, _ = env.reset(task=make_default_task(level=BenchmarkLevel.PARTIAL_INTENT))
        observation, _, _, _, info = env.step(SelectionAction.ask_clarification("comfort"))

        transcript = render_history_transcript(observation.history, offers=observation.offers)
        buyer_text = render_buyer_response(
            SelectionAction.ask_clarification("comfort"),
            info["last_response"],
            offers=observation.offers,
        )

        self.assertIn("Platform:", transcript)
        self.assertIn("Buyer:", transcript)
        self.assertIn("comfort", buyer_text)
        self.assertIn("comfort", transcript)
        self.assertNotIn("[[", buyer_text)
        self.assertNotIn("[[", transcript)
        self.assertNotIn("### ASK_CLARIFICATION", transcript)


if __name__ == "__main__":
    unittest.main()
