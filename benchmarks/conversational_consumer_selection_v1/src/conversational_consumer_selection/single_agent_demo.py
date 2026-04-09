"""Demo for the LLM-backed platform agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conversational_consumer_selection.agents import (
    DemoPlatformAgentModel,
    LLMPlatformAgent,
    OpenAIPlatformAgentModel,
)
from conversational_consumer_selection.env import BestOfferSelectionEnv
from conversational_consumer_selection.surfaces import (
    build_platform_opening_stage,
    build_user_initial_request,
)
from conversational_consumer_selection.tasks import make_v0_demo_task


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the prompt-style single-agent demo with a local demo backend or OpenAI."
    )
    parser.add_argument(
        "--backend",
        choices=("demo", "openai"),
        default="demo",
        help="Agent backend used to produce raw action JSON.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4-mini",
        help="OpenAI model name when --backend openai is used.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh"),
        default="none",
        help="GPT-5 reasoning effort when --backend openai is used.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Generation token cap when --backend openai is used.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full agent context, raw model output, and full structured response.",
    )
    args = parser.parse_args()

    task = make_v0_demo_task()
    env = BestOfferSelectionEnv()
    if args.backend == "openai":
        model_backend = OpenAIPlatformAgentModel(
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_tokens=args.max_tokens,
            base_url=args.base_url,
        )
    else:
        model_backend = DemoPlatformAgentModel()
    agent = LLMPlatformAgent(model=model_backend)

    observation, info = env.reset(task=task)

    print("Task")
    print(
        json.dumps(
            {
                "task_id": task.task_id,
                "mode": "v0_structured",
                "max_turns": task.max_turns,
                "hidden_preference_slots": task.hidden_preference_slots,
                "backend": args.backend,
                "model": args.model if args.backend == "openai" else "demo",
                "reasoning_effort": args.reasoning_effort if args.backend == "openai" else None,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print()

    opening_offset = 1
    print("Turn 1")
    print("stage =")
    print(
        json.dumps(
            {
                "platform": build_platform_opening_stage(task),
                "user": build_user_initial_request(task),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print()

    while not observation.terminated:
        trace = agent.decide(observation)
        turn_number = observation.turn_index + 1 + opening_offset

        print(f"Turn {turn_number}")
        print("state =")
        print(json.dumps(_compact_state(trace.context), indent=2, ensure_ascii=False))
        print("action =")
        print(
            json.dumps(
                {
                    "action_type": trace.action.action_type.value,
                    "slot": trace.action.slot,
                    "offer_id": trace.action.offer_id,
                    "comparison_offer_id": trace.action.comparison_offer_id,
                    "explanation": trace.action.explanation,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        if args.verbose:
            print("agent_context =")
            print(json.dumps(trace.context, indent=2, ensure_ascii=False))
            print("raw_model_output =")
            print(trace.raw_output)
            if trace.working_belief is not None:
                print("working_belief =")
                print(json.dumps(trace.working_belief, indent=2, ensure_ascii=False))
            if trace.used_fallback:
                print(f"fallback_reason = {trace.error}")

        observation, step_payoff, terminated, truncated, info = env.step(trace.action)
        print("result =")
        print(
            json.dumps(
                _compact_result(info, step_payoff),
                indent=2,
                ensure_ascii=False,
            )
        )
        if args.verbose:
            print("structured_response =")
            print(json.dumps(info["last_response"], indent=2, ensure_ascii=False))
        print()

        if terminated or truncated:
            break

    print("Final Summary")
    print(json.dumps(_compact_summary(info), indent=2, ensure_ascii=False))


def _compact_state(context: dict[str, object]) -> dict[str, object]:
    revealed_context = dict(context["revealed_context"])
    return {
        "mode": "v0_structured",
        "revealed_context": revealed_context,
        "available_clarification_slots": context["available_clarification_slots"],
        "offer_ids": [offer["offer_id"] for offer in context["offers"]],
        "turn_index": context["turn_index"],
        "remaining_turns": context["remaining_turns"],
    }


def _compact_result(info: dict[str, object], step_payoff: float) -> dict[str, object]:
    result = {
        "step_payoff": step_payoff,
        "response_status": info["last_response"].get("status"),
        "response_reason": info["last_response"].get("reason"),
        "rounds": info["rounds"],
        "commit_success": info["commit_success"],
        "termination_reason": info["termination_reason"],
    }
    preferred_offer_id = info["last_response"].get("preferred_offer_id")
    if preferred_offer_id is not None:
        result["preferred_offer_id"] = preferred_offer_id
    accepted = info["last_response"].get("accepted")
    if accepted is not None:
        result["accepted"] = accepted
    return result


def _compact_summary(info: dict[str, object]) -> dict[str, object]:
    return {
        "task_id": info["task_id"],
        "mode": "v0_structured",
        "commit_success": info["commit_success"],
        "consumer_utility": info["consumer_utility"],
        "controller_payoff": info["controller_payoff"],
        "rounds": info["rounds"],
        "clarification_count": info["clarification_count"],
        "termination_reason": info["termination_reason"],
        "committed_offer_id": info["committed_offer_id"],
        "best_offer_id": info["best_offer_id"],
    }


if __name__ == "__main__":
    main()
