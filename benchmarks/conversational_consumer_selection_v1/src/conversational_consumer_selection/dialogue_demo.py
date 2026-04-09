"""Dialogue-layer demos built on top of the structured benchmark core."""

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
from conversational_consumer_selection.policies import GreedySelectionPolicy
from conversational_consumer_selection.surfaces import (
    DialogueDecorator,
    OpenAIDialogueModel,
    render_buyer_opening,
    render_buyer_response,
    render_history_transcript,
    render_platform_opening,
    render_platform_action_surface,
)
from conversational_consumer_selection.tasks import (
    make_v1_direct_intent_task,
    make_v1_partial_intent_task,
    make_v2_hidden_intent_task,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the dialogue-layer demos built on top of the v0 structured core."
    )
    parser.add_argument(
        "--backend",
        choices=("rule", "demo", "openai"),
        default="demo",
        help="Decision backend for the platform-side action policy.",
    )
    parser.add_argument(
        "--mode",
        choices=("v1_direct_intent", "v1_partial_intent", "v2_hidden_intent"),
        default="v1_partial_intent",
        help="Dialogue-layer demo mode.",
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
        "--debug-actions",
        action="store_true",
        help="Print the internal structured action and structured response after each turn.",
    )
    parser.add_argument(
        "--dialogue-backend",
        choices=("template", "openai"),
        default="template",
        help="Dialogue surface backend. Use openai to let an LLM rewrite visible utterances.",
    )
    parser.add_argument(
        "--dialogue-model",
        default="gpt-5.4-mini",
        help="OpenAI model name when --dialogue-backend openai is used.",
    )
    args = parser.parse_args()

    if args.mode == "v2_hidden_intent":
        task = make_v2_hidden_intent_task()
    elif args.mode == "v1_direct_intent":
        task = make_v1_direct_intent_task()
    else:
        task = make_v1_partial_intent_task()
    env = BestOfferSelectionEnv()
    policy = _build_policy(args)
    decorator = _build_decorator(args)

    observation, info = env.reset(task=task)

    print("Task")
    print(
        json.dumps(
            {
                "task_id": task.task_id,
                "mode": args.mode,
                "level": task.level.value,
                "max_turns": task.max_turns,
                "hidden_preference_slots": task.hidden_preference_slots,
                "backend": args.backend,
                "dialogue_backend": args.dialogue_backend,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print()

    print("Offers")
    for offer in task.offers:
        print(
            json.dumps(
                {
                    "offer_id": offer.offer_id,
                    "title": offer.title,
                    "price": offer.price,
                    "features": offer.features,
                    "attribute_values": offer.attribute_values,
                },
                ensure_ascii=False,
            )
        )
    print()

    opening_history = "(no prior turns)"
    platform_opening = decorator.decorate_opening(
        speaker="platform",
        base_text=render_platform_opening(task),
        history_text=opening_history,
    )
    buyer_opening = decorator.decorate_opening(
        speaker="buyer",
        base_text=render_buyer_opening(task),
        history_text=f"Platform: {platform_opening}",
    )
    print("Turn 1")
    print(f"Platform: {platform_opening}")
    print(f"Buyer: {buyer_opening}")
    print()

    opening_offset = 1

    while not observation.terminated:
        action = policy.act(observation)
        history_text_before = render_history_transcript(observation.history, offers=observation.offers)
        base_platform_text = render_platform_action_surface(action, offers=observation.offers)
        platform_text = decorator.decorate_platform(
            base_text=base_platform_text,
            action=action,
            offers=observation.offers,
            history_text=history_text_before,
        )
        offers = observation.offers

        observation, _, terminated, truncated, info = env.step(action)
        base_buyer_text = render_buyer_response(
            action,
            info["last_response"],
            offers=offers,
        )
        history_text_after = render_history_transcript(observation.history, offers=offers)
        buyer_text = decorator.decorate_buyer(
            base_text=base_buyer_text,
            action=action,
            response=info["last_response"],
            platform_text=platform_text,
            offers=offers,
            history_text=history_text_after,
        )

        print(f"Turn {info['rounds'] + opening_offset}")
        print(f"Platform: {platform_text}")
        print(f"Buyer: {buyer_text}")
        if args.debug_actions:
            print("control_action =")
            print(_pretty_action(action))
            print("structured_response =")
            print(json.dumps(info["last_response"], indent=2, ensure_ascii=False))
        print()

        if terminated or truncated:
            break

    print("Final Summary")
    print(
        json.dumps(
            {
                "commit_success": info["commit_success"],
                "consumer_utility": info["consumer_utility"],
                "controller_payoff": info["controller_payoff"],
                "rounds": info["rounds"],
                "clarification_count": info["clarification_count"],
                "termination_reason": info["termination_reason"],
                "committed_offer_id": info["committed_offer_id"],
                "best_offer_id": info["best_offer_id"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _build_policy(args: argparse.Namespace) -> GreedySelectionPolicy | LLMPlatformAgent:
    if args.backend == "rule":
        return GreedySelectionPolicy(clarify_missing_preferences=True)
    if args.backend == "openai":
        model = OpenAIPlatformAgentModel(
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_tokens=args.max_tokens,
            base_url=args.base_url,
        )
        return LLMPlatformAgent(model=model, include_user_utterance_history=True)
    return LLMPlatformAgent(
        model=DemoPlatformAgentModel(),
        include_user_utterance_history=True,
    )


def _build_decorator(args: argparse.Namespace) -> DialogueDecorator:
    if args.dialogue_backend == "openai":
        return DialogueDecorator(
            model=OpenAIDialogueModel(
                model=args.dialogue_model,
                reasoning_effort=args.reasoning_effort,
                max_tokens=min(args.max_tokens, 120),
                base_url=args.base_url,
            )
        )
    return DialogueDecorator()


def _pretty_action(action: object) -> str:
    payload = {
        "action_type": action.action_type.value,
        "slot": action.slot,
        "offer_id": action.offer_id,
        "comparison_offer_id": action.comparison_offer_id,
        "explanation": action.explanation,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
