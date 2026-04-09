"""Single-episode walkthrough for the standalone benchmark."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conversational_consumer_selection.env import BestOfferSelectionEnv
from conversational_consumer_selection.schemas import (
    BenchmarkLevel,
    CLARIFICATION_PREFERENCE_PREFIX,
    SelectionAction,
)
from conversational_consumer_selection.tasks import make_default_task


def _render_action(action: SelectionAction) -> str:
    if action.slot is not None:
        return f"{action.action_type.value}(slot={action.slot})"
    if action.comparison_offer_id is not None:
        return (
            f"{action.action_type.value}(offer_id={action.offer_id}, "
            f"comparison_offer_id={action.comparison_offer_id})"
        )
    if action.offer_id is not None:
        return f"{action.action_type.value}(offer_id={action.offer_id})"
    return f"{action.action_type.value}()"


def main() -> None:
    task = make_default_task(level=BenchmarkLevel.PARTIAL_INTENT)
    env = BestOfferSelectionEnv()

    observation, info = env.reset(task=task)

    print("Task")
    print(
        json.dumps(
            {
                "task_id": task.task_id,
                "level": task.level.value,
                "budget_max": task.user_goal.budget_max,
                "must_have": task.user_goal.must_have,
                "hidden_preference_slots": task.hidden_preference_slots,
                "max_turns": task.max_turns,
                "turn_penalty": task.turn_penalty,
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
                    "price": offer.price,
                    "features": offer.features,
                    "attribute_values": offer.attribute_values,
                },
                ensure_ascii=False,
            )
        )
    print()

    print("Initial Observation")
    print(json.dumps(observation.revealed_context, indent=2, ensure_ascii=False))
    print()

    scripted_actions = [
        SelectionAction.ask_clarification(f"{CLARIFICATION_PREFERENCE_PREFIX}comfort"),
        SelectionAction.compare_options("offer_budget", "offer_travel"),
        SelectionAction.recommend_option("offer_budget"),
        SelectionAction.commit_selection("offer_budget"),
    ]

    for index, action in enumerate(scripted_actions, start=1):
        print(f"Turn {index}")
        print(f"system_action = {_render_action(action)}")
        observation, step_payoff, terminated, truncated, info = env.step(action)
        print("user_response =")
        print(json.dumps(info["last_response"], indent=2, ensure_ascii=False))
        print(f"step_payoff = {step_payoff}")
        print(
            json.dumps(
                {
                    "rounds": info["rounds"],
                    "clarification_count": info["clarification_count"],
                    "commit_success": info["commit_success"],
                    "consumer_utility": info["consumer_utility"],
                    "controller_payoff": info["controller_payoff"],
                    "termination_reason": info["termination_reason"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        print()
        if terminated or truncated:
            break

    print("Final Summary")
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
