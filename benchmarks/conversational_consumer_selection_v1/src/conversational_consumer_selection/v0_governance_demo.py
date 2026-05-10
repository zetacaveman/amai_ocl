"""V0 structured action-governance demo.

This demo is a mechanism vignette, not a full experimental comparison. It shows
how a thin OCL-style wrapper can gate/rewrite/escalate structured shopping
actions before they reach the consumer-selection environment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterable

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conversational_consumer_selection.env import BestOfferSelectionEnv
from conversational_consumer_selection.governance import SelectionGovernedPolicy
from conversational_consumer_selection.metrics import build_episode_record, summarize_records
from conversational_consumer_selection.policies import Policy
from conversational_consumer_selection.schemas import (
    BenchmarkLevel,
    CategorySchema,
    CLARIFICATION_BUDGET_MAX,
    CLARIFICATION_MUST_HAVE_PREFIX,
    CLARIFICATION_PREFERENCE_PREFIX,
    LatentConsumerModel,
    Observation,
    Offer,
    SelectionAction,
    SelectionTask,
    UserGoal,
    UserProfile,
)
from conversational_consumer_selection.tasks import make_v0_demo_task


@dataclass
class StaticActionPolicy:
    """Policy that keeps proposing the same raw action."""

    action: SelectionAction

    def act(self, observation: Observation) -> SelectionAction:
        del observation
        return self.action


@dataclass
class ScriptedPolicy:
    """Policy that emits a fixed action script, then repeats the last action."""

    actions: tuple[SelectionAction, ...]
    index: int = 0

    def act(self, observation: Observation) -> SelectionAction:
        del observation
        if not self.actions:
            return SelectionAction.escalate("empty script")
        action = self.actions[min(self.index, len(self.actions) - 1)]
        self.index += 1
        return action


@dataclass(frozen=True)
class DemoScenario:
    """One v0 governance-demo scenario."""

    name: str
    task: SelectionTask
    raw_policy: Policy
    description: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the V0 structured governance demo for consumer selection."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/conversational_consumer_selection_v1/v0_governance_demo",
        help="Directory used for records.json / summary.json / trace.json.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="Safety cap for each rollout.",
    )
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for scenario in _build_scenarios():
        for arm, policy in _build_arms(scenario).items():
            record, trace = _run_trace(
                scenario=scenario,
                arm=arm,
                policy=policy,
                max_steps=args.max_steps,
            )
            records.append(record)
            traces.append(trace)

    summaries = summarize_records(records, group_keys=("scenario", "arm"))
    _write_outputs(Path(args.output_dir), records=records, summaries=summaries, traces=traces)

    print("V0 governance demo summary")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    print()
    print(f"Wrote outputs to {Path(args.output_dir).resolve()}")


def _build_scenarios() -> list[DemoScenario]:
    safe_task = make_v0_demo_task()
    must_have_task = _make_must_have_gate_task()
    return [
        DemoScenario(
            name="safe_script",
            task=safe_task,
            raw_policy=ScriptedPolicy(
                actions=(
                    SelectionAction.ask_clarification(CLARIFICATION_BUDGET_MAX),
                    SelectionAction.recommend_option("offer_budget"),
                    SelectionAction.commit_selection("offer_budget"),
                )
            ),
            description="A safe raw script that already follows clarify -> recommend -> commit.",
        ),
        DemoScenario(
            name="missing_info_commit",
            task=safe_task,
            raw_policy=StaticActionPolicy(
                SelectionAction.commit_selection(
                    "offer_budget",
                    explanation="raw policy tries to commit before clarification/recommendation",
                )
            ),
            description="Raw policy commits a feasible offer too early; governance recovers evidence first.",
        ),
        DemoScenario(
            name="must_have_violation_commit",
            task=must_have_task,
            raw_policy=StaticActionPolicy(
                SelectionAction.commit_selection(
                    "offer_slim",
                    explanation="raw policy tries to commit an offer that violates hidden must-have",
                )
            ),
            description="Raw policy commits a non-foldable offer; governance asks constraints and redirects.",
        ),
    ]


def _build_arms(scenario: DemoScenario) -> dict[str, Policy]:
    return {
        "raw_policy": scenario.raw_policy,
        "governed_policy": SelectionGovernedPolicy(
            raw_policy=_clone_policy(scenario.raw_policy),
        ),
    }


def _clone_policy(policy: Policy) -> Policy:
    if isinstance(policy, StaticActionPolicy):
        return StaticActionPolicy(policy.action)
    if isinstance(policy, ScriptedPolicy):
        return ScriptedPolicy(actions=tuple(policy.actions))
    raise TypeError(f"unsupported demo policy type: {type(policy)!r}")


def _run_trace(
    *,
    scenario: DemoScenario,
    arm: str,
    policy: Policy,
    max_steps: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    env = BestOfferSelectionEnv()
    observation, _ = env.reset(task=scenario.task)
    steps: list[dict[str, Any]] = []

    for _ in range(max_steps):
        if observation.terminated:
            break
        action = policy.act(observation)
        decision = getattr(policy, "last_decision", None)
        observation, step_payoff, terminated, truncated, info = env.step(action)
        raw_action = decision.raw_action if decision is not None else action
        governed_action = decision.governed_action if decision is not None else action
        steps.append(
            {
                "turn_index": observation.turn_index,
                "raw_action": _action_to_dict(raw_action),
                "governed_action": _action_to_dict(governed_action),
                "intervention": decision.intervention if decision is not None else "none",
                "reason": decision.reason if decision is not None else "raw_policy",
                "step_payoff": step_payoff,
                "terminated": terminated,
                "truncated": truncated,
                "response": info["last_response"],
                "summary": _compact_summary(info),
            }
        )
        if terminated or truncated:
            break

    info = env.episode_summary()
    record = build_episode_record(
        info,
        arm=arm,
        setting="v0_structured",
    )
    record["scenario"] = scenario.name
    record["description"] = scenario.description
    record["intervention_count"] = sum(
        1 for step in steps if step["intervention"] not in {"none", "pass"}
    )
    trace = {
        "scenario": scenario.name,
        "arm": arm,
        "description": scenario.description,
        "task": _task_to_dict(scenario.task),
        "steps": steps,
        "final_summary": _compact_summary(info),
    }
    return record, trace


def _make_must_have_gate_task() -> SelectionTask:
    category_schema = CategorySchema(
        category="headphones",
        constraint_slots=("foldable", "wireless"),
        preference_slots=("comfort", "battery"),
    )
    latent_consumer_model = LatentConsumerModel(
        budget_max=100.0,
        must_have={"foldable": True, "wireless": True},
        preference_weights={"comfort": 1.0, "battery": 0.4},
        price_sensitivity=0.005,
        outside_option_threshold=0.20,
        turn_penalty=0.03,
    )
    return SelectionTask(
        task_id="v0_must_have_gate",
        level=BenchmarkLevel.PARTIAL_INTENT,
        user_goal=UserGoal(
            category=category_schema.category,
            budget_max=latent_consumer_model.budget_max,
            must_have=dict(latent_consumer_model.must_have),
        ),
        offers=(
            Offer(
                offer_id="offer_foldable",
                title="FoldGo ANC",
                category=category_schema.category,
                price=88.0,
                features={"foldable": True, "wireless": True},
                attribute_values={"comfort": 0.70, "battery": 0.75},
            ),
            Offer(
                offer_id="offer_slim",
                title="SlimStay ANC",
                category=category_schema.category,
                price=82.0,
                features={"foldable": False, "wireless": True},
                attribute_values={"comfort": 0.95, "battery": 0.80},
            ),
            Offer(
                offer_id="offer_wire",
                title="WireLite",
                category=category_schema.category,
                price=64.0,
                features={"foldable": True, "wireless": False},
                attribute_values={"comfort": 0.60, "battery": 0.30},
            ),
        ),
        preference_weights=dict(latent_consumer_model.preference_weights),
        price_sensitivity=latent_consumer_model.price_sensitivity,
        outside_option_threshold=latent_consumer_model.outside_option_threshold,
        turn_penalty=latent_consumer_model.turn_penalty,
        hidden_preference_slots=("comfort",),
        max_turns=8,
        initial_intent_reveal_ratio=0.0,
        initial_request_payload={"category": category_schema.category},
        initial_revealed_context={
            "category": category_schema.category,
            "budget_max": None,
            "must_have": {},
            "preference_weights": {},
        },
        category_schema=category_schema,
        user_profile=UserProfile(
            profile_id="v0_must_have_gate_user",
            initial_request_payload={"category": category_schema.category},
        ),
        latent_consumer_model=latent_consumer_model,
    )


def _write_outputs(
    output_dir: Path,
    *,
    records: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    traces: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "records.json", records)
    _write_json(output_dir / "summary.json", summaries)
    _write_json(output_dir / "trace.json", traces)
    (output_dir / "trace.md").write_text(_render_trace_markdown(traces), encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _render_trace_markdown(traces: Iterable[dict[str, Any]]) -> str:
    lines = ["# V0 Governance Demo Trace", ""]
    for trace in traces:
        lines.append(f"## {trace['scenario']} / {trace['arm']}")
        lines.append("")
        lines.append(str(trace["description"]))
        lines.append("")
        for step in trace["steps"]:
            lines.append(f"- turn {step['turn_index']}: {step['intervention']} ({step['reason']})")
            lines.append(f"  - raw: `{_compact_action_label(step['raw_action'])}`")
            lines.append(f"  - governed: `{_compact_action_label(step['governed_action'])}`")
            lines.append(f"  - response: `{step['response'].get('status')}`")
        lines.append("")
        lines.append(f"Final: `{trace['final_summary']['termination_reason']}`")
        lines.append("")
    return "\n".join(lines)


def _task_to_dict(task: SelectionTask) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "level": "v0_structured",
        "budget_max": task.user_goal.budget_max,
        "must_have": dict(task.user_goal.must_have),
        "hidden_preference_slots": list(task.hidden_preference_slots),
        "offers": [
            {
                "offer_id": offer.offer_id,
                "title": offer.title,
                "price": offer.price,
                "features": dict(offer.features),
                "attribute_values": dict(offer.attribute_values),
            }
            for offer in task.offers
        ],
    }


def _compact_summary(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "commit_success": info["commit_success"],
        "consumer_utility": info["consumer_utility"],
        "consumer_regret": info["consumer_regret"],
        "rounds": info["rounds"],
        "clarification_count": info["clarification_count"],
        "has_transient_violation": info["has_transient_violation"],
        "has_executed_violation": info["has_executed_violation"],
        "has_unrecovered_violation": info["has_unrecovered_violation"],
        "escalated": info["escalated"],
        "termination_reason": info["termination_reason"],
        "committed_offer_id": info["committed_offer_id"],
        "best_offer_id": info["best_offer_id"],
    }


def _action_to_dict(action: SelectionAction) -> dict[str, Any]:
    return {
        "action_type": action.action_type.value,
        "slot": action.slot,
        "offer_id": action.offer_id,
        "comparison_offer_id": action.comparison_offer_id,
        "explanation": action.explanation,
    }


def _compact_action_label(action: dict[str, Any]) -> str:
    action_type = action["action_type"]
    if action.get("slot"):
        return f"{action_type}({action['slot']})"
    if action.get("comparison_offer_id"):
        return f"{action_type}({action['offer_id']}, {action['comparison_offer_id']})"
    if action.get("offer_id"):
        return f"{action_type}({action['offer_id']})"
    return f"{action_type}()"


if __name__ == "__main__":
    main()
