"""V0 LLM-backed structured governance demo."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conversational_consumer_selection.agents import (
    build_platform_agent_context,
    build_platform_agent_system_prompt,
    build_platform_agent_user_prompt,
    DemoPlatformAgentModel,
    OpenAIPlatformAgentModel,
    PlatformAgentModel,
)
from conversational_consumer_selection.env import BestOfferSelectionEnv
from conversational_consumer_selection.governance import SelectionGovernedPolicy
from conversational_consumer_selection.metrics import build_episode_record
from conversational_consumer_selection.parsers import parse_action_payload
from conversational_consumer_selection.schemas import Observation, SelectionAction
from conversational_consumer_selection.tasks import make_v0_demo_task


@dataclass
class RawLLMDecisionTrace:
    """One ungoverned LLM decision trace."""

    context: dict[str, Any]
    system_prompt: str
    user_prompt: str
    raw_output: str
    action: SelectionAction
    working_belief: dict[str, Any] | None = None
    used_fallback: bool = False
    error: str | None = None


@dataclass
class TraceableLLMPolicy:
    """Adapter that parses raw LLM actions before governance.

    Unlike `LLMPlatformAgent`, this demo adapter deliberately does not validate
    actions against the observation. The governed wrapper is the validation and
    repair point being demonstrated.
    """

    model: PlatformAgentModel
    include_user_utterance_history: bool = False
    last_trace: RawLLMDecisionTrace | None = None

    def act(self, observation: Observation) -> SelectionAction:
        context = build_platform_agent_context(
            observation,
            include_user_utterance_history=self.include_user_utterance_history,
        )
        system_prompt = build_platform_agent_system_prompt()
        user_prompt = build_platform_agent_user_prompt(context)
        raw_output = ""
        try:
            raw_output = self.model.generate(system_prompt=system_prompt, user_prompt=user_prompt)
            payload = _load_json_object(raw_output)
            working_belief = _extract_working_belief(payload)
            action_payload = payload["next_action"] if "next_action" in payload else payload
            if not isinstance(action_payload, dict):
                raise ValueError("next_action must be a JSON object")
            action = parse_action_payload(action_payload)
            self.last_trace = RawLLMDecisionTrace(
                context=context,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_output=raw_output,
                action=action,
                working_belief=working_belief,
            )
            return action
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            fallback = SelectionAction.escalate("invalid_raw_model_output")
            self.last_trace = RawLLMDecisionTrace(
                context=context,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_output=raw_output,
                action=fallback,
                used_fallback=True,
                error=str(exc),
            )
            return fallback


@dataclass
class CommitFirstDemoModel:
    """Mock LLM backend that intentionally makes an unsafe early commit."""

    offer_id: str = "offer_budget"

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        del system_prompt, user_prompt
        return json.dumps(
            {
                "working_belief": {
                    "demo_mode": "commit_first",
                    "risk": "skips clarification and recommendation",
                },
                "next_action": {
                    "action_type": "commit_selection",
                    "offer_id": self.offer_id,
                    "explanation": "mock LLM jumps directly to commitment",
                },
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a V0 structured LLM -> governance -> environment demo."
    )
    parser.add_argument(
        "--backend",
        choices=("demo", "commit_first_demo", "openai"),
        default="demo",
        help="Raw platform-agent backend.",
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
        "--output-dir",
        default="outputs/conversational_consumer_selection_v1/v0_llm_governance_demo",
        help="Directory used for record.json / trace.json / trace.md.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="Safety cap for the rollout.",
    )
    parser.add_argument(
        "--include-user-utterance-history",
        action="store_true",
        help="Include rendered user utterances in the LLM context.",
    )
    args = parser.parse_args()

    raw_policy = TraceableLLMPolicy(
        model=_build_model(args),
        include_user_utterance_history=args.include_user_utterance_history,
    )
    governed_policy = SelectionGovernedPolicy(raw_policy=raw_policy)

    record, trace = _run_demo(
        backend=args.backend,
        raw_policy=raw_policy,
        governed_policy=governed_policy,
        max_steps=args.max_steps,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "trace.json").write_text(
        json.dumps(trace, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "trace.md").write_text(_render_trace_markdown(trace), encoding="utf-8")

    print("V0 LLM governance demo record")
    print(json.dumps(record, indent=2, ensure_ascii=False))
    print()
    print(f"Wrote outputs to {output_dir.resolve()}")


def _build_model(args: argparse.Namespace) -> PlatformAgentModel:
    if args.backend == "openai":
        return OpenAIPlatformAgentModel(
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_tokens=args.max_tokens,
            base_url=args.base_url,
        )
    if args.backend == "commit_first_demo":
        return CommitFirstDemoModel()
    return DemoPlatformAgentModel()


def _load_json_object(raw_output: str) -> dict[str, Any]:
    text = raw_output.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("model output must be a JSON object")
    return payload


def _extract_working_belief(payload: dict[str, Any]) -> dict[str, Any] | None:
    working_belief = payload.get("working_belief")
    if working_belief is None:
        return None
    if not isinstance(working_belief, dict):
        raise ValueError("working_belief must be a JSON object")
    return dict(working_belief)


def _run_demo(
    *,
    backend: str,
    raw_policy: TraceableLLMPolicy,
    governed_policy: SelectionGovernedPolicy,
    max_steps: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task = make_v0_demo_task()
    env = BestOfferSelectionEnv()
    observation, _ = env.reset(task=task)
    steps: list[dict[str, Any]] = []

    for _ in range(max_steps):
        if observation.terminated:
            break
        governed_action = governed_policy.act(observation)
        model_trace = raw_policy.last_trace
        decision = governed_policy.last_decision
        observation, step_payoff, terminated, truncated, info = env.step(governed_action)
        steps.append(
            {
                "turn_index": observation.turn_index,
                "raw_model_output": model_trace.raw_output if model_trace is not None else "",
                "used_model_fallback": bool(model_trace.used_fallback) if model_trace is not None else False,
                "model_error": model_trace.error if model_trace is not None else None,
                "working_belief": dict(model_trace.working_belief)
                if model_trace is not None and model_trace.working_belief is not None
                else None,
                "raw_action": _action_to_dict(decision.raw_action)
                if decision is not None
                else _action_to_dict(governed_action),
                "governed_action": _action_to_dict(governed_action),
                "intervention": decision.intervention if decision is not None else "none",
                "reason": decision.reason if decision is not None else "no_governance_decision",
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
    record = build_episode_record(info, arm=f"llm_governed:{backend}", setting="v0_structured")
    record["intervention_count"] = sum(
        1 for step in steps if step["intervention"] not in {"none", "pass"}
    )
    trace = {
        "backend": backend,
        "task_id": task.task_id,
        "setting": "v0_structured",
        "mode_note": (
            "LLM/raw backend selects structured actions; SelectionGovernedPolicy gates "
            "or rewrites them before env.step(). This is a demo vignette, not a full "
            "single-vs-OCL experiment."
        ),
        "steps": steps,
        "final_summary": _compact_summary(info),
    }
    return record, trace


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


def _render_trace_markdown(trace: dict[str, Any]) -> str:
    lines = [
        "# V0 LLM Governance Demo Trace",
        "",
        str(trace["mode_note"]),
        "",
        f"- backend: `{trace['backend']}`",
        f"- setting: `{trace['setting']}`",
        f"- task_id: `{trace['task_id']}`",
        "",
    ]
    for step in trace["steps"]:
        lines.append(f"## Turn {step['turn_index']}")
        lines.append("")
        lines.append(f"- intervention: `{step['intervention']}`")
        lines.append(f"- reason: `{step['reason']}`")
        lines.append(f"- raw action: `{_compact_action_label(step['raw_action'])}`")
        lines.append(f"- governed action: `{_compact_action_label(step['governed_action'])}`")
        lines.append(f"- response: `{step['response'].get('status')}`")
        if step["used_model_fallback"]:
            lines.append(f"- model fallback: `{step['model_error']}`")
        lines.append("")
    lines.append(f"Final: `{trace['final_summary']['termination_reason']}`")
    lines.append("")
    return "\n".join(lines)


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
