"""Episode-level metric helpers for the benchmark."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence


def build_episode_record(
    info: Mapping[str, Any],
    *,
    arm: str | None = None,
    setting: str | None = None,
) -> dict[str, Any]:
    """Normalize one terminal info payload into a record."""

    record = {
        "task_id": info["task_id"],
        "level": info["level"],
        "commit_success": float(bool(info["commit_success"])),
        "consumer_utility": float(info["consumer_utility"]),
        "consumer_regret": float(info["consumer_regret"]),
        "controller_payoff": float(info["controller_payoff"]),
        "rounds": int(info["rounds"]),
        "clarification_count": int(info["clarification_count"]),
        "has_transient_violation": float(bool(info["has_transient_violation"])),
        "has_executed_violation": float(bool(info["has_executed_violation"])),
        "has_unrecovered_violation": float(bool(info["has_unrecovered_violation"])),
        "escalated": float(bool(info["escalated"])),
        "termination_reason": info["termination_reason"],
    }
    if arm is not None:
        record["arm"] = arm
    if setting is not None:
        record["setting"] = setting
    return record


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    group_keys: Iterable[str] = ("arm", "setting"),
) -> list[dict[str, Any]]:
    """Aggregate episode records into grouped summaries."""

    if not records:
        return []

    group_keys_tuple = tuple(group_keys)
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[tuple(record.get(key) for key in group_keys_tuple)].append(record)

    summaries: list[dict[str, Any]] = []
    for key, group in grouped.items():
        count = len(group)
        summary: dict[str, Any] = {
            key_name: key_value for key_name, key_value in zip(group_keys_tuple, key)
        }
        summary.update(
            {
                "episodes": count,
                "commit_success_rate": _mean(group, "commit_success"),
                "avg_consumer_utility": _mean(group, "consumer_utility"),
                "avg_consumer_regret": _mean(group, "consumer_regret"),
                "avg_controller_payoff": _mean(group, "controller_payoff"),
                "avg_rounds": _mean(group, "rounds"),
                "avg_clarification_count": _mean(group, "clarification_count"),
                "transient_violation_rate": _mean(group, "has_transient_violation"),
                "executed_violation_rate": _mean(group, "has_executed_violation"),
                "unrecovered_violation_rate": _mean(group, "has_unrecovered_violation"),
                "escalation_rate": _mean(group, "escalated"),
            }
        )
        summaries.append(summary)

    summaries.sort(key=lambda row: tuple(str(row.get(key, "")) for key in group_keys_tuple))
    return summaries


def _mean(records: Sequence[Mapping[str, Any]], field_name: str) -> float:
    return sum(float(record[field_name]) for record in records) / float(len(records))
