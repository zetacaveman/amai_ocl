"""Shared metric utilities for batch experiment evaluation.

中文翻译：Shared metric utilities for batch experiment evaluation。"""

from __future__ import annotations

import json
from typing import Any

from aimai_ocl.schemas.audit import AuditEventType, EpisodeTrace
from aimai_ocl.schemas.constraints import ViolationType


def collect_violation_stats(
    trace: EpisodeTrace,
    *,
    actor_id: str = "seller",
) -> dict[str, Any]:
    """Collect seller-side violation and escalation stats from one trace.

    Input:
        trace:
            One completed episode trace.
        actor_id:
            Actor id filter for evaluation target. Defaults to seller-side
            evaluation.

    Output:
        Dict with keys:
        - ``failed_constraint_count``: number of failed checks
        - ``has_violation``: whether any failed check exists
        - ``violation_type_counts``: per-type failed-check counts
        - ``escalation_count``: count of escalation events

    Note:
        Failed checks are counted from ``CONSTRAINT_EVALUATED`` events only, to
        avoid double-counting checks that may also appear in ACTION_EXECUTED.
    

    中文翻译：收集 seller-side violation and escalation stats from one trace。"""
    violation_type_counts: dict[str, int] = {}
    failed_constraint_count = 0
    escalation_count = 0

    target_actor = str(actor_id).strip().lower()
    for event in trace.events:
        event_actor = str(event.actor_id or "").strip().lower()
        if event_actor != target_actor:
            continue

        if event.event_type == AuditEventType.ESCALATION_TRIGGERED:
            escalation_count += 1
            continue

        if event.event_type != AuditEventType.CONSTRAINT_EVALUATED:
            continue

        for check in event.constraint_checks:
            if check.passed:
                continue
            failed_constraint_count += 1
            key = (
                check.violation_type.value
                if check.violation_type is not None
                else ViolationType.UNKNOWN.value
            )
            violation_type_counts[key] = violation_type_counts.get(key, 0) + 1

    return {
        "failed_constraint_count": failed_constraint_count,
        "has_violation": failed_constraint_count > 0,
        "violation_type_counts": violation_type_counts,
        "escalation_count": escalation_count,
    }


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-run records into arm-level summary rows.

    Input:
        records:
            Flat list of per-run records containing at least:
            ``arm``, ``success``, ``has_violation``, ``round``,
            ``seller_reward``, ``latency_sec``, and ``violation_type_counts``.

    Output:
        Sorted list of summary dicts (one row per arm) including:
        ``episodes``, ``success_rate``, ``violation_rate``, ``avg_round``,
        ``avg_seller_reward``, ``avg_latency_sec``, and aggregated
        ``violation_type_counts``.
    

    中文翻译：Aggregate per-run records into arm-level summary rows。"""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        arm = str(row.get("arm", ""))
        if arm == "":
            continue
        grouped.setdefault(arm, []).append(row)

    summaries: list[dict[str, Any]] = []
    for arm in sorted(grouped):
        rows = grouped[arm]
        episodes = len(rows)

        success_rate = _safe_mean(_as_int(row.get("success")) for row in rows)
        violation_rate = _safe_mean(_as_int(row.get("has_violation")) for row in rows)
        avg_round = _safe_mean(_as_float(row.get("round")) for row in rows)
        avg_seller_reward = _safe_mean(_as_float(row.get("seller_reward")) for row in rows)
        avg_latency_sec = _safe_mean(_as_float(row.get("latency_sec")) for row in rows)
        avg_audit_events = _safe_mean(_as_float(row.get("audit_events")) for row in rows)
        total_failed_constraints = int(
            sum(_as_int(row.get("failed_constraint_count")) for row in rows)
        )
        total_escalations = int(sum(_as_int(row.get("escalation_count")) for row in rows))

        violation_type_counts: dict[str, int] = {}
        for row in rows:
            per_row = _as_violation_map(row.get("violation_type_counts"))
            for key, count in per_row.items():
                violation_type_counts[key] = violation_type_counts.get(key, 0) + int(count)

        summaries.append(
            {
                "arm": arm,
                "episodes": episodes,
                "success_rate": success_rate,
                "violation_rate": violation_rate,
                "avg_round": avg_round,
                "avg_seller_reward": avg_seller_reward,
                "avg_latency_sec": avg_latency_sec,
                "avg_audit_events": avg_audit_events,
                "total_failed_constraints": total_failed_constraints,
                "total_escalations": total_escalations,
                "violation_type_counts": violation_type_counts,
            }
        )

    return summaries


def success_from_status(status: Any) -> int:
    """Convert terminal status text into a binary success flag.

    Input:
        status:
            Final status string from AgenticPay ``final_info``.

    Output:
        ``1`` if status is ``agreed`` (case-insensitive), else ``0``.
    

    中文翻译：转换 terminal status text into a binary success flag。"""
    return 1 if str(status).strip().lower() == "agreed" else 0


def _safe_mean(values: Any) -> float:
    """Compute mean of numeric iterable, returning ``0.0`` for empty inputs.

中文翻译：计算 mean of numeric iterable, returning ``0.0`` for empty inputs。"""
    arr = [float(v) for v in values if v is not None]
    if not arr:
        return 0.0
    return sum(arr) / len(arr)


def _as_float(value: Any) -> float | None:
    """Best-effort float conversion for aggregation.

中文翻译：Best-effort float conversion for aggregation。"""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int:
    """Best-effort integer conversion for counters/flags.

中文翻译：Best-effort integer conversion for counters/flags。"""
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_violation_map(value: Any) -> dict[str, int]:
    """Normalize violation-count payload into ``dict[str, int]``.

中文翻译：规范化 violation-count payload into ``dict[str, int]``。"""
    if isinstance(value, dict):
        return {str(k): int(v) for k, v in value.items()}
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(payload, dict):
            return {str(k): int(v) for k, v in payload.items()}
    return {}
