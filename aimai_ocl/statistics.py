"""Statistical and metric utilities for experiment evaluation.

Provides bootstrap confidence intervals, sign-flip permutation tests,
paired metric delta computation, and batch metric aggregation used by
evaluation scripts and the offline experiment protocol.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any

from aimai_ocl.schemas import AuditEventType, EpisodeTrace, ViolationType


def mean(values: list[float]) -> float | None:
    """Compute arithmetic mean, returning None for empty inputs."""
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def to_float(value: Any) -> float:
    """Best-effort float conversion, returning 0.0 on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def to_float_or_none(value: Any) -> float | None:
    """Best-effort float conversion, returning None on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bootstrap_ci_mean(
    deltas: list[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Compute bootstrap 95% confidence interval for the mean of *deltas*.

    Args:
        deltas: Paired difference values.
        samples: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.

    Returns:
        Dict with method, samples, confidence_level, lower, and upper keys.
    """
    if samples <= 0:
        raise ValueError("bootstrap samples must be > 0")
    n = len(deltas)
    if n == 0:
        return {
            "method": "bootstrap",
            "samples": 0,
            "confidence_level": 0.95,
            "lower": None,
            "upper": None,
        }
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(samples):
        acc = 0.0
        for _j in range(n):
            acc += deltas[rng.randrange(n)]
        means.append(acc / float(n))
    means.sort()
    lo_idx = max(0, min(samples - 1, int(math.floor(0.025 * (samples - 1)))))
    hi_idx = max(0, min(samples - 1, int(math.ceil(0.975 * (samples - 1)))))
    return {
        "method": "bootstrap",
        "samples": samples,
        "confidence_level": 0.95,
        "lower": means[lo_idx],
        "upper": means[hi_idx],
    }


def sign_flip_pvalues(
    deltas: list[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Compute sign-flip permutation p-values for paired deltas.

    Uses exact enumeration when n <= 20, otherwise Monte Carlo approximation.

    Args:
        deltas: Paired difference values.
        samples: Number of Monte Carlo permutations (used when n > 20).
        seed: RNG seed for reproducibility.

    Returns:
        Dict with method, samples, p_one_sided, and p_two_sided keys.
    """
    if samples <= 0:
        raise ValueError("permutation samples must be > 0")
    n = len(deltas)
    if n == 0:
        return {
            "method": "none",
            "samples": 0,
            "p_one_sided": None,
            "p_two_sided": None,
        }

    observed = sum(deltas) / float(n)
    threshold = observed - 1e-12
    abs_threshold = abs(observed) - 1e-12

    if n <= 20:
        total = 1 << n
        ge = 0
        abs_ge = 0
        for mask in range(total):
            acc = 0.0
            for idx, delta in enumerate(deltas):
                sign = 1.0 if ((mask >> idx) & 1) else -1.0
                acc += sign * delta
            value = acc / float(n)
            if value >= threshold:
                ge += 1
            if abs(value) >= abs_threshold:
                abs_ge += 1
        return {
            "method": "exact_sign_flip",
            "samples": total,
            "p_one_sided": ge / float(total),
            "p_two_sided": abs_ge / float(total),
        }

    rng = random.Random(seed)
    ge = 0
    abs_ge = 0
    for _ in range(samples):
        acc = 0.0
        for delta in deltas:
            sign = 1.0 if rng.random() < 0.5 else -1.0
            acc += sign * delta
        value = acc / float(n)
        if value >= threshold:
            ge += 1
        if abs(value) >= abs_threshold:
            abs_ge += 1
    return {
        "method": "monte_carlo_sign_flip",
        "samples": samples,
        "p_one_sided": ge / float(samples),
        "p_two_sided": abs_ge / float(samples),
    }


def paired_metric_deltas(
    *,
    records: list[dict[str, Any]],
    target_arm: str,
    baseline_arm: str,
    metric_key: str,
) -> list[float]:
    """Compute paired deltas (target - baseline) for a given metric.

    Pairs are matched by (episode_index, seed).
    """
    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for row in records:
        arm = str(row.get("arm"))
        if arm not in {target_arm, baseline_arm}:
            continue
        episode_index = to_float_or_none(row.get("episode_index"))
        seed = to_float_or_none(row.get("seed"))
        if episode_index is None or seed is None:
            continue
        pair_key = (int(episode_index), int(seed))
        grouped.setdefault(pair_key, {})[arm] = row

    deltas: list[float] = []
    for pair_rows in grouped.values():
        if target_arm not in pair_rows or baseline_arm not in pair_rows:
            continue
        target_value = to_float_or_none(pair_rows[target_arm].get(metric_key))
        baseline_value = to_float_or_none(pair_rows[baseline_arm].get(metric_key))
        if target_value is None or baseline_value is None:
            continue
        deltas.append(target_value - baseline_value)
    return deltas


def paired_metric_stats(
    *,
    records: list[dict[str, Any]],
    target_arm: str,
    baseline_arm: str,
    metric_key: str,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Compute full paired statistics for one metric across two arms.

    Returns dict with pairs count, mean_delta, delta_ci95, and
    sign_flip_pvalues.
    """
    deltas = paired_metric_deltas(
        records=records,
        target_arm=target_arm,
        baseline_arm=baseline_arm,
        metric_key=metric_key,
    )
    return {
        "pairs": len(deltas),
        "mean_delta": mean(deltas),
        "delta_ci95": bootstrap_ci_mean(
            deltas,
            samples=bootstrap_samples,
            seed=seed + 17,
        ),
        "sign_flip_pvalues": sign_flip_pvalues(
            deltas,
            samples=permutation_samples,
            seed=seed + 29,
        ),
    }


# ---------------------------------------------------------------------------
# Batch evaluation metric helpers (merged from evaluation_metrics.py)
# ---------------------------------------------------------------------------


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
    """
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
        Sorted list of summary dicts (one row per arm).
    """
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
    """
    return 1 if str(status).strip().lower() == "agreed" else 0


def _safe_mean(values: Any) -> float:
    """Compute mean of numeric iterable, returning ``0.0`` for empty inputs."""
    arr = [float(v) for v in values if v is not None]
    if not arr:
        return 0.0
    return sum(arr) / len(arr)


def _as_float(value: Any) -> float | None:
    """Best-effort float conversion for aggregation."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int:
    """Best-effort integer conversion for counters/flags."""
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_violation_map(value: Any) -> dict[str, int]:
    """Normalize violation-count payload into ``dict[str, int]``."""
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
