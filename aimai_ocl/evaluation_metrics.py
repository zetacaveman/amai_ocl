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
    """Collect seller-side constraint, violation, and escalation stats.

    Input:
        trace:
            One completed episode trace.
        actor_id:
            Actor id filter for evaluation target. Defaults to seller-side
            evaluation.

    Output:
        Dict with keys:
        - ``total_constraint_check_count``: number of evaluated checks
        - ``passed_constraint_count``: number of passing checks
        - ``failed_constraint_count``: number of failed checks
        - ``constraint_satisfaction_rate``: passed / total checks
        - ``has_violation``: alias of transient violation for backward compatibility
        - ``transient_has_violation``: whether any failed check exists in trajectory checks
        - ``executed_has_violation``: whether an approved executable action still carried failed checks
        - ``violation_type_counts``: per-type failed-check counts for transient checks
        - ``executed_violation_type_counts``: per-type failed-check counts on approved executable actions
        - ``escalation_count``: count of escalation events

    Note:
        Failed checks are counted from ``CONSTRAINT_EVALUATED`` events only, to
        avoid double-counting checks that may also appear in ACTION_EXECUTED.
    

    中文翻译：收集 seller-side violation and escalation stats from one trace。"""
    violation_type_counts: dict[str, int] = {}
    executed_violation_type_counts: dict[str, int] = {}
    total_constraint_check_count = 0
    passed_constraint_count = 0
    failed_constraint_count = 0
    executed_failed_constraint_count = 0
    escalation_count = 0

    target_actor = str(actor_id).strip().lower()
    for event in trace.events:
        event_actor = str(event.actor_id or "").strip().lower()
        if event_actor != target_actor:
            continue

        if event.event_type == AuditEventType.ESCALATION_TRIGGERED:
            escalation_count += 1
            continue

        if event.event_type == AuditEventType.CONSTRAINT_EVALUATED:
            for check in event.constraint_checks:
                total_constraint_check_count += 1
                if check.passed:
                    passed_constraint_count += 1
                    continue
                failed_constraint_count += 1
                key = (
                    check.violation_type.value
                    if check.violation_type is not None
                    else ViolationType.UNKNOWN.value
                )
                violation_type_counts[key] = violation_type_counts.get(key, 0) + 1
            continue

        if event.event_type != AuditEventType.ACTION_EXECUTED:
            continue

        exec_action = event.executable_action
        if exec_action is None or not bool(exec_action.approved):
            continue
        for check in event.constraint_checks:
            if check.passed:
                continue
            executed_failed_constraint_count += 1
            key = (
                check.violation_type.value
                if check.violation_type is not None
                else ViolationType.UNKNOWN.value
            )
            executed_violation_type_counts[key] = (
                executed_violation_type_counts.get(key, 0) + 1
            )

    transient_has_violation = failed_constraint_count > 0
    executed_has_violation = executed_failed_constraint_count > 0
    constraint_satisfaction_rate = _satisfaction_rate(
        passed_count=passed_constraint_count,
        total_count=total_constraint_check_count,
        has_violation=transient_has_violation,
    )
    return {
        "total_constraint_check_count": total_constraint_check_count,
        "passed_constraint_count": passed_constraint_count,
        "failed_constraint_count": failed_constraint_count,
        "executed_failed_constraint_count": executed_failed_constraint_count,
        "constraint_satisfaction_rate": constraint_satisfaction_rate,
        "has_violation": transient_has_violation,
        "transient_has_violation": transient_has_violation,
        "executed_has_violation": executed_has_violation,
        "violation_type_counts": violation_type_counts,
        "transient_violation_type_counts": violation_type_counts,
        "executed_violation_type_counts": executed_violation_type_counts,
        "escalation_count": escalation_count,
    }


def build_episode_metrics(
    *,
    trace: EpisodeTrace,
    final_info: dict[str, Any],
    latency_sec: float,
    actor_id: str = "seller",
    buyer_max_price: float | None = None,
    seller_min_price: float | None = None,
) -> dict[str, Any]:
    """Build one per-episode metric payload aligned with paper reporting.

    Input:
        trace:
            Completed episode trace.
        final_info:
            Terminal environment info payload.
        latency_sec:
            Observed wall-clock latency for this run.
        actor_id:
            Evaluation target actor id. Defaults to seller-side evaluation.
        buyer_max_price:
            Optional hard upper bound for final deal price.
        seller_min_price:
            Optional hard lower bound for final deal price.

    Output:
        Dict containing environment success, strict price feasibility,
        constraint metrics, welfare, and cost-adjusted welfare fields suitable
        for run records.

    中文翻译：构建与论文指标口径一致的单 episode 指标。"""
    constraint_stats = collect_violation_stats(trace, actor_id=actor_id)
    success = success_from_status(final_info.get("status"))
    round_value = _as_float(final_info.get("round"))
    global_score = _as_float(final_info.get("global_score"))
    seller_reward = _as_float(final_info.get("seller_reward"))
    welfare = (
        global_score
        if global_score is not None
        else (seller_reward if seller_reward is not None else 0.0)
    )
    cost_proxy_rounds = round_value if round_value is not None and round_value > 0 else 1.0
    cost_adjusted_welfare = welfare / cost_proxy_rounds
    price_feasible = price_feasibility_from_final_info(
        final_info=final_info,
        buyer_max_price=buyer_max_price,
        seller_min_price=seller_min_price,
    )
    strict_success = int(success == 1 and price_feasible == 1)

    transient_has_violation = bool(constraint_stats["transient_has_violation"])
    executed_has_violation = bool(constraint_stats["executed_has_violation"])
    recovered_has_violation = (
        transient_has_violation
        and strict_success == 1
        and not executed_has_violation
    )
    unrecovered_has_violation = executed_has_violation or (
        transient_has_violation and strict_success == 0
    )

    return {
        "success": success,
        "env_success": success,
        "feasibility": price_feasible,
        "price_feasible": price_feasible,
        "strict_success": strict_success,
        "round": final_info.get("round"),
        "latency_sec": float(latency_sec),
        "seller_reward": final_info.get("seller_reward"),
        "buyer_reward": final_info.get("buyer_reward"),
        "global_score": final_info.get("global_score"),
        "seller_score": final_info.get("seller_score"),
        "buyer_score": final_info.get("buyer_score"),
        "welfare": welfare,
        "cost_adjusted_welfare": cost_adjusted_welfare,
        "transient_has_violation": int(transient_has_violation),
        "executed_has_violation": int(executed_has_violation),
        "recovered_has_violation": int(recovered_has_violation),
        "unrecovered_has_violation": int(unrecovered_has_violation),
        **constraint_stats,
    }


def price_feasibility_from_final_info(
    *,
    final_info: dict[str, Any],
    buyer_max_price: float | None = None,
    seller_min_price: float | None = None,
) -> int:
    """Check strict final-price feasibility under buyer/seller bounds.

    AgenticPay's native ``status=agreed`` means the two offers are close
    enough under the environment's agreement rule. This helper is stricter:
    a successful deal is price-feasible only when its final agreed price lies
    inside the known buyer/seller private-value bounds.

    中文翻译：按最终成交价和买卖双方边界检查严格价格可行性。
    """
    if success_from_status(final_info.get("status")) != 1:
        return 0

    agreed_price = _as_float(final_info.get("agreed_price"))
    buyer_max = _as_float(buyer_max_price)
    seller_min = _as_float(seller_min_price)

    # Batch runners should provide all three values. The permissive fallback
    # keeps legacy callers from being marked infeasible only because they do
    # not yet pass bounds into the metric helper.
    if agreed_price is None or (buyer_max is None and seller_min is None):
        return 1
    if buyer_max is not None and agreed_price > buyer_max:
        return 0
    if seller_min is not None and agreed_price < seller_min:
        return 0
    return 1


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-run records into arm-level summary rows.

    Input:
        records:
            Flat list of per-run records containing at least:
            ``arm``, ``success``, ``has_violation``, ``transient_has_violation``,
            ``executed_has_violation``, ``unrecovered_has_violation``, ``round``,
            ``seller_reward``, ``latency_sec``, ``welfare``,
            ``cost_adjusted_welfare``, and ``violation_type_counts``.

    Output:
        Sorted list of summary dicts (one row per arm) including:
        ``episodes``, ``violation_rate`` (transient alias),
        ``executed_violation_rate``, ``unrecovered_violation_rate``,
        ``avg_round``,
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
        strict_success_rate = _safe_mean(
            _as_int(row.get("strict_success", row.get("feasibility", row.get("success"))))
            for row in rows
        )
        feasibility_rate = _safe_mean(_as_int(row.get("feasibility", row.get("success"))) for row in rows)
        violation_rate = _safe_mean(_as_int(row.get("has_violation")) for row in rows)
        transient_violation_rate = _safe_mean(
            _as_int(row.get("transient_has_violation", row.get("has_violation"))) for row in rows
        )
        executed_violation_rate = _safe_mean(
            _as_int(row.get("executed_has_violation")) for row in rows
        )
        recovered_violation_rate = _safe_mean(
            _as_int(row.get("recovered_has_violation")) for row in rows
        )
        unrecovered_violation_rate = _safe_mean(
            _as_int(row.get("unrecovered_has_violation")) for row in rows
        )
        avg_round = _safe_mean(_as_float(row.get("round")) for row in rows)
        avg_seller_reward = _safe_mean(_as_float(row.get("seller_reward")) for row in rows)
        avg_latency_sec = _safe_mean(_as_float(row.get("latency_sec")) for row in rows)
        avg_global_score = _safe_mean(_as_float(row.get("global_score")) for row in rows)
        avg_welfare = _safe_mean(_as_float(row.get("welfare", row.get("global_score"))) for row in rows)
        avg_cost_adjusted_welfare = _safe_mean(
            _as_float(row.get("cost_adjusted_welfare")) for row in rows
        )
        avg_constraint_satisfaction_rate = _safe_mean(
            _as_float(row.get("constraint_satisfaction_rate")) for row in rows
        )
        avg_audit_events = _safe_mean(_as_float(row.get("audit_events")) for row in rows)
        repair_rate = _safe_mean(
            _as_int(row.get("seller_repair_applied")) for row in rows
        )
        avg_seller_repair_count = _safe_mean(
            _as_float(row.get("seller_repair_count")) for row in rows
        )
        raw_price_violation_rate = _safe_mean(
            _as_int(row.get("seller_raw_price_violation_count")) for row in rows
        )
        post_repair_violation_rate = _safe_mean(
            _as_int(row.get("seller_post_repair_violation_count")) for row in rows
        )
        total_seller_repairs = int(
            sum(_as_int(row.get("seller_repair_count")) for row in rows)
        )
        avg_total_constraint_checks = _safe_mean(
            _as_float(row.get("total_constraint_check_count")) for row in rows
        )
        avg_passed_constraint_checks = _safe_mean(
            _as_float(row.get("passed_constraint_count")) for row in rows
        )
        avg_failed_constraint_count = _safe_mean(
            _as_float(row.get("failed_constraint_count")) for row in rows
        )
        avg_executed_failed_constraint_count = _safe_mean(
            _as_float(row.get("executed_failed_constraint_count")) for row in rows
        )
        avg_escalation_count = _safe_mean(
            _as_float(row.get("escalation_count")) for row in rows
        )
        total_failed_constraints = int(
            sum(_as_int(row.get("failed_constraint_count")) for row in rows)
        )
        total_executed_failed_constraints = int(
            sum(_as_int(row.get("executed_failed_constraint_count")) for row in rows)
        )
        total_escalations = int(sum(_as_int(row.get("escalation_count")) for row in rows))

        violation_type_counts: dict[str, int] = {}
        executed_violation_type_counts: dict[str, int] = {}
        for row in rows:
            per_row = _as_violation_map(row.get("violation_type_counts"))
            for key, count in per_row.items():
                violation_type_counts[key] = violation_type_counts.get(key, 0) + int(count)
            executed_per_row = _as_violation_map(row.get("executed_violation_type_counts"))
            for key, count in executed_per_row.items():
                executed_violation_type_counts[key] = (
                    executed_violation_type_counts.get(key, 0) + int(count)
                )

        summaries.append(
            {
                "arm": arm,
                "episodes": episodes,
                "success_rate": success_rate,
                "strict_success_rate": strict_success_rate,
                "feasibility_rate": feasibility_rate,
                "violation_rate": violation_rate,
                "transient_violation_rate": transient_violation_rate,
                "executed_violation_rate": executed_violation_rate,
                "recovered_violation_rate": recovered_violation_rate,
                "unrecovered_violation_rate": unrecovered_violation_rate,
                "avg_constraint_satisfaction_rate": avg_constraint_satisfaction_rate,
                "avg_round": avg_round,
                "avg_seller_reward": avg_seller_reward,
                "avg_latency_sec": avg_latency_sec,
                "avg_global_score": avg_global_score,
                "avg_welfare": avg_welfare,
                "avg_cost_adjusted_welfare": avg_cost_adjusted_welfare,
                "avg_audit_events": avg_audit_events,
                "repair_rate": repair_rate,
                "avg_seller_repair_count": avg_seller_repair_count,
                "raw_price_violation_rate": raw_price_violation_rate,
                "post_repair_violation_rate": post_repair_violation_rate,
                "total_seller_repairs": total_seller_repairs,
                "avg_total_constraint_checks": avg_total_constraint_checks,
                "avg_passed_constraint_checks": avg_passed_constraint_checks,
                "avg_failed_constraint_count": avg_failed_constraint_count,
                "avg_executed_failed_constraint_count": avg_executed_failed_constraint_count,
                "avg_escalation_count": avg_escalation_count,
                "total_failed_constraints": total_failed_constraints,
                "total_executed_failed_constraints": total_executed_failed_constraints,
                "total_escalations": total_escalations,
                "violation_type_counts": violation_type_counts,
                "executed_violation_type_counts": executed_violation_type_counts,
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


def _satisfaction_rate(
    *,
    passed_count: int,
    total_count: int,
    has_violation: bool,
) -> float:
    """Compute one normalized constraint-satisfaction rate in ``[0, 1]``.

    中文翻译：计算约束满足率。"""
    if total_count > 0:
        return float(passed_count) / float(total_count)
    return 0.0 if has_violation else 1.0
