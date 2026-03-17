#!/usr/bin/env python3
"""Run a paired single-vs-ocl_full experiment with discounted success score.

This script is intentionally scoped to a direct baseline comparison:

- arm A: single
- arm B: ocl_full

It runs paired units (same scenario/run config + same seed) and reports:

- discounted success points per episode
- per-arm aggregated metrics
- paired delta (ocl_full - single)
- bootstrap confidence interval for mean delta
- paired sign-flip permutation p-values

`--dry-run` prints the resolved plan without model calls.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import csv
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime
from enum import Enum
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

# Prefer vendored AgenticPay in this repo when present.
VENDORED_AGENTICPAY_ROOT = Path(__file__).resolve().parent.parent / "agenticpay"
VENDORED_AGENTICPAY_PKG = VENDORED_AGENTICPAY_ROOT / "agenticpay" / "__init__.py"
if VENDORED_AGENTICPAY_PKG.exists() and str(VENDORED_AGENTICPAY_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDORED_AGENTICPAY_ROOT))

# Allow direct execution via `python scripts/run_paired_single_vs_ocl.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aimai_ocl import (
    AUDIT_ALGORITHM_REGISTRY,
    RunConfig,
    ExperimentConfig,
    resolve_algorithm_bundle,
    resolve_arm,
    compose_algorithm_bundle,
    run_ocl_negotiation_episode,
    run_single_negotiation_episode,
)
from aimai_ocl.evaluation_metrics import (
    collect_violation_stats,
    success_from_status,
)
from aimai_ocl.model_runtime import (
    build_agenticpay_agents,
)

DEFAULT_RUN = RunConfig(
    model=os.getenv("AIMAI_MODEL", os.getenv("OPENAI_MODEL", RunConfig.model)),
)
DEFAULT_ORDER_SEED = 20260312
DEFAULT_DISCOUNT_LAST_ROUND_VALUE = 0.25
DEFAULT_SCORE_SCALE = 100.0


def parse_args() -> argparse.Namespace:
    """Parse CLI args for paired single-vs-ocl experiment."""
    parser = argparse.ArgumentParser(
        description=(
            "Run paired single vs ocl_full experiment "
            "with discounted-success scoring."
        ),
    )
    parser.add_argument(
        "--units",
        type=int,
        default=20,
        help="Number of paired units to run (default: 20).",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=DEFAULT_RUN.seed,
        help="Base seed. Unit i uses seed_base + i.",
    )
    parser.add_argument(
        "--order-seed",
        type=int,
        default=DEFAULT_ORDER_SEED,
        help="Seed controlling per-unit arm order randomization.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory for artifacts. Defaults to "
            "outputs/paired_single_vs_ocl_<timestamp>."
        ),
    )
    parser.add_argument(
        "--save-traces",
        action="store_true",
        help="Also write one trace JSON per episode run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved plan and exit without model calls.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-run progress and keep underlying runtime stdout.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_RUN.model,
        help="OpenAI model id, for example gpt-4o-mini.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SCORE_SCALE,
        help="Score scale multiplier. 100 means points in [0, 100].",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Round discount gamma in (0,1]. If omitted, inferred from --discount-last-round-value.",
    )
    parser.add_argument(
        "--discount-last-round-value",
        type=float,
        default=DEFAULT_DISCOUNT_LAST_ROUND_VALUE,
        help=(
            "If --gamma is omitted: target discount value at round=DEFAULT max_rounds, "
            "in (0,1]. Default: 0.25."
        ),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Bootstrap sample count for delta CI.",
    )
    parser.add_argument(
        "--permutation-samples",
        type=int,
        default=20000,
        help="Monte Carlo sign-flip samples when exact test is too large.",
    )
    parser.add_argument(
        "--audit-algorithm",
        default=None,
        choices=sorted(AUDIT_ALGORITHM_REGISTRY.keys()),
        help=(
            "Optional audit algorithm override for both arms "
            "(for example audit_v1_off / audit_v1_weak)."
        ),
    )
    return parser.parse_args()


def _build_run_config(args: argparse.Namespace) -> RunConfig:
    return replace(
        DEFAULT_RUN,
        model=args.model,
        seed=args.seed_base,
    )


def _resolve_output_dir(raw_path: str | None) -> Path:
    if raw_path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (REPO_ROOT / "outputs" / f"paired_single_vs_ocl_{stamp}").resolve()


def _resolve_gamma(args: argparse.Namespace) -> float:
    if args.gamma is not None:
        return float(args.gamma)
    if DEFAULT_RUN.max_rounds <= 1:
        return 1.0
    return float(
        args.discount_last_round_value
        ** (1.0 / float(DEFAULT_RUN.max_rounds - 1))
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.units <= 0:
        raise ValueError("--units must be > 0.")
    if args.scale <= 0:
        raise ValueError("--scale must be > 0.")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be > 0.")
    if args.permutation_samples <= 0:
        raise ValueError("--permutation-samples must be > 0.")
    if args.gamma is not None and not (0.0 < args.gamma <= 1.0):
        raise ValueError("--gamma must be in (0, 1].")
    if not (0.0 < args.discount_last_round_value <= 1.0):
        raise ValueError("--discount-last-round-value must be in (0, 1].")


def _apply_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        return None


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_jsonable(payload), ensure_ascii=True, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _build_agents(
    run_config: RunConfig,
) -> tuple[Any, Any]:
    buyer, seller, _runtime = build_agenticpay_agents(
        run_config=run_config,
    )
    return buyer, seller


def _discounted_success_points(
    *,
    success: int | bool | None,
    round_value: Any,
    gamma: float,
    scale: float,
) -> float:
    if int(success or 0) != 1:
        return 0.0
    try:
        round_num = int(round_value)
    except Exception:
        return 0.0
    if round_num < 1:
        round_num = 1
    return float(scale * (gamma ** float(round_num - 1)))


def _arm_order_for_unit(*, order_seed: int, unit_index: int) -> list[str]:
    order = ["single", "ocl_full"]
    if random.Random(order_seed + unit_index).random() < 0.5:
        order.reverse()
    return order


def _run_one(
    *,
    run_config: RunConfig,
    arm_name: str,
    seed: int,
    unit_index: int,
    traces_dir: Path | None,
    verbose: bool,
    audit_algorithm_override: str | None,
) -> dict[str, Any]:
    arm = resolve_arm(arm_name)
    # Design note:
    # Paired comparison is now strictly about architecture (`single` vs `ocl_full`);
    # no environment-side seller implementation branch is part of the contract.
    base_bundle = resolve_algorithm_bundle(arm.algorithm_bundle_id)
    algorithm_bundle = compose_algorithm_bundle(
        bundle_id=base_bundle.bundle_id,
        role_algorithm_id=arm.role_algorithm_id,
        gate_algorithm_id=arm.gate_algorithm_id,
        escalation_algorithm_id=arm.escalation_algorithm_id,
        audit_algorithm_id=(audit_algorithm_override or arm.audit_algorithm_id),
        attribution_algorithm_id=arm.attribution_algorithm_id,
    )

    seeded_run = RunConfig(
        env_id=run_config.env_id,
        model=run_config.model,
        seed=seed,
        max_rounds=run_config.max_rounds,
        initial_seller_price=run_config.initial_seller_price,
        buyer_max_price=run_config.buyer_max_price,
        seller_min_price=run_config.seller_min_price,
        user_requirement=run_config.user_requirement,
        product_name=run_config.product_name,
        product_price=run_config.product_price,
        user_profile=run_config.user_profile,
    )
    exp = ExperimentConfig(run=seeded_run, arm=arm)

    _apply_seed(seed)
    start = time.perf_counter()

    def _execute_one() -> tuple[Any, dict[str, Any]]:
        buyer, seller = _build_agents(
            seeded_run,
        )
        common_kwargs = {
            "env_id": seeded_run.env_id,
            "buyer_agent": buyer,
            "seller_agent": seller,
            "env_kwargs": {
                "max_rounds": seeded_run.max_rounds,
                "initial_seller_price": seeded_run.initial_seller_price,
                "buyer_max_price": seeded_run.buyer_max_price,
                "seller_min_price": seeded_run.seller_min_price,
            },
            "reset_kwargs": {
                "user_requirement": seeded_run.user_requirement,
                "product_info": {
                    "name": seeded_run.product_name,
                    "price": seeded_run.product_price,
                },
                "user_profile": seeded_run.user_profile,
            },
            "trace_metadata": {
                "arm_name": arm.name,
                "runner_mode": arm.runner_mode,
                "config_digest": exp.digest(),
                "seed": seeded_run.seed,
                "unit_index": unit_index,
                "paired_mode": True,
            },
        }
        if arm.runner_mode == "ocl":
            return run_ocl_negotiation_episode(
                **common_kwargs,
                controller=algorithm_bundle.make_gate_algorithm(),
                coordinator=algorithm_bundle.make_role_algorithm(),
                escalation_manager=algorithm_bundle.make_escalation_algorithm(),
                audit_policy=algorithm_bundle.make_audit_algorithm(),
            )
        return run_single_negotiation_episode(
            **common_kwargs,
            audit_policy=algorithm_bundle.make_audit_algorithm(),
        )

    if verbose:
        trace, final_info = _execute_one()
    else:
        with redirect_stdout(io.StringIO()):
            trace, final_info = _execute_one()
    latency_sec = time.perf_counter() - start

    violation = collect_violation_stats(trace, actor_id="seller")
    trace_path: str | None = None
    if traces_dir is not None:
        trace_file = traces_dir / (
            f"unit{unit_index:03d}_{arm.name}_seed{seed}_{trace.episode_id}.json"
        )
        _write_json(trace_file, trace)
        trace_path = str(trace_file.resolve())

    return {
        "unit_index": unit_index,
        "arm": arm.name,
        "runner_mode": arm.runner_mode,
        "algorithm_bundle_id": arm.algorithm_bundle_id,
        "role_algorithm_id": algorithm_bundle.role_algorithm_id,
        "gate_algorithm_id": algorithm_bundle.gate_algorithm_id,
        "escalation_algorithm_id": algorithm_bundle.escalation_algorithm_id,
        "audit_algorithm_id": algorithm_bundle.audit_algorithm_id,
        "attribution_algorithm_id": algorithm_bundle.attribution_algorithm_id,
        "experiment_protocol_id": arm.experiment_protocol_id,
        "seed": seeded_run.seed,
        "env_id": seeded_run.env_id,
        "model": seeded_run.model,
        "config_digest": exp.digest(),
        "episode_id": trace.episode_id,
        "status": final_info.get("status"),
        "success": success_from_status(final_info.get("status")),
        "round": final_info.get("round"),
        "termination_reason": final_info.get("termination_reason"),
        "seller_reward": final_info.get("seller_reward"),
        "buyer_reward": final_info.get("buyer_reward"),
        "global_score": final_info.get("global_score"),
        "seller_score": final_info.get("seller_score"),
        "buyer_score": final_info.get("buyer_score"),
        "latency_sec": latency_sec,
        "audit_events": len(trace.events),
        "failed_constraint_count": violation["failed_constraint_count"],
        "has_violation": int(violation["has_violation"]),
        "escalation_count": violation["escalation_count"],
        "violation_type_counts": violation["violation_type_counts"],
        "trace_json": trace_path,
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def _bootstrap_ci_mean(
    deltas: list[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
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


def _sign_flip_pvalues(
    deltas: list[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    n = len(deltas)
    if n == 0:
        return {
            "method": "none",
            "samples": 0,
            "p_one_sided": None,
            "p_two_sided": None,
        }

    observed = sum(deltas) / float(n)

    if n <= 20:
        total = 1 << n
        ge = 0
        abs_ge = 0
        threshold = observed - 1e-12
        abs_threshold = abs(observed) - 1e-12
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
    threshold = observed - 1e-12
    abs_threshold = abs(observed) - 1e-12
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


def _build_pair_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for row in records:
        unit = int(row["unit_index"])
        grouped.setdefault(unit, {})[str(row["arm"])] = row

    rows: list[dict[str, Any]] = []
    for unit in sorted(grouped.keys()):
        pair = grouped[unit]
        if "single" not in pair or "ocl_full" not in pair:
            continue
        single = pair["single"]
        ocl = pair["ocl_full"]
        rows.append(
            {
                "unit_index": unit,
                "seed": single["seed"],
                "single_score": single["discounted_success_points"],
                "ocl_score": ocl["discounted_success_points"],
                "delta_score": (
                    float(ocl["discounted_success_points"])
                    - float(single["discounted_success_points"])
                ),
                "single_success": single["success"],
                "ocl_success": ocl["success"],
                "delta_success": int(ocl["success"]) - int(single["success"]),
                "single_round": single["round"],
                "ocl_round": ocl["round"],
                "single_status": single["status"],
                "ocl_status": ocl["status"],
            }
        )
    return rows


def _build_summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    arm_metrics = summary.get("arm_metrics", {})
    for arm in ("single", "ocl_full"):
        metric = arm_metrics.get(arm, {})
        rows.append(
            {
                "row_type": "arm",
                "arm": arm,
                "units": metric.get("units"),
                "success_rate": metric.get("success_rate"),
                "avg_discounted_success_points": metric.get(
                    "avg_discounted_success_points"
                ),
                "avg_round_on_success": metric.get("avg_round_on_success"),
                "avg_round_all": metric.get("avg_round_all"),
                "avg_latency_sec": metric.get("avg_latency_sec"),
                "speed_factor_given_success": metric.get("speed_factor_given_success"),
                "mean_delta_score": None,
                "delta_ci_lower": None,
                "delta_ci_upper": None,
                "p_one_sided": None,
                "p_two_sided": None,
            }
        )
    effect = summary.get("paired_effect", {})
    ci = effect.get("delta_score_ci95", {})
    pvals = effect.get("sign_flip_pvalues", {})
    rows.append(
        {
            "row_type": "paired_effect",
            "arm": "ocl_full-single",
            "units": effect.get("pairs"),
            "success_rate": None,
            "avg_discounted_success_points": None,
            "avg_round_on_success": None,
            "avg_round_all": None,
            "avg_latency_sec": None,
            "speed_factor_given_success": None,
            "mean_delta_score": effect.get("mean_delta_score"),
            "delta_ci_lower": ci.get("lower"),
            "delta_ci_upper": ci.get("upper"),
            "p_one_sided": pvals.get("p_one_sided"),
            "p_two_sided": pvals.get("p_two_sided"),
        }
    )
    return rows


def _print_key_results(
    *,
    total_pairs: int,
    total_runs: int,
    summary: dict[str, Any],
) -> None:
    arm_metrics = summary.get("arm_metrics", {})
    single = arm_metrics.get("single", {})
    ocl = arm_metrics.get("ocl_full", {})
    effect = summary.get("paired_effect", {})
    ci = effect.get("delta_score_ci95", {})
    pvals = effect.get("sign_flip_pvalues", {})

    print("=== Key Results ===")
    print(f"units: {total_pairs}")
    print(f"total_runs: {total_runs}")
    print("metric: discounted_success_points")
    print(f"single: {single.get('avg_discounted_success_points')}")
    print(f"ocl_full: {ocl.get('avg_discounted_success_points')}")
    print(f"mean_delta_score (ocl-single): {effect.get('mean_delta_score')}")
    print(f"p_one_sided: {pvals.get('p_one_sided')}")
    print(f"p_two_sided: {pvals.get('p_two_sided')}")
    print(f"test_method: {pvals.get('method')}")
    print(f"delta_score_ci95: [{ci.get('lower')}, {ci.get('upper')}]")
    print(f"success_rate_single: {single.get('success_rate')}")
    print(f"success_rate_ocl_full: {ocl.get('success_rate')}")
    print(f"avg_round_single: {single.get('avg_round_all')}")
    print(f"avg_round_ocl_full: {ocl.get('avg_round_all')}")
    print(f"win_rate: {effect.get('win_rate')}")


def _format_seconds(total_sec: float) -> str:
    sec = max(0, int(total_sec))
    minutes, seconds = divmod(sec, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _print_progress(*, completed: int, total: int, started: float) -> None:
    now = time.perf_counter()
    elapsed = now - started
    pct = (100.0 * float(completed) / float(total)) if total > 0 else 0.0
    avg = elapsed / float(completed) if completed > 0 else 0.0
    eta = avg * float(max(total - completed, 0))
    print(
        f"progress: {completed}/{total} units ({pct:.1f}%) | "
        f"elapsed={_format_seconds(elapsed)} | eta={_format_seconds(eta)}",
        flush=True,
    )


def _summarize(
    *,
    records: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    gamma: float,
    scale: float,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> dict[str, Any]:
    arm_metrics: dict[str, dict[str, Any]] = {}
    for arm in ("single", "ocl_full"):
        subset = [row for row in records if row["arm"] == arm]
        success_values = [int(row["success"]) for row in subset]
        rounds_all = [float(row["round"]) for row in subset if row.get("round") is not None]
        rounds_success = [
            float(row["round"])
            for row in subset
            if int(row["success"]) == 1 and row.get("round") is not None
        ]
        points = [float(row["discounted_success_points"]) for row in subset]
        latency = [float(row["latency_sec"]) for row in subset if row.get("latency_sec") is not None]
        speed_factors = [
            float(row["discounted_success_points"]) / scale
            for row in subset
            if int(row["success"]) == 1
        ]
        arm_metrics[arm] = {
            "units": len(subset),
            "success_rate": _mean([float(v) for v in success_values]),
            "avg_discounted_success_points": _mean(points),
            "avg_round_on_success": _mean(rounds_success),
            "avg_round_all": _mean(rounds_all),
            "avg_latency_sec": _mean(latency),
            "speed_factor_given_success": _mean(speed_factors),
        }

    deltas = [float(row["delta_score"]) for row in pair_rows]
    wins = sum(1 for row in pair_rows if float(row["delta_score"]) > 0.0)
    ties = sum(1 for row in pair_rows if abs(float(row["delta_score"])) <= 1e-12)

    return {
        "metric": {
            "name": "discounted_success_points",
            "definition": "score = scale * 1(success) * gamma^(round-1)",
            "scale": scale,
            "gamma": gamma,
        },
        "arm_metrics": arm_metrics,
        "paired_effect": {
            "pairs": len(pair_rows),
            "mean_delta_score": _mean(deltas),
            "median_delta_score": (
                sorted(deltas)[len(deltas) // 2] if deltas else None
            ),
            "win_rate": (wins / float(len(pair_rows))) if pair_rows else None,
            "tie_rate": (ties / float(len(pair_rows))) if pair_rows else None,
            "delta_score_ci95": _bootstrap_ci_mean(
                deltas,
                samples=bootstrap_samples,
                seed=seed + 17,
            ),
            "sign_flip_pvalues": _sign_flip_pvalues(
                deltas,
                samples=permutation_samples,
                seed=seed + 29,
            ),
        },
    }


def main() -> int:
    args = parse_args()
    try:
        _validate_args(args)
        resolve_arm("single")
        resolve_arm("ocl_full")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        run_config = _build_run_config(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    gamma = _resolve_gamma(args)
    output_dir = _resolve_output_dir(args.output_dir)
    traces_dir = output_dir / "traces" if args.save_traces else None
    plan = {
        "arms": ["single", "ocl_full"],
        "units": args.units,
        "seed_base": args.seed_base,
        "order_seed": args.order_seed,
        "output_dir": str(output_dir),
        "save_traces": args.save_traces,
        "audit_algorithm_override": args.audit_algorithm,
        "metric": {
            "scale": args.scale,
            "gamma": gamma,
            "discount_last_round_value": args.discount_last_round_value,
        },
        "bootstrap_samples": args.bootstrap_samples,
        "permutation_samples": args.permutation_samples,
        "run_config": run_config.to_dict(),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=True, sort_keys=True))
        return 0

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for unit_index in range(args.units):
        seed = args.seed_base + unit_index
        arm_order = _arm_order_for_unit(order_seed=args.order_seed, unit_index=unit_index)
        for arm_name in arm_order:
            try:
                record = _run_one(
                    run_config=run_config,
                    arm_name=arm_name,
                    seed=seed,
                    unit_index=unit_index,
                    traces_dir=traces_dir,
                    verbose=args.verbose,
                    audit_algorithm_override=args.audit_algorithm,
                )
            except RuntimeError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 2

            record["arm_order"] = ",".join(arm_order)
            record["discount_gamma"] = gamma
            record["score_scale"] = args.scale
            record["discounted_success_points"] = _discounted_success_points(
                success=record["success"],
                round_value=record["round"],
                gamma=gamma,
                scale=args.scale,
            )
            records.append(record)
            if args.verbose:
                print(
                    f"[{arm_name}] unit={unit_index} seed={seed} "
                    f"status={record['status']} success={record['success']} "
                    f"round={record['round']} points={record['discounted_success_points']:.4f} "
                    f"latency_sec={record['latency_sec']:.3f}"
                )
        if not args.verbose:
            _print_progress(
                completed=unit_index + 1,
                total=args.units,
                started=started,
            )

    pair_rows = _build_pair_rows(records)
    summary = _summarize(
        records=records,
        pair_rows=pair_rows,
        gamma=gamma,
        scale=args.scale,
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
        seed=args.seed_base,
    )
    elapsed = time.perf_counter() - started

    output_dir.mkdir(parents=True, exist_ok=True)
    runs_json = output_dir / "runs.json"
    runs_csv = output_dir / "runs.csv"
    pairs_json = output_dir / "pairs.json"
    pairs_csv = output_dir / "pairs.csv"
    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"

    runs_for_csv: list[dict[str, Any]] = []
    for row in records:
        csv_row = dict(row)
        csv_row["violation_type_counts"] = json.dumps(
            row.get("violation_type_counts", {}),
            ensure_ascii=True,
            sort_keys=True,
        )
        runs_for_csv.append(csv_row)

    _write_json(
        runs_json,
        {
            "plan": plan,
            "total_runs": len(records),
            "elapsed_sec": elapsed,
            "records": records,
        },
    )
    _write_json(
        pairs_json,
        {
            "plan": plan,
            "pairs": pair_rows,
        },
    )
    _write_json(
        summary_json,
        {
            "plan": plan,
            "total_runs": len(records),
            "total_pairs": len(pair_rows),
            "elapsed_sec": elapsed,
            "summary": summary,
        },
    )

    _write_csv(
        runs_csv,
        runs_for_csv,
        fields=[
            "unit_index",
            "arm",
            "arm_order",
            "runner_mode",
            "algorithm_bundle_id",
            "role_algorithm_id",
            "gate_algorithm_id",
            "escalation_algorithm_id",
            "audit_algorithm_id",
            "attribution_algorithm_id",
            "experiment_protocol_id",
            "seed",
            "env_id",
            "model",
            "config_digest",
            "episode_id",
            "status",
            "success",
            "round",
            "termination_reason",
            "discounted_success_points",
            "discount_gamma",
            "score_scale",
            "seller_reward",
            "buyer_reward",
            "global_score",
            "seller_score",
            "buyer_score",
            "latency_sec",
            "audit_events",
            "failed_constraint_count",
            "has_violation",
            "escalation_count",
            "violation_type_counts",
            "trace_json",
        ],
    )
    _write_csv(
        pairs_csv,
        pair_rows,
        fields=[
            "unit_index",
            "seed",
            "single_score",
            "ocl_score",
            "delta_score",
            "single_success",
            "ocl_success",
            "delta_success",
            "single_round",
            "ocl_round",
            "single_status",
            "ocl_status",
        ],
    )
    _write_csv(
        summary_csv,
        _build_summary_rows(summary),
        fields=[
            "row_type",
            "arm",
            "units",
            "success_rate",
            "avg_discounted_success_points",
            "avg_round_on_success",
            "avg_round_all",
            "avg_latency_sec",
            "speed_factor_given_success",
            "mean_delta_score",
            "delta_ci_lower",
            "delta_ci_upper",
            "p_one_sided",
            "p_two_sided",
        ],
    )

    print(f"output_dir: {output_dir}")
    print(f"summary_json: {summary_json}")
    print(f"elapsed_sec: {elapsed:.3f}")
    _print_key_results(
        total_pairs=len(pair_rows),
        total_runs=len(records),
        summary=summary,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "total_runs": len(records),
                "total_pairs": len(pair_rows),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
