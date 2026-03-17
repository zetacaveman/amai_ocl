#!/usr/bin/env python3
"""Run batch evaluation across configurable experiment arms.

中文翻译：运行 batch evaluation across configurable experiment arms。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime
from enum import Enum
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

# Prefer vendored AgenticPay in this repo when present.
# 中文：如果仓库内带有 vendored AgenticPay，则优先走本地导入路径。
VENDORED_AGENTICPAY_ROOT = Path(__file__).resolve().parent.parent / "agenticpay"
VENDORED_AGENTICPAY_PKG = VENDORED_AGENTICPAY_ROOT / "agenticpay" / "__init__.py"
if VENDORED_AGENTICPAY_PKG.exists() and str(VENDORED_AGENTICPAY_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDORED_AGENTICPAY_ROOT))

# Allow direct execution via `python scripts/run_batch_eval.py` from repo root.
# 中文：支持从仓库根目录直接执行，无需先安装为可编辑包。
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aimai_ocl import (
    ALGORITHM_BUNDLE_REGISTRY,
    ATTRIBUTION_ALGORITHM_REGISTRY,
    AUDIT_ALGORITHM_REGISTRY,
    EXPERIMENT_PROTOCOL_REGISTRY,
    GATE_ALGORITHM_REGISTRY,
    ROLE_ALGORITHM_REGISTRY,
    ESCALATION_ALGORITHM_REGISTRY,
    ExperimentConfig,
    RunConfig,
    compose_algorithm_bundle,
    resolve_algorithm_bundle,
    resolve_arm,
    resolve_experiment_protocol,
    run_ocl_negotiation_episode,
    run_single_negotiation_episode,
)
from aimai_ocl.evaluation_metrics import (
    collect_violation_stats,
    success_from_status,
    summarize_records,
)
from aimai_ocl.model_runtime import (
    build_agenticpay_agents,
)


DEFAULT_RUN = RunConfig(
    model=os.getenv("AIMAI_MODEL", os.getenv("OPENAI_MODEL", RunConfig.model)),
)


def parse_args() -> argparse.Namespace:
    """Parse CLI args for batch evaluation.

    Input:
        CLI flags from shell invocation.

    Output:
        Namespace with batch controls, run config overrides, and output options.
    

    中文翻译：解析 CLI args for batch evaluation。"""
    parser = argparse.ArgumentParser(
        description="Run batch evaluation over selected experiment arms.",
    )
    parser.add_argument(
        "--arms",
        default="single,ocl_full",
        help="Comma-separated arm list (default: single,ocl_full).",
    )
    parser.add_argument(
        "--episodes-per-arm",
        type=int,
        default=5,
        help="Number of episodes to run per arm.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=DEFAULT_RUN.seed,
        help="Base seed. Each episode uses seed_base + episode_index.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory for CSV/JSON reports. Defaults to "
            "outputs/batch_eval_<timestamp>."
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
        help="Print resolved batch plan and exit.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Bootstrap sample count for paired-delta confidence intervals.",
    )
    parser.add_argument(
        "--permutation-samples",
        type=int,
        default=20000,
        help="Sign-flip permutation sample count when exact test is too large.",
    )

    # Shared run config (kept fixed across arms for fair comparison).
    # 中文：这些运行参数在不同实验臂之间保持不变，确保对比公平。
    parser.add_argument("--env-id", default=DEFAULT_RUN.env_id)
    parser.add_argument("--model", default=DEFAULT_RUN.model)
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_RUN.max_rounds)
    parser.add_argument(
        "--initial-seller-price",
        type=float,
        default=DEFAULT_RUN.initial_seller_price,
    )
    parser.add_argument("--buyer-max-price", type=float, default=DEFAULT_RUN.buyer_max_price)
    parser.add_argument("--seller-min-price", type=float, default=DEFAULT_RUN.seller_min_price)
    parser.add_argument("--user-requirement", default=DEFAULT_RUN.user_requirement)
    parser.add_argument("--product-name", default=DEFAULT_RUN.product_name)
    parser.add_argument("--product-price", type=float, default=DEFAULT_RUN.product_price)
    parser.add_argument("--user-profile", default=DEFAULT_RUN.user_profile)
    parser.add_argument(
        "--algorithm-bundle",
        choices=sorted(ALGORITHM_BUNDLE_REGISTRY.keys()),
        default=None,
        help="Override algorithm bundle id for all selected arms.",
    )
    parser.add_argument(
        "--role-algorithm",
        choices=sorted(ROLE_ALGORITHM_REGISTRY.keys()),
        default=None,
        help="Optional role decomposition algorithm override for all arms.",
    )
    parser.add_argument(
        "--gate-algorithm",
        choices=sorted(GATE_ALGORITHM_REGISTRY.keys()),
        default=None,
        help="Optional risk gate algorithm override for all arms.",
    )
    parser.add_argument(
        "--escalation-algorithm",
        choices=sorted(ESCALATION_ALGORITHM_REGISTRY.keys()),
        default=None,
        help="Optional escalation algorithm override for all arms.",
    )
    parser.add_argument(
        "--audit-algorithm",
        choices=sorted(AUDIT_ALGORITHM_REGISTRY.keys()),
        default=None,
        help="Optional audit/trace policy algorithm override for all arms.",
    )
    parser.add_argument(
        "--attribution-algorithm",
        choices=sorted(ATTRIBUTION_ALGORITHM_REGISTRY.keys()),
        default=None,
        help="Optional attribution algorithm override for all arms.",
    )
    parser.add_argument(
        "--experiment-protocol",
        choices=sorted(EXPERIMENT_PROTOCOL_REGISTRY.keys()),
        default=None,
        help="Override experiment protocol id for all selected arms.",
    )
    return parser.parse_args()


def _parse_arms(raw: str) -> list[str]:
    """Resolve comma-separated arm names into validated registry keys.

    Input:
        raw:
            Comma-separated arm string from CLI.

    Output:
        Ordered validated arm names.
    

    中文翻译：解析 comma-separated arm names into validated registry keys。"""
    arms = [token.strip() for token in raw.split(",") if token.strip()]
    if not arms:
        raise ValueError("`--arms` must contain at least one arm name.")
    for arm in arms:
        resolve_arm(arm)
    return arms


def _build_base_run_config(args: argparse.Namespace) -> RunConfig:
    """Build run-level config shared by all arms in this batch.

    Input:
        Parsed CLI namespace.

    Output:
        ``RunConfig`` with fixed per-batch settings (seed set per episode).
    

    中文翻译：构建 run-level config shared by all arms in this batch。"""
    return RunConfig(
        env_id=args.env_id,
        model=args.model,
        seed=args.seed_base,
        max_rounds=args.max_rounds,
        initial_seller_price=args.initial_seller_price,
        buyer_max_price=args.buyer_max_price,
        seller_min_price=args.seller_min_price,
        user_requirement=args.user_requirement,
        product_name=args.product_name,
        product_price=args.product_price,
        user_profile=args.user_profile,
    )


def _resolve_output_dir(raw_path: str | None) -> Path:
    """Resolve output directory for batch artifacts.

    Input:
        raw_path:
            Optional user path. Relative paths are resolved against repo root.

    Output:
        Absolute ``Path`` to output directory.
    

    中文翻译：解析 output directory for batch artifacts。"""
    if raw_path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (REPO_ROOT / "outputs" / f"batch_eval_{stamp}").resolve()


def _apply_seed(seed: int) -> None:
    """Apply deterministic seed to Python and NumPy (when available).

中文翻译：应用 deterministic seed to Python and NumPy (when available)。"""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        return None


def _to_jsonable(value: Any) -> Any:
    """Convert nested objects into JSON-safe structures.

中文翻译：转换 nested objects into JSON-safe structures。"""
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
    """Write one JSON artifact with stable encoding settings.

中文翻译：写入 one JSON artifact with stable encoding settings。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_jsonable(payload), ensure_ascii=True, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Write one CSV artifact with fixed field order.

    Input:
        path:
            Destination CSV path.
        rows:
            Flat records to write.
        fields:
            Ordered field list.

    Output:
        Creates/overwrites CSV file.
    

    中文翻译：写入 one CSV artifact with fixed field order。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _build_agents(
    run_config: RunConfig,
) -> tuple[Any, Any]:
    """Create buyer and seller agents for one run.

    Input:
        run_config:
            Full run config containing model id and price bounds.

    Output:
        Tuple ``(buyer_agent, seller_agent)`` using AgenticPay classes.
    

    中文翻译：创建 buyer and seller agents for one run。"""
    buyer, seller, _runtime = build_agenticpay_agents(
        run_config=run_config,
    )
    return buyer, seller


def _run_one(
    *,
    run_config: RunConfig,
    arm_name: str,
    seed: int,
    episode_index: int,
    traces_dir: Path | None,
    algorithm_bundle_override: str | None,
    role_algorithm_override: str | None,
    gate_algorithm_override: str | None,
    escalation_algorithm_override: str | None,
    audit_algorithm_override: str | None,
    attribution_algorithm_override: str | None,
    experiment_protocol_override: str | None,
) -> dict[str, Any]:
    """Run one episode for one arm and return a flat metrics record.

    Input:
        run_config:
            Shared run-level config for this batch.
        arm_name:
            Arm key from ``ARM_REGISTRY``.
        seed:
            Deterministic run seed.
        episode_index:
            Zero-based episode index in this batch.
        traces_dir:
            Optional directory for per-run trace JSON export.
        algorithm_bundle_override:
            Optional algorithm bundle id applied to this arm run.
        role_algorithm_override:
            Optional role algorithm id override.
        gate_algorithm_override:
            Optional gate algorithm id override.
        escalation_algorithm_override:
            Optional escalation algorithm id override.
        audit_algorithm_override:
            Optional audit algorithm id override.
        attribution_algorithm_override:
            Optional attribution algorithm id override.
        experiment_protocol_override:
            Optional protocol bundle id applied to this arm run.

    Output:
        Flat record containing run metadata + requested metrics:
        success, violation, round, reward, latency.
    

    中文翻译：运行 one episode for one arm and return a flat metrics record。"""
    arm = resolve_arm(arm_name)
    # Design note:
    # `arm_name` now controls architecture only (single vs ocl).
    # Seller implementation is intentionally not a second axis anymore.
    arm = replace(
        arm,
        algorithm_bundle_id=algorithm_bundle_override or arm.algorithm_bundle_id,
        role_algorithm_id=role_algorithm_override or arm.role_algorithm_id,
        gate_algorithm_id=gate_algorithm_override or arm.gate_algorithm_id,
        escalation_algorithm_id=(
            escalation_algorithm_override or arm.escalation_algorithm_id
        ),
        audit_algorithm_id=(audit_algorithm_override or arm.audit_algorithm_id),
        attribution_algorithm_id=(
            attribution_algorithm_override or arm.attribution_algorithm_id
        ),
        experiment_protocol_id=(
            experiment_protocol_override or arm.experiment_protocol_id
        ),
    )
    base_bundle = resolve_algorithm_bundle(arm.algorithm_bundle_id)
    algorithm_bundle = compose_algorithm_bundle(
        bundle_id=base_bundle.bundle_id,
        role_algorithm_id=arm.role_algorithm_id,
        gate_algorithm_id=arm.gate_algorithm_id,
        escalation_algorithm_id=arm.escalation_algorithm_id,
        audit_algorithm_id=arm.audit_algorithm_id,
        attribution_algorithm_id=arm.attribution_algorithm_id,
    )
    experiment_protocol = resolve_experiment_protocol(arm.experiment_protocol_id)
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
    buyer, seller = _build_agents(
        seeded_run,
    )

    start = time.perf_counter()
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
            "episode_index": episode_index,
            "batch_mode": True,
        },
    }
    if arm.runner_mode == "ocl":
        trace, final_info = run_ocl_negotiation_episode(
            **common_kwargs,
            controller=algorithm_bundle.make_gate_algorithm(),
            coordinator=algorithm_bundle.make_role_algorithm(),
            escalation_manager=algorithm_bundle.make_escalation_algorithm(),
            audit_policy=algorithm_bundle.make_audit_algorithm(),
        )
    else:
        trace, final_info = run_single_negotiation_episode(
            **common_kwargs,
            audit_policy=algorithm_bundle.make_audit_algorithm(),
        )
    latency_sec = time.perf_counter() - start

    violation = collect_violation_stats(trace, actor_id="seller")
    trace_path: str | None = None
    if traces_dir is not None:
        trace_file = traces_dir / f"{arm.name}_ep{episode_index:03d}_seed{seed}_{trace.episode_id}.json"
        _write_json(trace_file, trace)
        trace_path = str(trace_file.resolve())

    record = {
        "arm": arm.name,
        "runner_mode": arm.runner_mode,
        "algorithm_bundle_id": arm.algorithm_bundle_id,
        "role_algorithm_id": algorithm_bundle.role_algorithm_id,
        "gate_algorithm_id": algorithm_bundle.gate_algorithm_id,
        "escalation_algorithm_id": algorithm_bundle.escalation_algorithm_id,
        "audit_algorithm_id": algorithm_bundle.audit_algorithm_id,
        "attribution_algorithm_id": algorithm_bundle.attribution_algorithm_id,
        "experiment_protocol_id": arm.experiment_protocol_id,
        "episode_index": episode_index,
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
    return record


def main() -> int:
    """Entry point for batch evaluation.

    Output:
        Returns process exit code and writes artifacts under output directory:
        ``runs.csv``, ``runs.json``, ``summary.csv``, ``summary.json``,
        and ``protocol_outputs.json``.
    

    中文翻译：batch evaluation 的入口函数。"""
    args = parse_args()
    try:
        arms = _parse_arms(args.arms)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.episodes_per_arm <= 0:
        print("ERROR: --episodes-per-arm must be > 0", file=sys.stderr)
        return 2
    if args.bootstrap_samples <= 0:
        print("ERROR: --bootstrap-samples must be > 0", file=sys.stderr)
        return 2
    if args.permutation_samples <= 0:
        print("ERROR: --permutation-samples must be > 0", file=sys.stderr)
        return 2
    try:
        if args.algorithm_bundle is not None:
            resolve_algorithm_bundle(args.algorithm_bundle)
        if any(
            value is not None
            for value in (
                args.role_algorithm,
                args.gate_algorithm,
                args.escalation_algorithm,
                args.audit_algorithm,
                args.attribution_algorithm,
            )
        ):
            compose_algorithm_bundle(
                bundle_id=args.algorithm_bundle or "v1_default",
                role_algorithm_id=args.role_algorithm,
                gate_algorithm_id=args.gate_algorithm,
                escalation_algorithm_id=args.escalation_algorithm,
                audit_algorithm_id=args.audit_algorithm,
                attribution_algorithm_id=args.attribution_algorithm,
            )
        if args.experiment_protocol is not None:
            resolve_experiment_protocol(args.experiment_protocol)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        base_run = _build_base_run_config(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    output_dir = _resolve_output_dir(args.output_dir)
    traces_dir = output_dir / "traces" if args.save_traces else None

    plan = {
        "arms": arms,
        "episodes_per_arm": args.episodes_per_arm,
        "seed_base": args.seed_base,
        "output_dir": str(output_dir),
        "save_traces": args.save_traces,
        "algorithm_bundle_override": args.algorithm_bundle,
        "role_algorithm_override": args.role_algorithm,
        "gate_algorithm_override": args.gate_algorithm,
        "escalation_algorithm_override": args.escalation_algorithm,
        "audit_algorithm_override": args.audit_algorithm,
        "attribution_algorithm_override": args.attribution_algorithm,
        "experiment_protocol_override": args.experiment_protocol,
        "bootstrap_samples": args.bootstrap_samples,
        "permutation_samples": args.permutation_samples,
        "base_run": base_run.to_dict(),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=True, sort_keys=True))
        return 0

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for episode_index in range(args.episodes_per_arm):
        seed = args.seed_base + episode_index
        for arm in arms:
            try:
                record = _run_one(
                    run_config=base_run,
                    arm_name=arm,
                    seed=seed,
                    episode_index=episode_index,
                    traces_dir=traces_dir,
                    algorithm_bundle_override=args.algorithm_bundle,
                    role_algorithm_override=args.role_algorithm,
                    gate_algorithm_override=args.gate_algorithm,
                    escalation_algorithm_override=args.escalation_algorithm,
                    audit_algorithm_override=args.audit_algorithm,
                    attribution_algorithm_override=args.attribution_algorithm,
                    experiment_protocol_override=args.experiment_protocol,
                )
            except RuntimeError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 2
            records.append(record)
            print(
                f"[{arm}] ep={episode_index} seed={seed} "
                f"status={record['status']} success={record['success']} "
                f"viol={record['has_violation']} round={record['round']} "
                f"bundle={record['algorithm_bundle_id']} "
                f"role_alg={record['role_algorithm_id']} "
                f"gate_alg={record['gate_algorithm_id']} "
                f"esc_alg={record['escalation_algorithm_id']} "
                f"audit_alg={record['audit_algorithm_id']} "
                f"attr_alg={record['attribution_algorithm_id']} "
                f"protocol={record['experiment_protocol_id']} "
                f"seller_reward={record['seller_reward']} "
                f"latency_sec={record['latency_sec']:.3f}"
            )

    summaries = summarize_records(records)
    protocol_outputs: dict[str, dict[str, Any]] = {}
    protocol_ids = sorted({row["experiment_protocol_id"] for row in records})
    for protocol_id in protocol_ids:
        protocol = resolve_experiment_protocol(protocol_id)
        protocol_outputs[protocol_id] = {
            "main": (
                protocol.run_main_fn(
                    records=records,
                    summaries=summaries,
                    plan=plan,
                    bootstrap_samples=args.bootstrap_samples,
                    permutation_samples=args.permutation_samples,
                    seed=args.seed_base,
                )
                if protocol.run_main_fn is not None
                else None
            ),
            "ablation": (
                protocol.run_ablation_fn(
                    records=records,
                    summaries=summaries,
                    plan=plan,
                )
                if protocol.run_ablation_fn is not None
                else None
            ),
            "adversarial": (
                protocol.run_adversarial_fn(
                    records=records,
                    summaries=summaries,
                    plan=plan,
                )
                if protocol.run_adversarial_fn is not None
                else None
            ),
            "repeated": (
                protocol.run_repeated_fn(
                    records=records,
                    summaries=summaries,
                    plan=plan,
                )
                if protocol.run_repeated_fn is not None
                else None
            ),
            "roi": (
                protocol.run_roi_fn(
                    records=records,
                    summaries=summaries,
                    plan=plan,
                )
                if protocol.run_roi_fn is not None
                else None
            ),
        }
    elapsed = time.perf_counter() - started

    output_dir.mkdir(parents=True, exist_ok=True)
    runs_json = output_dir / "runs.json"
    runs_csv = output_dir / "runs.csv"
    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    protocol_json = output_dir / "protocol_outputs.json"

    runs_for_csv = []
    for row in records:
        csv_row = dict(row)
        csv_row["violation_type_counts"] = json.dumps(
            row.get("violation_type_counts", {}),
            ensure_ascii=True,
            sort_keys=True,
        )
        runs_for_csv.append(csv_row)

    summary_for_csv = []
    for row in summaries:
        csv_row = dict(row)
        csv_row["violation_type_counts"] = json.dumps(
            row.get("violation_type_counts", {}),
            ensure_ascii=True,
            sort_keys=True,
        )
        summary_for_csv.append(csv_row)

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
        summary_json,
        {
            "plan": plan,
            "total_runs": len(records),
            "elapsed_sec": elapsed,
            "summary": summaries,
        },
    )
    _write_json(
        protocol_json,
        {
            "plan": plan,
            "total_runs": len(records),
            "protocol_outputs": protocol_outputs,
        },
    )
    _write_csv(
        runs_csv,
        runs_for_csv,
        fields=[
            "arm",
            "runner_mode",
            "algorithm_bundle_id",
            "role_algorithm_id",
            "gate_algorithm_id",
            "escalation_algorithm_id",
            "audit_algorithm_id",
            "attribution_algorithm_id",
            "experiment_protocol_id",
            "episode_index",
            "seed",
            "env_id",
            "model",
            "config_digest",
            "episode_id",
            "status",
            "success",
            "round",
            "termination_reason",
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
        summary_csv,
        summary_for_csv,
        fields=[
            "arm",
            "episodes",
            "success_rate",
            "violation_rate",
            "avg_round",
            "avg_seller_reward",
            "avg_latency_sec",
            "avg_audit_events",
            "total_failed_constraints",
            "total_escalations",
            "violation_type_counts",
        ],
    )

    print(f"output_dir: {output_dir}")
    print(f"runs_csv: {runs_csv}")
    print(f"runs_json: {runs_json}")
    print(f"summary_csv: {summary_csv}")
    print(f"summary_json: {summary_json}")
    print(f"protocol_json: {protocol_json}")
    print(f"total_runs: {len(records)}")
    print(f"elapsed_sec: {elapsed:.3f}")
    print(f"result_json: {json.dumps({'output_dir': str(output_dir), 'total_runs': len(records)}, ensure_ascii=True, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
