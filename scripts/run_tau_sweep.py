#!/usr/bin/env python3
"""Run a tau sweep over tau-controlled organizational-control strength.

This script standardizes the Phase 5 experiment:

- fixed comparison: `single` vs `ocl_full`
- fixed gate family: `gate_v3_tau_controlled`
- varying control strength: `gate_tau`

For each tau value it invokes `scripts/run_batch_eval.py`, then reads the
standardized `main_result.json` artifact and produces:

- `tau_sweep_summary.json`
- `tau_sweep_metrics.csv` (long form)
- `tau_sweep_curve.csv` (paper-figure source)
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
BATCH_EVAL_SCRIPT = REPO_ROOT / "scripts" / "run_batch_eval.py"
DEFAULT_TAU_VALUES = "0.0,0.25,0.5,0.75,1.0"
FOCUS_METRICS: tuple[str, ...] = (
    "success",
    "has_violation",
    "round",
    "cost_adjusted_welfare",
)


@dataclass(frozen=True, slots=True)
class TauSpec:
    tau: float
    run_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a tau-controlled gate sweep and produce curve-ready summary files."
        )
    )
    parser.add_argument(
        "--tau-values",
        default=DEFAULT_TAU_VALUES,
        help=(
            "Comma-separated tau values in [0,1]. "
            f"Default: {DEFAULT_TAU_VALUES}."
        ),
    )
    parser.add_argument("--episodes-per-arm", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--permutation-samples", type=int, default=20000)
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Root output dir. Defaults to outputs/tau_sweep_<timestamp>."
        ),
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--initial-seller-price", type=float, default=None)
    parser.add_argument("--buyer-max-price", type=float, default=None)
    parser.add_argument("--seller-min-price", type=float, default=None)
    parser.add_argument("--user-requirement", default=None)
    parser.add_argument("--product-name", default=None)
    parser.add_argument("--product-price", type=float, default=None)
    parser.add_argument("--user-profile", default=None)
    parser.add_argument("--algorithm-bundle", default=None)
    parser.add_argument("--experiment-protocol", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip one tau run if its main_result.json already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and planned sweep only.",
    )
    return parser.parse_args()


def _parse_tau_values(raw: str) -> list[TauSpec]:
    values: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if token == "":
            continue
        try:
            tau = float(token)
        except ValueError as exc:
            raise ValueError(f"Invalid tau value '{token}'.") from exc
        if not 0.0 <= tau <= 1.0:
            raise ValueError(f"Tau value {tau} is outside [0, 1].")
        values.append(tau)
    if not values:
        raise ValueError("At least one tau value is required.")

    ordered = sorted(set(values))
    return [TauSpec(tau=tau, run_id=_tau_run_id(tau)) for tau in ordered]


def _tau_run_id(tau: float) -> str:
    return f"tau_{tau:.2f}".replace(".", "p")


def _resolve_output_root(raw: str | None) -> Path:
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (REPO_ROOT / "outputs" / f"tau_sweep_{stamp}").resolve()


def _append_optional(cmd: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    cmd.extend([key, str(value)])


def _build_batch_eval_cmd(
    *,
    spec: TauSpec,
    args: argparse.Namespace,
    output_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(BATCH_EVAL_SCRIPT),
        "--arms",
        "single,ocl_full",
        "--gate-algorithm",
        "gate_v3_tau_controlled",
        "--gate-tau",
        str(spec.tau),
        "--episodes-per-arm",
        str(args.episodes_per_arm),
        "--seed-base",
        str(args.seed_base),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--permutation-samples",
        str(args.permutation_samples),
        "--output-dir",
        str(output_dir),
    ]
    _append_optional(cmd, "--model", args.model)
    _append_optional(cmd, "--env-id", args.env_id)
    _append_optional(cmd, "--max-rounds", args.max_rounds)
    _append_optional(cmd, "--initial-seller-price", args.initial_seller_price)
    _append_optional(cmd, "--buyer-max-price", args.buyer_max_price)
    _append_optional(cmd, "--seller-min-price", args.seller_min_price)
    _append_optional(cmd, "--user-requirement", args.user_requirement)
    _append_optional(cmd, "--product-name", args.product_name)
    _append_optional(cmd, "--product-price", args.product_price)
    _append_optional(cmd, "--user-profile", args.user_profile)
    _append_optional(cmd, "--algorithm-bundle", args.algorithm_bundle)
    _append_optional(cmd, "--experiment-protocol", args.experiment_protocol)
    return cmd


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_main_result(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    result = payload.get("main_result", {})
    if not isinstance(result, dict):
        raise ValueError(f"Malformed main_result payload: {path}")
    if not bool(result.get("available")):
        raise ValueError(
            f"Main result unavailable for tau sweep run {path}: "
            f"{result.get('reason')}"
        )
    return result


def _extract_gate_runtime(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    records = payload.get("records", [])
    if not isinstance(records, list):
        return {}
    for row in records:
        if str(row.get("arm")) != "ocl_full":
            continue
        return {
            "gate_tau": row.get("gate_tau"),
            "rewrite_threshold": row.get("rewrite_threshold"),
            "block_threshold": row.get("block_threshold"),
            "epsilon_miss": row.get("epsilon_miss"),
        }
    return {}


def _index_metric_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("metric_key")): row
        for row in rows
        if row.get("metric_key") is not None
    }


def _build_metric_rows(
    *,
    tau_specs: list[TauSpec],
    main_result_by_run_id: dict[str, dict[str, Any]],
    gate_runtime_by_run_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in tau_specs:
        main_result = main_result_by_run_id[spec.run_id]
        gate_runtime = gate_runtime_by_run_id.get(spec.run_id, {})
        for metric_row in list(main_result.get("rows") or []):
            rows.append(
                {
                    "run_id": spec.run_id,
                    "tau": spec.tau,
                    "rewrite_threshold": gate_runtime.get("rewrite_threshold"),
                    "block_threshold": gate_runtime.get("block_threshold"),
                    "epsilon_miss": gate_runtime.get("epsilon_miss"),
                    **metric_row,
                }
            )
    return rows


def _build_curve_rows(
    *,
    tau_specs: list[TauSpec],
    main_result_by_run_id: dict[str, dict[str, Any]],
    gate_runtime_by_run_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in tau_specs:
        main_result = main_result_by_run_id[spec.run_id]
        gate_runtime = gate_runtime_by_run_id.get(spec.run_id, {})
        metric_rows = _index_metric_rows(list(main_result.get("rows") or []))
        row: dict[str, Any] = {
            "run_id": spec.run_id,
            "tau": spec.tau,
            "rewrite_threshold": gate_runtime.get("rewrite_threshold"),
            "block_threshold": gate_runtime.get("block_threshold"),
            "epsilon_miss": gate_runtime.get("epsilon_miss"),
        }
        for metric_key in FOCUS_METRICS:
            metric = metric_rows.get(metric_key, {})
            label = str(metric.get("metric_label", metric_key))
            row[f"single_{label}"] = metric.get("single_value")
            row[f"ocl_full_{label}"] = metric.get("ocl_full_value")
            row[f"{label}_delta_ocl_minus_single"] = metric.get("delta_ocl_minus_single")
            row[f"{label}_improvement"] = metric.get("improvement_ocl_vs_single")
            row[f"{label}_p_one_sided"] = metric.get("p_one_sided")
            row[f"{label}_p_two_sided"] = metric.get("p_two_sided")
        rows.append(row)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _print_curve_preview(rows: list[dict[str, Any]]) -> None:
    print("=== Tau Sweep Preview ===")
    for row in rows:
        print(
            " | ".join(
                [
                    f"tau={row.get('tau')}",
                    f"success={row.get('ocl_full_success_rate')}",
                    f"violation={row.get('ocl_full_violation_rate')}",
                    f"round={row.get('ocl_full_avg_round')}",
                    f"cost_adj_welfare={row.get('ocl_full_avg_cost_adjusted_welfare')}",
                ]
            )
        )


def main() -> int:
    args = parse_args()
    if args.episodes_per_arm <= 0:
        print("ERROR: --episodes-per-arm must be > 0", file=sys.stderr)
        return 2
    if args.bootstrap_samples <= 0:
        print("ERROR: --bootstrap-samples must be > 0", file=sys.stderr)
        return 2
    if args.permutation_samples <= 0:
        print("ERROR: --permutation-samples must be > 0", file=sys.stderr)
        return 2
    if not BATCH_EVAL_SCRIPT.exists():
        print(f"ERROR: missing script {BATCH_EVAL_SCRIPT}", file=sys.stderr)
        return 2
    try:
        tau_specs = _parse_tau_values(args.tau_values)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    output_root = _resolve_output_root(args.output_root)
    commands: dict[str, list[str]] = {}
    for spec in tau_specs:
        run_dir = output_root / spec.run_id
        cmd = _build_batch_eval_cmd(spec=spec, args=args, output_dir=run_dir)
        commands[spec.run_id] = cmd
        print(f"[plan] {spec.run_id}: {' '.join(cmd)}")
        if args.dry_run:
            continue

        main_result_path = run_dir / "main_result.json"
        if args.resume and main_result_path.exists():
            print(f"[skip] {spec.run_id}: found {main_result_path}")
            continue

        print(f"[run] {spec.run_id}")
        completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
        if completed.returncode != 0:
            print(
                f"ERROR: {spec.run_id} failed with exit code {completed.returncode}",
                file=sys.stderr,
            )
            return completed.returncode

    if args.dry_run:
        plan_payload = {
            "output_root": str(output_root),
            "tau_runs": [
                {
                    "run_id": spec.run_id,
                    "tau": spec.tau,
                    "command": commands[spec.run_id],
                }
                for spec in tau_specs
            ],
        }
        print(json.dumps(plan_payload, ensure_ascii=True, sort_keys=True))
        return 0

    main_result_by_run_id: dict[str, dict[str, Any]] = {}
    gate_runtime_by_run_id: dict[str, dict[str, Any]] = {}
    for spec in tau_specs:
        run_dir = output_root / spec.run_id
        main_result_path = run_dir / "main_result.json"
        runs_path = run_dir / "runs.json"
        if not main_result_path.exists():
            print(
                f"ERROR: missing main result for {spec.run_id}: {main_result_path}",
                file=sys.stderr,
            )
            return 2
        if not runs_path.exists():
            print(
                f"ERROR: missing runs.json for {spec.run_id}: {runs_path}",
                file=sys.stderr,
            )
            return 2
        main_result_by_run_id[spec.run_id] = _extract_main_result(main_result_path)
        gate_runtime_by_run_id[spec.run_id] = _extract_gate_runtime(runs_path)

    metric_rows = _build_metric_rows(
        tau_specs=tau_specs,
        main_result_by_run_id=main_result_by_run_id,
        gate_runtime_by_run_id=gate_runtime_by_run_id,
    )
    curve_rows = _build_curve_rows(
        tau_specs=tau_specs,
        main_result_by_run_id=main_result_by_run_id,
        gate_runtime_by_run_id=gate_runtime_by_run_id,
    )

    summary_payload = {
        "output_root": str(output_root),
        "tau_runs": [
            {
                "run_id": spec.run_id,
                "tau": spec.tau,
                "output_dir": str((output_root / spec.run_id).resolve()),
                "command": commands[spec.run_id],
                "gate_runtime": gate_runtime_by_run_id.get(spec.run_id, {}),
                "main_result": main_result_by_run_id.get(spec.run_id),
            }
            for spec in tau_specs
        ],
        "curve_rows": curve_rows,
        "metric_rows": metric_rows,
    }

    summary_json = output_root / "tau_sweep_summary.json"
    metrics_csv = output_root / "tau_sweep_metrics.csv"
    curve_csv = output_root / "tau_sweep_curve.csv"
    _write_json(summary_json, summary_payload)
    _write_csv(
        metrics_csv,
        metric_rows,
        fieldnames=[
            "run_id",
            "tau",
            "rewrite_threshold",
            "block_threshold",
            "epsilon_miss",
            "metric_key",
            "metric_label",
            "summary_key",
            "higher_is_better",
            "direction_sign",
            "single_value",
            "ocl_full_value",
            "delta_ocl_minus_single",
            "delta_ci95_lower",
            "delta_ci95_upper",
            "improvement_ocl_vs_single",
            "improvement_ci95_lower",
            "improvement_ci95_upper",
            "pairs",
            "p_one_sided",
            "p_two_sided",
            "pvalue_method",
            "pvalue_samples",
            "alternative",
        ],
    )
    _write_csv(
        curve_csv,
        curve_rows,
        fieldnames=[
            "run_id",
            "tau",
            "rewrite_threshold",
            "block_threshold",
            "epsilon_miss",
            "single_success_rate",
            "ocl_full_success_rate",
            "success_rate_delta_ocl_minus_single",
            "success_rate_improvement",
            "success_rate_p_one_sided",
            "success_rate_p_two_sided",
            "single_violation_rate",
            "ocl_full_violation_rate",
            "violation_rate_delta_ocl_minus_single",
            "violation_rate_improvement",
            "violation_rate_p_one_sided",
            "violation_rate_p_two_sided",
            "single_avg_round",
            "ocl_full_avg_round",
            "avg_round_delta_ocl_minus_single",
            "avg_round_improvement",
            "avg_round_p_one_sided",
            "avg_round_p_two_sided",
            "single_avg_cost_adjusted_welfare",
            "ocl_full_avg_cost_adjusted_welfare",
            "avg_cost_adjusted_welfare_delta_ocl_minus_single",
            "avg_cost_adjusted_welfare_improvement",
            "avg_cost_adjusted_welfare_p_one_sided",
            "avg_cost_adjusted_welfare_p_two_sided",
        ],
    )

    print(f"output_root: {output_root}")
    print(f"summary_json: {summary_json}")
    print(f"metrics_csv: {metrics_csv}")
    print(f"curve_csv: {curve_csv}")
    _print_curve_preview(curve_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
