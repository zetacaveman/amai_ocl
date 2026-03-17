#!/usr/bin/env python3
"""Run a default ablation matrix and summarize component contributions.

This script executes a fixed set of OFAT (one-factor-at-a-time) ablations:

- E0: default
- E1: role seller-only
- E2: gate strict
- E3: gate lenient
- E4: escalation no-replan
- E5: audit weak/off

For each experiment it runs `scripts/run_batch_eval.py` with:

- arms = single,ocl_full
- shared seeds/config for fairness

Then it reads `protocol_outputs.json` and summarizes:

- per-metric paired deltas/p-values
- component contribution approximation:
  contribution_raw = delta_default - delta_ablation
  contribution_benefit = orient(delta_default) - orient(delta_ablation)
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

# +1 means larger delta is better; -1 means smaller delta is better.
METRIC_DIRECTIONS: dict[str, int] = {
    "success": 1,
    "seller_reward": 1,
    "has_violation": -1,
    "round": -1,
    "latency_sec": -1,
}


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    experiment_id: str
    description: str
    overrides: dict[str, str]


def _build_specs(audit_algorithm: str) -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            experiment_id="E0_default",
            description="Default bundle/components.",
            overrides={},
        ),
        ExperimentSpec(
            experiment_id="E1_role_seller_only",
            description="Role ablation with seller-only coordinator.",
            overrides={"--role-algorithm": "role_v1_seller_only"},
        ),
        ExperimentSpec(
            experiment_id="E2_gate_strict",
            description="Gate ablation with strict risk policy.",
            overrides={"--gate-algorithm": "gate_v1_strict"},
        ),
        ExperimentSpec(
            experiment_id="E3_gate_lenient",
            description="Gate ablation with lenient risk policy.",
            overrides={"--gate-algorithm": "gate_v1_lenient"},
        ),
        ExperimentSpec(
            experiment_id="E4_escalation_no_replan",
            description="Escalation ablation disabling replan.",
            overrides={"--escalation-algorithm": "escalation_v1_no_replan"},
        ),
        ExperimentSpec(
            experiment_id=f"E5_audit_{audit_algorithm}",
            description=f"Audit ablation using {audit_algorithm}.",
            overrides={"--audit-algorithm": audit_algorithm},
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run default ablation matrix (E0~E5) and summarize "
            "delta_default - delta_ablation contributions."
        )
    )
    parser.add_argument("--episodes-per-arm", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--permutation-samples", type=int, default=20000)
    parser.add_argument(
        "--audit-ablation",
        choices=("audit_v1_weak", "audit_v1_off"),
        default="audit_v1_weak",
        help="Which audit option to use for E5.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Root output dir. Defaults to "
            "outputs/ablation_matrix_<timestamp>."
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
        help="Skip experiment run if protocol_outputs.json already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and planned matrix only.",
    )
    return parser.parse_args()


def _resolve_output_root(raw: str | None) -> Path:
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (REPO_ROOT / "outputs" / f"ablation_matrix_{stamp}").resolve()


def _append_optional(cmd: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    cmd.extend([key, str(value)])


def _build_batch_eval_cmd(
    *,
    spec: ExperimentSpec,
    args: argparse.Namespace,
    output_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(BATCH_EVAL_SCRIPT),
        "--arms",
        "single,ocl_full",
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

    for key, value in spec.overrides.items():
        cmd.extend([key, value])
    return cmd


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_main_stats(protocol_outputs_path: Path) -> dict[str, Any]:
    payload = _load_json(protocol_outputs_path)
    protocol_outputs = payload.get("protocol_outputs", {})
    offline_v1 = protocol_outputs.get("offline_v1", {})
    main = offline_v1.get("main", {})
    paired = (main.get("paired_statistics") or {}).get("ocl_vs_single")
    return {
        "main": main,
        "paired": paired if isinstance(paired, dict) else None,
    }


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_metric_rows(
    *,
    specs: list[ExperimentSpec],
    main_stats_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        paired = main_stats_by_id.get(spec.experiment_id, {}).get("paired")
        metrics = (paired or {}).get("metrics", {})
        for metric, metric_stats in sorted(metrics.items()):
            pvals = metric_stats.get("sign_flip_pvalues", {})
            ci = metric_stats.get("delta_ci95", {})
            rows.append(
                {
                    "experiment_id": spec.experiment_id,
                    "description": spec.description,
                    "metric": metric,
                    "direction_sign": METRIC_DIRECTIONS.get(metric),
                    "pairs": metric_stats.get("pairs"),
                    "mean_delta": metric_stats.get("mean_delta"),
                    "ci_lower": ci.get("lower"),
                    "ci_upper": ci.get("upper"),
                    "p_one_sided": pvals.get("p_one_sided"),
                    "p_two_sided": pvals.get("p_two_sided"),
                    "test_method": pvals.get("method"),
                    "test_samples": pvals.get("samples"),
                }
            )
    return rows


def _build_contribution_rows(
    *,
    default_id: str,
    specs: list[ExperimentSpec],
    metric_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_exp_metric: dict[tuple[str, str], dict[str, Any]] = {}
    for row in metric_rows:
        by_exp_metric[(str(row["experiment_id"]), str(row["metric"]))] = row

    contribution_rows: list[dict[str, Any]] = []
    metrics = sorted({str(row["metric"]) for row in metric_rows})
    for spec in specs:
        if spec.experiment_id == default_id:
            continue
        for metric in metrics:
            default_row = by_exp_metric.get((default_id, metric))
            ablation_row = by_exp_metric.get((spec.experiment_id, metric))
            if default_row is None or ablation_row is None:
                continue
            delta_default = _safe_float(default_row.get("mean_delta"))
            delta_ablation = _safe_float(ablation_row.get("mean_delta"))
            direction = METRIC_DIRECTIONS.get(metric)

            contribution_raw = None
            contribution_benefit = None
            benefit_default = None
            benefit_ablation = None
            if delta_default is not None and delta_ablation is not None:
                contribution_raw = delta_default - delta_ablation
                if direction in (1, -1):
                    benefit_default = float(direction * delta_default)
                    benefit_ablation = float(direction * delta_ablation)
                    contribution_benefit = benefit_default - benefit_ablation

            contribution_rows.append(
                {
                    "default_id": default_id,
                    "ablation_id": spec.experiment_id,
                    "metric": metric,
                    "direction_sign": direction,
                    "delta_default": delta_default,
                    "delta_ablation": delta_ablation,
                    "contribution_raw": contribution_raw,
                    "benefit_default": benefit_default,
                    "benefit_ablation": benefit_ablation,
                    "contribution_benefit": contribution_benefit,
                }
            )
    return contribution_rows


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


def _print_key_contributions(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No contribution rows available.")
        return
    print("=== Component Contribution (benefit-oriented, larger is better) ===")
    focus = {"success", "has_violation", "round", "seller_reward", "latency_sec"}
    for row in rows:
        metric = str(row.get("metric"))
        if metric not in focus:
            continue
        print(
            f"{row.get('ablation_id')} | metric={metric} | "
            f"contribution_benefit={row.get('contribution_benefit')}"
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

    output_root = _resolve_output_root(args.output_root)
    specs = _build_specs(args.audit_ablation)
    default_id = specs[0].experiment_id

    commands: dict[str, list[str]] = {}
    for spec in specs:
        run_dir = output_root / spec.experiment_id
        cmd = _build_batch_eval_cmd(spec=spec, args=args, output_dir=run_dir)
        commands[spec.experiment_id] = cmd
        print(f"[plan] {spec.experiment_id}: {' '.join(cmd)}")
        if args.dry_run:
            continue

        protocol_path = run_dir / "protocol_outputs.json"
        if args.resume and protocol_path.exists():
            print(f"[skip] {spec.experiment_id}: found {protocol_path}")
            continue

        print(f"[run] {spec.experiment_id}")
        completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
        if completed.returncode != 0:
            print(
                f"ERROR: {spec.experiment_id} failed with exit code "
                f"{completed.returncode}",
                file=sys.stderr,
            )
            return completed.returncode

    if args.dry_run:
        plan_payload = {
            "output_root": str(output_root),
            "default_id": default_id,
            "experiments": [
                {
                    "experiment_id": spec.experiment_id,
                    "description": spec.description,
                    "overrides": spec.overrides,
                    "command": commands[spec.experiment_id],
                }
                for spec in specs
            ],
        }
        print(json.dumps(plan_payload, ensure_ascii=True, sort_keys=True))
        return 0

    main_stats_by_id: dict[str, dict[str, Any]] = {}
    for spec in specs:
        protocol_path = output_root / spec.experiment_id / "protocol_outputs.json"
        if not protocol_path.exists():
            print(
                f"ERROR: missing protocol outputs for {spec.experiment_id}: "
                f"{protocol_path}",
                file=sys.stderr,
            )
            return 2
        main_stats_by_id[spec.experiment_id] = _extract_main_stats(protocol_path)

    metric_rows = _build_metric_rows(specs=specs, main_stats_by_id=main_stats_by_id)
    contribution_rows = _build_contribution_rows(
        default_id=default_id,
        specs=specs,
        metric_rows=metric_rows,
    )

    summary_payload = {
        "output_root": str(output_root),
        "default_id": default_id,
        "experiments": [
            {
                "experiment_id": spec.experiment_id,
                "description": spec.description,
                "overrides": spec.overrides,
                "output_dir": str((output_root / spec.experiment_id).resolve()),
                "command": commands[spec.experiment_id],
                "main_stats": main_stats_by_id.get(spec.experiment_id, {}).get("main"),
            }
            for spec in specs
        ],
        "metric_rows": metric_rows,
        "contribution_rows": contribution_rows,
    }

    summary_json = output_root / "ablation_summary.json"
    metrics_csv = output_root / "ablation_metrics.csv"
    contrib_csv = output_root / "ablation_contributions.csv"
    _write_json(summary_json, summary_payload)
    _write_csv(
        metrics_csv,
        metric_rows,
        fieldnames=[
            "experiment_id",
            "description",
            "metric",
            "direction_sign",
            "pairs",
            "mean_delta",
            "ci_lower",
            "ci_upper",
            "p_one_sided",
            "p_two_sided",
            "test_method",
            "test_samples",
        ],
    )
    _write_csv(
        contrib_csv,
        contribution_rows,
        fieldnames=[
            "default_id",
            "ablation_id",
            "metric",
            "direction_sign",
            "delta_default",
            "delta_ablation",
            "contribution_raw",
            "benefit_default",
            "benefit_ablation",
            "contribution_benefit",
        ],
    )

    print(f"output_root: {output_root}")
    print(f"summary_json: {summary_json}")
    print(f"metrics_csv: {metrics_csv}")
    print(f"contributions_csv: {contrib_csv}")
    _print_key_contributions(contribution_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
