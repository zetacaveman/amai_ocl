#!/usr/bin/env python3
"""Run the paper-facing mechanism ablation matrix.

This script standardizes Phase 6 around four named OCL ablations:

- `ocl_full` (baseline)
- `w_o_role`
- `w_o_gate`
- `w_o_audit`
- `w_o_escalation`

Each experiment reuses `scripts/run_batch_eval.py` with the same
`single,ocl_full` comparison and then reads the canonical `main_result.json`
artifact. Outputs are standardized into:

- `ablation_summary.json`
- `ablation_rows.csv`
- `ablation_component_effects.csv`

Compatibility aliases are also written:

- `ablation_metrics.csv`
- `ablation_contributions.csv`
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


@dataclass(frozen=True, slots=True)
class AblationSpec:
    experiment_id: str
    component_removed: str
    description: str
    overrides: dict[str, str]


def _build_specs() -> list[AblationSpec]:
    return [
        AblationSpec(
            experiment_id="ocl_full",
            component_removed="none",
            description="Full OCL stack: role + gate + audit + escalation.",
            overrides={},
        ),
        AblationSpec(
            experiment_id="w_o_role",
            component_removed="role",
            description="Remove role decomposition via seller-only coordinator.",
            overrides={"--role-algorithm": "role_v1_seller_only"},
        ),
        AblationSpec(
            experiment_id="w_o_gate",
            component_removed="gate",
            description="Remove gate intervention via gate-off controller.",
            overrides={"--gate-algorithm": "gate_v0_off"},
        ),
        AblationSpec(
            experiment_id="w_o_audit",
            component_removed="audit",
            description="Remove audit trace recording.",
            overrides={"--audit-algorithm": "audit_v1_off"},
        ),
        AblationSpec(
            experiment_id="w_o_escalation",
            component_removed="escalation",
            description="Remove escalation/replan/handoff behavior.",
            overrides={"--escalation-algorithm": "escalation_v0_off"},
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the paper mechanism ablation matrix and summarize component loss."
        )
    )
    parser.add_argument("--episodes-per-arm", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--permutation-samples", type=int, default=20000)
    parser.add_argument(
        "--gate-tau",
        type=float,
        default=0.5,
        help="Shared gate_tau for baseline and ablations that still use the barrier gate.",
    )
    parser.add_argument(
        "--gate-algorithm",
        default=None,
        help=(
            "Optional gate algorithm override for the baseline and all ablations "
            "except ablations that explicitly replace the gate."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Root output dir. Defaults to outputs/ablation_matrix_<timestamp>."
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
        help="Skip one experiment if its main_result.json already exists.",
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
    spec: AblationSpec,
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
        "--gate-tau",
        str(args.gate_tau),
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
    if "--gate-algorithm" not in spec.overrides:
        _append_optional(cmd, "--gate-algorithm", args.gate_algorithm)
    for key, value in spec.overrides.items():
        cmd.extend([key, value])
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
            f"Main result unavailable for ablation run {path}: "
            f"{result.get('reason')}"
        )
    return result


def _build_ablation_rows(
    *,
    specs: list[AblationSpec],
    main_result_by_id: dict[str, dict[str, Any]],
    output_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        main_result = main_result_by_id[spec.experiment_id]
        for metric_row in list(main_result.get("rows") or []):
            rows.append(
                {
                    "experiment_id": spec.experiment_id,
                    "component_removed": spec.component_removed,
                    "description": spec.description,
                    "output_dir": str((output_root / spec.experiment_id).resolve()),
                    **metric_row,
                }
            )
    return rows


def _index_rows_by_experiment_metric(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row.get("experiment_id")), str(row.get("metric_key"))): row
        for row in rows
        if row.get("experiment_id") is not None and row.get("metric_key") is not None
    }


def _build_component_effect_rows(
    *,
    baseline_id: str,
    specs: list[AblationSpec],
    ablation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_key = _index_rows_by_experiment_metric(ablation_rows)
    baseline_metrics = {
        metric_key
        for experiment_id, metric_key in rows_by_key
        if experiment_id == baseline_id
    }
    effect_rows: list[dict[str, Any]] = []
    for spec in specs:
        if spec.experiment_id == baseline_id:
            continue
        for metric_key in sorted(baseline_metrics):
            baseline_row = rows_by_key.get((baseline_id, metric_key))
            ablation_row = rows_by_key.get((spec.experiment_id, metric_key))
            if baseline_row is None or ablation_row is None:
                continue
            baseline_improvement = _safe_float(
                baseline_row.get("improvement_ocl_vs_single")
            )
            ablation_improvement = _safe_float(
                ablation_row.get("improvement_ocl_vs_single")
            )
            component_benefit_loss = None
            if baseline_improvement is not None and ablation_improvement is not None:
                component_benefit_loss = baseline_improvement - ablation_improvement
            effect_rows.append(
                {
                    "baseline_id": baseline_id,
                    "ablation_id": spec.experiment_id,
                    "component_removed": spec.component_removed,
                    "metric_key": metric_key,
                    "metric_label": baseline_row.get("metric_label"),
                    "higher_is_better": baseline_row.get("higher_is_better"),
                    "direction_sign": baseline_row.get("direction_sign"),
                    "single_value": baseline_row.get("single_value"),
                    "baseline_ocl_full_value": baseline_row.get("ocl_full_value"),
                    "ablation_ocl_full_value": ablation_row.get("ocl_full_value"),
                    "baseline_delta_ocl_minus_single": baseline_row.get("delta_ocl_minus_single"),
                    "ablation_delta_ocl_minus_single": ablation_row.get("delta_ocl_minus_single"),
                    "baseline_improvement": baseline_improvement,
                    "ablation_improvement": ablation_improvement,
                    "component_benefit_loss": component_benefit_loss,
                }
            )
    return effect_rows


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _print_key_component_losses(rows: list[dict[str, Any]]) -> None:
    print("=== Ablation Component Loss ===")
    focus_metrics = {"success", "has_violation", "round", "cost_adjusted_welfare"}
    for row in rows:
        metric_key = str(row.get("metric_key"))
        if metric_key not in focus_metrics:
            continue
        print(
            " | ".join(
                [
                    f"ablation={row.get('ablation_id')}",
                    f"metric={metric_key}",
                    f"loss={row.get('component_benefit_loss')}",
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
    if not 0.0 <= float(args.gate_tau) <= 1.0:
        print("ERROR: --gate-tau must be in [0, 1]", file=sys.stderr)
        return 2
    if not BATCH_EVAL_SCRIPT.exists():
        print(f"ERROR: missing script {BATCH_EVAL_SCRIPT}", file=sys.stderr)
        return 2

    output_root = _resolve_output_root(args.output_root)
    specs = _build_specs()
    baseline_id = specs[0].experiment_id
    commands: dict[str, list[str]] = {}
    for spec in specs:
        run_dir = output_root / spec.experiment_id
        cmd = _build_batch_eval_cmd(spec=spec, args=args, output_dir=run_dir)
        commands[spec.experiment_id] = cmd
        print(f"[plan] {spec.experiment_id}: {' '.join(cmd)}")
        if args.dry_run:
            continue

        main_result_path = run_dir / "main_result.json"
        if args.resume and main_result_path.exists():
            print(f"[skip] {spec.experiment_id}: found {main_result_path}")
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
            "baseline_id": baseline_id,
            "experiments": [
                {
                    "experiment_id": spec.experiment_id,
                    "component_removed": spec.component_removed,
                    "description": spec.description,
                    "overrides": spec.overrides,
                    "command": commands[spec.experiment_id],
                }
                for spec in specs
            ],
        }
        print(json.dumps(plan_payload, ensure_ascii=True, sort_keys=True))
        return 0

    main_result_by_id: dict[str, dict[str, Any]] = {}
    for spec in specs:
        main_result_path = output_root / spec.experiment_id / "main_result.json"
        if not main_result_path.exists():
            print(
                f"ERROR: missing main result for {spec.experiment_id}: "
                f"{main_result_path}",
                file=sys.stderr,
            )
            return 2
        main_result_by_id[spec.experiment_id] = _extract_main_result(main_result_path)

    ablation_rows = _build_ablation_rows(
        specs=specs,
        main_result_by_id=main_result_by_id,
        output_root=output_root,
    )
    component_effect_rows = _build_component_effect_rows(
        baseline_id=baseline_id,
        specs=specs,
        ablation_rows=ablation_rows,
    )

    summary_payload = {
        "output_root": str(output_root),
        "baseline_id": baseline_id,
        "experiments": [
            {
                "experiment_id": spec.experiment_id,
                "component_removed": spec.component_removed,
                "description": spec.description,
                "overrides": spec.overrides,
                "output_dir": str((output_root / spec.experiment_id).resolve()),
                "command": commands[spec.experiment_id],
                "main_result": main_result_by_id.get(spec.experiment_id),
            }
            for spec in specs
        ],
        "ablation_rows": ablation_rows,
        "component_effect_rows": component_effect_rows,
    }

    summary_json = output_root / "ablation_summary.json"
    rows_csv = output_root / "ablation_rows.csv"
    effects_csv = output_root / "ablation_component_effects.csv"
    compat_metrics_csv = output_root / "ablation_metrics.csv"
    compat_contrib_csv = output_root / "ablation_contributions.csv"
    _write_json(summary_json, summary_payload)
    row_fieldnames = [
        "experiment_id",
        "component_removed",
        "description",
        "output_dir",
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
    ]
    effect_fieldnames = [
        "baseline_id",
        "ablation_id",
        "component_removed",
        "metric_key",
        "metric_label",
        "higher_is_better",
        "direction_sign",
        "single_value",
        "baseline_ocl_full_value",
        "ablation_ocl_full_value",
        "baseline_delta_ocl_minus_single",
        "ablation_delta_ocl_minus_single",
        "baseline_improvement",
        "ablation_improvement",
        "component_benefit_loss",
    ]
    _write_csv(rows_csv, ablation_rows, fieldnames=row_fieldnames)
    _write_csv(effects_csv, component_effect_rows, fieldnames=effect_fieldnames)
    _write_csv(compat_metrics_csv, ablation_rows, fieldnames=row_fieldnames)
    _write_csv(compat_contrib_csv, component_effect_rows, fieldnames=effect_fieldnames)

    print(f"output_root: {output_root}")
    print(f"summary_json: {summary_json}")
    print(f"rows_csv: {rows_csv}")
    print(f"effects_csv: {effects_csv}")
    _print_key_component_losses(component_effect_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
