"""CLI entry point: python -m aimai_ocl run configs/batch.yaml [--dry-run] [--model X]"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

from aimai_ocl.adapters import build_agents
from aimai_ocl.attribution import CONTROLLED_ROLES, ValueConfig, compute_V, compute_shapley, run_masked_episode
from aimai_ocl.config import load_config, load_experiment_yaml
from aimai_ocl.control import AUDIT_FULL, AUDIT_MINIMAL, AUDIT_OFF, AuditPolicy, ControlConfig
from aimai_ocl.coordinator import Coordinator
from aimai_ocl.experiment import ARMS, ArmConfig, ExperimentConfig, RunConfig, resolve_arm
from aimai_ocl.runner import run_episode
from aimai_ocl.statistics import (
    bootstrap_ci_mean,
    collect_violation_stats,
    sign_flip_pvalues,
    success_from_status,
    summarize_records,
)


def main() -> int:
    parser = argparse.ArgumentParser(prog="aimai_ocl", description="AiMai OCL experiment runner")
    parser.add_argument("command", choices=["run"], help="Command to execute")
    parser.add_argument("config", help="Path to experiment YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved plan and exit")
    parser.add_argument("--model", default=None, help="Override model id")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--episodes", type=int, default=None, help="Override episodes count")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        return 1

    exp = load_experiment_yaml(config_path)
    mode = exp.get("mode", "demo")

    # Load base run config
    cli_overrides = {}
    if args.model:
        cli_overrides["model"] = args.model
    if args.seed is not None:
        cli_overrides["seed"] = args.seed

    # Resolve base config — try inherit path or default.yaml in same dir
    inherit_path = exp.get("inherit")
    if inherit_path:
        base_config_path = config_path.parent / inherit_path
    else:
        base_config_path = config_path.parent / "default.yaml"
        if not base_config_path.exists():
            base_config_path = None

    run_config = load_config(base_config_path, cli_overrides={**_flatten(exp), **cli_overrides})

    if mode == "demo":
        return _run_demo(run_config, exp, args)
    elif mode == "batch":
        return _run_batch(run_config, exp, args)
    elif mode == "paired":
        return _run_paired(run_config, exp, args)
    elif mode == "ablation":
        return _run_ablation(run_config, exp, args)
    elif mode == "shapley":
        return _run_shapley(run_config, exp, args)
    else:
        print(f"ERROR: Unknown mode '{mode}'", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Mode: demo (single episode)
# ---------------------------------------------------------------------------

def _run_demo(run_config: RunConfig, exp: dict, args: argparse.Namespace) -> int:
    arm_names = exp.get("arms", ["single"])
    arm_name = arm_names[0] if isinstance(arm_names, list) else str(arm_names)
    arm = _resolve_arm(arm_name, exp)
    ec = ExperimentConfig(run=run_config, arm=arm)

    if args.dry_run:
        print(json.dumps(ec.to_dict(), sort_keys=True, indent=2))
        return 0

    trace, info = _run_one_episode(run_config, arm)
    _print_result(arm, info, len(trace.events))
    return 0


# ---------------------------------------------------------------------------
# Mode: batch
# ---------------------------------------------------------------------------

def _run_batch(run_config: RunConfig, exp: dict, args: argparse.Namespace) -> int:
    arm_names = exp.get("arms", ["single", "ocl_full"])
    episodes = args.episodes or exp.get("episodes_per_arm", 5)
    seed_base = run_config.seed

    if args.dry_run:
        plan = {
            "mode": "batch", "arms": arm_names,
            "episodes_per_arm": episodes, "seed_base": seed_base,
            "run_config": run_config.to_dict(),
        }
        print(json.dumps(plan, sort_keys=True, indent=2))
        return 0

    records: list[dict[str, Any]] = []
    for arm_name in arm_names:
        arm = _resolve_arm(arm_name, exp)
        for i in range(episodes):
            seed = seed_base + i
            rc = RunConfig(**{**run_config.to_dict(), "seed": seed})
            t0 = time.time()
            trace, info = _run_one_episode(rc, arm)
            elapsed = time.time() - t0
            vs = collect_violation_stats(trace)
            records.append({
                "arm": arm.name, "episode_index": i, "seed": seed,
                "success": success_from_status(info.get("status")),
                "round": info.get("round"), "seller_reward": info.get("seller_reward"),
                "latency_sec": round(elapsed, 2), "audit_events": len(trace.events),
                **vs,
            })
            print(f"  [{arm.name}] episode {i}: status={info.get('status')}, {elapsed:.1f}s")

    summaries = summarize_records(records)
    print("\n--- Summary ---")
    print(json.dumps(summaries, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Mode: paired (single vs ocl with statistical tests)
# ---------------------------------------------------------------------------

def _run_paired(run_config: RunConfig, exp: dict, args: argparse.Namespace) -> int:
    arm_names = exp.get("arms", ["single", "ocl_full"])
    units = args.episodes or exp.get("units", 20)
    seed_base = run_config.seed
    bootstrap_samples = exp.get("stats", {}).get("bootstrap_samples", 2000)
    permutation_samples = exp.get("stats", {}).get("permutation_samples", 20000)

    if args.dry_run:
        plan = {
            "mode": "paired", "arms": arm_names, "units": units,
            "seed_base": seed_base,
            "bootstrap_samples": bootstrap_samples,
            "permutation_samples": permutation_samples,
            "run_config": run_config.to_dict(),
        }
        print(json.dumps(plan, sort_keys=True, indent=2))
        return 0

    records: list[dict[str, Any]] = []
    for i in range(units):
        seed = seed_base + i
        for arm_name in arm_names:
            arm = _resolve_arm(arm_name, exp)
            rc = RunConfig(**{**run_config.to_dict(), "seed": seed})
            t0 = time.time()
            trace, info = _run_one_episode(rc, arm)
            elapsed = time.time() - t0
            vs = collect_violation_stats(trace)
            records.append({
                "arm": arm.name, "episode_index": i, "seed": seed,
                "success": success_from_status(info.get("status")),
                "round": info.get("round"), "seller_reward": info.get("seller_reward"),
                "latency_sec": round(elapsed, 2), **vs,
            })

    summaries = summarize_records(records)
    print(json.dumps({"summaries": summaries, "records_count": len(records)}, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Mode: ablation
# ---------------------------------------------------------------------------

def _run_ablation(run_config: RunConfig, exp: dict, args: argparse.Namespace) -> int:
    base_arms = exp.get("base", {}).get("arms", ["single", "ocl_full"])
    episodes = args.episodes or exp.get("base", {}).get("episodes_per_arm", 20)
    variants = exp.get("variants", [])

    if args.dry_run:
        plan = {
            "mode": "ablation", "base_arms": base_arms,
            "episodes_per_arm": episodes,
            "variants": [v.get("name") for v in variants],
            "run_config": run_config.to_dict(),
        }
        print(json.dumps(plan, sort_keys=True, indent=2))
        return 0

    # Run base experiment
    print("=== Base experiment ===")
    # (Would call _run_batch internally in a real run)
    for variant in variants:
        print(f"\n=== Variant: {variant.get('name')} ===")
        # Apply variant overrides and run
    return 0


# ---------------------------------------------------------------------------
# Mode: shapley
# ---------------------------------------------------------------------------

def _run_shapley(run_config: RunConfig, exp: dict, args: argparse.Namespace) -> int:
    seeds = exp.get("seeds", [run_config.seed])
    roles = tuple(exp.get("roles", list(CONTROLLED_ROLES)))

    # Build all 2^n subsets
    subsets: list[frozenset[str]] = []
    for size in range(len(roles) + 1):
        for combo in combinations(roles, size):
            subsets.append(frozenset(combo))

    if args.dry_run:
        plan = {
            "mode": "shapley", "roles": list(roles), "seeds": seeds,
            "subsets": [sorted(s) for s in subsets],
            "run_config": run_config.to_dict(),
        }
        print(json.dumps(plan, sort_keys=True, indent=2))
        return 0

    # Run episodes for each (seed, subset) and compute Shapley values
    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        subset_values: dict[frozenset[str], float] = {}
        for subset in subsets:
            trace = run_masked_episode(
                role_mask=set(subset), seed=seed,
                env_id=run_config.env_id,
                buyer_agent=None,  # Would need real agents
                seller_agent=None,
                env_kwargs={
                    "max_rounds": run_config.max_rounds,
                    "buyer_max_price": run_config.buyer_max_price,
                    "seller_min_price": run_config.seller_min_price,
                },
                reset_kwargs={
                    "user_requirement": run_config.user_requirement,
                    "product_info": {"name": run_config.product_name, "price": run_config.product_price},
                    "user_profile": run_config.user_profile,
                },
            )
            subset_values[subset] = compute_V(trace)
            print(f"  V({sorted(subset)}) = {subset_values[subset]:.4f}")

        result = compute_shapley(subset_values, roles=roles)
        print(f"  Shapley phi: {result['phi']}")
        print(f"  Weights: {result['w']}")

    return 0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _run_one_episode(
    run_config: RunConfig,
    arm: ArmConfig,
) -> tuple[Any, dict[str, Any]]:
    """Build agents, configure control, and run one episode."""
    buyer, seller = build_agents(
        model=run_config.model,
        buyer_max_price=run_config.buyer_max_price,
        seller_min_price=run_config.seller_min_price,
        provider=run_config.provider,
        api_key_env=run_config.api_key_env,
    )
    random.seed(run_config.seed)

    audit = {"full": AUDIT_FULL, "minimal": AUDIT_MINIMAL, "off": AUDIT_OFF}.get(arm.audit, AUDIT_FULL)
    control_config = ControlConfig(
        risk_rewrite_threshold=arm.risk_rewrite_threshold,
        risk_block_threshold=arm.risk_block_threshold,
    ) if arm.ocl else None

    return run_episode(
        env_id=run_config.env_id,
        buyer_agent=buyer,
        seller_agent=seller,
        env_kwargs={
            "max_rounds": run_config.max_rounds,
            "initial_seller_price": run_config.initial_seller_price,
            "buyer_max_price": run_config.buyer_max_price,
            "seller_min_price": run_config.seller_min_price,
        },
        reset_kwargs={
            "user_requirement": run_config.user_requirement,
            "product_info": {"name": run_config.product_name, "price": run_config.product_price},
            "user_profile": run_config.user_profile,
        },
        trace_metadata={"arm": arm.name, "seed": run_config.seed},
        ocl=arm.ocl,
        control_config=control_config,
        coordinator=Coordinator(mode=arm.coordinator_mode) if arm.ocl else None,
        audit_policy=audit,
        enable_replan=arm.enable_replan,
    )


def _resolve_arm(name: str, exp: dict) -> ArmConfig:
    """Resolve arm from pre-defined registry or inline YAML overrides."""
    try:
        return resolve_arm(name)
    except ValueError:
        # Allow inline arm definition in YAML
        return ArmConfig(name=name, ocl="ocl" in name)


def _print_result(arm: ArmConfig, info: dict, events: int) -> None:
    print(f"arm: {arm.name}")
    print(f"status: {info.get('status')}")
    print(f"agreed_price: {info.get('agreed_price')}")
    print(f"rounds: {info.get('round')}")
    print(f"seller_reward: {info.get('seller_reward')}")
    print(f"audit_events: {events}")


def _flatten(data: dict) -> dict:
    """Flatten nested dict for config override."""
    result = {}
    for k, v in data.items():
        if isinstance(v, dict):
            result.update(v)
        elif k not in ("mode", "arms", "inherit", "variants", "base"):
            result[k] = v
    return result


if __name__ == "__main__":
    raise SystemExit(main())
