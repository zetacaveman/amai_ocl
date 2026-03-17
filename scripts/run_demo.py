#!/usr/bin/env python3
"""Run one minimal AiMai OCL + AgenticPay negotiation episode.

中文翻译：运行 one minimal AiMai OCL + AgenticPay negotiation episode。"""

from __future__ import annotations

import argparse
from contextlib import suppress
from dataclasses import asdict, is_dataclass, replace
from enum import Enum
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

# Prefer vendored AgenticPay in this repo when present.
# 中文：如果仓库中存在 vendored AgenticPay，则优先使用本地实现。
VENDORED_AGENTICPAY_ROOT = Path(__file__).resolve().parent.parent / "agenticpay"
VENDORED_AGENTICPAY_PKG = VENDORED_AGENTICPAY_ROOT / "agenticpay" / "__init__.py"
if VENDORED_AGENTICPAY_PKG.exists() and str(VENDORED_AGENTICPAY_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDORED_AGENTICPAY_ROOT))

# Allow direct execution via `python scripts/run_demo.py` from repo root
# without requiring `pip install -e .` first. 中文：可在仓库根目录直接运行，
# 无需先执行 `pip install -e .`。
# 中文：支持在仓库根目录直接运行脚本。
# 中文：运行前不需要先执行 `pip install -e .`。
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aimai_ocl import (
    ALGORITHM_BUNDLE_REGISTRY,
    ATTRIBUTION_ALGORITHM_REGISTRY,
    AUDIT_ALGORITHM_REGISTRY,
    EXPERIMENT_PROTOCOL_REGISTRY,
    GATE_ALGORITHM_REGISTRY,
    ARM_REGISTRY,
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
from aimai_ocl.model_runtime import (
    build_agenticpay_agents,
)

DEFAULT_RUN = RunConfig(
    model=os.getenv("AIMAI_MODEL", os.getenv("OPENAI_MODEL", RunConfig.model)),
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for one experiment episode run.

    Returns:
        Parsed argparse namespace containing run config overrides and arm
        selection.
    

    中文翻译：解析 CLI arguments for one experiment episode run。"""
    parser = argparse.ArgumentParser(
        description="Run one reproducible AiMai OCL experiment episode.",
    )
    parser.add_argument(
        "--arm",
        choices=sorted(ARM_REGISTRY.keys()),
        default=None,
        help="Experiment arm name.",
    )
    parser.add_argument(
        "--mode",
        choices=("single", "ocl"),
        default=None,
        help="Deprecated alias for arm selection: single -> single, ocl -> ocl_full.",
    )

    # Shared run config (must stay fixed across arms for fair comparison).
    # 中文：运行层参数在各实验臂之间固定，避免实验比较出现混杂变量。
    parser.add_argument(
        "--env-id",
        default=DEFAULT_RUN.env_id,
        help="AgenticPay environment id.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_RUN.model,
        help="OpenAI model id (for example gpt-4o-mini).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RUN.seed)
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_RUN.max_rounds)
    parser.add_argument(
        "--initial-seller-price",
        type=float,
        default=DEFAULT_RUN.initial_seller_price,
    )
    parser.add_argument("--buyer-max-price", type=float, default=DEFAULT_RUN.buyer_max_price)
    parser.add_argument("--seller-min-price", type=float, default=DEFAULT_RUN.seller_min_price)
    parser.add_argument(
        "--user-requirement",
        default=DEFAULT_RUN.user_requirement,
    )
    parser.add_argument("--product-name", default=DEFAULT_RUN.product_name)
    parser.add_argument("--product-price", type=float, default=DEFAULT_RUN.product_price)
    parser.add_argument(
        "--user-profile",
        default=DEFAULT_RUN.user_profile,
    )
    parser.add_argument(
        "--algorithm-bundle",
        choices=sorted(ALGORITHM_BUNDLE_REGISTRY.keys()),
        default=None,
        help="Algorithm bundle id for role/gate/escalation/attribution stack.",
    )
    parser.add_argument(
        "--role-algorithm",
        choices=sorted(ROLE_ALGORITHM_REGISTRY.keys()),
        default=None,
        help="Optional role decomposition algorithm override.",
    )
    parser.add_argument(
        "--gate-algorithm",
        choices=sorted(GATE_ALGORITHM_REGISTRY.keys()),
        default=None,
        help="Optional risk gating algorithm override.",
    )
    parser.add_argument(
        "--escalation-algorithm",
        choices=sorted(ESCALATION_ALGORITHM_REGISTRY.keys()),
        default=None,
        help="Optional escalation/replan algorithm override.",
    )
    parser.add_argument(
        "--audit-algorithm",
        choices=sorted(AUDIT_ALGORITHM_REGISTRY.keys()),
        default=None,
        help="Optional audit/trace policy algorithm override.",
    )
    parser.add_argument(
        "--attribution-algorithm",
        choices=sorted(ATTRIBUTION_ALGORITHM_REGISTRY.keys()),
        default=None,
        help="Optional attribution algorithm override.",
    )
    parser.add_argument(
        "--experiment-protocol",
        choices=sorted(EXPERIMENT_PROTOCOL_REGISTRY.keys()),
        default=None,
        help="Experiment protocol bundle id (step-10 modular slot).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved experiment config and exit without running agents.",
    )
    parser.add_argument(
        "--trace-json",
        default=None,
        help=(
            "Optional path to write full episode trace JSON "
            "(for example outputs/trace.json)."
        ),
    )
    return parser.parse_args()


def _resolve_arm_name(args: argparse.Namespace) -> str:
    """Resolve arm name from ``--arm`` and legacy ``--mode`` arguments.

    Args:
        args: Parsed CLI args.

    Returns:
        Canonical arm name present in ``ARM_REGISTRY``.

    Raises:
        ValueError: If ``--arm`` and ``--mode`` conflict.
    

    中文翻译：解析 arm name from ``--arm`` and legacy ``--mode`` arguments。"""
    mode_to_arm = {"single": "single", "ocl": "ocl_full"}
    if args.mode is not None:
        mode_arm = mode_to_arm[args.mode]
        if args.arm is not None and args.arm != mode_arm:
            raise ValueError(
                f"Conflicting options: --arm {args.arm} and --mode {args.mode}.",
            )
        return mode_arm
    return args.arm or "single"


def _build_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    """Build an immutable experiment config from CLI args.

    Args:
        args: Parsed CLI args.

    Returns:
        Fully resolved ``ExperimentConfig`` with run-level and arm-level
        settings.
    

    中文翻译：构建 an immutable experiment config from CLI args。"""
    arm_name = _resolve_arm_name(args)
    arm_config = resolve_arm(arm_name)
    algorithm_bundle_id = args.algorithm_bundle or arm_config.algorithm_bundle_id
    experiment_protocol_id = (
        args.experiment_protocol or arm_config.experiment_protocol_id
    )
    arm_config = replace(
        arm_config,
        algorithm_bundle_id=algorithm_bundle_id,
        experiment_protocol_id=experiment_protocol_id,
        role_algorithm_id=args.role_algorithm or arm_config.role_algorithm_id,
        gate_algorithm_id=args.gate_algorithm or arm_config.gate_algorithm_id,
        escalation_algorithm_id=(
            args.escalation_algorithm or arm_config.escalation_algorithm_id
        ),
        audit_algorithm_id=(args.audit_algorithm or arm_config.audit_algorithm_id),
        attribution_algorithm_id=(
            args.attribution_algorithm or arm_config.attribution_algorithm_id
        ),
    )

    run_config = RunConfig(
        env_id=args.env_id,
        model=args.model,
        seed=args.seed,
        max_rounds=args.max_rounds,
        initial_seller_price=args.initial_seller_price,
        buyer_max_price=args.buyer_max_price,
        seller_min_price=args.seller_min_price,
        user_requirement=args.user_requirement,
        product_name=args.product_name,
        product_price=args.product_price,
        user_profile=args.user_profile,
    )
    return ExperimentConfig(run=run_config, arm=arm_config)


def _apply_seed(seed: int) -> None:
    """Apply random seed to Python and NumPy (if available).

    Args:
        seed: Integer random seed.

    Returns:
        None.
    

    中文翻译：应用 random seed to Python and NumPy (if available)。"""
    random.seed(seed)
    with suppress(Exception):
        import numpy as np

        np.random.seed(seed)


def _build_result_record(
    experiment_config: ExperimentConfig,
    algorithm_bundle: Any,
    final_info: dict[str, Any],
    audit_events: int,
    episode_id: str,
    trace_json_path: str | None = None,
) -> dict[str, Any]:
    """Build a stable result record for logging/comparison.

    Args:
        experiment_config: Resolved experiment config for this run.
        algorithm_bundle: Resolved algorithm bundle (possibly with component
            overrides) used for this run.
        final_info: Terminal info dict returned by env.
        audit_events: Number of audit events collected.
        episode_id: Trace episode id.

    Returns:
        Flat JSON-serializable dict used for stdout logging and later analysis.
    

    中文翻译：构建 a stable result record for logging/comparison。"""
    run = experiment_config.run
    arm = experiment_config.arm
    # Design note:
    # We intentionally keep only one architecture axis (`runner_mode`).
    # Environment-side seller implementation is no longer exposed as an
    # experimental variable, so result records avoid carrying that field.
    result = {
        "arm": arm.name,
        "runner_mode": arm.runner_mode,
        "algorithm_bundle_id": arm.algorithm_bundle_id,
        "role_algorithm_id": algorithm_bundle.role_algorithm_id,
        "gate_algorithm_id": algorithm_bundle.gate_algorithm_id,
        "escalation_algorithm_id": algorithm_bundle.escalation_algorithm_id,
        "audit_algorithm_id": algorithm_bundle.audit_algorithm_id,
        "attribution_algorithm_id": algorithm_bundle.attribution_algorithm_id,
        "experiment_protocol_id": arm.experiment_protocol_id,
        "evaluation_target": "seller_side",
        "buyer_mode": "external_user_simulator",
        "config_digest": experiment_config.digest(),
        "env_id": run.env_id,
        "model": run.model,
        "seed": run.seed,
        "episode_id": episode_id,
        "status": final_info.get("status"),
        "agreed_price": final_info.get("agreed_price"),
        "round": final_info.get("round"),
        "termination_reason": final_info.get("termination_reason"),
        "buyer_reward": final_info.get("buyer_reward"),
        "seller_reward": final_info.get("seller_reward"),
        "global_score": final_info.get("global_score"),
        "buyer_score": final_info.get("buyer_score"),
        "seller_score": final_info.get("seller_score"),
        "audit_events": audit_events,
    }
    if trace_json_path is not None:
        result["trace_json"] = trace_json_path
    return result


def _to_jsonable(value: Any) -> Any:
    """Convert nested objects into JSON-serializable primitives.

    Args:
        value: Arbitrary Python object from a trace payload.

    Returns:
        A recursively converted JSON-safe object.
    

    中文翻译：转换 nested objects into JSON-serializable primitives。"""
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


def _write_trace_json(trace: Any, output_path: str) -> str:
    """Write one full episode trace to JSON.

    Args:
        trace: Episode trace object returned by a runner.
        output_path: User-provided file path. Relative paths are resolved
            against the repository root.

    Returns:
        Absolute path string to the written JSON file.
    

    中文翻译：写入 one full episode trace to JSON。"""
    path = Path(output_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_jsonable(trace)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return str(path.resolve())


def main() -> int:
    """Run one episode according to resolved experiment config.

    Returns:
        Process exit code (``0`` for success, non-zero for config/runtime
        failures).
    

    中文翻译：运行 one episode according to resolved experiment config。"""
    args = parse_args()
    try:
        experiment_config = _build_experiment_config(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(
            json.dumps(
                {
                    "config_digest": experiment_config.digest(),
                    "experiment_config": experiment_config.to_dict(),
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        return 0

    _apply_seed(experiment_config.run.seed)
    try:
        _base_algorithm_bundle = resolve_algorithm_bundle(
            experiment_config.arm.algorithm_bundle_id
        )
        algorithm_bundle = compose_algorithm_bundle(
            bundle_id=_base_algorithm_bundle.bundle_id,
            role_algorithm_id=experiment_config.arm.role_algorithm_id,
            gate_algorithm_id=experiment_config.arm.gate_algorithm_id,
            escalation_algorithm_id=experiment_config.arm.escalation_algorithm_id,
            audit_algorithm_id=experiment_config.arm.audit_algorithm_id,
            attribution_algorithm_id=experiment_config.arm.attribution_algorithm_id,
        )
        experiment_protocol = resolve_experiment_protocol(
            experiment_config.arm.experiment_protocol_id
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        buyer, seller, _model_runtime = build_agenticpay_agents(
            run_config=experiment_config.run,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    common_kwargs = {
        "env_id": experiment_config.run.env_id,
        "buyer_agent": buyer,
        "seller_agent": seller,
        "env_kwargs": {
            "max_rounds": experiment_config.run.max_rounds,
            "initial_seller_price": experiment_config.run.initial_seller_price,
            "buyer_max_price": experiment_config.run.buyer_max_price,
            "seller_min_price": experiment_config.run.seller_min_price,
        },
        "reset_kwargs": {
            "user_requirement": experiment_config.run.user_requirement,
            "product_info": {
                "name": experiment_config.run.product_name,
                "price": experiment_config.run.product_price,
            },
            "user_profile": experiment_config.run.user_profile,
        },
        "trace_metadata": {
            "arm_name": experiment_config.arm.name,
            "runner_mode": experiment_config.arm.runner_mode,
            "config_digest": experiment_config.digest(),
            "seed": experiment_config.run.seed,
        },
    }
    if experiment_config.arm.runner_mode == "ocl":
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
    trace_json_path: str | None = None
    if args.trace_json:
        try:
            trace_json_path = _write_trace_json(trace, args.trace_json)
        except OSError as exc:
            print(f"ERROR: failed to write trace JSON: {exc}", file=sys.stderr)
            return 2

    result = _build_result_record(
        experiment_config=experiment_config,
        algorithm_bundle=algorithm_bundle,
        final_info=final_info,
        audit_events=len(trace.events),
        episode_id=trace.episode_id,
        trace_json_path=trace_json_path,
    )
    ordered_keys = [
        "arm",
        "runner_mode",
        "algorithm_bundle_id",
        "role_algorithm_id",
        "gate_algorithm_id",
        "escalation_algorithm_id",
        "audit_algorithm_id",
        "attribution_algorithm_id",
        "experiment_protocol_id",
        "evaluation_target",
        "buyer_mode",
        "config_digest",
        "seed",
        "env_id",
        "model",
        "episode_id",
        "status",
        "agreed_price",
        "round",
        "termination_reason",
        "audit_events",
        "trace_json",
    ]
    for key in ordered_keys:
        if key in result:
            print(f"{key}: {result[key]}")
    print(f"result_json: {json.dumps(result, ensure_ascii=True, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
