#!/usr/bin/env python3
"""Run minimal step-8 Shapley attribution over role-mask subsets.

This script intentionally stays close to the four interface points:

- run_episode(role_mask=S, seed=...) -> trace
- compute_V(trace) -> float
- fallback_policy(role, state) -> action
- compute_shapley({V(S)}) -> {phi_i, w_i}


中文翻译：运行 minimal step-8 Shapley attribution over role-mask subsets。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from enum import Enum
import json
import os
from pathlib import Path
import sys
from typing import Any

# Prefer vendored AgenticPay in this repo when present.
# 中文：如果仓库内自带了 vendored AgenticPay，则优先使用本地版本。
VENDORED_AGENTICPAY_ROOT = Path(__file__).resolve().parent.parent / "agenticpay"
VENDORED_AGENTICPAY_PKG = VENDORED_AGENTICPAY_ROOT / "agenticpay" / "__init__.py"
if VENDORED_AGENTICPAY_PKG.exists() and str(VENDORED_AGENTICPAY_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDORED_AGENTICPAY_ROOT))

# Allow direct execution via `python scripts/run_shapley_min.py` from repo root.
# 中文：允许在仓库根目录直接运行脚本，无需先执行 `pip install -e .`。
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aimai_ocl import (  # noqa: E402
    CONTROLLED_ROLES,
    RunConfig,
    ValueFunctionConfig,
    compute_V,
    compute_shapley,
    run_episode,
)
from aimai_ocl.model_runtime import (  # noqa: E402
    build_agenticpay_agents,
)


DEFAULT_RUN = RunConfig(
    model=os.getenv("AIMAI_MODEL", os.getenv("OPENAI_MODEL", RunConfig.model)),
)
DEFAULT_VALUE = ValueFunctionConfig()


def parse_args() -> argparse.Namespace:
    """Parse CLI args for minimal Shapley experiment.

    Input:
        CLI flags from shell invocation.

    Output:
        Namespace with scenario settings, seed list, and output controls.
    

    中文翻译：解析 CLI args for minimal Shapley experiment。"""
    parser = argparse.ArgumentParser(
        description="Run minimal role-mask Shapley attribution (step-8).",
    )
    parser.add_argument(
        "--roles",
        default=",".join(CONTROLLED_ROLES),
        help="Comma-separated role universe. Default: platform,seller,expert",
    )
    parser.add_argument(
        "--seeds",
        default=str(DEFAULT_RUN.seed),
        help="Comma-separated deterministic seed list. Example: 7,8,9",
    )
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

    parser.add_argument("--success-weight", type=float, default=DEFAULT_VALUE.success_weight)
    parser.add_argument(
        "--seller-reward-weight",
        type=float,
        default=DEFAULT_VALUE.seller_reward_weight,
    )
    parser.add_argument("--violation-penalty", type=float, default=DEFAULT_VALUE.violation_penalty)
    parser.add_argument("--round-penalty", type=float, default=DEFAULT_VALUE.round_penalty)
    parser.add_argument("--escalation-penalty", type=float, default=DEFAULT_VALUE.escalation_penalty)

    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write full result JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned subsets/seeds and exit without model calls.",
    )
    return parser.parse_args()


def _parse_csv_tokens(raw: str) -> list[str]:
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _parse_int_list(raw: str) -> list[int]:
    """Parse a comma-separated integer list from CLI string.

中文翻译：解析 a comma-separated integer list from CLI string。"""
    values: list[int] = []
    for token in _parse_csv_tokens(raw):
        values.append(int(token))
    if not values:
        raise ValueError("At least one seed is required.")
    return values


def _parse_roles(raw: str) -> tuple[str, ...]:
    """Parse and validate role universe against controlled role constants.

中文翻译：解析 and validate role universe against controlled role constants。"""
    roles = tuple(token.lower() for token in _parse_csv_tokens(raw))
    if not roles:
        raise ValueError("At least one role is required.")
    invalid = [role for role in roles if role not in CONTROLLED_ROLES]
    if invalid:
        raise ValueError(
            f"Invalid roles: {invalid}. Allowed: {list(CONTROLLED_ROLES)}"
        )
    if len(set(roles)) != len(roles):
        raise ValueError("Duplicate role names are not allowed.")
    return roles


def _all_subsets(roles: tuple[str, ...]) -> list[frozenset[str]]:
    """Enumerate all role subsets in deterministic order.

中文翻译：Enumerate all role subsets in deterministic order。"""
    subsets: list[frozenset[str]] = []
    n = len(roles)
    for mask in range(2**n):
        active: set[str] = set()
        for i, role in enumerate(roles):
            if (mask >> i) & 1:
                active.add(role)
        subsets.append(frozenset(active))
    return subsets


def _subset_key(subset: frozenset[str]) -> str:
    if not subset:
        return "{}"
    return "+".join(sorted(subset))


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


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_jsonable(payload), ensure_ascii=True, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return str(path.resolve())


def _build_agents(
    run_config: RunConfig,
) -> tuple[Any, Any]:
    """Create one buyer/seller pair for one episode execution.

中文翻译：创建 one buyer/seller pair for one episode execution。"""
    # Attribution sandbox uses the same upstream seller path as main experiments.
    # This keeps coalition-value estimates comparable with single/ocl runs.
    buyer, seller, _runtime = build_agenticpay_agents(
        run_config=run_config,
    )
    return buyer, seller


def main() -> int:
    """Entry point for minimal step-8 role-mask Shapley run.

中文翻译：minimal step-8 role-mask Shapley run 的入口函数。"""
    args = parse_args()
    try:
        roles = _parse_roles(args.roles)
        seeds = _parse_int_list(args.seeds)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    subsets = _all_subsets(roles)
    run_cfg = RunConfig(
        env_id=args.env_id,
        model=args.model,
        seed=seeds[0],
        max_rounds=args.max_rounds,
        initial_seller_price=args.initial_seller_price,
        buyer_max_price=args.buyer_max_price,
        seller_min_price=args.seller_min_price,
        user_requirement=args.user_requirement,
        product_name=args.product_name,
        product_price=args.product_price,
        user_profile=args.user_profile,
    )
    value_cfg = ValueFunctionConfig(
        success_weight=args.success_weight,
        seller_reward_weight=args.seller_reward_weight,
        violation_penalty=args.violation_penalty,
        round_penalty=args.round_penalty,
        escalation_penalty=args.escalation_penalty,
    )

    plan = {
        "roles": list(roles),
        "subsets": [sorted(list(s)) for s in subsets],
        "seeds": seeds,
        "run_config": run_cfg.to_dict(),
        "value_config": _to_jsonable(value_cfg),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=True, sort_keys=True))
        return 0

    subset_values: dict[frozenset[str], float] = {}
    subset_details: list[dict[str, Any]] = []
    for subset in subsets:
        seed_values: list[float] = []
        for seed in seeds:
            try:
                buyer, seller = _build_agents(
                    run_cfg,
                )
            except (RuntimeError, ValueError) as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 2
            trace = run_episode(
                role_mask=set(subset),
                seed=seed,
                env_id=run_cfg.env_id,
                buyer_agent=buyer,
                seller_agent=seller,
                env_kwargs={
                    "max_rounds": run_cfg.max_rounds,
                    "initial_seller_price": run_cfg.initial_seller_price,
                    "buyer_max_price": run_cfg.buyer_max_price,
                    "seller_min_price": run_cfg.seller_min_price,
                },
                reset_kwargs={
                    "user_requirement": run_cfg.user_requirement,
                    "product_info": {
                        "name": run_cfg.product_name,
                        "price": run_cfg.product_price,
                    },
                    "user_profile": run_cfg.user_profile,
                },
                trace_metadata={
                    "source": "run_shapley_min",
                    "subset_key": _subset_key(subset),
                },
            )
            v = compute_V(trace, config=value_cfg)
            seed_values.append(float(v))
            print(
                f"subset={_subset_key(subset)} seed={seed} "
                f"status={trace.final_status} V={v:.4f} "
                f"events={len(trace.events)}"
            )

        mean_v = sum(seed_values) / len(seed_values)
        subset_values[frozenset(subset)] = mean_v
        subset_details.append(
            {
                "subset": sorted(list(subset)),
                "subset_key": _subset_key(subset),
                "seed_values": seed_values,
                "mean_V": mean_v,
            }
        )

    shapley = compute_shapley(subset_values, roles=roles)
    result = {
        "plan": plan,
        "subset_values": {
            _subset_key(key): value
            for key, value in subset_values.items()
        },
        "subset_details": subset_details,
        "phi": shapley["phi"],
        "w": shapley["w"],
    }

    print(f"phi: {json.dumps(result['phi'], ensure_ascii=True, sort_keys=True)}")
    print(f"w: {json.dumps(result['w'], ensure_ascii=True, sort_keys=True)}")

    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        if not output_path.is_absolute():
            output_path = (REPO_ROOT / output_path).resolve()
        written = _write_json(output_path, result)
        print(f"output_json: {written}")

    print(f"result_json: {json.dumps(result, ensure_ascii=True, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
