"""Experiment configuration and run modes.

Defines RunConfig (shared across arms), ArmConfig (per-arm settings),
and the five experiment modes: demo, batch, paired, ablation, shapley.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    """Shared run config — stays fixed across experiment arms."""

    env_id: str = "Task1_basic_price_negotiation-v0"
    model: str = "gpt-4o-mini"
    provider: str = "openai"
    api_key_env: str = "OPENAI_API_KEY"
    seed: int = 42
    max_rounds: int = 10
    initial_seller_price: float = 180.0
    buyer_max_price: float = 120.0
    seller_min_price: float = 90.0
    user_requirement: str = "I need a winter jacket"
    product_name: str = "Winter Jacket"
    product_price: float = 180.0
    user_profile: str = "Budget-conscious and compares options before buying."

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ArmConfig:
    """Experiment arm: baseline (single) or OCL variant."""

    name: str
    ocl: bool = False
    coordinator_mode: str = "default"
    risk_rewrite_threshold: float = 0.45
    risk_block_threshold: float = 0.75
    enable_replan: bool = True
    audit: str = "full"  # "full", "minimal", "off"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# Pre-defined arms
ARMS: dict[str, ArmConfig] = {
    "single": ArmConfig(name="single", ocl=False),
    "ocl_full": ArmConfig(name="ocl_full", ocl=True),
    "ocl_strict": ArmConfig(
        name="ocl_strict", ocl=True,
        risk_rewrite_threshold=0.35, risk_block_threshold=0.60,
    ),
    "ocl_lenient": ArmConfig(
        name="ocl_lenient", ocl=True,
        risk_rewrite_threshold=0.55, risk_block_threshold=0.85,
    ),
    "ocl_no_replan": ArmConfig(name="ocl_no_replan", ocl=True, enable_replan=False),
    "ocl_seller_only": ArmConfig(name="ocl_seller_only", ocl=True, coordinator_mode="seller_only"),
    "ocl_state_machine": ArmConfig(name="ocl_state_machine", ocl=True, coordinator_mode="state_machine"),
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Combined experiment configuration."""

    run: RunConfig
    arm: ArmConfig

    def to_dict(self) -> dict[str, object]:
        return {"run": self.run.to_dict(), "arm": self.arm.to_dict()}

    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return sha256(payload).hexdigest()[:12]


def resolve_arm(name: str) -> ArmConfig:
    """Look up a pre-defined arm by name."""
    if name not in ARMS:
        raise ValueError(f"Unknown arm '{name}'. Available: {', '.join(sorted(ARMS))}")
    return ARMS[name]
