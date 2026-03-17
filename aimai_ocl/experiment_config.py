"""Experiment configuration objects for reproducible baseline runs.

This module separates:

- run-level controls shared across all experiment arms
- arm-level controls for baseline/OCL variants

中文翻译：
该模块把实验配置拆成两层：
1) 运行层（RunConfig）：所有实验臂共享，保证可比性；
2) 实验臂层（ArmConfig）：描述 baseline / OCL / ablation 的差异。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json


@dataclass(frozen=True)
class RunConfig:
    """Shared run configuration that must stay fixed across experiment arms.

    中文翻译：
    该配置是“横向对照”的固定项，跨实验臂应保持一致，避免混杂变量。
    """

    env_id: str = "Task1_basic_price_negotiation-v0"
    model: str = "gpt-4o-mini"
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
    """Experiment arm definition (baseline or OCL variant).

    Architecture note:
        ``runner_mode`` is the single architecture axis in this repository:
        ``single`` (baseline pass-through) vs ``ocl`` (role/gate/escalation).

        We intentionally removed environment-side seller implementation
        branching. Upstream AgenticPay exposes one stable seller path and
        mixing that with control-layer architecture caused conceptual drift in
        experiment interpretation.

    Note:
        Arm flags are the contract for ablation comparisons. Some controls are
        currently scaffold-level and will be fully activated in later steps.
        ``*_algorithm_id`` fields are optional component-level overrides that
        can specialize one bundle without creating a new bundle id.

    中文翻译：
    ArmConfig 是实验臂契约。``role_on/gate_on/...`` 定义机制开关，
    ``*_algorithm_id`` 允许在同一个 bundle 下做局部算法替换。
    """

    name: str
    runner_mode: str
    role_on: bool
    gate_on: bool
    audit_on: bool
    escalate_on: bool
    attribution_on: bool
    algorithm_bundle_id: str = "v1_default"
    experiment_protocol_id: str = "offline_v1"
    role_algorithm_id: str | None = None
    gate_algorithm_id: str | None = None
    escalation_algorithm_id: str | None = None
    audit_algorithm_id: str | None = None
    attribution_algorithm_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentConfig:
    """Combined experiment configuration.

    中文翻译：
    把运行层配置与实验臂配置组合成一个不可变对象，便于审计与复现。
    """

    run: RunConfig
    arm: ArmConfig

    def to_dict(self) -> dict[str, object]:
        return {
            "run": self.run.to_dict(),
            "arm": self.arm.to_dict(),
        }

    def digest(self) -> str:
        """Stable short fingerprint for run reproducibility checks.

        中文翻译：
        生成短哈希作为配置指纹，用于快速校验两次运行是否真的是同一配置。
        """
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(payload).hexdigest()[:12]


ARM_REGISTRY: dict[str, ArmConfig] = {
    "single": ArmConfig(
        name="single",
        runner_mode="single",
        role_on=False,
        gate_on=False,
        audit_on=True,
        escalate_on=False,
        attribution_on=False,
        algorithm_bundle_id="v1_default",
        experiment_protocol_id="offline_v1",
    ),
    "ocl_full": ArmConfig(
        name="ocl_full",
        runner_mode="ocl",
        role_on=True,
        gate_on=True,
        audit_on=True,
        escalate_on=True,
        attribution_on=True,
        algorithm_bundle_id="v1_default",
        experiment_protocol_id="offline_v1",
    ),
}


def resolve_arm(arm_name: str) -> ArmConfig:
    """Resolve a registered experiment arm.

    Args:
        arm_name: Arm name key, for example ``single`` or ``ocl_full``.

    Returns:
        Immutable ``ArmConfig`` for the requested arm.

    Raises:
        ValueError: If the arm name is not in ``ARM_REGISTRY``.

    中文翻译：
    从注册表中解析实验臂名称；若名称非法，会返回带候选列表的错误信息。
    """
    try:
        return ARM_REGISTRY[arm_name]
    except KeyError as exc:
        available = ", ".join(sorted(ARM_REGISTRY))
        raise ValueError(f"Unknown arm '{arm_name}'. Available: {available}") from exc
