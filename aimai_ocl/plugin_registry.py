"""Pluggable algorithm/protocol registry for steps 6-10.

Design goals:
- Step 6 role decomposition algorithm is replaceable
- Step 7 escalation/replan algorithm is replaceable
- Step 8/9 attribution algorithm is replaceable
- Step 3 risk-gating behavior is replaceable
- Step 10 experiment protocol is replaceable


中文翻译：Pluggable algorithm/protocol registry for steps 6-10。"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from statistics import mean
from typing import Any, Callable, Mapping, Protocol

from aimai_ocl.attribution_counterfactual import (
    compute_V as counterfactual_compute_V,
    compute_shapley as counterfactual_compute_shapley,
    fallback_policy as counterfactual_fallback_policy,
    run_episode as counterfactual_run_episode,
)
from aimai_ocl.attribution_shapley import (
    compute_V as default_compute_V,
    compute_shapley as default_compute_shapley,
    fallback_policy as default_fallback_policy,
    run_episode as default_run_episode,
)
from aimai_ocl.controllers import (
    AuditPolicy,
    BarrierRiskGate,
    Coordinator,
    DisabledEscalationManager,
    EscalationManager,
    OCLController,
    RiskGate,
    SellerOnlyCoordinator,
    StateMachineCoordinator,
    TauControlledRiskGate,
    barrier_config_from_tau,
    tau_control_surface_from_tau,
)
from aimai_ocl.schemas.actions import ActionIntent, RawAction
from aimai_ocl.schemas.audit import AuditEvent, AuditEventType, EpisodeTrace


class RoleAlgorithm(Protocol):
    """Protocol for round-level role decomposition algorithms.

中文翻译：Protocol for round-level role decomposition algorithms。"""

    def plan_turn(
        self,
        *,
        round_id: int,
        buyer_text: str | None,
        seller_actor_id: str,
        max_rounds: int | None = None,
    ):  # noqa: ANN201
        """Return one per-round coordination plan.

中文翻译：返回 one per-round coordination plan。"""

    def build_audit_event(self, plan) -> AuditEvent:  # noqa: ANN001
        """Convert one plan into a coordination audit event.

中文翻译：转换 one plan into a coordination audit event。"""


class GateAlgorithm(Protocol):
    """Protocol for seller-side control/gating algorithms.

中文翻译：Protocol for seller-side control/gating algorithms。"""

    def apply(
        self,
        raw_action: RawAction,
        *,
        round_id: int | None = None,
        history: list[dict[str, Any]] | None = None,
        state: dict[str, Any] | None = None,
    ):  # noqa: ANN201
        """Transform raw action into executable action plus checks/events.

中文翻译：Transform raw action into executable action plus checks/events。"""


class EscalationAlgorithm(Protocol):
    """Protocol for escalation/replan algorithms.

中文翻译：Protocol for escalation/replan algorithms。"""

    def resolve(
        self,
        *,
        round_id: int,
        actor_id: str,
        raw_action: RawAction,
        approved: bool,
        requires_confirmation: bool,
        requires_escalation: bool,
        violations: list[str],
        state: dict[str, Any],
        allow_replan: bool = True,
    ):  # noqa: ANN201
        """Resolve execution path after gate/controller decision.

中文翻译：解析 execution path after gate/controller decision。"""


class AuditAlgorithm(Protocol):
    """Protocol for audit/trace emission policy.

中文翻译：Protocol for audit/trace emission policy。"""

    def should_record(self, event_type: AuditEventType) -> bool:
        """Return whether one audit event type should be recorded.

中文翻译：返回某个审计事件类型是否应记录。"""


@dataclass(frozen=True, slots=True)
class AttributionModule:
    """Attribution algorithm module exposing the four step-8 interfaces.

中文翻译：Attribution algorithm module exposing the four step-8 interfaces。"""

    module_id: str
    description: str
    run_episode_fn: Callable[..., EpisodeTrace]
    compute_V_fn: Callable[..., float]
    fallback_policy_fn: Callable[[str, Mapping[str, Any]], str]
    compute_shapley_fn: Callable[..., dict[str, dict[str, float]]]


class AttributionAlgorithm(Protocol):
    """Protocol for step-8/9 attribution algorithm interface.

中文翻译：Protocol for step-8/9 attribution algorithm interface。"""

    def run_episode(self, **kwargs: Any) -> EpisodeTrace:
        """Run one role-mask episode and return trace.

中文翻译：运行 one role-mask episode and return trace。"""

    def compute_V(self, trace: EpisodeTrace, **kwargs: Any) -> float:
        """Compute one scalar value from trace.

中文翻译：计算 one scalar value from trace。"""

    def fallback_policy(self, role: str, state: Mapping[str, Any]) -> str:
        """Return deterministic fallback action for a role.

中文翻译：返回 deterministic fallback action for a role。"""

    def compute_shapley(
        self,
        subset_values: Mapping[frozenset[str], float],
        **kwargs: Any,
    ) -> dict[str, dict[str, float]]:
        """Compute role contribution values/shares from coalition values.

中文翻译：计算 role contribution values/shares from coalition values。"""


@dataclass(frozen=True, slots=True)
class AlgorithmBundleSpec:
    """Declarative bundle spec referencing component-level algorithm ids.

中文翻译：Declarative bundle spec referencing component-level algorithm ids。"""

    bundle_id: str
    description: str
    role_algorithm_id: str
    gate_algorithm_id: str
    escalation_algorithm_id: str
    audit_algorithm_id: str
    attribution_algorithm_id: str


@dataclass(frozen=True, slots=True)
class AlgorithmBundle:
    """Concrete bundle with resolved algorithm factories/functions.

中文翻译：Concrete bundle with resolved algorithm factories/functions。"""

    bundle_id: str
    description: str
    role_algorithm_id: str
    gate_algorithm_id: str
    escalation_algorithm_id: str
    audit_algorithm_id: str
    attribution_algorithm_id: str
    make_role_algorithm: Callable[[], RoleAlgorithm]
    make_gate_algorithm: Callable[[], GateAlgorithm]
    make_escalation_algorithm: Callable[[], EscalationAlgorithm]
    make_audit_algorithm: Callable[[], AuditAlgorithm]
    run_episode_fn: Callable[..., EpisodeTrace]
    compute_V_fn: Callable[..., float]
    fallback_policy_fn: Callable[[str, Mapping[str, Any]], str]
    compute_shapley_fn: Callable[..., dict[str, dict[str, float]]]
    gate_tau: float | None = None
    gate_runtime_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExperimentProtocolBundle:
    """Pluggable bundle for experiment protocol logic (step 10).

中文翻译：Pluggable bundle for experiment protocol logic (step 10)。"""

    protocol_id: str
    description: str
    run_main_fn: Callable[..., dict[str, Any]] | None = None
    run_ablation_fn: Callable[..., dict[str, Any]] | None = None
    run_adversarial_fn: Callable[..., dict[str, Any]] | None = None
    run_repeated_fn: Callable[..., dict[str, Any]] | None = None
    run_roi_fn: Callable[..., dict[str, Any]] | None = None


def _make_default_gate_algorithm() -> OCLController:
    return OCLController()


def _make_strict_gate_algorithm() -> OCLController:
    return OCLController(
        risk_gate=RiskGate(
            high_risk_intents=(
                ActionIntent.TOOL_CALL,
                ActionIntent.ESCALATE,
                ActionIntent.ACCEPT_DEAL,
                ActionIntent.NEGOTIATE_PRICE,
            ),
            require_confirmation_for_high_risk=True,
        )
    )


def _make_lenient_gate_algorithm() -> OCLController:
    return OCLController(
        risk_gate=RiskGate(
            high_risk_intents=(
                ActionIntent.TOOL_CALL,
                ActionIntent.ESCALATE,
            ),
            require_confirmation_for_high_risk=False,
        )
    )


def _make_off_gate_algorithm() -> OCLController:
    return OCLController(
        risk_gate=RiskGate(
            high_risk_intents=(),
            require_confirmation_for_high_risk=False,
        )
    )


_BARRIER_GATE_PRESET_CONFIGS: dict[str, dict[str, float]] = {
    "gate_v2_barrier": {
        "rewrite_threshold": 0.45,
        "block_threshold": 0.75,
        "epsilon_miss": 0.10,
    },
    "gate_v2_barrier_strict": {
        "rewrite_threshold": 0.35,
        "block_threshold": 0.60,
        "epsilon_miss": 0.05,
    },
}


def _make_barrier_gate_algorithm(
    *,
    rewrite_threshold: float,
    block_threshold: float,
    epsilon_miss: float,
) -> OCLController:
    return OCLController(
        risk_gate=BarrierRiskGate(
            rewrite_threshold=rewrite_threshold,
            block_threshold=block_threshold,
            epsilon_miss=epsilon_miss,
        )
    )


def _make_tau_controlled_gate_algorithm(
    *,
    gate_tau: float,
) -> OCLController:
    return OCLController(
        risk_gate=TauControlledRiskGate.from_tau(gate_tau=gate_tau)
    )


def _make_barrier_gate_algorithm_from_preset(gate_algorithm_id: str) -> OCLController:
    params = _BARRIER_GATE_PRESET_CONFIGS[gate_algorithm_id]
    return _make_barrier_gate_algorithm(**params)


def _make_default_escalation_algorithm() -> EscalationManager:
    return EscalationManager()


def _make_no_replan_escalation_algorithm() -> EscalationManager:
    return EscalationManager(enable_replan=False)


def _make_off_escalation_algorithm() -> DisabledEscalationManager:
    return DisabledEscalationManager()


def _make_full_audit_algorithm() -> AuditPolicy:
    return AuditPolicy(
        policy_id="audit_v1_full",
        enabled_event_types=None,
        description="Full trace with all audit event types.",
    )


def _make_weak_audit_algorithm() -> AuditPolicy:
    return AuditPolicy(
        policy_id="audit_v1_weak",
        enabled_event_types=frozenset(
            {
                AuditEventType.EPISODE_STARTED,
                AuditEventType.EPISODE_FINISHED,
                AuditEventType.CONSTRAINT_EVALUATED,
                AuditEventType.ESCALATION_TRIGGERED,
            }
        ),
        description=(
            "Minimal trace keeping lifecycle + constraint + escalation events."
        ),
    )


def _make_off_audit_algorithm() -> AuditPolicy:
    return AuditPolicy(
        policy_id="audit_v1_off",
        enabled_event_types=frozenset(),
        description="Disable audit event emission in episode traces.",
    )


def _compute_V_reward_only(trace: EpisodeTrace, **kwargs: Any) -> float:
    """Compute value only from terminal seller reward (simple baseline).

中文翻译：计算 value only from terminal seller reward (simple baseline)。"""
    del kwargs
    final_metrics = trace.final_metrics if isinstance(trace.final_metrics, dict) else {}
    reward = final_metrics.get("seller_reward", 0.0)
    try:
        return float(reward)
    except (TypeError, ValueError):
        return 0.0


ROLE_ALGORITHM_REGISTRY: dict[str, Callable[[], RoleAlgorithm]] = {
    "role_v1_rule": Coordinator,
    "role_v1_seller_only": SellerOnlyCoordinator,
    "role_v2_state_machine": StateMachineCoordinator,
}


GATE_ALGORITHM_REGISTRY: dict[str, Callable[[], GateAlgorithm]] = {
    "gate_v0_off": _make_off_gate_algorithm,
    "gate_v1_default": _make_default_gate_algorithm,
    "gate_v1_strict": _make_strict_gate_algorithm,
    "gate_v1_lenient": _make_lenient_gate_algorithm,
    "gate_v2_barrier": lambda: _make_barrier_gate_algorithm_from_preset("gate_v2_barrier"),
    "gate_v2_barrier_strict": (
        lambda: _make_barrier_gate_algorithm_from_preset("gate_v2_barrier_strict")
    ),
    "gate_v3_tau_controlled": lambda: _make_tau_controlled_gate_algorithm(gate_tau=0.5),
}


ESCALATION_ALGORITHM_REGISTRY: dict[str, Callable[[], EscalationAlgorithm]] = {
    "escalation_v0_off": _make_off_escalation_algorithm,
    "escalation_v1_default": _make_default_escalation_algorithm,
    "escalation_v1_no_replan": _make_no_replan_escalation_algorithm,
}


AUDIT_ALGORITHM_REGISTRY: dict[str, Callable[[], AuditAlgorithm]] = {
    "audit_v1_full": _make_full_audit_algorithm,
    "audit_v1_weak": _make_weak_audit_algorithm,
    "audit_v1_off": _make_off_audit_algorithm,
}


ATTRIBUTION_ALGORITHM_REGISTRY: dict[str, AttributionModule] = {
    "shapley_v1_exact": AttributionModule(
        module_id="shapley_v1_exact",
        description=(
            "Exact Shapley with trajectory value function including success/"
            "reward/violation/round/escalation terms."
        ),
        run_episode_fn=default_run_episode,
        compute_V_fn=default_compute_V,
        fallback_policy_fn=default_fallback_policy,
        compute_shapley_fn=default_compute_shapley,
    ),
    "shapley_v1_reward_only": AttributionModule(
        module_id="shapley_v1_reward_only",
        description=(
            "Exact Shapley with reward-only value function baseline."
        ),
        run_episode_fn=default_run_episode,
        compute_V_fn=_compute_V_reward_only,
        fallback_policy_fn=default_fallback_policy,
        compute_shapley_fn=default_compute_shapley,
    ),
    "counterfactual_v1": AttributionModule(
        module_id="counterfactual_v1",
        description=(
            "Counterfactual value + MC/Exact Shapley hybrid with "
            "role-specific fallback policy."
        ),
        run_episode_fn=counterfactual_run_episode,
        compute_V_fn=counterfactual_compute_V,
        fallback_policy_fn=counterfactual_fallback_policy,
        compute_shapley_fn=counterfactual_compute_shapley,
    ),
}


ALGORITHM_BUNDLE_SPEC_REGISTRY: dict[str, AlgorithmBundleSpec] = {
    "v1_default": AlgorithmBundleSpec(
        bundle_id="v1_default",
        description=(
            "Rule-based role decomposition + tau-controlled gate + default escalation + "
            "full audit + exact Shapley attribution."
        ),
        role_algorithm_id="role_v1_rule",
        gate_algorithm_id="gate_v3_tau_controlled",
        escalation_algorithm_id="escalation_v1_default",
        audit_algorithm_id="audit_v1_full",
        attribution_algorithm_id="shapley_v1_exact",
    ),
    "v1_role_ablation": AlgorithmBundleSpec(
        bundle_id="v1_role_ablation",
        description=(
            "Seller-only coordination ablation with tau-controlled gate/escalation/"
            "attribution."
        ),
        role_algorithm_id="role_v1_seller_only",
        gate_algorithm_id="gate_v3_tau_controlled",
        escalation_algorithm_id="escalation_v1_default",
        audit_algorithm_id="audit_v1_full",
        attribution_algorithm_id="shapley_v1_exact",
    ),
    "v2_research": AlgorithmBundleSpec(
        bundle_id="v2_research",
        description=(
            "State-machine role decomposition + tau-controlled risk gating + full audit + "
            "counterfactual attribution."
        ),
        role_algorithm_id="role_v2_state_machine",
        gate_algorithm_id="gate_v3_tau_controlled",
        escalation_algorithm_id="escalation_v1_default",
        audit_algorithm_id="audit_v1_full",
        attribution_algorithm_id="counterfactual_v1",
    ),
}


def resolve_role_algorithm_factory(role_algorithm_id: str) -> Callable[[], RoleAlgorithm]:
    """Resolve role-decomposition algorithm factory by id.

中文翻译：解析 role-decomposition algorithm factory by id。"""
    try:
        return ROLE_ALGORITHM_REGISTRY[role_algorithm_id]
    except KeyError as exc:
        available = ", ".join(sorted(ROLE_ALGORITHM_REGISTRY))
        raise ValueError(
            f"Unknown role algorithm '{role_algorithm_id}'. Available: {available}"
        ) from exc


def describe_gate_algorithm_runtime(
    gate_algorithm_id: str,
    *,
    gate_tau: float | None = None,
) -> dict[str, Any]:
    """Describe resolved runtime gate parameters for logging and traces.

    中文翻译：描述运行时 gate 参数，用于日志与 trace。"""
    if gate_algorithm_id in _BARRIER_GATE_PRESET_CONFIGS:
        params = (
            barrier_config_from_tau(gate_tau)
            if gate_tau is not None
            else dict(_BARRIER_GATE_PRESET_CONFIGS[gate_algorithm_id])
        )
        return {
            "gate_family": "barrier",
            "tau_applied": gate_tau is not None,
            "gate_tau": params.get("gate_tau"),
            "rewrite_threshold": params["rewrite_threshold"],
            "block_threshold": params["block_threshold"],
            "epsilon_miss": params["epsilon_miss"],
        }
    if gate_algorithm_id == "gate_v3_tau_controlled":
        surface = tau_control_surface_from_tau(0.5 if gate_tau is None else gate_tau)
        return surface.to_runtime_config()
    return {
        "gate_family": "legacy",
        "tau_applied": False,
        "gate_tau": gate_tau,
        "rewrite_threshold": None,
        "block_threshold": None,
        "epsilon_miss": None,
    }


def resolve_gate_algorithm_factory(
    gate_algorithm_id: str,
    *,
    gate_tau: float | None = None,
) -> Callable[[], GateAlgorithm]:
    """Resolve risk-gate/controller algorithm factory by id.

中文翻译：解析 risk-gate/controller algorithm factory by id。"""
    if gate_algorithm_id in _BARRIER_GATE_PRESET_CONFIGS and gate_tau is not None:
        params = barrier_config_from_tau(gate_tau)

        def _factory() -> GateAlgorithm:
            return _make_barrier_gate_algorithm(
                rewrite_threshold=params["rewrite_threshold"],
                block_threshold=params["block_threshold"],
                epsilon_miss=params["epsilon_miss"],
            )

        return _factory
    if gate_algorithm_id == "gate_v3_tau_controlled":
        resolved_tau = 0.5 if gate_tau is None else gate_tau

        def _factory() -> GateAlgorithm:
            return _make_tau_controlled_gate_algorithm(gate_tau=resolved_tau)

        return _factory
    try:
        return GATE_ALGORITHM_REGISTRY[gate_algorithm_id]
    except KeyError as exc:
        available = ", ".join(sorted(GATE_ALGORITHM_REGISTRY))
        raise ValueError(
            f"Unknown gate algorithm '{gate_algorithm_id}'. Available: {available}"
        ) from exc


def resolve_escalation_algorithm_factory(
    escalation_algorithm_id: str,
) -> Callable[[], EscalationAlgorithm]:
    """Resolve escalation/replan algorithm factory by id.

中文翻译：解析 escalation/replan algorithm factory by id。"""
    try:
        return ESCALATION_ALGORITHM_REGISTRY[escalation_algorithm_id]
    except KeyError as exc:
        available = ", ".join(sorted(ESCALATION_ALGORITHM_REGISTRY))
        raise ValueError(
            "Unknown escalation algorithm "
            f"'{escalation_algorithm_id}'. Available: {available}"
        ) from exc


def resolve_audit_algorithm_factory(audit_algorithm_id: str) -> Callable[[], AuditAlgorithm]:
    """Resolve audit policy algorithm factory by id.

中文翻译：解析 audit policy algorithm factory by id。"""
    try:
        return AUDIT_ALGORITHM_REGISTRY[audit_algorithm_id]
    except KeyError as exc:
        available = ", ".join(sorted(AUDIT_ALGORITHM_REGISTRY))
        raise ValueError(
            f"Unknown audit algorithm '{audit_algorithm_id}'. Available: {available}"
        ) from exc


def resolve_attribution_algorithm(attribution_algorithm_id: str) -> AttributionModule:
    """Resolve attribution algorithm module by id.

中文翻译：解析 attribution algorithm module by id。"""
    try:
        return ATTRIBUTION_ALGORITHM_REGISTRY[attribution_algorithm_id]
    except KeyError as exc:
        available = ", ".join(sorted(ATTRIBUTION_ALGORITHM_REGISTRY))
        raise ValueError(
            "Unknown attribution algorithm "
            f"'{attribution_algorithm_id}'. Available: {available}"
        ) from exc


def compose_algorithm_bundle(
    *,
    bundle_id: str,
    role_algorithm_id: str | None = None,
    gate_algorithm_id: str | None = None,
    gate_tau: float | None = None,
    escalation_algorithm_id: str | None = None,
    audit_algorithm_id: str | None = None,
    attribution_algorithm_id: str | None = None,
) -> AlgorithmBundle:
    """Compose one concrete algorithm bundle with optional component overrides.

中文翻译：Compose one concrete algorithm bundle with optional component overrides。"""
    try:
        spec = ALGORITHM_BUNDLE_SPEC_REGISTRY[bundle_id]
    except KeyError as exc:
        available = ", ".join(sorted(ALGORITHM_BUNDLE_SPEC_REGISTRY))
        raise ValueError(
            f"Unknown algorithm bundle '{bundle_id}'. Available: {available}"
        ) from exc

    resolved_role_id = role_algorithm_id or spec.role_algorithm_id
    resolved_gate_id = gate_algorithm_id or spec.gate_algorithm_id
    resolved_escalation_id = escalation_algorithm_id or spec.escalation_algorithm_id
    resolved_audit_id = audit_algorithm_id or spec.audit_algorithm_id
    resolved_attr_id = attribution_algorithm_id or spec.attribution_algorithm_id

    role_factory = resolve_role_algorithm_factory(resolved_role_id)
    gate_factory = resolve_gate_algorithm_factory(
        resolved_gate_id,
        gate_tau=gate_tau,
    )
    gate_runtime_config = describe_gate_algorithm_runtime(
        resolved_gate_id,
        gate_tau=gate_tau,
    )
    escalation_factory = resolve_escalation_algorithm_factory(resolved_escalation_id)
    audit_factory = resolve_audit_algorithm_factory(resolved_audit_id)
    attribution_module = resolve_attribution_algorithm(resolved_attr_id)

    return AlgorithmBundle(
        bundle_id=bundle_id,
        description=spec.description,
        role_algorithm_id=resolved_role_id,
        gate_algorithm_id=resolved_gate_id,
        escalation_algorithm_id=resolved_escalation_id,
        audit_algorithm_id=resolved_audit_id,
        attribution_algorithm_id=resolved_attr_id,
        gate_tau=gate_runtime_config.get("gate_tau"),
        gate_runtime_config=gate_runtime_config,
        make_role_algorithm=role_factory,
        make_gate_algorithm=gate_factory,
        make_escalation_algorithm=escalation_factory,
        make_audit_algorithm=audit_factory,
        run_episode_fn=attribution_module.run_episode_fn,
        compute_V_fn=attribution_module.compute_V_fn,
        fallback_policy_fn=attribution_module.fallback_policy_fn,
        compute_shapley_fn=attribution_module.compute_shapley_fn,
    )


ALGORITHM_BUNDLE_REGISTRY: dict[str, AlgorithmBundle] = {
    bundle_id: compose_algorithm_bundle(bundle_id=bundle_id)
    for bundle_id in ALGORITHM_BUNDLE_SPEC_REGISTRY
}


def _to_float(value: Any) -> float:
    """Best-effort float conversion for protocol metrics.

中文翻译：Best-effort float conversion for protocol metrics。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _summary_index(summaries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("arm")): row for row in summaries if row.get("arm") is not None}


_MAIN_RESULT_METRIC_SPECS: dict[str, dict[str, Any]] = {
    "success": {
        "summary_key": "success_rate",
        "label": "success_rate",
        "direction": 1,
    },
    "feasibility": {
        "summary_key": "feasibility_rate",
        "label": "feasibility_rate",
        "direction": 1,
    },
    "has_violation": {
        "summary_key": "violation_rate",
        "label": "transient_violation_rate",
        "direction": -1,
    },
    "transient_has_violation": {
        "summary_key": "transient_violation_rate",
        "label": "transient_violation_rate",
        "direction": -1,
    },
    "executed_has_violation": {
        "summary_key": "executed_violation_rate",
        "label": "executed_violation_rate",
        "direction": -1,
    },
    "unrecovered_has_violation": {
        "summary_key": "unrecovered_violation_rate",
        "label": "unrecovered_violation_rate",
        "direction": -1,
    },
    "constraint_satisfaction_rate": {
        "summary_key": "avg_constraint_satisfaction_rate",
        "label": "constraint_satisfaction_rate",
        "direction": 1,
    },
    "round": {
        "summary_key": "avg_round",
        "label": "avg_round",
        "direction": -1,
    },
    "seller_reward": {
        "summary_key": "avg_seller_reward",
        "label": "avg_seller_reward",
        "direction": 1,
    },
    "global_score": {
        "summary_key": "avg_global_score",
        "label": "avg_global_score",
        "direction": 1,
    },
    "welfare": {
        "summary_key": "avg_welfare",
        "label": "avg_welfare",
        "direction": 1,
    },
    "cost_adjusted_welfare": {
        "summary_key": "avg_cost_adjusted_welfare",
        "label": "avg_cost_adjusted_welfare",
        "direction": 1,
    },
    "escalation_count": {
        "summary_key": "avg_escalation_count",
        "label": "avg_escalation_count",
        "direction": -1,
    },
    "latency_sec": {
        "summary_key": "avg_latency_sec",
        "label": "avg_latency_sec",
        "direction": -1,
    },
}


def _metric_direction(metric_key: str) -> int:
    spec = _MAIN_RESULT_METRIC_SPECS.get(metric_key, {})
    direction = int(spec.get("direction", 1))
    return 1 if direction >= 0 else -1


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def _to_float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bootstrap_ci_mean(
    deltas: list[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be > 0")
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
    if samples <= 0:
        raise ValueError("permutation samples must be > 0")
    n = len(deltas)
    if n == 0:
        return {
            "method": "none",
            "samples": 0,
            "p_one_sided": None,
            "p_two_sided": None,
        }

    observed = sum(deltas) / float(n)
    threshold = observed - 1e-12
    abs_threshold = abs(observed) - 1e-12

    if n <= 20:
        total = 1 << n
        ge = 0
        abs_ge = 0
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


def _paired_metric_deltas(
    *,
    records: list[dict[str, Any]],
    target_arm: str,
    baseline_arm: str,
    metric_key: str,
) -> list[float]:
    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for row in records:
        arm = str(row.get("arm"))
        if arm not in {target_arm, baseline_arm}:
            continue
        episode_index = _to_float_or_none(row.get("episode_index"))
        seed = _to_float_or_none(row.get("seed"))
        if episode_index is None or seed is None:
            continue
        pair_key = (int(episode_index), int(seed))
        grouped.setdefault(pair_key, {})[arm] = row

    deltas: list[float] = []
    for pair_rows in grouped.values():
        if target_arm not in pair_rows or baseline_arm not in pair_rows:
            continue
        target_value = _to_float_or_none(pair_rows[target_arm].get(metric_key))
        baseline_value = _to_float_or_none(pair_rows[baseline_arm].get(metric_key))
        if target_value is None or baseline_value is None:
            continue
        deltas.append(target_value - baseline_value)
    return deltas


def _paired_metric_stats(
    *,
    records: list[dict[str, Any]],
    target_arm: str,
    baseline_arm: str,
    metric_key: str,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> dict[str, Any]:
    deltas = _paired_metric_deltas(
        records=records,
        target_arm=target_arm,
        baseline_arm=baseline_arm,
        metric_key=metric_key,
    )
    direction = _metric_direction(metric_key)
    oriented_deltas = [float(direction) * delta for delta in deltas]
    return {
        "pairs": len(deltas),
        "metric_direction": direction,
        "alternative": "target_better_than_baseline",
        "mean_delta": _mean(deltas),
        "mean_improvement": _mean(oriented_deltas),
        "delta_ci95": _bootstrap_ci_mean(
            deltas,
            samples=bootstrap_samples,
            seed=seed + 17,
        ),
        "improvement_ci95": _bootstrap_ci_mean(
            oriented_deltas,
            samples=bootstrap_samples,
            seed=seed + 23,
        ),
        "sign_flip_pvalues": _sign_flip_pvalues(
            oriented_deltas,
            samples=permutation_samples,
            seed=seed + 29,
        ),
    }


def _protocol_main_offline_v1(**kwargs: Any) -> dict[str, Any]:
    """Compute main-result deltas for offline comparison arms.

    Input:
        records:
            Per-run flat records from batch harness.
        summaries:
            Per-arm aggregate summaries.
        plan:
            Batch execution plan metadata.

    Output:
        Dict with arm-level summary plus pairwise deltas where available.
    

    中文翻译：计算 main-result deltas for offline comparison arms。"""
    records: list[dict[str, Any]] = list(kwargs.get("records") or [])
    summaries: list[dict[str, Any]] = list(kwargs.get("summaries") or [])
    plan = dict(kwargs.get("plan") or {})
    bootstrap_samples = int(_to_float(kwargs.get("bootstrap_samples", 2000)))
    permutation_samples = int(_to_float(kwargs.get("permutation_samples", 20000)))
    random_seed = int(_to_float(kwargs.get("seed", 42)))
    by_arm = _summary_index(summaries)

    def _delta(target: str, baseline: str) -> dict[str, float] | None:
        if target not in by_arm or baseline not in by_arm:
            return None
        t = by_arm[target]
        b = by_arm[baseline]
        return {
            "success_rate_delta": _to_float(t.get("success_rate")) - _to_float(b.get("success_rate")),
            "feasibility_rate_delta": _to_float(t.get("feasibility_rate")) - _to_float(b.get("feasibility_rate")),
            "violation_rate_delta": _to_float(t.get("violation_rate")) - _to_float(b.get("violation_rate")),
            "transient_violation_rate_delta": (
                _to_float(t.get("transient_violation_rate")) - _to_float(b.get("transient_violation_rate"))
            ),
            "executed_violation_rate_delta": (
                _to_float(t.get("executed_violation_rate")) - _to_float(b.get("executed_violation_rate"))
            ),
            "unrecovered_violation_rate_delta": (
                _to_float(t.get("unrecovered_violation_rate"))
                - _to_float(b.get("unrecovered_violation_rate"))
            ),
            "avg_constraint_satisfaction_rate_delta": (
                _to_float(t.get("avg_constraint_satisfaction_rate"))
                - _to_float(b.get("avg_constraint_satisfaction_rate"))
            ),
            "avg_round_delta": _to_float(t.get("avg_round")) - _to_float(b.get("avg_round")),
            "avg_seller_reward_delta": _to_float(t.get("avg_seller_reward")) - _to_float(b.get("avg_seller_reward")),
            "avg_global_score_delta": _to_float(t.get("avg_global_score")) - _to_float(b.get("avg_global_score")),
            "avg_welfare_delta": _to_float(t.get("avg_welfare")) - _to_float(b.get("avg_welfare")),
            "avg_cost_adjusted_welfare_delta": (
                _to_float(t.get("avg_cost_adjusted_welfare"))
                - _to_float(b.get("avg_cost_adjusted_welfare"))
            ),
            "avg_latency_sec_delta": _to_float(t.get("avg_latency_sec")) - _to_float(b.get("avg_latency_sec")),
        }

    paired_statistics: dict[str, dict[str, Any] | None] = {
        "ocl_vs_single": None,
    }
    if "ocl_full" in by_arm and "single" in by_arm:
        metric_keys = (
            "success",
            "feasibility",
            "has_violation",
            "transient_has_violation",
            "executed_has_violation",
            "unrecovered_has_violation",
            "constraint_satisfaction_rate",
            "round",
            "seller_reward",
            "global_score",
            "welfare",
            "cost_adjusted_welfare",
            "escalation_count",
            "latency_sec",
        )
        paired_statistics["ocl_vs_single"] = {
            "target_arm": "ocl_full",
            "baseline_arm": "single",
            "pair_definition": "Matched by (episode_index, seed).",
            "bootstrap_samples": bootstrap_samples,
            "permutation_samples": permutation_samples,
            "metrics": {
                metric: _paired_metric_stats(
                    records=records,
                    target_arm="ocl_full",
                    baseline_arm="single",
                    metric_key=metric,
                    bootstrap_samples=bootstrap_samples,
                    permutation_samples=permutation_samples,
                    seed=random_seed,
                )
                for metric in metric_keys
            },
        }

    return {
        "implemented": True,
        "protocol": "offline_v1.main",
        "num_records": len(records),
        "arms": sorted(by_arm.keys()),
        "summary_by_arm": by_arm,
        "deltas": {
            "ocl_vs_single": _delta("ocl_full", "single"),
        },
        "paired_statistics": paired_statistics,
        "plan": plan,
    }


def build_main_result_artifact(main_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Standardize main-result payload into one paper-facing table artifact.

    Input:
        main_payload:
            ``offline_v1.main`` payload or compatible mapping.

    Output:
        Dict with ``available`` flag plus canonical comparison rows when the
        payload contains both ``single`` and ``ocl_full`` arms.

    中文翻译：将 main protocol payload 标准化为论文主结果表。"""
    payload = dict(main_payload or {})
    by_arm = payload.get("summary_by_arm")
    paired_stats = payload.get("paired_statistics")
    if not isinstance(by_arm, Mapping) or not isinstance(paired_stats, Mapping):
        return {
            "available": False,
            "protocol": "offline_v1.main_result",
            "reason": "Missing summary_by_arm or paired_statistics in main payload.",
        }

    baseline_arm = "single"
    target_arm = "ocl_full"
    if baseline_arm not in by_arm or target_arm not in by_arm:
        return {
            "available": False,
            "protocol": "offline_v1.main_result",
            "reason": "Main payload does not contain both single and ocl_full.",
        }

    pair_payload = paired_stats.get("ocl_vs_single")
    if not isinstance(pair_payload, Mapping):
        return {
            "available": False,
            "protocol": "offline_v1.main_result",
            "reason": "Main payload does not contain paired ocl_vs_single statistics.",
        }

    metric_payloads = pair_payload.get("metrics")
    if not isinstance(metric_payloads, Mapping):
        return {
            "available": False,
            "protocol": "offline_v1.main_result",
            "reason": "Paired statistics do not contain metric payloads.",
        }

    baseline_summary = dict(by_arm[baseline_arm])
    target_summary = dict(by_arm[target_arm])
    rows: list[dict[str, Any]] = []
    for metric_key, spec in _MAIN_RESULT_METRIC_SPECS.items():
        stats = metric_payloads.get(metric_key)
        if not isinstance(stats, Mapping):
            continue
        summary_key = str(spec["summary_key"])
        delta_ci95 = dict(stats.get("delta_ci95") or {})
        improvement_ci95 = dict(stats.get("improvement_ci95") or {})
        pvalues = dict(stats.get("sign_flip_pvalues") or {})
        direction = _metric_direction(metric_key)
        rows.append(
            {
                "metric_key": metric_key,
                "metric_label": str(spec["label"]),
                "summary_key": summary_key,
                "higher_is_better": direction > 0,
                "direction_sign": direction,
                "single_value": baseline_summary.get(summary_key),
                "ocl_full_value": target_summary.get(summary_key),
                "delta_ocl_minus_single": stats.get("mean_delta"),
                "delta_ci95_lower": delta_ci95.get("lower"),
                "delta_ci95_upper": delta_ci95.get("upper"),
                "improvement_ocl_vs_single": stats.get("mean_improvement"),
                "improvement_ci95_lower": improvement_ci95.get("lower"),
                "improvement_ci95_upper": improvement_ci95.get("upper"),
                "pairs": stats.get("pairs"),
                "p_one_sided": pvalues.get("p_one_sided"),
                "p_two_sided": pvalues.get("p_two_sided"),
                "pvalue_method": pvalues.get("method"),
                "pvalue_samples": pvalues.get("samples"),
                "alternative": stats.get("alternative"),
            }
        )

    return {
        "available": True,
        "protocol": "offline_v1.main_result",
        "baseline_arm": baseline_arm,
        "target_arm": target_arm,
        "pair_definition": pair_payload.get("pair_definition"),
        "summary_by_arm": {
            baseline_arm: baseline_summary,
            target_arm: target_summary,
        },
        "rows": rows,
    }


def _protocol_ablation_offline_v1(**kwargs: Any) -> dict[str, Any]:
    """Compute algorithm-component ablation slices from run records.

    Input:
        records:
            Per-run records from batch harness.

    Output:
        Dict with grouped aggregates by algorithm-id tuple.
    

    中文翻译：计算 algorithm-component ablation slices from run records。"""
    records: list[dict[str, Any]] = list(kwargs.get("records") or [])
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        if str(row.get("runner_mode")) != "ocl":
            continue
        key = "|".join(
            (
                str(row.get("role_algorithm_id")),
                str(row.get("gate_algorithm_id")),
                str(row.get("escalation_algorithm_id")),
                str(row.get("audit_algorithm_id")),
                str(row.get("attribution_algorithm_id")),
            )
        )
        grouped.setdefault(key, []).append(row)

    def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
        episodes = len(rows)
        if episodes == 0:
            return {"episodes": 0}
        return {
            "episodes": episodes,
            "success_rate": mean(_to_float(r.get("success")) for r in rows),
            "feasibility_rate": mean(_to_float(r.get("feasibility", r.get("success"))) for r in rows),
            "violation_rate": mean(_to_float(r.get("has_violation")) for r in rows),
            "avg_constraint_satisfaction_rate": mean(
                _to_float(r.get("constraint_satisfaction_rate")) for r in rows
            ),
            "avg_round": mean(_to_float(r.get("round")) for r in rows),
            "avg_seller_reward": mean(_to_float(r.get("seller_reward")) for r in rows),
            "avg_global_score": mean(_to_float(r.get("global_score")) for r in rows),
            "avg_welfare": mean(_to_float(r.get("welfare")) for r in rows),
            "avg_cost_adjusted_welfare": mean(
                _to_float(r.get("cost_adjusted_welfare")) for r in rows
            ),
            "avg_latency_sec": mean(_to_float(r.get("latency_sec")) for r in rows),
        }

    return {
        "implemented": True,
        "protocol": "offline_v1.ablation",
        "grouped_results": {key: _agg(rows) for key, rows in grouped.items()},
    }


def _protocol_adversarial_offline_v1(**kwargs: Any) -> dict[str, Any]:
    """Compute adversarial robustness summary when tagged records exist.

    Input:
        records:
            Per-run records. Adversarial rows must carry
            ``scenario_tag=adversarial``.

    Output:
        Dict with robustness summary or unavailable marker.
    

    中文翻译：计算 adversarial robustness summary when tagged records exist。"""
    records: list[dict[str, Any]] = list(kwargs.get("records") or [])
    adv = [row for row in records if str(row.get("scenario_tag")) == "adversarial"]
    if not adv:
        return {
            "implemented": True,
            "protocol": "offline_v1.adversarial",
            "available": False,
            "reason": "No records with scenario_tag=adversarial.",
        }

    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in adv:
        by_arm.setdefault(str(row.get("arm")), []).append(row)

    return {
        "implemented": True,
        "protocol": "offline_v1.adversarial",
        "available": True,
        "arm_robustness": {
            arm: {
                "episodes": len(rows),
                "robust_success_rate": mean(_to_float(r.get("success")) for r in rows),
                "adversarial_violation_rate": mean(_to_float(r.get("has_violation")) for r in rows),
                "avg_constraint_satisfaction_rate": mean(
                    _to_float(r.get("constraint_satisfaction_rate")) for r in rows
                ),
            }
            for arm, rows in by_arm.items()
        },
    }


def _protocol_repeated_offline_v1(**kwargs: Any) -> dict[str, Any]:
    """Compute repeated-interaction trend over episode index.

    Input:
        records:
            Per-run records containing ``episode_index``.

    Output:
        Dict with per-arm per-episode trend rows.
    

    中文翻译：计算 repeated-interaction trend over episode index。"""
    records: list[dict[str, Any]] = list(kwargs.get("records") or [])
    by_arm_episode: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in records:
        arm = str(row.get("arm"))
        episode_index = int(_to_float(row.get("episode_index")))
        by_arm_episode.setdefault((arm, episode_index), []).append(row)

    trend_rows: list[dict[str, Any]] = []
    for (arm, episode_index), rows in sorted(by_arm_episode.items()):
        trend_rows.append(
            {
                "arm": arm,
                "episode_index": episode_index,
                "episodes": len(rows),
                "success_rate": mean(_to_float(r.get("success")) for r in rows),
                "violation_rate": mean(_to_float(r.get("has_violation")) for r in rows),
                "avg_seller_reward": mean(_to_float(r.get("seller_reward")) for r in rows),
                "avg_welfare": mean(_to_float(r.get("welfare")) for r in rows),
                "avg_cost_adjusted_welfare": mean(
                    _to_float(r.get("cost_adjusted_welfare")) for r in rows
                ),
            }
        )

    return {
        "implemented": True,
        "protocol": "offline_v1.repeated",
        "trend": trend_rows,
    }


def _protocol_roi_offline_v1(**kwargs: Any) -> dict[str, Any]:
    """Compute cost-adjusted ROI proxy from summary rows.

    Input:
        summaries:
            Per-arm summary rows.
        alpha/beta/gamma/delta:
            Optional scalar coefficients.

    Output:
        Dict with per-arm ROI proxy.
    

    中文翻译：计算 cost-adjusted ROI proxy from summary rows。"""
    summaries: list[dict[str, Any]] = list(kwargs.get("summaries") or [])
    alpha = _to_float(kwargs.get("alpha", 1.0))
    beta = _to_float(kwargs.get("beta", 0.2))
    gamma = _to_float(kwargs.get("gamma", 0.8))
    delta = _to_float(kwargs.get("delta", 0.1))

    roi_by_arm: dict[str, dict[str, float]] = {}
    for row in summaries:
        arm = str(row.get("arm"))
        avg_reward = _to_float(row.get("avg_seller_reward"))
        avg_welfare = _to_float(row.get("avg_welfare"))
        success_rate = _to_float(row.get("success_rate"))
        violation_rate = _to_float(row.get("violation_rate"))
        avg_latency = _to_float(row.get("avg_latency_sec"))

        gmv_proxy = success_rate * max(0.0, avg_reward)
        welfare_proxy = success_rate * max(0.0, avg_welfare)
        human_cost_proxy = (1.0 - success_rate) * max(0.0, avg_latency)
        risk_loss_proxy = violation_rate
        latency_cost_proxy = avg_latency

        roi_value = (
            alpha * gmv_proxy
            - beta * human_cost_proxy
            - gamma * risk_loss_proxy
            - delta * latency_cost_proxy
        )
        roi_by_arm[arm] = {
            "roi_proxy": roi_value,
            "gmv_proxy": gmv_proxy,
            "welfare_proxy": welfare_proxy,
            "human_cost_proxy": human_cost_proxy,
            "risk_loss_proxy": risk_loss_proxy,
            "latency_cost_proxy": latency_cost_proxy,
        }

    return {
        "implemented": True,
        "protocol": "offline_v1.roi",
        "coefficients": {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta},
        "roi_by_arm": roi_by_arm,
    }


EXPERIMENT_PROTOCOL_REGISTRY: dict[str, ExperimentProtocolBundle] = {
    "offline_v1": ExperimentProtocolBundle(
        protocol_id="offline_v1",
        description=(
            "Default offline protocol slot for main/ablation/adversarial/"
            "repeated/ROI experiments."
        ),
        run_main_fn=_protocol_main_offline_v1,
        run_ablation_fn=_protocol_ablation_offline_v1,
        run_adversarial_fn=_protocol_adversarial_offline_v1,
        run_repeated_fn=_protocol_repeated_offline_v1,
        run_roi_fn=_protocol_roi_offline_v1,
    ),
}


def resolve_algorithm_bundle(bundle_id: str) -> AlgorithmBundle:
    """Resolve one default algorithm bundle id from registry.

中文翻译：解析 one default algorithm bundle id from registry。"""
    try:
        return ALGORITHM_BUNDLE_REGISTRY[bundle_id]
    except KeyError as exc:
        available = ", ".join(sorted(ALGORITHM_BUNDLE_REGISTRY))
        raise ValueError(
            f"Unknown algorithm bundle '{bundle_id}'. Available: {available}"
        ) from exc


def resolve_experiment_protocol(protocol_id: str) -> ExperimentProtocolBundle:
    """Resolve one experiment protocol id from registry.

中文翻译：解析 one experiment protocol id from registry。"""
    try:
        return EXPERIMENT_PROTOCOL_REGISTRY[protocol_id]
    except KeyError as exc:
        available = ", ".join(sorted(EXPERIMENT_PROTOCOL_REGISTRY))
        raise ValueError(
            f"Unknown experiment protocol '{protocol_id}'. Available: {available}"
        ) from exc
