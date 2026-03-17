"""Counterfactual attribution algorithm module (step-8/9 research variant).

This module keeps the same four required interfaces while providing a more
algorithmic attribution baseline than the minimal exact-Shapley scaffold:

- ``run_episode(role_mask=S, seed=...) -> trace``
- ``compute_V(trace) -> float``
- ``fallback_policy(role, state) -> action``
- ``compute_shapley({V(S)}) -> {phi_i, w_i}``


中文翻译：Counterfactual attribution algorithm module (step-8/9 research variant)。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations
import math
import random
from typing import Any, Callable, Mapping

from aimai_ocl.controllers import StateMachineCoordinator
from aimai_ocl.runners.ocl_episode import run_ocl_negotiation_episode
from aimai_ocl.schemas.actions import ActionRole
from aimai_ocl.schemas.audit import AuditEventType, EpisodeTrace


CONTROLLED_ROLES: tuple[str, ...] = ("platform", "seller", "expert")


@dataclass(frozen=True, slots=True)
class CounterfactualValueConfig:
    """Weights for counterfactual trajectory value function.

    Input:
        success_weight:
            Reward weight for successful completion.
        seller_reward_weight:
            Weight for seller reward metric.
        global_score_weight:
            Weight for global score metric when available.
        violation_penalty:
            Penalty per failed hard constraint.
        escalation_penalty:
            Penalty per escalation event.
        round_penalty:
            Penalty per negotiation round.

    Output:
        Immutable config consumed by ``compute_V``.
    

    中文翻译：Weights for counterfactual trajectory value function。"""

    success_weight: float = 4.0
    seller_reward_weight: float = 1.0
    global_score_weight: float = 0.5
    violation_penalty: float = 2.0
    escalation_penalty: float = 0.25
    round_penalty: float = 0.15


def fallback_policy(role: str, state: Mapping[str, Any]) -> str:
    """Return deterministic role-specific fallback action.

    Input:
        role:
            Role label (``seller``, ``expert``, or ``platform``).
        state:
            Runtime state with optional ``buyer_max_price``,
            ``seller_min_price``, and ``product_name`` fields.

    Output:
        One deterministic action text for the requested role.
    

    中文翻译：返回 deterministic role-specific fallback action。"""
    normalized_role = str(role).strip().lower() or "seller"
    buyer_max = _coerce_float(state.get("buyer_max_price"))
    seller_min = _coerce_float(state.get("seller_min_price"))
    product_name = str(state.get("product_name") or "product")

    if (buyer_max is not None) and (seller_min is not None) and seller_min <= buyer_max:
        target_price = (buyer_max + seller_min) / 2.0
    elif buyer_max is not None:
        target_price = buyer_max
    elif seller_min is not None:
        target_price = seller_min
    else:
        target_price = 100.0

    if normalized_role == "expert":
        return (
            f"[expert] For {product_name}, balanced option is around "
            f"${target_price:.2f}."
        )
    if normalized_role == "platform":
        return (
            "[platform] To keep policy-safe execution, please confirm "
            f"final offer ${target_price:.2f}."
        )
    return f"[seller] deterministic counterfactual offer ${target_price:.2f}"


def run_episode(
    *,
    role_mask: set[str] | frozenset[str],
    seed: int,
    env_id: str,
    buyer_agent: Any,
    seller_agent: Any,
    env_kwargs: dict[str, Any],
    reset_kwargs: dict[str, Any],
    trace_metadata: dict[str, Any] | None = None,
    fallback_action: Callable[[str, Mapping[str, Any]], str] = fallback_policy,
) -> EpisodeTrace:
    """Run one OCL episode under role-mask counterfactual policy.

    Input:
        role_mask:
            Active role subset ``S``.
        seed:
            Deterministic run seed.
        env_id:
            AgenticPay env id.
        buyer_agent:
            External buyer/user simulator.
        seller_agent:
            Seller agent under evaluation.
        env_kwargs:
            Env construction kwargs.
        reset_kwargs:
            Scenario reset kwargs.
        trace_metadata:
            Optional metadata merged into final trace.
        fallback_action:
            Deterministic fallback action policy for masked roles.

    Output:
        Completed ``EpisodeTrace``.
    

    中文翻译：运行 one OCL episode under role-mask counterfactual policy。"""
    _apply_seed(seed)
    normalized_mask = _normalize_role_mask(role_mask)

    masked_seller = _RoleMaskedSellerAgent(
        seller_agent=seller_agent,
        role_mask=normalized_mask,
        fallback_action=fallback_action,
        base_state=dict(env_kwargs),
    )
    masked_coordinator = _RoleMaskStateMachineCoordinator(
        base=StateMachineCoordinator(),
        role_mask=normalized_mask,
    )

    trace, _final_info = run_ocl_negotiation_episode(
        env_id=env_id,
        buyer_agent=buyer_agent,
        seller_agent=masked_seller,
        env_kwargs=env_kwargs,
        reset_kwargs=reset_kwargs,
        trace_metadata={
            **(trace_metadata or {}),
            "role_mask": sorted(normalized_mask),
            "seed": seed,
            "attribution_algorithm": "counterfactual_v1",
        },
        coordinator=masked_coordinator,
    )
    return trace


def compute_V(
    trace: EpisodeTrace,
    *,
    config: CounterfactualValueConfig = CounterfactualValueConfig(),
) -> float:
    """Compute one scalar counterfactual trajectory value.

    Input:
        trace:
            Completed episode trace.
        config:
            Weight config for value components.

    Output:
        Scalar ``V(trace)``.
    

    中文翻译：计算 one scalar counterfactual trajectory value。"""
    final_metrics = trace.final_metrics if isinstance(trace.final_metrics, dict) else {}
    status = str(trace.final_status or final_metrics.get("status") or "").strip().lower()
    success = 1.0 if status == "agreed" else 0.0
    seller_reward = _coerce_float(final_metrics.get("seller_reward")) or 0.0
    global_score = _coerce_float(final_metrics.get("global_score")) or 0.0
    rounds = _coerce_float(final_metrics.get("round")) or 0.0

    violations = 0
    escalations = 0
    for event in trace.events:
        if event.event_type == AuditEventType.ESCALATION_TRIGGERED:
            escalations += 1
        if event.event_type != AuditEventType.CONSTRAINT_EVALUATED:
            continue
        violations += sum(1 for check in event.constraint_checks if not check.passed)

    value = (
        config.success_weight * success
        + config.seller_reward_weight * seller_reward
        + config.global_score_weight * global_score
        - config.violation_penalty * float(violations)
        - config.escalation_penalty * float(escalations)
        - config.round_penalty * rounds
    )
    return float(value)


def compute_shapley(
    subset_values: Mapping[frozenset[str], float],
    *,
    roles: tuple[str, ...] = CONTROLLED_ROLES,
    samples: int = 512,
    seed: int = 17,
    incentive_temperature: float = 1.0,
) -> dict[str, dict[str, float]]:
    """Compute role attribution via exact or Monte-Carlo Shapley.

    Input:
        subset_values:
            Coalition values map ``S -> V(S)``.
        roles:
            Controlled role universe.
        samples:
            Number of permutations for MC estimate when subsets are sparse.
        seed:
            RNG seed for deterministic MC estimate.
        incentive_temperature:
            Temperature for positive-share normalization.

    Output:
        Dict containing:
        - ``phi``: role contribution values
        - ``w``: normalized incentive shares
    

    中文翻译：计算 role attribution via exact or Monte-Carlo Shapley。"""
    role_set = tuple(str(role).strip().lower() for role in roles)
    if _has_complete_coalitions(subset_values, role_set):
        phi = _exact_shapley(subset_values, role_set)
    else:
        phi = _monte_carlo_shapley(
            subset_values=subset_values,
            roles=role_set,
            samples=max(1, int(samples)),
            seed=seed,
        )

    weights = _normalize_positive_shares(
        phi=phi,
        temperature=max(1e-6, float(incentive_temperature)),
    )
    return {
        "phi": phi,
        "w": weights,
    }


@dataclass(slots=True)
class _RoleMaskedSellerAgent:
    """Seller wrapper that applies fallback policy when seller role is masked.

中文翻译：Seller wrapper that applies fallback policy when seller role is masked。"""

    seller_agent: Any
    role_mask: set[str]
    fallback_action: Callable[[str, Mapping[str, Any]], str]
    base_state: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return str(getattr(self.seller_agent, "name", "seller"))

    def respond(
        self,
        conversation_history: list[dict[str, Any]],
        current_state: dict[str, Any],
    ) -> str:
        merged_state = {**self.base_state, **current_state}
        if "seller" not in self.role_mask:
            return self.fallback_action("seller", merged_state)
        return str(
            self.seller_agent.respond(
                conversation_history=conversation_history,
                current_state=merged_state,
            )
        )


@dataclass(slots=True)
class _RoleMaskStateMachineCoordinator:
    """State-machine coordinator adapter with role-mask fallback.

中文翻译：State-machine coordinator adapter with role-mask fallback。"""

    base: StateMachineCoordinator
    role_mask: set[str]

    @property
    def orchestrator_id(self) -> str:
        return self.base.orchestrator_id

    def plan_turn(
        self,
        *,
        round_id: int,
        buyer_text: str | None,
        seller_actor_id: str,
        max_rounds: int | None = None,
    ):  # noqa: ANN201
        plan = self.base.plan_turn(
            round_id=round_id,
            buyer_text=buyer_text,
            seller_actor_id=seller_actor_id,
            max_rounds=max_rounds,
        )
        preferred = plan.decision_role.value
        chosen = _choose_available_role(preferred=preferred, role_mask=self.role_mask)
        if chosen == preferred:
            return plan
        return replace(
            plan,
            decision_role=ActionRole(chosen),
            reason=(
                f"{plan.reason} Role `{preferred}` masked out; "
                f"fallback decision role=`{chosen}`."
            ),
            metadata={
                **plan.metadata,
                "masked_roles": sorted(set(CONTROLLED_ROLES) - self.role_mask),
            },
        )

    def build_audit_event(self, plan):  # noqa: ANN001, ANN201
        return self.base.build_audit_event(plan)


def _normalize_role_mask(role_mask: set[str] | frozenset[str]) -> set[str]:
    normalized = {str(role).strip().lower() for role in role_mask}
    return {role for role in normalized if role in CONTROLLED_ROLES}


def _choose_available_role(*, preferred: str, role_mask: set[str]) -> str:
    p = str(preferred).strip().lower()
    if p in role_mask:
        return p
    for fallback in ("seller", "expert", "platform"):
        if fallback in role_mask:
            return fallback
    return "seller"


def _has_complete_coalitions(
    subset_values: Mapping[frozenset[str], float],
    roles: tuple[str, ...],
) -> bool:
    expected: set[frozenset[str]] = set()
    role_list = list(roles)
    for size in range(len(role_list) + 1):
        for subset in combinations(role_list, size):
            expected.add(frozenset(subset))
    return all(subset in subset_values for subset in expected)


def _exact_shapley(
    subset_values: Mapping[frozenset[str], float],
    roles: tuple[str, ...],
) -> dict[str, float]:
    n = len(roles)
    factorial_n = math.factorial(n)
    phi: dict[str, float] = {role: 0.0 for role in roles}
    for role in roles:
        others = [r for r in roles if r != role]
        for size in range(len(others) + 1):
            weight = math.factorial(size) * math.factorial(n - size - 1) / factorial_n
            for subset in combinations(others, size):
                s = frozenset(subset)
                s_with_role = frozenset((*subset, role))
                marginal = float(subset_values[s_with_role]) - float(subset_values[s])
                phi[role] += weight * marginal
    return phi


def _monte_carlo_shapley(
    *,
    subset_values: Mapping[frozenset[str], float],
    roles: tuple[str, ...],
    samples: int,
    seed: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    phi: dict[str, float] = {role: 0.0 for role in roles}
    roles_list = list(roles)

    for _ in range(samples):
        perm = roles_list[:]
        rng.shuffle(perm)
        coalition: set[str] = set()
        prev_value = _estimate_value(frozenset(coalition), subset_values)
        for role in perm:
            coalition.add(role)
            cur_value = _estimate_value(frozenset(coalition), subset_values)
            phi[role] += cur_value - prev_value
            prev_value = cur_value

    inv = 1.0 / float(samples)
    for role in phi:
        phi[role] *= inv
    return phi


def _estimate_value(
    subset: frozenset[str],
    subset_values: Mapping[frozenset[str], float],
) -> float:
    if subset in subset_values:
        return float(subset_values[subset])

    candidates = [known for known in subset_values if known.issubset(subset)]
    if candidates:
        best = max(candidates, key=len)
        return float(subset_values[best])

    supersets = [known for known in subset_values if subset.issubset(known)]
    if supersets:
        best = min(supersets, key=len)
        return float(subset_values[best])

    return 0.0


def _normalize_positive_shares(
    *,
    phi: Mapping[str, float],
    temperature: float,
) -> dict[str, float]:
    positive = {role: max(float(value), 0.0) for role, value in phi.items()}
    if all(value <= 0.0 for value in positive.values()):
        return {role: 0.0 for role in positive}

    if temperature == 1.0:
        total = sum(positive.values())
        return {role: value / total for role, value in positive.items()}

    scaled = {role: value ** (1.0 / temperature) for role, value in positive.items()}
    total = sum(scaled.values())
    if total <= 0:
        return {role: 0.0 for role in scaled}
    return {role: value / total for role, value in scaled.items()}


def _apply_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
