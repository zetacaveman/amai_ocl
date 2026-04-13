"""Shapley value attribution for measuring each role's contribution.

Runs counterfactual episodes with roles masked out, then computes
exact Shapley values to determine how much each role (platform, seller,
expert) contributes to negotiation outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import math
import random
from typing import Any, Callable, Mapping

from aimai_ocl.coordinator import Coordinator, CoordinationPlan
from aimai_ocl.runner import run_episode
from aimai_ocl.schemas import ActionRole, AuditEventType, EpisodeTrace


CONTROLLED_ROLES: tuple[str, ...] = ("platform", "seller", "expert")


# ---------------------------------------------------------------------------
# Value function
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ValueConfig:
    """Weights for the trajectory value function."""
    success_weight: float = 5.0
    seller_reward_weight: float = 1.0
    violation_penalty: float = 2.0
    round_penalty: float = 0.1
    escalation_penalty: float = 0.2


def compute_V(trace: EpisodeTrace, *, config: ValueConfig = ValueConfig()) -> float:
    """Compute a scalar value from one episode trace."""
    metrics = trace.final_metrics if isinstance(trace.final_metrics, dict) else {}
    status = str(trace.final_status or metrics.get("status") or "").strip().lower()
    success = 1.0 if status == "agreed" else 0.0
    seller_reward = _float(metrics.get("seller_reward")) or 0.0
    rounds = _float(metrics.get("round")) or 0.0

    violations = 0
    escalations = 0
    for event in trace.events:
        if event.event_type == AuditEventType.ESCALATION_TRIGGERED:
            escalations += 1
        if event.event_type == AuditEventType.CONSTRAINT_EVALUATED:
            violations += sum(1 for c in event.constraint_checks if not c.passed)

    return float(
        config.success_weight * success
        + config.seller_reward_weight * seller_reward
        - config.violation_penalty * violations
        - config.round_penalty * rounds
        - config.escalation_penalty * escalations
    )


# ---------------------------------------------------------------------------
# Fallback policy
# ---------------------------------------------------------------------------


def fallback_policy(role: str, state: Mapping[str, Any]) -> str:
    """Deterministic fallback action when a role is masked out."""
    buyer_max = _float(state.get("buyer_max_price"))
    seller_min = _float(state.get("seller_min_price"))

    if buyer_max is not None and seller_min is not None and seller_min <= buyer_max:
        price = (buyer_max + seller_min) / 2.0
    elif buyer_max is not None:
        price = buyer_max
    elif seller_min is not None:
        price = seller_min
    else:
        price = 100.0

    tag = str(role).strip().lower() or "seller"
    return f"[{tag}] deterministic offer ${price:.2f}"


# ---------------------------------------------------------------------------
# Masked episode runner
# ---------------------------------------------------------------------------


def run_masked_episode(
    *,
    role_mask: set[str] | frozenset[str],
    seed: int,
    env_id: str,
    buyer_agent: Any,
    seller_agent: Any,
    env_kwargs: dict[str, Any],
    reset_kwargs: dict[str, Any],
    trace_metadata: dict[str, Any] | None = None,
    coordinator_mode: str = "default",
) -> EpisodeTrace:
    """Run one episode with inactive roles replaced by fallback policy."""
    _apply_seed(seed)
    mask = {r for r in (s.strip().lower() for s in role_mask) if r in CONTROLLED_ROLES}

    masked_seller = _MaskedSeller(
        seller=seller_agent, role_mask=mask, base_state=dict(env_kwargs),
    )
    masked_coord = _MaskedCoordinator(
        base=Coordinator(mode=coordinator_mode), role_mask=mask,
    )

    trace, _ = run_episode(
        env_id=env_id,
        buyer_agent=buyer_agent,
        seller_agent=masked_seller,
        env_kwargs=env_kwargs,
        reset_kwargs=reset_kwargs,
        trace_metadata={
            **(trace_metadata or {}),
            "role_mask": sorted(mask),
            "seed": seed,
        },
        ocl=True,
        coordinator=masked_coord,
    )
    return trace


# ---------------------------------------------------------------------------
# Shapley computation
# ---------------------------------------------------------------------------


def compute_shapley(
    subset_values: Mapping[frozenset[str], float],
    *,
    roles: tuple[str, ...] = CONTROLLED_ROLES,
) -> dict[str, dict[str, float]]:
    """Compute exact Shapley values and normalized weights.

    Args:
        subset_values: Mapping S -> V(S) for all 2^n role subsets.
        roles: Role universe.

    Returns:
        {"phi": {role: value}, "w": {role: weight}}
    """
    n = len(roles)
    factorial_n = math.factorial(n)
    phi: dict[str, float] = {r: 0.0 for r in roles}

    for role in roles:
        others = [r for r in roles if r != role]
        for size in range(len(others) + 1):
            weight = math.factorial(size) * math.factorial(n - size - 1) / factorial_n
            for subset in combinations(others, size):
                s = frozenset(subset)
                s_with = frozenset((*subset, role))
                if s in subset_values and s_with in subset_values:
                    phi[role] += weight * (subset_values[s_with] - subset_values[s])

    total_pos = sum(max(v, 0.0) for v in phi.values())
    if total_pos <= 0:
        weights = {r: 0.0 for r in phi}
    else:
        weights = {r: max(v, 0.0) / total_pos for r, v in phi.items()}

    return {"phi": phi, "w": weights}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _MaskedSeller:
    """Seller wrapper that uses fallback when seller role is masked."""
    seller: Any
    role_mask: set[str]
    base_state: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return str(getattr(self.seller, "name", "seller"))

    def respond(self, conversation_history: list, current_state: dict) -> str:
        merged = {**self.base_state, **current_state}
        if "seller" not in self.role_mask:
            return fallback_policy("seller", merged)
        return str(self.seller.respond(
            conversation_history=conversation_history, current_state=merged,
        ))

    def __getattr__(self, item: str) -> Any:
        return getattr(self.seller, item)


@dataclass(slots=True)
class _MaskedCoordinator:
    """Coordinator that falls back when preferred role is masked."""
    base: Coordinator
    role_mask: set[str]

    @property
    def orchestrator_id(self) -> str:
        return self.base.orchestrator_id

    def plan_turn(self, **kwargs: Any) -> CoordinationPlan:
        plan = self.base.plan_turn(**kwargs)
        preferred = plan.decision_role.value
        if preferred in self.role_mask:
            return plan
        # Fallback: pick first available role
        for fb in ("seller", "expert", "platform"):
            if fb in self.role_mask:
                plan.decision_role = ActionRole(fb)
                plan.reason += f" (fallback from masked {preferred})"
                return plan
        plan.decision_role = ActionRole.SELLER
        return plan

    def build_audit_event(self, plan: CoordinationPlan) -> Any:
        return self.base.build_audit_event(plan)


def _apply_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
