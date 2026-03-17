"""Step-8 role-level attribution baseline with exact Shapley interfaces.

This module intentionally exposes the four interface points requested in the
project TODO:

- ``run_episode(role_mask=S, seed=...) -> trace``
- ``compute_V(trace) -> float``
- ``fallback_policy(role, state) -> action`` (deterministic)
- ``compute_shapley({V(S)}) -> {phi_i, w_i}``

中文翻译：
该模块对应 TODO 中 step-8 的最小可运行归因闭环。它不追求复杂策略学习，
而是优先把反事实运行、价值函数、Shapley 计算这三层接口稳定下来，
方便后续替换算法实现。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations
import math
import random
from typing import Any, Callable, Mapping

from aimai_ocl.controllers import Coordinator
from aimai_ocl.runners.ocl_episode import run_ocl_negotiation_episode
from aimai_ocl.schemas.actions import ActionRole
from aimai_ocl.schemas.audit import AuditEventType, EpisodeTrace


CONTROLLED_ROLES: tuple[str, ...] = ("platform", "seller", "expert")


@dataclass(frozen=True, slots=True)
class ValueFunctionConfig:
    """Config for deterministic trajectory value function.

    Inputs:
        success_weight:
            Multiplier for terminal success flag.
        seller_reward_weight:
            Multiplier for seller reward from final metrics.
        violation_penalty:
            Penalty per failed hard constraint check.
        round_penalty:
            Penalty per negotiation round.
        escalation_penalty:
            Penalty per escalation event.

    Output:
        Immutable config used by ``compute_V``.

    中文翻译：
    这是轨迹价值函数的权重配置。默认值偏向“先可解释、再优化”，用于
    对比实验和 attribution 接口联调。
    """

    success_weight: float = 5.0
    seller_reward_weight: float = 1.0
    violation_penalty: float = 2.0
    round_penalty: float = 0.1
    escalation_penalty: float = 0.2


def fallback_policy(role: str, state: Mapping[str, Any]) -> str:
    """Return deterministic fallback seller-side action text.

    Input:
        role:
            One role label such as ``seller`` / ``platform`` / ``expert``.
        state:
            Runtime state that may include ``buyer_max_price``,
            ``seller_min_price``, and product metadata.

    Output:
        One deterministic action string that can be sent to AgenticPay env.

    中文翻译：
    当某角色在 role mask 中被关闭时，使用该策略生成确定性动作，
    确保 counterfactual 对比不会被随机性污染。
    """
    buyer_max = _coerce_float(state.get("buyer_max_price"))
    seller_min = _coerce_float(state.get("seller_min_price"))
    product_price = _coerce_float(state.get("product_price"))

    if (buyer_max is not None) and (seller_min is not None) and seller_min <= buyer_max:
        price = (buyer_max + seller_min) / 2.0
    elif buyer_max is not None:
        price = buyer_max
    elif seller_min is not None:
        price = seller_min
    elif product_price is not None:
        price = product_price
    else:
        price = 100.0

    role_tag = str(role).strip().lower() or "seller"
    return f"[{role_tag}] deterministic offer ${price:.2f}"


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
    """Run one OCL episode under role mask for counterfactual attribution.

    Input:
        role_mask:
            Active controlled role set ``S``. Roles not in ``S`` are replaced by
            deterministic fallback behavior.
        seed:
            Deterministic random seed for this episode run.
        env_id:
            AgenticPay environment id.
        buyer_agent:
            Buyer/user simulator instance.
        seller_agent:
            Seller agent instance for full-policy trajectory.
        env_kwargs:
            Environment kwargs forwarded to runner.
        reset_kwargs:
            Reset payload forwarded to runner.
        trace_metadata:
            Optional metadata merged into output trace metadata.
        fallback_action:
            Deterministic fallback policy callable.

    Output:
        Completed ``EpisodeTrace`` for the masked run.

    中文翻译：
    在固定 seed 下执行一次带 role mask 的 OCL 回合。被屏蔽角色会被
    fallback policy 接管，最后返回可审计的 EpisodeTrace。
    """
    _apply_seed(seed)
    normalized_mask = _normalize_role_mask(role_mask)

    masked_seller = _RoleMaskedSellerAgent(
        seller_agent=seller_agent,
        role_mask=normalized_mask,
        fallback_action=fallback_action,
        base_state=dict(env_kwargs),
    )
    masked_coordinator = _RoleMaskCoordinator(
        base=Coordinator(),
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
        },
        coordinator=masked_coordinator,
    )
    return trace


def compute_V(
    trace: EpisodeTrace,
    *,
    config: ValueFunctionConfig = ValueFunctionConfig(),
) -> float:
    """Compute one deterministic scalar value from trajectory trace.

    Input:
        trace:
            Completed episode trace.
        config:
            Value-function weights.

    Output:
        Scalar utility value ``V(trace)`` for Shapley attribution.

    中文翻译：
    将轨迹压缩成标量价值。当前实现综合 success、seller reward、
    violation、round 数与 escalation，便于后续做 Shapley credit assignment。
    """
    final_metrics = trace.final_metrics if isinstance(trace.final_metrics, dict) else {}
    status = str(trace.final_status or final_metrics.get("status") or "").strip().lower()
    success = 1.0 if status == "agreed" else 0.0
    seller_reward = _coerce_float(final_metrics.get("seller_reward")) or 0.0
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
        - config.violation_penalty * float(violations)
        - config.round_penalty * rounds
        - config.escalation_penalty * float(escalations)
    )
    return float(value)


def compute_shapley(
    subset_values: Mapping[frozenset[str], float],
    *,
    roles: tuple[str, ...] = CONTROLLED_ROLES,
) -> dict[str, dict[str, float]]:
    """Compute exact Shapley contribution and normalized weight for roles.

    Input:
        subset_values:
            Mapping ``S -> V(S)`` where ``S`` is a role subset represented by
            ``frozenset[str]``.
        roles:
            Ordered role universe used for exact Shapley computation.

    Output:
        Dict with:
        - ``phi``: raw Shapley values by role
        - ``w``: normalized positive-weight shares by role

    中文翻译：
    根据完整子集取值 ``V(S)`` 计算精确 Shapley。输出既包含原始贡献
    ``phi``，也包含便于做权重分配的归一化 ``w``。
    """
    role_set = tuple(str(role).strip().lower() for role in roles)
    _validate_subset_values(subset_values, role_set)

    n = len(role_set)
    factorial_n = math.factorial(n)
    phi: dict[str, float] = {role: 0.0 for role in role_set}

    for role in role_set:
        others = [r for r in role_set if r != role]
        for size in range(len(others) + 1):
            weight = math.factorial(size) * math.factorial(n - size - 1) / factorial_n
            for subset in combinations(others, size):
                s = frozenset(subset)
                s_with_role = frozenset((*subset, role))
                marginal = float(subset_values[s_with_role]) - float(subset_values[s])
                phi[role] += weight * marginal

    positive_mass = sum(max(v, 0.0) for v in phi.values())
    if positive_mass <= 0:
        weights = {role: 0.0 for role in phi}
    else:
        weights = {role: max(v, 0.0) / positive_mass for role, v in phi.items()}

    return {
        "phi": phi,
        "w": weights,
    }


@dataclass(slots=True)
class _RoleMaskedSellerAgent:
    """Seller wrapper applying deterministic fallback when seller role masked.

    Input:
        seller_agent:
            Original seller agent object.
        role_mask:
            Active controlled role set.
        fallback_action:
            Deterministic fallback policy callable.

    Output:
        Agent-like object with ``respond(...)`` interface.

    中文翻译：
    这是对 seller agent 的轻量包装器。它保持原 agent 行为不变，
    仅在 seller 角色被 mask 时注入确定性 fallback。
    """

    seller_agent: Any
    role_mask: set[str]
    fallback_action: Callable[[str, Mapping[str, Any]], str]
    base_state: dict[str, Any] = field(default_factory=dict)

    def initialize(self, context: dict[str, Any]) -> None:
        """Initialize wrapped seller agent and sync fallback state.

        Input:
            context:
                Seller context provided by AgenticPay environment reset.

        Output:
            None. Delegates initialization to wrapped seller and stores a
            normalized snapshot for deterministic fallback policy.

        中文翻译：
        透传 initialize 调用，避免与 AgenticPay 生命周期不兼容；
        同时缓存上下文，供 fallback policy 在无 seller 角色时使用。
        """
        self.base_state.update(context or {})
        initialize_fn = getattr(self.seller_agent, "initialize", None)
        if callable(initialize_fn):
            initialize_fn(context)

    @property
    def name(self) -> str:
        return str(getattr(self.seller_agent, "name", "seller"))

    def respond(
        self,
        conversation_history: list[dict[str, Any]],
        current_state: dict[str, Any],
    ) -> str:
        """Return seller action or deterministic fallback based on role mask.

        Input:
            conversation_history:
                Conversation history from runner.
            current_state:
                Current environment state from runner.

        Output:
            One action string.

        中文翻译：
        如果 seller 在 role mask 中启用，就调用原 seller_agent；
        否则返回确定性 fallback 动作文本。
        """
        merged_state = {**self.base_state, **current_state}
        if "seller" not in self.role_mask:
            return self.fallback_action("seller", merged_state)
        return str(
            self.seller_agent.respond(
                conversation_history=conversation_history,
                current_state=merged_state,
            )
        )

    def __getattr__(self, item: str) -> Any:
        """Delegate unknown attributes to wrapped seller agent.

        中文翻译：
        对未显式覆盖的属性统一委托给原 seller agent，尽量保持接口透明。
        """
        return getattr(self.seller_agent, item)


@dataclass(slots=True)
class _RoleMaskCoordinator:
    """Coordinator adapter enforcing role-mask counterfactual behavior.

    中文翻译：
    对 Coordinator 的最小适配层：优先保留原决策逻辑，只在角色被 mask
    时改写 decision role，确保回合仍可执行。
    """

    base: Coordinator
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
        """Plan round roles while masking unavailable roles deterministically.

        中文翻译：
        先调用原协调器出计划，再把不可用角色替换为可用角色，并在 reason/metadata
        中留下可追踪说明。
        """
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
    """Normalize user-provided role mask to controlled role vocabulary.

    中文翻译：
    对 role mask 做清洗和过滤，只保留 ``CONTROLLED_ROLES`` 中的合法角色。
    """
    normalized = {str(role).strip().lower() for role in role_mask}
    return {role for role in normalized if role in CONTROLLED_ROLES}


def _choose_available_role(*, preferred: str, role_mask: set[str]) -> str:
    """Choose one executable fallback role when preferred role is unavailable.

    中文翻译：
    若首选角色不可用，按 ``seller -> expert -> platform`` 顺序降级。
    """
    p = str(preferred).strip().lower()
    if p in role_mask:
        return p
    for fallback in ("seller", "expert", "platform"):
        if fallback in role_mask:
            return fallback
    return "seller"


def _validate_subset_values(
    subset_values: Mapping[frozenset[str], float],
    roles: tuple[str, ...],
) -> None:
    """Validate that exact-Shapley subset coverage is complete.

    中文翻译：
    精确 Shapley 需要完整幂集 ``2^N`` 的 ``V(S)``。若缺子集则直接报错。
    """
    expected: set[frozenset[str]] = set()
    role_list = list(roles)
    for size in range(len(role_list) + 1):
        for subset in combinations(role_list, size):
            expected.add(frozenset(subset))

    missing = [subset for subset in expected if subset not in subset_values]
    if missing:
        raise ValueError(
            "subset_values missing role subsets required for exact Shapley: "
            f"{missing}"
        )


def _apply_seed(seed: int) -> None:
    """Apply deterministic seed to Python and NumPy.

    中文翻译：
    设置随机种子，保证角色子集对比时的可复现性。
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        return None


def _coerce_float(value: Any) -> float | None:
    """Best-effort float parsing helper.

    中文翻译：
    宽松地把输入转成 float，失败则返回 ``None``。
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
