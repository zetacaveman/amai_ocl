"""OCL-controlled episode runner using role policy and risk gate decisions.

中文翻译：OCL-controlled episode runner using role policy and risk gate decisions。"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from aimai_ocl.adapters import AgenticPayEnvAdapter
from aimai_ocl.adapters.agenticpay_actions import raw_action_from_text
from aimai_ocl.controllers import (
    AuditPolicy,
    Coordinator,
    EscalationManager,
    OCLController,
)
from aimai_ocl.runners.scenario_validation import enforce_single_product_scenario
from aimai_ocl.schemas.actions import ActionRole
from aimai_ocl.schemas.audit import AuditEvent, AuditEventType, EpisodeTrace


@dataclass(slots=True)
class _ControlApplyResult:
    """Result of applying OCL control plus escalation policy to one action.

    Input:
        executable_text:
            Seller text approved for env execution, or ``None``.
        used_replan:
            Whether deterministic one-shot replan path was used.
        requires_human_handoff:
            Whether escalation policy recommends human/platform handoff.

    Output:
        Structured payload consumed by runner loop.
    

    中文翻译：Result of applying OCL control plus escalation policy to one action。"""

    executable_text: str | None
    used_replan: bool
    requires_human_handoff: bool


def _normalize_passthrough_text(text: str | None) -> str | None:
    """Normalize external pass-through text before env stepping.

    Input:
        text:
            Raw text returned by an external actor (buyer simulator), or
            ``None`` if no message is emitted.

    Output:
        Stripped non-empty string for env consumption, or ``None`` when empty.
    

    中文翻译：规范化 external pass-through text before env stepping。"""
    if text is None:
        return None
    normalized = text.strip()
    return normalized or None


def _apply_control_to_text(
    *,
    trace: EpisodeTrace,
    controller: OCLController,
    escalation_manager: EscalationManager,
    round_id: int,
    actor_id: str,
    actor_role: ActionRole,
    text: str | None,
    history: list[dict[str, Any]],
    state: dict[str, Any],
    audit_policy: AuditPolicy | None,
) -> _ControlApplyResult:
    """Run one text action through the OCL controller and append audit events.

    Args:
        trace: Episode trace mutated in place.
        controller: OCL controller instance.
        round_id: Logical round index of the action.
        actor_id: Stable actor identifier.
        actor_role: Actor role label (must not be ``BUYER`` in current scope).
        text: Raw text action, or ``None``.
        history: Conversation history visible to the controller.
        state: Current environment observation/state.

    Output:
        ``_ControlApplyResult`` including final executable text (if any),
        whether replan path was used, and whether human handoff is needed.
    

    中文翻译：运行 one text action through the OCL controller and append audit events。"""
    if actor_role == ActionRole.BUYER:
        raise ValueError(
            "Buyer actions must not enter OCL control path. "
            "Use passthrough flow for external user-simulator inputs.",
        )
    if text is None:
        return _ControlApplyResult(
            executable_text=None,
            used_replan=False,
            requires_human_handoff=False,
        )

    raw_action = raw_action_from_text(
        actor_id=actor_id,
        actor_role=actor_role,
        text=text,
    )
    control_result = controller.apply(
        raw_action,
        round_id=round_id,
        history=history,
        state=state,
    )

    for event in control_result.audit_events:
        _trace_add_event(trace, event, audit_policy)

    executable_action = control_result.executable_action
    violations = list(control_result.metadata.get("violations", []))
    outcome = escalation_manager.resolve(
        round_id=round_id,
        actor_id=actor_id,
        raw_action=raw_action,
        approved=executable_action.approved,
        requires_confirmation=executable_action.requires_confirmation,
        requires_escalation=executable_action.requires_escalation,
        violations=violations,
        state=state,
        allow_replan=True,
    )
    for event in outcome.audit_events:
        _trace_add_event(trace, event, audit_policy)

    if outcome.replan_text is None:
        final_text = (outcome.final_text or "").strip()
        return _ControlApplyResult(
            executable_text=(final_text or None),
            used_replan=False,
            requires_human_handoff=outcome.requires_human_handoff,
        )

    replanned_raw = raw_action_from_text(
        actor_id=actor_id,
        actor_role=actor_role,
        text=outcome.replan_text,
    )
    replanned_result = controller.apply(
        replanned_raw,
        round_id=round_id,
        history=history,
        state=state,
    )
    for event in replanned_result.audit_events:
        _trace_add_event(trace, event, audit_policy)

    replanned_exec = replanned_result.executable_action
    replanned_violations = list(replanned_result.metadata.get("violations", []))
    second_outcome = escalation_manager.resolve(
        round_id=round_id,
        actor_id=actor_id,
        raw_action=replanned_raw,
        approved=replanned_exec.approved,
        requires_confirmation=replanned_exec.requires_confirmation,
        requires_escalation=replanned_exec.requires_escalation,
        violations=replanned_violations,
        state=state,
        allow_replan=False,
    )
    for event in second_outcome.audit_events:
        _trace_add_event(trace, event, audit_policy)

    final_text = (second_outcome.final_text or "").strip()
    return _ControlApplyResult(
        executable_text=(final_text or None),
        used_replan=True,
        requires_human_handoff=second_outcome.requires_human_handoff,
    )


def _trace_add_event(
    trace: EpisodeTrace,
    event: AuditEvent,
    audit_policy: AuditPolicy | None,
) -> None:
    """Append one event when allowed by current audit policy."""
    if audit_policy is None or audit_policy.should_record(event.event_type):
        trace.add_event(event)


def run_ocl_negotiation_episode(
    env_id: str,
    buyer_agent: Any,
    seller_agent: Any,
    reset_kwargs: dict[str, Any],
    *,
    env_kwargs: dict[str, Any] | None = None,
    trace_metadata: dict[str, Any] | None = None,
    controller: OCLController | None = None,
    coordinator: Coordinator | None = None,
    escalation_manager: EscalationManager | None = None,
    audit_policy: AuditPolicy | None = None,
) -> tuple[EpisodeTrace, dict[str, Any]]:
    """Run one negotiation episode with seller-side OCL control decisions.

    Buyer actions are treated as external user-simulator inputs and are passed
    directly to the environment without OCL events/checks. Seller actions are
    transformed via OCL.

    Args:
        env_id: AgenticPay environment id.
        buyer_agent: External user-simulator-like buyer agent.
        seller_agent: Seller-side agent controlled through OCL.
        reset_kwargs: Scenario reset payload for the environment.
        env_kwargs: Optional environment construction kwargs.
        trace_metadata: Optional metadata stored on the episode trace.
        controller: Optional prebuilt OCL controller.
        coordinator: Optional prebuilt round coordinator for role assignment.
        escalation_manager: Optional escalation/replan policy manager.

    Returns:
        A tuple ``(trace, final_info)`` where ``trace`` is the full audit trace
        and ``final_info`` is terminal env info from the last step.
    

    中文翻译：运行 one negotiation episode with seller-side OCL control decisions。"""
    normalized_reset_kwargs = enforce_single_product_scenario(reset_kwargs)
    env_config = dict(env_kwargs or {})
    env_config["buyer_agent"] = buyer_agent
    env_config["seller_agent"] = seller_agent

    adapter = AgenticPayEnvAdapter(env_id=env_id, env_kwargs=env_config)
    observation, _reset_info = adapter.reset(**normalized_reset_kwargs)
    trace = adapter.new_episode_trace(
        scenario=normalized_reset_kwargs,
        metadata=trace_metadata,
    )
    if (
        audit_policy is not None
        and not audit_policy.should_record(AuditEventType.EPISODE_STARTED)
    ):
        trace.events = [
            event
            for event in trace.events
            if event.event_type != AuditEventType.EPISODE_STARTED
        ]

    ocl_controller = controller or OCLController()
    ocl_coordinator = coordinator or Coordinator()
    ocl_escalation = escalation_manager or EscalationManager()
    done = False
    final_info: dict[str, Any] = {}
    control_context = {
        "buyer_max_price": env_config.get("buyer_max_price"),
        "seller_min_price": env_config.get("seller_min_price"),
        "max_rounds": env_config.get("max_rounds"),
    }
    product_info = normalized_reset_kwargs.get("product_info")
    if isinstance(product_info, dict):
        control_context["product_name"] = product_info.get("name")
        control_context["product_price"] = product_info.get("price")

    while not done:
        round_id = int(observation.get("current_round", 0))

        buyer_action = buyer_agent.respond(
            conversation_history=observation["conversation_history"],
            current_state=observation,
        )
        buyer_exec_text = _normalize_passthrough_text(
            buyer_action if isinstance(buyer_action, str) else None,
        )

        seller_history = observation["conversation_history"].copy()
        if buyer_exec_text is not None:
            seller_history.append(
                {
                    "role": "buyer",
                    "content": buyer_exec_text,
                    "round": round_id,
                }
            )

        seller_actor_id = getattr(seller_agent, "name", "seller")
        coordination_plan = ocl_coordinator.plan_turn(
            round_id=round_id,
            buyer_text=buyer_exec_text,
            seller_actor_id=seller_actor_id,
            max_rounds=control_context.get("max_rounds"),
        )
        _trace_add_event(
            trace,
            ocl_coordinator.build_audit_event(coordination_plan),
            audit_policy,
        )

        coordination_plan_state = {
            "decision_role": coordination_plan.decision_role.value,
            "execution_role": coordination_plan.execution_role.value,
            "escalation_role": coordination_plan.escalation_role.value,
            "reason": coordination_plan.reason,
        }
        seller_state = dict(observation)
        seller_state.update(
            {
                key: value
                for key, value in control_context.items()
                if value is not None
            }
        )
        # Surface role-planning output to seller generation so role decomposition
        # can influence policy behavior, not only audit metadata. 中文：将角色规划
        # 输出传递给 seller 生成流程，使其影响策略行为而非仅做审计记录。
        # 中文：把角色规划结果传给 seller 生成阶段，使 role decomposition
        # 中文：真正影响策略行为，而不是只停留在审计元数据。
        seller_state["coordination_plan"] = coordination_plan_state
        seller_action = seller_agent.respond(
            conversation_history=seller_history,
            current_state=seller_state,
        )
        control_state = dict(seller_state)
        seller_exec_text = _apply_control_to_text(
            trace=trace,
            controller=ocl_controller,
            escalation_manager=ocl_escalation,
            round_id=round_id,
            actor_id=coordination_plan.control_actor_id,
            actor_role=coordination_plan.control_actor_role,
            text=seller_action,
            history=seller_history,
            state=control_state,
            audit_policy=audit_policy,
        ).executable_text

        observation, _reward, terminated, truncated, final_info = adapter.step(
            buyer_action=buyer_exec_text,
            seller_action=seller_exec_text,
        )
        done = terminated or truncated

    trace.final_status = str(final_info.get("status"))
    trace.final_metrics = {
        "round": final_info.get("round"),
        "status": final_info.get("status"),
        "termination_reason": final_info.get("termination_reason"),
        "agreed_price": final_info.get("agreed_price"),
        "buyer_price": final_info.get("buyer_price"),
        "seller_price": final_info.get("seller_price"),
        "buyer_reward": final_info.get("buyer_reward"),
        "seller_reward": final_info.get("seller_reward"),
        "global_score": final_info.get("global_score"),
        "buyer_score": final_info.get("buyer_score"),
        "seller_score": final_info.get("seller_score"),
    }
    _trace_add_event(
        trace,
        AuditEvent(
            event_type=AuditEventType.EPISODE_FINISHED,
            summary=f"Episode finished with status={trace.final_status}",
            metadata=trace.final_metrics,
        ),
        audit_policy,
    )

    with suppress(Exception):
        adapter.env.close()

    return trace, final_info
