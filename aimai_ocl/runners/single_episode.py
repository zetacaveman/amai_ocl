"""Minimal baseline runner for single-product AgenticPay negotiation.

Design note:
Buyer-side text is treated as external user-simulator input and is passed
through directly. This runner only records structured action events for the
seller side so evaluation stays seller-centric.


中文翻译：Minimal baseline runner for single-product AgenticPay negotiation。"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from aimai_ocl.adapters import AgenticPayEnvAdapter
from aimai_ocl.adapters.agenticpay_actions import (
    executable_action_from_raw,
    raw_action_from_text,
)
from aimai_ocl.controllers.audit_policy import AuditPolicy
from aimai_ocl.runners.scenario_validation import enforce_single_product_scenario
from aimai_ocl.schemas.actions import ActionRole, ControlDecision, ExecutableAction
from aimai_ocl.schemas.audit import AuditEvent, AuditEventType, EpisodeTrace


DEFAULT_SINGLE_REPAIR_TEMPLATE = "I can revise to ${price:.2f}."


def _normalize_passthrough_text(text: str | None) -> str | None:
    """Normalize pass-through text before sending it to ``env.step``.

    Args:
        text: Raw text returned by an external actor (for example buyer agent),
            or ``None`` when no action is emitted.

    Returns:
        Stripped non-empty text, or ``None`` if input is ``None``/empty.
    

    中文翻译：规范化 pass-through text before sending it to ``env.step``。"""
    if text is None:
        return None
    normalized = text.strip()
    return normalized or None


def _append_action_events(
    trace: EpisodeTrace,
    round_id: int,
    actor_id: str,
    actor_role: ActionRole,
    text: str | None,
    audit_policy: AuditPolicy | None,
    repair_price_state: dict[str, Any] | None = None,
) -> str | None:
    """Record raw and executable action events for one actor message.

    Args:
        trace: Episode trace being mutated in place.
        round_id: Logical round id for the action.
        actor_id: Stable identifier for the acting entity.
        actor_role: Functional role of the actor.
        text: The raw text produced by the actor, or ``None`` if no action was
            emitted.

    Returns:
        The executable text that should be sent to the environment, or ``None``
        if no input text was provided.
    

    中文翻译：Record raw and executable action events for one actor message。"""
    if text is None:
        return None

    raw_action = raw_action_from_text(
        actor_id=actor_id,
        actor_role=actor_role,
        text=text,
    )
    exec_action = executable_action_from_raw(raw_action)
    if repair_price_state is not None and actor_role == ActionRole.SELLER:
        exec_action = _repair_executable_action(
            raw_action=raw_action,
            exec_action=exec_action,
            repair_price_state=repair_price_state,
        )

    _trace_add_event(
        trace,
        AuditEvent(
            event_type=AuditEventType.RAW_ACTION_RECEIVED,
            round_id=round_id,
            actor_id=actor_id,
            summary=f"Received raw action from {actor_id}",
            raw_action=raw_action,
        ),
        audit_policy,
    )
    _trace_add_event(
        trace,
        AuditEvent(
            event_type=AuditEventType.ACTION_EXECUTED,
            round_id=round_id,
            actor_id=actor_id,
            summary=f"Executed action for {actor_id}",
            raw_action=raw_action,
            executable_action=exec_action,
        ),
        audit_policy,
    )
    return exec_action.final_text


def _repair_executable_action(
    *,
    raw_action: Any,
    exec_action: ExecutableAction,
    repair_price_state: dict[str, Any],
) -> ExecutableAction:
    """Apply seller-side floor repair without invoking OCL control."""
    seller_min_price = _coerce_float(repair_price_state.get("seller_min_price"))
    template = str(
        repair_price_state.get("repair_template") or DEFAULT_SINGLE_REPAIR_TEMPLATE
    )
    raw_price = raw_action.proposed_price
    repaired_price = _derive_repaired_price(
        proposed_price=raw_price,
        seller_min_price=seller_min_price,
    )
    raw_price_violation = _is_price_violation(
        price=raw_price,
        seller_min_price=seller_min_price,
    )
    post_repair_violation = _is_price_violation(
        price=repaired_price,
        seller_min_price=seller_min_price,
    )
    repair_applied = (
        raw_price is not None
        and repaired_price is not None
        and repaired_price != raw_price
        and raw_price_violation
        and not post_repair_violation
    )
    metadata = {
        **exec_action.metadata,
        "repair_mode": "single_repair",
        "repair_scope": "seller_side_private_floor",
        "repair_enabled": True,
        "repair_applied": repair_applied,
        "raw_price": raw_price,
        "repaired_price": repaired_price,
        "seller_min_price": seller_min_price,
        "raw_price_violation": raw_price_violation,
        "post_repair_violation": post_repair_violation,
    }
    if not repair_applied or repaired_price is None:
        return ExecutableAction(
            actor_id=exec_action.actor_id,
            actor_role=exec_action.actor_role,
            approved=exec_action.approved,
            decision=exec_action.decision,
            final_text=exec_action.final_text,
            intent=exec_action.intent,
            final_price=exec_action.final_price,
            blocked_reason=exec_action.blocked_reason,
            requires_confirmation=exec_action.requires_confirmation,
            requires_escalation=exec_action.requires_escalation,
            metadata=metadata,
        )

    repaired_text = template.format(price=repaired_price)
    metadata["repair_text"] = repaired_text
    return ExecutableAction(
        actor_id=exec_action.actor_id,
        actor_role=exec_action.actor_role,
        approved=True,
        decision=ControlDecision.REWRITE,
        final_text=repaired_text,
        intent=exec_action.intent,
        final_price=repaired_price,
        blocked_reason=None,
        requires_confirmation=False,
        requires_escalation=False,
        metadata=metadata,
    )


def _derive_repaired_price(
    *,
    proposed_price: float | None,
    seller_min_price: float | None,
) -> float | None:
    """Raise one proposed seller price to the seller floor when possible."""
    if proposed_price is None:
        return None

    repaired = proposed_price
    if seller_min_price is not None:
        repaired = max(repaired, seller_min_price)
    return repaired if repaired > 0 else None


def _is_price_violation(
    *,
    price: float | None,
    seller_min_price: float | None,
) -> bool:
    """Return whether a numeric price violates the seller-side floor."""
    if price is None:
        return False
    if seller_min_price is not None and price < seller_min_price:
        return True
    return False


def _coerce_float(value: Any) -> float | None:
    """Best-effort conversion for bound values coming from env kwargs."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _trace_add_event(
    trace: EpisodeTrace,
    event: AuditEvent,
    audit_policy: AuditPolicy | None,
) -> None:
    """Append one event when allowed by current audit policy."""
    if audit_policy is None or audit_policy.should_record(event.event_type):
        trace.add_event(event)


def run_single_negotiation_episode(
    env_id: str,
    buyer_agent: Any,
    seller_agent: Any,
    reset_kwargs: dict[str, Any],
    *,
    env_kwargs: dict[str, Any] | None = None,
    trace_metadata: dict[str, Any] | None = None,
    audit_policy: AuditPolicy | None = None,
    repair_seller_price: bool = False,
    repair_price_template: str = DEFAULT_SINGLE_REPAIR_TEMPLATE,
) -> tuple[EpisodeTrace, dict[str, Any]]:
    """Run one baseline episode with seller-side action tracing.

    Args:
        env_id: AgenticPay environment id to execute.
        buyer_agent: Any object compatible with the AgenticPay buyer interface.
            Buyer output is treated as pass-through user-simulator text.
        seller_agent: Any object compatible with the AgenticPay seller
            interface (``SellerAgent`` from upstream AgenticPay).
        reset_kwargs: Keyword arguments forwarded to ``env.reset`` for scenario
            setup.
        env_kwargs: Optional keyword arguments forwarded to ``agenticpay.make``.
            This is typically where the buyer and seller agents are attached.
        trace_metadata: Optional experiment bookkeeping fields stored on the
            resulting episode trace.
        repair_seller_price: Whether to apply deterministic seller-side floor
            repair before executing seller actions in the environment. This is
            used by the ``single_repair`` fairness baseline and does not invoke
            OCL or buyer-private price bounds.
        repair_price_template: Text template for repaired seller offers.

    Returns:
        A tuple ``(trace, final_info)`` where ``trace`` is the collected
        ``EpisodeTrace`` and ``final_info`` is the final AgenticPay ``info``
        dictionary from the terminal step.
    

    中文翻译：运行 one baseline episode with seller-side action tracing。"""
    normalized_reset_kwargs = enforce_single_product_scenario(reset_kwargs)
    env_config = dict(env_kwargs or {})
    env_config["buyer_agent"] = buyer_agent
    env_config["seller_agent"] = seller_agent
    repair_price_state = (
        {
            "seller_min_price": env_config.get("seller_min_price"),
            "repair_template": repair_price_template,
        }
        if repair_seller_price
        else None
    )

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

    done = False
    final_info: dict[str, Any] = {}

    while not done:
        round_id = int(observation.get("current_round", 0))

        buyer_action = buyer_agent.respond(
            conversation_history=observation["conversation_history"],
            current_state=observation,
        )
        buyer_exec_text = _normalize_passthrough_text(buyer_action)

        seller_history = observation["conversation_history"].copy()
        if buyer_exec_text is not None:
            seller_history.append(
                {
                    "role": "buyer",
                    "content": buyer_exec_text,
                    "round": round_id,
                }
            )

        seller_action = seller_agent.respond(
            conversation_history=seller_history,
            current_state=observation,
        )
        seller_exec_text = _append_action_events(
            trace=trace,
            round_id=round_id,
            actor_id=getattr(seller_agent, "name", "seller"),
            actor_role=ActionRole.SELLER,
            text=seller_action,
            audit_policy=audit_policy,
            repair_price_state=repair_price_state,
        )

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
