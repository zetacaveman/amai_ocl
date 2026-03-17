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
from aimai_ocl.schemas.actions import ActionRole
from aimai_ocl.schemas.audit import AuditEvent, AuditEventType, EpisodeTrace


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

    Returns:
        A tuple ``(trace, final_info)`` where ``trace`` is the collected
        ``EpisodeTrace`` and ``final_info`` is the final AgenticPay ``info``
        dictionary from the terminal step.
    

    中文翻译：运行 one baseline episode with seller-side action tracing。"""
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
