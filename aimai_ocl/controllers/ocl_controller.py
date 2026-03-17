"""Minimal Organizational Control Layer (OCL) controller skeleton.

This file defines the first concrete implementation of the proposal's central
mapping:

    g_Pi: (m_1:t, a_raw_1:t) -> a_exec_1:t

The current implementation is deliberately conservative. It wires together:

- role validation
- risk evaluation
- executable action shaping
- audit event generation

without yet introducing advanced negotiation strategy, escalation planning, or
credit assignment.


中文翻译：Minimal Organizational Control Layer (OCL) controller skeleton。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aimai_ocl.controllers.constraint_engine import ConstraintEngine
from aimai_ocl.controllers.risk_gate import RiskGate
from aimai_ocl.controllers.role_policy import RolePolicy
from aimai_ocl.schemas.actions import ExecutableAction, RawAction
from aimai_ocl.schemas.audit import AuditEvent, AuditEventType
from aimai_ocl.schemas.constraints import ConstraintCheck


@dataclass(slots=True)
class OCLControlResult:
    """Structured result returned by the minimal OCL controller.

    Inputs:
        raw_action: The untrusted action proposal received by the controller.
        executable_action: The resulting action after role and risk processing.
        checks: All constraint checks generated during control evaluation.
        audit_events: Ordered audit events produced while processing the action.
        metadata: Optional controller-specific details for future extensions.

    Outputs:
        A single object that callers can use to update traces, benchmark
        summaries, or downstream environment actions.
    

    中文翻译：Structured result returned by the minimal OCL controller。"""

    raw_action: RawAction
    executable_action: ExecutableAction
    checks: list[ConstraintCheck] = field(default_factory=list)
    audit_events: list[AuditEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OCLController:
    """Minimal OCL controller composed of role policy, hard constraints, and risk gate.

    Inputs:
        role_policy: Policy object deciding whether a role may emit an intent.
        constraint_engine: Deterministic rule engine for hard checks such as
            budget/floor/format/privacy constraints.
        risk_gate: Gate object deciding whether an otherwise valid action should
            pass through, require confirmation, or be blocked.

    Outputs:
        A controller object exposing ``apply(...)`` which transforms one
        ``RawAction`` into an ``ExecutableAction`` plus checks and audit events.
    

    中文翻译：Minimal OCL controller composed of role policy, hard constraints, and risk gate。"""

    role_policy: RolePolicy = field(default_factory=RolePolicy)
    constraint_engine: ConstraintEngine = field(default_factory=ConstraintEngine)
    risk_gate: RiskGate = field(default_factory=RiskGate)

    def apply(
        self,
        raw_action: RawAction,
        *,
        round_id: int | None = None,
        history: list[dict[str, Any]] | None = None,
        state: dict[str, Any] | None = None,
    ) -> OCLControlResult:
        """Apply the minimal OCL control pipeline to one action proposal.

        Input:
            raw_action:
                Untrusted pre-control proposal from one actor.
            round_id:
                Optional round index used in emitted audit events.
            history:
                Optional dialogue history visible to controllers.
            state:
                Optional control state, including runtime bounds like
                ``buyer_max_price`` and ``seller_min_price``.

        Output:
            ``OCLControlResult`` with:
            - ``executable_action``: final decision payload sent downstream
            - ``checks``: ordered checks from role + hard rules + risk gate
            - ``audit_events``: raw/constraint/execution events for trace replay
            - ``metadata``: summarized flags (decision, violations, failed checks)

        Decision flow:
            1. Role permission check
            2. Hard-constraint checks (format/privacy/price bounds)
            3. Risk check and executable-action shaping
        

        中文翻译：应用 the minimal OCL control pipeline to one action proposal。"""
        role_check = self.role_policy.evaluate(raw_action)
        hard_checks = self.constraint_engine.evaluate(
            raw_action,
            state=state,
            history=history,
        )
        risk_check = self.risk_gate.evaluate(
            raw_action,
            state=state,
            history=history,
        )
        checks = [role_check, *hard_checks, risk_check]

        executable_action = self.risk_gate.apply(
            raw_action,
            checks,
            state=state,
            history=history,
        )

        audit_events = [
            AuditEvent(
                event_type=AuditEventType.RAW_ACTION_RECEIVED,
                round_id=round_id,
                actor_id=raw_action.actor_id,
                summary=f"Controller received raw action from {raw_action.actor_id}",
                raw_action=raw_action,
                metadata={
                    "history_length": len(history or []),
                    "state_keys": sorted((state or {}).keys()),
                },
            ),
            AuditEvent(
                event_type=AuditEventType.CONSTRAINT_EVALUATED,
                round_id=round_id,
                actor_id=raw_action.actor_id,
                summary=f"Controller evaluated {len(checks)} checks for {raw_action.actor_id}",
                raw_action=raw_action,
                executable_action=executable_action,
                constraint_checks=checks,
            ),
            AuditEvent(
                event_type=AuditEventType.ACTION_EXECUTED,
                round_id=round_id,
                actor_id=raw_action.actor_id,
                summary=f"Controller produced executable action for {raw_action.actor_id}",
                raw_action=raw_action,
                executable_action=executable_action,
                constraint_checks=checks,
            ),
        ]

        return OCLControlResult(
            raw_action=raw_action,
            executable_action=executable_action,
            checks=checks,
            audit_events=audit_events,
            metadata={
                "approved": executable_action.approved,
                "decision": executable_action.decision.value,
                "requires_confirmation": executable_action.requires_confirmation,
                "requires_escalation": executable_action.requires_escalation,
                "check_count": len(checks),
                "failed_constraints": [
                    check.constraint_id
                    for check in checks
                    if not check.passed
                ],
                "violations": [
                    check.violation_type.value
                    for check in checks
                    if check.violation_type is not None
                ],
            },
        )
