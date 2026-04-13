"""Core data types for the OCL pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ActionRole(str, Enum):
    BUYER = "buyer"
    SELLER = "seller"
    PLATFORM = "platform"
    EXPERT = "expert"
    USER = "user"
    UNKNOWN = "unknown"


class ActionIntent(str, Enum):
    NEGOTIATE_PRICE = "negotiate_price"
    ACCEPT_DEAL = "accept_deal"
    REJECT_DEAL = "reject_deal"
    REQUEST_INFO = "request_info"
    EXPLAIN_POLICY = "explain_policy"
    ESCALATE = "escalate"
    TOOL_CALL = "tool_call"
    OTHER = "other"


class ControlDecision(str, Enum):
    APPROVE = "approve"
    REWRITE = "rewrite"
    BLOCK = "block"
    ESCALATE = "escalate"


class ConstraintSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ViolationType(str, Enum):
    ROLE_PERMISSION = "role_permission"
    HIGH_RISK_ACTION = "high_risk_action"
    BUDGET_EXCEEDED = "budget_exceeded"
    SELLER_FLOOR_BREACH = "seller_floor_breach"
    FORMAT_INVALID = "format_invalid"
    POLICY_PRIVACY = "policy_privacy"
    UNKNOWN = "unknown"


class AuditEventType(str, Enum):
    EPISODE_STARTED = "episode_started"
    COORDINATION_PLANNED = "coordination_planned"
    RAW_ACTION_RECEIVED = "raw_action_received"
    CONSTRAINT_EVALUATED = "constraint_evaluated"
    ACTION_EXECUTED = "action_executed"
    ESCALATION_TRIGGERED = "escalation_triggered"
    REPLAN_APPLIED = "replan_applied"
    EPISODE_FINISHED = "episode_finished"


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RawAction:
    """Proposed action before OCL validation."""

    actor_id: str
    actor_role: ActionRole = ActionRole.UNKNOWN
    utterance: str = ""
    intent: ActionIntent = ActionIntent.OTHER
    proposed_price: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutableAction:
    """Action approved or shaped by the OCL control layer."""

    actor_id: str
    actor_role: ActionRole = ActionRole.UNKNOWN
    approved: bool = True
    decision: ControlDecision = ControlDecision.APPROVE
    final_text: str = ""
    intent: ActionIntent = ActionIntent.OTHER
    final_price: float | None = None
    blocked_reason: str | None = None
    requires_confirmation: bool = False
    requires_escalation: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Constraint check
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ConstraintCheck:
    """Result of evaluating one constraint against an action."""

    constraint_id: str
    passed: bool
    severity: ConstraintSeverity = ConstraintSeverity.INFO
    reason: str = ""
    violation_type: ViolationType | None = None
    checked_fields: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Audit trace
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AuditEvent:
    """One event in the control-layer audit trail."""

    event_type: AuditEventType
    round_id: int | None = None
    actor_id: str | None = None
    summary: str = ""
    raw_action: RawAction | None = None
    executable_action: ExecutableAction | None = None
    constraint_checks: list[ConstraintCheck] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodeTrace:
    """Top-level audit container for one episode."""

    episode_id: str
    env_id: str
    scenario: dict[str, Any] = field(default_factory=dict)
    events: list[AuditEvent] = field(default_factory=list)
    final_status: str | None = None
    final_metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_event(self, event: AuditEvent) -> None:
        self.events.append(event)
