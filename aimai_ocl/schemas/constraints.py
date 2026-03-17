"""Constraint schemas for the minimal OCL interface.

中文翻译：Constraint schemas for the minimal OCL interface。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ConstraintSeverity(str, Enum):
    """Severity level associated with a failed or notable constraint check.

中文翻译：Severity level associated with a failed or notable constraint check。"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ViolationType(str, Enum):
    """Canonical violation taxonomy for OCL policy and safety evaluation.

中文翻译：Canonical violation taxonomy for OCL policy and safety evaluation。"""

    ROLE_PERMISSION = "role_permission"
    HIGH_RISK_ACTION = "high_risk_action"
    BUDGET_EXCEEDED = "budget_exceeded"
    SELLER_FLOOR_BREACH = "seller_floor_breach"
    FORMAT_INVALID = "format_invalid"
    POLICY_PRIVACY = "policy_privacy"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class ConstraintCheck:
    """Result of evaluating a single constraint against a raw or exec action.

    Inputs:
        constraint_id: Stable identifier for the evaluated constraint.
        passed: Whether the action satisfied the constraint.
        severity: Impact level if the constraint did not pass or needs
            attention.
        reason: Human-readable explanation of the evaluation result.
        violation_type: Optional canonical violation taxonomy label.
        checked_fields: Structured list of fields or dimensions that were
            examined.
        metadata: Additional structured details preserved for audit and
            downstream analysis.

    Outputs:
        A normalized record that can be attached to audit events, traces, and
        benchmarking metrics such as violation rate.
    

    中文翻译：Result of evaluating a single constraint against a raw or exec action。"""

    constraint_id: str
    passed: bool
    severity: ConstraintSeverity = ConstraintSeverity.INFO
    reason: str = ""
    violation_type: ViolationType | None = None
    checked_fields: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
