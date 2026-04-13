"""OCL control pipeline: role check, constraints, risk scoring, escalation.

The entire seller-side control layer in one file. Four checks run sequentially:
1. Role permission (dict lookup)
2. Price format validation
3. Privacy keyword scan
4. Price bounds + risk score

Then escalation handles blocked actions (clamp price or hand off).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from aimai_ocl.schemas import (
    ActionIntent,
    ActionRole,
    AuditEvent,
    AuditEventType,
    ConstraintCheck,
    ConstraintSeverity,
    ControlDecision,
    ExecutableAction,
    RawAction,
    ViolationType,
)

# ---------------------------------------------------------------------------
# Role permission table
# ---------------------------------------------------------------------------

ROLE_PERMISSIONS: dict[ActionRole, set[ActionIntent]] = {
    ActionRole.BUYER: {
        ActionIntent.NEGOTIATE_PRICE, ActionIntent.ACCEPT_DEAL,
        ActionIntent.REJECT_DEAL, ActionIntent.REQUEST_INFO, ActionIntent.OTHER,
    },
    ActionRole.SELLER: {
        ActionIntent.NEGOTIATE_PRICE, ActionIntent.ACCEPT_DEAL,
        ActionIntent.REJECT_DEAL, ActionIntent.REQUEST_INFO,
        ActionIntent.EXPLAIN_POLICY, ActionIntent.OTHER,
    },
    ActionRole.PLATFORM: {
        ActionIntent.EXPLAIN_POLICY, ActionIntent.ESCALATE,
        ActionIntent.TOOL_CALL, ActionIntent.OTHER,
    },
    ActionRole.EXPERT: {
        ActionIntent.REQUEST_INFO, ActionIntent.EXPLAIN_POLICY, ActionIntent.OTHER,
    },
    ActionRole.USER: {ActionIntent.REQUEST_INFO, ActionIntent.OTHER},
    ActionRole.UNKNOWN: {ActionIntent.OTHER},
}

_SENSITIVE_KEYWORDS = (
    "password", "passcode", "ssn", "social security",
    "credit card", "cvv", "bank account", "routing number",
)

_HIGH_RISK_INTENTS = (ActionIntent.TOOL_CALL, ActionIntent.ESCALATE)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_role(raw: RawAction) -> ConstraintCheck:
    """Check if the actor role is allowed to emit this intent."""
    allowed = ROLE_PERMISSIONS.get(raw.actor_role, set())
    if raw.intent in allowed:
        return ConstraintCheck(
            constraint_id="role_policy", passed=True,
            reason=f"Role `{raw.actor_role.value}` may emit `{raw.intent.value}`.",
        )
    return ConstraintCheck(
        constraint_id="role_policy", passed=False,
        severity=ConstraintSeverity.ERROR,
        violation_type=ViolationType.ROLE_PERMISSION,
        reason=f"Role `{raw.actor_role.value}` may not emit `{raw.intent.value}`.",
    )


def check_price_format(raw: RawAction) -> ConstraintCheck:
    """Check intent/price consistency."""
    needs_price = raw.intent in (ActionIntent.NEGOTIATE_PRICE, ActionIntent.ACCEPT_DEAL)
    if needs_price and raw.proposed_price is None:
        return ConstraintCheck(
            constraint_id="price_format", passed=False,
            severity=ConstraintSeverity.ERROR,
            violation_type=ViolationType.FORMAT_INVALID,
            reason=f"Intent `{raw.intent.value}` requires an explicit price.",
        )
    if raw.proposed_price is not None and raw.proposed_price <= 0:
        return ConstraintCheck(
            constraint_id="price_format", passed=False,
            severity=ConstraintSeverity.ERROR,
            violation_type=ViolationType.FORMAT_INVALID,
            reason="Proposed price must be positive.",
        )
    return ConstraintCheck(constraint_id="price_format", passed=True)


def check_privacy(raw: RawAction) -> ConstraintCheck:
    """Scan for PII-like content."""
    text = raw.utterance.lower()
    matched = [kw for kw in _SENSITIVE_KEYWORDS if kw in text]
    if matched:
        return ConstraintCheck(
            constraint_id="privacy_policy", passed=False,
            severity=ConstraintSeverity.CRITICAL,
            violation_type=ViolationType.POLICY_PRIVACY,
            reason="Utterance contains privacy-sensitive keywords.",
            metadata={"matched_keywords": matched},
        )
    has_long_digits = re.search(r"\b(?:\d[ -]*?){13,19}\b", raw.utterance) is not None
    mentions_payment = any(t in text for t in ("card", "account", "bank", "payment"))
    if has_long_digits and mentions_payment:
        return ConstraintCheck(
            constraint_id="privacy_policy", passed=False,
            severity=ConstraintSeverity.CRITICAL,
            violation_type=ViolationType.POLICY_PRIVACY,
            reason="Potential sensitive payment identifier detected.",
        )
    return ConstraintCheck(constraint_id="privacy_policy", passed=True)


def check_price_bounds(
    raw: RawAction, *, buyer_max: float | None, seller_min: float | None,
) -> list[ConstraintCheck]:
    """Check proposed price against buyer cap and seller floor."""
    checks: list[ConstraintCheck] = []
    price = raw.proposed_price

    if raw.actor_role != ActionRole.SELLER or price is None:
        checks.append(ConstraintCheck(constraint_id="budget_cap", passed=True))
        checks.append(ConstraintCheck(constraint_id="seller_floor", passed=True))
        return checks

    if buyer_max is not None and price > buyer_max:
        checks.append(ConstraintCheck(
            constraint_id="budget_cap", passed=False,
            severity=ConstraintSeverity.ERROR,
            violation_type=ViolationType.BUDGET_EXCEEDED,
            reason=f"Price {price:.2f} exceeds buyer max {buyer_max:.2f}.",
        ))
    else:
        checks.append(ConstraintCheck(constraint_id="budget_cap", passed=True))

    if seller_min is not None and price < seller_min:
        checks.append(ConstraintCheck(
            constraint_id="seller_floor", passed=False,
            severity=ConstraintSeverity.ERROR,
            violation_type=ViolationType.SELLER_FLOOR_BREACH,
            reason=f"Price {price:.2f} below seller floor {seller_min:.2f}.",
        ))
    else:
        checks.append(ConstraintCheck(constraint_id="seller_floor", passed=True))

    return checks


def compute_risk_score(raw: RawAction, *, state: dict[str, Any]) -> float:
    """Compute a deterministic risk score in [0, 1]."""
    score = 0.05
    if raw.intent in _HIGH_RISK_INTENTS:
        score += 0.35
    text = raw.utterance.lower()
    if any(t in text for t in ("password", "card", "bank", "account", "ssn", "cvv")):
        score += 0.45
    if "!" in raw.utterance:
        score += 0.05
    price = raw.proposed_price
    buyer_max = _float(state.get("buyer_max_price"))
    seller_min = _float(state.get("seller_min_price"))
    if price is not None:
        if buyer_max is not None and price > buyer_max:
            score += min(0.30, 0.10 + (price - buyer_max) / max(1.0, buyer_max))
        if seller_min is not None and price < seller_min:
            score += min(0.25, 0.10 + (seller_min - price) / max(1.0, seller_min))
    return min(1.0, max(0.0, score))


# ---------------------------------------------------------------------------
# Control pipeline
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ControlResult:
    """Output of the control pipeline for one action."""
    raw_action: RawAction
    executable: ExecutableAction
    checks: list[ConstraintCheck] = field(default_factory=list)
    audit_events: list[AuditEvent] = field(default_factory=list)


@dataclass(slots=True)
class ControlConfig:
    """Tunable control parameters."""
    risk_rewrite_threshold: float = 0.45
    risk_block_threshold: float = 0.75
    high_risk_intents: tuple[ActionIntent, ...] = _HIGH_RISK_INTENTS
    confirm_high_risk: bool = True


def apply_control(
    raw: RawAction,
    *,
    state: dict[str, Any] | None = None,
    round_id: int | None = None,
    config: ControlConfig | None = None,
) -> ControlResult:
    """Run the full control pipeline on one raw action.

    Returns ControlResult with executable action, checks, and audit events.
    """
    cfg = config or ControlConfig()
    st = state or {}

    # 1-4: run all checks
    checks = [
        check_role(raw),
        check_price_format(raw),
        check_privacy(raw),
        *check_price_bounds(
            raw,
            buyer_max=_float(st.get("buyer_max_price")),
            seller_min=_float(st.get("seller_min_price")),
        ),
    ]
    risk = compute_risk_score(raw, state=st)
    checks.append(ConstraintCheck(
        constraint_id="risk_score", passed=risk < cfg.risk_block_threshold,
        severity=(
            ConstraintSeverity.ERROR if risk >= cfg.risk_block_threshold
            else ConstraintSeverity.WARNING if risk >= cfg.risk_rewrite_threshold
            else ConstraintSeverity.INFO
        ),
        reason=f"Risk score={risk:.3f}",
        violation_type=ViolationType.HIGH_RISK_ACTION if risk >= cfg.risk_rewrite_threshold else None,
        metadata={"risk_score": risk},
    ))

    # Decide: block / rewrite / approve
    has_hard_fail = any(
        not c.passed and c.severity in (ConstraintSeverity.ERROR, ConstraintSeverity.CRITICAL)
        for c in checks
    )
    if has_hard_fail:
        executable = ExecutableAction(
            actor_id=raw.actor_id, actor_role=raw.actor_role,
            approved=False, decision=ControlDecision.BLOCK, final_text="",
            intent=raw.intent, blocked_reason="Blocked by control checks.",
            requires_escalation=True,
        )
    elif risk >= cfg.risk_rewrite_threshold or (
        cfg.confirm_high_risk and raw.intent in cfg.high_risk_intents
    ):
        executable = ExecutableAction(
            actor_id=raw.actor_id, actor_role=raw.actor_role,
            approved=True, decision=ControlDecision.REWRITE,
            final_text=raw.utterance, intent=raw.intent,
            final_price=raw.proposed_price,
            requires_confirmation=True,
            requires_escalation=raw.intent == ActionIntent.ESCALATE,
        )
    else:
        decision = ControlDecision.ESCALATE if raw.intent == ActionIntent.ESCALATE else ControlDecision.APPROVE
        executable = ExecutableAction(
            actor_id=raw.actor_id, actor_role=raw.actor_role,
            approved=True, decision=decision,
            final_text=raw.utterance, intent=raw.intent,
            final_price=raw.proposed_price,
            requires_escalation=raw.intent == ActionIntent.ESCALATE,
        )

    # Audit events (simplified: 2 events instead of 3)
    audit_events = [
        AuditEvent(
            event_type=AuditEventType.CONSTRAINT_EVALUATED,
            round_id=round_id, actor_id=raw.actor_id,
            summary=f"Evaluated {len(checks)} checks for {raw.actor_id}",
            raw_action=raw, executable_action=executable,
            constraint_checks=checks,
        ),
        AuditEvent(
            event_type=AuditEventType.ACTION_EXECUTED,
            round_id=round_id, actor_id=raw.actor_id,
            summary=f"Control decision: {executable.decision.value}",
            raw_action=raw, executable_action=executable,
        ),
    ]

    return ControlResult(
        raw_action=raw, executable=executable,
        checks=checks, audit_events=audit_events,
    )


# ---------------------------------------------------------------------------
# Escalation (replan or hand off)
# ---------------------------------------------------------------------------


def resolve_escalation(
    *,
    raw: RawAction,
    executable: ExecutableAction,
    state: dict[str, Any],
    round_id: int | None = None,
    enable_replan: bool = True,
) -> tuple[str | None, list[AuditEvent]]:
    """Resolve a blocked/flagged action. Returns (final_text, audit_events).

    If replan succeeds, returns replanned text to be re-validated.
    If not, returns None (human handoff / skip).
    """
    if executable.approved and not executable.requires_confirmation and not executable.requires_escalation:
        return executable.final_text.strip() or None, []

    events: list[AuditEvent] = [
        AuditEvent(
            event_type=AuditEventType.ESCALATION_TRIGGERED,
            round_id=round_id, actor_id=raw.actor_id,
            summary=f"Escalation: approved={executable.approved}",
            raw_action=raw,
            metadata={
                "approved": executable.approved,
                "requires_confirmation": executable.requires_confirmation,
            },
        ),
    ]

    if not executable.approved and enable_replan:
        # Try deterministic price clamp
        violations = [c.violation_type.value for c in [] if c.violation_type]  # placeholder
        buyer_max = _float(state.get("buyer_max_price"))
        seller_min = _float(state.get("seller_min_price"))
        price = raw.proposed_price

        can_replan = (
            price is not None
            and buyer_max is not None
            and seller_min is not None
            and seller_min <= buyer_max
        )
        if can_replan:
            clamped = max(seller_min, min(buyer_max, price))
            replan_text = f"I can revise to ${clamped:.2f}."
            events.append(AuditEvent(
                event_type=AuditEventType.REPLAN_APPLIED,
                round_id=round_id, actor_id=raw.actor_id,
                summary="Deterministic price replan applied.",
                metadata={"original_price": price, "replanned_price": clamped},
            ))
            return replan_text, events

    # Human handoff
    return None, events


# ---------------------------------------------------------------------------
# Audit policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AuditPolicy:
    """Controls which event types get recorded in traces."""
    enabled_types: frozenset[AuditEventType] | None = None

    def should_record(self, event_type: AuditEventType) -> bool:
        if self.enabled_types is None:
            return True
        return event_type in self.enabled_types


# Full and minimal presets
AUDIT_FULL = AuditPolicy()
AUDIT_MINIMAL = AuditPolicy(enabled_types=frozenset({
    AuditEventType.EPISODE_STARTED,
    AuditEventType.EPISODE_FINISHED,
    AuditEventType.CONSTRAINT_EVALUATED,
    AuditEventType.ESCALATION_TRIGGERED,
}))
AUDIT_OFF = AuditPolicy(enabled_types=frozenset())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
