"""Escalation and one-shot replan policy for seller-side OCL execution.

This module implements step-7 behavior:

- trigger escalation when actions are blocked/high-risk
- attempt one deterministic replan for recoverable price violations
- persist escalation strategy decisions as audit events for replay


中文翻译：Escalation and one-shot replan policy for seller-side OCL execution。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aimai_ocl.schemas.actions import RawAction
from aimai_ocl.schemas.audit import AuditEvent, AuditEventType
from aimai_ocl.schemas.constraints import ViolationType


@dataclass(slots=True)
class EscalationOutcome:
    """Resolved execution outcome after escalation/replan policy.

    Input:
        final_text:
            Text to execute in environment, or ``None`` when no execution.
        replan_text:
            Optional replanned text that should be re-validated by controller.
        requires_human_handoff:
            Whether policy recommends routing to human/platform escalation.
        strategy:
            Stable strategy label for trace and offline analysis.
        audit_events:
            One or more strategy events to append to episode trace.

    Output:
        Container consumed by runner to continue execution flow.
    

    中文翻译：Resolved execution outcome after escalation/replan policy。"""

    final_text: str | None
    replan_text: str | None
    requires_human_handoff: bool
    strategy: str
    audit_events: list[AuditEvent] = field(default_factory=list)


@dataclass(slots=True)
class EscalationManager:
    """Deterministic escalation manager with one-shot price replan.

    Inputs:
        enable_replan:
            Whether blocked actions may be repaired by deterministic rewrite.
        replan_price_template:
            Template used to produce replanned seller text.

    Outputs:
        ``resolve(...)`` returns ``EscalationOutcome`` for one control result.
    

    中文翻译：Deterministic escalation manager with one-shot price replan。"""

    enable_replan: bool = True
    replan_price_template: str = "I can revise to ${price:.2f}."

    def resolve(
        self,
        *,
        round_id: int,
        actor_id: str,
        raw_action: RawAction,
        approved: bool,
        requires_confirmation: bool,
        requires_escalation: bool,
        violations: list[str],
        state: dict[str, Any],
        allow_replan: bool = True,
    ) -> EscalationOutcome:
        """Resolve one action into execute/replan/handoff path.

        Input:
            round_id:
                Current negotiation round index.
            actor_id:
                Seller actor id for audit attribution.
            raw_action:
                Original or replanned raw action under evaluation.
            approved:
                Control approval flag from executable action.
            requires_confirmation:
                High-risk confirmation flag from executable action.
            requires_escalation:
                Explicit escalation flag from executable action.
            violations:
                Violation type values extracted from control metadata.
            state:
                Control state used for deterministic replan pricing.
            allow_replan:
                Whether this call may emit a replan candidate. Runner should set
                this to ``False`` on second-pass validation.

        Output:
            ``EscalationOutcome`` with one strategy:
            - ``direct_execute``
            - ``replan_and_retry``
            - ``human_handoff``
        

        中文翻译：解析 one action into execute/replan/handoff path。"""
        if approved and (not requires_confirmation) and (not requires_escalation):
            final_text = raw_action.utterance.strip() or None
            return EscalationOutcome(
                final_text=final_text,
                replan_text=None,
                requires_human_handoff=False,
                strategy="direct_execute",
            )

        events: list[AuditEvent] = []
        events.append(
            AuditEvent(
                event_type=AuditEventType.ESCALATION_TRIGGERED,
                round_id=round_id,
                actor_id=actor_id,
                summary=(
                    "Escalation policy triggered for seller action: "
                    f"approved={approved}, "
                    f"requires_confirmation={requires_confirmation}, "
                    f"requires_escalation={requires_escalation}"
                ),
                raw_action=raw_action,
                metadata={
                    "approved": approved,
                    "requires_confirmation": requires_confirmation,
                    "requires_escalation": requires_escalation,
                    "violations": violations,
                },
            )
        )

        can_try_replan = (
            self.enable_replan
            and allow_replan
            and (not approved)
            and self._has_recoverable_price_violation(violations)
        )
        if can_try_replan:
            replanned_price = self._derive_replanned_price(
                proposed_price=raw_action.proposed_price,
                state=state,
            )
            if replanned_price is not None:
                replan_text = self.replan_price_template.format(price=replanned_price)
                events.append(
                    AuditEvent(
                        event_type=AuditEventType.REPLAN_APPLIED,
                        round_id=round_id,
                        actor_id=actor_id,
                        summary=(
                            "Deterministic replan applied for blocked seller action."
                        ),
                        raw_action=raw_action,
                        metadata={
                            "strategy": "replan_and_retry",
                            "original_price": raw_action.proposed_price,
                            "replanned_price": replanned_price,
                            "replan_text": replan_text,
                            "violations": violations,
                        },
                    )
                )
                return EscalationOutcome(
                    final_text=None,
                    replan_text=replan_text,
                    requires_human_handoff=False,
                    strategy="replan_and_retry",
                    audit_events=events,
                )

        events[-1].metadata["strategy"] = "human_handoff"
        return EscalationOutcome(
            final_text=None,
            replan_text=None,
            requires_human_handoff=True,
            strategy="human_handoff",
            audit_events=events,
        )

    def _has_recoverable_price_violation(self, violations: list[str]) -> bool:
        """Check whether violations are compatible with deterministic replan.

        Input:
            violations:
                Violation type values from control metadata.

        Output:
            ``True`` when any recoverable price/format violation is present.
        

        中文翻译：Check whether violations are compatible with deterministic replan。"""
        recoverable = {
            ViolationType.BUDGET_EXCEEDED.value,
            ViolationType.SELLER_FLOOR_BREACH.value,
            ViolationType.FORMAT_INVALID.value,
        }
        return any(v in recoverable for v in violations)

    def _derive_replanned_price(
        self,
        *,
        proposed_price: float | None,
        state: dict[str, Any],
    ) -> float | None:
        """Derive one feasible price candidate from runtime bounds.

        Input:
            proposed_price:
                Parsed price from raw action, if available.
            state:
                Runtime state containing ``buyer_max_price`` and
                ``seller_min_price`` where available.

        Output:
            Feasible replanned price, or ``None`` when bounds are infeasible
            and deterministic replan is impossible.
        

        中文翻译：Derive one feasible price candidate from runtime bounds。"""
        buyer_max = _coerce_float(state.get("buyer_max_price"))
        seller_min = _coerce_float(state.get("seller_min_price"))

        if (buyer_max is not None) and (seller_min is not None) and seller_min > buyer_max:
            return None

        if proposed_price is None:
            if (buyer_max is not None) and (seller_min is not None):
                return (buyer_max + seller_min) / 2.0
            if buyer_max is not None:
                return buyer_max
            if seller_min is not None:
                return seller_min
            return None

        replanned = proposed_price
        if buyer_max is not None:
            replanned = min(replanned, buyer_max)
        if seller_min is not None:
            replanned = max(replanned, seller_min)
        return replanned if replanned > 0 else None


def _coerce_float(value: Any) -> float | None:
    """Best-effort conversion to float for state numeric fields.

    Input:
        value:
            Any runtime value from control state.

    Output:
        Parsed float or ``None`` when conversion fails.
    

    中文翻译：Best-effort conversion to float for state numeric fields。"""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
