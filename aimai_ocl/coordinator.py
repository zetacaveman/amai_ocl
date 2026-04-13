"""Round-level role routing: who owns each negotiation turn.

Three modes:
- "default": keyword-based routing (seller / expert / platform)
- "state_machine": explicit phase transitions (discovery → negotiation → closing)
- "seller_only": always seller (ablation baseline)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aimai_ocl.schemas import ActionRole, AuditEvent, AuditEventType


_EXPERT_KEYWORDS = (
    "compare", "difference", "which", "recommend", "spec",
    "feature", "material", "size", "fit", "warranty", "policy",
)
_ESCALATION_KEYWORDS = (
    "complaint", "refund", "angry", "unacceptable",
    "manager", "escalate", "human",
)
_CLOSING_KEYWORDS = (
    "deal", "final", "accept", "confirm", "checkout", "payment", "order",
)


@dataclass(slots=True)
class CoordinationPlan:
    """One round-level role assignment."""
    round_id: int
    decision_role: ActionRole
    execution_role: ActionRole = ActionRole.SELLER
    escalation_role: ActionRole = ActionRole.PLATFORM
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Coordinator:
    """Deterministic role router with pluggable mode.

    Args:
        mode: "default", "state_machine", or "seller_only"
        escalation_round_buffer: rounds near deadline where platform takes over
    """

    mode: str = "default"
    orchestrator_id: str = "platform_orchestrator"
    escalation_round_buffer: int = 1

    def plan_turn(
        self,
        *,
        round_id: int,
        buyer_text: str | None,
        seller_actor_id: str,
        max_rounds: int | None = None,
    ) -> CoordinationPlan:
        """Decide who owns this round."""
        if self.mode == "seller_only":
            return self._plan_seller_only(round_id, seller_actor_id)
        if self.mode == "state_machine":
            return self._plan_state_machine(round_id, buyer_text, seller_actor_id, max_rounds)
        return self._plan_default(round_id, buyer_text, seller_actor_id, max_rounds)

    def build_audit_event(self, plan: CoordinationPlan) -> AuditEvent:
        return AuditEvent(
            event_type=AuditEventType.COORDINATION_PLANNED,
            round_id=plan.round_id,
            actor_id=self.orchestrator_id,
            summary=f"Coordinator: decision={plan.decision_role.value}",
            metadata={
                "decision_role": plan.decision_role.value,
                "reason": plan.reason,
                "mode": self.mode,
                **plan.metadata,
            },
        )

    # -- Default mode --

    def _plan_default(
        self, round_id: int, buyer_text: str | None,
        seller_actor_id: str, max_rounds: int | None,
    ) -> CoordinationPlan:
        text = (buyer_text or "").strip().lower()
        near_deadline = self._near_deadline(round_id, max_rounds)

        if near_deadline:
            role, reason = ActionRole.PLATFORM, "Near deadline; platform owns closeout."
        elif self._has_keywords(text, _EXPERT_KEYWORDS) or "?" in text:
            role, reason = ActionRole.EXPERT, "Buyer needs clarification; expert assists."
        else:
            role, reason = ActionRole.SELLER, "Default seller-led turn."

        return CoordinationPlan(
            round_id=round_id, decision_role=role, reason=reason,
            metadata={"seller_actor_id": seller_actor_id},
        )

    # -- State machine mode --

    def _plan_state_machine(
        self, round_id: int, buyer_text: str | None,
        seller_actor_id: str, max_rounds: int | None,
    ) -> CoordinationPlan:
        text = (buyer_text or "").strip().lower()

        if self._has_keywords(text, _ESCALATION_KEYWORDS):
            phase, role = "escalation", ActionRole.PLATFORM
        elif self._near_deadline(round_id, max_rounds):
            phase, role = "closing", ActionRole.PLATFORM
        elif self._has_keywords(text, _CLOSING_KEYWORDS):
            phase, role = "closing", ActionRole.PLATFORM
        elif self._has_keywords(text, _EXPERT_KEYWORDS) or "?" in text:
            phase, role = "recommendation", ActionRole.EXPERT
        elif round_id <= 0:
            phase, role = "discovery", ActionRole.SELLER
        else:
            phase, role = "negotiation", ActionRole.SELLER

        return CoordinationPlan(
            round_id=round_id, decision_role=role,
            reason=f"Phase={phase}; owner={role.value}.",
            metadata={"phase": phase, "seller_actor_id": seller_actor_id},
        )

    # -- Seller only mode --

    def _plan_seller_only(self, round_id: int, seller_actor_id: str) -> CoordinationPlan:
        return CoordinationPlan(
            round_id=round_id, decision_role=ActionRole.SELLER,
            reason="Seller-only baseline.",
            metadata={"seller_actor_id": seller_actor_id, "ablation": True},
        )

    # -- Helpers --

    def _near_deadline(self, round_id: int, max_rounds: int | None) -> bool:
        return (
            max_rounds is not None
            and max_rounds > 0
            and round_id >= max(0, max_rounds - self.escalation_round_buffer)
        )

    def _has_keywords(self, text: str, keywords: tuple[str, ...]) -> bool:
        return text != "" and any(kw in text for kw in keywords)
