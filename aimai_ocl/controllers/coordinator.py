"""Minimal coordinator for seller-side multi-role OCL orchestration.

This module introduces a lightweight turn planner that maps the conceptual
roles from the paper draft into executable per-round assignments:

- A_u: external buyer/user simulator (passthrough)
- A_p: platform orchestrator (coordinator in this module)
- A_s: seller executor (actual env-facing actor)
- A_e: expert consultant (decision-support role)


中文翻译：Minimal coordinator for seller-side multi-role OCL orchestration。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from aimai_ocl.schemas.actions import ActionRole
from aimai_ocl.schemas.audit import AuditEvent, AuditEventType


_DEFAULT_EXPERT_KEYWORDS = (
    "compare",
    "difference",
    "which",
    "recommend",
    "spec",
    "feature",
    "material",
    "size",
    "fit",
    "warranty",
    "policy",
)

_ESCALATION_HINT_KEYWORDS = (
    "complaint",
    "refund",
    "angry",
    "unacceptable",
    "manager",
    "escalate",
    "human",
)

_CLOSING_HINT_KEYWORDS = (
    "deal",
    "final",
    "accept",
    "confirm",
    "checkout",
    "payment",
    "order",
)


class CoordinationPhase(str, Enum):
    """High-level round phase used by the role state-machine coordinator.

中文翻译：High-level round phase used by the role state-machine coordinator。"""

    DISCOVERY = "discovery"
    RECOMMENDATION = "recommendation"
    NEGOTIATION = "negotiation"
    CLOSING = "closing"
    ESCALATION = "escalation"


@dataclass(slots=True)
class CoordinationPlan:
    """One round-level role assignment emitted by the coordinator.

    Input:
        round_id:
            Current negotiation round.
        decision_role:
            Role that should own strategy/decision for this round.
        execution_role:
            Role allowed to emit env-facing executable text.
        escalation_role:
            Role that owns escalation when control blocks or risk rises.
        control_actor_id:
            Actor id that will be stamped on seller-side control events.
        control_actor_role:
            Role label used for role-policy and constraint checks.
        reason:
            Short rationale for why this assignment was chosen.
        metadata:
            Extra role mapping details for audit replay.

    Output:
        Structured coordination plan consumed by the OCL episode runner.
    

    中文翻译：One round-level role assignment emitted by the coordinator。"""

    round_id: int
    decision_role: ActionRole
    execution_role: ActionRole
    escalation_role: ActionRole
    control_actor_id: str
    control_actor_role: ActionRole
    reason: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class Coordinator:
    """Deterministic coordinator implementing a minimal role flow.

    Input:
        orchestrator_id:
            Stable actor id used for coordinator audit events (A_p).
        expert_keywords:
            Buyer-text keywords that trigger expert decision support (A_e).
        escalation_round_buffer:
            Number of rounds near deadline where platform should take decision
            ownership for safer closeout.

    Output:
        ``plan_turn(...)`` returns ``CoordinationPlan`` for each round.
        ``build_audit_event(...)`` converts one plan into a trace event.
    

    中文翻译：Deterministic coordinator implementing a minimal role flow。"""

    orchestrator_id: str = "platform_orchestrator"
    expert_keywords: tuple[str, ...] = _DEFAULT_EXPERT_KEYWORDS
    escalation_round_buffer: int = 1

    def plan_turn(
        self,
        *,
        round_id: int,
        buyer_text: str | None,
        seller_actor_id: str,
        max_rounds: int | None = None,
    ) -> CoordinationPlan:
        """Build one deterministic coordination plan for the current round.

        Input:
            round_id:
                Zero-based round index from environment observation.
            buyer_text:
                Latest buyer utterance already normalized by runner, or ``None``.
            seller_actor_id:
                Stable actor id of env-facing seller agent instance.
            max_rounds:
                Optional scenario round budget used for deadline routing.

        Output:
            ``CoordinationPlan`` describing:
            - who decides (A_s / A_e / A_p),
            - who executes (A_s),
            - who escalates (A_p),
            - which actor metadata is used for control checks.
        

        中文翻译：构建 one deterministic coordination plan for the current round。"""
        normalized = (buyer_text or "").strip().lower()
        near_deadline = (
            max_rounds is not None
            and (max_rounds > 0)
            and round_id >= max(0, max_rounds - self.escalation_round_buffer)
        )
        expert_needed = self._needs_expert_support(normalized)

        if near_deadline:
            decision_role = ActionRole.PLATFORM
            reason = "Near round budget limit; platform orchestrator owns closeout decision."
        elif expert_needed:
            decision_role = ActionRole.EXPERT
            reason = "Buyer request indicates product/policy clarification; expert assists."
        else:
            decision_role = ActionRole.SELLER
            reason = "Default seller-led negotiation turn."

        return CoordinationPlan(
            round_id=round_id,
            decision_role=decision_role,
            execution_role=ActionRole.SELLER,
            escalation_role=ActionRole.PLATFORM,
            control_actor_id=seller_actor_id,
            control_actor_role=ActionRole.SELLER,
            reason=reason,
            metadata={
                "A_u": "external_buyer_passthrough",
                "A_p": self.orchestrator_id,
                "A_s": seller_actor_id,
                "A_e": "expert_virtual_role",
                "buyer_text_present": bool(normalized),
                "expert_needed": expert_needed,
                "near_deadline": near_deadline,
            },
        )

    def build_audit_event(
        self,
        plan: CoordinationPlan,
    ) -> AuditEvent:
        """Convert one coordination plan into a structured trace event.

        Input:
            plan:
                Round-level assignment returned by ``plan_turn(...)``.

        Output:
            ``AuditEvent`` with ``event_type=COORDINATION_PLANNED`` and role
            ownership details in metadata.
        

        中文翻译：转换 one coordination plan into a structured trace event。"""
        return AuditEvent(
            event_type=AuditEventType.COORDINATION_PLANNED,
            round_id=plan.round_id,
            actor_id=self.orchestrator_id,
            summary=(
                "Coordinator planned round roles: "
                f"decision={plan.decision_role.value}, "
                f"execution={plan.execution_role.value}, "
                f"escalation={plan.escalation_role.value}"
            ),
            metadata={
                "decision_role": plan.decision_role.value,
                "execution_role": plan.execution_role.value,
                "escalation_role": plan.escalation_role.value,
                "control_actor_id": plan.control_actor_id,
                "control_actor_role": plan.control_actor_role.value,
                "reason": plan.reason,
                **plan.metadata,
            },
        )

    def _needs_expert_support(self, normalized_buyer_text: str) -> bool:
        """Decide whether buyer text should trigger expert participation.

        Input:
            normalized_buyer_text:
                Lowercased buyer text, or empty string.

        Output:
            ``True`` when text looks like product/policy clarification request.
        

        中文翻译：Decide whether buyer text should trigger expert participation。"""
        if normalized_buyer_text == "":
            return False
        if "?" in normalized_buyer_text:
            return True
        return any(keyword in normalized_buyer_text for keyword in self.expert_keywords)


@dataclass(slots=True)
class StateMachineCoordinator:
    """Role-decomposition algorithm with explicit phase-state transitions.

    This coordinator implements a deterministic finite-state policy over
    negotiation rounds. The policy is intentionally simple but explicit enough
    to serve as an algorithmic baseline for role-decomposition studies.

    Input:
        orchestrator_id:
            Stable actor id used in coordination audit events.
        expert_keywords:
            Keywords indicating that expert interpretation is needed.
        escalation_keywords:
            Keywords indicating dispute/risk requiring escalation ownership.
        closing_keywords:
            Keywords indicating intent to close the deal.
        escalation_round_buffer:
            Number of last rounds treated as near-deadline.

    Output:
        ``plan_turn(...)`` emits one ``CoordinationPlan`` with phase metadata:
        - ``phase``
        - ``next_phase``
        - ``transition_reason``
    

    中文翻译：Role-decomposition algorithm with explicit phase-state transitions。"""

    orchestrator_id: str = "platform_orchestrator"
    expert_keywords: tuple[str, ...] = _DEFAULT_EXPERT_KEYWORDS
    escalation_keywords: tuple[str, ...] = _ESCALATION_HINT_KEYWORDS
    closing_keywords: tuple[str, ...] = _CLOSING_HINT_KEYWORDS
    escalation_round_buffer: int = 1

    def plan_turn(
        self,
        *,
        round_id: int,
        buyer_text: str | None,
        seller_actor_id: str,
        max_rounds: int | None = None,
    ) -> CoordinationPlan:
        """Plan one round using explicit state-machine transitions.

        Input:
            round_id:
                Zero-based round index from the environment.
            buyer_text:
                Latest normalized buyer message, or ``None``.
            seller_actor_id:
                Stable seller actor id.
            max_rounds:
                Optional round budget used for deadline transition.

        Output:
            ``CoordinationPlan`` with state-machine phase annotations in
            ``metadata``.
        

        中文翻译：Plan one round using explicit state-machine transitions。"""
        normalized = (buyer_text or "").strip().lower()
        phase, transition_reason = self._infer_phase(
            round_id=round_id,
            normalized_buyer_text=normalized,
            max_rounds=max_rounds,
        )
        decision_role = self._phase_to_decision_role(phase)

        return CoordinationPlan(
            round_id=round_id,
            decision_role=decision_role,
            execution_role=ActionRole.SELLER,
            escalation_role=ActionRole.PLATFORM,
            control_actor_id=seller_actor_id,
            control_actor_role=ActionRole.SELLER,
            reason=(
                f"State-machine phase={phase.value}; "
                f"decision owner={decision_role.value}."
            ),
            metadata={
                "A_u": "external_buyer_passthrough",
                "A_p": self.orchestrator_id,
                "A_s": seller_actor_id,
                "A_e": "expert_virtual_role",
                "phase": phase.value,
                "next_phase": phase.value,
                "transition_reason": transition_reason,
                "buyer_text_present": bool(normalized),
            },
        )

    def build_audit_event(
        self,
        plan: CoordinationPlan,
    ) -> AuditEvent:
        """Serialize one state-machine plan as ``COORDINATION_PLANNED`` event.

        Input:
            plan:
                Coordination plan returned by ``plan_turn``.

        Output:
            One audit event including phase metadata for replay.
        

        中文翻译：序列化 one state-machine plan as ``COORDINATION_PLANNED`` event。"""
        return AuditEvent(
            event_type=AuditEventType.COORDINATION_PLANNED,
            round_id=plan.round_id,
            actor_id=self.orchestrator_id,
            summary=(
                "State-machine coordinator planned round roles: "
                f"decision={plan.decision_role.value}, "
                f"execution={plan.execution_role.value}, "
                f"escalation={plan.escalation_role.value}, "
                f"phase={plan.metadata.get('phase')}"
            ),
            metadata={
                "decision_role": plan.decision_role.value,
                "execution_role": plan.execution_role.value,
                "escalation_role": plan.escalation_role.value,
                "control_actor_id": plan.control_actor_id,
                "control_actor_role": plan.control_actor_role.value,
                "reason": plan.reason,
                **plan.metadata,
            },
        )

    def _infer_phase(
        self,
        *,
        round_id: int,
        normalized_buyer_text: str,
        max_rounds: int | None,
    ) -> tuple[CoordinationPhase, str]:
        """Infer state-machine phase from round position and buyer signal.

        Input:
            round_id:
                Current round index.
            normalized_buyer_text:
                Lowercased buyer text.
            max_rounds:
                Optional total round budget.

        Output:
            Tuple ``(phase, reason)`` describing chosen phase and transition
            rationale.
        

        中文翻译：推断 state-machine phase from round position and buyer signal。"""
        if self._contains_any(normalized_buyer_text, self.escalation_keywords):
            return (
                CoordinationPhase.ESCALATION,
                "Buyer text indicates dispute/risk requiring escalation ownership.",
            )

        near_deadline = (
            max_rounds is not None
            and (max_rounds > 0)
            and round_id >= max(0, max_rounds - self.escalation_round_buffer)
        )
        if near_deadline:
            return (
                CoordinationPhase.CLOSING,
                "Round near deadline; prioritize controlled closeout.",
            )

        if self._contains_any(normalized_buyer_text, self.closing_keywords):
            return (
                CoordinationPhase.CLOSING,
                "Buyer text indicates close/deal intent.",
            )

        if self._contains_any(normalized_buyer_text, self.expert_keywords) or "?" in normalized_buyer_text:
            return (
                CoordinationPhase.RECOMMENDATION,
                "Buyer asks for explanation/comparison; expert guidance needed.",
            )

        if round_id <= 0:
            return (
                CoordinationPhase.DISCOVERY,
                "First round defaults to discovery.",
            )

        return (
            CoordinationPhase.NEGOTIATION,
            "Default iterative negotiation phase.",
        )

    def _phase_to_decision_role(self, phase: CoordinationPhase) -> ActionRole:
        """Map one phase to the role owning round-level decision.

        Input:
            phase:
                State-machine phase label.

        Output:
            Decision owner role for this phase.
        

        中文翻译：映射 one phase to the role owning round-level decision。"""
        if phase == CoordinationPhase.RECOMMENDATION:
            return ActionRole.EXPERT
        if phase in {CoordinationPhase.CLOSING, CoordinationPhase.ESCALATION}:
            return ActionRole.PLATFORM
        return ActionRole.SELLER

    def _contains_any(self, text: str, keywords: tuple[str, ...]) -> bool:
        """Check if text contains any keyword from the provided tuple.

        Input:
            text:
                Lowercased text to inspect.
            keywords:
                Keyword tuple used for deterministic matching.

        Output:
            ``True`` when at least one keyword is present.
        

        中文翻译：Check if text contains any keyword from the provided tuple。"""
        if text == "":
            return False
        return any(keyword in text for keyword in keywords)


@dataclass(slots=True)
class SellerOnlyCoordinator:
    """Ablation coordinator that always routes decisions to seller role.

    Input:
        orchestrator_id:
            Stable actor id stamped on coordination audit events.

    Output:
        ``plan_turn(...)`` always emits seller-led plan, allowing clean
        role-decomposition ablation while keeping the same runner contract.
    

    中文翻译：Ablation coordinator that always routes decisions to seller role。"""

    orchestrator_id: str = "platform_orchestrator"

    def plan_turn(
        self,
        *,
        round_id: int,
        buyer_text: str | None,
        seller_actor_id: str,
        max_rounds: int | None = None,
    ) -> CoordinationPlan:
        """Return deterministic seller-only role plan for one round.

        Input:
            round_id:
                Zero-based round index.
            buyer_text:
                Latest buyer text (unused in this ablation policy).
            seller_actor_id:
                Env-facing seller actor id.
            max_rounds:
                Optional round budget (unused in this ablation policy).

        Output:
            ``CoordinationPlan`` with decision/execution role fixed to seller.
        

        中文翻译：返回 deterministic seller-only role plan for one round。"""
        del buyer_text, max_rounds
        return CoordinationPlan(
            round_id=round_id,
            decision_role=ActionRole.SELLER,
            execution_role=ActionRole.SELLER,
            escalation_role=ActionRole.PLATFORM,
            control_actor_id=seller_actor_id,
            control_actor_role=ActionRole.SELLER,
            reason="Seller-only coordination baseline (role decomposition off).",
            metadata={
                "A_u": "external_buyer_passthrough",
                "A_p": self.orchestrator_id,
                "A_s": seller_actor_id,
                "A_e": "disabled_in_seller_only_policy",
                "ablation": "role_decomposition_off",
            },
        )

    def build_audit_event(
        self,
        plan: CoordinationPlan,
    ) -> AuditEvent:
        """Serialize seller-only plan to one coordination audit event.

        Input:
            plan:
                Round plan returned by ``plan_turn``.

        Output:
            ``AuditEvent`` with event type ``COORDINATION_PLANNED``.
        

        中文翻译：序列化 seller-only plan to one coordination audit event。"""
        return AuditEvent(
            event_type=AuditEventType.COORDINATION_PLANNED,
            round_id=plan.round_id,
            actor_id=self.orchestrator_id,
            summary=(
                "Seller-only coordinator planned round roles: "
                f"decision={plan.decision_role.value}, "
                f"execution={plan.execution_role.value}, "
                f"escalation={plan.escalation_role.value}"
            ),
            metadata={
                "decision_role": plan.decision_role.value,
                "execution_role": plan.execution_role.value,
                "escalation_role": plan.escalation_role.value,
                "control_actor_id": plan.control_actor_id,
                "control_actor_role": plan.control_actor_role.value,
                "reason": plan.reason,
                **plan.metadata,
            },
        )
