"""Risk gate for the minimal OCL controller skeleton.

This module answers the second control question:

"Even if the actor is allowed to attempt this action, should the action pass
through immediately, require confirmation, or be blocked?"


中文翻译：Risk gate for the minimal OCL controller skeleton。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aimai_ocl.schemas.actions import (
    ActionIntent,
    ControlDecision,
    ExecutableAction,
    RawAction,
)
from aimai_ocl.schemas.constraints import ConstraintCheck, ConstraintSeverity, ViolationType


@dataclass(slots=True)
class RiskGate:
    """Minimal risk gate for shaping executable actions.

    Inputs:
        high_risk_intents: Intents that should not pass through silently.
        require_confirmation_for_high_risk: Whether high-risk intents should be
            marked as requiring confirmation.

    Outputs:
        A gate object that produces additional ``ConstraintCheck`` records and a
        first-pass ``ExecutableAction`` decision.
    

    中文翻译：Minimal risk gate for shaping executable actions。"""

    high_risk_intents: tuple[ActionIntent, ...] = (
        ActionIntent.TOOL_CALL,
        ActionIntent.ESCALATE,
    )
    require_confirmation_for_high_risk: bool = True

    def evaluate(
        self,
        raw_action: RawAction,
        *,
        state: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> ConstraintCheck:
        """Evaluate the risk posture of a raw action.

        Args:
            raw_action: The pre-control action proposed by an actor.
            state: Optional control state (unused in v1 risk gate).
            history: Optional conversation history (unused in v1 risk gate).

        Returns:
            A ``ConstraintCheck`` describing whether the action is considered
            low risk, warning-level high risk, or malformed.
        

        中文翻译：Evaluate the risk posture of a raw action。"""
        del state, history
        if raw_action.intent in self.high_risk_intents:
            return ConstraintCheck(
                constraint_id="risk_gate",
                passed=True,
                severity=ConstraintSeverity.WARNING,
                reason=(
                    f"Intent `{raw_action.intent.value}` is marked high risk and "
                    "should be reviewed by the controller."
                ),
                violation_type=ViolationType.HIGH_RISK_ACTION,
                checked_fields=["intent"],
            )

        return ConstraintCheck(
            constraint_id="risk_gate",
            passed=True,
            severity=ConstraintSeverity.INFO,
            reason=f"Intent `{raw_action.intent.value}` is low risk.",
            checked_fields=["intent"],
        )

    def apply(
        self,
        raw_action: RawAction,
        checks: list[ConstraintCheck],
        *,
        state: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> ExecutableAction:
        """Turn a raw action plus current checks into a first executable action.

        Args:
            raw_action: The original untrusted action proposal.
            checks: Constraint checks already collected for this action,
                typically including at least the role-policy result.
            state: Optional control state (unused in v1 risk gate).
            history: Optional conversation history (unused in v1 risk gate).

        Returns:
            An ``ExecutableAction`` that either passes through, is marked as
            requiring confirmation, or is blocked if an earlier check failed.
        

        中文翻译：Turn a raw action plus current checks into a first executable action。"""
        del state, history
        has_hard_failure = any(
            (not check.passed) and check.severity in {ConstraintSeverity.ERROR, ConstraintSeverity.CRITICAL}
            for check in checks
        )
        requires_confirmation = self.require_confirmation_for_high_risk and (
            raw_action.intent in self.high_risk_intents
        )

        if has_hard_failure:
            return ExecutableAction(
                actor_id=raw_action.actor_id,
                actor_role=raw_action.actor_role,
                approved=False,
                decision=ControlDecision.BLOCK,
                final_text="",
                intent=raw_action.intent,
                final_price=None,
                blocked_reason="Blocked by failed control-layer checks.",
                requires_confirmation=False,
                requires_escalation=False,
            )

        decision = ControlDecision.APPROVE
        if requires_confirmation:
            decision = ControlDecision.REWRITE
        elif raw_action.intent == ActionIntent.ESCALATE:
            decision = ControlDecision.ESCALATE

        return ExecutableAction(
            actor_id=raw_action.actor_id,
            actor_role=raw_action.actor_role,
            approved=True,
            decision=decision,
            final_text=raw_action.utterance,
            intent=raw_action.intent,
            final_price=raw_action.proposed_price,
            blocked_reason=None,
            requires_confirmation=requires_confirmation,
            requires_escalation=raw_action.intent == ActionIntent.ESCALATE,
        )


@dataclass(slots=True)
class BarrierRiskGate:
    """Risk-gating control barrier with explicit score thresholds.

    This algorithm estimates a bounded risk score in ``[0, 1]`` and maps the
    score to control decisions:
    - score < rewrite_threshold: approve
    - rewrite_threshold <= score < block_threshold: rewrite/confirm
    - score >= block_threshold: block + escalation

    Input:
        high_risk_intents:
            Intents that contribute strong prior risk.
        rewrite_threshold:
            Score threshold for confirmation/rewrite path.
        block_threshold:
            Score threshold for block/escalation path.
        epsilon_miss:
            Upper-bound proxy of false-pass probability on high-risk actions.
            Lower epsilon implies stronger barrier behavior.
        require_confirmation_for_high_risk:
            Whether high-risk intents should force confirmation below block
            threshold.

    Output:
        A gate algorithm exposing ``evaluate(...)`` and ``apply(...)`` that
        emits risk metadata suitable for audit and theory appendix logging.
    

    中文翻译：Risk-gating control barrier with explicit score thresholds。"""

    high_risk_intents: tuple[ActionIntent, ...] = (
        ActionIntent.TOOL_CALL,
        ActionIntent.ESCALATE,
        ActionIntent.ACCEPT_DEAL,
        ActionIntent.NEGOTIATE_PRICE,
    )
    rewrite_threshold: float = 0.45
    block_threshold: float = 0.75
    epsilon_miss: float = 0.10
    require_confirmation_for_high_risk: bool = True

    def evaluate(
        self,
        raw_action: RawAction,
        *,
        state: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> ConstraintCheck:
        """Evaluate action risk and return one barrier risk check.

        Input:
            raw_action:
                Pre-control action proposal.
            state:
                Optional control state with price bounds.
            history:
                Optional conversation history; only length is used in v1.

        Output:
            ``ConstraintCheck`` with:
            - ``constraint_id=risk_gate_barrier``
            - warning/error severity by threshold zone
            - metadata including ``risk_score`` and threshold config
        

        中文翻译：Evaluate action risk and return one barrier risk check。"""
        risk_score = self._risk_score(
            raw_action=raw_action,
            state=state or {},
            history=history or [],
        )
        severity = self._severity_from_score(risk_score)
        passed = risk_score < self.block_threshold
        reason = (
            f"Barrier risk score={risk_score:.3f} "
            f"(rewrite@{self.rewrite_threshold:.2f}, block@{self.block_threshold:.2f})."
        )
        return ConstraintCheck(
            constraint_id="risk_gate_barrier",
            passed=passed,
            severity=severity,
            reason=reason,
            violation_type=(
                ViolationType.HIGH_RISK_ACTION
                if risk_score >= self.rewrite_threshold
                else None
            ),
            checked_fields=["intent", "utterance", "proposed_price"],
            metadata={
                "risk_score": risk_score,
                "rewrite_threshold": self.rewrite_threshold,
                "block_threshold": self.block_threshold,
                "epsilon_miss": self.epsilon_miss,
                "violation_upper_bound_factor": self._violation_upper_bound_factor(),
            },
        )

    def apply(
        self,
        raw_action: RawAction,
        checks: list[ConstraintCheck],
        *,
        state: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> ExecutableAction:
        """Apply control-barrier decision from checks and risk score.

        Input:
            raw_action:
                Original action proposal.
            checks:
                Collected checks including role/constraint results and barrier
                risk check.
            state:
                Optional state passed for deterministic tie-breaking.
            history:
                Optional history; not used directly in decision.

        Output:
            ``ExecutableAction`` with decision in
            ``approve/rewrite/block/escalate`` space.
        

        中文翻译：应用 control-barrier decision from checks and risk score。"""
        del history  # Reserved for future trajectory-sensitive barriers.
        has_hard_failure = any(
            (not check.passed)
            and check.severity in {ConstraintSeverity.ERROR, ConstraintSeverity.CRITICAL}
            and check.constraint_id != "risk_gate_barrier"
            for check in checks
        )
        if has_hard_failure:
            return ExecutableAction(
                actor_id=raw_action.actor_id,
                actor_role=raw_action.actor_role,
                approved=False,
                decision=ControlDecision.BLOCK,
                final_text="",
                intent=raw_action.intent,
                final_price=None,
                blocked_reason="Blocked by failed hard checks before barrier pass.",
                requires_confirmation=False,
                requires_escalation=True,
                metadata={"barrier_decision": "hard_block"},
            )

        risk_score = self._extract_risk_score_from_checks(checks)
        if risk_score is None:
            risk_score = self._risk_score(
                raw_action=raw_action,
                state=state or {},
                history=[],
            )

        if raw_action.intent == ActionIntent.ESCALATE:
            return ExecutableAction(
                actor_id=raw_action.actor_id,
                actor_role=raw_action.actor_role,
                approved=True,
                decision=ControlDecision.ESCALATE,
                final_text=raw_action.utterance,
                intent=raw_action.intent,
                final_price=raw_action.proposed_price,
                blocked_reason=None,
                requires_confirmation=False,
                requires_escalation=True,
                metadata={"barrier_score": risk_score, "barrier_decision": "explicit_escalate"},
            )

        if risk_score >= self.block_threshold:
            return ExecutableAction(
                actor_id=raw_action.actor_id,
                actor_role=raw_action.actor_role,
                approved=False,
                decision=ControlDecision.BLOCK,
                final_text="",
                intent=raw_action.intent,
                final_price=None,
                blocked_reason=(
                    f"Blocked by barrier score {risk_score:.3f} "
                    f">= block threshold {self.block_threshold:.2f}."
                ),
                requires_confirmation=False,
                requires_escalation=True,
                metadata={"barrier_score": risk_score, "barrier_decision": "score_block"},
            )

        requires_confirmation = (
            risk_score >= self.rewrite_threshold
            or (
                self.require_confirmation_for_high_risk
                and raw_action.intent in self.high_risk_intents
            )
        )
        decision = ControlDecision.REWRITE if requires_confirmation else ControlDecision.APPROVE
        return ExecutableAction(
            actor_id=raw_action.actor_id,
            actor_role=raw_action.actor_role,
            approved=True,
            decision=decision,
            final_text=raw_action.utterance,
            intent=raw_action.intent,
            final_price=raw_action.proposed_price,
            blocked_reason=None,
            requires_confirmation=requires_confirmation,
            requires_escalation=False,
            metadata={"barrier_score": risk_score, "barrier_decision": decision.value},
        )

    def _risk_score(
        self,
        *,
        raw_action: RawAction,
        state: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> float:
        """Compute deterministic risk score in ``[0, 1]``.

        Input:
            raw_action:
                Action proposal to score.
            state:
                Control state containing optional budget/floor bounds.
            history:
                Conversation history (length contributes mild long-horizon risk).

        Output:
            Clamped risk score float within ``[0.0, 1.0]``.
        

        中文翻译：计算 deterministic risk score in ``[0, 1]``。"""
        score = 0.05
        if raw_action.intent in self.high_risk_intents:
            score += 0.35

        text = raw_action.utterance.lower()
        if any(token in text for token in ("password", "card", "bank", "account", "ssn", "cvv")):
            score += 0.45
        if "!" in raw_action.utterance:
            score += 0.05

        proposed = _coerce_float(raw_action.proposed_price)
        buyer_max = _coerce_float(state.get("buyer_max_price"))
        seller_min = _coerce_float(state.get("seller_min_price"))
        if proposed is not None:
            if (buyer_max is not None) and proposed > buyer_max:
                exceed_ratio = (proposed - buyer_max) / max(1.0, buyer_max)
                score += min(0.30, 0.10 + exceed_ratio)
            if (seller_min is not None) and proposed < seller_min:
                breach_ratio = (seller_min - proposed) / max(1.0, seller_min)
                score += min(0.25, 0.10 + breach_ratio)

        if len(history) >= 12:
            score += 0.05

        return min(1.0, max(0.0, score))

    def _extract_risk_score_from_checks(
        self,
        checks: list[ConstraintCheck],
    ) -> float | None:
        """Extract barrier risk score from existing checks when available.

        Input:
            checks:
                Collected checks emitted by the control pipeline.

        Output:
            Risk score if a barrier check with score metadata exists,
            otherwise ``None``.
        

        中文翻译：Extract barrier risk score from existing checks when available。"""
        for check in checks:
            if check.constraint_id != "risk_gate_barrier":
                continue
            value = check.metadata.get("risk_score")
            parsed = _coerce_float(value)
            if parsed is not None:
                return min(1.0, max(0.0, parsed))
        return None

    def _severity_from_score(self, risk_score: float) -> ConstraintSeverity:
        """Map risk score to check severity bucket.

        Input:
            risk_score:
                Score in ``[0, 1]``.

        Output:
            Severity level used by barrier check.
        

        中文翻译：映射 risk score to check severity bucket。"""
        if risk_score >= self.block_threshold:
            return ConstraintSeverity.ERROR
        if risk_score >= self.rewrite_threshold:
            return ConstraintSeverity.WARNING
        return ConstraintSeverity.INFO

    def _violation_upper_bound_factor(self) -> float:
        """Return simple monotonic upper-bound factor based on epsilon.

        Input:
            None.

        Output:
            Scalar factor ``(1 - epsilon_miss)`` in ``[0, 1]``.
        

        中文翻译：返回 simple monotonic upper-bound factor based on epsilon。"""
        eps = min(1.0, max(0.0, self.epsilon_miss))
        return 1.0 - eps


def _coerce_float(value: Any) -> float | None:
    """Best-effort conversion to float for risk-gate numeric parsing.

    Input:
        value:
            Any runtime value.

    Output:
        Parsed float or ``None``.
    

    中文翻译：Best-effort conversion to float for risk-gate numeric parsing。"""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
