"""Deterministic hard-constraint engine for seller-side OCL control.

This module is the step-3 rule layer in the OCL pipeline.

Contract:
- input: one ``RawAction`` plus optional control ``state/history``
- output: an ordered ``list[ConstraintCheck]``
- goal: convert rule outcomes into stable, auditable violation records


中文翻译：Deterministic hard-constraint engine for seller-side OCL control。"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from aimai_ocl.schemas.actions import ActionIntent, ActionRole, RawAction
from aimai_ocl.schemas.constraints import ConstraintCheck, ConstraintSeverity, ViolationType


# v1 privacy-sensitive keyword list (language/domain expansion planned later).
# 中文：v1 隐私敏感词列表（后续会按语言与业务域扩展）。
_SENSITIVE_KEYWORDS = (
    "password",
    "passcode",
    "ssn",
    "social security",
    "credit card",
    "cvv",
    "bank account",
    "routing number",
)


@dataclass(slots=True)
class ConstraintEngine:
    """Evaluate deterministic hard constraints for one raw action.

    Inputs:
        required_price_intents: Intents that must carry an explicit numeric
            price to be considered valid.

    Outputs:
        A rule engine exposing ``evaluate(...)`` that emits normalized
        ``ConstraintCheck`` records in a stable order:
        ``price_format -> privacy_policy -> budget_cap/seller_floor``.
    

    中文翻译：Evaluate deterministic hard constraints for one raw action。"""

    required_price_intents: tuple[ActionIntent, ...] = (
        ActionIntent.NEGOTIATE_PRICE,
        ActionIntent.ACCEPT_DEAL,
    )

    def evaluate(
        self,
        raw_action: RawAction,
        *,
        state: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> list[ConstraintCheck]:
        """Run all deterministic checks for one raw action.

        Input:
            raw_action:
                Proposed pre-control action.
            state:
                Optional environment/control state. Current rules read:
                ``buyer_max_price`` and ``seller_min_price``.
            history:
                Optional dialogue history. Kept as an API input for future
                history-based rules; not used by v1 logic.

        Output:
            Ordered ``list[ConstraintCheck]`` where each item has:
            - ``constraint_id`` (which rule ran)
            - ``passed`` and ``severity``
            - optional ``violation_type`` for failed checks
        

        中文翻译：运行 all deterministic checks for one raw action。"""
        _ = history  # Reserved for future deterministic history-based checks.
        checks: list[ConstraintCheck] = []
        checks.append(self._check_price_format(raw_action))
        checks.append(self._check_privacy_policy(raw_action))
        checks.extend(self._check_price_bounds(raw_action, state or {}))
        return checks

    def _check_price_format(self, raw_action: RawAction) -> ConstraintCheck:
        """Validate intent/price format consistency.

        Input:
            raw_action:
                Proposed action with ``intent`` and optional ``proposed_price``.

        Output:
            One ``ConstraintCheck`` with ``constraint_id=price_format``.
            Failure conditions:
            - intent is in ``required_price_intents`` but ``proposed_price`` is missing
            - ``proposed_price`` is non-positive
            On failure, emits ``violation_type=FORMAT_INVALID`` with ERROR severity.
        

        中文翻译：校验 intent/price format consistency。"""
        if (
            raw_action.intent in self.required_price_intents
            and raw_action.proposed_price is None
        ):
            return ConstraintCheck(
                constraint_id="price_format",
                passed=False,
                severity=ConstraintSeverity.ERROR,
                reason=(
                    f"Intent `{raw_action.intent.value}` requires an explicit "
                    "positive price."
                ),
                violation_type=ViolationType.FORMAT_INVALID,
                checked_fields=["intent", "proposed_price", "utterance"],
            )

        if raw_action.proposed_price is not None and raw_action.proposed_price <= 0:
            return ConstraintCheck(
                constraint_id="price_format",
                passed=False,
                severity=ConstraintSeverity.ERROR,
                reason="Proposed price must be positive.",
                violation_type=ViolationType.FORMAT_INVALID,
                checked_fields=["proposed_price"],
            )

        return ConstraintCheck(
            constraint_id="price_format",
            passed=True,
            severity=ConstraintSeverity.INFO,
            reason="Price format check passed.",
            checked_fields=["intent", "proposed_price"],
        )

    def _check_privacy_policy(self, raw_action: RawAction) -> ConstraintCheck:
        """Detect obvious privacy-sensitive content patterns.

        Input:
            raw_action:
                Proposed action text (``utterance``) to inspect.

        Output:
            One ``ConstraintCheck`` with ``constraint_id=privacy_policy``.
            Failure conditions:
            - sensitive keyword hit in text
            - likely payment identifier pattern (long digits + payment wording)
            On failure, emits ``violation_type=POLICY_PRIVACY`` with CRITICAL severity.
        

        中文翻译：Detect obvious privacy-sensitive content patterns。"""
        text = raw_action.utterance.lower()
        matched = [token for token in _SENSITIVE_KEYWORDS if token in text]
        if matched:
            return ConstraintCheck(
                constraint_id="privacy_policy",
                passed=False,
                severity=ConstraintSeverity.CRITICAL,
                reason=(
                    "Utterance contains privacy-sensitive keywords and cannot "
                    "be auto-executed."
                ),
                violation_type=ViolationType.POLICY_PRIVACY,
                checked_fields=["utterance"],
                metadata={"matched_keywords": matched},
            )

        # If there is a long digit run and explicit payment wording, treat it
        # as a likely sensitive identifier leakage attempt. 中文：当出现长数字串
        # 且包含支付语义时，按疑似敏感标识泄露处理。
        # 中文：若同时出现长数字串和支付语义词，按“疑似敏感标识泄露”处理，
        # 中文：并将其视为需要阻断的高风险输入。
        has_long_digits = re.search(r"\b(?:\d[ -]*?){13,19}\b", raw_action.utterance) is not None
        mentions_payment = any(token in text for token in ("card", "account", "bank", "payment"))
        if has_long_digits and mentions_payment:
            return ConstraintCheck(
                constraint_id="privacy_policy",
                passed=False,
                severity=ConstraintSeverity.CRITICAL,
                reason="Potential sensitive payment identifier detected in text.",
                violation_type=ViolationType.POLICY_PRIVACY,
                checked_fields=["utterance"],
            )

        return ConstraintCheck(
            constraint_id="privacy_policy",
            passed=True,
            severity=ConstraintSeverity.INFO,
            reason="Privacy policy check passed.",
            checked_fields=["utterance"],
        )

    def _check_price_bounds(
        self,
        raw_action: RawAction,
        state: dict[str, Any],
    ) -> list[ConstraintCheck]:
        """Validate seller proposal against buyer cap and seller floor bounds.

        Input:
            raw_action:
                Proposed action. Rule is active only for seller actions with
                a numeric ``proposed_price``.
            state:
                Control state that may provide ``buyer_max_price`` and
                ``seller_min_price``.

        Output:
            A two-item list in fixed order:
            1. ``budget_cap`` check
            2. ``seller_floor`` check
            Failure mapping:
            - price > buyer_max_price -> ``BUDGET_EXCEEDED`` (ERROR)
            - price < seller_min_price -> ``SELLER_FLOOR_BREACH`` (ERROR)
        

        中文翻译：校验 seller proposal against buyer cap and seller floor bounds。"""
        checks: list[ConstraintCheck] = []
        proposed_price = raw_action.proposed_price
        buyer_max_price = _coerce_float(state.get("buyer_max_price"))
        seller_min_price = _coerce_float(state.get("seller_min_price"))

        if raw_action.actor_role != ActionRole.SELLER or proposed_price is None:
            checks.append(
                ConstraintCheck(
                    constraint_id="budget_cap",
                    passed=True,
                    severity=ConstraintSeverity.INFO,
                    reason="Budget-cap check not applicable for this action.",
                    checked_fields=["actor_role", "proposed_price"],
                    metadata={"buyer_max_price": buyer_max_price},
                )
            )
            checks.append(
                ConstraintCheck(
                    constraint_id="seller_floor",
                    passed=True,
                    severity=ConstraintSeverity.INFO,
                    reason="Seller-floor check not applicable for this action.",
                    checked_fields=["actor_role", "proposed_price"],
                    metadata={"seller_min_price": seller_min_price},
                )
            )
            return checks

        if buyer_max_price is None:
            checks.append(
                ConstraintCheck(
                    constraint_id="budget_cap",
                    passed=True,
                    severity=ConstraintSeverity.INFO,
                    reason="buyer_max_price unavailable; budget-cap check skipped.",
                    checked_fields=["proposed_price", "buyer_max_price"],
                )
            )
        elif proposed_price > buyer_max_price:
            checks.append(
                ConstraintCheck(
                    constraint_id="budget_cap",
                    passed=False,
                    severity=ConstraintSeverity.ERROR,
                    reason=(
                        f"Seller proposed price {proposed_price:.2f} exceeds "
                        f"buyer max {buyer_max_price:.2f}."
                    ),
                    violation_type=ViolationType.BUDGET_EXCEEDED,
                    checked_fields=["proposed_price", "buyer_max_price"],
                )
            )
        else:
            checks.append(
                ConstraintCheck(
                    constraint_id="budget_cap",
                    passed=True,
                    severity=ConstraintSeverity.INFO,
                    reason=(
                        f"Seller proposed price {proposed_price:.2f} is within "
                        f"buyer max {buyer_max_price:.2f}."
                    ),
                    checked_fields=["proposed_price", "buyer_max_price"],
                )
            )

        if seller_min_price is None:
            checks.append(
                ConstraintCheck(
                    constraint_id="seller_floor",
                    passed=True,
                    severity=ConstraintSeverity.INFO,
                    reason="seller_min_price unavailable; seller-floor check skipped.",
                    checked_fields=["proposed_price", "seller_min_price"],
                )
            )
        elif proposed_price < seller_min_price:
            checks.append(
                ConstraintCheck(
                    constraint_id="seller_floor",
                    passed=False,
                    severity=ConstraintSeverity.ERROR,
                    reason=(
                        f"Seller proposed price {proposed_price:.2f} is below "
                        f"seller floor {seller_min_price:.2f}."
                    ),
                    violation_type=ViolationType.SELLER_FLOOR_BREACH,
                    checked_fields=["proposed_price", "seller_min_price"],
                )
            )
        else:
            checks.append(
                ConstraintCheck(
                    constraint_id="seller_floor",
                    passed=True,
                    severity=ConstraintSeverity.INFO,
                    reason=(
                        f"Seller proposed price {proposed_price:.2f} is above "
                        f"seller floor {seller_min_price:.2f}."
                    ),
                    checked_fields=["proposed_price", "seller_min_price"],
                )
            )

        return checks


def _coerce_float(value: Any) -> float | None:
    """Best-effort conversion to float for control-state numeric fields.

    Input:
        value:
            Raw field value from control state (for example ``int``, ``float``,
            numeric ``str``, or ``None``).

    Output:
        Parsed float when conversion succeeds; otherwise ``None``.
    

    中文翻译：Best-effort conversion to float for control-state numeric fields。"""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
