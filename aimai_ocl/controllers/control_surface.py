"""Control-surface parameterization for organizational-control intensity.

This module lifts paper-level ``tau`` into a structured runtime policy surface
instead of treating it as one threshold knob.

中文翻译：把论文中的 ``tau`` 提升为结构化控制面，而不是单一阈值旋钮。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aimai_ocl.schemas.actions import ActionIntent


def _normalize_tau(gate_tau: float) -> float:
    try:
        tau = float(gate_tau)
    except (TypeError, ValueError) as exc:
        raise ValueError("gate_tau must be a float in [0, 1].") from exc
    if not 0.0 <= tau <= 1.0:
        raise ValueError("gate_tau must be in [0, 1].")
    return tau


def _clip(value: float, *, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


@dataclass(frozen=True, slots=True)
class TauControlSurface:
    """Structured control surface derived from paper-level ``tau``.

    中文翻译：由论文级 ``tau`` 派生出的结构化控制面。"""

    gate_tau: float
    rewrite_threshold: float
    block_threshold: float
    epsilon_miss: float
    fixed_high_risk_intents: tuple[ActionIntent, ...]
    intent_risk_priors: dict[ActionIntent, float]
    price_penalty_in_score: bool = False

    def to_runtime_config(self) -> dict[str, Any]:
        priors = {
            intent.value: round(float(value), 6)
            for intent, value in sorted(
                self.intent_risk_priors.items(),
                key=lambda item: item[0].value,
            )
        }
        return {
            "gate_family": "tau_controlled",
            "tau_applied": True,
            "gate_tau": self.gate_tau,
            "rewrite_threshold": self.rewrite_threshold,
            "block_threshold": self.block_threshold,
            "epsilon_miss": self.epsilon_miss,
            "fixed_high_risk_intents": [intent.value for intent in self.fixed_high_risk_intents],
            "intent_risk_priors": priors,
            "price_penalty_in_score": self.price_penalty_in_score,
        }


def tau_control_surface_from_tau(gate_tau: float) -> TauControlSurface:
    """Map one scalar ``tau`` to a structured organizational-control surface.

    Design goals:
    - ``tau`` stays continuous in ``[0, 1]``
    - ``tau=0`` approximates a legacy low-control regime
    - ``tau=1`` corresponds to stronger organizational control
    - hard price constraints remain outside this soft-governance surface

    中文翻译：把标量 ``tau`` 映射为结构化组织控制面。"""
    tau = _normalize_tau(gate_tau)

    rewrite_threshold = _clip(0.92 - (0.27 * tau))
    block_threshold = _clip(1.02 - (0.32 * tau))
    if block_threshold <= rewrite_threshold:
        block_threshold = _clip(rewrite_threshold + 0.05)

    return TauControlSurface(
        gate_tau=tau,
        rewrite_threshold=rewrite_threshold,
        block_threshold=block_threshold,
        epsilon_miss=_clip(0.22 - (0.14 * tau)),
        fixed_high_risk_intents=(
            ActionIntent.TOOL_CALL,
            ActionIntent.ESCALATE,
        ),
        intent_risk_priors={
            ActionIntent.TOOL_CALL: 0.55,
            ActionIntent.ESCALATE: 0.60,
            ActionIntent.ACCEPT_DEAL: 0.05 + (0.25 * tau),
            ActionIntent.NEGOTIATE_PRICE: 0.02 + (0.12 * tau),
            ActionIntent.REJECT_DEAL: 0.02 + (0.05 * tau),
            ActionIntent.EXPLAIN_POLICY: 0.01 + (0.04 * tau),
        },
        price_penalty_in_score=False,
    )

