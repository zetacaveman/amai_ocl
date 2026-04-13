"""AgenticPay integration: env wrapper, action parsing, agent construction."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from aimai_ocl.schemas import (
    ActionIntent,
    ActionRole,
    AuditEvent,
    AuditEventType,
    ControlDecision,
    EpisodeTrace,
    ExecutableAction,
    RawAction,
)

# ---------------------------------------------------------------------------
# AgenticPay import helpers
# ---------------------------------------------------------------------------

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
SUPPORTED_PROVIDERS = ("openai",)


def load_agenticpay() -> Any:
    """Import the AgenticPay package."""
    import agenticpay  # noqa: PLC0415
    return agenticpay


def make_env(env_id: str, **kwargs: Any) -> Any:
    """Create an AgenticPay environment via ``agenticpay.make``."""
    return load_agenticpay().make(env_id, **kwargs)


# ---------------------------------------------------------------------------
# Text -> action parsing
# ---------------------------------------------------------------------------

_PRICE_PATTERNS = (
    r"###\s*(?:BUYER_PRICE|SELLER_PRICE)\s*\(\$([\d,]+\.?\d*)\)\s*###",
    r"###\s*\$([\d,]+\.?\d*)\s*###",
    r"\$([\d,]+\.?\d*)",
)


def extract_price_from_text(text: str) -> float | None:
    """Extract the last positive price from text."""
    for pattern in _PRICE_PATTERNS:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            try:
                value = float(matches[-1].replace(",", ""))
            except ValueError:
                continue
            if value > 0:
                return value
    return None


def infer_intent_from_text(text: str) -> ActionIntent:
    """Infer a coarse action intent from text."""
    if "MAKE_DEAL" in text.upper():
        return ActionIntent.ACCEPT_DEAL
    if "?" in text:
        return ActionIntent.REQUEST_INFO
    if extract_price_from_text(text) is not None:
        return ActionIntent.NEGOTIATE_PRICE
    return ActionIntent.OTHER


def raw_action_from_text(actor_id: str, actor_role: ActionRole, text: str) -> RawAction:
    """Build a RawAction from an AgenticPay text message."""
    return RawAction(
        actor_id=actor_id,
        actor_role=actor_role,
        utterance=text,
        intent=infer_intent_from_text(text),
        proposed_price=extract_price_from_text(text),
    )


def passthrough_executable(raw: RawAction) -> ExecutableAction:
    """Create a pass-through executable action (no gating)."""
    return ExecutableAction(
        actor_id=raw.actor_id,
        actor_role=raw.actor_role,
        approved=True,
        decision=ControlDecision.APPROVE,
        final_text=raw.utterance,
        intent=raw.intent,
        final_price=raw.proposed_price,
    )


# ---------------------------------------------------------------------------
# Environment adapter
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EnvAdapter:
    """Thin wrapper around an AgenticPay environment."""

    env_id: str
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    env: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.env = make_env(self.env_id, **self.env_kwargs)

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.env.reset(**kwargs)

    def step(
        self,
        buyer_action: str | None = None,
        seller_action: str | None = None,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        return self.env.step(buyer_action=buyer_action, seller_action=seller_action)

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass

    def new_trace(
        self,
        scenario: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EpisodeTrace:
        """Create a fresh episode trace."""
        trace = EpisodeTrace(
            episode_id=str(uuid4()),
            env_id=self.env_id,
            scenario=scenario or {},
            metadata=metadata or {},
        )
        trace.add_event(
            AuditEvent(
                event_type=AuditEventType.EPISODE_STARTED,
                summary=f"Episode started for {self.env_id}",
            )
        )
        return trace


# ---------------------------------------------------------------------------
# Model client + agent builders
# ---------------------------------------------------------------------------


def build_model_client(*, provider: str = "openai", model: str, api_key_env: str = OPENAI_API_KEY_ENV) -> Any:
    """Instantiate a model client for the configured provider."""
    if provider != "openai":
        raise RuntimeError(f"Unsupported provider: '{provider}'. Supported: {SUPPORTED_PROVIDERS}")
    try:
        from agenticpay.models.openai_llm import OpenAILLM  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"Missing dependency: {exc}") from exc
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set.")
    return OpenAILLM(model=model, api_key=api_key)


def build_agents(
    *,
    model: str,
    buyer_max_price: float,
    seller_min_price: float,
    provider: str = "openai",
    api_key_env: str = OPENAI_API_KEY_ENV,
) -> tuple[Any, Any]:
    """Build AgenticPay buyer/seller agents. Returns (buyer, seller)."""
    try:
        from agenticpay.agents.buyer_agent import BuyerAgent  # noqa: PLC0415
        from agenticpay.agents.seller_agent import SellerAgent  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"Missing dependency: {exc}") from exc

    buyer_model = build_model_client(provider=provider, model=model, api_key_env=api_key_env)
    seller_model = build_model_client(provider=provider, model=model, api_key_env=api_key_env)
    buyer = BuyerAgent(model=buyer_model, buyer_max_price=buyer_max_price)
    seller = SellerAgent(model=seller_model, seller_min_price=seller_min_price)
    return buyer, seller


# ---------------------------------------------------------------------------
# Scenario validation
# ---------------------------------------------------------------------------


def enforce_single_product(reset_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize single-product scenario inputs."""
    normalized = dict(reset_kwargs)
    if "products" in normalized:
        products = normalized["products"]
        if not isinstance(products, list) or len(products) != 1:
            raise ValueError("Current scope only supports exactly one product.")
        normalized.setdefault("product_info", products[0])
        normalized.pop("products", None)
    if not isinstance(normalized.get("product_info"), dict):
        raise ValueError("Single-product scope requires `product_info` dict.")
    return normalized
