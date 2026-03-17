"""Minimal adapter for calling AgenticPay environments from AiMai OCL.

中文翻译：Minimal adapter for calling AgenticPay environments from AiMai OCL。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from aimai_ocl.agenticpay_runtime import make_env
from aimai_ocl.schemas.audit import AuditEvent, AuditEventType, EpisodeTrace


@dataclass(slots=True)
class AgenticPayEnvAdapter:
    """Thin wrapper around ``agenticpay.make`` and the resulting environment.

    Inputs:
        env_id: AgenticPay environment id to instantiate.
        env_kwargs: Keyword arguments forwarded to ``agenticpay.make``.

    Outputs:
        An adapter object exposing explicit ``reset`` and ``step`` methods while
        keeping the selected environment id and construction kwargs visible for
        auditing and future wrappers.

    Notes:
        This class is intentionally thin. At this stage it exists mainly to make
        the AgenticPay call path obvious and to provide a home for trace
        creation.
    

    中文翻译：Thin wrapper around ``agenticpay.make`` and the resulting environment。"""

    env_id: str
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    env: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Instantiate the underlying AgenticPay environment.

中文翻译：Instantiate the underlying AgenticPay environment。"""
        self.env = make_env(self.env_id, **self.env_kwargs)

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the underlying AgenticPay environment.

        Args:
            **kwargs: Keyword arguments forwarded directly to ``env.reset``.

        Returns:
            A tuple ``(observation, info)`` from the underlying environment.
        

        中文翻译：Reset the underlying AgenticPay environment。"""
        return self.env.reset(**kwargs)

    def step(
        self,
        buyer_action: str | None = None,
        seller_action: str | None = None,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Advance the underlying AgenticPay environment by one step.

        Args:
            buyer_action: Text action to pass to the buyer side of the
                environment.
            seller_action: Text action to pass to the seller side of the
                environment.

        Returns:
            The standard AgenticPay step tuple:
            ``(observation, reward, terminated, truncated, info)``.
        

        中文翻译：Advance the underlying AgenticPay environment by one step。"""
        return self.env.step(
            buyer_action=buyer_action,
            seller_action=seller_action,
        )

    def new_episode_trace(
        self,
        scenario: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EpisodeTrace:
        """Create a fresh episode trace for one environment run.

        Args:
            scenario: Optional structured scenario payload captured at the start
                of the episode.
            metadata: Optional experiment bookkeeping fields.

        Returns:
            A new ``EpisodeTrace`` initialized with an ``EPISODE_STARTED`` audit
            event.
        

        中文翻译：创建 a fresh episode trace for one environment run。"""
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
