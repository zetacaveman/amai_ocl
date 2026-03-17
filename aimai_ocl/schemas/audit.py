"""Audit schemas for tracing OCL decisions over an episode.

中文翻译：Audit schemas for tracing OCL decisions over an episode。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from aimai_ocl.schemas.actions import ExecutableAction, RawAction
from aimai_ocl.schemas.constraints import ConstraintCheck


class AuditEventType(str, Enum):
    """Canonical event types emitted by the minimal OCL pipeline.

中文翻译：Canonical event types emitted by the minimal OCL pipeline。"""

    EPISODE_STARTED = "episode_started"
    COORDINATION_PLANNED = "coordination_planned"
    RAW_ACTION_RECEIVED = "raw_action_received"
    CONSTRAINT_EVALUATED = "constraint_evaluated"
    ACTION_EXECUTED = "action_executed"
    ESCALATION_TRIGGERED = "escalation_triggered"
    REPLAN_APPLIED = "replan_applied"
    EPISODE_FINISHED = "episode_finished"


@dataclass(slots=True)
class AuditEvent:
    """One structured event in the control-layer audit trail.

    Inputs:
        event_type: The type of event being recorded.
        round_id: Optional logical round index associated with this event.
        actor_id: Optional actor responsible for the event.
        summary: Short human-readable description of what happened.
        raw_action: Optional pre-control action associated with the event.
        executable_action: Optional post-control action associated with the
            event.
        constraint_checks: Zero or more constraint evaluation results relevant
            to this event.
        metadata: Additional structured details for debugging or offline
            analysis.

    Outputs:
        A normalized audit record suitable for trace replay, attribution, and
        failure analysis.
    

    中文翻译：One structured event in the control-layer audit trail。"""

    event_type: AuditEventType
    round_id: int | None = None
    actor_id: str | None = None
    summary: str = ""
    raw_action: RawAction | None = None
    executable_action: ExecutableAction | None = None
    constraint_checks: list[ConstraintCheck] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodeTrace:
    """Top-level audit container for one environment episode.

    Inputs:
        episode_id: Stable identifier for the run.
        env_id: AgenticPay environment id used for the episode.
        scenario: Structured scenario description or episode setup payload.
        events: Ordered audit events emitted during the run.
        final_status: Optional terminal status string from the environment or
            control layer.
        final_metrics: Optional terminal metrics such as reward, welfare,
            violation counts, or round count.
        metadata: Free-form episode-level data for experiment bookkeeping.

    Outputs:
        A replayable record of one run that can later feed debugging,
        benchmarking, or attribution code.
    

    中文翻译：one environment episode 的顶层审计容器。"""

    episode_id: str
    env_id: str
    scenario: dict[str, Any] = field(default_factory=dict)
    events: list[AuditEvent] = field(default_factory=list)
    final_status: str | None = None
    final_metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_event(self, event: AuditEvent) -> None:
        """Append one audit event to the episode trace.

        Args:
            event: The structured audit event to append.

        Returns:
            None. The trace is mutated in place.
        

        中文翻译：追加 one audit event to the episode trace。"""
        self.events.append(event)
