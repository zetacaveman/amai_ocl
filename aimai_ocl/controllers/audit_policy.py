"""Audit policy for controlling trace verbosity and event coverage.

中文翻译：用于控制 trace 详细程度与事件覆盖范围的审计策略。"""

from __future__ import annotations

from dataclasses import dataclass

from aimai_ocl.schemas.audit import AuditEventType


@dataclass(frozen=True, slots=True)
class AuditPolicy:
    """Policy object deciding which audit event types are recorded.

    Inputs:
        policy_id:
            Stable policy identifier used for experiment logs and ablations.
        enabled_event_types:
            Event-type whitelist. ``None`` means allow all event types.
        description:
            Human-readable summary for protocol docs and diagnostics.

    Output:
        Lightweight policy object consumed by runners before appending events
        into ``EpisodeTrace``.

    中文翻译：决定哪些审计事件会被写入 trace 的策略对象。"""

    policy_id: str
    enabled_event_types: frozenset[AuditEventType] | None = None
    description: str = ""

    def should_record(self, event_type: AuditEventType) -> bool:
        """Return whether one event type should be appended to trace.

        中文翻译：返回某个事件类型是否应写入 trace。"""
        if self.enabled_event_types is None:
            return True
        return event_type in self.enabled_event_types
