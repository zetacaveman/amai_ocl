"""Tests for batch evaluation metric utility functions."""

from __future__ import annotations

import unittest

from aimai_ocl.statistics import (
    collect_violation_stats,
    success_from_status,
    summarize_records,
)
from aimai_ocl.schemas import (
    AuditEvent,
    AuditEventType,
    ConstraintCheck,
    ConstraintSeverity,
    EpisodeTrace,
    ViolationType,
)


class EvaluationMetricTests(unittest.TestCase):
    """Coverage for violation extraction and arm-level summary math."""

    def test_collect_violation_stats_counts_constraint_events_once(self) -> None:
        trace = EpisodeTrace(episode_id="ep-1", env_id="Task1_basic_price_negotiation-v0")
        checks = [
            ConstraintCheck(
                constraint_id="budget_cap",
                passed=False,
                severity=ConstraintSeverity.ERROR,
                violation_type=ViolationType.BUDGET_EXCEEDED,
            ),
            ConstraintCheck(
                constraint_id="seller_floor",
                passed=True,
                severity=ConstraintSeverity.INFO,
            ),
        ]
        # Count this event.
        trace.add_event(
            AuditEvent(
                event_type=AuditEventType.CONSTRAINT_EVALUATED,
                actor_id="seller",
                constraint_checks=checks,
            )
        )
        # Do not double-count this event.
        trace.add_event(
            AuditEvent(
                event_type=AuditEventType.ACTION_EXECUTED,
                actor_id="seller",
                constraint_checks=checks,
            )
        )
        trace.add_event(
            AuditEvent(
                event_type=AuditEventType.ESCALATION_TRIGGERED,
                actor_id="seller",
            )
        )

        stats = collect_violation_stats(trace, actor_id="seller")
        self.assertEqual(1, stats["failed_constraint_count"])
        self.assertTrue(stats["has_violation"])
        self.assertEqual(1, stats["escalation_count"])
        self.assertEqual(1, stats["violation_type_counts"][ViolationType.BUDGET_EXCEEDED.value])

    def test_summarize_records_aggregates_by_arm(self) -> None:
        records = [
            {
                "arm": "single",
                "success": 1,
                "has_violation": 0,
                "round": 5,
                "seller_reward": 20.0,
                "latency_sec": 1.0,
                "audit_events": 10,
                "failed_constraint_count": 0,
                "escalation_count": 0,
                "violation_type_counts": {},
            },
            {
                "arm": "single",
                "success": 0,
                "has_violation": 1,
                "round": 9,
                "seller_reward": 5.0,
                "latency_sec": 3.0,
                "audit_events": 12,
                "failed_constraint_count": 2,
                "escalation_count": 1,
                "violation_type_counts": {ViolationType.BUDGET_EXCEEDED.value: 2},
            },
            {
                "arm": "ocl_full",
                "success": 1,
                "has_violation": 0,
                "round": 6,
                "seller_reward": 22.0,
                "latency_sec": 2.0,
                "audit_events": 30,
                "failed_constraint_count": 0,
                "escalation_count": 0,
                "violation_type_counts": {},
            },
        ]
        summary = summarize_records(records)
        by_arm = {row["arm"]: row for row in summary}

        single = by_arm["single"]
        self.assertEqual(2, single["episodes"])
        self.assertAlmostEqual(0.5, single["success_rate"])
        self.assertAlmostEqual(0.5, single["violation_rate"])
        self.assertAlmostEqual(7.0, single["avg_round"])
        self.assertAlmostEqual(12.5, single["avg_seller_reward"])
        self.assertAlmostEqual(2.0, single["avg_latency_sec"])
        self.assertEqual(2, single["total_failed_constraints"])
        self.assertEqual(1, single["total_escalations"])

        ocl = by_arm["ocl_full"]
        self.assertEqual(1, ocl["episodes"])
        self.assertAlmostEqual(1.0, ocl["success_rate"])
        self.assertAlmostEqual(0.0, ocl["violation_rate"])

    def test_success_from_status(self) -> None:
        self.assertEqual(1, success_from_status("agreed"))
        self.assertEqual(1, success_from_status("Agreed"))
        self.assertEqual(0, success_from_status("timeout"))
        self.assertEqual(0, success_from_status(None))

    def test_collect_violation_stats_actor_id_match_is_case_insensitive(self) -> None:
        """Input: trace event actor id uses different case than filter.

        Expected output:
        - seller-side violation is still counted by normalized actor-id match
        """
        trace = EpisodeTrace(episode_id="ep-2", env_id="Task1_basic_price_negotiation-v0")
        trace.add_event(
            AuditEvent(
                event_type=AuditEventType.CONSTRAINT_EVALUATED,
                actor_id="Seller",
                constraint_checks=[
                    ConstraintCheck(
                        constraint_id="budget_cap",
                        passed=False,
                        severity=ConstraintSeverity.ERROR,
                        violation_type=ViolationType.BUDGET_EXCEEDED,
                    )
                ],
            )
        )
        stats = collect_violation_stats(trace, actor_id="seller")
        self.assertEqual(1, stats["failed_constraint_count"])
        self.assertTrue(stats["has_violation"])


if __name__ == "__main__":
    unittest.main()
