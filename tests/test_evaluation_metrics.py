"""Tests for batch evaluation metric utility functions.

中文翻译：用于批量评估指标工具函数的测试。"""

from __future__ import annotations

import unittest

from aimai_ocl.evaluation_metrics import (
    build_episode_metrics,
    collect_violation_stats,
    price_feasibility_from_final_info,
    success_from_status,
    summarize_records,
)
from aimai_ocl.schemas.audit import AuditEvent, AuditEventType, EpisodeTrace
from aimai_ocl.schemas.actions import ControlDecision, ExecutableAction
from aimai_ocl.schemas.constraints import ConstraintCheck, ConstraintSeverity, ViolationType


class EvaluationMetricTests(unittest.TestCase):
    """Coverage for violation extraction and arm-level summary math.

中文翻译：覆盖 violation 提取与实验臂汇总计算相关逻辑。"""

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
        # 中文：该事件应被计入统计。
        trace.add_event(
            AuditEvent(
                event_type=AuditEventType.CONSTRAINT_EVALUATED,
                actor_id="seller",
                constraint_checks=checks,
            )
        )
        # Do not double-count this event.
        # 中文：该事件不应重复计数。
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
        self.assertEqual(2, stats["total_constraint_check_count"])
        self.assertEqual(1, stats["passed_constraint_count"])
        self.assertEqual(1, stats["failed_constraint_count"])
        self.assertAlmostEqual(0.5, stats["constraint_satisfaction_rate"])
        self.assertTrue(stats["has_violation"])
        self.assertTrue(stats["transient_has_violation"])
        self.assertFalse(stats["executed_has_violation"])
        self.assertEqual(1, stats["escalation_count"])
        self.assertEqual(1, stats["violation_type_counts"][ViolationType.BUDGET_EXCEEDED.value])
        self.assertEqual({}, stats["executed_violation_type_counts"])

    def test_collect_violation_stats_splits_transient_from_executed(self) -> None:
        trace = EpisodeTrace(episode_id="ep-1b", env_id="Task1_basic_price_negotiation-v0")
        failed_check = ConstraintCheck(
            constraint_id="privacy_policy",
            passed=False,
            severity=ConstraintSeverity.CRITICAL,
            violation_type=ViolationType.POLICY_PRIVACY,
        )
        trace.add_event(
            AuditEvent(
                event_type=AuditEventType.CONSTRAINT_EVALUATED,
                actor_id="seller",
                constraint_checks=[failed_check],
            )
        )
        trace.add_event(
            AuditEvent(
                event_type=AuditEventType.ACTION_EXECUTED,
                actor_id="seller",
                executable_action=ExecutableAction(
                    actor_id="seller",
                    approved=True,
                    decision=ControlDecision.APPROVE,
                ),
                constraint_checks=[failed_check],
            )
        )

        stats = collect_violation_stats(trace, actor_id="seller")
        self.assertTrue(stats["transient_has_violation"])
        self.assertTrue(stats["executed_has_violation"])
        self.assertEqual(1, stats["executed_failed_constraint_count"])
        self.assertEqual(
            1,
            stats["executed_violation_type_counts"][ViolationType.POLICY_PRIVACY.value],
        )

    def test_summarize_records_aggregates_by_arm(self) -> None:
        records = [
            {
                "arm": "single",
                "success": 1,
                "has_violation": 0,
                "transient_has_violation": 0,
                "executed_has_violation": 0,
                "recovered_has_violation": 0,
                "unrecovered_has_violation": 0,
                "round": 5,
                "seller_reward": 20.0,
                "latency_sec": 1.0,
                "global_score": 8.0,
                "welfare": 8.0,
                "cost_adjusted_welfare": 1.6,
                "audit_events": 10,
                "total_constraint_check_count": 4,
                "passed_constraint_count": 4,
                "failed_constraint_count": 0,
                "executed_failed_constraint_count": 0,
                "constraint_satisfaction_rate": 1.0,
                "escalation_count": 0,
                "violation_type_counts": {},
                "executed_violation_type_counts": {},
            },
            {
                "arm": "single",
                "success": 0,
                "has_violation": 1,
                "transient_has_violation": 1,
                "executed_has_violation": 0,
                "recovered_has_violation": 0,
                "unrecovered_has_violation": 1,
                "round": 9,
                "seller_reward": 5.0,
                "latency_sec": 3.0,
                "global_score": 3.0,
                "welfare": 3.0,
                "cost_adjusted_welfare": 1.0 / 3.0,
                "audit_events": 12,
                "total_constraint_check_count": 4,
                "passed_constraint_count": 2,
                "failed_constraint_count": 2,
                "executed_failed_constraint_count": 0,
                "constraint_satisfaction_rate": 0.5,
                "escalation_count": 1,
                "violation_type_counts": {ViolationType.BUDGET_EXCEEDED.value: 2},
                "executed_violation_type_counts": {},
            },
            {
                "arm": "ocl_full",
                "success": 1,
                "has_violation": 0,
                "transient_has_violation": 0,
                "executed_has_violation": 0,
                "recovered_has_violation": 0,
                "unrecovered_has_violation": 0,
                "round": 6,
                "seller_reward": 22.0,
                "latency_sec": 2.0,
                "global_score": 9.0,
                "welfare": 9.0,
                "cost_adjusted_welfare": 1.5,
                "audit_events": 30,
                "total_constraint_check_count": 6,
                "passed_constraint_count": 6,
                "failed_constraint_count": 0,
                "executed_failed_constraint_count": 0,
                "constraint_satisfaction_rate": 1.0,
                "escalation_count": 0,
                "violation_type_counts": {},
                "executed_violation_type_counts": {},
            },
        ]
        summary = summarize_records(records)
        by_arm = {row["arm"]: row for row in summary}

        single = by_arm["single"]
        self.assertEqual(2, single["episodes"])
        self.assertAlmostEqual(0.5, single["success_rate"])
        self.assertAlmostEqual(0.5, single["violation_rate"])
        self.assertAlmostEqual(0.5, single["transient_violation_rate"])
        self.assertAlmostEqual(0.0, single["executed_violation_rate"])
        self.assertAlmostEqual(0.5, single["unrecovered_violation_rate"])
        self.assertAlmostEqual(7.0, single["avg_round"])
        self.assertAlmostEqual(12.5, single["avg_seller_reward"])
        self.assertAlmostEqual(2.0, single["avg_latency_sec"])
        self.assertAlmostEqual(5.5, single["avg_global_score"])
        self.assertAlmostEqual(5.5, single["avg_welfare"])
        self.assertAlmostEqual((1.6 + (1.0 / 3.0)) / 2.0, single["avg_cost_adjusted_welfare"])
        self.assertAlmostEqual(0.75, single["avg_constraint_satisfaction_rate"])
        self.assertAlmostEqual(4.0, single["avg_total_constraint_checks"])
        self.assertAlmostEqual(3.0, single["avg_passed_constraint_checks"])
        self.assertAlmostEqual(1.0, single["avg_failed_constraint_count"])
        self.assertAlmostEqual(0.0, single["avg_executed_failed_constraint_count"])
        self.assertAlmostEqual(0.5, single["avg_escalation_count"])
        self.assertEqual(2, single["total_failed_constraints"])
        self.assertEqual(0, single["total_executed_failed_constraints"])
        self.assertEqual(1, single["total_escalations"])

        ocl = by_arm["ocl_full"]
        self.assertEqual(1, ocl["episodes"])
        self.assertAlmostEqual(1.0, ocl["success_rate"])
        self.assertAlmostEqual(1.0, ocl["feasibility_rate"])
        self.assertAlmostEqual(0.0, ocl["violation_rate"])
        self.assertAlmostEqual(0.0, ocl["transient_violation_rate"])
        self.assertAlmostEqual(0.0, ocl["executed_violation_rate"])
        self.assertAlmostEqual(1.0, ocl["avg_constraint_satisfaction_rate"])
        self.assertAlmostEqual(9.0, ocl["avg_welfare"])

    def test_success_from_status(self) -> None:
        self.assertEqual(1, success_from_status("agreed"))
        self.assertEqual(1, success_from_status("Agreed"))
        self.assertEqual(0, success_from_status("timeout"))
        self.assertEqual(0, success_from_status(None))

    def test_collect_violation_stats_actor_id_match_is_case_insensitive(self) -> None:
        """Input: trace event actor id uses different case than filter.

        Expected output:
        - seller-side violation is still counted by normalized actor-id match
        

        中文翻译：输入：trace 事件中的 actor id 与过滤条件大小写不同。"""
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

    def test_build_episode_metrics_uses_global_score_and_round_cost_proxy(self) -> None:
        """Input: trace without explicit constraint events and agreed final info.

        Expected output:
        - cost-adjusted welfare uses global_score / round
        - zero explicit violations yields full constraint satisfaction

        中文翻译：输入：无显式约束事件但有终局结果的 trace。"""
        trace = EpisodeTrace(episode_id="ep-3", env_id="Task1_basic_price_negotiation-v0")
        metrics = build_episode_metrics(
            trace=trace,
            final_info={
                "status": "agreed",
                "round": 5,
                "agreed_price": 118.0,
                "seller_reward": 4.0,
                "buyer_reward": 1.0,
                "global_score": 10.0,
                "seller_score": 0.5,
                "buyer_score": 0.6,
            },
            latency_sec=1.25,
            buyer_max_price=120.0,
            seller_min_price=90.0,
        )
        self.assertEqual(1, metrics["success"])
        self.assertEqual(1, metrics["feasibility"])
        self.assertEqual(1, metrics["price_feasible"])
        self.assertEqual(1, metrics["strict_success"])
        self.assertAlmostEqual(10.0, metrics["welfare"])
        self.assertAlmostEqual(2.0, metrics["cost_adjusted_welfare"])
        self.assertEqual(0, metrics["failed_constraint_count"])
        self.assertEqual(0, metrics["executed_failed_constraint_count"])
        self.assertEqual(0, metrics["transient_has_violation"])
        self.assertEqual(0, metrics["executed_has_violation"])
        self.assertEqual(0, metrics["unrecovered_has_violation"])
        self.assertEqual(0, metrics["total_constraint_check_count"])
        self.assertAlmostEqual(1.0, metrics["constraint_satisfaction_rate"])

    def test_build_episode_metrics_marks_recovered_vs_unrecovered(self) -> None:
        trace = EpisodeTrace(episode_id="ep-4", env_id="Task1_basic_price_negotiation-v0")
        failed_check = ConstraintCheck(
            constraint_id="budget_cap",
            passed=False,
            severity=ConstraintSeverity.ERROR,
            violation_type=ViolationType.BUDGET_EXCEEDED,
        )
        trace.add_event(
            AuditEvent(
                event_type=AuditEventType.CONSTRAINT_EVALUATED,
                actor_id="seller",
                constraint_checks=[failed_check],
            )
        )
        metrics = build_episode_metrics(
            trace=trace,
            final_info={
                "status": "agreed",
                "round": 3,
                "agreed_price": 100.0,
                "global_score": 9.0,
            },
            latency_sec=0.5,
            buyer_max_price=120.0,
            seller_min_price=90.0,
        )
        self.assertEqual(1, metrics["transient_has_violation"])
        self.assertEqual(0, metrics["executed_has_violation"])
        self.assertEqual(1, metrics["recovered_has_violation"])
        self.assertEqual(0, metrics["unrecovered_has_violation"])

    def test_price_feasibility_is_stricter_than_agenticpay_success(self) -> None:
        """Input: AgenticPay agreed deal below seller floor.

        Expected output:
        - native success remains true
        - strict price feasibility is false
        - strict_success is false

        中文翻译：输入：AgenticPay agreed 但成交价低于 seller floor。"""
        final_info = {
            "status": "agreed",
            "round": 5,
            "agreed_price": 114.5,
            "global_score": -0.59,
        }
        self.assertEqual(
            0,
            price_feasibility_from_final_info(
                final_info=final_info,
                buyer_max_price=120.0,
                seller_min_price=115.0,
            ),
        )
        metrics = build_episode_metrics(
            trace=EpisodeTrace(
                episode_id="ep-infeasible",
                env_id="Task1_basic_price_negotiation-v0",
            ),
            final_info=final_info,
            latency_sec=0.5,
            buyer_max_price=120.0,
            seller_min_price=115.0,
        )
        self.assertEqual(1, metrics["success"])
        self.assertEqual(0, metrics["feasibility"])
        self.assertEqual(0, metrics["price_feasible"])
        self.assertEqual(0, metrics["strict_success"])


if __name__ == "__main__":
    unittest.main()
