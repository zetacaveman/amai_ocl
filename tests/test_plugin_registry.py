"""Tests for algorithm/protocol plugin registries.

中文翻译：用于 algorithm/protocol plugin registries 的测试。"""

from __future__ import annotations

import unittest

from aimai_ocl.evaluation_metrics import summarize_records
from aimai_ocl.experiment_config import ARM_REGISTRY
from aimai_ocl.plugin_registry import (
    ALGORITHM_BUNDLE_REGISTRY,
    ATTRIBUTION_ALGORITHM_REGISTRY,
    AUDIT_ALGORITHM_REGISTRY,
    ESCALATION_ALGORITHM_REGISTRY,
    EXPERIMENT_PROTOCOL_REGISTRY,
    GATE_ALGORITHM_REGISTRY,
    ROLE_ALGORITHM_REGISTRY,
    compose_algorithm_bundle,
    build_main_result_artifact,
    resolve_attribution_algorithm,
    resolve_algorithm_bundle,
    resolve_audit_algorithm_factory,
    resolve_escalation_algorithm_factory,
    resolve_experiment_protocol,
    resolve_gate_algorithm_factory,
    resolve_role_algorithm_factory,
)


class PluginRegistryTests(unittest.TestCase):
    """Coverage for step-9/10 pluggable bundle/protocol contracts.

中文翻译：step-9/10 pluggable bundle/protocol contracts 的覆盖测试。"""

    def test_default_algorithm_bundle_is_resolvable(self) -> None:
        """Input: default bundle id.

        Expected output:
        - resolver returns bundle with expected id
        - role/gate/escalation factories are callable
        - attribution interface functions are callable
        

        中文翻译：输入：default bundle id。"""
        bundle = resolve_algorithm_bundle("v1_default")
        self.assertEqual("v1_default", bundle.bundle_id)
        self.assertEqual("role_v1_rule", bundle.role_algorithm_id)
        self.assertEqual("gate_v3_tau_controlled", bundle.gate_algorithm_id)
        self.assertEqual("escalation_v1_default", bundle.escalation_algorithm_id)
        self.assertEqual("audit_v1_full", bundle.audit_algorithm_id)
        self.assertEqual("shapley_v1_exact", bundle.attribution_algorithm_id)
        self.assertTrue(callable(bundle.make_role_algorithm))
        self.assertTrue(callable(bundle.make_gate_algorithm))
        self.assertTrue(callable(bundle.make_escalation_algorithm))
        self.assertTrue(callable(bundle.make_audit_algorithm))
        self.assertTrue(callable(bundle.run_episode_fn))
        self.assertTrue(callable(bundle.compute_V_fn))
        self.assertTrue(callable(bundle.fallback_policy_fn))
        self.assertTrue(callable(bundle.compute_shapley_fn))

    def test_research_bundle_is_resolvable(self) -> None:
        """Input: v2 research bundle id.

        Expected output:
        - role/gate/attribution point to v2 algorithm bodies
        

        中文翻译：输入：v2 research bundle id。"""
        bundle = resolve_algorithm_bundle("v2_research")
        self.assertEqual("role_v2_state_machine", bundle.role_algorithm_id)
        self.assertEqual("gate_v3_tau_controlled", bundle.gate_algorithm_id)
        self.assertEqual("counterfactual_v1", bundle.attribution_algorithm_id)

    def test_default_protocol_bundle_is_resolvable(self) -> None:
        """Input: default protocol id.

        Expected output:
        - resolver returns protocol with expected id
        - protocol hooks are available and return dict payload
        

        中文翻译：输入：default protocol id。"""
        protocol = resolve_experiment_protocol("offline_v1")
        self.assertEqual("offline_v1", protocol.protocol_id)
        self.assertIsNotNone(protocol.run_main_fn)
        self.assertIsNotNone(protocol.run_ablation_fn)
        self.assertIsNotNone(protocol.run_adversarial_fn)
        self.assertIsNotNone(protocol.run_repeated_fn)
        self.assertIsNotNone(protocol.run_roi_fn)
        payload = protocol.run_main_fn(foo=1) if protocol.run_main_fn else {}
        self.assertIsInstance(payload, dict)
        self.assertEqual(True, payload.get("implemented"))
        self.assertIn("summary_by_arm", payload)

    def test_all_registry_keys_resolve(self) -> None:
        """Input: every key in both registries.

        Expected output: each key resolves without error.
        

        中文翻译：输入：every key in both registries。"""
        for bundle_id in ALGORITHM_BUNDLE_REGISTRY:
            resolved = resolve_algorithm_bundle(bundle_id)
            self.assertEqual(bundle_id, resolved.bundle_id)
        for protocol_id in EXPERIMENT_PROTOCOL_REGISTRY:
            resolved = resolve_experiment_protocol(protocol_id)
            self.assertEqual(protocol_id, resolved.protocol_id)
        for role_id in ROLE_ALGORITHM_REGISTRY:
            self.assertTrue(callable(resolve_role_algorithm_factory(role_id)))
        for gate_id in GATE_ALGORITHM_REGISTRY:
            self.assertTrue(callable(resolve_gate_algorithm_factory(gate_id)))
        for escalation_id in ESCALATION_ALGORITHM_REGISTRY:
            self.assertTrue(callable(resolve_escalation_algorithm_factory(escalation_id)))
        for audit_id in AUDIT_ALGORITHM_REGISTRY:
            self.assertTrue(callable(resolve_audit_algorithm_factory(audit_id)))
        for attr_id in ATTRIBUTION_ALGORITHM_REGISTRY:
            module = resolve_attribution_algorithm(attr_id)
            self.assertEqual(attr_id, module.module_id)

    def test_arm_defaults_point_to_registered_plugins(self) -> None:
        """Input: all arm configs from ``ARM_REGISTRY``.

        Expected output: configured bundle/protocol ids are valid registry keys.
        

        中文翻译：输入：all arm configs from ``ARM_REGISTRY``。"""
        for arm in ARM_REGISTRY.values():
            self.assertIn(arm.algorithm_bundle_id, ALGORITHM_BUNDLE_REGISTRY)
            self.assertIn(arm.experiment_protocol_id, EXPERIMENT_PROTOCOL_REGISTRY)

    def test_compose_algorithm_bundle_applies_component_overrides(self) -> None:
        """Input: bundle id with component override ids.

        Expected output:
        - composed bundle keeps same bundle id
        - component ids match the requested override values
        

        中文翻译：输入：bundle id with component override ids。"""
        bundle = compose_algorithm_bundle(
            bundle_id="v1_default",
            role_algorithm_id="role_v1_seller_only",
            gate_algorithm_id="gate_v0_off",
            gate_tau=0.8,
            escalation_algorithm_id="escalation_v0_off",
            audit_algorithm_id="audit_v1_off",
            attribution_algorithm_id="counterfactual_v1",
        )
        self.assertEqual("v1_default", bundle.bundle_id)
        self.assertEqual("role_v1_seller_only", bundle.role_algorithm_id)
        self.assertEqual("gate_v0_off", bundle.gate_algorithm_id)
        self.assertAlmostEqual(0.8, bundle.gate_tau)
        self.assertEqual("legacy", bundle.gate_runtime_config["gate_family"])
        self.assertEqual(False, bundle.gate_runtime_config["tau_applied"])
        self.assertIsNone(bundle.gate_runtime_config["rewrite_threshold"])
        self.assertIsNone(bundle.gate_runtime_config["block_threshold"])
        self.assertEqual("escalation_v0_off", bundle.escalation_algorithm_id)
        self.assertEqual("audit_v1_off", bundle.audit_algorithm_id)
        self.assertEqual("counterfactual_v1", bundle.attribution_algorithm_id)

    def test_compose_algorithm_bundle_keeps_barrier_tau_runtime(self) -> None:
        """Input: bundle id with barrier gate override and tau.

        Expected output:
        - barrier runtime config records applied tau and lowered thresholds

        中文翻译：输入：带 barrier gate override 与 tau 的 bundle。"""
        bundle = compose_algorithm_bundle(
            bundle_id="v1_default",
            gate_algorithm_id="gate_v2_barrier_strict",
            gate_tau=0.8,
        )
        self.assertEqual("gate_v2_barrier_strict", bundle.gate_algorithm_id)
        self.assertAlmostEqual(0.8, bundle.gate_tau)
        self.assertEqual("barrier", bundle.gate_runtime_config["gate_family"])
        self.assertEqual(True, bundle.gate_runtime_config["tau_applied"])
        self.assertLess(bundle.gate_runtime_config["rewrite_threshold"], 0.45)
        self.assertLess(bundle.gate_runtime_config["block_threshold"], 0.75)

    def test_compose_algorithm_bundle_resolves_tau_controlled_runtime(self) -> None:
        """Input: default bundle with explicit tau under tau-controlled gate.

        Expected output:
        - runtime config exposes tau-controlled family
        - runtime thresholds and priors are recorded

        中文翻译：输入：tau-controlled gate 下的默认 bundle 与显式 tau。"""
        bundle = compose_algorithm_bundle(
            bundle_id="v1_default",
            gate_tau=0.8,
        )
        self.assertEqual("gate_v3_tau_controlled", bundle.gate_algorithm_id)
        self.assertAlmostEqual(0.8, bundle.gate_tau)
        self.assertEqual("tau_controlled", bundle.gate_runtime_config["gate_family"])
        self.assertIn("intent_risk_priors", bundle.gate_runtime_config)
        priors = bundle.gate_runtime_config["intent_risk_priors"]
        self.assertGreater(priors["accept_deal"], priors["negotiate_price"])

    def test_main_protocol_orients_lower_is_better_metrics_for_one_sided_pvalue(self) -> None:
        """Input: paired records where OCL always lowers violation and rounds.

        Expected output:
        - raw deltas for lower-is-better metrics stay negative
        - oriented improvement becomes positive
        - one-sided p-value tests "target better" after orientation

        中文翻译：输入：OCL 在 paired records 中始终降低 violation 与轮数。"""
        protocol = resolve_experiment_protocol("offline_v1")
        records: list[dict[str, object]] = []
        for idx, seed in enumerate((41, 42, 43)):
            records.extend(
                [
                    {
                        "arm": "single",
                        "episode_index": idx,
                        "seed": seed,
                        "success": 1,
                        "feasibility": 1,
                        "has_violation": 1,
                        "constraint_satisfaction_rate": 0.0,
                        "round": 8,
                        "seller_reward": 10.0,
                        "global_score": 2.0,
                        "welfare": 2.0,
                        "cost_adjusted_welfare": 0.25,
                        "escalation_count": 1,
                        "latency_sec": 2.0,
                    },
                    {
                        "arm": "ocl_full",
                        "episode_index": idx,
                        "seed": seed,
                        "success": 1,
                        "feasibility": 1,
                        "has_violation": 0,
                        "constraint_satisfaction_rate": 1.0,
                        "round": 4,
                        "seller_reward": 12.0,
                        "global_score": 5.0,
                        "welfare": 5.0,
                        "cost_adjusted_welfare": 1.25,
                        "escalation_count": 0,
                        "latency_sec": 1.0,
                    },
                ]
            )

        summaries = summarize_records(records)
        payload = protocol.run_main_fn(
            records=records,
            summaries=summaries,
            plan={},
            bootstrap_samples=32,
            permutation_samples=128,
            seed=7,
        )

        violation_stats = payload["paired_statistics"]["ocl_vs_single"]["metrics"]["has_violation"]
        self.assertAlmostEqual(-1.0, violation_stats["mean_delta"])
        self.assertAlmostEqual(1.0, violation_stats["mean_improvement"])
        self.assertEqual(-1, violation_stats["metric_direction"])
        self.assertLess(violation_stats["sign_flip_pvalues"]["p_one_sided"], 0.5)

    def test_build_main_result_artifact_emits_canonical_rows(self) -> None:
        """Input: valid offline main payload with single and ocl_full arms.

        Expected output:
        - canonical main-result artifact is available
        - rows expose arm values, deltas, CIs, and p-values per metric

        中文翻译：输入：包含 single 与 ocl_full 的有效 main payload。"""
        protocol = resolve_experiment_protocol("offline_v1")
        records = [
            {
                "arm": "single",
                "episode_index": 0,
                "seed": 42,
                "success": 0,
                "feasibility": 0,
                "has_violation": 1,
                "constraint_satisfaction_rate": 0.5,
                "round": 8,
                "seller_reward": 5.0,
                "global_score": 1.0,
                "welfare": 1.0,
                "cost_adjusted_welfare": 0.125,
                "escalation_count": 1,
                "latency_sec": 2.5,
            },
            {
                "arm": "ocl_full",
                "episode_index": 0,
                "seed": 42,
                "success": 1,
                "feasibility": 1,
                "has_violation": 0,
                "constraint_satisfaction_rate": 1.0,
                "round": 4,
                "seller_reward": 10.0,
                "global_score": 4.0,
                "welfare": 4.0,
                "cost_adjusted_welfare": 1.0,
                "escalation_count": 0,
                "latency_sec": 1.0,
            },
        ]
        payload = protocol.run_main_fn(
            records=records,
            summaries=summarize_records(records),
            plan={"arms": ["single", "ocl_full"]},
            bootstrap_samples=16,
            permutation_samples=16,
            seed=11,
        )
        artifact = build_main_result_artifact(payload)
        self.assertTrue(artifact["available"])
        self.assertEqual("single", artifact["baseline_arm"])
        self.assertEqual("ocl_full", artifact["target_arm"])
        by_metric = {row["metric_key"]: row for row in artifact["rows"]}
        self.assertIn("success", by_metric)
        self.assertIn("has_violation", by_metric)
        self.assertEqual(0.0, by_metric["success"]["single_value"])
        self.assertEqual(1.0, by_metric["success"]["ocl_full_value"])
        self.assertEqual(False, by_metric["has_violation"]["higher_is_better"])
        self.assertLess(by_metric["has_violation"]["delta_ocl_minus_single"], 0.0)
        self.assertGreater(by_metric["has_violation"]["improvement_ocl_vs_single"], 0.0)
        self.assertIn("p_one_sided", by_metric["has_violation"])

    def test_unknown_bundle_raises_value_error(self) -> None:
        """Input: unknown bundle id.

        Expected output: resolver raises ValueError.
        

        中文翻译：输入：unknown bundle id。"""
        with self.assertRaises(ValueError):
            resolve_algorithm_bundle("does_not_exist")

    def test_unknown_protocol_raises_value_error(self) -> None:
        """Input: unknown protocol id.

        Expected output: resolver raises ValueError.
        

        中文翻译：输入：unknown protocol id。"""
        with self.assertRaises(ValueError):
            resolve_experiment_protocol("does_not_exist")

    def test_unknown_component_algorithm_raises_value_error(self) -> None:
        """Input: unknown role/gate/escalation/attribution ids.

        Expected output: component resolvers raise ValueError.
        

        中文翻译：输入：unknown role/gate/escalation/attribution ids。"""
        with self.assertRaises(ValueError):
            resolve_role_algorithm_factory("bad_role")
        with self.assertRaises(ValueError):
            resolve_gate_algorithm_factory("bad_gate")
        with self.assertRaises(ValueError):
            resolve_escalation_algorithm_factory("bad_escalation")
        with self.assertRaises(ValueError):
            resolve_audit_algorithm_factory("bad_audit")
        with self.assertRaises(ValueError):
            resolve_attribution_algorithm("bad_attr")


if __name__ == "__main__":
    unittest.main()
