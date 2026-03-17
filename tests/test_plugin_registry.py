"""Tests for algorithm/protocol plugin registries.

中文翻译：用于 algorithm/protocol plugin registries 的测试。"""

from __future__ import annotations

import unittest

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
        self.assertEqual("gate_v1_default", bundle.gate_algorithm_id)
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
        self.assertEqual("gate_v2_barrier", bundle.gate_algorithm_id)
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
            gate_algorithm_id="gate_v2_barrier_strict",
            escalation_algorithm_id="escalation_v1_no_replan",
            audit_algorithm_id="audit_v1_off",
            attribution_algorithm_id="counterfactual_v1",
        )
        self.assertEqual("v1_default", bundle.bundle_id)
        self.assertEqual("role_v1_seller_only", bundle.role_algorithm_id)
        self.assertEqual("gate_v2_barrier_strict", bundle.gate_algorithm_id)
        self.assertEqual("escalation_v1_no_replan", bundle.escalation_algorithm_id)
        self.assertEqual("audit_v1_off", bundle.audit_algorithm_id)
        self.assertEqual("counterfactual_v1", bundle.attribution_algorithm_id)

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
