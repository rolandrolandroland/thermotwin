from contextlib import redirect_stderr
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import thermotwin.reports.operating_decision_prospective_development as phase_d_cli
import thermotwin.studies.operating_decision_prospective_phase_d as phase_d
import thermotwin.studies.operating_decision_prospective_phase_d_resources as phase_d_resources
from thermotwin.studies.operating_decision_prospective_pilot import (
    PROSPECTIVE_PILOT_PARTITION,
    _expected_corrected_stream_manifest,
)


class ProspectivePhaseDProtocolTests(unittest.TestCase):
    def setUp(self):
        self.payload = phase_d.prospective_phase_d_protocol_payload(
            source_revision="a" * 40,
            source_manifest_digest="b" * 64,
        )

    def test_protocol_digest_is_stable_and_tampering_is_rejected(self):
        self.assertEqual(
            phase_d.prospective_phase_d_protocol_digest(
                source_revision="a" * 40,
                source_manifest_digest="b" * 64,
            ),
            "35719eb7289a0e7cc49b4ef8623d8c0bd5956112c529f8ffc2f08c72ccb4f1c7",
        )
        self.assertEqual(
            phase_d.validate_prospective_phase_d_protocol(self.payload),
            self.payload,
        )
        tampered = deepcopy(self.payload)
        tampered["rule_grid"]["general_stopping_clearance_kelvin"][1] = 0.04
        with self.assertRaisesRegex(ValueError, "committed design"):
            phase_d.validate_prospective_phase_d_protocol(tampered)

    def test_only_tuning_is_authorized(self):
        partitions = self.payload["partitions"]
        self.assertEqual(
            [(item["block_count"], item["case_count"]) for item in partitions],
            [(20, 60), (10, 30)],
        )
        self.assertTrue(partitions[0]["authorized_by_this_protocol"])
        self.assertFalse(partitions[1]["authorized_by_this_protocol"])
        authorization = self.payload["authorization"]
        self.assertEqual(authorization["next_partition"], "p1_development_tuning")
        self.assertFalse(authorization["development_internal_check_authorized_now"])
        self.assertFalse(authorization["independent_calibration_authorized_now"])
        self.assertFalse(authorization["reserved_evaluation_authorized_now"])

    def test_offset_estimator_and_fallback_are_complete(self):
        estimator = self.payload["offset_estimator"]
        self.assertEqual(estimator["block_score_count"], 20)
        self.assertEqual(estimator["nearest_rank"], 18)
        self.assertEqual(estimator["round_up_to_kelvin"], 0.001)
        self.assertEqual(estimator["minimum_separate_support_blocks"], 20)
        self.assertEqual(estimator["actions"], list(phase_d.POLICY_NAMES))
        self.assertFalse(estimator["family_or_candidate_count_specific_offsets"])
        self.assertEqual(
            estimator["infinite_offset_rule"],
            "stop_primary_phase_d_as_infeasible",
        )

    def test_runtime_and_whole_workflow_budget_are_explicit(self):
        runtime = self.payload["runtime_identity"]
        self.assertEqual(runtime["python_version"], "3.10.12")
        requirements = Path(__file__).resolve().parents[1] / runtime["requirements_path"]
        self.assertEqual(
            hashlib.sha256(requirements.read_bytes()).hexdigest(),
            runtime["requirements_sha256"],
        )
        budget = self.payload["compute_budget"]
        self.assertIn("worker_only_peak_rss_estimate_bytes", budget)
        self.assertNotIn("concurrent_peak_rss_bytes", budget)
        self.assertTrue(budget["whole_workflow_measurement_required_before_tuning"])
        self.assertLess(
            budget["phase_d_process_tree_limit_bytes"],
            budget["machine_physical_memory_bytes"],
        )

    def test_rule_grid_has_exactly_eighty_one_unique_rules(self):
        grid = self.payload["rule_grid"]
        axes = (
            grid["general_stopping_clearance_kelvin"],
            grid["single_candidate_stopping_clearance_kelvin"],
            grid["minimum_expected_reduction_kelvin"],
            grid["minimum_utility_per_normalized_cost"],
        )
        combinations = {
            (a, b, c, d)
            for a in axes[0]
            for b in axes[1]
            for c in axes[2]
            for d in axes[3]
        }
        self.assertEqual(len(combinations), 81)
        self.assertEqual(grid["cartesian_rule_count"], 81)
        self.assertEqual(
            grid["selection_objective"]["loss"]["false_approval_weight"],
            100.0,
        )

    def test_cost_and_sensor_maps_are_exact(self):
        maps = self.payload["measurement_maps"]
        self.assertEqual(len(maps["cost_scenarios"]), 12)
        self.assertEqual(
            sum(
                item["name"] == maps["primary_cost_scenario"]
                for item in maps["cost_scenarios"]
            ),
            1,
        )
        sensors = {item["name"]: item for item in maps["sensor_scenarios"]}
        self.assertEqual(set(sensors), {
            "nominal",
            "high_voltage_noise",
            "high_face_noise",
            "high_probe_loading",
        })
        self.assertEqual(sensors["high_voltage_noise"]["voltage_noise"], 0.004)
        self.assertEqual(
            sensors["high_face_noise"]["face_temperature_noise"],
            0.04,
        )
        self.assertEqual(
            sensors["high_probe_loading"]["probe_capacitance"],
            10.0,
        )
        self.assertEqual(sum(item["primary"] for item in sensors.values()), 1)

    def test_draw_sensitivity_subset_and_gate_are_frozen(self):
        sensitivity = self.payload["draw_sensitivity"]
        self.assertEqual(sensitivity["tuning_block_indices"], [0, 5, 10, 15])
        self.assertEqual(sensitivity["case_count"], 12)
        self.assertEqual(sensitivity["candidate_draw_count"], 16)
        self.assertEqual(sensitivity["reference_draw_count"], 32)
        self.assertEqual(sensitivity["minimum_agreement_count"], 11)
        self.assertEqual(sensitivity["maximum_normalized_utility_regret"], 0.05)
        self.assertEqual(sensitivity["contingency_draw_count"], 64)

    def test_compute_budget_is_additive(self):
        budget = self.payload["compute_budget"]
        self.assertAlmostEqual(
            budget["planned_wall_seconds_before_draw_continuation"],
            budget["nominal_tuning_wall_seconds"]
            + budget["nominal_internal_check_wall_seconds"]
            + budget["three_sensor_stress_wall_seconds"],
        )
        self.assertAlmostEqual(
            budget["planned_cpu_seconds_before_draw_continuation"],
            budget["nominal_tuning_cpu_seconds"]
            + budget["nominal_internal_check_cpu_seconds"]
            + budget["three_sensor_stress_cpu_seconds"],
        )
        self.assertEqual(
            budget["planned_archive_bytes_before_draw_continuation"],
            budget["nominal_tuning_archive_bytes"]
            + budget["nominal_internal_check_archive_bytes"]
            + budget["three_sensor_stress_archive_bytes"],
        )

    def test_numerical_source_allowlist_is_sorted_complete_and_present(self):
        paths = phase_d.PHASE_D_NUMERICAL_SOURCE_PATHS
        self.assertEqual(paths, tuple(sorted(set(paths))))
        self.assertIn(
            "thermotwin/studies/operating_decision_prospective_phase_d.py",
            paths,
        )
        self.assertIn(
            "thermotwin/reports/operating_decision_prospective_development.py",
            paths,
        )
        self.assertIn(
            "thermotwin/studies/operating_decision_prospective_phase_d_resources.py",
            paths,
        )
        root = Path(__file__).resolve().parents[1]
        self.assertEqual([value for value in paths if not (root / value).is_file()], [])

    def test_disposable_stream_namespace_differs_from_closed_p4(self):
        rehearsal = _expected_corrected_stream_manifest(
            0,
            "parent pilot",
            campaign=phase_d.PROSPECTIVE_CAMPAIGN,
            partition=phase_d.PHASE_D_REHEARSAL_PARTITION,
        )
        closed = _expected_corrected_stream_manifest(0, "parent pilot")
        self.assertTrue(rehearsal)
        self.assertTrue(
            all(
                item["key"]["partition"] == phase_d.PHASE_D_REHEARSAL_PARTITION
                for item in rehearsal
            )
        )
        self.assertTrue(
            all(
                item["key"]["partition"] == PROSPECTIVE_PILOT_PARTITION
                for item in closed
            )
        )
        self.assertNotEqual(
            {item["seed"] for item in rehearsal},
            {item["seed"] for item in closed},
        )


class ProspectivePhaseDCliTests(unittest.TestCase):
    def test_cli_preflight_rejects_dirty_head(self):
        with patch("subprocess.run", return_value=Mock(stdout=" M changed.py\n")):
            with self.assertRaisesRegex(ValueError, "clean committed"):
                phase_d_cli._committed_source_revision(Path("."))

    def test_cli_rejects_unrelated_repository_root(self):
        with self.assertRaisesRegex(ValueError, "imported source tree"):
            phase_d_cli._require_project_root(Path("/tmp"))

    def test_cli_does_not_expose_a_scientific_partition_execution_mode(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            phase_d_cli.main(("--execute-development-tuning",))

    def test_replacement_rehearsal_requires_resource_output(self):
        with (
            patch.object(phase_d_cli, "_committed_source_revision", return_value="a" * 40),
            patch.object(phase_d_cli, "_require_project_root", return_value=Path(".")),
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            phase_d_cli.main(("--execute-disposable-rehearsal", "--json", "/tmp/x.json"))


class ProspectivePhaseDOffsetAndDecisionTests(unittest.TestCase):
    @staticmethod
    def _scores(overrides=None):
        values = {name: [0.0] * 20 for name in phase_d.POLICY_NAMES}
        if overrides:
            values.update(overrides)
        return values

    def test_finite_offsets_round_up_and_zero_behavior_is_preserved(self):
        scores = self._scores(
            {phase_d.POLICY_NAMES[1]: [0.0] * 17 + [0.00101, 0.5, 0.6]}
        )
        result = phase_d.estimate_phase_d_development_offsets(scores)
        self.assertEqual(result["status"], "finite_offsets_ready")
        values = result["development_offsets"]["values"]
        self.assertEqual(values[phase_d.POLICY_NAMES[0]], 0.0)
        self.assertEqual(values[phase_d.POLICY_NAMES[1]], 0.002)
        baseline = phase_d.phase_d_development_adjusted_decision([0.0, 0.2], 0.0)
        self.assertEqual(baseline["raw_decision"], baseline["development_adjusted_decision"])

    def test_any_infinite_required_offset_stops_phase_d(self):
        for actions in (
            (phase_d.POLICY_NAMES[0],),
            (phase_d.POLICY_NAMES[1],),
            (phase_d.POLICY_NAMES[2],),
            (phase_d.POLICY_NAMES[3],),
            (phase_d.POLICY_NAMES[0], phase_d.POLICY_NAMES[2]),
        ):
            overrides = {
                name: [0.0] * 17 + [math.inf] * 3 for name in actions
            }
            result = phase_d.estimate_phase_d_development_offsets(
                self._scores(overrides)
            )
            self.assertEqual(result["status"], "phase_d_infeasible")
            self.assertIsNone(result["development_offsets"])
            self.assertFalse(result["tuning_winner_selection_authorized"])
            self.assertEqual(set(result["infinite_actions"]), set(actions))

    def test_eighteenth_order_statistic_is_the_infinite_boundary(self):
        policy = phase_d.POLICY_NAMES[0]
        finite = phase_d.estimate_phase_d_development_offsets(
            self._scores({policy: [0.1] * 18 + [math.inf] * 2})
        )
        stopped = phase_d.estimate_phase_d_development_offsets(
            self._scores({policy: [0.1] * 17 + [math.inf] * 3})
        )
        self.assertEqual(finite["status"], "finite_offsets_ready")
        self.assertEqual(stopped["status"], "phase_d_infeasible")

    def test_adjusted_decision_boundaries_and_failures(self):
        approval = phase_d.phase_d_development_adjusted_decision([0.10, 0.20], 0.15)
        rejection = phase_d.phase_d_development_adjusted_decision([-0.20, -0.10], 0.15)
        unchanged = phase_d.phase_d_development_adjusted_decision([1.0, 2.0], 0.1)
        exact_approve = phase_d.phase_d_development_adjusted_decision([0.0, 0.2], 0.0)
        exact_abstain = phase_d.phase_d_development_adjusted_decision([-0.2, 0.0], 0.0)
        exact_reject = phase_d.phase_d_development_adjusted_decision([-0.2, -1e-12], 0.0)
        self.assertEqual(approval["raw_decision"], "approve")
        self.assertEqual(approval["development_adjusted_decision"], "insufficient_evidence")
        self.assertEqual(rejection["raw_decision"], "reject")
        self.assertEqual(rejection["development_adjusted_decision"], "insufficient_evidence")
        self.assertEqual(unchanged["development_adjusted_decision"], "approve")
        self.assertEqual(exact_approve["development_adjusted_decision"], "approve")
        self.assertEqual(exact_abstain["development_adjusted_decision"], "insufficient_evidence")
        self.assertEqual(exact_reject["development_adjusted_decision"], "reject")

        for record in (
            phase_d.phase_d_development_adjusted_decision(
                [1.0, 2.0], 0.1, verification_succeeded=False
            ),
            phase_d.phase_d_development_adjusted_decision(None, 0.1),
            phase_d.phase_d_development_adjusted_decision([0.0, math.inf], 0.1),
            phase_d.phase_d_development_adjusted_decision(
                [1.0, 2.0], 0.1, failure_reason="fit_failure"
            ),
        ):
            self.assertEqual(
                record["development_adjusted_decision"], "insufficient_evidence"
            )

    def test_all_eighty_one_objective_rows_match_independent_fixture(self):
        decisions = (
            phase_d.phase_d_development_adjusted_decision([0.2, 0.4], 0.0),
            phase_d.phase_d_development_adjusted_decision([-0.4, -0.2], 0.0),
            phase_d.phase_d_development_adjusted_decision([-0.1, 0.1], 0.0),
        )
        margins = (0.3, -0.3, 0.2)
        cases = [
            {
                "block": 0,
                "truth_family": family,
                "adjusted_decision": decision,
                "true_margin": margin,
                "added_run_count": 1,
                "added_sensor_count": 0,
                "normalized_incremental_energy": 0.5,
            }
            for family, decision, margin in zip(
                phase_d.STAGE3_TRUTH_CONDITIONS, decisions, margins
            )
        ]
        expected_case_losses = (0.15, 0.15, 1.15)
        expected_mean = sum(expected_case_losses) / 3.0
        rows = [
            phase_d.evaluate_phase_d_rule_objective(rule, cases)
            for rule in phase_d.phase_d_rule_grid()
        ]
        self.assertEqual(len(rows), 81)
        self.assertEqual(len({tuple(row["rule"]) for row in rows}), 81)
        for row in rows:
            self.assertAlmostEqual(row["mean_block_loss"], expected_mean)
            self.assertEqual(row["false_approvals"], 0)
            self.assertEqual(row["false_rejections"], 0)
            self.assertEqual(row["definitive_decisions"], 2)


class ProspectivePhaseDScientificIdentityTests(unittest.TestCase):
    def _payload(self):
        return {
            "source_revision": "a" * 40,
            "protocol_digest": "b" * 64,
            "runtime_identity": phase_d.phase_d_runtime_identity(),
            "block_result": {
                "block": 0,
                "cases": [{"observation": 1.25, "timing": {"wall_seconds": 2.0}}],
                "corrected_random_stream_manifest": [{"seed": 10}],
                "corrected_random_stream_audit": {"clean": True},
                "timing": {"block_wall_seconds": 3.0, "peak_rss_bytes": 100},
            },
            "performance": {"wall_seconds": 4.0, "cpu_seconds": 3.0, "peak_rss_bytes": 100},
        }

    def test_performance_changes_only_archive_identity(self):
        payload = self._payload()
        replay = {"block_evidence_digest": "c" * 64}
        with patch.object(phase_d, "_replay_summary", return_value=replay):
            scientific_before = phase_d.phase_d_rehearsal_scientific_digest(payload)
            sealed_before, _ = phase_d._archive_with_content_seal(payload)
            changed = deepcopy(payload)
            changed["performance"]["wall_seconds"] = 99.0
            changed["block_result"]["timing"]["block_wall_seconds"] = 88.0
            scientific_after = phase_d.phase_d_rehearsal_scientific_digest(changed)
            sealed_after, _ = phase_d._archive_with_content_seal(changed)
        self.assertEqual(scientific_before, scientific_after)
        self.assertNotEqual(
            sealed_before["archive_content_digest"],
            sealed_after["archive_content_digest"],
        )

    def test_scientific_changes_alter_scientific_identity(self):
        payload = self._payload()
        replay = {"block_evidence_digest": "c" * 64}
        mutations = []
        for key, value in (
            ("observation", 1.26),
            ("fit", {"parameter": 2.0}),
            ("decision", "approve"),
            ("failure", "nonconvergence"),
        ):
            changed = deepcopy(payload)
            changed["block_result"]["cases"][0][key] = value
            mutations.append(changed)
        changed_stream = deepcopy(payload)
        changed_stream["block_result"]["corrected_random_stream_manifest"][0][
            "seed"
        ] = 11
        mutations.append(changed_stream)
        changed_protocol = deepcopy(payload)
        changed_protocol["protocol_digest"] = "d" * 64
        mutations.append(changed_protocol)
        with patch.object(phase_d, "_replay_summary", return_value=replay):
            expected = phase_d.phase_d_rehearsal_scientific_digest(payload)
            for changed in mutations:
                with self.subTest(change=changed):
                    self.assertNotEqual(
                        expected,
                        phase_d.phase_d_rehearsal_scientific_digest(changed),
                    )

    def test_archive_seal_round_trip_and_tamper_detection(self):
        payload = self._payload()
        sealed, encoded = phase_d._archive_with_content_seal(payload)
        self.assertEqual(json.loads(encoded), sealed)
        material = dict(sealed)
        digest = material.pop("archive_content_digest")
        material.pop("archive_size_bytes")
        self.assertEqual(
            digest,
            phase_d._digest(
                f"{phase_d.PHASE_D_HASH_DOMAIN}.rehearsal_archive_content",
                material,
            ),
        )
        tampered = deepcopy(sealed)
        tampered["performance"]["wall_seconds"] = 123.0
        tampered_material = dict(tampered)
        tampered_material.pop("archive_content_digest")
        tampered_material.pop("archive_size_bytes")
        self.assertNotEqual(
            tampered["archive_content_digest"],
            phase_d._digest(
                f"{phase_d.PHASE_D_HASH_DOMAIN}.rehearsal_archive_content",
                tampered_material,
            ),
        )


class ProspectivePhaseDResourceProbeTests(unittest.TestCase):
    def test_constructed_twenty_block_archive_is_explicitly_non_scientific(self):
        seed = {
            "schema_version": 2,
            "protocol_version": "test",
            "source_revision": "a" * 40,
            "protocol_digest": "b" * 64,
            "scientific_result_digest": "c" * 64,
            "block_result": {"block": 0, "cases": [{"observation": 1.0}]},
        }
        payload = phase_d_resources.constructed_resource_archive_payload(seed)
        self.assertEqual(payload["block_count"], 20)
        self.assertEqual(len(payload["constructed_blocks"]), 20)
        self.assertEqual(
            payload["scientific_use"],
            "prohibited_constructed_resource_evidence_only",
        )
        self.assertEqual(
            phase_d_resources.validate_constructed_resource_archive(payload),
            payload,
        )
        tampered = deepcopy(payload)
        tampered["constructed_blocks"][4]["cases"][0]["observation"] = 2.0
        with self.assertRaisesRegex(ValueError, "differs from its seed"):
            phase_d_resources.validate_constructed_resource_archive(tampered)


if __name__ == "__main__":
    unittest.main()
