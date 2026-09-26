from contextlib import redirect_stderr
from copy import deepcopy
import io
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import thermotwin.reports.operating_decision_prospective_development as phase_d_cli
import thermotwin.studies.operating_decision_prospective_phase_d as phase_d
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
            "9277f9668f685871791dbb1caf25af48c5523dfd4aae33adf5a4871c4519001f",
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


if __name__ == "__main__":
    unittest.main()
