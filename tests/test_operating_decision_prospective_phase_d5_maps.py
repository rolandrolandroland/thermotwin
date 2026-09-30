from pathlib import Path
import tempfile
import unittest

from thermotwin.studies.operating_decision import POLICY_NAMES
from thermotwin.studies.operating_decision_prospective_phase_d import (
    PHASE_D_SENSOR_SCENARIOS,
)
import thermotwin.studies.operating_decision_prospective_phase_d5_maps as d5
from thermotwin.studies.sensor_model_discrimination import COLD_FACE, VOLTAGE


class PhaseD5MeasurementMapTests(unittest.TestCase):
    def test_sensor_scenarios_change_only_declared_physics(self):
        configs = {
            scenario.name: d5.phase_d5_physical_config(scenario)
            for scenario in PHASE_D_SENSOR_SCENARIOS
        }
        nominal = configs["nominal"]
        voltage = configs["high_voltage_noise"]
        face = configs["high_face_noise"]
        loading = configs["high_probe_loading"]

        self.assertEqual(dict(nominal.sensor.channel_noise)[VOLTAGE], 0.002)
        self.assertEqual(dict(voltage.sensor.channel_noise)[VOLTAGE], 0.004)
        self.assertEqual(dict(voltage.sensor.channel_noise)[COLD_FACE], 0.02)
        self.assertEqual(dict(face.sensor.channel_noise)[VOLTAGE], 0.002)
        self.assertEqual(dict(face.sensor.channel_noise)[COLD_FACE], 0.04)
        self.assertEqual(loading.face_sensor_capacitance_nominal, 10.0)
        self.assertEqual(loading.face_sensor_response_nominal, 2.5)
        self.assertEqual(nominal.face_sensor_capacitance_nominal, 5.0)

    def test_protocol_freezes_all_48_map_cells(self):
        payload = d5.phase_d5_protocol_payload(
            source_revision="a" * 40,
            source_manifest_digest="b" * 64,
        )
        self.assertEqual(len(payload["cost_scenarios"]), 12)
        self.assertEqual(len(payload["sensor_scenarios"]), 4)
        self.assertEqual(payload["map_cells"], 48)
        self.assertFalse(payload["internal_check_opened"])
        self.assertEqual(
            payload["no_useful_measurement_definition"],
            "selection_reason_equals_no_valuable_acquisition_stop_then_verify",
        )

    def test_measurement_map_summary_keeps_failures_and_outcomes_distinct(self):
        records = [
            {
                "selected_policy": "fixed_voltage",
                "definitive_decision": True,
                "ineligible_actions": [],
                "selection_failure": False,
                "verification_failure": False,
                "pipeline_failure": False,
                "no_useful_measurement": False,
            },
            {
                "selected_policy": "fixed_voltage",
                "definitive_decision": False,
                "ineligible_actions": ["fixed_face_temperature"],
                "selection_failure": False,
                "verification_failure": True,
                "pipeline_failure": False,
                "no_useful_measurement": False,
            },
            {
                "selected_policy": "stop_now",
                "definitive_decision": False,
                "ineligible_actions": [],
                "selection_failure": False,
                "verification_failure": False,
                "pipeline_failure": False,
                "no_useful_measurement": True,
            },
            {
                "selected_policy": None,
                "definitive_decision": False,
                "ineligible_actions": list(POLICY_NAMES[1:]),
                "selection_failure": True,
                "verification_failure": False,
                "pipeline_failure": True,
                "no_useful_measurement": False,
            },
        ]
        summary = d5.summarize_measurement_map(records)
        actions = {item["action"]: item for item in summary["actions"]}
        self.assertEqual(actions["fixed_voltage"]["selected_count"], 2)
        self.assertEqual(actions["fixed_voltage"]["definitive_count"], 1)
        self.assertEqual(actions["selection_failure"]["selected_count"], 1)
        self.assertEqual(summary["definitive_decision_count"], 1)
        self.assertEqual(summary["selection_failure_count"], 1)
        self.assertEqual(summary["verification_failure_count"], 1)
        self.assertEqual(summary["pipeline_failure_count"], 1)
        self.assertEqual(summary["no_useful_measurement_count"], 1)
        self.assertEqual(
            summary["ineligible_action_counts"]["fixed_face_temperature"],
            2,
        )

    def test_stress_path_rejects_nominal_and_unknown_scenarios(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "not predeclared"):
                d5.phase_d5_stress_paths(directory, "nominal")
            with self.assertRaisesRegex(ValueError, "not predeclared"):
                d5.phase_d5_stress_paths(directory, "invented")

    def test_source_allowlist_contains_both_d5_entry_points(self):
        self.assertIn(
            "thermotwin/studies/operating_decision_prospective_phase_d5_maps.py",
            d5.PHASE_D5_SOURCE_PATHS,
        )
        self.assertIn(
            "thermotwin/reports/operating_decision_prospective_phase_d5.py",
            d5.PHASE_D5_SOURCE_PATHS,
        )

    def test_final_save_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / d5.PHASE_D5_JSON_NAME).write_text("occupied")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                d5.save_phase_d5_artifacts({}, root)


if __name__ == "__main__":
    unittest.main()
