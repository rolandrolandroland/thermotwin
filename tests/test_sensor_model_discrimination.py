import unittest

from thermotwin.reports.sensor_model_discrimination import (
    format_sensor_model_discrimination_report,
)
from thermotwin.studies.sensor_model_discrimination import (
    COLD_EXCHANGER,
    COLD_FACE,
    HOT_EXCHANGER,
    SensorDiscriminationConfig,
    SensorPackage,
    run_sensor_model_discrimination,
)


class SensorModelDiscriminationTests(unittest.TestCase):
    @staticmethod
    def small_config():
        return SensorDiscriminationConfig(
            trial_count=1,
            fit_iterations=2,
            packages=(
                SensorPackage(
                    "baseline_one_pulse",
                    ("0.8A_20s",),
                    (COLD_EXCHANGER, HOT_EXCHANGER),
                    0,
                ),
                SensorPackage(
                    "add_cold_face_temperature",
                    ("0.8A_20s",),
                    (COLD_EXCHANGER, HOT_EXCHANGER, COLD_FACE),
                    1,
                ),
            ),
        )

    def test_study_is_paired_and_selects_between_both_topologies(self):
        result = run_sensor_model_discrimination(self.small_config())

        self.assertEqual(len(result.trials), 4)
        self.assertEqual(
            {item.truth_condition for item in result.trials},
            {"matched_four_state", "extra_interface_mass"},
        )
        for trial in result.trials:
            self.assertIn(trial.selected_model, {"four_state", "five_state"})
            self.assertGreaterEqual(trial.four_state_validation_mse, 0.0)
            self.assertGreaterEqual(trial.five_state_validation_mse, 0.0)

    def test_added_face_channel_increases_discrimination_margin_in_pilot(self):
        result = run_sensor_model_discrimination(self.small_config())
        by_key = {
            (item.truth_condition, item.package_name): item
            for item in result.summaries
        }

        for condition in ("matched_four_state", "extra_interface_mass"):
            self.assertGreater(
                by_key[(condition, "add_cold_face_temperature")]
                .mean_validation_mse_margin,
                by_key[(condition, "baseline_one_pulse")]
                .mean_validation_mse_margin,
            )

    def test_report_states_cost_and_synthetic_boundaries(self):
        text = format_sensor_model_discrimination_report(
            run_sensor_model_discrimination(self.small_config())
        )

        self.assertIn("common withheld validation run is additional", text)
        self.assertIn("not hardware validation", text)
        self.assertIn("training energy", text)

    def test_package_requires_both_existing_temperature_channels(self):
        with self.assertRaisesRegex(ValueError, "both exchanger"):
            SensorPackage(
                "invalid",
                ("0.8A_20s",),
                (COLD_EXCHANGER, COLD_FACE),
                1,
            )


if __name__ == "__main__":
    unittest.main()
