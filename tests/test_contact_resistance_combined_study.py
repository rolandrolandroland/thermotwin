import math
import unittest

from thermotwin.contact_resistance_combined_study import (
    ContactResistanceCombinedStudyConfig,
    format_contact_resistance_combined_study_report,
    run_contact_resistance_combined_study,
)
from thermotwin.measurement_bias import FixedTemperatureBias
from thermotwin.measurement_lag import FirstOrderTemperatureLag


class ContactResistanceCombinedStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.small_config = ContactResistanceCombinedStudyConfig(
            trial_count=5
        )
        cls.small_result = run_contact_resistance_combined_study(
            cls.small_config
        )

    def test_default_config_freezes_all_five_measurement_choices(self):
        config = ContactResistanceCombinedStudyConfig()

        self.assertEqual(config.noise_standard_deviation, 0.05)
        self.assertEqual(config.trial_count, 100)
        self.assertEqual(config.first_seed, 2026)
        self.assertEqual(
            config.bias_model.sensor_biases,
            (("cold_face_sensor", 0.10),),
        )
        self.assertEqual(
            config.lag_model.sensor_time_constants,
            (("cold_face_sensor", 2.0),),
        )
        self.assertEqual(config.turn_off_half_width, 2.0)
        self.assertEqual(
            config.available_sensor_names,
            ("cold_face_sensor", "cold_exchanger_sensor"),
        )
        self.assertEqual(config.search.resistance_tolerance, 1e-6)

    def test_config_rejects_invalid_values(self):
        invalid = (
            {"noise_standard_deviation": -1.0},
            {"noise_standard_deviation": float("nan")},
            {"noise_standard_deviation": True},
            {"trial_count": 0},
            {"trial_count": True},
            {"first_seed": 1.5},
            {"first_seed": -1},
            {"bias_model": "invalid"},
            {"lag_model": "invalid"},
            {"turn_off_half_width": -1.0},
            {"turn_off_half_width": True},
            {"available_sensor_names": ()},
            {"available_sensor_names": ("unknown",)},
            {"search": "invalid"},
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    ContactResistanceCombinedStudyConfig(**values)

    def test_all_zero_and_complete_configuration_recovers_ideal_limit(self):
        config = ContactResistanceCombinedStudyConfig(
            noise_standard_deviation=0.0,
            trial_count=1,
            bias_model=FixedTemperatureBias(),
            lag_model=FirstOrderTemperatureLag(),
            turn_off_half_width=None,
        )
        result = run_contact_resistance_combined_study(config)

        self.assertLess(result.summary.parameter_rmse, 1e-6)
        self.assertLess(result.summary.mean_training_truth_rmse, 1e-6)
        self.assertEqual(result.summary.search_bound_hits, 0)

    def test_first_combined_trial_is_frozen_regression(self):
        trial = self.small_result.trials[0]

        self.assertEqual(tuple(trial.seeds), (2026, 2027, 2028))
        self.assertAlmostEqual(
            trial.inferred_cold_contact_resistance,
            0.205764353,
            places=8,
        )
        self.assertAlmostEqual(
            trial.training_observation_rmse,
            0.147474,
            places=6,
        )
        self.assertAlmostEqual(
            trial.test_truth_rmse,
            0.059663,
            places=6,
        )
        self.assertFalse(trial.reached_search_bound)

    def test_small_combined_study_is_exactly_reproducible(self):
        repeated = run_contact_resistance_combined_study(
            self.small_config
        )

        self.assertEqual(repeated, self.small_result)

    def test_changed_first_seed_changes_empirical_distribution(self):
        changed = run_contact_resistance_combined_study(
            ContactResistanceCombinedStudyConfig(
                trial_count=2,
                first_seed=9000,
            )
        )

        self.assertNotEqual(
            changed.summary.mean_inferred_resistance,
            self.small_result.summary.mean_inferred_resistance,
        )

    def test_summary_parameter_rmse_matches_individual_trials(self):
        errors = tuple(
            trial.signed_parameter_error
            for trial in self.small_result.trials
        )
        expected = math.sqrt(
            sum(error * error for error in errors) / len(errors)
        )

        self.assertAlmostEqual(
            self.small_result.summary.parameter_rmse,
            expected,
        )

    def test_systematic_error_dominates_random_trial_spread(self):
        summary = self.small_result.summary

        self.assertGreater(
            abs(summary.mean_parameter_bias),
            5.0 * summary.parameter_standard_deviation,
        )
        self.assertGreater(summary.mean_test_truth_rmse, 0.05)
        self.assertEqual(summary.search_bound_hits, 0)

    def test_report_records_pipeline_controls_and_both_error_views(self):
        report = format_contact_resistance_combined_study_report(
            self.small_result
        )

        self.assertIn("combined-imperfections study", report)
        self.assertIn("dense lag -> sample -> bias -> noise", report)
        self.assertIn("turn-off outage half-width: 2.000 s", report)
        self.assertIn("available sensors: cold_face_sensor", report)
        self.assertIn("mean observation RMSE", report)
        self.assertIn("mean truth RMSE", report)


if __name__ == "__main__":
    unittest.main()
