import unittest

from thermotwin.contact_resistance_inference import (
    REFERENCE_COLD_CONTACT_RESISTANCE,
    reference_contact_resistance_dataset_split,
)
from thermotwin.contact_resistance_lag_study import (
    format_contact_resistance_lag_study_report,
    reference_contact_resistance_lag_cases,
    run_contact_resistance_lag_study,
)
from thermotwin.contact_resistance_robustness import (
    lag_contact_resistance_dataset_split,
)


class ContactResistanceLagStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_contact_resistance_lag_study()
        cls.by_name = {case.name: case for case in cls.result.cases}

    def test_reference_cases_freeze_zero_individual_and_pair_lags(self):
        cases = reference_contact_resistance_lag_cases()

        self.assertEqual(
            tuple(case.name for case in cases),
            (
                "zero_lag",
                "cold_face_tau_2s",
                "cold_exchanger_tau_2s",
                "cold_pair_common_tau_2s",
                "cold_pair_asymmetric_tau_2s_0p5s",
            ),
        )
        self.assertEqual(cases[0].lag_model.sensor_time_constants, ())
        self.assertEqual(
            cases[-1].lag_model.sensor_time_constants,
            (
                ("cold_face_sensor", 2.0),
                ("cold_exchanger_sensor", 0.5),
            ),
        )

    def test_zero_lag_dense_pipeline_matches_ideal_output_limit(self):
        ideal = reference_contact_resistance_dataset_split()
        zero_lag = reference_contact_resistance_lag_cases()[0].lag_model
        lagged = lag_contact_resistance_dataset_split(ideal, zero_lag)

        self.assertEqual(lagged, ideal)
        self.assertIsNot(lagged, ideal)

    def test_zero_lag_recovers_true_resistance(self):
        zero = self.by_name["zero_lag"]

        self.assertLess(zero.absolute_parameter_error, 1e-6)
        self.assertLess(zero.training_truth_rmse, 1e-6)
        self.assertEqual(zero.training_observation_count, 122)

    def test_face_and_exchanger_lag_have_different_parameter_effects(self):
        face = self.by_name["cold_face_tau_2s"]
        exchanger = self.by_name["cold_exchanger_tau_2s"]

        self.assertLess(
            face.inferred_cold_contact_resistance,
            REFERENCE_COLD_CONTACT_RESISTANCE,
        )
        self.assertGreater(
            exchanger.inferred_cold_contact_resistance,
            REFERENCE_COLD_CONTACT_RESISTANCE,
        )
        self.assertGreater(
            exchanger.absolute_parameter_error,
            face.absolute_parameter_error,
        )

    def test_lag_estimates_are_frozen_regressions(self):
        expected = {
            "cold_face_tau_2s": 0.246787415,
            "cold_exchanger_tau_2s": 0.271277687,
            "cold_pair_common_tau_2s": 0.271434083,
            "cold_pair_asymmetric_tau_2s_0p5s": 0.252630129,
        }
        for name, inferred in expected.items():
            with self.subTest(name=name):
                self.assertAlmostEqual(
                    self.by_name[name].inferred_cold_contact_resistance,
                    inferred,
                    places=8,
                )

    def test_resistance_shift_cannot_remove_dynamic_lag_residual(self):
        for case in self.result.cases[1:]:
            with self.subTest(name=case.name):
                self.assertGreater(case.training_observation_rmse, 0.04)
                self.assertGreater(case.test_observation_rmse, 0.07)

    def test_truth_errors_transfer_to_held_out_regimes(self):
        common = self.by_name["cold_pair_common_tau_2s"]

        self.assertGreater(common.validation_truth_rmse, 0.01)
        self.assertGreater(common.test_truth_rmse, 0.02)
        self.assertGreater(
            common.test_observation_rmse,
            common.training_observation_rmse,
        )

    def test_text_report_contains_lag_cases_and_error_views(self):
        report = format_contact_resistance_lag_study_report(self.result)

        self.assertIn("cold contact resistance sensor-lag study", report)
        self.assertIn("cold_face_tau_2s:", report)
        self.assertIn("cold_pair_common_tau_2s:", report)
        self.assertIn("observation RMSE train/validation/test", report)
        self.assertIn("truth RMSE train/validation/test", report)


if __name__ == "__main__":
    unittest.main()
