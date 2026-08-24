import unittest

from thermotwin.contact_resistance_bias_study import (
    format_contact_resistance_bias_study_report,
    reference_contact_resistance_bias_cases,
    run_contact_resistance_bias_study,
)
from thermotwin.contact_resistance_inference import (
    REFERENCE_COLD_CONTACT_RESISTANCE,
)


class ContactResistanceBiasStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_contact_resistance_bias_study()
        cls.by_name = {case.name: case for case in cls.result.cases}

    def test_reference_cases_freeze_zero_individual_and_pair_biases(self):
        cases = reference_contact_resistance_bias_cases()

        self.assertEqual(
            tuple(case.name for case in cases),
            (
                "zero_bias",
                "cold_face_plus_0p10_K",
                "cold_exchanger_plus_0p10_K",
                "cold_pair_common_plus_0p10_K",
                "cold_pair_differential_0p10_K",
            ),
        )
        self.assertEqual(cases[0].bias_model.sensor_biases, ())
        self.assertEqual(
            cases[4].bias_model.sensor_biases,
            (
                ("cold_face_sensor", 0.05),
                ("cold_exchanger_sensor", -0.05),
            ),
        )

    def test_zero_bias_recovers_ideal_limiting_case(self):
        zero = self.by_name["zero_bias"]

        self.assertLess(zero.absolute_parameter_error, 1e-6)
        self.assertLess(zero.training_truth_rmse, 1e-6)
        self.assertEqual(zero.training_observation_count, 122)
        self.assertEqual(zero.search_evaluations, 32)

    def test_face_and_exchanger_bias_shift_parameter_in_opposite_directions(self):
        face = self.by_name["cold_face_plus_0p10_K"]
        exchanger = self.by_name["cold_exchanger_plus_0p10_K"]

        self.assertLess(
            face.inferred_cold_contact_resistance,
            REFERENCE_COLD_CONTACT_RESISTANCE,
        )
        self.assertGreater(
            exchanger.inferred_cold_contact_resistance,
            REFERENCE_COLD_CONTACT_RESISTANCE,
        )

    def test_first_bias_estimates_are_frozen_regressions(self):
        expected = {
            "cold_face_plus_0p10_K": 0.208885282,
            "cold_exchanger_plus_0p10_K": 0.272817055,
            "cold_pair_common_plus_0p10_K": 0.228450366,
            "cold_pair_differential_0p10_K": 0.218889695,
        }
        for name, inferred in expected.items():
            with self.subTest(name=name):
                self.assertAlmostEqual(
                    self.by_name[name].inferred_cold_contact_resistance,
                    inferred,
                    places=8,
                )

    def test_common_mode_bias_does_not_cancel_from_absolute_temperature_fit(self):
        common = self.by_name["cold_pair_common_plus_0p10_K"]

        self.assertGreater(common.absolute_parameter_error, 0.02)
        self.assertGreater(common.training_observation_rmse, 0.09)

    def test_differential_bias_creates_systematic_truth_error(self):
        differential = self.by_name["cold_pair_differential_0p10_K"]

        self.assertGreater(differential.absolute_parameter_error, 0.03)
        self.assertGreater(differential.test_truth_rmse, 0.04)
        self.assertGreater(
            differential.test_observation_rmse,
            differential.training_observation_rmse,
        )

    def test_bias_persists_on_unseen_regimes(self):
        for case in self.result.cases[1:]:
            with self.subTest(name=case.name):
                self.assertGreater(case.validation_truth_rmse, 0.0)
                self.assertGreater(case.test_truth_rmse, 0.0)

    def test_text_report_contains_all_cases_and_both_error_views(self):
        report = format_contact_resistance_bias_study_report(self.result)

        self.assertIn("cold contact resistance fixed-bias study", report)
        self.assertIn("zero_bias:", report)
        self.assertIn("cold_pair_differential_0p10_K:", report)
        self.assertIn("observation RMSE train/validation/test", report)
        self.assertIn("truth RMSE train/validation/test", report)


if __name__ == "__main__":
    unittest.main()
