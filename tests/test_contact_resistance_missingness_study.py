import unittest

from thermotwin.contact_resistance_inference import (
    reference_contact_resistance_dataset_split,
)
from thermotwin.contact_resistance_missingness_study import (
    format_contact_resistance_missingness_study_report,
    missingness_for_contact_resistance_case,
    reference_contact_resistance_missingness_cases,
    run_contact_resistance_missingness_study,
)


class ContactResistanceMissingnessStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_contact_resistance_missingness_study()
        cls.by_name = {case.name: case for case in cls.result.cases}

    def test_reference_cases_freeze_control_and_turnoff_windows(self):
        cases = reference_contact_resistance_missingness_cases()

        self.assertEqual(
            tuple(case.name for case in cases),
            (
                "complete_readings",
                "pre_pulse_control_0_to_4s",
                "turn_off_instants",
                "turn_off_windows_plus_minus_2s",
                "turn_off_windows_plus_minus_5s",
            ),
        )
        self.assertIsNone(cases[0].turn_off_half_width)
        self.assertTrue(cases[1].remove_pre_pulse_control)
        self.assertEqual(cases[-1].turn_off_half_width, 5.0)

    def test_outages_align_to_each_regimes_nonzero_to_zero_switches(self):
        split = reference_contact_resistance_dataset_split()
        instant_case = reference_contact_resistance_missingness_cases()[2]
        training = missingness_for_contact_resistance_case(
            instant_case,
            split.train[0],
        )
        validation = missingness_for_contact_resistance_case(
            instant_case,
            split.validation[0],
        )
        test = missingness_for_contact_resistance_case(
            instant_case,
            split.test[0],
        )

        self.assertEqual(
            tuple(outage.start_time for outage in training.outages),
            (20.0, 20.0),
        )
        self.assertEqual(
            tuple(outage.start_time for outage in validation.outages),
            (30.0, 30.0),
        )
        self.assertEqual(
            tuple(outage.start_time for outage in test.outages),
            (20.0, 50.0, 20.0, 50.0),
        )

    def test_complete_and_prepulse_control_have_same_information(self):
        complete = self.by_name["complete_readings"]
        control = self.by_name["pre_pulse_control_0_to_4s"]

        self.assertEqual(complete.training_observation_count, 122)
        self.assertEqual(control.training_observation_count, 112)
        self.assertAlmostEqual(
            complete.training_information_curvature,
            control.training_information_curvature,
        )

    def test_turnoff_instant_removes_few_records_but_loses_information(self):
        complete = self.by_name["complete_readings"]
        instant = self.by_name["turn_off_instants"]

        self.assertEqual(instant.training_observation_count, 120)
        self.assertLess(
            instant.training_information_curvature,
            complete.training_information_curvature,
        )

    def test_wider_turnoff_windows_reduce_information_monotonically(self):
        cases = (
            self.by_name["complete_readings"],
            self.by_name["turn_off_instants"],
            self.by_name["turn_off_windows_plus_minus_2s"],
            self.by_name["turn_off_windows_plus_minus_5s"],
        )

        self.assertEqual(
            tuple(case.training_observation_count for case in cases),
            (122, 120, 112, 100),
        )
        self.assertTrue(
            all(
                left.training_information_curvature
                > right.training_information_curvature
                for left, right in zip(cases, cases[1:])
            )
        )

    def test_exact_remaining_data_still_recover_the_parameter(self):
        for case in self.result.cases:
            with self.subTest(name=case.name):
                self.assertLess(case.absolute_parameter_error, 1e-6)
                self.assertLess(case.training_observation_rmse, 1e-6)
                self.assertLess(case.validation_truth_rmse, 1e-6)
                self.assertLess(case.test_truth_rmse, 1e-6)

    def test_wide_window_loses_more_than_half_the_local_information(self):
        complete = self.by_name["complete_readings"]
        wide = self.by_name["turn_off_windows_plus_minus_5s"]

        self.assertLess(
            wide.training_information_curvature,
            0.5 * complete.training_information_curvature,
        )

    def test_text_report_contains_counts_curvature_and_error_views(self):
        report = format_contact_resistance_missingness_study_report(
            self.result
        )

        self.assertIn("turn-off missingness study", report)
        self.assertIn("turn_off_instants:", report)
        self.assertIn("records=120", report)
        self.assertIn("curvature=", report)
        self.assertIn("truth RMSE train/validation/test", report)


if __name__ == "__main__":
    unittest.main()
