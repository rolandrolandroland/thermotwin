import unittest

from thermotwin.contact_resistance_inference import (
    reference_contact_resistance_dataset_split,
)
from thermotwin.contact_resistance_robustness import (
    restrict_observation_dataset,
    validate_sensor_names,
)
from thermotwin.contact_resistance_sensor_study import (
    format_contact_resistance_sensor_study_report,
    reference_contact_resistance_sensor_cases,
    run_contact_resistance_sensor_study,
)


class ContactResistanceSensorStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_contact_resistance_sensor_study()
        cls.by_name = {case.name: case for case in cls.result.cases}

    def test_reference_cases_freeze_five_available_sensor_sets(self):
        cases = reference_contact_resistance_sensor_cases()

        self.assertEqual(
            tuple(case.name for case in cases),
            (
                "cold_pair",
                "cold_face_only",
                "cold_exchanger_only",
                "hot_pair_only",
                "all_four_sensors",
            ),
        )
        self.assertEqual(cases[1].sensor_names, ("cold_face_sensor",))
        self.assertEqual(
            cases[3].sensor_names,
            ("hot_face_sensor", "hot_exchanger_sensor"),
        )

    def test_restriction_removes_sensor_definitions_and_records(self):
        ideal = reference_contact_resistance_dataset_split().train[0]
        restricted = restrict_observation_dataset(
            ideal.observations,
            ("cold_face_sensor",),
        )

        self.assertEqual(
            tuple(sensor.name for sensor in restricted.sensors),
            ("cold_face_sensor",),
        )
        self.assertEqual(len(restricted.observations), 61)
        self.assertTrue(
            all(
                observation.sensor_name == "cold_face_sensor"
                for observation in restricted.observations
            )
        )

    def test_sensor_selection_rejects_empty_duplicate_and_unknown_names(self):
        invalid = (
            (),
            ("cold_face_sensor", "cold_face_sensor"),
            ("unknown_sensor",),
        )
        for sensor_names in invalid:
            with self.subTest(sensor_names=sensor_names):
                with self.assertRaises(ValueError):
                    validate_sensor_names(sensor_names)

    def test_training_record_counts_follow_available_sensor_count(self):
        self.assertEqual(
            tuple(case.training_observation_count for case in self.result.cases),
            (122, 61, 61, 122, 244),
        )

    def test_exact_data_recover_truth_from_every_sensor_set(self):
        for case in self.result.cases:
            with self.subTest(name=case.name):
                self.assertLess(case.absolute_parameter_error, 1e-6)
                self.assertLess(case.training_truth_rmse, 1e-6)
                self.assertLess(case.test_truth_rmse, 1e-6)

    def test_cold_face_is_more_informative_than_cold_exchanger(self):
        face = self.by_name["cold_face_only"]
        exchanger = self.by_name["cold_exchanger_only"]

        self.assertGreater(
            face.training_information_curvature,
            2.0 * exchanger.training_information_curvature,
        )

    def test_hot_pair_has_very_weak_cold_contact_information(self):
        cold = self.by_name["cold_pair"]
        hot = self.by_name["hot_pair_only"]

        self.assertLess(
            hot.training_information_curvature,
            0.01 * cold.training_information_curvature,
        )

    def test_all_sensor_information_is_sum_of_cold_and_hot_pairs(self):
        cold = self.by_name["cold_pair"]
        hot = self.by_name["hot_pair_only"]
        all_sensors = self.by_name["all_four_sensors"]

        self.assertAlmostEqual(
            all_sensors.training_information_curvature,
            cold.training_information_curvature
            + hot.training_information_curvature,
            places=6,
        )

    def test_text_report_identifies_sensors_counts_and_curvature(self):
        report = format_contact_resistance_sensor_study_report(self.result)

        self.assertIn("restricted-sensor study", report)
        self.assertIn("cold_face_only:", report)
        self.assertIn("hot_pair_only:", report)
        self.assertIn("records=61", report)
        self.assertIn("sensors=hot_face_sensor,hot_exchanger_sensor", report)


if __name__ == "__main__":
    unittest.main()
