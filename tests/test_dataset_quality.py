import unittest

from thermotwin import (
    DeterministicTemperatureMissingness,
    TemperatureSensorOutage,
    run_ideal_contact_reference_test_stand,
    run_missing_contact_reference_test_stand,
)
from thermotwin.contact_resistance_inference import (
    reference_contact_resistance_dataset_split,
)
from thermotwin.dataset_quality import (
    format_dataset_quality_report,
    reference_dataset_quality_report,
    summarize_dataset_collection,
    summarize_observation_dataset,
)


class DatasetQualityTests(unittest.TestCase):
    def test_ideal_reference_is_complete_and_has_ground_truth(self):
        summary = summarize_observation_dataset(
            run_ideal_contact_reference_test_stand()
        )

        self.assertEqual(summary.observation_count, 244)
        self.assertEqual(summary.expected_observation_count, 244)
        self.assertEqual(summary.missing_observation_count, 0)
        self.assertEqual(summary.completeness_fraction, 1.0)
        self.assertEqual(summary.sensor_count, 4)
        self.assertEqual(summary.expected_measurement_time_count, 61)
        self.assertTrue(summary.provenance_available)
        self.assertTrue(summary.ground_truth_available)
        self.assertEqual(summary.observation_process_steps, ("ideal_sampling",))
        self.assertTrue(all(item.observation_count == 61 for item in summary.sensors))

    def test_missingness_summary_counts_the_exact_outage(self):
        dataset = run_missing_contact_reference_test_stand().dataset
        summary = summarize_observation_dataset(dataset)

        self.assertEqual(summary.observation_count, 233)
        self.assertEqual(summary.expected_observation_count, 244)
        self.assertEqual(summary.missing_observation_count, 11)
        self.assertAlmostEqual(summary.completeness_fraction, 233 / 244)
        by_name = {item.sensor_name: item for item in summary.sensors}
        self.assertEqual(
            by_name["cold_face_sensor"].missing_observation_count,
            11,
        )
        self.assertEqual(
            by_name["hot_face_sensor"].missing_observation_count,
            0,
        )

    def test_quality_summary_accepts_one_completely_unavailable_sensor(self):
        dataset = run_missing_contact_reference_test_stand(
            missingness_model=DeterministicTemperatureMissingness(
                outages=(
                    TemperatureSensorOutage(
                        "cold_face_sensor",
                        0.0,
                        60.0,
                    ),
                )
            )
        ).dataset

        summary = summarize_observation_dataset(dataset)
        cold_face = {
            item.sensor_name: item for item in summary.sensors
        }["cold_face_sensor"]
        self.assertEqual(cold_face.observation_count, 0)
        self.assertEqual(cold_face.missing_observation_count, 61)
        self.assertEqual(cold_face.completeness_fraction, 0.0)
        self.assertIsNone(cold_face.minimum_temperature)
        self.assertIsNone(cold_face.maximum_temperature)

    def test_collection_confirms_whole_regime_split_integrity(self):
        split = reference_contact_resistance_dataset_split()
        datasets = tuple(
            item.observations
            for group in (split.train, split.validation, split.test)
            for item in group
        )
        summary = summarize_dataset_collection(datasets)

        self.assertEqual(summary.dataset_count, 3)
        self.assertEqual(summary.total_observation_count, 732)
        self.assertEqual(summary.total_expected_observation_count, 732)
        self.assertEqual(summary.overall_completeness_fraction, 1.0)
        self.assertTrue(summary.all_provenance_available)
        self.assertTrue(summary.all_ground_truth_available)
        self.assertTrue(summary.regime_names_unique)
        self.assertTrue(summary.required_splits_present)
        self.assertEqual(summary.train_regimes, ("unipolar_training_pulse",))
        self.assertEqual(
            summary.validation_regimes,
            ("lower_amplitude_validation_pulse",),
        )
        self.assertEqual(summary.test_regimes, ("bipolar_test_pulse",))

    def test_report_is_compact_and_reproducible(self):
        report = reference_dataset_quality_report()

        self.assertIn("ThermoTwin dataset quality report", report)
        self.assertIn("observations: 732/732 (100.00%)", report)
        self.assertIn("ground truth recorded: PASS", report)
        self.assertIn("train/validation/test present: PASS", report)
        self.assertIn("bipolar_test_pulse | test", report)
        self.assertEqual(report, reference_dataset_quality_report())

    def test_invalid_collection_and_summary_types_are_rejected(self):
        with self.assertRaises(ValueError):
            summarize_observation_dataset("invalid")
        for datasets in ((), None, ("invalid",)):
            with self.subTest(datasets=datasets):
                with self.assertRaises(ValueError):
                    summarize_dataset_collection(datasets)
        with self.assertRaises(ValueError):
            format_dataset_quality_report("invalid")


if __name__ == "__main__":
    unittest.main()
