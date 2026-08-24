from dataclasses import replace
import unittest

from thermotwin import (
    GaussianTemperatureNoise,
    apply_gaussian_temperature_noise,
    constant_current_contact_reference_experiment,
    run_incomplete_contact_reference_test_stand,
    run_ideal_contact_reference_test_stand,
    run_lagged_noisy_biased_contact_reference_test_stand,
)
from thermotwin.contact_resistance_inference import (
    reference_contact_resistance_dataset_split,
)
from thermotwin.dataset_metadata import (
    ContactExperimentMetadata,
    CurrentScheduleMetadata,
    DatasetProvenance,
    MetadataSetting,
    ObservationProcessStep,
)


class DatasetMetadataTests(unittest.TestCase):
    def test_reference_dataset_records_complete_reproducible_truth(self):
        dataset = run_ideal_contact_reference_test_stand()
        provenance = dataset.provenance

        self.assertIsInstance(provenance, DatasetProvenance)
        metadata = provenance.experiment
        self.assertEqual(metadata.experiment_name, "contact_reference")
        self.assertEqual(metadata.regime_name, "constant_current_reference")
        self.assertEqual(metadata.split, "unsplit")
        self.assertEqual(metadata.duration, 60.0)
        self.assertEqual(metadata.integration_time_step, 0.1)
        self.assertEqual(metadata.current_schedule.kind, "scalar")
        self.assertEqual(metadata.current_schedule.values, (1.0,))
        self.assertEqual(
            metadata.thermal_parameters.cold_contact_resistance,
            0.25,
        )
        self.assertEqual(
            metadata.thermoelectric_parameters.seebeck_coefficient,
            0.05,
        )
        self.assertEqual(
            tuple(step.name for step in provenance.observation_steps),
            ("ideal_sampling",),
        )
        self.assertFalse(hasattr(dataset, "trajectory"))
        self.assertFalse(hasattr(dataset, "truth"))

    def test_metadata_rebuilds_the_recorded_experiment(self):
        reference = constant_current_contact_reference_experiment()
        metadata = ContactExperimentMetadata.from_experiment(
            reference,
            experiment_name="reference",
            regime_name="constant",
        )

        self.assertEqual(metadata.to_experiment(), reference)

    def test_regime_datasets_record_split_schedule_and_parameter_truth(self):
        split = reference_contact_resistance_dataset_split()

        for expected_split, group in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        ):
            dataset = group[0]
            metadata = dataset.observations.provenance.experiment
            self.assertEqual(metadata.regime_name, dataset.regime.name)
            self.assertEqual(metadata.split, expected_split)
            self.assertEqual(
                metadata.current_schedule.to_current(),
                dataset.regime.current,
            )
            self.assertEqual(
                metadata.thermal_parameters.cold_contact_resistance,
                0.25,
            )

    def test_noise_seed_and_scale_are_appended_to_dataset_provenance(self):
        ideal = run_ideal_contact_reference_test_stand()
        noisy = apply_gaussian_temperature_noise(
            ideal,
            GaussianTemperatureNoise(
                default_standard_deviation=0.05,
                random_seed=17,
                sensor_standard_deviations=(("cold_face_sensor", 0.1),),
            ),
        ).dataset

        step = noisy.provenance.observation_steps[-1]
        self.assertEqual(step.name, "gaussian_temperature_noise")
        settings = {item.name: item.value for item in step.settings}
        self.assertEqual(settings["default_standard_deviation_K"], 0.05)
        self.assertEqual(settings["random_seed"], 17)
        self.assertEqual(
            settings["sensor_standard_deviation_K:cold_face_sensor"],
            0.1,
        )

    def test_combined_pipeline_records_steps_in_physical_order(self):
        complete = run_lagged_noisy_biased_contact_reference_test_stand()
        incomplete = run_incomplete_contact_reference_test_stand()

        self.assertEqual(
            tuple(
                step.name
                for step in complete.dataset.provenance.observation_steps
            ),
            (
                "ideal_sampling",
                "first_order_temperature_lag",
                "output_sampling",
                "fixed_temperature_bias",
                "gaussian_temperature_noise",
            ),
        )
        self.assertEqual(
            incomplete.dataset.provenance.observation_steps[-1].name,
            "deterministic_temperature_missingness",
        )
        outage_settings = {
            item.name: item.value
            for item in incomplete.dataset.provenance.observation_steps[-1].settings
        }
        self.assertEqual(outage_settings["outage_0_sensor"], "cold_face_sensor")
        self.assertEqual(outage_settings["outage_0_start_s"], 20.0)
        self.assertEqual(outage_settings["outage_0_end_s"], 30.0)

    def test_metadata_types_reject_invalid_or_ambiguous_values(self):
        with self.assertRaises(ValueError):
            MetadataSetting("", 1.0)
        with self.assertRaises(ValueError):
            MetadataSetting("value", float("nan"))
        with self.assertRaises(ValueError):
            ObservationProcessStep(
                "step",
                (MetadataSetting("same", 1), MetadataSetting("same", 2)),
            )
        with self.assertRaises(ValueError):
            CurrentScheduleMetadata("scalar", (1.0,), (1.0, 2.0))
        with self.assertRaises(ValueError):
            replace(
                run_ideal_contact_reference_test_stand(),
                provenance="invalid",
            )


if __name__ == "__main__":
    unittest.main()
