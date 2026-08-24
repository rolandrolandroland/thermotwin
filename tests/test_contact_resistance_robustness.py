from dataclasses import replace
import unittest

from thermotwin.contact_resistance_inference import (
    contact_resistance_observation_rmse,
    contact_resistance_training_loss,
    fit_cold_contact_resistance,
    reference_contact_resistance_dataset_split,
)
from thermotwin.contact_resistance_robustness import (
    contact_resistance_information_curvature,
    count_selected_observations,
    downsample_observation_dataset,
    lag_contact_resistance_dataset_split,
    map_contact_resistance_dataset_split,
    match_observation_schema,
    restrict_observation_dataset,
)
from thermotwin.measurement_lag import FirstOrderTemperatureLag
from thermotwin.measurement_missingness import (
    DeterministicTemperatureMissingness,
    TemperatureSensorOutage,
    apply_deterministic_temperature_missingness,
)


class ContactResistanceRobustnessUtilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ideal = reference_contact_resistance_dataset_split()

    def test_downsampling_preserves_regular_times_sensors_and_units(self):
        dense = self.ideal.train[0].observations
        downsampled = downsample_observation_dataset(dense, 5.0)

        self.assertEqual(len(downsampled.measurement_times), 13)
        self.assertEqual(len(downsampled.observations), 52)
        self.assertEqual(downsampled.sensors, dense.sensors)
        self.assertEqual(downsampled.temperature_unit, "K")

    def test_split_mapping_preserves_regimes_and_does_not_mutate_input(self):
        mapped = map_contact_resistance_dataset_split(
            self.ideal,
            lambda dataset: downsample_observation_dataset(
                dataset.observations,
                5.0,
            ),
        )

        self.assertEqual(mapped.train[0].regime, self.ideal.train[0].regime)
        self.assertEqual(len(mapped.train[0].observations.observations), 52)
        self.assertEqual(
            len(self.ideal.train[0].observations.observations),
            244,
        )

    def test_schema_matching_returns_ideal_values_at_available_keys(self):
        ideal = self.ideal.train[0].observations
        incomplete = apply_deterministic_temperature_missingness(
            ideal,
            DeterministicTemperatureMissingness(
                outages=(
                    TemperatureSensorOutage(
                        "cold_face_sensor",
                        18.0,
                        22.0,
                    ),
                )
            ),
        ).dataset
        restricted = restrict_observation_dataset(
            incomplete,
            ("cold_face_sensor", "cold_exchanger_sensor"),
        )
        matched = match_observation_schema(ideal, restricted)

        self.assertEqual(matched.observations, restricted.observations)
        self.assertEqual(matched.sensors, restricted.sensors)
        self.assertEqual(matched.sampling_interval, restricted.sampling_interval)
        self.assertEqual(
            matched.provenance.observation_steps[-1].name,
            "matched_observation_schema",
        )
        self.assertEqual(len(matched.observations), 117)

    def test_selected_sensor_fit_recovers_single_sensor_limit(self):
        fit = fit_cold_contact_resistance(
            self.ideal.train,
            fitted_sensor_names=("cold_face_sensor",),
        )

        self.assertAlmostEqual(
            fit.inferred_cold_contact_resistance,
            0.25,
            places=7,
        )

    def test_training_loss_rejects_empty_unknown_and_unavailable_sensors(self):
        for names in ((), ("unknown",)):
            with self.subTest(names=names):
                with self.assertRaises(ValueError):
                    contact_resistance_training_loss(
                        0.25,
                        self.ideal.train,
                        fitted_sensor_names=names,
                    )

        restricted_observations = restrict_observation_dataset(
            self.ideal.train[0].observations,
            ("cold_face_sensor",),
        )
        restricted_split = map_contact_resistance_dataset_split(
            self.ideal,
            lambda dataset: restrict_observation_dataset(
                dataset.observations,
                ("cold_face_sensor",),
            ),
        )
        self.assertEqual(
            restricted_split.train[0].observations,
            restricted_observations,
        )
        with self.assertRaisesRegex(ValueError, "unavailable"):
            contact_resistance_training_loss(
                0.25,
                restricted_split.train,
                fitted_sensor_names=("cold_exchanger_sensor",),
            )

    def test_observation_rmse_pairs_only_to_retained_times(self):
        ideal_dataset = self.ideal.train[0]
        incomplete = apply_deterministic_temperature_missingness(
            ideal_dataset.observations,
            DeterministicTemperatureMissingness(
                outages=(
                    TemperatureSensorOutage(
                        "cold_face_sensor",
                        20.0,
                        20.0,
                    ),
                )
            ),
        ).dataset
        incomplete_dataset = replace(
            ideal_dataset,
            observations=incomplete,
        )
        rmse = contact_resistance_observation_rmse(
            0.25,
            incomplete_dataset,
            sensor_names=("cold_face_sensor",),
        )

        self.assertAlmostEqual(rmse, 0.0)

    def test_information_curvature_uses_available_record_count(self):
        count = count_selected_observations(
            self.ideal.train,
            ("cold_face_sensor",),
        )
        curvature = contact_resistance_information_curvature(
            self.ideal.train,
            fitted_sensor_names=("cold_face_sensor",),
        )

        self.assertEqual(count, 61)
        self.assertGreater(curvature, 0.0)
        with self.assertRaises(ValueError):
            contact_resistance_information_curvature(
                self.ideal.train,
                fitted_sensor_names=("cold_face_sensor",),
                parameter_step=0.0,
            )

    def test_dense_lag_interval_cannot_exceed_output_interval(self):
        with self.assertRaises(ValueError):
            lag_contact_resistance_dataset_split(
                self.ideal,
                FirstOrderTemperatureLag(),
                dense_sampling_interval=2.0,
            )


if __name__ == "__main__":
    unittest.main()
