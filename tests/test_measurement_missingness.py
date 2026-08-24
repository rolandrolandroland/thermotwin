import unittest

from thermotwin import (
    DeterministicTemperatureMissingness,
    FixedTemperatureBias,
    IdealTemperatureSensor,
    ObservationDataset,
    TemperatureObservation,
    TemperatureSensorOutage,
    apply_deterministic_temperature_missingness,
    apply_fixed_temperature_bias,
    apply_gaussian_temperature_noise,
    reference_deterministic_temperature_missingness,
    reference_fixed_temperature_bias,
    reference_gaussian_temperature_noise,
    reference_first_order_temperature_lag,
    run_incomplete_contact_reference_test_stand,
    run_ideal_contact_reference_test_stand,
    run_lagged_contact_reference_test_stand,
    run_lagged_noisy_biased_contact_reference_test_stand,
    run_missing_contact_reference_test_stand,
)


def one_sensor_dataset(times):
    sensor = IdealTemperatureSensor("sensor", "cold_face")
    return ObservationDataset(
        observations=tuple(
            TemperatureObservation(
                time=time,
                sensor_name=sensor.name,
                location=sensor.location,
                temperature=300.0 + time,
                current=0.0,
            )
            for time in times
        ),
        sensors=(sensor,),
        sampling_interval=1.0,
    )


class MeasurementMissingnessTests(unittest.TestCase):
    def setUp(self):
        self.ideal = run_ideal_contact_reference_test_stand()

    def test_reference_outage_is_frozen_and_inclusive(self):
        missingness = reference_deterministic_temperature_missingness()

        self.assertEqual(
            missingness.outages,
            (
                TemperatureSensorOutage(
                    sensor_name="cold_face_sensor",
                    start_time=20.0,
                    end_time=30.0,
                ),
            ),
        )

    def test_outage_rejects_invalid_names_and_times(self):
        invalid_outages = (
            {"sensor_name": "", "start_time": 0.0, "end_time": 1.0},
            {"sensor_name": None, "start_time": 0.0, "end_time": 1.0},
            {"sensor_name": "sensor", "start_time": -1.0, "end_time": 1.0},
            {
                "sensor_name": "sensor",
                "start_time": float("inf"),
                "end_time": 1.0,
            },
            {
                "sensor_name": "sensor",
                "start_time": float("nan"),
                "end_time": 1.0,
            },
            {
                "sensor_name": "sensor",
                "start_time": "not a number",
                "end_time": 1.0,
            },
            {"sensor_name": "sensor", "start_time": 0.0, "end_time": -1.0},
            {
                "sensor_name": "sensor",
                "start_time": 0.0,
                "end_time": float("inf"),
            },
            {"sensor_name": "sensor", "start_time": 2.0, "end_time": 1.0},
        )
        for values in invalid_outages:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    TemperatureSensorOutage(**values)

    def test_missingness_rejects_malformed_or_overlapping_outages(self):
        outage = TemperatureSensorOutage("sensor", 1.0, 2.0)
        invalid_configurations = (
            None,
            (("sensor", 1.0, 2.0),),
            (outage, TemperatureSensorOutage("sensor", 2.0, 3.0)),
            (outage, TemperatureSensorOutage("sensor", 1.5, 1.75)),
        )
        for outages in invalid_configurations:
            with self.subTest(outages=outages):
                with self.assertRaises(ValueError):
                    DeterministicTemperatureMissingness(outages=outages)

    def test_overlapping_times_are_allowed_for_different_sensors(self):
        missingness = DeterministicTemperatureMissingness(
            outages=(
                TemperatureSensorOutage("cold_face_sensor", 1.0, 2.0),
                TemperatureSensorOutage("hot_face_sensor", 1.0, 2.0),
            )
        )

        self.assertEqual(len(missingness.outages), 2)

    def test_no_outages_are_the_exact_input_limiting_case(self):
        result = apply_deterministic_temperature_missingness(
            self.ideal,
            DeterministicTemperatureMissingness(),
        )

        self.assertIsNot(result.dataset, self.ideal)
        self.assertEqual(result.dataset, self.ideal)

    def test_outage_outside_measurement_times_has_no_effect(self):
        missingness = DeterministicTemperatureMissingness(
            outages=(
                TemperatureSensorOutage(
                    "cold_face_sensor",
                    100.0,
                    110.0,
                ),
            )
        )
        result = apply_deterministic_temperature_missingness(
            self.ideal,
            missingness,
        )

        self.assertEqual(result.dataset, self.ideal)

    def test_unknown_sensor_outage_is_rejected(self):
        missingness = DeterministicTemperatureMissingness(
            outages=(TemperatureSensorOutage("unknown", 1.0, 2.0),)
        )

        with self.assertRaisesRegex(ValueError, "unknown sensors"):
            apply_deterministic_temperature_missingness(
                self.ideal,
                missingness,
            )

    def test_outage_boundaries_are_inclusive_with_float_tolerance(self):
        dataset = one_sensor_dataset((0.0, 0.1 + 0.2, 0.6))
        missingness = DeterministicTemperatureMissingness(
            outages=(TemperatureSensorOutage("sensor", 0.3, 0.3),)
        )
        result = apply_deterministic_temperature_missingness(
            dataset,
            missingness,
        )

        self.assertEqual(
            result.dataset.measurement_times,
            (0.0, 0.6),
        )

    def test_reference_outage_removes_expected_cold_face_records(self):
        result = run_missing_contact_reference_test_stand()
        cold = result.dataset.observations_for("cold_face_sensor")

        self.assertEqual(len(result.dataset.measurement_times), 61)
        self.assertEqual(len(result.dataset.observations), 233)
        self.assertEqual(len(cold), 50)
        self.assertFalse(
            any(20.0 <= observation.time <= 30.0 for observation in cold)
        )
        self.assertEqual(cold[19].time, 19.0)
        self.assertEqual(cold[20].time, 31.0)

    def test_reference_outage_preserves_other_sensors_and_metadata(self):
        result = run_missing_contact_reference_test_stand().dataset

        for sensor_name in (
            "hot_face_sensor",
            "cold_exchanger_sensor",
            "hot_exchanger_sensor",
        ):
            self.assertEqual(
                result.observations_for(sensor_name),
                self.ideal.observations_for(sensor_name),
            )
        self.assertEqual(result.sensors, self.ideal.sensors)
        self.assertEqual(result.sampling_interval, self.ideal.sampling_interval)
        self.assertEqual(result.time_unit, self.ideal.time_unit)
        self.assertEqual(result.temperature_unit, self.ideal.temperature_unit)
        self.assertEqual(result.current_unit, self.ideal.current_unit)

    def test_retained_observations_are_unmodified_and_input_is_unchanged(self):
        original_observations = self.ideal.observations
        result = run_missing_contact_reference_test_stand().dataset
        original_set = set(original_observations)

        self.assertTrue(
            all(observation in original_set for observation in result.observations)
        )
        self.assertEqual(self.ideal.observations, original_observations)
        self.assertEqual(len(self.ideal.observations), 244)

    def test_removing_every_observation_is_rejected(self):
        dataset = one_sensor_dataset((0.0, 1.0, 2.0))
        missingness = DeterministicTemperatureMissingness(
            outages=(TemperatureSensorOutage("sensor", 0.0, 2.0),)
        )

        with self.assertRaisesRegex(ValueError, "every observation"):
            apply_deterministic_temperature_missingness(
                dataset,
                missingness,
            )

    def test_one_sensor_may_be_fully_unobserved_when_others_remain(self):
        missingness = DeterministicTemperatureMissingness(
            outages=(
                TemperatureSensorOutage(
                    "cold_face_sensor",
                    0.0,
                    60.0,
                ),
            )
        )
        result = apply_deterministic_temperature_missingness(
            self.ideal,
            missingness,
        ).dataset

        self.assertEqual(
            result.observations_for("cold_face_sensor"),
            (),
        )
        self.assertEqual(len(result.sensors), 4)
        self.assertEqual(len(result.measurement_times), 61)
        self.assertEqual(len(result.observations), 183)

    def test_reference_outage_respects_another_sampling_interval(self):
        result = run_missing_contact_reference_test_stand(
            sampling_interval=5.0
        ).dataset

        self.assertEqual(len(result.measurement_times), 13)
        self.assertEqual(len(result.observations), 49)
        self.assertEqual(
            len(result.observations_for("cold_face_sensor")),
            10,
        )

    def test_complete_pipeline_applies_missingness_last_and_retains_models(self):
        complete = run_lagged_noisy_biased_contact_reference_test_stand()
        missingness = reference_deterministic_temperature_missingness()
        expected = apply_deterministic_temperature_missingness(
            complete.dataset,
            missingness,
        )
        result = run_incomplete_contact_reference_test_stand()

        self.assertEqual(result.dataset, expected.dataset)
        self.assertEqual(
            result.lag_model,
            reference_first_order_temperature_lag(),
        )
        self.assertEqual(
            result.bias_model,
            reference_fixed_temperature_bias(),
        )
        self.assertEqual(
            result.noise_model,
            reference_gaussian_temperature_noise(),
        )
        self.assertEqual(result.missingness_model, missingness)
        self.assertFalse(hasattr(result, "ideal_dataset"))
        self.assertFalse(hasattr(result, "truth"))

        complete_cold_at_31 = next(
            observation
            for observation in complete.dataset.observations_for(
                "cold_face_sensor"
            )
            if observation.time == 31.0
        )
        incomplete_cold_at_31 = next(
            observation
            for observation in result.dataset.observations_for(
                "cold_face_sensor"
            )
            if observation.time == 31.0
        )
        self.assertEqual(incomplete_cold_at_31, complete_cold_at_31)

    def test_removing_records_before_seeded_noise_changes_later_errors(self):
        lagged = run_lagged_contact_reference_test_stand().dataset
        biased = apply_fixed_temperature_bias(
            lagged,
            reference_fixed_temperature_bias(),
        ).dataset
        missing_before_noise = apply_deterministic_temperature_missingness(
            biased,
            reference_deterministic_temperature_missingness(),
        ).dataset
        wrong_order = apply_gaussian_temperature_noise(
            missing_before_noise,
            reference_gaussian_temperature_noise(),
        ).dataset
        agreed_order = run_incomplete_contact_reference_test_stand().dataset

        self.assertNotEqual(wrong_order.observations, agreed_order.observations)

    def test_empty_missingness_in_complete_pipeline_preserves_all_records(self):
        complete = run_lagged_noisy_biased_contact_reference_test_stand()
        result = run_incomplete_contact_reference_test_stand(
            missingness_model=DeterministicTemperatureMissingness()
        )

        self.assertEqual(result.dataset, complete.dataset)
        self.assertEqual(len(result.dataset.observations), 244)


if __name__ == "__main__":
    unittest.main()
