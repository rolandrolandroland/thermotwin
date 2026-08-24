import math
import unittest

from thermotwin import (
    FirstOrderTemperatureLag,
    FixedTemperatureBias,
    GaussianTemperatureNoise,
    IdealTemperatureSensor,
    IdealVirtualTestStand,
    ObservationDataset,
    TemperatureObservation,
    apply_fixed_temperature_bias,
    apply_first_order_temperature_lag,
    apply_gaussian_temperature_noise,
    reference_first_order_temperature_lag,
    run_ideal_contact_reference_test_stand,
    run_lagged_contact_reference_test_stand,
    run_lagged_noisy_biased_contact_reference_test_stand,
)


def one_sensor_dataset(times, temperatures):
    sensor = IdealTemperatureSensor("sensor", "cold_face")
    return ObservationDataset(
        observations=tuple(
            TemperatureObservation(
                time=time,
                sensor_name=sensor.name,
                location=sensor.location,
                temperature=temperature,
                current=0.0,
            )
            for time, temperature in zip(times, temperatures)
        ),
        sensors=(sensor,),
        sampling_interval=1.0,
    )


class MeasurementLagTests(unittest.TestCase):
    def setUp(self):
        self.ideal = run_ideal_contact_reference_test_stand()

    def test_reference_lag_is_frozen_generic_and_one_sensor_only(self):
        lag = reference_first_order_temperature_lag()

        self.assertEqual(lag.default_time_constant, 0.0)
        self.assertEqual(
            lag.sensor_time_constants,
            (("cold_face_sensor", 2.0),),
        )

    def test_lag_configuration_rejects_invalid_values(self):
        invalid_configurations = (
            {"default_time_constant": -1.0},
            {"default_time_constant": float("inf")},
            {"default_time_constant": float("nan")},
            {"default_time_constant": "not a number"},
            {"sensor_time_constants": None},
            {"sensor_time_constants": (None,)},
            {"sensor_time_constants": (("cold",),)},
            {"sensor_time_constants": (("cold", 1.0, 2.0),)},
            {"sensor_time_constants": (("", 1.0),)},
            {"sensor_time_constants": (("cold", "not a number"),)},
            {"sensor_time_constants": (("cold", -1.0),)},
            {"sensor_time_constants": (("cold", float("inf")),)},
            {
                "sensor_time_constants": (
                    ("cold", 1.0),
                    ("cold", 2.0),
                )
            },
        )
        for values in invalid_configurations:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    FirstOrderTemperatureLag(**values)

    def test_zero_lag_is_the_exact_input_limiting_case(self):
        result = apply_first_order_temperature_lag(
            self.ideal,
            FirstOrderTemperatureLag(),
        )

        self.assertIsNot(result.dataset, self.ideal)
        self.assertEqual(result.dataset.observations, self.ideal.observations)

    def test_constant_temperature_is_unchanged_for_nonzero_lag(self):
        constant = one_sensor_dataset(
            (0.0, 1.0, 2.0, 3.0),
            (300.0, 300.0, 300.0, 300.0),
        )
        result = apply_first_order_temperature_lag(
            constant,
            FirstOrderTemperatureLag(default_time_constant=2.0),
        )

        self.assertEqual(
            tuple(
                observation.temperature
                for observation in result.dataset.observations
            ),
            (300.0, 300.0, 300.0, 300.0),
        )

    def test_piecewise_linear_then_constant_target_is_integrated_exactly(self):
        ramp_then_hold = one_sensor_dataset(
            (0.0, 1.0, 2.0),
            (0.0, 10.0, 10.0),
        )
        result = apply_first_order_temperature_lag(
            ramp_then_hold,
            FirstOrderTemperatureLag(default_time_constant=1.0),
        )
        decay = math.exp(-1.0)
        ramp_end = 10.0 * decay

        self.assertEqual(result.dataset.observations[0].temperature, 0.0)
        self.assertAlmostEqual(
            result.dataset.observations[1].temperature,
            ramp_end,
        )
        self.assertAlmostEqual(
            result.dataset.observations[2].temperature,
            decay * ramp_end + (1.0 - decay) * 10.0,
        )

    def test_linear_ramp_matches_continuous_first_order_solution(self):
        time_step = 0.1
        times = tuple(index * time_step for index in range(101))
        ramp = one_sensor_dataset(times, times)
        time_constant = 2.0

        result = apply_first_order_temperature_lag(
            ramp,
            FirstOrderTemperatureLag(
                default_time_constant=time_constant,
            ),
        )

        for observation in result.dataset.observations:
            expected = (
                observation.time
                - time_constant
                + time_constant
                * math.exp(-observation.time / time_constant)
            )
            self.assertAlmostEqual(observation.temperature, expected, places=12)

    def test_irregular_intervals_use_their_actual_time_differences(self):
        irregular = one_sensor_dataset(
            (0.0, 0.5, 2.0),
            (0.0, 10.0, 10.0),
        )
        result = apply_first_order_temperature_lag(
            irregular,
            FirstOrderTemperatureLag(default_time_constant=1.0),
        )
        first = 20.0 * (0.5 - (1.0 - math.exp(-0.5)))
        second = math.exp(-1.5) * first + (1.0 - math.exp(-1.5)) * 10.0

        self.assertAlmostEqual(result.dataset.observations[1].temperature, first)
        self.assertAlmostEqual(result.dataset.observations[2].temperature, second)

    def test_unknown_sensor_override_is_rejected(self):
        lag = FirstOrderTemperatureLag(
            sensor_time_constants=(("unknown_sensor", 1.0),),
        )

        with self.assertRaises(ValueError):
            apply_first_order_temperature_lag(self.ideal, lag)

    def test_reference_lag_affects_only_cold_face_and_has_expected_direction(self):
        result = run_lagged_contact_reference_test_stand()
        cold_ideal = self.ideal.observations_for("cold_face_sensor")
        cold_lagged = result.dataset.observations_for("cold_face_sensor")

        self.assertEqual(cold_lagged[0].temperature, cold_ideal[0].temperature)
        self.assertTrue(
            all(
                lagged.temperature >= ideal.temperature
                for ideal, lagged in zip(cold_ideal, cold_lagged)
            )
        )
        self.assertGreater(
            cold_lagged[-1].temperature,
            cold_ideal[-1].temperature,
        )
        for sensor_name in (
            "hot_face_sensor",
            "cold_exchanger_sensor",
            "hot_exchanger_sensor",
        ):
            self.assertEqual(
                result.dataset.observations_for(sensor_name),
                self.ideal.observations_for(sensor_name),
            )

    def test_lag_changes_only_temperature_values(self):
        lagged = run_lagged_contact_reference_test_stand().dataset

        for ideal, observed in zip(
            self.ideal.observations,
            lagged.observations,
        ):
            self.assertEqual(observed.time, ideal.time)
            self.assertEqual(observed.sensor_name, ideal.sensor_name)
            self.assertIs(observed.location, ideal.location)
            self.assertEqual(observed.current, ideal.current)
        self.assertEqual(lagged.sensors, self.ideal.sensors)
        self.assertEqual(lagged.sampling_interval, self.ideal.sampling_interval)
        self.assertEqual(lagged.time_unit, self.ideal.time_unit)
        self.assertEqual(lagged.temperature_unit, self.ideal.temperature_unit)
        self.assertEqual(lagged.current_unit, self.ideal.current_unit)

    def test_larger_time_constant_produces_more_cold_face_lag(self):
        short = run_lagged_contact_reference_test_stand(
            lag_model=FirstOrderTemperatureLag(
                sensor_time_constants=(("cold_face_sensor", 1.0),),
            )
        )
        long = run_lagged_contact_reference_test_stand(
            lag_model=FirstOrderTemperatureLag(
                sensor_time_constants=(("cold_face_sensor", 4.0),),
            )
        )

        self.assertGreater(
            long.dataset.observations_for("cold_face_sensor")[-1].temperature,
            short.dataset.observations_for("cold_face_sensor")[-1].temperature,
        )

    def test_high_level_lag_is_computed_before_output_downsampling(self):
        one_second = run_lagged_contact_reference_test_stand(
            sampling_interval=1.0
        )
        five_seconds = run_lagged_contact_reference_test_stand(
            sampling_interval=5.0
        )
        one_second_cold = {
            observation.time: observation.temperature
            for observation in one_second.dataset.observations_for(
                "cold_face_sensor"
            )
        }

        self.assertEqual(len(five_seconds.dataset.measurement_times), 13)
        for observation in five_seconds.dataset.observations_for(
            "cold_face_sensor"
        ):
            self.assertEqual(
                observation.temperature,
                one_second_cold[observation.time],
            )

    def test_combined_workflow_retains_order_and_all_configurations(self):
        result = run_lagged_noisy_biased_contact_reference_test_stand()

        self.assertEqual(len(result.dataset.measurement_times), 61)
        self.assertEqual(len(result.dataset.observations), 244)
        self.assertEqual(
            result.lag_model,
            reference_first_order_temperature_lag(),
        )
        self.assertEqual(result.bias_model.sensor_biases, (("cold_face_sensor", 0.1),))
        self.assertEqual(result.noise_model.default_standard_deviation, 0.05)
        self.assertFalse(hasattr(result, "ideal_dataset"))
        self.assertFalse(hasattr(result, "truth"))

    def test_combined_workflow_adds_random_noise_after_lag(self):
        lag = reference_first_order_temperature_lag()
        bias = FixedTemperatureBias(
            sensor_biases=(("cold_face_sensor", 0.25),)
        )
        noise = GaussianTemperatureNoise(random_seed=12)
        lagged = run_lagged_contact_reference_test_stand(
            lag_model=lag
        ).dataset
        biased = apply_fixed_temperature_bias(lagged, bias).dataset
        expected = apply_gaussian_temperature_noise(biased, noise).dataset
        combined = run_lagged_noisy_biased_contact_reference_test_stand(
            lag_model=lag,
            bias_model=bias,
            noise_model=noise,
        ).dataset

        self.assertEqual(combined.observations, expected.observations)

        noise_then_lag = apply_first_order_temperature_lag(
            apply_gaussian_temperature_noise(self.ideal, noise).dataset,
            lag,
        ).dataset
        lag_then_noise = apply_gaussian_temperature_noise(
            apply_first_order_temperature_lag(self.ideal, lag).dataset,
            noise,
        ).dataset
        self.assertNotEqual(
            noise_then_lag.observations,
            lag_then_noise.observations,
        )


if __name__ == "__main__":
    unittest.main()
