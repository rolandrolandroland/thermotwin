import unittest

from thermotwin import (
    FixedTemperatureBias,
    GaussianTemperatureNoise,
    apply_fixed_temperature_bias,
    apply_gaussian_temperature_noise,
    reference_fixed_temperature_bias,
    reference_gaussian_temperature_noise,
    run_biased_contact_reference_test_stand,
    run_ideal_contact_reference_test_stand,
    run_noisy_biased_contact_reference_test_stand,
)


class MeasurementBiasTests(unittest.TestCase):
    def setUp(self):
        self.ideal = run_ideal_contact_reference_test_stand()

    def test_reference_bias_is_frozen_generic_and_one_sensor_only(self):
        bias = reference_fixed_temperature_bias()

        self.assertEqual(bias.default_bias, 0.0)
        self.assertEqual(
            bias.sensor_biases,
            (("cold_face_sensor", 0.10),),
        )

    def test_bias_configuration_rejects_invalid_values(self):
        invalid_configurations = (
            {"default_bias": float("inf")},
            {"default_bias": float("nan")},
            {"sensor_biases": (("", 0.1),)},
            {"sensor_biases": (("cold", float("inf")),)},
            {"sensor_biases": (("cold", float("nan")),)},
            {
                "sensor_biases": (
                    ("cold", 0.1),
                    ("cold", 0.2),
                )
            },
        )
        for values in invalid_configurations:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    FixedTemperatureBias(**values)

    def test_zero_bias_is_the_exact_ideal_limiting_case(self):
        result = apply_fixed_temperature_bias(
            self.ideal,
            FixedTemperatureBias(),
        )

        self.assertIsNot(result.dataset, self.ideal)
        self.assertEqual(result.dataset.observations, self.ideal.observations)

    def test_default_and_per_sensor_biases_are_additive_and_persistent(self):
        bias = FixedTemperatureBias(
            default_bias=-0.05,
            sensor_biases=(("cold_face_sensor", 0.10),),
        )
        result = apply_fixed_temperature_bias(self.ideal, bias)

        for ideal, observed in zip(
            self.ideal.observations,
            result.dataset.observations,
        ):
            expected_bias = (
                0.10
                if ideal.sensor_name == "cold_face_sensor"
                else -0.05
            )
            self.assertAlmostEqual(
                observed.temperature - ideal.temperature,
                expected_bias,
            )

    def test_bias_changes_only_temperature_values(self):
        biased = apply_fixed_temperature_bias(
            self.ideal,
            reference_fixed_temperature_bias(),
        ).dataset

        for ideal, observed in zip(
            self.ideal.observations,
            biased.observations,
        ):
            self.assertEqual(observed.time, ideal.time)
            self.assertEqual(observed.sensor_name, ideal.sensor_name)
            self.assertIs(observed.location, ideal.location)
            self.assertEqual(observed.current, ideal.current)
        self.assertEqual(biased.sensors, self.ideal.sensors)
        self.assertEqual(biased.sampling_interval, self.ideal.sampling_interval)
        self.assertEqual(biased.time_unit, self.ideal.time_unit)
        self.assertEqual(biased.temperature_unit, self.ideal.temperature_unit)
        self.assertEqual(biased.current_unit, self.ideal.current_unit)

    def test_reference_bias_affects_only_cold_face_readings(self):
        result = run_biased_contact_reference_test_stand()

        for ideal, observed in zip(
            self.ideal.observations,
            result.dataset.observations,
        ):
            if ideal.sensor_name == "cold_face_sensor":
                self.assertAlmostEqual(
                    observed.temperature - ideal.temperature,
                    0.10,
                )
            else:
                self.assertEqual(observed.temperature, ideal.temperature)

    def test_bias_does_not_average_away_over_repeated_measurements(self):
        result = run_biased_contact_reference_test_stand()
        cold_ideal = self.ideal.observations_for("cold_face_sensor")
        cold_biased = result.dataset.observations_for("cold_face_sensor")
        errors = tuple(
            observed.temperature - ideal.temperature
            for ideal, observed in zip(cold_ideal, cold_biased)
        )

        self.assertEqual(len(errors), 61)
        self.assertAlmostEqual(sum(errors) / len(errors), 0.10)

    def test_unknown_sensor_override_is_rejected(self):
        bias = FixedTemperatureBias(
            sensor_biases=(("unknown_sensor", 0.10),),
        )

        with self.assertRaises(ValueError):
            apply_fixed_temperature_bias(self.ideal, bias)

    def test_noise_and_bias_order_agree_to_floating_point_precision(self):
        noise = GaussianTemperatureNoise(random_seed=12)
        bias = reference_fixed_temperature_bias()
        noise_then_bias = apply_fixed_temperature_bias(
            apply_gaussian_temperature_noise(self.ideal, noise).dataset,
            bias,
        ).dataset
        bias_then_noise = apply_gaussian_temperature_noise(
            apply_fixed_temperature_bias(self.ideal, bias).dataset,
            noise,
        ).dataset

        for first, second in zip(
            noise_then_bias.observations,
            bias_then_noise.observations,
        ):
            self.assertAlmostEqual(first.temperature, second.temperature)

    def test_combined_reference_retains_both_effect_configurations(self):
        result = run_noisy_biased_contact_reference_test_stand()

        self.assertEqual(len(result.dataset.measurement_times), 61)
        self.assertEqual(len(result.dataset.observations), 244)
        self.assertEqual(
            result.noise_model,
            reference_gaussian_temperature_noise(),
        )
        self.assertEqual(
            result.bias_model,
            reference_fixed_temperature_bias(),
        )
        self.assertFalse(hasattr(result, "ideal_dataset"))
        self.assertFalse(hasattr(result, "truth"))

    def test_bias_reference_supports_downsampling(self):
        result = run_biased_contact_reference_test_stand(
            sampling_interval=5.0
        )

        self.assertEqual(len(result.dataset.measurement_times), 13)
        self.assertEqual(len(result.dataset.observations), 52)


if __name__ == "__main__":
    unittest.main()
