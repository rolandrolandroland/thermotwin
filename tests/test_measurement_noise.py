import math
import unittest

from thermotwin import (
    GaussianTemperatureNoise,
    apply_gaussian_temperature_noise,
    reference_gaussian_temperature_noise,
    run_ideal_contact_reference_test_stand,
    run_noisy_contact_reference_test_stand,
)


class MeasurementNoiseTests(unittest.TestCase):
    def setUp(self):
        self.ideal = run_ideal_contact_reference_test_stand()

    def test_reference_noise_configuration_is_frozen_and_generic(self):
        noise = reference_gaussian_temperature_noise()

        self.assertEqual(noise.default_standard_deviation, 0.05)
        self.assertEqual(noise.random_seed, 2026)
        self.assertEqual(noise.sensor_standard_deviations, ())

    def test_noise_configuration_rejects_invalid_values(self):
        invalid_configurations = (
            {"default_standard_deviation": -0.1},
            {"default_standard_deviation": float("inf")},
            {"default_standard_deviation": float("nan")},
            {"random_seed": 1.5},
            {"random_seed": True},
            {"sensor_standard_deviations": (("", 0.1),)},
            {"sensor_standard_deviations": (("cold", -0.1),)},
            {
                "sensor_standard_deviations": (
                    ("cold", 0.1),
                    ("cold", 0.2),
                )
            },
        )
        for values in invalid_configurations:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    GaussianTemperatureNoise(**values)

    def test_zero_noise_is_the_exact_limiting_case(self):
        result = apply_gaussian_temperature_noise(
            self.ideal,
            GaussianTemperatureNoise(default_standard_deviation=0.0),
        )

        self.assertIsNot(result.dataset, self.ideal)
        self.assertEqual(result.dataset.observations, self.ideal.observations)

    def test_same_seed_repeats_and_different_seed_changes_noise(self):
        first = apply_gaussian_temperature_noise(
            self.ideal,
            GaussianTemperatureNoise(random_seed=10),
        )
        repeated = apply_gaussian_temperature_noise(
            self.ideal,
            GaussianTemperatureNoise(random_seed=10),
        )
        different = apply_gaussian_temperature_noise(
            self.ideal,
            GaussianTemperatureNoise(random_seed=11),
        )

        self.assertEqual(
            first.dataset.observations,
            repeated.dataset.observations,
        )
        self.assertNotEqual(
            first.dataset.observations,
            different.dataset.observations,
        )

    def test_noise_changes_only_temperature_values(self):
        noisy = apply_gaussian_temperature_noise(
            self.ideal,
            reference_gaussian_temperature_noise(),
        ).dataset

        for ideal, observed in zip(
            self.ideal.observations,
            noisy.observations,
        ):
            self.assertEqual(observed.time, ideal.time)
            self.assertEqual(observed.sensor_name, ideal.sensor_name)
            self.assertIs(observed.location, ideal.location)
            self.assertEqual(observed.current, ideal.current)
        self.assertEqual(noisy.sensors, self.ideal.sensors)
        self.assertEqual(noisy.sampling_interval, self.ideal.sampling_interval)
        self.assertEqual(noisy.time_unit, self.ideal.time_unit)
        self.assertEqual(noisy.temperature_unit, self.ideal.temperature_unit)
        self.assertEqual(noisy.current_unit, self.ideal.current_unit)

    def test_reference_noise_has_expected_sample_statistics(self):
        noisy = apply_gaussian_temperature_noise(
            self.ideal,
            reference_gaussian_temperature_noise(),
        ).dataset
        errors = tuple(
            observed.temperature - ideal.temperature
            for ideal, observed in zip(
                self.ideal.observations,
                noisy.observations,
            )
        )
        mean_error = sum(errors) / len(errors)
        rms_error = math.sqrt(
            sum(error * error for error in errors) / len(errors)
        )

        self.assertTrue(any(error < 0.0 for error in errors))
        self.assertTrue(any(error > 0.0 for error in errors))
        self.assertLess(abs(mean_error), 0.015)
        self.assertGreater(rms_error, 0.035)
        self.assertLess(rms_error, 0.065)

    def test_per_sensor_override_changes_only_selected_sensor(self):
        noise = GaussianTemperatureNoise(
            default_standard_deviation=0.0,
            sensor_standard_deviations=(("cold_face_sensor", 0.1),),
        )
        noisy = apply_gaussian_temperature_noise(
            self.ideal,
            noise,
        ).dataset

        changed_names = {
            observed.sensor_name
            for ideal, observed in zip(
                self.ideal.observations,
                noisy.observations,
            )
            if observed.temperature != ideal.temperature
        }
        self.assertEqual(changed_names, {"cold_face_sensor"})
        self.assertEqual(noise.standard_deviation_for("cold_face_sensor"), 0.1)
        self.assertEqual(noise.standard_deviation_for("hot_face_sensor"), 0.0)

    def test_override_does_not_shift_other_sensors_random_draws(self):
        baseline = apply_gaussian_temperature_noise(
            self.ideal,
            GaussianTemperatureNoise(random_seed=10),
        ).dataset
        cold_face_exact = apply_gaussian_temperature_noise(
            self.ideal,
            GaussianTemperatureNoise(
                random_seed=10,
                sensor_standard_deviations=(("cold_face_sensor", 0.0),),
            ),
        ).dataset

        for ideal, baseline_value, override_value in zip(
            self.ideal.observations,
            baseline.observations,
            cold_face_exact.observations,
        ):
            if ideal.sensor_name == "cold_face_sensor":
                self.assertEqual(
                    override_value.temperature,
                    ideal.temperature,
                )
            else:
                self.assertEqual(
                    override_value.temperature,
                    baseline_value.temperature,
                )

    def test_unknown_sensor_override_is_rejected(self):
        noise = GaussianTemperatureNoise(
            sensor_standard_deviations=(("unknown_sensor", 0.1),),
        )

        with self.assertRaises(ValueError):
            apply_gaussian_temperature_noise(self.ideal, noise)

    def test_high_level_noisy_reference_preserves_counts_and_provenance(self):
        result = run_noisy_contact_reference_test_stand()

        self.assertEqual(len(result.dataset.measurement_times), 61)
        self.assertEqual(len(result.dataset.observations), 244)
        self.assertEqual(
            result.noise_model,
            reference_gaussian_temperature_noise(),
        )
        self.assertFalse(hasattr(result, "ideal_dataset"))
        self.assertFalse(hasattr(result, "truth"))

    def test_high_level_reference_supports_downsampling(self):
        result = run_noisy_contact_reference_test_stand(
            sampling_interval=5.0
        )

        self.assertEqual(len(result.dataset.measurement_times), 13)
        self.assertEqual(len(result.dataset.observations), 52)


if __name__ == "__main__":
    unittest.main()
