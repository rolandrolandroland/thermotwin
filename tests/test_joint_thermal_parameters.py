import math
import unittest

from thermotwin.core.controls import PiecewiseConstantCurrent
from thermotwin.inference.experiment_selection import candidate_current
from thermotwin.inference.joint_thermal_parameters import (
    JointThermalFitConfig,
    JointThermalTruth,
    analyze_joint_thermal_identifiability,
    fit_joint_thermal_parameters,
    generate_joint_thermal_observations,
)


class JointThermalParameterTests(unittest.TestCase):
    def test_nonlinear_fit_recovers_off_nominal_truth_and_profiles_biases(self):
        config = JointThermalFitConfig(gauss_newton_iterations=8)
        truth = JointThermalTruth(
            cold_contact_resistance=0.27,
            cold_face_thermal_capacitance=54.0,
            sensor_time_constant=1.7,
            cold_sensor_bias=0.06,
            hot_sensor_bias=-0.03,
        )
        current = candidate_current(0.8, 20.0)
        observations = generate_joint_thermal_observations(
            current, truth, config, noise_seed=None
        )

        fit = fit_joint_thermal_parameters(current, observations, config)

        for estimate, expected in zip(fit.physical_values, truth.physical_values):
            self.assertLess(abs(math.log(estimate / expected)), 0.02)
        self.assertAlmostEqual(fit.cold_sensor_bias, truth.cold_sensor_bias, places=3)
        self.assertAlmostEqual(fit.hot_sensor_bias, truth.hot_sensor_bias, places=3)
        self.assertEqual(fit.identifiability.supported_rank, 3)
        self.assertFalse(fit.reached_bound)

    def test_selected_pulse_is_supported_and_zero_current_is_rank_zero(self):
        config = JointThermalFitConfig(gauss_newton_iterations=1)
        selected = analyze_joint_thermal_identifiability(
            candidate_current(0.8, 20.0), config
        )
        zero = analyze_joint_thermal_identifiability(
            PiecewiseConstantCurrent.constant(0.0), config
        )

        self.assertEqual(selected.supported_rank, 3)
        self.assertTrue(math.isfinite(selected.condition_number))
        self.assertEqual(zero.effective_rank, 0)
        self.assertEqual(zero.supported_rank, 0)
        self.assertTrue(math.isinf(zero.condition_number))

    def test_invalid_configuration_and_truth_are_rejected(self):
        with self.assertRaises(ValueError):
            JointThermalTruth(cold_contact_resistance=0.0)
        with self.assertRaises(ValueError):
            JointThermalFitConfig(value_bounds=((0.1, 0.2),))
        with self.assertRaises(ValueError):
            JointThermalFitConfig(gauss_newton_iterations=0)


if __name__ == "__main__":
    unittest.main()
