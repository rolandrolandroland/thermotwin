import unittest

from thermotwin import (
    constant_current_reference_experiment,
    run_two_node_experiment,
)


class TwoNodeExperimentTests(unittest.TestCase):
    def test_reference_case_freezes_the_agreed_inputs(self):
        experiment = constant_current_reference_experiment()

        self.assertEqual(experiment.current, 1.0)
        self.assertEqual(experiment.duration, 60.0)
        self.assertEqual(experiment.time_step, 0.1)
        self.assertEqual(experiment.initial_cold_temperature, 300.0)
        self.assertEqual(experiment.initial_hot_temperature, 300.0)
        self.assertEqual(experiment.cold_reservoir_temperature, 300.0)
        self.assertEqual(experiment.hot_reservoir_temperature, 300.0)
        self.assertEqual(experiment.cold_external_heat, 0.0)
        self.assertEqual(experiment.hot_external_heat, 0.0)

    def test_reference_run_matches_hand_calculation_and_regression_values(self):
        result = run_two_node_experiment(
            constant_current_reference_experiment()
        )

        self.assertEqual(result.trajectory.time[0], 0.0)
        self.assertEqual(result.trajectory.time[-1], 60.0)
        self.assertEqual(result.diagnostics.cold_heat[0], 14.0)
        self.assertEqual(result.diagnostics.hot_heat[0], 16.0)
        self.assertAlmostEqual(
            result.trajectory.cold[-1], 295.971976, places=6
        )
        self.assertAlmostEqual(
            result.trajectory.hot[-1], 302.404041, places=6
        )

    def test_reference_diagnostics_obey_module_energy_identity(self):
        result = run_two_node_experiment(
            constant_current_reference_experiment()
        )

        for cold_heat, hot_heat, electrical_power in zip(
            result.diagnostics.cold_heat,
            result.diagnostics.hot_heat,
            result.diagnostics.electrical_power,
        ):
            self.assertAlmostEqual(
                hot_heat - cold_heat,
                electrical_power,
            )


if __name__ == "__main__":
    unittest.main()
