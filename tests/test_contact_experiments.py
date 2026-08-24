import unittest

from thermotwin import (
    constant_current_contact_reference_experiment,
    run_four_node_contact_experiment,
)


class FourNodeContactExperimentTests(unittest.TestCase):
    def test_reference_case_freezes_agreed_generic_inputs(self):
        experiment = constant_current_contact_reference_experiment()
        thermal = experiment.thermal_parameters

        self.assertEqual(experiment.current, 1.0)
        self.assertEqual(experiment.duration, 60.0)
        self.assertEqual(experiment.time_step, 0.1)
        self.assertEqual(experiment.initial_cold_face_temperature, 300.0)
        self.assertEqual(experiment.initial_hot_face_temperature, 300.0)
        self.assertEqual(experiment.initial_cold_exchanger_temperature, 300.0)
        self.assertEqual(experiment.initial_hot_exchanger_temperature, 300.0)
        self.assertEqual(thermal.cold_face_thermal_capacitance, 50.0)
        self.assertEqual(thermal.cold_exchanger_thermal_capacitance, 50.0)
        self.assertEqual(thermal.hot_face_thermal_capacitance, 100.0)
        self.assertEqual(thermal.hot_exchanger_thermal_capacitance, 100.0)
        self.assertEqual(thermal.cold_contact_resistance, 0.25)
        self.assertEqual(thermal.hot_contact_resistance, 0.25)
        self.assertEqual(experiment.cold_external_heat, 0.0)
        self.assertEqual(experiment.hot_external_heat, 0.0)

    def test_reference_run_matches_initial_prediction_and_regression_values(
        self,
    ):
        result = run_four_node_contact_experiment(
            constant_current_contact_reference_experiment()
        )
        trajectory = result.trajectory
        diagnostics = result.diagnostics

        self.assertEqual(trajectory.time[0], 0.0)
        self.assertEqual(trajectory.time[-1], 60.0)
        self.assertEqual(diagnostics.cold_contact_heat[0], 0.0)
        self.assertEqual(diagnostics.hot_contact_heat[0], 0.0)
        self.assertEqual(diagnostics.cold_heat[0], 14.0)
        self.assertEqual(diagnostics.hot_heat[0], 16.0)
        self.assertAlmostEqual(
            trajectory.cold_face[-1],
            294.795190,
            places=6,
        )
        self.assertAlmostEqual(
            trajectory.hot_face[-1],
            303.959316,
            places=6,
        )
        self.assertAlmostEqual(
            trajectory.cold_exchanger[-1],
            296.735891,
            places=6,
        )
        self.assertAlmostEqual(
            trajectory.hot_exchanger[-1],
            301.743400,
            places=6,
        )
        self.assertGreater(
            diagnostics.face_temperature_difference[-1],
            diagnostics.exchanger_temperature_difference[-1],
        )

    def test_final_contact_heat_differs_from_module_heat_during_transient(self):
        result = run_four_node_contact_experiment(
            constant_current_contact_reference_experiment()
        )
        diagnostics = result.diagnostics

        self.assertNotAlmostEqual(
            diagnostics.cold_contact_heat[-1],
            diagnostics.cold_heat[-1],
        )
        self.assertNotAlmostEqual(
            diagnostics.hot_contact_heat[-1],
            diagnostics.hot_heat[-1],
        )
        self.assertLess(
            diagnostics.exchanger_cooling_cop[-1],
            diagnostics.module_cooling_cop[-1],
        )


if __name__ == "__main__":
    unittest.main()
