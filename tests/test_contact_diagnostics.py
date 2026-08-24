import unittest

from thermotwin import (
    FourNodeContactTemperatureTrajectory,
    constant_current_contact_reference_experiment,
    evaluate_contact_trajectory,
    run_four_node_contact_experiment,
)


class ContactTrajectoryDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.experiment = constant_current_contact_reference_experiment()

    def test_initial_diagnostics_match_hand_calculation(self):
        trajectory = FourNodeContactTemperatureTrajectory(
            time=(0.0,),
            cold_face=(300.0,),
            hot_face=(300.0,),
            cold_exchanger=(300.0,),
            hot_exchanger=(300.0,),
        )
        diagnostics = evaluate_contact_trajectory(
            self.experiment.thermoelectric_parameters,
            self.experiment.thermal_parameters,
            trajectory,
            current=1.0,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
        )

        self.assertEqual(diagnostics.current, (1.0,))
        self.assertEqual(diagnostics.face_temperature_difference, (0.0,))
        self.assertEqual(
            diagnostics.exchanger_temperature_difference,
            (0.0,),
        )
        self.assertEqual(diagnostics.cold_contact_heat, (0.0,))
        self.assertEqual(diagnostics.hot_contact_heat, (0.0,))
        self.assertEqual(diagnostics.cold_heat, (14.0,))
        self.assertEqual(diagnostics.hot_heat, (16.0,))
        self.assertEqual(diagnostics.voltage, (2.0,))
        self.assertEqual(diagnostics.electrical_power, (2.0,))
        self.assertEqual(diagnostics.module_cooling_cop, (7.0,))
        self.assertEqual(diagnostics.exchanger_cooling_cop, (0.0,))
        self.assertAlmostEqual(diagnostics.stored_energy_rate[0], 2.0)
        self.assertEqual(diagnostics.external_energy_rate, (2.0,))
        self.assertAlmostEqual(diagnostics.energy_balance_residual[0], 0.0)

    def test_reference_histories_are_aligned_and_energy_closes(self):
        result = run_four_node_contact_experiment(self.experiment)
        sample_count = len(result.trajectory.time)

        self.assertEqual(sample_count, 601)
        for history in result.diagnostics:
            self.assertEqual(len(history), sample_count)
        self.assertLess(
            max(
                abs(value)
                for value in result.diagnostics.energy_balance_residual
            ),
            1e-12,
        )
        for cold_heat, hot_heat, power in zip(
            result.diagnostics.cold_heat,
            result.diagnostics.hot_heat,
            result.diagnostics.electrical_power,
        ):
            self.assertAlmostEqual(hot_heat - cold_heat, power)

    def test_zero_power_makes_both_cop_histories_undefined(self):
        trajectory = FourNodeContactTemperatureTrajectory(
            time=(0.0, 1.0),
            cold_face=(300.0, 299.0),
            hot_face=(310.0, 309.0),
            cold_exchanger=(301.0, 300.0),
            hot_exchanger=(309.0, 308.0),
        )
        diagnostics = evaluate_contact_trajectory(
            self.experiment.thermoelectric_parameters,
            self.experiment.thermal_parameters,
            trajectory,
            current=0.0,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
        )

        self.assertEqual(diagnostics.module_cooling_cop, (None, None))
        self.assertEqual(diagnostics.exchanger_cooling_cop, (None, None))

    def test_mismatched_histories_are_rejected(self):
        malformed = FourNodeContactTemperatureTrajectory(
            time=(0.0, 1.0),
            cold_face=(300.0,),
            hot_face=(300.0, 301.0),
            cold_exchanger=(300.0, 300.0),
            hot_exchanger=(300.0, 300.0),
        )

        with self.assertRaises(ValueError):
            evaluate_contact_trajectory(
                self.experiment.thermoelectric_parameters,
                self.experiment.thermal_parameters,
                malformed,
                current=1.0,
                cold_reservoir_temperature=300.0,
                hot_reservoir_temperature=300.0,
            )


if __name__ == "__main__":
    unittest.main()
