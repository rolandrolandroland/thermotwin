import unittest

from thermotwin import (
    PiecewiseConstantCurrent,
    TemperatureTrajectory,
    ThermoelectricParameters,
    evaluate_trajectory,
)


class TrajectoryDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.parameters = ThermoelectricParameters(
            seebeck_coefficient=0.05,
            electrical_resistance=2.0,
            thermal_conductance=0.5,
        )

    def test_reference_initial_diagnostics_match_hand_calculation(self):
        trajectory = TemperatureTrajectory(
            time=(0.0,),
            cold=(300.0,),
            hot=(300.0,),
        )

        diagnostics = evaluate_trajectory(
            self.parameters,
            trajectory,
            current=1.0,
        )

        self.assertEqual(diagnostics.temperature_difference, (0.0,))
        self.assertEqual(diagnostics.current, (1.0,))
        self.assertEqual(diagnostics.cold_heat, (14.0,))
        self.assertEqual(diagnostics.hot_heat, (16.0,))
        self.assertEqual(diagnostics.voltage, (2.0,))
        self.assertEqual(diagnostics.electrical_power, (2.0,))
        self.assertEqual(diagnostics.cooling_cop, (7.0,))

    def test_energy_identity_holds_at_every_sample(self):
        trajectory = TemperatureTrajectory(
            time=(0.0, 1.0, 2.0),
            cold=(300.0, 298.0, 295.0),
            hot=(300.0, 302.0, 305.0),
        )
        diagnostics = evaluate_trajectory(
            self.parameters,
            trajectory,
            current=1.0,
        )

        for cold_heat, hot_heat, power in zip(
            diagnostics.cold_heat,
            diagnostics.hot_heat,
            diagnostics.electrical_power,
        ):
            self.assertAlmostEqual(hot_heat - cold_heat, power)

    def test_cop_is_undefined_at_zero_electrical_power(self):
        trajectory = TemperatureTrajectory(
            time=(0.0, 1.0),
            cold=(300.0, 301.0),
            hot=(310.0, 309.0),
        )

        diagnostics = evaluate_trajectory(
            self.parameters,
            trajectory,
            current=0.0,
        )

        self.assertEqual(diagnostics.cooling_cop, (None, None))

    def test_scheduled_current_is_recorded_at_every_sample(self):
        trajectory = TemperatureTrajectory(
            time=(0.0, 0.5, 1.0, 2.0, 3.0, 4.0),
            cold=(300.0,) * 6,
            hot=(300.0,) * 6,
        )
        pulse = PiecewiseConstantCurrent.pulse(
            start_time=1.0,
            end_time=3.0,
            pulse_current=2.0,
        )

        diagnostics = evaluate_trajectory(
            self.parameters,
            trajectory,
            current=pulse,
        )

        self.assertEqual(diagnostics.current, (0.0, 0.0, 2.0, 2.0, 0.0, 0.0))
        self.assertEqual(
            diagnostics.cooling_cop,
            (None, None, 3.25, 3.25, None, None),
        )

    def test_mismatched_history_lengths_are_rejected(self):
        malformed_trajectory = TemperatureTrajectory(
            time=(0.0, 1.0),
            cold=(300.0,),
            hot=(300.0, 301.0),
        )

        with self.assertRaises(ValueError):
            evaluate_trajectory(
                self.parameters,
                malformed_trajectory,
                current=1.0,
            )


if __name__ == "__main__":
    unittest.main()
