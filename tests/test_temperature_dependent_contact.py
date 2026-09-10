import math
import unittest

from thermotwin.core.controls import PiecewiseConstantCurrent
from thermotwin.physics.thermoelectric import electrical_power
from thermotwin.simulation.four_node_experiments import (
    constant_current_contact_reference_experiment,
)
from thermotwin.simulation.interface_mass_mismatch import (
    InterfaceMassMismatch,
    InterfaceMassTrajectory,
    integrate_interface_mass_truth,
    interface_mass_rhs,
)
from thermotwin.simulation.temperature_dependent_contact import (
    TemperatureDependentColdContact,
    integrate_temperature_dependent_contact_truth,
    temperature_dependent_contact_rhs,
)


class TemperatureDependentContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = constant_current_contact_reference_experiment()
        cls.mismatch = InterfaceMassMismatch(
            thermal_capacitance=20.0,
            face_side_resistance_fraction=0.5,
        )

    def test_law_is_positive_monotone_and_validated(self):
        contact = TemperatureDependentColdContact(beta=0.12)
        low = contact.resistance(0.25, 295.0)
        reference = contact.resistance(0.25, 300.0)
        high = contact.resistance(0.25, 305.0)

        self.assertGreater(low, 0.0)
        self.assertLess(low, reference)
        self.assertEqual(reference, 0.25)
        self.assertGreater(high, reference)
        self.assertAlmostEqual(high / reference, reference / low, places=14)

        for beta in (-0.01, math.inf, math.nan):
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                TemperatureDependentColdContact(beta=beta)
        for temperature in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(reference_temperature=temperature), self.assertRaises(
                ValueError
            ):
                TemperatureDependentColdContact(reference_temperature=temperature)
        for resistance in (0.0, -0.1, math.inf, math.nan):
            with self.subTest(reference_resistance=resistance), self.assertRaises(
                ValueError
            ):
                contact.resistance(resistance, 300.0)
        for temperature in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(interface_temperature=temperature), self.assertRaises(
                ValueError
            ):
                contact.resistance(0.25, temperature)

    def test_zero_beta_rhs_and_trajectory_equal_constant_contact_truth(self):
        thermal = self.reference.thermal_parameters
        thermoelectric = self.reference.thermoelectric_parameters
        rhs_arguments = dict(
            cold_face_temperature=297.0,
            hot_face_temperature=304.0,
            cold_interface_temperature=298.5,
            cold_exchanger_temperature=299.0,
            hot_exchanger_temperature=302.0,
            current=0.7,
            cold_reservoir_temperature=296.0,
            hot_reservoir_temperature=307.0,
            cold_external_heat=0.3,
            hot_external_heat=-0.2,
        )
        expected_rates = interface_mass_rhs(
            thermoelectric,
            thermal,
            self.mismatch,
            **rhs_arguments,
        )
        actual_rates = temperature_dependent_contact_rhs(
            thermoelectric,
            thermal,
            self.mismatch,
            TemperatureDependentColdContact(beta=0.0),
            **rhs_arguments,
        )
        self.assertEqual(actual_rates, expected_rates)

        current = PiecewiseConstantCurrent(
            transition_times=(0.3, 1.1, 2.2),
            values=(0.0, 0.8, -0.4, 0.0),
        )
        integration_arguments = dict(
            initial_temperature=300.0,
            duration=3.0,
            time_step=0.2,
            current=current,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
        )
        expected_trajectory = integrate_interface_mass_truth(
            thermoelectric,
            thermal,
            self.mismatch,
            **integration_arguments,
        )
        actual_trajectory = integrate_temperature_dependent_contact_truth(
            thermoelectric,
            thermal,
            self.mismatch,
            TemperatureDependentColdContact(beta=0.0),
            **integration_arguments,
        )
        self.assertEqual(actual_trajectory, expected_trajectory)

    def test_rhs_closes_total_energy_balance(self):
        thermal = self.reference.thermal_parameters
        thermoelectric = self.reference.thermoelectric_parameters
        temperatures = (296.5, 306.0, 298.0, 300.5, 303.0)
        current = 0.9
        cold_reservoir_temperature = 294.0
        hot_reservoir_temperature = 309.0
        cold_external_heat = 0.4
        hot_external_heat = -0.15
        rates = temperature_dependent_contact_rhs(
            thermoelectric,
            thermal,
            self.mismatch,
            TemperatureDependentColdContact(beta=0.16),
            cold_face_temperature=temperatures[0],
            hot_face_temperature=temperatures[1],
            cold_interface_temperature=temperatures[2],
            cold_exchanger_temperature=temperatures[3],
            hot_exchanger_temperature=temperatures[4],
            current=current,
            cold_reservoir_temperature=cold_reservoir_temperature,
            hot_reservoir_temperature=hot_reservoir_temperature,
            cold_external_heat=cold_external_heat,
            hot_external_heat=hot_external_heat,
        )
        stored_energy_rate = sum(
            capacitance * rate
            for capacitance, rate in zip(
                (
                    thermal.cold_face_thermal_capacitance,
                    thermal.hot_face_thermal_capacitance,
                    self.mismatch.thermal_capacitance,
                    thermal.cold_exchanger_thermal_capacitance,
                    thermal.hot_exchanger_thermal_capacitance,
                ),
                rates,
            )
        )
        cold_reservoir_heat = thermal.cold_reservoir_conductance * (
            cold_reservoir_temperature - temperatures[3]
        )
        hot_reservoir_heat = thermal.hot_reservoir_conductance * (
            hot_reservoir_temperature - temperatures[4]
        )
        expected_energy_rate = (
            electrical_power(
                thermoelectric,
                current,
                temperatures[1],
                temperatures[0],
            )
            + cold_reservoir_heat
            + hot_reservoir_heat
            + cold_external_heat
            + hot_external_heat
        )
        self.assertAlmostEqual(stored_energy_rate, expected_energy_rate, places=12)

    def test_zero_current_shared_equilibrium_remains_constant(self):
        trajectory = integrate_temperature_dependent_contact_truth(
            self.reference.thermoelectric_parameters,
            self.reference.thermal_parameters,
            self.mismatch,
            TemperatureDependentColdContact(beta=0.16),
            initial_temperature=300.0,
            duration=4.0,
            time_step=0.37,
            current=0.0,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
        )

        self.assertIsInstance(trajectory, InterfaceMassTrajectory)
        for history in trajectory[1:]:
            self.assertTrue(all(value == 300.0 for value in history))

    def test_piecewise_transitions_are_exact_integration_times(self):
        current = PiecewiseConstantCurrent(
            transition_times=(0.35, 0.9),
            values=(0.0, 0.7, 0.0),
        )
        trajectory = integrate_temperature_dependent_contact_truth(
            self.reference.thermoelectric_parameters,
            self.reference.thermal_parameters,
            self.mismatch,
            TemperatureDependentColdContact(),
            initial_temperature=300.0,
            duration=1.2,
            time_step=0.4,
            current=current,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
        )

        self.assertIn(0.35, trajectory.time)
        self.assertIn(0.9, trajectory.time)
        self.assertEqual(trajectory.time[-1], 1.2)
        self.assertTrue(
            all(
                right > left
                for left, right in zip(trajectory.time, trajectory.time[1:])
            )
        )

    def test_rk4_time_step_refinement_converges(self):
        current = PiecewiseConstantCurrent(
            transition_times=(5.0, 25.0, 38.0, 58.0),
            values=(0.0, 1.0, 0.0, -0.8, 0.0),
        )

        def integrate(time_step):
            return integrate_temperature_dependent_contact_truth(
                self.reference.thermoelectric_parameters,
                self.reference.thermal_parameters,
                self.mismatch,
                TemperatureDependentColdContact(beta=0.12),
                initial_temperature=300.0,
                duration=80.0,
                time_step=time_step,
                current=current,
                cold_reservoir_temperature=300.0,
                hot_reservoir_temperature=300.0,
            )

        def common_grid_error(coarser, finer):
            finer_indices = {
                round(time, 9): index for index, time in enumerate(finer.time)
            }
            return max(
                abs(coarse_value - finer[node][finer_indices[round(time, 9)]])
                for node in range(1, len(coarser))
                for time, coarse_value in zip(coarser.time, coarser[node])
            )

        coarse = integrate(0.5)
        medium = integrate(0.25)
        fine = integrate(0.125)
        coarse_error = common_grid_error(coarse, medium)
        fine_error = common_grid_error(medium, fine)

        self.assertGreater(coarse_error, 8.0 * fine_error)
        self.assertLess(fine_error, 5.0e-6)


if __name__ == "__main__":
    unittest.main()
