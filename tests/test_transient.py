import unittest

from thermotwin import (
    IntegrationDivergenceError,
    PiecewiseConstantCurrent,
    ThermoelectricParameters,
    TwoNodeThermalParameters,
    electrical_power,
    integrate_two_node,
    two_node_rhs,
    two_node_steady_state,
)


class TwoNodeTransientTests(unittest.TestCase):
    def setUp(self):
        self.thermoelectric = ThermoelectricParameters(
            seebeck_coefficient=0.05,
            electrical_resistance=2.0,
            thermal_conductance=0.5,
        )
        self.thermal = TwoNodeThermalParameters(
            cold_thermal_capacitance=100.0,
            hot_thermal_capacitance=200.0,
            cold_reservoir_conductance=2.0,
            hot_reservoir_conductance=4.0,
        )

    def test_rhs_matches_the_agreed_node_balances(self):
        rates = two_node_rhs(
            self.thermoelectric,
            self.thermal,
            cold_temperature=300.0,
            hot_temperature=320.0,
            current=3.0,
            cold_reservoir_temperature=295.0,
            hot_reservoir_temperature=300.0,
            cold_external_heat=5.0,
            hot_external_heat=7.0,
        )

        # Q_c = 26 W and Q_h = 47 W for this operating point.
        # Cold net heat: -10 + 5 - 26 = -31 W.
        # Hot net heat: -80 + 7 + 47 = -26 W.
        self.assertAlmostEqual(rates.cold, -31.0 / 100.0)
        self.assertAlmostEqual(rates.hot, -26.0 / 200.0)
        self.assertEqual(tuple(rates), (rates.cold, rates.hot))

    def test_total_stored_energy_rate_matches_all_external_inputs(self):
        cold_temperature = 300.0
        hot_temperature = 320.0
        current = 3.0
        cold_reservoir_temperature = 295.0
        hot_reservoir_temperature = 300.0
        cold_external_heat = 5.0
        hot_external_heat = 7.0

        rates = two_node_rhs(
            self.thermoelectric,
            self.thermal,
            cold_temperature=cold_temperature,
            hot_temperature=hot_temperature,
            current=current,
            cold_reservoir_temperature=cold_reservoir_temperature,
            hot_reservoir_temperature=hot_reservoir_temperature,
            cold_external_heat=cold_external_heat,
            hot_external_heat=hot_external_heat,
        )

        stored_energy_rate = (
            self.thermal.cold_thermal_capacitance * rates.cold
            + self.thermal.hot_thermal_capacitance * rates.hot
        )
        expected_energy_rate = (
            self.thermal.cold_reservoir_conductance
            * (cold_reservoir_temperature - cold_temperature)
            + self.thermal.hot_reservoir_conductance
            * (hot_reservoir_temperature - hot_temperature)
            + cold_external_heat
            + hot_external_heat
            + electrical_power(
                self.thermoelectric,
                current,
                hot_temperature,
                cold_temperature,
            )
        )
        self.assertAlmostEqual(stored_energy_rate, expected_energy_rate)

    def test_equal_temperatures_with_no_current_or_load_are_equilibrium(self):
        rates = two_node_rhs(
            self.thermoelectric,
            self.thermal,
            cold_temperature=300.0,
            hot_temperature=300.0,
            current=0.0,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
        )

        self.assertEqual(rates.cold, 0.0)
        self.assertEqual(rates.hot, 0.0)

    def test_passive_module_conduction_warms_cold_and_cools_hot_node(self):
        insulated_nodes = TwoNodeThermalParameters(
            cold_thermal_capacitance=100.0,
            hot_thermal_capacitance=200.0,
            cold_reservoir_conductance=0.0,
            hot_reservoir_conductance=0.0,
        )
        rates = two_node_rhs(
            self.thermoelectric,
            insulated_nodes,
            cold_temperature=300.0,
            hot_temperature=320.0,
            current=0.0,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=320.0,
        )

        self.assertGreater(rates.cold, 0.0)
        self.assertLess(rates.hot, 0.0)
        self.assertAlmostEqual(
            insulated_nodes.cold_thermal_capacitance * rates.cold
            + insulated_nodes.hot_thermal_capacitance * rates.hot,
            0.0,
        )

    def test_positive_refrigeration_current_cools_cold_and_heats_hot_node(self):
        insulated_nodes = TwoNodeThermalParameters(
            cold_thermal_capacitance=100.0,
            hot_thermal_capacitance=200.0,
            cold_reservoir_conductance=0.0,
            hot_reservoir_conductance=0.0,
        )
        rates = two_node_rhs(
            self.thermoelectric,
            insulated_nodes,
            cold_temperature=300.0,
            hot_temperature=300.0,
            current=1.0,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
        )

        self.assertLess(rates.cold, 0.0)
        self.assertGreater(rates.hot, 0.0)

    def test_doubling_capacitance_halves_rate_for_same_net_heat(self):
        base = TwoNodeThermalParameters(
            cold_thermal_capacitance=100.0,
            hot_thermal_capacitance=200.0,
            cold_reservoir_conductance=0.0,
            hot_reservoir_conductance=0.0,
        )
        doubled = TwoNodeThermalParameters(
            cold_thermal_capacitance=200.0,
            hot_thermal_capacitance=400.0,
            cold_reservoir_conductance=0.0,
            hot_reservoir_conductance=0.0,
        )
        inputs = dict(
            cold_temperature=300.0,
            hot_temperature=300.0,
            current=0.0,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
            cold_external_heat=10.0,
            hot_external_heat=20.0,
        )

        base_rates = two_node_rhs(
            self.thermoelectric, base, **inputs
        )
        doubled_rates = two_node_rhs(
            self.thermoelectric, doubled, **inputs
        )

        self.assertAlmostEqual(doubled_rates.cold, 0.5 * base_rates.cold)
        self.assertAlmostEqual(doubled_rates.hot, 0.5 * base_rates.hot)

    def test_invalid_thermal_parameters_are_rejected(self):
        invalid_cases = (
            dict(
                cold_thermal_capacitance=0.0,
                hot_thermal_capacitance=1.0,
                cold_reservoir_conductance=0.0,
                hot_reservoir_conductance=0.0,
            ),
            dict(
                cold_thermal_capacitance=1.0,
                hot_thermal_capacitance=-1.0,
                cold_reservoir_conductance=0.0,
                hot_reservoir_conductance=0.0,
            ),
            dict(
                cold_thermal_capacitance=1.0,
                hot_thermal_capacitance=1.0,
                cold_reservoir_conductance=-1.0,
                hot_reservoir_conductance=0.0,
            ),
            dict(
                cold_thermal_capacitance=1.0,
                hot_thermal_capacitance=1.0,
                cold_reservoir_conductance=0.0,
                hot_reservoir_conductance=-1.0,
            ),
        )

        for values in invalid_cases:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    TwoNodeThermalParameters(**values)

    def test_integrator_reproduces_constant_heat_rates_and_partial_step(self):
        no_thermoelectric_effects = ThermoelectricParameters(
            seebeck_coefficient=0.0,
            electrical_resistance=0.0,
            thermal_conductance=0.0,
        )
        insulated_nodes = TwoNodeThermalParameters(
            cold_thermal_capacitance=100.0,
            hot_thermal_capacitance=200.0,
            cold_reservoir_conductance=0.0,
            hot_reservoir_conductance=0.0,
        )

        trajectory = integrate_two_node(
            no_thermoelectric_effects,
            insulated_nodes,
            initial_cold_temperature=300.0,
            initial_hot_temperature=310.0,
            duration=2.5,
            time_step=1.0,
            current=0.0,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=310.0,
            cold_external_heat=10.0,
            hot_external_heat=-20.0,
        )

        self.assertEqual(trajectory.time, (0.0, 1.0, 2.0, 2.5))
        self.assertAlmostEqual(trajectory.cold[-1], 300.25)
        self.assertAlmostEqual(trajectory.hot[-1], 309.75)

    def test_integrator_preserves_an_equilibrium(self):
        trajectory = integrate_two_node(
            self.thermoelectric,
            self.thermal,
            initial_cold_temperature=300.0,
            initial_hot_temperature=300.0,
            duration=1.0,
            time_step=0.3,
            current=0.0,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
        )

        self.assertEqual(trajectory.time[-1], 1.0)
        self.assertTrue(all(value == 300.0 for value in trajectory.cold))
        self.assertTrue(all(value == 300.0 for value in trajectory.hot))

    def test_integrator_matches_initial_current_step_direction(self):
        trajectory = integrate_two_node(
            self.thermoelectric,
            self.thermal,
            initial_cold_temperature=300.0,
            initial_hot_temperature=300.0,
            duration=0.1,
            time_step=0.1,
            current=1.0,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
        )

        self.assertLess(trajectory.cold[-1], trajectory.cold[0])
        self.assertGreater(trajectory.hot[-1], trajectory.hot[0])

    def test_integrator_conserves_energy_for_passive_insulated_nodes(self):
        insulated_nodes = TwoNodeThermalParameters(
            cold_thermal_capacitance=100.0,
            hot_thermal_capacitance=200.0,
            cold_reservoir_conductance=0.0,
            hot_reservoir_conductance=0.0,
        )
        trajectory = integrate_two_node(
            self.thermoelectric,
            insulated_nodes,
            initial_cold_temperature=300.0,
            initial_hot_temperature=320.0,
            duration=10.0,
            time_step=0.5,
            current=0.0,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=320.0,
        )
        initial_energy = (
            insulated_nodes.cold_thermal_capacitance
            * trajectory.cold[0]
            + insulated_nodes.hot_thermal_capacitance
            * trajectory.hot[0]
        )

        for cold_temperature, hot_temperature in zip(
            trajectory.cold, trajectory.hot
        ):
            stored_energy = (
                insulated_nodes.cold_thermal_capacitance * cold_temperature
                + insulated_nodes.hot_thermal_capacitance * hot_temperature
            )
            self.assertAlmostEqual(stored_energy, initial_energy)

        self.assertGreater(trajectory.cold[-1], trajectory.cold[0])
        self.assertLess(trajectory.hot[-1], trajectory.hot[0])

    def test_integrator_rejects_invalid_time_settings(self):
        base_inputs = dict(
            initial_cold_temperature=300.0,
            initial_hot_temperature=300.0,
            current=0.0,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
        )
        invalid_cases = (
            dict(duration=-1.0, time_step=0.1),
            dict(duration=float("nan"), time_step=0.1),
            dict(duration=float("inf"), time_step=0.1),
            dict(duration=1.0, time_step=0.0),
            dict(duration=1.0, time_step=-0.1),
            dict(duration=1.0, time_step=float("nan")),
            dict(duration=1.0, time_step=float("inf")),
        )

        for time_settings in invalid_cases:
            with self.subTest(time_settings=time_settings):
                with self.assertRaises(ValueError):
                    integrate_two_node(
                        self.thermoelectric,
                        self.thermal,
                        **base_inputs,
                        **time_settings,
                    )

    def test_steady_state_solves_both_node_balances(self):
        inputs = dict(
            current=1.0,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
            cold_external_heat=3.0,
            hot_external_heat=-4.0,
        )
        steady_state = two_node_steady_state(
            self.thermoelectric,
            self.thermal,
            **inputs,
        )

        rates = two_node_rhs(
            self.thermoelectric,
            self.thermal,
            cold_temperature=steady_state.cold,
            hot_temperature=steady_state.hot,
            **inputs,
        )

        self.assertAlmostEqual(rates.cold, 0.0)
        self.assertAlmostEqual(rates.hot, 0.0)

    def test_steady_state_matches_decoupled_reservoir_balances(self):
        no_thermoelectric_effects = ThermoelectricParameters(
            seebeck_coefficient=0.0,
            electrical_resistance=0.0,
            thermal_conductance=0.0,
        )
        steady_state = two_node_steady_state(
            no_thermoelectric_effects,
            self.thermal,
            current=0.0,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=310.0,
            cold_external_heat=10.0,
            hot_external_heat=-20.0,
        )

        self.assertAlmostEqual(steady_state.cold, 305.0)
        self.assertAlmostEqual(steady_state.hot, 305.0)

    def test_long_integration_approaches_independent_steady_state(self):
        inputs = dict(
            current=1.0,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
        )
        steady_state = two_node_steady_state(
            self.thermoelectric,
            self.thermal,
            **inputs,
        )
        trajectory = integrate_two_node(
            self.thermoelectric,
            self.thermal,
            initial_cold_temperature=300.0,
            initial_hot_temperature=300.0,
            duration=600.0,
            time_step=0.5,
            **inputs,
        )

        self.assertAlmostEqual(
            trajectory.cold[-1], steady_state.cold, places=4
        )
        self.assertAlmostEqual(
            trajectory.hot[-1], steady_state.hot, places=4
        )

    def test_steady_state_rejects_nonunique_insulated_system(self):
        no_thermoelectric_effects = ThermoelectricParameters(
            seebeck_coefficient=0.0,
            electrical_resistance=0.0,
            thermal_conductance=0.0,
        )
        insulated_nodes = TwoNodeThermalParameters(
            cold_thermal_capacitance=100.0,
            hot_thermal_capacitance=200.0,
            cold_reservoir_conductance=0.0,
            hot_reservoir_conductance=0.0,
        )

        with self.assertRaises(ValueError):
            two_node_steady_state(
                no_thermoelectric_effects,
                insulated_nodes,
                current=0.0,
                cold_reservoir_temperature=300.0,
                hot_reservoir_temperature=300.0,
            )

    def test_steady_state_rejects_sub_absolute_zero_solution(self):
        with self.assertRaisesRegex(ValueError, "positive-kelvin"):
            two_node_steady_state(
                self.thermoelectric,
                self.thermal,
                current=91.0,
                cold_reservoir_temperature=300.0,
                hot_reservoir_temperature=300.0,
            )

    def test_numerical_overflow_is_reported_as_integration_divergence(self):
        with self.assertRaisesRegex(
            IntegrationDivergenceError,
            "integration overflowed.*reduce the time step",
        ):
            integrate_two_node(
                self.thermoelectric,
                self.thermal,
                initial_cold_temperature=300.0,
                initial_hot_temperature=300.0,
                duration=1.0,
                time_step=1.0,
                current=1e308,
                cold_reservoir_temperature=300.0,
                hot_reservoir_temperature=300.0,
            )

    def test_constant_schedule_matches_scalar_current(self):
        inputs = dict(
            initial_cold_temperature=300.0,
            initial_hot_temperature=300.0,
            duration=10.0,
            time_step=0.3,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
        )

        scalar_trajectory = integrate_two_node(
            self.thermoelectric,
            self.thermal,
            current=1.0,
            **inputs,
        )
        scheduled_trajectory = integrate_two_node(
            self.thermoelectric,
            self.thermal,
            current=PiecewiseConstantCurrent.constant(1.0),
            **inputs,
        )

        self.assertEqual(scheduled_trajectory, scalar_trajectory)

    def test_pulse_boundaries_split_steps_and_integrate_exact_heating(self):
        joule_only = ThermoelectricParameters(
            seebeck_coefficient=0.0,
            electrical_resistance=2.0,
            thermal_conductance=0.0,
        )
        insulated_nodes = TwoNodeThermalParameters(
            cold_thermal_capacitance=1.0,
            hot_thermal_capacitance=1.0,
            cold_reservoir_conductance=0.0,
            hot_reservoir_conductance=0.0,
        )
        pulse = PiecewiseConstantCurrent.pulse(
            start_time=1.0,
            end_time=3.0,
            pulse_current=2.0,
        )

        trajectory = integrate_two_node(
            joule_only,
            insulated_nodes,
            initial_cold_temperature=300.0,
            initial_hot_temperature=300.0,
            duration=4.0,
            time_step=3.0,
            current=pulse,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
        )

        self.assertEqual(trajectory.time, (0.0, 1.0, 3.0, 4.0))
        self.assertEqual(trajectory.cold, (300.0, 300.0, 308.0, 308.0))
        self.assertEqual(trajectory.hot, (300.0, 300.0, 308.0, 308.0))


if __name__ == "__main__":
    unittest.main()
