from dataclasses import replace
import math
import unittest

from thermotwin.physics.distributed import (
    ConstantProperty,
    DistributedFaceThermalParameters,
    DistributedLegGeometry,
    DistributedThermoelectricMaterial,
    PiecewiseLinearProperty,
    distributed_leg_rhs,
    evaluate_distributed_state,
    linear_cell_temperatures,
    recommended_explicit_time_step,
)


class TemperaturePropertyTests(unittest.TestCase):
    def test_constant_property_integral_is_oriented(self):
        prop = ConstantProperty(2.5)
        self.assertEqual(prop.value(300.0), 2.5)
        self.assertEqual(prop.integral(290.0, 310.0), 50.0)
        self.assertEqual(prop.integral(310.0, 290.0), -50.0)

    def test_piecewise_linear_property_interpolates_and_integrates(self):
        prop = PiecewiseLinearProperty(
            temperatures=(290.0, 300.0, 320.0),
            values=(1.0, 3.0, 5.0),
        )
        self.assertEqual(prop.value(280.0), 1.0)
        self.assertEqual(prop.value(295.0), 2.0)
        self.assertEqual(prop.value(310.0), 4.0)
        self.assertEqual(prop.value(330.0), 5.0)
        self.assertAlmostEqual(prop.integral(290.0, 320.0), 100.0)
        self.assertAlmostEqual(prop.integral(320.0, 290.0), -100.0)

    def test_invalid_piecewise_basis_is_rejected(self):
        with self.assertRaises(ValueError):
            PiecewiseLinearProperty((300.0, 300.0), (1.0, 2.0))
        with self.assertRaises(ValueError):
            PiecewiseLinearProperty((290.0, 300.0), (1.0,))


class DistributedPhysicsTests(unittest.TestCase):
    def setUp(self):
        self.alpha = 200.0e-6
        self.resistivity = 1.0e-5
        self.conductivity = 1.5
        self.material = DistributedThermoelectricMaterial(
            seebeck_coefficient=ConstantProperty(self.alpha),
            electrical_resistivity=ConstantProperty(self.resistivity),
            thermal_conductivity=ConstantProperty(self.conductivity),
            mass_density=7700.0,
            specific_heat_capacity=150.0,
        )
        self.geometry = DistributedLegGeometry(
            length=1.5e-3,
            area=2.25e-6,
        )
        self.faces = DistributedFaceThermalParameters(
            cold_thermal_capacitance=0.05,
            hot_thermal_capacitance=0.07,
            cold_reservoir_conductance=0.01,
            hot_reservoir_conductance=0.02,
        )

    def _analytic_joule_profile(
        self,
        cold_temperature: float,
        hot_temperature: float,
        current: float,
        cell_count: int,
    ):
        current_density = current / self.geometry.area
        return tuple(
            cold_temperature
            + (hot_temperature - cold_temperature) * x / self.geometry.length
            + self.resistivity
            * current_density**2
            * x
            * (self.geometry.length - x)
            / (2.0 * self.conductivity)
            for x in (
                self.geometry.length * (index + 0.5) / cell_count
                for index in range(cell_count)
            )
        )

    def test_zero_current_equal_temperature_is_equilibrium(self):
        cells = (300.0,) * 6
        rates = distributed_leg_rhs(
            self.material,
            self.geometry,
            self.faces,
            cold_face_temperature=300.0,
            cell_temperatures=cells,
            hot_face_temperature=300.0,
            current=0.0,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
        )
        self.assertEqual(rates.cold_face, 0.0)
        self.assertEqual(rates.hot_face, 0.0)
        self.assertTrue(all(rate == 0.0 for rate in rates.cells))

    def test_passive_gradient_heats_cold_and_cools_hot(self):
        cells = linear_cell_temperatures(295.0, 305.0, 8)
        rates = distributed_leg_rhs(
            self.material,
            self.geometry,
            replace(
                self.faces,
                cold_reservoir_conductance=0.0,
                hot_reservoir_conductance=0.0,
            ),
            cold_face_temperature=295.0,
            cell_temperatures=cells,
            hot_face_temperature=305.0,
            current=0.0,
            cold_reservoir_temperature=295.0,
            hot_reservoir_temperature=305.0,
        )
        self.assertGreater(rates.cold_face, 0.0)
        self.assertLess(rates.hot_face, 0.0)
        self.assertTrue(all(abs(rate) < 1e-10 for rate in rates.cells))

    def test_positive_current_pumps_heat_and_initial_joule_heat_is_stored(self):
        cells = (300.0,) * 8
        diagnostics = evaluate_distributed_state(
            self.material,
            self.geometry,
            self.faces,
            cold_face_temperature=300.0,
            cell_temperatures=cells,
            hot_face_temperature=300.0,
            current=0.5,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
        )
        self.assertGreater(diagnostics.cold_side_heat, 0.0)
        self.assertGreater(diagnostics.hot_side_heat, 0.0)
        self.assertAlmostEqual(
            diagnostics.hot_side_heat,
            diagnostics.cold_side_heat,
        )
        self.assertGreater(diagnostics.electrical_power, 0.0)
        self.assertAlmostEqual(
            diagnostics.stored_energy_rate,
            diagnostics.electrical_power,
        )

    def test_constant_property_steady_solution_recovers_lumped_face_heats(self):
        cold_temperature = 295.0
        hot_temperature = 305.0
        current = 0.8
        cells = self._analytic_joule_profile(
            cold_temperature, hot_temperature, current, 10
        )
        diagnostics = evaluate_distributed_state(
            self.material,
            self.geometry,
            self.faces,
            cold_face_temperature=cold_temperature,
            cell_temperatures=cells,
            hot_face_temperature=hot_temperature,
            current=current,
            cold_reservoir_temperature=cold_temperature,
            hot_reservoir_temperature=hot_temperature,
            cold_external_heat=0.0,
            hot_external_heat=0.0,
        )
        resistance = (
            self.resistivity * self.geometry.length / self.geometry.area
        )
        conductance = (
            self.conductivity * self.geometry.area / self.geometry.length
        )
        expected_cold = (
            self.alpha * current * cold_temperature
            - 0.5 * current**2 * resistance
            - conductance * (hot_temperature - cold_temperature)
        )
        expected_hot = (
            self.alpha * current * hot_temperature
            + 0.5 * current**2 * resistance
            - conductance * (hot_temperature - cold_temperature)
        )
        expected_voltage = (
            self.alpha * (hot_temperature - cold_temperature)
            + current * resistance
        )
        self.assertAlmostEqual(diagnostics.cold_side_heat, expected_cold, places=11)
        self.assertAlmostEqual(diagnostics.hot_side_heat, expected_hot, places=11)
        self.assertAlmostEqual(diagnostics.voltage, expected_voltage, places=13)

        balanced_rates = distributed_leg_rhs(
            self.material,
            self.geometry,
            self.faces,
            cold_face_temperature=cold_temperature,
            cell_temperatures=cells,
            hot_face_temperature=hot_temperature,
            current=current,
            cold_reservoir_temperature=cold_temperature,
            hot_reservoir_temperature=hot_temperature,
            cold_external_heat=expected_cold,
            hot_external_heat=-expected_hot,
        )
        self.assertTrue(all(abs(rate) < 1e-10 for rate in balanced_rates.cells))
        self.assertAlmostEqual(balanced_rates.cold_face, 0.0, places=11)
        self.assertAlmostEqual(balanced_rates.hot_face, 0.0, places=11)

    def test_semidiscrete_whole_system_energy_closes_to_roundoff(self):
        diagnostics = evaluate_distributed_state(
            self.material,
            self.geometry,
            self.faces,
            cold_face_temperature=296.0,
            cell_temperatures=(297.0, 298.5, 300.0, 301.0),
            hot_face_temperature=303.0,
            current=-0.4,
            cold_reservoir_temperature=294.0,
            hot_reservoir_temperature=306.0,
            cold_external_heat=0.03,
            hot_external_heat=-0.01,
        )
        self.assertAlmostEqual(
            diagnostics.stored_energy_rate,
            diagnostics.expected_energy_rate,
            places=13,
        )
        self.assertLess(abs(diagnostics.energy_balance_residual), 1e-13)

    def test_recommended_step_shrinks_quadratically_with_cell_size(self):
        coarse = recommended_explicit_time_step(
            self.material,
            self.geometry,
            cell_count=5,
            temperature_range=(290.0, 310.0),
        )
        fine = recommended_explicit_time_step(
            self.material,
            self.geometry,
            cell_count=10,
            temperature_range=(290.0, 310.0),
        )
        self.assertAlmostEqual(fine / coarse, 0.25)


if __name__ == "__main__":
    unittest.main()
