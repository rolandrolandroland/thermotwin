from dataclasses import replace
import math
import unittest

from thermotwin.core.controls import PiecewiseConstantCurrent
from thermotwin.physics.four_node import integrate_four_node_contact
from thermotwin.physics.thermoelectric import electrical_power
from thermotwin.simulation.four_node_experiments import (
    constant_current_contact_reference_experiment,
)
from thermotwin.simulation.interface_mass_mismatch import InterfaceMassMismatch
from thermotwin.simulation.temporary_face_sensor import (
    TemporaryFaceSensor,
    four_node_temporary_face_sensor_rhs,
    integrate_four_node_temporary_face_sensor,
    integrate_interface_mass_temporary_face_sensor,
    interface_mass_temporary_face_sensor_rhs,
    temporary_face_sensor_coupling,
)


class TemporaryFaceSensorTests(unittest.TestCase):
    def setUp(self):
        self.reference = constant_current_contact_reference_experiment()
        self.thermoelectric = self.reference.thermoelectric_parameters
        self.thermal = self.reference.thermal_parameters
        self.sensor = TemporaryFaceSensor()
        self.mismatch = InterfaceMassMismatch(thermal_capacitance=20.0)

    def test_default_sensor_and_parameter_validation(self):
        self.assertEqual(self.sensor.thermal_capacitance, 5.0)
        self.assertEqual(self.sensor.response_time_constant, 2.5)
        self.assertEqual(self.sensor.thermal_conductance, 2.0)

        for field in ("thermal_capacitance", "response_time_constant"):
            for value in (0.0, -1.0, math.inf, math.nan, "invalid"):
                with self.subTest(field=field, value=value):
                    values = {
                        "thermal_capacitance": 5.0,
                        "response_time_constant": 2.5,
                        field: value,
                    }
                    with self.assertRaisesRegex(
                        ValueError, "finite and positive"
                    ):
                        TemporaryFaceSensor(**values)

    def test_coupling_is_equal_and_opposite(self):
        coupling = temporary_face_sensor_coupling(
            self.sensor,
            cold_face_temperature=310.0,
            face_sensor_temperature=300.0,
            cold_face_thermal_capacitance=50.0,
        )

        self.assertEqual(coupling.heat_into_sensor, 20.0)
        self.assertEqual(coupling.cold_face_rate_adjustment, -0.4)
        self.assertEqual(coupling.face_sensor_rate, 4.0)
        self.assertEqual(
            50.0 * coupling.cold_face_rate_adjustment
            + self.sensor.thermal_capacitance * coupling.face_sensor_rate,
            0.0,
        )

    def test_equal_temperature_zero_current_is_equilibrium(self):
        common = dict(
            cold_face_temperature=300.0,
            hot_face_temperature=300.0,
            cold_exchanger_temperature=300.0,
            hot_exchanger_temperature=300.0,
            face_sensor_temperature=300.0,
            current=0.0,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
        )
        four = four_node_temporary_face_sensor_rhs(
            self.thermoelectric,
            self.thermal,
            self.sensor,
            **common,
        )
        five = interface_mass_temporary_face_sensor_rhs(
            self.thermoelectric,
            self.thermal,
            self.mismatch,
            self.sensor,
            cold_interface_temperature=300.0,
            **common,
        )

        self.assertEqual(four, (0.0, 0.0, 0.0, 0.0, 0.0))
        self.assertEqual(five, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    def test_whole_system_energy_rate_conserves_sensor_coupling(self):
        inputs = dict(
            cold_face_temperature=294.0,
            hot_face_temperature=308.0,
            cold_exchanger_temperature=299.0,
            hot_exchanger_temperature=303.0,
            face_sensor_temperature=301.0,
            current=1.2,
            cold_reservoir_temperature=301.0,
            hot_reservoir_temperature=310.0,
            cold_external_heat=3.0,
            hot_external_heat=-4.0,
        )
        four = four_node_temporary_face_sensor_rhs(
            self.thermoelectric,
            self.thermal,
            self.sensor,
            **inputs,
        )
        five = interface_mass_temporary_face_sensor_rhs(
            self.thermoelectric,
            self.thermal,
            self.mismatch,
            self.sensor,
            cold_interface_temperature=296.0,
            **inputs,
        )
        expected = (
            self.thermal.cold_reservoir_conductance
            * (
                inputs["cold_reservoir_temperature"]
                - inputs["cold_exchanger_temperature"]
            )
            + self.thermal.hot_reservoir_conductance
            * (
                inputs["hot_reservoir_temperature"]
                - inputs["hot_exchanger_temperature"]
            )
            + inputs["cold_external_heat"]
            + inputs["hot_external_heat"]
            + electrical_power(
                self.thermoelectric,
                inputs["current"],
                inputs["hot_face_temperature"],
                inputs["cold_face_temperature"],
            )
        )
        four_stored = (
            self.thermal.cold_face_thermal_capacitance * four.cold_face
            + self.thermal.hot_face_thermal_capacitance * four.hot_face
            + self.thermal.cold_exchanger_thermal_capacitance
            * four.cold_exchanger
            + self.thermal.hot_exchanger_thermal_capacitance
            * four.hot_exchanger
            + self.sensor.thermal_capacitance * four.face_sensor
        )
        five_stored = (
            self.thermal.cold_face_thermal_capacitance * five.cold_face
            + self.thermal.hot_face_thermal_capacitance * five.hot_face
            + self.mismatch.thermal_capacitance * five.cold_interface
            + self.thermal.cold_exchanger_thermal_capacitance
            * five.cold_exchanger
            + self.thermal.hot_exchanger_thermal_capacitance
            * five.hot_exchanger
            + self.sensor.thermal_capacitance * five.face_sensor
        )

        self.assertAlmostEqual(four_stored, expected)
        self.assertAlmostEqual(five_stored, expected)

    @staticmethod
    def _four_node_loaded(reference, sensor, current, time_step):
        return integrate_four_node_temporary_face_sensor(
            reference.thermoelectric_parameters,
            reference.thermal_parameters,
            sensor,
            initial_cold_face_temperature=300.0,
            initial_hot_face_temperature=300.0,
            initial_cold_exchanger_temperature=300.0,
            initial_hot_exchanger_temperature=300.0,
            initial_face_sensor_temperature=300.0,
            duration=8.0,
            time_step=time_step,
            current=current,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
        )

    def test_vanishing_sensor_mass_recovers_unloaded_device(self):
        current = PiecewiseConstantCurrent.pulse(
            start_time=1.3,
            end_time=6.7,
            pulse_current=0.8,
        )
        unloaded = integrate_four_node_contact(
            self.thermoelectric,
            self.thermal,
            initial_cold_face_temperature=300.0,
            initial_hot_face_temperature=300.0,
            initial_cold_exchanger_temperature=300.0,
            initial_hot_exchanger_temperature=300.0,
            duration=8.0,
            time_step=0.05,
            current=current,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
        )
        loaded = self._four_node_loaded(
            self.reference,
            TemporaryFaceSensor(
                thermal_capacitance=1.0e-6,
                response_time_constant=2.5,
            ),
            current,
            0.05,
        )

        self.assertEqual(loaded.time, unloaded.time)
        for loaded_values, unloaded_values in zip(
            (
                loaded.cold_face,
                loaded.hot_face,
                loaded.cold_exchanger,
                loaded.hot_exchanger,
            ),
            (
                unloaded.cold_face,
                unloaded.hot_face,
                unloaded.cold_exchanger,
                unloaded.hot_exchanger,
            ),
        ):
            self.assertLess(
                max(
                    abs(left - right)
                    for left, right in zip(loaded_values, unloaded_values)
                ),
                1.0e-6,
            )

    def test_sensor_temperature_is_a_distinct_lagged_state(self):
        current = PiecewiseConstantCurrent.pulse(
            start_time=1.0,
            end_time=7.0,
            pulse_current=0.8,
        )
        trajectory = self._four_node_loaded(
            self.reference,
            self.sensor,
            current,
            0.05,
        )
        skeleton = trajectory.four_node_projection

        self.assertEqual(skeleton.time, trajectory.time)
        self.assertGreater(
            max(
                abs(face - measured)
                for face, measured in zip(
                    trajectory.cold_face, trajectory.face_sensor
                )
            ),
            0.05,
        )
        self.assertGreater(min(trajectory.face_sensor), min(trajectory.cold_face))

    def test_fast_sensor_limit_matches_lumped_added_capacitance(self):
        sensor = TemporaryFaceSensor(
            thermal_capacitance=5.0,
            response_time_constant=0.02,
        )
        current = PiecewiseConstantCurrent.constant(0.8)
        loaded = integrate_four_node_temporary_face_sensor(
            self.thermoelectric,
            self.thermal,
            sensor,
            initial_cold_face_temperature=300.0,
            initial_hot_face_temperature=300.0,
            initial_cold_exchanger_temperature=300.0,
            initial_hot_exchanger_temperature=300.0,
            initial_face_sensor_temperature=300.0,
            duration=2.0,
            time_step=0.001,
            current=current,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
        )
        lumped_thermal = replace(
            self.thermal,
            cold_face_thermal_capacitance=(
                self.thermal.cold_face_thermal_capacitance
                + sensor.thermal_capacitance
            ),
        )
        lumped = integrate_four_node_contact(
            self.thermoelectric,
            lumped_thermal,
            initial_cold_face_temperature=300.0,
            initial_hot_face_temperature=300.0,
            initial_cold_exchanger_temperature=300.0,
            initial_hot_exchanger_temperature=300.0,
            duration=2.0,
            time_step=0.001,
            current=current,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
        )

        self.assertLess(abs(loaded.cold_face[-1] - lumped.cold_face[-1]), 0.005)
        self.assertLess(abs(loaded.face_sensor[-1] - loaded.cold_face[-1]), 0.005)

    def test_both_integrators_are_switch_aligned_and_converge(self):
        current = PiecewiseConstantCurrent(
            transition_times=(1.3, 4.7),
            values=(0.0, 0.8, 0.0),
        )

        def four(step):
            return self._four_node_loaded(
                self.reference, self.sensor, current, step
            )

        def five(step):
            return integrate_interface_mass_temporary_face_sensor(
                self.thermoelectric,
                self.thermal,
                self.mismatch,
                self.sensor,
                initial_cold_face_temperature=300.0,
                initial_hot_face_temperature=300.0,
                initial_cold_interface_temperature=300.0,
                initial_cold_exchanger_temperature=300.0,
                initial_hot_exchanger_temperature=300.0,
                initial_face_sensor_temperature=300.0,
                duration=8.0,
                time_step=step,
                current=current,
                cold_reservoir_temperature=300.0,
                hot_reservoir_temperature=300.0,
            )

        for name, simulate in (("four", four), ("five", five)):
            with self.subTest(model=name):
                coarse = simulate(0.4)
                medium = simulate(0.2)
                fine = simulate(0.1)
                self.assertIn(1.3, coarse.time)
                self.assertIn(4.7, coarse.time)
                coarse_error = max(
                    abs(left[-1] - right[-1])
                    for left, right in zip(coarse[1:], fine[1:])
                )
                medium_error = max(
                    abs(left[-1] - right[-1])
                    for left, right in zip(medium[1:], fine[1:])
                )
                self.assertLess(medium_error, coarse_error)


if __name__ == "__main__":
    unittest.main()
