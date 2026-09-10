import unittest
from dataclasses import replace

from thermotwin.core.controls import PiecewiseConstantCurrent
from thermotwin.physics.thermoelectric import electrical_power
from thermotwin.simulation.four_node_experiments import (
    constant_current_contact_reference_experiment,
)
from thermotwin.simulation.interface_mass_mismatch import InterfaceMassMismatch
from thermotwin.simulation.operating_realism import (
    integrate_temperature_dependent_contact_temporary_sensor,
    temperature_dependent_contact_temporary_sensor_rhs,
)
from thermotwin.simulation.temperature_dependent_contact import (
    TemperatureDependentColdContact,
)
from thermotwin.simulation.temporary_face_sensor import TemporaryFaceSensor


class OperatingRealismCompositionTests(unittest.TestCase):
    def setUp(self):
        reference = constant_current_contact_reference_experiment()
        self.thermal = reference.thermal_parameters
        self.thermoelectric = replace(
            reference.thermoelectric_parameters,
            electrical_resistance=(
                reference.thermoelectric_parameters.electrical_resistance + 0.10
            ),
        )
        self.mismatch = InterfaceMassMismatch(thermal_capacitance=20.0)
        self.contact = TemperatureDependentColdContact(beta=0.12)
        self.sensor = TemporaryFaceSensor(5.0, 2.5)

    def test_composed_six_state_rhs_closes_physical_energy_balance(self):
        temperatures = (295.0, 306.0, 297.0, 300.0, 303.0, 299.0)
        current = 0.9
        cold_reservoir = 294.0
        hot_reservoir = 308.0
        cold_external = 0.3
        hot_external = -0.1
        rates = temperature_dependent_contact_temporary_sensor_rhs(
            self.thermoelectric,
            self.thermal,
            self.mismatch,
            self.contact,
            self.sensor,
            cold_face_temperature=temperatures[0],
            hot_face_temperature=temperatures[1],
            cold_interface_temperature=temperatures[2],
            cold_exchanger_temperature=temperatures[3],
            hot_exchanger_temperature=temperatures[4],
            face_sensor_temperature=temperatures[5],
            current=current,
            cold_reservoir_temperature=cold_reservoir,
            hot_reservoir_temperature=hot_reservoir,
            cold_external_heat=cold_external,
            hot_external_heat=hot_external,
        )
        stored = sum(
            capacitance * rate
            for capacitance, rate in zip(
                (
                    self.thermal.cold_face_thermal_capacitance,
                    self.thermal.hot_face_thermal_capacitance,
                    self.mismatch.thermal_capacitance,
                    self.thermal.cold_exchanger_thermal_capacitance,
                    self.thermal.hot_exchanger_thermal_capacitance,
                    self.sensor.thermal_capacitance,
                ),
                rates,
            )
        )
        expected = (
            electrical_power(
                self.thermoelectric,
                current,
                temperatures[1],
                temperatures[0],
            )
            + self.thermal.cold_reservoir_conductance
            * (cold_reservoir - temperatures[3])
            + self.thermal.hot_reservoir_conductance
            * (hot_reservoir - temperatures[4])
            + cold_external
            + hot_external
        )
        self.assertAlmostEqual(stored, expected, places=12)

    def test_composed_equilibrium_and_switch_alignment(self):
        zero = integrate_temperature_dependent_contact_temporary_sensor(
            self.thermoelectric,
            self.thermal,
            self.mismatch,
            self.contact,
            self.sensor,
            initial_temperature=300.0,
            duration=1.0,
            time_step=0.3,
            current=0.0,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
        )
        self.assertTrue(all(value == 300.0 for history in zero[1:] for value in history))

        schedule = PiecewiseConstantCurrent(
            transition_times=(0.35, 0.9),
            values=(0.0, 0.8, 0.0),
        )
        switched = integrate_temperature_dependent_contact_temporary_sensor(
            self.thermoelectric,
            self.thermal,
            self.mismatch,
            self.contact,
            self.sensor,
            initial_temperature=300.0,
            duration=1.2,
            time_step=0.4,
            current=schedule,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
        )
        self.assertIn(0.35, switched.time)
        self.assertIn(0.9, switched.time)
        self.assertEqual(switched.time[-1], 1.2)


if __name__ == "__main__":
    unittest.main()
