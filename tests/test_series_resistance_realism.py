import unittest
from dataclasses import replace

from thermotwin.core.controls import PiecewiseConstantCurrent
from thermotwin.inference.experiment_selection import candidate_current
from thermotwin.physics.four_node import four_node_contact_rhs
from thermotwin.physics.thermoelectric import (
    electrical_power,
    joule_heating,
    voltage,
)
from thermotwin.simulation.four_node_experiments import (
    constant_current_contact_reference_experiment,
)
from thermotwin.simulation.interface_mass_mismatch import (
    InterfaceMassMismatch,
    interface_mass_rhs,
)
from thermotwin.studies.sensor_model_discrimination import (
    COLD_EXCHANGER,
    FIVE_STATE_MODEL,
    FOUR_STATE_MODEL,
    HOT_EXCHANGER,
    VOLTAGE,
    SensorDiscriminationConfig,
    _simulate_observables,
)


class SeriesResistanceRealismTests(unittest.TestCase):
    def setUp(self):
        self.config = SensorDiscriminationConfig(trial_count=1)
        self.current = candidate_current(0.8, 20.0)
        self.physical_values = self.config.fit.nominal_values

    def simulate(
        self,
        model_name,
        current=None,
        *,
        series_resistance=0.0,
        config=None,
    ):
        return _simulate_observables(
            model_name,
            self.current if current is None else current,
            (COLD_EXCHANGER, HOT_EXCHANGER, VOLTAGE),
            self.physical_values,
            (
                self.config.interface_mass_nominal
                if model_name == FIVE_STATE_MODEL
                else None
            ),
            self.config if config is None else config,
            series_resistance=series_resistance,
        )

    def test_zero_series_resistance_is_exactly_the_existing_simulation(self):
        for model_name in (FOUR_STATE_MODEL, FIVE_STATE_MODEL):
            with self.subTest(model_name=model_name):
                legacy = _simulate_observables(
                    model_name,
                    self.current,
                    (COLD_EXCHANGER, HOT_EXCHANGER, VOLTAGE),
                    self.physical_values,
                    (
                        self.config.interface_mass_nominal
                        if model_name == FIVE_STATE_MODEL
                        else None
                    ),
                    self.config,
                )
                explicit_zero = self.simulate(model_name, series_resistance=0.0)
                self.assertEqual(explicit_zero, legacy)

    def test_series_voltage_drop_is_odd_and_added_joule_heat_is_even(self):
        series_resistance = 0.10
        current_magnitude = 0.8
        base = constant_current_contact_reference_experiment().thermoelectric_parameters
        effective = replace(
            base,
            electrical_resistance=base.electrical_resistance + series_resistance,
        )

        positive_drop = voltage(effective, current_magnitude, 305.0, 295.0) - voltage(
            base, current_magnitude, 305.0, 295.0
        )
        negative_drop = voltage(effective, -current_magnitude, 305.0, 295.0) - voltage(
            base, -current_magnitude, 305.0, 295.0
        )
        positive_joule = joule_heating(effective, current_magnitude) - joule_heating(
            base, current_magnitude
        )
        negative_joule = joule_heating(effective, -current_magnitude) - joule_heating(
            base, -current_magnitude
        )

        self.assertAlmostEqual(positive_drop, current_magnitude * series_resistance)
        self.assertAlmostEqual(negative_drop, -positive_drop)
        self.assertAlmostEqual(
            positive_joule,
            current_magnitude**2 * series_resistance,
        )
        self.assertAlmostEqual(negative_joule, positive_joule)

        for model_name in (FOUR_STATE_MODEL, FIVE_STATE_MODEL):
            for signed_current in (current_magnitude, -current_magnitude):
                schedule = PiecewiseConstantCurrent.pulse(
                    start_time=5.0,
                    end_time=25.0,
                    pulse_current=signed_current,
                )
                baseline = self.simulate(model_name, schedule)
                with_series = self.simulate(
                    model_name,
                    schedule,
                    series_resistance=series_resistance,
                )
                baseline_at_switch = next(
                    item.value
                    for item in baseline.values
                    if item.channel == VOLTAGE and item.time == 5.0
                )
                series_at_switch = next(
                    item.value
                    for item in with_series.values
                    if item.channel == VOLTAGE and item.time == 5.0
                )
                self.assertAlmostEqual(
                    series_at_switch - baseline_at_switch,
                    signed_current * series_resistance,
                )

    def test_invalid_series_resistance_is_rejected(self):
        for value in (-0.01, float("inf"), float("-inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "series resistance must be finite and nonnegative",
                ):
                    self.simulate(FOUR_STATE_MODEL, series_resistance=value)

    def test_effective_resistance_preserves_four_and_five_state_energy_closure(self):
        reference = constant_current_contact_reference_experiment()
        series_resistance = 0.10
        thermoelectric = replace(
            reference.thermoelectric_parameters,
            electrical_resistance=(
                reference.thermoelectric_parameters.electrical_resistance
                + series_resistance
            ),
        )
        thermal = reference.thermal_parameters
        current = -0.8
        cold_face = 295.0
        hot_face = 305.0
        cold_exchanger = 300.0
        hot_exchanger = 301.0
        reservoir_rate = (
            thermal.cold_reservoir_conductance * (300.0 - cold_exchanger)
            + thermal.hot_reservoir_conductance * (300.0 - hot_exchanger)
        )
        expected_rate = reservoir_rate + electrical_power(
            thermoelectric,
            current,
            hot_face,
            cold_face,
        )

        four_rates = four_node_contact_rhs(
            thermoelectric,
            thermal,
            cold_face_temperature=cold_face,
            hot_face_temperature=hot_face,
            cold_exchanger_temperature=cold_exchanger,
            hot_exchanger_temperature=hot_exchanger,
            current=current,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
        )
        four_stored_rate = (
            thermal.cold_face_thermal_capacitance * four_rates.cold_face
            + thermal.hot_face_thermal_capacitance * four_rates.hot_face
            + thermal.cold_exchanger_thermal_capacitance * four_rates.cold_exchanger
            + thermal.hot_exchanger_thermal_capacitance * four_rates.hot_exchanger
        )
        self.assertAlmostEqual(four_stored_rate, expected_rate)

        mismatch = InterfaceMassMismatch(thermal_capacitance=20.0)
        five_rates = interface_mass_rhs(
            thermoelectric,
            thermal,
            mismatch,
            cold_face_temperature=cold_face,
            hot_face_temperature=hot_face,
            cold_interface_temperature=298.0,
            cold_exchanger_temperature=cold_exchanger,
            hot_exchanger_temperature=hot_exchanger,
            current=current,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
        )
        five_stored_rate = (
            thermal.cold_face_thermal_capacitance * five_rates.cold_face
            + thermal.hot_face_thermal_capacitance * five_rates.hot_face
            + mismatch.thermal_capacitance * five_rates.cold_interface
            + thermal.cold_exchanger_thermal_capacitance * five_rates.cold_exchanger
            + thermal.hot_exchanger_thermal_capacitance * five_rates.hot_exchanger
        )
        self.assertAlmostEqual(five_stored_rate, expected_rate)

    def test_series_resistance_trajectories_converge_under_time_step_refinement(self):
        schedule = PiecewiseConstantCurrent(
            transition_times=(5.0, 25.0, 38.0, 58.0),
            values=(0.0, 1.0, 0.0, -0.8, 0.0),
        )
        for model_name in (FOUR_STATE_MODEL, FIVE_STATE_MODEL):
            final_temperatures = []
            for time_step in (0.25, 0.125, 0.0625):
                config = replace(self.config, dense_time_step=time_step)
                result = self.simulate(
                    model_name,
                    schedule,
                    series_resistance=0.50,
                    config=config,
                )
                final_temperatures.append(result.cold_face[-1])
            coarse_change = abs(final_temperatures[1] - final_temperatures[0])
            fine_change = abs(final_temperatures[2] - final_temperatures[1])
            with self.subTest(model_name=model_name):
                self.assertLess(fine_change, coarse_change)
                self.assertLess(fine_change, 1.0e-6)


if __name__ == "__main__":
    unittest.main()
