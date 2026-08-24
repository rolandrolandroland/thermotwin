from dataclasses import replace
import math
import unittest

from thermotwin.control_comparison import (
    ControlComparisonConfig,
    evaluate_control_schedule,
    first_rising_crossing_bracket,
    piecewise_electrical_energy,
    run_control_comparison,
    trapezoidal_integral,
)
from thermotwin.controls import PiecewiseConstantCurrent
from thermotwin.contact_experiments import constant_current_contact_reference_experiment
from thermotwin.thermoelectric import ThermoelectricParameters


class ControlComparisonTests(unittest.TestCase):
    def test_default_duty_grid_exposes_continuous_limit(self):
        duties = ControlComparisonConfig().pulse_duty_cycles

        self.assertIn(0.75, duties)
        self.assertEqual(duties[-1], 0.99)

    def test_trapezoidal_integral_clips_both_interval_edges(self):
        value = trapezoidal_integral(
            (0.0, 1.0, 2.0),
            (0.0, 1.0, 2.0),
            start_time=0.5,
            end_time=1.5,
        )

        self.assertAlmostEqual(value, 1.0)

    def test_piecewise_power_integral_preserves_switch_limits(self):
        parameters = ThermoelectricParameters(0.05, 2.0, 0.5)
        schedule = PiecewiseConstantCurrent.pulse(
            start_time=0.0,
            end_time=0.75,
            pulse_current=1.0,
        )
        energy = piecewise_electrical_energy(
            (0.0, 0.5, 1.0),
            (295.0, 295.0, 295.0),
            (305.0, 305.0, 305.0),
            parameters,
            schedule,
            start_time=0.0,
            end_time=1.0,
        )

        # On-state power is alpha*dT*I + R*I^2 = 2.5 W for 0.75 s.
        self.assertAlmostEqual(energy, 1.875)

    def test_piecewise_power_integral_is_independent_of_output_alignment(self):
        parameters = ThermoelectricParameters(0.05, 2.0, 0.5)
        schedule = PiecewiseConstantCurrent.periodic_pulse(
            duration=4.0,
            period=1.0,
            duty_cycle=0.75,
            pulse_current=1.0,
        )
        coarse = piecewise_electrical_energy(
            (0.0, 1.0, 2.0, 3.0, 4.0),
            (295.0,) * 5,
            (305.0,) * 5,
            parameters,
            schedule,
            start_time=0.0,
            end_time=4.0,
        )
        shifted = piecewise_electrical_energy(
            (0.0, 0.6, 1.4, 2.2, 3.1, 4.0),
            (295.0,) * 6,
            (305.0,) * 6,
            parameters,
            schedule,
            start_time=0.0,
            end_time=4.0,
        )

        self.assertAlmostEqual(coarse, 7.5)
        self.assertAlmostEqual(shifted, coarse)

    def test_short_period_average_power_is_time_step_stable(self):
        base = constant_current_contact_reference_experiment()
        powers = []
        for time_step in (0.2, 0.05):
            config = ControlComparisonConfig(
                warmup_duration=20.0,
                evaluation_duration=4.0,
                time_step=time_step,
                target_cooling_rates=(2.0,),
                pulse_periods=(1.0,),
                pulse_duty_cycles=(0.75,),
                maximum_storage_drift=10.0,
            )
            schedule = PiecewiseConstantCurrent.periodic_pulse(
                duration=config.total_duration,
                period=1.0,
                duty_cycle=0.75,
                pulse_current=0.8,
            )
            experiment = replace(
                base,
                duration=config.total_duration,
                time_step=time_step,
                current=schedule,
            )
            point = evaluate_control_schedule(
                experiment,
                control_kind="regression",
                current_amplitude=0.8,
                period=1.0,
                duty_cycle=0.75,
                config=config,
            )
            powers.append(point.average_electrical_power)

        self.assertAlmostEqual(powers[0], powers[1], delta=2e-6)

    def test_configuration_rejects_nonphysical_values(self):
        for keyword, value in (
            ("time_step", 0.0),
            ("pulse_duty_cycles", (1.0,)),
            ("maximum_current", -1.0),
            ("amplitude_bracket_subdivisions", 1),
        ):
            with self.subTest(keyword=keyword):
                with self.assertRaises(ValueError):
                    ControlComparisonConfig(**{keyword: value})

    def test_first_crossing_bracket_handles_nonmonotonic_capacity(self):
        # The endpoint response is below target even though the rising branch
        # crosses it: q(5)=-5, while q(1)=3.
        bracket = first_rising_crossing_bracket(
            lambda current: 4.0 * current - current**2,
            target=3.0,
            maximum_input=5.0,
            subdivisions=20,
        )

        self.assertIsNotNone(bracket)
        self.assertLess(bracket[0], 1.0)
        self.assertGreaterEqual(bracket[1], 1.0)

    def test_equal_capacity_comparison_reports_storage_and_safety(self):
        config = ControlComparisonConfig(
            warmup_duration=160.0,
            evaluation_duration=40.0,
            time_step=0.4,
            target_cooling_rates=(2.0,),
            pulse_periods=(10.0,),
            pulse_duty_cycles=(0.75, 0.99),
            maximum_storage_drift=0.2,
        )

        result = run_control_comparison(
            config,
            cold_contact_resistance_samples=(0.25,),
        )
        comparison = result.comparisons[0]

        self.assertLess(
            abs(comparison.continuous.average_cooling_rate - 2.0),
            config.cooling_match_tolerance,
        )
        self.assertLess(
            abs(comparison.best_pulsed.average_cooling_rate - 2.0),
            config.cooling_match_tolerance,
        )
        self.assertTrue(comparison.continuous.safe)
        self.assertTrue(comparison.best_pulsed.safe)
        self.assertTrue(comparison.continuous.cyclically_settled)
        self.assertTrue(comparison.best_pulsed.cyclically_settled)
        self.assertEqual(comparison.best_pulsed.duty_cycle, 0.99)
        point_by_duty = {
            point.duty_cycle: point for point in comparison.pulsed_candidates
        }
        continuous_cop = comparison.continuous.delivered_cooling_cop
        self.assertLess(
            abs(point_by_duty[0.99].delivered_cooling_cop - continuous_cop),
            abs(point_by_duty[0.75].delivered_cooling_cop - continuous_cop),
        )
        self.assertTrue(math.isfinite(comparison.pulsed_cop_change_percent))
        self.assertLess(
            abs(
                comparison.equal_power_continuous.average_electrical_power
                - comparison.best_pulsed.average_electrical_power
            ),
            0.001,
        )
        self.assertTrue(
            math.isfinite(
                comparison.pulsed_cooling_change_at_equal_power_percent
            )
        )
        self.assertEqual(len(result.uncertainty_cases), 1)


if __name__ == "__main__":
    unittest.main()
