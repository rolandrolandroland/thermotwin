import unittest

from thermotwin.pwm_power_electronics import (
    PWMPowerElectronicsConfig,
    averaged_thermoelectric_rates,
    direct_pwm_current_moments,
    evaluate_pwm_operating_point,
    ideal_dc_current_moments,
    run_pwm_power_electronics_experiment,
    smoothed_pwm_current_moments,
)


class PWMPowerElectronicsTests(unittest.TestCase):
    def test_averaged_heat_rates_close_electrical_power(self):
        current = direct_pwm_current_moments(0.6, 1.5)
        rates = averaged_thermoelectric_rates(
            seebeck_coefficient=0.05,
            electrical_resistance=2.0,
            thermal_conductance=0.5,
            current=current,
            hot_temperature=305.0,
            cold_temperature=295.0,
        )

        self.assertAlmostEqual(
            rates.hot_heat - rates.cold_heat,
            rates.module_electrical_power,
        )

    def test_direct_pwm_uses_mean_for_peltier_and_rms_for_joule(self):
        current = direct_pwm_current_moments(0.6, 1.5)

        self.assertAlmostEqual(current.duty_cycle, 0.4)
        self.assertAlmostEqual(current.mean_square_current, 0.9)
        self.assertAlmostEqual(current.joule_multiplier_over_dc, 2.5)
        self.assertGreater(current.rms_current, current.mean_current)

    def test_zero_ripple_smoothed_pwm_matches_dc_module_physics(self):
        config = PWMPowerElectronicsConfig(
            external_temperature_lifts=(10.0,),
            mean_currents=(0.6,),
            smoothed_ripple_peak_to_peak_fraction=0.0,
        )
        dc = evaluate_pwm_operating_point(
            ideal_dc_current_moments(0.6), 10.0, config
        )
        smoothed = evaluate_pwm_operating_point(
            smoothed_pwm_current_moments(0.6, 0.0), 10.0, config
        )

        self.assertAlmostEqual(smoothed.delivered_cooling_rate, dc.delivered_cooling_rate)
        self.assertAlmostEqual(smoothed.module_electrical_power, dc.module_electrical_power)
        self.assertLess(smoothed.wall_cooling_cop, dc.wall_cooling_cop)

    def test_full_duty_direct_pwm_matches_dc_module_physics(self):
        config = PWMPowerElectronicsConfig(
            external_temperature_lifts=(0.0,),
            mean_currents=(1.5,),
        )
        dc = evaluate_pwm_operating_point(
            ideal_dc_current_moments(1.5), 0.0, config
        )
        direct = evaluate_pwm_operating_point(
            direct_pwm_current_moments(1.5, 1.5), 0.0, config
        )

        self.assertAlmostEqual(direct.delivered_cooling_rate, dc.delivered_cooling_rate)
        self.assertAlmostEqual(direct.module_electrical_power, dc.module_electrical_power)

    def test_direct_pwm_is_worse_than_smoothed_at_same_mean_current(self):
        result = run_pwm_power_electronics_experiment(
            PWMPowerElectronicsConfig(
                external_temperature_lifts=(10.0,),
                mean_currents=(0.6,),
            )
        )
        smoothed = next(point for point in result.points if point.mode == "smoothed_pwm")
        direct = next(point for point in result.points if point.mode == "direct_pwm")

        self.assertLess(direct.delivered_cooling_rate, smoothed.delivered_cooling_rate)
        self.assertLess(direct.wall_cooling_cop, smoothed.wall_cooling_cop)
        self.assertGreater(direct.current.mean_square_current, smoothed.current.mean_square_current)

    def test_default_experiment_has_three_modes_per_grid_point(self):
        result = run_pwm_power_electronics_experiment()
        self.assertEqual(
            len(result.points),
            3 * len(result.config.external_temperature_lifts) * len(result.config.mean_currents),
        )

    def test_configuration_rejects_invalid_electronics(self):
        for keyword, value in (
            ("converter_efficiency", 0.0),
            ("converter_efficiency", 1.1),
            ("fixed_switching_loss", -0.1),
            ("smoothed_ripple_peak_to_peak_fraction", -0.1),
            ("mean_currents", (1.6,)),
        ):
            with self.subTest(keyword=keyword):
                with self.assertRaises(ValueError):
                    PWMPowerElectronicsConfig(**{keyword: value})


if __name__ == "__main__":
    unittest.main()
