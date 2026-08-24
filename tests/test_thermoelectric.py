import unittest

from thermotwin import (
    ThermoelectricParameters,
    coefficient_of_performance,
    cold_side_heat,
    conductive_heat_leak,
    electrical_power,
    hot_side_heat,
    heating_coefficient_of_performance,
    joule_heating,
    peltier_heat,
    voltage,
)


class ThermoelectricModelTests(unittest.TestCase):
    def setUp(self):
        self.parameters = ThermoelectricParameters(
            seebeck_coefficient=0.05,
            electrical_resistance=2.0,
            thermal_conductance=0.5,
        )
        self.hot_temperature = 320.0
        self.cold_temperature = 300.0

    def test_nominal_terms_have_expected_values_and_units(self):
        current = 3.0

        self.assertAlmostEqual(
            peltier_heat(self.parameters, current, self.cold_temperature),
            45.0,
        )
        self.assertAlmostEqual(joule_heating(self.parameters, current), 18.0)
        self.assertAlmostEqual(
            conductive_heat_leak(
                self.parameters,
                self.hot_temperature,
                self.cold_temperature,
            ),
            10.0,
        )
        self.assertAlmostEqual(
            cold_side_heat(
                self.parameters,
                current,
                self.hot_temperature,
                self.cold_temperature,
            ),
            26.0,
        )
        self.assertAlmostEqual(
            hot_side_heat(
                self.parameters,
                current,
                self.hot_temperature,
                self.cold_temperature,
            ),
            47.0,
        )
        self.assertAlmostEqual(
            voltage(
                self.parameters,
                current,
                self.hot_temperature,
                self.cold_temperature,
            ),
            7.0,
        )

    def test_energy_identity_holds_for_positive_zero_and_negative_current(self):
        for current in (3.0, 0.0, -0.1, -3.0):
            with self.subTest(current=current):
                cold_heat = cold_side_heat(
                    self.parameters,
                    current,
                    self.hot_temperature,
                    self.cold_temperature,
                )
                hot_heat = hot_side_heat(
                    self.parameters,
                    current,
                    self.hot_temperature,
                    self.cold_temperature,
                )
                power = electrical_power(
                    self.parameters,
                    current,
                    self.hot_temperature,
                    self.cold_temperature,
                )
                self.assertAlmostEqual(hot_heat - cold_heat, power)

    def test_heating_cop_is_cooling_cop_plus_one(self):
        current = 3.0

        cooling_cop = coefficient_of_performance(
            self.parameters,
            current,
            self.hot_temperature,
            self.cold_temperature,
        )
        heating_cop = heating_coefficient_of_performance(
            self.parameters,
            current,
            self.hot_temperature,
            self.cold_temperature,
        )

        self.assertAlmostEqual(heating_cop, cooling_cop + 1.0)

    def test_heating_cop_is_undefined_at_zero_power(self):
        with self.assertRaises(ZeroDivisionError):
            heating_coefficient_of_performance(
                self.parameters,
                0.0,
                self.hot_temperature,
                self.cold_temperature,
            )

    def test_zero_current_is_passive_hot_to_cold_conduction(self):
        current = 0.0
        expected_heat = -self.parameters.thermal_conductance * (
            self.hot_temperature - self.cold_temperature
        )

        self.assertAlmostEqual(
            cold_side_heat(
                self.parameters,
                current,
                self.hot_temperature,
                self.cold_temperature,
            ),
            expected_heat,
        )
        self.assertAlmostEqual(
            hot_side_heat(
                self.parameters,
                current,
                self.hot_temperature,
                self.cold_temperature,
            ),
            expected_heat,
        )
        self.assertGreater(
            voltage(
                self.parameters,
                current,
                self.hot_temperature,
                self.cold_temperature,
            ),
            0.0,
        )
        self.assertEqual(
            electrical_power(
                self.parameters,
                current,
                self.hot_temperature,
                self.cold_temperature,
            ),
            0.0,
        )

    def test_current_reversal_flips_peltier_but_not_joule_heating(self):
        current = 3.0
        positive_peltier = peltier_heat(
            self.parameters, current, self.cold_temperature
        )
        negative_peltier = peltier_heat(
            self.parameters, -current, self.cold_temperature
        )

        self.assertAlmostEqual(negative_peltier, -positive_peltier)
        self.assertAlmostEqual(
            joule_heating(self.parameters, -current),
            joule_heating(self.parameters, current),
        )

    def test_equal_face_temperatures_remove_thermal_leak_and_seebeck_voltage(self):
        current = 3.0
        temperature = 300.0

        self.assertEqual(
            conductive_heat_leak(
                self.parameters, temperature, temperature
            ),
            0.0,
        )
        self.assertAlmostEqual(
            voltage(
                self.parameters, current, temperature, temperature
            ),
            current * self.parameters.electrical_resistance,
        )

        cold_heat = cold_side_heat(
            self.parameters, current, temperature, temperature
        )
        hot_heat = hot_side_heat(
            self.parameters, current, temperature, temperature
        )
        self.assertAlmostEqual(
            hot_heat - cold_heat,
            joule_heating(self.parameters, current),
        )

    def test_cold_side_heat_peaks_at_predicted_current(self):
        peak_current = (
            self.parameters.seebeck_coefficient
            * self.cold_temperature
            / self.parameters.electrical_resistance
        )

        heat_at_peak = cold_side_heat(
            self.parameters,
            peak_current,
            self.hot_temperature,
            self.cold_temperature,
        )
        for offset in (-1.0, 1.0):
            with self.subTest(offset=offset):
                self.assertGreater(
                    heat_at_peak,
                    cold_side_heat(
                        self.parameters,
                        peak_current + offset,
                        self.hot_temperature,
                        self.cold_temperature,
                    ),
                )

        excessive_current = 2.0 * peak_current
        self.assertLess(
            cold_side_heat(
                self.parameters,
                excessive_current,
                self.hot_temperature,
                self.cold_temperature,
            ),
            0.0,
        )

    def test_cop_is_cooling_divided_by_electrical_power(self):
        current = 3.0
        cold_heat = cold_side_heat(
            self.parameters,
            current,
            self.hot_temperature,
            self.cold_temperature,
        )
        power = electrical_power(
            self.parameters,
            current,
            self.hot_temperature,
            self.cold_temperature,
        )

        self.assertAlmostEqual(
            coefficient_of_performance(
                self.parameters,
                current,
                self.hot_temperature,
                self.cold_temperature,
            ),
            cold_heat / power,
        )

    def test_cop_is_undefined_at_zero_power(self):
        with self.assertRaisesRegex(ZeroDivisionError, "undefined"):
            coefficient_of_performance(
                self.parameters,
                0.0,
                self.hot_temperature,
                self.cold_temperature,
            )


if __name__ == "__main__":
    unittest.main()
