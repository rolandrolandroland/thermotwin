import unittest

from thermotwin.cop_operating_map import (
    COPOperatingMapConfig,
    _match_heat_rate,
    contact_steady_operating_point,
    points_for,
    run_cop_operating_map,
)


class COPOperatingMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_cop_operating_map()

    def test_complete_grid_has_expected_size(self):
        config = self.result.config
        self.assertEqual(
            len(self.result.points),
            len(config.external_temperature_lifts)
            * len(config.currents)
            * (1 + len(config.symmetric_contact_resistances)),
        )

    def test_heating_cop_is_cooling_cop_plus_one(self):
        point = contact_steady_operating_point(0.8, 10.0, 0.25)

        self.assertIsNotNone(point.cooling_cop)
        self.assertIsNotNone(point.heating_cop)
        self.assertAlmostEqual(point.heating_cop, point.cooling_cop + 1.0)

    def test_more_contact_resistance_hurts_same_current_cooling(self):
        low = contact_steady_operating_point(1.0, 10.0, 0.10)
        high = contact_steady_operating_point(1.0, 10.0, 0.50)

        self.assertLess(high.delivered_cooling_rate, low.delivered_cooling_rate)
        self.assertLess(high.cooling_cop, low.cooling_cop)
        self.assertGreater(high.face_temperature_lift, low.face_temperature_lift)

    def test_reported_cop_optimum_excludes_tiny_capacity_points(self):
        summary = next(
            item
            for item in self.result.optima
            if item.topology == "four_node_contact"
            and item.symmetric_contact_resistance == 0.25
            and item.external_temperature_lift == 0.0
            and item.mode == "cooling"
        )

        self.assertGreaterEqual(
            summary.heat_rate_at_maximum_cop,
            self.result.config.minimum_useful_heat_rate,
        )
        self.assertGreater(summary.current_at_maximum_heat_rate, 0.0)

    def test_equal_load_comparison_exposes_contact_penalty(self):
        comparison = next(
            item
            for item in self.result.equal_load_comparisons
            if item.mode == "cooling"
            and item.external_temperature_lift == 10.0
            and item.symmetric_contact_resistance == 0.25
        )

        self.assertTrue(comparison.feasible)
        self.assertGreater(comparison.contact_current, comparison.reduced_current)
        self.assertLess(comparison.contact_cop_penalty_percent, 0.0)
        self.assertGreater(comparison.extra_face_lift, 0.0)

    def test_high_lift_cooling_target_can_be_infeasible(self):
        comparison = next(
            item
            for item in self.result.equal_load_comparisons
            if item.mode == "cooling"
            and item.external_temperature_lift == 30.0
            and item.symmetric_contact_resistance == 0.25
        )

        self.assertFalse(comparison.feasible)
        self.assertIsNone(comparison.contact_cop)

    def test_heat_matching_finds_rising_branch_after_endpoint_turnover(self):
        config = COPOperatingMapConfig(
            currents=(0.1,),
            maximum_current=12.0,
        )

        matched = _match_heat_rate(
            "reduced_no_explicit_contact",
            0.0,
            None,
            3.0,
            "cooling",
            config,
        )

        self.assertIsNotNone(matched)
        self.assertAlmostEqual(matched.delivered_cooling_rate, 3.0, places=5)
        self.assertLess(matched.current, 1.0)

    def test_points_for_selects_one_curve(self):
        points = points_for(
            self.result,
            topology="four_node_contact",
            external_temperature_lift=15.0,
            symmetric_contact_resistance=0.25,
        )

        self.assertEqual(len(points), len(self.result.config.currents))
        self.assertEqual(tuple(point.current for point in points), self.result.config.currents)

    def test_configuration_rejects_invalid_grids(self):
        for keyword, value in (
            ("external_temperature_lifts", (5.0, 0.0)),
            ("currents", (0.0,)),
            ("symmetric_contact_resistances", (0.0,)),
            ("minimum_useful_heat_rate", 0.0),
            ("heat_rate_bracket_subdivisions", 1),
        ):
            with self.subTest(keyword=keyword):
                with self.assertRaises(ValueError):
                    COPOperatingMapConfig(**{keyword: value})


if __name__ == "__main__":
    unittest.main()
