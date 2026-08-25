from dataclasses import replace
import unittest

from thermotwin.design.codesign.models import APPLICATION_SPECIFICATIONS
from thermotwin.design.contact_process_window import (
    ContactProcessWindowConfig,
    default_process_material_pairs,
    format_contact_process_window_report,
    logarithmic_grid,
    run_contact_process_window,
)
from thermotwin.design.literature_materials import PUBLISHED_AG2SE_UNICOUPLE


class ContactProcessWindowTests(unittest.TestCase):
    def test_logarithmic_grid_is_inclusive_and_geometric(self):
        values = logarithmic_grid(1.0, 100.0, 3)
        self.assertEqual(values[0], 1.0)
        self.assertAlmostEqual(values[1], 10.0)
        self.assertAlmostEqual(values[2], 100.0)

    def test_small_tensor_grid_and_published_sweep(self):
        config = ContactProcessWindowConfig(
            leg_lengths=(0.5e-3, 1.5e-3),
            specific_contact_resistivities=(0.0, 2.0e-10, 1.0e-8),
            current_density_limits=(1.0e6, 3.0e6),
            material_pairs=default_process_material_pairs()[:1],
            applications=APPLICATION_SPECIFICATIONS[:1],
        )
        result = run_contact_process_window(config)
        self.assertEqual(len(result.points), 12)
        self.assertEqual(len(result.published_electrical_sweep), 3)
        self.assertEqual(
            result.published_electrical_sweep[0].normalized_zt_retention,
            1.0,
        )

    def test_contact_share_is_monotone_and_zero_limit_closes(self):
        config = ContactProcessWindowConfig(
            leg_lengths=(1.5e-3,),
            specific_contact_resistivities=(0.0, 2.0e-10, 1.0e-8),
            current_density_limits=(1.0e6,),
            material_pairs=default_process_material_pairs()[:1],
            applications=APPLICATION_SPECIFICATIONS[:1],
        )
        points = run_contact_process_window(config).points
        shares = tuple(point.electrical_contact_fraction for point in points)
        self.assertEqual(shares[0], 0.0)
        self.assertEqual(tuple(sorted(shares)), shares)
        self.assertGreater(points[0].device_zt_300k, points[-1].device_zt_300k)

    def test_published_half_loss_point_is_analytical_half(self):
        case = PUBLISHED_AG2SE_UNICOUPLE
        self.assertAlmostEqual(
            case.modeled_contact_share(case.half_contact_share_resistivity),
            0.5,
        )

    def test_unmatched_target_distinguishes_current_cap_from_physics(self):
        cap_limited_config = ContactProcessWindowConfig(
            leg_lengths=(1.5e-3,),
            specific_contact_resistivities=(2.0e-10,),
            current_density_limits=(5.0e4,),
            material_pairs=default_process_material_pairs()[:1],
            applications=APPLICATION_SPECIFICATIONS[:1],
        )
        cap_limited = run_contact_process_window(cap_limited_config).points[0]
        self.assertFalse(cap_limited.matched_required_cooling)
        self.assertEqual(
            cap_limited.infeasibility_reasons,
            ("cooling_target_current_density_limited",),
        )
        self.assertIn(
            "current-density cap=1/1 (100.0%); physical maximum=0/1 (0.0%)",
            format_contact_process_window_report(
                run_contact_process_window(cap_limited_config)
            ),
        )

        impossible_application = replace(
            APPLICATION_SPECIFICATIONS[0],
            name="impossible_cooling",
            label="impossible cooling",
            minimum_cooling_rate=100.0,
        )
        physics_limited_config = replace(
            cap_limited_config,
            applications=(impossible_application,),
        )
        physics_limited = run_contact_process_window(physics_limited_config).points[0]
        self.assertFalse(physics_limited.matched_required_cooling)
        self.assertEqual(
            physics_limited.infeasibility_reasons,
            ("cooling_target_physics_limited",),
        )
        self.assertIn(
            "current-density cap=0/1 (0.0%); physical maximum=1/1 (100.0%)",
            format_contact_process_window_report(
                run_contact_process_window(physics_limited_config)
            ),
        )


if __name__ == "__main__":
    unittest.main()
