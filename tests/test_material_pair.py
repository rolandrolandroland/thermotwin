from dataclasses import replace
import unittest

from thermotwin.design.codesign.evaluation import (
    evaluate_design_current,
    optimize_design_current,
)
from thermotwin.design.codesign.models import (
    APPLICATION_SPECIFICATIONS,
    ModuleAssemblyAssumptions,
)
from thermotwin.design.codesign.sampling import generate_space_filling_designs
from thermotwin.design.material_pair import (
    MaterialPairDesign,
    evaluate_material_pair_current,
    match_required_cooling_current,
    optimize_material_pair_current,
)


class ExplicitMaterialPairTests(unittest.TestCase):
    def setUp(self):
        self.prototype = generate_space_filling_designs(
            1,
            seed=7,
            prefix="parity",
        )[0]
        self.explicit = MaterialPairDesign.from_prototype(self.prototype)
        self.application = APPLICATION_SPECIFICATIONS[0]

    def test_fixed_current_matches_frozen_prototype_path(self):
        original = evaluate_design_current(self.prototype, self.application, 0.4)
        explicit = evaluate_material_pair_current(self.explicit, self.application, 0.4)
        for field in (
            "delivered_cooling_rate",
            "delivered_heating_rate",
            "module_electrical_power",
            "supply_electrical_power",
            "wall_cooling_cop",
            "peak_voltage",
            "peak_current_density",
            "prototype_cost_index",
            "utility",
        ):
            self.assertEqual(getattr(explicit, field), getattr(original, field))
        self.assertEqual(explicit.feasible, original.feasible)

    def test_grid_optimizer_matches_frozen_prototype_path(self):
        original = optimize_design_current(
            self.prototype,
            self.application,
            grid_size=9,
        )
        explicit = optimize_material_pair_current(
            self.explicit,
            self.application,
            grid_size=9,
        )
        self.assertEqual(explicit.mean_current, original.mean_current)
        self.assertEqual(explicit.utility, original.utility)
        self.assertEqual(explicit.delivered_cooling_rate, original.delivered_cooling_rate)

    def test_higher_current_ceiling_preserves_first_rising_solution(self):
        assembly = ModuleAssemblyAssumptions(
            specific_electrical_contact_resistivity=2.0e-10,
            maximum_current_density=1.0e6,
        )
        low = match_required_cooling_current(
            self.explicit,
            self.application,
            assembly=assembly,
        )
        high = match_required_cooling_current(
            self.explicit,
            self.application,
            assembly=replace(assembly, maximum_current_density=3.0e6),
        )
        self.assertIsNotNone(low)
        self.assertIsNotNone(high)
        self.assertAlmostEqual(low.mean_current, high.mean_current, places=5)
        self.assertGreaterEqual(
            low.delivered_cooling_rate,
            self.application.minimum_cooling_rate,
        )
        self.assertGreaterEqual(
            high.delivered_cooling_rate,
            self.application.minimum_cooling_rate,
        )

    def test_zero_current_is_valid_for_limiting_case_diagnostics(self):
        point = evaluate_material_pair_current(self.explicit, self.application, 0.0)
        self.assertEqual(point.mean_current, 0.0)
        self.assertEqual(point.module_electrical_power, 0.0)
        self.assertIsNone(point.wall_cooling_cop)


if __name__ == "__main__":
    unittest.main()
