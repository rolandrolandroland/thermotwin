import unittest

from thermotwin.material_catalog import (
    N_TYPE_SAMPLES,
    P_TYPE_SAMPLES,
    STARRYDATA_SNAPSHOT_DOI,
    STARRYDATA_SNAPSHOT_MD5,
    STARRYDATA_SNAPSHOT_URL,
    material_sample,
)
from thermotwin.material_geometry_codesign import (
    ModuleAssemblyAssumptions,
    ModuleGeometry,
    PrototypeDesign,
    module_electrical_resistance_components,
    module_thermoelectric_parameters,
    prototype_cost_index,
)


class MaterialCatalogTests(unittest.TestCase):
    def test_catalog_has_six_unique_samples_of_each_sign(self):
        self.assertEqual(len(P_TYPE_SAMPLES), 6)
        self.assertEqual(len(N_TYPE_SAMPLES), 6)
        samples = P_TYPE_SAMPLES + N_TYPE_SAMPLES
        self.assertEqual(len({sample.sample_id for sample in samples}), 12)
        self.assertTrue(all(sample.seebeck_coefficient > 0 for sample in P_TYPE_SAMPLES))
        self.assertTrue(all(sample.seebeck_coefficient < 0 for sample in N_TYPE_SAMPLES))
        self.assertTrue(all(sample.temperature == 300.0 for sample in samples))

    def test_same_row_figure_of_merit_is_positive_and_plausible(self):
        for sample in P_TYPE_SAMPLES + N_TYPE_SAMPLES:
            self.assertGreater(sample.power_factor, 0.0)
            self.assertGreater(sample.calculated_zt, 0.10)
            self.assertLess(sample.calculated_zt, 1.50)

    def test_lookup_and_fixed_snapshot_provenance(self):
        self.assertIs(material_sample(9107), P_TYPE_SAMPLES[0])
        with self.assertRaises(KeyError):
            material_sample(999999)
        self.assertEqual(STARRYDATA_SNAPSHOT_DOI, "10.6084/m9.figshare.11340935.v1")
        self.assertEqual(STARRYDATA_SNAPSHOT_MD5, "5ae1d38f76fd872d40bff37c2bec29f6")
        self.assertTrue(STARRYDATA_SNAPSHOT_URL.startswith("https://figshare.com/"))


class ModuleScalingTests(unittest.TestCase):
    def setUp(self):
        self.p_material = P_TYPE_SAMPLES[0]
        self.n_material = N_TYPE_SAMPLES[0]
        self.ideal_assembly = ModuleAssemblyAssumptions(
            specific_electrical_contact_resistivity=0.0,
            parasitic_thermal_conductance=0.0,
            pwm_ripple_peak_to_peak_fraction=0.0,
            converter_efficiency=1.0,
            fixed_converter_loss=0.0,
        )

    def parameters(self, count=100, length=1.0e-3, area=1.0e-6):
        return module_thermoelectric_parameters(
            self.p_material,
            self.n_material,
            ModuleGeometry(count, length, area),
            assembly=self.ideal_assembly,
        )

    def test_doubling_couple_count_doubles_alpha_resistance_and_conductance(self):
        base = self.parameters(count=100)
        doubled = self.parameters(count=200)
        self.assertAlmostEqual(doubled.seebeck_coefficient, 2 * base.seebeck_coefficient)
        self.assertAlmostEqual(doubled.electrical_resistance, 2 * base.electrical_resistance)
        self.assertAlmostEqual(doubled.thermal_conductance, 2 * base.thermal_conductance)

    def test_doubling_length_doubles_resistance_and_halves_conductance(self):
        base = self.parameters(length=1.0e-3)
        doubled = self.parameters(length=2.0e-3)
        self.assertAlmostEqual(doubled.seebeck_coefficient, base.seebeck_coefficient)
        self.assertAlmostEqual(doubled.electrical_resistance, 2 * base.electrical_resistance)
        self.assertAlmostEqual(doubled.thermal_conductance, 0.5 * base.thermal_conductance)

    def test_doubling_area_halves_resistance_and_doubles_conductance(self):
        base = self.parameters(area=1.0e-6)
        doubled = self.parameters(area=2.0e-6)
        self.assertAlmostEqual(doubled.seebeck_coefficient, base.seebeck_coefficient)
        self.assertAlmostEqual(doubled.electrical_resistance, 0.5 * base.electrical_resistance)
        self.assertAlmostEqual(doubled.thermal_conductance, 2 * base.thermal_conductance)

    def test_areal_contact_resistance_is_independent_of_leg_length(self):
        assembly = ModuleAssemblyAssumptions(
            specific_electrical_contact_resistivity=2.0e-10
        )
        short = module_electrical_resistance_components(
            self.p_material,
            self.n_material,
            ModuleGeometry(100, 0.8e-3, 1.0e-6),
            assembly=assembly,
        )
        long = module_electrical_resistance_components(
            self.p_material,
            self.n_material,
            ModuleGeometry(100, 2.4e-3, 1.0e-6),
            assembly=assembly,
        )

        self.assertAlmostEqual(
            short.electrical_contact_resistance,
            long.electrical_contact_resistance,
        )
        self.assertAlmostEqual(
            long.bulk_leg_resistance,
            3.0 * short.bulk_leg_resistance,
        )
        self.assertGreater(short.contact_fraction, long.contact_fraction)

    def test_areal_contact_resistance_scales_with_count_and_inverse_area(self):
        assembly = ModuleAssemblyAssumptions(
            specific_electrical_contact_resistivity=2.0e-10
        )
        base = module_electrical_resistance_components(
            self.p_material,
            self.n_material,
            ModuleGeometry(100, 1.0e-3, 1.0e-6),
            assembly=assembly,
        )
        doubled_count = module_electrical_resistance_components(
            self.p_material,
            self.n_material,
            ModuleGeometry(200, 1.0e-3, 1.0e-6),
            assembly=assembly,
        )
        doubled_area = module_electrical_resistance_components(
            self.p_material,
            self.n_material,
            ModuleGeometry(100, 1.0e-3, 2.0e-6),
            assembly=assembly,
        )

        self.assertAlmostEqual(
            doubled_count.electrical_contact_resistance,
            2.0 * base.electrical_contact_resistance,
        )
        self.assertAlmostEqual(
            doubled_area.electrical_contact_resistance,
            0.5 * base.electrical_contact_resistance,
        )

    def test_assembly_rejects_negative_specific_contact_resistivity(self):
        with self.assertRaises(ValueError):
            ModuleAssemblyAssumptions(
                specific_electrical_contact_resistivity=-1.0e-10
            )

    def test_cost_proxy_increases_with_material_and_exchanger_burden(self):
        base = PrototypeDesign(
            "base",
            0,
            0,
            ModuleGeometry(100, 1.0e-3, 1.0e-6),
            0.25,
            2.0,
            4.0,
        )
        larger = PrototypeDesign(
            "larger",
            0,
            0,
            ModuleGeometry(140, 1.5e-3, 1.5e-6),
            0.25,
            3.0,
            6.0,
        )
        self.assertGreater(prototype_cost_index(larger), prototype_cost_index(base))


if __name__ == "__main__":
    unittest.main()
