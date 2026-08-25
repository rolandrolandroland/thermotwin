import unittest

from thermotwin.design.literature_materials import (
    AG2SE_2026_DOI,
    AG2SE_2026_OPTIMIZED,
    PUBLISHED_AG2SE_UNICOUPLE,
)
from thermotwin.design.materials import N_TYPE_SAMPLES


class LiteratureMaterialTests(unittest.TestCase):
    def test_optimized_ag2se_triplet_and_provenance(self):
        record = AG2SE_2026_OPTIMIZED
        material = record.material
        self.assertEqual(record.source_doi, AG2SE_2026_DOI)
        self.assertEqual(material.carrier_type, "n")
        self.assertAlmostEqual(material.seebeck_coefficient, -153.3e-6)
        self.assertAlmostEqual(material.electrical_conductivity, 117_400.0)
        self.assertAlmostEqual(material.thermal_conductivity, 0.85)
        self.assertAlmostEqual(material.power_factor, 2.759004486e-3)
        self.assertAlmostEqual(material.calculated_zt, 0.9737662891764707)

    def test_opt_in_record_does_not_mutate_baseline_catalog(self):
        self.assertEqual(len(N_TYPE_SAMPLES), 6)
        self.assertNotIn(
            AG2SE_2026_OPTIMIZED.material.sample_id,
            tuple(sample.sample_id for sample in N_TYPE_SAMPLES),
        )

    def test_published_unicouple_electrical_landmarks(self):
        case = PUBLISHED_AG2SE_UNICOUPLE
        self.assertAlmostEqual(case.p_electrical_conductivity, 47_619.04761904762)
        self.assertAlmostEqual(
            case.half_contact_share_resistivity,
            1.1069207836456561e-8,
        )
        self.assertAlmostEqual(
            case.inferred_specific_contact_resistivity,
            1.665e-8,
        )
        self.assertAlmostEqual(case.reported_contact_share, 0.592)
        self.assertAlmostEqual(
            case.modeled_contact_share(case.inferred_specific_contact_resistivity),
            0.6006665160936441,
        )
        self.assertAlmostEqual(
            case.modeled_contact_share(case.half_contact_share_resistivity),
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
