import unittest

from thermotwin.design.ag2se_substitution import (
    Ag2SeSubstitutionConfig,
    run_ag2se_substitution_study,
)
from thermotwin.design.codesign.models import APPLICATION_SPECIFICATIONS
from thermotwin.design.literature_materials import AG2SE_2026_OPTIMIZED
from thermotwin.design.materials import N_TYPE_SAMPLES


class Ag2SeSubstitutionTests(unittest.TestCase):
    def test_small_matched_study_preserves_geometry_and_p_material(self):
        config = Ag2SeSubstitutionConfig(
            initial_design_count=1,
            candidate_design_count=1,
            current_grid_size=4,
            specific_contact_resistivities=(2.0e-10,),
            applications=APPLICATION_SPECIFICATIONS[:1],
        )
        result = run_ag2se_substitution_study(config)
        self.assertEqual(len(result.comparisons), 2)
        self.assertEqual(len(result.summaries), 1)
        for comparison in result.comparisons:
            self.assertEqual(
                comparison.original.design.geometry,
                comparison.ag2se.design.geometry,
            )
            self.assertEqual(
                comparison.original.design.p_material.sample_id,
                comparison.ag2se.design.p_material.sample_id,
            )
            self.assertEqual(
                comparison.ag2se.design.n_material.sample_id,
                AG2SE_2026_OPTIMIZED.material.sample_id,
            )

    def test_baseline_catalog_remains_frozen(self):
        self.assertEqual(len(N_TYPE_SAMPLES), 6)
        self.assertNotIn(
            AG2SE_2026_OPTIMIZED.material,
            N_TYPE_SAMPLES,
        )


if __name__ == "__main__":
    unittest.main()
