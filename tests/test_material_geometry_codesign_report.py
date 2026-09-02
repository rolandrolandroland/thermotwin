import tempfile
import unittest
from pathlib import Path

from thermotwin.material_geometry_codesign import CodesignCampaignConfig
from thermotwin.material_geometry_codesign_report import (
    DEFAULT_MATERIAL_GEOMETRY_CODESIGN_PATH,
    build_and_save_material_geometry_codesign_report,
)


class MaterialGeometryCodesignReportTests(unittest.TestCase):
    def test_default_output_uses_package_figures_directory(self):
        self.assertEqual(
            DEFAULT_MATERIAL_GEOMETRY_CODESIGN_PATH.parent.name,
            "MATERIAL_GEOMETRY_BAYESIAN_CODESIGN",
        )
        self.assertEqual(
            DEFAULT_MATERIAL_GEOMETRY_CODESIGN_PATH.parent.parent.name,
            "figures",
        )
        self.assertEqual(
            DEFAULT_MATERIAL_GEOMETRY_CODESIGN_PATH.name,
            "material_geometry_bayesian_codesign.png",
        )

    def test_small_report_writes_png(self):
        config = CodesignCampaignConfig(
            initial_design_count=4,
            candidate_design_count=6,
            bayesian_iterations=1,
            random_search_repetitions=2,
            robustness_trials=4,
            current_grid_size=5,
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "report.png"
            result, saved = build_and_save_material_geometry_codesign_report(
                destination,
                config=config,
            )
            self.assertEqual(saved, destination.resolve())
            self.assertEqual(saved.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(len(result.bayesian_results), 3)


if __name__ == "__main__":
    unittest.main()
