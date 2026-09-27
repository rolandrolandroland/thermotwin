import hashlib
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "reports/hax_research_2026_09_17_v1"
V2 = ROOT / "reports/hax_research_2026_09_27_v2"


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HaxResearchBundleTests(unittest.TestCase):
    def test_v2_manifest_binds_every_output_and_retains_v1_science(self):
        manifest = json.loads((V2 / "source_manifest.json").read_text())
        self.assertFalse(manifest["scientific_inputs_changed"])
        v1_sources = json.loads((V1 / "source_manifest.json").read_text())["sources"]
        self.assertEqual(manifest["unchanged_scientific_sources_from_v1"], v1_sources)
        expected = {
            path.relative_to(ROOT).as_posix(): (path.stat().st_size, _digest(path))
            for path in V2.rglob("*")
            if path.is_file() and path.name != "source_manifest.json"
        }
        observed = {
            item["path"]: (item["bytes"], item["sha256"])
            for item in manifest["revised_outputs"]
        }
        self.assertEqual(observed, expected)

    def test_active_markdown_links_resolve_and_legacy_figure_is_not_selected(self):
        for report in V2.glob("*.md"):
            text = report.read_text()
            for target in re.findall(r"!\[[^]]*\]\(([^)]+)\)", text):
                self.assertTrue((report.parent / target).is_file(), target)
            self.assertNotIn("next_experiment_story.png", text)
        legacy = ROOT / "thermotwin/figures/NONLINEAR_EXPERIMENT_SELECTION/superseded/next_experiment_story.png"
        self.assertTrue(legacy.is_file())

    def test_corrected_disclosures_are_in_active_sources(self):
        pinn = (V2 / "thermotwin_pinn_research_report_2026_09_27_v2.md").read_text()
        self.assertIn("same ten paired trials", pinn)
        self.assertIn("5.0747%", pinn)
        self.assertIn("roughly two-thirds", pinn)
        self.assertIn("weight 10 was then frozen before trials 1 and 2", pinn)
        selection = (
            V2 / "thermotwin_selection_research_report_2026_09_27_v2.md"
        ).read_text()
        for value in ("6.55", "6.60", "3.92", "3.9–6.6"):
            self.assertIn(value, selection)


if __name__ == "__main__":
    unittest.main()
