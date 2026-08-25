from pathlib import Path
import tempfile
import unittest

try:
    from thermotwin.design.ag2se_substitution import Ag2SeSubstitutionConfig
    from thermotwin.design.codesign.models import APPLICATION_SPECIFICATIONS
    from thermotwin.reports.ag2se_substitution import (
        DEFAULT_AG2SE_SUBSTITUTION_PATH,
        build_and_save_ag2se_substitution_report,
    )
except ModuleNotFoundError:
    build_and_save_ag2se_substitution_report = None


@unittest.skipIf(
    build_and_save_ag2se_substitution_report is None,
    "optional Matplotlib dependency is not installed",
)
class Ag2SeSubstitutionReportTests(unittest.TestCase):
    def test_default_path_uses_ignored_figures_directory(self):
        self.assertEqual(
            DEFAULT_AG2SE_SUBSTITUTION_PATH.name,
            "ag2se_matched_substitution.png",
        )
        self.assertEqual(DEFAULT_AG2SE_SUBSTITUTION_PATH.parent.name, "figures")

    def test_small_report_writer_creates_png(self):
        config = Ag2SeSubstitutionConfig(
            initial_design_count=1,
            candidate_design_count=1,
            current_grid_size=4,
            specific_contact_resistivities=(2.0e-10,),
            applications=APPLICATION_SPECIFICATIONS[:1],
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "substitution.png"
            result, written = build_and_save_ag2se_substitution_report(
                destination,
                config=config,
            )
            self.assertEqual(len(result.comparisons), 2)
            self.assertEqual(written, destination.resolve())
            self.assertGreater(destination.stat().st_size, 1_000)
            self.assertEqual(destination.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
