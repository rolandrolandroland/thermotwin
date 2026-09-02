from pathlib import Path
import tempfile
import unittest

try:
    from thermotwin.design.codesign.models import APPLICATION_SPECIFICATIONS
    from thermotwin.design.contact_process_window import (
        ContactProcessWindowConfig,
        default_process_material_pairs,
    )
    from thermotwin.reports.contact_process_window import (
        DEFAULT_CONTACT_PROCESS_WINDOW_PATH,
        build_and_save_contact_process_window_report,
    )
except ModuleNotFoundError:
    build_and_save_contact_process_window_report = None


@unittest.skipIf(
    build_and_save_contact_process_window_report is None,
    "optional Matplotlib dependency is not installed",
)
class ContactProcessWindowReportTests(unittest.TestCase):
    def test_default_path_uses_ignored_figures_directory(self):
        self.assertEqual(
            DEFAULT_CONTACT_PROCESS_WINDOW_PATH.name,
            "electrical_contact_process_window.png",
        )
        self.assertEqual(
            DEFAULT_CONTACT_PROCESS_WINDOW_PATH.parent.name,
            "ELECTRICAL_CONTACT_PROCESS_WINDOW",
        )
        self.assertEqual(
            DEFAULT_CONTACT_PROCESS_WINDOW_PATH.parent.parent.name, "figures"
        )

    def test_small_report_writer_creates_png(self):
        config = ContactProcessWindowConfig(
            leg_lengths=(0.5e-3, 1.5e-3),
            specific_contact_resistivities=(0.0, 2.0e-10, 1.0e-8),
            current_density_limits=(1.0e6, 3.0e6),
            material_pairs=default_process_material_pairs()[:2],
            applications=APPLICATION_SPECIFICATIONS,
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "process-window.png"
            result, written = build_and_save_contact_process_window_report(
                destination,
                config=config,
            )
            self.assertEqual(len(result.points), 72)
            self.assertEqual(written, destination.resolve())
            self.assertGreater(destination.stat().st_size, 1_000)
            self.assertEqual(destination.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
