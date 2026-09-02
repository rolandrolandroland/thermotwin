from pathlib import Path
import tempfile
import unittest

try:
    from thermotwin.cop_operating_map import run_cop_operating_map
    from thermotwin.cop_operating_map_report import (
        DEFAULT_COP_OPERATING_MAP_PATH,
        save_cop_operating_map_report,
    )
except ModuleNotFoundError:
    run_cop_operating_map = None


@unittest.skipIf(
    run_cop_operating_map is None,
    "optional Matplotlib dependency is not installed",
)
class COPOperatingMapReportTests(unittest.TestCase):
    def test_default_path_uses_figures_directory(self):
        self.assertEqual(DEFAULT_COP_OPERATING_MAP_PATH.name, "cop_operating_map.png")
        self.assertEqual(
            DEFAULT_COP_OPERATING_MAP_PATH.parent.name,
            "COP_OPERATING_MAP_EXPERIMENT",
        )
        self.assertEqual(DEFAULT_COP_OPERATING_MAP_PATH.parent.parent.name, "figures")

    def test_report_writer_creates_png(self):
        result = run_cop_operating_map()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "map.png"
            written = save_cop_operating_map_report(result, destination)

            self.assertEqual(written, destination.resolve())
            self.assertGreater(destination.stat().st_size, 10_000)
            self.assertEqual(destination.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertTrue(destination.with_suffix(".json").exists())


if __name__ == "__main__":
    unittest.main()
