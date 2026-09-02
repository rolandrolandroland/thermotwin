from pathlib import Path
import tempfile
import unittest

try:
    from thermotwin.control_comparison import ControlComparisonConfig
    from thermotwin.pulse_operating_map import run_pulse_operating_map
    from thermotwin.pulse_operating_map_report import (
        DEFAULT_PULSE_OPERATING_MAP_PATH,
        save_pulse_operating_map_report,
    )
except ModuleNotFoundError:
    run_pulse_operating_map = None


@unittest.skipIf(
    run_pulse_operating_map is None,
    "optional Matplotlib dependency is not installed",
)
class PulseOperatingMapReportTests(unittest.TestCase):
    def test_default_path_uses_figures_directory(self):
        self.assertEqual(
            DEFAULT_PULSE_OPERATING_MAP_PATH.parent.name,
            "PULSE_OPERATING_MAP_EXPERIMENT",
        )
        self.assertEqual(DEFAULT_PULSE_OPERATING_MAP_PATH.parent.parent.name, "figures")

    def test_report_writer_creates_png(self):
        config = ControlComparisonConfig(
            warmup_duration=160.0,
            evaluation_duration=40.0,
            time_step=0.4,
            target_cooling_rates=(2.0,),
            pulse_periods=(10.0,),
            pulse_duty_cycles=(0.75,),
            maximum_storage_drift=0.2,
        )
        result = run_pulse_operating_map(config)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "pulse.png"
            save_pulse_operating_map_report(result, destination)
            self.assertGreater(destination.stat().st_size, 10_000)
            self.assertEqual(destination.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
