from pathlib import Path
import tempfile
import unittest

try:
    from thermotwin.pwm_power_electronics import run_pwm_power_electronics_experiment
    from thermotwin.pwm_power_electronics_report import (
        DEFAULT_PWM_POWER_ELECTRONICS_PATH,
        save_pwm_power_electronics_report,
    )
except ModuleNotFoundError:
    run_pwm_power_electronics_experiment = None


@unittest.skipIf(
    run_pwm_power_electronics_experiment is None,
    "optional Matplotlib dependency is not installed",
)
class PWMPowerElectronicsReportTests(unittest.TestCase):
    def test_default_path_uses_figures_directory(self):
        self.assertEqual(
            DEFAULT_PWM_POWER_ELECTRONICS_PATH.parent.name,
            "PWM_POWER_ELECTRONICS_EXPERIMENT",
        )
        self.assertEqual(DEFAULT_PWM_POWER_ELECTRONICS_PATH.parent.parent.name, "figures")

    def test_report_writer_creates_png(self):
        result = run_pwm_power_electronics_experiment()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "pwm.png"
            save_pwm_power_electronics_report(result, destination)
            self.assertGreater(destination.stat().st_size, 10_000)
            self.assertEqual(destination.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
