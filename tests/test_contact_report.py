from pathlib import Path
import tempfile
import unittest

try:
    from thermotwin import (
        constant_current_contact_reference_experiment,
        constant_current_reference_experiment,
    )
    from thermotwin.contact_report import (
        DEFAULT_CONTACT_REPORT_PATH,
        build_contact_comparison_report_data,
        save_contact_comparison_report,
    )
except ModuleNotFoundError:
    build_contact_comparison_report_data = None


@unittest.skipIf(
    build_contact_comparison_report_data is None,
    "optional Matplotlib dependency is not installed",
)
class ContactComparisonReportTests(unittest.TestCase):
    def setUp(self):
        self.report = build_contact_comparison_report_data(
            constant_current_contact_reference_experiment(),
            constant_current_reference_experiment(),
        )

    def test_default_report_path_uses_package_figures_directory(self):
        self.assertEqual(
            DEFAULT_CONTACT_REPORT_PATH.name,
            "contact_model_comparison.png",
        )
        self.assertEqual(DEFAULT_CONTACT_REPORT_PATH.parent.name, "figures")
        self.assertEqual(
            DEFAULT_CONTACT_REPORT_PATH.parent.parent.name,
            "thermotwin",
        )

    def test_report_histories_and_sweep_are_aligned(self):
        contact_count = len(self.report.contact_time)
        two_node_count = len(self.report.two_node_time)
        for history in (
            self.report.cold_face_temperature,
            self.report.hot_face_temperature,
            self.report.cold_exchanger_temperature,
            self.report.hot_exchanger_temperature,
            self.report.cold_contact_temperature_drop,
            self.report.hot_contact_temperature_drop,
            self.report.cold_contact_heat,
            self.report.hot_contact_heat,
            self.report.cold_module_heat,
            self.report.hot_module_heat,
        ):
            self.assertEqual(len(history), contact_count)
        self.assertEqual(
            len(self.report.two_node_cold_temperature),
            two_node_count,
        )
        self.assertEqual(
            len(self.report.two_node_hot_temperature),
            two_node_count,
        )
        self.assertEqual(
            tuple(point.contact_resistance for point in self.report.sweep),
            (0.1, 0.25, 0.5, 1.0),
        )
        self.assertLess(
            self.report.max_absolute_energy_balance_residual,
            1e-12,
        )

    def test_sweep_shows_larger_drops_and_less_delivered_cold_heat(self):
        first = self.report.sweep[0]
        last = self.report.sweep[-1]

        self.assertGreater(
            last.cold_contact_temperature_drop,
            first.cold_contact_temperature_drop,
        )
        self.assertGreater(
            last.hot_contact_temperature_drop,
            first.hot_contact_temperature_drop,
        )
        self.assertLess(last.cold_contact_heat, first.cold_contact_heat)
        self.assertGreater(
            last.cold_exchanger_temperature,
            first.cold_exchanger_temperature,
        )

    def test_invalid_sweep_resistances_are_rejected(self):
        for resistances in ((), (0.0,), (-1.0,), (float("nan"),)):
            with self.subTest(resistances=resistances):
                with self.assertRaises(ValueError):
                    build_contact_comparison_report_data(
                        constant_current_contact_reference_experiment(),
                        constant_current_reference_experiment(),
                        sweep_resistances=resistances,
                    )

    def test_report_writer_creates_png(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "contact.png"
            written = save_contact_comparison_report(
                self.report,
                destination,
            )

            self.assertEqual(written, destination.resolve())
            self.assertGreater(destination.stat().st_size, 1_000)
            self.assertEqual(
                destination.read_bytes()[:8],
                b"\x89PNG\r\n\x1a\n",
            )


if __name__ == "__main__":
    unittest.main()
