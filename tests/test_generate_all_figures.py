import unittest
from unittest.mock import patch

from thermotwin.reports.generate_all import (
    REPORT_ARGUMENTS,
    REPORT_MODULES,
    generate_all_figures,
)


class GenerateAllFiguresTests(unittest.TestCase):
    def test_catalog_has_unique_report_modules(self):
        self.assertEqual(len(REPORT_MODULES), len(set(REPORT_MODULES)))
        self.assertIn("thermotwin.control_comparison_report", REPORT_MODULES)
        self.assertIn("thermotwin.sparse_sensor_report", REPORT_MODULES)
        self.assertIn("thermotwin.adaptive_experiment_campaign", REPORT_MODULES)
        self.assertIn("thermotwin.sensor_model_discrimination", REPORT_MODULES)

    @patch("thermotwin.reports.generate_all.subprocess.run")
    def test_selected_modules_run_with_active_python(self, run):
        selected = (
            "thermotwin.assembly_fingerprint_report",
            "thermotwin.experiment_selection_report",
        )
        self.assertEqual(generate_all_figures(selected), selected)
        self.assertEqual(run.call_count, 2)
        for call, module in zip(run.call_args_list, selected):
            self.assertEqual(call.args[0][1:], ("-m", module))
            self.assertTrue(call.kwargs["check"])

    @patch("thermotwin.reports.generate_all.subprocess.run")
    def test_completed_distributed_report_enables_inverse_panels(self, run):
        module = "thermotwin.distributed_property_report"
        generate_all_figures((module,))
        self.assertEqual(
            run.call_args.args[0][1:],
            ("-m", module, *REPORT_ARGUMENTS[module]),
        )

    def test_unknown_or_empty_module_selection_is_rejected(self):
        with self.assertRaises(ValueError):
            generate_all_figures(())
        with self.assertRaises(ValueError):
            generate_all_figures(("thermotwin.not_a_report",))


if __name__ == "__main__":
    unittest.main()
