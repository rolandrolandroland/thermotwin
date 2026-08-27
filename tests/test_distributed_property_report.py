from pathlib import Path
import tempfile
import unittest

from thermotwin.reports.distributed_properties import (
    format_distributed_property_study,
    run_distributed_property_study,
    save_distributed_property_figure,
)


class DistributedPropertyReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_distributed_property_study(
            run_forward_pinn=False,
            quick=True,
        )

    def test_report_states_the_local_identifiability_boundary(self):
        text = format_distributed_property_study(self.result)
        self.assertIn("joint: rank", text)
        self.assertIn("training gate", text)
        self.assertIn("not trained", text)

    def test_figure_is_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "distributed.png"
            returned = save_distributed_property_figure(self.result, output)
            self.assertEqual(returned, output.resolve())
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 1_000)


if __name__ == "__main__":
    unittest.main()
