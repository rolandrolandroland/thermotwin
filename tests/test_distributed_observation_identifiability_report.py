from pathlib import Path
import tempfile
import unittest

from thermotwin.reports.distributed_observation_identifiability import (
    format_distributed_observation_identifiability_report,
    save_distributed_observation_identifiability_figure,
)
from thermotwin.studies.distributed_observation_identifiability import (
    DistributedObservationIdentifiabilityConfig,
    run_distributed_observation_identifiability_study,
)


class DistributedObservationIdentifiabilityReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_distributed_observation_identifiability_study(
            DistributedObservationIdentifiabilityConfig(
                truth_node_count=9,
                truth_time_step=5.0e-4,
                inverse_pinn_epochs=1,
            ),
            fit_models=False,
        )

    def test_report_separates_gate_from_optimizer_output(self):
        text = format_distributed_observation_identifiability_report(self.result)
        self.assertIn("must inference refuse the estimate", text)
        self.assertIn("zero_current_only", text)
        self.assertIn("structurally_non_identifiable", text)
        self.assertIn("practically_non_identifiable", text)
        self.assertIn("not a global proof", text)
        self.assertIn("explicitly rejected", text)

    def test_figure_is_written_without_fitted_curves(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "identifiability.png"
            returned = save_distributed_observation_identifiability_figure(
                self.result, output
            )
            self.assertEqual(returned, output.resolve())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
