from pathlib import Path
import tempfile
import unittest

from thermotwin.reports.distributed_inverse_robustness import (
    format_distributed_inverse_robustness_report,
    save_distributed_inverse_robustness_figure,
)
from thermotwin.studies.distributed_inverse_robustness import (
    DistributedInverseRobustnessConfig,
    DistributedInverseRobustnessSeeds,
    DistributedInverseRobustnessStudyResult,
    DistributedInverseRobustnessTrial,
    summarize_distributed_inverse_robustness,
)


def _result():
    trial = DistributedInverseRobustnessTrial(
        trial_index=0,
        seeds=DistributedInverseRobustnessSeeds(100, (101, 102, 103, 104)),
        conventional_multipliers=(1.041, 1.069, 1.031),
        conventional_initial_normalized_loss=100.0,
        conventional_final_normalized_loss=1.0,
        conventional_loss_reduction_fraction=0.99,
        conventional_maximum_absolute_multiplier_error=0.001,
        conventional_reached_search_bound=False,
        conventional_success=True,
        conventional_failure_reasons=(),
        inverse_pinn_multipliers=(1.05, 1.06, 1.04),
        inverse_pinn_initial_normalized_loss=100.0,
        inverse_pinn_final_normalized_loss=1.5,
        inverse_pinn_loss_reduction_fraction=0.985,
        inverse_pinn_initial_observation_loss=80.0,
        inverse_pinn_final_observation_loss=0.8,
        inverse_pinn_observation_loss_reduction_fraction=0.99,
        inverse_pinn_initial_physics_loss=20.0,
        inverse_pinn_final_physics_loss=0.7,
        inverse_pinn_maximum_absolute_multiplier_error=0.01,
        inverse_pinn_success=True,
        inverse_pinn_failure_reasons=(),
    )
    return DistributedInverseRobustnessStudyResult(
        config=DistributedInverseRobustnessConfig(trial_count=1),
        truth_multipliers=(1.04, 1.07, 1.03),
        trials=(trial,),
        summary=summarize_distributed_inverse_robustness((trial,)),
    )


class DistributedInverseRobustnessReportTests(unittest.TestCase):
    def test_report_prints_criteria_seeds_every_trial_and_boundary(self):
        text = format_distributed_inverse_robustness_report(_result())
        self.assertIn("Predeclared success gate", text)
        self.assertIn("noise seeds=(101, 102, 103, 104)", text)
        self.assertIn("conventional: PASS", text)
        self.assertIn("inverse PINN: PASS", text)
        self.assertIn("reasons=('none',)", text)
        self.assertIn("observation loss=0.800000", text)
        self.assertIn("physics loss=0.700000", text)
        self.assertIn("total objective=1.500000", text)
        self.assertIn("no failed trial is dropped", text)
        self.assertIn("not independent-model or hardware", text)

    def test_figure_is_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "robustness.png"
            returned = save_distributed_inverse_robustness_figure(
                _result(), output
            )
            self.assertEqual(returned, output.resolve())
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 1_000)


if __name__ == "__main__":
    unittest.main()
