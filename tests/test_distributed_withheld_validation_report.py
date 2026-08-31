from pathlib import Path
import tempfile
import unittest

from thermotwin.reports.distributed_withheld_validation import (
    format_distributed_withheld_validation_report,
    save_distributed_withheld_validation_figure,
)
from thermotwin.studies.distributed_inverse_robustness import (
    DistributedInverseRobustnessSeeds,
)
from thermotwin.studies.distributed_withheld_validation import (
    DistributedWithheldPredictionMetrics,
    DistributedWithheldValidationConfig,
    DistributedWithheldValidationStudyResult,
    DistributedWithheldValidationTrial,
    summarize_distributed_withheld_validation,
)


def _result():
    metrics = DistributedWithheldPredictionMetrics(
        cold_face_rmse=0.001,
        hot_face_rmse=0.002,
        internal_temperature_rmse=0.0015,
        voltage_rmse=2.0e-6,
        maximum_absolute_temperature_error=0.004,
        maximum_energy_balance_residual=1.0e-15,
    )
    trial = DistributedWithheldValidationTrial(
        trial_index=0,
        seeds=DistributedInverseRobustnessSeeds(100, (101, 102, 103)),
        conventional_multipliers=(1.04, 1.07, 1.03),
        conventional_maximum_absolute_multiplier_error=0.0,
        conventional_reached_search_bound=False,
        conventional_prediction=metrics,
        conventional_prediction_success=True,
        conventional_prediction_failure_reasons=(),
        inverse_pinn_multipliers=(1.05, 1.06, 1.04),
        inverse_pinn_maximum_absolute_multiplier_error=0.01,
        inverse_pinn_prediction=metrics,
        inverse_pinn_prediction_success=True,
        inverse_pinn_prediction_failure_reasons=(),
    )
    return DistributedWithheldValidationStudyResult(
        config=DistributedWithheldValidationConfig(trial_count=1),
        withheld_experiment_name="positive_0.4A_20K_lift",
        truth_multipliers=(1.04, 1.07, 1.03),
        trials=(trial,),
        summary=summarize_distributed_withheld_validation((trial,)),
    )


class DistributedWithheldValidationReportTests(unittest.TestCase):
    def test_report_states_exclusion_no_refit_gate_and_every_trial(self):
        text = format_distributed_withheld_validation_report(_result())
        self.assertIn("withheld complete regime: positive_0.4A_20K_lift", text)
        self.assertIn("no withheld refitting", text)
        self.assertIn("internal-field RMSE", text)
        self.assertIn("conventional: PASS", text)
        self.assertIn("inverse PINN: PASS", text)
        self.assertIn("internal field is hidden during fitting", text)

    def test_figure_is_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "withheld.png"
            returned = save_distributed_withheld_validation_figure(
                _result(), output
            )
            self.assertEqual(returned, output.resolve())
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 1_000)


if __name__ == "__main__":
    unittest.main()
