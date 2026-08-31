from pathlib import Path
import tempfile
import unittest

from thermotwin.reports.distributed_independent_validation import (
    format_distributed_independent_validation_report,
    save_distributed_independent_validation_figure,
)
from thermotwin.studies.distributed_independent_validation import (
    ESTIMATOR_NAMES,
    DistributedIndependentEstimatorResult,
    DistributedIndependentEstimatorSummary,
    DistributedIndependentValidationConfig,
    DistributedIndependentValidationSeeds,
    DistributedIndependentValidationStudyResult,
    DistributedIndependentValidationTrial,
    DistributedMismatchPredictionMetrics,
    smooth_resistivity_truth,
)


def _result():
    metrics = DistributedMismatchPredictionMetrics(
        cold_face_rmse=0.001,
        hot_face_rmse=0.001,
        internal_temperature_rmse=0.002,
        voltage_rmse=1.0e-5,
        maximum_absolute_temperature_error=0.004,
        maximum_prediction_energy_balance_residual=1.0e-15,
    )
    estimators = tuple(
        DistributedIndependentEstimatorResult(
            name=name,
            smoothness_weight=25.0 if name.endswith("matched") else 0.0,
            multipliers=(1.04, 1.07, 1.03),
            final_normalized_observation_loss=1.0,
            log_multiplier_roughness=0.004,
            in_support_property_relative_rmse=0.01,
            in_support_property_maximum_relative_error=0.02,
            extended_property_relative_rmse=0.03,
            predictions=(
                ("constant", metrics),
                ("pulse", metrics),
                ("outside", metrics),
            ),
            success=True,
            failure_reasons=(),
        )
        for name in ESTIMATOR_NAMES
    )
    summaries = tuple(
        DistributedIndependentEstimatorSummary(
            name=name,
            success_count=1,
            completed_count=1,
            mean_in_support_property_relative_rmse=0.01,
            mean_in_support_property_maximum_relative_error=0.02,
            mean_extended_property_relative_rmse=0.03,
            mean_in_support_internal_temperature_rmse=0.002,
            mean_in_support_voltage_rmse=1.0e-5,
        )
        for name in ESTIMATOR_NAMES
    )
    return DistributedIndependentValidationStudyResult(
        config=DistributedIndependentValidationConfig(
            trial_count=1, inverse_pinn_epochs=1
        ),
        truth_property=smooth_resistivity_truth(),
        truth_knot_multipliers=(1.04, 1.07, 1.03),
        training_experiment_names=("zero", "positive", "negative"),
        holdout_names=("constant", "pulse", "outside"),
        trials=(
            DistributedIndependentValidationTrial(
                trial_index=0,
                seeds=DistributedIndependentValidationSeeds(
                    100, (101, 102, 103)
                ),
                estimators=estimators,
            ),
        ),
        summaries=summaries,
    )


class DistributedIndependentValidationReportTests(unittest.TestCase):
    def test_report_states_numerical_independence_matched_prior_and_limits(self):
        text = format_distributed_independent_validation_report(_result())
        self.assertIn("node-centred grid", text)
        self.assertIn("matched weight", text)
        self.assertIn("implicit neural regularization", text)
        self.assertIn("outside-knot case is diagnostic", text)
        self.assertIn("same continuum thermoelectric equations", text)

    def test_figure_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "independent.png"
            returned = save_distributed_independent_validation_figure(
                _result(), output
            )
            self.assertEqual(returned, output.resolve())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
