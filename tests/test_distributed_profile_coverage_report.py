from pathlib import Path
import tempfile
import unittest

from thermotwin.reports.distributed_profile_coverage import (
    build_parser,
    format_distributed_profile_coverage_report,
    save_distributed_profile_coverage_figure,
)
from thermotwin.studies.distributed_independent_validation import (
    smooth_resistivity_truth,
)
from thermotwin.studies.distributed_profile_coverage import (
    DistributedProfileCoverageConfig,
    DistributedProfileCoverageEstimatorResult,
    DistributedProfileCoverageSeeds,
    DistributedProfileCoverageStudyResult,
    DistributedProfileCoverageTrial,
    summarize_distributed_profile_coverage,
)


def _result():
    estimator = DistributedProfileCoverageEstimatorResult(
        name="conventional_unregularized",
        multipliers=(1.04, 1.07, 1.03),
        property_relative_rmse=0.01,
        property_maximum_relative_error=0.02,
        normalized_observation_loss=1.0,
        intervals_68=(),
        intervals_95=(),
        coefficient_coverage_68=(),
        coefficient_coverage_95=(),
        holdout_internal_temperature_rmse=0.001,
        holdout_voltage_rmse=1.0e-5,
    )
    trial = DistributedProfileCoverageTrial(
        trial_index=0,
        seeds=DistributedProfileCoverageSeeds(100, (101, 102, 103)),
        estimators=(estimator,),
    )
    summaries = summarize_distributed_profile_coverage((trial,))
    return DistributedProfileCoverageStudyResult(
        config=DistributedProfileCoverageConfig(
            trial_count=1,
            pinn_trial_count=0,
            inverse_pinn_epochs=1,
            truth_node_count=9,
            truth_time_step=5.0e-4,
            representative_profile_points=3,
        ),
        truth_property=smooth_resistivity_truth(),
        truth_knot_multipliers=(1.04, 1.07, 1.03),
        representative_profiles=(),
        trials=(trial,),
        summaries=summaries,
    )


class DistributedProfileCoverageReportTests(unittest.TestCase):
    def test_report_states_profile_approximation_and_coverage_limits(self):
        text = format_distributed_profile_coverage_report(_result())
        self.assertIn("how wide are the supported", text)
        self.assertIn("nonlinear re-optimized profiles", text)
        self.assertIn("quadratic profile approximation", text)
        self.assertIn("not classical confidence", text)
        self.assertIn("wide binomial uncertainty", text)

    def test_figure_is_written_when_representative_profiles_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "coverage.png"
            returned = save_distributed_profile_coverage_figure(_result(), output)
            self.assertEqual(returned, output.resolve())
            self.assertGreater(output.stat().st_size, 0)

    def test_parser_accepts_persistent_text_report_path(self):
        arguments = build_parser().parse_args(
            ["--report-output", "coverage.txt", "--skip-profiles"]
        )
        self.assertEqual(arguments.report_output, Path("coverage.txt"))
        self.assertTrue(arguments.skip_profiles)


if __name__ == "__main__":
    unittest.main()
