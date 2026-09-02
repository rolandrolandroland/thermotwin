from functools import lru_cache
from pathlib import Path
import tempfile
import unittest

try:
    import matplotlib  # noqa: F401

    from thermotwin.inference.joint_thermal_parameters import JointThermalFitConfig
    from thermotwin.reports.nonlinear_experiment_selection import (
        format_nonlinear_experiment_selection_report,
        save_nonlinear_experiment_selection_figure,
    )
    from thermotwin.studies.nonlinear_experiment_selection import (
        NonlinearExperimentSelectionConfig,
        run_nonlinear_experiment_selection_study,
    )
except ModuleNotFoundError:
    matplotlib = None


@lru_cache(maxsize=1)
def _result():
    return run_nonlinear_experiment_selection_study(
        NonlinearExperimentSelectionConfig(
            trial_count=1,
            profile_log_offsets=(0.0,),
            fit=JointThermalFitConfig(gauss_newton_iterations=3),
        ),
        include_profiles=True,
    )


@unittest.skipIf(matplotlib is None, "optional Matplotlib dependency is absent")
class NonlinearExperimentSelectionReportTests(unittest.TestCase):
    def test_text_names_resource_control_rank_and_synthetic_boundary(self):
        text = format_nonlinear_experiment_selection_report(_result())

        self.assertIn("closest-energy grid control", text)
        self.assertIn("zero-current supported rank: 0/3", text)
        self.assertIn("not hardware identifiability", text)

    def test_figure_writes_png_and_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nonlinear.png"
            returned = save_nonlinear_experiment_selection_figure(_result(), output)

            self.assertEqual(returned, output.resolve())
            self.assertTrue(output.exists())
            self.assertTrue(output.with_suffix(".json").exists())
            self.assertTrue(output.with_suffix(".txt").exists())


if __name__ == "__main__":
    unittest.main()
