from pathlib import Path
import tempfile
import unittest

try:
    import matplotlib

    from thermotwin.assembly_fingerprint import AssemblySpecification
    from thermotwin.control_comparison import ControlComparisonConfig
    from thermotwin.experiment_selection import ExperimentSelectionConfig
    from thermotwin.reports.assembly_fingerprint import (
        DEFAULT_ASSEMBLY_FINGERPRINT_PATH,
        save_assembly_fingerprint_report,
    )
    from thermotwin.reports.control_comparison import (
        DEFAULT_CONTROL_COMPARISON_PATH,
        save_control_comparison_report,
    )
    from thermotwin.reports.engineering_showcase import run_engineering_showcase
    from thermotwin.reports.experiment_selection import (
        DEFAULT_EXPERIMENT_SELECTION_PATH,
        save_experiment_selection_report,
    )
    from thermotwin.reports.sparse_sensor import (
        DEFAULT_SPARSE_SENSOR_PATH,
        save_sparse_sensor_report,
    )
    from thermotwin.sparse_sensor_inference import SparseSensorInferenceConfig
except ModuleNotFoundError:
    matplotlib = None


@unittest.skipIf(matplotlib is None, "optional Matplotlib dependency is unavailable")
class ExperimentResultReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if matplotlib is None:
            return
        cls.result = run_engineering_showcase(
            sparse_config=SparseSensorInferenceConfig(
                grid_points_per_axis=5,
                refinement_count=2,
            ),
            control_config=ControlComparisonConfig(
                warmup_duration=160.0,
                evaluation_duration=40.0,
                time_step=0.4,
                target_cooling_rates=(2.0,),
                pulse_periods=(10.0,),
                pulse_duty_cycles=(0.75,),
                maximum_storage_drift=0.2,
            ),
            selection_config=ExperimentSelectionConfig(
                current_amplitudes=(0.4, 0.8),
                pulse_durations=(5.0, 20.0),
                monte_carlo_trials=20,
            ),
            assemblies=(
                AssemblySpecification("reference", 0.25),
                AssemblySpecification("elevated", 0.4),
            ),
        )

    def test_default_paths_match_walkthrough_directories(self):
        expected = (
            (DEFAULT_ASSEMBLY_FINGERPRINT_PATH, "ASSEMBLY_FINGERPRINT_EXPERIMENT"),
            (DEFAULT_CONTROL_COMPARISON_PATH, "CONTROL_COMPARISON_EXPERIMENT"),
            (DEFAULT_EXPERIMENT_SELECTION_PATH, "NEXT_EXPERIMENT_WALKTHROUGH"),
            (DEFAULT_SPARSE_SENSOR_PATH, "SPARSE_SENSOR_EXPERIMENT"),
        )
        for path, directory in expected:
            with self.subTest(path=path):
                self.assertEqual(path.parent.name, directory)

    def test_each_report_writes_png_json_and_explanation(self):
        savers = (
            (save_assembly_fingerprint_report, self.result.assembly_fingerprints),
            (save_control_comparison_report, self.result.control_comparison),
            (save_experiment_selection_report, self.result.experiment_selection),
            (save_sparse_sensor_report, self.result.sparse_inference),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, (saver, result) in enumerate(savers):
                destination = Path(directory) / f"report_{index}.png"
                with self.subTest(saver=saver.__name__):
                    saver(result, destination)
                    self.assertEqual(
                        destination.read_bytes()[:8],
                        b"\x89PNG\r\n\x1a\n",
                    )
                    self.assertTrue(destination.with_suffix(".json").is_file())
                    explanation = destination.with_suffix(".txt")
                    self.assertTrue(explanation.is_file())
                    self.assertIn(
                        "Interpretation boundary",
                        explanation.read_text(encoding="utf-8"),
                    )


if __name__ == "__main__":
    unittest.main()
