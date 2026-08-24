from pathlib import Path
import tempfile
import unittest

try:
    import matplotlib

    from thermotwin.assembly_fingerprint import AssemblySpecification
    from thermotwin.control_comparison import ControlComparisonConfig
    from thermotwin.engineering_showcase import (
        DEFAULT_ENGINEERING_SHOWCASE_PATH,
        run_engineering_showcase,
        save_engineering_showcase,
    )
    from thermotwin.experiment_selection import ExperimentSelectionConfig
    from thermotwin.sparse_sensor_inference import SparseSensorInferenceConfig
except ModuleNotFoundError:
    matplotlib = None


@unittest.skipIf(matplotlib is None, "optional Matplotlib dependency is unavailable")
class EngineeringShowcaseTests(unittest.TestCase):
    def test_default_output_uses_ignored_figures_directory(self):
        self.assertEqual(
            DEFAULT_ENGINEERING_SHOWCASE_PATH.name,
            "engineering_decision_showcase.png",
        )
        self.assertEqual(DEFAULT_ENGINEERING_SHOWCASE_PATH.parent.name, "figures")

    def test_small_showcase_runs_and_writes_png(self):
        result = run_engineering_showcase(
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

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "showcase.png"
            written = save_engineering_showcase(result, destination)

            self.assertEqual(written, destination.resolve())
            self.assertGreater(destination.stat().st_size, 5_000)
            self.assertEqual(destination.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
