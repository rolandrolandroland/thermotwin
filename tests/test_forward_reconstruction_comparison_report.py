from functools import lru_cache
from pathlib import Path
import tempfile
import unittest

try:
    import matplotlib  # noqa: F401
    import torch  # noqa: F401

    from thermotwin.reports.forward_reconstruction_comparison import (
        format_forward_reconstruction_comparison_report,
        save_forward_reconstruction_comparison_figure,
    )
    from thermotwin.studies.forward_reconstruction_comparison import (
        ForwardReconstructionComparisonConfig,
        run_forward_reconstruction_comparison,
    )
except ModuleNotFoundError:
    matplotlib = None


@lru_cache(maxsize=1)
def _result():
    return run_forward_reconstruction_comparison(
        ForwardReconstructionComparisonConfig(
            trial_count=1,
            epochs=1,
            hidden_width=4,
            hidden_layers=1,
            collocation_points=12,
            energy_sampling_interval=2.0,
        )
    )


@unittest.skipIf(matplotlib is None, "optional report dependencies are absent")
class ForwardReconstructionComparisonReportTests(unittest.TestCase):
    def test_text_states_matching_and_energy_boundary(self):
        text = format_forward_reconstruction_comparison_report(_result())

        self.assertIn("bit-identical weights", text)
        self.assertIn("not an independent", text)
        self.assertIn("physical law", text)
        self.assertIn("not hardware validation", text)

    def test_figure_writes_png_and_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.png"
            returned = save_forward_reconstruction_comparison_figure(
                _result(), output
            )

            self.assertEqual(returned, output.resolve())
            self.assertTrue(output.exists())
            self.assertTrue(output.with_suffix(".json").exists())
            self.assertTrue(output.with_suffix(".txt").exists())


if __name__ == "__main__":
    unittest.main()
