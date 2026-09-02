from functools import lru_cache
from pathlib import Path
import tempfile
import unittest

try:
    import matplotlib  # noqa: F401
    import torch  # noqa: F401

    from thermotwin.reports.imperfect_inverse_pinn import (
        format_imperfect_inverse_pinn_report,
        save_imperfect_inverse_pinn_figure,
    )
    from thermotwin.studies.imperfect_inverse_pinn import (
        ImperfectInversePINNConfig,
        run_imperfect_inverse_pinn_study,
    )
except ModuleNotFoundError:
    matplotlib = None


@lru_cache(maxsize=1)
def _result():
    return run_imperfect_inverse_pinn_study(
        ImperfectInversePINNConfig(
            trial_count_per_case=1,
            initial_resistances=(0.5,),
            training_epochs=1,
            hidden_width=4,
            hidden_layers=1,
            collocation_points=8,
        )
    )


@unittest.skipIf(matplotlib is None, "optional report dependencies are absent")
class ImperfectInversePINNReportTests(unittest.TestCase):
    def test_text_names_identical_rows_and_mismatch_boundary(self):
        text = format_imperfect_inverse_pinn_report(_result())

        self.assertIn("identical transformed rows", text)
        self.assertIn("unmodeled_cold_face_bias", text)
        self.assertIn("not hardware validation", text)

    def test_figure_writes_png_and_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit.png"
            returned = save_imperfect_inverse_pinn_figure(_result(), output)

            self.assertEqual(returned, output.resolve())
            self.assertTrue(output.exists())
            self.assertTrue(output.with_suffix(".json").exists())
            self.assertTrue(output.with_suffix(".txt").exists())


if __name__ == "__main__":
    unittest.main()
