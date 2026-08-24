from pathlib import Path
import tempfile
import unittest

from thermotwin import constant_current_reference_experiment

try:
    import torch

    from thermotwin.forward_pinn import ForwardPINN, PINNTrainingResult
    from thermotwin.forward_pinn_report import (
        DEFAULT_FORWARD_PINN_REPORT_PATH,
        build_forward_pinn_report_data,
        save_forward_pinn_comparison_report,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(
    torch is None,
    "optional PyTorch and Matplotlib dependencies are not installed",
)
class ForwardPINNReportTests(unittest.TestCase):
    def setUp(self):
        self.experiment = constant_current_reference_experiment()
        model = ForwardPINN(
            duration=self.experiment.duration,
            initial_cold_temperature=self.experiment.initial_cold_temperature,
            initial_hot_temperature=self.experiment.initial_hot_temperature,
            hidden_width=8,
            hidden_layers=1,
        )
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        self.training = PINNTrainingResult(
            model=model,
            loss_history=(1.0, 0.5),
            device="cpu",
        )

    def test_default_report_path_uses_package_figures_directory(self):
        self.assertEqual(
            DEFAULT_FORWARD_PINN_REPORT_PATH.name,
            "forward_pinn_comparison.png",
        )
        self.assertEqual(
            DEFAULT_FORWARD_PINN_REPORT_PATH.parent.name,
            "figures",
        )
        self.assertEqual(
            DEFAULT_FORWARD_PINN_REPORT_PATH.parent.parent.name,
            "thermotwin",
        )

    def test_report_data_are_aligned_and_initial_error_is_zero(self):
        report = build_forward_pinn_report_data(
            self.training,
            self.experiment,
        )

        sample_count = len(report.time)
        self.assertGreater(sample_count, 1)
        for history in (
            report.reference_cold,
            report.reference_hot,
            report.predicted_cold,
            report.predicted_hot,
            report.cold_error,
            report.hot_error,
            report.cold_residual,
            report.hot_residual,
        ):
            self.assertEqual(len(history), sample_count)
        self.assertEqual(report.cold_error[0], 0.0)
        self.assertEqual(report.hot_error[0], 0.0)
        self.assertEqual(report.epoch, (1, 2))
        self.assertEqual(report.loss, (1.0, 0.5))

    def test_report_writer_creates_a_png(self):
        report = build_forward_pinn_report_data(
            self.training,
            self.experiment,
        )

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "comparison.png"
            written_path = save_forward_pinn_comparison_report(
                report,
                destination,
            )

            self.assertEqual(written_path, destination.resolve())
            self.assertGreater(destination.stat().st_size, 1_000)
            self.assertEqual(
                destination.read_bytes()[:8],
                b"\x89PNG\r\n\x1a\n",
            )


if __name__ == "__main__":
    unittest.main()
