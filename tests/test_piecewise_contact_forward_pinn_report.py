from pathlib import Path
import tempfile
import unittest

try:
    import torch

    from thermotwin.piecewise_contact_forward_pinn import (
        PiecewiseContactForwardPINN,
        PiecewiseContactPINNTrainingResult,
        unipolar_pulse_contact_experiment,
    )
    from thermotwin.piecewise_contact_forward_pinn_report import (
        DEFAULT_PIECEWISE_CONTACT_FORWARD_PINN_REPORT_PATH,
        build_piecewise_contact_forward_pinn_report_data,
        save_piecewise_contact_forward_pinn_report,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(
    torch is None,
    "optional PyTorch and Matplotlib dependencies are not installed",
)
class PiecewiseContactForwardPINNReportTests(unittest.TestCase):
    def setUp(self):
        self.experiment = unipolar_pulse_contact_experiment()
        model = PiecewiseContactForwardPINN(
            duration=self.experiment.duration,
            transition_times=(5.0, 20.0),
            initial_temperatures=(300.0, 300.0, 300.0, 300.0),
            hidden_width=8,
            hidden_layers=1,
        )
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        self.training = PiecewiseContactPINNTrainingResult(
            model=model,
            loss_history=(2.0, 1.0),
            device="cpu",
            collocation_time=(1.0, 10.0, 40.0),
        )

    def test_default_path_uses_package_figures_directory(self):
        self.assertEqual(
            DEFAULT_PIECEWISE_CONTACT_FORWARD_PINN_REPORT_PATH.name,
            "piecewise_contact_forward_pinn_comparison.png",
        )
        self.assertEqual(
            DEFAULT_PIECEWISE_CONTACT_FORWARD_PINN_REPORT_PATH.parent.name,
            "CONTACT_RESISTANCE_EXPERIMENT",
        )
        self.assertEqual(
            DEFAULT_PIECEWISE_CONTACT_FORWARD_PINN_REPORT_PATH.parent.parent.name,
            "figures",
        )

    def test_report_histories_are_aligned_and_switches_are_right_continuous(self):
        report = build_piecewise_contact_forward_pinn_report_data(
            self.training,
            self.experiment,
        )

        sample_count = len(report.time)
        self.assertGreater(sample_count, 1)
        for history in (
            report.current,
            report.reference_cold_face,
            report.reference_hot_face,
            report.reference_cold_exchanger,
            report.reference_hot_exchanger,
            report.predicted_cold_face,
            report.predicted_hot_face,
            report.predicted_cold_exchanger,
            report.predicted_hot_exchanger,
            report.cold_face_error,
            report.hot_face_error,
            report.cold_exchanger_error,
            report.hot_exchanger_error,
            report.cold_face_residual,
            report.hot_face_residual,
            report.cold_exchanger_residual,
            report.hot_exchanger_residual,
        ):
            self.assertEqual(len(history), sample_count)
        def current_near(target):
            index = min(
                range(sample_count),
                key=lambda item: abs(report.time[item] - target),
            )
            self.assertAlmostEqual(report.time[index], target, places=10)
            return report.current[index]

        self.assertEqual(current_near(4.9), 0.0)
        self.assertEqual(current_near(5.0), 1.0)
        self.assertEqual(current_near(19.9), 1.0)
        self.assertEqual(current_near(20.0), 0.0)
        self.assertEqual(report.switch_times, (5.0, 20.0))
        self.assertEqual(report.max_boundary_temperature_jump, 0.0)
        self.assertEqual(report.epoch, (1, 2))

    def test_report_writer_creates_png(self):
        report = build_piecewise_contact_forward_pinn_report_data(
            self.training,
            self.experiment,
        )

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "piecewise_contact.png"
            written_path = save_piecewise_contact_forward_pinn_report(
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
