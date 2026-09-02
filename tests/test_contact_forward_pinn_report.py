from pathlib import Path
import tempfile
import unittest

from thermotwin import constant_current_contact_reference_experiment

try:
    import torch

    from thermotwin.contact_forward_pinn import (
        ContactForwardPINN,
        ContactPINNTrainingResult,
    )
    from thermotwin.contact_forward_pinn_report import (
        DEFAULT_CONTACT_FORWARD_PINN_REPORT_PATH,
        build_contact_forward_pinn_report_data,
        save_contact_forward_pinn_report,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(
    torch is None,
    "optional PyTorch and Matplotlib dependencies are not installed",
)
class ContactForwardPINNReportTests(unittest.TestCase):
    def setUp(self):
        self.experiment = constant_current_contact_reference_experiment()
        model = ContactForwardPINN(
            duration=self.experiment.duration,
            initial_temperatures=(300.0, 300.0, 300.0, 300.0),
            hidden_width=8,
            hidden_layers=1,
        )
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        self.training = ContactPINNTrainingResult(
            model=model,
            loss_history=(2.0, 1.0),
            device="cpu",
        )

    def test_default_path_uses_package_figures_directory(self):
        self.assertEqual(
            DEFAULT_CONTACT_FORWARD_PINN_REPORT_PATH.name,
            "contact_forward_pinn_comparison.png",
        )
        self.assertEqual(
            DEFAULT_CONTACT_FORWARD_PINN_REPORT_PATH.parent.name,
            "CONTACT_RESISTANCE_EXPERIMENT",
        )
        self.assertEqual(
            DEFAULT_CONTACT_FORWARD_PINN_REPORT_PATH.parent.parent.name,
            "figures",
        )

    def test_report_histories_are_aligned_and_initial_errors_are_zero(self):
        report = build_contact_forward_pinn_report_data(
            self.training,
            self.experiment,
        )

        sample_count = len(report.time)
        self.assertGreater(sample_count, 1)
        excluded_fields = {"epoch", "loss", "validation"}
        for name, history in zip(report._fields, report):
            if name not in excluded_fields:
                self.assertEqual(len(history), sample_count, name)
        for error in (
            report.cold_face_error[0],
            report.hot_face_error[0],
            report.cold_exchanger_error[0],
            report.hot_exchanger_error[0],
        ):
            self.assertEqual(error, 0.0)
        self.assertEqual(report.reference_cold_contact_drop[0], 0.0)
        self.assertEqual(report.reference_hot_contact_drop[0], 0.0)
        self.assertEqual(report.epoch, (1, 2))
        self.assertEqual(report.loss, (2.0, 1.0))

    def test_report_writer_creates_png(self):
        report = build_contact_forward_pinn_report_data(
            self.training,
            self.experiment,
        )

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "contact_pinn.png"
            written_path = save_contact_forward_pinn_report(
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
