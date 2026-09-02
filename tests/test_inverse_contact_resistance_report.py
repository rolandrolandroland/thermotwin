from pathlib import Path
import tempfile
import unittest

try:
    import torch

    from thermotwin.inverse_contact_resistance import (
        InverseContactResistancePINN,
        InverseContactTrainingHistory,
        InverseContactTrainingResult,
        ideal_inverse_contact_problem,
    )
    from thermotwin.inverse_contact_resistance_report import (
        DEFAULT_INVERSE_CONTACT_RESISTANCE_REPORT_PATH,
        build_inverse_contact_resistance_report_data,
        save_inverse_contact_resistance_report,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(
    torch is None,
    "optional PyTorch and Matplotlib dependencies are not installed",
)
class InverseContactResistanceReportTests(unittest.TestCase):
    def setUp(self):
        self.problem = ideal_inverse_contact_problem()
        experiment = self.problem.experiment
        model = InverseContactResistancePINN(
            duration=experiment.duration,
            initial_temperatures=(300.0, 300.0, 300.0, 300.0),
            initial_cold_contact_resistance=0.5,
            hidden_width=8,
            hidden_layers=1,
        )
        for parameter in model.temperature_model.parameters():
            torch.nn.init.zeros_(parameter)
        self.training = InverseContactTrainingResult(
            model=model,
            history=InverseContactTrainingHistory(
                total_loss=(2.0, 1.0),
                physics_loss=(1.5, 0.7),
                observation_loss=(0.5, 0.3),
                cold_contact_resistance=(0.5, 0.4),
            ),
            device="cpu",
        )

    def test_default_path_uses_package_figures_directory(self):
        self.assertEqual(
            DEFAULT_INVERSE_CONTACT_RESISTANCE_REPORT_PATH.name,
            "inverse_contact_resistance_comparison.png",
        )
        self.assertEqual(
            DEFAULT_INVERSE_CONTACT_RESISTANCE_REPORT_PATH.parent.name,
            "CONTACT_RESISTANCE_EXPERIMENT",
        )
        self.assertEqual(
            DEFAULT_INVERSE_CONTACT_RESISTANCE_REPORT_PATH.parent.parent.name,
            "figures",
        )

    def test_report_histories_are_aligned_and_initial_errors_are_zero(self):
        report = build_inverse_contact_resistance_report_data(
            self.training,
            self.problem,
        )

        sample_count = len(report.time)
        self.assertGreater(sample_count, 1)
        for history in (
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
            report.reference_cold_contact_drop,
            report.predicted_cold_contact_drop,
        ):
            self.assertEqual(len(history), sample_count)
        for error in (
            report.cold_face_error[0],
            report.hot_face_error[0],
            report.cold_exchanger_error[0],
            report.hot_exchanger_error[0],
        ):
            self.assertEqual(error, 0.0)
        self.assertEqual(len(report.observation_time), 13)
        self.assertEqual(report.epoch, (1, 2))
        self.assertEqual(report.inferred_resistance, (0.5, 0.4))

    def test_report_writer_creates_png(self):
        report = build_inverse_contact_resistance_report_data(
            self.training,
            self.problem,
        )

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "inverse_contact.png"
            written_path = save_inverse_contact_resistance_report(
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
