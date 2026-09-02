from pathlib import Path
import tempfile
import unittest

try:
    import torch

    from thermotwin.inverse_contact_resistance import (
        InverseContactTrainingHistory,
    )
    from thermotwin.piecewise_inverse_contact_resistance import (
        PiecewiseInverseContactResistancePINN,
        PiecewiseInverseContactTrainingResult,
        ideal_piecewise_inverse_contact_problem,
    )
    from thermotwin.piecewise_inverse_contact_resistance_report import (
        DEFAULT_PIECEWISE_INVERSE_CONTACT_RESISTANCE_REPORT_PATH,
        build_piecewise_inverse_contact_resistance_report_data,
        save_piecewise_inverse_contact_resistance_report,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(
    torch is None,
    "optional PyTorch and Matplotlib dependencies are not installed",
)
class PiecewiseInverseContactResistanceReportTests(unittest.TestCase):
    def setUp(self):
        self.problem = ideal_piecewise_inverse_contact_problem()
        experiment = self.problem.experiment
        model = PiecewiseInverseContactResistancePINN(
            duration=experiment.duration,
            transition_times=(5.0, 20.0),
            initial_temperatures=(300.0,) * 4,
            initial_cold_contact_resistance=0.5,
            hidden_width=8,
            hidden_layers=1,
        )
        for parameter in model.temperature_model.parameters():
            torch.nn.init.zeros_(parameter)
        self.training = PiecewiseInverseContactTrainingResult(
            model=model,
            history=InverseContactTrainingHistory(
                total_loss=(2.0, 1.0),
                physics_loss=(1.5, 0.7),
                observation_loss=(0.5, 0.3),
                cold_contact_resistance=(0.5, 0.4),
            ),
            device="cpu",
            collocation_time=(1.0, 10.0, 40.0),
        )

    def test_default_path_uses_package_figures_directory(self):
        self.assertEqual(
            DEFAULT_PIECEWISE_INVERSE_CONTACT_RESISTANCE_REPORT_PATH.name,
            "piecewise_inverse_contact_resistance_comparison.png",
        )
        self.assertEqual(
            DEFAULT_PIECEWISE_INVERSE_CONTACT_RESISTANCE_REPORT_PATH.parent.name,
            "CONTACT_RESISTANCE_EXPERIMENT",
        )
        self.assertEqual(
            DEFAULT_PIECEWISE_INVERSE_CONTACT_RESISTANCE_REPORT_PATH.parent.parent.name,
            "figures",
        )

    def test_report_histories_align_and_switches_are_right_continuous(self):
        report = build_piecewise_inverse_contact_resistance_report_data(
            self.training,
            self.problem,
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
            report.reference_cold_contact_drop,
            report.predicted_cold_contact_drop,
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
        self.assertEqual(len(report.observation_time), 61)
        self.assertEqual(report.validation.max_boundary_temperature_jump, 0.0)
        self.assertEqual(report.epoch, (1, 2))
        self.assertEqual(report.inferred_resistance, (0.5, 0.4))

    def test_misaligned_history_is_rejected(self):
        invalid_training = self.training._replace(
            history=self.training.history._replace(
                physics_loss=(1.0,),
            )
        )

        with self.assertRaisesRegex(ValueError, "equal lengths"):
            build_piecewise_inverse_contact_resistance_report_data(
                invalid_training,
                self.problem,
            )

    def test_report_writer_creates_png(self):
        report = build_piecewise_inverse_contact_resistance_report_data(
            self.training,
            self.problem,
        )

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "piecewise_inverse_contact.png"
            written_path = save_piecewise_inverse_contact_resistance_report(
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
