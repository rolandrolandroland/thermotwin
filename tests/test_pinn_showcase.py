from pathlib import Path
import tempfile
import unittest

try:
    import torch

    from thermotwin.inverse_contact_resistance import (
        InverseContactTrainingHistory,
    )
    from thermotwin.piecewise_contact_forward_pinn import (
        PiecewiseContactForwardPINN,
        PiecewiseContactPINNTrainingResult,
    )
    from thermotwin.piecewise_inverse_contact_resistance import (
        PiecewiseInverseContactResistancePINN,
        PiecewiseInverseContactTrainingResult,
        ideal_piecewise_inverse_contact_problem,
    )
    from thermotwin.pinn_showcase import (
        DEFAULT_PINN_SHOWCASE_PATH,
        PINNShowcaseConfig,
        build_pinn_showcase_data,
        save_pinn_showcase,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(
    torch is None,
    "optional PyTorch and Matplotlib dependencies are not installed",
)
class PINNShowcaseTests(unittest.TestCase):
    def setUp(self):
        forward_model = PiecewiseContactForwardPINN(
            duration=60.0,
            transition_times=(5.0, 20.0),
            initial_temperatures=(300.0,) * 4,
            hidden_width=8,
            hidden_layers=1,
        )
        for parameter in forward_model.parameters():
            torch.nn.init.zeros_(parameter)
        self.forward_training = PiecewiseContactPINNTrainingResult(
            model=forward_model,
            loss_history=(2.0, 1.0),
            device="cpu",
            collocation_time=(1.0, 10.0, 40.0),
        )

        self.inverse_problem = ideal_piecewise_inverse_contact_problem()
        inverse_model = PiecewiseInverseContactResistancePINN(
            duration=60.0,
            transition_times=(5.0, 20.0),
            initial_temperatures=(300.0,) * 4,
            initial_cold_contact_resistance=0.5,
            hidden_width=8,
            hidden_layers=1,
        )
        for parameter in inverse_model.temperature_model.parameters():
            torch.nn.init.zeros_(parameter)
        self.inverse_training = PiecewiseInverseContactTrainingResult(
            model=inverse_model,
            history=InverseContactTrainingHistory(
                total_loss=(2.0, 1.0),
                physics_loss=(1.5, 0.7),
                observation_loss=(0.5, 0.3),
                cold_contact_resistance=(0.5, 0.4),
            ),
            device="cpu",
            collocation_time=(1.0, 10.0, 40.0),
        )

    def test_default_path_uses_ignored_figures_directory(self):
        self.assertEqual(DEFAULT_PINN_SHOWCASE_PATH.name, "pinn_showcase.png")
        self.assertEqual(DEFAULT_PINN_SHOWCASE_PATH.parent.name, "figures")

    def test_configuration_rejects_invalid_values(self):
        for keyword, value in (
            ("forward_epochs", 0),
            ("inverse_epochs", True),
            ("device", "cuda"),
        ):
            with self.subTest(keyword=keyword):
                with self.assertRaises(ValueError):
                    PINNShowcaseConfig(**{keyword: value})

    def test_showcase_combines_aligned_forward_and_inverse_evidence(self):
        showcase = build_pinn_showcase_data(
            self.forward_training,
            self.inverse_training,
            self.inverse_problem,
        )

        self.assertEqual(showcase.forward.time, showcase.inverse.time)
        self.assertEqual(showcase.forward.current, showcase.inverse.current)
        self.assertEqual(showcase.forward.switch_times, (5.0, 20.0))
        self.assertEqual(len(showcase.inverse.observation_time), 61)
        self.assertEqual(
            showcase.forward.max_boundary_temperature_jump,
            0.0,
        )
        self.assertEqual(
            showcase.inverse.validation.max_boundary_temperature_jump,
            0.0,
        )

    def test_showcase_writer_creates_png(self):
        showcase = build_pinn_showcase_data(
            self.forward_training,
            self.inverse_training,
            self.inverse_problem,
        )

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "showcase.png"
            written_path = save_pinn_showcase(showcase, destination)

            self.assertEqual(written_path, destination.resolve())
            self.assertGreater(destination.stat().st_size, 1_000)
            self.assertEqual(
                destination.read_bytes()[:8],
                b"\x89PNG\r\n\x1a\n",
            )


if __name__ == "__main__":
    unittest.main()
