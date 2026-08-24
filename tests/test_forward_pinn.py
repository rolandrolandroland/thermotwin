from dataclasses import replace
import unittest

from thermotwin import (
    PiecewiseConstantCurrent,
    constant_current_reference_experiment,
)

try:
    import torch

    from thermotwin.forward_pinn import (
        ForwardPINN,
        ForwardPINNConfig,
        physics_residuals,
        predict_trajectory,
        train_forward_pinn,
        validate_against_rk4,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
class ForwardPINNTests(unittest.TestCase):
    def setUp(self):
        self.experiment = constant_current_reference_experiment()

    def test_initial_temperatures_are_enforced_exactly(self):
        model = ForwardPINN(
            duration=self.experiment.duration,
            initial_cold_temperature=300.0,
            initial_hot_temperature=300.0,
            hidden_width=8,
            hidden_layers=1,
        )

        temperatures = model(torch.tensor([[0.0], [60.0]]))

        self.assertEqual(tuple(temperatures.shape), (2, 2))
        self.assertEqual(float(temperatures[0, 0].detach()), 300.0)
        self.assertEqual(float(temperatures[0, 1].detach()), 300.0)

    def test_initial_hand_calculated_rates_make_both_residuals_zero(self):
        class InitialLinearRates(torch.nn.Module):
            def forward(self, time):
                return torch.cat(
                    (
                        300.0 - 0.14 * time,
                        300.0 + 0.08 * time,
                    ),
                    dim=1,
                )

        residuals = physics_residuals(
            InitialLinearRates(),
            torch.tensor([[0.0]]),
            self.experiment,
        )

        self.assertAlmostEqual(
            float(residuals.cold.detach()), 0.0, places=6
        )
        self.assertAlmostEqual(
            float(residuals.hot.detach()), 0.0, places=6
        )

    def test_zero_current_equilibrium_has_zero_residual_for_all_times(self):
        equilibrium_experiment = replace(self.experiment, current=0.0)
        model = ForwardPINN(
            duration=equilibrium_experiment.duration,
            initial_cold_temperature=300.0,
            initial_hot_temperature=300.0,
            hidden_width=8,
            hidden_layers=1,
        )
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)

        residuals = physics_residuals(
            model,
            torch.tensor([[0.0], [30.0], [60.0]]),
            equilibrium_experiment,
        )

        self.assertTrue(
            torch.allclose(
                residuals.cold,
                torch.zeros_like(residuals.cold),
            )
        )
        self.assertTrue(
            torch.allclose(
                residuals.hot,
                torch.zeros_like(residuals.hot),
            )
        )

    def test_first_forward_pinn_rejects_switching_current(self):
        pulse_experiment = replace(
            self.experiment,
            current=PiecewiseConstantCurrent.pulse(
                start_time=10.0,
                end_time=30.0,
                pulse_current=1.0,
            ),
        )

        with self.assertRaisesRegex(ValueError, "constant current only"):
            train_forward_pinn(
                pulse_experiment,
                ForwardPINNConfig(epochs=1),
            )

    def test_short_cpu_training_reduces_physics_loss(self):
        training = train_forward_pinn(
            self.experiment,
            ForwardPINNConfig(
                hidden_width=16,
                hidden_layers=2,
                collocation_points=64,
                epochs=300,
                learning_rate=5e-3,
                seed=7,
                device="cpu",
            ),
        )

        self.assertEqual(training.device, "cpu")
        self.assertLess(training.loss_history[-1], training.loss_history[0])
        trajectory = predict_trajectory(
            training.model,
            (0.0, 30.0, 60.0),
        )
        self.assertEqual(trajectory.time, (0.0, 30.0, 60.0))
        self.assertEqual(trajectory.cold[0], 300.0)
        self.assertEqual(trajectory.hot[0], 300.0)

        validation = validate_against_rk4(
            training.model,
            self.experiment,
        )
        self.assertLess(validation.cold_rmse, 0.5)
        self.assertLess(validation.hot_rmse, 0.5)


if __name__ == "__main__":
    unittest.main()
