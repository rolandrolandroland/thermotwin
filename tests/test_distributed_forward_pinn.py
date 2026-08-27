from dataclasses import replace
import unittest

from thermotwin.core.controls import PiecewiseConstantCurrent
from thermotwin.simulation.distributed import distributed_reference_experiment

try:
    import torch

    from thermotwin.pinn.distributed_forward import (
        DistributedForwardPINN,
        DistributedForwardPINNConfig,
        distributed_physics_residuals,
        predict_distributed_temperature,
        train_distributed_forward_pinn,
        validate_distributed_forward_pinn,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
class DistributedForwardPINNTests(unittest.TestCase):
    def test_initial_linear_profile_is_enforced_exactly(self):
        model = DistributedForwardPINN(
            length=1.5e-3,
            duration=1.0,
            initial_cold_temperature=295.0,
            initial_hot_temperature=305.0,
            hidden_width=8,
            hidden_layers=1,
        )
        position = torch.tensor([[0.0], [0.75e-3], [1.5e-3]])
        temperature = model(position, torch.zeros_like(position))
        self.assertTrue(
            torch.allclose(
                temperature.detach().reshape(-1),
                torch.tensor([295.0, 300.0, 305.0]),
            )
        )

    def test_zero_current_uniform_equilibrium_has_zero_residual(self):
        experiment = distributed_reference_experiment(
            current=0.0,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
            duration=0.1,
            cell_count=4,
            time_step=0.001,
        )
        experiment = replace(
            experiment,
            initial_cold_face_temperature=300.0,
            initial_hot_face_temperature=300.0,
        )
        model = DistributedForwardPINN(
            length=experiment.geometry.length,
            duration=experiment.duration,
            initial_cold_temperature=300.0,
            initial_hot_temperature=300.0,
            hidden_width=8,
            hidden_layers=1,
        )
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        residuals = distributed_physics_residuals(
            model,
            torch.tensor([[0.5e-3], [1.0e-3]]),
            torch.tensor([[0.02], [0.08]]),
            torch.tensor([[0.02], [0.08]]),
            experiment,
        )
        self.assertTrue(torch.allclose(residuals.interior, torch.zeros_like(residuals.interior)))
        self.assertTrue(torch.allclose(residuals.cold_boundary, torch.zeros_like(residuals.cold_boundary)))
        self.assertTrue(torch.allclose(residuals.hot_boundary, torch.zeros_like(residuals.hot_boundary)))

    def test_switching_current_is_rejected_by_first_distributed_pinn(self):
        experiment = distributed_reference_experiment(
            current=PiecewiseConstantCurrent.step(
                transition_time=0.05,
                before_current=0.0,
                after_current=0.5,
            ),
            duration=0.1,
            cell_count=4,
            time_step=0.001,
        )
        with self.assertRaisesRegex(ValueError, "constant current only"):
            train_distributed_forward_pinn(
                experiment,
                DistributedForwardPINNConfig(epochs=1),
            )

    def test_short_cpu_training_reduces_physics_loss(self):
        experiment = distributed_reference_experiment(
            current=0.4,
            duration=0.08,
            cell_count=4,
            time_step=0.001,
        )
        training = train_distributed_forward_pinn(
            experiment,
            DistributedForwardPINNConfig(
                hidden_width=12,
                hidden_layers=2,
                interior_space_points=5,
                time_points=10,
                epochs=120,
                learning_rate=3e-3,
                seed=9,
                device="cpu",
            ),
        )
        self.assertLess(
            training.history.total_loss[-1],
            training.history.total_loss[0],
        )
        values = predict_distributed_temperature(
            training.model,
            positions=(0.0, experiment.geometry.length),
            times=(0.0, experiment.duration),
        )
        self.assertEqual(values[0], (295.0, 305.0))
        validation = validate_distributed_forward_pinn(
            training.model, experiment
        )
        self.assertLess(validation.maximum_absolute_temperature_error, 1.0)


if __name__ == "__main__":
    unittest.main()
