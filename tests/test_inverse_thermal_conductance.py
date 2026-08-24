from dataclasses import replace
import unittest

from thermotwin import (
    constant_current_reference_experiment,
    run_two_node_experiment,
)

try:
    import torch

    from thermotwin.forward_pinn import physics_residuals
    from thermotwin.inverse_thermal_conductance import (
        InverseThermalConductanceConfig,
        InverseThermalConductancePINN,
        synthetic_temperature_observations,
        train_inverse_thermal_conductance,
        validate_inverse_thermal_conductance,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
class InverseThermalConductanceTests(unittest.TestCase):
    def setUp(self):
        self.experiment = constant_current_reference_experiment()

    def test_synthetic_observations_sample_the_rk4_reference(self):
        observations = synthetic_temperature_observations(
            self.experiment,
            observation_interval=5.0,
        )
        reference = run_two_node_experiment(self.experiment).trajectory

        self.assertEqual(len(observations.time), 13)
        self.assertEqual(observations.time[0], 0.0)
        self.assertEqual(observations.time[-1], 60.0)
        self.assertEqual(observations.cold[0], 300.0)
        self.assertEqual(observations.hot[0], 300.0)
        self.assertAlmostEqual(observations.cold[-1], reference.cold[-1])
        self.assertAlmostEqual(observations.hot[-1], reference.hot[-1])

    def test_inferred_conductance_starts_positive_and_preserves_initial_state(self):
        model = InverseThermalConductancePINN(
            duration=60.0,
            initial_cold_temperature=300.0,
            initial_hot_temperature=300.0,
            initial_thermal_conductance=0.2,
            hidden_width=8,
            hidden_layers=1,
        )

        temperatures = model(torch.tensor([[0.0]]))

        self.assertAlmostEqual(
            float(model.thermal_conductance.detach()),
            0.2,
            places=6,
        )
        self.assertGreater(float(model.thermal_conductance.detach()), 0.0)
        self.assertEqual(float(temperatures[0, 0].detach()), 300.0)
        self.assertEqual(float(temperatures[0, 1].detach()), 300.0)

    def test_larger_k_changes_fixed_path_residuals_in_expected_directions(self):
        class FixedTemperatures(torch.nn.Module):
            def forward(self, time):
                return torch.cat(
                    (
                        295.0 + 0.0 * time,
                        305.0 + 0.0 * time,
                    ),
                    dim=1,
                )

        time = torch.tensor([[10.0]])
        low_conductance = physics_residuals(
            FixedTemperatures(),
            time,
            self.experiment,
            thermal_conductance=torch.tensor(0.5),
        )
        high_conductance = physics_residuals(
            FixedTemperatures(),
            time,
            self.experiment,
            thermal_conductance=torch.tensor(1.0),
        )

        self.assertLess(
            float(high_conductance.cold.detach()),
            float(low_conductance.cold.detach()),
        )
        self.assertGreater(
            float(high_conductance.hot.detach()),
            float(low_conductance.hot.detach()),
        )

    def test_k_is_unidentifiable_when_temperature_difference_is_always_zero(self):
        equilibrium_experiment = replace(self.experiment, current=0.0)
        model = InverseThermalConductancePINN(
            duration=60.0,
            initial_cold_temperature=300.0,
            initial_hot_temperature=300.0,
            initial_thermal_conductance=0.2,
            hidden_width=8,
            hidden_layers=1,
        )
        for parameter in model.temperature_model.parameters():
            torch.nn.init.zeros_(parameter)

        residuals = physics_residuals(
            model,
            torch.tensor([[0.0], [30.0], [60.0]]),
            equilibrium_experiment,
            thermal_conductance=model.thermal_conductance,
        )
        loss = residuals.cold.square().mean() + residuals.hot.square().mean()
        conductance_gradient = torch.autograd.grad(
            loss,
            model.raw_thermal_conductance,
        )[0]

        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(float(conductance_gradient.detach()), 0.0)

    def test_cpu_training_recovers_k_from_sparse_noise_free_data(self):
        observations = synthetic_temperature_observations(
            self.experiment,
            observation_interval=5.0,
        )
        training = train_inverse_thermal_conductance(
            self.experiment,
            observations,
            InverseThermalConductanceConfig(
                hidden_width=16,
                hidden_layers=2,
                collocation_points=64,
                epochs=1_000,
                initial_thermal_conductance=0.2,
                device="cpu",
            ),
        )
        validation = validate_inverse_thermal_conductance(
            training,
            self.experiment,
            observations,
        )

        self.assertEqual(training.device, "cpu")
        self.assertTrue(
            all(value > 0.0 for value in training.history.thermal_conductance)
        )
        self.assertLess(validation.absolute_parameter_error, 0.02)
        self.assertLess(validation.cold_trajectory_rmse, 0.05)
        self.assertLess(validation.hot_trajectory_rmse, 0.05)


if __name__ == "__main__":
    unittest.main()
