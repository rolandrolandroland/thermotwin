from dataclasses import replace
import unittest

from thermotwin import (
    PiecewiseConstantCurrent,
    constant_current_contact_reference_experiment,
    four_node_contact_rhs,
)

try:
    import torch

    from thermotwin.contact_forward_pinn import (
        ContactForwardPINN,
        ContactForwardPINNConfig,
        contact_physics_loss,
        contact_physics_residuals,
        predict_contact_trajectory,
        train_contact_forward_pinn,
        validate_contact_pinn_against_rk4,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
class ContactForwardPINNTests(unittest.TestCase):
    def setUp(self):
        self.experiment = constant_current_contact_reference_experiment()

    def test_initial_temperatures_are_enforced_exactly(self):
        model = ContactForwardPINN(
            duration=self.experiment.duration,
            initial_temperatures=(299.0, 301.0, 298.0, 302.0),
            hidden_width=8,
            hidden_layers=1,
        )

        temperatures = model(torch.tensor([[0.0], [60.0]]))

        self.assertEqual(tuple(temperatures.shape), (2, 4))
        self.assertEqual(
            tuple(float(value) for value in temperatures[0].detach()),
            (299.0, 301.0, 298.0, 302.0),
        )

    def test_initial_hand_calculated_rates_make_all_residuals_zero(self):
        class InitialLinearRates(torch.nn.Module):
            def forward(self, time):
                return torch.cat(
                    (
                        300.0 - 0.28 * time,
                        300.0 + 0.16 * time,
                        300.0 + 0.0 * time,
                        300.0 + 0.0 * time,
                    ),
                    dim=1,
                )

        residuals = contact_physics_residuals(
            InitialLinearRates(),
            torch.tensor([[0.0]]),
            self.experiment,
        )

        for residual in residuals:
            self.assertAlmostEqual(float(residual.detach()), 0.0, places=6)

    def test_residuals_match_conventional_rhs_at_an_arbitrary_state(self):
        state = (295.0, 305.0, 300.0, 300.0)
        rates = four_node_contact_rhs(
            self.experiment.thermoelectric_parameters,
            self.experiment.thermal_parameters,
            cold_face_temperature=state[0],
            hot_face_temperature=state[1],
            cold_exchanger_temperature=state[2],
            hot_exchanger_temperature=state[3],
            current=1.0,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
        )

        class MatchingLinearRates(torch.nn.Module):
            def forward(self, time):
                return torch.cat(
                    tuple(
                        value + rate * time
                        for value, rate in zip(state, rates)
                    ),
                    dim=1,
                )

        residuals = contact_physics_residuals(
            MatchingLinearRates(),
            torch.tensor([[0.0]]),
            self.experiment,
        )

        for residual in residuals:
            self.assertAlmostEqual(float(residual.detach()), 0.0, places=6)

    def test_zero_current_equal_temperature_state_is_equilibrium(self):
        experiment = replace(self.experiment, current=0.0)
        model = ContactForwardPINN(
            duration=experiment.duration,
            initial_temperatures=(300.0, 300.0, 300.0, 300.0),
            hidden_width=8,
            hidden_layers=1,
        )
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)

        residuals = contact_physics_residuals(
            model,
            torch.tensor([[0.0], [30.0], [60.0]]),
            experiment,
        )

        for residual in residuals:
            self.assertTrue(
                torch.allclose(residual, torch.zeros_like(residual))
            )

    def test_loss_sums_all_four_mean_squared_residuals(self):
        residuals = tuple(
            torch.full((3, 1), value)
            for value in (1.0, 2.0, 3.0, 4.0)
        )

        loss = contact_physics_loss(residuals)

        self.assertEqual(float(loss), 30.0)

    def test_first_contact_pinn_rejects_switching_current(self):
        pulse_experiment = replace(
            self.experiment,
            current=PiecewiseConstantCurrent.pulse(
                start_time=10.0,
                end_time=30.0,
                pulse_current=1.0,
            ),
        )

        with self.assertRaisesRegex(ValueError, "constant current only"):
            train_contact_forward_pinn(
                pulse_experiment,
                ContactForwardPINNConfig(epochs=1),
            )

    def test_training_performs_backpropagation_and_reduces_loss(self):
        training = train_contact_forward_pinn(
            self.experiment,
            ContactForwardPINNConfig(
                hidden_width=16,
                hidden_layers=2,
                collocation_points=64,
                epochs=400,
                learning_rate=5e-3,
                seed=11,
                device="cpu",
            ),
        )

        self.assertEqual(training.device, "cpu")
        self.assertLess(training.loss_history[-1], training.loss_history[0])
        trajectory = predict_contact_trajectory(
            training.model,
            (0.0, 30.0, 60.0),
        )
        self.assertEqual(trajectory.time, (0.0, 30.0, 60.0))
        self.assertEqual(
            (
                trajectory.cold_face[0],
                trajectory.hot_face[0],
                trajectory.cold_exchanger[0],
                trajectory.hot_exchanger[0],
            ),
            (300.0, 300.0, 300.0, 300.0),
        )

        validation = validate_contact_pinn_against_rk4(
            training.model,
            self.experiment,
        )
        for rmse in (
            validation.cold_face_rmse,
            validation.hot_face_rmse,
            validation.cold_exchanger_rmse,
            validation.hot_exchanger_rmse,
        ):
            self.assertLess(rmse, 0.5)

    def test_invalid_configuration_and_model_inputs_are_rejected(self):
        for keyword, value in (
            ("hidden_width", 0),
            ("hidden_layers", True),
            ("collocation_points", 1),
            ("epochs", 0),
            ("learning_rate", float("nan")),
            ("temperature_scale", 0.0),
            ("seed", 1.5),
            ("device", "cuda"),
        ):
            with self.subTest(keyword=keyword):
                with self.assertRaises(ValueError):
                    ContactForwardPINNConfig(**{keyword: value})

        with self.assertRaises(ValueError):
            ContactForwardPINN(
                duration=0.0,
                initial_temperatures=(300.0,) * 4,
            )
        with self.assertRaises(ValueError):
            ContactForwardPINN(
                duration=60.0,
                initial_temperatures=(300.0,) * 3,
            )


if __name__ == "__main__":
    unittest.main()
