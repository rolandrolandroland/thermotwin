import unittest

try:
    import torch

    from thermotwin.contact_forward_pinn import contact_physics_residuals
    from thermotwin.piecewise_contact_forward_pinn import (
        PiecewiseContactForwardPINN,
        PiecewiseContactForwardPINNConfig,
        current_segment_boundaries,
        piecewise_collocation_times,
        scheduled_current_tensor,
        train_piecewise_contact_forward_pinn,
        unipolar_pulse_contact_experiment,
        validate_piecewise_contact_pinn,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
class PiecewiseContactForwardPINNTests(unittest.TestCase):
    def setUp(self):
        self.experiment = unipolar_pulse_contact_experiment()

    def test_pulse_boundaries_match_established_regime(self):
        boundaries = current_segment_boundaries(
            self.experiment.current,
            self.experiment.duration,
        )

        self.assertEqual(boundaries, (0.0, 5.0, 20.0, 60.0))

    def test_scheduled_current_tensor_is_right_continuous(self):
        times = torch.tensor(
            [[0.0], [4.999], [5.0], [19.999], [20.0], [60.0]]
        )

        values = scheduled_current_tensor(self.experiment.current, times)

        self.assertEqual(
            tuple(float(value) for value in values.reshape(-1)),
            (0.0, 0.0, 1.0, 1.0, 0.0, 0.0),
        )

    def test_collocation_coordinates_exclude_every_boundary(self):
        boundaries = (0.0, 5.0, 20.0, 60.0)

        times = piecewise_collocation_times(boundaries, 24).reshape(-1)

        self.assertEqual(len(times), 24)
        self.assertTrue(torch.all(times > 0.0))
        self.assertTrue(torch.all(times < 60.0))
        for boundary in boundaries[1:-1]:
            self.assertFalse(torch.any(times == boundary))
        self.assertGreaterEqual(int(torch.sum(times < 5.0)), 2)
        self.assertGreaterEqual(
            int(torch.sum((times > 5.0) & (times < 20.0))),
            2,
        )
        self.assertGreaterEqual(int(torch.sum(times > 20.0)), 2)

    def test_temperatures_are_continuous_while_rates_can_jump(self):
        model = PiecewiseContactForwardPINN(
            duration=60.0,
            transition_times=(5.0, 20.0),
            initial_temperatures=(300.0, 300.0, 300.0, 300.0),
            hidden_width=8,
            hidden_layers=1,
            temperature_scale=10.0,
        )
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        with torch.no_grad():
            model.networks[1][-1].bias[0] = 1.0

        switch_time = torch.tensor([[5.0]], requires_grad=True)
        left_temperature = model.predict_in_segment(0, switch_time)
        right_temperature = model.predict_in_segment(1, switch_time)
        left_rate = torch.autograd.grad(
            left_temperature[:, 0],
            switch_time,
            grad_outputs=torch.ones_like(left_temperature[:, 0]),
            retain_graph=True,
        )[0]
        right_rate = torch.autograd.grad(
            right_temperature[:, 0],
            switch_time,
            grad_outputs=torch.ones_like(right_temperature[:, 0]),
        )[0]

        self.assertTrue(torch.equal(left_temperature, right_temperature))
        self.assertEqual(float(left_rate), 0.0)
        self.assertGreater(float(right_rate), 0.0)
        self.assertTrue(
            torch.equal(
                model.boundary_temperature_jumps(),
                torch.zeros((2, 4)),
            )
        )

    def test_right_side_initial_slopes_satisfy_pulse_physics(self):
        class RightSideLinearRates(torch.nn.Module):
            def forward(self, time):
                relative_time = time - 5.0
                return torch.cat(
                    (
                        300.0 - 0.28 * relative_time,
                        300.0 + 0.16 * relative_time,
                        300.0 + 0.0 * relative_time,
                        300.0 + 0.0 * relative_time,
                    ),
                    dim=1,
                )

        time = torch.tensor([[5.0]])
        residuals = contact_physics_residuals(
            RightSideLinearRates(),
            time,
            self.experiment,
            current_values=scheduled_current_tensor(
                self.experiment.current,
                time,
            ),
        )

        for residual in residuals:
            self.assertAlmostEqual(float(residual.detach()), 0.0, places=6)

    def test_zero_current_first_segment_preserves_equilibrium(self):
        model = PiecewiseContactForwardPINN(
            duration=60.0,
            transition_times=(5.0, 20.0),
            initial_temperatures=(300.0, 300.0, 300.0, 300.0),
            hidden_width=8,
            hidden_layers=1,
        )
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        time = torch.tensor([[1.0], [2.5], [4.999]])

        residuals = contact_physics_residuals(
            model,
            time,
            self.experiment,
            current_values=scheduled_current_tensor(
                self.experiment.current,
                time,
            ),
        )

        for residual in residuals:
            self.assertTrue(
                torch.allclose(residual, torch.zeros_like(residual))
            )

    def test_short_cpu_training_reduces_loss_and_matches_rk4(self):
        training = train_piecewise_contact_forward_pinn(
            self.experiment,
            PiecewiseContactForwardPINNConfig(
                hidden_width=16,
                hidden_layers=2,
                collocation_points=48,
                epochs=300,
                learning_rate=5e-3,
                seed=17,
                device="cpu",
            ),
        )
        validation = validate_piecewise_contact_pinn(
            training,
            self.experiment,
        )

        self.assertEqual(training.device, "cpu")
        self.assertLess(training.loss_history[-1], training.loss_history[0])
        self.assertEqual(len(training.collocation_time), 48)
        self.assertEqual(
            float(
                training.model.boundary_temperature_jumps()
                .abs()
                .max()
                .detach()
            ),
            0.0,
        )
        for rmse in validation[:4]:
            self.assertLess(rmse, 0.1)

    def test_invalid_inputs_are_rejected(self):
        for keyword, value in (
            ("hidden_width", 0),
            ("hidden_layers", True),
            ("collocation_points", 1),
            ("epochs", 0),
            ("learning_rate", "invalid"),
            ("temperature_scale", 0.0),
            ("seed", 1.5),
            ("device", "cuda"),
        ):
            with self.subTest(keyword=keyword):
                with self.assertRaises(ValueError):
                    PiecewiseContactForwardPINNConfig(**{keyword: value})

        with self.assertRaises(ValueError):
            PiecewiseContactForwardPINN(
                duration=60.0,
                transition_times=(20.0, 5.0),
                initial_temperatures=(300.0,) * 4,
            )
        with self.assertRaises(ValueError):
            piecewise_collocation_times((0.0, 5.0, 20.0), 3)
        with self.assertRaisesRegex(ValueError, "current values"):
            contact_physics_residuals(
                PiecewiseContactForwardPINN(
                    duration=60.0,
                    transition_times=(5.0, 20.0),
                    initial_temperatures=(300.0,) * 4,
                    hidden_width=8,
                    hidden_layers=1,
                ),
                torch.tensor([[1.0]]),
                self.experiment,
                current_values=torch.tensor([[0.0], [1.0]]),
            )


if __name__ == "__main__":
    unittest.main()
