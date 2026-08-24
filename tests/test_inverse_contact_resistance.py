from dataclasses import replace
import unittest

from thermotwin import PiecewiseConstantCurrent

try:
    import torch

    from thermotwin.contact_forward_pinn import contact_physics_residuals
    from thermotwin.inverse_contact_resistance import (
        ColdContactTemperatureObservations,
        InverseContactResistanceConfig,
        InverseContactResistancePINN,
        ideal_inverse_contact_problem,
        train_inverse_contact_resistance,
        validate_inverse_contact_resistance,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
class InverseContactResistanceTests(unittest.TestCase):
    def setUp(self):
        self.problem = ideal_inverse_contact_problem()

    def test_ideal_problem_has_sparse_aligned_cold_pair_observations(self):
        observations = self.problem.observations

        self.assertEqual(len(observations.time), 13)
        self.assertEqual(observations.time, tuple(range(0, 61, 5)))
        self.assertEqual(observations.cold_face[0], 300.0)
        self.assertEqual(observations.cold_exchanger[0], 300.0)
        self.assertEqual(
            self.problem.experiment.thermal_parameters.cold_contact_resistance,
            0.25,
        )
        self.assertEqual(
            self.problem.dataset.regime.current.transition_times,
            (),
        )
        self.assertEqual(
            self.problem.dataset.regime.current.values,
            (1.0,),
        )

    def test_parameter_starts_positive_and_initial_state_is_exact(self):
        model = InverseContactResistancePINN(
            duration=60.0,
            initial_temperatures=(299.0, 301.0, 298.0, 302.0),
            initial_cold_contact_resistance=0.5,
            hidden_width=8,
            hidden_layers=1,
        )

        temperatures = model(torch.tensor([[0.0]]))

        self.assertAlmostEqual(
            float(model.cold_contact_resistance.detach()),
            0.5,
            places=6,
        )
        self.assertGreater(float(model.cold_contact_resistance.detach()), 0.0)
        self.assertEqual(
            tuple(float(value) for value in temperatures[0].detach()),
            (299.0, 301.0, 298.0, 302.0),
        )

    def test_larger_resistance_changes_contact_residuals_as_expected(self):
        class FixedTemperatures(torch.nn.Module):
            def forward(self, time):
                return torch.cat(
                    (
                        295.0 + 0.0 * time,
                        305.0 + 0.0 * time,
                        300.0 + 0.0 * time,
                        300.0 + 0.0 * time,
                    ),
                    dim=1,
                )

        low = contact_physics_residuals(
            FixedTemperatures(),
            torch.tensor([[10.0]]),
            self.problem.experiment,
            cold_contact_resistance=torch.tensor(0.25),
        )
        high = contact_physics_residuals(
            FixedTemperatures(),
            torch.tensor([[10.0]]),
            self.problem.experiment,
            cold_contact_resistance=torch.tensor(0.5),
        )

        self.assertGreater(
            float(high.cold_face.detach()),
            float(low.cold_face.detach()),
        )
        self.assertLess(
            float(high.cold_exchanger.detach()),
            float(low.cold_exchanger.detach()),
        )
        self.assertEqual(
            float(high.hot_face.detach()),
            float(low.hot_face.detach()),
        )
        self.assertEqual(
            float(high.hot_exchanger.detach()),
            float(low.hot_exchanger.detach()),
        )

    def test_resistance_is_unidentifiable_without_a_cold_contact_drop(self):
        equilibrium_problem = self.problem._replace(
            experiment=replace(
                self.problem.experiment,
                current=0.0,
            )
        )
        model = InverseContactResistancePINN(
            duration=60.0,
            initial_temperatures=(300.0, 300.0, 300.0, 300.0),
            initial_cold_contact_resistance=0.5,
            hidden_width=8,
            hidden_layers=1,
        )
        for parameter in model.temperature_model.parameters():
            torch.nn.init.zeros_(parameter)

        residuals = contact_physics_residuals(
            model,
            torch.tensor([[0.0], [30.0], [60.0]]),
            equilibrium_problem.experiment,
            cold_contact_resistance=model.cold_contact_resistance,
        )
        loss = sum(residual.square().mean() for residual in residuals)
        gradient = torch.autograd.grad(
            loss,
            model.raw_cold_contact_resistance,
        )[0]

        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(float(gradient.detach()), 0.0)

    def test_switching_current_is_rejected_by_first_inverse_contact_pinn(self):
        switched_problem = self.problem._replace(
            experiment=replace(
                self.problem.experiment,
                current=PiecewiseConstantCurrent.pulse(
                    start_time=5.0,
                    end_time=20.0,
                    pulse_current=1.0,
                ),
            )
        )

        with self.assertRaisesRegex(ValueError, "constant current only"):
            train_inverse_contact_resistance(
                switched_problem,
                InverseContactResistanceConfig(epochs=1),
            )

    def test_cpu_training_recovers_resistance_and_transfers_to_pulses(self):
        training = train_inverse_contact_resistance(
            self.problem,
            InverseContactResistanceConfig(
                hidden_width=16,
                hidden_layers=2,
                collocation_points=64,
                epochs=1_800,
                initial_cold_contact_resistance=0.5,
                seed=13,
                device="cpu",
            ),
        )
        validation = validate_inverse_contact_resistance(
            training,
            self.problem,
        )

        self.assertEqual(training.device, "cpu")
        self.assertLess(
            training.history.total_loss[-1],
            training.history.total_loss[0],
        )
        self.assertTrue(
            all(
                value > 0.0
                for value in training.history.cold_contact_resistance
            )
        )
        self.assertLess(validation.absolute_parameter_error, 0.02)
        self.assertAlmostEqual(
            validation.conventional_cold_contact_resistance,
            0.25,
            places=7,
        )
        self.assertLess(validation.cold_face_trajectory_rmse, 0.1)
        self.assertLess(validation.cold_exchanger_trajectory_rmse, 0.1)
        self.assertLess(
            validation.validation_regime_metrics.all_sensor_rmse,
            0.03,
        )
        self.assertLess(
            validation.test_regime_metrics.all_sensor_rmse,
            0.03,
        )

    def test_invalid_observations_and_configurations_are_rejected(self):
        invalid_observations = (
            ((0.0,), (), (300.0,)),
            ((0.0, 0.0), (300.0, 300.0), (300.0, 300.0)),
            ((float("nan"),), (300.0,), (300.0,)),
        )
        for values in invalid_observations:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    ColdContactTemperatureObservations(*values)

        for keyword, value in (
            ("hidden_width", 0),
            ("hidden_layers", True),
            ("collocation_points", 1),
            ("epochs", 0),
            ("network_learning_rate", float("nan")),
            ("parameter_learning_rate", 0.0),
            ("initial_cold_contact_resistance", -1.0),
            ("temperature_scale", "invalid"),
            ("residual_rate_scale", 0.0),
            ("observation_weight", float("inf")),
            ("seed", 2.5),
            ("device", "cuda"),
        ):
            with self.subTest(keyword=keyword):
                with self.assertRaises(ValueError):
                    InverseContactResistanceConfig(**{keyword: value})


if __name__ == "__main__":
    unittest.main()
