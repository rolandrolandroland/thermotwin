from dataclasses import replace
import unittest

try:
    import torch

    from thermotwin.contact_forward_pinn import contact_physics_residuals
    from thermotwin.inverse_contact_resistance import (
        ColdContactTemperatureObservations,
    )
    from thermotwin.piecewise_contact_forward_pinn import (
        scheduled_current_tensor,
    )
    from thermotwin.piecewise_inverse_contact_resistance import (
        PiecewiseInverseContactResistanceConfig,
        PiecewiseInverseContactResistancePINN,
        ideal_piecewise_inverse_contact_problem,
        train_piecewise_inverse_contact_resistance,
        validate_piecewise_inverse_contact_resistance,
    )
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
class PiecewiseInverseContactResistanceTests(unittest.TestCase):
    def setUp(self):
        self.problem = ideal_piecewise_inverse_contact_problem()

    def test_ideal_problem_reuses_training_pulse_and_one_second_cold_data(self):
        observations = self.problem.observations
        regime = self.problem.dataset.regime

        self.assertEqual(regime.name, "unipolar_training_pulse")
        self.assertEqual(regime.split, "train")
        self.assertEqual(regime.current.transition_times, (5.0, 20.0))
        self.assertEqual(regime.current.values, (0.0, 1.0, 0.0))
        self.assertEqual(len(observations.time), 61)
        self.assertEqual(observations.time, tuple(range(61)))
        self.assertEqual(observations.cold_face[0], 300.0)
        self.assertEqual(observations.cold_exchanger[0], 300.0)
        self.assertEqual(
            self.problem.experiment.thermal_parameters.cold_contact_resistance,
            0.25,
        )
        face_observations = (
            self.problem.dataset.observations.observations_for(
                "cold_face_sensor"
            )
        )
        current_by_time = {
            item.time: item.current for item in face_observations
        }
        self.assertEqual(current_by_time[5.0], 1.0)
        self.assertEqual(current_by_time[20.0], 0.0)

    def test_parameter_is_positive_and_all_interfaces_are_exact(self):
        model = PiecewiseInverseContactResistancePINN(
            duration=60.0,
            transition_times=(5.0, 20.0),
            initial_temperatures=(299.0, 301.0, 298.0, 302.0),
            initial_cold_contact_resistance=0.5,
            hidden_width=8,
            hidden_layers=1,
        )

        initial = model(torch.tensor([[0.0]]))

        self.assertAlmostEqual(
            float(model.cold_contact_resistance.detach()),
            0.5,
            places=6,
        )
        self.assertGreater(float(model.cold_contact_resistance.detach()), 0.0)
        self.assertEqual(
            tuple(float(value) for value in initial[0].detach()),
            (299.0, 301.0, 298.0, 302.0),
        )
        self.assertTrue(
            torch.equal(
                model.boundary_temperature_jumps(),
                torch.zeros((2, 4)),
            )
        )

    def test_one_resistance_parameter_is_shared_by_all_segments(self):
        model = PiecewiseInverseContactResistancePINN(
            duration=60.0,
            transition_times=(5.0, 20.0),
            initial_temperatures=(300.0,) * 4,
            initial_cold_contact_resistance=0.5,
            hidden_width=8,
            hidden_layers=1,
        )

        resistance_parameters = tuple(
            name for name, _ in model.named_parameters() if "resistance" in name
        )

        self.assertEqual(
            resistance_parameters,
            ("raw_cold_contact_resistance",),
        )
        self.assertEqual(len(model.temperature_model.networks), 3)

    def test_resistance_is_unidentifiable_at_zero_current_equilibrium(self):
        equilibrium_experiment = replace(
            self.problem.experiment,
            current=0.0,
        )
        model = PiecewiseInverseContactResistancePINN(
            duration=60.0,
            transition_times=(),
            initial_temperatures=(300.0,) * 4,
            initial_cold_contact_resistance=0.5,
            hidden_width=8,
            hidden_layers=1,
        )
        for parameter in model.temperature_model.parameters():
            torch.nn.init.zeros_(parameter)
        time = torch.tensor([[1.0], [30.0], [59.0]])
        residuals = contact_physics_residuals(
            model,
            time,
            equilibrium_experiment,
            cold_contact_resistance=model.cold_contact_resistance,
            current_values=scheduled_current_tensor(
                equilibrium_experiment.current,
                time,
            ),
        )
        loss = sum(residual.square().mean() for residual in residuals)
        gradient = torch.autograd.grad(
            loss,
            model.raw_cold_contact_resistance,
        )[0]

        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(float(gradient.detach()), 0.0)

    def test_short_cpu_training_recovers_the_shared_resistance(self):
        training = train_piecewise_inverse_contact_resistance(
            self.problem,
            PiecewiseInverseContactResistanceConfig(
                hidden_width=16,
                hidden_layers=2,
                collocation_points=48,
                epochs=3_000,
                network_learning_rate=2e-3,
                parameter_learning_rate=1e-2,
                initial_cold_contact_resistance=0.5,
                observation_weight=20.0,
                seed=19,
                device="cpu",
            ),
        )
        validation = validate_piecewise_inverse_contact_resistance(
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
        self.assertEqual(validation.max_boundary_temperature_jump, 0.0)
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

    def test_out_of_range_observations_and_too_few_points_are_rejected(self):
        invalid_observations = ColdContactTemperatureObservations(
            time=(-1.0, 0.0),
            cold_face=(300.0, 300.0),
            cold_exchanger=(300.0, 300.0),
        )
        invalid_problem = self.problem._replace(
            observations=invalid_observations
        )

        with self.assertRaisesRegex(ValueError, "observation times"):
            train_piecewise_inverse_contact_resistance(
                invalid_problem,
                PiecewiseInverseContactResistanceConfig(epochs=1),
            )
        with self.assertRaisesRegex(ValueError, "two points per segment"):
            train_piecewise_inverse_contact_resistance(
                self.problem,
                PiecewiseInverseContactResistanceConfig(
                    collocation_points=2,
                    epochs=1,
                ),
            )

    def test_invalid_configurations_are_rejected(self):
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
                    PiecewiseInverseContactResistanceConfig(
                        **{keyword: value}
                    )


if __name__ == "__main__":
    unittest.main()
