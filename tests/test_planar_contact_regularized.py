"""Constructed full-rank readout/gradient and transactional controller checks."""
from copy import deepcopy
from dataclasses import asdict, replace
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from thermotwin.physics.planar_contact import PlanarContactConfig
from thermotwin.pinn import planar_contact_regularized as ridge
from thermotwin.pinn import planar_contact_continuity as frozen
from thermotwin.pinn import planar_contact_optimized as old_optimizer
from thermotwin.pinn.planar_contact_inverse import PlanarObservations
from thermotwin.pinn.planar_contact_step_control import matched_step_control_model, StepControlConfig


class IllConditionedModel(nn.Module):
    def __init__(self, **options):
        super().__init__()
        self.config = ridge.RegularizedConfig(width=2, layers=1, **options)
        self.physics, self.known_h = PlanarContactConfig(), 800.
        self.networks = nn.ModuleList([nn.Linear(2, 1, bias=False, dtype=torch.float64)])
        with torch.no_grad():
            self.networks[0].weight.copy_(torch.tensor([[1., 1.3]], dtype=torch.float64))
        self.coefficients = nn.Parameter(torch.zeros(2, dtype=torch.float64), requires_grad=False)
        self.register_buffer("readout_scale", torch.ones(2, dtype=torch.float64))
        self.register_buffer("readout_lambda", torch.zeros((), dtype=torch.float64))
        self.register_buffer("readout_calibrated", torch.tensor(False))


def toy_matrix(model):
    w = model.networks[0].weight.flatten()
    one, zero = torch.ones((), dtype=torch.float64), torch.zeros((), dtype=torch.float64)
    return torch.stack((torch.stack((one, one)), torch.stack((1.e-7*w[0], zero)), torch.stack((zero, 1.e-7*w[1]))))


def toy_objective(model, points, observations):
    value = torch.sum((toy_matrix(model)@model.coefficients-torch.tensor([0., 1., .5], dtype=torch.float64))**2)
    return value, value, value, {"constructed": value}


def toy_system(model, points, observations):
    return toy_matrix(model).detach().numpy(), np.array([0., 1., .5]), {"families": [], "coefficient_count": 2}


class RegularizedContactTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.observations = PlanarObservations(np.array([[.004, .002, 0.], [.03, .002, 20.]]), np.array([300., 299.]), .02)

    def toy_context(self):
        return patch.multiple(ridge, objective_continuity=toy_objective, linear_system=toy_system)

    def toy(self, **options):
        model = IllConditionedModel(**options)
        ridge.calibrate_readout(model, {}, self.observations)
        ridge.project_regularized(model, {}, self.observations)
        return model

    def assert_nested_equal(self, left, right):
        if torch.is_tensor(left):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def check_directional_derivative(self, model, points, observations):
        ridge.project_regularized(model, points, observations)
        parameters = list(model.networks.parameters())
        gradients = torch.autograd.grad(ridge.objective_regularized(model, points, observations)[0], parameters)
        maximum = max(float(abs(value).max()) for value in gradients)
        self.assertGreater(maximum, 0.)
        direction = [-value.detach()/maximum for value in gradients]
        derivative = sum(float(torch.sum(value*vector)) for value, vector in zip(gradients, direction))
        original = deepcopy(model.state_dict())
        values = []
        for radius in (1.e-4, 1.e-5, 1.e-6):
            losses = []
            for sign in (-1, 1):
                model.load_state_dict(original)
                with torch.no_grad():
                    for parameter, vector in zip(parameters, direction):
                        parameter.add_(sign*radius*vector)
                projection = ridge.project_regularized(model, points, observations)
                self.assertEqual(projection["rank"], len(model.coefficients))
                self.assertLessEqual(projection["stationarity_relative"], 1.e-8)
                self.assertLessEqual(projection["algebraic_objective_abs_difference"], 1.e-8+1.e-8*abs(projection["objective_after"]))
                losses.append(projection["objective_after"])
            values.append((losses[1]-losses[0])/(2*radius))
        model.load_state_dict(original)
        self.assertLessEqual(abs(values[-1]-derivative), 1.e-4+5.e-3*max(abs(values[-1]), abs(derivative)))
        self.assertLessEqual(abs(values[-1]-values[-2]), 2.e-4+1.e-2*max(abs(values[-1]), abs(values[-2])))
        return derivative, values

    def test_fixed_calibration_counts_reuse_identity_and_field_architecture(self):
        config = ridge.RegularizedConfig(width=2, layers=1, interior_points=32, bottom_points=8)
        model, identity = ridge.matched_regularized_model(PlanarContactConfig(), config, 800., 13)
        base = {key: value for key, value in asdict(config).items() if key in StepControlConfig.__dataclass_fields__}
        old, expected_identity = matched_step_control_model(model.physics, StepControlConfig(**base), 800., 13)
        self.assertEqual(identity, expected_identity)
        for name, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]))
        points = ridge.graded_training_points(model.physics, config, 71)
        system, spectral = ridge.linear_system, np.linalg.svd
        with patch.object(ridge, "linear_system", wraps=system) as builds, patch.object(np.linalg, "svd", wraps=spectral) as svds:
            calibration = ridge.calibrate_readout(model, points, self.observations)
        self.assertEqual(builds.call_count, 1)
        self.assertEqual(svds.call_count, 1)
        self.assertEqual(calibration["calibration_matrix_builds"], 1)
        self.assertEqual(calibration["fixed_lambda"], 1.e-7*calibration["initial_sigma_max"])
        copy, _ = ridge.matched_regularized_model(model.physics, config, 800., 13)
        with patch.object(ridge, "linear_system", side_effect=AssertionError("prepared arrays must not rebuild A0")):
            ridge.apply_readout_calibration(copy, calibration, points, self.observations)
        self.assertTrue(torch.equal(copy.readout_scale, model.readout_scale))
        self.assertTrue(torch.equal(copy.readout_lambda, model.readout_lambda))
        with self.assertRaisesRegex(RuntimeError, "immutable"):
            ridge.calibrate_readout(model, points, self.observations)
        different, _ = ridge.matched_regularized_model(model.physics, config, 800., 14)
        with self.assertRaisesRegex(ValueError, "identity"):
            ridge.apply_readout_calibration(different, calibration, points, self.observations)
        with self.assertRaises(ValueError):
            replace(config, regularization_relative_scale=1.e-6)

    def test_ill_conditioned_continuous_stationary_head_and_profiled_gradient(self):
        with self.toy_context():
            model = self.toy()
            before = model.readout_scale.clone(), model.readout_lambda.clone()
            derivative, differences = self.check_directional_derivative(model, {}, self.observations)
            self.assertGreater(abs(derivative), .01)
            self.assertTrue(all(np.isfinite(differences)))
            self.assertTrue(torch.equal(model.readout_scale, before[0]))
            self.assertTrue(torch.equal(model.readout_lambda, before[1]))
            model.coefficients.requires_grad_(True)
            total = ridge.objective_regularized(model, {}, self.observations)[0]
            coefficient_gradient = torch.autograd.grad(total, model.coefficients)[0]/model.readout_scale
            self.assertLess(float(torch.linalg.vector_norm(coefficient_gradient)), 1.e-7)

    def test_actual_thermal_operator_profiled_gradient_uses_fixed_penalty(self):
        config = ridge.RegularizedConfig(width=2, layers=1, interior_points=128, bottom_points=16, projection_chunk=32)
        model, _ = ridge.matched_regularized_model(PlanarContactConfig(), config, 800., 13)
        points = ridge.graded_training_points(model.physics, config, 71)
        ridge.calibrate_readout(model, points, self.observations)
        self.check_directional_derivative(model, points, self.observations)

    def test_stationary_ridge_head_can_raise_unpenalized_loss_without_fallback(self):
        with self.toy_context():
            model = self.toy()
            matrix, rhs, _ = toy_system(model, {}, None)
            with torch.no_grad():
                model.coefficients.copy_(torch.tensor(np.linalg.lstsq(matrix, rhs, rcond=0)[0]))
            before = ridge.objective_regularized(model, {}, self.observations)
            before_total, before_old = float(before[0].detach()), float(before[1].detach())
            projection = ridge.project_regularized(model, {}, self.observations)
            self.assertLess(projection["objective_after"], before_total)
            self.assertGreater(projection["unpenalized_objective"], before_old)
            self.assertFalse(projection["hard_singular_value_truncation"])
            self.assertIn("unconditional", projection["head_policy"])

    def test_failed_and_interrupted_trials_restore_hidden_head_moments_groups_and_gradients(self):
        with self.toy_context():
            model = self.toy()
            optimizer = torch.optim.Adam(model.networks.parameters(), lr=model.config.learning_rate)
            first = ridge.transactional_adam_step(model, {}, self.observations, optimizer)
            self.assertTrue(first["accepted"])
            snapshot = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict()), [value.grad.clone() for value in model.networks.parameters()]
            def corrupt(*args):
                with torch.no_grad():
                    model.coefficients.fill_(float("nan"))
                raise FloatingPointError("constructed failed readout")
            with patch.object(ridge, "project_regularized", side_effect=corrupt):
                event = ridge.transactional_adam_step(model, {}, self.observations, optimizer)
            self.assertFalse(event["accepted"])
            self.assertEqual(len(event["trials"]), 5)
            self.assertTrue(event["optimizer_state_rolled_back"])
            self.assert_nested_equal(model.state_dict(), snapshot[0])
            self.assert_nested_equal(optimizer.state_dict(), snapshot[1])
            self.assert_nested_equal([value.grad for value in model.networks.parameters()], snapshot[2])
            with patch.object(ridge, "project_regularized", side_effect=RuntimeError("constructed interruption")):
                with self.assertRaisesRegex(RuntimeError, "interruption"):
                    ridge.transactional_adam_step(model, {}, self.observations, optimizer)
            self.assert_nested_equal(model.state_dict(), snapshot[0])
            self.assert_nested_equal(optimizer.state_dict(), snapshot[1])
            json.dumps(event, allow_nan=False)

    def test_controller_restart_and_eight_failure_bound_use_new_solve_path(self):
        with self.toy_context():
            model = self.toy()
            optimizer = torch.optim.Adam(model.networks.parameters(), lr=model.config.learning_rate)
            state = ridge.StepControlState(model.config.learning_rate)
            with patch.object(ridge, "project_regularized", side_effect=FloatingPointError("constructed reject")), \
                 patch.object(old_optimizer, "transactional_adam_step", side_effect=AssertionError("old hard-cutoff path forbidden")):
                events = [ridge.controller_step(model, {}, self.observations, optimizer, state) for _ in range(8)]
            self.assertEqual(state.stop_reason, "consecutive exhausted searches")
            self.assertEqual(state.moment_restarts, 8)
            self.assertEqual(state.trial_solves, 40)
            self.assertTrue(events[-1]["controller_stop"])
            self.assertTrue(all(event["optimizer_state_rolled_back"] for event in events))
            self.assertEqual([event["next_learning_rate"] for event in events], [model.config.learning_rate*2.**(-index) for index in range(1, 9)])

    def test_prepared_calibration_tiny_fit_budget_accounting_and_checkpoint(self):
        config = ridge.RegularizedConfig(width=2, layers=1, adam_epochs=5, interior_points=32,
            bottom_points=8, projection_chunk=16, max_backtracks=1, maximum_trial_solves=2)
        fresh, _ = ridge.matched_regularized_model(PlanarContactConfig(), config, 800., 13)
        points = ridge.graded_training_points(fresh.physics, config, 71)
        calibration = ridge.calibrate_readout(fresh, points, self.observations)
        with patch.object(np.linalg, "svd", side_effect=AssertionError("worker cannot recalibrate")), \
             patch.object(frozen, "project_output", side_effect=AssertionError("old TSVD forbidden")), \
             patch.object(ridge, "diagnose_regularized", side_effect=AssertionError("diagnostic leakage")):
            model, report = ridge.fit_regularized(self.observations, config=config, known_h=800., neural_seed=13,
                sampling_seed=71, readout_calibration=calibration)
        self.assertEqual(report["completed_updates"], 1)
        self.assertEqual(report["stop_reason"], "projection_budget")
        self.assertEqual(report["initial_calibration_matrix_builds"], 0)
        self.assertEqual(report["initial_calibration_spectral_calculations"], 0)
        self.assertLessEqual(report["trial_solve_count"], 2)
        self.assertEqual(report["total_projection_solve_calls"], report["trial_solve_count"]+2)
        self.assertEqual(report["hidden_accepted_objective_increases"], 0)
        self.assertEqual(report["readout_calibration"], calibration)
        self.assertAlmostEqual(report["final_objective"], report["final_unpenalized_objective"]+report["final_regularization_penalty"])
        self.assertLessEqual(report["final_head_stationarity"], 1.e-8)
        checkpoint = torch.load(io.BytesIO(ridge.checkpoint_bytes(model)), weights_only=False)
        self.assertEqual(checkpoint["model_class"], "RegularizedPINN")
        self.assertEqual(checkpoint["readout_calibration"], calibration)
        self.assert_nested_equal(checkpoint["state_dict"], model.state_dict())
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
