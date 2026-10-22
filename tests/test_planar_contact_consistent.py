"""Constructed checks for the single corrective top-consistent PINN method."""
from dataclasses import replace
import inspect
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from thermotwin.physics.planar_contact import PlanarContactConfig
from thermotwin.pinn.planar_contact_inverse import PlanarObservations
from thermotwin.pinn.planar_contact_continuity import (
    linear_system,
    objective_continuity,
    project_output,
    residuals_continuity,
    training_points,
)
import thermotwin.pinn.planar_contact_consistent as consistent_module
from thermotwin.pinn.planar_contact_consistent import (
    ConsistentConfig,
    ConsistentPINN,
    consistent_training_points,
    fit_consistent,
    variable_projection_step,
)


def derivative(value, q):
    return torch.autograd.grad(value, q, torch.ones_like(value), create_graph=True)[0]


class PlanarConsistentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7293)
        self.config = ConsistentConfig(width=3, layers=1, adam_epochs=1,
                                       projection_every=1, interior_points=16,
                                       bottom_points=8, projection_chunk=16)
        self.physics = PlanarContactConfig()

    def model(self):
        model = ConsistentPINN(physics=self.physics, config=self.config, known_h=800.)
        with torch.no_grad():
            model.coefficients.copy_(torch.randn_like(model.coefficients) * .1)
        return model

    def observations(self):
        return PlanarObservations(
            np.array([[.004, .003, 11.], [.018, .001, 45.], [.034, .003, 89.]]),
            np.array([299.8, 299.6, 299.9]),
        )

    def points(self):
        return {
            "interior": np.array([[.1, .3, .2], [.4, .7, .4], [.8, .5, .8]]),
            "bottom_contact": np.array([[.3, 0., .2], [.5, 0., .7]]),
            "bottom_insulated": np.array([[.1, 0., .4], [.9, 0., .6]]),
        }

    def test_nonzero_paired_channels_preserve_interfaces_top_sides_and_initial_field(self):
        model = self.model()
        for index, edge in enumerate((self.physics.contact_start, self.physics.contact_end)):
            q = torch.tensor([[edge / self.physics.width, z, t]
                              for z in (.001, .17, .61, 1.)
                              for t in (.002, .27, .84)], dtype=torch.float64,
                             requires_grad=True)
            left = model.dimensionless_temperature(q, region=index)
            right = model.dimensionless_temperature(q, region=index + 1)
            torch.testing.assert_close(left, right, rtol=1e-11, atol=2e-12)
            torch.testing.assert_close(derivative(left, q)[:, 0],
                                       derivative(right, q)[:, 0],
                                       rtol=1e-10, atol=2e-11)
        exterior = {
            "top": np.array([[.11, 1., .2], [.5, 1., .6], [.91, 1., .9]]),
            "left": np.array([[0., .01, .2], [0., .8, .7]]),
            "right": np.array([[1., .01, .2], [1., .8, .7]]),
        }
        for name, value in residuals_continuity(model, exterior).items():
            self.assertLess(float(value.abs().max().detach()), 2e-10, name)
        physical = torch.tensor([[.003, .001, 0.], [.021, .002, 0.], [.037, .004, 0.]],
                                dtype=torch.float64)
        np.testing.assert_array_equal(model(physical).detach().numpy(), np.full(3, 300.))

    def test_paired_feature_frees_top_third_derivative_without_changing_top_value(self):
        model = self.model()
        base_count = len(model.descriptors) * (self.config.width + 1)
        self.assertEqual(len(model.coefficients), 2 * base_count)
        with torch.no_grad():
            model.coefficients.zero_()
            for index, (kind, _) in enumerate(model.descriptors):
                if kind == "A":
                    bias = index * (self.config.width + 1) + self.config.width
                    model.coefficients[bias] = -1.
                    model.coefficients[base_count + bias] = 1.
        q = torch.tensor([[x, 1., t] for x in (.1, .37, .8) for t in (.04, .3, .9)],
                         dtype=torch.float64, requires_grad=True)
        correction = model.dimensionless_temperature(q) - model.base_temperature(q)
        first = derivative(correction, q)[:, 1:2]
        second = derivative(first, q)[:, 1:2]
        third = derivative(second, q)[:, 1:2]
        ramp = -torch.expm1(-q[:, 2:3] * self.config.duration / self.config.ramp_seconds)
        torch.testing.assert_close(correction, torch.zeros_like(correction), rtol=0., atol=2e-14)
        torch.testing.assert_close(first, torch.zeros_like(first), rtol=0., atol=2e-13)
        torch.testing.assert_close(third, -12 * ramp, rtol=2e-11, atol=2e-11)
        self.assertGreater(float(third.abs().min().detach()), 1.)

    def test_linear_basis_and_strong_operators_reconstruct_paired_field_and_gradient(self):
        model = self.model()
        model.coefficients.requires_grad_(True)
        points = self.points()
        observed = self.observations()
        matrix, target, _ = linear_system(model, points, observed)
        matrix, target = torch.tensor(matrix), torch.tensor(target)
        direct = objective_continuity(model, points, observed)[0]
        vector = matrix @ model.coefficients - target
        torch.testing.assert_close(direct, vector.square().sum(), rtol=2e-11, atol=1e-8)
        actual_gradient = torch.autograd.grad(direct, model.coefficients)[0]
        torch.testing.assert_close(actual_gradient, 2 * matrix.T @ vector,
                                   rtol=2e-10, atol=2e-7)
        q = torch.tensor(points["interior"], dtype=torch.float64)
        reconstructed = model.base_temperature(q).reshape(-1) + model.basis(q) @ model.coefficients
        torch.testing.assert_close(model.dimensionless_temperature(q).reshape(-1),
                                   reconstructed, rtol=2e-12, atol=2e-12)

    def test_public_forward_training_does_not_accept_hidden_field_labels(self):
        for callable_ in (ConsistentPINN, fit_consistent):
            self.assertFalse(set(inspect.signature(callable_).parameters) & {
                "truth", "true_h", "reference_field", "hidden_field",
                "true_contact", "verification_result", "best_epoch",
            })
        with self.assertRaises(TypeError):
            fit_consistent(self.observations(), known_h=800., neural_seed=2,
                           sampling_seed=3, hidden_field=np.zeros(1))

    def tiny_projected_fixture(self):
        config = replace(self.config, width=2, max_backtracks=2)
        physics = replace(self.physics, contact_start=0., contact_end=self.physics.width)
        model = ConsistentPINN(physics=physics, config=config, known_h=800.)
        points = training_points(physics, config, 183)
        observed = self.observations()
        project_output(model, points, observed)
        return model, points, observed

    def test_variable_projection_acceptance_uses_reprojected_same_point_objective(self):
        model, points, observed = self.tiny_projected_fixture()
        parameters = list(model.networks.parameters())
        optimizer = torch.optim.Adam(parameters, lr=model.config.learning_rate)
        before = float(objective_continuity(model, points, observed)[0].detach())
        actual_project = consistent_module.project_output
        projections = []
        def capture_projection(*args, **kwargs):
            record = actual_project(*args, **kwargs)
            projections.append(record)
            return record
        with patch.object(consistent_module, "project_output", capture_projection):
            report = variable_projection_step(model, points, observed, optimizer,
                                               parameters=parameters)
        after = float(objective_continuity(model, points, observed)[0].detach())
        self.assertGreaterEqual(len(projections), 1)
        self.assertLessEqual(after, before)
        self.assertEqual(report["objective_before"], before)
        self.assertAlmostEqual(report["objective_after"], after, places=10)
        self.assertTrue(report["optimizer_reset_required"])
        if report["accepted"]:
            self.assertLessEqual(report["objective_after"], report["objective_before"])
        json.dumps(report, allow_nan=False)

    def test_rejected_variable_projection_restores_readout_hidden_and_field(self):
        for failure in ("nonfinite_hidden", "nonfinite_projection", "finite_worse_projection"):
            with self.subTest(failure=failure):
                torch.manual_seed(7293)
                model, points, observed = self.tiny_projected_fixture()
                parameters = list(model.networks.parameters())
                optimizer = torch.optim.Adam(parameters, lr=model.config.learning_rate)
                initial = {name: value.detach().clone()
                           for name, value in model.named_parameters()}
                held = torch.tensor([[.007, .002, 17.], [.025, .003, 68.]],
                                    dtype=torch.float64)
                initial_field = model(held).detach().clone()
                before = float(objective_continuity(model, points, observed)[0].detach())
                actual_step = optimizer.step
                def bad_step():
                    if failure == "nonfinite_hidden":
                        with torch.no_grad():
                            for parameter in parameters:
                                parameter.fill_(float("nan"))
                    else:
                        actual_step()
                def bad_projection(passed_model, passed_points, passed_observed):
                    with torch.no_grad():
                        passed_model.coefficients.fill_(float("nan") if failure == "nonfinite_projection"
                                                        else 1e10)
                    if failure == "nonfinite_projection":
                        raise FloatingPointError("constructed nonfinite projection")
                    return {"accepted": True, "rank": 1, "resolved_condition": 1.}
                with patch.object(optimizer, "step", bad_step), \
                     patch.object(consistent_module, "project_output", bad_projection):
                    report = variable_projection_step(model, points, observed, optimizer,
                                                       parameters=parameters)
                self.assertFalse(report["accepted"])
                self.assertTrue(report["optimizer_reset_required"])
                self.assertEqual(report["objective_after"], before)
                if failure == "finite_worse_projection":
                    self.assertTrue(all(
                        trial["reason"] == "complete objective did not decrease after readout solve"
                        and "error_type" not in trial for trial in report["trials"]))
                for name, value in model.named_parameters():
                    torch.testing.assert_close(value, initial[name], rtol=0., atol=0.)
                torch.testing.assert_close(model(held), initial_field, rtol=0., atol=0.)
                json.dumps(report, allow_nan=False)

    def test_tiny_fit_returns_strict_json_without_diagnostics_or_hidden_truth(self):
        config = replace(self.config, width=2, adam_epochs=1)
        physics = replace(self.physics, contact_start=0., contact_end=self.physics.width)
        with patch.object(consistent_module, "diagnose_consistent",
                          side_effect=AssertionError("diagnostics entered training")):
            model, report = fit_consistent(
                self.observations(), physics=physics, config=config,
                known_h=800., neural_seed=12, sampling_seed=18,
            )
        self.assertFalse(report["hidden_truth_labels_used"])
        self.assertFalse(report["independent_diagnostics_used_for_training"])
        self.assertEqual(len(report["hidden_update_history"]), 1)
        self.assertEqual(report["optimizer_resets"], 2)
        self.assertEqual(report["hidden_accepted_objective_increases"], 0)
        points = {name: np.asarray(value) for name, value in report["training_points"].items()}
        actual = float(objective_continuity(model, points, self.observations())[0].detach())
        self.assertAlmostEqual(actual, report["final_objective"], places=9)
        json.dumps(report, allow_nan=False)

    def test_fit_recreates_optimizer_and_restores_complete_field_after_failed_trials(self):
        config = replace(self.config, width=2, adam_epochs=1, max_backtracks=2)
        physics = replace(self.physics, contact_start=0., contact_end=self.physics.width)
        actual_project = consistent_module.project_output
        actual_adam = torch.optim.Adam
        optimizers, initial = [], {}
        calls = 0
        def track_adam(*args, **kwargs):
            instance = actual_adam(*args, **kwargs)
            optimizers.append(instance)
            return instance
        def corrupt_trial_head(model, points, observed):
            nonlocal calls
            calls += 1
            if 1 < calls <= config.max_backtracks + 2:
                with torch.no_grad():
                    model.coefficients.fill_(1e10)
                return {"accepted": True, "rank": 1, "resolved_condition": 1.}
            report = actual_project(model, points, observed)
            if not initial:
                initial.update({name: value.detach().clone()
                                for name, value in model.named_parameters()})
            return report
        with patch.object(consistent_module.torch.optim, "Adam", track_adam), \
             patch.object(consistent_module, "project_output", corrupt_trial_head):
            model, report = fit_consistent(
                self.observations(), physics=physics, config=config, known_h=800.,
                neural_seed=12, sampling_seed=18,
            )
        self.assertEqual(report["hidden_rejected_updates"], 1)
        self.assertEqual(report["optimizer_resets"], 2)
        self.assertEqual(len(optimizers), 2)
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, initial[name], rtol=0., atol=0.)
        json.dumps(report, allow_nan=False)

    def test_denser_vertical_training_grid_keeps_all_v3_bottom_points(self):
        config = ConsistentConfig()
        old = training_points(self.physics, replace(config, interior_points=4096), 927)
        new = consistent_training_points(self.physics, config, 927)
        self.assertEqual(new["interior"].shape, (8192, 3))
        self.assertEqual([len(np.unique(new["interior"][:, axis])) for axis in range(3)],
                         [32, 16, 16])
        for family in ("bottom_contact", "bottom_insulated"):
            np.testing.assert_array_equal(new[family], old[family])


if __name__ == "__main__":
    unittest.main()
