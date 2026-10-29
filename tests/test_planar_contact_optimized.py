"""Constructed tests of transactional optimizer history and frozen field math."""
from copy import deepcopy
from dataclasses import asdict, replace
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from thermotwin.physics.planar_contact import PlanarContactConfig
from thermotwin.pinn.planar_contact_inverse import PlanarObservations
from thermotwin.pinn import planar_contact_optimized as optimized
from thermotwin.pinn.planar_contact_graded import GradedConfig, graded_point_metadata, matched_graded_model


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.networks = nn.ModuleList([nn.Linear(1, 1, bias=False, dtype=torch.float64)])
        with torch.no_grad():
            self.networks[0].weight.fill_(2.)
        self.coefficients = nn.Parameter(torch.tensor([6.], dtype=torch.float64), requires_grad=False)
        self.config = SimpleNamespace(gradient_clip=100., rank_rcond=1e-7, max_backtracks=4)


def toy_objective(model, points, observations):
    weight = model.networks[0].weight.reshape(())
    head = model.coefficients[0]
    value = weight*weight+(head-3*weight)**2
    return value, value, value, {"constructed": value}


def toy_projection(model, points, observations):
    before = float(toy_objective(model, points, observations)[0].detach())
    with torch.no_grad():
        model.coefficients.copy_(3*model.networks[0].weight.reshape(1))
    after = float(toy_objective(model, points, observations)[0].detach())
    return {"accepted": True, "rank": 1, "resolved_condition": 1.,
            "rank_rcond": 1e-7, "objective_before": before, "objective_after": after}


class OptimizedContactTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def assert_nested_equal(self, left, right):
        if torch.is_tensor(left):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def step(self, model, optimizer, projection=toy_projection):
        with patch.object(optimized, "objective_continuity", toy_objective), \
             patch.object(optimized, "project_output", projection):
            return optimized.transactional_adam_step(model, {}, None, optimizer)

    def snapshot(self, model, optimizer):
        return (deepcopy(model.state_dict()), deepcopy(optimizer.state_dict()),
                [None if value.grad is None else value.grad.clone() for value in model.networks.parameters()])

    def assert_snapshot(self, model, optimizer, snapshot):
        self.assert_nested_equal(model.state_dict(), snapshot[0])
        self.assert_nested_equal(optimizer.state_dict(), snapshot[1])
        self.assert_nested_equal([value.grad for value in model.networks.parameters()], snapshot[2])

    def test_two_accepted_steps_preserve_adam_moments(self):
        model = ToyModel()
        optimizer = torch.optim.Adam(model.networks.parameters(), lr=.1)
        first = self.step(model, optimizer)
        first_moment = optimizer.state[next(model.networks.parameters())]["exp_avg"].clone()
        second_gradient = 2*model.networks[0].weight.detach().clone()
        second = self.step(model, optimizer)
        state = optimizer.state[next(model.networks.parameters())]
        self.assertEqual(int(state["step"]), 2)
        torch.testing.assert_close(state["exp_avg"], .9*first_moment+.1*second_gradient, rtol=1e-12, atol=1e-12)
        for event in (first, second):
            self.assertTrue(event["accepted"])
            self.assertLess(event["objective_after"], event["objective_before"])
            self.assertTrue(event["optimizer_state_committed"])
        self.assertEqual(second["optimizer_step_counts_before"], [1])
        self.assertEqual(second["optimizer_step_counts_after"], [2])

    def test_scaled_acceptance_commits_only_one_gradient_step(self):
        model = ToyModel()
        optimizer = torch.optim.Adam(model.networks.parameters(), lr=10.)
        event = self.step(model, optimizer)
        self.assertTrue(event["accepted"])
        self.assertEqual(event["backtracks"], 2)
        self.assertEqual(event["accepted_scale"], .25)
        state = optimizer.state[next(model.networks.parameters())]
        self.assertEqual(int(state["step"]), 1)
        torch.testing.assert_close(state["exp_avg"], torch.tensor([[.4]], dtype=torch.float64))
        self.assertEqual(len(event["trials"]), 3)
        json.dumps(event, allow_nan=False)

    def test_rejection_restores_nonempty_moments_groups_and_gradient_buffers(self):
        model = ToyModel()
        optimizer = torch.optim.Adam(model.networks.parameters(), lr=.1)
        self.step(model, optimizer)
        original = self.snapshot(model, optimizer)
        def worse(model, points, observations):
            with torch.no_grad():
                model.coefficients.fill_(100.)
            optimizer.param_groups[0]["lr"] = .789
            return {"accepted": True, "rank": 1, "resolved_condition": 1.}
        event = self.step(model, optimizer, worse)
        self.assertFalse(event["accepted"])
        self.assertEqual(len(event["trials"]), 5)
        self.assertTrue(event["optimizer_state_rolled_back"])
        self.assertEqual(event["optimizer_step_counts_before"], event["optimizer_step_counts_after"])
        self.assert_snapshot(model, optimizer, original)
        json.dumps(event, allow_nan=False)

    def test_numeric_readout_failure_restores_all_state(self):
        model = ToyModel()
        optimizer = torch.optim.Adam(model.networks.parameters(), lr=.1)
        self.step(model, optimizer)
        original = self.snapshot(model, optimizer)
        def nonfinite(model, points, observations):
            with torch.no_grad():
                model.coefficients.fill_(float("nan"))
            raise FloatingPointError("constructed nonfinite readout")
        event = self.step(model, optimizer, nonfinite)
        self.assertFalse(event["accepted"])
        self.assert_snapshot(model, optimizer, original)
        json.dumps(event, allow_nan=False)

    def test_unexpected_exception_rolls_back_then_propagates(self):
        model = ToyModel()
        optimizer = torch.optim.Adam(model.networks.parameters(), lr=.1)
        self.step(model, optimizer)
        original = self.snapshot(model, optimizer)
        def interruption(model, points, observations):
            with torch.no_grad():
                model.coefficients.fill_(83.)
            raise RuntimeError("constructed interruption")
        with self.assertRaisesRegex(RuntimeError, "constructed interruption"):
            self.step(model, optimizer, interruption)
        self.assert_snapshot(model, optimizer, original)

    def test_nonfinite_optimizer_moments_cannot_be_committed(self):
        model = ToyModel()
        optimizer = torch.optim.Adam(model.networks.parameters(), lr=.1)
        self.step(model, optimizer)
        original = self.snapshot(model, optimizer)
        actual = optimizer.step
        def corrupted():
            actual()
            optimizer.state[next(model.networks.parameters())]["exp_avg_sq"].fill_(float("inf"))
        with patch.object(optimizer, "step", corrupted):
            event = self.step(model, optimizer)
        self.assertFalse(event["accepted"])
        self.assertFalse(event["proposed_optimizer_state_finite"])
        self.assertTrue(event["proposed_hidden_displacement_finite"])
        self.assert_snapshot(model, optimizer, original)
        json.dumps(event, allow_nan=False)

    def test_optimized_model_and_grid_are_exactly_the_existing_log_off_field(self):
        config = optimized.OptimizedConfig(width=3, layers=1, adam_epochs=2)
        base = asdict(config); base.pop("optimizer_history")
        physics = PlanarContactConfig()
        actual, identity = optimized.matched_optimized_model(physics, config, 800., 91)
        expected, expected_identity = matched_graded_model(physics, GradedConfig(**base), 800., 91)
        self.assertEqual(identity, expected_identity)
        self.assert_nested_equal(actual.state_dict(), expected.state_dict())
        self.assertEqual(graded_point_metadata(physics, config, 73), graded_point_metadata(physics, GradedConfig(**base), 73))
        from thermotwin.pinn.planar_contact_continuity import objective_continuity, project_output
        self.assertIs(optimized.objective_continuity, objective_continuity)
        self.assertIs(optimized.project_output, project_output)
        with self.assertRaises(ValueError):
            replace(config, bottom_log_feature=True)
        with self.assertRaises(ValueError):
            replace(config, optimizer_history="unknown")

    def test_tiny_fit_exposes_history_policy_and_finite_evidence(self):
        observations = PlanarObservations(np.array([[.004, .002, 0.], [.03, .002, 20.]]), np.array([300., 299.]), .02)
        for history, instances in (("persistent", 1), ("reset", 2)):
            config = optimized.OptimizedConfig(width=2, layers=1, adam_epochs=2, interior_points=32,
                bottom_points=8, projection_chunk=16, max_backtracks=1, optimizer_history=history)
            with patch.object(optimized, "diagnose_optimized", side_effect=AssertionError("diagnostic leakage")):
                model, report = optimized.fit_optimized(observations, config=config, known_h=800., neural_seed=13, sampling_seed=71)
            self.assertEqual(report["optimizer_instances"], instances)
            self.assertEqual(report["optimizer_resets"], instances-1)
            self.assertEqual(len(report["hidden_update_history"]), 2)
            self.assertEqual(report["hidden_accepted_objective_increases"], 0)
            json.dumps(report, allow_nan=False)
            checkpoint = torch.load(io.BytesIO(optimized.checkpoint_bytes(model)), weights_only=False)
            self.assertEqual(checkpoint["model_class"], "OptimizedPINN")
            self.assert_nested_equal(checkpoint["optimizer_state"], model.final_optimizer_state)


if __name__ == "__main__":
    unittest.main()
