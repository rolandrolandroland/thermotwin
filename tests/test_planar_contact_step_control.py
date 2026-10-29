"""Constructed tests of controller state outside frozen Adam transactions."""
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
from thermotwin.pinn.planar_contact_inverse import PlanarObservations
from thermotwin.pinn import planar_contact_optimized as optimized
from thermotwin.pinn import planar_contact_step_control as controlled
from thermotwin.pinn.planar_contact_graded import graded_point_metadata


class ToyModel(nn.Module):
    def __init__(self, **options):
        super().__init__()
        self.config = controlled.StepControlConfig(gradient_clip=100., **options)
        self.networks = nn.ModuleList([nn.Linear(1, 1, bias=False, dtype=torch.float64)])
        with torch.no_grad():
            self.networks[0].weight.fill_(2.)
        self.coefficients = nn.Parameter(torch.tensor([6.], dtype=torch.float64), requires_grad=False)


def toy_objective(model, points, observations):
    weight = model.networks[0].weight.reshape(())
    value = weight*weight+(model.coefficients[0]-3*weight)**2
    return value, value, value, {"constructed": value}


def toy_projection(model, points, observations):
    with torch.no_grad():
        model.coefficients.copy_(3*model.networks[0].weight.reshape(1))
    return {"accepted": True, "rank": 1, "resolved_condition": 1., "rank_rcond": 1e-7}


def bad_projection(model, points, observations):
    with torch.no_grad():
        model.coefficients.fill_(100.)
    return {"accepted": True, "rank": 1, "resolved_condition": 1., "rank_rcond": 1e-7}


class StepControlTests(unittest.TestCase):
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

    def setup_model(self, **options):
        model = ToyModel(**options)
        optimizer = torch.optim.Adam(model.networks.parameters(), lr=model.config.learning_rate)
        state = controlled.StepControlState(model.config.learning_rate)
        return model, optimizer, state

    def step(self, model, optimizer, state, projection=toy_projection):
        with patch.object(optimized, "objective_continuity", toy_objective), \
             patch.object(optimized, "project_output", projection):
            return controlled.controller_step(model, {}, None, optimizer, state)

    def test_exhausted_transaction_restores_field_then_changes_only_controller_state(self):
        model, optimizer, state = self.setup_model(learning_rate=.1)
        self.step(model, optimizer, state)
        snapshot = deepcopy(model.state_dict())
        gradients = [value.grad.clone() for value in model.networks.parameters()]
        before_groups = deepcopy(optimizer.param_groups)
        event = self.step(model, optimizer, state, bad_projection)
        self.assertFalse(event["accepted"])
        self.assertTrue(event["optimizer_state_rolled_back"])
        self.assertEqual(event["optimizer_step_counts_before"], [1])
        self.assertEqual(event["optimizer_step_counts_after"], [1])
        self.assertEqual(event["controller_optimizer_step_counts_after"], [])
        self.assertEqual(state.learning_rate, .05)
        self.assertEqual(state.moment_restarts, 1)
        self.assertEqual(state.consecutive_failures, 1)
        self.assertEqual(optimizer.state, {})
        self.assert_nested_equal(model.state_dict(), snapshot)
        self.assert_nested_equal([value.grad for value in model.networks.parameters()], gradients)
        self.assertEqual(optimizer.param_groups[0]["betas"], before_groups[0]["betas"])
        self.assertEqual(optimizer.param_groups[0]["lr"], .05)
        self.assertTrue(event["identical_rejected_next_attempt_precluded"])
        json.dumps(event, allow_nan=False)

    def test_three_exhausted_searches_have_changed_rates_then_stop(self):
        model, optimizer, state = self.setup_model(learning_rate=.1)
        events = [self.step(model, optimizer, state, bad_projection) for _ in range(3)]
        self.assertEqual([event["controller_before"]["learning_rate"] for event in events], [.1, .05, .025])
        self.assertEqual([event["next_learning_rate"] for event in events], [.05, .025, .0125])
        self.assertEqual([event["controller_stop"] for event in events], [False, False, True])
        self.assertEqual(state.stop_reason, "consecutive exhausted searches")
        self.assertEqual(state.trial_solves, 15)
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            self.step(model, optimizer, state)

    def test_floor_is_attempted_before_rejection_at_floor_stops(self):
        model, optimizer, state = self.setup_model(learning_rate=1.5e-7)
        first = self.step(model, optimizer, state, bad_projection)
        self.assertEqual(state.learning_rate, 1.e-7)
        self.assertFalse(first["controller_stop"])
        second = self.step(model, optimizer, state, bad_projection)
        self.assertTrue(second["controller_stop"])
        self.assertEqual(second["controller_stop_reason"], "learning-rate floor reached")

    def test_backtracked_acceptance_carries_scale_and_preserves_committed_moments(self):
        model, optimizer, state = self.setup_model(learning_rate=10.)
        event = self.step(model, optimizer, state)
        self.assertTrue(event["accepted"])
        self.assertEqual(event["backtracks"], 2)
        self.assertEqual(state.learning_rate, 2.5)
        self.assertEqual(event["optimizer_step_counts_after"], [1])
        self.assertEqual(event["controller_optimizer_step_counts_after"], [1])
        self.assertFalse(event["controller_moments_restarted"])
        self.assertLess(event["objective_after"], event["objective_before"])

    def test_accepted_carry_to_floor_does_not_stop(self):
        model, optimizer, state = self.setup_model(learning_rate=2.e-7)
        event = {"accepted": True, "backtracks": 4, "accepted_scale": .0625,
            "trials": [{}]*5, "proposed_hidden_displacement_finite": True,
            "proposed_optimizer_state_finite": True}
        with patch.object(controlled, "transactional_adam_step", return_value=event):
            actual = controlled.controller_step(model, {}, None, optimizer, state)
        self.assertEqual(state.learning_rate, 1.e-7)
        self.assertFalse(actual["controller_stop"])

    def test_growth_requires_full_scale_streak_and_is_capped_at_initial_rate(self):
        model, optimizer, state = self.setup_model(learning_rate=.1)
        state.learning_rate = .025
        optimizer.param_groups[0]["lr"] = .025
        for _ in range(7):
            self.step(model, optimizer, state)
        self.assertEqual(state.learning_rate, .025)
        self.step(model, optimizer, state)
        self.assertEqual(state.learning_rate, .05)
        self.assertEqual(state.full_scale_successes, 0)
        for _ in range(24):
            self.step(model, optimizer, state)
        self.assertEqual(state.learning_rate, .1)
        self.assertEqual(state.full_scale_successes, 0)
        self.assertEqual(state.moment_restarts, 0)

    def test_rejection_resets_growth_streak_and_success_resets_failure_streak(self):
        model, optimizer, state = self.setup_model(learning_rate=.1)
        for _ in range(3):
            self.step(model, optimizer, state)
        self.assertEqual(state.full_scale_successes, 3)
        self.step(model, optimizer, state, bad_projection)
        self.assertEqual(state.full_scale_successes, 0)
        self.assertEqual(state.consecutive_failures, 1)
        self.step(model, optimizer, state)
        self.assertEqual(state.consecutive_failures, 0)
        self.assertEqual(state.full_scale_successes, 1)

    def test_unexpected_exception_restores_transaction_and_leaves_controller_untouched(self):
        model, optimizer, state = self.setup_model(learning_rate=.1)
        self.step(model, optimizer, state)
        model_before, optimizer_before, controller_before = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict()), asdict(state)
        def interrupted(model, points, observations):
            with torch.no_grad():
                model.coefficients.fill_(100.)
            raise RuntimeError("constructed interruption")
        with self.assertRaisesRegex(RuntimeError, "constructed interruption"):
            self.step(model, optimizer, state, interrupted)
        self.assert_nested_equal(model.state_dict(), model_before)
        self.assert_nested_equal(optimizer.state_dict(), optimizer_before)
        self.assertEqual(asdict(state), controller_before)

    def test_nonfinite_adam_proposal_restarts_after_exact_rollback(self):
        model, optimizer, state = self.setup_model(learning_rate=.1)
        self.step(model, optimizer, state)
        snapshot = deepcopy(model.state_dict())
        original = optimizer.step
        def corrupted():
            original()
            optimizer.state[next(model.networks.parameters())]["exp_avg_sq"].fill_(float("inf"))
        with patch.object(optimizer, "step", corrupted):
            event = self.step(model, optimizer, state)
        self.assertFalse(event["accepted"])
        self.assertFalse(event["proposed_optimizer_state_finite"])
        self.assertEqual(event["trial_solve_count"], 0)
        self.assert_nested_equal(model.state_dict(), snapshot)
        self.assertEqual(optimizer.state, {})
        self.assertEqual(state.learning_rate, .05)
        json.dumps(event, allow_nan=False)

    def test_field_initialization_grid_and_solver_are_inherited_without_changes(self):
        config = controlled.StepControlConfig(width=3, layers=1, adam_epochs=2)
        base = {key: value for key, value in asdict(config).items() if key in optimized.OptimizedConfig.__dataclass_fields__}
        physics = PlanarContactConfig()
        model, identity = controlled.matched_step_control_model(physics, config, 800., 91)
        expected, expected_identity = optimized.matched_optimized_model(physics, optimized.OptimizedConfig(**base), 800., 91)
        self.assertEqual(identity, expected_identity)
        self.assert_nested_equal(model.state_dict(), expected.state_dict())
        self.assertEqual(graded_point_metadata(physics, config, 73), graded_point_metadata(physics, optimized.OptimizedConfig(**base), 73))
        self.assertIs(controlled.objective_continuity, optimized.objective_continuity)
        self.assertIs(controlled.project_output, optimized.project_output)
        self.assertIs(controlled.transactional_adam_step, optimized.transactional_adam_step)
        for name, value in (("controller_reduction", 1.), ("controller_minimum_lr", 0.),
                            ("controller_max_failures", 0), ("controller_growth_streak", True),
                            ("controller_growth_factor", float("nan")), ("maximum_trial_solves", 0)):
            with self.assertRaises(ValueError):
                replace(config, **{name: value})

    def test_tiny_fit_budget_guard_checkpoint_and_final_state_only(self):
        observations = PlanarObservations(np.array([[.004, .002, 0.], [.03, .002, 20.]]), np.array([300., 299.]), .02)
        config = controlled.StepControlConfig(width=2, layers=1, adam_epochs=5, interior_points=32,
            bottom_points=8, projection_chunk=16, max_backtracks=1, maximum_trial_solves=2)
        events = []
        with patch.object(controlled, "diagnose_step_control", side_effect=AssertionError("diagnostic leakage")):
            model, report = controlled.fit_step_control(observations, config=config, known_h=800.,
                neural_seed=13, sampling_seed=71, progress=events.append)
        self.assertEqual(report["completed_updates"], 1)
        self.assertEqual(report["requested_updates"], 5)
        self.assertEqual(report["stop_reason"], "projection_budget")
        self.assertLessEqual(report["trial_solve_count"], 2)
        self.assertEqual(report["total_projection_solve_calls"], report["trial_solve_count"]+2)
        self.assertEqual(report["hidden_accepted_objective_increases"], 0)
        self.assertEqual(len(report["controller_history"]), 1)
        self.assertFalse(report["independent_diagnostics_used_for_training"])
        json.dumps(report, allow_nan=False)
        checkpoint = torch.load(io.BytesIO(controlled.checkpoint_bytes(model)), weights_only=False)
        self.assertEqual(checkpoint["model_class"], "StepControlPINN")
        self.assert_nested_equal(checkpoint["optimizer_state"], model.final_optimizer_state)
        self.assertEqual(checkpoint["controller_state"], report["final_controller_state"])
        self.assertEqual(events[-1]["stage"], "final_projection")


if __name__ == "__main__":
    unittest.main()
