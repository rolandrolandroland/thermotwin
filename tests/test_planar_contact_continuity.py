"""Constructed checks for the additive continuity-constrained 2D PINN.

These tests exercise identities and numerical operators, not a scientific
cohort or a reference-field training run.
"""
from dataclasses import asdict, replace
import inspect
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import thermotwin.pinn.planar_contact_continuity as continuity_module

from thermotwin.physics.planar_contact import PlanarContactConfig
from thermotwin.pinn.planar_contact_inverse import PlanarObservations
from thermotwin.pinn.planar_contact_continuity import (
    ContinuityConfig,
    ContinuityPINN,
    diagnose_continuity,
    fit_continuity,
    linear_system,
    project_output,
    residuals_continuity,
)
from thermotwin.pinn.planar_contact_revision import (
    diagnose_planar_revision,
    revision_residuals,
)


def gradient(value, coordinates):
    return torch.autograd.grad(
        value, coordinates, torch.ones_like(value), create_graph=True,
    )[0]


class PlanarContinuityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(8721)
        self.config = ContinuityConfig(width=5, layers=2, adam_epochs=1,
                                       interior_points=12, bottom_points=8)
        self.physics = PlanarContactConfig()

    def model(self, physics=None):
        model = ContinuityPINN(physics=physics or self.physics,
                               config=self.config, known_h=800.)
        with torch.no_grad():
            model.coefficients.copy_(torch.randn_like(model.coefficients) * .3)
        return model

    def test_arbitrary_coefficients_share_temperature_and_normal_heat_flux(self):
        model = self.model()
        with torch.no_grad():
            model.coefficients.copy_(torch.randn_like(model.coefficients) * 2.7)
        for index, edge in enumerate((self.physics.contact_start,
                                      self.physics.contact_end)):
            q = torch.tensor([[edge / self.physics.width, z, t]
                              for z in (.0001, .03, .4, .91, 1.)
                              for t in (.001, .12, .8)], dtype=torch.float64,
                             requires_grad=True)
            left = model.dimensionless_temperature(q, region=index)
            right = model.dimensionless_temperature(q, region=index + 1)
            torch.testing.assert_close(left, right, rtol=1e-12, atol=2e-12)
            left_dx = gradient(left, q)[:, 0]
            right_dx = gradient(right, q)[:, 0]
            torch.testing.assert_close(left_dx, right_dx, rtol=1e-11, atol=2e-11)

    def test_top_side_fluxes_and_initial_temperature_are_exact_but_bottom_is_soft(self):
        for physics in (self.physics, replace(self.physics, contact_start=0.,
                                              contact_end=self.physics.width)):
            with self.subTest(full_contact=physics.contact_start == 0.):
                model = self.model(physics)
                times = np.array([.002, .13, .88])
                points = {
                    "left": np.column_stack((np.zeros(3), [.1, .5, .9], times)),
                    "right": np.column_stack((np.ones(3), [.1, .5, .9], times)),
                    "top": np.column_stack(([.13, .51, .87], np.ones(3), times)),
                    "bottom_contact": np.array([[.3, 0., .3], [.4, 0., .6]]),
                }
                if physics.contact_start:
                    points["bottom_insulated"] = np.array([[.1, 0., .3], [.9, 0., .6]])
                residuals = residuals_continuity(model, points)
                for family in ("left", "right", "top"):
                    self.assertLess(float(abs(residuals[family]).max().detach()),
                                    2e-10, family)
                self.assertGreater(float(abs(residuals["bottom_contact"]).max().detach()),
                                   1e-4)
                coordinates = torch.tensor([[.003, .001, 0.], [.018, .003, 0.],
                                            [.037, .004, 0.]], dtype=torch.float64)
                np.testing.assert_array_equal(model(coordinates).detach().numpy(),
                                              np.full(3, 300.))

    def test_ad_first_and_second_derivatives_agree_with_central_differences(self):
        model = self.model()
        location = np.array([[.37, .43, .28]])
        q = torch.tensor(location, dtype=torch.float64, requires_grad=True)
        value = model.dimensionless_temperature(q)
        first = gradient(value, q)
        second = [gradient(first[:, axis:axis + 1], q)[:, axis]
                  for axis in range(3)]
        step = 2e-4
        for axis in range(3):
            delta = np.zeros_like(location)
            delta[0, axis] = step
            with torch.no_grad():
                upper = float(model.dimensionless_temperature(torch.tensor(
                    location + delta, dtype=torch.float64)))
                lower = float(model.dimensionless_temperature(torch.tensor(
                    location - delta, dtype=torch.float64)))
                center = float(value.detach())
            self.assertAlmostEqual((upper - lower) / (2 * step),
                                   float(first[0, axis].detach()), delta=2e-5)
            self.assertAlmostEqual((upper - 2 * center + lower) / step ** 2,
                                   float(second[axis].detach()), delta=2e-4)

    def test_linear_basis_reconstructs_field_for_arbitrary_coefficients(self):
        model = self.model()
        q = torch.tensor([[.1, .2, .1], [.3, .6, .4], [.8, .4, .9]],
                         dtype=torch.float64)
        for _ in range(2):
            with torch.no_grad():
                model.coefficients.copy_(torch.randn_like(model.coefficients))
            expected = model.base_temperature(q).reshape(-1) + model.basis(q) @ model.coefficients
            torch.testing.assert_close(model.dimensionless_temperature(q).reshape(-1),
                                       expected, rtol=2e-12, atol=2e-12)

    def test_residual_implementation_keeps_v2_units_signs_and_families(self):
        model = self.model()
        points = {
            "interior": np.array([[.1, .3, .2], [.5, .8, .6], [.8, .4, .9]]),
            "left": np.array([[0., .2, .3]]),
            "right": np.array([[1., .8, .6]]),
            "top": np.array([[.4, 1., .6]]),
            "bottom_contact": np.array([[.3, 0., .2]]),
            "bottom_insulated": np.array([[.9, 0., .6]]),
            "interface_0": np.array([[.2, .3, .6]]),
            "interface_1": np.array([[.6, .7, .3]]),
        }
        expected = revision_residuals(model, points)
        actual = residuals_continuity(model, points)
        self.assertEqual(set(actual), set(expected))
        for family in expected:
            torch.testing.assert_close(actual[family], expected[family],
                                       rtol=2e-12, atol=2e-12)

    def test_near_corner_field_and_ad_remain_finite_but_corner_residual_is_rejected(self):
        model = self.model()
        for edge in (self.physics.contact_start, self.physics.contact_end):
            x = edge / self.physics.width
            q = torch.tensor([[x + offset, z, .4]
                              for offset in (-1e-5, 0., 1e-5)
                              for z in (1e-6, 1e-3, .04)], dtype=torch.float64,
                             requires_grad=True)
            value = model.dimensionless_temperature(q)
            first = gradient(value, q)
            self.assertTrue(bool(torch.isfinite(value).all()))
            self.assertTrue(bool(torch.isfinite(first).all()))
            for axis in (0, 1):
                second = gradient(first[:, axis:axis + 1], q)
                self.assertTrue(bool(torch.isfinite(second).all()))
            corner = np.array([[x, 0., .4]])
            with self.assertRaisesRegex(ValueError, "corner|edge"):
                residuals_continuity(model, {"bottom_contact": corner})

    def test_corner_enrichment_value_has_zero_corner_limit(self):
        model = self.model()
        for edge in (self.physics.contact_start, self.physics.contact_end):
            x = edge / self.physics.width
            at_corner = torch.tensor([[x, 0., .4]], dtype=torch.float64)
            value = float(model._corner_feature(at_corner, edge))
            self.assertTrue(np.isfinite(value))
            self.assertAlmostEqual(value, 0., delta=1e-12)
            absolute_values = []
            for distance in (1e-3, 1e-5, 1e-7):
                q = torch.tensor([[x + distance, distance, .4]], dtype=torch.float64)
                absolute_values.append(abs(float(model._corner_feature(q, edge))))
            self.assertGreater(absolute_values[0], absolute_values[1])
            self.assertGreater(absolute_values[1], absolute_values[2])
            self.assertLess(absolute_values[-1], 1e-5)

    def test_independent_diagnostics_use_unchanged_v2_points_scales_and_rms(self):
        model = self.model()
        # The facade only supplies v2's architecture label. Its field remains
        # the new model, and all physical trace constants are identical.
        facade = SimpleNamespace(
            physics=model.physics,
            config=SimpleNamespace(**asdict(model.config),
                                   architecture="hard_boundary_decomposed"),
            regions=model.regions,
            contact=model.contact,
            dimensionless_temperature=model.dimensionless_temperature,
        )
        captured = {}
        def record_points(passed_model, points):
            for family, coordinates in points.items():
                captured.setdefault(family, []).append(coordinates.copy())
            return revision_residuals(passed_model, points)
        with patch("thermotwin.pinn.planar_contact_revision.revision_residuals",
                   side_effect=record_points):
            expected = diagnose_planar_revision(facade, independent_seed=19237)
        points = {family: np.concatenate(chunks)
                  for family, chunks in captured.items()}
        actual = diagnose_continuity(model, points)
        self.assertEqual(actual["point_counts"], expected["point_counts"])
        self.assertEqual(actual["normalization_scales"], expected["normalization_scales"])
        self.assertEqual(set(actual["normalized_rms"]), set(expected["normalized_rms"]))
        for family in expected["normalized_rms"]:
            self.assertAlmostEqual(actual["normalized_rms"][family],
                                   expected["normalized_rms"][family],
                                   delta=1e-10 * (1 + expected["normalized_rms"][family]))

    def test_public_training_boundary_does_not_accept_hidden_field_labels(self):
        for callable_ in (ContinuityPINN, project_output):
            parameters = inspect.signature(callable_).parameters
            self.assertFalse(set(parameters) & {
                "true_h", "true_contact", "reference_field", "hidden_field",
                "truth", "family", "best_epoch",
            })
        with self.assertRaises(TypeError):
            self.model().forward(torch.zeros((1, 3), dtype=torch.float64),
                                 reference_field=np.ones(1))

    def linear_fixture(self):
        model = self.model()
        points = {
            "interior": np.array([[.1, .3, .2], [.4, .7, .4], [.8, .5, .8]]),
            "bottom_contact": np.array([[.3, 0., .2], [.5, 0., .7]]),
            "bottom_insulated": np.array([[.1, 0., .4], [.9, 0., .6]]),
        }
        observations = PlanarObservations(
            np.array([[.004, .003, 10.], [.018, .001, 42.], [.034, .003, 91.]]),
            np.array([299.8, 299.6, 299.9]),
        )
        return model, points, observations

    @staticmethod
    def direct_objective(model, points, observations):
        values = residuals_continuity(model, points)
        physics = sum(torch.mean(value.square()) for value in values.values())
        coordinates = torch.tensor(observations.coordinates.copy(), dtype=torch.float64)
        target = torch.tensor(observations.temperature.copy(), dtype=torch.float64)
        observation = torch.mean(((model(coordinates) - target) /
                                  observations.standard_deviation).square())
        return (model.config.physics_weight * physics +
                model.config.observation_weight * observation)

    def test_linear_system_matches_pde_bottom_observation_values_and_gradients(self):
        model, points, observations = self.linear_fixture()
        model.coefficients.requires_grad_(True)
        matrix, target, _ = linear_system(model, points, observations)
        matrix = torch.as_tensor(matrix, dtype=torch.float64)
        target = torch.as_tensor(target, dtype=torch.float64)
        for scale in (.1, .7):
            with torch.no_grad():
                model.coefficients.copy_(torch.randn_like(model.coefficients) * scale)
            actual = self.direct_objective(model, points, observations)
            expected_residual = matrix @ model.coefficients - target
            expected = expected_residual.square().sum()
            torch.testing.assert_close(actual, expected, rtol=2e-11, atol=1e-8)
            actual_gradient = torch.autograd.grad(actual, model.coefficients)[0]
            torch.testing.assert_close(actual_gradient,
                                       2 * matrix.T @ expected_residual,
                                       rtol=2e-10, atol=2e-7)

    def test_readout_projection_cannot_increase_same_point_objective(self):
        model, points, observations = self.linear_fixture()
        before_parameters = {
            key: value.detach().clone() for key, value in model.named_parameters()
            if value is not model.coefficients
        }
        before = float(self.direct_objective(model, points, observations).detach())
        project_output(model, points, observations)
        after = float(self.direct_objective(model, points, observations).detach())
        self.assertTrue(np.isfinite(after))
        self.assertLessEqual(after, before * (1 + 1e-12) + 1e-10)
        self.assertLess(after, before * .99)
        for key, value in model.named_parameters():
            if key in before_parameters:
                torch.testing.assert_close(value, before_parameters[key], rtol=0., atol=0.)

    def test_finite_overflowing_readout_is_rejected_restored_and_serializable(self):
        model, points, observations = self.linear_fixture()
        original = {name: value.detach().clone()
                    for name, value in model.named_parameters()}
        before = float(self.direct_objective(model, points, observations).detach())
        def overflowing_solution(matrix, target, rcond):
            rank = min(matrix.shape)
            return (np.full(matrix.shape[1], 1e200), np.empty(0),
                    rank, np.ones(rank))
        with patch.object(continuity_module.np.linalg, "lstsq", overflowing_solution), \
             np.errstate(over="ignore", invalid="ignore"):
            report = project_output(model, points, observations)
        self.assertFalse(report["accepted"])
        self.assertFalse(report["proposed_objective_finite"])
        self.assertIsNone(report["objective_proposed"])
        self.assertEqual(report["objective_after"], before)
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, original[name], rtol=0., atol=0.)
        json.dumps(report, allow_nan=False)

    def test_tiny_fit_uses_only_training_points_and_nonincreasing_hidden_updates(self):
        _, _, observations = self.linear_fixture()
        config = replace(self.config, width=2, layers=1, adam_epochs=2,
                         projection_every=1, interior_points=16,
                         bottom_points=8, projection_chunk=16)
        with patch.object(continuity_module, "diagnose_continuity",
                          side_effect=AssertionError("diagnostics entered training")):
            model, report = fit_continuity(
                observations, config=config, known_h=800.,
                neural_seed=511, sampling_seed=712,
            )
        self.assertEqual(len(report["hidden_update_history"]), 2)
        self.assertEqual(len(report["projections"]), 4)
        self.assertEqual(report["hidden_accepted_objective_increases"], 0)
        for update in report["hidden_update_history"]:
            self.assertLessEqual(update["objective_after"], update["objective_before"])
        for projection in report["projections"]:
            self.assertLessEqual(projection["objective_after"], projection["objective_before"])
        points = {name: np.asarray(values)
                  for name, values in report["training_points"].items()}
        actual = float(self.direct_objective(model, points, observations).detach())
        self.assertAlmostEqual(actual, report["final_objective"], places=9)
        self.assertFalse(report["hidden_truth_labels_used"])
        self.assertFalse(report["independent_diagnostics_used_for_training"])

    def test_failed_hidden_proposals_restore_field_readout_and_reset_optimizer(self):
        _, _, observations = self.linear_fixture()
        config = replace(self.config, width=2, layers=1, adam_epochs=1,
                         projection_every=2, interior_points=16,
                         bottom_points=8, projection_chunk=16, max_backtracks=3)
        actual_project = project_output
        held_coordinates = torch.tensor([[.007, .002, 15.], [.023, .003, 71.]],
                                        dtype=torch.float64)
        for failure in ("nonfinite", "finite_ascent"):
            with self.subTest(failure=failure):
                optimizers = []
                initial = {}
                class DeliberatelyBadAdam:
                    def __init__(self, parameters, lr):
                        self.parameters = list(parameters)
                        optimizers.append(self)
                    def zero_grad(self):
                        for parameter in self.parameters:
                            parameter.grad = None
                    def step(self):
                        with torch.no_grad():
                            for parameter in self.parameters:
                                if failure == "nonfinite":
                                    parameter.fill_(float("nan"))
                                else:
                                    parameter.add_(parameter.grad * 1e-4)
                def capture_first_projection(model, points, observed):
                    report = actual_project(model, points, observed)
                    if not initial:
                        initial["parameters"] = {
                            name: value.detach().clone()
                            for name, value in model.named_parameters()
                        }
                        initial["field"] = model(held_coordinates).detach().clone()
                    return report
                with patch.object(continuity_module.torch.optim, "Adam", DeliberatelyBadAdam), \
                     patch.object(continuity_module, "project_output", capture_first_projection):
                    model, report = fit_continuity(
                        observations, config=config, known_h=800.,
                        neural_seed=511, sampling_seed=712,
                    )
                self.assertEqual(report["hidden_rejected_updates"], 1)
                self.assertEqual(report["optimizer_resets"], 2)
                self.assertEqual(len(optimizers), 2)
                update = report["hidden_update_history"][0]
                self.assertFalse(update["accepted"])
                self.assertEqual(update["backtracks"], config.max_backtracks)
                self.assertEqual(update["objective_after"], update["objective_before"])
                for name, value in model.named_parameters():
                    torch.testing.assert_close(value, initial["parameters"][name],
                                               rtol=0., atol=0.)
                torch.testing.assert_close(model(held_coordinates), initial["field"],
                                           rtol=0., atol=0.)


if __name__ == "__main__":
    unittest.main()
