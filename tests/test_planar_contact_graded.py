"""Constructed scientific checks for matched graded log-input arms."""
from dataclasses import replace
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from thermotwin.physics.planar_contact import PlanarContactConfig
from thermotwin.pinn.planar_contact_inverse import PlanarObservations
from thermotwin.pinn.planar_contact_continuity import residuals_continuity, training_points
from thermotwin.pinn import planar_contact_graded as graded


class GradedContactTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.physics = PlanarContactConfig()
        self.config = graded.GradedConfig(width=3, layers=1)

    def pair(self):
        on, left = graded.matched_graded_model(self.physics, self.config, 800., 123)
        off, right = graded.matched_graded_model(self.physics, replace(self.config, bottom_log_feature=False), 800., 123)
        return on, off, left, right

    def test_defaults_freeze_larger_budget_and_existing_solver(self):
        config = graded.GradedConfig()
        self.assertEqual((config.adam_epochs, config.projection_every, config.interior_points, config.bottom_points),
                         (80, 1, 8192, 512))
        self.assertEqual((config.width, config.layers, config.rank_rcond, config.max_backtracks), (24, 2, 1e-7, 2))
        from thermotwin.pinn.planar_contact_consistent import variable_projection_step
        from thermotwin.pinn.planar_contact_continuity import objective_continuity, project_output
        self.assertIs(graded.variable_projection_step, variable_projection_step)
        self.assertIs(graded.objective_continuity, objective_continuity)
        self.assertIs(graded.project_output, project_output)

    def test_common_parameters_match_exactly_and_probes_use_nonzero_readout(self):
        on, off, left, right = self.pair()
        self.assertEqual(left["common_weights_sha256"], right["common_weights_sha256"])
        self.assertEqual(left["shared_function_sha256"], right["shared_function_sha256"])
        self.assertNotEqual(left["complete_state_sha256"], right["complete_state_sha256"])
        for source, target, (kind, _) in zip(on.networks, off.networks, on.descriptors):
            for name, value in source.state_dict().items():
                expected = value[:, :-1] if name == "0.weight" and kind != "corner" else value
                torch.testing.assert_close(target.state_dict()[name], expected, rtol=0, atol=0)
            if kind != "corner":
                self.assertTrue(bool(torch.all(source[0].weight[:, -1] == 0)))
        proof = graded.verify_matched_initialization(self.physics, self.config,
            replace(self.config, bottom_log_feature=False), 800., 123)
        self.assertTrue(proof["probe_agreement"]["passed"])
        self.assertLess(proof["probe_agreement"]["basis_max_abs"], 1e-12)
        self.assertLess(proof["probe_agreement"]["normalized_hessian_max_abs"], 1e-10)
        self.assertEqual(proof["probe_coordinates_normalized"], graded.INITIALIZATION_PROBE_POINTS.tolist())
        self.assertTrue(bool(torch.all(on.coefficients == 0)))
        self.assertTrue(bool(torch.all(off.coefficients == 0)))
        json.dumps(proof, allow_nan=False)

    def test_zero_log_columns_remain_trainable_and_can_change_function(self):
        on, off, _, _ = self.pair()
        with torch.no_grad():
            head = .1*torch.cos(torch.arange(len(on.coefficients), dtype=torch.float64))
            on.coefficients.copy_(head); off.coefficients.copy_(head)
        q = torch.tensor([[.11, .002, .09], [.39, .07, .31], [.81, .41, .83]], dtype=torch.float64)
        before = on.dimensionless_temperature(q).detach().clone()
        loss = torch.sum((on.dimensionless_temperature(q)-.7)**2)
        loss.backward()
        gradients = [network[0].weight.grad[:, -1] for network, (kind, _) in zip(on.networks, on.descriptors) if kind != "corner"]
        self.assertTrue(any(float(torch.max(abs(value))) > 1e-7 for value in gradients))
        with torch.no_grad():
            on.networks[0][0].weight[:, -1].add_(.03)
        self.assertGreater(float(torch.max(abs(on.dimensionless_temperature(q)-before)).detach()), 1e-8)
        torch.testing.assert_close(off.dimensionless_temperature(q), before, rtol=1e-12, atol=1e-12)

    def test_both_arms_use_identical_graded_grid_with_fixed_strata_weights(self):
        for physics, counts, weights in (
            (self.physics, [4096, 2048, 2048], [.5, .25, .25]),
            (replace(self.physics, contact_start=0., contact_end=.04), [6144, 2048], [.75, .25]),
        ):
            on_points = graded.graded_training_points(physics, self.config, 421)
            off_config = replace(self.config, bottom_log_feature=False)
            off_points = graded.graded_training_points(physics, off_config, 421)
            metadata = graded.graded_point_metadata(physics, self.config, 421)
            self.assertEqual(metadata, graded.graded_point_metadata(physics, off_config, 421))
            self.assertEqual([row["count"] for row in metadata["strata"]], counts)
            self.assertEqual([row["mean_weight"] for row in metadata["strata"]], weights)
            self.assertEqual(len(on_points["interior"]), 8192)
            self.assertLess(metadata["interior_min_height_m"], 8e-6)
            self.assertGreater(metadata["interior_max_height_m"], .0037)
            self.assertTrue(np.all((on_points["interior"][:, :2] > 0) & (on_points["interior"][:, :2] < 1)))
            old = training_points(physics, replace(self.config, interior_points=4096), 421)
            for name in on_points:
                self.assertEqual(on_points[name].tobytes(), off_points[name].tobytes())
                if name != "interior":
                    self.assertEqual(on_points[name].tobytes(), old[name].tobytes())

    def test_matched_models_retain_hard_conditions_after_log_column_changes(self):
        on, off, _, _ = self.pair()
        for model in (on, off):
            with torch.no_grad():
                model.coefficients.copy_(.1*torch.sin(torch.arange(len(model.coefficients), dtype=torch.float64)))
                if model.config.bottom_log_feature:
                    for network, (kind, _) in zip(model.networks, model.descriptors):
                        if kind != "corner":
                            network[0].weight[:, -1].fill_(.03)
            points = {"left": np.array([[0., .31, .27]]), "right": np.array([[1., .71, .37]]),
                      "top": np.array([[.13, 1., .57], [.73, 1., .81]]),
                      "interface_0": np.array([[.2, .003, .21], [.2, .59, .41]]),
                      "interface_1": np.array([[.6, .013, .31], [.6, .79, .61]])}
            for name, values in residuals_continuity(model, points).items():
                self.assertLess(float(torch.max(abs(values)).detach()), 1e-9, name)
            coords = torch.tensor([[.004, .002, 0.], [.017, .001, 0.]], dtype=torch.float64)
            torch.testing.assert_close(model(coords), torch.full((2,), 300., dtype=torch.float64), rtol=0, atol=0)

    def test_tiny_fit_preserves_inherited_acceptance_and_saves_full_evidence(self):
        observations = PlanarObservations(np.array([[.004, .002, 0.], [.03, .002, 20.]]), np.array([300., 299.]), .02)
        config = replace(self.config, width=2, adam_epochs=1, interior_points=32, bottom_points=8, projection_chunk=16)
        with patch.object(graded, "diagnose_graded", side_effect=AssertionError("independent diagnostics entered fitting")):
            model, report = graded.fit_graded(observations, config=config, known_h=800., neural_seed=13, sampling_seed=71)
        self.assertEqual(len(report["hidden_update_history"]), 1)
        self.assertEqual(report["hidden_accepted_objective_increases"], 0)
        self.assertEqual(report["optimizer_resets"], 2)
        self.assertEqual(report["initialization"]["arm"], "on")
        for name, rows in report["training_points"].items():
            self.assertEqual(len(rows), len(report["training_point_weights"][name]))
            self.assertAlmostEqual(sum(report["training_point_weights"][name]), 1.)
        json.dumps(report, allow_nan=False)
        saved = torch.load(io.BytesIO(graded.checkpoint_bytes(model)), weights_only=False)
        self.assertEqual(saved["model_class"], "GradedPINN")
        self.assertEqual(saved["initialization"]["common_weights_sha256"], report["initialization"]["common_weights_sha256"])


if __name__ == "__main__":
    unittest.main()
