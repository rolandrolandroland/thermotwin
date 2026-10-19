import math
import unittest
from dataclasses import replace

import numpy as np
import torch

from thermotwin.physics.planar_contact import PlanarContactConfig
from thermotwin.pinn.planar_contact_inverse import (
    PlanarContactPINN, PlanarPINNConfig, PlanarObservations, residual_tensors,
    train_planar_contact_pinn, predict_planar_temperature,
)


class PlanarPINNTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = replace(PlanarPINNConfig(), width=6, layers=2, adam_epochs=2,
                              lbfgs_iterations=0, interior_points=8, boundary_points_per_side=5)

    def test_hard_initial_condition_and_matched_field_initialization(self):
        digests = []
        for mode in ('inverse', 'known_contact', 'data_only'):
            torch.manual_seed(818)
            model = PlanarContactPINN(config=self.config, mode=mode,
                                     known_h=1000. if mode == 'known_contact' else None)
            points = torch.tensor([[.001, .001, 0.], [.039, .003, 0.]], dtype=torch.float64)
            np.testing.assert_array_equal(model(points).detach().numpy(), [300., 300.])
            digests.append(model.initial_field_digest())
        self.assertEqual(len(set(digests)), 1)

    def test_truth_contact_is_forbidden_outside_explicit_oracle(self):
        for mode in ('inverse', 'data_only'):
            with self.assertRaisesRegex(ValueError, 'forbidden'):
                PlanarContactPINN(config=self.config, mode=mode, known_h=1000.)
        with self.assertRaises(ValueError):
            PlanarContactPINN(config=self.config, mode='known_contact')

    def test_residuals_match_manufactured_polynomial_in_physical_units(self):
        model = PlanarContactPINN(config=self.config)
        a, b, c = .2, .3, -.1
        model.dimensionless_temperature = lambda q: a*q[:, 0:1]**2 + b*q[:, 1:2]**2 + c*q[:, 2:3]
        points = {'interior': np.array([[.5, .4, .5]]), 'top': np.array([[.5, 1., .5]]),
                  'bottom': np.array([[.4, 0., .5]]), 'right': np.array([[1., .4, .5]])}
        residual = residual_tensors(model, points)
        p, cfg = model.physics, model.config
        expected_rate = cfg.temperature_scale * (c/cfg.duration - p.diffusivity*(2*a/p.width**2+2*b/p.thickness**2))
        self.assertAlmostEqual(float(residual['interior'].detach()), expected_rate, places=12)
        ramp = 1-math.exp(-.5*cfg.duration/cfg.ramp_seconds)
        self.assertAlmostEqual(float(residual['top'].detach()),
            p.thermal_conductivity*cfg.temperature_scale/p.thickness*2*b-cfg.top_flux_watts_per_square_metre*ramp, places=10)
        expected_bottom = -cfg.initial_h*(cfg.temperature_scale*(a*.4**2+c*.5)+cfg.bath_drop_kelvin*ramp)
        self.assertAlmostEqual(float(residual['bottom'].detach()), expected_bottom, places=10)
        self.assertAlmostEqual(float(residual['right'].detach()),
            p.thermal_conductivity*cfg.temperature_scale/p.width*2*a, places=10)

    def test_blinded_training_emits_no_fake_data_only_contact_estimate(self):
        observed = PlanarObservations(np.array([[.004, .003, 0.], [.02, .003, 10.], [.036, .003, 20.]]),
                                      np.array([300., 299.9, 299.8]))
        model, result = train_planar_contact_pinn(observed, config=self.config, mode='data_only')
        self.assertIsNone(result['contact_estimate'])
        self.assertIsNone(result['independent_residuals']['normalized_rms']['bottom'])
        self.assertEqual(len(result['history']), 2)
        self.assertTrue(np.isfinite(predict_planar_temperature(model, observed.coordinates)).all())
        with self.assertRaises(TypeError):
            train_planar_contact_pinn({'temperature': observed.temperature, 'true_h': 1000.}, config=self.config)

    def test_positive_contact_bounds_and_finite_inverse_gradients(self):
        observations = PlanarObservations(np.array([[.004, .003, 10.], [.02, .003, 20.]]),
                                         np.array([299.9, 299.8]))
        model, result = train_planar_contact_pinn(observations, config=self.config)
        self.assertTrue(self.config.h_bounds[0] < result['contact_estimate'] < self.config.h_bounds[1])
        self.assertTrue(all(math.isfinite(v) for v in result['independent_residuals']['normalized_rms'].values()))
        self.assertEqual(result['final_checkpoint'], 'optimizer-returned final state; no truth-selected checkpoint')

    def test_rejects_invalid_bounds_and_shared_diagnostic_seed(self):
        for bounds in ((0., 2000.), (400., float('nan')), (400.,), (2000., 400.)):
            with self.assertRaises(ValueError):
                replace(self.config, h_bounds=bounds)
        original = [400., 2000.]
        frozen = replace(self.config, h_bounds=original)
        original[0] = -1.
        self.assertEqual(frozen.h_bounds, (400., 2000.))
        observations = PlanarObservations(np.array([[.02, .003, 10.]]), np.array([299.9]))
        with self.assertRaisesRegex(ValueError, 'distinct'):
            train_planar_contact_pinn(observations, config=self.config,
                                      collocation_seed=7, diagnostic_seed=7)

    def test_lbfgs_reports_returned_state_loss_and_actual_iterations(self):
        observations = PlanarObservations(np.array([[.004, .003, 10.], [.02, .003, 20.]]),
                                         np.array([299.9, 299.8]))
        config = replace(self.config, lbfgs_iterations=2)
        model, result = train_planar_contact_pinn(observations, config=config)
        predictions = predict_planar_temperature(model, observations.coordinates)
        final_mse = np.mean(((predictions-observations.temperature)/observations.standard_deviation)**2)
        self.assertAlmostEqual(final_mse, result['normalized_observation_mse'], places=10)
        self.assertTrue(0 < result['lbfgs_actual_iterations'] <= 2)
        self.assertGreaterEqual(result['lbfgs_closure_evaluations'], result['lbfgs_actual_iterations'])


if __name__ == '__main__':
    unittest.main()
