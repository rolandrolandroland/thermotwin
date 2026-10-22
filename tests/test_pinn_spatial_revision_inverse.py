"""Constructed information-boundary and stage-gate checks; no scientific fits."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np
import torch

from thermotwin.pinn.planar_contact_inverse import PlanarObservations
from thermotwin.pinn.planar_contact_revision import PlanarRevisionConfig
from thermotwin.studies.pinn_spatial_revision_inverse import fit_delayed_inverse, physical_gate, INVERSE_SPEC


class ConstructedModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.raw_h = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))
        self.field = torch.nn.Linear(1, 1, dtype=torch.float64)

    @property
    def contact(self):
        return torch.tensor(INVERSE_SPEC['initial_h'], dtype=torch.float64)


def report(config, *, failed=False):
    values = {key: .01 for key in ('interior', 'left', 'right', 'top', 'bottom_contact', 'bottom_insulated')}
    if failed:
        values['interior'] = .2
    return {'architecture': 'soft_mlp',
            'physics': {'width': .04, 'contact_start': .008, 'contact_end': .024},
            'config': {}, 'independent_residuals': {'normalized_rms': values}}


class RevisedInverseBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.config = PlanarRevisionConfig(architecture='soft_mlp', adam_epochs=1, lbfgs_iterations=0)
        self.observations = PlanarObservations(np.array([[.02, .003, 10.]]), np.array([299.9]))

    def test_nominal_warm_start_sees_only_known_initial_condition(self):
        calls = []
        def trainer(observed, **kwargs):
            calls.append((observed, kwargs))
            return kwargs.get('initial_model') or ConstructedModel(kwargs['config']), report(kwargs['config'])
        with patch('thermotwin.studies.pinn_spatial_revision_inverse.train_planar_revision', side_effect=trainer):
            _, result = fit_delayed_inverse(self.observations, config=self.config)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(len(calls), 2)
        self.assertTrue(np.all(calls[0][0].coordinates[:, 2] == 0.))
        self.assertTrue(np.all(calls[0][0].temperature == 300.))
        self.assertEqual(calls[0][1]['config'].observation_weight, 0.)
        self.assertTrue(calls[0][1]['freeze_contact'])
        self.assertEqual(calls[1][1]['config'].observation_weight, 1.)
        self.assertIs(calls[1][0], self.observations)
        self.assertFalse(calls[1][1]['freeze_contact'])
        self.assertTrue(all(item[1]['known_h'] is None and item[1]['mode'] == 'inverse' for item in calls))

    def test_failed_nominal_physics_blocks_parameter_release(self):
        with patch('thermotwin.studies.pinn_spatial_revision_inverse.train_planar_revision',
                   return_value=(ConstructedModel(self.config), report(self.config, failed=True))) as train:
            _, result = fit_delayed_inverse(self.observations, config=self.config)
        self.assertEqual(train.call_count, 1)
        self.assertEqual(result['status'], 'blocked_nominal_forward_gate')

    def test_alternation_trains_only_the_declared_parameter_block(self):
        blocks = []
        def trainer(observed, **kwargs):
            model = kwargs.get('initial_model') or ConstructedModel(kwargs['config'])
            blocks.append({name: parameter.requires_grad for name, parameter in model.named_parameters()})
            return model, report(kwargs['config'])
        with patch('thermotwin.studies.pinn_spatial_revision_inverse.train_planar_revision', side_effect=trainer):
            _, result = fit_delayed_inverse(self.observations, config=self.config, strategy='alternating')
        self.assertEqual(len(blocks), 1+2*INVERSE_SPEC['alternating_cycles'])
        for index in range(INVERSE_SPEC['alternating_cycles']):
            self.assertTrue(blocks[1+2*index]['raw_h'])
            self.assertFalse(blocks[1+2*index]['field.weight'])
            self.assertFalse(blocks[2+2*index]['raw_h'])
            self.assertTrue(blocks[2+2*index]['field.weight'])
        self.assertFalse(result['unknown_truth_used'])

    def test_missing_region_or_interface_cannot_pass_physics_gate(self):
        value = report(self.config)
        self.assertTrue(physical_gate(value)['passed'])
        del value['independent_residuals']['normalized_rms']['bottom_insulated']
        self.assertFalse(physical_gate(value)['passed'])
        value = report(self.config)
        value['architecture'] = 'hard_boundary_decomposed'
        self.assertFalse(physical_gate(value)['passed'])

    def test_truth_parameter_is_not_a_fitting_argument(self):
        with self.assertRaises(TypeError):
            fit_delayed_inverse(self.observations, config=self.config, true_h=1000.)

    def test_completed_stage_is_retained_before_later_training_fails(self):
        events = []
        def trainer(observed, **kwargs):
            events.append('train')
            if len(events) > 1:
                raise RuntimeError('constructed later-stage interruption')
            return ConstructedModel(kwargs['config']), report(kwargs['config'])
        def retained(model, value, index):
            events.append(('retained', index, value['stage']))
        with patch('thermotwin.studies.pinn_spatial_revision_inverse.train_planar_revision', side_effect=trainer):
            with self.assertRaisesRegex(RuntimeError, 'later-stage interruption'):
                fit_delayed_inverse(self.observations, config=self.config, completed_stage=retained)
        self.assertEqual(events, ['train', ('retained', 0, 'nominal_physics_only_warm_start'), 'train'])


if __name__ == '__main__':
    unittest.main()
