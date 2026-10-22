"""Execution-boundary checks using constructed observations; no numerical fits."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from thermotwin.pinn.planar_contact_inverse import PlanarObservations
from thermotwin.pinn import planar_contact_revision as prior
from thermotwin.studies import pinn_spatial_continuity as runner


class ContinuityRunnerTests(unittest.TestCase):
    def prepare(self, directory, **kwargs):
        paired = directory.parent/'paired-v2'
        paired.mkdir(exist_ok=True)
        for case in runner.CASES:
            original = paired/case['id'] if case['id'] in ('t01', 't02') else paired/'resource-continuation'/case['id']
            original.mkdir(parents=True, exist_ok=True)
            np.savez(original/'observations.npz', coordinates=np.array([[.02, .003, 10.]]),
                temperature=np.array([300.]), clean_temperature=np.array([-99999.]))
            np.savez(original/'field_scoring.npz', coordinates=np.array([[.02, .002, 20.]]),
                truth=np.array([300.]), prediction=np.array([-99999.]))
        return runner.prepare_development(directory, model_config={'width': 24, 'duration': 120.},
            source_paths=['tests/test_pinn_spatial_continuity.py'], paired_v2_directory=paired, **kwargs)

    def callbacks(self, directory, failures=()):
        calls = []
        def fit(observations, **kwargs):
            calls.append(kwargs)
            self.assertIsInstance(observations, PlanarObservations)
            self.assertEqual(set(kwargs), {'physics', 'config', 'known_h', 'neural_seed', 'sampling_seed', 'progress'})
            if len(calls) in failures:
                raise ValueError('constructed fit failure')
            kwargs['progress']({'stage': 'constructed', 'step': 1})
            return SimpleNamespace(physics=kwargs['physics'], ordinal=len(calls)), {'training_only': True}
        def diagnose(model, points):
            case_directory = directory/runner.CASES[model.ordinal-1]['id']
            self.assertEqual((case_directory/'final_checkpoint.pt').read_bytes(), b'constructed checkpoint')
            self.assertEqual(json.loads((case_directory/'training-report.json').read_text()), {'training_only': True})
            self.assertFalse((case_directory/'field_scoring.npz').exists())
            expected = runner.independent_points(model.physics) if 'top' in points else runner.near_interface_points(model.physics)
            self.assertEqual(runner.point_digest(points), runner.point_digest(expected))
            return {'normalized_rms': {name: 0. for name in runner.expected_families(model.physics)}}
        callbacks = runner.ForwardCallbacks(fit=fit, predict=lambda model, coordinates: np.full(len(coordinates), 300.),
            diagnose=diagnose, checkpoint=lambda model: b'constructed checkpoint')
        return callbacks, calls

    def test_independent_coordinates_are_byte_identical_to_v2_diagnostics(self):
        for full in (False, True):
            physics = runner.physics_for(runner.CASES[0 if full else 1])
            model = SimpleNamespace(physics=physics, config=prior.PlanarRevisionConfig())
            model.regions = [(0., .04, True)] if full else [(0., .008, False), (.008, .024, True), (.024, .04, False)]
            captured = {}
            def residuals(model, points):
                for key, values in points.items():
                    captured.setdefault(key, []).append(values.copy())
                return {key: torch.zeros((len(values), 1), dtype=torch.float64) for key, values in points.items()}
            with patch.object(prior, 'revision_residuals', side_effect=residuals):
                prior.diagnose_planar_revision(model, independent_seed=runner.SEEDS['diagnostic'])
            expected = runner.independent_points(physics)
            self.assertEqual(set(captured), set(expected))
            for key, chunks in captured.items():
                np.testing.assert_array_equal(np.concatenate(chunks), expected[key])

    def test_gate_requires_every_applicable_family_and_unchanged_scales(self):
        physics = runner.physics_for(runner.CASES[1])
        residuals = {name: .05 for name in runner.expected_families(physics)}
        field = {'rmse_K': .05, 'maximum_error_K': .15}
        self.assertTrue(runner.acceptance(physics, 2., residuals, field)['all'])
        for key, value in (('interior', .050001), ('interface_0_temperature', float('nan')),
                           ('interface_1_flux', -.01)):
            changed = dict(residuals, **{key: value})
            self.assertFalse(runner.acceptance(physics, 2., changed, field)['all'])
        missing = deepcopy(residuals)
        missing.pop('interface_0_flux')
        self.assertFalse(runner.acceptance(physics, 2., missing, field)['all'])
        self.assertFalse(runner.acceptance(physics, float('nan'), residuals, field)['all'])
        self.assertFalse(runner.acceptance(physics, 2., residuals, dict(field, rmse_K=float('inf')))['all'])
        self.assertEqual(runner.GATES['interface_temperature_scale_K']*.05, .005000000000000001)

    def test_near_interface_audit_has_fixed_noncorner_points_and_is_not_a_gate(self):
        self.assertEqual(runner.near_interface_points(runner.physics_for(runner.CASES[0])), {})
        points = runner.near_interface_points(runner.physics_for(runner.CASES[1]))
        self.assertEqual(set(points), {'interior', 'interface_0', 'interface_1'})
        self.assertEqual(points['interior'].shape, (360, 3))
        self.assertEqual(points['interface_0'].shape, (30, 3))
        for values in points.values():
            self.assertTrue((values > 0).all())
            self.assertTrue((values < 1).all())
        np.testing.assert_array_equal(np.unique(points['interior'][:, 1]), [.002, .01, .05, .2, .5])
        np.testing.assert_allclose(np.unique(points['interior'][:, 2])*120., [.4, 1.1, 4.5, 14., 45., 105.])

    def test_plan_is_prepared_before_reference_or_fit_and_is_single_use(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/'run'
            plan = self.prepare(directory)
            self.assertEqual(set(path.name for path in directory.iterdir()), {'plan.json', 'state.json'})
            self.assertEqual(plan['cases'], list(runner.CASES))
            self.assertEqual(plan['seeds'], runner.SEEDS)
            callbacks, calls = self.callbacks(directory)
            result = runner.run_development(directory, callbacks)
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(len(calls), 4)
            self.assertEqual(len(result['cases']), 4)
            self.assertEqual(len({json.dumps(row['config'], sort_keys=True) for row in result['cases']}), 1)
            self.assertFalse(result['fresh_evaluation_opened'])
            for case in runner.CASES:
                audit = json.loads((directory/case['id']/'near-interface-audit.json').read_text())
                self.assertEqual(audit['status'], 'not_applicable' if case['geometry'] == 'full_contact' else 'completed')
            self.assertTrue((directory/'artifact-index.json').exists())
            with self.assertRaisesRegex(ValueError, 'consumed'):
                runner.run_development(directory, callbacks)

    def test_fit_failure_is_retained_and_other_predeclared_cases_continue(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/'run'
            self.prepare(directory)
            callbacks, calls = self.callbacks(directory, failures=(2,))
            result = runner.run_development(directory, callbacks)
            self.assertEqual(len(calls), 4)
            self.assertEqual(result['status'], 'blocked_forward_gate')
            self.assertEqual([row['status'] for row in result['cases']], ['completed', 'failed', 'completed', 'completed'])
            self.assertTrue((directory/'t02/failure.json').exists())
            self.assertIsNone(result['selected_config'])

    def test_resource_limit_retains_failure_and_unrun_cases_without_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/'run'
            self.prepare(directory, maximum_rss_bytes=1)
            callbacks, calls = self.callbacks(directory)
            result = runner.run_development(directory, callbacks)
            self.assertEqual(calls, [])
            self.assertEqual([row['status'] for row in result['cases']], ['failed', 'not_run', 'not_run', 'not_run'])
            self.assertEqual(result['status'], 'blocked_forward_gate')
            self.assertFalse((directory/'t01/observations.npz').exists())

    def test_mutated_prepared_plan_and_protected_evidence_block_before_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protected = root/'v2'
            protected.mkdir()
            (protected/'old-result.json').write_text('immutable')
            directory = root/'run'
            self.prepare(directory, protected_directories=[protected])
            callbacks, _ = self.callbacks(directory)
            (protected/'old-result.json').write_text('changed')
            with patch.object(runner, 'paired_observations') as reference:
                with self.assertRaisesRegex(RuntimeError, 'protected'):
                    runner.run_development(directory, callbacks)
                reference.assert_not_called()
            (protected/'old-result.json').write_text('immutable')
            plan = json.loads((directory/'plan.json').read_text())
            plan['cases'][0]['known_h'] = 1000.
            (directory/'plan.json').write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'plan is changed'):
                runner.run_development(directory, callbacks)

    def test_changed_source_blocks_before_data_or_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/'run'
            plan = self.prepare(directory)
            callbacks, _ = self.callbacks(directory)
            changed = dict(plan['source_sha256'])
            changed[next(iter(changed))] = 'changed'
            with patch.object(runner, '_source_records', return_value=changed), \
                 patch.object(runner, 'paired_observations') as reference:
                with self.assertRaisesRegex(RuntimeError, 'source changed'):
                    runner.run_development(directory, callbacks)
                reference.assert_not_called()

    def test_sealed_records_ignore_unindexed_cache_but_reject_changed_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            evidence = directory/'result.json'
            evidence.write_text('immutable')
            records = [{'path': evidence.name, 'bytes': evidence.stat().st_size,
                        'sha256': runner.digest_file(evidence)}]
            (directory/'evidence-index.json').write_text(json.dumps({'files': records,
                'digest': runner.canonical_digest(records)}))
            before = runner._indexed_context(directory)
            (directory/'incidental-cache').write_text('not part of prior seal')
            self.assertEqual(runner._indexed_context(directory), before)
            evidence.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'sealed v2 evidence changed'):
                runner._indexed_context(directory)

    def test_plan_change_during_fit_retains_raw_state_and_blocks_scoring_and_later_arms(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/'run'
            self.prepare(directory)
            callbacks, calls = self.callbacks(directory)
            def change_plan(observations, **kwargs):
                returned = callbacks.fit(observations, **kwargs)
                plan = json.loads((directory/'plan.json').read_text())
                plan['model_config']['width'] += 1
                (directory/'plan.json').write_text(json.dumps(plan))
                return returned
            changed = runner.ForwardCallbacks(fit=change_plan, predict=callbacks.predict,
                diagnose=callbacks.diagnose, checkpoint=callbacks.checkpoint)
            with patch.object(runner, 'paired_scoring_field') as hidden:
                result = runner.run_development(directory, changed)
                hidden.assert_not_called()
            self.assertEqual(len(calls), 1)
            self.assertFalse(result['source_unchanged'])
            self.assertEqual([row['status'] for row in result['cases']], ['failed', 'not_run', 'not_run', 'not_run'])
            self.assertTrue((directory/'t01/final_checkpoint.pt').exists())
            self.assertTrue((directory/'t01/training-report.json').exists())
            self.assertFalse((directory/'t01/independent-diagnostics.json').exists())

    def test_supplemental_audit_failure_does_not_change_primary_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/'run'
            self.prepare(directory)
            callbacks, _ = self.callbacks(directory)
            def diagnose(model, points):
                if 'top' not in points:
                    raise ValueError('constructed supplemental audit error')
                return callbacks.diagnose(model, points)
            changed = runner.ForwardCallbacks(fit=callbacks.fit, predict=callbacks.predict,
                diagnose=diagnose, checkpoint=callbacks.checkpoint)
            result = runner.run_development(directory, changed)
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['cases'][1]['supplemental_audit_status'], 'failed')


if __name__ == '__main__':
    unittest.main()
