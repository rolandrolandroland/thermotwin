"""Optimizer-study boundary tests using constructed arrays and mocked fits."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from thermotwin.pinn.planar_contact_graded import GradedConfig
from thermotwin.pinn.planar_contact_optimized import OptimizedConfig
from thermotwin.studies import pinn_spatial_continuity as common
from thermotwin.studies import pinn_spatial_optimized as study


class OptimizedStudyTests(unittest.TestCase):
    def temporary(self):
        return tempfile.TemporaryDirectory(dir=common.ROOT/'reports', prefix='constructed-optimizer-runner-')

    def old_context(self, directory):
        return {'directory': str(Path(directory).resolve()), 'constructed': True}

    def old_forward(self, directory):
        return {'source_sha256': {}}, self.old_context(directory)

    def preflight(self, config, old_plan):
        values = np.array([[.1, .1, .1]])
        identities = {'interior': {'shape': [1, 3], 'sha256': study.graded._model_api()._array_digest(values)}}
        grid = {'point_digest': 'constructed-grid', 'point_identities': identities,
                'point_counts': {'interior': 1}, 'strata': [{'name': 'constructed', 'count': 1}]}
        return {case['id']: {'initialization': {'arm': 'off', 'common_weights_sha256': 'constructed-common',
            'complete_state_sha256': 'constructed-state', 'log_column_present': False,
            'probe_agreement': {'passed': True}}, 'training_grid': deepcopy(grid)} for case in common.CASES}

    def fixture(self, root, policy='persistent'):
        paired = root/'paired'/'two-d'
        for case in common.CASES:
            directory = paired/case['id'] if case['id'] in ('t01', 't02') else paired/'resource-continuation'/case['id']
            directory.mkdir(parents=True)
            np.savez(directory/'observations.npz', coordinates=np.array([[.02, .003, 10.]]),
                temperature=np.array([300.]), clean_temperature=np.array([-99999.]))
            np.savez(directory/'field_scoring.npz', coordinates=np.array([[.02, .002, 20.]]),
                truth=np.array([300.]), prediction=np.array([-99999.]))
        prior = root/'graded'
        prior.mkdir()
        common._write(prior/'plan.json', {'model_configs': {'log_off': asdict(GradedConfig(bottom_log_feature=False))},
            'source_sha256': {'thermotwin/studies/pinn_spatial_continuity.py': common.digest_file(common.ROOT/'thermotwin/studies/pinn_spatial_continuity.py'),
                'thermotwin/physics/planar_contact.py': common.digest_file(common.ROOT/'thermotwin/physics/planar_contact.py'),
                'thermotwin/pinn/planar_contact_inverse.py': common.digest_file(common.ROOT/'thermotwin/pinn/planar_contact_inverse.py')}})
        old = []
        for name in ('v3', 'v4'):
            directory = root/name
            directory.mkdir()
            (directory/'immutable').write_text('constructed preserved artifact')
            old.append(directory)
        config = asdict(OptimizedConfig(adam_epochs=320, learning_rate=.0004, optimizer_history=policy))
        evidence = root/'training-only-probe.json'
        common._write(evidence, {'constructed': True, 'training_only': True})
        decision_path = root/'decision.json'
        decision = {'status': 'approved_for_forward_development', 'training_only': True,
            'selected_config': config, 'reference_fields_used': False,
            'independent_diagnostics_used_for_selection': False,
            'probe_evidence': [{'path': str(evidence.relative_to(common.ROOT)), 'sha256': common.digest_file(evidence)}]}
        common._write(decision_path, decision)
        return dict(output_directory=root/'optimized', model_config=config,
            probe_decision_path=decision_path, source_paths=[], graded_comparison=prior,
            paired_v2_directory=paired, v2_development_record=None, preserved_campaign=None,
            protected_campaigns=old, protected_files=[])

    def prepare(self, arguments):
        with patch.object(study, '_preflight', side_effect=self.preflight):
            return study.prepare_optimized(**arguments)

    def callbacks(self, plan, nonfinite=None):
        calls = []
        def fit(observations, **kwargs):
            self.assertTrue(np.all(observations.temperature == 300.))
            self.assertNotIn('reference', kwargs)
            model = SimpleNamespace(physics=kwargs['physics'], known_h=kwargs['known_h'])
            case_id = study.graded._case_for_model(model)['id']
            calls.append(case_id)
            expected = plan['optimization_provenance']['preflight_by_case'][case_id]
            kwargs['progress']({'stage': 'constructed optimizer endpoint', 'step': plan['fixed_optimizer_updates']})
            return model, {'config': kwargs['config'], 'initialization': expected['initialization'],
                'graded_point_metadata': expected['training_grid'],
                'training_points': {'interior': [[.1, .1, .1]]},
                'training_point_weights': {'interior': [1.]},
                'optimizer_history': plan['optimizer_history'],
                'optimizer_instances': 1 if plan['optimizer_history'] == 'persistent' else plan['fixed_optimizer_updates'],
                'hidden_update_history': [{'accepted': False} for _ in range(plan['fixed_optimizer_updates'])]}
        def predict(model, coordinates):
            case_id = study.graded._case_for_model(model)['id']
            return np.full(len(coordinates), np.nan if case_id == nonfinite else 300.)
        def diagnose(model, points):
            return {'normalized_rms': {key: 0. for key in common.expected_families(model.physics)}}
        return common.ForwardCallbacks(fit=fit, predict=predict, diagnose=diagnose,
            checkpoint=lambda model: b'constructed final checkpoint'), calls

    def test_explicit_training_only_decision_required_before_preparation(self):
        with self.temporary() as temporary, patch.object(study.graded, 'validate_comparison', side_effect=self.old_context), \
             patch.object(study.graded, '_validate_old_campaign', side_effect=self.old_forward):
            arguments = self.fixture(Path(temporary))
            path = arguments['probe_decision_path']
            decision = study._read(path)
            for key, bad in (('training_only', False), ('reference_fields_used', True),
                             ('independent_diagnostics_used_for_selection', True), ('status', 'pending')):
                common._write(path, dict(decision, **{key: bad}))
                with self.assertRaisesRegex(ValueError, 'training-only'):
                    self.prepare(arguments)
                self.assertFalse(arguments['output_directory'].exists())
            common._write(path, decision)
            changed = deepcopy(arguments)
            changed['model_config']['learning_rate'] *= 2
            with self.assertRaisesRegex(ValueError, 'differs from the probe decision'):
                self.prepare(changed)

    def test_both_supported_history_policies_use_explicit_fixed_budget_and_once_only_execution(self):
        for policy in ('persistent', 'reset'):
            with self.subTest(policy=policy), self.temporary() as temporary, \
                 patch.object(study.graded, 'validate_comparison', side_effect=self.old_context), \
                 patch.object(study.graded, '_validate_old_campaign', side_effect=self.old_forward):
                arguments = self.fixture(Path(temporary), policy)
                plan = self.prepare(arguments)
                self.assertEqual(plan['fixed_optimizer_updates'], 320)
                self.assertEqual(plan['optimizer_history'], policy)
                self.assertEqual(plan['maximum_wall_seconds'], 14400.)
                self.assertFalse((arguments['output_directory']/'t01').exists())
                callbacks, calls = self.callbacks(plan)
                result = study.run_optimized(arguments['output_directory'], callbacks)
                self.assertEqual(result['status'], 'passed')
                self.assertEqual(calls, ['t01', 't02', 't03', 't04'])
                self.assertEqual(study.validate_optimized(arguments['output_directory'])['optimizer_history'], policy)
                with self.assertRaisesRegex(ValueError, 'already consumed'):
                    study.run_optimized(arguments['output_directory'], callbacks)

    def test_optimizer_revision_cannot_change_cutoff_grid_or_implicit_defaults(self):
        old = asdict(GradedConfig(bottom_log_feature=False))
        config = asdict(OptimizedConfig(adam_epochs=320))
        for key, value in (('rank_rcond', 1e-6), ('interior_points', 4096), ('physics_weight', 50.), ('width', 25), ('gradient_clip', 2.)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'frozen model, objective, grid, or rank'):
                study._config(dict(config, **{key: value}), old)
        incomplete = dict(config)
        incomplete.pop('learning_rate')
        with self.assertRaisesRegex(ValueError, 'undeclared defaults'):
            study._config(incomplete, old)

    def test_probe_evidence_change_blocks_run_before_fit(self):
        with self.temporary() as temporary, patch.object(study.graded, 'validate_comparison', side_effect=self.old_context), \
             patch.object(study.graded, '_validate_old_campaign', side_effect=self.old_forward):
            arguments = self.fixture(Path(temporary))
            plan = self.prepare(arguments)
            callbacks, calls = self.callbacks(plan)
            (Path(temporary)/'training-only-probe.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'probe evidence identity differs'):
                study.run_optimized(arguments['output_directory'], callbacks)
            self.assertEqual(calls, [])

    def test_nonfinite_failure_keeps_raw_checkpoint_and_remaining_cases(self):
        with self.temporary() as temporary, patch.object(study.graded, 'validate_comparison', side_effect=self.old_context), \
             patch.object(study.graded, '_validate_old_campaign', side_effect=self.old_forward):
            arguments = self.fixture(Path(temporary))
            plan = self.prepare(arguments)
            callbacks, calls = self.callbacks(plan, nonfinite='t02')
            result = study.run_optimized(arguments['output_directory'], callbacks)
            self.assertEqual(result['status'], 'blocked_forward_gate')
            self.assertEqual([row['status'] for row in result['cases']], ['completed', 'failed', 'completed', 'completed'])
            self.assertEqual(len(calls), 4)
            self.assertTrue((arguments['output_directory']/'t02/final_checkpoint.pt').is_file())
            self.assertTrue((arguments['output_directory']/'t02/training-report.json').is_file())

    def test_actual_training_points_weights_and_optimizer_mechanics_are_verified(self):
        with self.temporary() as temporary, patch.object(study.graded, 'validate_comparison', side_effect=self.old_context), \
             patch.object(study.graded, '_validate_old_campaign', side_effect=self.old_forward):
            arguments = self.fixture(Path(temporary))
            plan = self.prepare(arguments)
            callbacks, _ = self.callbacks(plan)
            observations = common.PlanarObservations(np.array([[.02, .003, 10.]]), np.array([300.]), .02)
            _, raw = callbacks.fit(observations, physics=common.physics_for(common.CASES[0]),
                config=plan['model_config'], known_h=800., neural_seed=1, sampling_seed=2, progress=lambda event: None)
            self.assertTrue(study._training_proof(raw, plan, 't01')['verified'])
            for mutation, error in ((lambda value: value['training_points']['interior'][0].__setitem__(1, .2), 'coordinates'),
                                    (lambda value: value['training_point_weights']['interior'].__setitem__(0, .5), 'row weights'),
                                    (lambda value: value.__setitem__('optimizer_instances', 5), 'history mechanics')):
                changed = deepcopy(raw)
                mutation(changed)
                with self.assertRaisesRegex(ValueError, error):
                    study._training_proof(changed, plan, 't01')

    def test_fresh_preflight_requires_original_common_weights_and_grid_not_old_optimizer_digest(self):
        config = asdict(OptimizedConfig(adam_epochs=320, learning_rate=.0004))
        grid = {'point_digest': 'same-grid', 'point_counts': {'interior': 8192}}
        initialization = {'common_weights_sha256': 'same-original-weights',
            'shared_function_sha256': 'new-optimizer-config-dependent-digest',
            'log_column_present': False, 'probe_agreement': {'passed': True}}
        prior = {'preflight_by_case': {case['id']: {'initialization': {'off': {
            'common_weights_sha256': 'same-original-weights', 'shared_function_sha256': 'old-digest'}},
            'training_grid': {'log_off': grid}} for case in common.CASES}}
        api = SimpleNamespace(OptimizedConfig=OptimizedConfig,
            matched_optimized_model=lambda *args: (None, deepcopy(initialization)))
        grid_api = SimpleNamespace(graded_point_metadata=lambda *args: deepcopy(grid))
        with patch.object(study, '_api', return_value=api), patch.object(study.graded, '_model_api', return_value=grid_api):
            records = study._preflight(config, prior)
            self.assertEqual(len(records), 4)
            initialization['common_weights_sha256'] = 'changed-initial-weights'
            with self.assertRaisesRegex(ValueError, 'original log-off common weights'):
                study._preflight(config, prior)

    def test_primary_diagnostics_require_checkpoint_and_verified_raw_training_proof(self):
        with self.temporary() as temporary, patch.object(study.graded, 'validate_comparison', side_effect=self.old_context), \
             patch.object(study.graded, '_validate_old_campaign', side_effect=self.old_forward):
            arguments = self.fixture(Path(temporary))
            plan = self.prepare(arguments)
            toy, _ = self.callbacks(plan)
            observations = common.PlanarObservations(np.array([[.02, .003, 10.]]), np.array([300.]), .02)
            model, raw = toy.fit(observations, physics=common.physics_for(common.CASES[0]),
                config=plan['model_config'], known_h=800., neural_seed=1, sampling_seed=2, progress=lambda event: None)
            calls = []
            inherited = SimpleNamespace(diagnose=lambda *args: calls.append('independent diagnostic'))
            api = SimpleNamespace(OptimizedConfig=OptimizedConfig, fit_optimized=None,
                predict_optimized=None, diagnose_optimized=None, checkpoint_bytes=None)
            with patch.object(study, '_api', return_value=api), \
                 patch.object(study.graded, 'graded_callbacks', return_value=inherited):
                callbacks = study.optimized_callbacks(arguments['output_directory'], plan)
            points = {'top': np.zeros((1, 3))}
            with self.assertRaisesRegex(RuntimeError, 'retained final checkpoint'):
                callbacks.diagnose(model, points)
            case = arguments['output_directory']/'t01'
            case.mkdir()
            (case/'final_checkpoint.pt').write_bytes(b'constructed')
            common._write(case/'training-report.json', raw)
            callbacks.diagnose(model, points)
            self.assertEqual(calls, ['independent diagnostic'])
            raw['optimizer_instances'] = 99
            common._write(case/'training-report.json', raw)
            with self.assertRaisesRegex(ValueError, 'history mechanics'):
                callbacks.diagnose(model, points)
            self.assertEqual(calls, ['independent diagnostic'])


if __name__ == '__main__':
    unittest.main()
