"""Isolated controller orchestration tests with constructed, unfit evidence."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import signal
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from thermotwin.pinn.planar_contact_optimized import OptimizedConfig
from thermotwin.pinn.planar_contact_step_control import StepControlConfig, StepControlState
from thermotwin.studies import pinn_spatial_continuity as common
from thermotwin.studies import pinn_spatial_step_control as study


class StepControlStudyTests(unittest.TestCase):
    def temporary(self):
        return tempfile.TemporaryDirectory(dir=common.ROOT/'reports', prefix='constructed-controller-runner-')

    def context(self, path):
        return {'directory': str(Path(path).resolve()), 'constructed': True}

    def preflight(self, config, prior):
        points = np.array([[.1, .1, .1]])
        grid = {'point_identities': {'interior': {'shape': [1, 3],
            'sha256': study.graded._model_api()._array_digest(points)}},
            'point_counts': {'interior': 1}, 'point_digest': 'constructed'}
        return {case['id']: {'initialization': {'common_weights_sha256': 'original-common',
            'complete_state_sha256': 'fresh-state', 'log_column_present': False,
            'probe_agreement': {'passed': True}}, 'training_grid': deepcopy(grid)} for case in common.CASES}

    def fixture(self, root, **config_changes):
        prior = root/'prior-v6'
        prior.mkdir()
        bindings = {}
        for case in common.CASES:
            directory = root/'paired'/case['id']
            directory.mkdir(parents=True)
            np.savez(directory/'observations.npz', coordinates=[[.02, .003, 10.]],
                temperature=[300.], clean_temperature=[-99999.])
            np.savez(directory/'field_scoring.npz', coordinates=[[.02, .002, 20.]],
                truth=[300.], prediction=[-99999.])
            bindings[case['id']] = {name: {'path': str(directory/name),
                'sha256': common.digest_file(directory/name)} for name in ('observations.npz', 'field_scoring.npz')}
        prior_config = asdict(OptimizedConfig(adam_epochs=320, learning_rate=.0064, max_backtracks=4))
        config = asdict(StepControlConfig(**dict(prior_config, adam_epochs=4), **config_changes))
        sources = {'thermotwin/studies/pinn_spatial_continuity.py':
            common.digest_file(common.ROOT/'thermotwin/studies/pinn_spatial_continuity.py')}
        old_plan = {'model_config': prior_config, 'paired_evidence': bindings, 'source_sha256': sources,
            'optimization_provenance': {'graded_comparison': {'directory': str(root/'graded'),
                'context': {'constructed': True}, 'plan_sha256': 'constructed-graded-plan'},
                'preflight_by_case': self.preflight(config, None)}}
        common._write(prior/'plan.json', old_plan)
        (prior/'retained-failure.txt').write_text('complete prior failed attempt is immutable')
        evidence = root/'probe.json'
        common._write(evidence, {'training_only': True, 'scientific_recovery_supported': False})
        decision = root/'decision.json'
        common._write(decision, {'status': 'approved_for_forward_development', 'training_only': True,
            'selected_config': config, 'reference_fields_used': False,
            'independent_diagnostics_used_for_selection': False,
            'scientific_recovery_supported': False,
            'probe_evidence': [{'path': str(evidence), 'sha256': common.digest_file(evidence)}]})
        return {'output_directory': root/'controlled', 'model_config': config,
            'probe_decision_path': decision, 'source_paths': [], 'optimized_campaign': prior,
            'protected_files': []}

    def prepare(self, arguments):
        with patch.object(study, '_preflight', side_effect=self.preflight):
            return study.prepare_step_control(**arguments)

    def raw(self, plan, case_id, *, reason='fixed_budget_exhausted', actual=None):
        config = plan['model_config']
        actual = config['adam_epochs'] if actual is None else actual
        state, history, transitions = asdict(StepControlState(config['learning_rate'])), [], []
        for step in range(1, actual+1):
            before = deepcopy(state)
            state['attempts'], state['trial_solves'] = step, step
            state['full_scale_successes'] = step
            if reason == 'consecutive_failure_stop':
                state.update(learning_rate=state['learning_rate']*.5, consecutive_failures=step,
                    moment_restarts=step, full_scale_successes=0)
                if step == actual:
                    state.update(stopped=True, stop_reason='consecutive exhausted searches')
            transition = {'step': step, 'before': before, 'after': deepcopy(state), 'transitions': []}
            transitions.append(transition)
            history.append({'step': step, 'accepted': reason != 'consecutive_failure_stop',
                'trials': [{}], 'trial_solve_count': 1,
                'proposed_hidden_displacement_finite': True, 'proposed_optimizer_state_finite': True,
                'controller_before': before, 'controller_after': deepcopy(state), 'controller_transitions': []})
        if reason == 'projection_budget':
            state.update(stopped=True, stop_reason='projection budget')
        expected = plan['preflight_by_case'][case_id]
        return {'config': deepcopy(config), 'requested_updates': config['adam_epochs'], 'completed_updates': actual,
            'hidden_update_history': history, 'controller_history': transitions, 'stop_reason': reason,
            'final_controller_state': state, 'trial_solve_count': actual,
            'additional_initial_final_solves': 2, 'total_projection_solve_calls': actual+2,
            'optimizer_instances': 1, 'optimizer_history': config['optimizer_history'],
            'initialization': deepcopy(expected['initialization']),
            'graded_point_metadata': deepcopy(expected['training_grid']),
            'training_points': {'interior': [[.1, .1, .1]]},
            'training_point_weights': {'interior': [1.]}}

    def callbacks(self, plan, output, *, failed=None, nonfinite=None, terminated=None, proof_mutation=None):
        calls = []
        def fit(observations, **kwargs):
            self.assertEqual(observations.temperature.tolist(), [300.])
            self.assertNotIn('truth', kwargs)
            model = SimpleNamespace(physics=kwargs['physics'], known_h=kwargs['known_h'])
            case_id = study.graded._case_for_model(model)['id']
            calls.append(case_id)
            if case_id == terminated:
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            if case_id == failed:
                raise common.ResourceLimit('constructed case-only cap')
            raw = self.raw(plan, case_id)
            if proof_mutation:
                proof_mutation(raw)
            kwargs['progress']({'stage': 'constructed terminal training state', 'step': len(raw['hidden_update_history'])})
            return model, raw
        def diagnose(model, points):
            case_id = study.graded._case_for_model(model)['id']
            self.assertTrue((output/case_id/'final_checkpoint.pt').is_file())
            self.assertTrue((output/case_id/'training-report.json').is_file())
            return {'normalized_rms': {family: 0. for family in common.expected_families(model.physics)}}
        def predict(model, coordinates):
            case_id = study.graded._case_for_model(model)['id']
            return np.full(len(coordinates), np.nan if case_id == nonfinite else 300.)
        return common.ForwardCallbacks(fit=fit, diagnose=diagnose, predict=predict,
            checkpoint=lambda model: b'constructed terminal checkpoint'), calls

    def test_preparation_requires_explicit_probe_and_all_frozen_grid_math(self):
        with self.temporary() as temporary, patch.object(study.optimized, 'validate_optimized', side_effect=self.context):
            args = self.fixture(Path(temporary))
            decision = study._read(args['probe_decision_path'])
            common._write(args['probe_decision_path'], dict(decision, training_only=False))
            with self.assertRaisesRegex(ValueError, 'training-only'):
                self.prepare(args)
            self.assertFalse(args['output_directory'].exists())
            old = asdict(OptimizedConfig(adam_epochs=320, learning_rate=.0064, max_backtracks=4))
            for key, value in (('learning_rate', .0032), ('width', 25), ('rank_rcond', 1e-6), ('interior_points', 4096)):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'frozen architecture'):
                    study._config(dict(args['model_config'], **{key: value}), old)
            with self.assertRaisesRegex(ValueError, '320 attempts'):
                study._config(dict(args['model_config'], adam_epochs=321), old)
            partial = dict(args['model_config'])
            partial.pop('controller_reduction')
            with self.assertRaisesRegex(ValueError, 'explicitly supplied'):
                study._config(partial, old)

    def test_four_once_only_workers_waves_and_complete_index(self):
        with self.temporary() as temporary, patch.object(study.optimized, 'validate_optimized', side_effect=self.context):
            args = self.fixture(Path(temporary))
            plan = self.prepare(args)
            self.assertEqual(plan['topology']['maximum_concurrent_workers'], 2)
            self.assertEqual(plan['topology']['seconds_per_worker'], 3000.)
            callbacks, calls = self.callbacks(plan, args['output_directory'])
            study.start_execution(args['output_directory'])
            with self.assertRaisesRegex(ValueError, 'second wave'):
                study.run_case(args['output_directory'], 't03', callbacks)
            for case in common.CASES:
                self.assertEqual(study.run_case(args['output_directory'], case['id'], callbacks)['status'], 'completed')
            with self.assertRaisesRegex(ValueError, 'already consumed'):
                study.run_case(args['output_directory'], 't01', callbacks)
            result = study.finalize_step_control(args['output_directory'])
            self.assertEqual(calls, ['t01', 't02', 't03', 't04'])
            self.assertEqual(result['forward_status'], 'passed')
            self.assertEqual(study.validate_step_control(args['output_directory'])['method_fit_count'], 4)
            with self.assertRaisesRegex(ValueError, 'already finalized'):
                study.finalize_step_control(args['output_directory'])
            (args['output_directory']/'t04/unindexed.txt').write_text('tampering')
            with self.assertRaisesRegex(ValueError, 'completely indexed'):
                study.validate_step_control(args['output_directory'])

    def test_resource_signal_failure_and_nonfinite_prediction_do_not_remove_other_opportunities(self):
        with self.temporary() as temporary, patch.object(study.optimized, 'validate_optimized', side_effect=self.context):
            args = self.fixture(Path(temporary))
            plan = self.prepare(args)
            callbacks, calls = self.callbacks(plan, args['output_directory'], terminated='t01', nonfinite='t02')
            study.start_execution(args['output_directory'])
            for case in common.CASES:
                study.run_case(args['output_directory'], case['id'], callbacks)
            result = study.finalize_step_control(args['output_directory'])
            self.assertEqual([row['status'] for row in result['cases']], ['failed', 'failed', 'completed', 'completed'])
            self.assertEqual(len(calls), 4)
            self.assertIn('terminated', result['cases'][0]['error'])
            self.assertTrue((args['output_directory']/'t02/final_checkpoint.pt').is_file())
            self.assertTrue((args['output_directory']/'t02/training-report.json').is_file())
            self.assertNotIn('gate', result['cases'][1])
            study.validate_step_control(args['output_directory'])

    def test_observed_child_exit_and_never_run_cases_are_retained_without_retry(self):
        with self.temporary() as temporary, patch.object(study.optimized, 'validate_optimized', side_effect=self.context):
            args = self.fixture(Path(temporary))
            plan = self.prepare(args)
            callbacks, calls = self.callbacks(plan, args['output_directory'])
            study.start_execution(args['output_directory'])
            first = study.record_worker_exit(args['output_directory'], 't01', -9, 'parent observed child exit')
            self.assertEqual(study.record_worker_exit(args['output_directory'], 't01', -9, 'same exit'), first)
            study.run_case(args['output_directory'], 't02', callbacks)
            study.run_case(args['output_directory'], 't03', callbacks)
            study.mark_unrun(args['output_directory'], 't04', 'aggregate deadline reached')
            with self.assertRaisesRegex(ValueError, 'already consumed'):
                study.run_case(args['output_directory'], 't01', callbacks)
            result = study.finalize_step_control(args['output_directory'])
            self.assertEqual(calls, ['t02', 't03'])
            self.assertEqual([row['status'] for row in result['cases']], ['failed', 'completed', 'completed', 'not_run'])
            self.assertEqual(result['cases'][-1]['reason'], 'aggregate deadline reached')
            study.validate_step_control(args['output_directory'])

    def test_declared_training_stops_allow_shorter_runs_and_bind_terminal_state(self):
        with self.temporary() as temporary, patch.object(study.optimized, 'validate_optimized', side_effect=self.context):
            args = self.fixture(Path(temporary))
            plan = self.prepare(args)
            raw = self.raw(plan, 't01', reason='consecutive_failure_stop', actual=3)
            self.assertEqual(study._training_proof(raw, plan, 't01')['actual_updates'], 3)
            bad = deepcopy(raw)
            bad['stop_reason'] = 'fixed_budget_exhausted'
            with self.assertRaisesRegex(ValueError, 'training-only stop'):
                study._training_proof(bad, plan, 't01')
            bad = deepcopy(raw)
            bad['final_controller_state']['trial_solves'] = 999
            with self.assertRaisesRegex(ValueError, 'terminal state'):
                study._training_proof(bad, plan, 't01')
            for key in ('training_points', 'training_point_weights'):
                bad = deepcopy(raw)
                bad[key]['interior'][0] = [.2, .1, .1] if key == 'training_points' else .5
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'coordinates|row weights'):
                    study._training_proof(bad, plan, 't01')
            bad = deepcopy(raw)
            bad['controller_history'][1]['before']['learning_rate'] = .0064
            with self.assertRaisesRegex(ValueError, 'transition chain'):
                study._training_proof(bad, plan, 't01')

    def test_projection_budget_may_stop_between_attempts_with_a_partial_reservation(self):
        with self.temporary() as temporary, patch.object(study.optimized, 'validate_optimized', side_effect=self.context):
            args = self.fixture(Path(temporary), maximum_trial_solves=5)
            plan = self.prepare(args)
            raw = self.raw(plan, 't01', reason='projection_budget', actual=1)
            self.assertTrue(study._training_proof(raw, plan, 't01')['verified'])
            raw['trial_solve_count'] = 0
            raw['total_projection_solve_calls'] = 2
            with self.assertRaisesRegex(ValueError, 'projection budget'):
                study._training_proof(raw, plan, 't01')

    def test_changed_prior_evidence_and_probe_block_execution_before_fit(self):
        with self.temporary() as temporary, patch.object(study.optimized, 'validate_optimized', side_effect=self.context):
            args = self.fixture(Path(temporary))
            self.prepare(args)
            (args['optimized_campaign']/'retained-failure.txt').write_text('changed')
            with self.assertRaisesRegex(RuntimeError, 'complete V6 attempt'):
                study.start_execution(args['output_directory'])
        with self.temporary() as temporary, patch.object(study.optimized, 'validate_optimized', side_effect=self.context):
            args = self.fixture(Path(temporary))
            self.prepare(args)
            (Path(temporary)/'probe.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'probe evidence identity'):
                study.start_execution(args['output_directory'])

    def test_checkpoint_and_training_proof_precede_every_independent_read(self):
        with self.temporary() as temporary, patch.object(study.optimized, 'validate_optimized', side_effect=self.context):
            args = self.fixture(Path(temporary))
            plan = self.prepare(args)
            callbacks, _ = self.callbacks(plan, args['output_directory'], proof_mutation=lambda raw: raw.__setitem__('completed_updates', 999))
            study.start_execution(args['output_directory'])
            with patch.object(common, 'paired_scoring_field', side_effect=AssertionError('hidden field read before proof')), \
                 patch.object(common, 'independent_points', side_effect=AssertionError('diagnostics read before proof')):
                row = study.run_case(args['output_directory'], 't01', callbacks)
            self.assertEqual(row['status'], 'failed')
            self.assertIn('update and transition counts', row['error'])
            self.assertTrue((args['output_directory']/'t01/final_checkpoint.pt').is_file())
            self.assertTrue((args['output_directory']/'t01/training-report.json').is_file())

    def test_aggregate_and_worker_resource_guards_preclude_fitting_after_deadline(self):
        with self.temporary() as temporary, patch.object(study.optimized, 'validate_optimized', side_effect=self.context):
            args = self.fixture(Path(temporary))
            plan = self.prepare(args)
            callbacks, calls = self.callbacks(plan, args['output_directory'])
            study.start_execution(args['output_directory'])
            execution = study._read(args['output_directory']/'execution.json')
            execution['started_monotonic'] -= study.TOP_SECONDS+1
            common._write(args['output_directory']/'execution.json', execution)
            self.assertEqual(study.run_case(args['output_directory'], 't01', callbacks)['status'], 'failed')
            self.assertEqual(calls, [])
        with self.temporary() as temporary, patch.object(study.optimized, 'validate_optimized', side_effect=self.context):
            args = self.fixture(Path(temporary))
            plan = self.prepare(args)
            callbacks, calls = self.callbacks(plan, args['output_directory'])
            study.start_execution(args['output_directory'])
            with patch.object(common, '_rss_bytes', return_value=study.RSS_BYTES+1):
                self.assertEqual(study.run_case(args['output_directory'], 't01', callbacks)['status'], 'failed')
            self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
