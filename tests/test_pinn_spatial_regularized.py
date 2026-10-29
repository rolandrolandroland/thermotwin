"""Constructed lifecycle checks for the additive continuous-readout study."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from thermotwin.pinn.planar_contact_regularized import RegularizedConfig
from thermotwin.pinn.planar_contact_step_control import StepControlConfig
from thermotwin.studies import pinn_spatial_continuity as common
from thermotwin.studies import pinn_spatial_regularized as study
from tests import test_pinn_spatial_step_control as fixtures


class RegularizedStudyTests(unittest.TestCase):
    def temporary(self):
        return tempfile.TemporaryDirectory(dir=common.ROOT/'reports', prefix='constructed-regularized-study-')

    def context(self, directory):
        return {'directory': str(Path(directory).resolve()), 'constructed': True}

    def calibration(self, config, grid):
        scales = np.array([1., 2.])
        value = {'schema': 'fixed_initial_column_scale_ridge_v1', 'column_scale': scales.tolist(),
            'column_scale_sha256': study.graded._model_api()._array_digest(scales),
            'fixed_lambda': 2e-7, 'relative_scale': 1e-7, 'initial_sigma_max': 2.,
            'initial_matrix_shape': [3, 2], 'initial_matrix_sha256': 'constructed-A0',
            'initial_rhs_sha256': 'constructed-b0', 'calibration_matrix_builds': 1,
            'calibration_spectral_calculations': 1, 'frozen_for_entire_fit': True,
            'hidden_parameters_differentiated_through_calibration': False,
            'bindings': {'config': deepcopy(config), 'points': {
                key: entry['sha256'] for key, entry in grid['point_identities'].items()}}}
        value['sha256'] = common.canonical_digest(value)
        return value

    def preflight(self, config, prior, progress=None):
        records = fixtures.StepControlStudyTests().preflight(config, None)
        for value in records.values():
            calibration = self.calibration(config, value['training_grid'])
            value.update(readout_calibration=calibration, calibration_sha256=common.canonical_digest(calibration))
        if progress:
            for case_id, value in records.items():
                progress(case_id, None)
                progress(case_id, value)
        return records

    def fixture(self, root):
        config = asdict(RegularizedConfig())
        prior = root/'prior-v7'
        prior.mkdir()
        cutoff = root/'closed-cutoff'
        cutoff.mkdir()
        common._write(cutoff/'plan.json', {'source_sha256': {}, 'protected_trees': {}})
        bindings = {}
        for case in common.CASES:
            directory = root/'paired'/case['id']
            directory.mkdir(parents=True)
            np.savez(directory/'observations.npz', coordinates=[[.02, .003, 10.]], temperature=[300.], clean_temperature=[-99999.])
            np.savez(directory/'field_scoring.npz', coordinates=[[.02, .002, 20.]], truth=[300.], prediction=[-99999.])
            bindings[case['id']] = {name: {'path': str(directory/name), 'sha256': common.digest_file(directory/name)}
                for name in ('observations.npz', 'field_scoring.npz')}
        old = asdict(StepControlConfig(adam_epochs=320, learning_rate=.0064, controller_max_failures=8))
        common._write(prior/'plan.json', {'model_config': old, 'paired_evidence': bindings,
            'source_sha256': {}, 'preflight_by_case': self.preflight(config, None)})
        (prior/'immutable').write_text('closed V7/V6 proof')
        probe = root/'gradient-probe'
        probe.mkdir()
        common._write(probe/'plan.json', {'declared_points': 14, 'calibrations': 2})
        common._write(probe/'result.json', {'method_checks_passed': True, 'retained_points': 14, 'calibrations': 2})
        validator = root/'gradient-proof.py'
        validator.write_text('''from pathlib import Path
import hashlib,json
def verify_gradient_probe(directory):
    directory=Path(directory).resolve()
    result=json.loads((directory/'result.json').read_text())
    if result['retained_points'] != 14 or result['calibrations'] != 2:
        raise RuntimeError('incomplete constructed gradient proof')
    return {'directory':str(directory),'plan_sha256':hashlib.sha256((directory/'plan.json').read_bytes()).hexdigest(),
        'result_sha256':hashlib.sha256((directory/'result.json').read_bytes()).hexdigest(),
        'method_checks_passed':result['method_checks_passed']}
''')
        decision = {'schema': 'regularized_training_only_method_decision_v1',
            'status': 'approved_for_forward_development', 'method_checks_passed': True, 'training_only': True,
            'reference_fields_used': False, 'independent_diagnostics_used_for_selection': False,
            'selected_config': config, 'gradient_probe': {'directory': str(probe),
                'plan_sha256': common.digest_file(probe/'plan.json'), 'result_sha256': common.digest_file(probe/'result.json'),
                'method_checks_passed': True},
            'followup': {'cases': list(common.CASES), 'maximum_attempts_per_case': 320,
                'maximum_trial_solves_per_case': 480, 'maximum_backtracks': 4,
                'acceptance_thresholds': common.GATES, 'parallel_workers': 2, 'worker_threads': 1,
                'case_wall_seconds': 3000., 'case_rss_bytes': 2*1024**3,
                'supervisor_wall_seconds': 7200., 'automatic_retries': 0},
            'no_guarantee_of_forward_gate_success': True, 'no_further_method_search': True}
        decision_path = root/'decision.json'
        common._write(decision_path, decision)
        return {'output_directory': root/'regularized', 'model_config': config,
            'method_decision_path': decision_path, 'gradient_probe_validator_path': validator,
            'step_control_campaign': prior, 'cutoff_audit_directory': cutoff,
            'source_paths': [], 'protected_files': []}

    def prepare(self, arguments):
        with patch.object(study, '_preflight', side_effect=self.preflight):
            return study.prepare_regularized(**arguments)

    def raw(self, plan, case_id):
        raw = fixtures.StepControlStudyTests().raw(plan, case_id,
            reason='consecutive_failure_stop', actual=plan['model_config']['controller_max_failures'])
        calibration = plan['preflight_by_case'][case_id]['readout_calibration']
        projection = {'head_policy': 'unconditional unique stationary ridge readout',
            'hard_singular_value_truncation': False, 'rank_rcond': None, 'rank': 2, 'columns': 2,
            'head_solve_calls': 1, 'fixed_lambda': calibration['fixed_lambda'],
            'column_scale_sha256': calibration['column_scale_sha256'],
            'stationarity_tolerance': plan['model_config']['head_stationarity_tolerance'], 'stationarity_relative': 1e-12,
            'objective_before': 3., 'objective_after': 2., 'objective_proposed': 2.,
            'unpenalized_objective': 1.5, 'regularization_penalty': .5, 'algebraic_objective': 2.}
        for event in raw['hidden_update_history']:
            event.update(objective_before=2., objective_after=2.)
            for trial in event['trials']:
                trial.update(head_solve_attempted=True, accepted=False, objective=None)
        raw.update(readout_calibration=deepcopy(calibration), initial_calibration_matrix_builds=0,
            initial_calibration_spectral_calculations=0, final_objective=2., final_unpenalized_objective=1.5,
            final_regularization_penalty=.5, final_head_stationarity=1e-12,
            head_stationarity_required=True, hard_singular_value_truncation=False,
            hidden_truth_labels_used=False, independent_diagnostics_used_for_training=False,
            gradient_semantics=study.GRADIENT_SEMANTICS,
            projections=[dict(projection, stage='initial_projection'), dict(projection, stage='final_projection')])
        return raw

    def callbacks(self, plan, output, *, fail=None, bad_field=None, mutate_raw=None):
        calls = []
        def fit(observations, **kwargs):
            self.assertEqual(observations.temperature.tolist(), [300.])
            model = SimpleNamespace(physics=kwargs['physics'], known_h=kwargs['known_h'])
            case_id = study.graded._case_for_model(model)['id']
            calls.append(case_id)
            self.assertEqual(kwargs['readout_calibration'], plan['preflight_by_case'][case_id]['readout_calibration'])
            if case_id == fail:
                raise common.ResourceLimit('constructed worker cap')
            raw = self.raw(plan, case_id)
            if mutate_raw:
                mutate_raw(raw)
            kwargs['progress']({'stage': 'constructed regularized endpoint'})
            return model, raw
        def diagnose(model, points):
            case = study.graded._case_for_model(model)['id']
            self.assertTrue((output/case/'final_checkpoint.pt').is_file())
            self.assertTrue((output/case/'training-report.json').is_file())
            return {'normalized_rms': {key: 0. for key in common.expected_families(model.physics)}}
        def predict(model, coordinates):
            case = study.graded._case_for_model(model)['id']
            return np.full(len(coordinates), 300.2 if case == bad_field else 300.)
        return common.ForwardCallbacks(fit=fit, predict=predict, diagnose=diagnose,
            checkpoint=lambda model: b'constructed calibrated terminal state'), calls

    def test_method_checks_must_be_verified_before_any_forward_preparation(self):
        with self.temporary() as temporary, patch.object(study.previous, 'validate_step_control', side_effect=self.context), \
             patch.object(study, '_cutoff_context', side_effect=self.context):
            args = self.fixture(Path(temporary))
            probe = Path(study._read(args['method_decision_path'])['gradient_probe']['directory'])
            common._write(probe/'result.json', {'method_checks_passed': True, 'retained_points': 13, 'calibrations': 2})
            with self.assertRaisesRegex(RuntimeError, 'incomplete constructed gradient proof'):
                self.prepare(args)
            self.assertFalse(args['output_directory'].exists())

    def test_lifecycle_retains_four_fresh_fits_and_separates_method_from_forward_gates(self):
        with self.temporary() as temporary, patch.object(study.previous, 'validate_step_control', side_effect=self.context), \
             patch.object(study, '_cutoff_context', side_effect=self.context):
            args = self.fixture(Path(temporary))
            plan = self.prepare(args)
            self.assertEqual(plan['calibration_accounting']['preparation_matrix_builds'], 4)
            self.assertEqual(plan['calibration_accounting']['worker_matrix_builds'], 0)
            callbacks, calls = self.callbacks(plan, args['output_directory'], bad_field='t02')
            study.start_execution(args['output_directory'])
            with self.assertRaisesRegex(ValueError, 'second wave'):
                study.run_case(args['output_directory'], 't03', callbacks)
            for case in common.CASES:
                self.assertEqual(study.run_case(args['output_directory'], case['id'], callbacks)['status'], 'completed')
            result = study.finalize_regularized(args['output_directory'])
            self.assertEqual(calls, ['t01', 't02', 't03', 't04'])
            self.assertTrue(result['initial_method_checks_passed'])
            self.assertEqual(result['forward_status'], 'blocked_forward_gate')
            self.assertEqual(sum(row['gate']['all'] for row in result['cases']), 3)
            self.assertEqual(study.validate_regularized(args['output_directory'])['method_fit_count'], 4)
            with self.assertRaises(ValueError):
                study.run_case(args['output_directory'], 't01', callbacks)

    def test_worker_failure_and_observed_exit_do_not_remove_later_opportunities(self):
        with self.temporary() as temporary, patch.object(study.previous, 'validate_step_control', side_effect=self.context), \
             patch.object(study, '_cutoff_context', side_effect=self.context):
            args = self.fixture(Path(temporary))
            plan = self.prepare(args)
            callbacks, calls = self.callbacks(plan, args['output_directory'], fail='t01')
            study.start_execution(args['output_directory'])
            study.run_case(args['output_directory'], 't01', callbacks)
            study.record_worker_exit(args['output_directory'], 't02', -9, 'observed abnormal child exit')
            study.run_case(args['output_directory'], 't03', callbacks)
            study.mark_unrun(args['output_directory'], 't04', 'aggregate cap before launch')
            result = study.finalize_regularized(args['output_directory'])
            self.assertEqual([row['status'] for row in result['cases']], ['failed', 'failed', 'completed', 'not_run'])
            self.assertEqual(calls, ['t01', 't03'])
            self.assertTrue(result['initial_method_checks_passed'])
            study.validate_regularized(args['output_directory'])

    def test_frozen_calibration_objective_stationarity_and_no_hard_cutoff_are_verified(self):
        with self.temporary() as temporary, patch.object(study.previous, 'validate_step_control', side_effect=self.context), \
             patch.object(study, '_cutoff_context', side_effect=self.context):
            args = self.fixture(Path(temporary))
            plan = self.prepare(args)
            raw = self.raw(plan, 't01')
            self.assertTrue(study._training_proof(raw, plan, 't01')['unique_stationary_readout_verified'])
            mutations = (
                lambda value: value['readout_calibration'].__setitem__('fixed_lambda', 1.),
                lambda value: value.__setitem__('initial_calibration_spectral_calculations', 1),
                lambda value: value.__setitem__('final_objective', 1.5),
                lambda value: value.__setitem__('final_head_stationarity', 1e-3),
                lambda value: value['projections'][0].__setitem__('hard_singular_value_truncation', True),
                lambda value: value['training_points']['interior'][0].__setitem__(1, .9),
            )
            for index, mutate in enumerate(mutations):
                changed = deepcopy(raw)
                mutate(changed)
                with self.subTest(mutation=index), self.assertRaises(ValueError):
                    study._training_proof(changed, plan, 't01')

    def test_invalid_method_report_is_saved_before_diagnostics_and_never_reads_truth(self):
        with self.temporary() as temporary, patch.object(study.previous, 'validate_step_control', side_effect=self.context), \
             patch.object(study, '_cutoff_context', side_effect=self.context):
            args = self.fixture(Path(temporary))
            plan = self.prepare(args)
            callbacks, _ = self.callbacks(plan, args['output_directory'], mutate_raw=lambda raw: raw.__setitem__('final_objective', 99.))
            study.start_execution(args['output_directory'])
            with patch.object(common, 'independent_points', side_effect=AssertionError('diagnostics before method proof')), \
                 patch.object(common, 'paired_scoring_field', side_effect=AssertionError('truth before method proof')):
                row = study.run_case(args['output_directory'], 't01', callbacks)
            self.assertEqual(row['status'], 'failed')
            self.assertIn('training objective', row['error'])
            self.assertTrue((args['output_directory']/'t01/final_checkpoint.pt').is_file())
            self.assertTrue((args['output_directory']/'t01/training-report.json').is_file())

    def test_probe_tamper_and_original_tree_changes_block_fitting(self):
        with self.temporary() as temporary, patch.object(study.previous, 'validate_step_control', side_effect=self.context), \
             patch.object(study, '_cutoff_context', side_effect=self.context):
            args = self.fixture(Path(temporary))
            self.prepare(args)
            (args['step_control_campaign']/'immutable').write_text('changed')
            with self.assertRaisesRegex(RuntimeError, 'older experiment chain changed'):
                study.start_execution(args['output_directory'])
        with self.temporary() as temporary, patch.object(study.previous, 'validate_step_control', side_effect=self.context), \
             patch.object(study, '_cutoff_context', side_effect=self.context):
            args = self.fixture(Path(temporary))
            self.prepare(args)
            probe = Path(study._read(args['method_decision_path'])['gradient_probe']['directory'])
            common._write(probe/'result.json', {'method_checks_passed': False, 'retained_points': 14, 'calibrations': 2})
            with self.assertRaisesRegex(ValueError, 'gradient-probe proof differs'):
                study.start_execution(args['output_directory'])

    def test_configuration_cannot_change_geometry_cost_budget_or_old_controller(self):
        old = asdict(StepControlConfig(adam_epochs=320, learning_rate=.0064, controller_max_failures=8))
        config = asdict(RegularizedConfig())
        for key, value in (('adam_epochs', 80), ('maximum_trial_solves', 400), ('max_backtracks', 3),
                           ('width', 25), ('physics_weight', 50.), ('controller_max_failures', 3)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'requires 320|frozen architecture'):
                study._config(dict(config, **{key: value}), old)
        incomplete = dict(config)
        incomplete.pop('head_stationarity_tolerance')
        with self.assertRaisesRegex(ValueError, 'must be explicit'):
            study._config(incomplete, old)

    def test_preflight_calibrates_each_fresh_initial_field_once_from_noisy_training_data_only(self):
        with self.temporary() as temporary:
            args = self.fixture(Path(temporary))
            config = args['model_config']
            prior = study._read(args['step_control_campaign']/'plan.json')
            declared = self.preflight(config, None)
            grid = declared['t01']['training_grid']
            made, calibrated = [], []
            def matched(physics, config, known_h, seed):
                model = SimpleNamespace(physics=physics, known_h=known_h)
                made.append(model)
                return model, deepcopy(declared['t01']['initialization'])
            def calibrate(model, points, observations):
                self.assertEqual(observations.temperature.tolist(), [300.])
                self.assertEqual(list(points), ['interior'])
                calibrated.append(model)
                return self.calibration(config, grid)
            array_digest = study.graded._model_api()._array_digest
            api = SimpleNamespace(RegularizedConfig=RegularizedConfig,
                matched_regularized_model=matched, calibrate_readout=calibrate)
            grid_api = SimpleNamespace(graded_point_metadata=lambda *args: deepcopy(grid),
                graded_training_points=lambda *args: {'interior': np.array([[.1, .1, .1]])},
                _array_digest=array_digest)
            with patch.object(study, '_api', return_value=api), patch.object(study.graded, '_model_api', return_value=grid_api), \
                 patch.object(common, 'paired_scoring_field', side_effect=AssertionError('preflight may not read truth')):
                records = study._preflight(config, prior)
            self.assertEqual(len(made), 4)
            self.assertEqual(calibrated, made)
            self.assertEqual(len({id(value) for value in made}), 4)
            self.assertEqual(set(records), {'t01', 't02', 't03', 't04'})

    def test_partial_preparation_is_retained_and_cannot_recalibrate_on_retry(self):
        with self.temporary() as temporary, patch.object(study.previous, 'validate_step_control', side_effect=self.context), \
             patch.object(study, '_cutoff_context', side_effect=self.context):
            args = self.fixture(Path(temporary))
            def interrupted(config, prior, progress):
                record = self.preflight(config, prior)['t01']
                progress('t01', None)
                progress('t01', record)
                raise common.ResourceLimit('constructed calibration preparation cap')
            with patch.object(study, '_preflight', side_effect=interrupted):
                with self.assertRaisesRegex(common.ResourceLimit, 'preparation cap'):
                    study.prepare_regularized(**args)
            saved = study._read(args['output_directory']/'calibration-state.json')
            self.assertEqual(saved['status'], 'failed')
            self.assertEqual(saved['attempted_cases'], ['t01'])
            self.assertEqual(saved['completed_cases'], ['t01'])
            self.assertFalse((args['output_directory']/'plan.json').exists())
            with patch.object(study, '_preflight') as preflight:
                with self.assertRaisesRegex(FileExistsError, 'once-only'):
                    study.prepare_regularized(**args)
                preflight.assert_not_called()

    def test_inherited_and_other_sealed_trees_cannot_receive_preparation_outputs(self):
        with self.temporary() as temporary, patch.object(study.previous, 'validate_step_control', side_effect=self.context), \
             patch.object(study, '_cutoff_context', side_effect=self.context):
            args = self.fixture(Path(temporary))
            inherited = Path(temporary)/'old-v3'
            inherited.mkdir()
            common._write(args['cutoff_audit_directory']/'plan.json', {'source_sha256': {}, 'protected_trees': {str(inherited): {}}})
            args['output_directory'] = inherited/'new-child'
            with patch.object(study, '_preflight') as preflight:
                with self.assertRaisesRegex(ValueError, 'inherited protected'):
                    study.prepare_regularized(**args)
                preflight.assert_not_called()
            self.assertFalse(args['output_directory'].exists())
            sealed = Path(temporary)/'other-sealed-report'
            sealed.mkdir()
            common._write(sealed/'report-index.json', {'constructed': True})
            args['output_directory'] = sealed/'new-child'
            with patch.object(study, '_preflight') as preflight:
                with self.assertRaisesRegex(ValueError, 'sealed ancestor'):
                    study.prepare_regularized(**args)
                preflight.assert_not_called()
            self.assertFalse(args['output_directory'].exists())

    def test_scalar_objectives_must_match_the_full_stationary_readout_history(self):
        with self.temporary() as temporary, patch.object(study.previous, 'validate_step_control', side_effect=self.context), \
             patch.object(study, '_cutoff_context', side_effect=self.context):
            args = self.fixture(Path(temporary))
            plan = self.prepare(args)
            original = self.raw(plan, 't01')
            mutations = (
                lambda value: value.update(final_objective=3., final_unpenalized_objective=2.5),
                lambda value: value['projections'][0].__setitem__('algebraic_objective', 4.),
                lambda value: value['hidden_update_history'][0].__setitem__('objective_before', 3.),
                lambda value: value['hidden_update_history'][0].__setitem__('objective_after', 1.),
            )
            for index, mutate in enumerate(mutations):
                changed = deepcopy(original)
                mutate(changed)
                with self.subTest(mutation=index), self.assertRaisesRegex(ValueError, 'objective'):
                    study._training_proof(changed, plan, 't01')


if __name__ == '__main__':
    unittest.main()
