"""Paired execution tests with constructed arrays and mocked numerical fits."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from thermotwin.studies import pinn_spatial_graded as study
from thermotwin.studies import pinn_spatial_continuity as common


class GradedStudyTests(unittest.TestCase):
    def preflight(self, configs):
        records = {}
        for case in common.CASES:
            on = {'arm': 'on', 'common_weights_sha256': 'common-'+case['id'],
                  'shared_function_sha256': 'function-'+case['id'], 'complete_state_sha256': 'on-'+case['id'],
                  'log_column_zero': True, 'log_column_present': True,
                  'on_only_log_column_initially_zero': True, 'probe_agreement': {'passed': True}}
            off = dict(on, arm='off', complete_state_sha256='off-'+case['id'],
                       log_column_zero=None, log_column_present=False)
            grid = {'point_digest': 'grid-'+case['id'], 'point_counts': {'interior': 8192},
                    'strata': [{'name': 'bulk', 'count': 4096, 'mean_weight': .5}]}
            records[case['id']] = {'initialization': {'on': on, 'off': off,
                'probe_agreement': {'passed': True}, 'probe_coordinates_normalized': [[.0731, .0173, .0047]]},
                'training_grid': {arm: deepcopy(grid) for arm in study.ARMS}}
        return records

    def old_campaign(self, directory):
        return {'source_sha256': {}}, {'directory': str(directory), 'constructed': True}

    def prepare(self, root):
        paired = root/'paired-v2'/'two-d'
        for case in common.CASES:
            directory = paired/case['id'] if case['id'] in ('t01', 't02') else paired/'resource-continuation'/case['id']
            directory.mkdir(parents=True)
            np.savez(directory/'observations.npz', coordinates=np.array([[.02, .003, 10.]]),
                temperature=np.array([300.]), clean_temperature=np.array([-99999.]))
            np.savez(directory/'field_scoring.npz', coordinates=np.array([[.02, .002, 20.]]),
                truth=np.array([300.]), prediction=np.array([-99999.]))
        protected = []
        for name in ('old-v3', 'old-v4'):
            directory = root/name
            directory.mkdir()
            (directory/'immutable').write_text('preserved')
            protected.append(directory)
        output = root/'comparison'
        with patch.object(study, '_preflight', side_effect=self.preflight):
            plan = study.prepare_comparison(output, source_paths=[], paired_v2_directory=paired,
                v2_development_record=None, preserved_campaign=None,
                protected_campaigns=protected, protected_files=[])
        return output, plan

    def callbacks(self, plan, arm, *, nonfinite_case=None, bad_pair=False):
        calls = []
        def fit(observations, **kwargs):
            self.assertTrue(np.all(observations.temperature == 300.))
            self.assertEqual(set(kwargs), {'physics', 'config', 'known_h', 'neural_seed', 'sampling_seed', 'progress'})
            model = SimpleNamespace(physics=kwargs['physics'], known_h=kwargs['known_h'])
            case = study._case_for_model(model)
            calls.append(case['id'])
            frozen = plan['preflight_by_case'][case['id']]
            initialization = deepcopy(frozen['initialization']['on' if arm == 'log_on' else 'off'])
            if bad_pair:
                initialization['common_weights_sha256'] = 'wrong common initial state'
            kwargs['progress']({'stage': 'constructed fit', 'step': 80})
            return model, {'config': kwargs['config'], 'initialization': initialization,
                'graded_point_metadata': frozen['training_grid'][arm],
                'hidden_update_history': [{'accepted': False} for _ in range(80)]}
        def predict(model, coordinates):
            bad = study._case_for_model(model)['id'] == nonfinite_case
            return np.full(len(coordinates), np.nan if bad else 300.)
        def diagnose(model, points):
            return {'normalized_rms': {name: 0. for name in common.expected_families(model.physics)}}
        return common.ForwardCallbacks(fit=fit, predict=predict, diagnose=diagnose,
            checkpoint=lambda model: b'constructed checkpoint'), calls

    def test_both_complete_subplans_and_pairing_proofs_exist_before_any_fit(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(study, '_validate_old_campaign', side_effect=self.old_campaign):
            output, plan = self.prepare(Path(temporary))
            self.assertEqual(plan['maximum_method_fits'], 8)
            self.assertEqual(plan['reused_paired_datasets'], 4)
            self.assertEqual(plan['topology']['aggregate_child_threads'], 2)
            self.assertEqual(plan['topology']['aggregate_child_rss_limit_bytes'], 4*1024**3)
            for arm in study.ARMS:
                self.assertEqual({path.name for path in (output/arm).iterdir()}, {'plan.json', 'state.json'})
                self.assertEqual(common.digest_file(output/arm/'plan.json'), plan['subplans'][arm]['plan_sha256'])
            self.assertEqual(plan['preflight_by_case']['t01']['training_grid']['log_on'],
                             plan['preflight_by_case']['t01']['training_grid']['log_off'])
            with self.assertRaisesRegex(ValueError, 'has not completed'):
                study.finalize_comparison(output)

    def test_eight_mocked_fits_are_retained_without_winner_and_cannot_be_repeated(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(study, '_validate_old_campaign', side_effect=self.old_campaign):
            output, plan = self.prepare(Path(temporary))
            all_calls = []
            for arm in study.ARMS:
                callbacks, calls = self.callbacks(plan, arm)
                result = study.run_arm(output, arm, callbacks)
                self.assertEqual(result['status'], 'passed')
                self.assertEqual(calls, ['t01', 't02', 't03', 't04'])
                all_calls.extend((arm, case) for case in calls)
                with self.assertRaisesRegex(ValueError, 'consumed'):
                    study.run_arm(output, arm, callbacks)
            result = study.finalize_comparison(output)
            self.assertEqual(len(all_calls), 8)
            self.assertEqual(len(result['paired_case_rows']), 4)
            self.assertEqual(result['method_fit_count'], 8)
            self.assertTrue(result['no_winner_selected'])
            self.assertFalse(result['fresh_evaluation_opened'])
            self.assertEqual(study.validate_comparison(output)['method_fit_count'], 8)
            with self.assertRaisesRegex(ValueError, 'finalized'):
                study.finalize_comparison(output)
            with self.assertRaisesRegex(ValueError, 'finalized'):
                study.run_arm(output, 'log_on')

    def test_nonfinite_prediction_retains_failure_and_other_log_arm(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(study, '_validate_old_campaign', side_effect=self.old_campaign):
            output, plan = self.prepare(Path(temporary))
            for arm in study.ARMS:
                callbacks, calls = self.callbacks(plan, arm, nonfinite_case='t02' if arm == 'log_on' else None)
                study.run_arm(output, arm, callbacks)
                self.assertEqual(len(calls), 4)
            result = study.finalize_comparison(output)
            row = result['paired_case_rows'][1]
            self.assertEqual(row['methods']['log_on']['status'], 'failed')
            self.assertEqual(row['methods']['log_off']['status'], 'completed')
            self.assertEqual(result['arms']['log_on']['scientific_status'], 'blocked_forward_gate')
            self.assertEqual(result['arms']['log_off']['scientific_status'], 'passed')
            self.assertTrue((output/'log_on/t02/final_checkpoint.pt').exists())

    def test_executed_pairing_metadata_cannot_be_substituted(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(study, '_validate_old_campaign', side_effect=self.old_campaign):
            output, plan = self.prepare(Path(temporary))
            for arm in study.ARMS:
                callbacks, _ = self.callbacks(plan, arm, bad_pair=arm == 'log_on')
                study.run_arm(output, arm, callbacks)
            with self.assertRaisesRegex(ValueError, 'initialization or graded points'):
                study.finalize_comparison(output)

    def test_source_and_topology_changes_block_before_fitting(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(study, '_validate_old_campaign', side_effect=self.old_campaign):
            output, plan = self.prepare(Path(temporary))
            changed = deepcopy(plan)
            changed['topology']['aggregate_child_threads'] = 8
            with self.assertRaisesRegex(ValueError, 'topology'):
                study.verify_comparison_plan(changed)
            callbacks, calls = self.callbacks(plan, 'log_on')
            with patch.object(common, '_source_records', return_value={'constructed': 'changed'}):
                with self.assertRaisesRegex(RuntimeError, 'source changed'):
                    study.run_arm(output, 'log_on', callbacks)
            self.assertEqual(calls, [])
            for overrides in ({'adam_epochs': 79}, {'interior_points': 4096}, {'bottom_points': 256}):
                with self.assertRaisesRegex(ValueError, '80 updates'):
                    study._configs(overrides)

    def test_whole_plate_bottom_audit_follows_raw_checkpoint_and_records_height_strata(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            def diagnose(model, points):
                values = points.get('interior', np.zeros((1, 3)))
                return {'normalized_rms': {'interior': float(np.mean(values[:, 1]))},
                        'physical_rms': {'interior': float(np.mean(values[:, 1]))*5/120}}
            api = SimpleNamespace(GradedConfig=lambda **kwargs: kwargs, fit_graded=None,
                predict_graded=None, diagnose_graded=diagnose, checkpoint_bytes=None)
            with patch.object(study, '_model_api', return_value=api):
                callbacks = study.graded_callbacks(directory)
            for case in common.CASES[:2]:
                model = SimpleNamespace(physics=common.physics_for(case), known_h=case['known_h'])
                with self.assertRaisesRegex(RuntimeError, 'requires retained'):
                    callbacks.diagnose(model, {'top': np.zeros((1, 3))})
                case_directory = directory/case['id']
                case_directory.mkdir()
                (case_directory/'final_checkpoint.pt').write_bytes(b'constructed')
                (case_directory/'training-report.json').write_text('{}')
                callbacks.diagnose(model, {'top': np.zeros((1, 3))})
                audit = study._read(case_directory/'near-bottom-audit.json')
                self.assertEqual(audit['status'], 'completed')
                self.assertEqual(audit['role'], 'report only')
                self.assertEqual(len(audit['by_height']), 6)
                self.assertEqual(sum(row['point_count'] for row in audit['by_height']), 1152)
                self.assertEqual(audit['point_digest'], common.point_digest(study.near_bottom_points()))

    def test_parent_can_retain_native_exit_failure_without_retry_or_fabricated_completion(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(study, '_validate_old_campaign', side_effect=self.old_campaign):
            output, _ = self.prepare(Path(temporary))
            receipt = study.record_arm_exit(output, 'log_on', -9)
            self.assertEqual(receipt['return_code'], -9)
            self.assertTrue(receipt['no_retry'])
            self.assertTrue((output/'process-exit-log_on.json').exists())
            self.assertEqual(study._read(output/'state.json')['status'], 'prepared')
            with self.assertRaisesRegex(ValueError, 'already recorded'):
                study.record_arm_exit(output, 'log_on', -9)
            with self.assertRaisesRegex(ValueError, 'cannot be retried'):
                study.run_arm(output, 'log_on')
            with self.assertRaisesRegex(ValueError, 'has not completed'):
                study.finalize_comparison(output)


if __name__ == '__main__':
    unittest.main()
