"""Constructed recorder evidence checks; no numerical solve or audit run."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('constructed_cutoff_recorder', ROOT/'tools/pinn_spatial/record-cutoff-audit.py')
RECORDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECORDER)


class CutoffRecordTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='constructed-cutoff-record-')
        self.root = Path(self.temporary.name).resolve()
        self.raw = self.root/'reports/audit'
        self.raw.mkdir(parents=True)
        self.records = self.root/'thermotwin'
        self.records.mkdir()
        self.protected = self.root/'old-sealed'
        self.protected.mkdir()
        (self.protected/'immutable').write_bytes(b'constructed original evidence')
        self.protected_before = RECORDER.audit.common._tree_records(self.protected)
        self.plan = {'protocol': RECORDER.audit._protocol(), 'cases': list(RECORDER.audit.CASES),
            'model_config': {'rank_rcond': 1e-7},
            'protected_trees': {str(self.protected): self.protected_before}}
        self.write(self.raw/'plan.json', self.plan)
        self.write(self.raw/'state.json', {'status': 'prepared', 'plan_sha256': RECORDER.audit.common.digest_file(self.raw/'plan.json')})
        self.patches = [patch.multiple(RECORDER, ROOT=self.root,
            PLAN=self.records/'PLAN.json', RESULT=self.records/'RESULT.json'),
            patch.object(RECORDER.audit, 'verify_plan', return_value=True),
            patch.object(RECORDER.audit, '_one_thread'),
            patch.object(RECORDER.audit, 'solve_readout_once', side_effect=AssertionError('Recorder must never solve')),
            patch.object(RECORDER.audit, 'run_cutoff_audit', side_effect=AssertionError('Recorder must never run the audit'))]
        self.started_patches = [value.start() for value in self.patches]
        RECORDER.freeze(self.raw)

    def tearDown(self):
        self.assertEqual(RECORDER.audit.common._tree_records(self.protected), self.protected_before)
        for value in reversed(self.patches):
            value.stop()
        self.temporary.cleanup()

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value, allow_nan=False, indent=2)+'\n')

    def result(self, statuses=None, failed_before_claim=False):
        statuses = ['solved']*14 if statuses is None else statuses
        rows, attempted, successful = [], 0, 0
        for case_index, case in enumerate(RECORDER.audit.CASES):
            samples = []
            points = [point for point in self.plan['protocol'] if point['case'] == case]
            for offset, point in enumerate(points):
                status = statuses[case_index*7+offset]
                if status == 'not_run':
                    sample = dict(point, status=status, readout_attempted=False,
                        attempt_before=None, attempt_after=None, reason='resource cap before this opportunity')
                else:
                    before = attempted
                    attempted += int(not (failed_before_claim and status == 'failed'))
                    successful += int(status == 'solved')
                    sample = dict(point, status=status, readout_attempted=attempted > before,
                        attempt_before=before, attempt_after=attempted,
                        pure_tsvd_objective=1.25, singular_values=[1., .5], system={'constructed': True})
                    if attempted > before:
                        sample['solve_number'] = attempted
                    if status == 'failed':
                        sample.update(error_type='ResourceLimit', error='interrupted constructed solve')
                samples.append(sample)
            status = 'completed' if all(value['status'] == 'solved' for value in samples) else (
                'not_run' if all(value['status'] == 'not_run' for value in samples) else 'failed')
            rows.append({'case': case, 'status': status, 'samples': samples,
                'original_state_restored_exactly': status != 'not_run'})
        return {'status': 'completed' if successful == 14 else 'failed',
            'plan_sha256': RECORDER.audit.common.digest_file(self.raw/'plan.json'), 'cases': rows,
            'attempted_readout_solves': attempted, 'successful_readout_solves': successful,
            'maximum_readout_solves': 14, 'training_only': True, 'neural_training_updates': 0,
            'new_checkpoints_written': 0, 'inputs_and_sources_preserved': True,
            'reference_fields_used': False, 'independent_diagnostics_used': False,
            'independent_diagnostics_used_for_selection': False, 'automatic_followup_launched': False}

    def seal_index(self, result):
        index = {str(path.relative_to(self.raw)): {'sha256': RECORDER.audit.common.digest_file(path), 'bytes': path.stat().st_size}
            for path in self.raw.rglob('*') if path.is_file() and path not in (self.raw/'state.json', self.raw/'artifact-index.json')}
        self.write(self.raw/'artifact-index.json', index)
        self.write(self.raw/'state.json', {'status': result['status'], 'plan_sha256': result['plan_sha256'],
            'artifact_index_sha256': RECORDER.audit.common.digest_file(self.raw/'artifact-index.json')})

    def seal(self, result, progress=None):
        for row in result['cases']:
            self.write(self.raw/(row['case']+'.json'), row)
        self.write(self.raw/'result.json', result)
        if progress is None:
            progress = [sample for row in result['cases'] for sample in row['samples'] if sample['status'] != 'not_run']
        (self.raw/'progress.jsonl').write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in progress))
        self.seal_index(result)

    def test_freeze_and_record_are_once_only_and_preserve_complete_raw_evidence(self):
        declaration = RECORDER.PLAN.read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'Refusing to replace'):
            RECORDER.freeze(self.raw)
        self.assertEqual(RECORDER.PLAN.read_bytes(), declaration)
        result = self.result()
        self.seal(result)
        before = RECORDER.audit.common._tree_records(self.raw)
        compact = RECORDER.record(self.raw)
        self.assertEqual(compact['denominator'], {'declared_points': 14, 'solved': 14, 'failed': 0, 'not_run': 0})
        self.assertEqual(RECORDER.audit.common._tree_records(self.raw), before)
        original = RECORDER.RESULT.read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'Refusing to replace'):
            RECORDER.record(self.raw)
        self.assertEqual(RECORDER.RESULT.read_bytes(), original)
        self.assertTrue(all('singular_values' not in point and 'system' not in point
            for row in compact['cases'] for point in row['samples']))
        self.started_patches[2].assert_called()
        self.started_patches[3].assert_not_called()
        self.started_patches[4].assert_not_called()

    def test_compact_declaration_checks_every_field_except_timestamp(self):
        result = self.result()
        self.seal(result)
        original = RECORDER.read(RECORDER.PLAN)
        for key, value in (('model_config', {'rank_rcond': .1}), ('protocol', []),
                           ('protected_trees', {}), ('invented_claim', True)):
            changed = deepcopy(original)
            changed[key] = value
            self.write(RECORDER.PLAN, changed)
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'compact prospective declaration'):
                RECORDER.record(self.raw)
            self.assertFalse(RECORDER.RESULT.exists())
        original['recorded_utc'] = 'constructed different record timestamp'
        self.write(RECORDER.PLAN, original)
        self.assertEqual(RECORDER.record(self.raw)['denominator']['solved'], 14)

    def test_duplicate_unexpected_and_missing_cases_or_opportunities_fail_closed(self):
        original = self.result()
        mutations = (
            lambda value: value['cases'].append(deepcopy(value['cases'][0])),
            lambda value: value['cases'][0].__setitem__('case', 't99'),
            lambda value: value['cases'].pop(),
            lambda value: value['cases'][0]['samples'].append(deepcopy(value['cases'][0]['samples'][0])),
            lambda value: value['cases'][0]['samples'][0].__setitem__('radius', 1.),
            lambda value: value['cases'][0]['samples'].pop(),
        )
        for index, mutate in enumerate(mutations):
            changed = deepcopy(original)
            mutate(changed)
            self.seal(changed)
            with self.subTest(mutation=index), self.assertRaisesRegex(RuntimeError, 'Duplicate|unexpected|Unexpected|Both declared|All seven'):
                RECORDER.record(self.raw)
            self.assertFalse(RECORDER.RESULT.exists())

    def test_resource_interruption_retains_solved_failed_and_unattempted_opportunities(self):
        result = self.result(['solved', 'solved', 'failed']+['not_run']*11)
        self.seal(result)
        compact = RECORDER.record(self.raw)
        self.assertEqual(compact['denominator'], {'declared_points': 14, 'solved': 2, 'failed': 1, 'not_run': 11})
        self.assertEqual(compact['attempted_readout_solves'], 3)
        self.assertEqual(compact['successful_readout_solves'], 2)
        failure = compact['cases'][0]['samples'][2]
        self.assertEqual((failure['attempt_before'], failure['attempt_after'], failure['solve_number']), (2, 3, 3))
        self.assertEqual(compact['cases'][1]['status'], 'not_run')

    def test_pre_solve_failure_does_not_invent_a_readout_call(self):
        result = self.result(['solved', 'failed']+['not_run']*12, failed_before_claim=True)
        self.seal(result)
        compact = RECORDER.record(self.raw)
        self.assertEqual(compact['attempted_readout_solves'], 1)
        self.assertEqual(compact['successful_readout_solves'], 1)
        self.assertEqual(compact['denominator']['failed'], 1)

    def test_aggregate_and_per_point_counter_disagreement_never_produces_a_record(self):
        original = self.result(['solved', 'solved', 'failed']+['not_run']*11)
        mutations = (
            lambda value: value.__setitem__('successful_readout_solves', 0),
            lambda value: value.__setitem__('attempted_readout_solves', 2),
            lambda value: value.__setitem__('maximum_readout_solves', True),
            lambda value: value['cases'][0]['samples'][2].__setitem__('attempt_after', 4),
            lambda value: value['cases'][0]['samples'][2].__setitem__('readout_attempted', False),
            lambda value: value['cases'][0]['samples'][2].__setitem__('solve_number', 4),
            lambda value: value['cases'][0]['samples'][3].__setitem__('attempt_after', 4),
        )
        for index, mutate in enumerate(mutations):
            changed = deepcopy(original)
            mutate(changed)
            self.seal(changed)
            with self.subTest(mutation=index), self.assertRaisesRegex(RuntimeError, 'accounting|counts|solve identity|unrun'):
                RECORDER.record(self.raw)
            self.assertFalse(RECORDER.RESULT.exists())

    def test_progress_must_exactly_retain_the_canonical_attempted_prefix(self):
        result = self.result(['solved', 'solved', 'failed']+['not_run']*11)
        progress = result['cases'][0]['samples'][:3]
        for values in (progress[:2], progress+[progress[-1]], list(reversed(progress))):
            self.seal(result, progress=values)
            with self.assertRaisesRegex(RuntimeError, 'Progress differs'):
                RECORDER.record(self.raw)
            self.assertFalse(RECORDER.RESULT.exists())

    def test_nested_state_basename_is_not_excluded_from_the_file_seal(self):
        result = self.result()
        self.seal(result)
        nested = self.raw/'nested'
        nested.mkdir()
        (nested/'state.json').write_text('unindexed addition')
        with self.assertRaisesRegex(RuntimeError, 'file set changed'):
            RECORDER.record(self.raw)
        self.seal_index(result)
        self.assertEqual(RECORDER.record(self.raw)['denominator']['solved'], 14)

    def test_resealed_case_mismatch_scope_change_and_false_completion_fail_closed(self):
        original = self.result()
        self.seal(original)
        changed_case = deepcopy(original['cases'][0])
        changed_case['original_state_restored_exactly'] = False
        self.write(self.raw/'t02.json', changed_case)
        self.seal_index(original)
        with self.assertRaisesRegex(RuntimeError, 'case file differs'):
            RECORDER.record(self.raw)
        for key, value in (('training_only', False), ('reference_fields_used', True),
                           ('inputs_and_sources_preserved', False), ('neural_training_updates', True)):
            changed = dict(original, **{key: value})
            self.seal(changed)
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'scope or denominator'):
                RECORDER.record(self.raw)
        changed = self.result(['solved']*13+['not_run'])
        changed['status'] = 'completed'
        self.seal(changed)
        with self.assertRaisesRegex(RuntimeError, 'completed audit'):
            RECORDER.record(self.raw)

    def test_record_targets_cannot_point_inside_raw_or_protected_evidence(self):
        result = self.result()
        self.seal(result)
        for target in (self.raw/'new-result.json', self.protected/'new-result.json'):
            with patch.object(RECORDER, 'RESULT', target), self.assertRaisesRegex(RuntimeError, 'sealed or protected'):
                RECORDER.record(self.raw)
            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
