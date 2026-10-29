"""Constructed checks for terminal software evidence; no scientific execution."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT/'tools/pinn_spatial/finalize-step-control-result.py'
SPEC = importlib.util.spec_from_file_location('pinn_spatial_completion_finalizer', HELPER)
FINALIZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FINALIZER)


class CompletionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='thermotwin-completion-evidence-')
        self.root = Path(self.temporary.name)
        self.validation = self.root/'validation'
        self.validation.mkdir()
        self.source = self.root/'constructed-source.py'
        self.source.write_text('constructed software source\n')
        self.state_path = self.validation/'full-suite-state.json'
        self.log_path = self.validation/'full-suite.log'
        self.manifest_path = self.validation/'full-suite-source-manifest.json'
        self.patcher = patch.multiple(FINALIZER, ROOT=self.root, VALIDATION=self.validation)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.temporary.cleanup()

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

    def fixture(self, *, legacy=False):
        self.log_path.write_text('constructed test output\n\nRan 7 tests in 0.123s\n\nOK\n')
        sources = {'constructed-source.py': FINALIZER.digest(self.source)}
        self.write(self.manifest_path, {'created_at': 'constructed', 'source_sha256': sources})
        state = {'status': 'completed', 'passed': True, 'tests_run': 7,
            'discovered_test_count': 7, 'expected_test_count': 7,
            'failures': 0, 'errors': 0, 'unexpected_successes': 0,
            'source_unchanged': True, 'source_sha256': sources,
            'log_sha256': FINALIZER.digest(self.log_path),
            'source_manifest_sha256': FINALIZER.digest(self.manifest_path)}
        if legacy:
            for key in ('passed', 'tests_run', 'discovered_test_count', 'expected_test_count',
                        'log_sha256', 'source_manifest_sha256'):
                state.pop(key)
            state.update(status='passed', tests=7, exit_code=0)
        self.write(self.state_path, state)
        return state

    def evidence(self):
        return FINALIZER.test_evidence('full-suite')

    def test_canonical_completed_proof_uses_all_declared_counts_and_hashes(self):
        state = self.fixture()
        original = self.state_path.read_bytes()
        result = self.evidence()
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(result['completed'])
        self.assertIs(result['passed'], True)
        self.assertEqual(result['test_count'], 7)
        self.assertEqual(result['declared_test_counts'], {key: state[key] for key in
            ('tests_run', 'discovered_test_count', 'expected_test_count')})
        self.assertTrue(result['declared_log_hash_matches'])
        self.assertTrue(result['source_records_match_current'])
        self.assertTrue(result['source_manifest_matches'])
        self.assertEqual(self.state_path.read_bytes(), original)

    def test_legacy_passed_proof_without_optional_hashes_remains_supported(self):
        self.fixture(legacy=True)
        result = self.evidence()
        self.assertEqual(result['status'], 'passed')
        self.assertIs(result['passed'], True)
        self.assertEqual(result['test_count'], 7)
        self.assertNotIn('declared_log_hash_matches', result)

    def test_legacy_failed_proof_remains_failed(self):
        state = self.fixture(legacy=True)
        state.update(status='failed', failures=1, exit_code=1)
        self.log_path.write_text('Ran 7 tests in 0.123s\n\nFAILED (failures=1)\n')
        self.write(self.state_path, state)
        result = self.evidence()
        self.assertTrue(result['completed'])
        self.assertIs(result['passed'], False)

    def test_completed_false_cannot_be_promoted_by_ok_log(self):
        state = self.fixture()
        state['passed'] = False
        self.write(self.state_path, state)
        self.assertIs(self.evidence()['passed'], False)

    def test_completed_unknown_or_nonboolean_pass_value_fails_closed(self):
        for value in (None, 'true', 1):
            with self.subTest(value=value):
                state = self.fixture()
                state['passed'] = value
                self.write(self.state_path, state)
                result = self.evidence()
                self.assertIsNone(result['passed'])
                self.assertIn('explicit boolean', result['evidence_error'])
        state = self.fixture()
        state.pop('passed')
        self.write(self.state_path, state)
        self.assertIsNone(self.evidence()['passed'])

    def test_every_declared_count_must_agree_with_terminal_log(self):
        for key in ('tests_run', 'discovered_test_count', 'expected_test_count', 'tests', 'test_count'):
            with self.subTest(key=key):
                state = self.fixture()
                state[key] = 8
                self.write(self.state_path, state)
                result = self.evidence()
                self.assertIsNone(result['passed'])
                self.assertFalse(result['declared_counts_match_log'])

    def test_completed_schema_requires_run_discovery_and_expected_counts(self):
        for key in ('tests_run', 'discovered_test_count', 'expected_test_count'):
            with self.subTest(key=key):
                state = self.fixture()
                state.pop(key)
                self.write(self.state_path, state)
                self.assertIsNone(self.evidence()['passed'])

    def test_invalid_count_types_are_not_coerced(self):
        for value in (True, '7', -1, 7.0):
            with self.subTest(value=value):
                state = self.fixture()
                state['tests_run'] = value
                self.write(self.state_path, state)
                self.assertIsNone(self.evidence()['passed'])

    def test_declared_log_hash_rejects_tamper_with_same_terminal_ok_and_count(self):
        self.fixture()
        self.log_path.write_text('changed header\n\nRan 7 tests in 0.123s\n\nOK\n')
        result = self.evidence()
        self.assertIsNone(result['passed'])
        self.assertFalse(result['declared_log_hash_matches'])

    def test_passing_state_requires_terminal_ok(self):
        state = self.fixture()
        self.log_path.write_text('Ran 7 tests in 0.123s\n\nFAILED (failures=1)\n')
        state['log_sha256'] = FINALIZER.digest(self.log_path)
        self.write(self.state_path, state)
        self.assertIsNone(self.evidence()['passed'])

    def test_unittest_summary_must_be_the_last_nonempty_log_line(self):
        state = self.fixture()
        self.log_path.write_text(self.log_path.read_text()+'unexpected trailing output\n')
        state['log_sha256'] = FINALIZER.digest(self.log_path)
        self.write(self.state_path, state)
        self.assertIsNone(self.evidence()['passed'])

    def test_changed_source_rejects_pass_despite_unchanged_flag(self):
        self.fixture()
        self.source.write_text('changed after the software check\n')
        result = self.evidence()
        self.assertIsNone(result['passed'])
        self.assertFalse(result['source_records_match_current'])

    def test_missing_or_empty_source_records_do_not_authenticate_a_pass(self):
        for value in (None, {}):
            with self.subTest(value=value):
                state = self.fixture()
                state['source_sha256'] = value
                self.write(self.state_path, state)
                self.assertIsNone(self.evidence()['passed'])

    def test_escaped_source_path_is_rejected_without_exposing_external_content(self):
        state = self.fixture()
        state['source_sha256'] = {'../outside.py': 'constructed hash'}
        self.write(self.state_path, state)
        self.assertIsNone(self.evidence()['passed'])

    def test_false_or_missing_canonical_unchanged_source_flag_fails_closed(self):
        for value in (False, None, 'true'):
            with self.subTest(value=value):
                state = self.fixture()
                state['source_unchanged'] = value
                self.write(self.state_path, state)
                self.assertIsNone(self.evidence()['passed'])

    def test_declared_source_manifest_is_bound_by_bytes_and_contents(self):
        state = self.fixture()
        self.manifest_path.write_text('{}\n')
        result = self.evidence()
        self.assertIsNone(result['passed'])
        self.assertFalse(result['source_manifest_matches'])
        state['source_manifest_sha256'] = FINALIZER.digest(self.manifest_path)
        self.write(self.state_path, state)
        self.assertIsNone(self.evidence()['passed'])

    def test_missing_declared_manifest_or_log_rejects_pass(self):
        self.fixture()
        self.manifest_path.unlink()
        self.assertIsNone(self.evidence()['passed'])
        self.fixture()
        self.log_path.unlink()
        result = self.evidence()
        self.assertIsNone(result['passed'])
        self.assertFalse(result['log_read'])

    def test_failure_error_or_bad_exit_cannot_be_hidden_by_passed_flag(self):
        for key, value in (('failures', 1), ('errors', 1), ('unexpected_successes', 1),
                           ('exit_code', 1), ('return_code', 1), ('exit_code', False)):
            with self.subTest(key=key, value=value):
                state = self.fixture()
                state[key] = value
                self.write(self.state_path, state)
                self.assertIsNone(self.evidence()['passed'])

    def test_conflicting_legacy_status_and_boolean_flag_fails_closed(self):
        state = self.fixture(legacy=True)
        state['passed'] = False
        self.write(self.state_path, state)
        self.assertIsNone(self.evidence()['passed'])
        state.update(status='failed', passed=True)
        self.write(self.state_path, state)
        self.assertIs(self.evidence()['passed'], False)

    def test_running_or_unknown_status_never_reads_partial_log(self):
        for status in ('running', 'prepared', 'unknown'):
            with self.subTest(status=status):
                self.write(self.state_path, {'status': status})
                if not self.log_path.exists():
                    self.log_path.mkdir()
                result = self.evidence()
                self.assertFalse(result['completed'])
                self.assertFalse(result['log_read'])
                self.assertIsNone(result['test_count'])
                self.assertIsNone(result['passed'])


if __name__ == '__main__':
    unittest.main()
