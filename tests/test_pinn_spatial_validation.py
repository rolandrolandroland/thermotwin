import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from thermotwin.studies.pinn_spatial_validation import (
    canonical_digest, digest_file, verify_ci, validate_evidence, geometry_assessment,
    run_spatial_campaign, runtime_context,
)


class SpatialCampaignIntegrityTests(unittest.TestCase):
    def _mock_campaign(self, directory, *, reference_passed, planar_status='completed'):
        """Constructed records only: no scientific data or fits are generated."""
        ci = dict(head_sha='a'*40, status='completed', conclusion='success', workflow_id=348759149,
                  repository={'full_name': 'rolandrolandroland/thermotwin'})
        ci_path = directory/'ci.json'
        ci_path.write_text(json.dumps(ci))
        protocol = {'geometry_3d': {'geometry': 'synthetic extrusion'}}
        context = {'source_commit': 'a'*40, 'source_digest': 'constructed',
                   'operating_decision_frozen_files_unchanged': 117}
        def one_d(path):
            path.mkdir()
            result = {'status': 'completed' if reference_passed else 'blocked',
                      'numerical_reference_passed': reference_passed, 'denominator': 3,
                      'completed_count': 0, 'trials': []}
            (path/'result.json').write_text(json.dumps(result))
            return result
        def two_d(path, *, progress):
            path.mkdir()
            return {'status': planar_status, 'trials': []}
        with patch('thermotwin.studies.pinn_spatial_validation.validated_protocol', return_value=protocol), \
             patch('thermotwin.studies.pinn_spatial_validation.runtime_context', return_value={}), \
             patch('thermotwin.studies.pinn_spatial_validation.source_context', return_value=context), \
             patch('thermotwin.studies.pinn_spatial_fresh_1d.run_fresh_1d_validation', side_effect=one_d), \
             patch('thermotwin.studies.pinn_spatial_2d.run_planar_benchmark', side_effect=two_d) as planar:
            output = directory/'output'
            result = run_spatial_campaign(output, source_commit='a'*40, ci_evidence=ci_path)
            return output, result, planar.call_count

    def test_staged_dry_run_stops_on_reference_failure_and_validates_portable_evidence(self):
        try:
            import thermotwin.studies.pinn_spatial_2d  # optional scientific dependencies
        except ModuleNotFoundError:
            self.skipTest('optional scientific dependencies unavailable')
        with tempfile.TemporaryDirectory() as temporary:
            output, result, calls = self._mock_campaign(Path(temporary), reference_passed=False)
            self.assertEqual(calls, 0)
            self.assertEqual(result['status'], 'blocked_1d_reference')
            self.assertGreaterEqual(validate_evidence(output)['validated_files'], 5)
            self.assertTrue((output/'exact-source-ci.json').is_file())

    def test_staged_dry_run_never_labels_failed_2d_execution_complete(self):
        try:
            import thermotwin.studies.pinn_spatial_2d
        except ModuleNotFoundError:
            self.skipTest('optional scientific dependencies unavailable')
        with tempfile.TemporaryDirectory() as temporary:
            _, result, calls = self._mock_campaign(Path(temporary), reference_passed=True, planar_status='failed')
            self.assertEqual(calls, 1)
            self.assertEqual(result['status'], 'stopped_2d_failed')

    def test_runtime_mismatch_blocks_generation(self):
        protocol = {'execution': {'runtime_requirements': {
            'implementation': 'CPython', 'python': '0.0.0', 'packages': {}}}}
        with self.assertRaisesRegex(RuntimeError, 'runtime differs'):
            runtime_context(protocol)

    def test_ci_requires_exact_repository_workflow_commit_and_success(self):
        ci = dict(head_sha='a'*40, status='completed', conclusion='success', workflow_id=348759149,
                  repository={'full_name': 'rolandrolandroland/thermotwin'})
        self.assertEqual(verify_ci(ci, 'a'*40)['head_sha'], 'a'*40)
        for changes in ({'head_sha': 'b'*40}, {'status': 'in_progress'}, {'conclusion': 'failure'},
                        {'workflow_id': 1}, {'repository': {'full_name': 'someone/else'}}):
            with self.assertRaises(ValueError):
                verify_ci(dict(ci, **changes), 'a'*40)

    def test_evidence_reload_detects_corruption_and_path_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            target = directory/'result.json'
            target.write_text('{}\n')
            for name in ('manifest.json', 'exact-source-ci.json', 'REPORT.md'):
                (directory/name).write_text('{}\n')
            records = [{'path': path.name, 'bytes': path.stat().st_size, 'sha256': digest_file(path)}
                       for path in sorted(directory.iterdir())]
            index = directory/'evidence-index.json'
            index.write_text(json.dumps({'files': records, 'digest': canonical_digest(records)}))
            self.assertEqual(validate_evidence(directory)['validated_files'], 4)
            extra = directory/'unindexed.txt'
            extra.write_text('extra')
            with self.assertRaisesRegex(ValueError, 'artifact set'):
                validate_evidence(directory)
            extra.unlink()
            target.write_text('[]\n')
            with self.assertRaisesRegex(ValueError, 'changed'):
                validate_evidence(directory)
            target.write_text('{}\n')
            records.append({'path': '../result.json', 'bytes': 3, 'sha256': digest_file(target)})
            index.write_text(json.dumps({'files': records, 'digest': canonical_digest(records)}))
            with self.assertRaisesRegex(ValueError, 'path'):
                validate_evidence(directory)

    def test_3d_assessment_is_conditional_and_not_hardware_validation(self):
        assessment = geometry_assessment({'geometry_3d': {'geometry': 'synthetic extrusion'}})
        self.assertFalse(assessment['hardware_geometry_available'])
        self.assertFalse(assessment['numerical_3d_run'])
        self.assertIn('Uniqueness', assessment['basis'])
        self.assertIn('finite-depth', assessment['basis'])


if __name__ == '__main__':
    unittest.main()
