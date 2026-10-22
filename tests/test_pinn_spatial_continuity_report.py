"""Constructed evidence checks; no scientific data generation or numerical fits."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from thermotwin.reports import pinn_spatial_continuity as report


class ContinuityReportTests(unittest.TestCase):
    def write(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, allow_nan=False))

    def fixture(self, root):
        campaign, prior_root = root/'campaign', root/'old-v2'
        campaign.mkdir()
        prior_root.mkdir()
        declared = [{'id': name, 'geometry': geometry, 'known_h': h}
            for name, geometry, h in (('t01', 'full_contact', 800.), ('t02', 'partial_contact', 800.),
                                     ('t03', 'full_contact', 1400.), ('t04', 'partial_contact', 1400.))]
        def completed(case, current):
            normalized = {'interior': .9 if current else .04, 'bottom_contact': .01,
                          'left': 0., 'right': 0., 'top': 0.}
            physical = {'interior': .0375, 'bottom_contact': 5., 'left': 0., 'right': 0., 'top': 0.}
            if case['geometry'] == 'partial_contact':
                normalized['bottom_insulated'] = .02
                physical['bottom_insulated'] = 10.
                for index in (0, 1):
                    normalized[f'interface_{index}_temperature'] = 0. if current else .1
                    normalized[f'interface_{index}_flux'] = 0. if current else .02
                    physical[f'interface_{index}_temperature'] = 0. if current else .01
                    physical[f'interface_{index}_flux'] = 0. if current else 10.
            return dict(case, status='completed', gate={'all': True}, field_metrics={'rmse_K': .01, 'maximum_error_K': .04},
                        independent_residuals={'normalized_rms': normalized, 'physical_rms': physical},
                        **({'normalized_observation_mse': .8} if current else {'wall_seconds': 300.}))
        prior = {'status': 'blocked_forward_gate', 'cases': [completed(case, False) for case in declared]}
        prior_path = prior_root/'final-forward-selection.json'
        self.write(prior_path, prior)
        plan = {'candidate_id': 'continuity-v3', 'cases': declared, 'model_config': {'width': 24}, 'protected_indices': [{
            'directory': str(prior_root), 'files': [{'path': prior_path.name, 'sha256': report.digest_file(prior_path),
                'bytes': prior_path.stat().st_size}]}]}
        self.write(campaign/'plan.json', plan)
        cases = [completed(case, True) for case in declared[:2]]
        cases.extend([dict(declared[2], status='failed', error='constructed failure'),
                      dict(declared[3], status='not_run', reason='constructed resource stop')])
        result = {'status': 'blocked_forward_gate', 'cases': cases, 'plan_sha256': report.digest_file(campaign/'plan.json'),
            'wall_seconds': 123., 'peak_rss_bytes': 456789, 'source_unchanged': True,
            'source_sha256': {'constructed_source.py': 'constructed'}, 'inverse_opened': False, 'fresh_evaluation_opened': False}
        self.write(campaign/'final-forward-selection.json', result)
        for case in declared[:2]:
            self.write(campaign/case['id']/'training-report.json', {'wall_seconds': 50., 'coefficient_count': 100,
                'projections': [{'rank': 10, 'columns': 100, 'coefficient_norm_returned': 100., 'accepted': True},
                                {'rank': 12, 'columns': 100, 'coefficient_norm_returned': 2., 'accepted': True}],
                'hidden_backtrack_count': 4, 'hidden_rejected_updates': 1, 'optimizer_resets': 3,
                'hidden_accepted_objective_increases': 0})
            self.write(campaign/case['id']/'near-interface-audit.json',
                {'status': 'not_applicable'} if case['geometry'] == 'full_contact' else {'status': 'completed', 'role': 'report only',
                    'diagnostics': {'normalized_rms': {'interior': .7},
                        'physical_rms': {'interface_0_temperature': 0., 'interface_0_flux': 0.,
                            'interface_1_temperature': 0., 'interface_1_flux': 0.}}})
        self.seal(campaign)
        return campaign, prior_path

    def seal(self, directory):
        records = {str(path.relative_to(directory)): {'bytes': path.stat().st_size, 'sha256': report.digest_file(path)}
                   for path in sorted(directory.rglob('*')) if path.is_file() and path.name not in ('state.json', 'artifact-index.json')}
        self.write(directory/'artifact-index.json', records)
        self.write(directory/'state.json', {'status': 'completed',
            'plan_sha256': report.digest_file(directory/'plan.json'),
            'artifact_index_sha256': report.digest_file(directory/'artifact-index.json')})

    def plot_stub(self, output, comparison):
        (output/'comparison.png').write_bytes(b'constructed PNG')
        (output/'comparison.pdf').write_bytes(b'constructed PDF')
        return ['comparison.png', 'comparison.pdf']

    def test_supplied_gates_are_displayed_without_recalculation_and_missing_values_remain_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign, prior = self.fixture(root)
            comparison = report.load_comparison(campaign, prior)
            self.assertTrue(comparison['rows'][0]['candidate']['gate']['all'])
            self.assertEqual(comparison['rows'][0]['candidate']['independent_residuals']['normalized_rms']['interior'], .9)
            self.assertEqual(comparison['rows'][0]['training']['final_projection_rank'], 12)
            self.assertEqual(comparison['rows'][0]['training']['final_coefficient_norm'], 2.)
            self.assertIsNone(comparison['rows'][2]['training']['projection_count'])
            markdown = report._markdown(comparison)
            self.assertIn('| V3 | completed | pass | 0.9 |', markdown)
            self.assertIn('failed | not recorded | not recorded', markdown)
            self.assertIn('not_run | not recorded | not recorded', markdown)
            self.assertIn('report only', markdown)
            self.assertIn('constructed failure', markdown)

    def test_renderer_writes_only_separate_outputs_and_indexes_each_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign, prior = self.fixture(root)
            original = {str(path): path.read_bytes() for path in campaign.rglob('*') if path.is_file()}
            old_bytes = prior.read_bytes()
            with patch.object(report, '_plots', side_effect=self.plot_stub):
                summary = report.render_report(campaign, prior_selection=prior, output_directory=root/'report')
            self.assertEqual(summary['files'], 4)
            index = report._json(root/'report/report-index.json')
            self.assertEqual(index['digest'], report.canonical_digest(index['files']))
            self.assertEqual({record['path'] for record in index['files']}, {'REPORT.md', 'result-summary.json', 'comparison.png', 'comparison.pdf'})
            for record in index['files']:
                self.assertEqual(record['sha256'], report.digest_file(root/'report'/record['path']))
            for path, content in original.items():
                self.assertEqual(Path(path).read_bytes(), content)
            self.assertEqual(prior.read_bytes(), old_bytes)
            for forbidden in (campaign/'nested-report', prior.parent/'nested-report', root):
                with self.assertRaisesRegex(ValueError, 'separate'):
                    report.render_report(campaign, prior_selection=prior, output_directory=forbidden)

    def test_corrupted_missing_unindexed_or_incomplete_evidence_is_rejected(self):
        for mutation in ('changed', 'missing', 'extra', 'running', 'index'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                campaign, prior = self.fixture(root)
                if mutation == 'changed':
                    (campaign/'t01/training-report.json').write_text('changed')
                elif mutation == 'missing':
                    (campaign/'t01/training-report.json').unlink()
                elif mutation == 'extra':
                    (campaign/'unindexed.json').write_text('{}')
                elif mutation == 'running':
                    state = report._json(campaign/'state.json')
                    state['status'] = 'running'
                    self.write(campaign/'state.json', state)
                else:
                    self.write(campaign/'artifact-index.json', {})
                with self.assertRaises(ValueError):
                    report.load_comparison(campaign, prior)

    def test_prior_selection_cannot_be_substituted_and_duplicate_cases_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign, prior = self.fixture(root)
            original = prior.read_bytes()
            prior.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'old v2 selection differs'):
                report.load_comparison(campaign, prior)
            prior.write_bytes(original)
            result = report._json(campaign/'final-forward-selection.json')
            result['cases'].append(deepcopy(result['cases'][0]))
            self.write(campaign/'final-forward-selection.json', result)
            self.seal(campaign)
            with self.assertRaisesRegex(ValueError, 'duplicate case'):
                report.load_comparison(campaign, prior)

    def test_new_candidate_version_is_taken_from_saved_plan_and_never_mislabeled_v3(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign, prior = self.fixture(root)
            plan = report._json(campaign/'plan.json')
            plan['candidate_id'] = 'corrective-v4-shared-traces'
            self.write(campaign/'plan.json', plan)
            result = report._json(campaign/'final-forward-selection.json')
            result['plan_sha256'] = report.digest_file(campaign/'plan.json')
            self.write(campaign/'final-forward-selection.json', result)
            self.seal(campaign)
            comparison = report.load_comparison(campaign, prior)
            self.assertEqual(comparison['candidate_label'], 'V4')
            self.assertEqual(comparison['candidate_id'], plan['candidate_id'])
            markdown = report._markdown(comparison)
            self.assertIn('# Paired 2D continuity repair: V4 development', markdown)
            self.assertIn('| V4 | completed | pass |', markdown)
            self.assertNotIn('V3', markdown)
            self.assertNotIn('saved v3 plan', markdown)
            self.assertEqual(report._candidate_label({'candidate_id': 'unversioned-correction'}), 'Candidate')
            self.assertEqual(report._candidate_label({'candidate_id': 'v3-to-v4'}), 'Candidate')
            self.assertEqual(report._candidate_label({'candidate_id': 'v4', 'report_label': 'Declared revision'}), 'Declared revision')


if __name__ == '__main__':
    unittest.main()
