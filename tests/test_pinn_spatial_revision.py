"""Review gates only: constructed records, no scientific datasets or training."""
from dataclasses import asdict
from copy import deepcopy
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from thermotwin.pinn.planar_contact_revision import PlanarRevisionConfig
from thermotwin.studies import pinn_spatial_revision as revision


class RevisionFreezeTests(unittest.TestCase):
    inverse_source = 'thermotwin/studies/pinn_spatial_revision_inverse.py'

    def selection(self):
        config = json.loads(json.dumps(asdict(PlanarRevisionConfig())))
        return {'status': 'passed', 'selected_config': config, 'source_sha256': {'constructed': 'fixture'},
                'cases': [{'geometry': geometry, 'known_h': h, 'config': config, 'gate': {'all': True}}
                          for geometry in ('full_contact', 'partial_contact') for h in (800., 1400.)]}

    def inverse_selection(self):
        return {'status': 'passed', 'selected_strategy': 'delayed_release', 'source_unchanged': True,
                'source_sha256': {path: 'fixture' for path in (self.inverse_source,
                    'thermotwin/pinn/planar_contact_revision.py', 'thermotwin/physics/planar_contact.py',
                    'thermotwin/pinn/planar_contact_inverse.py')},
                'forward_config': deepcopy(self.selection()['selected_config']),
                'rows': [{'strategy': strategy, 'development_h': h,
                          'passed': strategy == 'delayed_release',
                          'score': {'passed': strategy == 'delayed_release'}}
                         for strategy in ('delayed_release', 'alternating') for h in (800., 1400.)]}

    def frozen_protocol(self, inverse=None):
        one_d = {'status': 'selected', 'candidate_score': {'passed': True},
                 'selected_fd_refinement': [{'passed': True}],
                 'source_sha256': {'one-d-constructed': 'fixture'},
                 'spec_sha256': revision.canonical_digest(revision.ONE_D_SPEC),
                 'candidate': {'experiment_count': 6}}
        with patch.object(revision, 'digest_file', return_value='fixture'), \
             patch.object(revision, 'preserved_context', return_value={'constructed': True}):
            return revision.build_frozen_protocol(one_d_selection=one_d,
                forward_selection=self.selection(), inverse_selection=inverse or self.inverse_selection(),
                preserved_campaign='never-opened-constructed-campaign')

    def validate_fixture(self, protocol, digest=None):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'constructed-protocol.json'
            path.write_text(json.dumps(protocol, allow_nan=False))
            with patch.object(revision, 'PROTOCOL_PATH', path), \
                 patch.object(revision, 'digest_file', side_effect=digest or (lambda path: 'fixture')):
                return revision.validated_revision_protocol()

    def test_forward_gate_requires_same_config_for_all_geometries_and_contacts(self):
        selection = self.selection()
        with patch.object(revision, 'verify_selection_sources'):
            config = revision.verified_forward_selection(selection)
            self.assertEqual(config.width, selection['selected_config']['width'])
            selection['cases'][0]['config'] = dict(selection['selected_config'], width=config.width+1)
            with self.assertRaisesRegex(ValueError, 'same selected configuration'):
                revision.verified_forward_selection(selection)

    def test_forward_failure_is_never_reinterpreted_as_an_eligible_config(self):
        for status in ('blocked_forward_gate', 'running', 'failed'):
            value = self.selection()
            value['status'] = status
            with self.assertRaisesRegex(ValueError, 'has not passed'):
                revision.verified_forward_selection(value)

    def test_build_retains_complete_inverse_proof_and_runtime_accepts_it(self):
        inverse = self.inverse_selection()
        protocol = self.frozen_protocol(inverse)
        selected = protocol['selected_inverse']
        self.assertEqual(selected['development_selection'], inverse)
        self.assertEqual(len(selected['development_selection']['rows']), 4)
        self.assertEqual(selected['development_selection']['source_sha256'], inverse['source_sha256'])
        self.assertEqual(selected['development_evidence_digest'], revision.canonical_digest(inverse))
        self.assertEqual(self.validate_fixture(protocol), json.loads(json.dumps(protocol)))

    def test_build_rejects_incomplete_failed_or_changed_inverse_proof(self):
        mutations = {
            'strategy': lambda value: value.update(selected_strategy='unregistered_strategy'),
            'source_changed_during_fits': lambda value: value.update(source_unchanged=False),
            'source_hash': lambda value: value['source_sha256'].update({self.inverse_source: 'changed'}),
            'source_records_missing': lambda value: value.pop('source_sha256'),
            'partial_source_records': lambda value: value['source_sha256'].pop(self.inverse_source),
            'config': lambda value: value['forward_config'].update(width=33),
            'missing_contact': lambda value: value['rows'].pop(1),
            'duplicate_contact': lambda value: value['rows'][1].update(development_h=800.),
            'failed_row': lambda value: value['rows'][0].update(passed=False),
            'failed_score': lambda value: value['rows'][0]['score'].update(passed=False),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                inverse = self.inverse_selection()
                mutate(inverse)
                with self.assertRaises(ValueError):
                    self.frozen_protocol(inverse)

    def test_runtime_rejects_execution_and_retained_proof_mutations(self):
        mutations = {
            'execution_strategy': lambda selected: selected.update(strategy='alternating'),
            'execution_config': lambda selected: selected['config'].update(width=33),
            'missing_proof': lambda selected: selected.pop('development_selection'),
            'proof_strategy': lambda selected: selected['development_selection'].update(selected_strategy='alternating'),
            'proof_config': lambda selected: selected['development_selection']['forward_config'].update(width=33),
            'proof_contact': lambda selected: selected['development_selection']['rows'][1].update(development_h=800.),
            'proof_row': lambda selected: selected['development_selection']['rows'][0].update(passed=False),
            'proof_score': lambda selected: selected['development_selection']['rows'][0]['score'].update(passed=False),
            'proof_source_records': lambda selected: selected['development_selection'].pop('source_sha256'),
            'proof_partial_source_records': lambda selected: selected['development_selection']['source_sha256'].pop(self.inverse_source),
            'proof_source_unchanged': lambda selected: selected['development_selection'].update(source_unchanged=False),
            'proof_source_hash': lambda selected: selected['development_selection']['source_sha256'].update({self.inverse_source: 'changed'}),
            'proof_digest': lambda selected: selected.update(development_evidence_digest='changed'),
            'scorecard': lambda selected: selected['development_scorecards'][0]['score'].update(passed=False),
            'unselected_evidence': lambda selected: selected['development_selection']['rows'][2].update(failure='changed'),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                protocol = self.frozen_protocol()
                mutate(protocol['selected_inverse'])
                with self.assertRaises(ValueError):
                    self.validate_fixture(protocol)

    def test_runtime_rechecks_inverse_numerical_source_after_freeze(self):
        protocol = self.frozen_protocol()
        def digest(path):
            return 'changed-after-freeze' if path.name == 'pinn_spatial_revision_inverse.py' else 'fixture'
        with self.assertRaisesRegex(ValueError, 'source changed: thermotwin/studies/pinn_spatial_revision_inverse.py'):
            self.validate_fixture(protocol, digest)

    def test_planned_unrun_rows_retain_every_method_and_h_repeat(self):
        rows = revision.pending_two_d_rows('reference_failure')
        self.assertEqual(len(rows), 6)
        self.assertEqual([row['true_h'] for row in rows], [650., 650., 1100., 1100., 1600., 1600.])
        self.assertTrue(all(len(row['methods']) == 3 for row in rows))
        self.assertTrue(all(method['status'] == 'not_run' for row in rows for method in row['methods'].values()))

    def test_seed_schedule_separates_stages_cases_and_paired_family_noise(self):
        for strategy in ('delayed_release', 'alternating'):
            protocol = {'selected_inverse': {'config': asdict(PlanarRevisionConfig()), 'strategy': strategy},
                        'one_d_selection': {'candidate': {'experiment_count': 6}}}
            audit = revision.seed_schedule(protocol)
            seeds = [row['seed'] for row in audit['streams']]
            self.assertEqual(len(seeds), len(set(seeds)))
            self.assertEqual(sum(row['semantic_key'].startswith('one_d/') for row in audit['streams']), 18)
            self.assertIn('paired', audit['intentional_pairing'])
            with patch('thermotwin.pinn.planar_contact_revision._derived_seed', return_value=123):
                with self.assertRaisesRegex(ValueError, 'collision'):
                    revision.seed_schedule(protocol)


if __name__ == '__main__':
    unittest.main()
