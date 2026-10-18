import ast
from concurrent.futures import Future
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import inspect
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout, redirect_stderr

from thermotwin.studies.operating_decision import POLICY_NAMES
from thermotwin.studies.operating_decision_prospective import ProspectiveSelectorRule
from thermotwin.studies.operating_decision_prospective_phase_e_calibration import PHASE_E_PROCEDURES
from thermotwin.studies.operating_decision_prospective_phase_f import (
    PHASE_F_REHEARSAL_PARTITION, RULE_DIGEST, load_complete_rule,
)
from thermotwin.studies.operating_decision_prospective_phase_f_decisions import (
    blinded_inputs_from_saved, calibrated_saves, persist_saved_decisions,
)
from thermotwin.studies.operating_decision_prospective_phase_f_worker import run_calibrated_family
from thermotwin.studies.operating_decision_prospective_pilot import (
    PROSPECTIVE_RESERVED_PARTITION, PROSPECTIVE_CALIBRATION_PARTITION,
)
from thermotwin.studies.operating_decision_realism import STAGE3_TRUTH_CONDITIONS
import thermotwin.studies.operating_decision_prospective_phase_f_execution as f
import thermotwin.reports.operating_decision_prospective_phase_f as cli


def context_fixture():
    return {'source_revision': 'a'*40, 'source_manifest_digest': 'b'*64,
            'protocol_digest': 'c'*64, 'runtime_identity': {'runtime': 'fixture'}}


def block_fixture(root, name=PHASE_F_REHEARSAL_PARTITION):
    partition = f.phase_f_partition(name)
    manifest = f.e._expected_corrected_stream_manifest(
        0, 'parent pilot', campaign=partition.campaign, partition=partition.name)
    registry = f.d6.RandomStreamRegistry()
    for use in manifest:
        registry.register(f.e.RandomStream(f.e.RandomStreamKey(**use['key'])),
            consumer=use['consumer'], pairing_member=use['pairing_member'], pairing_id=use['pairing_id'])
    cases, before = [], {}
    for family in STAGE3_TRUTH_CONDITIONS:
        raw = {p: {'margin_envelope': {'lower': -.5, 'upper': -.1},
                   'verifications': [{'passed': True}]} for p in POLICY_NAMES}
        fixed = {p: {'saved_before_reveal': raw[p],
                     'post_reveal_score': {'revealed': {'true_margin': -.25},
                         'scored': {'total_diagnostic_energy': 60., 'diagnostic_run_count': 2,
                                    'extra_sensor_count': 0, 'energized_schedule_time_seconds': 160.}}}
                 for p in POLICY_NAMES}
        case = {'block': 0, 'truth_condition': family, 'device_token': f'device-{family}',
                'selected_policy_at_n16_primary': 'stop_now', 'fixed_policy_results': fixed,
                'pipeline_failures': []}
        inputs, actions = blinded_inputs_from_saved(raw, 'stop_now')
        before[family] = persist_saved_decisions(f.before_path(Path(root), 0, family),
            calibrated_saves(inputs, actions), context=f.decision_context(context_fixture(), name),
            block=0, family=family)
        cases.append(case)
    return {'numerical_block': {'block': 0, 'cases': cases,
        'corrected_random_stream_manifest': manifest,
        'corrected_random_stream_audit': f.d6._strict_json_value(registry.audit()),
        'timing': {'block_cpu_seconds': 1.}}, 'before_reveal_by_family': before}


class PhaseFFreezeTests(unittest.TestCase):
    def test_complete_rule_preserves_endpoints_and_closed_calibration(self):
        rule = load_complete_rule()
        self.assertEqual(rule['artifact_digest'], RULE_DIGEST)
        self.assertEqual(rule['design']['reserved_block_count'], 100)
        self.assertEqual(rule['design']['analysis']['confidence']['primary_bound_count'], 73)
        self.assertFalse(rule['reserved_generation_gate']['reserved_generation_authorized_by_this_rule_artifact'])
        self.assertFalse(rule['calibrated_decision']['corrections_change_selection_stop_fits_offsets_or_verification'])
        self.assertEqual(set(rule['final_corrections_kelvin']), set(PHASE_E_PROCEDURES))

    def test_partition_entry_point_keeps_development_and_calibration_closed(self):
        self.assertEqual(f.phase_f_partition(PROSPECTIVE_RESERVED_PARTITION).block_count, 100)
        self.assertEqual(f.phase_f_partition(PHASE_F_REHEARSAL_PARTITION).block_count, 1)
        for name in (PROSPECTIVE_CALIBRATION_PARTITION, 'p1_development_tuning',
                     'p0_disposable_phase_e_roundtrip_v2', 'custom'):
            with self.assertRaises(ValueError):
                f.phase_f_partition(name)

    def test_static_worker_recovers_original_when_only_calibration_insertion_removed(self):
        original = ast.parse(inspect.getsource(f.d6._run_internal_check_family)).body[0]
        revised = ast.parse(inspect.getsource(run_calibrated_family)).body[0]
        revised.name = original.name
        revised.args.kwonlyargs.pop()
        revised.args.kw_defaults.pop()
        revised.returns = original.returns
        begin = next(i for i, n in enumerate(revised.body) if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'selected_before_reveal')
        end = next(i for i, n in enumerate(revised.body) if isinstance(n, ast.Assign)
                   and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'fixed_results')
        del revised.body[begin:end]
        revised.body[-1].value = revised.body[-1].value.elts[0]
        class Unqualify(ast.NodeTransformer):
            def visit_Attribute(self, node):
                if isinstance(node.value, ast.Name) and node.value.id == 'd6':
                    return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
                return self.generic_visit(node)
        revised = Unqualify().visit(revised)
        self.assertEqual(ast.dump(original, include_attributes=False), ast.dump(revised, include_attributes=False))

    def test_wrong_ci_revision_and_incomplete_generation_gate_precede_any_work(self):
        ci = {'head_sha': 'a'*40, 'status': 'completed', 'conclusion': 'success',
              'workflow_id': 348759149, 'repository': {'full_name': 'rolandrolandroland/thermotwin'}}
        self.assertEqual(f.validate_ci_evidence(ci, source_revision='a'*40), ci)
        for delta in ({'head_sha': 'b'*40}, {'status': 'in_progress'}, {'workflow_id': 0}):
            with self.assertRaises(ValueError):
                f.validate_ci_evidence({**ci, **delta}, source_revision='a'*40)
        with self.assertRaises(ValueError):
            f.validate_generation_gate({'reserved_generation_authorized': True}, context=context_fixture())


class PhaseFBoundaryTests(unittest.TestCase):
    def worker_run(self, sink):
        case = SimpleNamespace(case_id=SimpleNamespace(device_token='device'),
                               acquisition_runs=[None], final_regime=None)
        raw = {'margin_envelope': {'lower': -.5, 'upper': -.1},
               'verifications': [{'passed': True}]}
        with ExitStack() as stack:
            for name, kwargs in (
                ('build_corrected_blinded_case', {'return_value': case}),
                ('prepare_prospective_acquisition_evidence', {'side_effect': ValueError('retained acquisition failure')}),
                ('decide_corrected_blinded_case', {'return_value': raw}),
                ('_saved_payload', {'side_effect': lambda x: x}),
                ('_case_payload', {'return_value': {}}),
                ('_scored_record_payload', {'side_effect': lambda x: x}),
            ):
                stack.enter_context(patch.object(f.d6, name, **kwargs))
            def reveal(*args, **kwargs):
                self.assertTrue(self.path.exists(), 'target revealed before durable calibrated save')
                self.reveals += 1
                return {}
            stack.enter_context(patch.object(f.d6, 'score_corrected_saved_decision', side_effect=reveal))
            return run_calibrated_family(truth_condition=STAGE3_TRUTH_CONDITIONS[0], block=0,
                truth={}, partition=f.phase_f_partition(PHASE_F_REHEARSAL_PARTITION),
                physical_config=None, registry=None, selector_rule=ProspectiveSelectorRule(),
                frozen_rule=f.d6.phase_d6_frozen_selector_rule(),
                cost_scenario=f.d6.PRIMARY_PROSPECTIVE_COST_SCENARIO, before_reveal_sink=sink)

    def test_actual_worker_writes_calibrated_decisions_before_every_reveal(self):
        with tempfile.TemporaryDirectory() as directory:
            self.path, self.reveals = Path(directory) / 'saved.json', 0
            def sink(value):
                self.assertEqual(self.reveals, 0)
                return persist_saved_decisions(self.path, value, context={}, block=0,
                                               family=STAGE3_TRUTH_CONDITIONS[0])
            case, saved = self.worker_run(sink)
            self.assertEqual(self.reveals, 4)
            self.assertEqual(len(saved['saves']['procedures']), 5)
            self.assertIsNone(case['selected_policy_at_n16_primary'])

    def test_failed_durable_save_prevents_all_target_reveals(self):
        with tempfile.TemporaryDirectory() as directory:
            self.path, self.reveals = Path(directory) / 'saved.json', 0
            def broken_sink(value):
                raise OSError('disk full')
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.worker_run(broken_sink)
            self.assertEqual(self.reveals, 0)

    def test_archive_roundtrip_keeps_real_stream_and_calibrated_decision_validation(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(f.e, '_validate_phase_e_case'):
            root, context = Path(directory), context_fixture()
            result = block_fixture(root)
            path = root / 'blocks' / 'block-000.json'
            saved = f._save_block(path, result, context=context, partition_name=PHASE_F_REHEARSAL_PARTITION)
            self.assertEqual(saved, f.load_phase_f_block(path, context=context, partition_name=PHASE_F_REHEARSAL_PARTITION))
            rows = f.scored_records(result, context=context, partition_name=PHASE_F_REHEARSAL_PARTITION)
            self.assertEqual({k: len(v) for k, v in rows.items()}, {'0.0': 15, '0.05': 15, '0.1': 15, '0.2': 15})
            raw = path.read_bytes()
            with self.assertRaises(FileExistsError):
                f._save_block(path, result, context=context, partition_name=PHASE_F_REHEARSAL_PARTITION)
            self.assertEqual(path.read_bytes(), raw)
            with self.assertRaises(ValueError):
                f.load_phase_f_block(path, context=context, partition_name=PROSPECTIVE_RESERVED_PARTITION)

    def test_changed_post_reveal_raw_interval_cannot_rewrite_saved_decision(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(f.e, '_validate_phase_e_case'):
            result = block_fixture(directory)
            result['numerical_block']['cases'][0]['fixed_policy_results']['stop_now']['saved_before_reveal']['margin_envelope']['lower'] = -.8
            with self.assertRaisesRegex(ValueError, 'before-reveal decision'):
                f.validate_phase_f_block(result, context=context_fixture(), partition_name=PHASE_F_REHEARSAL_PARTITION)

    def test_missing_durable_file_and_reveal_failure_are_material(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(f.e, '_validate_phase_e_case'):
            result = block_fixture(directory)
            completed = {0: {'block_result': result}}
            f.before_path(Path(directory), 0, STAGE3_TRUTH_CONDITIONS[0]).unlink()
            with self.assertRaises(ValueError):
                f._check_before_files(Path(directory), context=context_fixture(),
                                      partition_name=PHASE_F_REHEARSAL_PARTITION, completed=completed)
            result['numerical_block']['cases'][0]['pipeline_failures'] = [{'stage': 'fixed_policy_reveal:stop_now'}]
            with self.assertRaisesRegex(ValueError, 'material target incident'):
                f.validate_phase_f_block(result, context=context_fixture(), partition_name=PHASE_F_REHEARSAL_PARTITION)

    def test_final_analysis_reloads_from_saved_raw_blocks_and_rejects_changed_endpoint(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(f.e, '_validate_phase_e_case'), patch.object(
            f, 'phase_f_context', return_value=context_fixture()):
            root, context = Path(directory), context_fixture()
            result = block_fixture(root)
            blockpath = root / 'blocks' / 'block-000.json'
            block = f._save_block(blockpath, result, context=context, partition_name=PHASE_F_REHEARSAL_PARTITION)
            inventory = [{'block': 0, 'path': 'blocks/block-000.json',
                          'sha256': hashlib.sha256(blockpath.read_bytes()).hexdigest(),
                          'scientific_block_digest': block['scientific_block_digest']}]
            rows = f.scored_records(result, context=context, partition_name=PHASE_F_REHEARSAL_PARTITION)
            value = f._analysis_payload(context, inventory, rows, partition_name=PHASE_F_REHEARSAL_PARTITION)
            value['performance'] = {}
            value['scientific_result_digest'] = f.d6._digest(f'{f.PHASE_F_DOMAIN}.evaluation_scientific_v1', {
                'context': context, 'partition': PHASE_F_REHEARSAL_PARTITION,
                'scientific_block_digests': [block['scientific_block_digest']],
                'calibrated_case_records_by_clearance': rows, 'primary_analysis': value['primary_analysis'],
                'secondary_analysis_by_clearance': value['secondary_analysis_by_clearance']})
            _, raw = f.d6._sealed_payload(value, domain=f'{f.PHASE_F_DOMAIN}.evaluation_v1')
            path = root / 'evaluation.json'
            path.write_bytes(raw)
            self.assertEqual(f.validate_phase_f_archive(path, repository_root=root)['primary_analysis']['independent_block_count'], 1)
            value['primary_analysis']['primary_success'] = True
            _, changed = f.d6._sealed_payload(value, domain=f'{f.PHASE_F_DOMAIN}.evaluation_v1')
            path.write_bytes(changed)
            with self.assertRaisesRegex(ValueError, 'does not reproduce'):
                f.validate_phase_f_archive(path, repository_root=root)

    def test_complete_disposable_execution_and_completed_resume_do_not_rerun(self):
        class ImmediateExecutor:
            def __init__(self, max_workers):
                self.max_workers = max_workers
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def submit(self, function, arguments):
                future = Future()
                try:
                    future.set_result(function(arguments))
                except Exception as error:
                    future.set_exception(error)
                return future
        with tempfile.TemporaryDirectory() as directory, patch.object(f.e, '_validate_phase_e_case'), patch.object(
            f, 'phase_f_context', return_value=context_fixture()), patch.object(f, 'ProcessPoolExecutor', ImmediateExecutor):
            root = Path(directory)
            with patch.object(f, '_run_block', side_effect=lambda args: block_fixture(root)) as worker:
                result = f._execute_partition(root=root, repository_root=root, context=context_fixture(),
                                              name=PHASE_F_REHEARSAL_PARTITION)
                worker.assert_called_once()
            with patch.object(f, '_run_block') as worker:
                second = f._execute_partition(root=root, repository_root=root, context=context_fixture(),
                                              name=PHASE_F_REHEARSAL_PARTITION)
                worker.assert_not_called()
            self.assertEqual(result, second)
            seed = f.load_phase_f_block(root / 'blocks' / 'block-000.json', context=context_fixture(),
                                        partition_name=PHASE_F_REHEARSAL_PARTITION)
            probe = f.constructed_archive_probe({'block_result': seed['block_result'],
                'calibrated_records': result['calibrated_case_records_by_clearance']['0.0']}, output_directory=root)
            self.assertTrue(probe['roundtrip_passed'])
            self.assertEqual(probe['constructed_analysis_records'], 1500)
            self.assertEqual(probe['scientific_use'], 'constructed_resource_evidence_only')


class PhaseFCommandTests(unittest.TestCase):
    def test_missing_reserved_gate_is_rejected_before_any_worker(self):
        args = ['--execute-reserved', '--output-directory', '/unused',
                '--calibration-archive', '/unused/calibration.json', '--calibration-source-root', '/unused/source']
        with patch.object(cli, '_clean_revision', return_value='a'*40), patch.object(
            cli, '_validate_launch_environment'), patch.object(f, 'run_phase_f_reserved') as run, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                cli.main(args)
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()

    def test_readonly_validation_reports_actual_partition_without_writing_evidence(self):
        class Monitor:
            peak_bytes, sample_count, error = 100, 2, None
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        resource_module = 'thermotwin.studies.operating_decision_prospective_phase_d_resources'
        with tempfile.TemporaryDirectory() as directory, patch.object(cli, '_clean_revision', return_value='a'*40), patch.object(
            cli, '_validate_launch_environment'), patch.object(f, 'validate_calibration_evidence'), patch.object(
            f, 'validate_phase_f_archive', return_value={'partition': PROSPECTIVE_RESERVED_PARTITION,
                                                       'scientific_result_digest': 'd'*64}), patch(
            resource_module + '.ProcessTreeMonitor', Monitor), patch(resource_module + '._process_tree_rss_bytes', return_value=100):
            output = io.StringIO()
            with redirect_stdout(output):
                cli.main(['--validate-evaluation', '/unused/evaluation.json', '--output-directory', directory,
                          '--calibration-archive', '/unused/calibration.json', '--calibration-source-root', '/unused/source'])
            resources = json.loads(output.getvalue().splitlines()[-1])['resources']
            self.assertEqual(resources['partition'], PROSPECTIVE_RESERVED_PARTITION)
            self.assertTrue(resources['resource_gate_passed'])
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
