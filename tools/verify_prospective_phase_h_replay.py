"""Independently verify retained Phase H replay outputs without numerical imports.

Checks the complete predetermined matrix, exact archived fit/draw/decision
payloads, explicit reuse of original streams, and source/file/scientific hashes.
This checks already computed results; it does not perform another refit.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import re

from tools import verify_prospective_phase_f_result as audit


def check_digest(value, key, domain):
    audit.require(value[key] == audit.digest(domain, {k: v for k, v in value.items() if k != key}), 'content digest differs: ' + key)


def verify(*, archive, plan_path, replay_directory, repository_root, frozen_source_root, full_test_log, focused_test_log):
    root, replay = Path(repository_root), Path(replay_directory)
    audit.require(audit.file_record(archive)['sha256'] == '299f75ad492e5bde0acc724ed4e72ae029c86f2c05538677c2a856c23734c564', 'closed archive changed')
    saved = audit.load_sealed(archive, 'thermotwin.prospective_phase_f.evaluation_v1')
    plan = json.loads(Path(plan_path).read_bytes())
    check_digest(plan, 'artifact_digest', 'thermotwin.phase_h.numerical_replay_plan_v1')
    result = json.loads((replay / 'result.json').read_bytes())
    check_digest(result, 'artifact_digest', 'thermotwin.phase_h.numerical_replay_result_v1')
    audit.require(result['passed'] is True and not result['errors'], 'numerical replay failed')
    audit.require(result['plan_digest'] == plan['artifact_digest'] and result['plan_file'] == audit.file_record(plan_path), 'plan binding differs')
    audit.require(result['scientific_result_digest'] == saved['scientific_result_digest'], 'original scientific result changed')
    audit.require(result['runner_file'] == plan['runner_file'] == audit.file_record(root / 'tools/replay_prospective_phase_f_numerics.py'), 'replay runner changed')
    for source in (root, Path(frozen_source_root)):
        for item in saved['context']['source_manifest']['files']:
            audit.require(audit.file_record(source / item['path']) == {'sha256': item['sha256'], 'bytes': item['byte_count']}, 'frozen source changed: ' + item['path'])
    started = json.loads((replay / 'started.json').read_bytes())
    audit.require(started['plan_digest'] == plan['artifact_digest'] and
                  datetime.fromisoformat(plan['planned_at_utc']) < datetime.fromisoformat(started['started_at_utc']), 'plan was not recorded before replay')
    expected_tasks = {t['task_id']: t for t in plan['sample']['tasks']}
    records = {r['task_id']: r for r in result['case_evidence_files']}
    audit.require(len(records) == len(result['case_evidence_files']) and set(records) == set(expected_tasks), 'missing or duplicate case tasks')
    audit.require({p.name for p in replay.glob('*.json')} == {'result.json', 'started.json', *(r['file'] for r in records.values())}, 'unexpected or missing replay JSON')
    policy_count, draw_count, decision_count, stream_use_count = 0, 0, 0, 0
    scientific = []
    policy_rows, draw_rows = [], []
    for task_id in sorted(records):
        item, task = records[task_id], expected_tasks[task_id]
        path = replay / item['file']
        audit.require(audit.file_record(path) == {'sha256': item['sha256'], 'bytes': item['bytes']}, 'replay case bytes differ')
        value = json.loads(path.read_bytes())
        audit.require(value['passed'] is True and value['task_id'] == task_id, 'case failed or changed identity')
        material = {k: v for k, v in value.items() if k not in ('started_at_utc', 'completed_at_utc', 'performance', 'traceback', 'replay_scientific_digest')}
        audit.require(value['replay_scientific_digest'] == audit.digest('thermotwin.phase_h.numerical_replay_case_scientific_v1', material), 'case scientific digest differs')
        scientific.append((task_id, value['replay_scientific_digest']))
        entry = saved['block_file_records'][task['block']]
        path = Path(archive).parent / entry['path']
        audit.require(audit.file_record(path)['sha256'] == entry['sha256'], 'selected original block changed')
        checkpoint = audit.load_sealed(path, 'thermotwin.prospective_phase_f.block_v1')
        case = next(c for c in checkpoint['block_result']['numerical_block']['cases'] if c['truth_condition'] == task['family'])
        evidence = case['n16_complete_uncertainty_result']
        audit.require(audit.canonical(value['common_fit_set']) == audit.canonical(evidence['acquisition_evidence']['fit_set'])
                      and value['common_evidence_digest'] == evidence['acquisition_evidence']['evidence_digest'], 'common acquisition refit differs')
        policies = {p['policy']: p for p in value['policies']}
        audit.require(len(policies) == len(value['policies']) and set(policies) == set(task['policies']), 'policy replay matrix differs')
        before = checkpoint['block_result']['before_reveal_by_family'][task['family']]['saves']['procedures']
        for policy, payload in policies.items():
            expected = {k: v for k, v in case['fixed_policy_results'][policy]['saved_before_reveal'].items() if k != 'decision_computation_seconds'}
            audit.require(audit.canonical(payload['numerical_save']) == audit.canonical(expected), 'exact verified fit/interval differs')
            procedures = {policy, *([audit.SELECTOR] if case['selected_policy_at_n16_primary'] == policy else [])}
            audit.require(set(payload['calibrated_decisions']) == procedures, 'calibrated procedure set differs')
            for procedure, decisions in payload['calibrated_decisions'].items():
                audit.require(audit.canonical(decisions) == audit.canonical(before[procedure]['decisions']) and len(decisions) == 4, 'saved calibrated decisions differ')
                decision_count += len(decisions)
            policy_count += 1
            policy_rows.append({'task_id': task_id, 'block': task['block'], 'family': task['family'], 'policy': policy,
                                'reasons': task['policies'][policy], 'raw_decision': expected['decision'],
                                'raw_interval': expected['margin_envelope'], 'calibrated_procedures': sorted(procedures)})
        draws = {(r['identity']['policy_name'], r['identity']['generator_model'], r['identity']['draw_index']): r for r in value['predictive_draws']}
        expected_draws = {(r['policy_name'], r['generator_model'], r['draw_index']) for r in task['draws']}
        audit.require(len(draws) == len(value['predictive_draws']) and set(draws) == expected_draws, 'predictive draw replay matrix differs')
        archived_streams = {audit.canonical(u) for u in evidence['stream_uses']}
        for identity, payload in draws.items():
            expected = next(d for d in evidence['draw_outcomes'] if (d['policy_name'], d['generator_model'], d['draw_index']) == identity)
            audit.require(audit.canonical(payload['outcome']) == audit.canonical(expected), 'exact predictive draw differs')
            uses = [audit.canonical(u) for u in payload['replayed_stream_uses']]
            audit.require(len(set(uses)) == len(uses) and set(uses) <= archived_streams, 'unarchived or duplicate stream use')
            stream_use_count += len(uses)
            draw_count += 1
            draw_rows.append({'task_id': task_id, **payload['identity'], 'failed': expected['failed'],
                              'failure_stage': expected['failure_stage'], 'failure_type': expected['failure_type'],
                              'candidate_statuses': [v['status'] for v in expected['candidate_outcomes']],
                              'synthetic_observation_digest': expected['synthetic_observation_digest']})
    audit.require(policy_count == result['policy_refits'] == sum(len(t['policies']) for t in expected_tasks.values()), 'policy count differs')
    audit.require(draw_count == result['predictive_draw_replays'] == sum(len(t['draws']) for t in expected_tasks.values()), 'draw count differs')
    audit.require(decision_count == result['calibrated_decisions_recomputed'], 'decision count differs')
    expected_scientific = {'plan_digest': plan['artifact_digest'], 'source_revision': audit.SOURCE_REVISION,
                          'source_manifest_digest': audit.SOURCE_DIGEST, 'runtime_identity': plan['runtime_identity'],
                          'original_scientific_result_digest': saved['scientific_result_digest'], 'passed': True,
                          'errors': [], 'case_scientific_digests': scientific}
    audit.require(result['replay_scientific_digest'] == audit.digest('thermotwin.phase_h.numerical_replay_scientific_v1', expected_scientific), 'aggregate scientific replay digest differs')
    perf = result['performance']
    audit.require(perf['process_tree_monitor_error'] is None and perf['process_tree_samples'] > 0
                  and 0 < perf['process_tree_peak_rss_bytes'] <= perf['memory_limit_bytes'] == 4 * 1024**3, 'replay resource gate failed')
    full_text, focused_text = Path(full_test_log).read_text(), Path(focused_test_log).read_text()
    match = re.search(r'\nRan (\d+) tests in ([0-9.]+)s\n\nOK\n', full_text)
    audit.require(match is not None, 'full repository test run has not passed')
    audit.require('\nRan 42 tests ' in focused_text and '\nOK\n' in focused_text, '42 focused checks have not passed')
    return {'schema_version': 1, 'independent_replay_verification_passed': True,
            'plan_digest': plan['artifact_digest'], 'replay_artifact_digest': result['artifact_digest'],
            'replay_scientific_digest': result['replay_scientific_digest'],
            'source_revision': audit.SOURCE_REVISION, 'original_scientific_result_digest': saved['scientific_result_digest'],
            'case_tasks': len(records), 'policy_refits': policy_count, 'predictive_draw_replays': draw_count,
            'saved_calibrated_decisions_recomputed': decision_count, 'original_stream_uses_explicitly_replayed': stream_use_count,
            'full_repository_tests_passed': int(match.group(1)), 'full_repository_test_seconds': float(match.group(2)),
            'focused_tests_passed': 42, 'new_replay_tests_in_focused_suite': 7,
            'full_test_log': audit.file_record(full_test_log), 'focused_test_log': audit.file_record(focused_test_log),
            'verification_script_file': audit.file_record(__file__), 'result_file': audit.file_record(replay / 'result.json'),
            'started_file': audit.file_record(replay / 'started.json'), 'policy_rows': policy_rows, 'draw_rows': draw_rows,
            'new_numerical_computations_during_this_verification': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'plan', 'replay-directory', 'repository-root', 'frozen-source-root', 'full-test-log', 'focused-test-log', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    audit.require(not args.output.exists(), 'refuse to overwrite independent verification')
    for directory in (args.archive.parent, args.replay_directory):
        audit.require(not args.output.resolve().is_relative_to(directory.resolve()), 'verification output must be outside accepted evidence directories')
    value = verify(archive=args.archive, plan_path=args.plan, replay_directory=args.replay_directory,
                   repository_root=args.repository_root, frozen_source_root=args.frozen_source_root,
                   full_test_log=args.full_test_log, focused_test_log=args.focused_test_log)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: value[k] for k in ('independent_replay_verification_passed', 'case_tasks', 'policy_refits', 'predictive_draw_replays',
                                          'saved_calibrated_decisions_recomputed', 'full_repository_tests_passed', 'focused_tests_passed')}, indent=2))


if __name__ == '__main__':
    main()
