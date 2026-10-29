"""Plan and execute a bounded computational replay of the CLOSED Phase F archive.

This is a post hoc implementation check, not a new scientific partition. Freeze
the deterministic target list with `plan`, then use `run` with the same tool and
the original clean numerical source. Outputs must be new and outside the raw
archive. A mismatch is retained and fails the check; there are no automatic
retries, replacement targets, or relaxed numerical tolerances.
"""

import argparse
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

from tools import verify_prospective_phase_f_result as audit


EVALUATION_SHA256 = '299f75ad492e5bde0acc724ed4e72ae029c86f2c05538677c2a856c23734c564'
PLAN_DOMAIN = 'thermotwin.phase_h.numerical_replay_plan_v1'
RESULT_DOMAIN = 'thermotwin.phase_h.numerical_replay_result_v1'
CASE_DOMAIN = 'thermotwin.phase_h.numerical_replay_case_scientific_v1'


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


class ReplayDifference(ValueError):
    def __init__(self, actual, expected, label):
        super().__init__('exact replay differs: ' + label)
        self.details = {'label': label, 'actual': actual, 'expected': expected}


def exact(actual, expected, label):
    if audit.canonical(actual) != audit.canonical(expected):
        raise ReplayDifference(actual, expected, label)


def numerical_save(value):
    # Exclude precisely the single elapsed-time field, not arbitrary keys.
    return {k: v for k, v in value.items() if k != 'decision_computation_seconds'}


def worker_scientific_payload(value):
    return {k: v for k, v in value.items() if k not in (
        'started_at_utc', 'completed_at_utc', 'performance', 'traceback', 'replay_scientific_digest')}


def verify_sources(saved, source_root):
    root = Path(source_root).resolve()
    for item in saved['context']['source_manifest']['files']:
        exact(audit.file_record(root / item['path']),
              {'sha256': item['sha256'], 'bytes': item['byte_count']}, item['path'])
    return root


def read_archive(archive, source_root, validation_record):
    audit.require(audit.file_record(archive)['sha256'] == EVALUATION_SHA256, 'not the accepted closed archive')
    saved, rows = audit.reconstruct(archive, source_root, validation_record)
    cases = []
    for entry in saved['block_file_records']:
        path = Path(archive).parent / entry['path']
        audit.require(audit.file_record(path)['sha256'] == entry['sha256'], 'checkpoint changed during extraction')
        cases.extend(json.loads(path.read_bytes())['block_result']['numerical_block']['cases'])
    return saved, rows['0.0'], cases


def choose_targets(rows, cases):
    """Freeze audit coverage using saved identities and labels, never replay results."""
    selector = [r for r in rows if r['procedure'] == audit.SELECTOR]
    order = {f: i for i, f in enumerate(audit.FAMILIES)}
    key = lambda r: (r['block'], order[r['truth_family']])
    tasks = {}

    def task(block, family):
        return tasks.setdefault((block, family), {'block': block, 'family': family, 'policies': {}, 'draws': []})

    def policy(row, name, reason):
        reasons = task(row['block'], row['truth_family'])['policies'].setdefault(name, [])
        if reason not in reasons:
            reasons.append(reason)

    absent = []
    for family in audit.FAMILIES:
        group = [r for r in selector if r['truth_family'] == family]
        audit.require(bool(group), 'missing family in replay sampling frame')
        for action in audit.POLICIES:
            matches = [r for r in group if r['selected_action'] == action]
            row = min(matches or group, key=key)
            policy(row, action, 'first_selected_family_action' if matches else 'unselected_fixed_fallback_for_absent_cell')
            if not matches:
                absent.append({'family': family, 'action': action})
    for decision in ('approve', 'reject', 'insufficient_evidence'):
        group = [r for r in selector if r['decision'] == decision]
        audit.require(bool(group), 'missing decision label in replay frame')
        row = min(group, key=lambda r: (min(abs(r['final_interval']['lower']), abs(r['final_interval']['upper'])), *key(r)))
        policy(row, row['selected_action'], 'nearest_saved_zero_endpoint_' + decision)

    all_draws = []
    for case in cases:
        for draw in case['n16_complete_uncertainty_result']['draw_outcomes']:
            all_draws.append((case['block'], case['truth_condition'], draw))
    all_draws.sort(key=lambda r: (r[0], order[r[1]], audit.POLICIES.index(r[2]['policy_name']),
                                 r[2]['generator_model'], r[2]['draw_index']))

    def first_draw(predicate, reason):
        matches = [r for r in all_draws if predicate(r[2])]
        audit.require(bool(matches), 'missing predictive replay stratum: ' + reason)
        block, family, draw = matches[0]
        identity = {k: draw[k] for k in ('policy_name', 'generator_model', 'draw_index')}
        targets = task(block, family)['draws']
        existing = next((r for r in targets if all(r[k] == v for k, v in identity.items())), None)
        if existing is None:
            existing = {**identity, 'reasons': []}
            targets.append(existing)
        existing['reasons'].append(reason)

    for action in audit.POLICIES[1:]:
        first_draw(lambda d, action=action: d['policy_name'] == action and not d['failed']
                   and not d['initially_admissible_became_inadmissible']
                   and not d['initially_excluded_became_admissible'], 'stable_predictive_' + action)
    first_draw(lambda d: d['failed'], 'whole_predictive_draw_failure')
    for status in ('fit_reached_bound', 'optimizer_not_converged'):
        first_draw(lambda d, status=status: not d['failed'] and bool(d['initially_admissible_became_inadmissible'])
                   and any(v['status'] == status for v in d['candidate_outcomes']), 'usable_candidate_exclusion_' + status)
    result = [tasks[k] for k in sorted(tasks, key=lambda k: (k[0], order[k[1]]))]
    for index, item in enumerate(result):
        item['task_id'] = f'case-{index:02d}-block-{item["block"]:03d}-{item["family"]}'
    return {'tasks': result, 'absent_selected_family_action_cells': absent,
            'primary_pipeline_failure_count': sum(r['pipeline_failure'] for r in selector),
            'available_whole_predictive_failure_count': sum(d['failed'] for _, _, d in all_draws)}


def make_plan(args):
    audit.require(not args.output.exists(), 'refuse to overwrite a replay plan')
    audit.require(not args.output.resolve().is_relative_to(args.archive.parent.resolve()), 'plan must be outside closed raw archive')
    saved, rows, cases = read_archive(args.archive, args.repository_root, args.validation_record)
    verify_sources(saved, args.frozen_source_root)
    targets = choose_targets(rows, cases)
    value = {'schema_version': 1, 'version': 'closed_phase_f_bounded_numerical_replay_v1',
             'planned_at_utc': now(), 'scientific_use': 'post_hoc_implementation_validation_only',
             'selection_is_after_original_outcomes_but_before_any_new_numerical_replay': True,
             'source_revision': audit.SOURCE_REVISION, 'source_manifest_digest': audit.SOURCE_DIGEST,
             'scientific_result_digest': saved['scientific_result_digest'],
             'evaluation_file': audit.file_record(args.archive), 'validation_file': audit.file_record(args.validation_record),
             'runner_file': audit.file_record(__file__), 'runtime_identity': saved['context']['runtime_identity'],
             'math_environment': saved['context']['protocol']['complete_calibrated_rule']['math_environment'],
             'sample': targets,
             'rule': {'family_action': 'first block with each selected family/action; if absent replay that fixed action on first family block and record fallback',
                      'boundary': 'minimum absolute final-interval endpoint for each selector decision label; ties block then frozen family order',
                      'draws': 'first stable draw per measurement action, first whole failure, first usable bound and nonconvergence candidate exclusion; ties block/family/action/source/index',
                      'comparison': 'exact canonical numerical payloads; only raw decision_computation_seconds excluded',
                      'no_retuning_no_replacement_targets_no_automatic_retries': True,
                      'replay_reuses_original_semantic_keys_explicitly': True,
                      'full_n16_rescoring_or_new_target_truth_simulation': False},
             'limits': {'workers': 2, 'process_tree_memory_bytes': 4 * 1024**3,
                        'task_timeout_seconds': 1800, 'total_timeout_seconds': 5400},
             'new_cohort_or_partition_authorized': False}
    value['artifact_digest'] = audit.digest(PLAN_DOMAIN, value)
    write_new(args.output, value)
    print(json.dumps({'plan': str(args.output), 'artifact_digest': value['artifact_digest'],
                      'case_tasks': len(targets['tasks']),
                      'policy_refits': sum(len(t['policies']) for t in targets['tasks']),
                      'predictive_draws': sum(len(t['draws']) for t in targets['tasks'])}, indent=2), flush=True)


def frozen_imports(source_root, manifest):
    root = Path(source_root).resolve()
    sys.path.insert(0, str(root))
    modules = {name: importlib.import_module('thermotwin.' + name) for name in (
        'core.controls', 'studies.operating_decision', 'studies.operating_decision_realism',
        'studies.sensor_model_discrimination', 'studies.operating_decision_replication',
        'studies.operating_decision_prospective_uncertainty', 'studies.operating_decision_prospective_random_streams',
        'studies.operating_decision_prospective_phase_f_execution',
        'studies.operating_decision_prospective_phase_e_calibration',
        'studies.operating_decision_prospective_pilot', 'studies.operating_decision_prospective_phase_d_resources')}
    paths = {item['path'] for item in manifest['files']}
    for name, module in list(sys.modules.items()):
        if name == 'thermotwin' or name.startswith('thermotwin.'):
            path = Path(module.__file__).resolve()
            audit.require(path.is_relative_to(root) and path.relative_to(root).as_posix() in paths,
                          'import outside frozen numerical closure: ' + name)
    return modules


def decode_case(payload, modules):
    c = modules['core.controls']
    o = modules['studies.operating_decision']
    r = modules['studies.operating_decision_realism']
    s = modules['studies.sensor_model_discrimination']

    def current(value):
        return c.PiecewiseConstantCurrent(transition_times=tuple(value['transition_times']), values=tuple(value['values']))

    def regime(value):
        return o.OperatingRegime(name=value['name'], phase=value['phase'], current=current(value['current']), channels=tuple(value['channels']))

    def run(value):
        obs = value['observations']
        return r.RealisticOperatingRun(regime(value['regime']),
                                      s.ObservableRun(obs['name'], current(obs['current']), tuple(s.ObservableValue(**v) for v in obs['values'])),
                                      r.RunInstrumentation(**value['instrumentation']))

    policy = next(p for p in o.default_fixed_policies() if p.name == payload['policy']['name'])
    return r.RealisticBlindedOperatingCase(case_id=r.Stage3CaseId(**payload['case_id']), policy=policy,
                                         acquisition_runs=tuple(run(v) for v in payload['acquisition_runs']),
                                         verification_run=run(payload['verification_run']), final_regime=regime(payload['final_regime']),
                                         final_instrumentation=r.RunInstrumentation(**payload['final_instrumentation']))


def check_runtime(saved, source_root):
    expected = saved['context']['runtime_identity']
    observed = {**expected, 'python_implementation': platform.python_implementation(), 'python_version': platform.python_version(),
                'operating_system': platform.system(), 'machine': platform.machine(),
                'scientific_packages': {name: importlib.metadata.version(name) for name in expected['scientific_packages']},
                'requirements_sha256': audit.file_record(Path(source_root) / expected['requirements_path'])['sha256']}
    exact(observed, expected, 'frozen runtime identity')
    for name, value in saved['context']['protocol']['complete_calibrated_rule']['math_environment'].items():
        audit.require(os.environ.get(name) == value, 'frozen math environment differs: ' + name)


def worker(args):
    plan = json.loads(args.plan.read_bytes())
    material = {k: v for k, v in plan.items() if k != 'artifact_digest'}
    audit.require(audit.digest(PLAN_DOMAIN, material) == plan['artifact_digest'], 'worker plan digest differs')
    task = next(t for t in plan['sample']['tasks'] if t['task_id'] == args.task_id)
    started, cpu = time.perf_counter(), time.process_time()
    result = {'task_id': task['task_id'], 'started_at_utc': now(), 'passed': False}
    try:
        exact(audit.file_record(__file__), plan['runner_file'], 'planned runner')
        saved = json.loads(args.archive.read_bytes())
        exact(audit.file_record(args.archive), plan['evaluation_file'], 'raw evaluation')
        verify_sources(saved, args.frozen_source_root)
        check_runtime(saved, args.frozen_source_root)
        m = frozen_imports(args.frozen_source_root, saved['context']['source_manifest'])
        o, r, u, rng, f, cal, p = (m['studies.' + name] for name in (
            'operating_decision', 'operating_decision_replication', 'operating_decision_prospective_uncertainty',
            'operating_decision_prospective_random_streams', 'operating_decision_prospective_phase_f_execution',
            'operating_decision_prospective_phase_e_calibration', 'operating_decision_prospective_pilot'))
        entry = saved['block_file_records'][task['block']]
        path = args.archive.parent / entry['path']
        audit.require(audit.file_record(path)['sha256'] == entry['sha256'], 'selected checkpoint changed')
        checkpoint = json.loads(path.read_bytes())
        case = next(c for c in checkpoint['block_result']['numerical_block']['cases'] if c['truth_condition'] == task['family'])
        # Reconstruct observed inputs only. No physical-truth or cohort generator is called.
        common = decode_case(case['fixed_policy_results']['stop_now']['case'], m)
        config = f.e._physical_config(f.phase_f_partition(saved['partition']))
        evidence = u.prepare_prospective_acquisition_evidence(common.acquisition_runs[0], common.final_regime, config)
        original = case['n16_complete_uncertainty_result']['acquisition_evidence']
        regenerated = {'schema_version': u.PROSPECTIVE_UNCERTAINTY_SCHEMA_VERSION,
                       'protocol_version': u.PROSPECTIVE_UNCERTAINTY_PROTOCOL_VERSION,
                       'common_initial_run': u._run_payload(evidence.common_initial_run),
                       'fit_set': u._fit_set_payload(evidence.fit_set), 'snapshot': u._snapshot_payload(evidence.snapshot),
                       'final_regime': u._current_payload(evidence.final_regime), 'evidence_digest': evidence.evidence_digest,
                       'physical_config': u._physical_config_payload(config), 'physical_protocol_digest': evidence.physical_protocol_digest}
        exact(regenerated, original, 'common acquisition evidence')
        policies, draws = [], []
        before = checkpoint['block_result']['before_reveal_by_family'][task['family']]['saves']['procedures']
        rule = saved['context']['protocol']['complete_calibrated_rule']
        offsets = rule['design']['selector_rule']['development_offsets']['values']
        for policy in task['policies']:
            recorded = case['fixed_policy_results'][policy]
            observed_case = decode_case(recorded['case'], m)
            exact(p._case_payload(observed_case), recorded['case'], 'observed input round trip')
            replayed = p._saved_payload(r.decide_corrected_blinded_case(observed_case, config=config))
            expected = recorded['saved_before_reveal']
            exact(numerical_save(replayed), numerical_save(expected), 'verified fit/interval/' + policy)
            inp = {'raw_interval': replayed['margin_envelope'], 'development_offset_kelvin': offsets[policy],
                   'verification_succeeded': any(v['passed'] for v in replayed['verifications']), 'pipeline_failure': False}
            procedures = [policy] + ([audit.SELECTOR] if case['selected_policy_at_n16_primary'] == policy else [])
            decisions = {}
            for procedure in procedures:
                exact(inp, before[procedure]['input'], 'blinded calibrated inputs')
                decisions[procedure] = [cal.apply_calibrated_decision(inp, rule['final_corrections_kelvin'][procedure],
                                                                      decision_clearance_kelvin=clearance) for clearance in audit.CLEARANCES]
                exact(decisions[procedure], before[procedure]['decisions'], 'calibrated decisions/' + procedure)
            policies.append({'policy': policy, 'numerical_save': numerical_save(replayed), 'calibrated_decisions': decisions})
        namespace = rng.ProspectiveRandomStreamNamespace(campaign=rule['design']['campaign'], partition=saved['partition'],
                                                        block=task['block'], acquisition_evidence_digest=evidence.evidence_digest)
        fit_by_model = {fit.model_name: fit for fit in evidence.fit_set.fits}
        for target in task['draws']:
            expected = next(d for d in case['n16_complete_uncertainty_result']['draw_outcomes']
                            if all(d[k] == target[k] for k in ('policy_name', 'generator_model', 'draw_index')))
            registry = rng.ProspectiveRandomStreamRegistry()
            fit = fit_by_model[target['generator_model']]
            cfg = u.ProspectiveUncertaintyConfig(draw_count=16, max_unstable_draws_per_source_action=1)
            parameters = u._generator_parameter_draw(fit, namespace, target['draw_index'], config, cfg, registry)
            probe = u._face_probe_draw(namespace, target['generator_model'], target['draw_index'], config, cfg, registry)
            exact(list(parameters), expected['generator_log_offsets'], 'predictive parameter draw')
            probe = probe if target['policy_name'] == o.FIXED_FACE_TEMPERATURE else None
            exact(list(probe) if probe else None, expected['face_probe_log_offsets'], 'predictive probe draw')
            policy = next(v for v in o.default_fixed_policies() if v.name == target['policy_name'])
            baseline = evidence.snapshot.provisional_margin_envelope.upper - evidence.snapshot.provisional_margin_envelope.lower
            draw = u._evaluate_hypothetical_draw(policy=policy, generator_fit=fit, generator_offsets=parameters,
                                                face_probe_offsets=probe, draw_index=target['draw_index'], evidence=evidence,
                                                namespace=namespace, physical_config=config, registry=registry, baseline_width=baseline)
            actual = p._strict_json_value(u._draw_payload(draw))
            exact(actual, expected, 'predictive refit/outcome')
            registry.audit().assert_clean()
            stream_uses = [u._stream_use_payload(v) for v in registry.uses]
            archived_uses = {audit.canonical(v) for v in case['n16_complete_uncertainty_result']['stream_uses']}
            audit.require(all(audit.canonical(v) in archived_uses for v in stream_uses), 'replay used an unarchived semantic stream')
            draws.append({'identity': target, 'outcome': actual, 'replayed_stream_uses': stream_uses})
        result.update(passed=True, common_evidence_digest=evidence.evidence_digest,
                      common_fit_set=regenerated['fit_set'], policies=policies, predictive_draws=draws,
                      selected_n16_policy_replayed=False)
    except Exception as error:
        result.update(error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        if isinstance(error, ReplayDifference):
            result['mismatch'] = error.details
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['performance'] = {'wall_seconds': time.perf_counter() - started, 'cpu_seconds': time.process_time() - cpu,
                             'process_peak_rss_bytes': peak if sys.platform == 'darwin' else peak * 1024}
    result['completed_at_utc'] = now()
    result['replay_scientific_digest'] = audit.digest(CASE_DOMAIN, worker_scientific_payload(result))
    write_new(args.output, result)
    return 0 if result['passed'] else 1


def run(args):
    audit.require(not args.output_directory.exists(), 'refuse to reuse a replay output directory')
    audit.require(not args.output_directory.resolve().is_relative_to(args.archive.parent.resolve()), 'replay output must be outside closed raw archive')
    plan = json.loads(args.plan.read_bytes())
    material = {k: v for k, v in plan.items() if k != 'artifact_digest'}
    audit.require(audit.digest(PLAN_DOMAIN, material) == plan['artifact_digest'], 'replay plan digest differs')
    exact(audit.file_record(__file__), plan['runner_file'], 'planned runner')
    saved, rows, cases = read_archive(args.archive, args.repository_root, args.validation_record)
    exact(choose_targets(rows, cases), plan['sample'], 'predetermined targets')
    exact(audit.file_record(args.validation_record), plan['validation_file'], 'original validation record')
    verify_sources(saved, args.frozen_source_root)
    check_runtime(saved, args.frozen_source_root)
    m = frozen_imports(args.frozen_source_root, saved['context']['source_manifest'])
    resources = m['studies.operating_decision_prospective_phase_d_resources']
    # Preflight precedes output creation and all replay fits. Permission failure
    # can be retried outside the sandbox without repeating numerical work.
    resources._process_tree_rss_bytes(os.getpid())
    args.output_directory.mkdir(parents=True)
    started, start_cpu = time.perf_counter(), time.process_time()
    write_new(args.output_directory / 'started.json', {'started_at_utc': now(), 'plan_digest': plan['artifact_digest'],
                                                      'plan_file': audit.file_record(args.plan), 'runner_file': plan['runner_file']})
    waiting, active, completed, errors = list(plan['sample']['tasks']), {}, [], []
    monitor = resources.ProcessTreeMonitor(interval_seconds=1.)
    with monitor:
        while waiting or active:
            if (monitor.error or monitor.peak_bytes > plan['limits']['process_tree_memory_bytes']
                    or time.perf_counter() - started > plan['limits']['total_timeout_seconds']):
                errors.append('resource monitor/limit/total timeout: ' + str(monitor.error))
                break
            while waiting and len(active) < plan['limits']['workers'] and not errors:
                task = waiting.pop(0)
                output = args.output_directory / (task['task_id'] + '.json')
                log = (args.output_directory / (task['task_id'] + '.log')).open('xb')
                command = [sys.executable, '-u', '-m', 'tools.replay_prospective_phase_f_numerics', '_worker',
                           '--archive', str(args.archive.resolve()), '--plan', str(args.plan.resolve()),
                           '--frozen-source-root', str(args.frozen_source_root.resolve()), '--task-id', task['task_id'], '--output', str(output.resolve())]
                process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1], stdout=log, stderr=subprocess.STDOUT)
                active[task['task_id']] = (process, log, output, time.perf_counter())
            for task_id, (process, log, output, task_started) in list(active.items()):
                if time.perf_counter() - task_started > plan['limits']['task_timeout_seconds']:
                    process.terminate()
                    errors.append('task timeout: ' + task_id)
                if process.poll() is None:
                    continue
                log.close()
                del active[task_id]
                record = json.loads(output.read_bytes()) if output.exists() else {'passed': False, 'error': 'missing worker result'}
                if output.exists():
                    exact(record['replay_scientific_digest'], audit.digest(CASE_DOMAIN, worker_scientific_payload(record)), 'worker scientific digest')
                completed.append({'task_id': task_id, 'file': output.name, **audit.file_record(output)} if output.exists()
                                 else {'task_id': task_id, 'file': None})
                if process.returncode != 0 or not record['passed']:
                    errors.append(task_id + ': ' + record.get('error', 'nonzero worker status'))
                print(f'Numerical replay: {len(completed)}/{len(plan["sample"]["tasks"])} cases saved; {task_id}; passed={record["passed"]}', flush=True)
            if errors:
                break
            time.sleep(.5)
        for process, log, _, _ in active.values():
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
    verify_sources(saved, args.repository_root)
    verify_sources(saved, args.frozen_source_root)
    exact(audit.file_record(args.archive), plan['evaluation_file'], 'closed evaluation after replay')
    details = [json.loads((args.output_directory / r['file']).read_bytes()) for r in completed if r['file']]
    passed = not errors and not monitor.error and len(completed) == len(plan['sample']['tasks'])
    result = {'schema_version': 1, 'version': 'closed_phase_f_bounded_numerical_replay_result_v1',
              'completed_at_utc': now(), 'passed': passed, 'scientific_use': plan['scientific_use'],
              'plan_digest': plan['artifact_digest'], 'plan_file': audit.file_record(args.plan), 'runner_file': plan['runner_file'],
              'source_revision': audit.SOURCE_REVISION, 'source_manifest_digest': audit.SOURCE_DIGEST,
              'scientific_result_digest': saved['scientific_result_digest'], 'runtime_identity': plan['runtime_identity'],
              'evaluation_file': plan['evaluation_file'], 'frozen_numerical_files_unchanged_in_both_source_roots': 117,
              'completed_case_tasks': len(completed), 'planned_case_tasks': len(plan['sample']['tasks']),
              'policy_refits': sum(len(v.get('policies', [])) for v in details),
              'predictive_draw_replays': sum(len(v.get('predictive_draws', [])) for v in details),
              'calibrated_decisions_recomputed': sum(len(d) for v in details for p in v.get('policies', []) for d in p['calibrated_decisions'].values()),
              'exact_scientific_equality_required': True, 'only_excluded_fit_field': 'decision_computation_seconds',
              'sample': plan['sample'], 'case_evidence_files': sorted(completed, key=lambda r: r['task_id']),
              'errors': errors, 'full_n16_action_selection_rerun': False,
              'full_300_case_numerical_replay': False, 'original_primary_success_changed': False,
              'new_reserved_observations_or_target_truth_generated': False,
              'same_saved_predictive_streams_explicitly_replayed': True,
              'performance': {'wall_seconds': time.perf_counter() - started,
                              'worker_cpu_seconds': sum(v['performance']['cpu_seconds'] for v in details),
                              'coordinator_cpu_seconds': time.process_time() - start_cpu,
                              'process_tree_peak_rss_bytes': monitor.peak_bytes, 'process_tree_samples': monitor.sample_count,
                              'process_tree_monitor_error': monitor.error, 'memory_limit_bytes': plan['limits']['process_tree_memory_bytes'],
                              'worker_count': plan['limits']['workers']}}
    result['replay_scientific_digest'] = audit.digest('thermotwin.phase_h.numerical_replay_scientific_v1', {
        'plan_digest': plan['artifact_digest'], 'source_revision': audit.SOURCE_REVISION,
        'source_manifest_digest': audit.SOURCE_DIGEST, 'runtime_identity': plan['runtime_identity'],
        'original_scientific_result_digest': saved['scientific_result_digest'],
        'passed': passed, 'errors': errors,
        'case_scientific_digests': sorted((v['task_id'], v['replay_scientific_digest']) for v in details)})
    result['artifact_digest'] = audit.digest(RESULT_DOMAIN, result)
    write_new(args.output_directory / 'result.json', result)
    print(json.dumps({'passed': passed, 'result': str(args.output_directory / 'result.json'),
                      'artifact_digest': result['artifact_digest'], 'errors': errors}, indent=2), flush=True)
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('plan', 'run', '_worker'):
        item = sub.add_parser(command)
        item.add_argument('--archive', type=Path, required=True)
        item.add_argument('--frozen-source-root', type=Path, required=True)
        if command == 'plan':
            item.add_argument('--output', type=Path, required=True)
        else:
            item.add_argument('--plan', type=Path, required=True)
        if command == 'run':
            item.add_argument('--output-directory', type=Path, required=True)
        if command == '_worker':
            item.add_argument('--task-id', required=True)
            item.add_argument('--output', type=Path, required=True)
        else:
            item.add_argument('--repository-root', type=Path, required=True)
            item.add_argument('--validation-record', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'plan':
        make_plan(args)
        return 0
    return run(args) if args.command == 'run' else worker(args)


if __name__ == '__main__':
    sys.exit(main())
