"""Finalize the sealed last step-control comparison, without numerical work.

Only a completed four-case campaign, supervisor, and separately sealed report
can be finalized. Prior failures remain failures. This helper never fits,
recomputes a gate, chooses a checkpoint, or opens a new scientific partition.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BASE = ROOT/'reports/pinn-spatial-validation'
CAMPAIGN = BASE/'step-control-development'
VALIDATION = BASE/'step-control-validation'
REPORT = BASE/'step-control-report'
FINALIZATION = BASE/'step-control-result-finalization'
TRACKED_PLAN = ROOT/'thermotwin/PINN_SPATIAL_STEP_CONTROL_PLAN.json'
TRACKED_PROBE = ROOT/'thermotwin/PINN_SPATIAL_STEP_CONTROL_PROBE.json'
TRACKED_RESULT = ROOT/'thermotwin/PINN_SPATIAL_STEP_CONTROL_RESULT.json'
V6_RECORD = ROOT/'thermotwin/PINN_SPATIAL_OPTIMIZATION_RESULT.json'
EXPECTED_REPORT_FILES = {'REPORT.md', 'result-summary.json', 'comparison.png',
    'comparison.pdf', 'trajectories.png', 'trajectories.pdf',
    'controller-history.png', 'controller-history.pdf'}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT.resolve()))


def bound_path(path):
    path = Path(path)
    path = path.resolve() if path.is_absolute() else (ROOT/path).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise RuntimeError('Evidence path escapes the repository')
    return path


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def by_id(rows):
    result = {}
    for row in rows:
        if row['id'] in result:
            raise RuntimeError('Duplicate declared or saved case')
        result[row['id']] = row
    return result


def completion_gate():
    # The barrier reads terminal flags only. No outcomes, trajectories, or
    # supplemental audits are consulted before all completion flags exist.
    if read(CAMPAIGN/'state.json').get('status') != 'completed':
        raise RuntimeError('Step-control campaign is not completed')
    execution = read(CAMPAIGN/'execution.json')
    wrapper = read(VALIDATION/'execution-state.json')
    if execution.get('status') != 'completed' or wrapper.get('status') != 'completed':
        raise RuntimeError('Campaign execution and supervisor must both be completed')
    if not (REPORT/'report-index.json').is_file():
        raise RuntimeError('The separately rendered report must already be sealed')
    return execution, wrapper


def verify_campaign(execution, wrapper):
    from thermotwin.studies.pinn_spatial_step_control import validate_step_control
    if digest(TRACKED_PLAN) != digest(CAMPAIGN/'plan.json'):
        raise RuntimeError('Tracked and executed step-control plans differ byte for byte')
    context = validate_step_control(CAMPAIGN)
    if (execution.get('plan_sha256') != context['plan_sha256']
        or wrapper.get('plan_sha256') != context['plan_sha256']
        or wrapper.get('evidence') != context):
        raise RuntimeError('Execution provenance refers to a different campaign')
    return dict(context, directory=relative(CAMPAIGN), tracked_plan=relative(TRACKED_PLAN),
        tracked_plan_sha256=digest(TRACKED_PLAN),
        artifact_index_digest=canonical(read(CAMPAIGN/'artifact-index.json')))


def verify_sources(records):
    for name, expected in records.items():
        path = bound_path(name)
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError('A frozen source or probe input changed: '+str(name))


def probe_proof(plan):
    bound = plan['decision']
    decision_path = bound_path(bound['path'])
    decision = read(decision_path)
    if digest(decision_path) != bound['sha256'] or decision != bound['decision']:
        raise RuntimeError('The exact bound terminal probe decision changed')
    required_flags = {'training_only': True, 'reference_fields_used': False,
        'independent_diagnostics_used_for_selection': False,
        'previous_probe_C_strict_recovery_passed': False, 'mechanics_valid': True,
        'development_comparison_only': True, 'no_further_optimizer_search': True,
        'no_guarantee_of_forward_gate_success': True}
    if (decision.get('status') != 'approved_for_forward_development'
        or any(decision.get(key) is not value for key, value in required_flags.items())
        or decision.get('selected_config') != plan['model_config']):
        raise RuntimeError('The campaign lacks its exact training-only terminal decision')
    c, d = BASE/'step-control-recovery-probe', BASE/'step-control-terminal-probe'
    records = decision['probe_evidence']
    required_paths = {relative(directory/name) for directory in (c, d) for name in ('plan.json', 'result.json')}
    paths = [relative(bound_path(item['path'])) for item in records]
    if len(paths) != 4 or len(set(paths)) != 4 or set(paths) != required_paths:
        raise RuntimeError('The terminal decision must bind both complete probe plans and results')
    for item in records:
        if digest(bound_path(item['path'])) != item['sha256']:
            raise RuntimeError('A bound terminal probe artifact changed')
    for directory in (c, d):
        expected = {'status': 'completed', 'plan_sha256': digest(directory/'plan.json')}
        if read(directory/'state.json') != expected or read(directory/'result.json')['plan_sha256'] != expected['plan_sha256']:
            raise RuntimeError('A probe lacks matching terminal plan and result identities')
    cp, cr, dp, dr = (read(c/'plan.json'), read(c/'result.json'), read(d/'plan.json'), read(d/'result.json'))
    verify_sources(cp['source_sha256'])
    verify_sources(dp['source_sha256'])
    for inputs in cp['inputs'].values():
        verify_sources({item['path']: item['sha256'] for item in inputs.values()})
    if (cr.get('approved_for_forward_development') is not False
        or dp.get('previous_strict_recovery_criterion_passed') is not False
        or dr.get('previous_probe_C_strict_recovery_passed') is not False
        or dp.get('previous_plan_sha256') != digest(c/'plan.json')
        or dp.get('previous_result_sha256') != digest(c/'result.json')
        or dp.get('full_case_inputs') != cp['inputs']['stalled_full_800']):
        raise RuntimeError('The failed strict Probe C and its successor linkage changed')
    prior_config, config = cp['selected_config_if_supported'], dp['selected_config']
    if (set(config) != set(prior_config)
        or {key for key in config if config[key] != prior_config[key]} != {'controller_max_failures'}
        or prior_config['controller_max_failures'] != 3 or config['controller_max_failures'] != 8
        or config != plan['model_config']):
        raise RuntimeError('Probe D may change only the consecutive-failure limit from three to eight')
    if (dr.get('mechanics_valid') is not True or dr.get('no_further_optimizer_search') is not True
        or dr.get('scientific_recovery_supported') != decision.get('scientific_recovery_supported')
        or dr.get('partial_prefixes_reused') != dp['partial_prefixes']
        or dr['full_case'].get('copied_weights_restored_exactly') is not True
        or not 1 <= dr['full_case']['attempts'] <= dp['maximum_full_attempts']
        or dp['maximum_full_attempts'] != 8 or dp['maximum_trial_readout_solves'] != 40
        or 'no further optimizer probes' not in dp['terminal_decision']
        or decision.get('decision_basis') != dp['terminal_decision']):
        raise RuntimeError('The fixed terminal mechanics check or no-more-search promise changed')
    if (len(cr['cases']) != 3 or {row['case'] for row in cr['cases']} !=
        {'stalled_full_800', 'partial_800', 'partial_1400'}):
        raise RuntimeError('The failed Probe C lost a declared training case')
    partials = {row['case']: row for row in cr['cases'] if row['case'].startswith('partial_')}
    if set(partials) != {'partial_800', 'partial_1400'} or len(dp['partial_prefixes']) != 2:
        raise RuntimeError('Both declared partial training prefixes must be retained')
    for prefix in dp['partial_prefixes']:
        path, row = bound_path(prefix['source']), partials[prefix['case']]
        if (digest(path) != prefix['sha256'] or read(path) != row
            or row['attempts'] != 16 or row.get('monotone') is not True
            or row.get('copied_weights_restored_exactly') is not True
            or len(row['history']) != 16 or not all(event['accepted'] for event in row['history'])):
            raise RuntimeError('A reused partial prefix does not prove unchanged accepted execution')
    followup = dp['followup']
    if (decision.get('followup') != followup or cp['followup'] != followup
        or followup['cases'] != plan['cases']
        or followup['maximum_attempts_per_case'] != config['adam_epochs']
        or followup['maximum_trial_solves_per_case'] != config['maximum_trial_solves']
        or followup['acceptance_thresholds'] != plan['gates']
        or followup['waves'] != plan['waves']
        or followup['maximum_case_wall_seconds'] != plan['topology']['seconds_per_worker']
        or followup['maximum_case_rss_bytes'] != plan['topology']['rss_bytes_per_worker']
        or followup['maximum_supervisor_wall_seconds'] != plan['topology']['top_seconds']
        or followup['parallel_workers'] != plan['topology']['maximum_concurrent_workers']
        or followup['worker_threads'] != plan['topology']['threads_per_worker']
        or followup['aggregate_worker_rss_bytes'] != plan['topology']['aggregate_child_rss_bytes']
        or followup['automatic_retries'] != plan['automatic_retries']):
        raise RuntimeError('The all-four comparison differs from its prospective follow-up declaration')
    compact = read(TRACKED_PROBE)
    if (compact['decision'] != decision
        or compact['failed_strict_probe_C'] != {'plan': cp, 'result': cr}
        or compact['terminal_probe_D'] != {'plan': dp, 'result': dr}
        or compact['configuration_freeze']['path'] != relative(TRACKED_PLAN)
        or compact['configuration_freeze']['sha256'] != digest(TRACKED_PLAN)):
        raise RuntimeError('The tracked probe record does not exactly bind C, D, and the final rule')
    def row_summary(row, source, pointer):
        return {key: value for key, value in row.items() if key != 'history'} | {
            'history': {'path': relative(source), 'sha256': digest(source),
                'json_pointer': pointer, 'entries': len(row['history']),
                'identity_digest': canonical(row['history'])}}
    return {'tracked_record': {'path': relative(TRACKED_PROBE), 'sha256': digest(TRACKED_PROBE)},
        'decision': {'path': relative(decision_path), 'sha256': digest(decision_path), 'saved': decision},
        'evidence': records,
        'probe_C': {'plan_sha256': digest(c/'plan.json'), 'result_sha256': digest(c/'result.json'),
            'strict_recovery_criterion_passed': False,
            'cases': [row_summary(row, c/'result.json', '/cases/'+str(index)+'/history')
                for index, row in enumerate(cr['cases'])]},
        'probe_D': {'plan_sha256': digest(d/'plan.json'), 'result_sha256': digest(d/'result.json'),
            'mechanics_valid': True, 'scientific_recovery_supported': dr['scientific_recovery_supported'],
            'full_case': row_summary(dr['full_case'], d/'result.json', '/full_case/history'),
            'partial_prefixes_reused': dp['partial_prefixes'],
            'wall_seconds': dr.get('wall_seconds'), 'peak_rss_bytes': dr.get('peak_rss_bytes')},
        'no_further_optimizer_search': True, 'selection_recomputed': False}


def sealed_directory(directory, index_name, expected_names=None):
    directory = bound_path(directory)
    index = read(directory/index_name)
    files = index['files']
    if canonical(files) != index['digest']:
        raise RuntimeError('A report or finalization index digest differs')
    names = set()
    for item in files:
        path = (directory/item['path']).resolve()
        if (item['path'] in names or not path.is_relative_to(directory) or not path.is_file()
            or path.stat().st_size != item['bytes'] or digest(path) != item['sha256']):
            raise RuntimeError('A sealed artifact is duplicated, invalid, or changed')
        names.add(item['path'])
    actual = {str(path.relative_to(directory)) for path in directory.rglob('*')
        if path.is_file() and path != directory/index_name}
    if actual != names or (expected_names is not None and names != expected_names):
        raise RuntimeError('The seal does not cover the exact artifact file set')
    return index


def legacy_snapshot(plan):
    # The existing V6 finalizer supplies hash-bound, read-only verification of
    # its report and the earlier graded/V3/V4/V2/V1/Phase-F evidence/report chain.
    record = read(V6_RECORD)
    identity = record['finalizer']
    helper = bound_path(identity['path'])
    if not helper.is_file():
        helper = bound_path(identity['archive_path'])
    if digest(helper) != identity['sha256']:
        raise RuntimeError('The older V6 finalizer identity changed')
    spec = importlib.util.spec_from_file_location('step_control_prior_finalizer', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = module.verify_campaign(module.completion_gate())
    expected, observed = dict(record['raw_campaign']), dict(context)
    recorded_directory = bound_path(expected.pop('directory'))
    observed_directory = bound_path(observed.pop('directory'))
    if (expected != observed or recorded_directory != observed_directory
        or observed_directory != bound_path(plan['preserved_v6_attempt']['directory'])):
        raise RuntimeError('The sealed prior V6 attempt changed')
    prior_plan = read(Path(plan['preserved_v6_attempt']['directory'])/'plan.json')
    report = module.validate_report(context, prior_plan)
    older = module.legacy_snapshot(prior_plan)
    if (report != record['report'] or module.verify_probe(prior_plan) != record['optimizer_probe']
        or canonical(older) != record['preservation']['after_identity_digest']):
        raise RuntimeError('A V6 report, probe, or older evidence/report chain changed')
    directory = BASE/'optimized-result-finalization'
    manifest = sealed_directory(directory, 'manifest.json')
    if (manifest['tracked_result_sha256'] != digest(V6_RECORD)
        or digest(directory/'result.json') != digest(V6_RECORD)
        or manifest['plan_sha256'] != context['plan_sha256']
        or manifest['raw_artifact_index_sha256'] != context['artifact_index_sha256']
        or manifest['report_index_sha256'] != report['report_index_sha256']):
        raise RuntimeError('The V6 finalization manifest does not bind the retained record')
    if plan['protected_files'].get(relative(V6_RECORD)) != digest(V6_RECORD):
        raise RuntimeError('The V7 plan did not freeze the retained V6 result')
    return {'v6_record': {'path': relative(V6_RECORD), 'sha256': digest(V6_RECORD)},
        'v6_raw': context, 'v6_report': report,
        'v6_finalization_manifest': {'path': relative(directory/'manifest.json'),
            'sha256': digest(directory/'manifest.json'), 'digest': manifest['digest']},
        'v6_original_denominator': record['denominator'],
        'older_chain_identity_digest': canonical(older),
        'older_chain_proof': older, 'protected_files': plan['protected_files'],
        'v6_preserved_tree': {'directory': plan['preserved_v6_attempt']['directory'],
            'file_count': len(plan['preserved_v6_attempt']['files']),
            'file_records_digest': canonical(plan['preserved_v6_attempt']['files'])},
        'validator': 'thermotwin.studies.pinn_spatial_step_control.validate_step_control'}


def method_directories(plan):
    provenance = plan['step_control_provenance']
    return {'baseline': bound_path(provenance['graded_comparison']['directory'])/'log_off',
        'prior_attempt': bound_path(provenance['prior_optimized_campaign']['directory']),
        'controlled': CAMPAIGN}


def method_results(plan):
    return {method: read(directory/('result.json' if method == 'controlled' else 'final-forward-selection.json'))
        for method, directory in method_directories(plan).items()}


def report_proof(context, plan):
    index = sealed_directory(REPORT, 'report-index.json', EXPECTED_REPORT_FILES)
    provenance = plan['step_control_provenance']
    checks = {'campaign_evidence': {key: context[key] for key in
        ('plan_sha256', 'artifact_index_sha256', 'method_fit_count')},
        'baseline_evidence': provenance['graded_comparison']['context'],
        'prior_attempt_evidence': provenance['prior_optimized_campaign']['context']}
    if (any(index.get(key) != expected for key, expected in checks.items())
        or index.get('plan_sha256') != context['plan_sha256']
        or index.get('renderer_sha256') != digest(ROOT/'thermotwin/reports/pinn_spatial_step_control.py')):
        raise RuntimeError('The report has different campaign, baseline, prior-attempt, or renderer identities')
    summary = read(REPORT/'result-summary.json')
    rows = by_id(summary['rows'])
    declared = by_id(plan['cases'])
    results = method_results(plan)
    if len(rows) != 4 or set(rows) != set(declared):
        raise RuntimeError('The report lost a declared case')
    for method, result in results.items():
        saved = by_id(result['cases'])
        if set(saved) != set(declared):
            raise RuntimeError('A reported attempt lost a declared case')
        for case_id in saved:
            if rows[case_id]['methods'][method]['saved'] != saved[case_id]:
                raise RuntimeError('A report row differs from the exact saved coordinator row')
    return {'directory': relative(REPORT), 'report_index_sha256': digest(REPORT/'report-index.json'),
        'report_index_digest': index['digest'], 'files': len(index['files']),
        'renderer_sha256': index['renderer_sha256'], 'summary_sha256': digest(REPORT/'result-summary.json')}


def training_summary(directory):
    path = directory/'training-report.json'
    if not path.is_file():
        return {'status': 'not_recorded_after_failure_or_not_run',
            'raw_report': relative(path), 'raw_report_sha256': None}
    raw = read(path)
    keep = ('architecture', 'model_module', 'wall_seconds', 'final_objective', 'final_physics_loss',
        'normalized_observation_mse', 'training_family_mse', 'coefficient_count',
        'initialization', 'graded_point_metadata', 'training_point_counts',
        'hidden_rejected_updates', 'hidden_backtrack_count', 'hidden_accepted_objective_increases',
        'optimizer_history', 'optimizer_instances', 'optimizer_resets', 'controller_moment_restarts',
        'policy_moment_resets', 'optimizer_state_rollbacks', 'final_optimizer_step_counts',
        'final_controller_state', 'trial_solve_count', 'additional_initial_final_solves',
        'total_projection_solve_calls', 'maximum_trial_solves', 'requested_updates',
        'completed_updates', 'termination', 'stop_reason', 'gradient_semantics')
    result = {key: raw.get(key) for key in keep}
    updates, projections = raw.get('hidden_update_history'), raw.get('projections')
    result.update(status='recorded', raw_report=relative(path), raw_report_sha256=digest(path),
        timing_scope='Training only; later independent diagnostics and scoring are excluded.',
        hidden_update_count=len(updates) if updates is not None else None,
        hidden_accepted_updates=sum(event.get('accepted') is True for event in updates) if updates is not None else None,
        initial_projection_objective=projections[0].get('objective_after') if projections else None,
        checkpoint_rule=raw.get('checkpoint'))
    for name in ('hidden_update_history', 'controller_history', 'projections'):
        values = raw.get(name)
        result[name] = None if values is None else {'json_pointer': '/'+name,
            'entries': len(values), 'identity_digest': canonical(values)}
    final = projections[-1] if projections else {}
    result['final_projection'] = {key: final.get(key) for key in ('rank', 'columns', 'rows',
        'rank_rcond', 'resolved_condition', 'smallest_singular_value',
        'coefficient_norm_returned', 'accepted', 'objective_before', 'objective_after')}
    result['training_arrays'] = {'points_json_pointer': '/training_points',
        'weights_json_pointer': '/training_point_weights', 'included_in_compact_result': False}
    return result


def progress_provenance(directory):
    path = directory/'progress.jsonl'
    if not path.is_file():
        return {'status': 'not_recorded'}
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    updates = [event for event in records if 'trials' in event and 'objective_before' in event]
    return {'path': relative(path), 'sha256': digest(path), 'records': len(records),
        'recorded_update_events': len(updates), 'update_identity_digest': canonical(updates),
        'role': 'Partial training provenance; never a completed fit or substitute primary gate.'}


def optional_audit(directory, name):
    path = directory/name
    return {'path': relative(path), 'sha256': digest(path), 'saved': read(path)} if path.is_file() else {
        'path': relative(path), 'sha256': None, 'saved': {'status': 'not_recorded', 'role': 'report only'}}


def compact_rows(plan, results):
    declared = by_id(plan['cases'])
    mapped = {method: by_id(result['cases']) for method, result in results.items()}
    if len(declared) != 4 or any(set(rows) != set(declared) for rows in mapped.values()):
        raise RuntimeError('Every attempt must retain all four declared cases')
    directories = method_directories(plan)
    rows = []
    for case_id, case in declared.items():
        methods = {}
        for method, saved_rows in mapped.items():
            saved, directory = saved_rows[case_id], directories[method]/case_id
            if any(saved.get(key) != case[key] for key in ('geometry', 'known_h')):
                raise RuntimeError('A saved outcome changed physical case identity')
            diagnostics = saved.get('independent_residuals') or {}
            methods[method] = {key: saved.get(key) for key in ('status', 'gate', 'field_metrics',
                'normalized_observation_mse', 'checkpoint_sha256', 'controller_proof',
                'wall_seconds', 'peak_rss_bytes', 'return_code', 'terminal_recorded_by')}
            methods[method].update(error_or_reason=saved.get('error') or saved.get('reason'),
                primary_diagnostics={key: diagnostics.get(key) for key in ('normalized_rms',
                    'physical_rms', 'normalization_scales', 'point_counts', 'validation_scheme', 'all_families_pass')},
                training=training_summary(directory), progress_provenance=progress_provenance(directory),
                near_interface_audit=optional_audit(directory, 'near-interface-audit.json'),
                near_bottom_audit=optional_audit(directory, 'near-bottom-audit.json'))
        rows.append(dict(case, methods=methods))
    return rows


def test_evidence(stem):
    state_path, log_path = VALIDATION/(stem+'-state.json'), VALIDATION/(stem+'.log')
    if not state_path.is_file():
        return {'status': 'not_recorded', 'completed': False, 'test_count': None, 'passed': None, 'log_read': False}
    state = read(state_path)
    status = state.get('status', 'not_recorded')
    terminal = status in ('passed', 'failed', 'completed')
    record = {'status': state.get('status', 'not_recorded'), 'completed': terminal,
        'state_path': relative(state_path), 'state_sha256': digest(state_path), 'saved': state,
        'test_count': None, 'passed': None, 'log_read': False}
    if not terminal:
        return record
    problems = []
    if status == 'completed':
        claimed = state.get('passed') if type(state.get('passed')) is bool else None
        if claimed is None:
            problems.append('completed state requires an explicit boolean passed value')
    else:
        claimed = status == 'passed'
        if 'passed' in state and state['passed'] is not claimed:
            problems.append('legacy status and explicit passed value disagree')
    if claimed is False:
        record['passed'] = False
    if not log_path.is_file():
        record['evidence_error'] = 'terminal test state lacks a terminal log'
        return record
    text = log_path.read_text(errors='replace')
    counts = re.findall(r'^Ran ([0-9]+) tests? in ([0-9.]+)s\s*$', text, flags=re.MULTILINE)
    endings = re.findall(r'^(OK(?: \([^\n]*\))?|FAILED \([^\n]*\))\s*$', text, flags=re.MULTILINE)
    actual_log_sha256 = digest(log_path)
    record.update(log_read=True, log_path=relative(log_path), log_sha256=digest(log_path),
        test_count=int(counts[-1][0]) if counts else None,
        unittest_reported_seconds=float(counts[-1][1]) if counts else None,
        terminal_log_summary=endings[-1] if endings else None)
    last_line = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), None)
    if not counts or not endings or last_line != endings[-1]:
        problems.append('log lacks a terminal unittest count and summary')
    if 'log_sha256' in state:
        record['declared_log_hash_matches'] = state['log_sha256'] == actual_log_sha256
        if not record['declared_log_hash_matches']:
            problems.append('declared log hash differs from the retained log')
    count_keys = ('tests_run', 'discovered_test_count', 'expected_test_count', 'tests', 'test_count')
    declared = {key: state[key] for key in count_keys if key in state}
    record['declared_test_counts'] = declared
    counts_match = bool(declared) and all(type(value) is int and value >= 0
        and value == record['test_count'] for value in declared.values())
    if status == 'completed' and any(key not in declared for key in count_keys[:3]):
        counts_match = False
    record['declared_counts_match_log'] = counts_match
    if not counts_match:
        problems.append('declared run, discovery, expected, or legacy counts disagree with the log')
    identities = state.get('source_sha256')
    sources_match = isinstance(identities, dict) and bool(identities)
    if sources_match:
        try:
            sources_match = all(isinstance(name, str) and isinstance(expected, str)
                and bound_path(name).is_file() and digest(bound_path(name)) == expected
                for name, expected in identities.items())
        except (OSError, RuntimeError, ValueError, TypeError):
            sources_match = False
    record['source_records_match_current'] = sources_match
    if not sources_match or ('source_unchanged' in state and state['source_unchanged'] is not True):
        problems.append('tested source identities changed or are unavailable')
    if status == 'completed' and state.get('source_unchanged') is not True:
        problems.append('completed state lacks an explicit unchanged-source proof')
    if 'source_manifest_sha256' in state:
        manifest_path = VALIDATION/(stem+'-source-manifest.json')
        manifest_matches = False
        try:
            manifest_matches = (manifest_path.is_file()
                and digest(manifest_path) == state['source_manifest_sha256']
                and read(manifest_path).get('source_sha256') == identities)
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        record['source_manifest_matches'] = manifest_matches
        if not manifest_matches:
            problems.append('declared source manifest is missing, changed, or disagrees with source records')
    if claimed is True:
        if not endings or not endings[-1].startswith('OK'):
            problems.append('a passing state requires a terminal OK summary')
        for key in ('failures', 'errors', 'unexpected_successes'):
            if key in state and (type(state[key]) is not int or state[key] != 0):
                problems.append('passing state records nonzero or invalid '+key)
        if status == 'completed' and any(type(state.get(key)) is not int or state[key] != 0
            for key in ('failures', 'errors')):
            problems.append('completed passing state lacks zero failures and errors')
        for key in ('exit_code', 'return_code'):
            if key in state and (type(state[key]) is not int or state[key] != 0):
                problems.append('passing state records nonzero or invalid '+key)
        if status != 'completed' and not ('exit_code' in state or 'return_code' in state
            or all(type(state.get(key)) is int and state[key] == 0 for key in ('failures', 'errors'))):
            problems.append('legacy passing state lacks successful exit or result statistics')
        if not problems:
            record['passed'] = True
    if problems:
        record['evidence_error'] = '; '.join(problems)
    return record


def finalize():
    execution, wrapper = completion_gate()
    if TRACKED_RESULT.exists() or FINALIZATION.exists():
        raise RuntimeError('Finalization is one shot; existing result or manifest is not overwritten')
    suite = test_evidence('full-suite')
    if suite.get('completed') is not True or suite.get('passed') is not True:
        raise RuntimeError('A terminal passing full-repository test proof is required before finalization')
    before = verify_campaign(execution, wrapper)
    plan = read(CAMPAIGN/'plan.json')
    probes, legacy_before = probe_proof(plan), legacy_snapshot(plan)
    report = report_proof(before, plan)
    results = method_results(plan)
    saved = results['controlled']
    rows = compact_rows(plan, results)
    declared_ids = {case['id'] for case in plan['cases']}
    never_run = {case['id'] for case in saved['cases'] if case.get('status') == 'not_run'}
    if (saved.get('status') != 'completed' or saved.get('plan_sha256') != before['plan_sha256']
        or saved.get('source_sha256') != plan['source_sha256']
        or set(wrapper.get('return_codes') or {}) != declared_ids-never_run):
        raise RuntimeError('The saved execution does not preserve all four once-only opportunities')
    after, legacy_after = verify_campaign(execution, wrapper), legacy_snapshot(plan)
    if before != after or legacy_before != legacy_after or report_proof(before, plan) != report:
        raise RuntimeError('A source, scientific artifact, or report changed during finalization')
    parent = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    counts = {method: {'declared': 4,
        'completed': sum(row.get('status') == 'completed' for row in result['cases']),
        'failed': sum(row.get('status') == 'failed' for row in result['cases']),
        'not_run_or_not_recorded': sum(row.get('status') not in ('completed', 'failed') for row in result['cases']),
        'saved_gate_passes': sum((row.get('gate') or {}).get('all') is True for row in result['cases'])}
        for method, result in results.items()}
    result = {'schema_version': 1, 'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'parent_commit': parent, 'development_only': True,
        'scope': 'One final bounded step-controller comparison on four reused 2D development datasets; saved gates only.',
        'recorded_status': saved.get('status'), 'recorded_forward_status': saved.get('forward_status'),
        'no_winner_selected': saved.get('no_winner_selected'),
        'denominator': {'new_case_opportunities': 4, 'comparison_case_opportunities': 12,
            'retained_prior_failed_attempt_opportunities': 4, 'reused_paired_datasets': 4,
            'independent_noise_replications_claimed': False, 'automatic_retries': plan['automatic_retries'],
            'per_method': counts},
        'plan': {'path': relative(TRACKED_PLAN), 'sha256': before['plan_sha256'],
            'source_sha256': plan['source_sha256'], 'runtime': plan['runtime'],
            'model_config': plan['model_config'], 'gates': plan['gates'], 'seeds': plan['seeds'],
            'waves': plan['waves'], 'topology': plan['topology'],
            'allowed_training_stop_reasons': plan['allowed_training_stop_reasons'],
            'checkpoint_rule': plan['checkpoint_rule'], 'initialization_policy': plan['initialization_policy'],
            'selection_boundary': plan['selection_boundary']},
        'raw_campaign': before, 'case_comparisons': rows, 'training_only_probes': probes,
        'costs': {'per_attempt': {method: {key: value.get(key) for key in
            ('wall_seconds', 'peak_rss_bytes', 'peak_worker_rss_bytes')} for method, value in results.items()},
            'supervisor_wall_seconds': wrapper.get('wall_seconds'),
            'scope': 'Per-fit training times exclude diagnostics. Concurrent worker RSS maxima and elapsed totals are not additive CPU-time or simultaneous aggregate-RSS measurements.'},
        'execution': {'campaign_execution': {'path': relative(CAMPAIGN/'execution.json'),
            'sha256': digest(CAMPAIGN/'execution.json'), 'saved': execution},
            'supervisor': {'path': relative(VALIDATION/'execution-state.json'),
                'sha256': digest(VALIDATION/'execution-state.json'), 'saved': wrapper}},
        'report': report,
        'quality_checks': {'focused': test_evidence('focused'), 'reporter_focused': test_evidence('report-focused'),
            'full_repository_suite': suite,
            'overlap_policy': 'Separate test invocations are not added together; running logs never support a pass or count.',
            'exact_head_ci': 'Not asserted; verify after the resulting source commit.'},
        'preservation': {'identical_before_and_after': True,
            'before_identity_digest': canonical(legacy_before), 'after_identity_digest': canonical(legacy_after),
            'proof': relative(FINALIZATION/'preservation-after.json'),
            'campaign_before': before, 'campaign_after': after},
        'source_unchanged': saved.get('source_unchanged'),
        'saved_scope_flags': {key: saved.get(key) for key in ('inverse_opened', 'fresh_evaluation_opened', 'three_d_opened')},
        'finalizer': {'path': relative(__file__), 'sha256': digest(__file__)},
        'interpretation_limits': [
            'All primary gates are saved coordinator decisions. This helper performs no numerical fitting, scoring, re-gating, or checkpoint selection.',
            'The failed strict Probe C and the incomplete V6 attempt remain explicitly preserved; no partial progress is promoted to a completed fit.',
            'Probe D tested one declared failure-threshold extension. Mechanical validity and bounded training recovery are distinct from the unchanged primary gates.',
            'The declared terminal promise permits no further optimizer probes or outcome-driven budget extensions.',
            'Every attempt retains all four cases, including failures and cases not run. These reused datasets are not independent noise replications or fresh-cohort evaluation.',
            'Controller and resource stops describe bounded training endpoints; they do not certify stationarity or convergence.',
            'Supplemental near-interface and near-bottom audits remain report only. Exact continuity does not establish heat-equation accuracy.',
            'Coordinate arrays and full training/controller trajectories remain in sealed raw artifacts; compact hashes and pointers bind them.',
            'No inverse recovery, reserved evaluation, hardware validation, or 3D validity is established.']}
    FINALIZATION.mkdir(parents=True)
    for name, value in (('preservation-before.json', legacy_before), ('preservation-after.json', legacy_after),
        ('probe-proof.json', probes), ('report-proof.json', report), ('result.json', result)):
        write(FINALIZATION/name, value)
    (FINALIZATION/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    write(TRACKED_RESULT, result)
    files = [{'path': path.name, 'bytes': path.stat().st_size, 'sha256': digest(path)}
        for path in sorted(FINALIZATION.iterdir()) if path.is_file()]
    write(FINALIZATION/'manifest.json', {'files': files, 'digest': canonical(files),
        'tracked_result_sha256': digest(TRACKED_RESULT), 'plan_sha256': before['plan_sha256'],
        'raw_artifact_index_sha256': before['artifact_index_sha256'],
        'report_index_sha256': report['report_index_sha256'], 'finalizer': result['finalizer']})
    if (verify_campaign(execution, wrapper) != before or probe_proof(plan) != probes
        or legacy_snapshot(plan) != legacy_after or report_proof(before, plan) != report
        or test_evidence('full-suite') != suite):
        raise RuntimeError('Preserved evidence changed after writing; inspect finalization records')
    output = {'tracked_result': relative(TRACKED_RESULT), 'tracked_result_sha256': digest(TRACKED_RESULT),
        'manifest': relative(FINALIZATION/'manifest.json'), 'manifest_sha256': digest(FINALIZATION/'manifest.json'),
        'report': relative(REPORT/'REPORT.md'), 'forward_status': result['recorded_forward_status'],
        'saved_gate_passes': {method: value['saved_gate_passes'] for method, value in counts.items()},
        'full_suite_status': result['quality_checks']['full_repository_suite']['status']}
    print(json.dumps(output, indent=2))
    return output


if __name__ == '__main__':
    finalize()
