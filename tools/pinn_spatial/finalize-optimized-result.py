"""Seal completed optimizer evidence without fitting, scoring, or selection.

The campaign, supervisor, and separately rendered report must already have
finished. This one-shot helper writes a compact tracked result and its own
ignored manifest; it never changes a scientific campaign or an older report.
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
CAMPAIGN = BASE/'optimized-v6-development'
VALIDATION = BASE/'optimization-validation'
REPORT = BASE/'optimized-v6-report'
FINALIZATION = BASE/'optimized-result-finalization'
TRACKED_PLAN = ROOT/'thermotwin/PINN_SPATIAL_OPTIMIZATION_PLAN.json'
TRACKED_RESULT = ROOT/'thermotwin/PINN_SPATIAL_OPTIMIZATION_RESULT.json'
TRACKED_PROBE = ROOT/'thermotwin/PINN_SPATIAL_OPTIMIZATION_PROBE.json'
ARCHIVE = ROOT/'tools/pinn_spatial/finalize-optimized-result.py'
EXPECTED_REPORT_FILES = {'REPORT.md', 'result-summary.json', 'comparison.png',
    'comparison.pdf', 'trajectories.png', 'trajectories.pdf',
    'optimizer-history.png', 'optimizer-history.pdf'}


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


def completion_gate():
    # Only terminal flags and the existence of the sealed report are consulted
    # before any outcome, training trajectory, or supplemental audit is read.
    if read(CAMPAIGN/'state.json').get('status') != 'completed':
        raise RuntimeError('Optimizer campaign is not completed; nothing was finalized')
    execution = read(VALIDATION/'execution-state.json')
    if execution.get('status') != 'completed' or execution.get('return_code') != 0:
        raise RuntimeError('Execution supervisor has not recorded successful completion')
    if not (REPORT/'report-index.json').is_file():
        raise RuntimeError('The separately rendered report must already be sealed')
    return execution


def verify_campaign(execution):
    from thermotwin.studies.pinn_spatial_optimized import validate_optimized
    if digest(TRACKED_PLAN) != digest(CAMPAIGN/'plan.json'):
        raise RuntimeError('Tracked and executed optimizer plans differ byte for byte')
    context = validate_optimized(CAMPAIGN)
    if execution.get('plan_sha256') != context['plan_sha256']:
        raise RuntimeError('Execution supervisor is bound to another plan')
    if execution.get('evidence') != context:
        raise RuntimeError('Execution supervisor evidence differs from the sealed campaign')
    return dict(context, tracked_plan=relative(TRACKED_PLAN),
        tracked_plan_sha256=digest(TRACKED_PLAN),
        artifact_index_digest=canonical(read(CAMPAIGN/'artifact-index.json')))


def verify_probe(plan):
    provenance = plan['optimization_provenance']
    bound = provenance['decision']
    path = bound_path(bound['path'])
    decision = read(path)
    if digest(path) != bound['sha256'] or decision != bound['decision']:
        raise RuntimeError('The bound training-only decision changed')
    if (decision.get('status') != 'approved_for_forward_development'
        or decision.get('training_only') is not True
        or decision.get('reference_fields_used') is not False
        or decision.get('independent_diagnostics_used_for_selection') is not False
        or decision.get('selected_config') != plan['model_config']):
        raise RuntimeError('The executed rule lacks the exact approved training-only decision')
    records = []
    for item in decision['probe_evidence']:
        artifact = bound_path(item['path'])
        if digest(artifact) != item['sha256']:
            raise RuntimeError('A frozen probe artifact changed')
        records.append({'path': relative(artifact), 'sha256': digest(artifact)})
    if len(records) != 4 or len({item['path'] for item in records}) != 4:
        raise RuntimeError('Exactly two frozen probe plans and results are required')
    a, b = BASE/'optimization-directional-probe', BASE/'optimization-history-probe'
    required = {relative(directory/name) for directory in (a, b) for name in ('plan.json', 'result.json')}
    if {item['path'] for item in records} != required:
        raise RuntimeError('The optimizer decision is bound to unexpected probe artifacts')
    for directory in (a, b):
        expected = {'status': 'completed', 'plan_sha256': digest(directory/'plan.json')}
        if read(directory/'state.json') != expected or read(directory/'result.json')['plan_sha256'] != expected['plan_sha256']:
            raise RuntimeError('A probe lacks matching terminal plan and result proofs')
    aplan, ar, br = read(a/'plan.json'), read(a/'result.json'), read(b/'result.json')
    followup, config = aplan['four_case_followup'], plan['model_config']
    if (config['learning_rate'] != ar['selected_learning_rate']
        or config['optimizer_history'] != br['selected_optimizer_history']
        or config['adam_epochs'] != followup['fixed_updates']
        or config['max_backtracks'] != followup['maximum_backtracks']
        or decision['start_policy'] != followup['start_policy']
        or decision['maximum_wall_seconds'] != followup['maximum_wall_seconds']
        or decision['maximum_rss_bytes'] != followup['maximum_rss_bytes']
        or plan['maximum_wall_seconds'] != followup['maximum_wall_seconds']
        or plan['maximum_rss_bytes'] != followup['maximum_rss_bytes']
        or plan['topology']['threads'] != followup['threads']
        or plan['retries'] != followup['automatic_retries']):
        raise RuntimeError('The frozen probe selections do not reproduce the declared campaign rule')
    if (len(ar['cases']) != 2 or {row['case'] for row in ar['cases']} != {'t02', 't04'}
        or len(br['arms']) != 4
        or {(row['case'], row['arm']) for row in br['arms']} !=
            {(case, arm) for case in ('t02', 't04') for arm in ('reset', 'persistent')}
        or not all(row['copied_weights_restored_exactly'] for row in ar['cases']+br['arms'])):
        raise RuntimeError('A probe lost a declared case, policy, or restoration proof')
    compact = read(TRACKED_PROBE)
    if compact.get('decision') != decision or compact.get('four_case_plan_sha256') != digest(CAMPAIGN/'plan.json'):
        raise RuntimeError('Tracked probe summary does not bind the decision and executed plan')
    return {'tracked_summary': {'path': relative(TRACKED_PROBE), 'sha256': digest(TRACKED_PROBE)},
        'decision': {'path': relative(path), 'sha256': digest(path), 'saved': decision},
        'evidence': records, 'frozen_followup': followup,
        'selection_recomputed': False,
        'limits': 'Training-only continuation probes on two already opened partial datasets; no independent diagnostics or hidden-field labels selected the rule.'}


def legacy_snapshot(plan):
    # Reuse the old, hash-bound read-only verifiers, including their report seals.
    # The optimized study validator independently verifies the complete numerical
    # source/evidence chain through graded, V3/V4, V2, V1 and Phase F.
    path = ROOT/'thermotwin/PINN_SPATIAL_GRADED_LOG_RESULT.json'
    recorded = read(path)
    identity = recorded['finalizer']
    helper_path = bound_path(identity['path'])
    if digest(helper_path) != identity['sha256']:
        raise RuntimeError('The prior finalizer differs from its recorded identity')
    verifier_id = recorded['legacy_preservation']['verifier']
    if digest(bound_path(verifier_id['path'])) != verifier_id['sha256']:
        raise RuntimeError('The prior legacy verifier differs from its recorded identity')
    spec = importlib.util.spec_from_file_location('optimized_prior_finalizer', helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    previous = module.verify_campaign()
    if previous != recorded['comparison_preservation']['after']:
        raise RuntimeError('The prior graded comparison changed')
    old_report = module.validate_report(previous)
    if old_report != recorded['report']:
        raise RuntimeError('The prior graded report changed')
    verify_legacy, current_id = module.legacy_verifier()
    older = verify_legacy()
    if current_id != verifier_id or canonical(older) != recorded['legacy_preservation']['after_identity_digest']:
        raise RuntimeError('The V3/V4 and older report/evidence chain changed')
    prior = plan['prior_development']
    return {'graded_record': {'path': relative(path), 'sha256': digest(path)},
        'graded_raw': previous, 'graded_report': old_report,
        'older_evidence_identity_digest': canonical(older), 'legacy_verifier': current_id,
        'prior_development_record': {'path': prior['path'], 'sha256': prior['sha256']},
        'preserved_v1_and_phase_f_evidence': prior['preserved_evidence'],
        'protected_tracked_files': plan['optimization_provenance']['protected_files'],
        'protected_directory_proofs': [{'directory': item['directory'],
            'files': len(item['files']), 'file_records_digest': canonical(item['files'])}
            for item in plan['protected_directories']],
        'protected_index_proofs': [{'directory': item['directory'],
            'index_sha256': item['index_sha256'], 'digest': item['digest'], 'files': len(item['files'])}
            for item in plan['protected_indices']],
        'numerical_preservation_validator': 'thermotwin.studies.pinn_spatial_optimized.validate_optimized'}


def validate_report(context, plan):
    path = REPORT/'report-index.json'
    index = read(path)
    files = index['files']
    if canonical(files) != index['digest']:
        raise RuntimeError('Optimizer report index digest differs')
    names = set()
    for item in files:
        name = item['path']
        artifact = (REPORT/name).resolve()
        if name in names or not artifact.is_relative_to(REPORT.resolve()) or not artifact.is_file():
            raise RuntimeError('Invalid or duplicated optimizer report artifact')
        names.add(name)
        if artifact.stat().st_size != item['bytes'] or digest(artifact) != item['sha256']:
            raise RuntimeError('A sealed optimizer report artifact changed')
    actual = {str(item.relative_to(REPORT)) for item in REPORT.rglob('*')
        if item.is_file() and item != path}
    if names != EXPECTED_REPORT_FILES or actual != names:
        raise RuntimeError('Optimizer report does not have the exact eight-file sealed set')
    for key in ('artifact_index_sha256', 'plan_sha256'):
        if (index.get('campaign_evidence') or {}).get(key) != context[key]:
            raise RuntimeError('Optimizer report refers to another campaign')
    if (index.get('plan_sha256') != context['plan_sha256']
        or index.get('baseline_evidence') != plan['optimization_provenance']['graded_comparison']['context']):
        raise RuntimeError('Optimizer report plan or baseline binding differs')
    renderer = ROOT/'thermotwin/reports/pinn_spatial_optimized.py'
    if index.get('renderer_sha256') != digest(renderer):
        raise RuntimeError('Optimizer report renderer identity changed')
    summary = read(REPORT/'result-summary.json')
    saved = by_id(read(CAMPAIGN/'final-forward-selection.json')['cases'])
    old = bound_path(plan['optimization_provenance']['graded_comparison']['directory'])
    baseline = by_id(read(old/'log_off/final-forward-selection.json')['cases'])
    rows = by_id(summary['rows'])
    if set(rows) != set(saved) or set(rows) != set(baseline) or len(rows) != 4:
        raise RuntimeError('The report lost a declared case')
    for case_id in saved:
        for method, expected in (('optimized', saved[case_id]), ('baseline', baseline[case_id])):
            if rows[case_id]['methods'][method]['saved'] != expected:
                raise RuntimeError('A report row differs from its saved coordinator row')
    return {'directory': relative(REPORT), 'report_index_sha256': digest(path),
        'report_index_digest': index['digest'], 'files': len(files),
        'renderer_sha256': index['renderer_sha256'],
        'summary_sha256': digest(REPORT/'result-summary.json')}


def by_id(rows):
    result = {}
    for row in rows:
        if row['id'] in result:
            raise RuntimeError('Duplicate declared or saved case')
        result[row['id']] = row
    return result


def optional_audit(directory, name):
    path = directory/name
    return {'path': relative(path), 'sha256': digest(path), 'saved': read(path)} if path.is_file() else {
        'path': relative(path), 'sha256': None, 'saved': {'status': 'not_recorded', 'role': 'report only'}}


def training_summary(directory):
    path = directory/'training-report.json'
    if not path.is_file():
        return {'status': 'not_recorded_after_failure', 'raw_report': relative(path),
            'raw_report_sha256': None, 'trajectory': None}
    raw = read(path)
    updates, projections = raw.get('hidden_update_history'), raw.get('projections')
    final = projections[-1] if projections else {}
    def pointer(name, values):
        return None if values is None else {'entries': len(values),
            'json_pointer': '/'+name, 'identity_digest': canonical(values)}
    keep = ('architecture', 'model_module', 'wall_seconds', 'final_objective',
        'final_physics_loss', 'normalized_observation_mse', 'training_family_mse',
        'coefficient_count', 'base_coefficient_count', 'initialization',
        'graded_point_metadata', 'training_point_counts', 'hidden_rejected_updates',
        'hidden_backtrack_count', 'hidden_accepted_objective_increases',
        'optimizer_history', 'optimizer_instances', 'optimizer_resets',
        'optimizer_state_rollbacks', 'optimizer_state_policy',
        'final_optimizer_step_counts', 'gradient_semantics')
    result = {key: raw.get(key) for key in keep}
    result.update(status='recorded', raw_report=relative(path), raw_report_sha256=digest(path),
        timing_scope='Fit only; independent diagnostics and hidden-field scoring follow.',
        initial_projection_objective=projections[0].get('objective_after') if projections else None,
        trajectory=pointer('hidden_update_history', updates),
        hidden_update_count=len(updates) if updates is not None else None,
        hidden_accepted_updates=sum(item.get('accepted') is True for item in updates) if updates is not None else None,
        projection_records=pointer('projections', projections),
        final_projection={key: final.get(key) for key in ('rank', 'columns', 'rows', 'rank_rcond',
            'resolved_condition', 'smallest_singular_value', 'coefficient_norm_returned',
            'accepted', 'objective_before', 'objective_after')},
        checkpoint_rule=raw.get('checkpoint'),
        training_point_arrays={'json_pointer': '/training_points', 'included_in_compact_result': False},
        training_point_weights={'json_pointer': '/training_point_weights',
            'policy': 'The validator verifies the frozen uniform within-family mean weights; full row arrays remain in the indexed raw report.'})
    return result


def compact_outcomes(result, plan, index):
    declared, saved = by_id(plan['cases']), by_id(result['cases'])
    if len(declared) != 4 or set(declared) != set(saved):
        raise RuntimeError('All four declared outcomes, including failed and unrun cases, must be retained')
    rows = []
    for case_id, case in declared.items():
        value, directory = saved[case_id], CAMPAIGN/case_id
        if any(value.get(key) != case[key] for key in ('geometry', 'known_h')):
            raise RuntimeError('A saved outcome changed physical case identity')
        diagnostics = value.get('independent_residuals') or {}
        rows.append({'id': case_id, 'geometry': case['geometry'], 'known_h': case['known_h'],
            'status': value.get('status', 'not_recorded'), 'gate': value.get('gate'),
            'normalized_observation_mse': value.get('normalized_observation_mse'),
            'field_metrics': value.get('field_metrics'),
            'primary_diagnostics': {key: diagnostics.get(key) for key in ('normalized_rms',
                'physical_rms', 'normalization_scales', 'point_counts', 'validation_scheme', 'all_families_pass')},
            'checkpoint_sha256': value.get('checkpoint_sha256'),
            'indexed_artifacts': {name: index.get(case_id+'/'+name) for name in
                ('result.json', 'final_checkpoint.pt', 'training-report.json', 'observations.npz', 'field_scoring.npz')},
            'error_or_reason': value.get('error') or value.get('reason'),
            'training': training_summary(directory),
            'near_interface_audit': optional_audit(directory, 'near-interface-audit.json'),
            'near_bottom_audit': optional_audit(directory, 'near-bottom-audit.json')})
    return rows


def test_evidence(stem):
    state_path, log_path = VALIDATION/(stem+'-state.json'), VALIDATION/(stem+'.log')
    if not state_path.is_file():
        return {'status': 'not_recorded', 'completed': False, 'test_count': None, 'passed': None, 'log_read': False}
    state = read(state_path)
    terminal = state.get('status') in ('passed', 'failed')
    record = {'status': state.get('status', 'not_recorded'), 'completed': terminal,
        'state_path': relative(state_path), 'state_sha256': digest(state_path),
        'state': state, 'test_count': None, 'passed': None, 'log_read': False}
    if not terminal:
        return record
    if not log_path.is_file():
        record['evidence_error'] = 'terminal test state lacks its log'
        return record
    text = log_path.read_text(errors='replace')
    counts = re.findall(r'^Ran ([0-9]+) tests? in ([0-9.]+)s\s*$', text, flags=re.MULTILINE)
    endings = re.findall(r'^(OK(?: \([^\n]*\))?|FAILED \([^\n]*\))\s*$', text, flags=re.MULTILINE)
    record.update(log_read=True, log_path=relative(log_path), log_sha256=digest(log_path),
        test_count=int(counts[-1][0]) if counts else None,
        unittest_reported_seconds=float(counts[-1][1]) if counts else None,
        terminal_log_summary=endings[-1] if endings else None)
    identities = state.get('source_sha256')
    if isinstance(identities, dict):
        record['source_records_match_current'] = all(
            bound_path(name).is_file() and digest(bound_path(name)) == expected
            for name, expected in identities.items())
    declared_count = state.get('tests', state.get('test_count'))
    count_ok = declared_count is None or record['test_count'] == declared_count
    if state.get('status') == 'passed':
        # Some focused harnesses record failures/errors rather than an exit code.
        code_ok = state.get('exit_code') == 0 if 'exit_code' in state else (
            state.get('failures') == 0 and state.get('errors') == 0)
        if (code_ok and counts and endings and endings[-1].startswith('OK') and count_ok
            and state.get('source_unchanged') is not False
            and record.get('source_records_match_current') is not False):
            record['passed'] = True
        else:
            record['evidence_error'] = 'passed state and terminal unittest evidence do not agree'
    else:
        record['passed'] = False
    return record


def finalizer_identity():
    source = Path(__file__).resolve()
    original = BASE/'finalize-optimized-result.py'
    if source not in (original.resolve(), ARCHIVE.resolve()):
        raise RuntimeError('Execute the reviewed repository helper or its identical archive')
    if not ARCHIVE.is_file() or digest(original) != digest(ARCHIVE):
        raise RuntimeError('The portable finalizer archive must be byte identical')
    return {'path': relative(original), 'archive_path': relative(ARCHIVE),
        'sha256': digest(original), 'archive_sha256': digest(ARCHIVE)}


def finalize():
    execution = completion_gate()
    if TRACKED_RESULT.exists() or FINALIZATION.exists():
        raise RuntimeError('Optimizer result finalization is one shot; existing output is not overwritten')
    identity = finalizer_identity()
    before = verify_campaign(execution)
    plan = read(CAMPAIGN/'plan.json')
    probe = verify_probe(plan)
    legacy_before = legacy_snapshot(plan)
    report = validate_report(before, plan)
    saved = read(CAMPAIGN/'final-forward-selection.json')
    rows = compact_outcomes(saved, plan, read(CAMPAIGN/'artifact-index.json'))
    if saved.get('plan_sha256') != before['plan_sha256'] or saved.get('source_sha256') != plan['source_sha256']:
        raise RuntimeError('Saved outcomes do not bind the declared plan and sources')
    after, legacy_after = verify_campaign(execution), legacy_snapshot(plan)
    if before != after or legacy_before != legacy_after or validate_report(before, plan) != report:
        raise RuntimeError('Source, evidence, or report changed during finalization')
    parent = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    result = {'schema_version': 1, 'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'parent_commit': parent, 'development_only': True,
        'scope': 'Training-probe-selected optimizer on four reused 2D forward development datasets; saved gates only.',
        'recorded_status': saved.get('status'),
        'denominator': {'declared_cases': 4, 'new_method_fits': 4, 'reused_paired_datasets': 4,
            'completed': sum(row['status'] == 'completed' for row in rows),
            'failed': sum(row['status'] == 'failed' for row in rows),
            'not_run_or_not_recorded': sum(row['status'] not in ('completed', 'failed') for row in rows),
            'saved_gate_passes': sum((row['gate'] or {}).get('all') is True for row in rows),
            'fixed_updates_per_completed_fit': plan['fixed_optimizer_updates'],
            'automatic_retries': plan['retries'], 'independent_noise_replications_claimed': False},
        'plan': {'path': relative(TRACKED_PLAN), 'sha256': before['plan_sha256'],
            'source_sha256': plan['source_sha256'], 'runtime': plan['runtime'],
            'model_config': plan['model_config'], 'gates': plan['gates'], 'seeds': plan['seeds'],
            'topology': plan['topology'], 'near_bottom_audit': plan['near_bottom_audit'],
            'initialization_policy': plan['optimization_provenance']['initialization_policy'],
            'selection_boundary': plan['optimization_provenance']['selection_boundary']},
        'raw_campaign': {'directory': relative(CAMPAIGN), **before},
        'case_outcomes': rows, 'optimizer_probe': probe,
        'costs': {'campaign_wall_seconds': saved.get('wall_seconds'),
            'peak_rss_bytes': saved.get('peak_rss_bytes'),
            'wrapper_wall_seconds': execution.get('wall_seconds'),
            'scope': 'Fit times in each row exclude later scoring. Probe and wrapper elapsed times are distinct records and are not summed into CPU time.'},
        'execution': {'path': relative(VALIDATION/'execution-state.json'),
            'sha256': digest(VALIDATION/'execution-state.json'), 'saved': execution},
        'report': report,
        'quality_checks': {'spatial_focused': test_evidence('spatial-focused'),
            'reporter_focused': test_evidence('report-focused'),
            'full_repository_suite': test_evidence('full-suite'),
            'overlap_policy': 'Separate invocations are reported independently; overlapping focused tests are not added to the full-suite count.',
            'exact_head_ci': 'Not asserted; verify after the resulting source commit.'},
        'preservation': {'identical_before_and_after': True,
            'before_identity_digest': canonical(legacy_before), 'after_identity_digest': canonical(legacy_after),
            'proof': relative(FINALIZATION/'preservation-after.json'),
            'campaign_before': before, 'campaign_after': after},
        'source_unchanged': saved.get('source_unchanged'),
        'saved_scope_flags': {key: saved.get(key) for key in ('inverse_opened', 'fresh_evaluation_opened', 'three_d_opened')},
        'no_inverse_or_reserved_or_three_d_declared': plan.get('no_inverse_or_reserved_or_three_d'),
        'finalizer': identity,
        'interpretation_limits': [
            'Every pass/fail is a saved coordinator gate; this helper never fits, scores, re-gates, or selects a checkpoint.',
            'All four declared cases remain in the denominator, including failed or unrun cases.',
            'The rule jointly changes learning rate, optimizer history and fixed budget relative to the 80-update log-off baseline; this is not an isolated causal test of optimizer history.',
            'Training-only probes and fresh initializations reuse already opened development datasets; no independent replication or fresh-cohort generalization is claimed.',
            'Fixed final updates are budget endpoints and do not certify convergence.',
            'Near-interface and near-bottom audits remain report only, even when their values exceed primary limits.',
            'Full coordinate arrays, readout records and update trajectories remain in indexed raw reports; compact hashes and JSON pointers bind them.',
            'No inverse recovery, untouched reserved-cohort evaluation, hardware validation, or 3D validation is established.']}
    FINALIZATION.mkdir(parents=True)
    for name, value in (('preservation-before.json', legacy_before), ('preservation-after.json', legacy_after),
                        ('probe-proof.json', probe), ('report-proof.json', report), ('result.json', result)):
        write(FINALIZATION/name, value)
    (FINALIZATION/'finalize-optimized-result.py').write_bytes((BASE/'finalize-optimized-result.py').read_bytes())
    write(TRACKED_RESULT, result)
    files = [{'path': path.name, 'bytes': path.stat().st_size, 'sha256': digest(path)}
        for path in sorted(FINALIZATION.iterdir()) if path.is_file()]
    write(FINALIZATION/'manifest.json', {'files': files, 'digest': canonical(files),
        'tracked_result_sha256': digest(TRACKED_RESULT), 'plan_sha256': before['plan_sha256'],
        'raw_artifact_index_sha256': before['artifact_index_sha256'],
        'report_index_sha256': report['report_index_sha256'], 'finalizer': identity})
    if (verify_campaign(execution) != before or legacy_snapshot(plan) != legacy_after
        or validate_report(before, plan) != report or verify_probe(plan) != probe):
        raise RuntimeError('Preserved evidence changed after result writing; inspect the finalization record')
    output = {'tracked_result': relative(TRACKED_RESULT), 'tracked_result_sha256': digest(TRACKED_RESULT),
        'finalization_manifest': relative(FINALIZATION/'manifest.json'),
        'manifest_sha256': digest(FINALIZATION/'manifest.json'),
        'report': relative(REPORT/'REPORT.md'), 'recorded_status': result['recorded_status'],
        'saved_gate_passes': result['denominator']['saved_gate_passes'],
        'full_suite_status': result['quality_checks']['full_repository_suite']['status']}
    print(json.dumps(output, indent=2))
    return output


if __name__ == '__main__':
    finalize()
