"""Create the compact closed Phase F result from already validated evidence.

Run as `python -m tools.record_prospective_phase_f_result --help` from the repo.
This packages existing observations; it performs no simulation or rule changes.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from tools.verify_prospective_phase_f_result import (
    SOURCE_REVISION, SOURCE_DIGEST, PROTOCOL_DIGEST, RULE_DIGEST, PROCEDURES,
    SELECTOR, CLEARANCES, canonical, file_record, load_sealed, require,
)


def build(*, repository_root, evidence_root, audit_path, test_log, recorded_date):
    repository_root, evidence_root = Path(repository_root).resolve(), Path(evidence_root).resolve()
    raw = evidence_root / 'prospective-phase-f-reserved-7e3b9a3-v1'
    journal = evidence_root / 'phase-f-durable-reserved-7e3b9a3-20261018'
    audit = json.loads(Path(audit_path).read_bytes())
    validation = json.loads((journal / 'independent-validation.json').read_bytes())
    resources = json.loads((raw / 'evaluation.resources.json').read_bytes())
    state = json.loads((journal / 'state.json').read_bytes())
    evaluation = load_sealed(raw / 'evaluation.json', 'thermotwin.prospective_phase_f.evaluation_v1')
    require(audit['audit_passed'] is True and audit['exact_saved_decisions_and_outcome_rows_checked'] == 6000,
            'independent result audit must pass first')
    require(audit['audit_script_file'] == file_record(repository_root / 'tools/verify_prospective_phase_f_result.py'),
            'audit script changed after the independent check')
    require(audit['evaluation_file'] == file_record(raw / 'evaluation.json')
            and validation['evaluation_json_sha256'] == audit['evaluation_file']['sha256']
            and validation['evaluation_resources_sha256'] == file_record(raw / 'evaluation.resources.json')['sha256'],
            'accepted evidence hashes differ')
    require(validation['independent_full_archive_validation_passed'] is True
            and resources['resource_gate_passed'] is True and resources['execution_completed'] is True
            and state['status'] == 'reserved_and_independent_validation_complete'
            and not state['monitor_errors'], 'execution/validation/resource acceptance incomplete')
    require(audit['scientific_result_digest'] == validation['scientific_result_digest']
            == evaluation['scientific_result_digest'], 'scientific result bindings differ')
    context = evaluation['context']
    require(context['source_revision'] == SOURCE_REVISION and context['source_manifest_digest'] == SOURCE_DIGEST
            and context['protocol_digest'] == PROTOCOL_DIGEST, 'closed numerical source/protocol differs')
    for row in context['source_manifest']['files']:
        require(file_record(repository_root / row['path']) == {'bytes': row['byte_count'], 'sha256': row['sha256']},
                'frozen numerical source changed: ' + row['path'])
    text = Path(test_log).read_text()
    require('\nRan 28 tests ' in text and '\nOK\n' in text, 'focused closeout checks have not passed')
    primary = evaluation['primary_analysis']
    selector = primary['procedures'][SELECTOR]
    observed = selector['overall']
    require(primary['primary_success'] is False and not primary['qualifying_fixed_policies'],
            'review changed result classification before packaging')
    selected_rows = [r for r in evaluation['calibrated_case_records_by_clearance']['0.0'] if r['procedure'] == SELECTOR]
    files = {
        'evaluation': raw / 'evaluation.json', 'execution_resources': raw / 'evaluation.resources.json',
        'original_independent_validation': journal / 'independent-validation.json',
        'execution_journal_final_state': journal / 'state.json',
        'execution_stdout': journal / 'stdout.log', 'execution_stderr': journal / 'stderr.log',
        'independent_result_audit': Path(audit_path), 'focused_closeout_tests': Path(test_log),
    }
    evidence = {name: {'path': str(path.resolve().relative_to(evidence_root)), **file_record(path)}
                for name, path in files.items()}
    evidence['independent_audit_script'] = {'path': 'tools/verify_prospective_phase_f_result.py',
                                           **file_record(repository_root / 'tools/verify_prospective_phase_f_result.py')}
    evidence['result_builder_script'] = {'path': 'tools/record_prospective_phase_f_result.py', **file_record(__file__)}
    result = {
        'schema_version': 1, 'protocol_version': 'operating_decision_prospective_phase_f_result_v1',
        'result_record_date': recorded_date, 'scientific_status': 'closed_primary_success_not_established',
        'campaign': evaluation['generation_gate']['campaign'], 'partition': evaluation['partition'],
        'source_revision': SOURCE_REVISION, 'source_manifest_digest': SOURCE_DIGEST,
        'protocol_digest': PROTOCOL_DIGEST, 'complete_rule_artifact_digest': RULE_DIGEST,
        'calibration_artifact_digest': evaluation['generation_gate']['calibration_artifact_digest'],
        'generation_gate_artifact_digest': evaluation['generation_gate']['artifact_digest'],
        'generation_gate_commit': validation['gate_commit'], 'runtime_identity': context['runtime_identity'],
        'scientific_result_digest': evaluation['scientific_result_digest'],
        'execution': {'started_at_utc': state['started_at_utc'],
                      'independent_validation_completed_at_utc': validation['completed_at_utc'],
                      'block_count': 100, 'case_count': 300, 'procedure_count': 5,
                      'primary_case_procedure_outcome_count': 1500,
                      'all_clearance_outcome_count': 6000, 'before_reveal_file_count': 300,
                      'resumed_blocks': evaluation['performance']['resumed_blocks'],
                      'cohort_closed_against_regeneration': True,
                      'sample_size_extended_after_outcomes': False},
        'predeclared_primary_targets': {
            'decision_clearance_kelvin': 0., 'false_approval_risk_upper_at_most': .10,
            'decision_coverage_lower_at_least': .70, 'paired_coverage_loss_upper_at_most': .05,
            'false_approval_risk_increase_upper_at_most': .02, 'paired_energy_saving_lower_strictly_above_joules': 0.,
            'familywise_alpha': .05, 'one_sided_primary_bounds': 73,
            'all_qualifying_fixed_comparators_required': True, 'at_least_one_qualifying_fixed_comparator_required': True,
            'energy_inference_is_approximate': True},
        'primary_analysis': primary,
        'secondary_observed_results_by_clearance': {
            str(c): {'scientific_use': 'descriptive_only_no_threshold_selection_or_confirmatory_inference',
                     'procedures': {p: evaluation['secondary_analysis_by_clearance'][str(c)]['analysis']['procedures'][p]['overall']
                                    for p in PROCEDURES}}
            for c in CLEARANCES if c != 0.},
        'primary_selected_action_counts': dict(Counter(r['selected_action'] for r in selected_rows)),
        'scientific_conclusion': {
            'primary_success': False, 'observed_selector_decision_rate': observed['decision_coverage']['estimate'],
            'observed_decisiveness_target_missed': observed['decision_coverage']['estimate'] < .70,
            'selector_passed_absolute_quality_qualification': selector['qualifies_for_primary_comparison'],
            'at_least_one_fixed_policy_qualified': False, 'approval_risk_ceiling_established': False,
            'strongest_fixed_strategy_superiority_established': False,
            'energy_saving_check_passed_against': [p for p, c in primary['comparisons'].items() if c['checks']['energy_saving']],
            'risk_assurance_is_inconclusive_not_an_observed_31_percent_error_rate': True,
            'diagnostic_energy_is_simulated_not_computer_energy_or_hardware_validation': True,
            'interval_coverage_and_decision_coverage_are_different_endpoints': True},
        'independent_verification': audit,
        'compute_and_resources': {'completed_block_cpu_seconds': evaluation['performance']['aggregate_completed_block_cpu_seconds'],
                                  'execution_invocation': resources,
                                  'external_final_state': {k: state[k] for k in (
                                      'wall_seconds_this_invocation', 'calendar_elapsed_seconds',
                                      'process_tree_peak_rss_bytes', 'process_tree_sample_count', 'monitor_errors')},
                                  'cpu_cost_is_for_whole_benchmark_not_deployed_single_case': True},
        'evidence_root': '.artifacts/operating-decision', 'evidence': evidence,
        'remaining_closeout': {'exploratory_abstention_analysis_complete': False,
                              'final_three_figures_complete': False, 'final_report_complete': False,
                              'presentation_complete': False, 'external_evidence_deposit_complete': False,
                              'predetermined_numerical_refit_replay_complete': False},
        'new_observations_generated_for_this_result_record': False,
    }
    result['artifact_digest'] = hashlib.sha256(canonical(result)).hexdigest()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--test-log', type=Path, required=True)
    parser.add_argument('--recorded-date', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), 'refuse to overwrite an accepted result record')
    result = build(repository_root=args.repository_root, evidence_root=args.evidence_root,
                   audit_path=args.audit, test_log=args.test_log, recorded_date=args.recorded_date)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'artifact_digest': result['artifact_digest'],
                      'scientific_status': result['scientific_status'], 'result_bytes': args.output.stat().st_size}, indent=2))


if __name__ == '__main__':
    main()
