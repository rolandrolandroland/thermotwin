"""Post hoc diagnosis of the CLOSED Phase F archive; no fitting or simulation.

Use python -m tools.analyze_prospective_phase_f_abstentions --help.
The summary is suitable for version control; the detailed rows belong in the
ignored evidence directory. Both output paths must be new and outside the raw
archive. Nothing here changes a decision or validates a redesigned selector.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from tools import verify_prospective_phase_f_result as audit


ABSTAIN = 'insufficient_evidence'
ACTIONS = audit.POLICIES[1:]
EVALUATION_SHA256 = '299f75ad492e5bde0acc724ed4e72ae029c86f2c05538677c2a856c23734c564'
RESULT_DIGEST = 'd565c0c6368b602eeb3d2721c128aab88221bce9a6b5abe68fcda4c727d4edf0'
STAGES = ('raw_interval_crosses_zero', 'development_padding', 'final_calibration',
          'invalid_or_unverified_interval', 'definitive')


def describe(values):
    values = list(values)
    if not values:
        return {'count': 0, 'mean': None, 'median': None, 'minimum': None, 'maximum': None}
    audit.require(all(audit.finite(v) for v in values), 'nonfinite diagnostic value')
    return {'count': len(values), 'mean': audit.average(values),
            'median': statistics.median(values), 'minimum': min(values), 'maximum': max(values)}


def width(interval):
    audit.require(isinstance(interval, dict) and audit.finite(interval['lower'])
                  and audit.finite(interval['upper']) and interval['lower'] <= interval['upper'],
                  'invalid interval in width diagnosis')
    return interval['upper'] - interval['lower']


def classify(interval):
    if interval is None:
        return ABSTAIN
    width(interval)
    return 'approve' if interval['lower'] >= 0 else 'reject' if interval['upper'] < 0 else ABSTAIN


def layers(row):
    """First layer at which the interval becomes indecisive; exact saved rounding."""
    inp = {k: row[k] for k in audit.BLINDED_FIELDS}
    raw = audit.final_decision({**inp, 'development_offset_kelvin': 0.}, 0., 0.)
    development = audit.final_decision(inp, 0., 0.)
    final = audit.final_decision(inp, row['correction_kelvin'], 0.)
    audit.require(final['decision'] == row['decision'] and final['final_interval'] == row['final_interval'],
                  'primary interval/decision differs during layer diagnosis')
    if final['decision'] != ABSTAIN:
        stage = 'definitive'
    elif final['final_interval'] is None:
        stage = 'invalid_or_unverified_interval'
    elif raw['decision'] == ABSTAIN:
        stage = 'raw_interval_crosses_zero'
    elif development['decision'] == ABSTAIN:
        stage = 'development_padding'
    else:
        stage = 'final_calibration'
    return {'first_abstaining_layer': stage, 'raw': raw, 'development': development, 'final': final}


def envelope_geometry(intervals):
    audit.require(bool(intervals), 'no retained model intervals')
    hull = {'lower': min(v['lower'] for v in intervals), 'upper': max(v['upper'] for v in intervals)}
    individual = Counter(classify(i) for i in intervals)
    if classify(hull) != ABSTAIN:
        ambiguity = 'raw_envelope_definitive'
    elif individual[ABSTAIN]:
        ambiguity = 'at_least_one_individual_interval_crosses_zero'
    else:
        audit.require(individual['approve'] and individual['reject'], 'unexpected union ambiguity')
        ambiguity = 'opposite_definitive_model_intervals_only'
    return {'model_count': len(intervals), 'envelope': hull, 'ambiguity': ambiguity,
            'maximum_individual_width_kelvin': max(width(i) for i in intervals),
            'excess_envelope_span_kelvin': max(0., width(hull) - max(width(i) for i in intervals))}


def padded_draw_width(draw, baseline, offset):
    """Independent arithmetic copy of the documented frozen draw-scoring convention."""
    if draw['failed']:
        return baseline
    value = draw['raw_after_width'] + 2. * offset
    return max(baseline, value) if draw['initially_admissible_became_inadmissible'] else value


def forecast_record(case, action, offsets):
    result = case['n16_complete_uncertainty_result']
    selection = case['frozen_selection_before_reveal']
    item = next(v for v in selection['action_evaluations'] if v['policy_name'] == action)
    audit.require(item['eligible'] is True, 'unexpected ineligible action in closed cohort')
    baseline = width(result['acquisition_evidence']['snapshot']['provisional_margin_envelope']) + 2. * offsets['stop_now']
    draws = [d for d in result['draw_outcomes'] if d['policy_name'] == action]
    sources = result['acquisition_evidence']['snapshot']['admissible_candidate_models']
    audit.require(len(draws) == 16 * len(sources), 'missing predictive draw')
    source_means = {}
    for source in sources:
        group = [d for d in draws if d['generator_model'] == source]
        audit.require(len(group) == 16 and sorted(d['draw_index'] for d in group) == list(range(16)),
                      'incomplete predictive source matrix')
        source_means[source] = audit.average([padded_draw_width(d, baseline, offsets[action]) for d in group])
    after = max(source_means.values())
    gain = baseline - after
    for actual, expected in ((item['uncertainty_before'], baseline),
                             (item['expected_uncertainty_after'], after),
                             (item['expected_uncertainty_reduction'], gain),
                             (item['utility_per_cost'], gain / item['declared_cost'])):
        audit.require(math.isclose(actual, expected, rel_tol=2e-12, abs_tol=2e-10),
                      'retained forecast does not match independently aggregated draws')
    return {'action': action, 'eligible': item['eligible'], 'declared_cost': item['declared_cost'],
            'baseline_padded_width_kelvin': baseline,
            'predicted_padded_width_kelvin': item['expected_uncertainty_after'],
            'predicted_reduction_kelvin': item['expected_uncertainty_reduction'],
            'predicted_utility_per_cost': item['utility_per_cost'],
            'draw_count': len(draws), 'source_count': len(sources),
            'failed_draw_count': sum(d['failed'] for d in draws),
            'candidate_loss_draw_count': sum(bool(d['initially_admissible_became_inadmissible']) for d in draws),
            'candidate_recovery_draw_count': sum(bool(d['initially_excluded_became_admissible']) for d in draws)}


def case_record(case, keyed, offsets):
    block, family = case['block'], case['truth_condition']
    row = keyed[block, family, audit.SELECTOR]
    action = row['selected_action']
    selection = case['frozen_selection_before_reveal']
    audit.require(selection['selection_succeeded'] and selection['selected_policy'] == action,
                  'saved selection differs from scored row')
    snapshot = case['n16_complete_uncertainty_result']['acquisition_evidence']['snapshot']
    initial = set(snapshot['admissible_candidate_models'])
    saved = case['fixed_policy_results'][action]['saved_before_reveal']
    geometry = envelope_geometry(saved['model_intervals'])
    audit.require(geometry['envelope'] == row['raw_interval'], 'model interval hull differs from saved envelope')
    verified = {v['model_name'] for v in saved['model_intervals']}
    numerically_admissible = {v['fit']['model_name'] for v in saved['verifications']
                             if v['fit']['converged'] and not v['fit']['reached_bound']}
    layer = layers(row)
    forecasts = []
    fixed_outcomes = {}
    for policy in audit.POLICIES:
        fixed = keyed[block, family, policy]
        fixed_outcomes[policy] = {'decision': fixed['decision'],
                                  'wrong_decision': fixed['false_approval'] or fixed['false_rejection'],
                                  'correction_kelvin': fixed['correction_kelvin']}
        if policy == 'stop_now':
            continue
        forecast = forecast_record(case, policy, offsets)
        # This is the saved verified envelope, not a new acquisition-only refit.
        observed_width = width(fixed['raw_interval']) + 2. * offsets[policy]
        realized = forecast['baseline_padded_width_kelvin'] - observed_width
        forecasts.append({**forecast, 'selected': policy == action,
                          'realized_verified_padded_width_kelvin': observed_width,
                          'realized_verified_reduction_kelvin': realized,
                          'realized_minus_predicted_width_kelvin': observed_width - forecast['predicted_padded_width_kelvin'],
                          'realized_minus_predicted_reduction_kelvin': realized - forecast['predicted_reduction_kelvin'],
                          'fixed_procedure_decision': fixed['decision'],
                          'selector_decision_if_selected': row['decision'] if policy == action else None})
    return {'block': block, 'truth_family': family, 'device_token': row['device_token'],
            'selected_action': action, 'decision': row['decision'],
            'first_abstaining_layer': layer['first_abstaining_layer'],
            'raw_decision': layer['raw']['decision'], 'development_decision': layer['development']['decision'],
            'raw_interval': row['raw_interval'], 'development_interval': layer['development']['final_interval'],
            'final_interval': row['final_interval'], 'development_offset_kelvin': row['development_offset_kelvin'],
            'selector_correction_kelvin': row['correction_kelvin'],
            'true_margin_kelvin': row['true_margin'],
            'final_interval_covered': row['interval_covered'],
            'wrong_decision': row['false_approval'] or row['false_rejection'],
            'selection_reason': selection['reason'], 'provisional_decision': selection['provisional_decision'],
            'initial_admissible_models': sorted(initial),
            'initial_candidate_stratum': selection['candidate_reliability_stratum'],
            'final_verified_models': sorted(verified),
            'lost_initial_models_at_numerical_fit': sorted(initial - numerically_admissible),
            'lost_initial_models_after_verification': sorted(initial - verified),
            'recovered_models_after_verification': sorted(verified - initial),
            'candidate_verification': [{'model': v['fit']['model_name'], 'passed': v['passed'],
                                        'failure_reason': v['failure_reason'],
                                        'converged': v['fit']['converged'], 'reached_bound': v['fit']['reached_bound']}
                                       for v in saved['verifications']],
            'raw_geometry': geometry, 'fixed_outcomes': fixed_outcomes, 'action_diagnostics': forecasts}


def count_outcomes(rows):
    counts = Counter(r['decision'] for r in rows)
    return {'cases': len(rows), 'definitive': len(rows) - counts[ABSTAIN], 'abstentions': counts[ABSTAIN],
            'approvals': counts['approve'], 'rejections': counts['reject'],
            'wrong_decisions': sum(r['wrong_decision'] for r in rows),
            'first_abstaining_layer': {k: sum(r['first_abstaining_layer'] == k for r in rows) for k in STAGES},
            'absolute_true_margin_kelvin': describe(abs(r['true_margin_kelvin']) for r in rows),
            'raw_width_kelvin': describe(width(r['raw_interval']) for r in rows),
            'final_width_kelvin': describe(width(r['final_interval']) for r in rows)}


def forecast_summary(rows):
    positive = [r for r in rows if r['predicted_reduction_kelvin'] > 0]
    selected = [r for r in rows if r['selected']]
    return {'case_action_count': len(rows),
            'predicted_reduction_kelvin': describe(r['predicted_reduction_kelvin'] for r in rows),
            'realized_verified_reduction_kelvin': describe(r['realized_verified_reduction_kelvin'] for r in rows),
            'realized_minus_predicted_width_kelvin': describe(r['realized_minus_predicted_width_kelvin'] for r in rows),
            'absolute_width_difference_kelvin': describe(abs(r['realized_minus_predicted_width_kelvin']) for r in rows),
            'predicted_positive_gain_count': len(positive),
            'predicted_positive_but_no_realized_narrowing_count': sum(r['realized_verified_reduction_kelvin'] <= 0 for r in positive),
            'realized_narrowing_count': sum(r['realized_verified_reduction_kelvin'] > 0 for r in rows),
            'selected_case_count': len(selected),
            'selected_abstentions': sum(r['selector_decision_if_selected'] == ABSTAIN for r in selected),
            'selected_narrowed_but_abstained': sum(r['realized_verified_reduction_kelvin'] > 0 and
                                                  r['selector_decision_if_selected'] == ABSTAIN for r in selected),
            'predictive_draw_count': sum(r['draw_count'] for r in rows),
            'predictive_failed_draw_count': sum(r['failed_draw_count'] for r in rows),
            'predictive_candidate_loss_draw_count': sum(r['candidate_loss_draw_count'] for r in rows),
            'predictive_candidate_recovery_draw_count': sum(r['candidate_recovery_draw_count'] for r in rows)}


def summarize(cases):
    audit.require(len(cases) == 300 and len({(r['block'], r['truth_family']) for r in cases}) == 300,
                  'diagnosis must retain all 300 cases')
    abstentions = [r for r in cases if r['decision'] == ABSTAIN]
    definitive = [r for r in cases if r['decision'] != ABSTAIN]
    measured = [r for r in cases if r['selected_action'] != 'stop_now']
    actions = [r for c in cases for r in c['action_diagnostics']]
    selected = [r for r in actions if r['selected']]
    unchanged_models = [r for c in measured if c['initial_admissible_models'] == c['final_verified_models']
                        for r in c['action_diagnostics'] if r['selected']]
    fixed_overlap = {}
    for policy in audit.POLICIES:
        resolved = [r for r in abstentions if r['fixed_outcomes'][policy]['decision'] != ABSTAIN]
        lost = [r for r in definitive if r['fixed_outcomes'][policy]['decision'] == ABSTAIN]
        fixed_overlap[policy] = {
            'selector_abstentions_with_fixed_definitive': len(resolved),
            'wrong_among_those_fixed_definitive': sum(r['fixed_outcomes'][policy]['wrong_decision'] for r in resolved),
            'selector_definitive_with_fixed_abstention': len(lost),
            'selector_and_fixed_same_case_definitive': sum(r['fixed_outcomes'][policy]['decision'] != ABSTAIN for r in definitive)}
    patterns = Counter('|'.join(p for p in audit.POLICIES if r['fixed_outcomes'][p]['decision'] != ABSTAIN) or 'none'
                       for r in abstentions)
    all_fixed_abstain = [r for r in abstentions if all(v['decision'] == ABSTAIN for v in r['fixed_outcomes'].values())]
    same_action_differences = Counter()
    for row in cases:
        fixed = row['fixed_outcomes'][row['selected_action']]['decision']
        if row['decision'] != fixed:
            same_action_differences[row['selected_action'] + ':' + row['decision'] + '->' + fixed] += 1
    # Removing padding is an uncalibrated diagnostic, not a substitute result.
    uncalibrated = {}
    for stage, key in (('raw', 'raw_decision'), ('development_only', 'development_decision'), ('final', 'decision')):
        approvals = [r for r in cases if r[key] == 'approve']
        rejections = [r for r in cases if r[key] == 'reject']
        uncalibrated[stage] = {'definitive': len(approvals) + len(rejections), 'approvals': len(approvals),
                               'false_approvals': sum(r['true_margin_kelvin'] < 0 for r in approvals),
                               'false_rejections': sum(r['true_margin_kelvin'] >= 0 for r in rejections)}
    return {
        'overall': count_outcomes(cases), 'abstentions': count_outcomes(abstentions),
        'definitive': count_outcomes(definitive), 'measured': count_outcomes(measured),
        'by_selected_action': {p: count_outcomes([r for r in cases if r['selected_action'] == p]) for p in audit.POLICIES},
        'by_truth_family': {f: count_outcomes([r for r in cases if r['truth_family'] == f]) for f in audit.FAMILIES},
        'by_initial_candidate_stratum': {s: count_outcomes([r for r in cases if r['initial_candidate_stratum'] == s])
                                         for s in sorted({r['initial_candidate_stratum'] for r in cases})},
        'by_selection_reason': {s: count_outcomes([r for r in cases if r['selection_reason'] == s])
                                for s in sorted({r['selection_reason'] for r in cases})},
        'by_family_and_selected_action': {f: {p: count_outcomes([r for r in cases if r['truth_family'] == f and
                                                               r['selected_action'] == p]) for p in audit.POLICIES}
                                          for f in audit.FAMILIES},
        'abstaining_raw_geometry_counts': dict(sorted(Counter(r['raw_geometry']['ambiguity'] for r in abstentions).items())),
        'abstentions_with_numerical_candidate_loss': sum(bool(r['lost_initial_models_at_numerical_fit']) for r in abstentions),
        'abstentions_with_any_candidate_loss_after_verification': sum(bool(r['lost_initial_models_after_verification']) for r in abstentions),
        'abstentions_with_recovered_models': sum(bool(r['recovered_models_after_verification']) for r in abstentions),
        'fixed_outcome_overlap_among_same_cases': fixed_overlap,
        'fixed_definitive_patterns_among_selector_abstentions': dict(sorted(patterns.items())),
        'selector_abstentions_all_fixed_also_abstain': len(all_fixed_abstain),
        'selector_abstentions_any_fixed_definitive': len(abstentions) - len(all_fixed_abstain),
        'retrospective_fixed_resolutions_by_selection_reason': {
            reason: {'selector_abstentions': sum(r['selection_reason'] == reason for r in abstentions),
                     'any_fixed_definitive': sum(r['selection_reason'] == reason and
                                                any(v['decision'] != ABSTAIN for v in r['fixed_outcomes'].values())
                                                for r in abstentions)}
            for reason in sorted({r['selection_reason'] for r in cases})},
        'same_acquisition_different_procedure_correction_decision_changes': dict(sorted(same_action_differences.items())),
        'uncalibrated_padding_removal_diagnostic': uncalibrated,
        'forecast_vs_observed_all_900_case_actions': forecast_summary(actions),
        'forecast_vs_observed_selected_125_measurements': forecast_summary(selected),
        'forecast_vs_observed_selected_with_unchanged_verified_model_set': forecast_summary(unchanged_models),
        'forecast_vs_observed_by_action_all_cases': {p: forecast_summary([r for r in actions if r['action'] == p]) for p in ACTIONS},
        'forecast_vs_observed_by_selected_action': {p: forecast_summary([r for r in selected if r['action'] == p]) for p in ACTIONS},
        'forecast_vs_observed_selected_by_truth_family': {
            f: forecast_summary([r for c in cases if c['truth_family'] == f for r in c['action_diagnostics'] if r['selected']])
            for f in audit.FAMILIES},
    }


def diagnose(*, archive, source_root, validation_record, recorded_date):
    audit.require(audit.file_record(archive)['sha256'] == EVALUATION_SHA256, 'not the accepted closed archive')
    saved, reconstructed = audit.reconstruct(archive, source_root, validation_record)
    audit.require(saved['scientific_result_digest'] == RESULT_DIGEST, 'scientific result identity changed')
    keyed = audit.matrix(reconstructed['0.0'])
    rule_path = Path(source_root) / 'thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_F_RULE.json'
    rule = json.loads(rule_path.read_bytes())
    offsets = rule['design']['selector_rule']['development_offsets']['values']
    cases = []
    for entry in saved['block_file_records']:
        path = Path(archive).parent / entry['path']
        # Recheck bytes at the point of diagnostic extraction as well as in the
        # independent reconstruction. No unchecked side archive is accepted.
        audit.require(audit.file_record(path)['sha256'] == entry['sha256'], 'checkpoint changed during read')
        checkpoint = json.loads(path.read_bytes())
        cases.extend(case_record(c, keyed, offsets) for c in checkpoint['block_result']['numerical_block']['cases'])
    audit.require(audit.file_record(archive)['sha256'] == EVALUATION_SHA256, 'evaluation changed during diagnosis')
    result = {
        'schema_version': 1, 'analysis_version': 'closed_phase_f_exploratory_abstention_diagnosis_v1',
        'recorded_date': recorded_date,
        'scientific_use': 'post_hoc_descriptive_only_no_confirmatory_inference_or_policy_selection',
        'closed_campaign': rule['design']['campaign'], 'closed_partition': saved['partition'],
        'scientific_result_digest': RESULT_DIGEST, 'evaluation_file': audit.file_record(archive),
        'original_validation_file': audit.file_record(validation_record),
        'analysis_script_file': audit.file_record(__file__),
        'independent_archive_reconstruction_script_file': audit.file_record(audit.__file__),
        'complete_rule_file': audit.file_record(rule_path), 'complete_rule_digest': rule['artifact_digest'],
        'source_revision': audit.SOURCE_REVISION, 'source_manifest_digest': audit.SOURCE_DIGEST,
        'frozen_numerical_paths_verified': len(saved['context']['source_manifest']['files']),
        'raw_blocks_verified': 100, 'before_reveal_files_verified': 300, 'saved_outcomes_reconstructed': 6000,
        'new_observations_generated': False, 'fits_or_physical_simulations_run': False,
        'selector_rule_or_calibration_changed': False, 'reserved_cohort_extended_or_reopened': False,
        'primary_success_reclassified': False, 'primary_success': saved['primary_analysis']['primary_success'],
        'frozen_development_offsets_kelvin': offsets, 'frozen_final_corrections_kelvin': rule['final_corrections_kelvin'],
        'conventions': {
            'abstention_layer': 'first crossing at raw, development-padded, or calibrated interval; exact outward rounding, approve lower >= 0, reject upper < 0',
            'prediction': 'saved N16 development-padded width, mean over every draw within source then maximum across initial sources; failures and candidate attrition retain frozen penalties',
            'realized_width': 'saved post-verification raw candidate envelope plus twice the frozen action development offset; no final q in either width reduction',
            'shared_baseline': 'same acquisition-only initial envelope plus twice stop development offset for every action, including negative realized gains',
            'stage_difference': 'forecasts precede independent verification; observed widths include its candidate filtering. Their difference is descriptive, not a calibrated prediction-error test',
            'selected_groups': 'selected cases differ in difficulty; action-group contrasts are not causal measurement effects',
            'fixed_overlap': 'each fixed procedure retains its own q; retrospective resolutions are not an executable selector or a guarantee for alternative actions under selector q',
            'geometry': 'excess envelope span is hull width minus maximum individual width; decomposition is descriptive, not a causal attribution',
            'independence': '100 paired blocks, three families; no p-values, new confidence bounds, threshold tuning, or iid-case inference',
            'zero_boundary': 'a raw interval ending exactly at zero still abstains unless its lower endpoint is zero or above',
        },
        'summary': summarize(cases),
    }
    return result, {'schema_version': 1, 'scientific_use': result['scientific_use'],
                    'scientific_result_digest': RESULT_DIGEST, 'cases': cases}


def write_new(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--validation-record', type=Path, required=True)
    parser.add_argument('--recorded-date', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case-output', type=Path, required=True)
    args = parser.parse_args(argv)
    audit.require(args.output.resolve() != args.case_output.resolve(), 'outputs must be distinct')
    for path in (args.output, args.case_output):
        audit.require(not path.exists(), 'refuse to overwrite an existing record')
        audit.require(not path.resolve().is_relative_to(args.archive.parent.resolve()), 'outputs must be outside closed raw archive')
    result, details = diagnose(archive=args.archive, source_root=args.source_root,
                              validation_record=args.validation_record, recorded_date=args.recorded_date)
    write_new(args.case_output, details)
    result['case_detail_file'] = audit.file_record(args.case_output)
    result['case_detail_digest'] = audit.digest('thermotwin.phase_f.exploratory_abstention_cases_v1', details)
    result['artifact_digest'] = audit.digest('thermotwin.phase_f.exploratory_abstention_summary_v1', result)
    write_new(args.output, result)
    print(json.dumps({'summary_path': str(args.output), 'case_detail_path': str(args.case_output),
                      'artifact_digest': result['artifact_digest'], 'cases': 300,
                      'abstentions': result['summary']['abstentions']['cases']}, indent=2))


if __name__ == '__main__':
    main()
