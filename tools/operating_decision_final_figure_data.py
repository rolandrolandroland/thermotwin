"""Prepare auditable, self-contained inputs for the three final study figures.

Uses only the accepted Phase F outcomes and Phase D5 maps. The illustrative
curves are forward projections of retained candidate fits from the original
source/runtime: no optimization, observation generation, or target-truth replay.
The resulting compact JSON can render all figures without the large archives.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from tools import verify_prospective_phase_f_result as audit


DATA_DOMAIN = 'thermotwin.operating_decision.final_figure_inputs_v1'
F_SHA = '299f75ad492e5bde0acc724ed4e72ae029c86f2c05538677c2a856c23734c564'
D5_SHA = '8ee535b760be9254bb9ddba8b82a909ee6fce6dd2f8c07b5790e47bdf4bf95c2'
D5_SCIENCE = 'b26352040b44c595f86d76003dd55bd7a2d8b43c0ccf9b825dda53c0a6edf4f3'


def illustrative_identity(rows, snapshots):
    """Post hoc illustrative rule fixed before any plotted curve is computed."""
    candidates = []
    order = {family: index for index, family in enumerate(audit.FAMILIES)}
    for row in rows:
        if row['procedure'] != audit.SELECTOR or row['selected_action'] == 'stop_now' or row['decision'] != 'insufficient_evidence':
            continue
        snapshot = snapshots[row['block'], row['truth_family']]
        if len(snapshot['admissible_candidate_models']) != 2:
            continue
        initial = snapshot['provisional_margin_envelope']
        current = row['raw_interval']
        if current is not None and (current['upper'] - current['lower'] + 2 * row['development_offset_kelvin']
                                    < initial['upper'] - initial['lower']):
            candidates.append(row)
    audit.require(bool(candidates), 'no case satisfies the declared illustrative rule')
    row = min(candidates, key=lambda r: (r['block'], order[r['truth_family']]))
    return row['block'], row['truth_family']


def profiled_offset(observed, predicted, noise, bias_ratio):
    audit.require(len(observed) == len(predicted) and bool(observed), 'profiled-offset vectors differ')
    audit.require(noise > 0 and bias_ratio > 0, 'nonpositive channel-noise setting')
    variance = (noise * bias_ratio)**2
    return variance * sum(a - b for a, b in zip(observed, predicted)) / (noise**2 + len(observed) * variance)


def map_cell(cell):
    rows = cell['case_records']
    expected = {(b, f) for b in range(20) for f in audit.FAMILIES}
    audit.require(len(rows) == 60 and {(r['block'], r['truth_family']) for r in rows} == expected,
                  'development map does not retain the complete 60-case matrix')
    actions = []
    for action in (*audit.POLICIES, 'selection_failure'):
        selected = [r for r in rows if (r['selection_failure'] if action == 'selection_failure'
                                       else not r['selection_failure'] and r['selected_policy'] == action)]
        actions.append({'action': action, 'selected': len(selected),
                        'definitive': sum(r['definitive_decision'] for r in selected),
                        'abstentions': sum(not r['definitive_decision'] for r in selected)})
    audit.require(sum(a['selected'] for a in actions) == 60, 'a map case was dropped')
    summary = cell['summary']
    for actual, original in zip(actions, summary['actions']):
        audit.require(actual['action'] == original['action'] and actual['selected'] == original['selected_count']
                      and actual['definitive'] == original['definitive_count'], 'map action summary differs from case records')
    counts = {'definitive': sum(r['definitive_decision'] for r in rows),
              'no_useful_measurement': sum(r['no_useful_measurement'] for r in rows),
              'selection_failure': sum(r['selection_failure'] for r in rows),
              'verification_failure': sum(r['verification_failure'] for r in rows),
              'pipeline_failure': sum(r['pipeline_failure'] for r in rows),
              'ineligible_actions': {p: sum(p in r['ineligible_actions'] for r in rows) for p in audit.POLICIES[1:]}}
    for name in ('definitive', 'no_useful_measurement', 'selection_failure', 'verification_failure', 'pipeline_failure'):
        stored = 'definitive_decision_count' if name == 'definitive' else name + '_count'
        audit.require(counts[name] == summary[stored], 'map outcome summary differs: ' + name)
    audit.require(counts['ineligible_actions'] == summary['ineligible_action_counts'], 'map ineligibility differs')
    return {'sensor_scenario': cell['sensor_scenario'], 'cost_scenario': cell['cost_scenario'],
            'primary': cell['primary'], 'case_count': 60, 'actions': actions, **counts}


def development_inputs(payload):
    protocol = payload['phase_d5_protocol']
    sensors, costs = protocol['sensor_scenarios'], protocol['cost_scenarios']
    audit.require(len(sensors) == 4 and len(costs) == 12 and payload['measurement_map_count'] == 48,
                  'not the complete declared map')
    cells = [map_cell(cell) for cell in payload['measurement_maps']]
    expected = {(s['name'], c['name']) for s in sensors for c in costs}
    audit.require(len(cells) == 48 and {(c['sensor_scenario'], c['cost_scenario']) for c in cells} == expected,
                  'missing or duplicate sensor/cost cell')
    audit.require(sum(c['primary'] for c in cells) == 1, 'primary map setting is ambiguous')
    for cost in costs:
        audit.require(math.isclose(sum(cost[k] for k in ('energy_weight', 'bench_time_weight', 'instrumentation_weight')), 1., abs_tol=1e-12),
                      'cost weights are not convex')
    return {'scientific_use': 'development_sensitivity_only_before_final_calibration_not_reserved_comparisons',
            'block_count': 20, 'cases_per_cell': 60, 'map_cell_count': 48,
            'cells_are_correlated_reuses_of_development_blocks': True,
            'sensor_scenarios': sensors, 'cost_scenarios': costs, 'cells': cells,
            'frozen_design_correction_q_applied': False,
            'reserved_setting': {'sensor_scenario': 'nominal', 'cost_scenario': 'balanced_face_equal'}}


def comparison_inputs(saved, rows, accepted):
    audit.require(audit.canonical(accepted['primary_analysis']) == audit.canonical(saved['primary_analysis']),
                  'accepted result and raw analysis differ')
    data = []
    for procedure in audit.PROCEDURES:
        group = [r for r in rows if r['procedure'] == procedure]
        audit.require(len(group) == 300 and all(r['final_interval'] is not None for r in group), 'missing comparison outcome')
        summary = saved['primary_analysis']['procedures'][procedure]
        audit.compare(audit.metrics(group), summary['overall'])
        data.append({'procedure': procedure, 'observed': summary['overall'],
                     'decision_coverage_lower': summary['population_equal_family_decision_coverage_lower'],
                     'approval_risk_bounds': summary['population_mixture_risk_bounds'],
                     'block_interval_coverage': summary['simultaneous_block_interval_coverage'],
                     'final_interval_widths_kelvin': [r['final_interval']['upper'] - r['final_interval']['lower'] for r in group],
                     'qualifies': summary['qualifies_for_primary_comparison']})
    audit.require(not saved['primary_analysis']['primary_success'] and not any(r['qualifies'] for r in data),
                  'review a changed primary scientific conclusion')
    return {'scientific_use': 'predeclared_primary_reserved_comparison', 'block_count': 100,
            'case_count': 300, 'cases_are_not_iid_replicates': True, 'procedures': data,
            'targets': accepted['predeclared_primary_targets'],
            'compute': accepted['compute_and_resources'], 'primary_success': False,
            'qualifying_fixed_policies': saved['primary_analysis']['qualifying_fixed_policies'],
            'paired_comparisons': saved['primary_analysis']['comparisons']}


def illustrative_curves(case, row, saved, modules):
    from tools.replay_prospective_phase_f_numerics import decode_case
    o, r, u, f = (modules['studies.' + n] for n in (
        'operating_decision', 'operating_decision_realism', 'operating_decision_prospective_uncertainty',
        'operating_decision_prospective_phase_f_execution'))
    config = f.e._physical_config(f.phase_f_partition(saved['partition']))
    action = row['selected_action']
    observed = decode_case(case['fixed_policy_results'][action]['case'], modules)
    initial = observed.acquisition_runs[0]
    added = observed.acquisition_runs[1]
    audit.require(not added.instrumentation.temporary_face_sensor, 'illustrative acquisition projection needs an explicit probe model')
    evidence = case['n16_complete_uncertainty_result']['acquisition_evidence']
    raw_saved = case['fixed_policy_results'][action]['saved_before_reveal']
    calls = 0

    def fit(payload):
        args = dict(payload)
        for name in ('log_multipliers', 'parameter_names', 'physical_values'):
            args[name] = tuple(args[name])
        args['covariance'] = tuple(tuple(row) for row in args['covariance'])
        args['face_sensor'] = None if args['face_sensor'] is None else r.TemporaryFaceSensor(**args['face_sensor'])
        value = r.RealisticCandidateFit(**args)
        audit.require(audit.canonical(u._fit_payload(value)) == audit.canonical(payload), 'candidate fit transport differs')
        audit.require(value.model_name in ('four_state', 'five_state'), 'plot cannot simulate a truth-family model')
        return value

    def predict(candidate, regime, loaded=False):
        nonlocal calls
        calls += 1
        audit.require(not loaded, 'this example must remain uninstrumented for projections')
        return r._simulate_realistic_observables(candidate.model_name, regime.current, regime.channels,
                                                candidate.physical_values, candidate.interface_mass,
                                                candidate.series_resistance, None, config)

    def measured(run):
        return {channel: {'time': [v.time for v in run.observations.values if v.channel == channel],
                          'value': [v.value for v in run.observations.values if v.channel == channel]}
                for channel in run.regime.channels}

    def channel_series(prediction, run, profile):
        result = {}
        for channel, obs in measured(run).items():
            values = {v.time: v.value for v in prediction.values if v.channel == channel}
            predicted = [values[t] for t in obs['time']]
            bias = profiled_offset(obs['value'], predicted, dict(config.sensor.channel_noise)[channel], config.sensor.run_bias_noise_ratio) if profile else 0.
            result[channel] = {'time': obs['time'], 'value': [v + bias for v in predicted], 'profiled_offset': bias}
        return result

    initial_curves, updated_curves = [], []
    intervals = {v['model_name']: v for v in evidence['snapshot']['model_intervals']}
    for payload in evidence['fit_set']['fits']:
        if payload['model_name'] not in evidence['snapshot']['admissible_candidate_models']:
            continue
        candidate = fit(payload)
        a = predict(candidate, initial.regime)
        b = predict(candidate, observed.final_regime)
        c = predict(candidate, added.regime)
        audit.require(o.operating_margin(b.cold_face, config.band) == intervals[candidate.model_name]['estimate'],
                      'plotted initial forecast disagrees with saved margin estimate')
        initial_curves.append({'model': candidate.model_name, 'acquisition': channel_series(a, initial, True),
                               'added_run_forecast': channel_series(c, added, False),
                               'target_time': list(b.time), 'target_cold_face': list(b.cold_face)})
    intervals = {v['model_name']: v for v in raw_saved['model_intervals']}
    for verification in raw_saved['verifications']:
        payload = verification['fit']
        if payload['model_name'] not in intervals:
            continue
        candidate = fit(payload)
        prediction = predict(candidate, observed.final_regime)
        audit.require(o.operating_margin(prediction.cold_face, config.band) == intervals[candidate.model_name]['estimate'],
                      'plotted verified forecast disagrees with saved margin estimate')
        updated_curves.append({'model': candidate.model_name, 'target_time': list(prediction.time),
                               'target_cold_face': list(prediction.cold_face)})
    revealed = case['fixed_policy_results'][action]['post_reveal_score']['revealed']
    times = initial_curves[0]['target_time']
    audit.require(len(times) == len(revealed['cold_face_temperature']) and times[0] == 0. and times[-1] == o.RUN_DURATION_SECONDS,
                  'saved truth and candidate target grids differ')
    audit.require(o.operating_margin(revealed['cold_face_temperature'], config.band) == row['true_margin'], 'stored target margin differs')
    initial_interval = evidence['snapshot']['provisional_margin_envelope']
    audit.require(row['raw_interval']['upper'] < 0 <= row['final_interval']['upper'],
                  'review the calibration-induced-abstention interpretation for a changed illustrative case')
    band = {'lower': config.band.lower_temperature, 'upper': config.band.upper_temperature}
    return {'block': case['block'], 'family': case['truth_condition'], 'selected_action': action,
            'selection_rule': 'first block then frozen family order: two admissible initial models; additional acquisition selected; calibrated abstention; observed development-padded width narrower than initial width',
            'scientific_use': 'post_hoc_illustration_of_calibration_induced_abstention_not_a_population_estimate',
            'initial_observations': measured(initial), 'added_run_observations': measured(added),
            'added_run_name': added.regime.name, 'added_run_count': len(observed.acquisition_runs) - 1,
            'initial_model_curves': initial_curves, 'verified_model_curves': updated_curves,
            'saved_target_truth': {'time': times, 'cold_face': revealed['cold_face_temperature'], 'margin': row['true_margin']},
            'operating_band_kelvin': band,
            'initial_interval': initial_interval, 'raw_after_interval': row['raw_interval'],
            'development_offset_kelvin': row['development_offset_kelvin'],
            'final_correction_kelvin': row['correction_kelvin'], 'final_interval': row['final_interval'],
            'final_decision': row['decision'], 'action_evaluations': case['frozen_selection_before_reveal']['action_evaluations'],
            'minimum_utility_per_cost': saved['context']['protocol']['complete_calibrated_rule']['design']['selector_rule']['minimum_utility_per_cost'],
            'new_optimizations': 0, 'candidate_forward_integrations': calls,
            'new_observations_or_target_truth_generated': False}


def build(args):
    audit.require(not args.output.exists(), 'refuse to overwrite figure inputs')
    for directory in (args.archive.parent, args.development_maps.parent):
        audit.require(not args.output.resolve().is_relative_to(directory.resolve()), 'figure output must be outside frozen evidence')
    audit.require(audit.file_record(args.archive)['sha256'] == F_SHA, 'not the accepted reserved archive')
    audit.require(audit.file_record(args.development_maps)['sha256'] == D5_SHA, 'not the accepted development maps')
    saved, rows_by_clearance = audit.reconstruct(args.archive, args.repository_root, args.validation_record)
    for entry in saved['context']['source_manifest']['files']:
        audit.require(audit.file_record(args.frozen_source_root / entry['path']) == {'sha256': entry['sha256'], 'bytes': entry['byte_count']}, 'original source clone differs')
    accepted_path = args.repository_root / 'thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_F_RESULT.json'
    accepted = json.loads(accepted_path.read_bytes())
    audit.require(accepted['artifact_digest'] == hashlib.sha256(audit.canonical({k: v for k, v in accepted.items() if k != 'artifact_digest'})).hexdigest(),
                  'accepted compact result digest differs')
    maps = audit.load_sealed(args.development_maps, 'thermotwin.prospective_phase_d.phase_d5_maps_archive_v1')
    audit.require(maps['scientific_result_digest'] == D5_SCIENCE, 'development scientific identity changed')
    cases = []
    for entry in saved['block_file_records']:
        path = args.archive.parent / entry['path']
        audit.require(audit.file_record(path)['sha256'] == entry['sha256'], 'raw block changed before figure extraction')
        cases.extend(json.loads(path.read_bytes())['block_result']['numerical_block']['cases'])
    snapshots = {(c['block'], c['truth_condition']): c['n16_complete_uncertainty_result']['acquisition_evidence']['snapshot'] for c in cases}
    identity = illustrative_identity(rows_by_clearance['0.0'], snapshots)
    case = next(c for c in cases if (c['block'], c['truth_condition']) == identity)
    row = next(r for r in rows_by_clearance['0.0'] if (r['block'], r['truth_family']) == identity and r['procedure'] == audit.SELECTOR)
    from tools.replay_prospective_phase_f_numerics import check_runtime, frozen_imports
    check_runtime(saved, args.frozen_source_root)
    modules = frozen_imports(args.frozen_source_root, saved['context']['source_manifest'])
    story = illustrative_curves(case, row, saved, modules)
    value = {'schema_version': 1, 'version': 'operating_decision_final_figure_inputs_v1',
             'source_revision': audit.SOURCE_REVISION, 'source_manifest_digest': audit.SOURCE_DIGEST,
             'reserved_scientific_result_digest': saved['scientific_result_digest'],
             'development_scientific_result_digest': D5_SCIENCE, 'data_builder_file': audit.file_record(__file__),
             'evidence_files': {'reserved_evaluation': audit.file_record(args.archive), 'development_maps': audit.file_record(args.development_maps),
                                'accepted_compact_result': audit.file_record(accepted_path)},
             'runtime_identity': saved['context']['runtime_identity'],
             'figure_1': story, 'figure_2': comparison_inputs(saved, rows_by_clearance['0.0'], accepted),
             'figure_3': development_inputs(maps),
             'new_cohort_or_fitted_rule': False, 'frozen_scientific_result_changed': False}
    value['artifact_digest'] = audit.digest(DATA_DOMAIN, value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'artifact_digest': value['artifact_digest'],
                      'illustrative_case': identity, 'new_optimizations': 0,
                      'candidate_forward_integrations': story['candidate_forward_integrations'], 'map_cells': 48}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'development-maps', 'repository-root', 'frozen-source-root', 'validation-record', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    build(parser.parse_args(argv))


if __name__ == '__main__':
    main()
