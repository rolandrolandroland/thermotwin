"""Supporting post hoc diagnostics from accepted, saved reserved evidence.

No fitting, ODE integration, observation generation or new partition. Scalar
parameter truths are replayed from their authenticated original RNG seeds.
Parameter errors are restricted to passed four-state fits in the matched
family; exclusions remain explicit and are not silently treated as successes.
"""
import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

from tools import verify_prospective_phase_f_result as audit

DOMAIN = 'thermotwin.operating_decision.report_diagnostics_v1'


def original_seed(entries, purpose, block):
    rows = [e for e in entries if e['key']['stream_kind'] == 'device_truth'
            and e['key']['purpose'] == purpose and e['key']['block'] == block]
    audit.require(len(rows) == 3 and {r['pairing_member'] for r in rows} == set(audit.FAMILIES), 'incomplete shared parameter stream')
    audit.require(len({e['seed'] for e in rows}) == 1, 'shared parameter seeds differ')
    for row in rows:
        material = {**row['key'], 'domain': 'thermotwin.operating_decision.random_stream'}
        audit.require(row['seed'] == int.from_bytes(hashlib.sha256(audit.canonical(material)).digest(), 'big'), 'scalar seed does not authenticate')
    return rows[0]['seed']


def summary(values):
    return {'count': len(values), 'median': statistics.median(values) if values else None,
            'mean': statistics.mean(values) if values else None, 'maximum': max(values) if values else None}


def build(args):
    from tools.replay_prospective_phase_f_numerics import check_runtime, frozen_imports
    saved = json.loads(args.archive.read_bytes())
    audit.require(audit.file_record(args.archive)['sha256'] == '299f75ad492e5bde0acc724ed4e72ae029c86f2c05538677c2a856c23734c564', 'wrong reserved archive')
    check_runtime(saved, args.frozen_source_root)
    modules = frozen_imports(args.frozen_source_root, saved['context']['source_manifest'])
    f = modules['studies.operating_decision_prospective_phase_f_execution']
    config = f.e._physical_config(f.phase_f_partition(saved['partition']))
    names = ('cold_contact_resistance', 'cold_face_capacitance', 'shared_sensor_lag', 'series_resistance')
    errors = {p: {k: [] for k in names} for p in audit.PROCEDURES}
    statuses = {p: {'passed_four_state_fits': 0, 'excluded_or_unverified': 0, 'total_cases': 100} for p in audit.PROCEDURES}
    details = []
    for entry in saved['block_file_records']:
        path = args.archive.parent / entry['path']
        audit.require(audit.file_record(path)['sha256'] == entry['sha256'], 'raw block changed')
        block = json.loads(path.read_bytes())['block_result']['numerical_block']
        index = block['block']
        manifest = block['corrected_random_stream_manifest']
        truth = [n * math.exp(random.Random(original_seed(manifest, f'physical_parameter_{i}', index)).gauss(0., sd))
                 for i, (n, sd) in enumerate(zip(config.sensor.fit.nominal_values, config.sensor.truth_log_standard_deviations))]
        source = random.Random(original_seed(manifest, 'series_resistance', index))
        for _ in range(10000):
            resistance = config.series_resistance_nominal * math.exp(source.gauss(0., config.series_resistance_truth_log_standard_deviation))
            if config.series_resistance_bounds[0] <= resistance <= config.series_resistance_bounds[1]:
                break
        else:
            raise ValueError('original scalar rejection sampler failed')
        truth.append(resistance)
        case = next(c for c in block['cases'] if c['truth_condition'] == audit.FAMILIES[0])
        for procedure in audit.PROCEDURES:
            policy = case['selected_policy_at_n16_primary'] if procedure == audit.SELECTOR else procedure
            result = case['fixed_policy_results'][policy]['saved_before_reveal']
            fit = next(v for v in result['verifications'] if v['fit']['model_name'] == 'four_state')
            row = {'block': index, 'procedure': procedure, 'selected_policy': policy, 'passed': fit['passed']}
            if fit['passed']:
                estimates = [*fit['fit']['physical_values'], fit['fit']['series_resistance']]
                row['absolute_relative_errors'] = {name: abs(estimate / actual - 1) for name, estimate, actual in zip(names, estimates, truth)}
                for name, value in row['absolute_relative_errors'].items():
                    errors[procedure][name].append(value)
                statuses[procedure]['passed_four_state_fits'] += 1
            else:
                statuses[procedure]['excluded_or_unverified'] += 1
            details.append(row)
    inputs = json.loads(args.figure_inputs.read_bytes())
    story = inputs['figure_1']
    actual = story['saved_target_truth']['cold_face']
    trajectory = []
    for phase in ('initial', 'verified'):
        for curve in story[phase + '_model_curves']:
            delta = [a - b for a, b in zip(curve['target_cold_face'], actual)]
            audit.require(len(delta) == len(actual) == 321, 'trajectory grids differ')
            trajectory.append({'phase': phase, 'candidate': curve['model'], 'time_points': 321,
                               'rmse_kelvin': math.sqrt(statistics.mean(v*v for v in delta)),
                               'maximum_absolute_error_kelvin': max(abs(v) for v in delta)})
    value = {'schema_version': 1, 'scientific_use': 'post_hoc_supporting_diagnostics_not_primary_or_model_identification',
             'scientific_result_digest': saved['scientific_result_digest'], 'archive_file': audit.file_record(args.archive),
             'source_revision': audit.SOURCE_REVISION, 'analysis_script_file': audit.file_record(__file__),
             'matched_family_parameter_errors': {p: {'status': statuses[p], 'absolute_relative_error': {k: summary(v) for k,v in errors[p].items()}} for p in audit.PROCEDURES},
             'parameter_case_records_digest': audit.digest(DOMAIN + '.cases', details),
             'scalar_parameter_replayed_blocks': 100, 'new_parameter_draws': False,
             'parameter_scope': 'passed four-state verifications in matched_four_state only; excluded fits counted separately; conditional descriptive errors',
             'illustrative_trajectory': {'block': story['block'], 'family': story['family'], 'records': trajectory},
             'new_fits_ode_integrations_observations_or_partitions': False}
    value['artifact_digest'] = audit.digest(DOMAIN, value)
    with args.output.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'artifact_digest':value['artifact_digest'],'parameter_cases':len(details),'trajectory_records':len(trajectory)},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','frozen-source-root','figure-inputs','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    build(parser.parse_args())
