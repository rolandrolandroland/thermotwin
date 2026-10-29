"""Independently audit the closed Phase F archive using only Python's standard library.

No ThermoTwin, NumPy or SciPy imports; no fitting, simulation or cohort generation.
Binomial limits invert Decimal binomial tails, rather than beta quantiles.
The Student-t critical value uses an angular density integral recurrence.
Run with --archive, --source-root, --validation-record and a NEW --output path.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path


SOURCE_REVISION = '7e3b9a3be22d991ec2e588cbd4a53a9d4d2d27d7'
SOURCE_DIGEST = '002d8afd7641dfa2e78269d195a5375c37b3f8aa547eb86d574dc439e7858662'
PROTOCOL_DIGEST = '7f692093108b9e0c0765e5497475a89d1d27eea3d00bf13e4568d6f68fb4301b'
RULE_DIGEST = 'd8ae3d9ea3748d4de776a672cf19798b145b589d681211ba88b8789186dd6341'
SELECTOR = 'prospective_selector'
POLICIES = ('stop_now', 'fixed_thermal', 'fixed_voltage', 'fixed_face_temperature')
PROCEDURES = (SELECTOR, *POLICIES)
FAMILIES = ('matched_four_state', 'extra_interface_mass', 'temperature_dependent_contact')
CLEARANCES = (0., .05, .1, .2)
RESOURCES = ('total_diagnostic_energy', 'diagnostic_run_count', 'extra_sensor_count',
             'energized_schedule_time_seconds')
BLINDED_FIELDS = {'raw_interval', 'development_offset_kelvin', 'verification_succeeded', 'pipeline_failure'}
PERFORMANCE_KEYS = {'decision_computation_seconds', 'performance', 'timing', 'wall_seconds',
                    'cpu_seconds', 'peak_rss_bytes', 'rss_delta_bytes', 'block_wall_seconds',
                    'block_cpu_seconds', 'case_wall_seconds', 'case_cpu_seconds'}
ALPHA = .05 / 73


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode()


def digest(domain, value):
    return hashlib.sha256(domain.encode() + b'\0' + canonical(value)).hexdigest()


def file_record(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return {'sha256': h.hexdigest(), 'bytes': path.stat().st_size}


def load_sealed(path, domain):
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    material = dict(value)
    content = material.pop('archive_content_digest')
    size = material.pop('archive_size_bytes')
    require(size == len(raw) and raw == canonical(value) + b'\n', 'noncanonical archive bytes')
    require(content == digest(domain, material), 'archive content seal differs')
    return value


def without_performance(value):
    if isinstance(value, dict):
        return {k: without_performance(v) for k, v in value.items() if k not in PERFORMANCE_KEYS}
    if isinstance(value, list):
        return [without_performance(v) for v in value]
    return value


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def average(values):
    return float(sum((Fraction(x) for x in values), Fraction()) / len(values))


def outward(value, lower):
    rounded = float(value)
    if lower and Fraction(rounded) > value:
        rounded = math.nextafter(rounded, -math.inf)
    if not lower and Fraction(rounded) < value:
        rounded = math.nextafter(rounded, math.inf)
    return rounded


def final_decision(inp, correction, clearance):
    require(set(inp) == BLINDED_FIELDS, 'unexpected blinded decision input')
    require(type(inp['verification_succeeded']) is bool and type(inp['pipeline_failure']) is bool,
            'nonboolean verification/failure')
    offset = inp['development_offset_kelvin']
    require(finite(offset) and offset >= 0 and finite(correction) and correction >= 0,
            'invalid frozen correction')
    raw, interval, decision = inp['raw_interval'], None, 'insufficient_evidence'
    valid_raw = (isinstance(raw, dict) and finite(raw.get('lower')) and finite(raw.get('upper'))
                 and raw['lower'] <= raw['upper'])
    if valid_raw and inp['verification_succeeded'] and not inp['pipeline_failure']:
        interval = {'lower': outward(Fraction(raw['lower']) - Fraction(offset) - Fraction(correction), True),
                    'upper': outward(Fraction(raw['upper']) + Fraction(offset) + Fraction(correction), False)}
        if interval['lower'] >= clearance:
            decision = 'approve'
        elif interval['upper'] < -clearance:
            decision = 'reject'
    return {'correction_kelvin': correction, 'decision_clearance_kelvin': clearance,
            'final_interval': interval, 'decision': decision}


@lru_cache(maxsize=None)
def binomial_bounds(x, n):
    """Invert direct finite binomial sums at 60 decimal digits."""
    require(type(x) is int and type(n) is int and 0 <= x <= n, 'invalid binomial counts')
    if n == 0:
        return {'estimate': None, 'lower': None, 'upper': None, 'numerator': x, 'denominator': n}
    with localcontext() as ctx:
        ctx.prec = 60
        target = Decimal.from_float(ALPHA)

        def tail(p, lower_bound):
            q = 1 - p
            term, total = q**n, Decimal(0)
            for k in range(n + 1):
                if (k >= x if lower_bound else k <= x):
                    total += term
                if k < n:
                    term *= Decimal(n - k) * p / (Decimal(k + 1) * q)
            return total

        def solve(lower_bound):
            lo, hi = Decimal(0), Decimal(1)
            for _ in range(165):
                mid = (lo + hi) / 2
                value = tail(mid, lower_bound)
                if (value < target if lower_bound else value > target):
                    lo = mid
                else:
                    hi = mid
            return float((lo + hi) / 2)

        lower = 0. if x == 0 else solve(True)
        upper = 1. if x == n else solve(False)
    return {'estimate': x / n, 'lower': lower, 'upper': upper, 'numerator': x, 'denominator': n}


def student_t_tail(t, df):
    """Positive-t survival integral; odd df covers the frozen df=99."""
    require(t >= 0 and type(df) is int and df > 0 and df % 2 == 1, 'positive t and odd df required')
    angle = math.atan(t / math.sqrt(df))
    sine, cosine = math.sin(angle), math.cos(angle)
    integral = math.pi / 2 - angle
    for power in range(2, df, 2):
        integral = ((power - 1) * integral - sine * cosine**(power - 1)) / power
    normalization = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(math.pi)
    return normalization * integral


def student_t_critical(df):
    lo, hi = 0., 8.
    while student_t_tail(hi, df) > ALPHA:
        hi *= 2
    for _ in range(90):
        mid = (lo + hi) / 2
        if student_t_tail(mid, df) > ALPHA:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def rate(x, n):
    return {'numerator': x, 'denominator': n, 'estimate': x / n if n else None}


def metrics(rows):
    n = len(rows)
    decisions = Counter(r['decision'] for r in rows)
    approvals, rejections = decisions['approve'], decisions['reject']
    false_approvals = sum(r['false_approval'] for r in rows)
    false_rejections = sum(r['false_rejection'] for r in rows)
    passes = sum(r['true_pass'] for r in rows)
    resources = {}
    for name in RESOURCES:
        values = [r['realized_resources'].get(name) for r in rows]
        recorded = sum(finite(v) and v >= 0 for v in values)
        resources[name] = {'mean': average(values) if values and recorded == n else None,
                           'recorded_count': recorded, 'required_count': n}
    return {
        'case_count': n, 'approvals': approvals, 'rejections': rejections,
        'definitive_decisions': approvals + rejections, 'abstentions': n - approvals - rejections,
        'true_passes': passes, 'true_violations': n - passes,
        'false_approvals': false_approvals, 'false_rejections': false_rejections,
        'false_approval_risk': rate(false_approvals, approvals),
        'false_rejection_rate_given_pass': rate(false_rejections, passes),
        'missed_violation_rate': rate(sum(r['missed_violation'] for r in rows), n - passes),
        'decision_error_rate': rate(false_approvals + false_rejections, approvals + rejections),
        'decision_coverage': rate(approvals + rejections, n),
        'case_interval_coverage': rate(sum(r['interval_covered'] for r in rows), n),
        'pipeline_failures': sum(r['pipeline_failure'] for r in rows),
        'verification_failures': sum(not r['verification_succeeded'] for r in rows),
        'missing_or_invalid_final_intervals': sum(r['final_interval'] is None for r in rows),
        'realized_resources': resources,
    }


def matrix(rows):
    expected = {(b, f, p) for b in range(100) for f in FAMILIES for p in PROCEDURES}
    result = {}
    for row in rows:
        key = row['block'], row['truth_family'], row['procedure']
        require(key in expected and key not in result, 'duplicate or undeclared analysis cell')
        result[key] = row
    require(set(result) == expected, 'incomplete 1500-cell matrix')
    return result


def analyze(rows):
    keyed = matrix(rows)
    summaries = {}
    for procedure in PROCEDURES:
        selected = [r for r in rows if r['procedure'] == procedure]
        groups = {}
        for family in FAMILIES:
            group = metrics([r for r in selected if r['truth_family'] == family])
            coverage = binomial_bounds(group['definitive_decisions'], 100).copy()
            coverage.pop('upper')
            groups[family] = {**group, 'false_approval_risk_bounds': binomial_bounds(
                group['false_approvals'], group['approvals']), 'decision_coverage_bounds': coverage}
        risks = [g['false_approval_risk_bounds'] for g in groups.values()]
        known = all(r['estimate'] is not None for r in risks)
        risk_lower = min(r['lower'] for r in risks) if known else None
        risk_upper = max(r['upper'] for r in risks) if known else None
        coverage_lower = average([g['decision_coverage_bounds']['lower'] for g in groups.values()])
        summaries[procedure] = {
            'overall': metrics(selected), 'by_truth_family': groups,
            'simultaneous_block_interval_coverage': rate(sum(
                all(keyed[b, f, procedure]['interval_covered'] for f in FAMILIES) for b in range(100)), 100),
            'population_mixture_risk_bounds': {'lower': risk_lower, 'upper': risk_upper,
                'method': 'family extrema; unknown population approval weights'},
            'population_equal_family_decision_coverage_lower': coverage_lower,
            'qualifies_for_primary_comparison': bool(risk_upper is not None and risk_upper <= .10 and coverage_lower >= .70),
        }
    comparisons = {}
    for policy in POLICIES:
        pairs, differences = {}, []
        for family in FAMILIES:
            flags = [(keyed[b, family, SELECTOR]['decision'] != 'insufficient_evidence',
                      keyed[b, family, policy]['decision'] != 'insufficient_evidence') for b in range(100)]
            losses = sum(f and not s for s, f in flags)
            gains = sum(s and not f for s, f in flags)
            pairs[family] = {'fixed_only_decisions': losses, 'selector_only_decisions': gains,
                             'coverage_loss_estimate': (losses - gains) / 100,
                             'coverage_loss_upper': binomial_bounds(losses, 100)['upper'] - binomial_bounds(gains, 100)['lower']}
        for block in range(100):
            values = []
            for family in FAMILIES:
                a = keyed[block, family, policy]['realized_resources']['total_diagnostic_energy']
                b = keyed[block, family, SELECTOR]['realized_resources']['total_diagnostic_energy']
                values.append(a - b if finite(a) and finite(b) else None)
            differences.append(average(values) if all(v is not None for v in values) else None)
        if all(finite(v) for v in differences):
            exact_mean = sum(map(Fraction, differences), Fraction()) / 100
            variance = sum(((Fraction(v) - exact_mean)**2 for v in differences), Fraction()) / 99
            se = math.sqrt(float(variance)) / 10
            energy = {'paired_blocks': 100, 'complete': True, 'mean_saving_joules': float(exact_mean),
                      'lower_saving_joules': float(exact_mean) - student_t_critical(99) * se,
                      'standard_error_joules': se, 'method': 'one-sided paired-block Student t; approximate'}
        else:
            energy = {'paired_blocks': 100, 'complete': False, 'mean_saving_joules': None, 'lower_saving_joules': None}
        coverage_loss = average([r['coverage_loss_upper'] for r in pairs.values()])
        su, fl = summaries[SELECTOR]['population_mixture_risk_bounds']['upper'], summaries[policy]['population_mixture_risk_bounds']['lower']
        risk_increase = su - fl if su is not None and fl is not None else None
        checks = {'energy_saving': energy['lower_saving_joules'] is not None and energy['lower_saving_joules'] > 0,
                  'coverage_noninferiority': coverage_loss <= .05,
                  'risk_noninferiority': risk_increase is not None and risk_increase <= .02}
        comparisons[policy] = {'energy': energy, 'coverage_pairs_by_family': pairs,
                               'coverage_loss_upper': coverage_loss, 'risk_increase_upper': risk_increase,
                               'checks': checks, 'all_comparison_checks_pass': all(checks.values())}
    qualifying = [p for p in POLICIES if summaries[p]['qualifies_for_primary_comparison']]
    # No policy qualifies in this closed archive; do not silently skip frontier
    # verification if this audit is pointed at a materially different result.
    require(not qualifying, 'this closed-result audit requires reviewing a changed qualifying set')
    return {
        'protocol_version': 'prospective_phase_e_paired_analysis_v1', 'independent_block_count': 100,
        'cases_are_not_independent_replicates': True, 'familywise_alpha': .05,
        'primary_bound_count': 73, 'one_sided_alpha_per_bound': ALPHA,
        'energy_inference_is_approximate': True, 'procedures': summaries, 'comparisons': comparisons,
        'qualifying_fixed_policies': qualifying, 'qualifying_fixed_frontier': [],
        'primary_success': bool(summaries[SELECTOR]['qualifies_for_primary_comparison'] and qualifying
                                and all(comparisons[p]['all_comparison_checks_pass'] for p in qualifying)),
        'named_comparisons_passing_all_checks': [p for p in POLICIES if comparisons[p]['all_comparison_checks_pass']],
        'conclusion_rule': 'success requires selector qualification, at least one qualifying fixed policy, and all comparisons against every qualifying fixed policy',
    }


def compare(expected, stored, path='', differences=None):
    """Tolerance applies to independent floating calculations, never pass/fail rules."""
    if differences is None:
        differences = []
    if isinstance(expected, dict):
        require(isinstance(stored, dict) and set(expected) == set(stored), 'schema differs: ' + path)
        for key in expected:
            compare(expected[key], stored[key], path + '/' + key, differences)
    elif isinstance(expected, list):
        require(isinstance(stored, list) and len(expected) == len(stored), 'length differs: ' + path)
        for i, (a, b) in enumerate(zip(expected, stored)):
            compare(a, b, path + '/' + str(i), differences)
    elif type(expected) is float:
        require(finite(stored) and math.isclose(expected, stored, rel_tol=2e-12, abs_tol=2e-10),
                'independent numerical calculation differs: ' + path)
        differences.append(abs(expected - stored))
    else:
        require(type(expected) is type(stored) and expected == stored, 'value differs: ' + path)
    return differences


def reconstruct(archive, source_root, validation_record):
    validation = json.loads(Path(validation_record).read_bytes())
    require(validation['independent_full_archive_validation_passed'] is True
            and validation['complete_block_count'] == 100, 'independent archive validation did not pass')
    require(file_record(archive)['sha256'] == validation['evaluation_json_sha256'], 'accepted summary hash differs')
    saved = load_sealed(archive, 'thermotwin.prospective_phase_f.evaluation_v1')
    root, context = Path(archive).parent, saved['context']
    require(context['source_revision'] == SOURCE_REVISION and context['source_manifest_digest'] == SOURCE_DIGEST
            and context['protocol_digest'] == PROTOCOL_DIGEST, 'source/protocol differs from accepted execution')
    require(saved['partition'] == 'p1_reserved_evaluation' and saved['reserved_sample_size_extended_after_outcomes'] is False,
            'reserved identity or sample-size rule differs')
    for row in context['source_manifest']['files']:
        actual = file_record(Path(source_root) / row['path'])
        require(actual == {'sha256': row['sha256'], 'bytes': row['byte_count']}, 'frozen source changed: ' + row['path'])
    rule = json.loads((Path(source_root) / 'thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_F_RULE.json').read_bytes())
    require(rule['artifact_digest'] == RULE_DIGEST, 'complete calibrated rule differs')
    q, offsets = rule['final_corrections_kelvin'], rule['design']['selector_rule']['development_offsets']['values']
    records = {str(c): [] for c in CLEARANCES}
    inventory = saved['block_file_records']
    require([r['block'] for r in inventory] == list(range(100)), 'incomplete ordered checkpoint inventory')
    require({p.name for p in (root / 'blocks').iterdir()} == {f'block-{b:03d}.json' for b in range(100)},
            'unexpected checkpoint files')
    expected_before, scientific_digests, block_cpu = set(), [], []
    case_ids = set()
    for entry in inventory:
        block = entry['block']
        relative = f'blocks/block-{block:03d}.json'
        require(entry['path'] == relative, 'invalid checkpoint path')
        p = root / relative
        require(file_record(p)['sha256'] == entry['sha256'], 'checkpoint hash differs')
        checkpoint = load_sealed(p, 'thermotwin.prospective_phase_f.block_v1')
        result, envelope = checkpoint['block_result'], checkpoint['context']
        require(checkpoint['block'] == block and envelope['source_revision'] == SOURCE_REVISION
                and envelope['partition'] == saved['partition'] and envelope['complete_rule_digest'] == RULE_DIGEST,
                'checkpoint source/case identity differs')
        scientific = digest('thermotwin.prospective_phase_f.scientific_block_v1', without_performance(result))
        require(scientific == checkpoint['scientific_block_digest'] == entry['scientific_block_digest'],
                'scientific block digest differs')
        scientific_digests.append(scientific)
        block_cpu.append(result['numerical_block']['timing']['block_cpu_seconds'])
        cases = result['numerical_block']['cases']
        require([c['truth_condition'] for c in cases] == list(FAMILIES), 'incomplete family block')
        for case in cases:
            family = case['truth_condition']
            require(case['block'] == block and (block, family) not in case_ids, 'duplicate or changed case identity')
            case_ids.add((block, family))
            p = root / 'before-reveal' / f'block-{block:03d}' / f'{family}.json'
            expected_before.add(p.relative_to(root).as_posix())
            before = load_sealed(p, 'thermotwin.prospective_phase_f.before_reveal_v1')
            require(before == result['before_reveal_by_family'][family] and before['context'] == envelope
                    and before['block'] == block and before['truth_condition'] == family
                    and before['phase'] == 'before_target_reveal', 'before-reveal identity differs')
            require(set(before['saves']['procedures']) == set(PROCEDURES)
                    and before['saves']['clearances_kelvin'] == list(CLEARANCES), 'incomplete saved decision set')
            fixed = case['fixed_policy_results']
            require(set(fixed) == set(POLICIES), 'fixed comparator missing')
            truths = [v['post_reveal_score']['revealed']['true_margin'] for v in fixed.values()
                      if isinstance(v.get('post_reveal_score'), dict)]
            require(truths and all(finite(t) and t == truths[0] for t in truths), 'missing or inconsistent target truth')
            for procedure in PROCEDURES:
                action = case['selected_policy_at_n16_primary'] if procedure == SELECTOR else procedure
                fixed_record = fixed.get(action)
                raw = fixed_record.get('saved_before_reveal') if fixed_record else None
                post = fixed_record.get('post_reveal_score') if fixed_record else None
                verified = bool(isinstance(raw, dict) and any(v.get('passed') is True for v in raw.get('verifications', [])))
                inp = {'raw_interval': raw.get('margin_envelope') if isinstance(raw, dict) else None,
                       'development_offset_kelvin': offsets[action] if action is not None else 0.,
                       'verification_succeeded': verified,
                       'pipeline_failure': action is None or not isinstance(raw, dict) or 'failure' in fixed_record}
                saved_procedure = before['saves']['procedures'][procedure]
                require(saved_procedure['input'] == inp and saved_procedure['selected_action'] == action,
                        'raw blinded inputs differ from saved calibrated decision')
                require(len(saved_procedure['decisions']) == 4, 'missing clearance decision')
                scored = post.get('scored') if isinstance(post, dict) else None
                resources = {name: scored.get(name) if isinstance(scored, dict) else None for name in RESOURCES}
                for i, clearance in enumerate(CLEARANCES):
                    decision = final_decision(inp, q[procedure], clearance)
                    require(decision == saved_procedure['decisions'][i], 'exact calibrated decision differs')
                    interval, label, margin = decision['final_interval'], decision['decision'], truths[0]
                    row = {'block': block, 'truth_family': family, 'device_token': case['device_token'],
                           'procedure': procedure, 'selected_action': action, **inp, 'true_margin': margin,
                           'realized_resources': resources, **decision,
                           'interval_covered': bool(interval is not None and interval['lower'] <= margin <= interval['upper']),
                           'true_pass': margin >= 0, 'false_approval': label == 'approve' and margin < 0,
                           'false_rejection': label == 'reject' and margin >= 0,
                           'missed_violation': margin < 0 and label != 'reject'}
                    records[str(clearance)].append(row)
        if (block + 1) % 20 == 0:
            print(f'Independent audit: {block+1}/100 raw blocks checked', flush=True)
    require(expected_before == {p.relative_to(root).as_posix() for p in (root / 'before-reveal').rglob('*') if p.is_file()},
            'missing or unexpected before-reveal files')
    require(records == saved['calibrated_case_records_by_clearance'], '6000 saved outcome records differ from raw reconstruction')
    require(average(block_cpu) * 100 == saved['performance']['aggregate_completed_block_cpu_seconds']
            or math.isclose(sum(block_cpu), saved['performance']['aggregate_completed_block_cpu_seconds'], abs_tol=1e-7),
            'completed-block compute total differs')
    scientific = digest('thermotwin.prospective_phase_f.evaluation_scientific_v1', {
        'context': context, 'partition': saved['partition'], 'scientific_block_digests': scientific_digests,
        'calibrated_case_records_by_clearance': records, 'primary_analysis': saved['primary_analysis'],
        'secondary_analysis_by_clearance': saved['secondary_analysis_by_clearance']})
    require(scientific == saved['scientific_result_digest'] == validation['scientific_result_digest'], 'final scientific digest differs')
    return saved, records


def audit(*, archive, source_root, validation_record):
    saved, rows = reconstruct(archive, source_root, validation_record)
    differences = []
    analyses = {}
    for clearance in CLEARANCES:
        result = analyze(rows[str(clearance)])
        analyses[str(clearance)] = result
        stored = saved['primary_analysis'] if clearance == 0 else saved['secondary_analysis_by_clearance'][str(clearance)]['analysis']
        compare(result, stored, str(clearance), differences)
        if clearance != 0:
            require(saved['secondary_analysis_by_clearance'][str(clearance)]['scientific_use']
                    == 'descriptive_only_no_confirmatory_inference', 'secondary inference label differs')
    return {
        'schema_version': 1, 'protocol_version': 'prospective_phase_f_independent_result_audit_v1',
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'audit_passed': True,
        'source_revision': SOURCE_REVISION, 'source_manifest_digest': SOURCE_DIGEST, 'protocol_digest': PROTOCOL_DIGEST,
        'scientific_result_digest': saved['scientific_result_digest'],
        'evaluation_file': file_record(archive), 'original_validation_file': file_record(validation_record),
        'audit_script_file': file_record(__file__), 'standard_library_only_no_thermotwin_numpy_or_scipy_imports': True,
        'raw_blocks_checked': 100, 'distinct_cases_checked': 300, 'before_reveal_files_checked': 300,
        'exact_saved_decisions_and_outcome_rows_checked': 6000,
        'analysis_clearances_checked': list(CLEARANCES), 'primary_simultaneous_bound_count': 73,
        'secondary_analyses_are_descriptive': True, 'primary_success': analyses['0.0']['primary_success'],
        'binomial_method': 'direct Decimal binomial-tail inversion at 60 decimal digits',
        'energy_critical_value_method': 'Student-t angular density integral recurrence and bisection, df=99',
        'student_t_critical_value': student_t_critical(99),
        'independently_compared_numeric_fields': len(differences),
        'maximum_absolute_numeric_difference': max(differences),
        'audit_comparison_absolute_tolerance': 2e-10, 'audit_comparison_relative_tolerance': 2e-12,
        'tolerances_change_scientific_decisions_or_pass_fail_thresholds': False,
        'new_observations_generated': False, 'numerical_fits_or_physical_simulations_replayed': False,
        'frozen_numerical_source_changed': False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--validation-record', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), 'refuse to overwrite an existing audit record')
    result = audit(archive=args.archive, source_root=args.source_root, validation_record=args.validation_record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, sort_keys=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
