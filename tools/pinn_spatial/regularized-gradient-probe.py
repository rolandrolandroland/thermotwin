"""One frozen, training-only verification of the stationary ridge method.

Two fresh-field calibration SVDs and fourteen head solves; no optimizer,
checkpoint selection, independent physics audit, or automatic method search.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from thermotwin.studies import pinn_spatial_cutoff_audit as prior
from thermotwin.pinn import planar_contact_regularized as api
from thermotwin.pinn.planar_contact_graded import graded_training_points
from thermotwin.pinn.planar_contact_inverse import PlanarObservations

common = prior.common
BASE = ROOT/'reports/pinn-spatial-validation'
PREVIOUS = BASE/'cutoff-directional-audit'
OUT = BASE/'regularized-gradient-probe'
TRACKED_PLAN = ROOT/'thermotwin/PINN_SPATIAL_REGULARIZED_METHOD_PLAN.json'
TRACKED_RESULT = ROOT/'thermotwin/PINN_SPATIAL_REGULARIZED_METHOD_RESULT.json'
CASES, RADII = ('t02', 't04'), (1.e-4, 1.e-5, 1.e-6)
GATES = {'stationarity_relative': 1.e-8, 'objective_atol': 1.e-8,
    'objective_rtol': 1.e-8, 'finest_fd_atol': 1.e-4, 'finest_fd_rtol': .005,
    'two_finest_fd_atol': 2.e-4, 'two_finest_fd_rtol': .01}
FOLLOWUP = {'cases': list(common.CASES), 'maximum_attempts_per_case': 320,
    'maximum_trial_solves_per_case': 480, 'maximum_backtracks': 4,
    'acceptance_thresholds': common.GATES, 'parallel_workers': 2, 'worker_threads': 1,
    'case_wall_seconds': 3000., 'case_rss_bytes': 2*1024**3,
    'supervisor_wall_seconds': 7200., 'automatic_retries': 0}


def read(path):
    return json.loads(Path(path).read_text())


def record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': common.digest_file(path), 'bytes': path.stat().st_size}


def protocol():
    return [{'case': case, 'kind': kind, 'radius': radius, 'sign': sign}
        for case in CASES for kind, radius, sign in
        [('baseline', 0., 0)]+[('perturbation', radius, sign) for radius in RADII for sign in (-1, 1)]]


def close(left, right, atol, rtol):
    return (math.isfinite(left) and math.isfinite(right)
            and abs(left-right) <= atol+rtol*max(abs(left), abs(right)))


def case_checks(row):
    samples, direction = row['samples'], row.get('direction')
    point_checks = []
    for sample in samples:
        p = sample.get('projection', {})
        values = [sample.get(name) for name in ('total_objective', 'physical_objective', 'penalty')]
        finite = sample.get('status') == 'solved' and all(
            type(value) in (int, float) and math.isfinite(value) for value in values)
        point_checks.append(finite and type(p.get('columns')) is int and p['columns'] > 0
            and p.get('rank') == p['columns']
            and p.get('hard_singular_value_truncation') is False
            and type(p.get('head_solve_calls')) is int and p['head_solve_calls'] == 1
            and 0 <= p.get('stationarity_relative', math.inf) <= GATES['stationarity_relative']
            and close(sample['total_objective'], p.get('algebraic_objective', math.inf),
                      GATES['objective_atol'], GATES['objective_rtol'])
            and close(sample['total_objective'], sample['physical_objective']+sample['penalty'],
                      GATES['objective_atol'], GATES['objective_rtol']))
    differences = []
    for radius in RADII:
        pair = [sample for sample in samples if sample['radius'] == radius]
        fd = None
        if len(pair) == 2 and all(sample['status'] == 'solved' for sample in pair):
            minus = next(sample for sample in pair if sample['sign'] == -1)
            plus = next(sample for sample in pair if sample['sign'] == 1)
            fd = (plus['total_objective']-minus['total_objective'])/(2*radius)
        differences.append({'radius': radius, 'centered_fd': fd})
    ad = None if direction is None else direction['ad_directional_derivative']
    finest, middle = differences[-1]['centered_fd'], differences[-2]['centered_fd']
    derivative = ad is not None and finest is not None and close(
        finest, ad, GATES['finest_fd_atol'], GATES['finest_fd_rtol'])
    consistency = finest is not None and middle is not None and close(
        finest, middle, GATES['two_finest_fd_atol'], GATES['two_finest_fd_rtol'])
    passed = (len(point_checks) == 7 and all(point_checks) and derivative and consistency
              and direction is not None and direction.get('gradient_clipped') is False
              and row.get('original_state_restored_exactly') is True)
    return {'point_checks': point_checks, 'directional_differences': differences,
        'unclipped_ad_directional_derivative': ad,
        'finest_fd_agrees_with_ad': bool(derivative),
        'two_finest_fd_agree': bool(consistency), 'passed': bool(passed)}


def configuration():
    result = asdict(api.RegularizedConfig())
    original = read(BASE/'step-control-development/plan.json')['model_config']
    result.update(original)
    result = asdict(api.RegularizedConfig(**result))
    if any(result[name] != value for name, value in original.items()):
        raise RuntimeError('The regularized method changed an undeclared field/controller value')
    return result


def prepare():
    prior._one_thread()
    previous = read(PREVIOUS/'plan.json')
    prior.verify_plan(previous, PREVIOUS)
    if read(PREVIOUS/'state.json')['status'] != 'completed':
        raise RuntimeError('The completed cutoff audit must remain sealed')
    sources = dict(previous['source_sha256'])
    sources.update(common._source_records([
        'thermotwin/pinn/planar_contact_regularized.py',
        'tests/test_planar_contact_regularized.py',
        'tools/pinn_spatial/regularized-gradient-probe.py',
        'tests/test_pinn_spatial_regularized_probe.py',
        'tests/test_regularized_gradient_lifecycle.py']))
    trees = dict(previous['protected_trees'])
    trees[str(PREVIOUS.resolve())] = common._tree_records(PREVIOUS)
    protected = dict(previous['protected_files'])
    protected.update(common._source_records([
        'thermotwin/PINN_SPATIAL_CUTOFF_AUDIT_PLAN.json',
        'thermotwin/PINN_SPATIAL_CUTOFF_AUDIT_RESULT.json']))
    validation = BASE/'regularized-method-validation'
    for name in ('focused-tests.log', 'focused-tests-state.json'):
        if (validation/name).is_file():
            protected.update(common._source_records([str((validation/name).relative_to(ROOT))]))
    if TRACKED_PLAN.exists() or TRACKED_RESULT.exists():
        raise RuntimeError('The one-candidate regularized method declaration already exists')
    plan = {'schema': 'fixed_ridge_training_gradient_verification_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'parent_commit': subprocess.run(['git','rev-parse','HEAD'], cwd=ROOT,
            check=True, capture_output=True, text=True).stdout.strip(),
        'runtime': prior.runtime_context(), 'source_sha256': sources,
        'protected_trees': trees, 'protected_files': protected,
        'previous_cutoff_plan': record(PREVIOUS/'plan.json'),
        'previous_cutoff_result': record(PREVIOUS/'result.json'),
        'inputs': previous['inputs'], 'model_config': configuration(),
        'protocol': protocol(), 'gates': GATES, 'followup': FOLLOWUP,
        'maximum_calibration_spectral_calculations': 2, 'maximum_head_solve_attempts': 14,
        'maximum_wall_seconds': 900., 'maximum_rss_bytes': 2*1024**3,
        'neural_training_updates': 0, 'automatic_retries': 0, 'new_checkpoints': 0,
        'training_only': True, 'reference_fields_used': False,
        'independent_diagnostics_used_for_selection': False,
        'calibration_policy': 'fixed initial D and lambda; fresh matched common field before terminal-field transfer',
        'direction_policy': 'negative unclipped gradient of regularized total at the baseline stationary ridge head, normalized Linf to one',
        'reset_policy': 'same baseline hidden weights and solved ridge head before each independent perturbation; calibration stays fixed',
        'derivative_policy': 'predeclared finest AD/FD and two-finest FD checks only; no best-radius selection',
        'failure_policy': 'any solve or method check failure blocks the four-case campaign; no candidate or budget search'}
    OUT.mkdir(exist_ok=False)
    common._write(OUT/'plan.json', plan)
    for name in sources:
        target = OUT/'source-snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    # Exclusive creation; the prospective record exists before the first solve.
    with TRACKED_PLAN.open('xb') as handle:
        handle.write((OUT/'plan.json').read_bytes())
    common._write(OUT/'state.json', {'status': 'prepared', 'plan_sha256': common.digest_file(OUT/'plan.json')})
    verify_plan(plan)
    return plan


def verify_plan(plan):
    prior._one_thread()
    expected = {'schema': 'fixed_ridge_training_gradient_verification_v1',
        'protocol': protocol(), 'gates': GATES, 'followup': FOLLOWUP,
        'maximum_calibration_spectral_calculations': 2, 'maximum_head_solve_attempts': 14,
        'maximum_wall_seconds': 900., 'maximum_rss_bytes': 2*1024**3,
        'neural_training_updates': 0, 'automatic_retries': 0, 'new_checkpoints': 0,
        'training_only': True, 'reference_fields_used': False,
        'independent_diagnostics_used_for_selection': False}
    if any(plan.get(key) != value for key, value in expected.items()):
        raise RuntimeError('The single-candidate verification protocol changed')
    if plan['model_config'] != configuration() or plan['runtime'] != prior.runtime_context():
        raise RuntimeError('The method configuration or one-thread runtime changed')
    if read(OUT/'plan.json') != plan or (OUT/'plan.json').read_bytes() != TRACKED_PLAN.read_bytes():
        raise RuntimeError('The prospective declaration changed')
    if common._source_records(plan['source_sha256']) != plan['source_sha256']:
        raise RuntimeError('A frozen verification source changed')
    if common._source_records(plan['protected_files']) != plan['protected_files']:
        raise RuntimeError('A protected scientific record changed')
    for name, records in plan['protected_trees'].items():
        if common._tree_records(name) != records:
            raise RuntimeError('A protected scientific evidence tree changed')
    for name, expected_hash in plan['source_sha256'].items():
        if common.digest_file(OUT/'source-snapshot'/name) != expected_hash:
            raise RuntimeError('A source snapshot changed')
    for records in plan['inputs'].values():
        for saved in records.values():
            if record(saved['path']) != saved:
                raise RuntimeError('A terminal training input changed')
    if record(plan['previous_cutoff_plan']['path']) != plan['previous_cutoff_plan'] or record(
            plan['previous_cutoff_result']['path']) != plan['previous_cutoff_result']:
        raise RuntimeError('The preceding cutoff audit changed')
    return True


def point_result(point, model, points, observations, counters):
    if counters['head_solves'] >= 14:
        raise RuntimeError('The fourteen-solve method verification budget is exhausted')
    counters['head_solves'] += 1
    projection = api.project_regularized(model, points, observations)
    total, physical, penalty, _, _, _ = api.objective_regularized(model, points, observations)
    if not all(math.isfinite(float(value.detach())) for value in (total, physical, penalty)):
        raise FloatingPointError('A verification objective is nonfinite')
    json.dumps(projection, allow_nan=False)
    return dict(point, status='solved', solve_number=counters['head_solves'],
        total_objective=float(total.detach()), physical_objective=float(physical.detach()),
        penalty=float(penalty.detach()), projection=projection)


def run_case(case, plan, counters, progress):
    records = plan['inputs'][case]
    old = torch.load(records['final_checkpoint.pt']['path'], map_location='cpu', weights_only=False)
    raw = read(records['training-report.json']['path'])
    declared = next(value for value in common.CASES if value['id'] == case)
    config = api.RegularizedConfig(**plan['model_config'])
    model, initialization = api.matched_regularized_model(common.physics_for(declared), config,
        declared['known_h'], common.SEEDS['neural'])
    points = graded_training_points(model.physics, config, common.SEEDS['sampling'])
    if common.point_digest(points) != common.point_digest({name: np.asarray(value)
            for name, value in raw['training_points'].items()}):
        raise RuntimeError('The terminal copy used different training coordinates')
    if (old['known_h'] != declared['known_h'] or old['physics'] != asdict(model.physics)
        or any(plan['model_config'][name] != value for name, value in old['config'].items())):
        raise RuntimeError('Terminal physical/configuration metadata differs')
    with np.load(records['observations.npz']['path']) as data:
        observations = PlanarObservations(data['coordinates'].copy(), data['temperature'].copy(), .02)
    if counters['calibrations'] >= 2:
        raise RuntimeError('The two-calibration verification budget is exhausted')
    counters['calibrations'] += 1
    calibration_attempt = {'case': case, 'attempt': counters['calibrations'], 'status': 'running'}
    common._write(OUT/(case+'-calibration-attempt.json'), calibration_attempt)
    try:
        calibration = api.calibrate_readout(model, points, observations)
    except BaseException as error:
        common._write(OUT/(case+'-calibration-attempt.json'), dict(calibration_attempt,
            status='failed', error_type=type(error).__name__, error=str(error)))
        raise
    common._write(OUT/(case+'-calibration.json'), calibration)
    common._write(OUT/(case+'-calibration-attempt.json'), dict(calibration_attempt,
        status='completed', calibration=record(OUT/(case+'-calibration.json')),
        metadata_sha256=calibration['sha256'],
        calibration_spectral_calculations=calibration['calibration_spectral_calculations']))
    api.load_field_state(model, old['state_dict'])
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    samples = [dict(point, status='not_run') for point in protocol() if point['case'] == case]
    direction_metadata, failure, current_index, attempts_before = None, None, None, None
    try:
        for index, point in enumerate([value for value in protocol() if value['case'] == case]):
            current_index, attempts_before = index, counters['head_solves']
            if index:
                model.load_state_dict(baseline)
                with torch.no_grad():
                    for parameter, terminal, vector in zip(model.networks.parameters(), hidden, direction):
                        parameter.copy_(terminal+point['sign']*point['radius']*vector)
            try:
                value = point_result(point, model, points, observations, counters)
            except BaseException as error:
                value = dict(point, status='failed', solve_number=counters['head_solves'],
                             error_type=type(error).__name__, error=str(error))
                samples[index] = value
                progress(value)
                raise
            samples[index] = value
            progress(value)
            if index == 0:
                baseline = {name: value.detach().clone() for name, value in model.state_dict().items()}
                hidden = [parameter.detach().clone() for parameter in model.networks.parameters()]
                objective = api.objective_regularized(model, points, observations)[0]
                gradients = torch.autograd.grad(objective, list(model.networks.parameters()), allow_unused=True)
                gradients = [torch.zeros_like(parameter) if gradient is None else gradient.detach()
                    for parameter, gradient in zip(model.networks.parameters(), gradients)]
                if any(not bool(torch.all(torch.isfinite(value))) for value in gradients):
                    raise FloatingPointError('Nonfinite verification gradient')
                maximum = max(float(torch.max(abs(value))) for value in gradients)
                if not math.isfinite(maximum) or maximum == 0:
                    raise FloatingPointError('The fixed verification direction is undefined')
                direction = [-value/maximum for value in gradients]
                derivative = sum(float(torch.sum(g*d)) for g, d in zip(gradients, direction))
                if not math.isfinite(derivative):
                    raise FloatingPointError('Nonfinite verification derivative')
                direction_metadata = {'ad_directional_derivative': derivative,
                    'gradient_max_abs': maximum, 'gradient_clipped': False,
                    'direction_sha256': hashlib.sha256(b''.join(
                        value.cpu().numpy().astype('<f8').tobytes() for value in direction)).hexdigest()}
    except BaseException as error:
        failure = {'case': case, 'error_type': type(error).__name__, 'error': str(error)}
        if (current_index is not None and samples[current_index]['status'] == 'not_run'
            and counters['head_solves'] > attempts_before):
            samples[current_index] = dict(samples[current_index], status='failed',
                solve_number=counters['head_solves'], **failure)
    finally:
        model.load_state_dict(original)
    row = {'case': case, 'status': 'completed' if failure is None else 'failed',
        'calibration': record(OUT/(case+'-calibration.json')),
        'calibration_sha256': calibration['sha256'], 'fixed_lambda': calibration['fixed_lambda'],
        'calibration_spectral_calculations': calibration['calibration_spectral_calculations'],
        'initialization': initialization, 'direction': direction_metadata, 'samples': samples,
        'failure': failure, 'original_state_restored_exactly': all(
            torch.equal(value, original[name]) for name, value in model.state_dict().items())}
    row['checks'] = case_checks(row)
    return row


def run():
    plan = read(OUT/'plan.json')
    sha = common.digest_file(OUT/'plan.json')
    verify_plan(plan)
    if read(OUT/'state.json') != {'status': 'prepared', 'plan_sha256': sha}:
        raise RuntimeError('The once-only probe was already consumed')
    descriptor = os.open(OUT/'run.lock', os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, 'w') as handle:
        handle.write(str(os.getpid())+'\n')
    started, rows, failure = time.perf_counter(), [], None
    counters = {'calibrations': 0, 'head_solves': 0}
    def guard(signum, frame):
        if time.perf_counter()-started > 900. or common._rss_bytes() > 2*1024**3:
            signal.setitimer(signal.ITIMER_REAL, 0.)
            raise common.ResourceLimit('Regularized verification resource limit')
    def progress(value):
        with (OUT/'progress.jsonl').open('a') as handle:
            handle.write(json.dumps(value, allow_nan=False)+'\n')
    previous = signal.signal(signal.SIGALRM, guard)
    signal.setitimer(signal.ITIMER_REAL, .5, .5)
    common._write(OUT/'state.json', {'status': 'running', 'plan_sha256': sha})
    try:
        for case in CASES:
            row = run_case(case, plan, counters, progress)
            rows.append(row)
            common._write(OUT/(case+'.json'), row)
            if row['status'] != 'completed':
                failure = row['failure'] or {'error': 'Method execution failed'}
                break
    except BaseException as error:
        failure = {'case': case, 'error_type': type(error).__name__, 'error': str(error)}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, previous)
    preserved = False
    try:
        verify_plan(plan)
        preserved = True
    except BaseException as error:
        failure = dict(failure or {}, preservation_error=str(error))
    for case in CASES:
        if not any(row['case'] == case for row in rows):
            failed_before_points = failure is not None and failure.get('case') == case
            row = {'case': case, 'status': 'failed' if failed_before_points else 'not_run',
                'failure': failure if failed_before_points else None, 'direction': None, 'samples': [
                dict(point, status='not_run') for point in protocol() if point['case'] == case]}
            row['checks'] = case_checks(row)
            rows.append(row)
            common._write(OUT/(case+'.json'), row)
    # Canonical terminal accounting retains any interrupted attempted point.
    progress_rows = [sample for row in rows for sample in row['samples'] if sample['status'] != 'not_run']
    common._atomic(OUT/'progress.jsonl', ''.join(json.dumps(value, allow_nan=False)+'\n'
        for value in progress_rows).encode())
    calibration_attempts = []
    for case in CASES:
        path = OUT/(case+'-calibration-attempt.json')
        if path.is_file():
            value = read(path)
            if value['status'] == 'running':
                value.update(status='failed', error='Calibration bookkeeping interrupted',
                             interruption=failure)
                common._write(path, value)
            calibration_attempts.append(value)
    completed = failure is None and counters == {'calibrations': 2, 'head_solves': 14}
    passed = completed and preserved and all(row['checks']['passed'] for row in rows)
    result = {'schema': 'fixed_ridge_training_gradient_result_v1',
        'status': 'completed' if completed else 'failed', 'plan_sha256': sha,
        'method_checks_passed': bool(passed), 'counters': counters, 'cases': rows,
        'calibration_attempts': calibration_attempts,
        'training_only': True, 'neural_training_updates': 0, 'new_checkpoints': 0,
        'reference_fields_used': False, 'independent_diagnostics_used_for_selection': False,
        'preservation_verified': preserved, 'failure': failure,
        'wall_seconds': time.perf_counter()-started, 'peak_rss_bytes': common._rss_bytes()}
    common._write(OUT/'result.json', result)
    context = {'directory': str(OUT.resolve()), 'plan_sha256': sha,
        'result_sha256': common.digest_file(OUT/'result.json'), 'method_checks_passed': bool(passed)}
    decision = {'schema': 'regularized_training_only_method_decision_v1',
        'status': 'approved_for_forward_development' if passed else 'blocked_method_checks',
        'method_checks_passed': bool(passed), 'training_only': True, 'reference_fields_used': False,
        'independent_diagnostics_used_for_selection': False, 'selected_config': plan['model_config'],
        'gradient_probe': context, 'followup': FOLLOWUP,
        'no_guarantee_of_forward_gate_success': True, 'no_further_method_search': True}
    common._write(OUT/'method-decision.json', decision)
    index = {str(path.relative_to(OUT)): {'sha256': common.digest_file(path), 'bytes': path.stat().st_size}
        for path in sorted(OUT.rglob('*')) if path.is_file() and path not in (OUT/'state.json', OUT/'artifact-index.json')}
    common._write(OUT/'artifact-index.json', index)
    common._write(OUT/'state.json', {'status': result['status'], 'plan_sha256': sha,
        'artifact_index_sha256': common.digest_file(OUT/'artifact-index.json')})
    # Numerical evidence is sealed first; the tracked record is an exact copy.
    with TRACKED_RESULT.open('xb') as handle:
        handle.write((OUT/'result.json').read_bytes())
    return result


def verify_gradient_probe(directory):
    """Read-only authentication and gate recomputation; never performs a solve."""
    directory = Path(directory).resolve()
    if directory != OUT.resolve():
        raise RuntimeError('Only the single declared method probe is supported')
    plan, result, state, index = (read(directory/name) for name in
        ('plan.json', 'result.json', 'state.json', 'artifact-index.json'))
    verify_plan(plan)
    sha = common.digest_file(directory/'plan.json')
    if (state['status'] != result['status'] or state['plan_sha256'] != sha
        or result['plan_sha256'] != sha or state['artifact_index_sha256'] != common.digest_file(directory/'artifact-index.json')
        or (directory/'result.json').read_bytes() != TRACKED_RESULT.read_bytes()):
        raise RuntimeError('The saved method completion identities disagree')
    actual = {str(path.relative_to(directory)) for path in directory.rglob('*')
        if path.is_file() and path not in (directory/'state.json', directory/'artifact-index.json')}
    if set(index) != actual:
        raise RuntimeError('The sealed method artifact set changed')
    for name, expected in index.items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory) or {'sha256': common.digest_file(path), 'bytes': path.stat().st_size} != expected:
            raise RuntimeError('A sealed method artifact changed')
    if [row['case'] for row in result['cases']] != list(CASES):
        raise RuntimeError('The two declared verification cases were not retained')
    all_samples = []
    for row in result['cases']:
        expected = [point for point in protocol() if point['case'] == row['case']]
        if ([{key: sample[key] for key in ('case','kind','radius','sign')} for sample in row['samples']] != expected
            or row['checks'] != case_checks(row) or read(directory/(row['case']+'.json')) != row):
            raise RuntimeError('A saved opportunity or recomputed method check differs')
        if row.get('calibration') and record(row['calibration']['path']) != row['calibration']:
            raise RuntimeError('A calibration artifact changed')
        all_samples.extend(row['samples'])
    solved = [sample for sample in all_samples if sample['status'] == 'solved']
    attempts = [sample for sample in all_samples if sample['status'] != 'not_run']
    progress = [json.loads(line) for line in (directory/'progress.jsonl').read_text().splitlines()
                if line.strip()] if (directory/'progress.jsonl').is_file() else []
    if progress != attempts or [sample['solve_number'] for sample in attempts] != list(range(1,len(attempts)+1)):
        raise RuntimeError('Per-opportunity solve accounting differs from progress')
    counters = result['counters']
    calibration_attempts = result.get('calibration_attempts', [])
    if (len(calibration_attempts) != counters['calibrations']
        or [value['attempt'] for value in calibration_attempts] != list(range(1,len(calibration_attempts)+1))
        or len({value['case'] for value in calibration_attempts}) != len(calibration_attempts)):
        raise RuntimeError('Calibration attempts were not retained accurately')
    for value in calibration_attempts:
        if (value['case'] not in CASES or value['status'] not in ('completed','failed')
            or read(directory/(value['case']+'-calibration-attempt.json')) != value):
            raise RuntimeError('A calibration attempt identity is invalid')
        if value['status'] == 'completed':
            calibration = read(value['calibration']['path'])
            scales = np.asarray(calibration['column_scale'], dtype=float)
            bindings = calibration['bindings']
            if (record(value['calibration']['path']) != value['calibration']
                or calibration['sha256'] != value['metadata_sha256']
                or calibration['sha256'] != api._digest({key:item for key,item in calibration.items() if key != 'sha256'})
                or calibration['calibration_spectral_calculations'] != 1
                or value['calibration_spectral_calculations'] != 1
                or calibration['relative_scale'] != plan['model_config']['regularization_relative_scale']
                or calibration['fixed_lambda'] != calibration['relative_scale']*calibration['initial_sigma_max']
                or not math.isfinite(calibration['fixed_lambda']) or calibration['fixed_lambda'] <= 0
                or calibration['calibration_matrix_builds'] != 1
                or not np.isfinite(scales).all() or (scales < 1.e-12).any()
                or api._array_digest(scales) != calibration['column_scale_sha256']
                or bindings['config'] != plan['model_config']
                or bindings['known_h'] != next(row['known_h'] for row in common.CASES if row['id'] == value['case'])):
                raise RuntimeError('A completed calibration does not match the frozen method')
            case_row = next(row for row in result['cases'] if row['case'] == value['case'])
            if (case_row.get('calibration_sha256') != calibration['sha256']
                or case_row.get('fixed_lambda') != calibration['fixed_lambda']
                or any(sample['projection'].get('fixed_lambda') != calibration['fixed_lambda']
                    or sample['projection'].get('column_scale_sha256') != calibration['column_scale_sha256']
                    for sample in case_row['samples'] if sample['status'] == 'solved')):
                raise RuntimeError('A solved point changed the frozen ridge calibration')
    passed = (result['status'] == 'completed' and result['preservation_verified'] is True
        and counters == {'calibrations': 2, 'head_solves': 14} and len(solved) == 14
        and all(value['status'] == 'completed' for value in calibration_attempts)
        and all(row['checks']['passed'] for row in result['cases']))
    if (result['method_checks_passed'] is not bool(passed) or counters['head_solves'] != len(attempts)
        or not 0 <= counters['calibrations'] <= 2 or len(attempts) > 14
        or result['neural_training_updates'] != 0 or result['new_checkpoints'] != 0
        or result['reference_fields_used'] is not False
        or result['independent_diagnostics_used_for_selection'] is not False):
        raise RuntimeError('Method scope, counts or passing flag is inconsistent')
    context = {'directory': str(directory), 'plan_sha256': sha,
        'result_sha256': common.digest_file(directory/'result.json'), 'method_checks_passed': bool(passed)}
    decision = read(directory/'method-decision.json')
    expected_decision = {'schema': 'regularized_training_only_method_decision_v1',
        'status': 'approved_for_forward_development' if passed else 'blocked_method_checks',
        'method_checks_passed': bool(passed), 'training_only': True, 'reference_fields_used': False,
        'independent_diagnostics_used_for_selection': False, 'selected_config': plan['model_config'],
        'gradient_probe': context, 'followup': FOLLOWUP,
        'no_guarantee_of_forward_gate_success': True, 'no_further_method_search': True}
    if decision != expected_decision:
        raise RuntimeError('The method decision changed')
    return context


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','run','verify'))
    args = parser.parse_args()
    value = prepare() if args.command == 'prepare' else run() if args.command == 'run' else verify_gradient_probe(OUT)
    print(json.dumps({key: value[key] for key in ('status','method_checks_passed','counters') if key in value}, allow_nan=False))
