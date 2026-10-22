"""Predeclared, paired development checks for an additive 2D field revision.

The runner supplies only sparse observations and known physical conditions to
the fit callback.  It retains the optimizer-returned checkpoint before hidden
field scoring.  It never generates a reserved evaluation cohort or retries an
unsuccessful arm.  The v2 data, diagnostic coordinates, scales, and gates are
kept fixed, making an architecture repair directly comparable to phase B.
"""
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time
import traceback
from typing import Callable

import numpy as np

from ..physics.planar_contact import PlanarContactConfig
from ..pinn.planar_contact_inverse import PlanarObservations

ROOT = Path(__file__).resolve().parents[2]
CASES = tuple({'id': case, 'geometry': geometry, 'known_h': h} for case, geometry, h in (
    ('t01', 'full_contact', 800.), ('t02', 'partial_contact', 800.),
    ('t03', 'full_contact', 1400.), ('t04', 'partial_contact', 1400.)))
SEEDS = {'noise': 2026102201, 'neural': 2026102202,
         'sampling': 2026102203, 'diagnostic': 2026102204}
GATES = {'normalized_observation_mse': 2., 'each_normalized_residual_rms': .05,
         'field_rmse_K': .05, 'field_maximum_error_K': .15,
         'interface_temperature_scale_K': .1, 'flux_scale_W_m2': 500.,
         'interior_scale_K_s': 5./120.}


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()


def _atomic(path, data):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write(path, payload):
    _atomic(path, _json_bytes(payload))


def _npz(path, **values):
    handle = io.BytesIO()
    np.savez_compressed(handle, **values)
    _atomic(path, handle.getvalue())


def runtime_context():
    return {'implementation': platform.python_implementation(),
            'python': platform.python_version(),
            'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'torch')},
            'float': 'float64', 'threads': 1}


def physics_for(case):
    physics = PlanarContactConfig()
    if case['geometry'] == 'full_contact':
        return replace(physics, contact_start=0., contact_end=physics.width)
    if case['geometry'] != 'partial_contact':
        raise ValueError('unknown development geometry')
    return physics


def independent_points(physics, seed=SEEDS['diagnostic']):
    """Exact v2 shifted grids, in normalized (x/width,z/thickness,t/120)."""
    rng = np.random.default_rng(seed)
    x = (np.arange(40)+rng.uniform(.3, .7))/40
    z = (np.arange(12)+rng.uniform(.3, .7))/12
    times = np.array([.25, .75, 1.5, 3., 6., 12., 22., 38., 60., 88., 115.])/120.
    xx, zz, tt = np.meshgrid(x, z, times, indexing='ij')
    points = {'interior': np.column_stack((xx.ravel(), zz.ravel(), tt.ravel()))}
    xb = (np.arange(160)+rng.uniform(.3, .7))/160
    xx, tt = np.meshgrid(xb, times, indexing='ij')
    flatx, flatt = xx.ravel(), tt.ravel()
    points['top'] = np.column_stack((flatx, np.ones(len(flatx)), flatt))
    contact = (flatx*physics.width >= physics.contact_start) & (flatx*physics.width <= physics.contact_end)
    points['bottom_contact'] = np.column_stack((flatx[contact], np.zeros(contact.sum()), flatt[contact]))
    if (~contact).any():
        points['bottom_insulated'] = np.column_stack((flatx[~contact], np.zeros((~contact).sum()), flatt[~contact]))
    zz, tt = np.meshgrid((np.arange(24)+.5)/24, times, indexing='ij')
    for side, boundary in (('left', 0.), ('right', 1.)):
        points[side] = np.column_stack((np.full(zz.size, boundary), zz.ravel(), tt.ravel()))
    interfaces = []
    if physics.contact_start > 0:
        interfaces.append(physics.contact_start)
    if physics.contact_end < physics.width:
        interfaces.append(physics.contact_end)
    for i, location in enumerate(interfaces):
        points[f'interface_{i}'] = np.column_stack((np.full(zz.size, location/physics.width), zz.ravel(), tt.ravel()))
    return points


def point_digest(points):
    return canonical_digest({key: {'shape': list(value.shape),
        'sha256': hashlib.sha256(np.asarray(value, dtype='<f8').tobytes()).hexdigest()}
        for key, value in sorted(points.items())})


def near_interface_points(physics):
    """Fixed supplemental points; no corner points and no acceptance role."""
    edges = []
    if physics.contact_start > 0:
        edges.append(physics.contact_start/physics.width)
    if physics.contact_end < physics.width:
        edges.append(physics.contact_end/physics.width)
    if not edges:
        return {}
    offsets = np.array([-.005, -.001, -.0002, .0002, .001, .005])
    heights = np.array([.002, .01, .05, .2, .5])
    times = np.array([.4, 1.1, 4.5, 14., 45., 105.])/120.
    x = np.concatenate([edge+offsets for edge in edges])
    xx, zz, tt = np.meshgrid(x, heights, times, indexing='ij')
    points = {'interior': np.column_stack((xx.ravel(), zz.ravel(), tt.ravel()))}
    zz, tt = np.meshgrid(heights, times, indexing='ij')
    for i, edge in enumerate(edges):
        points[f'interface_{i}'] = np.column_stack((np.full(zz.size, edge), zz.ravel(), tt.ravel()))
    return points


def expected_families(physics):
    names = {'interior', 'left', 'right', 'top', 'bottom_contact'}
    if physics.contact_start > 0 or physics.contact_end < physics.width:
        names.add('bottom_insulated')
    for i in range(int(physics.contact_start > 0)+int(physics.contact_end < physics.width)):
        names.update((f'interface_{i}_temperature', f'interface_{i}_flux'))
    return names


def acceptance(physics, observation_mse, residuals, field):
    """Missing, extra, negative, and nonfinite diagnostics never pass."""
    finite = lambda value: not isinstance(value, (bool, np.bool_)) and isinstance(value, (float, int, np.floating, np.integer)) and math.isfinite(value) and value >= 0
    gate = {'observations': bool(finite(observation_mse) and observation_mse <= 2.),
            'each_independent_residual': bool(set(residuals) == expected_families(physics)
                and all(finite(value) and value <= .05 for value in residuals.values())),
            'field': bool(finite(field.get('rmse_K')) and finite(field.get('maximum_error_K'))
                and field['rmse_K'] <= .05 and field['maximum_error_K'] <= .15)}
    return dict(gate, all=all(gate.values()))


def _source_records(paths):
    records = {}
    for path in paths:
        path = Path(path)
        path = (ROOT/path).resolve() if not path.is_absolute() else path.resolve()
        relative = str(path.relative_to(ROOT))
        records[relative] = digest_file(path)
    return dict(sorted(records.items()))


def _tree_records(path):
    path = Path(path).resolve()
    if not path.is_dir():
        raise ValueError('protected directory does not exist: '+str(path))
    return {str(item.relative_to(path)): digest_file(item) for item in sorted(path.rglob('*')) if item.is_file()}


def _indexed_context(directory):
    directory = Path(directory).resolve()
    index_path = directory/'evidence-index.json'
    index = json.loads(index_path.read_text())
    records = index['files']
    if canonical_digest(records) != index['digest']:
        raise ValueError('protected evidence index digest differs')
    names = [record['path'] for record in records]
    if len(names) != len(set(names)):
        raise ValueError('duplicate protected evidence path')
    for record in records:
        path = (directory/record['path']).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError('invalid protected evidence path')
        if path.stat().st_size != record['bytes'] or digest_file(path) != record['sha256']:
            raise ValueError('sealed v2 evidence changed: '+record['path'])
    return {'directory': str(directory), 'index_sha256': digest_file(index_path),
            'digest': index['digest'], 'files': records}


def prepare_development(output_directory, *, model_config, source_paths, paired_v2_directory,
                        protected_directories=(), maximum_wall_seconds=3600.,
                        maximum_rss_bytes=2*1024**3, candidate_id='continuity-v3',
                        v2_development_record=None, preserved_campaign=None):
    """Persist the complete bounded plan before data or fits exist."""
    if not math.isfinite(maximum_wall_seconds) or maximum_wall_seconds <= 0:
        raise ValueError('wall limit must be positive')
    if type(maximum_rss_bytes) is not int or maximum_rss_bytes <= 0:
        raise ValueError('RSS limit must be a positive integer')
    physical_trace = {'duration': 120., 'initial_temperature': 300., 'temperature_scale': 5.,
        'ramp_seconds': 10., 'bath_drop_kelvin': 5.,
        'top_flux_watts_per_square_metre': 500., 'interface_temperature_scale': .1}
    if any(key in model_config and model_config[key] != value for key, value in physical_trace.items()):
        raise ValueError('model config changes the paired physical trace or normalization')
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=False)
    fixed = ('thermotwin/studies/pinn_spatial_continuity.py',
             'thermotwin/physics/planar_contact.py', 'thermotwin/pinn/planar_contact_inverse.py')
    sources = _source_records(tuple(source_paths)+fixed)
    protected = [{'directory': str(Path(path).resolve()), 'files': _tree_records(path)}
                 for path in protected_directories]
    paired_root = Path(paired_v2_directory).resolve()
    indexed = [_indexed_context(paired_root.parent)] if (paired_root.parent/'evidence-index.json').is_file() else []
    prior_development = None
    if v2_development_record is not None:
        record_path = (ROOT/v2_development_record).resolve()
        record = json.loads(record_path.read_text())
        for name, expected in record['verification']['source_sha256'].items():
            if digest_file(ROOT/name) != expected:
                raise ValueError('v2 development source changed: '+name)
        if indexed and (len(indexed[0]['files']) != record['development_artifact_index']['files']
                        or indexed[0]['digest'] != record['development_artifact_index']['digest']):
            raise ValueError('v2 evidence seal differs from tracked development record')
        prior_development = {'path': str(record_path), 'sha256': digest_file(record_path),
            'source_sha256': record['verification']['source_sha256'],
            'preserved_evidence': record['preserved_evidence'],
            'preserved_campaign': str(Path(preserved_campaign).resolve()) if preserved_campaign else None}
    paired = {}
    for case in CASES:
        original = paired_root/case['id'] if case['id'] in ('t01', 't02') else paired_root/'resource-continuation'/case['id']
        paired[case['id']] = {name: {'path': str(original/name), 'sha256': digest_file(original/name)}
            for name in ('observations.npz', 'field_scoring.npz')}
    plan = {'schema': 'paired_2d_continuity_development_v3', 'candidate_id': candidate_id,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'development only; no inverse fitting or fresh evaluation',
        'model_config': model_config, 'cases': list(CASES), 'seeds': SEEDS, 'gates': GATES,
        'maximum_fits': 4, 'retries': 0, 'maximum_wall_seconds': maximum_wall_seconds,
        'maximum_rss_bytes': maximum_rss_bytes, 'threads': 1,
        'selection': 'same predeclared configuration must pass all four arms; retain every failure',
        'checkpoint_rule': 'optimizer-returned final state; atomically retained before independent diagnostics or field scoring',
        'runtime': runtime_context(), 'source_sha256': sources,
        'protected_directories': protected, 'protected_indices': indexed,
        'prior_development': prior_development, 'paired_evidence': paired,
        'data_policy': 'reuse exact saved v2 observations; load hidden reference field only after final checkpoint and raw training report',
        'reference': {'nx': 160, 'nz': 40, 'max_step_s': .0625, 'initial_temperature_K': 300.,
                      'bath': '300+5*expm1(-t/10)', 'top_flux': '-500*expm1(-t/10)',
                      'output_times_s': [0., 1., 2., 5.]+list(np.arange(10., 121., 10.)),
                      'hidden_field': '40x20 cell centers at every reference output'},
        'observations': {'x_m': [.004, .020, .036], 'z_m': .003,
            'times_s': list(range(0, 121, 10)), 'omit': 'center sensor at 40,50,60 seconds',
            'standard_deviation_K': .02, 'count': 36, 'pairing': 'exact same phase-B noise seed in every arm'},
        'diagnostic_point_digests': {case['geometry']: point_digest(independent_points(physics_for(case))) for case in CASES},
        'supplemental_near_interface_audit': {'role': 'report only; no acceptance, fitting, or checkpoint selection',
            'x_offsets_normalized': [-.005, -.001, -.0002, .0002, .001, .005],
            'z_normalized': [.002, .01, .05, .2, .5],
            'times_s': [.4, 1.1, 4.5, 14., 45., 105.],
            'point_digests': {case['geometry']: point_digest(near_interface_points(physics_for(case))) for case in CASES}}}
    _write(directory/'plan.json', plan)
    _write(directory/'state.json', {'status': 'prepared', 'plan_sha256': digest_file(directory/'plan.json')})
    return plan


def verify_plan(plan):
    if plan.get('cases') != list(CASES) or plan.get('seeds') != SEEDS or plan.get('gates') != GATES:
        raise ValueError('paired cases, random streams, or acceptance gates changed')
    if plan.get('maximum_fits') != 4 or plan.get('retries') != 0 or plan.get('threads') != 1:
        raise ValueError('invalid bounded execution policy')
    if plan.get('runtime') != runtime_context():
        raise RuntimeError('runtime differs from prepared plan')
    if _source_records(plan['source_sha256']) != plan['source_sha256']:
        raise RuntimeError('source changed since preparation')
    for protected in plan['protected_directories']:
        if _tree_records(protected['directory']) != protected['files']:
            raise RuntimeError('protected development evidence changed')
    for indexed in plan['protected_indices']:
        if _indexed_context(indexed['directory']) != indexed:
            raise RuntimeError('sealed v2 development evidence changed')
    prior_development = plan.get('prior_development')
    if prior_development is not None:
        if digest_file(prior_development['path']) != prior_development['sha256']:
            raise RuntimeError('tracked v2 development record changed')
        if _source_records(prior_development['source_sha256']) != prior_development['source_sha256']:
            raise RuntimeError('v2 numerical source or tests changed')
        if prior_development['preserved_campaign']:
            from .pinn_spatial_revision import preserved_context
            preserved_context(prior_development['preserved_campaign'], expected=prior_development['preserved_evidence'])
    for pair in plan['paired_evidence'].values():
        for record in pair.values():
            if digest_file(record['path']) != record['sha256']:
                raise RuntimeError('paired v2 observation or reference artifact changed')
    for case in CASES:
        if point_digest(independent_points(physics_for(case))) != plan['diagnostic_point_digests'][case['geometry']]:
            raise RuntimeError('independent diagnostic coordinates changed')
        if point_digest(near_interface_points(physics_for(case))) != plan['supplemental_near_interface_audit']['point_digests'][case['geometry']]:
            raise RuntimeError('supplemental near-interface audit coordinates changed')


def paired_observations(case, plan):
    """Read only observed coordinates/temperatures, never the stored clean array."""
    record = plan['paired_evidence'][case['id']]['observations.npz']
    if digest_file(record['path']) != record['sha256']:
        raise RuntimeError('paired observations changed')
    with np.load(record['path'], allow_pickle=False) as stored:
        return PlanarObservations(stored['coordinates'], stored['temperature'], .02)


def paired_scoring_field(case, plan):
    """Called only after the final model and raw training report are retained."""
    record = plan['paired_evidence'][case['id']]['field_scoring.npz']
    if digest_file(record['path']) != record['sha256']:
        raise RuntimeError('paired reference field changed')
    with np.load(record['path'], allow_pickle=False) as stored:
        return stored['coordinates'].copy(), stored['truth'].copy()


@dataclass(frozen=True)
class ForwardCallbacks:
    fit: Callable
    predict: Callable
    diagnose: Callable
    checkpoint: Callable


class ResourceLimit(RuntimeError):
    pass


def _rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value*1024)


def run_development(output_directory, callbacks):
    """Execute an already prepared plan once, with immutable sparse-fit inputs."""
    import torch
    directory = Path(output_directory)
    plan = json.loads((directory/'plan.json').read_text())
    state = json.loads((directory/'state.json').read_text())
    if state.get('status') != 'prepared' or state.get('plan_sha256') != digest_file(directory/'plan.json'):
        raise ValueError('plan is changed, already consumed, or not prepared')
    verify_plan(plan)
    descriptor = os.open(directory/'run.lock', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as handle:
        handle.write(json.dumps({'pid': os.getpid(), 'plan_sha256': state['plan_sha256']})+'\n')
    prepared_sha256 = state['plan_sha256']
    def verify_live_plan():
        if digest_file(directory/'plan.json') != prepared_sha256:
            raise RuntimeError('prepared plan changed during execution')
        verify_plan(plan)
    torch.set_num_threads(1)
    started = time.perf_counter()
    rows = [dict(case, status='not_run') for case in CASES]
    current = None
    previous_alarm = signal.getsignal(signal.SIGALRM)
    previous_terminate = signal.getsignal(signal.SIGTERM)
    def check_resource(event=None):
        if time.perf_counter()-started > plan['maximum_wall_seconds']:
            raise ResourceLimit('predeclared campaign wall limit reached')
        if _rss_bytes() > plan['maximum_rss_bytes']:
            raise ResourceLimit('predeclared campaign RSS limit reached')
        if event is not None:
            with (current/'progress.jsonl').open('a') as handle:
                handle.write(json.dumps(event, allow_nan=False)+'\n')
    def alarm(signum, frame):
        try:
            check_resource()
        except ResourceLimit:
            signal.setitimer(signal.ITIMER_REAL, 0.)
            raise
    def terminate(signum, frame):
        signal.setitimer(signal.ITIMER_REAL, 0.)
        raise ResourceLimit('development process interrupted; no retry')
    _write(directory/'state.json', dict(status='running', plan_sha256=digest_file(directory/'plan.json')))
    signal.signal(signal.SIGALRM, alarm)
    signal.signal(signal.SIGTERM, terminate)
    signal.setitimer(signal.ITIMER_REAL, min(plan['maximum_wall_seconds'], .5), .5)
    blocked = False
    try:
        for index, case in enumerate(CASES):
            if blocked:
                rows[index]['reason'] = 'campaign resource or preservation limit'
                continue
            current = directory/case['id']
            current.mkdir(exist_ok=False)
            _write(current/'declared-config.json', {'case': case, 'config': plan['model_config'],
                'physics': asdict(physics_for(case)), 'seeds': SEEDS,
                'source_sha256': plan['source_sha256'], 'truth_use': 'known-h forward only; field temperatures score only'})
            try:
                check_resource()
                verify_live_plan()
                physics = physics_for(case)
                observations = paired_observations(case, plan)
                source_observations = plan['paired_evidence'][case['id']]['observations.npz']['path']
                _atomic(current/'observations.npz', Path(source_observations).read_bytes())
                model, raw = callbacks.fit(observations, physics=physics,
                    config=plan['model_config'], known_h=case['known_h'],
                    neural_seed=SEEDS['neural'], sampling_seed=SEEDS['sampling'],
                    progress=check_resource)
                check_resource()
                _atomic(current/'final_checkpoint.pt', callbacks.checkpoint(model))
                _write(current/'training-report.json', raw)
                verify_live_plan()
                diagnostics = callbacks.diagnose(model, independent_points(physics))
                _write(current/'independent-diagnostics.json', diagnostics)
                audit_points = near_interface_points(physics)
                audit = {'status': 'not_applicable', 'role': 'report only', 'point_digest': point_digest(audit_points)}
                if audit_points:
                    _npz(current/'near-interface-audit-points.npz', **audit_points)
                    try:
                        supplemental = callbacks.diagnose(model, audit_points)
                        audit.update(status='completed', diagnostics=supplemental)
                        _json_bytes(audit)
                    except ResourceLimit:
                        raise
                    except Exception as error:
                        audit.update(status='failed', error=repr(error), traceback=traceback.format_exc())
                        audit.pop('diagnostics', None)
                _write(current/'near-interface-audit.json', audit)
                coordinates, truth = paired_scoring_field(case, plan)
                predicted = np.asarray(callbacks.predict(model, coordinates), dtype=float)
                obs_prediction = np.asarray(callbacks.predict(model, observations.coordinates), dtype=float)
                if predicted.shape != truth.shape or obs_prediction.shape != observations.temperature.shape:
                    raise ValueError('prediction shapes differ from declared coordinates')
                if not np.isfinite(predicted).all() or not np.isfinite(obs_prediction).all():
                    raise FloatingPointError('nonfinite field or observation predictions')
                field = {'rmse_K': float(np.sqrt(np.mean((predicted-truth)**2))),
                         'maximum_error_K': float(np.max(abs(predicted-truth)))}
                observation_mse = float(np.mean(((obs_prediction-observations.temperature)/.02)**2))
                gate = acceptance(physics, observation_mse, diagnostics['normalized_rms'], field)
                _npz(current/'field_scoring.npz', coordinates=coordinates, truth=truth, prediction=predicted)
                row = dict(case, status='completed', config=plan['model_config'], gate=gate,
                    field_metrics=field, normalized_observation_mse=observation_mse,
                    independent_residuals=diagnostics, raw_checkpoint='final_checkpoint.pt',
                    supplemental_audit_status=audit['status'],
                    checkpoint_sha256=digest_file(current/'final_checkpoint.pt'))
                verify_live_plan()
                rows[index] = row
                _write(current/'result.json', row)
            except Exception as error:
                if isinstance(error, ResourceLimit):
                    signal.setitimer(signal.ITIMER_REAL, 0.)
                    blocked = True
                rows[index] = dict(case, status='failed', error=repr(error), traceback=traceback.format_exc())
                _write(current/'failure.json', rows[index])
                try:
                    verify_live_plan()
                except Exception:
                    blocked = True
            _write(directory/'partial-results.json', rows)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, previous_alarm)
        signal.signal(signal.SIGTERM, previous_terminate)
    source_unchanged = True
    try:
        verify_live_plan()
    except Exception:
        source_unchanged = False
    passed = source_unchanged and all(row.get('gate', {}).get('all') is True for row in rows)
    result = {'status': 'passed' if passed else 'blocked_forward_gate',
        'selected_config': plan['model_config'] if passed else None,
        'selected_candidate_id': plan['candidate_id'] if passed else None,
        'source_sha256': plan['source_sha256'], 'source_unchanged': source_unchanged,
        'cases': rows, 'wall_seconds': time.perf_counter()-started, 'peak_rss_bytes': _rss_bytes(),
        'plan_sha256': digest_file(directory/'plan.json'), 'all_failures_retained': True,
        'inverse_opened': False, 'fresh_evaluation_opened': False,
        'hidden_truth_labels_used_in_neural_loss': False}
    _write(directory/'final-forward-selection.json', result)
    index = {str(path.relative_to(directory)): {'sha256': digest_file(path), 'bytes': path.stat().st_size}
             for path in sorted(directory.rglob('*')) if path.is_file() and path.name not in ('state.json', 'artifact-index.json')}
    _write(directory/'artifact-index.json', index)
    _write(directory/'state.json', {'status': 'completed', 'scientific_status': result['status'],
        'plan_sha256': result['plan_sha256'], 'artifact_index_sha256': digest_file(directory/'artifact-index.json')})
    return result


def continuity_callbacks():
    """Bind the additive model lazily, after the preexecution plan is reviewed."""
    from ..pinn.planar_contact_continuity import (
        ContinuityConfig, fit_continuity, predict_continuity,
        diagnose_continuity, save_checkpoint,
    )
    def fit(observations, **kwargs):
        kwargs['config'] = ContinuityConfig(**kwargs['config'])
        return fit_continuity(observations, **kwargs)
    def checkpoint(model):
        stream = io.BytesIO()
        save_checkpoint(model, stream)
        return stream.getvalue()
    return ForwardCallbacks(fit=fit, predict=predict_continuity,
        diagnose=diagnose_continuity, checkpoint=checkpoint)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Execute a prepared, paired 2D continuity development plan once.')
    parser.add_argument('--output-directory', required=True)
    args = parser.parse_args(argv)
    result = run_development(args.output_directory, continuity_callbacks())
    print(json.dumps({'status': result['status'], 'case_count': len(result['cases']),
        'wall_seconds': result['wall_seconds']}, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
