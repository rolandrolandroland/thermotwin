"""Development and frozen evaluation primitives for delayed-contact PINNs.

The fitting boundary accepts observations and known inputs only. A nominal-h,
physics-only warm start receives just the already-known initial temperature.
True contact and hidden fields enter the separate scoring boundary only.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from ..physics.planar_contact import PlanarContactConfig, PlanarContactModel
from ..pinn.planar_contact_inverse import PlanarObservations
from ..pinn.planar_contact_revision import (
    PlanarRevisionConfig, train_planar_revision, predict_planar_revision,
)


INVERSE_SPEC = {
    'namespace': 'pinn_spatial_revision_v2_2026_10',
    'strategies': ['delayed_release', 'alternating'],
    'warm_start': 'physics only at frozen nominal h; known IC only; no oracle checkpoint',
    'initial_h': math.sqrt(400.*2000.), 'bounds': [400., 2000.],
    'alternating_cycles': 3, 'contact_only_adam_epochs': 150, 'contact_only_lbfgs_iterations': 100,
    'strategy_selection': 'all development gates, then smallest total training wall-independent epoch cap; fixed order tie break',
    'development_h': [800., 1400.],
    'evaluation_h': [650., 1100., 1600.], 'evaluation_repeats_per_h': 2,
    'evaluation_methods': ['selected_revised_pinn', 'conventional', 'frozen_initial_h'],
    'trials_retained': True, 'retries': 0,
    'gates': {'observation_mse': 2., 'each_normalized_physics_rms': .05,
              'hidden_field_rmse_K': .05, 'hidden_field_max_K': .15,
              'relative_h_error': .1, 'minimum_initial_absolute_error_reduction': .5,
              'holdout_rmse_K': .05, 'holdout_max_K': .15},
    'field_times_s': [0, 1, 2, 5] + list(range(10, 121, 10)),
    'field_grid': [40, 20], 'reference_grid': [160, 40], 'reference_max_step_s': .0625,
    'conventional_grid': [120, 30], 'conventional_max_step_s': .0625,
    'reference_checks': {'finest_grid': [320, 80], 'refined_time_step_s': .03125,
                         'sensor_max_K': .002, 'field_rmse_K': .005, 'field_max_K': .01},
}


def save_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')


def semantic_seed(partition, case, purpose):
    text = f"{INVERSE_SPEC['namespace']}/{partition}/{case}/{purpose}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'big')


def physical_gate(report):
    physics = report['physics']
    required = {'interior', 'left', 'right', 'top', 'bottom_contact'}
    sides = int(physics['contact_start'] > 0) + int(physics['contact_end'] < physics['width'])
    if sides:
        required.add('bottom_insulated')
    if report['architecture'] == 'hard_boundary_decomposed':
        for index in range(sides):
            required.update((f'interface_{index}_temperature', f'interface_{index}_flux'))
    values = report['independent_residuals']['normalized_rms']
    missing = sorted(required-set(values))
    passed = not missing and all(isinstance(values[key], (float, int)) and math.isfinite(values[key])
                                and 0 <= values[key] <= .05 for key in required)
    return {'passed': bool(passed), 'required': sorted(required), 'missing': missing,
            'normalized_rms': values}


def fit_delayed_inverse(observations, *, config, physics=PlanarContactConfig(),
                        strategy='delayed_release', seed_namespace='development/0', progress=None,
                        completed_stage=None):
    """No truth label, true-h value, verification field, or oracle is accepted."""
    if strategy not in INVERSE_SPEC['strategies'] or not isinstance(observations, PlanarObservations):
        raise ValueError('unknown inverse strategy or unauthenticated observed data')
    if not isinstance(config, PlanarRevisionConfig):
        config = PlanarRevisionConfig(**config)
    config = replace(config, initial_h=INVERSE_SPEC['initial_h'])
    seeds = {name: semantic_seed(seed_namespace, 0, name) for name in ('neural', 'sampling', 'independent')}
    if len(set(seeds.values())) != len(seeds):
        raise ValueError('semantic seed collision')
    known_ic = PlanarObservations(np.array([[.004, .003, 0.], [.036, .003, 0.]]),
                                 np.array([config.initial_temperature]*2), .02)
    started, cpu = time.monotonic(), time.process_time()
    stages = []
    def retain_stage(model, report):
        stages.append(report)
        if completed_stage is not None:
            completed_stage(model, report, len(stages)-1)

    warm = replace(config, observation_weight=0.)
    model, report = train_planar_revision(known_ic, physics=physics, config=warm, mode='inverse',
        known_h=None, neural_seed=seeds['neural'], sampling_seed=seeds['sampling'],
        independent_seed=seeds['independent'], freeze_contact=True, progress=progress)
    report['stage'] = 'nominal_physics_only_warm_start'
    report['warm_start_observation_information'] = 'known initial condition only'
    retain_stage(model, report)
    gate = physical_gate(report)
    if not gate['passed']:
        return model, {'status': 'blocked_nominal_forward_gate', 'stages': stages, 'warm_gate': gate,
            'strategy': strategy, 'contact_estimate': float(model.contact.detach()),
            'wall_seconds': time.monotonic()-started, 'cpu_seconds': time.process_time()-cpu,
            'unknown_truth_used': False}
    if strategy == 'delayed_release':
        model, report = train_planar_revision(observations, physics=physics,
            config=replace(config, observation_weight=1.), mode='inverse', known_h=None,
            neural_seed=seeds['neural'], sampling_seed=semantic_seed(seed_namespace, 1, 'sampling'),
            independent_seed=semantic_seed(seed_namespace, 1, 'independent'), initial_model=model,
            freeze_contact=False, progress=progress)
        report['stage'] = 'joint_after_nominal_field_warm_start'
        retain_stage(model, report)
    else:
        for cycle in range(INVERSE_SPEC['alternating_cycles']):
            # Hard Robin lifting depends on h; differentiating this stage is
            # meaningful. A Robin least-squares update on that field is an identity.
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name == 'raw_h')
            contact_config = replace(config, observation_weight=1.,
                adam_epochs=INVERSE_SPEC['contact_only_adam_epochs'],
                lbfgs_iterations=INVERSE_SPEC['contact_only_lbfgs_iterations'])
            model, report = train_planar_revision(observations, physics=physics, config=contact_config,
                mode='inverse', known_h=None, neural_seed=seeds['neural'],
                sampling_seed=semantic_seed(seed_namespace, cycle, 'contact_sampling'),
                independent_seed=semantic_seed(seed_namespace, cycle, 'contact_independent'),
                initial_model=model, freeze_contact=False, progress=progress)
            report['stage'] = f'cycle_{cycle}_contact_only'
            retain_stage(model, report)
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name != 'raw_h')
            model, report = train_planar_revision(observations, physics=physics,
                config=replace(config, observation_weight=1.), mode='inverse', known_h=None,
                neural_seed=seeds['neural'], sampling_seed=semantic_seed(seed_namespace, cycle, 'field_sampling'),
                independent_seed=semantic_seed(seed_namespace, cycle, 'field_independent'),
                initial_model=model, freeze_contact=True, progress=progress)
            report['stage'] = f'cycle_{cycle}_field_only'
            retain_stage(model, report)
    return model, {'status': 'completed', 'stages': stages, 'warm_gate': gate, 'strategy': strategy,
        'contact_estimate': float(model.contact.detach()), 'final_report': stages[-1],
        'wall_seconds': time.monotonic()-started, 'cpu_seconds': time.process_time()-cpu,
        'unknown_truth_used': False,
        'initial_h': INVERSE_SPEC['initial_h'], 'checkpoint': 'returned final state; no truth-selected checkpoint'}


def coordinates_and_reference(h, *, physics=PlanarContactConfig(), holdout=False):
    times = np.array(INVERSE_SPEC['field_times_s'], dtype=float)
    trace = dict(ramp=15., drop=7., flux=750.) if holdout else dict(ramp=10., drop=5., flux=500.)
    trajectory = PlanarContactModel(physics, nx=160, nz=40, contact_conductance=h).solve(times,
        initial_temperature=300., top_heat_flux=lambda t: trace['flux']*(-np.expm1(-t/trace['ramp'])),
        te_temperature=lambda t: 300.-trace['drop']*(-np.expm1(-t/trace['ramp'])), max_step=.0625)
    xx = (np.arange(40)+.5)*physics.width/40
    zz = (np.arange(20)+.5)*physics.thickness/20
    tt, z, x = np.meshgrid(times, zz, xx, indexing='ij')
    coordinates = np.column_stack([x.ravel(), z.ravel(), tt.ravel()])
    field = trajectory.sample(*coordinates.T)
    return coordinates, field, trajectory


def numerical_preflight(output_directory):
    """Frozen numerical validation before any new observations or inverse fits."""
    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    points = np.array([(x, .003, t) for t in range(0, 121, 10) for x in (.004, .020, .036)])
    for h in INVERSE_SPEC['evaluation_h']:
        for holdout in (False, True):
            times = np.asarray(INVERSE_SPEC['field_times_s'], dtype=float)
            tt, z, x = np.meshgrid(times, (np.arange(20)+.5)*.004/20,
                                  (np.arange(40)+.5)*.04/40, indexing='ij')
            coords = np.column_stack((x.ravel(), z.ravel(), tt.ravel()))
            trace = dict(ramp=15., drop=7., flux=750.) if holdout else dict(ramp=10., drop=5., flux=500.)
            trajectories = {}
            for label, nx, nz, step in (('canonical', 160, 40, .0625), ('time', 160, 40, .03125),
                                        ('finest', 320, 80, .03125), ('conventional', 120, 30, .0625)):
                trajectories[label] = PlanarContactModel(nx=nx, nz=nz, contact_conductance=h).solve(
                    INVERSE_SPEC['field_times_s'], initial_temperature=300., max_step=step,
                    top_heat_flux=lambda t: trace['flux']*(-np.expm1(-t/trace['ramp'])),
                    te_temperature=lambda t: 300.-trace['drop']*(-np.expm1(-t/trace['ramp'])))
            comparisons = {}
            for name, left, right in (('time_refinement', 'canonical', 'time'),
                                      ('space_refinement', 'time', 'finest'),
                                      ('combined_truth', 'canonical', 'finest'),
                                      ('conventional_vs_truth', 'conventional', 'canonical')):
                a, b = trajectories[left], trajectories[right]
                field = field_metrics(a.sample(*coords.T), b.sample(*coords.T))
                sensor = float(np.max(np.abs(a.sample(*points.T)-b.sample(*points.T))))
                comparisons[name] = dict(field, sensor_max_K=sensor,
                    passed=field['rmse_K'] <= .005 and field['maximum_error_K'] <= .01 and sensor <= .002)
            case = {'h': h, 'holdout_trace': holdout, 'comparisons': comparisons,
                    'passed': all(value['passed'] for value in comparisons.values())}
            cases.append(case)
            save_json(out/f"h-{h:g}-{'holdout' if holdout else 'training'}.json", case)
    result = {'passed': all(case['passed'] for case in cases), 'cases': cases,
              'data_or_inverse_fits_generated': False}
    save_json(out/'result.json', result)
    return result


def observations_from_reference(trajectory, noise_seed):
    points = np.array([(x, .003, t) for t in range(0, 121, 10)
                      for x in (.004, .020, .036) if not (x == .020 and t in (40, 50, 60))])
    clean = trajectory.sample(*points.T)
    return PlanarObservations(points, clean+np.random.default_rng(noise_seed).normal(0., .02, len(clean)), .02)


def field_metrics(prediction, truth):
    error = np.asarray(prediction)-np.asarray(truth)
    if error.shape != np.shape(truth) or not np.isfinite(error).all():
        raise ValueError('invalid scored field')
    return {'rmse_K': float(np.sqrt(np.mean(error**2))), 'maximum_error_K': float(np.max(np.abs(error)))}


def score_inverse(model, report, observations, coordinates, truth, true_h, *, physics=PlanarContactConfig()):
    prediction = predict_planar_revision(model, coordinates)
    field = field_metrics(prediction, truth)
    h = float(model.contact.detach())
    relative = abs(h/true_h-1.)
    reduction = 1.-abs(h-true_h)/abs(INVERSE_SPEC['initial_h']-true_h)
    observation_mse = float(np.mean(((predict_planar_revision(model, observations.coordinates)
                                    -observations.temperature)/observations.standard_deviation)**2))
    final = report.get('final_report', report['stages'][-1])
    physical = physical_gate(final)
    holdout_coords, holdout_truth, _ = coordinates_and_reference(true_h, physics=physics, holdout=True)
    _, _, transfer = coordinates_and_reference(h, physics=physics, holdout=True)
    holdout = field_metrics(transfer.sample(*holdout_coords.T), holdout_truth)
    gates = {'execution': report['status'] == 'completed', 'observations': observation_mse <= 2.,
             'physical': physical['passed'], 'field': field['rmse_K'] <= .05 and field['maximum_error_K'] <= .15,
             'parameter': relative <= .1 and reduction >= .5,
             'holdout': holdout['rmse_K'] <= .05 and holdout['maximum_error_K'] <= .15}
    return {'contact_estimate': h, 'relative_h_error': relative, 'initial_error_reduction': reduction,
            'observation_mse': observation_mse, 'field': field, 'physical': physical,
            'holdout_parameter_transfer': holdout, 'gates': gates, 'passed': all(gates.values())}


def run_inverse_development(output_directory, forward_selection, *, progress=None):
    # Import after this module has loaded: the coordinator also imports the
    # fitting primitives, but development must pass the same four-case gate.
    from .pinn_spatial_revision import verified_forward_selection
    forward_config = verified_forward_selection(forward_selection)
    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    source_paths = ('thermotwin/studies/pinn_spatial_revision_inverse.py',
                    'thermotwin/pinn/planar_contact_revision.py', 'thermotwin/physics/planar_contact.py',
                    'thermotwin/pinn/planar_contact_inverse.py')
    sources = {path: hashlib.sha256((root/path).read_bytes()).hexdigest() for path in source_paths}
    save_json(out/'plan-before-fitting.json', dict(INVERSE_SPEC, source_sha256=sources,
        forward_selection=forward_selection,
        supersedes='The unexecuted c07/c08 placeholders in the forward-development plan are delegated here.'))
    rows = []
    for strategy in INVERSE_SPEC['strategies']:
        for index, true_h in enumerate(INVERSE_SPEC['development_h']):
            directory = out/f'{strategy}-{index}'
            directory.mkdir()
            coords, truth, trajectory = coordinates_and_reference(true_h)
            observations = observations_from_reference(trajectory, semantic_seed('development', index, 'noise'))
            save_json(directory/'observations.json', observations.payload())
            try:
                model, report = fit_delayed_inverse(observations, config=forward_config, strategy=strategy,
                    seed_namespace=f'development/{index}', progress=progress,
                    completed_stage=lambda model, report, stage: retain_stage_evidence(directory, model, report, stage))
                save_json(directory/'fit-report.json', report)
                retain_checkpoint(directory/'final-checkpoint.pt', model)
                score = score_inverse(model, report, observations, coords, truth, true_h)
                row = {'strategy': strategy, 'development_h': true_h, 'report': report, 'score': score,
                       'passed': score['passed']}
            except Exception as error:
                row = {'strategy': strategy, 'development_h': true_h, 'passed': False,
                       'failure': type(error).__name__+': '+str(error)}
            save_json(directory/'result.json', row)
            rows.append(row)
            save_json(out/'partial-results.json', rows)
    passing = [strategy for strategy in INVERSE_SPEC['strategies']
               if all(row['passed'] for row in rows if row['strategy'] == strategy)]
    selected = passing[0] if passing else None
    result = {'status': 'passed' if selected else 'blocked_inverse_development_gate',
              'selected_strategy': selected, 'forward_config': asdict(forward_config) if isinstance(forward_config, PlanarRevisionConfig) else forward_config,
              'rows': rows, 'all_failures_retained': True, 'source_sha256': sources,
              'source_unchanged': all(hashlib.sha256((root/path).read_bytes()).hexdigest() == value
                                      for path, value in sources.items())}
    if not result['source_unchanged']:
        result['status'] = 'blocked_source_changed'
        result['selected_strategy'] = None
    save_json(out/'result.json', result)
    return result


def retain_checkpoint(path, model):
    """Persist the returned state before validation can fail or a new stage starts."""
    torch.save({'state_dict': model.state_dict(), 'config': asdict(model.config),
                'physics': asdict(model.physics), 'mode': model.mode}, path)


def retain_stage_evidence(directory, model, report, stage):
    directory = Path(directory)/f'completed-stage-{stage:02d}'
    directory.mkdir(exist_ok=False)
    save_json(directory/'report.json', report)
    retain_checkpoint(directory/'checkpoint.pt', model)
