"""Bounded, training-only re-solved objective probe on copied development states.

No independent diagnostic, reference field, or scientific checkpoint is read or
written. This script and its declaration are sealed before the first trial.
"""
from pathlib import Path
import argparse
import json
import math
import os
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from thermotwin.studies import pinn_spatial_continuity as common
from thermotwin.studies import pinn_spatial_graded as graded_study
from thermotwin.pinn.planar_contact_graded import GradedConfig, GradedPINN, graded_training_points
from thermotwin.pinn.planar_contact_continuity import objective_continuity, project_output
from thermotwin.pinn.planar_contact_inverse import PlanarObservations
from thermotwin.physics.planar_contact import PlanarContactConfig

OLD = ROOT/'reports/pinn-spatial-validation/graded-log-development'
OUT = ROOT/'reports/pinn-spatial-validation/optimization-directional-probe'
MULTIPLIERS = (1., 4., 16., 64.)
EPSILONS = (1.e-5, 1.e-6)
CASES = ('t02', 't04')


def read(path):
    return json.loads(Path(path).read_text())


def prepare():
    old_plan = read(OLD/'plan.json')
    graded_study.verify_comparison_plan(old_plan, OLD)
    sources = dict(old_plan['source_sha256'])
    sources[str(Path(__file__).resolve().relative_to(ROOT))] = common.digest_file(__file__)
    inputs = {case: {name: {'path': str(OLD/'log_off'/case/name),
        'sha256': common.digest_file(OLD/'log_off'/case/name)}
        for name in ('final_checkpoint.pt', 'training-report.json', 'observations.npz')}
        for case in CASES}
    plan = {'schema': 'training_only_optimization_direction_probe_v1',
        'created_utc': common.datetime.now(common.timezone.utc).isoformat(),
        'source_sha256': sources, 'runtime': common.runtime_context(),
        'cases': list(CASES), 'inputs': inputs,
        'old_comparison_plan_sha256': common.digest_file(OLD/'plan.json'),
        'old_comparison_artifact_index_sha256': common.digest_file(OLD/'artifact-index.json'),
        'training_only': True, 'reference_fields_used': False,
        'independent_diagnostics_used_for_selection': False,
        'fresh_adam_base_learning_rate': .0001, 'gradient_clip': 1.,
        'multipliers': list(MULTIPLIERS), 'fd_epsilons_Linf': list(EPSILONS),
        'maximum_readout_solves_per_case': 9, 'maximum_wall_seconds': 900.,
        'maximum_rss_bytes': 2*1024**3, 'threads': 1, 'automatic_retries': 0,
        'checkpoint_policy': 'copies only; every trial starts at same saved final state; no new checkpoint retained',
        'selection_rule': 'eligible only if finite strict complete-objective decrease on both cases; minimize worst-case objective ratio, ties prefer smaller multiplier',
        'finite_difference_role': 'diagnostic only; compare fixed-head proposal derivative to re-solved objective, no exact profiled-gradient assertion',
        'followup_history_probe': {'arms': ['reset', 'persistent'], 'updates_per_case_per_arm': 8,
            'cases': list(CASES), 'maximum_backtracks': 4,
            'maximum_trial_readout_solves': 160, 'additional_baseline_readout_solves': 4,
            'maximum_wall_seconds': 1800., 'maximum_rss_bytes': 2*1024**3,
            'learning_rate': 'materialize the common directional-probe selection only',
            'selection_rule': 'persistent only if finite monotone trajectories, no worse on both cases within relative 1e-8, and worst terminal objective ratio improves by at least 1e-4; otherwise reset; reject followup if neither is finite monotone on both'},
        'four_case_followup': {'start_policy': 'fresh matched log-off initialization in every case; probe checkpoint states are not reused',
            'fixed_updates': 320, 'maximum_backtracks': 4, 'maximum_wall_seconds': 14400.,
            'maximum_rss_bytes': 2*1024**3, 'threads': 1, 'automatic_retries': 0,
            'architecture_grid_loss_cutoff_and_gates': 'unchanged log-off baseline',
            'checkpoint_policy': 'final state only, retained before independent scoring'},
        'failure_rule': 'retain every failed trial; no new case, seed, cutoff, grid, loss, gate, or retry'}
    OUT.mkdir(exist_ok=False)
    common._write(OUT/'plan.json', plan)
    common._write(OUT/'state.json', {'status': 'prepared', 'plan_sha256': common.digest_file(OUT/'plan.json')})
    return plan


def verify(plan):
    if common._source_records(plan['source_sha256']) != plan['source_sha256']:
        raise RuntimeError('probe numerical source changed')
    if common.runtime_context() != plan['runtime']:
        raise RuntimeError('probe runtime changed')
    if common.digest_file(OLD/'plan.json') != plan['old_comparison_plan_sha256']:
        raise RuntimeError('old comparison plan changed')
    if common.digest_file(OLD/'artifact-index.json') != plan['old_comparison_artifact_index_sha256']:
        raise RuntimeError('old comparison index changed')
    for records in plan['inputs'].values():
        for record in records.values():
            if common.digest_file(record['path']) != record['sha256']:
                raise RuntimeError('saved probe input changed')
    if plan['cases'] != list(CASES) or plan['multipliers'] != list(MULTIPLIERS) or plan['fd_epsilons_Linf'] != list(EPSILONS):
        raise RuntimeError('probe design changed')


def run():
    plan = read(OUT/'plan.json')
    state = read(OUT/'state.json')
    plan_sha = common.digest_file(OUT/'plan.json')
    if state != {'status': 'prepared', 'plan_sha256': plan_sha}:
        raise RuntimeError('probe already consumed or changed')
    verify(plan)
    fd = os.open(OUT/'run.lock', os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as handle:
        handle.write(str(os.getpid())+'\n')
    started = time.perf_counter()
    torch.set_num_threads(1)
    def alarm(signum, frame):
        if time.perf_counter()-started > plan['maximum_wall_seconds'] or common._rss_bytes() > plan['maximum_rss_bytes']:
            raise common.ResourceLimit('probe resource limit')
    previous = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, .5, .5)
    common._write(OUT/'state.json', {'status': 'running', 'plan_sha256': plan_sha})
    rows = []
    try:
        for case in CASES:
            records = plan['inputs'][case]
            saved = torch.load(records['final_checkpoint.pt']['path'], map_location='cpu', weights_only=False)
            config = GradedConfig(**saved['config'])
            if config.bottom_log_feature or config.learning_rate != .0001:
                raise ValueError('probe must use unchanged log-off baseline')
            model = GradedPINN(PlanarContactConfig(**saved['physics']), config, known_h=saved['known_h'])
            model.load_state_dict(saved['state_dict'])
            original = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
            raw = read(records['training-report.json']['path'])
            points = {name: np.asarray(value, dtype=np.float64) for name, value in raw['training_points'].items()}
            expected_points = graded_training_points(model.physics, config, common.SEEDS['sampling'])
            if common.point_digest(points) != common.point_digest(expected_points):
                raise ValueError('saved training points differ from frozen graded construction')
            with np.load(records['observations.npz']['path']) as data:
                observations = PlanarObservations(data['coordinates'].copy(), data['temperature'].copy(), .02)
            baseline_projection = project_output(model, points, observations)
            baseline = float(objective_continuity(model, points, observations)[0].detach())
            if not math.isclose(baseline, raw['final_objective'], rel_tol=1.e-8, abs_tol=1.e-9):
                raise ValueError('saved training objective not reproduced')
            base = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
            parameters = list(model.networks.parameters())
            optimizer = torch.optim.Adam(parameters, lr=.0001)
            optimizer.zero_grad()
            loss = objective_continuity(model, points, observations)[0]
            loss.backward()
            gradient = [parameter.grad.detach().clone() for parameter in parameters]
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.))
            before = [parameter.detach().clone() for parameter in parameters]
            optimizer.step()
            delta = [parameter.detach()-value for parameter, value in zip(parameters, before)]
            maximum = max(float(torch.max(abs(value))) for value in delta)
            direction = [value/maximum for value in delta]
            fixed_derivative = sum(float(torch.sum(g*d)) for g, d in zip(gradient, direction))
            def trial(difference):
                model.load_state_dict(base)
                with torch.no_grad():
                    for parameter, value in zip(parameters, difference):
                        parameter.add_(value)
                tick = time.perf_counter()
                result = {}
                try:
                    projection = project_output(model, points, observations)
                    total, obs, physical, components = objective_continuity(model, points, observations)
                    result = {'status': 'completed', 'objective': float(total.detach()),
                        'objective_ratio': float(total.detach())/baseline,
                        'training_family_mse': {name: float(value.detach()) for name, value in components.items()},
                        'projection': projection}
                    if not math.isfinite(result['objective']):
                        raise FloatingPointError('nonfinite re-solved objective')
                except common.ResourceLimit:
                    raise
                except Exception as error:
                    result = {'status': 'failed', 'objective': None, 'error': repr(error)}
                finally:
                    model.load_state_dict(base)
                result['wall_seconds'] = time.perf_counter()-tick
                return result
            finite_differences = []
            for epsilon in EPSILONS:
                plus = trial([epsilon*d for d in direction])
                minus = trial([-epsilon*d for d in direction])
                fdvalue = ((plus['objective']-minus['objective'])/(2*epsilon)
                    if plus['status'] == minus['status'] == 'completed' else None)
                finite_differences.append({'epsilon_Linf': epsilon, 'plus': plus, 'minus': minus,
                    'profiled_directional_derivative': fdvalue,
                    'fixed_head_directional_derivative': fixed_derivative})
            multiplier_rows = []
            for multiplier in MULTIPLIERS:
                result = trial([multiplier*d for d in delta])
                result.update(multiplier=multiplier, learning_rate=multiplier*.0001)
                multiplier_rows.append(result)
                print(json.dumps({'case': case, 'multiplier': multiplier,
                    'objective': result['objective'], 'baseline': baseline}), flush=True)
            model.load_state_dict(original)
            restored = all(torch.equal(value, original[name]) for name, value in model.state_dict().items())
            rows.append({'case': case, 'baseline_objective': baseline,
                'baseline_projection': baseline_projection, 'gradient_norm_before_clip': norm,
                'fresh_adam_max_parameter_step': maximum, 'finite_differences': finite_differences,
                'step_trials': multiplier_rows, 'copied_weights_restored_exactly': restored,
                'new_checkpoints_written': 0})
            common._write(OUT/(case+'.json'), rows[-1])
        eligible = []
        for index, multiplier in enumerate(MULTIPLIERS):
            paired = [row['step_trials'][index] for row in rows]
            if all(row['status'] == 'completed' and row['objective_ratio'] < 1. for row in paired):
                eligible.append((max(row['objective_ratio'] for row in paired), multiplier))
        chosen = min(eligible) if eligible else None
        verify(plan)
        graded_study.verify_comparison_plan(read(OLD/'plan.json'), OLD)
        result = {'status': 'completed', 'plan_sha256': plan_sha, 'training_only': True,
            'reference_fields_used': False, 'independent_diagnostics_used_for_selection': False,
            'cases': rows, 'selection_rule': plan['selection_rule'],
            'selected_multiplier': chosen[1] if chosen else None,
            'selected_learning_rate': chosen[1]*.0001 if chosen else None,
            'selected_worst_objective_ratio': chosen[0] if chosen else None,
            'wall_seconds': time.perf_counter()-started, 'peak_rss_bytes': common._rss_bytes(),
            'input_and_source_preserved': True, 'scientific_fits_started': 0}
        common._write(OUT/'result.json', result)
        common._write(OUT/'state.json', {'status': 'completed', 'plan_sha256': plan_sha})
        return result
    except BaseException as error:
        common._write(OUT/'state.json', {'status': 'failed', 'plan_sha256': plan_sha, 'error': repr(error)})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, previous)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run'))
    args = parser.parse_args()
    result = prepare() if args.command == 'prepare' else run()
    print(json.dumps({'command': args.command, 'status': result.get('status', 'prepared'),
        'selected_learning_rate': result.get('selected_learning_rate')}, allow_nan=False), flush=True)
