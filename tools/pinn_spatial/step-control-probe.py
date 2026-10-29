"""One frozen training-only recovery probe; no fields or acceptance diagnostics."""
from dataclasses import asdict
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
from thermotwin.pinn import planar_contact_step_control as api
from thermotwin.pinn.planar_contact_continuity import objective_continuity
from thermotwin.pinn.planar_contact_graded import graded_training_points
from thermotwin.pinn.planar_contact_inverse import PlanarObservations
from thermotwin.physics.planar_contact import PlanarContactConfig

BASE = ROOT/'reports/pinn-spatial-validation'
OUT = BASE/'step-control-recovery-probe'
CASES = (
    {'id': 'stalled_full_800', 'directory': 'optimized-v6-development/t01', 'attempts': 8, 'restore_moments': True},
    {'id': 'partial_800', 'directory': 'graded-log-development/log_off/t02', 'attempts': 16, 'restore_moments': False},
    {'id': 'partial_1400', 'directory': 'graded-log-development/log_off/t04', 'attempts': 16, 'restore_moments': False},
)


def read(path):
    return json.loads(Path(path).read_text())


def prepare():
    v6 = read(BASE/'optimized-v6-development/plan.json')
    config = asdict(api.StepControlConfig(adam_epochs=320, learning_rate=.0064, max_backtracks=4,
        bottom_log_feature=False, optimizer_history='persistent', maximum_trial_solves=480))
    sources = dict(v6['source_sha256'])
    sources.update(common._source_records([
        'thermotwin/pinn/planar_contact_step_control.py', 'tests/test_planar_contact_step_control.py',
        str(Path(__file__).resolve().relative_to(ROOT))]))
    inputs = {case['id']: {name: {'path': str(BASE/case['directory']/name),
        'sha256': common.digest_file(BASE/case['directory']/name)}
        for name in ('final_checkpoint.pt', 'training-report.json', 'observations.npz')} for case in CASES}
    plan = {'schema': 'training_only_step_control_recovery_probe_v1',
        'created_utc': common.datetime.now(common.timezone.utc).isoformat(),
        'source_sha256': sources, 'runtime': common.runtime_context(),
        'cases': list(CASES), 'inputs': inputs, 'selected_config_if_supported': config,
        'stalled_attempt_plan_sha256': common.digest_file(BASE/'optimized-v6-development/plan.json'),
        'stalled_attempt_index_sha256': common.digest_file(BASE/'optimized-v6-development/artifact-index.json'),
        'training_only': True, 'reference_fields_used': False,
        'independent_diagnostics_used_for_selection': False,
        'observation_keys_read': ['coordinates', 'temperature'], 'noise_standard_deviation_K': .02,
        'maximum_controller_attempts': 40, 'maximum_trial_readout_solves': 200,
        'maximum_wall_seconds': 1800., 'maximum_rss_bytes': 2*1024**3, 'threads': 1,
        'automatic_retries': 0, 'new_checkpoints_written': 0,
        'approval_rule': 'all three finite monotone training trajectories strictly reduce final objective and accept at least one step; stalled full case must also record a post-rollback controller change and a later accepted step',
        'selection_rule': 'one declared controller only; no alternative rates, seeds, architectures, loss weights, grids, cutoffs, or acceptance outcomes',
        'followup': {'cases': list(common.CASES), 'start_policy': 'all four original fresh matched log-off initializations',
            'maximum_attempts_per_case': 320, 'maximum_trial_solves_per_case': 480,
            'maximum_case_wall_seconds': 3000., 'maximum_case_rss_bytes': 2*1024**3,
            'parallel_workers': 2, 'worker_threads': 1, 'aggregate_worker_rss_bytes': 4*1024**3,
            'waves': [['t01', 't02'], ['t03', 't04']], 'maximum_supervisor_wall_seconds': 7200.,
            'automatic_retries': 0, 'acceptance_thresholds': common.GATES,
            'checkpoint_policy': 'controller-returned final state retained before unchanged scoring, including declared early stopping'}}
    OUT.mkdir(exist_ok=False)
    common._write(OUT/'plan.json', plan)
    common._write(OUT/'state.json', {'status': 'prepared', 'plan_sha256': common.digest_file(OUT/'plan.json')})
    return plan


def verify(plan):
    if common._source_records(plan['source_sha256']) != plan['source_sha256']:
        raise RuntimeError('recovery-probe source changed')
    if common.runtime_context() != plan['runtime']:
        raise RuntimeError('recovery-probe runtime changed')
    for records in plan['inputs'].values():
        for record in records.values():
            if common.digest_file(record['path']) != record['sha256']:
                raise RuntimeError('recovery-probe input changed')
    for filename, key in (('plan.json', 'stalled_attempt_plan_sha256'), ('artifact-index.json', 'stalled_attempt_index_sha256')):
        if common.digest_file(BASE/'optimized-v6-development'/filename) != plan[key]:
            raise RuntimeError('preserved V6 attempt changed')


def run():
    plan = read(OUT/'plan.json')
    sha = common.digest_file(OUT/'plan.json')
    if read(OUT/'state.json') != {'status': 'prepared', 'plan_sha256': sha}:
        raise RuntimeError('recovery probe already consumed or changed')
    verify(plan)
    descriptor = os.open(OUT/'run.lock', os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as handle:
        handle.write(str(os.getpid())+'\n')
    torch.set_num_threads(1)
    started = time.perf_counter()
    def alarm(signum, frame):
        if time.perf_counter()-started > plan['maximum_wall_seconds'] or common._rss_bytes() > plan['maximum_rss_bytes']:
            raise common.ResourceLimit('recovery-probe resource limit')
    previous = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, .5, .5)
    common._write(OUT/'state.json', {'status': 'running', 'plan_sha256': sha})
    rows = []
    try:
        for case in CASES:
            tick = time.perf_counter()
            records = plan['inputs'][case['id']]
            saved = torch.load(records['final_checkpoint.pt']['path'], map_location='cpu', weights_only=False)
            raw = read(records['training-report.json']['path'])
            config = api.StepControlConfig(**plan['selected_config_if_supported'])
            model = api.StepControlPINN(PlanarContactConfig(**saved['physics']), config, known_h=saved['known_h'])
            model.load_state_dict(saved['state_dict'])
            original = {name: value.detach().clone() for name, value in model.state_dict().items()}
            points = {name: np.asarray(value, dtype=np.float64) for name, value in raw['training_points'].items()}
            if common.point_digest(points) != common.point_digest(graded_training_points(model.physics, config, common.SEEDS['sampling'])):
                raise RuntimeError('recovery probe changed the graded training points')
            with np.load(records['observations.npz']['path']) as data:
                observations = PlanarObservations(data['coordinates'].copy(), data['temperature'].copy(), .02)
            baseline = float(objective_continuity(model, points, observations)[0].detach())
            if not math.isclose(baseline, raw['final_objective'], rel_tol=1.e-8, abs_tol=1.e-9):
                raise RuntimeError('recovery probe did not reproduce saved training objective')
            optimizer = torch.optim.Adam(model.networks.parameters(), lr=.0064)
            if case['restore_moments']:
                optimizer.load_state_dict(saved['optimizer_state'])
            state = api.StepControlState(learning_rate=float(optimizer.param_groups[0]['lr']))
            history = []
            for attempt in range(case['attempts']):
                if state.stopped:
                    break
                event = api.controller_step(model, points, observations, optimizer, state)
                event['probe_attempt'] = attempt+1
                history.append(event)
                with (OUT/'progress.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(case=case['id'], **event), allow_nan=False)+'\n')
                print(json.dumps({'case': case['id'], 'attempt': attempt+1, 'accepted': event['accepted'],
                    'objective': event['objective_after'], 'next_learning_rate': state.learning_rate,
                    'controller_stopped': state.stopped}), flush=True)
            terminal = float(objective_continuity(model, points, observations)[0].detach())
            changed_at = [index for index, event in enumerate(history) if not event['accepted']
                and event.get('controller_after') != event.get('controller_before')]
            recovered = any(event['accepted'] for index, event in enumerate(history)
                if any(previous < index for previous in changed_at))
            row = {'case': case['id'], 'status': 'completed', 'baseline_objective': baseline,
                'terminal_objective': terminal, 'terminal_objective_ratio': terminal/baseline,
                'moment_state_loaded_from_v6': case['restore_moments'],
                'attempts': len(history), 'accepted_updates': sum(event['accepted'] for event in history),
                'monotone': all(math.isfinite(event['objective_after']) and event['objective_after'] <= event['objective_before'] for event in history),
                'controller_change_followed_by_acceptance': recovered,
                'controller_final': asdict(state), 'history': history,
                'wall_seconds': time.perf_counter()-tick, 'new_checkpoints_written': 0}
            model.load_state_dict(original)
            row['copied_weights_restored_exactly'] = all(torch.equal(value, original[name]) for name, value in model.state_dict().items())
            rows.append(row)
            common._write(OUT/(case['id']+'.json'), row)
        approved = all(row['status'] == 'completed' and row['monotone'] and row['accepted_updates'] > 0
            and row['terminal_objective'] < row['baseline_objective'] and row['copied_weights_restored_exactly'] for row in rows)
        approved = approved and rows[0]['controller_change_followed_by_acceptance']
        verify(plan)
        result = {'status': 'completed', 'plan_sha256': sha, 'training_only': True,
            'reference_fields_used': False, 'independent_diagnostics_used_for_selection': False,
            'cases': rows, 'approved_for_forward_development': approved,
            'wall_seconds': time.perf_counter()-started, 'peak_rss_bytes': common._rss_bytes(),
            'inputs_and_sources_preserved': True, 'original_checkpoints_untouched': True}
        common._write(OUT/'result.json', result)
        if approved:
            evidence = [{'path': str(OUT/name), 'sha256': common.digest_file(OUT/name)} for name in ('plan.json', 'result.json')]
            common._write(OUT/'decision.json', {'status': 'approved_for_forward_development',
                'training_only': True, 'reference_fields_used': False,
                'independent_diagnostics_used_for_selection': False,
                'selected_config': plan['selected_config_if_supported'], 'probe_evidence': evidence,
                'followup': plan['followup'], 'no_guarantee_of_forward_gate_success': True})
        common._write(OUT/'state.json', {'status': 'completed', 'plan_sha256': sha})
        return result
    except BaseException as error:
        common._write(OUT/'state.json', {'status': 'failed', 'plan_sha256': sha, 'error': repr(error)})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, previous)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run'))
    args = parser.parse_args()
    value = prepare() if args.command == 'prepare' else run()
    print(json.dumps({'command': args.command, 'status': value.get('status', 'prepared'),
        'approved': value.get('approved_for_forward_development')}, allow_nan=False), flush=True)
