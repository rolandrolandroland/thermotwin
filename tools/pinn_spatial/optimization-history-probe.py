"""Materialize the history comparison declared before the directional probe."""
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
from thermotwin.studies import pinn_spatial_graded as graded_study
from thermotwin.pinn.planar_contact_optimized import OptimizedConfig, OptimizedPINN, transactional_adam_step
from thermotwin.pinn.planar_contact_graded import graded_training_points
from thermotwin.pinn.planar_contact_continuity import objective_continuity, project_output
from thermotwin.pinn.planar_contact_inverse import PlanarObservations
from thermotwin.physics.planar_contact import PlanarContactConfig

OLD = ROOT/'reports/pinn-spatial-validation/graded-log-development'
A = ROOT/'reports/pinn-spatial-validation/optimization-directional-probe'
OUT = ROOT/'reports/pinn-spatial-validation/optimization-history-probe'


def read(path):
    return json.loads(Path(path).read_text())


def prepare():
    aplan, aresult = read(A/'plan.json'), read(A/'result.json')
    if aresult['status'] != 'completed' or aresult['selected_learning_rate'] is None or aresult['plan_sha256'] != common.digest_file(A/'plan.json'):
        raise ValueError('directional probe is not complete with a supported learning rate')
    specification = aplan['followup_history_probe']
    if specification['updates_per_case_per_arm'] != 8 or specification['arms'] != ['reset', 'persistent']:
        raise ValueError('previous history-probe declaration differs')
    sources = dict(aplan['source_sha256'])
    paths = ['thermotwin/pinn/planar_contact_optimized.py', 'tests/test_planar_contact_optimized.py',
        str(Path(__file__).resolve().relative_to(ROOT))]
    sources.update(common._source_records(paths))
    plan = {'schema': 'training_only_optimizer_history_probe_v1',
        'created_utc': common.datetime.now(common.timezone.utc).isoformat(),
        'directional_plan_sha256': common.digest_file(A/'plan.json'),
        'directional_result_sha256': common.digest_file(A/'result.json'),
        'source_sha256': sources, 'runtime': common.runtime_context(),
        'specification': specification, 'four_case_followup': aplan['four_case_followup'],
        'learning_rate': aresult['selected_learning_rate'], 'inputs': aplan['inputs'],
        'training_only': True, 'reference_fields_used': False,
        'independent_diagnostics_used_for_selection': False,
        'checkpoint_policy': 'each arm starts from a separate copy of the same saved final log-off checkpoint; no new scientific checkpoint written',
        'accepted_backtrack_state': 'commit the once-updated Adam moments and counter with accepted scaled parameter displacement; rollback all state on rejection',
        'automatic_retries': 0}
    OUT.mkdir(exist_ok=False)
    common._write(OUT/'plan.json', plan)
    common._write(OUT/'state.json', {'status': 'prepared', 'plan_sha256': common.digest_file(OUT/'plan.json')})
    return plan


def verify(plan):
    if common._source_records(plan['source_sha256']) != plan['source_sha256']:
        raise RuntimeError('history-probe source changed')
    if common.runtime_context() != plan['runtime']:
        raise RuntimeError('history-probe runtime changed')
    if common.digest_file(A/'plan.json') != plan['directional_plan_sha256'] or common.digest_file(A/'result.json') != plan['directional_result_sha256']:
        raise RuntimeError('directional probe changed')
    for records in plan['inputs'].values():
        for record in records.values():
            if common.digest_file(record['path']) != record['sha256']:
                raise RuntimeError('history-probe input changed')


def choose(rows):
    by_arm = {arm: [row for row in rows if row['arm'] == arm] for arm in ('reset', 'persistent')}
    good = {arm: len(records) == 2 and all(row['status'] == 'completed' and row['monotone']
        and math.isfinite(row['terminal_objective_ratio']) for row in records) for arm, records in by_arm.items()}
    if not good['reset']:
        return None
    if good['persistent']:
        reset = {row['case']: row['terminal_objective_ratio'] for row in by_arm['reset']}
        persistent = {row['case']: row['terminal_objective_ratio'] for row in by_arm['persistent']}
        no_worse = all(persistent[case] <= reset[case]*(1+1.e-8) for case in reset)
        meaningful = max(persistent.values()) <= max(reset.values())-1.e-4
        if no_worse and meaningful:
            return 'persistent'
    return 'reset'


def run():
    plan = read(OUT/'plan.json')
    sha = common.digest_file(OUT/'plan.json')
    if read(OUT/'state.json') != {'status': 'prepared', 'plan_sha256': sha}:
        raise ValueError('history probe already consumed or plan changed')
    verify(plan)
    descriptor = os.open(OUT/'run.lock', os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as handle:
        handle.write(str(os.getpid())+'\n')
    started = time.perf_counter()
    torch.set_num_threads(1)
    def alarm(signum, frame):
        if time.perf_counter()-started > plan['specification']['maximum_wall_seconds'] or common._rss_bytes() > plan['specification']['maximum_rss_bytes']:
            raise common.ResourceLimit('history-probe resource limit')
    previous = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, .5, .5)
    common._write(OUT/'state.json', {'status': 'running', 'plan_sha256': sha})
    rows = []
    try:
        for case in plan['specification']['cases']:
            records = plan['inputs'][case]
            saved = torch.load(records['final_checkpoint.pt']['path'], map_location='cpu', weights_only=False)
            raw = read(records['training-report.json']['path'])
            points = {name: np.asarray(value, dtype=np.float64) for name, value in raw['training_points'].items()}
            with np.load(records['observations.npz']['path']) as data:
                observations = PlanarObservations(data['coordinates'].copy(), data['temperature'].copy(), .02)
            for arm in plan['specification']['arms']:
                tick = time.perf_counter()
                configuration = dict(saved['config'], learning_rate=plan['learning_rate'],
                    adam_epochs=8, max_backtracks=4, optimizer_history=arm)
                config = OptimizedConfig(**configuration)
                model = OptimizedPINN(PlanarContactConfig(**saved['physics']), config, known_h=saved['known_h'])
                model.load_state_dict(saved['state_dict'])
                original = {name: value.detach().clone() for name, value in model.state_dict().items()}
                expected_points = graded_training_points(model.physics, config, common.SEEDS['sampling'])
                if common.point_digest(points) != common.point_digest(expected_points):
                    raise ValueError('history probe changed the training grid')
                baseline_projection = project_output(model, points, observations)
                baseline = float(objective_continuity(model, points, observations)[0].detach())
                if not math.isclose(baseline, raw['final_objective'], rel_tol=1.e-8, abs_tol=1.e-9):
                    raise ValueError('history probe failed to reproduce the original training objective')
                optimizer = torch.optim.Adam(model.networks.parameters(), lr=config.learning_rate)
                history = []
                for step in range(8):
                    if step and arm == 'reset':
                        optimizer = torch.optim.Adam(model.networks.parameters(), lr=config.learning_rate)
                    event = transactional_adam_step(model, points, observations, optimizer)
                    event['step'] = step+1
                    history.append(event)
                    with (OUT/'progress.jsonl').open('a') as handle:
                        handle.write(json.dumps(dict(case=case, arm=arm, **event), allow_nan=False)+'\n')
                    print(json.dumps({'case': case, 'arm': arm, 'step': step+1,
                        'objective': event['objective_after'], 'accepted': event['accepted'],
                        'backtracks': event['backtracks']}), flush=True)
                total, obs, physics, components = objective_continuity(model, points, observations)
                terminal = float(total.detach())
                row = {'case': case, 'arm': arm, 'status': 'completed', 'config': configuration,
                    'baseline_objective': baseline, 'baseline_projection': baseline_projection,
                    'terminal_objective': terminal, 'terminal_objective_ratio': terminal/baseline,
                    'training_family_mse': {name: float(value.detach()) for name, value in components.items()},
                    'history': history, 'wall_seconds': time.perf_counter()-tick,
                    'monotone': all(event['objective_after'] <= event['objective_before'] for event in history)}
                model.load_state_dict(original)
                row['copied_weights_restored_exactly'] = all(torch.equal(value, original[name]) for name, value in model.state_dict().items())
                row['new_checkpoints_written'] = 0
                rows.append(row)
                common._write(OUT/(case+'-'+arm+'.json'), row)
        selected = choose(rows)
        verify(plan)
        graded_study.verify_comparison_plan(read(OLD/'plan.json'), OLD)
        result = {'status': 'completed', 'plan_sha256': sha, 'training_only': True,
            'reference_fields_used': False, 'independent_diagnostics_used_for_selection': False,
            'arms': rows, 'selected_optimizer_history': selected,
            'wall_seconds': time.perf_counter()-started, 'peak_rss_bytes': common._rss_bytes(),
            'input_and_source_preserved': True, 'new_scientific_checkpoints_written': 0}
        common._write(OUT/'result.json', result)
        if selected is not None:
            baseline_config = dict(torch.load(plan['inputs']['t02']['final_checkpoint.pt']['path'], map_location='cpu', weights_only=False)['config'])
            baseline_config.update(adam_epochs=plan['four_case_followup']['fixed_updates'],
                learning_rate=plan['learning_rate'], max_backtracks=plan['four_case_followup']['maximum_backtracks'],
                optimizer_history=selected)
            selected_config = asdict(OptimizedConfig(**baseline_config))
            evidence_paths = [A/'plan.json', A/'result.json', OUT/'plan.json', OUT/'result.json']
            decision = {'status': 'approved_for_forward_development', 'training_only': True,
                'reference_fields_used': False, 'independent_diagnostics_used_for_selection': False,
                'selected_config': selected_config,
                'probe_evidence': [{'path': str(path), 'sha256': common.digest_file(path)} for path in evidence_paths],
                'start_policy': plan['four_case_followup']['start_policy'],
                'maximum_wall_seconds': plan['four_case_followup']['maximum_wall_seconds'],
                'maximum_rss_bytes': plan['four_case_followup']['maximum_rss_bytes'],
                'selection_rule': plan['specification']['selection_rule'],
                'no_guarantee_of_forward_gate_success': True}
            common._write(OUT/'decision.json', decision)
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
    result = prepare() if args.command == 'prepare' else run()
    print(json.dumps({'command': args.command, 'status': result.get('status', 'prepared'),
        'selected_optimizer_history': result.get('selected_optimizer_history')}, allow_nan=False), flush=True)
