"""Final bounded training mechanics check; recovery and validity are separate."""
from dataclasses import asdict
from pathlib import Path
import argparse
import importlib.util
import json
import math
import os
import signal
import time

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('step_control_probe_c', Path(__file__).with_name('step-control-probe.py'))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
common, api, torch, np = C.common, C.api, C.torch, C.np
BASE, PREVIOUS = C.BASE, C.OUT
OUT = BASE/'step-control-terminal-probe'


def prepare():
    cplan, cresult = C.read(PREVIOUS/'plan.json'), C.read(PREVIOUS/'result.json')
    if C.read(PREVIOUS/'state.json') != {'status': 'completed', 'plan_sha256': common.digest_file(PREVIOUS/'plan.json')}:
        raise RuntimeError('Probe C must be terminal and retained')
    C.verify(cplan)
    if cresult['approved_for_forward_development'] is not False:
        raise RuntimeError('this successor explicitly follows the failed strict C criterion')
    partials = [row for row in cresult['cases'] if row['case'].startswith('partial_')]
    if len(partials) != 2 or any(row['attempts'] != 16 or not row['copied_weights_restored_exactly']
        or not row['monotone'] or not all(event['accepted'] for event in row['history']) for row in partials):
        raise RuntimeError('the declared partial-prefix execution equivalence is unavailable')
    config = dict(cplan['selected_config_if_supported'], controller_max_failures=8)
    config = asdict(api.StepControlConfig(**config))
    differences = {key for key in config if config[key] != cplan['selected_config_if_supported'][key]}
    if differences != {'controller_max_failures'}:
        raise RuntimeError('terminal probe may change only the rejection limit from three to eight')
    sources = dict(cplan['source_sha256'])
    sources.update(common._source_records([str(Path(__file__).resolve().relative_to(ROOT))]))
    plan = {'schema': 'terminal_training_only_step_control_mechanics_probe_v1',
        'created_utc': common.datetime.now(common.timezone.utc).isoformat(),
        'source_sha256': sources, 'runtime': common.runtime_context(),
        'previous_plan_sha256': common.digest_file(PREVIOUS/'plan.json'),
        'previous_result_sha256': common.digest_file(PREVIOUS/'result.json'),
        'previous_strict_recovery_criterion_passed': False,
        'selected_config': config, 'full_case_inputs': cplan['inputs']['stalled_full_800'],
        'maximum_full_attempts': 8, 'maximum_trial_readout_solves': 40,
        'maximum_wall_seconds': 600., 'maximum_rss_bytes': 2*1024**3, 'threads': 1,
        'training_only': True, 'reference_fields_used': False,
        'independent_diagnostics_used_for_selection': False,
        'partial_prefixes': [{'case': row['case'], 'source': str(PREVIOUS/(row['case']+'.json')),
            'sha256': common.digest_file(PREVIOUS/(row['case']+'.json')), 'attempts': 16,
            'execution_equivalence': 'only max-failure threshold differs; every saved attempt accepted so rejection branch and threshold were never consulted; same sources, observations, grid, initial checkpoint and fresh moments'} for row in partials],
        'mechanics_gate': 'finite monotone objective, exact copied-state restoration, full rollback before each controller change, and changed next rate or declared stop after rejection',
        'recovery_classification': 'report yes/no separately; unsuccessful bounded proposals do not establish stationarity',
        'terminal_decision': 'if mechanics valid freeze this one max8 controller uniformly for one all-four development comparison regardless recovery classification; no further optimizer probes or outcome-driven extensions',
        'followup': cplan['followup'], 'automatic_retries': 0, 'new_scientific_checkpoints': 0}
    OUT.mkdir(exist_ok=False)
    common._write(OUT/'plan.json', plan)
    common._write(OUT/'state.json', {'status': 'prepared', 'plan_sha256': common.digest_file(OUT/'plan.json')})
    return plan


def verify(plan):
    if common._source_records(plan['source_sha256']) != plan['source_sha256'] or common.runtime_context() != plan['runtime']:
        raise RuntimeError('terminal probe source/runtime changed')
    for path, key in ((PREVIOUS/'plan.json', 'previous_plan_sha256'), (PREVIOUS/'result.json', 'previous_result_sha256')):
        if common.digest_file(path) != plan[key]:
            raise RuntimeError('Probe C changed')
    C.verify(C.read(PREVIOUS/'plan.json'))
    for record in plan['partial_prefixes']:
        if common.digest_file(record['source']) != record['sha256']:
            raise RuntimeError('bound partial prefix changed')


def run():
    plan = C.read(OUT/'plan.json')
    sha = common.digest_file(OUT/'plan.json')
    if C.read(OUT/'state.json') != {'status': 'prepared', 'plan_sha256': sha}:
        raise RuntimeError('terminal probe already consumed or changed')
    verify(plan)
    descriptor = os.open(OUT/'run.lock', os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as handle:
        handle.write(str(os.getpid())+'\n')
    torch.set_num_threads(1)
    started = time.perf_counter()
    def alarm(signum, frame):
        if time.perf_counter()-started > plan['maximum_wall_seconds'] or common._rss_bytes() > plan['maximum_rss_bytes']:
            raise common.ResourceLimit('terminal probe resource limit')
    previous = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, .5, .5)
    common._write(OUT/'state.json', {'status': 'running', 'plan_sha256': sha})
    try:
        records = plan['full_case_inputs']
        saved = torch.load(records['final_checkpoint.pt']['path'], map_location='cpu', weights_only=False)
        raw = C.read(records['training-report.json']['path'])
        config = api.StepControlConfig(**plan['selected_config'])
        model = api.StepControlPINN(C.PlanarContactConfig(**saved['physics']), config, known_h=saved['known_h'])
        model.load_state_dict(saved['state_dict'])
        original = {name: value.detach().clone() for name, value in model.state_dict().items()}
        points = {name: np.asarray(value, dtype=np.float64) for name, value in raw['training_points'].items()}
        with np.load(records['observations.npz']['path']) as data:
            observations = C.PlanarObservations(data['coordinates'].copy(), data['temperature'].copy(), .02)
        baseline = float(C.objective_continuity(model, points, observations)[0].detach())
        if not math.isclose(baseline, raw['final_objective'], rel_tol=1.e-8, abs_tol=1.e-9):
            raise RuntimeError('stalled training objective not reproduced')
        optimizer = torch.optim.Adam(model.networks.parameters(), lr=config.learning_rate)
        optimizer.load_state_dict(saved['optimizer_state'])
        state = api.StepControlState(learning_rate=float(optimizer.param_groups[0]['lr']))
        history = []
        for attempt in range(8):
            if state.stopped:
                break
            event = api.controller_step(model, points, observations, optimizer, state)
            history.append(event)
            common._write(OUT/'partial-history.json', history)
            print(json.dumps({'attempt': attempt+1, 'accepted': event['accepted'],
                'objective': event['objective_after'], 'next_learning_rate': state.learning_rate,
                'stopped': state.stopped}), flush=True)
        terminal = float(C.objective_continuity(model, points, observations)[0].detach())
        model.load_state_dict(original)
        restored = all(torch.equal(value, original[name]) for name, value in model.state_dict().items())
        mechanics = restored and all(math.isfinite(event['objective_after'])
            and event['objective_after'] <= event['objective_before']
            and event['identical_rejected_next_attempt_precluded']
            and (event['accepted'] or event['optimizer_state_rolled_back']) for event in history)
        recovery = any(event['accepted'] for event in history) and terminal < baseline
        verify(plan)
        result = {'status': 'completed', 'plan_sha256': sha, 'training_only': True,
            'reference_fields_used': False, 'independent_diagnostics_used_for_selection': False,
            'previous_probe_C_strict_recovery_passed': False,
            'mechanics_valid': mechanics, 'scientific_recovery_supported': recovery,
            'full_case': {'baseline_objective': baseline, 'terminal_objective': terminal,
                'attempts': len(history), 'history': history, 'controller_final': asdict(state),
                'copied_weights_restored_exactly': restored},
            'partial_prefixes_reused': plan['partial_prefixes'],
            'no_further_optimizer_search': True, 'wall_seconds': time.perf_counter()-started,
            'peak_rss_bytes': common._rss_bytes(), 'new_checkpoints_written': 0}
        common._write(OUT/'result.json', result)
        if mechanics:
            paths = [PREVIOUS/'plan.json', PREVIOUS/'result.json', OUT/'plan.json', OUT/'result.json']
            decision = {'status': 'approved_for_forward_development', 'training_only': True,
                'reference_fields_used': False, 'independent_diagnostics_used_for_selection': False,
                'selected_config': plan['selected_config'], 'followup': plan['followup'],
                'probe_evidence': [{'path': str(path), 'sha256': common.digest_file(path)} for path in paths],
                'previous_probe_C_strict_recovery_passed': False, 'mechanics_valid': True,
                'scientific_recovery_supported': recovery, 'development_comparison_only': True,
                'no_further_optimizer_search': True, 'no_guarantee_of_forward_gate_success': True,
                'decision_basis': plan['terminal_decision']}
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
        'mechanics_valid': result.get('mechanics_valid'), 'recovery': result.get('scientific_recovery_supported')}, allow_nan=False), flush=True)
