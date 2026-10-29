"""Once-only two-wave execution of the final bounded development comparison."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from thermotwin.studies import pinn_spatial_continuity as common
from thermotwin.studies import pinn_spatial_step_control as study

BASE = ROOT/'reports/pinn-spatial-validation'
OUT = BASE/'step-control-development'
VALIDATION = BASE/'step-control-validation'
PROBE = BASE/'step-control-terminal-probe'
PREVIOUS_PROBE = BASE/'step-control-recovery-probe'


def read(path):
    return json.loads(Path(path).read_text())


def prepare():
    for directory in (PREVIOUS_PROBE, PROBE):
        sha = common.digest_file(directory/'plan.json')
        if read(directory/'state.json') != {'status': 'completed', 'plan_sha256': sha} or read(directory/'result.json')['plan_sha256'] != sha:
            raise RuntimeError('terminal recovery probe proofs differ')
    plan_d, result_d, decision = read(PROBE/'plan.json'), read(PROBE/'result.json'), read(PROBE/'decision.json')
    if (decision['selected_config'] != plan_d['selected_config'] or not result_d['mechanics_valid']
        or decision['scientific_recovery_supported'] != result_d['scientific_recovery_supported']
        or decision['previous_probe_C_strict_recovery_passed'] is not False
        or decision['no_further_optimizer_search'] is not True
        or decision['followup'] != plan_d['followup']):
        raise RuntimeError('final development rule does not reproduce the declared terminal decision')
    sources = ['thermotwin/pinn/planar_contact_step_control.py', 'tests/test_planar_contact_step_control.py',
        'thermotwin/studies/pinn_spatial_step_control.py', 'tests/test_pinn_spatial_step_control.py',
        'thermotwin/reports/pinn_spatial_step_control.py', 'tests/test_pinn_spatial_step_control_report.py',
        'tools/pinn_spatial/step-control-probe.py', 'tools/pinn_spatial/step-control-terminal-probe.py',
        str(Path(__file__).resolve().relative_to(ROOT))]
    old_plan = read(BASE/'optimized-v6-development/plan.json')
    protected = set(old_plan['optimization_provenance']['protected_files']) | {
        'thermotwin/PINN_SPATIAL_OPTIMIZATION_PLAN.json', 'thermotwin/PINN_SPATIAL_OPTIMIZATION_PROBE.json',
        'thermotwin/PINN_SPATIAL_OPTIMIZATION_RESULT.json'}
    plan = study.prepare_step_control(OUT, model_config=decision['selected_config'],
        probe_decision_path=PROBE/'decision.json', source_paths=sources,
        optimized_campaign=BASE/'optimized-v6-development', protected_files=sorted(protected))
    target = ROOT/'thermotwin/PINN_SPATIAL_STEP_CONTROL_PLAN.json'
    if target.exists():
        raise RuntimeError('tracked step-control declaration already exists')
    common._atomic(target, (OUT/'plan.json').read_bytes())
    compact = {'schema': 'training_only_step_control_probe_record_v1',
        'failed_strict_probe_C': {'plan': read(PREVIOUS_PROBE/'plan.json'), 'result': read(PREVIOUS_PROBE/'result.json')},
        'terminal_probe_D': {'plan': plan_d, 'result': result_d}, 'decision': decision,
        'configuration_freeze': {'path': str(target.relative_to(ROOT)), 'sha256': common.digest_file(target)},
        'scope': 'one final uniform development comparison; all prior failures retained; no independent replication, inverse fit, reserved evaluation, or further optimizer search'}
    common._write(ROOT/'thermotwin/PINN_SPATIAL_STEP_CONTROL_PROBE.json', compact)
    return {'status': 'prepared', 'plan_sha256': common.digest_file(OUT/'plan.json')}


def run():
    plan = read(OUT/'plan.json')
    study.verify_step_control_plan(plan, OUT)
    sha = common.digest_file(OUT/'plan.json')
    if common.digest_file(ROOT/'thermotwin/PINN_SPATIAL_STEP_CONTROL_PLAN.json') != sha:
        raise RuntimeError('tracked and executed step-control plans differ')
    VALIDATION.mkdir(exist_ok=True)
    descriptor = os.open(VALIDATION/'supervisor.lock', os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    os.close(descriptor)
    execution = study.start_execution(OUT)
    env = dict(os.environ, PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', MPLCONFIGDIR='/private/tmp/thermotwin-mpl',
        XDG_CACHE_HOME='/private/tmp/thermotwin-xdg-cache')
    started = time.monotonic()
    state = {'status': 'running', 'plan_sha256': sha, 'started_utc': common.datetime.now(common.timezone.utc).isoformat(),
        'maximum_parallel_workers': 2, 'automatic_retries': 0, 'return_codes': {}, 'interventions': []}
    def save():
        common._write(VALIDATION/'execution-state.json', state)
    save()
    print(json.dumps({'status': 'started', 'plan_sha256': sha, 'waves': plan['waves']}), flush=True)
    for wave in plan['waves']:
        processes = {}
        for case in wave:
            if time.monotonic()-started >= plan['topology']['top_seconds']:
                study.mark_unrun(OUT, case, 'predeclared aggregate supervisor wall limit before launch')
                continue
            handle = (VALIDATION/(case+'-process.log')).open('w')
            child = subprocess.Popen([sys.executable, '-m', 'thermotwin.studies.pinn_spatial_step_control',
                'run-case', '--output-directory', str(OUT), '--case-id', case],
                cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            processes[case] = {'child': child, 'handle': handle, 'started': time.monotonic(), 'terminated': None}
        while processes:
            for case, record in list(processes.items()):
                child = record['child']
                now = time.monotonic()
                overdue = (now-record['started'] > plan['topology']['seconds_per_worker']+30
                    or now-started > plan['topology']['top_seconds'])
                if child.poll() is None and overdue:
                    if record['terminated'] is None:
                        child.terminate()
                        record['terminated'] = now
                        state['interventions'].append({'case': case, 'action': 'SIGTERM', 'reason': 'predeclared external resource guard'})
                        save()
                    elif now-record['terminated'] > 10:
                        child.kill()
                        state['interventions'].append({'case': case, 'action': 'SIGKILL', 'reason': 'resource guard termination grace elapsed'})
                        save()
                code = child.poll()
                if code is not None:
                    record['handle'].close()
                    study.record_worker_exit(OUT, case, code, 'observed once-only worker exit; no retry')
                    state['return_codes'][case] = code
                    save()
                    print(json.dumps({'case': case, 'return_code': code, 'worker_terminal': True}), flush=True)
                    del processes[case]
            if processes:
                time.sleep(1.)
    result = study.finalize_step_control(OUT)
    evidence = study.validate_step_control(OUT)
    state.update(status='completed', wall_seconds=time.monotonic()-started, evidence=evidence,
        finished_utc=common.datetime.now(common.timezone.utc).isoformat())
    save()
    print(json.dumps({'status': 'completed', 'wall_seconds': state['wall_seconds'],
        'forward_status': result['forward_status']}), flush=True)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run'))
    args = parser.parse_args()
    value = prepare() if args.command == 'prepare' else run()
    print(json.dumps({'command': args.command, 'status': value['status']}), flush=True)
