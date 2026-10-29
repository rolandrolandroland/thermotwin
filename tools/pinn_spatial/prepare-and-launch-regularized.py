"""Freeze and supervise one four-case regularized development comparison."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from thermotwin.studies import pinn_spatial_continuity as common
from thermotwin.studies import pinn_spatial_regularized as study

BASE = ROOT/'reports/pinn-spatial-validation'
OUT = BASE/'regularized-v8-development'
VALIDATION = BASE/'regularized-validation'
PROBE = BASE/'regularized-gradient-probe'
METHOD_VALIDATION = BASE/'regularized-method-validation'
TRACKED_PLAN = ROOT/'thermotwin/PINN_SPATIAL_REGULARIZED_PLAN.json'


def read(path):
    return json.loads(Path(path).read_text())


def new_json(path, value):
    with path.open('xb') as handle:
        handle.write(common._json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def test_proof(state_path, log_path, count, required_sources):
    state = read(state_path)
    text = log_path.read_text()
    counts = re.findall(r'^Ran ([0-9]+) tests? in ([0-9.]+)s\s*$', text, re.MULTILINE)
    sources = state.get('source_sha256')
    if (state.get('status') != 'passed' or type(state.get('return_code')) is not int
        or state['return_code'] != 0 or type(state.get('tests_run')) is not int
        or state['tests_run'] != count or state.get('source_unchanged') is not True
        or not counts or int(counts[-1][0]) != count or text.rstrip().splitlines()[-1] != 'OK'
        or state.get('log_sha256') != common.digest_file(log_path)
        or not isinstance(sources, dict) or not set(required_sources).issubset(sources)
        or common._source_records(sources) != sources
        or any(state.get(key, 0) != 0 for key in ('failures', 'errors', 'unexpected_successes', 'skipped'))):
        raise RuntimeError('A terminal passing, source-bound software proof is required: '+str(state_path))
    return {'state': {'path': str(state_path.relative_to(ROOT)), 'sha256': common.digest_file(state_path)},
        'log': {'path': str(log_path.relative_to(ROOT)), 'sha256': common.digest_file(log_path)},
        'tests_run': count, 'source_sha256': sources, 'passed': True}


def preparation_checks():
    return {'method_tests': test_proof(METHOD_VALIDATION/'focused-tests-state.json',
        METHOD_VALIDATION/'focused-tests.log', 17, (
            'thermotwin/pinn/planar_contact_regularized.py', 'tests/test_planar_contact_regularized.py',
            'tools/pinn_spatial/regularized-gradient-probe.py',
            'tests/test_pinn_spatial_regularized_probe.py', 'tests/test_regularized_gradient_lifecycle.py')),
        'runner_tests': test_proof(VALIDATION/'runner-tests-state.json', VALIDATION/'runner-tests.log', 11, (
            'thermotwin/studies/pinn_spatial_regularized.py', 'tests/test_pinn_spatial_regularized.py'))}


def prepare():
    study.cutoff._one_thread()
    if OUT.exists() or TRACKED_PLAN.exists():
        raise RuntimeError('The once-only regularized comparison was already prepared or reserved')
    checks = preparation_checks()
    decision_path = PROBE/'method-decision.json'
    decision = read(decision_path)
    # Independently authenticate all two-calibration/fourteen-point method
    # evidence before reserving four forward opportunities or calibrating them.
    method = study._method_gate(decision_path, decision['selected_config'],
        ROOT/'tools/pinn_spatial/regularized-gradient-probe.py')
    if method['method_checks_passed'] is not True:
        raise RuntimeError('The method gate did not authorize forward development')
    VALIDATION.mkdir(parents=True, exist_ok=True)
    quality_path = VALIDATION/'forward-preparation-tests.json'
    new_json(quality_path, {'schema': 'regularized_forward_software_prerequisites_v1', 'checks': checks,
        'method_checks': method, 'full_repository_suite_required_for_closeout': True})
    numerical_sources = [
        'thermotwin/pinn/planar_contact_regularized.py', 'tests/test_planar_contact_regularized.py',
        'thermotwin/studies/pinn_spatial_regularized.py', 'tests/test_pinn_spatial_regularized.py',
        'tools/pinn_spatial/regularized-gradient-probe.py', 'tests/test_pinn_spatial_regularized_probe.py',
        'tests/test_regularized_gradient_lifecycle.py', str(Path(__file__).resolve().relative_to(ROOT))]
    proof_paths = [str(quality_path.relative_to(ROOT))]
    for proof in checks.values():
        proof_paths.extend((proof['state']['path'], proof['log']['path']))
    # Including prerequisite bytes in the prospective source closure binds them
    # before calibration as well as in the later full comparison declaration.
    sources = numerical_sources+proof_paths
    protected = proof_paths+[
        'thermotwin/PINN_SPATIAL_REGULARIZED_METHOD_PLAN.json',
        'thermotwin/PINN_SPATIAL_REGULARIZED_METHOD_RESULT.json']
    plan = study.prepare_regularized(OUT, model_config=decision['selected_config'],
        method_decision_path=decision_path,
        gradient_probe_validator_path=ROOT/'tools/pinn_spatial/regularized-gradient-probe.py',
        source_paths=sources, step_control_campaign=BASE/'step-control-development',
        cutoff_audit_directory=BASE/'cutoff-directional-audit', protected_files=protected)
    with TRACKED_PLAN.open('xb') as handle:
        handle.write((OUT/'plan.json').read_bytes())
        handle.flush()
        os.fsync(handle.fileno())
    ledger = read(OUT/'calibration-state.json')
    if (ledger['status'] != 'completed' or ledger['attempted_cases'] != ['t01', 't02', 't03', 't04']
        or ledger['completed_cases'] != ledger['attempted_cases']):
        raise RuntimeError('The four initial calibration opportunities were not completed exactly once')
    new_json(VALIDATION/'preparation-state.json', {'status': 'prepared',
        'plan_sha256': common.digest_file(OUT/'plan.json'),
        'tracked_plan_sha256': common.digest_file(TRACKED_PLAN), 'calibration_ledger': ledger,
        'calibration_accounting': plan['calibration_accounting'], 'quality_checks': checks})
    return {'status': 'prepared', 'plan_sha256': common.digest_file(OUT/'plan.json')}


def run():
    study.cutoff._one_thread()
    plan = read(OUT/'plan.json')
    study.verify_regularized_plan(plan, OUT)
    sha = common.digest_file(OUT/'plan.json')
    if common.digest_file(TRACKED_PLAN) != sha:
        raise RuntimeError('Tracked and executed regularized declarations differ')
    prepared = read(VALIDATION/'preparation-state.json')
    if (prepared.get('status') != 'prepared' or prepared.get('plan_sha256') != sha
        or prepared.get('tracked_plan_sha256') != sha or prepared.get('quality_checks') != preparation_checks()):
        raise RuntimeError('Prepared software, source and calibration proofs changed')
    descriptor = os.open(VALIDATION/'supervisor.lock', os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    os.close(descriptor)
    execution = study.start_execution(OUT)
    env = dict(os.environ, PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        MPLCONFIGDIR='/private/tmp/thermotwin-mpl', XDG_CACHE_HOME='/private/tmp/thermotwin-xdg-cache')
    started = time.monotonic()
    state = {'status': 'running', 'plan_sha256': sha,
        'started_utc': common.datetime.now(common.timezone.utc).isoformat(),
        'maximum_parallel_workers': 2, 'automatic_retries': 0, 'return_codes': {},
        'interventions': [], 'launch_failures': {}, 'initial_method_checks_passed': True,
        'calibration_accounting': plan['calibration_accounting']}
    def save():
        common._write(VALIDATION/'execution-state.json', state)
    save()
    print(json.dumps({'status': 'started', 'plan_sha256': sha, 'waves': plan['waves']}), flush=True)
    for wave in plan['waves']:
        processes = {}
        for case in wave:
            if time.monotonic()-execution['started_monotonic'] >= plan['topology']['top_seconds']:
                study.mark_unrun(OUT, case, 'predeclared aggregate supervisor wall limit before launch')
                continue
            handle = (VALIDATION/(case+'-process.log')).open('x')
            try:
                child = subprocess.Popen([sys.executable, '-m', 'thermotwin.studies.pinn_spatial_regularized',
                    'run-case', '--output-directory', str(OUT), '--case-id', case],
                    cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            except OSError as error:
                handle.close()
                state['launch_failures'][case] = repr(error)
                study.mark_unrun(OUT, case, 'child launch failed; no retry: '+repr(error))
                save()
                continue
            processes[case] = {'child': child, 'handle': handle, 'started': time.monotonic(), 'terminated': None}
        while processes:
            for case, record in list(processes.items()):
                child, now = record['child'], time.monotonic()
                overdue = (now-record['started'] > plan['topology']['seconds_per_worker']+30
                    or now-execution['started_monotonic'] > plan['topology']['top_seconds'])
                if child.poll() is None and overdue:
                    if record['terminated'] is None:
                        child.terminate()
                        record['terminated'] = now
                        state['interventions'].append({'case': case, 'action': 'SIGTERM', 'reason': 'predeclared external resource guard'})
                        save()
                    elif now-record['terminated'] > 10:
                        child.kill()
                        state['interventions'].append({'case': case, 'action': 'SIGKILL', 'reason': 'termination grace elapsed'})
                        save()
                code = child.poll()
                if code is not None:
                    record['handle'].close()
                    study.record_worker_exit(OUT, case, code, 'observed once-only child exit; no retry')
                    state['return_codes'][case] = code
                    save()
                    print(json.dumps({'case': case, 'return_code': code, 'worker_terminal': True}), flush=True)
                    del processes[case]
            if processes:
                time.sleep(1.)
    result = study.finalize_regularized(OUT)
    evidence = study.validate_regularized(OUT)
    state.update(status='completed', wall_seconds=time.monotonic()-started, evidence=evidence,
        finished_utc=common.datetime.now(common.timezone.utc).isoformat())
    save()
    print(json.dumps({'status': 'completed', 'wall_seconds': state['wall_seconds'],
        'initial_method_checks_passed': True, 'forward_status': result['forward_status']}), flush=True)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run'))
    args = parser.parse_args()
    value = prepare() if args.command == 'prepare' else run()
    print(json.dumps({'command': args.command, 'status': value['status']}), flush=True)
