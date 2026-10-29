"""Prepare and execute one probe-selected four-case optimizer development run."""
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
from thermotwin.studies import pinn_spatial_optimized as study

BASE = ROOT/'reports/pinn-spatial-validation'
OLD = BASE/'graded-log-development'
OUT = BASE/'optimized-v6-development'
VALIDATION = BASE/'optimization-validation'
A = BASE/'optimization-directional-probe'
B = BASE/'optimization-history-probe'


def read(path):
    return json.loads(Path(path).read_text())


def prepare():
    decision = read(B/'decision.json')
    old = read(OLD/'plan.json')
    for directory in (A, B):
        expected = {'status': 'completed', 'plan_sha256': common.digest_file(directory/'plan.json')}
        if read(directory/'state.json') != expected or read(directory/'result.json')['plan_sha256'] != expected['plan_sha256']:
            raise RuntimeError('both probes require matching terminal plan and result proofs')
    aplan, aresult, bresult = read(A/'plan.json'), read(A/'result.json'), read(B/'result.json')
    declared = aplan['four_case_followup']
    if (decision['selected_config']['learning_rate'] != aresult['selected_learning_rate']
        or decision['selected_config']['optimizer_history'] != bresult['selected_optimizer_history']
        or decision['start_policy'] != declared['start_policy']
        or decision['maximum_wall_seconds'] != declared['maximum_wall_seconds']
        or decision['maximum_rss_bytes'] != declared['maximum_rss_bytes']):
        raise RuntimeError('selected rule does not reproduce the frozen probe decisions')
    if ({(row['case'], row['arm']) for row in bresult['arms']}
        != {(case, arm) for case in ('t02', 't04') for arm in ('reset', 'persistent')}
        or len(bresult['arms']) != 4
        or not all(row['copied_weights_restored_exactly'] for row in bresult['arms'])
        or not all(row['copied_weights_restored_exactly'] for row in aresult['cases'])):
        raise RuntimeError('probe case denominator or restoration proof differs')
    if decision['selected_config']['adam_epochs'] != 320 or decision['selected_config']['max_backtracks'] != 4:
        raise RuntimeError('follow-up budget differs from the pre-probe declaration')
    protected = sorted(set(old['protected_files']) | {
        'thermotwin/PINN_SPATIAL_GRADED_LOG_PLAN.json',
        'thermotwin/PINN_SPATIAL_GRADED_LOG_RESULT.json'})
    sources = ['thermotwin/pinn/planar_contact_optimized.py',
        'tests/test_planar_contact_optimized.py', 'tests/test_pinn_spatial_optimized.py',
        str(Path(__file__).resolve().relative_to(ROOT)),
        str((BASE/'optimization-directional-probe.py').relative_to(ROOT)),
        str((BASE/'optimization-history-probe.py').relative_to(ROOT))]
    plan = study.prepare_optimized(OUT, model_config=decision['selected_config'],
        probe_decision_path=B/'decision.json', source_paths=sources,
        graded_comparison=OLD, paired_v2_directory=BASE/'revision-v2-development/two-d',
        v2_development_record='thermotwin/PINN_SPATIAL_REVISION_V2_DEVELOPMENT.json',
        preserved_campaign=BASE/'campaign-0bfe84ea-v1',
        protected_campaigns=(BASE/'continuity-v3-development', BASE/'continuity-v4-development'),
        protected_files=protected)
    target = ROOT/'thermotwin/PINN_SPATIAL_OPTIMIZATION_PLAN.json'
    if target.exists():
        raise RuntimeError('tracked optimizer declaration already exists')
    common._atomic(target, (OUT/'plan.json').read_bytes())
    compact = {'schema': 'training_only_optimizer_probe_summary_v1',
        'status': 'completed', 'development_only': True,
        'training_only': True, 'reference_fields_used': False,
        'independent_diagnostics_used_for_selection': False,
        'directional_plan': read(A/'plan.json'),
        'directional_result': aresult,
        'history_plan': read(B/'plan.json'),
        'history_rows': [{key: value for key, value in row.items() if key != 'history'}
            | {'accepted_updates': sum(event['accepted'] for event in row['history']),
               'rejected_updates': sum(not event['accepted'] for event in row['history']),
               'backtracks': sum(event['backtracks'] for event in row['history']),
               'objective_trajectory': [event['objective_after'] for event in row['history']],
               'optimizer_step_counts': [event['optimizer_step_counts_after'] for event in row['history']]}
            for row in bresult['arms']],
        'history_wall_seconds': bresult['wall_seconds'],
        'history_peak_rss_bytes': bresult['peak_rss_bytes'], 'decision': decision,
        'limits': 'two already-open partial development datasets; short checkpoint continuation probes do not predict a converged fresh fit or independent replication',
        'four_case_plan_sha256': common.digest_file(OUT/'plan.json')}
    common._write(ROOT/'thermotwin/PINN_SPATIAL_OPTIMIZATION_PROBE.json', compact)
    return {'status': 'prepared', 'plan_sha256': common.digest_file(OUT/'plan.json')}


def run():
    plan = read(OUT/'plan.json')
    study.verify_optimized_plan(plan, OUT)
    sha = common.digest_file(OUT/'plan.json')
    if common.digest_file(ROOT/'thermotwin/PINN_SPATIAL_OPTIMIZATION_PLAN.json') != sha:
        raise RuntimeError('tracked and executed declarations differ')
    VALIDATION.mkdir(exist_ok=True)
    descriptor = os.open(VALIDATION/'execution.lock', os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    os.close(descriptor)
    env = dict(os.environ, PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', MPLCONFIGDIR='/private/tmp/thermotwin-mpl',
        XDG_CACHE_HOME='/private/tmp/thermotwin-xdg-cache')
    started = time.perf_counter()
    state = {'status': 'running', 'plan_sha256': sha, 'scientific_processes': 1,
        'threads': 1, 'automatic_retries': 0, 'started_utc': common.datetime.now(common.timezone.utc).isoformat()}
    common._write(VALIDATION/'execution-state.json', state)
    with (VALIDATION/'scientific-process.log').open('w') as handle:
        child = subprocess.Popen([sys.executable, '-m', 'thermotwin.studies.pinn_spatial_optimized',
            'run', '--output-directory', str(OUT)], cwd=ROOT, env=env,
            stdout=handle, stderr=subprocess.STDOUT)
        state['child_pid'] = child.pid
        common._write(VALIDATION/'execution-state.json', state)
        print(json.dumps({'status': 'started', 'plan_sha256': sha, 'child_pid': child.pid}), flush=True)
        code = child.wait()
    state.update(return_code=code, wall_seconds=time.perf_counter()-started,
        finished_utc=common.datetime.now(common.timezone.utc).isoformat())
    if code:
        state['status'] = 'process_failure'
        common._write(VALIDATION/'execution-state.json', state)
        raise SystemExit(code)
    evidence = study.validate_optimized(OUT)
    state.update(status='completed', evidence=evidence)
    common._write(VALIDATION/'execution-state.json', state)
    print(json.dumps({'status': 'completed', 'wall_seconds': state['wall_seconds']}), flush=True)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run'))
    args = parser.parse_args()
    result = prepare() if args.command == 'prepare' else run()
    print(json.dumps({'command': args.command, 'status': result['status']}), flush=True)
