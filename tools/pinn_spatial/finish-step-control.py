"""Bounded local completion pipeline; never starts or retries a scientific fit."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from thermotwin.studies import pinn_spatial_continuity as common

BASE = ROOT/'reports/pinn-spatial-validation'
VALIDATION = BASE/'step-control-validation'
CAMPAIGN = BASE/'step-control-development'
REPORT = BASE/'step-control-report'
MAX_WAIT_SECONDS = 9000.


def read(path):
    return json.loads(Path(path).read_text())


def load_finalizer():
    path = ROOT/'tools/pinn_spatial/finalize-step-control-result.py'
    spec = importlib.util.spec_from_file_location('step_control_finalizer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(*, resume_completed=False):
    finalizer = load_finalizer()
    state_path = VALIDATION/'completion-pipeline-state.json'
    lock_path = VALIDATION/'completion-pipeline.lock'
    resumed = None
    if resume_completed:
        failed = read(state_path)
        if failed.get('status') != 'failed' or 'test suite did not pass: completed' not in failed.get('error', ''):
            raise RuntimeError('only the recorded test-status bookkeeping failure may resume here')
        if read(VALIDATION/'execution-state.json').get('status') != 'completed':
            raise RuntimeError('report-only recovery requires the completed scientific supervisor')
        proof = finalizer.test_evidence('full-suite')
        if proof.get('completed') is not True or proof.get('passed') is not True:
            raise RuntimeError('report-only recovery requires a verified passing test proof')
        resumed = {'failed_state_path': str(state_path.relative_to(ROOT)),
            'failed_state_sha256': common.digest_file(state_path),
            'reason': 'postprocessing test-status schema compatibility; no scientific rerun'}
        repair = VALIDATION/'completion-repair-20261029'
        if not (repair/'preservation-before.json').is_file():
            raise RuntimeError('original failed scripts and evidence must be archived before recovery')
        state_path, lock_path = repair/'state.json', repair/'run.lock'
    descriptor = os.open(lock_path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    os.close(descriptor)
    started = time.monotonic()
    record = {'status': 'waiting', 'started_utc': common.datetime.now(common.timezone.utc).isoformat(),
        'maximum_wait_seconds': MAX_WAIT_SECONDS, 'scientific_fits_started': 0,
        'automatic_retries': 0, 'source_sha256': common.digest_file(__file__),
        'finalizer_sha256': common.digest_file(ROOT/'tools/pinn_spatial/finalize-step-control-result.py'),
        'report_only_recovery': resumed}
    def save(status):
        record.update(status=status, updated_utc=common.datetime.now(common.timezone.utc).isoformat())
        common._write(state_path, record)
        print(json.dumps({'status': status}), flush=True)
    save('waiting')
    try:
        while True:
            supervisor = read(VALIDATION/'execution-state.json')
            tests = finalizer.test_evidence('full-suite')
            if supervisor['status'] not in ('running', 'completed'):
                raise RuntimeError('scientific supervisor ended abnormally: '+supervisor['status'])
            if tests.get('completed') is True and tests.get('passed') is not True:
                raise RuntimeError('complete repository test suite lacks a valid passing proof')
            if tests.get('completed') is not True and tests.get('status') not in ('running', 'prepared'):
                raise RuntimeError('unrecognized repository test state: '+str(tests.get('status')))
            if supervisor['status'] == 'completed' and tests.get('completed') is True and tests.get('passed') is True:
                break
            if time.monotonic()-started > MAX_WAIT_SECONDS:
                raise RuntimeError('completion pipeline wait cap reached; no scientific retry or budget extension')
            time.sleep(5.)
        if read(CAMPAIGN/'state.json')['status'] != 'completed' or read(CAMPAIGN/'execution.json')['status'] != 'completed':
            raise RuntimeError('scientific completion flags disagree')
        os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/thermotwin-mpl')
        save('rendering_saved_results')
        from thermotwin.reports.pinn_spatial_step_control import render_report
        report = render_report(CAMPAIGN, output_directory=REPORT)
        record['report'] = report
        save('validating_and_finalizing')
        if common.digest_file(ROOT/'tools/pinn_spatial/finalize-step-control-result.py') != record['finalizer_sha256']:
            raise RuntimeError('completion finalizer changed after this pipeline started')
        result = finalizer.finalize()
        record.update(result=result, wall_seconds=time.monotonic()-started,
            scientific_status=read(CAMPAIGN/'result.json')['forward_status'])
        save('completed')
        return 0
    except BaseException as error:
        record.update(error=repr(error), traceback=traceback.format_exc(), wall_seconds=time.monotonic()-started)
        save('failed')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-completed', action='store_true',
        help='repair the recorded postprocessing status mismatch without rerunning science')
    args = parser.parse_args()
    raise SystemExit(main(resume_completed=args.resume_completed))
