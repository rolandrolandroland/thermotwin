"""One-shot exact-source-CI gate and resource monitor for the v2 revision."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .pinn_spatial_supervisor import successful_ci, rss_bytes
from ..studies.pinn_spatial_validation import ROOT, now, write_json, source_context
from ..studies.pinn_spatial_revision import (
    validated_revision_protocol, preserved_context, validate_revision_evidence,
)


def supervise(*, output_directory, monitor_directory, source_commit, preserved_campaign):
    protocol = validated_revision_protocol()
    source_context(source_commit)
    preserved_context(preserved_campaign, expected=protocol['preservation'])
    output, monitor = Path(output_directory).resolve(), Path(monitor_directory).resolve()
    if output.exists():
        raise FileExistsError('fresh output already exists; no automatic restart or replacement')
    monitor.mkdir(parents=True, exist_ok=True)
    with (monitor/'supervisor.lock').open('x') as stream:
        stream.write(str(os.getpid())+'\n')
    state = {'status': 'waiting_for_exact_source_ci', 'source_commit': source_commit,
             'started_utc': now(), 'supervisor_pid': os.getpid(), 'peak_sampled_rss_bytes': 0,
             'automatic_restart': False, 'output_directory': str(output), 'limits': protocol['execution']}
    def update(**changes):
        state.update(changes, updated_utc=now())
        write_json(monitor/'state.json', state)
    child, inhibitor = None, None
    def stop_child():
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
    def interrupted(signum, frame):
        update(status='interrupted_preserve_evidence', signal=signum)
        stop_child()
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    update()
    try:
        ci_path = successful_ci(source_commit, monitor, update)
        environment = dict(os.environ)
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            environment[key] = '1'
        environment.pop('PYTHONPATH', None)
        environment['PYTHONUNBUFFERED'] = '1'
        environment['MPLBACKEND'] = 'Agg'
        cache = monitor/'matplotlib-cache'
        cache.mkdir(exist_ok=True)
        environment['MPLCONFIGDIR'] = str(cache)
        environment['XDG_CACHE_HOME'] = str(cache)
        command = [sys.executable, '-u', '-m', 'thermotwin.reports.pinn_spatial_revision',
            'evaluate', '--output-directory', str(output), '--source-commit', source_commit,
            '--ci-evidence', str(ci_path), '--preserved-campaign', str(Path(preserved_campaign).resolve())]
        started = time.monotonic()
        with (monitor/'campaign.log').open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            if sys.platform == 'darwin':
                inhibitor = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(child.pid)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            update(status='campaign_running', child_pid=child.pid, command=command,
                   computation_started_utc=now())
            while child.poll() is None:
                memory = rss_bytes([os.getpid(), child.pid])
                elapsed = time.monotonic()-started
                update(peak_sampled_rss_bytes=max(memory, state['peak_sampled_rss_bytes']), wall_seconds=elapsed)
                if (memory > protocol['execution']['rss_limit_bytes']
                    or elapsed > protocol['execution']['wall_limit_seconds']):
                    raise RuntimeError('declared resource limit exceeded; preserve evidence')
                time.sleep(5)
        if child.returncode:
            raise RuntimeError(f'fresh campaign exited {child.returncode}; do not replace trials')
        checked = validate_revision_evidence(output)
        write_json(monitor/'independent-artifact-validation.json', checked)
        result = json.loads((output/'result.json').read_text())
        update(status=result['status'], completed_utc=now(), evidence_validation=checked,
               last_child_return_code=child.returncode, wall_seconds=time.monotonic()-started)
    except BaseException as error:
        stop_child()
        update(status='stopped_preserve_evidence', error=type(error).__name__+': '+str(error))
        raise
    finally:
        if inhibitor is not None and inhibitor.poll() is None:
            inhibitor.terminate()
            inhibitor.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('output-directory', 'monitor-directory', 'source-commit', 'preserved-campaign'):
        parser.add_argument('--'+option, required=True)
    supervise(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
