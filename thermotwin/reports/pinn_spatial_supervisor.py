"""One-shot, CI-gated supervisor for the frozen spatial PINN campaign.

The process runs one scientific child, preserves evidence on interruption and
never restarts or replaces a trial. It is suitable for a native one-shot job.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from ..studies.pinn_spatial_validation import (ROOT, now, source_context, validated_protocol,
                                              verify_ci, write_json, validate_evidence)


def successful_ci(revision, monitor, progress, *, attempts=75):
    """Bounded wait for this exact commit; no scientific observations are made."""
    url = ('https://api.github.com/repos/rolandrolandroland/thermotwin/actions/runs'
           f'?head_sha={revision}&per_page=20')
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={
                'Accept': 'application/vnd.github+json', 'User-Agent': 'ThermoTwin-Spatial-Validation'})
            with urllib.request.urlopen(request, timeout=25) as response:
                runs = json.load(response)['workflow_runs']
            candidates = [run for run in runs if run.get('head_sha') == revision
                          and run.get('workflow_id') == 348759149]
            if candidates:
                run = max(candidates, key=lambda value: (value['run_number'], value.get('run_attempt', 1)))
                progress(ci_run_id=run['id'], ci_status=run['status'], ci_url=run['html_url'])
                if run['status'] == 'completed':
                    verify_ci(run, revision)
                    path = monitor/'exact-source-ci.json'
                    write_json(path, run)
                    return path
        except (urllib.error.URLError, TimeoutError) as error:
            progress(last_ci_network_error=f'{type(error).__name__}: {error}')
        if attempt+1 < attempts:
            time.sleep(60)
    raise RuntimeError('exact-source CI wait expired; no scientific data generated')


def rss_bytes(pids):
    result = subprocess.run(('ps', '-o', 'rss=', '-p', ','.join(map(str, pids))),
                            check=True, capture_output=True, text=True)
    values = [int(value) for value in result.stdout.split()]
    if not values or any(value < 0 for value in values):
        raise RuntimeError('could not sample job memory')
    return sum(values)*1024


def supervise(output, monitor, revision):
    protocol = validated_protocol()
    context = source_context(revision)
    output, monitor = Path(output).resolve(), Path(monitor).resolve()
    if output.exists():
        raise FileExistsError('scientific output already exists; no automatic resume or replacement')
    monitor.mkdir(parents=True, exist_ok=True)
    with (monitor/'supervisor.lock').open('x') as stream:
        stream.write(str(os.getpid())+'\n')
    limits = protocol['execution']
    state = dict(source_commit=revision, source_digest=context['source_digest'],
        supervisor_pid=os.getpid(), child_pid=None, status='waiting_for_exact_source_ci',
        started_utc=now(), output_directory=str(output), limits=limits,
        automatic_restart=False, peak_rss_bytes=0)
    def update(**changes):
        state.update(changes, updated_utc=now())
        write_json(monitor/'state.json', state)
    child = None
    inhibitor = None
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
        ci_path = successful_ci(revision, monitor, update)
        # These are new-child settings; the running operating-decision job is untouched.
        environment = dict(os.environ)
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            environment[key] = '1'
        environment['PYTHONUNBUFFERED'] = '1'
        environment.pop('PYTHONPATH', None)
        environment['MPLBACKEND'] = 'Agg'
        cache = monitor/'matplotlib-cache'
        cache.mkdir(exist_ok=True)
        environment['MPLCONFIGDIR'] = str(cache)
        environment['XDG_CACHE_HOME'] = str(cache)
        command = [sys.executable, '-u', '-m', 'thermotwin.reports.pinn_spatial_validation',
                   'run', '--output-directory', str(output), '--source-commit', revision,
                   '--ci-evidence', str(ci_path)]
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
                update(peak_rss_bytes=max(memory, state['peak_rss_bytes']), wall_seconds=elapsed)
                if memory > limits['rss_limit_bytes'] or elapsed > limits['wall_limit_seconds']:
                    raise RuntimeError('predeclared resource limit exceeded; preserve incomplete evidence')
                time.sleep(5)
        if child.returncode:
            raise RuntimeError(f'campaign exited {child.returncode}; preserve evidence, do not retry')
        evidence = validate_evidence(output)
        write_json(monitor/'independent-evidence-validation.json', dict(evidence, completed_utc=now()))
        result = json.loads((output/'result.json').read_text())
        update(status=result['status'], evidence_validation=evidence, completed_utc=now(),
               wall_seconds=time.monotonic()-started, last_child_return_code=child.returncode)
    except BaseException as error:
        stop_child()
        update(status='stopped_preserve_evidence', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if inhibitor is not None and inhibitor.poll() is None:
            inhibitor.terminate()
            inhibitor.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', required=True)
    parser.add_argument('--monitor-directory', required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    supervise(args.output_directory, args.monitor_directory, args.source_commit)


if __name__ == '__main__':
    main()
