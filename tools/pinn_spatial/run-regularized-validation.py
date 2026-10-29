"""One bounded full software suite with frozen source and terminal-log proof."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from thermotwin.studies import pinn_spatial_continuity as common
from thermotwin.studies.pinn_spatial_cutoff_audit import _one_thread

BASE = ROOT/'reports/pinn-spatial-validation'
OUT = BASE/'regularized-validation'
CAMPAIGN = BASE/'regularized-v8-development'
MAXIMUM_WALL_SECONDS = 2700.


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_snapshot(plan):
    sources = set(plan['source_sha256'])
    for directory in ('thermotwin', 'tests', 'tools'):
        sources.update(str(path.relative_to(ROOT)) for path in (ROOT/directory).rglob('*.py')
                       if '__pycache__' not in path.parts)
    sources.add('pyproject.toml')
    return {name: digest(ROOT/name) for name in sorted(sources)}


def run():
    _one_thread()
    if (OUT/'full-suite-state.json').exists() or (OUT/'full-suite.log').exists():
        raise RuntimeError('The once-only full validation attempt already exists')
    OUT.mkdir(exist_ok=True)
    plan = json.loads((CAMPAIGN/'plan.json').read_text())
    # Capture the complete current Python tree without touching ignored data.
    before = source_snapshot(plan)
    bootstrap = ("import torch,unittest; torch.set_num_threads(1); "
        "torch.set_num_interop_threads(1); "
        "unittest.main(module=None,argv=['unittest','discover','-s','tests','-t','.','-v'])")
    command = [sys.executable, '-c', bootstrap]
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), top_level_dir=str(ROOT))
    expected = suite.countTestCases()
    state = {'schema': 'regularized_full_software_suite_v1', 'status': 'running',
        'passed': False, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'expected_test_count': expected, 'discovered_test_count': expected,
        'source_sha256': before, 'source_manifest_sha256': common.canonical_digest(before),
        'command': command, 'maximum_wall_seconds': MAXIMUM_WALL_SECONDS,
        'campaign_plan_sha256': digest(CAMPAIGN/'plan.json')}
    common._write(OUT/'full-suite-state.json', state)
    code, failure = None, None
    with (OUT/'full-suite.log').open('x') as handle:
        try:
            process = subprocess.run(command, cwd=ROOT, stdout=handle,
                stderr=subprocess.STDOUT, timeout=MAXIMUM_WALL_SECONDS)
            code = process.returncode
        except BaseException as error:
            failure = {'error_type': type(error).__name__, 'error': str(error)}
    log = (OUT/'full-suite.log').read_text()
    counts = list(re.finditer(r'^Ran (\d+) tests? in ([\d.]+)s$', log, re.MULTILINE))
    count = int(counts[-1].group(1)) if counts else None
    trailer = log[counts[-1].end():] if counts else ''
    terminal_ok = bool(trailer.strip()) and trailer.strip().splitlines()[-1] == 'OK'
    try:
        after = source_snapshot(plan)
        unchanged = before == after
    except BaseException as error:
        unchanged = False
        failure = dict(failure or {}, source_snapshot_error=str(error))
    counters = {}
    for key in ('failures', 'errors', 'expected_failures', 'unexpected_successes', 'skipped'):
        label = key.replace('_', ' ')
        match = re.search(r'(?:'+re.escape(key)+'|'+re.escape(label)+r')=(\d+)', trailer)
        counters[key] = int(match.group(1)) if match else 0 if terminal_ok else None
    passed = (failure is None and code == 0 and count == expected and expected > 0 and terminal_ok
              and unchanged and all(value == 0 for value in counters.values()))
    state.update(status='completed' if failure is None else 'failed', passed=bool(passed),
        return_code=code, tests_run=count, terminal_log_ok=terminal_ok,
        log_sha256=digest(OUT/'full-suite.log'), source_unchanged=unchanged,
        wall_seconds=time.perf_counter()-started,
        unittest_wall_seconds=float(counts[-1].group(2)) if counts else None,
        finished_utc=datetime.now(timezone.utc).isoformat(), failure=failure, **counters)
    common._write(OUT/'full-suite-state.json', state)
    print(json.dumps({key:state[key] for key in ('status','passed','tests_run','wall_seconds')}, allow_nan=False), flush=True)
    return state


if __name__ == '__main__':
    result = run()
    sys.exit(0 if result['passed'] else 1)
