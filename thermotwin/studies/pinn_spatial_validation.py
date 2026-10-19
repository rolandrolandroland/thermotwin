"""Frozen, serial fresh-1D and bounded-2D synthetic validation campaign.

No operating-decision cohort or result is imported or opened. The old numerical
source manifest is checked only to ensure this additive study left it intact.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / 'thermotwin/PINN_SPATIAL_VALIDATION_PROTOCOL.json'
CAMPAIGN = 'pinn_spatial_v1_2026_10'


def now():
    return datetime.now(timezone.utc).isoformat()


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def campaign_protocol():
    from .pinn_spatial_fresh_1d import Fresh1DValidationConfig
    from .pinn_spatial_2d import BENCHMARK_SPEC
    return {
        'schema_version': 1, 'campaign': CAMPAIGN,
        'stages': ['fresh_1d', 'verified_2d_reference', 'bounded_2d_inverse_pinn', 'geometry_3d_assessment'],
        'fresh_1d': asdict(Fresh1DValidationConfig()),
        'bounded_2d': BENCHMARK_SPEC,
        'execution': {
            'source': 'clean committed clone; passing exact-source tests workflow before data generation',
            'checkpoint': 'final epoch or optimizer-returned state; never truth-selected',
            'threads': 1, 'wall_limit_seconds': 10800, 'rss_limit_bytes': 2 * 1024**3,
            'automatic_restart': False, 'evaluation_retries': 0,
            'failure_policy': 'retain every declared trial and failed method; no replacement seeds',
            'reference_failure': 'stop before fitting; preserve evidence and declare a revised protocol separately',
            '1d_scientific_gate_failure': 'report honestly; independent 2D scalar-contact study can proceed',
            '2d_development_gate_failure': 'retain fixed evaluation as negative evidence; no evaluation-driven tuning',
            'runtime_requirements': {'implementation': 'CPython', 'python': '3.10.12',
                'packages': {'numpy': '2.2.6', 'scipy': '1.15.3', 'torch': '2.9.1', 'matplotlib': '3.10.8'}},
        },
        'geometry_3d': {
            'geometry': 'synthetic extruded plate, not a hardware drawing',
            'dimensions_m': [.04, .01, .004],
            'contact_footprint_m': {'x': [.008, .024], 'y': [0., .01]},
            'out_of_plane_conditions': 'uniform initial state, material, loads and full-depth contact; insulated y faces',
            'decision': '2D suffices for this exact extrusion; no numerical 3D benchmark is justified yet',
            'new_3d_protocol_requires': ['device dimensions and material stack',
                'contact footprint in both face directions', 'localized loads and side boundary conditions',
                'sensor locations', 'predeclared decision or temperature-error tolerance'],
        },
        'claims_excluded': ['hardware validation', 'general PINN superiority',
            'precise failure-rate estimates from three trials', 'full electrothermal 2D module identification',
            'time-only PINN prediction under arbitrary new controls'],
    }


def validated_protocol():
    frozen = json.loads(PROTOCOL_PATH.read_text())
    actual = json.loads(json.dumps(campaign_protocol(), allow_nan=False))
    if frozen != actual:
        raise RuntimeError('frozen protocol differs from implementation; do not generate named data')
    return frozen


def verify_ci(ci, revision):
    if (ci.get('head_sha') != revision or ci.get('status') != 'completed'
        or ci.get('conclusion') != 'success' or ci.get('workflow_id') != 348759149
        or ci.get('repository', {}).get('full_name') != 'rolandrolandroland/thermotwin'):
        raise ValueError('passing exact-source ThermoTwin tests workflow evidence is required')
    return {key: ci.get(key) for key in ('id', 'head_sha', 'status', 'conclusion',
            'workflow_id', 'html_url', 'run_attempt', 'run_started_at', 'updated_at')}


def source_context(revision):
    def git(*arguments):
        return subprocess.run(('git', *arguments), cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    if git('rev-parse', 'HEAD') != revision or git('status', '--porcelain'):
        raise RuntimeError('campaign must execute from its exact clean source clone')
    tracked = git('ls-files').splitlines()
    records = [{'path': path, 'sha256': digest_file(ROOT/path)} for path in tracked
               if path.endswith(('.py', '.json')) or path == 'pyproject.toml']
    gate_path = ROOT / 'thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_F_GENERATION_GATE.json'
    old = json.loads(gate_path.read_text())
    frozen_files = old['source_manifest']['files']
    if (len(frozen_files) != 117 or old['source_revision'] != '7e3b9a3be22d991ec2e588cbd4a53a9d4d2d27d7'
        or old['source_manifest_digest'] != '002d8afd7641dfa2e78269d195a5375c37b3f8aa547eb86d574dc439e7858662'):
        raise RuntimeError('operating-decision frozen manifest identity differs')
    mismatches = [item['path'] for item in frozen_files
                  if digest_file(ROOT/item['path']) != item['sha256']]
    if mismatches:
        raise RuntimeError('operating-decision frozen source changed: ' + ', '.join(mismatches))
    return {'source_commit': revision, 'source_files': records,
            'source_digest': canonical_digest(records), 'operating_decision_frozen_files_unchanged': len(frozen_files)}


def geometry_assessment(protocol):
    return dict(protocol['geometry_3d'], status='assessed',
        basis='Translation-invariant coefficients, data and boundaries admit a y-independent solution. '
              'Uniqueness of the linear heat equation with positive Robin contact makes that extrusion '
              'the 3D solution under these assumptions. This argument does not apply to finite-depth '
              'contact patches, localized heating, legs or side losses.',
        hardware_geometry_available=False, numerical_3d_run=False)


def runtime_context(protocol):
    required = protocol['execution']['runtime_requirements']
    actual = {name: importlib.metadata.version(name) for name in required['packages']}
    if (platform.python_implementation() != required['implementation']
        or platform.python_version() != required['python'] or actual != required['packages']):
        raise RuntimeError('runtime differs from the frozen CPython/dependency versions; no data generated')
    return dict(actual, python=sys.version, implementation=platform.python_implementation(),
                executable=sys.executable, platform=platform.platform())


def run_spatial_campaign(output_directory, *, source_commit, ci_evidence):
    """Generate once after provenance checks; all scientific limits are frozen."""
    protocol = validated_protocol()
    runtime = runtime_context(protocol)
    ci_path = Path(ci_evidence)
    ci = verify_ci(json.loads(ci_path.read_text()), source_commit)
    context = source_context(source_commit)
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = {'campaign': CAMPAIGN, 'status': 'started', 'started_utc': now(),
             'source_commit': source_commit, 'protocol_digest': canonical_digest(protocol)}
    def progress(event):
        state.update(last_progress=event, updated_utc=now())
        write_json(output/'state.json', state)
        print(json.dumps(dict(utc=now(), **event), allow_nan=False), flush=True)
    manifest = dict(context, protocol=protocol, protocol_digest=canonical_digest(protocol),
                    exact_source_ci=ci, ci_evidence_sha256=digest_file(ci_path),
                    runtime=runtime,
                    started_utc=now(), campaign=CAMPAIGN)
    write_json(output/'manifest.json', manifest)
    (output/'exact-source-ci.json').write_bytes(ci_path.read_bytes())
    write_json(output/'state.json', state)
    result = {'campaign': CAMPAIGN, 'source_commit': source_commit,
              'protocol_digest': canonical_digest(protocol), 'stages': {}}
    def interrupted(signum, frame):
        raise SystemExit(128+signum)
    previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
    try:
        from .pinn_spatial_fresh_1d import run_fresh_1d_validation
        from .pinn_spatial_2d import run_planar_benchmark
        state['status'] = 'fresh_1d_running'
        progress({'stage': 'fresh_1d', 'event': 'started'})
        run_fresh_1d_validation(output/'fresh-1d')
        # The 1D Python return value includes dataclasses; use its saved portable record.
        fresh = json.loads((output/'fresh-1d/result.json').read_text())
        if source_context(source_commit)['source_digest'] != context['source_digest']:
            raise RuntimeError('source changed during fresh 1D validation; stop before 2D')
        result['stages']['fresh_1d'] = fresh
        write_json(output/'partial-results.json', result)
        if not fresh['numerical_reference_passed']:
            result['status'] = 'blocked_1d_reference'
        else:
            state['status'] = 'bounded_2d_running'
            progress({'stage': 'bounded_2d', 'event': 'started'})
            planar = run_planar_benchmark(output/'bounded-2d', progress=progress)
            result['stages']['bounded_2d'] = planar
            result['status'] = 'completed' if planar['status'] == 'completed' else 'stopped_2d_' + planar['status']
        if source_context(source_commit)['source_digest'] != context['source_digest']:
            raise RuntimeError('source changed during campaign; preserve evidence without acceptance')
        result['stages']['geometry_3d_assessment'] = geometry_assessment(protocol)
        result['completed_utc'] = now()
        write_json(output/'result.json', result)
        from ..reports.pinn_spatial_validation import render_report
        render_report(output, result)
        records = [{'path': str(path.relative_to(output)), 'bytes': path.stat().st_size,
                    'sha256': digest_file(path)} for path in sorted(output.rglob('*'))
                   if path.is_file() and path.name not in ('state.json', 'evidence-index.json')]
        write_json(output/'evidence-index.json', {'files': records, 'digest': canonical_digest(records)})
        state['status'] = result['status']
        progress({'stage': 'campaign', 'event': result['status']})
        return result
    except BaseException as error:
        state['status'] = 'failed_preserve_evidence'
        state['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
        state['updated_utc'] = now()
        write_json(output/'state.json', state)
        result.update(status='failed_preserve_evidence', failure=state['error'])
        write_json(output/'partial-results.json', result)
        partial = [{'path': str(path.relative_to(output)), 'bytes': path.stat().st_size,
                    'sha256': digest_file(path)} for path in sorted(output.rglob('*'))
                   if path.is_file() and path.name not in ('state.json', 'incomplete-evidence-index.json')]
        write_json(output/'incomplete-evidence-index.json', {'files': partial,
            'digest': canonical_digest(partial), 'incomplete': True, 'terminal_status': state['status']})
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


def validate_evidence(output_directory):
    """Reload artifact hashes; this is integrity checking, not scientific recomputation."""
    output = Path(output_directory)
    index = json.loads((output/'evidence-index.json').read_text())
    if canonical_digest(index['files']) != index['digest']:
        raise ValueError('evidence index digest differs')
    names = [record['path'] for record in index['files']]
    if not {'manifest.json', 'result.json', 'exact-source-ci.json', 'REPORT.md'}.issubset(names):
        raise ValueError('required campaign evidence is missing')
    if len(set(names)) != len(names):
        raise ValueError('duplicate evidence path')
    for record in index['files']:
        path = (output/record['path']).resolve()
        if not path.is_relative_to(output.resolve()) or not path.is_file():
            raise ValueError('invalid evidence path')
        if path.stat().st_size != record['bytes'] or digest_file(path) != record['sha256']:
            raise ValueError('evidence changed: ' + record['path'])
    actual = {str(path.relative_to(output)) for path in output.rglob('*') if path.is_file()
              and path.name not in ('state.json', 'evidence-index.json')}
    if set(names) != actual:
        raise ValueError('evidence index does not cover the artifact set')
    return {'validated_files': len(index['files']), 'evidence_digest': index['digest']}
