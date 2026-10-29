"""Record a frozen cutoff-audit declaration and sealed saved results; never solve."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from thermotwin.studies import pinn_spatial_cutoff_audit as audit

PLAN = ROOT/'thermotwin/PINN_SPATIAL_CUTOFF_AUDIT_PLAN.json'
RESULT = ROOT/'thermotwin/PINN_SPATIAL_CUTOFF_AUDIT_RESULT.json'


def read(path):
    return json.loads(Path(path).read_text())


def reference(path):
    path = Path(path).resolve()
    return {'path': str(path.relative_to(ROOT)),
            'sha256': audit.common.digest_file(path), 'bytes': path.stat().st_size}


def write_new(path, value):
    """Publish complete bytes atomically without replacing any existing path."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+'.', suffix='.pending', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(audit.common._json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise RuntimeError('Refusing to replace an existing scientific record: '+str(path)) from error
    finally:
        if temporary is not None:
            temporary.unlink()


def record_target(path, directory, plan):
    target = path.parent.resolve()/path.name
    if not target.is_relative_to(ROOT.resolve()):
        raise RuntimeError('Compact records must remain inside the project')
    if target.is_relative_to(directory) or any(target.is_relative_to(Path(name).resolve()) for name in plan['protected_trees']):
        raise RuntimeError('Compact records may not alter sealed or protected evidence')


def compact_declaration(plan, directory):
    compact = {key: value for key, value in plan.items() if key != 'protected_trees'}
    compact.update(authoritative_plan=reference(directory/'plan.json'),
        protected_trees={path: {'file_count': len(records),
            'manifest_sha256': audit.common.canonical_digest(records)}
            for path, records in plan['protected_trees'].items()}, compact_record_only=True)
    return compact


def freeze(directory):
    audit._one_thread()
    directory = Path(directory).resolve()
    plan = read(directory/'plan.json')
    audit.verify_plan(plan, directory)
    if read(directory/'state.json') != {'status': 'prepared',
            'plan_sha256': audit.common.digest_file(directory/'plan.json')}:
        raise RuntimeError('The compact declaration must precede audit execution')
    record_target(PLAN, directory, plan)
    compact = dict(compact_declaration(plan, directory), recorded_utc=datetime.now(timezone.utc).isoformat())
    write_new(PLAN, compact)
    return compact


def point_key(value):
    if (type(value.get('case')) is not str or type(value.get('kind')) is not str
        or type(value.get('sign')) is not int or type(value.get('radius')) not in (int, float)
        or not math.isfinite(value['radius'])):
        raise RuntimeError('Malformed declared opportunity identity')
    return tuple(value[key] for key in ('case', 'kind', 'radius', 'sign'))


def retained_opportunities(directory, plan, result):
    """Authenticate the saved fourteen rows, including interrupted attempts."""
    cases = result.get('cases')
    if not isinstance(cases, list):
        raise RuntimeError('Audit cases must be an explicit list')
    saved, expected_cases = {}, list(audit.CASES)
    for row in cases:
        case = row.get('case')
        if case in saved or case not in expected_cases:
            raise RuntimeError('Duplicate or unexpected audit case')
        saved[case] = row
    if list(saved) != expected_cases:
        raise RuntimeError('Both declared audit cases must be retained in order')
    rows, progress_expected, attempted, solved, failed, not_run = [], [], 0, 0, 0, 0
    for case in expected_cases:
        row = saved[case]
        if row.get('status') not in ('completed', 'failed', 'not_run'):
            raise RuntimeError('Unknown saved case status')
        if not (directory/(case+'.json')).is_file() or read(directory/(case+'.json')) != row:
            raise RuntimeError('Saved case file differs from the aggregate result')
        samples = row.get('samples')
        if not isinstance(samples, list):
            raise RuntimeError('Every case must explicitly retain its opportunities')
        expected = [point_key(point) for point in plan['protocol'] if point['case'] == case]
        identities = [point_key(value) for value in samples]
        if len(set(identities)) != len(identities):
            raise RuntimeError('Duplicate saved audit opportunity')
        if any(key not in expected for key in identities):
            raise RuntimeError('Unexpected saved audit opportunity')
        if identities != expected:
            raise RuntimeError('All seven declared opportunities must be retained in order')
        opportunities = []
        for value in samples:
            status = value.get('status')
            if status not in ('solved', 'failed', 'not_run'):
                raise RuntimeError('Unknown saved opportunity status')
            if status == 'not_run':
                if (value.get('readout_attempted') is not False or value.get('attempt_before') is not None
                    or value.get('attempt_after') is not None or value.get('solve_number') is not None):
                    raise RuntimeError('An unrun opportunity cannot claim a readout attempt')
                not_run += 1
            else:
                before, after = value.get('attempt_before'), value.get('attempt_after')
                if (type(before) is not int or type(after) is not int or before != attempted
                    or after-before not in (0, 1) or after > 14
                    or value.get('readout_attempted') is not bool(after-before)):
                    raise RuntimeError('Saved per-opportunity readout accounting is inconsistent')
                if ((after > before and value.get('solve_number') != after)
                    or (after == before and value.get('solve_number') is not None)
                    or (status == 'solved' and after != before+1)
                    or ('attempted_readout_solves' in value and value['attempted_readout_solves'] != after)):
                    raise RuntimeError('Saved solve identity or cumulative count is inconsistent')
                attempted = after
                solved += status == 'solved'
                failed += status == 'failed'
                progress_expected.append(value)
            opportunities.append({key: item for key, item in value.items() if key not in ('singular_values', 'system')})
        if row['status'] == 'completed' and (any(value['status'] != 'solved' for value in samples)
            or row.get('original_state_restored_exactly') is not True):
            raise RuntimeError('A completed case requires seven solved opportunities and exact restoration')
        if row['status'] == 'not_run' and any(value['status'] != 'not_run' for value in samples):
            raise RuntimeError('An unrun case contains attempted opportunities')
        rows.append(dict({key: value for key, value in row.items() if key != 'samples'},
            samples=opportunities, full_saved_row=reference(directory/(case+'.json'))))
    progress_path = directory/'progress.jsonl'
    if not progress_path.is_file():
        raise RuntimeError('The sealed progress account is missing')
    progress = [json.loads(line) for line in progress_path.read_text().splitlines() if line.strip()]
    if progress != progress_expected:
        raise RuntimeError('Progress differs from the complete retained per-opportunity account')
    counters = {key: result.get(key) for key in ('attempted_readout_solves', 'successful_readout_solves', 'maximum_readout_solves')}
    if (any(type(value) is not int for value in counters.values())
        or counters != {'attempted_readout_solves': attempted, 'successful_readout_solves': solved, 'maximum_readout_solves': 14}
        or solved+failed+not_run != 14):
        raise RuntimeError('Aggregate solve counts disagree with saved opportunities')
    return rows, {'declared_points': 14, 'solved': solved, 'failed': failed, 'not_run': not_run}


def record(directory):
    audit._one_thread()
    directory = Path(directory).resolve()
    plan, result, state, index = (read(directory/name) for name in
        ('plan.json', 'result.json', 'state.json', 'artifact-index.json'))
    audit.verify_plan(plan, directory)
    declaration = read(PLAN)
    if (type(declaration.get('recorded_utc')) is not str
        or {key: value for key, value in declaration.items() if key != 'recorded_utc'} != compact_declaration(plan, directory)):
        raise RuntimeError('The result does not match the compact prospective declaration')
    if (state.get('status') not in ('completed', 'failed')
        or state['status'] != result['status']
        or state.get('plan_sha256') != audit.common.digest_file(directory/'plan.json')
        or result.get('plan_sha256') != state['plan_sha256']
        or state.get('artifact_index_sha256') != audit.common.digest_file(directory/'artifact-index.json')):
        raise RuntimeError('Saved audit completion identities disagree')
    files = {str(path.relative_to(directory)) for path in directory.rglob('*')
             if path.is_file() and path not in (directory/'state.json', directory/'artifact-index.json')}
    if set(index) != files:
        raise RuntimeError('The sealed diagnostic file set changed')
    for name, expected in index.items():
        if Path(name).is_absolute() or not (directory/name).resolve().is_relative_to(directory):
            raise RuntimeError('The artifact index escapes the diagnostic directory')
        actual = reference(directory/name)
        if {key: actual[key] for key in ('sha256', 'bytes')} != expected:
            raise RuntimeError('A sealed diagnostic artifact changed: '+name)
    rows, denominator = retained_opportunities(directory, plan, result)
    if (result.get('training_only') is not True or result.get('inputs_and_sources_preserved') is not True
        or type(result['neural_training_updates']) is not int or result['neural_training_updates'] != 0
        or type(result['new_checkpoints_written']) is not int or result['new_checkpoints_written'] != 0
        or result['reference_fields_used'] is not False
        or result['independent_diagnostics_used'] is not False
        or result.get('independent_diagnostics_used_for_selection') is not False
        or result['automatic_followup_launched'] is not False):
        raise RuntimeError('Audit scope or denominator changed')
    if result['status'] == 'completed' and (denominator['solved'] != 14
        or result['attempted_readout_solves'] != 14
        or result['successful_readout_solves'] != 14
        or any(row.get('status') != 'completed' or row.get('original_state_restored_exactly') is not True for row in rows)):
        raise RuntimeError('A completed audit needs all fourteen solves and exact restoration')
    compact = {key: value for key, value in result.items() if key != 'cases'}
    compact.update(recorded_utc=datetime.now(timezone.utc).isoformat(),
        declaration=reference(PLAN), raw_result=reference(directory/'result.json'),
        artifact_index=reference(directory/'artifact-index.json'),
        preservation_verified=True,
        denominator=denominator,
        cases=rows)
    record_target(RESULT, directory, plan)
    write_new(RESULT, compact)
    return compact


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'record'))
    parser.add_argument('--directory', required=True)
    args = parser.parse_args()
    value = (freeze if args.command == 'freeze' else record)(Path(args.directory).resolve())
    print(json.dumps({'command': args.command, 'status': value.get('status', 'prepared')}, allow_nan=False))
