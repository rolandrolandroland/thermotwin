"""Phase F complete-rule replay and separately gated final reserved comparison."""

import argparse
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _clean_revision(root: Path) -> str:
    if root.resolve(strict=True) != PROJECT_ROOT.resolve():
        raise ValueError('Phase F root must match the imported source clone')
    status = subprocess.run(('git', 'status', '--porcelain'), cwd=root,
                            check=True, capture_output=True, text=True).stdout.strip()
    if status:
        raise ValueError('Phase F requires a clean committed source clone')
    return subprocess.run(('git', 'rev-parse', 'HEAD'), cwd=root,
                          check=True, capture_output=True, text=True).stdout.strip()


def _validate_launch_environment() -> None:
    # Read the committed rule without importing NumPy/SciPy first. The full rule
    # and its immutable digest are authenticated by the numerical preflight.
    rule = json.loads((PROJECT_ROOT / 'thermotwin' / 'OPERATING_DECISION_PROSPECTIVE_PHASE_F_RULE.json').read_bytes())
    expected = rule['math_environment']
    if any(os.environ.get(k) != v for k, v in expected.items()):
        raise ValueError('launch math environment differs from the frozen Phase F environment')


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--rehearsal', action='store_true')
    mode.add_argument('--execute-reserved', action='store_true')
    mode.add_argument('--validate-evaluation', type=Path)
    parser.add_argument('--repository-root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--calibration-archive', type=Path, required=True)
    parser.add_argument('--calibration-source-root', type=Path, required=True)
    parser.add_argument('--ci-evidence', type=Path)
    parser.add_argument('--generation-gate', type=Path)
    parser.add_argument('--gate-repository-root', type=Path)
    parser.add_argument('--recover-interrupted-lock', action='store_true')
    args = parser.parse_args(argv)
    _validate_launch_environment()
    revision = _clean_revision(args.repository_root)
    if args.execute_reserved and (args.generation_gate is None or args.gate_repository_root is None):
        parser.error('reserved evaluation needs its separate committed generation gate and repository')
    if args.rehearsal and args.ci_evidence is None:
        parser.error('disposable rehearsal requires --ci-evidence for this exact source revision')

    # These imports follow the environment check so a native one-time job
    # cannot silently initialize numerical libraries under different settings.
    from ..studies.operating_decision_prospective_phase_f_execution import (
        phase_f_preflight, run_phase_f_rehearsal, run_phase_f_reserved,
        validate_calibration_evidence, validate_phase_f_archive,
    )
    from ..studies.operating_decision_prospective_phase_f import PHASE_F_REHEARSAL_PARTITION
    from ..studies.operating_decision_prospective_pilot import PROSPECTIVE_RESERVED_PARTITION
    from ..studies.operating_decision_prospective_phase_e_execution import recover_interrupted_execution_lock
    from ..studies.operating_decision_prospective_phase_d_resources import ProcessTreeMonitor, _process_tree_rss_bytes

    if args.recover_interrupted_lock:
        if not (args.rehearsal or args.execute_reserved):
            parser.error('lock recovery is only for interrupted execution')
        for path in recover_interrupted_execution_lock(args.output_directory, source_revision=revision):
            print(f'Preserved interrupted coordinator lock: {path}', flush=True)
    name = PROSPECTIVE_RESERVED_PARTITION if args.execute_reserved else PHASE_F_REHEARSAL_PARTITION
    common = dict(repository_root=args.repository_root, source_revision=revision,
                  output_directory=args.output_directory, calibration_archive=args.calibration_archive,
                  calibration_source_root=args.calibration_source_root)
    # Include original archive validation, child processes, final reload and
    # constructed storage in the whole-workflow resource measurement.
    synchronous_rss = _process_tree_rss_bytes(os.getpid())
    monitor, started, completed = ProcessTreeMonitor(), perf_counter(), False
    try:
        with monitor:
            if args.preflight:
                result = phase_f_preflight(**common, partition_name=name)
                print(json.dumps(result, sort_keys=True, indent=2), flush=True)
            elif args.validate_evaluation:
                validate_calibration_evidence(args.calibration_archive, args.calibration_source_root)
                result = validate_phase_f_archive(args.validate_evaluation, repository_root=args.repository_root)
                name = result['partition']
                print('Phase F saved evaluation: VALID', flush=True)
                print(result['scientific_result_digest'], flush=True)
            elif args.rehearsal:
                result = run_phase_f_rehearsal(**common,
                    ci_evidence=json.loads(args.ci_evidence.read_bytes()),
                    progress=lambda s: print(s, flush=True))
            else:
                result = run_phase_f_reserved(**common, generation_gate_path=args.generation_gate,
                    gate_repository_root=args.gate_repository_root, progress=lambda s: print(s, flush=True))
            completed = True
    finally:
        resources = {'schema_version': 1, 'protocol_version': 'phase_f_resources_v1',
                     'source_revision': revision, 'partition': name, 'execution_completed': completed,
                     'wall_seconds_this_invocation': perf_counter()-started,
                     'process_tree_peak_rss_bytes': monitor.peak_bytes,
                     'process_tree_sample_count': monitor.sample_count,
                     'process_tree_sampling_error': monitor.error,
                     'synchronous_monitor_preflight_rss_bytes': synchronous_rss,
                     'process_tree_limit_bytes': 4 * 1024**3, 'worker_count': 4,
                     'resource_gate_passed': completed and monitor.error is None and monitor.sample_count > 0
                                            and 0 < monitor.peak_bytes <= 4 * 1024**3}
        # Executions retain a final invocation record; read-only modes emit it
        # without rewriting any archive evidence or requiring an output file.
        if args.rehearsal or args.execute_reserved:
            directory = args.output_directory.resolve()
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / ('rehearsal.resources.json' if args.rehearsal else 'evaluation.resources.json')
            with target.open('x', encoding='utf-8') as stream:
                json.dump(resources, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write('\n')
        print(json.dumps({'resources': resources}, sort_keys=True), flush=True)
    if not resources['resource_gate_passed']:
        raise RuntimeError('Phase F resource gate failed; preserve evidence and stop interpretation')
    if args.rehearsal:
        print('Phase F disposable roundtrip, exact scientific replay and constructed archive: PASS', flush=True)
    elif args.execute_reserved:
        print(f'Phase F primary success: {result["primary_analysis"]["primary_success"]}', flush=True)


if __name__ == '__main__':
    main()
