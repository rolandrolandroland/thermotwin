"""Phase E protocol, disposable rehearsal, and gated independent calibration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Sequence

from ..studies.operating_decision_prospective_phase_d_resources import (
    ProcessTreeMonitor, _process_tree_rss_bytes,
)
from ..studies.operating_decision_prospective_phase_e import PHASE_E_REHEARSAL_PARTITION
from ..studies.operating_decision_prospective_phase_e_execution import (
    phase_e_preflight, run_phase_e_calibration, run_phase_e_rehearsal,
    validate_phase_e_calibration_archive, recover_interrupted_execution_lock,
)
from ..studies.operating_decision_prospective_pilot import PROSPECTIVE_CALIBRATION_PARTITION


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _clean_revision(root: Path) -> str:
    root = root.resolve(strict=True)
    if root != PROJECT_ROOT.resolve():
        raise ValueError("Phase E root must match imported source")
    status = subprocess.run(("git", "status", "--porcelain"), cwd=root,
                            check=True, capture_output=True, text=True).stdout.strip()
    if status:
        raise ValueError("Phase E science requires a clean committed source clone")
    return subprocess.run(("git", "rev-parse", "HEAD"), cwd=root,
                          check=True, capture_output=True, text=True).stdout.strip()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--rehearsal", action="store_true")
    mode.add_argument("--execute-calibration", action="store_true")
    mode.add_argument("--validate-calibration", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--generation-gate", type=Path)
    parser.add_argument("--gate-repository-root", type=Path)
    parser.add_argument("--recover-interrupted-lock", action="store_true")
    args = parser.parse_args(argv)
    revision = _clean_revision(args.repository_root)
    if args.recover_interrupted_lock:
        if not (args.execute_calibration or args.rehearsal):
            parser.error("lock recovery is permitted only for a resumed execution")
        recovered = recover_interrupted_execution_lock(args.output_directory, source_revision=revision)
        for path in recovered:
            print(f"Preserved interrupted coordinator lock: {path}")
    partition = (PROSPECTIVE_CALIBRATION_PARTITION if args.execute_calibration
                 or args.generation_gate else PHASE_E_REHEARSAL_PARTITION)
    if args.preflight:
        print(json.dumps(phase_e_preflight(
            repository_root=args.repository_root, source_revision=revision,
            output_directory=args.output_directory, partition_name=partition,
        ), sort_keys=True, indent=2))
        return
    if args.validate_calibration:
        validated = validate_phase_e_calibration_archive(args.validate_calibration,
                                                        repository_root=args.repository_root)
        print("Phase E calibration archive: VALID")
        print(validated["calibration"]["scientific_status"])
        return
    if args.execute_calibration and (args.generation_gate is None or args.gate_repository_root is None):
        parser.error("calibration needs --generation-gate and --gate-repository-root")
    directory = args.output_directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # Sampling permission is checked synchronously before scientific computation.
    synchronous_rss = _process_tree_rss_bytes(os.getpid())
    started = perf_counter()
    monitor = ProcessTreeMonitor()
    execution_completed = False
    try:
        with monitor:
            if args.rehearsal:
                result = run_phase_e_rehearsal(
                    repository_root=args.repository_root, source_revision=revision,
                    output_directory=directory, progress=print,
                )
            else:
                result = run_phase_e_calibration(
                    repository_root=args.repository_root, source_revision=revision,
                    output_directory=directory, generation_gate_path=args.generation_gate,
                    gate_repository_root=args.gate_repository_root, progress=print,
                )
            execution_completed = True
    finally:
        # A failed run still retains its measured resource cost. It cannot pass
        # the execution gate, and its original exception continues to propagate.
        resources = {"schema_version": 1, "protocol_version": "phase_e_resource_record_v2",
                     "source_revision": revision, "partition": partition,
                     "execution_completed": execution_completed,
                     "wall_seconds_this_invocation": perf_counter()-started,
                     "process_tree_peak_rss_bytes": monitor.peak_bytes,
                     "process_tree_sample_count": monitor.sample_count,
                     "process_tree_sampling_error": monitor.error,
                     "synchronous_monitor_preflight_rss_bytes": synchronous_rss,
                     "worker_count": 4, "process_tree_limit_bytes": 4 * 1024**3,
                     "resource_gate_passed": execution_completed and monitor.error is None
                                            and monitor.sample_count > 0
                                            and 0 < monitor.peak_bytes <= 4 * 1024**3}
        target = directory / ("rehearsal.resources.json" if args.rehearsal else "calibration.resources.json")
        with target.open("x", encoding="utf-8") as stream:
            json.dump(resources, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    if not resources["resource_gate_passed"]:
        raise RuntimeError("Phase E resource gate failed; preserve evidence and stop interpretation")
    if args.rehearsal:
        print("Phase E disposable roundtrip, exact scientific replay, and constructed 100-block probe: PASS")
    else:
        print(result["calibration"]["scientific_status"])
        print((directory / "calibration.txt").read_text())


if __name__ == "__main__":
    main()
