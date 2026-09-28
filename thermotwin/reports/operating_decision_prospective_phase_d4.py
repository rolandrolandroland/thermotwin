"""Execute or validate the frozen Phase D4 draw-count sensitivity check."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter, process_time
from typing import Optional, Sequence

from ..studies.operating_decision_prospective_phase_d4_sensitivity import (
    phase_d4_output_paths,
    run_phase_d4_sensitivity,
    save_phase_d4_artifacts,
    validate_phase_d4_archive,
)
from ..studies.operating_decision_prospective_phase_d_resources import (
    ProcessTreeMonitor,
    _process_tree_rss_bytes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _clean_revision(repository_root: Path) -> str:
    root = repository_root.expanduser().resolve(strict=True)
    if root != PROJECT_ROOT.resolve():
        raise ValueError("the Phase D4 repository root must match imported source")
    status = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError("Phase D4 requires a clean committed working tree")
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(revision) != 40:
        raise ValueError("Phase D4 could not resolve a full source revision")
    return revision


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute-sensitivity", action="store_true")
    mode.add_argument("--validate-sensitivity", type=Path)
    parser.add_argument("--phase-d1-archive", type=Path, required=True)
    parser.add_argument("--phase-d2-result", type=Path, required=True)
    parser.add_argument("--phase-d3-result", type=Path, required=True)
    parser.add_argument("--rule-freeze", type=Path, required=True)
    parser.add_argument("--monitor-preflight", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--repository-root", type=Path, default=PROJECT_ROOT)
    arguments = parser.parse_args(argv)

    revision = _clean_revision(arguments.repository_root)
    common = {
        "phase_d1_archive_path": arguments.phase_d1_archive,
        "phase_d2_result_path": arguments.phase_d2_result,
        "phase_d3_result_path": arguments.phase_d3_result,
        "rule_freeze_path": arguments.rule_freeze,
        "monitor_preflight_path": arguments.monitor_preflight,
        "repository_root": arguments.repository_root,
    }
    if arguments.validate_sensitivity is not None:
        if arguments.output_directory is not None:
            parser.error("--output-directory cannot be used during validation")
        archive_path = arguments.validate_sensitivity.expanduser().resolve(strict=True)
        validated = validate_phase_d4_archive(
            json.loads(archive_path.read_bytes()),
            block_directory=archive_path.parent / "blocks",
            **common,
        )
        if validated["analysis_source_revision"] != revision:
            raise ValueError("Phase D4 archive was not generated at clean HEAD")
        print("Phase D4 draw-count sensitivity: VALID")
        print(f"gate passed: {validated['acceptance']['gate_passed']}")
        print(f"scientific result digest: {validated['scientific_result_digest']}")
        return

    if arguments.output_directory is None:
        parser.error("--output-directory is required for D4 execution")
    paths = phase_d4_output_paths(arguments.output_directory)
    if paths["resources"].exists():
        raise FileExistsError(f"Phase D4 resource record exists: {paths['resources']}")

    # This call is intentionally synchronous and precedes every simulation.
    synchronous_rss = _process_tree_rss_bytes(os.getpid())
    wall_started, cpu_started = perf_counter(), process_time()
    with ProcessTreeMonitor() as monitor:
        payload = run_phase_d4_sensitivity(
            source_revision=revision,
            output_directory=arguments.output_directory,
            progress=print,
            **common,
        )
        saved = save_phase_d4_artifacts(payload, arguments.output_directory)
        loaded = json.loads(saved.json_path.read_bytes())
        validate_phase_d4_archive(
            loaded,
            block_directory=paths["blocks"],
            **common,
        )
    resource = {
        "schema_version": 1,
        "protocol_version": "operating_decision_phase_d4_resource_record_v1",
        "analysis_source_revision": revision,
        "scientific_result_digest": payload["scientific_result_digest"],
        "json_sha256": saved.json_sha256,
        "archive_size_bytes": saved.archive_size_bytes,
        "wall_seconds": perf_counter() - wall_started,
        "coordinator_cpu_seconds": process_time() - cpu_started,
        "synchronous_monitor_preflight_rss_bytes": synchronous_rss,
        "process_tree_peak_rss_bytes": monitor.peak_bytes,
        "process_tree_sample_count": monitor.sample_count,
        "process_tree_sampling_error": monitor.error,
        "worker_count": 4,
        "gate_passed": payload["acceptance"]["gate_passed"],
    }
    paths["resources"].write_text(
        json.dumps(resource, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    if monitor.error is not None or monitor.sample_count == 0:
        raise RuntimeError(
            "Phase D4 process-tree monitor failed during execution: "
            f"{monitor.error}"
        )
    print(f"complete JSON: {saved.json_path}")
    print(f"compact report: {saved.report_path}")
    print(f"SHA-256 manifest: {saved.hash_path}")
    print(f"resource record: {paths['resources']}")
    print(f"JSON SHA-256: {saved.json_sha256}")
    print(f"process-tree peak RSS: {monitor.peak_bytes}")
    print(f"gate passed: {payload['acceptance']['gate_passed']}")
    print(f"next required action: {payload['acceptance']['next_required_action']}")


if __name__ == "__main__":
    main()
