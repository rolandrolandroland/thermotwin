"""Execute or validate the frozen Phase D6 internal check."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter, process_time
from typing import Optional, Sequence

from ..studies.operating_decision_prospective_phase_d6_internal_check import (
    phase_d6_internal_check_output_paths,
    phase_d6_internal_check_preflight_payload,
    run_phase_d_development_internal_check,
    save_phase_d6_internal_check_artifacts,
    validate_phase_d6_internal_check_archive,
)
from ..studies.operating_decision_prospective_phase_d_resources import (
    ProcessTreeMonitor,
    _process_tree_rss_bytes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _clean_revision(repository_root: Path) -> str:
    root = repository_root.expanduser().resolve(strict=True)
    if root != PROJECT_ROOT.resolve():
        raise ValueError("the Phase D6 repository root must match imported source")
    status = subprocess.run(
        ("git", "status", "--porcelain"), cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise ValueError("Phase D6 requires a clean committed working tree")
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if len(revision) != 40:
        raise ValueError("Phase D6 could not resolve a full source revision")
    return revision


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--validate", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=PROJECT_ROOT)
    arguments = parser.parse_args(argv)
    revision = _clean_revision(arguments.repository_root)

    if arguments.preflight:
        print(json.dumps(phase_d6_internal_check_preflight_payload(
            repository_root=arguments.repository_root,
            source_revision=revision,
            output_directory=arguments.output_directory,
        ), sort_keys=True, indent=2))
        return

    if arguments.validate is not None:
        archive_path = arguments.validate.expanduser().resolve(strict=True)
        validated = validate_phase_d6_internal_check_archive(
            json.loads(archive_path.read_bytes()),
            repository_root=arguments.repository_root,
            block_directory=archive_path.parent / "blocks",
        )
        if validated["source_revision"] != revision:
            raise ValueError("Phase D6 archive was not generated at clean HEAD")
        print("Phase D6 internal-check archive: VALID")
        print(f"scientific result digest: {validated['scientific_result_digest']}")
        print(f"phase status: {validated['analysis']['phase_d_status']}")
        return

    paths = phase_d6_internal_check_output_paths(arguments.output_directory)
    if paths["resources"].exists():
        raise FileExistsError(f"Phase D6 resource record exists: {paths['resources']}")
    synchronous_rss = _process_tree_rss_bytes(os.getpid())
    wall_started, cpu_started = perf_counter(), process_time()
    with ProcessTreeMonitor() as monitor:
        payload = run_phase_d_development_internal_check(
            repository_root=arguments.repository_root,
            source_revision=revision,
            output_directory=arguments.output_directory,
            progress=print,
        )
        saved = save_phase_d6_internal_check_artifacts(
            payload,
            output_directory=arguments.output_directory,
            repository_root=arguments.repository_root,
        )
    resource = {
        "schema_version": 1,
        "protocol_version": "operating_decision_phase_d6_resource_record_v1",
        "source_revision": revision,
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
    }
    paths["resources"].write_text(
        json.dumps(resource, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(saved.report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
