"""Execute or validate Phase D2 offset analysis without simulation or refitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Optional, Sequence

from ..studies.operating_decision_prospective_phase_d2_offsets import (
    build_phase_d2_offset_analysis,
    save_phase_d2_offset_artifacts,
    validate_phase_d2_offset_analysis,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _clean_committed_revision(repository_root: Path) -> str:
    root = repository_root.expanduser().resolve(strict=True)
    if root != PROJECT_ROOT.resolve():
        raise ValueError("the Phase D2 repository root must match imported source")
    status = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError("Phase D2 requires a clean committed working tree")
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(revision) != 40:
        raise ValueError("Phase D2 could not resolve a full Git commit")
    return revision


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute-offset-analysis", action="store_true")
    mode.add_argument("--validate-offset-analysis", type=Path)
    parser.add_argument("--phase-d1-archive", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--repository-root", type=Path, default=PROJECT_ROOT)
    arguments = parser.parse_args(argv)

    revision = _clean_committed_revision(arguments.repository_root)
    if arguments.execute_offset_analysis:
        if arguments.output_directory is None:
            parser.error("--output-directory is required for D2 execution")
        payload = build_phase_d2_offset_analysis(
            phase_d1_archive_path=arguments.phase_d1_archive,
            repository_root=arguments.repository_root,
            analysis_source_revision=revision,
        )
        saved = save_phase_d2_offset_artifacts(
            payload,
            output_directory=arguments.output_directory,
        )
        estimate = payload["offset_estimation"]
        print(f"Phase D2 status: {estimate['status']}")
        print(f"D2 scientific digest: {payload['scientific_result_digest']}")
        print(f"complete JSON: {saved.json_path}")
        print(f"compact report: {saved.report_path}")
        print(f"SHA-256 manifest: {saved.hash_path}")
        print(f"JSON SHA-256: {saved.json_sha256}")
        return

    if arguments.output_directory is not None:
        parser.error("--output-directory applies only to D2 execution")
    result_path = arguments.validate_offset_analysis.expanduser().resolve(strict=True)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    validated = validate_phase_d2_offset_analysis(
        payload,
        phase_d1_archive_path=arguments.phase_d1_archive,
        repository_root=arguments.repository_root,
    )
    if validated["analysis_source_revision"] != revision:
        raise ValueError("Phase D2 result was not generated at clean HEAD")
    print("Phase D2 offset analysis: VALID")
    print(f"status: {validated['offset_estimation']['status']}")
    print(f"scientific result digest: {validated['scientific_result_digest']}")


if __name__ == "__main__":
    main()
