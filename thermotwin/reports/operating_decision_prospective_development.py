"""Prepare and rehearse the prospective operating-decision Phase D protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Optional, Sequence, Tuple

from ..studies.operating_decision_prospective_phase_d import (
    PHASE_D_NUMERICAL_SOURCE_PATHS,
    PHASE_D_REHEARSAL_PARTITION,
    prospective_phase_d_protocol_payload,
    run_phase_d_disposable_rehearsal,
    save_phase_d_rehearsal_artifacts,
    validate_phase_d_rehearsal_archive,
)
from ..studies.operating_decision_provenance import create_source_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _require_project_root(repository_root: Path) -> Path:
    root = repository_root.expanduser().resolve()
    if root != PROJECT_ROOT.resolve():
        raise ValueError(
            "the Phase D repository root must match its imported source tree"
        )
    return root


def _require_clean_head(repository_root: Path, source_revision: str) -> None:
    root = _require_project_root(repository_root)
    status = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise ValueError("Phase D requires a clean committed working tree")
    observed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(observed) != 40:
        raise ValueError("Phase D could not resolve a full Git commit")
    if observed != source_revision:
        raise ValueError("Phase D source revision does not equal clean HEAD")


def _committed_source_revision(repository_root: Path) -> str:
    root = _require_project_root(repository_root)
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require_clean_head(root, revision)
    return revision


def _output_paths(
    json_path: Path,
    report_path: Optional[Path],
    hash_path: Optional[Path],
) -> Tuple[Path, Path, Path]:
    json_path = json_path.expanduser().resolve()
    report = (
        report_path.expanduser().resolve()
        if report_path is not None
        else json_path.with_suffix(".txt")
    )
    hashes = (
        hash_path.expanduser().resolve()
        if hash_path is not None
        else json_path.with_suffix(".sha256")
    )
    if len({json_path, report, hashes}) != 3:
        raise ValueError("Phase D JSON, report, and hash paths must be distinct")
    return json_path, report, hashes


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--print-protocol",
        action="store_true",
        help="print the source-bound Phase D protocol without opening a partition",
    )
    mode.add_argument(
        "--execute-disposable-rehearsal",
        action="store_true",
        help=(
            "run only the one-block disposable archive rehearsal "
            f"{PHASE_D_REHEARSAL_PARTITION}"
        ),
    )
    mode.add_argument(
        "--validate-rehearsal",
        type=Path,
        help="load and strictly replay an existing disposable rehearsal JSON",
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--hashes", type=Path)
    parser.add_argument("--repository-root", type=Path, default=PROJECT_ROOT)
    arguments = parser.parse_args(argv)

    root = _require_project_root(arguments.repository_root)
    revision = _committed_source_revision(root)
    if arguments.print_protocol:
        if any(
            value is not None
            for value in (arguments.json, arguments.report, arguments.hashes)
        ):
            parser.error("output paths are valid only for the disposable rehearsal")
        manifest = create_source_manifest(root, PHASE_D_NUMERICAL_SOURCE_PATHS)
        payload = prospective_phase_d_protocol_payload(
            source_revision=revision,
            source_manifest_digest=manifest.digest,
        )
        print(json.dumps(payload, sort_keys=True, indent=2))
        return

    if arguments.validate_rehearsal is not None:
        if any(
            value is not None
            for value in (arguments.json, arguments.report, arguments.hashes)
        ):
            parser.error("output paths cannot be used while validating an archive")
        payload = json.loads(
            arguments.validate_rehearsal.expanduser().resolve().read_text(
                encoding="utf-8"
            )
        )
        validated = validate_phase_d_rehearsal_archive(
            payload,
            repository_root=root,
        )
        if validated["source_revision"] != revision:
            raise ValueError("rehearsal archive was not generated at clean HEAD")
        print("Phase D disposable rehearsal archive: VALID")
        print(f"source revision: {revision}")
        print(f"protocol digest: {validated['protocol_digest']}")
        return

    if arguments.json is None:
        parser.error("--json is required for --execute-disposable-rehearsal")
    json_path, report_path, hash_path = _output_paths(
        arguments.json,
        arguments.report,
        arguments.hashes,
    )
    _require_clean_head(root, revision)
    result = run_phase_d_disposable_rehearsal(
        repository_root=root,
        source_revision=revision,
    )
    _require_clean_head(root, revision)
    saved = save_phase_d_rehearsal_artifacts(
        result,
        json_path=json_path,
        report_path=report_path,
        hash_path=hash_path,
        repository_root=root,
    )
    print(f"complete JSON: {saved.json_path}")
    print(f"compact report: {saved.report_path}")
    print(f"SHA-256 manifest: {saved.hash_path}")
    print(f"archive bytes: {saved.archive_size_bytes}")
    print(f"JSON SHA-256: {saved.json_sha256}")


if __name__ == "__main__":
    main()
