"""Command-line entry point for the four-block disposable prospective pilot."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
from typing import Optional, Sequence, Tuple

from ..studies.operating_decision_prospective_pilot import (
    PROSPECTIVE_PILOT_PARTITION,
    run_prospective_draw_count_pilot,
    save_prospective_pilot_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _require_project_root(repository_root: Path) -> Path:
    root = repository_root.expanduser().resolve()
    if root != PROJECT_ROOT.resolve():
        raise ValueError(
            "the prospective pilot repository root must match its imported source tree"
        )
    return root


def _require_clean_head(repository_root: Path, source_revision: str) -> None:
    """Require a clean checkout whose HEAD is the revision being recorded."""

    root = _require_project_root(repository_root)
    status = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise ValueError(
            "the prospective pilot requires a clean committed working tree"
        )
    observed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(observed) != 40:
        raise ValueError("the prospective pilot could not resolve a Git commit")
    if observed != source_revision:
        raise ValueError(
            "the prospective pilot source revision does not equal clean HEAD"
        )


def _committed_source_revision(repository_root: Path) -> str:
    """Return clean HEAD; the disposable evidence must bind committed source."""

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
        raise ValueError("pilot JSON, report, and hash paths must be distinct")
    return json_path, report, hashes


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the predeclared four-block disposable prospective draw-count pilot"
        )
    )
    parser.add_argument(
        "--execute-disposable-pilot",
        action="store_true",
        help=(
            "required acknowledgement; this opens only "
            f"{PROSPECTIVE_PILOT_PARTITION}"
        ),
    )
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--hashes", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--repository-root", type=Path, default=PROJECT_ROOT)
    arguments = parser.parse_args(argv)
    if not arguments.execute_disposable_pilot:
        parser.error("--execute-disposable-pilot is required")
    json_path, report_path, hash_path = _output_paths(
        arguments.json,
        arguments.report,
        arguments.hashes,
    )
    revision = _committed_source_revision(arguments.repository_root)
    _require_clean_head(arguments.repository_root, revision)
    result = run_prospective_draw_count_pilot(
        source_revision=revision,
        workers=arguments.workers,
        progress=print,
    )
    # A pilot can run for hours.  Refuse to publish evidence if the checkout
    # changed after the preflight that supplied its recorded source revision.
    _require_clean_head(arguments.repository_root, revision)
    saved = save_prospective_pilot_artifacts(
        result,
        json_path=json_path,
        report_path=report_path,
        hash_path=hash_path,
    )
    print(f"complete JSON: {saved.json_path}")
    print(f"compact report: {saved.report_path}")
    print(f"SHA-256 manifest: {saved.hash_path}")
    print(f"archive bytes: {saved.archive_size_bytes}")
    print(f"JSON SHA-256: {saved.json_sha256}")


if __name__ == "__main__":
    main()
