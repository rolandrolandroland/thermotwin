"""Archive complete corrected-replication diagnostics outside Git history.

This postprocessor is deliberately outside the numerical source manifest. It
does not generate, alter, or reinterpret scientific rows. It validates the
saved diagnostic payload, creates a deterministic content-addressed gzip file,
and writes a compact manifest suitable for committing to Git.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


MANIFEST_SCHEMA_VERSION = 1
MANIFEST_TYPE = "thermotwin_operating_decision_external_evidence"
DIAGNOSTIC_SCHEMA_VERSION = 1
EXPECTED_RECORDS_PER_BLOCK = 12
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_SHA1_PATTERN = re.compile(r"[0-9a-f]{40}")
COMMAND_BY_PARTITION = {
    "r2_parent_rehearsal": "rehearse-parent",
    "r2_guard_development": "freeze-guard",
    "r2_guard_calibration": "freeze-guard",
    "r2_reserved_evaluation": "evaluate-reserved",
}


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _load_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def recompute_diagnostic_evidence_digest(payload: Mapping[str, Any]) -> str:
    """Recompute the deterministic digest used by the frozen diagnostic writer."""

    evidence = copy.deepcopy(dict(payload))
    evidence.pop("deterministic_evidence_digest", None)
    records = evidence.get("records")
    if not isinstance(records, list):
        raise ValueError("diagnostics records must be a list")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("diagnostic record must be an object")
        saved = record.get("saved_decision")
        if not isinstance(saved, dict):
            raise ValueError("diagnostic record lacks saved_decision")
        if "decision_computation_seconds" not in saved:
            raise ValueError("diagnostic record lacks computation timer")
        saved.pop("decision_computation_seconds")
    return _sha256_bytes(_canonical_json_bytes(evidence))


def validate_diagnostic_payload(
    payload: Mapping[str, Any],
    *,
    command: str,
) -> dict[str, Any]:
    """Validate one complete corrected-replication diagnostic payload."""

    if payload.get("schema_version") != DIAGNOSTIC_SCHEMA_VERSION:
        raise ValueError("unsupported corrected diagnostic schema")
    partition = payload.get("partition")
    if not isinstance(partition, dict) or set(partition) != {
        "campaign",
        "name",
        "paired_blocks_per_family",
    }:
        raise ValueError("diagnostic partition identity is malformed")
    campaign = partition["campaign"]
    name = partition["name"]
    block_count = partition["paired_blocks_per_family"]
    if not isinstance(campaign, str) or not campaign.strip():
        raise ValueError("diagnostic campaign is malformed")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("diagnostic partition name is malformed")
    if not isinstance(block_count, int) or isinstance(block_count, bool) or block_count <= 0:
        raise ValueError("diagnostic block count is malformed")
    if COMMAND_BY_PARTITION.get(name) != command:
        raise ValueError("partition does not belong to the declared command")

    records = payload.get("records")
    expected_records = block_count * EXPECTED_RECORDS_PER_BLOCK
    if not isinstance(records, list) or len(records) != expected_records:
        raise ValueError("diagnostic record count is incomplete")
    audit = payload.get("stream_audit")
    if not isinstance(audit, dict):
        raise ValueError("diagnostic stream audit is missing")
    if audit.get("ok") is not True or audit.get("unintended_reuses") != []:
        raise ValueError("diagnostic stream audit is not clean")
    for key in ("use_count", "unique_key_count", "unique_seed_count"):
        value = audit.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("diagnostic stream-audit counts are malformed")
    if audit["unique_key_count"] != audit["unique_seed_count"]:
        raise ValueError("diagnostic keys and seeds are not one-to-one")

    claimed_digest = payload.get("deterministic_evidence_digest")
    if not isinstance(claimed_digest, str) or not SHA256_PATTERN.fullmatch(
        claimed_digest
    ):
        raise ValueError("diagnostic evidence digest is malformed")
    actual_digest = recompute_diagnostic_evidence_digest(payload)
    if actual_digest != claimed_digest:
        raise ValueError("diagnostic evidence digest does not match its rows")
    for key in ("generator_version", "random_stream_protocol_version"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ValueError(f"diagnostic {key} is malformed")
    return {
        "campaign": campaign,
        "name": name,
        "paired_blocks_per_family": block_count,
        "record_count": len(records),
        "deterministic_evidence_digest": claimed_digest,
        "generator_version": payload["generator_version"],
        "random_stream_protocol_version": payload[
            "random_stream_protocol_version"
        ],
        "stream_audit": {
            key: audit[key]
            for key in ("use_count", "unique_key_count", "unique_seed_count")
        },
    }


def _current_commit(repository_root: Path) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repository_root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _validate_release_location(base_url: str, release_tag: str) -> str:
    parsed = urlparse(base_url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("retrieval base URL must be an HTTPS URL without query or fragment")
    if not release_tag or release_tag != release_tag.strip() or "/" in release_tag:
        raise ValueError("release tag must be one portable path component")
    normalized = base_url.rstrip("/")
    if normalized.rsplit("/", 1)[-1] != release_tag:
        raise ValueError("retrieval base URL must end with the release tag")
    return normalized


def _validate_canonical_arguments(arguments: Sequence[str]) -> list[str]:
    if not arguments:
        raise ValueError("canonical command arguments are required")
    result = []
    for value in arguments:
        if not isinstance(value, str) or not value or "\n" in value or "\r" in value:
            raise ValueError("canonical command arguments must be nonempty single lines")
        if value.startswith("/") or value.startswith("~"):
            raise ValueError("canonical command arguments must not contain local paths")
        result.append(value)
    return result


def _deterministic_gzip(source: Path, archive_directory: Path, partition: str) -> Path:
    archive_directory.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=".thermotwin-evidence-",
            suffix=".tmp",
            dir=archive_directory,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            with source.open("rb") as raw, gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=temporary,
                mtime=0,
            ) as compressed:
                shutil.copyfileobj(raw, compressed, length=1024 * 1024)
        temporary_path = Path(temporary_name)
        archive_digest, _ = _sha256_path(temporary_path)
        destination = archive_directory / f"{partition}-{archive_digest}.json.gz"
        if destination.exists():
            raise FileExistsError(f"evidence archive already exists: {destination}")
        os.link(temporary_path, destination)
        temporary_path.unlink()
        temporary_name = None
        return destination
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _verify_round_trip(source: Path, archive: Path) -> None:
    raw_digest, raw_size = _sha256_path(source)
    expanded_digest = hashlib.sha256()
    expanded_size = 0
    with gzip.open(archive, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            expanded_digest.update(chunk)
            expanded_size += len(chunk)
    if expanded_digest.hexdigest() != raw_digest or expanded_size != raw_size:
        raise ValueError("evidence archive failed gzip round-trip verification")


def archive_diagnostics(
    diagnostic_paths: Sequence[Path | str],
    *,
    archive_directory: Path | str,
    manifest_path: Path | str,
    repository_root: Path | str,
    command: str,
    source_commit: str,
    canonical_arguments: Sequence[str],
    release_tag: str,
    retrieval_base_url: str,
) -> dict[str, Any]:
    """Validate, archive, and manifest one corrected scientific command."""

    if command not in set(COMMAND_BY_PARTITION.values()):
        raise ValueError("unsupported corrected replication command")
    if not GIT_SHA1_PATTERN.fullmatch(source_commit):
        raise ValueError("source commit must be a full lowercase Git SHA-1")
    root = Path(repository_root).expanduser().resolve(strict=True)
    if _current_commit(root) != source_commit:
        raise ValueError("source commit does not equal repository HEAD")
    arguments = _validate_canonical_arguments(canonical_arguments)
    base_url = _validate_release_location(retrieval_base_url, release_tag)
    archive_root = Path(archive_directory).expanduser().resolve()
    manifest_destination = Path(manifest_path).expanduser().resolve()
    if manifest_destination.exists():
        raise FileExistsError(f"evidence manifest already exists: {manifest_destination}")
    try:
        manifest_destination.relative_to(root)
    except ValueError as error:
        raise ValueError("evidence manifest must be inside the repository") from error

    sources = tuple(Path(path).expanduser().resolve(strict=True) for path in diagnostic_paths)
    if not sources:
        raise ValueError("at least one diagnostic file is required")
    if len(set(sources)) != len(sources):
        raise ValueError("diagnostic files must be unique")

    entries = []
    seen_partitions = set()
    for source in sources:
        payload = _load_object(source)
        validated = validate_diagnostic_payload(payload, command=command)
        partition_name = validated["name"]
        partition_key = (validated["campaign"], partition_name)
        if partition_key in seen_partitions:
            raise ValueError("diagnostic partitions must be unique")
        seen_partitions.add(partition_key)
        archive = _deterministic_gzip(source, archive_root, partition_name)
        _verify_round_trip(source, archive)
        raw_digest, raw_size = _sha256_path(source)
        archive_digest, archive_size = _sha256_path(archive)
        entries.append(
            {
                "partition": {
                    key: validated[key]
                    for key in ("campaign", "name", "paired_blocks_per_family")
                },
                "diagnostic_schema_version": payload["schema_version"],
                "generator_version": validated["generator_version"],
                "random_stream_protocol_version": validated[
                    "random_stream_protocol_version"
                ],
                "deterministic_evidence_digest": validated[
                    "deterministic_evidence_digest"
                ],
                "record_count": validated["record_count"],
                "stream_audit": validated["stream_audit"],
                "raw": {
                    "logical_name": source.name,
                    "sha256": raw_digest,
                    "byte_count": raw_size,
                    "media_type": "application/json",
                },
                "archive": {
                    "asset_name": archive.name,
                    "sha256": archive_digest,
                    "byte_count": archive_size,
                    "format": "gzip",
                    "compression_level": 9,
                    "mtime": 0,
                    "media_type": "application/gzip",
                },
                "retrieval": {
                    "release_tag": release_tag,
                    "url": f"{base_url}/{archive.name}",
                },
            }
        )
    entries.sort(key=lambda item: (item["partition"]["campaign"], item["partition"]["name"]))

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_type": MANIFEST_TYPE,
        "producer": {
            "module": "thermotwin.operating_decision_replication",
            "command": command,
            "canonical_arguments": arguments,
            "source_commit": source_commit,
        },
        "evidence": entries,
    }
    manifest["manifest_digest"] = _sha256_bytes(_canonical_json_bytes(manifest))
    manifest_destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with manifest_destination.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", action="append", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument(
        "--command",
        choices=sorted(set(COMMAND_BY_PARTITION.values())),
        required=True,
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--canonical-argument", action="append", default=[])
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--retrieval-base-url", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    manifest = archive_diagnostics(
        arguments.diagnostics,
        archive_directory=arguments.archive_dir,
        manifest_path=arguments.manifest,
        repository_root=arguments.repository_root,
        command=arguments.command,
        source_commit=arguments.source_commit,
        canonical_arguments=arguments.canonical_argument,
        release_tag=arguments.release_tag,
        retrieval_base_url=arguments.retrieval_base_url,
    )
    for item in manifest["evidence"]:
        print(
            f"archived {item['partition']['name']}: "
            f"{item['archive']['asset_name']}"
        )
    print(f"manifest digest: {manifest['manifest_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
