"""Disposable whole-workflow resource probe for prospective Phase D.

The probe expands one retained rehearsal block into a constructed twenty-block
JSON archive.  The constructed archive is storage and serialization evidence
only: it is never admissible as tuning, calibration, or reserved evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
from threading import Event, Thread
from time import perf_counter, process_time, sleep
from typing import Mapping, Optional

from .operating_decision_prospective_phase_d import (
    PHASE_D_MACHINE_MEMORY_BYTES,
    PHASE_D_PROCESS_TREE_LIMIT_BYTES,
    phase_d_runtime_identity,
    validate_executing_phase_d_runtime,
)
from .operating_decision_prospective_pilot import _canonical_bytes


RESOURCE_PROBE_SCHEMA_VERSION = 1
RESOURCE_PROBE_PROTOCOL_VERSION = "phase_d_constructed_20_block_resource_probe_v1"
RESOURCE_PROBE_BLOCK_COUNT = 20
RESOURCE_PROBE_PARTITION = "p0_disposable_constructed_phase_d_resource_probe_v1"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _self_peak_rss_bytes() -> int:
    # Darwin reports bytes; Linux reports KiB.
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if os.uname().sysname == "Darwin" else value * 1024


def _process_tree_rss_bytes(root_pid: int) -> int:
    completed = subprocess.run(
        ("ps", "-axo", "pid=,ppid=,rss="),
        check=True,
        capture_output=True,
        text=True,
    )
    parents: dict[int, int] = {}
    rss_bytes: dict[int, int] = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        pid, parent, rss_kib = (int(value) for value in fields)
        parents[pid] = parent
        rss_bytes[pid] = rss_kib * 1024
    members = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in members and pid not in members:
                members.add(pid)
                changed = True
    return sum(rss_bytes.get(pid, 0) for pid in members)


@dataclass
class _ProcessTreeMonitor:
    interval_seconds: float = 0.05

    def __post_init__(self) -> None:
        self._stop = Event()
        self._thread: Optional[Thread] = None
        self.peak_bytes = 0
        self.sample_count = 0
        self.error: Optional[str] = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak_bytes = max(
                    self.peak_bytes, _process_tree_rss_bytes(os.getpid())
                )
                self.sample_count += 1
            except Exception as exc:  # pragma: no cover - platform permission path
                self.error = f"{type(exc).__name__}: {exc}"
                return
            sleep(self.interval_seconds)

    def __enter__(self) -> "_ProcessTreeMonitor":
        self._thread = Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 4 * self.interval_seconds))


def _stage(started_wall: float, started_cpu: float) -> dict:
    return {
        "wall_seconds": perf_counter() - started_wall,
        "cpu_seconds": process_time() - started_cpu,
        "coordinator_peak_rss_bytes": _self_peak_rss_bytes(),
    }


def constructed_resource_archive_payload(seed: Mapping[str, object]) -> dict:
    """Create a twenty-block-sized, explicitly non-scientific payload."""

    block = seed.get("block_result")
    if not isinstance(block, Mapping):
        raise ValueError("resource probe seed must contain one rehearsal block")
    block_bytes = _canonical_bytes(block)
    block_digest = _sha256(block_bytes)
    return {
        "schema_version": RESOURCE_PROBE_SCHEMA_VERSION,
        "protocol_version": RESOURCE_PROBE_PROTOCOL_VERSION,
        "partition": RESOURCE_PROBE_PARTITION,
        "scientific_use": "prohibited_constructed_resource_evidence_only",
        "source_seed": {
            "schema_version": seed.get("schema_version"),
            "protocol_version": seed.get("protocol_version"),
            "source_revision": seed.get("source_revision"),
            "protocol_digest": seed.get("protocol_digest"),
            "scientific_result_digest": seed.get("scientific_result_digest"),
            "block_digest": block_digest,
            "block_canonical_bytes": len(block_bytes),
        },
        "runtime_identity": phase_d_runtime_identity(),
        "block_count": RESOURCE_PROBE_BLOCK_COUNT,
        "constructed_blocks": [block] * RESOURCE_PROBE_BLOCK_COUNT,
    }


def validate_constructed_resource_archive(payload: Mapping[str, object]) -> dict:
    """Validate the constructed archive without treating it as science."""

    if payload.get("schema_version") != RESOURCE_PROBE_SCHEMA_VERSION:
        raise ValueError("resource probe schema is invalid")
    if payload.get("protocol_version") != RESOURCE_PROBE_PROTOCOL_VERSION:
        raise ValueError("resource probe protocol is invalid")
    if payload.get("partition") != RESOURCE_PROBE_PARTITION:
        raise ValueError("resource probe partition is invalid")
    if payload.get("scientific_use") != "prohibited_constructed_resource_evidence_only":
        raise ValueError("constructed resource evidence lacks its prohibition")
    if payload.get("runtime_identity") != phase_d_runtime_identity():
        raise ValueError("resource probe runtime identity is invalid")
    blocks = payload.get("constructed_blocks")
    if not isinstance(blocks, list) or len(blocks) != RESOURCE_PROBE_BLOCK_COUNT:
        raise ValueError("resource probe must contain twenty constructed blocks")
    seed = payload.get("source_seed")
    if not isinstance(seed, Mapping):
        raise ValueError("resource probe seed identity is missing")
    expected_digest = seed.get("block_digest")
    expected_size = seed.get("block_canonical_bytes")
    for block in blocks:
        if not isinstance(block, Mapping):
            raise ValueError("constructed resource block is invalid")
        encoded = _canonical_bytes(block)
        if _sha256(encoded) != expected_digest or len(encoded) != expected_size:
            raise ValueError("constructed resource block differs from its seed")
    return dict(payload)


def run_constructed_resource_probe(
    seed_path: Path,
    archive_path: Path,
    result_path: Path,
) -> dict:
    """Write, load, and validate a representative twenty-block JSON archive."""

    validate_executing_phase_d_runtime()
    seed_path = seed_path.expanduser().resolve()
    archive_path = archive_path.expanduser().resolve()
    result_path = result_path.expanduser().resolve()
    if archive_path == result_path or archive_path.exists() or result_path.exists():
        raise FileExistsError("resource-probe outputs must be new and distinct")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    stage_records: dict[str, dict] = {}

    with _ProcessTreeMonitor() as monitor:
        started_wall, started_cpu = perf_counter(), process_time()
        seed_bytes = seed_path.read_bytes()
        seed = json.loads(seed_bytes)
        stage_records["seed_load"] = _stage(started_wall, started_cpu)

        started_wall, started_cpu = perf_counter(), process_time()
        payload = constructed_resource_archive_payload(seed)
        stage_records["retained_results"] = _stage(started_wall, started_cpu)

        started_wall, started_cpu = perf_counter(), process_time()
        archive_bytes = _canonical_bytes(payload)
        archive_sha256 = _sha256(archive_bytes)
        stage_records["canonical_serialization"] = _stage(started_wall, started_cpu)

        started_wall, started_cpu = perf_counter(), process_time()
        archive_path.write_bytes(archive_bytes)
        stage_records["archive_write"] = _stage(started_wall, started_cpu)

        started_wall, started_cpu = perf_counter(), process_time()
        loaded_bytes = archive_path.read_bytes()
        loaded = json.loads(loaded_bytes)
        stage_records["json_load"] = _stage(started_wall, started_cpu)

        started_wall, started_cpu = perf_counter(), process_time()
        validate_constructed_resource_archive(loaded)
        if _sha256(loaded_bytes) != archive_sha256:
            raise ValueError("resource archive changed during write/load")
        stage_records["archive_validation"] = _stage(started_wall, started_cpu)

    measured_peak = max(monitor.peak_bytes, _self_peak_rss_bytes())
    result = {
        "schema_version": RESOURCE_PROBE_SCHEMA_VERSION,
        "protocol_version": RESOURCE_PROBE_PROTOCOL_VERSION,
        "partition": RESOURCE_PROBE_PARTITION,
        "scientific_use": "prohibited_constructed_resource_evidence_only",
        "runtime_identity": phase_d_runtime_identity(),
        "machine_physical_memory_bytes": PHASE_D_MACHINE_MEMORY_BYTES,
        "process_tree_limit_bytes": PHASE_D_PROCESS_TREE_LIMIT_BYTES,
        "minimum_required_headroom_fraction": 0.50,
        "source_seed_path": str(seed_path),
        "source_seed_sha256": _sha256(seed_bytes),
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha256,
        "archive_size_bytes": len(archive_bytes),
        "block_count": RESOURCE_PROBE_BLOCK_COUNT,
        "logical_intermediate_copy_bytes": len(archive_bytes) * 3,
        "process_tree_peak_rss_bytes": measured_peak,
        "process_tree_sample_count": monitor.sample_count,
        "process_tree_sampling_error": monitor.error,
        "within_process_tree_limit": measured_peak <= PHASE_D_PROCESS_TREE_LIMIT_BYTES,
        "headroom_fraction": 1.0 - measured_peak / PHASE_D_PROCESS_TREE_LIMIT_BYTES,
        "worker_count": 4,
        "stages": stage_records,
        "stop_resume_behavior": (
            "persist complete validated blocks atomically; resume only from blocks "
            "matching source, protocol, and partition identities"
        ),
    }
    result_path.write_bytes(_canonical_bytes(result))
    return result


__all__ = [
    "RESOURCE_PROBE_BLOCK_COUNT",
    "RESOURCE_PROBE_PARTITION",
    "constructed_resource_archive_payload",
    "run_constructed_resource_probe",
    "validate_constructed_resource_archive",
]
