"""Source-bound Phase D2 offset analysis of the immutable Phase D1 archive.

This module performs no simulation or refitting.  It validates the complete
Phase D1 archive, reveals only the truth already stored after every policy was
saved, computes the four predeclared block-max nonconformity sequences, and
applies the frozen finite-offset feasibility gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence

from .operating_decision import POLICY_NAMES
from .operating_decision_prospective_phase_d import (
    PHASE_D_HASH_DOMAIN,
    PHASE_D_NUMERICAL_SOURCE_PATHS,
    estimate_phase_d_development_offsets,
    phase_d_runtime_identity,
    validate_executing_phase_d_runtime,
)
from .operating_decision_prospective_phase_d_tuning import (
    validate_phase_d_tuning_archive,
)
from .operating_decision_prospective_pilot import (
    DEVELOPMENT_TUNING_BLOCK_COUNT,
    PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION,
    _canonical_bytes,
    _digest,
    _exact_mapping,
    _validate_revision,
    _validate_sha256,
)
from .operating_decision_provenance import (
    create_source_manifest,
    source_manifest_from_payload,
    source_manifest_payload,
    verify_source_manifest,
)
from .operating_decision_realism import STAGE3_TRUTH_CONDITIONS


PHASE_D2_SCHEMA_VERSION = 1
PHASE_D2_PROTOCOL_VERSION = "operating_decision_phase_d2_offsets_v1"
PHASE_D2_JSON_NAME = "p1_development_offsets.json"
PHASE_D2_REPORT_NAME = "p1_development_offsets.txt"
PHASE_D2_HASH_NAME = "p1_development_offsets.sha256"
PHASE_D2_ANALYSIS_SOURCE_PATHS = tuple(
    sorted(
        set(PHASE_D_NUMERICAL_SOURCE_PATHS).union(
            {
                "thermotwin/reports/operating_decision_prospective_phase_d2.py",
                "thermotwin/studies/operating_decision_prospective_phase_d2_offsets.py",
            }
        )
    )
)


@dataclass(frozen=True)
class SavedPhaseD2Artifacts:
    json_path: Path
    report_path: Path
    hash_path: Path
    json_sha256: str
    report_sha256: str
    archive_size_bytes: int


def _serialized_score(value: float) -> float | str:
    return "positive_infinity" if math.isinf(value) else value


def phase_d2_case_nonconformity(
    raw_interval: object,
    true_margin: object,
) -> tuple[float, str]:
    """Return the frozen case score and its auditable interval status."""

    if not isinstance(raw_interval, Mapping) or set(raw_interval) != {
        "lower",
        "upper",
    }:
        return math.inf, "missing_or_malformed_interval"
    lower, upper = raw_interval["lower"], raw_interval["upper"]
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (lower, upper)
    ):
        return math.inf, "nonnumeric_interval"
    lower, upper = float(lower), float(upper)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        return math.inf, "invalid_or_nonfinite_interval"
    if (
        isinstance(true_margin, bool)
        or not isinstance(true_margin, (int, float))
        or not math.isfinite(float(true_margin))
    ):
        return math.inf, "missing_or_invalid_true_margin"
    margin = float(true_margin)
    return max(0.0, lower - margin, margin - upper), "valid"


def _case_policy_score(
    case: Mapping[str, object],
    *,
    policy_name: str,
) -> dict:
    record = case["fixed_policy_results"][policy_name]
    saved = record.get("saved_before_reveal")
    post = record.get("post_reveal_score")
    interval = saved.get("margin_envelope") if isinstance(saved, Mapping) else None
    revealed = post.get("revealed") if isinstance(post, Mapping) else None
    true_margin = revealed.get("true_margin") if isinstance(revealed, Mapping) else None
    score, status = phase_d2_case_nonconformity(interval, true_margin)
    if record.get("failure") is not None:
        score, status = math.inf, "fixed_policy_failure"
    return {
        "policy": policy_name,
        "raw_interval": dict(interval) if isinstance(interval, Mapping) else None,
        "true_margin": (
            float(true_margin)
            if isinstance(true_margin, (int, float))
            and not isinstance(true_margin, bool)
            and math.isfinite(float(true_margin))
            else None
        ),
        "interval_status": status,
        "nonconformity": _serialized_score(score),
    }


def analyze_validated_phase_d1_archive(payload: Mapping[str, object]) -> dict:
    """Apply D2 to an archive that has already passed the strict D1 validator."""

    if payload["partition"] != PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION:
        raise ValueError("Phase D2 requires the p1 development-tuning archive")
    block_scores = {name: [] for name in POLICY_NAMES}
    block_records = []
    for expected_block, block in enumerate(payload["block_results"]):
        if block["block"] != expected_block:
            raise ValueError("Phase D2 requires all twenty ordered paired blocks")
        cases = block["cases"]
        if [case["truth_condition"] for case in cases] != list(
            STAGE3_TRUTH_CONDITIONS
        ):
            raise ValueError("Phase D2 block truth-family ordering is invalid")
        per_policy = {name: [] for name in POLICY_NAMES}
        case_records = []
        for case in cases:
            policy_records = []
            for policy_name in POLICY_NAMES:
                record = _case_policy_score(case, policy_name=policy_name)
                policy_records.append(record)
                value = record["nonconformity"]
                per_policy[policy_name].append(
                    math.inf if value == "positive_infinity" else float(value)
                )
            case_records.append(
                {
                    "truth_condition": case["truth_condition"],
                    "device_token": case["device_token"],
                    "policy_scores": policy_records,
                }
            )
        maxima = {}
        for policy_name in POLICY_NAMES:
            maximum = max(per_policy[policy_name])
            block_scores[policy_name].append(maximum)
            maxima[policy_name] = _serialized_score(maximum)
        block_records.append(
            {
                "block": expected_block,
                "case_scores": case_records,
                "block_max_nonconformity": maxima,
            }
        )
    offsets = estimate_phase_d_development_offsets(block_scores)
    return {
        "block_records": block_records,
        "block_scores_in_partition_order": {
            name: [_serialized_score(value) for value in block_scores[name]]
            for name in POLICY_NAMES
        },
        "offset_estimation": offsets,
    }


def build_phase_d2_offset_analysis(
    *,
    phase_d1_archive_path: Path | str,
    repository_root: Path | str,
    analysis_source_revision: str,
) -> dict:
    """Validate the immutable D1 input and build a sealed-source D2 result."""

    _validate_revision(analysis_source_revision)
    validate_executing_phase_d_runtime()
    root = Path(repository_root).expanduser().resolve(strict=True)
    archive_path = Path(phase_d1_archive_path).expanduser().resolve(strict=True)
    raw = archive_path.read_bytes()
    phase_d1 = validate_phase_d_tuning_archive(
        json.loads(raw),
        repository_root=root,
        block_directory=archive_path.parent / "blocks",
    )
    analysis_manifest = create_source_manifest(root, PHASE_D2_ANALYSIS_SOURCE_PATHS)
    analysis = analyze_validated_phase_d1_archive(phase_d1)
    payload = {
        "schema_version": PHASE_D2_SCHEMA_VERSION,
        "protocol_version": PHASE_D2_PROTOCOL_VERSION,
        "analysis_source_revision": analysis_source_revision,
        "analysis_source_manifest": source_manifest_payload(analysis_manifest),
        "runtime_identity": phase_d_runtime_identity(),
        "input_phase_d1": {
            "path_name": archive_path.name,
            "json_sha256": hashlib.sha256(raw).hexdigest(),
            "archive_content_digest": phase_d1["archive_content_digest"],
            "scientific_result_digest": phase_d1["scientific_result_digest"],
            "source_revision": phase_d1["source_revision"],
            "source_manifest_digest": phase_d1["source_manifest"]["digest"],
            "protocol_digest": phase_d1["protocol_digest"],
            "partition": phase_d1["partition"],
            "block_count": phase_d1["block_count"],
            "case_count": phase_d1["case_count"],
        },
        "method": {
            "case_nonconformity": "max(0,L-m,m-U)",
            "missing_invalid_or_nonfinite_interval": "positive_infinity",
            "fixed_policy_failure": "positive_infinity",
            "within_block_aggregation": "maximum_across_three_truth_families",
            "block_score_count_per_policy": DEVELOPMENT_TUNING_BLOCK_COUNT,
            "nearest_rank": 18,
            "quantile": 0.90,
            "round_up_to_kelvin": 0.001,
            "refitting_performed": False,
            "new_partition_opened": False,
        },
        **analysis,
    }
    payload["scientific_result_digest"] = _digest(
        f"{PHASE_D_HASH_DOMAIN}.phase_d2_offset_result_v1",
        payload,
    )
    return payload


def validate_phase_d2_offset_analysis(
    payload: Mapping[str, object],
    *,
    phase_d1_archive_path: Path | str,
    repository_root: Path | str,
) -> dict:
    """Strictly replay D2 from the authenticated D1 input."""

    expected_keys = {
        "schema_version",
        "protocol_version",
        "analysis_source_revision",
        "analysis_source_manifest",
        "runtime_identity",
        "input_phase_d1",
        "method",
        "block_records",
        "block_scores_in_partition_order",
        "offset_estimation",
        "scientific_result_digest",
    }
    item = _exact_mapping(payload, expected_keys, "Phase D2 offset analysis")
    if (
        item["schema_version"] != PHASE_D2_SCHEMA_VERSION
        or item["protocol_version"] != PHASE_D2_PROTOCOL_VERSION
        or item["runtime_identity"] != phase_d_runtime_identity()
    ):
        raise ValueError("Phase D2 result header is invalid")
    _validate_revision(item["analysis_source_revision"])
    manifest = source_manifest_from_payload(item["analysis_source_manifest"])
    verify_source_manifest(
        manifest,
        repository_root,
        PHASE_D2_ANALYSIS_SOURCE_PATHS,
    ).assert_valid()
    rebuilt = build_phase_d2_offset_analysis(
        phase_d1_archive_path=phase_d1_archive_path,
        repository_root=repository_root,
        analysis_source_revision=item["analysis_source_revision"],
    )
    if dict(item) != rebuilt:
        raise ValueError("Phase D2 result does not match strict replay")
    _validate_sha256(
        "Phase D2 scientific result digest", item["scientific_result_digest"]
    )
    return dict(item)


def format_phase_d2_offset_report(payload: Mapping[str, object]) -> str:
    estimate = payload["offset_estimation"]
    lines = [
        "Prospective operating-decision Phase D2 offset analysis",
        "=========================================================",
        "",
        f"Analysis source revision: {payload['analysis_source_revision']}",
        f"Input D1 JSON SHA-256: {payload['input_phase_d1']['json_sha256']}",
        "Input D1 scientific digest: "
        f"{payload['input_phase_d1']['scientific_result_digest']}",
        f"D2 scientific digest: {payload['scientific_result_digest']}",
        "Refitting performed: no",
        "New partition opened: no",
        "",
        "Offsets and 18th order statistics",
    ]
    offsets = estimate.get("development_offsets")
    for policy_name in POLICY_NAMES:
        statistic = estimate["order_statistics"][policy_name]
        offset = None if offsets is None else offsets["values"][policy_name]
        lines.append(
            f"- {policy_name}: statistic={statistic}; offset={offset}"
        )
    lines.extend(("", f"Status: {estimate['status']}"))
    if estimate["status"] == "finite_offsets_ready":
        lines.append("PASS: all four required development offsets are finite.")
        lines.append(
            "Next permitted step: Phase D3 grid rescoring on this same archive."
        )
    else:
        lines.append("STOP: at least one required development offset is infinite.")
        lines.append("Phase D3 and the internal check are not authorized.")
    return "\n".join(lines)


def save_phase_d2_offset_artifacts(
    payload: Mapping[str, object],
    *,
    output_directory: Path | str,
) -> SavedPhaseD2Artifacts:
    root = Path(output_directory).expanduser().resolve()
    json_path = root / PHASE_D2_JSON_NAME
    report_path = root / PHASE_D2_REPORT_NAME
    hash_path = root / PHASE_D2_HASH_NAME
    existing = [path for path in (json_path, report_path, hash_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "Phase D2 refuses to overwrite artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    archive = _canonical_bytes(dict(payload)) + b"\n"
    report = (format_phase_d2_offset_report(payload) + "\n").encode("utf-8")
    json_sha = hashlib.sha256(archive).hexdigest()
    report_sha = hashlib.sha256(report).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    json_path.write_bytes(archive)
    report_path.write_bytes(report)
    hash_path.write_text(
        f"{json_sha}  {json_path.name}\n{report_sha}  {report_path.name}\n",
        encoding="utf-8",
    )
    return SavedPhaseD2Artifacts(
        json_path=json_path,
        report_path=report_path,
        hash_path=hash_path,
        json_sha256=json_sha,
        report_sha256=report_sha,
        archive_size_bytes=len(archive),
    )


__all__ = [
    "PHASE_D2_ANALYSIS_SOURCE_PATHS",
    "PHASE_D2_HASH_NAME",
    "PHASE_D2_JSON_NAME",
    "PHASE_D2_REPORT_NAME",
    "SavedPhaseD2Artifacts",
    "analyze_validated_phase_d1_archive",
    "build_phase_d2_offset_analysis",
    "format_phase_d2_offset_report",
    "phase_d2_case_nonconformity",
    "save_phase_d2_offset_artifacts",
    "validate_phase_d2_offset_analysis",
]
