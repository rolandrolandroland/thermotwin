"""Validate the completed Phase D design without opening another partition."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Mapping

from .operating_decision_prospective_phase_d_freeze import (
    prospective_phase_d_freeze_artifact,
)


PHASE_D_FINAL_FREEZE_ARTIFACT_DIGEST = (
    "ecb490919a271357749101f7db7c30b98b445b522c2e39203508c3f716c137f6"
)
PHASE_D_FINAL_FREEZE_PROTOCOL_VERSION = (
    "operating_decision_prospective_phase_d_final_freeze_v1"
)
PHASE_D_FINAL_FREEZE_PATH = (
    Path(__file__).resolve().parents[1]
    / "OPERATING_DECISION_PROSPECTIVE_PHASE_D_FINAL_FREEZE.json"
)


def validate_prospective_phase_d_final_freeze(payload: Mapping[str, object]) -> dict:
    """Authenticate the immutable closeout record and its predecessor chain.

    This verifies the committed record; it does not rerun numerical fits or
    authorize calibration generation. Validate the full D6 archive at its
    historical clean source to verify raw evidence and checkpoint inventories.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("Phase D final freeze must be a mapping")
    record = deepcopy(dict(payload))
    digest = record.pop("artifact_digest", None)
    if digest != PHASE_D_FINAL_FREEZE_ARTIFACT_DIGEST:
        raise ValueError("Phase D final freeze does not match the committed record")
    encoded = json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != digest:
        raise ValueError("Phase D final freeze content digest is invalid")
    predecessor = prospective_phase_d_freeze_artifact()
    if (
        record["schema_version"] != 1
        or record["protocol_version"] != PHASE_D_FINAL_FREEZE_PROTOCOL_VERSION
        or record["provisional_freeze_artifact_digest"]
        != predecessor["artifact_digest"]
        or record["selector"] != predecessor["selector"]
        or record["campaign"] != predecessor["campaign"]
        or record["map_sensitivity"] != predecessor["map_sensitivity"]
        or any(
            record["evidence"][key] != value
            for key, value in predecessor["evidence"].items()
        )
    ):
        raise ValueError("Phase D final freeze predecessor chain is invalid")
    authorization = record["authorization"]
    if (
        authorization["phase_e_protocol_work_authorized_now"] is not True
        or authorization["independent_calibration_authorized_now"] is not False
        or authorization["reserved_evaluation_authorized_now"] is not False
        or authorization["development_internal_check_closed_against_regeneration"]
        is not True
        or record["internal_check_result"]["redesign_required"] is not False
    ):
        raise ValueError("Phase D final freeze authorization is invalid")
    record["artifact_digest"] = digest
    return record


def load_prospective_phase_d_final_freeze() -> dict:
    return validate_prospective_phase_d_final_freeze(
        json.loads(PHASE_D_FINAL_FREEZE_PATH.read_bytes())
    )


def validate_phase_d_final_freeze_evidence(archive_path: Path | str) -> dict:
    """Reproduce descriptive comparator metrics from authenticated D6 bytes."""

    from .operating_decision import POLICY_NAMES, STOP_NOW
    from .operating_decision_prospective_phase_d3_grid import (
        _case_loss_records,
        _fixed_policy_case,
        _group_summaries,
        _objective_for_records,
        _summary,
    )
    from .operating_decision_prospective_phase_d6_internal_check import (
        phase_d6_frozen_selector_rule,
    )

    freeze = load_prospective_phase_d_final_freeze()
    raw = Path(archive_path).expanduser().resolve(strict=True).read_bytes()
    expected = freeze["evidence"]["phase_d6"]
    if (
        hashlib.sha256(raw).hexdigest() != expected["json_sha256"]
        or len(raw) != expected["archive_size_bytes"]
    ):
        raise ValueError("Phase D final freeze D6 archive bytes do not match")
    archive = json.loads(raw)
    result = freeze["internal_check_result"]
    if (
        archive["scientific_result_digest"] != expected["scientific_result_digest"]
        or archive["analysis"]["metrics"] != result["metrics"]
        or archive["analysis"]["redesign_triggers"] != result["redesign_triggers"]
        or archive["analysis"]["action_counts"] != result["selected_action_counts"]
    ):
        raise ValueError("Phase D final freeze D6 result binding is invalid")
    offsets = phase_d6_frozen_selector_rule().development_offsets
    for policy in POLICY_NAMES:
        records = [
            _fixed_policy_case(
                case, block=block["block"], policy_name=policy, offsets=offsets
            )
            for block in archive["block_results"]
            for case in block["cases"]
        ]
        objective = _objective_for_records((0.0, 0.0, 0.0, 0.025), records)
        metrics = _group_summaries(_case_loss_records(records, objective))
        if metrics != freeze["fixed_policy_comparators"][policy]:
            raise ValueError("Phase D final freeze comparator metrics do not recompute")
    selected = archive["analysis"]["case_results"]
    one_stop = _summary([
        record for record in selected
        if record["selected_policy"] == STOP_NOW
        and record["candidate_reliability_stratum"] == "one_admissible_candidate"
    ])
    block_coverage = {
        "count": sum(
            all(record["development_adjusted_interval_covered"] for record in selected
                if record["block"] == block)
            for block in range(expected["block_count"])
        ),
        "denominator": expected["block_count"],
    }
    if (
        one_stop != result["one_candidate_stopping"]
        or block_coverage
        != result["simultaneous_development_adjusted_interval_block_coverage"]
    ):
        raise ValueError("Phase D final freeze stopping or block coverage is invalid")
    return freeze


__all__ = [
    "PHASE_D_FINAL_FREEZE_ARTIFACT_DIGEST",
    "PHASE_D_FINAL_FREEZE_PATH",
    "PHASE_D_FINAL_FREEZE_PROTOCOL_VERSION",
    "load_prospective_phase_d_final_freeze",
    "validate_phase_d_final_freeze_evidence",
    "validate_prospective_phase_d_final_freeze",
]
