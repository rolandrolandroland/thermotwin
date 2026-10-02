"""Frozen provisional operating-decision design entering Phase D6."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Mapping


PHASE_D_FREEZE_SCHEMA_VERSION = 1
PHASE_D_FREEZE_PROTOCOL_VERSION = (
    "operating_decision_prospective_phase_d_provisional_freeze_v1"
)
PHASE_D_FREEZE_ARTIFACT_DIGEST = (
    "a8b4cc22f8bf1d868a743c2427c1225483f954a8aeb9f751e38bec60dd949ca7"
)


def prospective_phase_d_freeze_payload() -> dict:
    """Return the immutable provisional design authorized for D6."""

    return {
        "schema_version": PHASE_D_FREEZE_SCHEMA_VERSION,
        "protocol_version": PHASE_D_FREEZE_PROTOCOL_VERSION,
        "campaign": "operating_decision_prospective_v1_2026_09",
        "status": "provisional_design_frozen_internal_check_authorized",
        "evidence": {
            "phase_c_freeze": {
                "payload_digest": (
                    "f9fd7d6760549f2b69ca4eac8fecabc3fc5783f3d2309e0d20fd527cd13830a8"
                ),
            },
            "phase_d1": {
                "source_revision": "f848a9583e0b3a345971401e08047b4769edbb9c",
                "json_sha256": (
                    "d83ae0e8ed52f328bf0a2accf5419dbd3a9ae10939552a14ccc0a445a1f30a2e"
                ),
                "scientific_result_digest": (
                    "efe860298104352fd0b6b35fc685e2320e6cfebd6db31fe8efc9876c509b1b5a"
                ),
            },
            "phase_d2": {
                "analysis_source_revision": (
                    "ed450411c6652a1e35a601ec8ecb9c426b761c51"
                ),
                "json_sha256": (
                    "3b69f063de593a48af8430e94f802f43bf81dc02e0bec999b3b510ed143b0c2f"
                ),
                "scientific_result_digest": (
                    "242aa34e72cff260c7a16efc0ce8041928bebb63eeedc5dab88be86817d61b43"
                ),
            },
            "phase_d3": {
                "analysis_source_revision": (
                    "b0fd0632a8c94ac262029aa122ca2bd9fecd9103"
                ),
                "json_sha256": (
                    "8664fb237efba6854ed46ea600d9ad88d82117d795d108839c7a8deaeff69397"
                ),
                "scientific_result_digest": (
                    "dd349111b64722877a27536cbe5711fcffc8e91f0a2cb6c38f7c9337f1945c1b"
                ),
                "rule_json_sha256": (
                    "993e408d876f3ffebcf488228a27735df72093455e25af6e9670dcc2b49006c6"
                ),
                "rule_artifact_digest": (
                    "a5eaf1fa7f9f800e1b8938cac04bcb5bcd635f242d846d36c117c94f0ef38f66"
                ),
            },
            "phase_d4": {
                "analysis_source_revision": (
                    "028fbca2c4e7fb4099478e46cc2f7a1a645db67a"
                ),
                "json_sha256": (
                    "c63c56e654eef5ab3f7be4de5e92dff05e32c4baf13d976fb193446d226dc26b"
                ),
                "scientific_result_digest": (
                    "1282a74f71073cffc1960d54a63a088c68f0d8e19838bb57b583c7617d4613b9"
                ),
                "selected_action_agreement": {"count": 12, "denominator": 12},
            },
            "phase_d5": {
                "analysis_source_revision": (
                    "72e711bc29bded623f7f196f519c3a0d41b7622c"
                ),
                "analysis_source_manifest_digest": (
                    "c793021767ccf28cfa86a5f65f686e2d56b51270808c43f504c175a0edbd2845"
                ),
                "protocol_digest": (
                    "57d9cc8c4a5f737e297339092df224d4987493fa52c7ad289ddc58779bb0375b"
                ),
                "json_sha256": (
                    "8ee535b760be9254bb9ddba8b82a909ee6fce6dd2f8c07b5790e47bdf4bf95c2"
                ),
                "scientific_result_digest": (
                    "b26352040b44c595f86d76003dd55bd7a2d8b43c0ccf9b825dda53c0a6edf4f3"
                ),
                "archive_content_digest": (
                    "0e6af05733f93399be5fa7f9559b20fd5e47ff93398ee86310085d519b06d22c"
                ),
                "map_cell_count": 48,
                "stress_archive_bytes": 228560784,
                "stress_archives": {
                    "high_voltage_noise": {
                        "scientific_result_digest": (
                            "99258c61098c9a5fd727abb096f6eaf296cff7c8a699749eb63bd411ee2badcc"
                        ),
                        "json_sha256": (
                            "6f5fa1ce82ca60b1f54471e776f7420d3fe3f65c807e44ed3432fabb94d56bde"
                        ),
                    },
                    "high_face_noise": {
                        "scientific_result_digest": (
                            "99b916a4cad310cf116f34aeeb34e3633f1df294d55ad88b9f17eef3186cbd9b"
                        ),
                        "json_sha256": (
                            "713365ec40849dc59490f1259c5fec5e6ec0d42e3596b5f7df02c62b70510587"
                        ),
                    },
                    "high_probe_loading": {
                        "scientific_result_digest": (
                            "6d843b999df068b4ea51f0a9d020e1949e5f23bc8e083ea9913f66e08a37462f"
                        ),
                        "json_sha256": (
                            "66cfb27aeb83b01d3ea769b274e932773dd3d8f0ad83cbcb2a866f9d17c3d9f1"
                        ),
                    },
                },
            },
        },
        "selector": {
            "procedure_name": "prospective_four_action_selector_v2",
            "algorithm_version": "development-padded-stop-and-four-key-ranking-v2",
            "action_order": [
                "stop_now",
                "fixed_thermal",
                "fixed_voltage",
                "fixed_face_temperature",
            ],
            "draw_count": 16,
            "max_unstable_draws_per_source_action": 1,
            "stopping_clearance": 0.0,
            "single_candidate_stopping_clearance": 0.0,
            "minimum_expected_reduction": 0.0,
            "minimum_utility_per_cost": 0.025,
            "development_offsets": {
                "stop_now": 0.0,
                "fixed_thermal": 0.0,
                "fixed_voltage": 0.074,
                "fixed_face_temperature": 0.098,
            },
            "worker_count": 4,
        },
        "primary_design": {
            "sensor_scenario": "nominal",
            "cost_scenario": "balanced_face_equal",
            "case_count": 60,
            "selected_action_counts": {
                "stop_now": 34,
                "fixed_thermal": 16,
                "fixed_voltage": 5,
                "fixed_face_temperature": 5,
                "selection_failure": 0,
            },
            "definitive_decision_count": 32,
            "no_useful_measurement_count": 9,
            "selection_failure_count": 0,
            "verification_failure_count": 0,
            "pipeline_failure_count": 0,
        },
        "map_sensitivity": {
            "sensor_scenario_count": 4,
            "cost_scenario_count": 12,
            "map_cell_count": 48,
            "selected_count_ranges": {
                "stop_now": [29, 42],
                "fixed_thermal": [8, 31],
                "fixed_voltage": [0, 10],
                "fixed_face_temperature": [0, 6],
            },
            "definitive_decision_count_range": [28, 32],
            "no_useful_measurement_count_range": [4, 17],
            "selection_failure_count_all_cells": 0,
            "verification_failure_count_all_cells": 0,
            "pipeline_failure_count_all_cells": 0,
            "maximum_ineligible_case_count": {
                "fixed_thermal": 0,
                "fixed_voltage": 0,
                "fixed_face_temperature": 1,
            },
        },
        "runtime": {
            "python_implementation": "CPython",
            "python_version": "3.10.12",
            "operating_system": "Darwin",
            "machine": "arm64",
            "requirements_sha256": (
                "be75b5b62c2dad9d48a412ecd857882a20cd10f8f03a60ea24c53debe05dea12"
            ),
            "process_tree_peak_rss_bytes": 1630978048,
            "process_tree_limit_bytes": 4294967296,
            "process_tree_sampling_error": None,
        },
        "authorization": {
            "next_partition": "p1_development_internal_check",
            "development_internal_check_authorized_now": True,
            "internal_check_block_count": 10,
            "internal_check_case_count": 30,
            "independent_calibration_authorized_now": False,
            "reserved_evaluation_authorized_now": False,
            "condition": (
                "apply_this_frozen_primary_design_once_under_a_committed_"
                "phase_d6_analysis_protocol"
            ),
        },
        "interpretation": (
            "development_only_provisional_design_not_calibration_reserved_"
            "evaluation_or_hardware_evidence"
        ),
    }


def prospective_phase_d_freeze_digest() -> str:
    encoded = json.dumps(
        prospective_phase_d_freeze_payload(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prospective_phase_d_freeze_artifact() -> dict:
    artifact = deepcopy(prospective_phase_d_freeze_payload())
    artifact["artifact_digest"] = prospective_phase_d_freeze_digest()
    return artifact


def validate_prospective_phase_d_freeze(payload: Mapping[str, object]) -> dict:
    """Reject any alteration of the committed Phase D freeze record."""

    if not isinstance(payload, Mapping):
        raise ValueError("Phase D freeze must be a mapping")
    expected = prospective_phase_d_freeze_artifact()
    if dict(payload) != expected:
        raise ValueError("Phase D freeze does not match the committed record")
    if expected["artifact_digest"] != PHASE_D_FREEZE_ARTIFACT_DIGEST:
        raise ValueError("Phase D freeze digest constant is stale")
    return expected


__all__ = [
    "PHASE_D_FREEZE_ARTIFACT_DIGEST",
    "PHASE_D_FREEZE_PROTOCOL_VERSION",
    "PHASE_D_FREEZE_SCHEMA_VERSION",
    "prospective_phase_d_freeze_artifact",
    "prospective_phase_d_freeze_digest",
    "prospective_phase_d_freeze_payload",
    "validate_prospective_phase_d_freeze",
]
