"""Frozen Phase C settings for prospective operating-decision development."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping


PHASE_C_FREEZE_SCHEMA_VERSION = 1
PHASE_C_FREEZE_PROTOCOL_VERSION = (
    "operating_decision_prospective_phase_c_freeze_v1"
)
PHASE_C_EVIDENCE_SOURCE_REVISION = (
    "c0518f5885f8421a1f6f95d4f60c5dc5744cfb4a"
)
FROZEN_PROSPECTIVE_DRAW_COUNT = 16
FROZEN_MAX_UNSTABLE_DRAWS_PER_SOURCE_ACTION = 1
FROZEN_PROSPECTIVE_WORKER_COUNT = 4
PHASE_C_FREEZE_PAYLOAD_DIGEST = (
    "f9fd7d6760549f2b69ca4eac8fecabc3fc5783f3d2309e0d20fd527cd13830a8"
)


def prospective_phase_c_freeze_payload() -> dict:
    """Return the immutable Phase C result used to enter development."""

    return {
        "schema_version": PHASE_C_FREEZE_SCHEMA_VERSION,
        "protocol_version": PHASE_C_FREEZE_PROTOCOL_VERSION,
        "campaign": "operating_decision_prospective_v1_2026_09",
        "evidence_source_revision": PHASE_C_EVIDENCE_SOURCE_REVISION,
        "evidence": {
            "parent": {
                "partition": "p4_disposable_bounded_instability_pilot",
                "json_sha256": (
                    "91f7d21c9e3a72bb8f341efc78ff7ce0a1a570fef0ff8540ed998e1655770bad"
                ),
                "json_bytes": 15395653,
                "protocol_digest": (
                    "5316dd5383001955c93b753748a8301f7f4a404709eec90b31bd6a6ca3f51864"
                ),
                "scientific_result_digest": (
                    "ef07528b0fee6da440090f357b97845399b84aaddeaccaf0db30393204a97f00"
                ),
            },
            "n32_followup": {
                "artifact_id": (
                    "p4_disposable_bounded_instability_pilot_"
                    "n32_all_cases_v1"
                ),
                "json_sha256": (
                    "749e6896bbd90503faf75c6deaffd24a5ede63baf5e6e15fa84e2d08c12f66cf"
                ),
                "json_bytes": 34992556,
                "protocol_digest": (
                    "8ec7b1c0bb0529ff5b02fd8bd98e24f57e9171880e7beea86aab2e8e016f1ec3"
                ),
                "scientific_result_digest": (
                    "0845d230b3de5d4678bc2c23995a5d721f91d449b4eb1bdc643b326e5b9a3e7e"
                ),
            },
        },
        "selection": {
            "draw_count": FROZEN_PROSPECTIVE_DRAW_COUNT,
            "max_unstable_draws_per_source_action": (
                FROZEN_MAX_UNSTABLE_DRAWS_PER_SOURCE_ACTION
            ),
            "failed_draw_scoring": "pre_action_width_no_gain",
            "complete_denominator_required": True,
            "all_reference_measurement_actions_must_be_eligible": True,
            "phase_d_revision_boundary": (
                "increase_draw_count_only_under_the_predeclared_development_"
                "sensitivity_rule_before_internal_check"
            ),
            "n16_to_n32_action_agreement": {
                "count": 12,
                "denominator": 12,
            },
            "n16_to_n32_maximum_changed_choice_normalized_regret": 0.0,
        },
        "compute_plan": {
            "worker_count": FROZEN_PROSPECTIVE_WORKER_COUNT,
            "runtime": {
                "python_implementation": "CPython",
                "python_version": "3.10.12",
                "operating_system": "Darwin",
                "machine": "arm64",
                "host_node_sha256": (
                    "67e831cdcbebfdb59ee481d5b3c76b5fab055ef7ea8eb8a434594e339ddbf426"
                ),
            },
            "planned_partitions": [
                {
                    "partition": "p1_development_tuning",
                    "block_count": 20,
                    "case_count": 60,
                    "estimated_wall_seconds": 35876.15561791521,
                    "estimated_cpu_seconds": 131802.28392,
                    "estimated_archive_bytes": 76965610,
                },
                {
                    "partition": "p1_development_internal_check",
                    "block_count": 10,
                    "case_count": 30,
                    "estimated_wall_seconds": 17938.077808957605,
                    "estimated_cpu_seconds": 65901.14196,
                    "estimated_archive_bytes": 38482805,
                },
                {
                    "partition": "p1_independent_calibration",
                    "block_count": 100,
                    "case_count": 300,
                    "estimated_wall_seconds": 179380.77808957605,
                    "estimated_cpu_seconds": 659011.4195999999,
                    "estimated_archive_bytes": 384828050,
                },
                {
                    "partition": "p1_reserved_evaluation",
                    "block_count": 100,
                    "case_count": 300,
                    "estimated_wall_seconds": 179380.77808957605,
                    "estimated_cpu_seconds": 659011.4195999999,
                    "estimated_archive_bytes": 384828050,
                },
            ],
            "total_planned_blocks": 230,
            "total_planned_cases": 690,
            "estimated_total_wall_seconds": 412575.7896060249,
            "estimated_total_cpu_seconds": 1515726.26508,
            "estimated_total_archive_bytes": 885104515,
            "measured_linear_n16_total_wall_seconds": 412291.383835042,
            "measured_linear_n16_total_cpu_seconds": 1514707.13462,
            "conservative_concurrent_peak_rss_bytes": 185204736,
            "scope": (
                "primary_nominal_sensor_campaign_with_all_fixed_comparators_"
                "and_verification"
            ),
            "cost_only_scenarios_reuse_authenticated_predictions": True,
            "additional_sensor_quality_simulations_included": False,
        },
        "authorization": {
            "phase_d_entry_authorized": True,
            "next_partition": "p1_development_tuning",
            "development_internal_check_authorized_now": False,
            "independent_calibration_authorized_now": False,
            "reserved_evaluation_authorized_now": False,
            "condition": (
                "open_only_the_development_tuning_partition_under_a_separately_"
                "committed_phase_d_protocol"
            ),
        },
        "interpretation": (
            "engineering_stability_and_compute_freeze_not_accuracy_coverage_"
            "or_selector_benefit_evidence"
        ),
    }


def prospective_phase_c_freeze_digest() -> str:
    payload = prospective_phase_c_freeze_payload()
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_prospective_phase_c_freeze(payload: Mapping[str, object]) -> dict:
    """Reject any alteration of the committed Phase C freeze record."""

    if not isinstance(payload, Mapping):
        raise ValueError("Phase C freeze must be a mapping")
    expected = prospective_phase_c_freeze_payload()
    if dict(payload) != expected:
        raise ValueError("Phase C freeze does not match the committed record")
    return expected


__all__ = [
    "FROZEN_MAX_UNSTABLE_DRAWS_PER_SOURCE_ACTION",
    "FROZEN_PROSPECTIVE_DRAW_COUNT",
    "FROZEN_PROSPECTIVE_WORKER_COUNT",
    "PHASE_C_EVIDENCE_SOURCE_REVISION",
    "PHASE_C_FREEZE_PROTOCOL_VERSION",
    "PHASE_C_FREEZE_PAYLOAD_DIGEST",
    "PHASE_C_FREEZE_SCHEMA_VERSION",
    "prospective_phase_c_freeze_digest",
    "prospective_phase_c_freeze_payload",
    "validate_prospective_phase_c_freeze",
]
