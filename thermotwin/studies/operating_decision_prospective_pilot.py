"""Disposable draw-count pilot for the prospective operating-decision study.

The default runner opens only the predeclared disposable pilot.  It generates
sixteen predictive draws once for every case and derives the matched 4-, 8-,
and 16-draw analyses from that authenticated result.  A separate conditional
runner can revisit the same twelve cases at N=32 only when the authenticated
parent artifact satisfies the frozen trigger.  Development, calibration, and
reserved namespaces cannot be executed through either interface.

Truth and target-response objects are deliberately kept outside the action
scoring call.  All fixed-policy decisions are saved before their corresponding
target response is revealed and scored.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import tempfile
from time import perf_counter, process_time
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

from ..observations.test_stand import regular_measurement_times
from ..simulation.temporary_face_sensor import TemporaryFaceSensor

from .operating_decision import (
    MarginEnvelope,
    MarginInterval,
    POLICY_NAMES,
    RUN_DURATION_SECONDS,
    STOP_NOW,
    default_fixed_policies,
    initial_acquisition_regime,
)
from .operating_decision_prospective import (
    CandidateExclusion,
    ProspectiveActionEvaluation,
    ProspectiveAcquisitionSnapshot,
    ProspectiveSelection,
    ProspectiveSelectorRule,
    prospective_action_catalog_digest,
    prospective_selector_protocol_digest,
    prospective_selector_rule_from_payload,
    prospective_selector_rule_payload,
    select_prospective_action,
)
from .operating_decision_prospective_costs import (
    PRIMARY_PROSPECTIVE_COST_SCENARIO,
    PROSPECTIVE_COST_PROTOCOL_VERSION,
    PROSPECTIVE_COST_SCHEMA_VERSION,
    PROSPECTIVE_COST_ENERGY_CONVENTION,
    PROSPECTIVE_COST_FORMULA,
    PROSPECTIVE_COST_INSTRUMENTATION_CONVENTION,
    PROSPECTIVE_COST_TIME_CONVENTION,
    ProspectiveCostScenario,
    _action_evaluation_payload as _cost_action_evaluation_payload,
    _action_evaluation_from_payload,
    _action_cost_payload,
    _action_resources_payload,
    _build_action_costs,
    _build_action_resources,
    _scenario_payload,
    cost_prospective_uncertainty,
    prospective_cost_protocol_digest,
    prospective_costed_scorecard_payload,
    select_costed_prospective_action,
)
from .operating_decision_prospective_random_streams import (
    PARAMETER_DRAW,
    PROBE_DRAW,
    PROSPECTIVE_RANDOM_STREAM_DOMAIN,
    RUN_BIAS,
    WHITE_NOISE,
    ProspectiveRandomStream,
    ProspectiveRandomStreamKey,
    ProspectiveRandomStreamNamespace,
    ProspectiveRandomStreamRegistry,
)
from .operating_decision_prospective_uncertainty import (
    PROSPECTIVE_ELIGIBILITY_PROTOCOL,
    PROSPECTIVE_PADDED_SCORING_PROTOCOL,
    PROSPECTIVE_UNCERTAINTY_PROTOCOL_VERSION,
    PROSPECTIVE_UNCERTAINTY_SCHEMA_VERSION,
    ProspectiveCandidateOutcome,
    ProspectiveDrawOutcome,
    ProspectiveUncertaintyConfig,
    _ProspectiveDrawSamplingError,
    _ACQUISITION_EVIDENCE_DOMAIN,
    _canonical_digest as _uncertainty_canonical_digest,
    _face_probe_draw,
    _generator_parameter_draw,
    _physical_config_payload,
    _positive_semidefinite_cholesky,
    _parameter_spec,
    _stream_audit_payload as _uncertainty_stream_audit_payload,
    estimate_prospective_action_uncertainty,
    prefix_prospective_uncertainty_result,
    prepare_prospective_acquisition_evidence,
    prospective_uncertainty_protocol_digest,
    prospective_uncertainty_result_payload,
    prospective_uncertainty_summary_payload,
)
from .operating_decision_random_streams import (
    RANDOM_STREAM_PROTOCOL_VERSION,
    RandomStream,
    RandomStreamKey,
    RandomStreamRegistry,
)
from .operating_decision_resources import NOMINAL_SELECTION_COST_PROTOCOL
from .operating_decision_realism import (
    FIT_BOUND_TOLERANCE,
    FIT_RELATIVE_OBJECTIVE_TOLERANCE,
    FIT_SCALED_GRADIENT_TOLERANCE,
    FIT_STEP_TOLERANCE,
    STAGE3_TRUTH_CONDITIONS,
    OperatingDecisionRealismConfig,
    RealisticCandidateFit,
    _decoded_parameters,
)
from .operating_decision_replication import (
    CORRECTED_REPLICATION_CONFIG,
    CorrectedPartition,
    build_corrected_blinded_case,
    corrected_partition_config,
    corrected_case_id,
    corrected_physical_protocol_digest,
    corrected_truth_for_block,
    decide_corrected_blinded_case,
    score_corrected_saved_decision,
)
from .sensor_model_discrimination import (
    ALL_CHANNELS,
    COLD_FACE,
    COLD_EXCHANGER,
    HOT_EXCHANGER,
    MODEL_NAMES,
)


PROSPECTIVE_PILOT_SCHEMA_VERSION = 3
PROSPECTIVE_PILOT_PROTOCOL_VERSION = (
    "operating_decision_prospective_draw_count_pilot_v4"
)
PROSPECTIVE_N32_FOLLOWUP_SCHEMA_VERSION = 1
PROSPECTIVE_N32_FOLLOWUP_PROTOCOL_VERSION = (
    "operating_decision_prospective_n32_followup_v2"
)
PROSPECTIVE_CAMPAIGN = "operating_decision_prospective_v1_2026_09"
PROSPECTIVE_SUPERSEDED_PILOT_PARTITIONS = (
    "p1_disposable_draw_count_pilot",
    "p2_disposable_candidate_exclusion_pilot",
)
PROSPECTIVE_PILOT_PARTITION = "p3_disposable_archive_roundtrip_replacement_pilot"
PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION = "p1_development_tuning"
PROSPECTIVE_DEVELOPMENT_CHECK_PARTITION = "p1_development_internal_check"
PROSPECTIVE_CALIBRATION_PARTITION = "p1_independent_calibration"
PROSPECTIVE_RESERVED_PARTITION = "p1_reserved_evaluation"

PILOT_BLOCK_COUNT = 4
DEVELOPMENT_TUNING_BLOCK_COUNT = 20
DEVELOPMENT_CHECK_BLOCK_COUNT = 10
CALIBRATION_BLOCK_COUNT = 100
RESERVED_BLOCK_COUNT = 100
PILOT_DRAW_COUNTS = (4, 8, 16)
PILOT_GENERATED_DRAW_COUNT = 16
PILOT_STRICT_MAX_UNSTABLE = 0
PILOT_SENSITIVITY_MAX_UNSTABLE = 1
PILOT_N32_FOLLOWUP_DRAW_COUNT = 32
PILOT_N32_FOLLOWUP_ARTIFACT_ID = (
    "p3_disposable_archive_roundtrip_replacement_pilot_n32_all_cases_v1"
)
PILOT_N32_FOLLOWUP_CASE_SELECTION = (
    "all_four_blocks_all_three_truth_families"
)

_PIPELINE_FAILURE_PREFIX = "pipeline_failure:"
_SELECTION_FAILURE_PREFIX = "selection_failure:"
_ACTION_PREFIX = "action:"
_HASH_DOMAIN = "thermotwin.prospective_pilot"
_SELECTION_INFORMATION_BOUNDARY = {
    "inputs": [
        "common_initial_acquisition",
        "candidate_fits_from_common_acquisition",
        "declared_physical_and_cost_protocols",
        "semantic_prospective_random_streams",
    ],
    "excluded": [
        "truth_condition",
        "truth_parameters",
        "verification_observations",
        "future_actual_observations",
        "target_response",
        "true_margin",
    ],
}


@dataclass(frozen=True)
class ProspectivePartitionSpec:
    name: str
    block_count: int
    role: str

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise ValueError("prospective partition needs a trimmed name")
        if (
            not isinstance(self.block_count, int)
            or isinstance(self.block_count, bool)
            or self.block_count <= 0
        ):
            raise ValueError("prospective partition needs a positive block count")
        if not self.role or self.role != self.role.strip():
            raise ValueError("prospective partition needs a trimmed role")


PROSPECTIVE_PARTITION_PLAN = (
    ProspectivePartitionSpec(
        PROSPECTIVE_PILOT_PARTITION,
        PILOT_BLOCK_COUNT,
        "disposable_runtime_and_draw_count_only",
    ),
    ProspectivePartitionSpec(
        PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION,
        DEVELOPMENT_TUNING_BLOCK_COUNT,
        "development_tuning",
    ),
    ProspectivePartitionSpec(
        PROSPECTIVE_DEVELOPMENT_CHECK_PARTITION,
        DEVELOPMENT_CHECK_BLOCK_COUNT,
        "development_internal_check",
    ),
    ProspectivePartitionSpec(
        PROSPECTIVE_CALIBRATION_PARTITION,
        CALIBRATION_BLOCK_COUNT,
        "independent_calibration",
    ),
    ProspectivePartitionSpec(
        PROSPECTIVE_RESERVED_PARTITION,
        RESERVED_BLOCK_COUNT,
        "reserved_evaluation",
    ),
)


@dataclass(frozen=True)
class PilotAcceptanceRule:
    """Predeclared engineering heuristic for choosing the smallest tested N."""

    minimum_action_agreement: float = 0.90
    maximum_normalized_utility_regret: float = 0.05
    reference_draw_count: int = PILOT_GENERATED_DRAW_COUNT
    n32_followup_trigger_draw_count: int = PILOT_GENERATED_DRAW_COUNT

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_action_agreement <= 1.0:
            raise ValueError("pilot action agreement must lie in (0, 1]")
        if not 0.0 <= self.maximum_normalized_utility_regret <= 1.0:
            raise ValueError("pilot utility regret must lie in [0, 1]")
        if self.reference_draw_count != PILOT_GENERATED_DRAW_COUNT:
            raise ValueError("pilot reference must use the generated 16 draws")
        if self.n32_followup_trigger_draw_count != PILOT_GENERATED_DRAW_COUNT:
            raise ValueError("the pilot must require N=32 follow-up when N=16 wins")


def prospective_n32_followup_design_payload(
    rule: PilotAcceptanceRule = PilotAcceptanceRule(),
) -> dict:
    """Return the predeclared conditional N=32 design bound before the pilot opens."""

    if not isinstance(rule, PilotAcceptanceRule):
        raise ValueError("N=32 follow-up design needs the pilot acceptance rule")
    return {
        "artifact_id": PILOT_N32_FOLLOWUP_ARTIFACT_ID,
        "trigger": {
            "tested_prefix_recommendation": (
                rule.n32_followup_trigger_draw_count
            ),
            "requires_zero_parent_pilot_pipeline_failures": True,
            "requires_zero_parent_pilot_n16_selection_failures": True,
            "required_before_phase_d": True,
        },
        "campaign": PROSPECTIVE_CAMPAIGN,
        "partition": PROSPECTIVE_PILOT_PARTITION,
        "case_selection_rule": PILOT_N32_FOLLOWUP_CASE_SELECTION,
        "blocks": list(range(PILOT_BLOCK_COUNT)),
        "truth_families": list(STAGE3_TRUTH_CONDITIONS),
        "case_count": PILOT_BLOCK_COUNT * len(STAGE3_TRUTH_CONDITIONS),
        "generated_draw_count": PILOT_N32_FOLLOWUP_DRAW_COUNT,
        "candidate_prefix_draw_count": PILOT_GENERATED_DRAW_COUNT,
        "reference_draw_count": PILOT_N32_FOLLOWUP_DRAW_COUNT,
        "max_whole_draw_failures_per_source_action": (
            PILOT_STRICT_MAX_UNSTABLE
        ),
        "minimum_action_agreement": rule.minimum_action_agreement,
        "maximum_normalized_utility_regret": (
            rule.maximum_normalized_utility_regret
        ),
        "generation_rule": (
            "generate_32_once_for_all_12_cases_and_compare_authenticated_"
            "n16_prefix_to_n32"
        ),
        "acceptance_rule": (
            "agreement_and_regret_pass_with_zero_n32_pipeline_failures_and_"
            "zero_n32_selection_failures_and_zero_n32_whole_draw_failures"
        ),
    }


@dataclass(frozen=True)
class ProspectivePilotResult:
    """Complete in-memory result before archive serialization is benchmarked."""

    source_revision: str
    worker_count: int
    protocol_digest: str
    scientific_result_digest: str
    block_results: Tuple[dict, ...]
    acceptance: dict
    compute_budget_inputs: dict
    wall_seconds: float
    cpu_seconds: float
    peak_rss_bytes: int
    selector_rule: ProspectiveSelectorRule = ProspectiveSelectorRule()
    cost_scenario: ProspectiveCostScenario = PRIMARY_PROSPECTIVE_COST_SCENARIO

    def __post_init__(self) -> None:
        _validate_revision(self.source_revision)
        _validate_sha256("pilot protocol digest", self.protocol_digest)
        _validate_sha256(
            "pilot scientific result digest", self.scientific_result_digest
        )
        if (
            not isinstance(self.worker_count, int)
            or isinstance(self.worker_count, bool)
            or self.worker_count <= 0
        ):
            raise ValueError("pilot worker count must be positive")
        blocks = tuple(self.block_results)
        object.__setattr__(self, "block_results", blocks)
        if tuple(item.get("block") for item in blocks) != tuple(
            range(PILOT_BLOCK_COUNT)
        ):
            raise ValueError("pilot blocks must be complete and ordered")
        cases = tuple(
            case for block in blocks for case in block.get("cases", ())
        )
        if len(cases) != PILOT_BLOCK_COUNT * len(STAGE3_TRUTH_CONDITIONS):
            raise ValueError("pilot must retain all twelve cases")
        expected_cases = {
            (block, truth_condition)
            for block in range(PILOT_BLOCK_COUNT)
            for truth_condition in STAGE3_TRUTH_CONDITIONS
        }
        actual_cases = {
            (case.get("block"), case.get("truth_condition")) for case in cases
        }
        if actual_cases != expected_cases or len(actual_cases) != len(cases):
            raise ValueError("pilot case identities must be complete and unique")
        for name, value in (
            ("pilot wall seconds", self.wall_seconds),
            ("pilot CPU seconds", self.cpu_seconds),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isinstance(self.peak_rss_bytes, int) or self.peak_rss_bytes < 0:
            raise ValueError("pilot peak RSS must be a nonnegative integer")
        if not isinstance(self.selector_rule, ProspectiveSelectorRule):
            raise ValueError("pilot result needs its selector rule")
        if not isinstance(self.cost_scenario, ProspectiveCostScenario):
            raise ValueError("pilot result needs its cost scenario")
        expected_scientific_digest = _digest(
            f"{_HASH_DOMAIN}.scientific_result",
            prospective_pilot_scientific_payload(
                source_revision=self.source_revision,
                protocol_digest=self.protocol_digest,
                block_results=blocks,
                acceptance=self.acceptance,
            ),
        )
        if self.scientific_result_digest != expected_scientific_digest:
            raise ValueError("pilot scientific result digest is invalid")


@dataclass(frozen=True)
class ProspectiveN32FollowupResult:
    """Source-bound result of the conditional all-case N=32 continuation."""

    source_revision: str
    worker_count: int
    protocol_digest: str
    parent_payload: dict
    parent_payload_digest: str
    parent_protocol_digest: str
    parent_scientific_result_digest: str
    scientific_result_digest: str
    block_results: Tuple[dict, ...]
    acceptance: dict
    wall_seconds: float
    cpu_seconds: float
    peak_rss_bytes: int

    def __post_init__(self) -> None:
        _validate_revision(self.source_revision)
        for name, value in (
            ("N=32 protocol digest", self.protocol_digest),
            ("parent payload digest", self.parent_payload_digest),
            ("parent protocol digest", self.parent_protocol_digest),
            ("parent scientific result digest", self.parent_scientific_result_digest),
            ("N=32 scientific result digest", self.scientific_result_digest),
        ):
            _validate_sha256(name, value)
        if (
            not isinstance(self.worker_count, int)
            or isinstance(self.worker_count, bool)
            or self.worker_count <= 0
        ):
            raise ValueError("N=32 worker count must be positive")
        blocks = tuple(self.block_results)
        object.__setattr__(self, "block_results", blocks)
        _validate_followup_case_matrix(blocks)
        if not isinstance(self.parent_payload, dict):
            raise ValueError("N=32 result needs the complete parent pilot archive")
        parent_payload_digest = _digest(
            f"{_HASH_DOMAIN}.n32.parent_payload",
            self.parent_payload,
        )
        if self.parent_payload_digest != parent_payload_digest:
            raise ValueError("N=32 parent payload digest is invalid")
        if (
            self.parent_protocol_digest
            != self.parent_payload.get("protocol_digest")
            or self.parent_scientific_result_digest
            != self.parent_payload.get("scientific_result_digest")
        ):
            raise ValueError("N=32 parent archive references are inconsistent")
        partition = CorrectedPartition(
            name=PROSPECTIVE_PILOT_PARTITION,
            block_count=PILOT_BLOCK_COUNT,
            campaign=PROSPECTIVE_CAMPAIGN,
        )
        physical_config = corrected_partition_config(
            partition,
            CORRECTED_REPLICATION_CONFIG,
        )
        selector_rule, cost_scenario = _parent_selector_and_cost(
            self.parent_payload
        )
        expected_protocol = prospective_n32_followup_protocol_digest(
            self.parent_payload,
            physical_config,
            selector_rule,
            cost_scenario,
            self.source_revision,
        )
        if self.protocol_digest != expected_protocol:
            raise ValueError("N=32 protocol digest is invalid")
        recomputed_acceptance = evaluate_n32_followup_acceptance(
            self.parent_payload,
            blocks,
            physical_config=physical_config,
            selector_rule=selector_rule,
            cost_scenario=cost_scenario,
        )
        if self.acceptance != recomputed_acceptance:
            raise ValueError("N=32 acceptance does not recompute from raw evidence")
        for name, value in (
            ("N=32 wall seconds", self.wall_seconds),
            ("N=32 CPU seconds", self.cpu_seconds),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isinstance(self.peak_rss_bytes, int) or self.peak_rss_bytes < 0:
            raise ValueError("N=32 peak RSS must be a nonnegative integer")
        expected = _digest(
            f"{_HASH_DOMAIN}.n32.scientific_result",
            prospective_n32_followup_scientific_payload(
                source_revision=self.source_revision,
                protocol_digest=self.protocol_digest,
                parent_payload_digest=self.parent_payload_digest,
                parent_protocol_digest=self.parent_protocol_digest,
                parent_scientific_result_digest=(
                    self.parent_scientific_result_digest
                ),
                block_results=blocks,
                acceptance=self.acceptance,
            ),
        )
        if self.scientific_result_digest != expected:
            raise ValueError("N=32 scientific result digest is invalid")


@dataclass(frozen=True)
class SavedProspectivePilotArtifacts:
    json_path: Path
    report_path: Path
    hash_path: Path
    json_sha256: str
    report_sha256: str
    archive_size_bytes: int


def _validate_revision(value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("source revision must be a lowercase Git/SHA digest")


def _validate_sha256(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _decoded_canonical_archive(archive_bytes: bytes, label: str) -> dict:
    """Return the exact JSON-reader view of canonical archive bytes."""

    decoded = json.loads(archive_bytes)
    if not isinstance(decoded, dict):
        raise RuntimeError(f"{label} canonical JSON did not decode to an object")
    if _canonical_bytes(decoded) + b"\n" != archive_bytes:
        raise RuntimeError(f"{label} canonical JSON did not round-trip exactly")
    return decoded


def _digest(domain: str, value: object) -> str:
    return hashlib.sha256(
        domain.encode("utf-8") + b"\0" + _canonical_bytes(value)
    ).hexdigest()


def _strict_json_value(value: object) -> object:
    """Convert nested study objects to JSON without emitting NaN/Infinity."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return {"nonfinite": "nan"}
        return {
            "nonfinite": (
                "positive_infinity" if value > 0.0 else "negative_infinity"
            )
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _strict_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _strict_json_value(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_")
        }
    if hasattr(value, "_asdict"):
        return _strict_json_value(value._asdict())  # type: ignore[attr-defined]
    if isinstance(value, (tuple, list)):
        return [_strict_json_value(item) for item in value]
    raise TypeError(f"cannot archive value of type {type(value).__name__}")


def prospective_partition_plan_payload() -> dict:
    return {
        "campaign": PROSPECTIVE_CAMPAIGN,
        "cases_per_block": len(STAGE3_TRUTH_CONDITIONS),
        "partitions": [asdict(item) for item in PROSPECTIVE_PARTITION_PLAN],
    }


def prospective_pilot_protocol_digest(
    physical_config: OperatingDecisionRealismConfig,
    selector_rule: ProspectiveSelectorRule,
    cost_scenario: ProspectiveCostScenario,
    source_revision: str,
) -> str:
    """Bind source, physics, scoring, cost, selector, and pilot decisions."""

    _validate_revision(source_revision)
    strict_config = ProspectiveUncertaintyConfig(
        draw_count=PILOT_GENERATED_DRAW_COUNT,
        max_unstable_draws_per_source_action=PILOT_STRICT_MAX_UNSTABLE,
    )
    material = {
        "schema_version": PROSPECTIVE_PILOT_SCHEMA_VERSION,
        "protocol_version": PROSPECTIVE_PILOT_PROTOCOL_VERSION,
        "source_revision": source_revision,
        "partition_plan": prospective_partition_plan_payload(),
        "superseded_pilot_partitions": list(
            PROSPECTIVE_SUPERSEDED_PILOT_PARTITIONS
        ),
        "draw_counts": list(PILOT_DRAW_COUNTS),
        "generated_draw_count": PILOT_GENERATED_DRAW_COUNT,
        "strict_max_unstable_draws_per_source_action": (
            PILOT_STRICT_MAX_UNSTABLE
        ),
        "sensitivity_max_unstable_draws_per_source_action": (
            PILOT_SENSITIVITY_MAX_UNSTABLE
        ),
        "acceptance_rule": asdict(PilotAcceptanceRule()),
        "conditional_n32_followup": prospective_n32_followup_design_payload(),
        "physical_protocol_digest": corrected_physical_protocol_digest(
            physical_config
        ),
        "uncertainty_protocol_digest": prospective_uncertainty_protocol_digest(
            physical_config,
            strict_config,
        ),
        "cost_protocol_digest": prospective_cost_protocol_digest(
            physical_config,
            strict_config,
            cost_scenario,
            selector_rule,
        ),
        "selector_rule": prospective_selector_rule_payload(selector_rule),
        "sequence": {
            "predictive_draws": "generate_16_once_then_authenticated_prefixes",
            "selection_inputs": "common_acquisition_only",
            "reveal": "after_every_fixed_policy_decision_is_saved",
            "parallel_unit": "paired_block",
            "output_order": "block_then_truth_family_then_draw_count",
        },
    }
    return _digest(f"{_HASH_DOMAIN}.protocol", material)


def _validate_followup_case_matrix(
    block_results: Sequence[Mapping[str, object]],
) -> Tuple[Mapping[str, object], ...]:
    ordered = tuple(sorted(block_results, key=lambda item: int(item["block"])))
    if tuple(int(item["block"]) for item in ordered) != tuple(
        range(PILOT_BLOCK_COUNT)
    ):
        raise ValueError("N=32 follow-up needs all four ordered blocks")
    cases = tuple(
        case
        for block in ordered
        for case in block.get("cases", ())  # type: ignore[union-attr]
    )
    if any(
        case.get("block") != block.get("block")
        for block in ordered
        for case in block.get("cases", ())  # type: ignore[union-attr]
    ):
        raise ValueError("N=32 follow-up case is nested under the wrong block")
    expected = {
        (block, truth_condition)
        for block in range(PILOT_BLOCK_COUNT)
        for truth_condition in STAGE3_TRUTH_CONDITIONS
    }
    actual = {
        (case.get("block"), case.get("truth_condition")) for case in cases
    }
    if len(cases) != len(expected) or actual != expected:
        raise ValueError("N=32 follow-up case identities must be complete and unique")
    return cases


def _parent_selector_and_cost(
    payload: Mapping[str, object],
) -> Tuple[ProspectiveSelectorRule, ProspectiveCostScenario]:
    """Recover the frozen selector/cost pair retained in every pilot scorecard."""

    try:
        selector_rule = prospective_selector_rule_from_payload(
            payload["selector_rule"]
        )
        scenario = ProspectiveCostScenario(**payload["cost_scenario"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "parent pilot archive does not retain its selector/cost configuration"
        ) from error
    return selector_rule, scenario


def _expected_corrected_stream_manifest(block: int, label: str) -> list:
    registry = RandomStreamRegistry()
    shared_truth_purposes = (
        "physical_parameter_0",
        "physical_parameter_1",
        "physical_parameter_2",
        "interface_mass",
        "series_resistance",
        "face_sensor_capacitance",
        "face_sensor_response",
    )
    for purpose in shared_truth_purposes:
        stream = RandomStream(
            RandomStreamKey(
                protocol_version=RANDOM_STREAM_PROTOCOL_VERSION,
                campaign=PROSPECTIVE_CAMPAIGN,
                partition=PROSPECTIVE_PILOT_PARTITION,
                block=block,
                stream_kind="device_truth",
                purpose=purpose,
            )
        )
        pairing_id = (
            f"{PROSPECTIVE_CAMPAIGN}/{PROSPECTIVE_PILOT_PARTITION}/{block}/"
            f"shared-truth/{purpose}"
        )
        for family in STAGE3_TRUTH_CONDITIONS:
            registry.register(
                stream,
                consumer=f"{family}/block-{block}/truth/{purpose}",
                pairing_member=family,
                pairing_id=pairing_id,
            )
    contact_family = STAGE3_TRUTH_CONDITIONS[2]
    registry.register(
        RandomStream(
            RandomStreamKey(
                protocol_version=RANDOM_STREAM_PROTOCOL_VERSION,
                campaign=PROSPECTIVE_CAMPAIGN,
                partition=PROSPECTIVE_PILOT_PARTITION,
                block=block,
                stream_kind="device_truth",
                purpose="contact_beta",
                family=contact_family,
            )
        ),
        consumer=f"{contact_family}/block-{block}/truth/contact_beta",
    )
    policies = default_fixed_policies()
    if label == "N=32":
        policies = tuple(policy for policy in policies if policy.name == STOP_NOW)
    elif label != "parent pilot":
        raise ValueError("corrected random-stream inventory label is unknown")
    initial = initial_acquisition_regime()
    for family in STAGE3_TRUTH_CONDITIONS:
        for policy in policies:
            acquisition_regimes = (initial, *policy.additional_regimes)
            installed = {
                channel for regime in acquisition_regimes for channel in regime.channels
            }
            regimes = (*acquisition_regimes, ("fixed_verification", tuple(
                channel for channel in ALL_CHANNELS if channel in installed
            )))
            for regime in regimes:
                if isinstance(regime, tuple):
                    regime_name, channels = regime
                else:
                    regime_name, channels = regime.name, regime.channels
                instrumented = COLD_FACE in channels
                run_name = (
                    f"{regime_name}|temporary_face_sensor={int(instrumented)}"
                )
                for channel in channels:
                    for purpose in ("run_bias", "white_noise"):
                        stream = RandomStream(
                            RandomStreamKey(
                                protocol_version=RANDOM_STREAM_PROTOCOL_VERSION,
                                campaign=PROSPECTIVE_CAMPAIGN,
                                partition=PROSPECTIVE_PILOT_PARTITION,
                                block=block,
                                stream_kind="observation",
                                purpose=purpose,
                                family=family,
                                run=run_name,
                                channel=channel,
                            )
                        )
                        pairing_id = (
                            f"{PROSPECTIVE_CAMPAIGN}/{PROSPECTIVE_PILOT_PARTITION}/"
                            f"{family}/{block}/{run_name}/{channel}/{purpose}"
                        )
                        registry.register(
                            stream,
                            consumer=(
                                f"{family}/block-{block}/{policy.name}/{run_name}/"
                                f"{channel}/{purpose}"
                            ),
                            pairing_member=policy.name,
                            pairing_id=pairing_id,
                        )
    return _stream_manifest_payload(registry.uses)


def _validate_corrected_stream_records(
    manifest: object,
    audit: object,
    *,
    block: int,
    label: str,
    cases: Optional[Sequence[Mapping[str, object]]] = None,
) -> None:
    if not isinstance(manifest, list) or not manifest:
        raise ValueError(f"{label} corrected random-stream manifest is missing")
    if not isinstance(audit, Mapping):
        raise ValueError(f"{label} corrected random-stream audit is missing")
    registry = RandomStreamRegistry()
    for raw in manifest:
        item = _exact_mapping(
            raw,
            {"key", "seed", "consumer", "pairing_member", "pairing_id"},
            f"{label} corrected random-stream use",
        )
        key_payload = _exact_mapping(
            item["key"],
            {
                "protocol_version",
                "campaign",
                "partition",
                "block",
                "stream_kind",
                "purpose",
                "family",
                "run",
                "channel",
            },
            f"{label} corrected random-stream key",
        )
        key = RandomStreamKey(**key_payload)
        stream = RandomStream(key)
        if (
            stream.seed != item["seed"]
            or key.campaign != PROSPECTIVE_CAMPAIGN
            or key.partition != PROSPECTIVE_PILOT_PARTITION
            or key.block != block
        ):
            raise ValueError(f"{label} corrected random-stream namespace is invalid")
        registry.register(
            stream,
            consumer=item["consumer"],
            pairing_member=item["pairing_member"],
            pairing_id=item["pairing_id"],
        )
    recomputed = registry.audit()
    recomputed.assert_clean()
    if audit != _strict_json_value(recomputed):
        raise ValueError(f"{label} corrected random-stream audit is invalid")
    expected = _expected_corrected_stream_manifest(block, label)
    actual_multiset = sorted(
        json.dumps(item, sort_keys=True, separators=(",", ":"))
        for item in manifest
    )
    expected_multiset = sorted(
        json.dumps(item, sort_keys=True, separators=(",", ":"))
        for item in expected
    )
    if label == "N=32" and cases is not None:
        actual_truth = sorted(
            value
            for value, item in zip(actual_multiset, sorted(
                manifest,
                key=lambda entry: json.dumps(
                    entry, sort_keys=True, separators=(",", ":")
                ),
            ))
            if item["key"]["stream_kind"] == "device_truth"
        )
        expected_truth = sorted(
            json.dumps(item, sort_keys=True, separators=(",", ":"))
            for item in expected
            if item["key"]["stream_kind"] == "device_truth"
        )
        if actual_truth != expected_truth:
            raise ValueError(f"{label} corrected random-stream inventory is invalid")
        expected_by_family = {
            family: [
                json.dumps(item, sort_keys=True, separators=(",", ":"))
                for item in expected
                if item["key"]["stream_kind"] == "observation"
                and item["key"]["family"] == family
            ]
            for family in STAGE3_TRUTH_CONDITIONS
        }
        actual_by_family = {
            family: [
                json.dumps(item, sort_keys=True, separators=(",", ":"))
                for item in manifest
                if item["key"]["stream_kind"] == "observation"
                and item["key"]["family"] == family
            ]
            for family in STAGE3_TRUTH_CONDITIONS
        }
        if sum(len(items) for items in actual_by_family.values()) != sum(
            item["key"]["stream_kind"] == "observation" for item in manifest
        ):
            raise ValueError(f"{label} corrected random-stream inventory is invalid")
        case_by_family = {case["truth_condition"]: case for case in cases}
        for family in STAGE3_TRUTH_CONDITIONS:
            case = case_by_family[family]
            actual_family = actual_by_family[family]
            expected_family = expected_by_family[family]
            if case.get("device_token") is not None:
                valid = actual_family == expected_family
            else:
                valid = actual_family == expected_family[: len(actual_family)]
            if not valid:
                raise ValueError(
                    f"{label} corrected random-stream inventory is invalid"
                )
    elif actual_multiset != expected_multiset:
        raise ValueError(f"{label} corrected random-stream inventory is invalid")


_UNCERTAINTY_PAYLOAD_KEYS = {
    "schema_version",
    "protocol_version",
    "protocol_digest",
    "result_digest",
    "physical_protocol_digest",
    "acquisition_evidence_digest",
    "physical_config",
    "acquisition_evidence",
    "config",
    "action_uncertainties",
    "draw_outcomes",
    "stream_uses",
    "stream_audit",
}


def _exact_mapping(value: object, keys: set, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} is malformed")
    return value


def _uncertainty_config_payload_for(config: ProspectiveUncertaintyConfig) -> dict:
    return {
        "schema_version": PROSPECTIVE_UNCERTAINTY_SCHEMA_VERSION,
        "protocol_version": PROSPECTIVE_UNCERTAINTY_PROTOCOL_VERSION,
        "estimator": config.estimator,
        "uncertainty_metric": config.uncertainty_metric,
        "draw_count": config.draw_count,
        "max_parameter_draw_attempts": config.max_parameter_draw_attempts,
        "max_unstable_draws_per_source_action": (
            config.max_unstable_draws_per_source_action
        ),
        "eligibility_protocol": PROSPECTIVE_ELIGIBILITY_PROTOCOL,
        "within_generator_aggregation": config.within_generator_aggregation,
        "across_generator_aggregation": config.across_generator_aggregation,
        "draw_failure_policy": config.draw_failure_policy,
        "candidate_attrition_policy": config.candidate_attrition_policy,
        "refit_start_protocol": config.refit_start_protocol,
    }


def _validated_candidate_fit_from_payload(
    fit: Mapping[str, object],
    physical_config: OperatingDecisionRealismConfig,
) -> RealisticCandidateFit:
    """Reconstruct a fit and authenticate its inexpensive numerical invariants."""

    objective = fit["objective"]
    if (
        isinstance(objective, bool)
        or not isinstance(objective, (int, float))
        or not math.isfinite(float(objective))
        or float(objective) < 0.0
    ):
        raise ValueError("complete acquisition fit objective is invalid")
    model_name = str(fit["model_name"])
    spec = _parameter_spec(model_name, False, physical_config)
    offsets = tuple(float(value) for value in fit["log_multipliers"])
    covariance = tuple(
        tuple(float(value) for value in row) for row in fit["covariance"]
    )
    _positive_semidefinite_cholesky(covariance)
    face_payload = fit["face_sensor"]
    if face_payload is None:
        face_sensor = None
    else:
        face_payload = _exact_mapping(
            face_payload,
            {"thermal_capacitance", "response_time_constant"},
            "complete acquisition fit face sensor",
        )
        face_sensor = TemporaryFaceSensor(
            thermal_capacitance=face_payload["thermal_capacitance"],
            response_time_constant=face_payload["response_time_constant"],
        )
    decoded = _decoded_parameters(model_name, offsets, spec)
    serialized_decoded = (
        tuple(float(value) for value in fit["physical_values"]),
        None if fit["interface_mass"] is None else float(fit["interface_mass"]),
        float(fit["series_resistance"]),
        face_sensor,
    )
    if decoded != serialized_decoded:
        raise ValueError("complete acquisition fit values do not decode from offsets")
    return RealisticCandidateFit(
        model_name=model_name,
        log_multipliers=offsets,
        parameter_names=tuple(fit["parameter_names"]),
        physical_values=serialized_decoded[0],
        interface_mass=serialized_decoded[1],
        series_resistance=serialized_decoded[2],
        face_sensor=face_sensor,
        objective=float(objective),
        covariance=covariance,
        reached_bound=fit["reached_bound"],
        evaluation_count=fit["evaluation_count"],
        converged=fit["converged"],
        termination_reason=fit["termination_reason"],
        completed_iterations=fit["completed_iterations"],
        accepted_iterations=fit["accepted_iterations"],
        scaled_gradient_infinity_norm=fit["scaled_gradient_infinity_norm"],
        last_step_infinity_norm=fit["last_step_infinity_norm"],
        last_relative_objective_reduction=fit[
            "last_relative_objective_reduction"
        ],
    )


def _validate_interval_formula(
    interval: Mapping[str, object],
    multiplier: float,
    label: str,
) -> None:
    estimate = float(interval["estimate"])
    standard_error = float(interval["local_standard_error"])
    lower = float(interval["lower"])
    upper = float(interval["upper"])
    if (
        not all(
            math.isfinite(value)
            for value in (estimate, standard_error, lower, upper)
        )
        or standard_error < 0.0
        or not math.isclose(
            lower,
            estimate - multiplier * standard_error,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            upper,
            estimate + multiplier * standard_error,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError(f"{label} does not match estimate +/- multiplier * SE")


def _validate_exact_prospective_draws(
    draws: Sequence[Mapping[str, object]],
    *,
    fits_by_model: Mapping[str, RealisticCandidateFit],
    source_models: Sequence[str],
    acquisition_evidence_digest: str,
    block: int,
    config: ProspectiveUncertaintyConfig,
    physical_config: OperatingDecisionRealismConfig,
) -> None:
    """Replay the production parameter/probe RNG without rerunning simulations.

    Archive replay authenticates serialized evidence and cheap deterministic
    invariants. Predictive simulations and candidate refits remain outside this
    validator; the normal runner validates those in memory and deterministically
    regenerates the N=16 prefix from the complete N=32 result.
    """

    namespace = ProspectiveRandomStreamNamespace(
        campaign=PROSPECTIVE_CAMPAIGN,
        partition=PROSPECTIVE_PILOT_PARTITION,
        block=block,
        acquisition_evidence_digest=acquisition_evidence_digest,
    )
    by_source_draw = {}
    for draw in draws:
        by_source_draw.setdefault(
            (draw["generator_model"], draw["draw_index"]), []
        ).append(draw)
    for source in source_models:
        fit = fits_by_model[source]
        for draw_index in range(config.draw_count):
            selected = tuple(by_source_draw[(source, draw_index)])
            registry = ProspectiveRandomStreamRegistry()
            try:
                expected_parameters = _generator_parameter_draw(
                    fit,
                    namespace,
                    draw_index,
                    physical_config,
                    config,
                    registry,
                )
            except _ProspectiveDrawSamplingError:
                if not all(
                    draw["failed"] is True
                    and draw["failure_stage"] == "prospective_parameter_draw"
                    and draw["generator_log_offsets"] == []
                    for draw in selected
                ):
                    raise ValueError(
                        "parameter sampling failure does not match deterministic stream"
                    )
                continue
            if any(
                tuple(draw["generator_log_offsets"]) != expected_parameters
                for draw in selected
            ):
                raise ValueError(
                    "prospective parameter draw does not match deterministic stream"
                )
            if any(
                draw["failure_stage"] == "prospective_parameter_draw"
                for draw in selected
            ):
                raise ValueError(
                    "successful parameter sampling cannot retain a parameter failure"
                )
            face = next(
                draw
                for draw in selected
                if draw["policy_name"] == "fixed_face_temperature"
            )
            if any(
                draw["policy_name"] != "fixed_face_temperature"
                and draw["failure_stage"] == "prospective_probe_draw"
                for draw in selected
            ):
                raise ValueError("probe sampling failure belongs only to face action")
            try:
                expected_probe = _face_probe_draw(
                    namespace,
                    source,
                    draw_index,
                    physical_config,
                    config,
                    registry,
                )
            except _ProspectiveDrawSamplingError:
                if not (
                    face["failed"] is True
                    and face["failure_stage"] == "prospective_probe_draw"
                    and face["face_probe_log_offsets"] is None
                ):
                    raise ValueError(
                        "probe sampling failure does not match deterministic stream"
                    )
            else:
                probe = face["face_probe_log_offsets"]
                if (
                    face["failure_stage"] == "prospective_probe_draw"
                    or probe is None
                    or tuple(probe) != expected_probe
                ):
                    raise ValueError(
                        "prospective probe draw does not match deterministic stream"
                    )


def _recomputed_action_uncertainties(
    draws: Sequence[Mapping[str, object]],
    *,
    source_models: Sequence[str],
    baseline_width: float,
    config: ProspectiveUncertaintyConfig,
    initially_admissible: Optional[Sequence[str]] = None,
    initially_excluded: Sequence[str] = (),
    physical_config: Optional[OperatingDecisionRealismConfig] = None,
) -> list:
    expected_ids = tuple(
        sorted(
            (source, policy, draw_index)
            for source in source_models
            for policy in POLICY_NAMES
            if policy != STOP_NOW
            for draw_index in range(config.draw_count)
        )
    )
    observed_ids = tuple(
        (
            str(draw.get("generator_model")),
            str(draw.get("policy_name")),
            draw.get("draw_index"),
        )
        for draw in draws
    )
    if observed_ids != expected_ids:
        raise ValueError("complete uncertainty draw matrix is incomplete or unordered")
    draw_keys = {
        "policy_name",
        "generator_model",
        "draw_index",
        "generator_log_offsets",
        "face_probe_log_offsets",
        "candidate_outcomes",
        "initially_admissible_became_inadmissible",
        "initially_excluded_became_admissible",
        "stable",
        "failed",
        "failure_stage",
        "failure_model",
        "failure_type",
        "raw_after_width",
        "scored_after_width",
        "synthetic_observation_digest",
    }
    candidate_keys = {"model_name", "status", "objective", "interval"}
    initial_admissible_set = set(
        source_models if initially_admissible is None else initially_admissible
    )
    initial_excluded_set = set(initially_excluded)
    for draw in draws:
        _exact_mapping(draw, draw_keys, "prospective raw draw")
        candidate_outcomes = []
        for candidate in draw["candidate_outcomes"]:
            candidate = _exact_mapping(
                candidate,
                candidate_keys,
                "prospective raw candidate outcome",
            )
            interval = candidate["interval"]
            if interval is not None and physical_config is not None:
                _validate_interval_formula(
                    interval,
                    physical_config.local_interval_multiplier,
                    "prospective candidate interval",
                )
            candidate_outcomes.append(
                ProspectiveCandidateOutcome(
                    model_name=candidate["model_name"],
                    status=candidate["status"],
                    objective=candidate["objective"],
                    interval=(
                        None
                        if interval is None
                        else MarginInterval(**interval)
                    ),
                )
            )
        loaded_draw = ProspectiveDrawOutcome(
            policy_name=draw["policy_name"],
            generator_model=draw["generator_model"],
            draw_index=draw["draw_index"],
            generator_log_offsets=tuple(draw["generator_log_offsets"]),
            face_probe_log_offsets=(
                None
                if draw["face_probe_log_offsets"] is None
                else tuple(draw["face_probe_log_offsets"])
            ),
            candidate_outcomes=tuple(candidate_outcomes),
            initially_admissible_became_inadmissible=tuple(
                draw["initially_admissible_became_inadmissible"]
            ),
            initially_excluded_became_admissible=tuple(
                draw["initially_excluded_became_admissible"]
            ),
            stable=draw["stable"],
            failed=draw["failed"],
            failure_stage=draw["failure_stage"],
            failure_model=draw["failure_model"],
            failure_type=draw["failure_type"],
            raw_after_width=draw["raw_after_width"],
            scored_after_width=draw["scored_after_width"],
            synthetic_observation_digest=draw[
                "synthetic_observation_digest"
            ],
        )
        if loaded_draw.failed:
            allowed_failure_stages = {
                "prospective_parameter_draw",
                "prospective_probe_draw",
                "prospective_simulation",
                "prospective_refit",
                "prospective_uncertainty",
            }
            if loaded_draw.failure_stage not in allowed_failure_stages:
                raise ValueError("prospective draw has an unknown failure stage")
            upstream_failure = loaded_draw.failure_stage in {
                "prospective_parameter_draw",
                "prospective_probe_draw",
                "prospective_simulation",
            }
            if upstream_failure:
                if (
                    loaded_draw.candidate_outcomes
                    or loaded_draw.synthetic_observation_digest is not None
                    or loaded_draw.failure_model is not None
                ):
                    raise ValueError(
                        "upstream prospective failure retained downstream evidence"
                    )
                if (
                    loaded_draw.failure_stage == "prospective_probe_draw"
                    and loaded_draw.policy_name != "fixed_face_temperature"
                ):
                    raise ValueError(
                        "prospective probe failure belongs only to the face action"
                    )
                if (
                    loaded_draw.failure_stage
                    in {"prospective_parameter_draw", "prospective_probe_draw"}
                    and loaded_draw.failure_type
                    != "_ProspectiveDrawSamplingError"
                ):
                    raise ValueError(
                        "prospective sampling failure type is inconsistent"
                    )
                if (
                    loaded_draw.failure_stage == "prospective_simulation"
                    and loaded_draw.failure_type
                    not in {
                        "ArithmeticError",
                        "IntegrationDivergenceError",
                        "ValueError",
                    }
                ):
                    raise ValueError(
                        "prospective simulation failure type is inconsistent"
                    )
            else:
                if (
                    tuple(
                        item.model_name
                        for item in loaded_draw.candidate_outcomes
                    )
                    != tuple(sorted(MODEL_NAMES))
                    or loaded_draw.synthetic_observation_digest is None
                ):
                    raise ValueError(
                        "downstream prospective failure lacks candidate evidence"
                    )
                status_by_model = {
                    item.model_name: item.status
                    for item in loaded_draw.candidate_outcomes
                }
                refit_failures = tuple(
                    sorted(
                        model
                        for model, status in status_by_model.items()
                        if status == "acquisition_fit_failure"
                    )
                )
                uncertainty_failures = tuple(
                    sorted(
                        model
                        for model, status in status_by_model.items()
                        if status == "uncertainty_failure"
                    )
                )
                if loaded_draw.failure_stage == "prospective_refit":
                    if (
                        not refit_failures
                        or loaded_draw.failure_model != refit_failures[0]
                    ):
                        raise ValueError(
                            "prospective refit failure model is inconsistent"
                        )
                elif refit_failures:
                    raise ValueError(
                        "prospective uncertainty failure retained a refit failure"
                    )
                elif uncertainty_failures:
                    if loaded_draw.failure_model != uncertainty_failures[0]:
                        raise ValueError(
                            "prospective uncertainty failure model is inconsistent"
                        )
                elif (
                    loaded_draw.failure_model is not None
                    or loaded_draw.failure_type != "NoAdmissibleCandidate"
                ):
                    raise ValueError(
                        "no-envelope prospective failure is inconsistent"
                    )
            if (
                loaded_draw.scored_after_width != baseline_width
                or loaded_draw.initially_admissible_became_inadmissible
                or loaded_draw.initially_excluded_became_admissible
            ):
                raise ValueError(
                    "failed prospective draw must retain the baseline width"
                )
        else:
            intervals = tuple(
                outcome.interval
                for outcome in loaded_draw.candidate_outcomes
                if outcome.status == "admissible" and outcome.interval is not None
            )
            lower = min(interval.lower for interval in intervals)
            upper = max(interval.upper for interval in intervals)
            raw_width = upper - lower
            now_admissible = {interval.model_name for interval in intervals}
            newly_excluded = tuple(
                sorted(initial_admissible_set - now_admissible)
            )
            recovered = tuple(
                sorted(initial_excluded_set.intersection(now_admissible))
            )
            scored_width = (
                max(raw_width, baseline_width) if newly_excluded else raw_width
            )
            if (
                loaded_draw.raw_after_width != raw_width
                or loaded_draw.scored_after_width != scored_width
                or loaded_draw.initially_admissible_became_inadmissible
                != newly_excluded
                or loaded_draw.initially_excluded_became_admissible != recovered
            ):
                raise ValueError(
                    "prospective draw does not match candidate outcomes or "
                    "the conservative attrition floor"
                )
    if physical_config is not None:
        by_source_draw = {}
        for draw in draws:
            by_source_draw.setdefault(
                (draw["generator_model"], draw["draw_index"]),
                [],
            ).append(draw)
        for source in source_models:
            spec = _parameter_spec(source, False, physical_config)
            for draw_index in range(config.draw_count):
                selected = tuple(by_source_draw[(source, draw_index)])
                parameter_failed = all(
                    draw["failed"] is True
                    and draw["failure_stage"] == "prospective_parameter_draw"
                    for draw in selected
                )
                if parameter_failed:
                    if any(draw["generator_log_offsets"] for draw in selected):
                        raise ValueError(
                            "failed parameter draw cannot retain offsets"
                        )
                else:
                    offsets = {
                        tuple(draw["generator_log_offsets"])
                        for draw in selected
                    }
                    if len(offsets) != 1:
                        raise ValueError(
                            "physical draw must be shared across actions"
                        )
                    values = next(iter(offsets))
                    if len(values) != len(spec.names) or any(
                        value < lower or value > upper
                        for value, (lower, upper) in zip(
                            values,
                            spec.log_bounds,
                        )
                    ):
                        raise ValueError(
                            "prospective physical draw is outside fit bounds"
                        )
                face = next(
                    draw
                    for draw in selected
                    if draw["policy_name"] == "fixed_face_temperature"
                )
                if parameter_failed or face["failure_stage"] == "prospective_probe_draw":
                    if face["face_probe_log_offsets"] is not None:
                        raise ValueError(
                            "failed probe draw cannot retain probe offsets"
                        )
                else:
                    probe = face["face_probe_log_offsets"]
                    bounds = (
                        tuple(
                            math.log(
                                value
                                / physical_config.face_sensor_capacitance_nominal
                            )
                            for value in physical_config.face_sensor_capacitance_bounds
                        ),
                        tuple(
                            math.log(
                                value
                                / physical_config.face_sensor_response_nominal
                            )
                            for value in physical_config.face_sensor_response_bounds
                        ),
                    )
                    if probe is None or len(probe) != 2 or any(
                        value < lower or value > upper
                        for value, (lower, upper) in zip(probe, bounds)
                    ):
                        raise ValueError(
                            "prospective probe draw is outside prior bounds"
                        )
    actions = [
        {
            "policy_name": STOP_NOW,
            "eligible": True,
            "failure_reason": None,
            "uncertainty_before": baseline_width,
            "expected_uncertainty_after": baseline_width,
            "expected_uncertainty_reduction": 0.0,
            "prospective_draw_count": 0,
            "source_summaries": [],
        }
    ]
    for policy in POLICY_NAMES:
        if policy == STOP_NOW:
            continue
        summaries = []
        for source in sorted(source_models):
            selected = tuple(
                draw
                for draw in draws
                if draw["policy_name"] == policy
                and draw["generator_model"] == source
            )
            stable_count = sum(draw.get("stable") is True for draw in selected)
            failed_count = sum(draw.get("failed") is True for draw in selected)
            if any(
                type(draw.get("stable")) is not bool
                or type(draw.get("failed")) is not bool
                or draw.get("stable") == draw.get("failed")
                or not isinstance(draw.get("scored_after_width"), (int, float))
                or isinstance(draw.get("scored_after_width"), bool)
                or not math.isfinite(float(draw["scored_after_width"]))
                for draw in selected
            ):
                raise ValueError("complete uncertainty draw status is invalid")
            for draw in selected:
                scored_width = float(draw["scored_after_width"])
                lost = tuple(draw["initially_admissible_became_inadmissible"])
                if draw["failed"] is True:
                    if scored_width != baseline_width:
                        raise ValueError(
                            "failed uncertainty draw must retain the baseline floor"
                        )
                else:
                    raw_width = draw.get("raw_after_width")
                    if (
                        not isinstance(raw_width, (int, float))
                        or isinstance(raw_width, bool)
                        or not math.isfinite(float(raw_width))
                        or scored_width
                        != max(
                            baseline_width if lost else float(raw_width),
                            float(raw_width),
                        )
                    ):
                        raise ValueError(
                            "completed uncertainty draw violates attrition scoring"
                        )
            eligible = (
                len(selected) - stable_count
                <= config.max_unstable_draws_per_source_action
            )
            summaries.append(
                {
                    "policy_name": policy,
                    "generator_model": source,
                    "draw_count": len(selected),
                    "stable_draw_count": stable_count,
                    "failed_draw_count": failed_count,
                    "unstable_draw_count": len(selected) - stable_count,
                    "max_unstable_draws": (
                        config.max_unstable_draws_per_source_action
                    ),
                    "mean_after_width": statistics.fmean(
                        float(draw["scored_after_width"]) for draw in selected
                    ),
                    "eligible": eligible,
                }
            )
        after = max(item["mean_after_width"] for item in summaries)
        eligible = all(item["eligible"] for item in summaries)
        actions.append(
            {
                "policy_name": policy,
                "eligible": eligible,
                "failure_reason": (
                    None if eligible else "insufficient_stable_prospective_draws"
                ),
                "uncertainty_before": baseline_width,
                "expected_uncertainty_after": after,
                "expected_uncertainty_reduction": baseline_width - after,
                "prospective_draw_count": sum(
                    item["draw_count"] for item in summaries
                ),
                "source_summaries": summaries,
            }
        )
    return actions


def _stream_audit_from_payload(
    stream_uses: Sequence[Mapping[str, object]],
    *,
    draws: Optional[Sequence[Mapping[str, object]]] = None,
    source_models: Sequence[str],
    draw_count: int,
    acquisition_evidence_digest: str,
    block: int,
) -> dict:
    registry = ProspectiveRandomStreamRegistry()
    parameter_sharing = set()
    reconstructed = []
    actions = tuple(name for name in POLICY_NAMES if name != STOP_NOW)
    for item in stream_uses:
        use = _exact_mapping(
            item,
            {"key", "seed", "consumer", "shared_for_action"},
            "prospective stream use",
        )
        key_payload = _exact_mapping(
            use["key"],
            {
                "domain",
                "protocol_version",
                "campaign",
                "partition",
                "block",
                "acquisition_evidence_digest",
                "generator_model",
                "draw_index",
                "purpose",
                "action",
                "run",
                "channel",
            },
            "prospective stream key",
        )
        if key_payload["domain"] != PROSPECTIVE_RANDOM_STREAM_DOMAIN:
            raise ValueError("prospective stream domain is invalid")
        key = ProspectiveRandomStreamKey(
            **{
                name: value
                for name, value in key_payload.items()
                if name != "domain"
            }
        )
        stream = ProspectiveRandomStream(key)
        if (
            str(stream.seed) != use["seed"]
            or key.block != block
            or key.campaign != PROSPECTIVE_CAMPAIGN
            or key.partition != PROSPECTIVE_PILOT_PARTITION
            or key.acquisition_evidence_digest != acquisition_evidence_digest
            or key.generator_model not in source_models
            or key.draw_index >= draw_count
        ):
            raise ValueError("prospective stream use is outside its case namespace")
        shared_for_action = use["shared_for_action"]
        if key.purpose == PARAMETER_DRAW:
            if shared_for_action not in actions:
                raise ValueError(
                    "prospective parameter stream sharing is incomplete"
                )
            expected_consumer = (
                f"{key.generator_model}/{key.draw_index}/"
                f"{shared_for_action}/physical_parameters"
            )
        elif key.purpose == PROBE_DRAW:
            if shared_for_action is not None:
                raise ValueError("prospective probe stream cannot be shared")
            expected_consumer = (
                f"{key.generator_model}/{key.draw_index}/"
                f"fixed_face_temperature/probe"
            )
        else:
            if shared_for_action is not None:
                raise ValueError("prospective observation stream cannot be shared")
            expected_consumer = (
                f"{key.generator_model}/{key.draw_index}/{key.action}/"
                f"{key.run}/{key.channel}/{key.purpose}"
            )
        if use["consumer"] != expected_consumer:
            raise ValueError("prospective stream consumer identity is invalid")
        registry.register(
            stream,
            consumer=use["consumer"],  # type: ignore[arg-type]
            shared_for_action=shared_for_action,  # type: ignore[arg-type]
        )
        reconstructed.append((key, shared_for_action))
        if key.purpose == "parameter_draw":
            parameter_sharing.add(
                (key.generator_model, key.draw_index, use["shared_for_action"])
            )
    expected_sharing = {
        (source, draw_index, policy)
        for source in source_models
        for draw_index in range(draw_count)
        for policy in POLICY_NAMES
        if policy != STOP_NOW
    }
    if parameter_sharing != expected_sharing:
        raise ValueError("prospective parameter stream sharing is incomplete")
    if draws is not None:
        policies = {
            policy.name: policy
            for policy in default_fixed_policies()
            if policy.name != STOP_NOW
        }
        draw_by_id = {
            (draw["generator_model"], draw["draw_index"], draw["policy_name"]): draw
            for draw in draws
        }
        for source in sorted(source_models):
            for draw_index in range(draw_count):
                selected = tuple(
                    (key, shared)
                    for key, shared in reconstructed
                    if key.generator_model == source
                    and key.draw_index == draw_index
                )
                parameter_uses = tuple(
                    (key, shared)
                    for key, shared in selected
                    if key.purpose == PARAMETER_DRAW
                )
                if (
                    len(parameter_uses) != len(actions)
                    or {shared for _, shared in parameter_uses} != set(actions)
                    or len({key for key, _ in parameter_uses}) != 1
                ):
                    raise ValueError(
                        "prospective parameter stream sharing is incomplete"
                    )
                action_draws = tuple(
                    draw_by_id[(source, draw_index, action)]
                    for action in actions
                )
                parameter_failed = all(
                    draw["failed"] is True
                    and draw["failure_stage"] == "prospective_parameter_draw"
                    for draw in action_draws
                )
                probe_uses = tuple(
                    key for key, _ in selected if key.purpose == PROBE_DRAW
                )
                if len(probe_uses) != (0 if parameter_failed else 1):
                    raise ValueError(
                        "prospective probe stream inventory is incomplete"
                    )
                observation_keys = tuple(
                    key
                    for key, _ in selected
                    if key.purpose in (RUN_BIAS, WHITE_NOISE)
                )
                for action in actions:
                    draw = draw_by_id[(source, draw_index, action)]
                    actual = tuple(
                        (key.purpose, key.action, key.run, key.channel)
                        for key in observation_keys
                        if key.action == action
                    )
                    expected = tuple(
                        (purpose, action, regime.name, channel)
                        for regime in policies[action].additional_regimes
                        for channel in regime.channels
                        for purpose in (RUN_BIAS, WHITE_NOISE)
                    )
                    if draw["failure_stage"] in (
                        "prospective_parameter_draw",
                        "prospective_probe_draw",
                    ):
                        valid = not actual
                    elif draw["failure_stage"] == "prospective_simulation":
                        valid = (
                            len(actual) < len(expected)
                            and actual == expected[: len(actual)]
                        )
                    else:
                        valid = actual == expected
                    if not valid:
                        raise ValueError(
                            "prospective observation stream inventory is incomplete"
                        )
    audit = registry.audit()
    audit.assert_clean()
    return _uncertainty_stream_audit_payload(audit)


def _validate_complete_uncertainty_payload(
    payload: object,
    *,
    physical_config: OperatingDecisionRealismConfig,
    block: int,
    draw_count: int,
    max_unstable_draws: int,
) -> dict:
    item = _exact_mapping(
        payload,
        _UNCERTAINTY_PAYLOAD_KEYS,
        "complete prospective uncertainty payload",
    )
    config = ProspectiveUncertaintyConfig(
        draw_count=draw_count,
        max_unstable_draws_per_source_action=max_unstable_draws,
    )
    if (
        item["schema_version"] != PROSPECTIVE_UNCERTAINTY_SCHEMA_VERSION
        or item["protocol_version"] != PROSPECTIVE_UNCERTAINTY_PROTOCOL_VERSION
        or item["config"] != _uncertainty_config_payload_for(config)
        or item["physical_config"] != _physical_config_payload(physical_config)
    ):
        raise ValueError("complete uncertainty protocol/config is invalid")
    physical_digest = corrected_physical_protocol_digest(physical_config)
    expected_protocol = prospective_uncertainty_protocol_digest(
        physical_config,
        config,
    )
    if (
        item["physical_protocol_digest"] != physical_digest
        or item["protocol_digest"] != expected_protocol
    ):
        raise ValueError("complete uncertainty physical/protocol digest is invalid")
    evidence = _exact_mapping(
        item["acquisition_evidence"],
        {
            "schema_version",
            "protocol_version",
            "physical_config",
            "physical_protocol_digest",
            "evidence_digest",
            "common_initial_run",
            "fit_set",
            "snapshot",
            "final_regime",
        },
        "complete acquisition evidence",
    )
    if (
        evidence["schema_version"] != PROSPECTIVE_UNCERTAINTY_SCHEMA_VERSION
        or evidence["protocol_version"] != PROSPECTIVE_UNCERTAINTY_PROTOCOL_VERSION
        or evidence["physical_protocol_digest"] != physical_digest
        or evidence["evidence_digest"] != item["acquisition_evidence_digest"]
        or evidence["physical_config"] != item["physical_config"]
    ):
        raise ValueError("complete acquisition evidence references are invalid")
    common_run = _exact_mapping(
        evidence["common_initial_run"],
        {"regime", "temporary_face_sensor", "observations"},
        "complete common initial run",
    )
    common_regime = _exact_mapping(
        common_run["regime"],
        {"name", "phase", "transition_times", "current_values", "channels"},
        "complete common initial regime",
    )
    final_regime = _exact_mapping(
        evidence["final_regime"],
        {"name", "phase", "transition_times", "current_values", "channels"},
        "complete final regime",
    )
    observations = common_run["observations"]
    initial_regime = initial_acquisition_regime()
    expected_common_regime = {
        "name": initial_regime.name,
        "phase": initial_regime.phase,
        "transition_times": [
            float(value) for value in initial_regime.current.transition_times
        ],
        "current_values": [
            float(value) for value in initial_regime.current.values
        ],
        "channels": list(initial_regime.channels),
    }
    expected_final_regime = {
        "name": "untouched_final_operating_schedule",
        "phase": "final_evaluation",
        "transition_times": [
            float(value)
            for value in physical_config.final_current.transition_times
        ],
        "current_values": [
            float(value) for value in physical_config.final_current.values
        ],
        "channels": [],
    }
    expected_observation_pairs = [
        (channel, time)
        for time in regular_measurement_times(
            RUN_DURATION_SECONDS,
            physical_config.sensor.sampling_interval,
        )
        for channel in (COLD_EXCHANGER, HOT_EXCHANGER)
    ]
    if (
        common_regime != expected_common_regime
        or common_run["temporary_face_sensor"] is not False
        or not isinstance(observations, list)
        or [
            (value.get("channel"), value.get("time"))
            for value in observations
            if isinstance(value, Mapping)
        ]
        != expected_observation_pairs
        or final_regime != expected_final_regime
    ):
        raise ValueError("complete acquisition/final regimes are invalid")
    for observation in observations:
        observation = _exact_mapping(
            observation,
            {"channel", "time", "value"},
            "complete acquisition observation",
        )
        if (
            observation["channel"] not in common_regime["channels"]
            or isinstance(observation["time"], bool)
            or not isinstance(observation["time"], (int, float))
            or not math.isfinite(float(observation["time"]))
            or isinstance(observation["value"], bool)
            or not isinstance(observation["value"], (int, float))
            or not math.isfinite(float(observation["value"]))
        ):
            raise ValueError("complete acquisition observation is invalid")
    fit_set = _exact_mapping(
        evidence["fit_set"],
        {"fits", "failures"},
        "complete acquisition fit set",
    )
    if not isinstance(fit_set["fits"], list) or not isinstance(
        fit_set["failures"], list
    ):
        raise ValueError("complete acquisition fit set is invalid")
    fit_keys = {
        "model_name",
        "log_multipliers",
        "parameter_names",
        "physical_values",
        "interface_mass",
        "series_resistance",
        "face_sensor",
        "objective",
        "covariance",
        "reached_bound",
        "evaluation_count",
        "converged",
        "termination_reason",
        "completed_iterations",
        "accepted_iterations",
        "scaled_gradient_infinity_norm",
        "last_step_infinity_norm",
        "last_relative_objective_reduction",
    }
    fits_by_model = {}
    candidate_fits_by_model = {}
    for raw_fit in fit_set["fits"]:
        fit = _exact_mapping(raw_fit, fit_keys, "complete acquisition fit")
        model_name = fit["model_name"]
        if model_name in fits_by_model or model_name not in MODEL_NAMES:
            raise ValueError("complete acquisition fits are not unique")
        spec = _parameter_spec(model_name, False, physical_config)
        multipliers = fit["log_multipliers"]
        covariance = fit["covariance"]
        if (
            fit["parameter_names"] != list(spec.names)
            or not isinstance(multipliers, list)
            or len(multipliers) != len(spec.names)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < lower
                or float(value) > upper
                for value, (lower, upper) in zip(multipliers, spec.log_bounds)
            )
            or not isinstance(covariance, list)
            or len(covariance) != len(spec.names)
            or any(
                not isinstance(row, list)
                or len(row) != len(spec.names)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in row
                )
                for row in covariance
            )
        ):
            raise ValueError("complete acquisition fit parameters are invalid")
        counts = (
            fit["evaluation_count"],
            fit["completed_iterations"],
            fit["accepted_iterations"],
        )
        optional_diagnostics = (
            fit["last_step_infinity_norm"],
            fit["last_relative_objective_reduction"],
        )
        allowed_termination_reasons = {
            "scaled_projected_gradient_tolerance",
            "step_and_objective_stagnation",
            "fixed_iteration_limit",
        }
        if (
            not isinstance(fit["reached_bound"], bool)
            or not isinstance(fit["converged"], bool)
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for value in counts
            )
            or fit["evaluation_count"] == 0
            or fit["completed_iterations"]
            != physical_config.sensor.fit_iterations
            or fit["accepted_iterations"] > fit["completed_iterations"]
            or not isinstance(fit["termination_reason"], str)
            or fit["termination_reason"] not in allowed_termination_reasons
            or fit["converged"]
            != (
                fit["termination_reason"]
                == "scaled_projected_gradient_tolerance"
            )
            or isinstance(fit["scaled_gradient_infinity_norm"], bool)
            or not isinstance(
                fit["scaled_gradient_infinity_norm"], (int, float)
            )
            or not math.isfinite(float(fit["scaled_gradient_infinity_norm"]))
            or float(fit["scaled_gradient_infinity_norm"]) < 0.0
            or any(
                value is not None
                and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                )
                for value in optional_diagnostics
            )
        ):
            raise ValueError("complete acquisition fit diagnostics are invalid")
        expected_reached_bound = any(
            abs(float(value) - lower) <= FIT_BOUND_TOLERANCE
            or abs(upper - float(value)) <= FIT_BOUND_TOLERANCE
            for value, (lower, upper) in zip(multipliers, spec.log_bounds)
        )
        gradient_converged = (
            float(fit["scaled_gradient_infinity_norm"])
            <= FIT_SCALED_GRADIENT_TOLERANCE
        )
        step_converged = (
            fit["last_step_infinity_norm"] is not None
            and fit["last_relative_objective_reduction"] is not None
            and float(fit["last_step_infinity_norm"]) <= FIT_STEP_TOLERANCE
            and float(fit["last_relative_objective_reduction"])
            <= FIT_RELATIVE_OBJECTIVE_TOLERANCE
        )
        expected_termination_reason = (
            "scaled_projected_gradient_tolerance"
            if gradient_converged
            else (
                "step_and_objective_stagnation"
                if step_converged
                else "fixed_iteration_limit"
            )
        )
        if (
            fit["reached_bound"] is not expected_reached_bound
            or fit["converged"] is not gradient_converged
            or fit["termination_reason"] != expected_termination_reason
        ):
            raise ValueError(
                "complete acquisition fit status does not match its diagnostics"
            )
        fits_by_model[model_name] = fit
        candidate_fits_by_model[model_name] = _validated_candidate_fit_from_payload(
            fit,
            physical_config,
        )
    failures_by_model = {}
    for raw_failure in fit_set["failures"]:
        failure = _exact_mapping(
            raw_failure,
            {"model_name", "stage", "error_type"},
            "complete acquisition fit failure",
        )
        model_name = failure["model_name"]
        if model_name in failures_by_model or model_name not in MODEL_NAMES:
            raise ValueError("complete acquisition failures are not unique")
        if (
            failure["stage"] != "acquisition_fit"
            or not isinstance(failure["error_type"], str)
            or not failure["error_type"].strip()
        ):
            raise ValueError("complete acquisition fit failure is invalid")
        failures_by_model[model_name] = failure
    if set(fits_by_model).intersection(failures_by_model) or set(
        fits_by_model
    ).union(failures_by_model) != set(MODEL_NAMES):
        raise ValueError("complete acquisition fit set does not cover candidates")
    snapshot = _exact_mapping(
        evidence["snapshot"],
        {
            "candidate_models",
            "admissible_candidate_models",
            "excluded_candidates",
            "failed_candidate_models",
            "model_intervals",
            "provisional_margin_envelope",
            "selection_failure_reason",
        },
        "complete acquisition snapshot",
    )
    source_models = tuple(snapshot["admissible_candidate_models"])  # type: ignore[arg-type]
    _snapshot_from_complete(item)
    for interval in snapshot["model_intervals"]:
        _validate_interval_formula(
            interval,
            physical_config.local_interval_multiplier,
            "acquisition snapshot interval",
        )
    exclusions = {
        value["model_name"]: value["reason"]
        for value in snapshot["excluded_candidates"]
    }
    expected_exclusions = {
        model: (
            "fit_reached_bound"
            if fit["reached_bound"] is True
            else "optimizer_not_converged"
        )
        for model, fit in fits_by_model.items()
        if fit["reached_bound"] is True or fit["converged"] is not True
    }
    expected_sources = tuple(
        sorted(set(fits_by_model).difference(expected_exclusions))
    )
    if (
        tuple(source_models) != expected_sources
        or exclusions != expected_exclusions
        or any(model not in fits_by_model for model in source_models)
        or set(snapshot["failed_candidate_models"]) != set(failures_by_model)
        or any(model not in fits_by_model for model in exclusions)
        or any(
            fits_by_model[model]["reached_bound"] is not False
            or fits_by_model[model]["converged"] is not True
            for model in source_models
        )
        or any(
            (reason == "fit_reached_bound" and fit["reached_bound"] is not True)
            or (
                reason == "optimizer_not_converged"
                and fit["converged"] is not False
            )
            for model, reason in exclusions.items()
            for fit in (fits_by_model[model],)
        )
    ):
        raise ValueError("acquisition snapshot does not match its fit set")
    expected_evidence_digest = _uncertainty_canonical_digest(
        {
            "domain": _ACQUISITION_EVIDENCE_DOMAIN,
            "common_initial_run": evidence["common_initial_run"],
            "fit_set": evidence["fit_set"],
            "snapshot": evidence["snapshot"],
            "final_regime": evidence["final_regime"],
            "physical_protocol_digest": physical_digest,
        }
    )
    if item["acquisition_evidence_digest"] != expected_evidence_digest:
        raise ValueError("complete acquisition evidence digest is invalid")
    envelope = snapshot["provisional_margin_envelope"]
    if not source_models or not isinstance(envelope, Mapping):
        raise ValueError("complete uncertainty needs a usable acquisition envelope")
    baseline_width = float(envelope["upper"]) - float(envelope["lower"])
    draws = item["draw_outcomes"]
    uses = item["stream_uses"]
    if not isinstance(draws, list) or not isinstance(uses, list):
        raise ValueError("complete uncertainty raw records must be lists")
    actions = _recomputed_action_uncertainties(
        draws,
        source_models=source_models,
        baseline_width=baseline_width,
        config=config,
        initially_admissible=source_models,
        initially_excluded=tuple(
            item["model_name"] for item in snapshot["excluded_candidates"]
        ),
        physical_config=physical_config,
    )
    _validate_exact_prospective_draws(
        draws,
        fits_by_model=candidate_fits_by_model,
        source_models=source_models,
        acquisition_evidence_digest=str(item["acquisition_evidence_digest"]),
        block=block,
        config=config,
        physical_config=physical_config,
    )
    if item["action_uncertainties"] != actions:
        raise ValueError("uncertainty summaries do not recompute from raw draws")
    audit = _stream_audit_from_payload(
        uses,
        draws=draws,
        source_models=source_models,
        draw_count=draw_count,
        acquisition_evidence_digest=str(item["acquisition_evidence_digest"]),
        block=block,
    )
    if item["stream_audit"] != audit:
        raise ValueError("uncertainty stream audit does not recompute")
    result_material = {
        "domain": "thermotwin.prospective_uncertainty_result",
        "config": item["config"],
        "physical_config": item["physical_config"],
        "acquisition_evidence_digest": item["acquisition_evidence_digest"],
        "protocol_digest": item["protocol_digest"],
        "action_uncertainties": actions,
        "draw_outcomes": draws,
        "stream_uses": uses,
        "stream_audit": audit,
    }
    result_digest = _uncertainty_canonical_digest(result_material)
    if item["result_digest"] != result_digest:
        raise ValueError("complete uncertainty result digest is invalid")
    return dict(item)


def _derive_uncertainty_prefix_payload(
    complete: Mapping[str, object],
    *,
    physical_config: OperatingDecisionRealismConfig,
    block: int,
    draw_count: int,
    max_unstable_draws: int,
) -> dict:
    derived = dict(complete)
    derived["draw_outcomes"] = [
        draw
        for draw in complete["draw_outcomes"]  # type: ignore[union-attr]
        if int(draw["draw_index"]) < draw_count
    ]
    derived["stream_uses"] = [
        use
        for use in complete["stream_uses"]  # type: ignore[union-attr]
        if int(use["key"]["draw_index"]) < draw_count
    ]
    config = ProspectiveUncertaintyConfig(
        draw_count=draw_count,
        max_unstable_draws_per_source_action=max_unstable_draws,
    )
    derived["config"] = _uncertainty_config_payload_for(config)
    derived["protocol_digest"] = prospective_uncertainty_protocol_digest(
        physical_config,
        config,
    )
    evidence = derived["acquisition_evidence"]
    snapshot = evidence["snapshot"]  # type: ignore[index]
    source_models = tuple(snapshot["admissible_candidate_models"])
    envelope = snapshot["provisional_margin_envelope"]
    baseline = float(envelope["upper"]) - float(envelope["lower"])
    derived["action_uncertainties"] = _recomputed_action_uncertainties(
        derived["draw_outcomes"],
        source_models=source_models,
        baseline_width=baseline,
        config=config,
        initially_admissible=source_models,
        initially_excluded=tuple(
            item["model_name"] for item in snapshot["excluded_candidates"]
        ),
        physical_config=physical_config,
    )
    derived["stream_audit"] = _stream_audit_from_payload(
        derived["stream_uses"],
        draws=derived["draw_outcomes"],
        source_models=source_models,
        draw_count=draw_count,
        acquisition_evidence_digest=str(derived["acquisition_evidence_digest"]),
        block=block,
    )
    result_material = {
        "domain": "thermotwin.prospective_uncertainty_result",
        "config": derived["config"],
        "physical_config": derived["physical_config"],
        "acquisition_evidence_digest": derived["acquisition_evidence_digest"],
        "protocol_digest": derived["protocol_digest"],
        "action_uncertainties": derived["action_uncertainties"],
        "draw_outcomes": derived["draw_outcomes"],
        "stream_uses": derived["stream_uses"],
        "stream_audit": derived["stream_audit"],
    }
    derived["result_digest"] = _uncertainty_canonical_digest(result_material)
    return _validate_complete_uncertainty_payload(
        derived,
        physical_config=physical_config,
        block=block,
        draw_count=draw_count,
        max_unstable_draws=max_unstable_draws,
    )


def _uncertainty_summary_from_complete(complete: Mapping[str, object]) -> dict:
    return {
        "schema_version": complete["schema_version"],
        "protocol_version": complete["protocol_version"],
        "protocol_digest": complete["protocol_digest"],
        "result_digest": complete["result_digest"],
        "physical_protocol_digest": complete["physical_protocol_digest"],
        "acquisition_evidence_digest": complete["acquisition_evidence_digest"],
        "config": complete["config"],
        "action_uncertainties": complete["action_uncertainties"],
        "stream_audit": complete["stream_audit"],
        "retained_draw_outcome_count": len(complete["draw_outcomes"]),  # type: ignore[arg-type]
        "retained_stream_use_count": len(complete["stream_uses"]),  # type: ignore[arg-type]
    }


def _snapshot_from_complete(complete: Mapping[str, object]) -> ProspectiveAcquisitionSnapshot:
    snapshot = complete["acquisition_evidence"]["snapshot"]  # type: ignore[index]
    envelope = snapshot["provisional_margin_envelope"]
    return ProspectiveAcquisitionSnapshot(
        candidate_models=tuple(snapshot["candidate_models"]),
        admissible_candidate_models=tuple(snapshot["admissible_candidate_models"]),
        excluded_candidates=tuple(
            CandidateExclusion(item["model_name"], item["reason"])
            for item in snapshot["excluded_candidates"]
        ),
        failed_candidate_models=tuple(snapshot["failed_candidate_models"]),
        model_intervals=tuple(
            MarginInterval(**item) for item in snapshot["model_intervals"]
        ),
        provisional_margin_envelope=(
            None if envelope is None else MarginEnvelope(**envelope)
        ),
        selection_failure_reason=snapshot["selection_failure_reason"],
    )


def _draw_diagnostics_from_payload(draws: Sequence[Mapping[str, object]]) -> dict:
    failed_count = 0
    transition_count = 0
    loss_count = 0
    recovery_count = 0
    statuses: Dict[str, int] = {}
    by_action: Dict[str, dict] = {}
    for draw in draws:
        policy = str(draw["policy_name"])
        action = by_action.setdefault(
            policy,
            {
                "retained_draw_count": 0,
                "failed_draw_count": 0,
                "candidate_transition_draw_count": 0,
                "candidate_loss_draw_count": 0,
                "candidate_recovery_draw_count": 0,
            },
        )
        failed = draw["failed"] is True
        lost = tuple(draw["initially_admissible_became_inadmissible"])
        recovered = tuple(draw["initially_excluded_became_admissible"])
        transitioned = bool(lost or recovered)
        action["retained_draw_count"] += 1
        action["failed_draw_count"] += failed
        action["candidate_transition_draw_count"] += transitioned
        action["candidate_loss_draw_count"] += bool(lost)
        action["candidate_recovery_draw_count"] += bool(recovered)
        failed_count += failed
        transition_count += transitioned
        loss_count += bool(lost)
        recovery_count += bool(recovered)
        outcomes = {
            outcome["model_name"]: outcome["status"]
            for outcome in draw["candidate_outcomes"]
        }
        for model_name in lost:
            status = outcomes.get(model_name, "missing_candidate_outcome")
            statuses[status] = statuses.get(status, 0) + 1
    return {
        "available": True,
        "retained_draw_count": len(draws),
        "failed_draw_count": failed_count,
        "candidate_transition_draw_count": transition_count,
        "candidate_loss_draw_count": loss_count,
        "candidate_recovery_draw_count": recovery_count,
        "candidate_loss_status_counts": dict(sorted(statuses.items())),
        "by_action": {name: by_action[name] for name in sorted(by_action)},
    }


def _validate_costed_scorecard_payload(
    payload: object,
    *,
    complete: Mapping[str, object],
    physical_config: OperatingDecisionRealismConfig,
    selector_rule: ProspectiveSelectorRule,
    cost_scenario: ProspectiveCostScenario,
) -> Tuple[object, ...]:
    keys = {
        "schema_version",
        "protocol_version",
        "formula",
        "padded_scoring_protocol",
        "energy_convention",
        "time_convention",
        "instrumentation_convention",
        "nominal_energy_protocol",
        "physical_protocol_digest",
        "uncertainty_protocol_digest",
        "uncertainty_result_digest",
        "action_catalog_digest",
        "selector_protocol_digest",
        "selector_rule",
        "acquisition_evidence_digest",
        "protocol_digest",
        "result_digest",
        "scenario",
        "energy_reference_joules",
        "bench_time_reference_seconds",
        "action_resources",
        "action_costs",
        "action_evaluations",
    }
    item = _exact_mapping(payload, keys, "prospective costed scorecard")
    expected_resources = _build_action_resources(physical_config, cost_scenario)
    expected_costs, expected_energy_reference, expected_time_reference = (
        _build_action_costs(expected_resources, cost_scenario)
    )
    if (
        item["schema_version"] != PROSPECTIVE_COST_SCHEMA_VERSION
        or item["protocol_version"] != PROSPECTIVE_COST_PROTOCOL_VERSION
        or item["formula"] != PROSPECTIVE_COST_FORMULA
        or item["padded_scoring_protocol"]
        != PROSPECTIVE_PADDED_SCORING_PROTOCOL
        or item["energy_convention"] != PROSPECTIVE_COST_ENERGY_CONVENTION
        or item["time_convention"] != PROSPECTIVE_COST_TIME_CONVENTION
        or item["instrumentation_convention"]
        != PROSPECTIVE_COST_INSTRUMENTATION_CONVENTION
        or item["nominal_energy_protocol"]
        != NOMINAL_SELECTION_COST_PROTOCOL
        or item["physical_protocol_digest"]
        != complete["physical_protocol_digest"]
        or item["uncertainty_protocol_digest"] != complete["protocol_digest"]
        or item["uncertainty_result_digest"] != complete["result_digest"]
        or item["acquisition_evidence_digest"]
        != complete["acquisition_evidence_digest"]
        or item["scenario"] != _scenario_payload(cost_scenario)
        or item["energy_reference_joules"] != expected_energy_reference
        or item["bench_time_reference_seconds"] != expected_time_reference
        or item["action_resources"]
        != [_action_resources_payload(value) for value in expected_resources]
        or item["action_costs"]
        != [_action_cost_payload(value) for value in expected_costs]
        or item["action_catalog_digest"]
        != prospective_action_catalog_digest()
        or item["selector_protocol_digest"]
        != prospective_selector_protocol_digest(selector_rule)
    ):
        raise ValueError("costed scorecard does not match uncertainty evidence")
    loaded_rule = prospective_selector_rule_from_payload(
        item["selector_rule"]  # type: ignore[arg-type]
    )
    if loaded_rule != selector_rule:
        raise ValueError("costed scorecard selector rule is inconsistent")
    if any(
        selector_rule.development_offsets.for_policy(policy_name) != 0.0
        for policy_name in POLICY_NAMES
    ):
        raise ValueError("the disposable pilot requires zero development offsets")
    config_payload = complete["config"]
    config = ProspectiveUncertaintyConfig(
        draw_count=int(config_payload["draw_count"]),  # type: ignore[index]
        max_unstable_draws_per_source_action=int(
            config_payload["max_unstable_draws_per_source_action"]  # type: ignore[index]
        ),
    )
    expected_protocol = prospective_cost_protocol_digest(
        physical_config,
        config,
        cost_scenario,
        selector_rule,
    )
    if item["protocol_digest"] != expected_protocol:
        raise ValueError("costed scorecard protocol digest is invalid")
    material = {
        "domain": "thermotwin.prospective_costed_scorecard",
        "uncertainty_result_digest": complete["result_digest"],
        "acquisition_evidence_digest": complete["acquisition_evidence_digest"],
        "scenario": item["scenario"],
        "selector_rule": item["selector_rule"],
        "protocol_digest": item["protocol_digest"],
        "energy_reference_joules": item["energy_reference_joules"],
        "bench_time_reference_seconds": item["bench_time_reference_seconds"],
        "action_resources": item["action_resources"],
        "action_costs": item["action_costs"],
        "action_evaluations": item["action_evaluations"],
    }
    if item["result_digest"] != _uncertainty_canonical_digest(material):
        raise ValueError("costed scorecard result digest is invalid")
    evaluations = item["action_evaluations"]
    if not isinstance(evaluations, list):
        raise ValueError("costed scorecard evaluations must be a list")
    loaded_evaluations = tuple(
        _action_evaluation_from_payload(value) for value in evaluations
    )
    action_uncertainties = tuple(complete["action_uncertainties"])
    if tuple(value["policy_name"] for value in action_uncertainties) != POLICY_NAMES:
        raise ValueError("uncertainty actions are not in canonical order")
    expected_evaluations = []
    for uncertainty, cost in zip(action_uncertainties, expected_costs):
        policy_name = uncertainty["policy_name"]
        if policy_name == STOP_NOW:
            expected_evaluations.append(
                ProspectiveActionEvaluation(STOP_NOW, True)
            )
        elif uncertainty["eligible"] is not True:
            expected_evaluations.append(
                ProspectiveActionEvaluation(
                    policy_name=policy_name,
                    eligible=False,
                    failure_reason=uncertainty["failure_reason"],
                )
            )
        else:
            offset = selector_rule.development_offsets.for_policy(policy_name)
            raw_before = float(uncertainty["uncertainty_before"])
            raw_after = float(uncertainty["expected_uncertainty_after"])
            expected_evaluations.append(
                ProspectiveActionEvaluation(
                    policy_name=policy_name,
                    eligible=True,
                    uncertainty_before=raw_before + offset,
                    expected_uncertainty_after=raw_after + offset,
                    declared_cost=cost.declared_cost,
                    prospective_draw_count=int(
                        uncertainty["prospective_draw_count"]
                    ),
                    raw_uncertainty_before=raw_before,
                    raw_expected_uncertainty_after=raw_after,
                    development_offset=offset,
                )
            )
    if loaded_evaluations != tuple(expected_evaluations):
        raise ValueError(
            "costed scorecard evaluations do not match uncertainty and cost"
        )
    return loaded_evaluations


def _validate_authenticated_prefix(
    prefix: object,
    *,
    complete: Mapping[str, object],
    physical_config: OperatingDecisionRealismConfig,
    selector_rule: ProspectiveSelectorRule,
    cost_scenario: ProspectiveCostScenario,
) -> Mapping[str, object]:
    item = _exact_mapping(
        prefix,
        {
            "draw_count",
            "max_unstable_draws_per_source_action",
            "choice_token",
            "eligibility",
            "draw_diagnostics",
            "selection",
            "uncertainty_summary",
            "costed_scorecard",
        },
        "authenticated pilot prefix",
    )
    config = complete["config"]
    if (
        item["draw_count"] != config["draw_count"]  # type: ignore[index]
        or item["max_unstable_draws_per_source_action"]
        != config["max_unstable_draws_per_source_action"]  # type: ignore[index]
    ):
        raise ValueError("pilot prefix config does not match complete evidence")
    expected_summary = _uncertainty_summary_from_complete(complete)
    if item["uncertainty_summary"] != expected_summary:
        raise ValueError("pilot uncertainty summary is not authenticated")
    expected_diagnostics = _draw_diagnostics_from_payload(
        complete["draw_outcomes"]  # type: ignore[arg-type]
    )
    if item["draw_diagnostics"] != expected_diagnostics:
        raise ValueError("pilot draw diagnostics do not match raw draws")
    actions = complete["action_uncertainties"]
    expected_eligibility = {
        action["policy_name"]: {
            "eligible": action["eligible"],
            "failure_reason": action["failure_reason"],
        }
        for action in actions  # type: ignore[union-attr]
    }
    if item["eligibility"] != expected_eligibility:
        raise ValueError("pilot eligibility does not match raw uncertainty")
    evaluations = _validate_costed_scorecard_payload(
        item["costed_scorecard"],
        complete=complete,
        physical_config=physical_config,
        selector_rule=selector_rule,
        cost_scenario=cost_scenario,
    )
    selection = select_prospective_action(
        _snapshot_from_complete(complete),
        evaluations,
        selector_rule,
    )
    expected_selection = _selection_payload(selection)
    if item["selection"] != expected_selection:
        raise ValueError("pilot selection does not recompute from its scorecard")
    if item["choice_token"] != _choice_token(selection, None):
        raise ValueError("pilot choice token does not match its selection")
    return item


def _validate_fixed_policy_record(
    record: Mapping[str, object],
    *,
    policy_name: str,
    truth_condition: str,
    block: int,
    device_token: str,
    physical_config: OperatingDecisionRealismConfig,
) -> None:
    partition = CorrectedPartition(
        name=PROSPECTIVE_PILOT_PARTITION,
        block_count=PILOT_BLOCK_COUNT,
        campaign=PROSPECTIVE_CAMPAIGN,
    )
    expected_id = _strict_json_value(
        corrected_case_id(
            partition,
            truth_condition,
            block,
            policy_name,
            physical_config,
        )
    )
    if expected_id["device_token"] != device_token:
        raise ValueError("parent pilot device identity is not reproducible")
    case_payload = _exact_mapping(
        record["case"],
        {
            "acquisition_runs",
            "case_id",
            "final_instrumentation",
            "final_regime",
            "policy",
            "verification_run",
        },
        "parent pilot fixed-policy case",
    )
    policy_payload = case_payload["policy"]
    if (
        case_payload["case_id"] != expected_id
        or not isinstance(policy_payload, Mapping)
        or policy_payload.get("name") != policy_name
    ):
        raise ValueError("parent pilot fixed-policy case identity is inconsistent")
    saved = record.get("saved_before_reveal")
    post = record.get("post_reveal_score")
    if "failure" in record:
        if not (
            record["failure"] == "decision_not_saved"
            and saved is None
            and post is None
        ) and not (
            isinstance(record["failure"], Mapping)
            and saved is not None
            and post is None
        ):
            raise ValueError("parent pilot fixed-policy failure record is inconsistent")
        return
    saved = _exact_mapping(
        saved,
        {
            "case_id",
            "decision",
            "decision_computation_seconds",
            "decision_reason",
            "failures",
            "margin_envelope",
            "model_intervals",
            "verifications",
        },
        "parent pilot saved fixed-policy decision",
    )
    post = _exact_mapping(
        post,
        {"nominal_selection_energy", "realized_energy", "revealed", "scored"},
        "parent pilot post-reveal score",
    )
    revealed = post["revealed"]
    scored = post["scored"]
    energy = post["realized_energy"]
    if (
        saved["case_id"] != expected_id
        or not isinstance(revealed, Mapping)
        or revealed.get("case_id") != expected_id
        or revealed.get("truth_condition") != truth_condition
        or not isinstance(scored, Mapping)
        or scored.get("saved") != saved
        or scored.get("truth_condition") != truth_condition
        or not isinstance(energy, Mapping)
        or energy.get("case_id") != expected_id
        or energy.get("truth_condition") != truth_condition
        or energy.get("trial_index") != block
    ):
        raise ValueError("parent pilot fixed-policy cross-identity is inconsistent")


def _validate_parent_pilot_payload(
    payload: Mapping[str, object],
    *,
    source_revision: str,
    physical_config: OperatingDecisionRealismConfig,
    selector_rule: ProspectiveSelectorRule,
    cost_scenario: ProspectiveCostScenario,
    require_n32_trigger: bool = True,
) -> Tuple[Mapping[str, object], ...]:
    """Authenticate complete pilot evidence and optionally its N=32 trigger."""

    required_top_level = {
        "schema_version",
        "protocol_version",
        "campaign",
        "partition",
        "source_revision",
        "protocol_digest",
        "selector_rule",
        "cost_scenario",
        "scientific_result_digest",
        "partition_plan",
        "worker_count",
        "case_count",
        "block_results",
        "acceptance",
        "compute_budget_inputs",
        "performance",
        "scientific_use",
        "archive_benchmark",
        "archive_content_digest",
        "archive_size_bytes",
    }
    _exact_mapping(payload, required_top_level, "complete parent pilot archive")

    expected_header = {
        "schema_version": PROSPECTIVE_PILOT_SCHEMA_VERSION,
        "protocol_version": PROSPECTIVE_PILOT_PROTOCOL_VERSION,
        "campaign": PROSPECTIVE_CAMPAIGN,
        "partition": PROSPECTIVE_PILOT_PARTITION,
        "source_revision": source_revision,
        "case_count": PILOT_BLOCK_COUNT * len(STAGE3_TRUTH_CONDITIONS),
    }
    for name, expected in expected_header.items():
        if payload.get(name) != expected:
            raise ValueError(f"parent pilot {name} does not match the frozen protocol")
    if (
        payload.get("selector_rule")
        != prospective_selector_rule_payload(selector_rule)
        or payload.get("cost_scenario") != _scenario_payload(cost_scenario)
    ):
        raise ValueError("parent pilot selector/cost configuration is invalid")
    archive_digest = payload.get("archive_content_digest")
    archive_size = payload.get("archive_size_bytes")
    _validate_sha256(
        "parent pilot archive content digest",
        archive_digest,  # type: ignore[arg-type]
    )
    if not isinstance(archive_size, int) or isinstance(archive_size, bool):
        raise ValueError("parent pilot archive size is missing")
    archive_material = dict(payload)
    archive_material.pop("archive_content_digest", None)
    archive_material.pop("archive_size_bytes", None)
    expected_archive_digest = _digest(
        f"{_HASH_DOMAIN}.archive_content",
        archive_material,
    )
    if archive_digest != expected_archive_digest:
        raise ValueError("parent pilot archive content digest is invalid")
    if archive_size != len(_canonical_bytes(payload) + b"\n"):
        raise ValueError("parent pilot archive size does not match canonical JSON")
    expected_protocol = prospective_pilot_protocol_digest(
        physical_config,
        selector_rule,
        cost_scenario,
        source_revision,
    )
    if payload.get("protocol_digest") != expected_protocol:
        raise ValueError("parent pilot protocol digest does not match source/config")
    blocks = payload.get("block_results")
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
        raise ValueError("parent pilot block results are missing")
    cases = _validate_followup_case_matrix(blocks)  # type: ignore[arg-type]
    if payload.get("partition_plan") != prospective_partition_plan_payload():
        raise ValueError("parent pilot partition plan is invalid")
    if not isinstance(payload.get("worker_count"), int) or (
        payload.get("scientific_use")
        != "disposable_engineering_evidence_only"
    ):
        raise ValueError("parent pilot runtime/archive records are incomplete")
    performance = _exact_mapping(
        payload.get("performance"),
        {"wall_seconds", "cpu_seconds", "peak_rss_bytes"},
        "parent pilot performance record",
    )
    if any(
        isinstance(performance[name], bool)
        or not isinstance(performance[name], (int, float))
        or not math.isfinite(float(performance[name]))
        or float(performance[name]) < 0.0
        for name in performance
    ):
        raise ValueError("parent pilot performance record is invalid")
    budget = _exact_mapping(
        payload.get("compute_budget_inputs"),
        {
            "pilot_block_wall_seconds",
            "pilot_block_cpu_seconds",
            "mean_block_wall_seconds",
            "p90_block_wall_seconds_nearest_rank",
            "mean_block_cpu_seconds",
            "measured_block_throughput_per_second",
            "measured_worker_count",
            "maximum_recorded_worker_peak_rss_bytes",
            "conservative_concurrent_peak_rss_bytes",
            "measured_pilot_throughput_scope",
            "predictive_n16_wall_seconds",
            "predictive_n16_cpu_seconds",
            "prefix_study_overhead_wall_seconds",
            "prefix_study_overhead_cpu_seconds",
            "retained_final_scoring_wall_seconds",
            "retained_final_scoring_cpu_seconds",
            "draw_count_linear_projections",
            "one_source_case_count",
            "two_source_case_count",
            "zero_source_case_count",
            "unknown_source_case_count",
            "fixed_policy_wall_seconds_including_verification_and_reveal",
            "fixed_policy_comparators_included",
            "verification_included",
            "runtime_host_manifest",
            "conservative_measured_throughput_phase_estimates",
            "assumptions",
        },
        "parent pilot compute-budget record",
    )
    if (
        budget["measured_worker_count"] != payload["worker_count"]
        or len(budget["pilot_block_wall_seconds"]) != PILOT_BLOCK_COUNT
        or len(budget["pilot_block_cpu_seconds"]) != PILOT_BLOCK_COUNT
        or budget["fixed_policy_comparators_included"] != list(POLICY_NAMES)
        or set(budget["fixed_policy_wall_seconds_including_verification_and_reveal"])
        != set(POLICY_NAMES)
        or budget["verification_included"] is not True
        or [item["draw_count"] for item in budget["draw_count_linear_projections"]]
        != list(PILOT_DRAW_COUNTS)
        or [item["partition"] for item in budget[
            "conservative_measured_throughput_phase_estimates"
        ]]
        != [item.name for item in PROSPECTIVE_PARTITION_PLAN[1:]]
    ):
        raise ValueError("parent pilot compute-budget record is invalid")
    _exact_mapping(
        budget["runtime_host_manifest"],
        {
            "python_version",
            "python_implementation",
            "operating_system",
            "operating_system_release",
            "machine",
            "processor",
            "logical_cpu_count",
            "host_node_sha256",
        },
        "parent pilot runtime host manifest",
    )
    archive_benchmark = _exact_mapping(
        payload.get("archive_benchmark"),
        {
            "benchmark_bytes",
            "serialization_wall_seconds",
            "serialization_cpu_seconds",
            "write_wall_seconds",
            "write_cpu_seconds",
            "bytes_per_write_wall_second",
            "bytes_per_serialization_wall_second",
            "estimated_bytes_per_paired_block",
            "full_campaign_partition_estimates",
            "full_campaign_draw_count_estimates",
            "note",
        },
        "parent pilot archive benchmark",
    )
    if (
        [item["partition"] for item in archive_benchmark[
            "full_campaign_partition_estimates"
        ]]
        != [item.name for item in PROSPECTIVE_PARTITION_PLAN[1:]]
        or [item["draw_count"] for item in archive_benchmark[
            "full_campaign_draw_count_estimates"
        ]]
        != list(PILOT_DRAW_COUNTS)
    ):
        raise ValueError("parent pilot archive benchmark is invalid")
    for block_record in blocks:
        _exact_mapping(
            block_record,
            {
                "block",
                "cases",
                "corrected_random_stream_manifest",
                "corrected_random_stream_audit",
                "timing",
            },
            "parent pilot block",
        )
        block_index = int(block_record["block"])
        manifest = block_record["corrected_random_stream_manifest"]
        audit = block_record["corrected_random_stream_audit"]
        if (
            isinstance(audit, Mapping)
            and set(audit) == {"clean", "not_run_due_to_block_failure"}
        ):
            _validate_canonical_failed_pilot_block(block_record, block_index)
            continue
        _validate_corrected_stream_records(
            manifest,
            audit,
            block=block_index,
            label="parent pilot",
        )
        for case in block_record["cases"]:
            if case.get("block") != block_index:
                raise ValueError("parent pilot case is nested under the wrong block")
            _exact_mapping(
                case,
                {
                    "block",
                    "truth_condition",
                    "device_token",
                    "truth_revealed_only_after_saves",
                    "selection_information_boundary",
                    "admissible_source_models",
                    "admissible_source_model_count",
                    "n16_complete_uncertainty_result",
                    "strict_prefixes",
                    "n16_max_unstable_1_sensitivity",
                    "selected_policy_at_n16_strict",
                    "selected_policy_outcome",
                    "fixed_policy_results",
                    "pipeline_failures",
                    "timing",
                },
                "parent pilot case",
            )
            if (
                not isinstance(case.get("device_token"), str)
                or not case.get("device_token")
                or not isinstance(case.get("pipeline_failures"), list)
                or case.get("selection_information_boundary")
                != _SELECTION_INFORMATION_BOUNDARY
                or case.get("truth_revealed_only_after_saves") is None
            ):
                raise ValueError("parent pilot case evidence is incomplete")
            fixed = case.get("fixed_policy_results")
            if not isinstance(fixed, Mapping) or set(fixed) != set(POLICY_NAMES):
                raise ValueError("parent pilot fixed-policy evidence is incomplete")
            for policy_name, fixed_record in fixed.items():
                if not isinstance(fixed_record, Mapping) or set(fixed_record) not in (
                    {"case", "saved_before_reveal", "post_reveal_score"},
                    {"case", "saved_before_reveal", "post_reveal_score", "failure"},
                ):
                    raise ValueError("parent pilot fixed-policy result is malformed")
                if fixed_record.get("case") is None:
                    raise ValueError("parent pilot fixed-policy case is incomplete")
                if "failure" not in fixed_record and any(
                    fixed_record[name] is None
                    for name in ("saved_before_reveal", "post_reveal_score")
                ):
                    raise ValueError("parent pilot fixed-policy result is incomplete")
                _validate_fixed_policy_record(
                    fixed_record,
                    policy_name=policy_name,
                    truth_condition=case["truth_condition"],
                    block=block_index,
                    device_token=case["device_token"],
                    physical_config=physical_config,
                )
            failures = tuple(
                _validate_pipeline_failure(
                    failure,
                    "parent pilot case pipeline failure",
                )
                for failure in case["pipeline_failures"]
            )
            for policy_name, fixed_record in fixed.items():
                fixed_failure = fixed_record.get("failure")
                if fixed_failure == "decision_not_saved":
                    if not any(
                        failure["stage"] == f"fixed_policy_save:{policy_name}"
                        for failure in failures
                    ):
                        raise ValueError(
                            "failed fixed-policy save has no pipeline failure"
                        )
                elif isinstance(fixed_failure, Mapping):
                    if not any(
                        failure["stage"] == f"fixed_policy_reveal:{policy_name}"
                        and failure["error_type"]
                        == fixed_failure.get("error_type")
                        and failure["message"] == fixed_failure.get("message")
                        for failure in failures
                    ):
                        raise ValueError(
                            "failed fixed-policy reveal has no pipeline failure"
                        )
            if case.get("n16_complete_uncertainty_result") is None:
                n16_failures = tuple(
                    failure
                    for failure in failures
                    if failure["stage"] == "n16_acquisition_or_scoring"
                )
                if len(n16_failures) != 1:
                    raise ValueError("parent pilot missing N=16 evidence is not canonical")
                n16_failure = n16_failures[0]
                for prefix, draw_count in zip(
                    case["strict_prefixes"], PILOT_DRAW_COUNTS
                ):
                    validated = _validate_failed_prefix_record(
                        prefix,
                        draw_count=draw_count,
                        max_unstable_draws=PILOT_STRICT_MAX_UNSTABLE,
                    )
                    if validated["pipeline_failure"] != n16_failure:
                        raise ValueError(
                            "failed pilot prefix does not match its N=16 failure"
                        )
                validated_sensitivity = _validate_failed_prefix_record(
                    case["n16_max_unstable_1_sensitivity"],
                    draw_count=PILOT_GENERATED_DRAW_COUNT,
                    max_unstable_draws=PILOT_SENSITIVITY_MAX_UNSTABLE,
                )
                if validated_sensitivity["pipeline_failure"] != n16_failure:
                    raise ValueError(
                        "failed pilot sensitivity does not match its N=16 failure"
                    )
                if (
                    case["admissible_source_models"] is not None
                    or case["admissible_source_model_count"] is not None
                    or case["selected_policy_at_n16_strict"] is not None
                    or case["selected_policy_outcome"] is not None
                ):
                    raise ValueError("parent pilot failed N=16 case is inconsistent")
                continue
            complete = _validate_complete_uncertainty_payload(
                case.get("n16_complete_uncertainty_result"),
                physical_config=physical_config,
                block=block_index,
                draw_count=PILOT_GENERATED_DRAW_COUNT,
                max_unstable_draws=PILOT_STRICT_MAX_UNSTABLE,
            )
            source_models = list(
                complete["acquisition_evidence"]["snapshot"][
                    "admissible_candidate_models"
                ]
            )
            if (
                case.get("admissible_source_models") != source_models
                or case.get("admissible_source_model_count")
                != len(source_models)
            ):
                raise ValueError("parent pilot source-model record is inconsistent")
            prefixes = case.get("strict_prefixes")
            if not isinstance(prefixes, list) or [
                prefix.get("draw_count") for prefix in prefixes
            ] != list(PILOT_DRAW_COUNTS):
                raise ValueError("parent pilot strict prefixes are incomplete")
            for prefix, draw_count in zip(prefixes, PILOT_DRAW_COUNTS):
                expected_complete = _derive_uncertainty_prefix_payload(
                    complete,
                    physical_config=physical_config,
                    block=block_index,
                    draw_count=draw_count,
                    max_unstable_draws=PILOT_STRICT_MAX_UNSTABLE,
                )
                if isinstance(prefix, Mapping) and "pipeline_failure" in prefix:
                    failed_prefix = _validate_failed_prefix_record(
                        prefix,
                        draw_count=draw_count,
                        max_unstable_draws=PILOT_STRICT_MAX_UNSTABLE,
                    )
                    prefix_failure = failed_prefix["pipeline_failure"]
                    if not any(
                        failure["stage"]
                        == f"{prefix_failure['stage']}_n{draw_count}"
                        and failure["error_type"] == prefix_failure["error_type"]
                        and failure["message"] == prefix_failure["message"]
                        for failure in failures
                    ):
                        raise ValueError("failed pilot prefix has no pipeline failure")
                else:
                    _validate_authenticated_prefix(
                        prefix,
                        complete=expected_complete,
                        physical_config=physical_config,
                        selector_rule=selector_rule,
                        cost_scenario=cost_scenario,
                    )
            relaxed_complete = _derive_uncertainty_prefix_payload(
                complete,
                physical_config=physical_config,
                block=block_index,
                draw_count=PILOT_GENERATED_DRAW_COUNT,
                max_unstable_draws=PILOT_SENSITIVITY_MAX_UNSTABLE,
            )
            sensitivity = case.get("n16_max_unstable_1_sensitivity")
            if isinstance(sensitivity, Mapping) and "pipeline_failure" in sensitivity:
                failed_sensitivity = _validate_failed_prefix_record(
                    sensitivity,
                    draw_count=PILOT_GENERATED_DRAW_COUNT,
                    max_unstable_draws=PILOT_SENSITIVITY_MAX_UNSTABLE,
                )
                sensitivity_failure = failed_sensitivity["pipeline_failure"]
                if not any(
                    failure["stage"]
                    == (
                        f"{sensitivity_failure['stage']}_n16_"
                        "max_unstable_1"
                    )
                    and failure["error_type"]
                    == sensitivity_failure["error_type"]
                    and failure["message"] == sensitivity_failure["message"]
                    for failure in failures
                ):
                    raise ValueError(
                        "failed pilot sensitivity has no pipeline failure"
                    )
            else:
                _validate_authenticated_prefix(
                    sensitivity,
                    complete=relaxed_complete,
                    physical_config=physical_config,
                    selector_rule=selector_rule,
                    cost_scenario=cost_scenario,
                )
            reference = _prefix_by_count(case, PILOT_GENERATED_DRAW_COUNT)
            selected_policy = (
                None
                if "pipeline_failure" in reference
                else reference["selection"]["selected_policy"]
            )
            if (
                case.get("selected_policy_at_n16_strict") != selected_policy
                or case.get("selected_policy_outcome")
                != (None if selected_policy is None else fixed.get(selected_policy))
            ):
                raise ValueError("parent pilot selected-policy outcome is inconsistent")
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, Mapping):
        raise ValueError("parent pilot acceptance record is missing")
    recomputed_acceptance = evaluate_pilot_acceptance(blocks)  # type: ignore[arg-type]
    if acceptance != recomputed_acceptance:
        raise ValueError("parent pilot acceptance record does not recompute exactly")
    if require_n32_trigger and (
        acceptance.get("feasibility_gate_passed") is not True
        or acceptance.get("stability_recommended_draw_count")
        != PILOT_GENERATED_DRAW_COUNT
        or acceptance.get("n32_followup_required") is not True
        or int(acceptance.get("pipeline_failure_count", -1)) != 0
        or int(acceptance.get("n16_selection_failure_count", -1)) != 0
    ):
        raise ValueError("parent pilot does not satisfy the frozen N=32 trigger")
    scientific = prospective_pilot_scientific_payload(
        source_revision=source_revision,
        protocol_digest=expected_protocol,
        block_results=blocks,  # type: ignore[arg-type]
        acceptance=acceptance,
    )
    expected_scientific = _digest(
        f"{_HASH_DOMAIN}.scientific_result",
        scientific,
    )
    if payload.get("scientific_result_digest") != expected_scientific:
        raise ValueError("parent pilot scientific result digest is invalid")
    for case in cases:
        if case.get("pipeline_failures"):
            continue
        prefix = _prefix_by_count(case, PILOT_GENERATED_DRAW_COUNT)
        if prefix.get("max_unstable_draws_per_source_action") != 0:
            raise ValueError("parent pilot N=16 prefix is not the strict prefix")
        summary = prefix.get("uncertainty_summary")
        if not isinstance(summary, Mapping):
            raise ValueError("parent pilot N=16 uncertainty summary is missing")
        config = summary.get("config")
        if not isinstance(config, Mapping) or (
            config.get("draw_count") != PILOT_GENERATED_DRAW_COUNT
            or config.get("max_unstable_draws_per_source_action") != 0
        ):
            raise ValueError("parent pilot N=16 uncertainty config is invalid")
    return cases


def validate_prospective_pilot_archive(
    payload: Mapping[str, object],
) -> dict:
    """Replay a complete saved pilot archive without requiring the N=32 trigger."""

    if not isinstance(payload, Mapping):
        raise ValueError("prospective pilot archive must be a mapping")
    source_revision = payload.get("source_revision")
    if not isinstance(source_revision, str):
        raise ValueError("prospective pilot archive source revision is missing")
    partition = CorrectedPartition(
        name=PROSPECTIVE_PILOT_PARTITION,
        block_count=PILOT_BLOCK_COUNT,
        campaign=PROSPECTIVE_CAMPAIGN,
    )
    physical_config = corrected_partition_config(
        partition,
        CORRECTED_REPLICATION_CONFIG,
    )
    selector_rule, cost_scenario = _parent_selector_and_cost(payload)
    _validate_parent_pilot_payload(
        payload,
        source_revision=source_revision,
        physical_config=physical_config,
        selector_rule=selector_rule,
        cost_scenario=cost_scenario,
        require_n32_trigger=False,
    )
    return dict(payload)


def prospective_n32_followup_protocol_digest(
    parent_payload: Mapping[str, object],
    physical_config: OperatingDecisionRealismConfig,
    selector_rule: ProspectiveSelectorRule,
    cost_scenario: ProspectiveCostScenario,
    source_revision: str,
) -> str:
    """Bind the continuation to source, exact parent evidence, and frozen design."""

    _validate_parent_pilot_payload(
        parent_payload,
        source_revision=source_revision,
        physical_config=physical_config,
        selector_rule=selector_rule,
        cost_scenario=cost_scenario,
    )
    material = {
        "schema_version": PROSPECTIVE_N32_FOLLOWUP_SCHEMA_VERSION,
        "protocol_version": PROSPECTIVE_N32_FOLLOWUP_PROTOCOL_VERSION,
        "source_revision": source_revision,
        "parent_payload_digest": _digest(
            f"{_HASH_DOMAIN}.n32.parent_payload",
            parent_payload,
        ),
        "parent_protocol_digest": parent_payload["protocol_digest"],
        "parent_scientific_result_digest": parent_payload[
            "scientific_result_digest"
        ],
        "design": prospective_n32_followup_design_payload(),
        "physical_protocol_digest": corrected_physical_protocol_digest(
            physical_config
        ),
        "uncertainty_protocol_digest": prospective_uncertainty_protocol_digest(
            physical_config,
            ProspectiveUncertaintyConfig(
                draw_count=PILOT_N32_FOLLOWUP_DRAW_COUNT,
                max_unstable_draws_per_source_action=PILOT_STRICT_MAX_UNSTABLE,
            ),
        ),
        "cost_protocol_digest": prospective_cost_protocol_digest(
            physical_config,
            ProspectiveUncertaintyConfig(
                draw_count=PILOT_N32_FOLLOWUP_DRAW_COUNT,
                max_unstable_draws_per_source_action=PILOT_STRICT_MAX_UNSTABLE,
            ),
            cost_scenario,
            selector_rule,
        ),
        "selector_rule": prospective_selector_rule_payload(selector_rule),
        "comparison": {
            "candidate": "authenticated_n16_prefix_from_n32_run",
            "reference": "strict_n32",
            "algorithm": "compare_pilot_choices_then_summarize_comparisons",
            "require_exact_match_to_parent_pilot_strict_n16": True,
            "require_zero_parent_pipeline_failures": True,
            "require_zero_parent_n16_selection_failures": True,
            "require_zero_n32_pipeline_failures": True,
            "require_zero_n32_selection_failures": True,
            "require_zero_n32_whole_draw_failures": True,
        },
    }
    return _digest(f"{_HASH_DOMAIN}.n32.protocol", material)


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _runtime_host_manifest() -> dict:
    """Describe the host used for performance projections without naming it."""

    host = platform.uname()
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "operating_system": host.system,
        "operating_system_release": host.release,
        "machine": host.machine,
        "processor": host.processor or platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "host_node_sha256": hashlib.sha256(
            host.node.encode("utf-8")
        ).hexdigest(),
    }


def _timed(callable_: Callable[[], object]) -> Tuple[object, dict]:
    wall_started = perf_counter()
    cpu_started = process_time()
    result = callable_()
    return result, {
        "wall_seconds": perf_counter() - wall_started,
        "cpu_seconds": process_time() - cpu_started,
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def _choice_token(selection: Optional[ProspectiveSelection], error: Optional[str]) -> str:
    if error is not None:
        return _PIPELINE_FAILURE_PREFIX + error
    if selection is None:
        return _PIPELINE_FAILURE_PREFIX + "missing_selection"
    if selection.selection_succeeded:
        return _ACTION_PREFIX + str(selection.selected_policy)
    return _SELECTION_FAILURE_PREFIX + selection.reason


def _action_evaluation_payload(evaluation) -> dict:
    return {
        "policy_name": evaluation.policy_name,
        "eligible": evaluation.eligible,
        "failure_reason": evaluation.failure_reason,
        "uncertainty_before": evaluation.uncertainty_before,
        "expected_uncertainty_after": evaluation.expected_uncertainty_after,
        "declared_cost": evaluation.declared_cost,
        "prospective_draw_count": evaluation.prospective_draw_count,
        "raw_uncertainty_before": evaluation.raw_uncertainty_before,
        "raw_expected_uncertainty_after": (
            evaluation.raw_expected_uncertainty_after
        ),
        "development_offset": evaluation.development_offset,
        "utility_per_cost": evaluation.utility_per_cost,
    }


def _selection_payload(selection: ProspectiveSelection) -> dict:
    return {
        "selected_policy": selection.selected_policy,
        "selection_succeeded": selection.selection_succeeded,
        "requires_additional_acquisition": (
            selection.requires_additional_acquisition
        ),
        "verification_required": selection.verification_required,
        "reason": selection.reason,
        "provisional_decision": selection.provisional_decision,
        "ranked_policies": list(selection.ranked_policies),
        "tied_policies": list(selection.tied_policies),
        "admissible_candidate_models": list(
            selection.admissible_candidate_models
        ),
        "candidate_reliability_stratum": (
            selection.candidate_reliability_stratum
        ),
        "padded_initial_margin_envelope": _strict_json_value(
            selection.padded_initial_margin_envelope
        ),
        "effective_stopping_clearance": selection.effective_stopping_clearance,
        "selector_protocol_digest": selection.selector_protocol_digest,
        "action_evaluations": [
            _action_evaluation_payload(item)
            for item in selection.action_evaluations
        ],
    }


def _prefix_record(
    result,
    *,
    selector_rule: ProspectiveSelectorRule,
    cost_scenario: ProspectiveCostScenario,
) -> dict:
    scorecard = cost_prospective_uncertainty(
        result,
        cost_scenario,
        selector_rule=selector_rule,
    )
    selection = select_costed_prospective_action(scorecard)
    uncertainty_payload = prospective_uncertainty_summary_payload(result)
    scorecard_payload = prospective_costed_scorecard_payload(scorecard)
    draw_diagnostics = _draw_diagnostics(result.draw_outcomes)
    return {
        "draw_count": result.config.draw_count,
        "max_unstable_draws_per_source_action": (
            result.config.max_unstable_draws_per_source_action
        ),
        "choice_token": _choice_token(selection, None),
        "eligibility": {
            item.policy_name: {
                "eligible": item.eligible,
                "failure_reason": item.failure_reason,
            }
            for item in result.action_uncertainties
        },
        "draw_diagnostics": draw_diagnostics,
        "selection": _selection_payload(selection),
        "uncertainty_summary": {
            "schema_version": uncertainty_payload["schema_version"],
            "protocol_version": uncertainty_payload["protocol_version"],
            "protocol_digest": uncertainty_payload["protocol_digest"],
            "result_digest": uncertainty_payload["result_digest"],
            "physical_protocol_digest": uncertainty_payload[
                "physical_protocol_digest"
            ],
            "acquisition_evidence_digest": uncertainty_payload[
                "acquisition_evidence_digest"
            ],
            "config": uncertainty_payload["config"],
            "action_uncertainties": uncertainty_payload[
                "action_uncertainties"
            ],
            "stream_audit": uncertainty_payload["stream_audit"],
            "retained_draw_outcome_count": uncertainty_payload[
                "retained_draw_outcome_count"
            ],
            "retained_stream_use_count": uncertainty_payload[
                "retained_stream_use_count"
            ],
        },
        "costed_scorecard": scorecard_payload,
    }


def _failed_prefix_record(
    draw_count: int,
    max_unstable_draws: int,
    error: BaseException,
    stage: str,
) -> dict:
    failure = f"{stage}:{type(error).__name__}"
    return {
        "draw_count": draw_count,
        "max_unstable_draws_per_source_action": max_unstable_draws,
        "choice_token": _choice_token(None, failure),
        "eligibility": {},
        "draw_diagnostics": {
            "available": False,
            "retained_draw_count": None,
            "failed_draw_count": None,
            "candidate_transition_draw_count": None,
            "candidate_loss_draw_count": None,
            "candidate_recovery_draw_count": None,
            "candidate_loss_status_counts": {},
            "by_action": {},
        },
        "selection": None,
        "uncertainty_summary": None,
        "costed_scorecard": None,
        "pipeline_failure": {
            "stage": stage,
            "error_type": type(error).__name__,
            "message": str(error),
        },
    }


def _validate_pipeline_failure(value: object, label: str) -> Mapping[str, object]:
    item = _exact_mapping(
        value,
        {"stage", "error_type", "message"},
        label,
    )
    if any(not isinstance(item[name], str) or not item[name] for name in item):
        raise ValueError(f"{label} is malformed")
    return item


def _validate_failed_prefix_record(
    prefix: object,
    *,
    draw_count: int,
    max_unstable_draws: int,
) -> Mapping[str, object]:
    item = _exact_mapping(
        prefix,
        {
            "draw_count",
            "max_unstable_draws_per_source_action",
            "choice_token",
            "eligibility",
            "draw_diagnostics",
            "selection",
            "uncertainty_summary",
            "costed_scorecard",
            "pipeline_failure",
        },
        "canonical failed N=32 prefix",
    )
    failure = _validate_pipeline_failure(
        item["pipeline_failure"],
        "canonical failed N=32 prefix failure",
    )
    expected_diagnostics = {
        "available": False,
        "retained_draw_count": None,
        "failed_draw_count": None,
        "candidate_transition_draw_count": None,
        "candidate_loss_draw_count": None,
        "candidate_recovery_draw_count": None,
        "candidate_loss_status_counts": {},
        "by_action": {},
    }
    if (
        item["draw_count"] != draw_count
        or item["max_unstable_draws_per_source_action"] != max_unstable_draws
        or item["choice_token"]
        != f"{_PIPELINE_FAILURE_PREFIX}{failure['stage']}:{failure['error_type']}"
        or item["eligibility"] != {}
        or item["draw_diagnostics"] != expected_diagnostics
        or item["selection"] is not None
        or item["uncertainty_summary"] is not None
        or item["costed_scorecard"] is not None
    ):
        raise ValueError("canonical failed N=32 prefix is inconsistent")
    return item


def _validate_canonical_failed_pilot_block(
    block_record: Mapping[str, object],
    block: int,
) -> None:
    if block_record["corrected_random_stream_manifest"] != []:
        raise ValueError("canonical failed pilot block retained random streams")
    audit = _exact_mapping(
        block_record["corrected_random_stream_audit"],
        {"clean", "not_run_due_to_block_failure"},
        "canonical failed pilot block audit",
    )
    failure = _validate_pipeline_failure(
        audit["not_run_due_to_block_failure"],
        "canonical failed pilot block failure",
    )
    if audit["clean"] is not False or failure["stage"] != "paired_block_generation":
        raise ValueError("canonical failed pilot block audit is inconsistent")
    for case in block_record["cases"]:
        _exact_mapping(
            case,
            {
                "block",
                "truth_condition",
                "device_token",
                "truth_revealed_only_after_saves",
                "selection_information_boundary",
                "admissible_source_models",
                "admissible_source_model_count",
                "n16_complete_uncertainty_result",
                "strict_prefixes",
                "n16_max_unstable_1_sensitivity",
                "selected_policy_at_n16_strict",
                "selected_policy_outcome",
                "fixed_policy_results",
                "pipeline_failures",
                "timing",
            },
            "canonical failed pilot case",
        )
        if (
            case["block"] != block
            or case["device_token"] is not None
            or case["truth_revealed_only_after_saves"] is not None
            or case["selection_information_boundary"] is not None
            or case["admissible_source_models"] is not None
            or case["admissible_source_model_count"] is not None
            or case["n16_complete_uncertainty_result"] is not None
            or case["selected_policy_at_n16_strict"] is not None
            or case["selected_policy_outcome"] is not None
            or case["fixed_policy_results"] != {}
            or case["pipeline_failures"] != [failure]
        ):
            raise ValueError("canonical failed pilot case is inconsistent")
        prefixes = case["strict_prefixes"]
        if not isinstance(prefixes, list) or len(prefixes) != len(PILOT_DRAW_COUNTS):
            raise ValueError("canonical failed pilot prefixes are incomplete")
        for prefix, draw_count in zip(prefixes, PILOT_DRAW_COUNTS):
            validated = _validate_failed_prefix_record(
                prefix,
                draw_count=draw_count,
                max_unstable_draws=PILOT_STRICT_MAX_UNSTABLE,
            )
            if validated["pipeline_failure"] != failure:
                raise ValueError(
                    "canonical failed pilot prefix does not match its block failure"
                )
        validated_sensitivity = _validate_failed_prefix_record(
            case["n16_max_unstable_1_sensitivity"],
            draw_count=PILOT_GENERATED_DRAW_COUNT,
            max_unstable_draws=PILOT_SENSITIVITY_MAX_UNSTABLE,
        )
        if validated_sensitivity["pipeline_failure"] != failure:
            raise ValueError(
                "canonical failed pilot sensitivity does not match its block failure"
            )


def _draw_diagnostics(draws: Sequence[object]) -> dict:
    """Separate whole-draw failures from completed candidate transitions."""

    retained = tuple(draws)
    failed_count = 0
    transition_count = 0
    loss_count = 0
    recovery_count = 0
    status_counts: Dict[str, int] = {}
    by_action: Dict[str, dict] = {}
    for draw in retained:
        policy_name = str(getattr(draw, "policy_name"))
        action = by_action.setdefault(
            policy_name,
            {
                "retained_draw_count": 0,
                "failed_draw_count": 0,
                "candidate_transition_draw_count": 0,
                "candidate_loss_draw_count": 0,
                "candidate_recovery_draw_count": 0,
            },
        )
        action["retained_draw_count"] += 1
        failed = bool(getattr(draw, "failed"))
        lost = tuple(
            getattr(draw, "initially_admissible_became_inadmissible")
        )
        recovered = tuple(
            getattr(draw, "initially_excluded_became_admissible")
        )
        transitioned = bool(lost or recovered)
        failed_count += failed
        transition_count += transitioned
        loss_count += bool(lost)
        recovery_count += bool(recovered)
        action["failed_draw_count"] += failed
        action["candidate_transition_draw_count"] += transitioned
        action["candidate_loss_draw_count"] += bool(lost)
        action["candidate_recovery_draw_count"] += bool(recovered)
        outcomes = {
            item.model_name: item.status
            for item in getattr(draw, "candidate_outcomes")
        }
        for model_name in lost:
            status = outcomes.get(model_name, "missing_candidate_outcome")
            status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "available": True,
        "retained_draw_count": len(retained),
        "failed_draw_count": failed_count,
        "candidate_transition_draw_count": transition_count,
        "candidate_loss_draw_count": loss_count,
        "candidate_recovery_draw_count": recovery_count,
        "candidate_loss_status_counts": dict(sorted(status_counts.items())),
        "by_action": {
            policy_name: by_action[policy_name]
            for policy_name in sorted(by_action)
        },
    }


def _case_payload(case) -> dict:
    return _strict_json_value(case)  # type: ignore[return-value]


def _saved_payload(saved) -> dict:
    return _strict_json_value(saved)  # type: ignore[return-value]


def _scored_record_payload(record) -> dict:
    return {
        "revealed": _strict_json_value(record.revealed),
        "scored": _strict_json_value(record.scored),
        "nominal_selection_energy": record.nominal_selection_energy,
        "realized_energy": _strict_json_value(record.realized_energy),
    }


def _stream_manifest_payload(uses: Sequence[object]) -> list:
    payload = []
    for use in uses:
        payload.append(
            {
                "key": _strict_json_value(use.stream.key),  # type: ignore[attr-defined]
                "seed": use.stream.seed,  # type: ignore[attr-defined]
                "consumer": use.consumer,  # type: ignore[attr-defined]
                "pairing_member": use.pairing_member,  # type: ignore[attr-defined]
                "pairing_id": use.pairing_id,  # type: ignore[attr-defined]
            }
        )
    return payload


def _run_pilot_family(
    *,
    truth_condition: str,
    block: int,
    truth,
    partition: CorrectedPartition,
    physical_config: OperatingDecisionRealismConfig,
    registry: RandomStreamRegistry,
    selector_rule: ProspectiveSelectorRule,
    cost_scenario: ProspectiveCostScenario,
) -> dict:
    case_wall_started = perf_counter()
    case_cpu_started = process_time()
    case_rss_started = _peak_rss_bytes()

    policies = default_fixed_policies()
    cases = {}
    generation_timings = {}
    for policy in policies:
        case, timing = _timed(
            lambda policy=policy: build_corrected_blinded_case(
                truth_condition,
                block,
                policy,
                truth,
                partition,
                physical_config,
                registry,
            )
        )
        cases[policy.name] = case
        generation_timings[policy.name] = timing

    common_case = cases[STOP_NOW]
    prefix_records: Dict[int, dict] = {}
    sensitivity_record = None
    pipeline_failures = []
    evidence_timing = None
    n16_timing = None
    prefix_timings = {}
    n16_complete_payload = None
    admissible_source_models = None

    try:
        evidence, evidence_timing = _timed(
            lambda: prepare_prospective_acquisition_evidence(
                common_case.acquisition_runs[0],
                common_case.final_regime,
                physical_config,
            )
        )
        admissible_source_models = tuple(
            evidence.snapshot.admissible_candidate_models
        )
        namespace = ProspectiveRandomStreamNamespace(
            campaign=partition.campaign,
            partition=partition.name,
            block=block,
            acquisition_evidence_digest=evidence.evidence_digest,
        )
        n16, n16_timing = _timed(
            lambda: estimate_prospective_action_uncertainty(
                evidence,
                namespace,
                physical_config,
                ProspectiveUncertaintyConfig(
                    draw_count=PILOT_GENERATED_DRAW_COUNT,
                    max_unstable_draws_per_source_action=(
                        PILOT_STRICT_MAX_UNSTABLE
                    ),
                ),
            )
        )
        n16_complete_payload = prospective_uncertainty_result_payload(n16)
        for draw_count in PILOT_DRAW_COUNTS:
            try:
                def derive_and_score(draw_count=draw_count):
                    prefix = prefix_prospective_uncertainty_result(
                        n16,
                        draw_count=draw_count,
                        max_unstable_draws_per_source_action=(
                            PILOT_STRICT_MAX_UNSTABLE
                        ),
                    )
                    return _prefix_record(
                        prefix,
                        selector_rule=selector_rule,
                        cost_scenario=cost_scenario,
                    )

                prefix_records[draw_count], prefix_timings[str(draw_count)] = _timed(
                    derive_and_score
                )
            except Exception as error:  # retained as a failed pilot case
                prefix_records[draw_count] = _failed_prefix_record(
                    draw_count,
                    PILOT_STRICT_MAX_UNSTABLE,
                    error,
                    "prefix_score",
                )
                pipeline_failures.append(
                    {
                        "stage": f"prefix_score_n{draw_count}",
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                )
        try:
            def derive_sensitivity():
                relaxed = prefix_prospective_uncertainty_result(
                    n16,
                    draw_count=PILOT_GENERATED_DRAW_COUNT,
                    max_unstable_draws_per_source_action=(
                        PILOT_SENSITIVITY_MAX_UNSTABLE
                    ),
                )
                return _prefix_record(
                    relaxed,
                    selector_rule=selector_rule,
                    cost_scenario=cost_scenario,
                )

            sensitivity_record, prefix_timings["16_max_unstable_1"] = _timed(
                derive_sensitivity
            )
        except Exception as error:  # retained as a failed sensitivity result
            sensitivity_record = _failed_prefix_record(
                PILOT_GENERATED_DRAW_COUNT,
                PILOT_SENSITIVITY_MAX_UNSTABLE,
                error,
                "sensitivity_score",
            )
            pipeline_failures.append(
                {
                    "stage": "sensitivity_score_n16_max_unstable_1",
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
    except Exception as error:  # retain all 12 cases if acquisition/scoring fails
        failure = {
            "stage": "n16_acquisition_or_scoring",
            "error_type": type(error).__name__,
            "message": str(error),
        }
        pipeline_failures.append(failure)
        for draw_count in PILOT_DRAW_COUNTS:
            prefix_records[draw_count] = _failed_prefix_record(
                draw_count,
                PILOT_STRICT_MAX_UNSTABLE,
                error,
                failure["stage"],
            )
        sensitivity_record = _failed_prefix_record(
            PILOT_GENERATED_DRAW_COUNT,
            PILOT_SENSITIVITY_MAX_UNSTABLE,
            error,
            failure["stage"],
        )

    # All four decisions are saved before any target response is revealed.
    saved = {}
    decision_timings = {}
    for policy in policies:
        try:
            decision, timing = _timed(
                lambda policy=policy: decide_corrected_blinded_case(
                    cases[policy.name],
                    config=physical_config,
                )
            )
            saved[policy.name] = decision
            decision_timings[policy.name] = timing
        except Exception as error:
            decision_timings[policy.name] = {
                "wall_seconds": 0.0,
                "cpu_seconds": 0.0,
                "peak_rss_bytes": _peak_rss_bytes(),
                "failed": True,
            }
            pipeline_failures.append(
                {
                    "stage": f"fixed_policy_save:{policy.name}",
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )

    fixed_results = {}
    reveal_timings = {}
    for policy in policies:
        if policy.name not in saved:
            reveal_timings[policy.name] = {
                "wall_seconds": 0.0,
                "cpu_seconds": 0.0,
                "peak_rss_bytes": _peak_rss_bytes(),
                "skipped": "decision_not_saved",
            }
            fixed_results[policy.name] = {
                "case": _case_payload(cases[policy.name]),
                "saved_before_reveal": None,
                "post_reveal_score": None,
                "failure": "decision_not_saved",
            }
            continue
        try:
            record, timing = _timed(
                lambda policy=policy: score_corrected_saved_decision(
                    saved[policy.name],
                    truth_condition=truth_condition,
                    block=block,
                    case=cases[policy.name],
                    truth=truth,
                    partition=partition,
                    config=physical_config,
                )
            )
            reveal_timings[policy.name] = timing
            fixed_results[policy.name] = {
                "case": _case_payload(cases[policy.name]),
                "saved_before_reveal": _saved_payload(saved[policy.name]),
                "post_reveal_score": _scored_record_payload(record),
            }
        except Exception as error:
            reveal_timings[policy.name] = {
                "wall_seconds": 0.0,
                "cpu_seconds": 0.0,
                "peak_rss_bytes": _peak_rss_bytes(),
                "failed": True,
            }
            pipeline_failures.append(
                {
                    "stage": f"fixed_policy_reveal:{policy.name}",
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
            fixed_results[policy.name] = {
                "case": _case_payload(cases[policy.name]),
                "saved_before_reveal": _saved_payload(saved[policy.name]),
                "post_reveal_score": None,
                "failure": {
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            }

    reference = prefix_records[PILOT_GENERATED_DRAW_COUNT]
    selected_policy = None
    if reference["selection"] is not None:
        selected_policy = reference["selection"]["selected_policy"]
    selected_outcome = (
        fixed_results.get(selected_policy) if selected_policy is not None else None
    )

    return {
        "block": block,
        "truth_condition": truth_condition,
        "device_token": common_case.case_id.device_token,
        "truth_revealed_only_after_saves": _strict_json_value(truth),
        "selection_information_boundary": _strict_json_value(
            _SELECTION_INFORMATION_BOUNDARY
        ),
        "admissible_source_models": (
            None
            if admissible_source_models is None
            else list(admissible_source_models)
        ),
        "admissible_source_model_count": (
            None
            if admissible_source_models is None
            else len(admissible_source_models)
        ),
        "n16_complete_uncertainty_result": n16_complete_payload,
        "strict_prefixes": [prefix_records[item] for item in PILOT_DRAW_COUNTS],
        "n16_max_unstable_1_sensitivity": sensitivity_record,
        "selected_policy_at_n16_strict": selected_policy,
        "selected_policy_outcome": selected_outcome,
        "fixed_policy_results": fixed_results,
        "pipeline_failures": pipeline_failures,
        "timing": {
            "case_generation_by_policy": generation_timings,
            "acquisition_evidence": evidence_timing,
            "n16_generation": n16_timing,
            "prefix_scoring": prefix_timings,
            "decision_and_verification_by_policy": decision_timings,
            "post_save_reveal_and_scoring_by_policy": reveal_timings,
            "case_wall_seconds": perf_counter() - case_wall_started,
            "case_cpu_seconds": process_time() - case_cpu_started,
            "peak_rss_before_bytes": case_rss_started,
            "peak_rss_after_bytes": _peak_rss_bytes(),
        },
    }


def _run_pilot_block(
    block: int,
    physical_config: OperatingDecisionRealismConfig,
    selector_rule: ProspectiveSelectorRule,
    cost_scenario: ProspectiveCostScenario,
) -> dict:
    block_wall_started = perf_counter()
    block_cpu_started = process_time()
    partition = CorrectedPartition(
        name=PROSPECTIVE_PILOT_PARTITION,
        block_count=PILOT_BLOCK_COUNT,
        campaign=PROSPECTIVE_CAMPAIGN,
    )
    registry = RandomStreamRegistry()
    truth, truth_timing = _timed(
        lambda: corrected_truth_for_block(
            partition,
            block,
            physical_config,
            registry,
        )
    )
    cases = tuple(
        _run_pilot_family(
            truth_condition=truth_condition,
            block=block,
            truth=truth,
            partition=partition,
            physical_config=physical_config,
            registry=registry,
            selector_rule=selector_rule,
            cost_scenario=cost_scenario,
        )
        for truth_condition in STAGE3_TRUTH_CONDITIONS
    )
    audit = registry.audit()
    audit.assert_clean()
    return {
        "block": block,
        "cases": list(cases),
        "corrected_random_stream_manifest": _stream_manifest_payload(registry.uses),
        "corrected_random_stream_audit": _strict_json_value(audit),
        "timing": {
            "truth_generation": truth_timing,
            "block_wall_seconds": perf_counter() - block_wall_started,
            "block_cpu_seconds": process_time() - block_cpu_started,
            "peak_rss_bytes": _peak_rss_bytes(),
        },
    }


def _run_pilot_block_worker(arguments) -> dict:
    return _run_pilot_block(*arguments)


def _failed_block_result(block: int, error: BaseException) -> dict:
    failure = {
        "stage": "paired_block_generation",
        "error_type": type(error).__name__,
        "message": str(error),
    }
    cases = []
    for truth_condition in STAGE3_TRUTH_CONDITIONS:
        prefixes = [
            _failed_prefix_record(
                draw_count,
                PILOT_STRICT_MAX_UNSTABLE,
                error,
                failure["stage"],
            )
            for draw_count in PILOT_DRAW_COUNTS
        ]
        cases.append(
            {
                "block": block,
                "truth_condition": truth_condition,
                "device_token": None,
                "truth_revealed_only_after_saves": None,
                "selection_information_boundary": None,
                "admissible_source_models": None,
                "admissible_source_model_count": None,
                "n16_complete_uncertainty_result": None,
                "strict_prefixes": prefixes,
                "n16_max_unstable_1_sensitivity": _failed_prefix_record(
                    PILOT_GENERATED_DRAW_COUNT,
                    PILOT_SENSITIVITY_MAX_UNSTABLE,
                    error,
                    failure["stage"],
                ),
                "selected_policy_at_n16_strict": None,
                "selected_policy_outcome": None,
                "fixed_policy_results": {},
                "pipeline_failures": [failure],
                "timing": {
                    "case_generation_by_policy": {},
                    "acquisition_evidence": None,
                    "n16_generation": None,
                    "prefix_scoring": {},
                    "decision_and_verification_by_policy": {},
                    "post_save_reveal_and_scoring_by_policy": {},
                    "case_wall_seconds": 0.0,
                    "case_cpu_seconds": 0.0,
                    "peak_rss_before_bytes": _peak_rss_bytes(),
                    "peak_rss_after_bytes": _peak_rss_bytes(),
                },
            }
        )
    return {
        "block": block,
        "cases": cases,
        "corrected_random_stream_manifest": [],
        "corrected_random_stream_audit": {
            "clean": False,
            "not_run_due_to_block_failure": failure,
        },
        "timing": {
            "truth_generation": None,
            "block_wall_seconds": 0.0,
            "block_cpu_seconds": 0.0,
            "peak_rss_bytes": _peak_rss_bytes(),
        },
    }


def _run_n32_followup_family(
    *,
    truth_condition: str,
    block: int,
    truth,
    partition: CorrectedPartition,
    physical_config: OperatingDecisionRealismConfig,
    registry: RandomStreamRegistry,
    selector_rule: ProspectiveSelectorRule,
    cost_scenario: ProspectiveCostScenario,
) -> dict:
    """Generate one conditional case at N=32 and its authenticated N=16 prefix."""

    wall_started = perf_counter()
    cpu_started = process_time()
    stop_policy = next(
        policy for policy in default_fixed_policies() if policy.name == STOP_NOW
    )
    pipeline_failures = []
    evidence_timing = None
    generation_timing = None
    n32_timing = None
    prefix_timings = {}
    complete_payload = None
    admissible_source_models = None
    case = None
    prefixes: Dict[int, dict] = {}
    try:
        case, generation_timing = _timed(
            lambda: build_corrected_blinded_case(
                truth_condition,
                block,
                stop_policy,
                truth,
                partition,
                physical_config,
                registry,
            )
        )
        evidence, evidence_timing = _timed(
            lambda: prepare_prospective_acquisition_evidence(
                case.acquisition_runs[0],
                case.final_regime,
                physical_config,
            )
        )
        admissible_source_models = tuple(
            evidence.snapshot.admissible_candidate_models
        )
        namespace = ProspectiveRandomStreamNamespace(
            campaign=partition.campaign,
            partition=partition.name,
            block=block,
            acquisition_evidence_digest=evidence.evidence_digest,
        )
        n32, n32_timing = _timed(
            lambda: estimate_prospective_action_uncertainty(
                evidence,
                namespace,
                physical_config,
                ProspectiveUncertaintyConfig(
                    draw_count=PILOT_N32_FOLLOWUP_DRAW_COUNT,
                    max_unstable_draws_per_source_action=PILOT_STRICT_MAX_UNSTABLE,
                ),
            )
        )
        complete_payload = prospective_uncertainty_result_payload(n32)
        for draw_count in (
            PILOT_GENERATED_DRAW_COUNT,
            PILOT_N32_FOLLOWUP_DRAW_COUNT,
        ):
            try:
                def derive_and_score(draw_count=draw_count):
                    prefix = prefix_prospective_uncertainty_result(
                        n32,
                        draw_count=draw_count,
                        max_unstable_draws_per_source_action=(
                            PILOT_STRICT_MAX_UNSTABLE
                        ),
                    )
                    return _prefix_record(
                        prefix,
                        selector_rule=selector_rule,
                        cost_scenario=cost_scenario,
                    )

                prefixes[draw_count], prefix_timings[str(draw_count)] = _timed(
                    derive_and_score
                )
            except Exception as error:
                prefixes[draw_count] = _failed_prefix_record(
                    draw_count,
                    PILOT_STRICT_MAX_UNSTABLE,
                    error,
                    "n32_prefix_score",
                )
                pipeline_failures.append(
                    {
                        "stage": f"n32_prefix_score_n{draw_count}",
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                )
    except Exception as error:
        failure = {
            "stage": "n32_acquisition_or_scoring",
            "error_type": type(error).__name__,
            "message": str(error),
        }
        pipeline_failures.append(failure)
        for draw_count in (
            PILOT_GENERATED_DRAW_COUNT,
            PILOT_N32_FOLLOWUP_DRAW_COUNT,
        ):
            prefixes[draw_count] = _failed_prefix_record(
                draw_count,
                PILOT_STRICT_MAX_UNSTABLE,
                error,
                failure["stage"],
            )
    return {
        "block": block,
        "truth_condition": truth_condition,
        "device_token": (
            case.case_id.device_token if case is not None else None
        ),
        "admissible_source_models": (
            list(admissible_source_models)
            if admissible_source_models is not None
            else None
        ),
        "admissible_source_model_count": (
            len(admissible_source_models)
            if admissible_source_models is not None
            else None
        ),
        "n32_complete_uncertainty_result": complete_payload,
        "strict_prefixes": [
            prefixes[PILOT_GENERATED_DRAW_COUNT],
            prefixes[PILOT_N32_FOLLOWUP_DRAW_COUNT],
        ],
        "pipeline_failures": pipeline_failures,
        "timing": {
            "case_generation": generation_timing,
            "acquisition_evidence": evidence_timing,
            "n32_generation": n32_timing,
            "prefix_scoring": prefix_timings,
            "case_wall_seconds": perf_counter() - wall_started,
            "case_cpu_seconds": process_time() - cpu_started,
        },
    }


def _run_n32_followup_block(
    block: int,
    physical_config: OperatingDecisionRealismConfig,
    selector_rule: ProspectiveSelectorRule,
    cost_scenario: ProspectiveCostScenario,
) -> dict:
    wall_started = perf_counter()
    cpu_started = process_time()
    partition = CorrectedPartition(
        name=PROSPECTIVE_PILOT_PARTITION,
        block_count=PILOT_BLOCK_COUNT,
        campaign=PROSPECTIVE_CAMPAIGN,
    )
    registry = RandomStreamRegistry()
    truth, truth_timing = _timed(
        lambda: corrected_truth_for_block(
            partition,
            block,
            physical_config,
            registry,
        )
    )
    cases = tuple(
        _run_n32_followup_family(
            truth_condition=truth_condition,
            block=block,
            truth=truth,
            partition=partition,
            physical_config=physical_config,
            registry=registry,
            selector_rule=selector_rule,
            cost_scenario=cost_scenario,
        )
        for truth_condition in STAGE3_TRUTH_CONDITIONS
    )
    audit = registry.audit()
    audit.assert_clean()
    return {
        "block": block,
        "cases": list(cases),
        "corrected_random_stream_manifest": _stream_manifest_payload(registry.uses),
        "corrected_random_stream_audit": _strict_json_value(audit),
        "timing": {
            "truth_generation": truth_timing,
            "block_wall_seconds": perf_counter() - wall_started,
            "block_cpu_seconds": process_time() - cpu_started,
            "peak_rss_bytes": _peak_rss_bytes(),
        },
    }


def _run_n32_followup_block_worker(arguments) -> dict:
    return _run_n32_followup_block(*arguments)


def _failed_n32_followup_block(block: int, error: BaseException) -> dict:
    failure = {
        "stage": "n32_paired_block_generation",
        "error_type": type(error).__name__,
        "message": str(error),
    }
    cases = []
    for truth_condition in STAGE3_TRUTH_CONDITIONS:
        cases.append(
            {
                "block": block,
                "truth_condition": truth_condition,
                "device_token": None,
                "admissible_source_models": None,
                "admissible_source_model_count": None,
                "n32_complete_uncertainty_result": None,
                "strict_prefixes": [
                    _failed_prefix_record(
                        draw_count,
                        PILOT_STRICT_MAX_UNSTABLE,
                        error,
                        failure["stage"],
                    )
                    for draw_count in (
                        PILOT_GENERATED_DRAW_COUNT,
                        PILOT_N32_FOLLOWUP_DRAW_COUNT,
                    )
                ],
                "pipeline_failures": [failure],
                "timing": {
                    "case_generation": None,
                    "acquisition_evidence": None,
                    "n32_generation": None,
                    "prefix_scoring": {},
                    "case_wall_seconds": 0.0,
                    "case_cpu_seconds": 0.0,
                },
            }
        )
    return {
        "block": block,
        "cases": cases,
        "corrected_random_stream_manifest": [],
        "corrected_random_stream_audit": {
            "clean": False,
            "not_run_due_to_block_failure": failure,
        },
        "timing": {
            "truth_generation": None,
            "block_wall_seconds": 0.0,
            "block_cpu_seconds": 0.0,
            "peak_rss_bytes": _peak_rss_bytes(),
        },
    }


def _prefix_by_count(case: Mapping[str, object], draw_count: int) -> Mapping[str, object]:
    selected = tuple(
        item
        for item in case["strict_prefixes"]  # type: ignore[index]
        if item["draw_count"] == draw_count
    )
    if len(selected) != 1:
        raise ValueError("pilot case has no unique draw-count prefix")
    return selected[0]


def _selection_utility(
    prefix: Mapping[str, object],
    policy_name: Optional[str],
) -> Optional[float]:
    if policy_name is None:
        return None
    if policy_name == STOP_NOW:
        return 0.0
    selection = prefix.get("selection")
    if not isinstance(selection, Mapping):
        return None
    evaluations = selection.get("action_evaluations")
    if not isinstance(evaluations, Sequence):
        return None
    matches = tuple(
        item
        for item in evaluations
        if isinstance(item, Mapping) and item.get("policy_name") == policy_name
    )
    if len(matches) != 1:
        return None
    value = matches[0].get("utility_per_cost")
    eligible = matches[0].get("eligible")
    if eligible is not True:
        return None
    return (
        float(value)
        if isinstance(value, (int, float)) and math.isfinite(value)
        else None
    )


def _selected_policy(prefix: Mapping[str, object]) -> Optional[str]:
    selection = prefix.get("selection")
    if not isinstance(selection, Mapping):
        return None
    selected = selection.get("selected_policy")
    return selected if isinstance(selected, str) else None


def _finite_reference_utilities(prefix: Mapping[str, object]) -> Tuple[float, ...]:
    selection = prefix.get("selection")
    if not isinstance(selection, Mapping):
        return ()
    evaluations = selection.get("action_evaluations")
    if not isinstance(evaluations, Sequence):
        return ()
    values = []
    for item in evaluations:
        if not isinstance(item, Mapping):
            continue
        value = item.get("utility_per_cost")
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return tuple(values)


def _action_is_eligible(
    eligibility: object,
    policy_name: Optional[str],
) -> Optional[bool]:
    if policy_name is None or policy_name == STOP_NOW:
        return None
    if not isinstance(eligibility, Mapping):
        return None
    item = eligibility.get(policy_name)
    if not isinstance(item, Mapping) or not isinstance(item.get("eligible"), bool):
        return None
    return bool(item["eligible"])


def compare_pilot_choices(
    candidate: Mapping[str, object],
    reference: Mapping[str, object],
) -> dict:
    """Compare a candidate prefix using only the reference action utilities."""

    candidate_token = str(candidate["choice_token"])
    reference_token = str(reference["choice_token"])
    agreement = candidate_token == reference_token
    candidate_policy = _selected_policy(candidate)
    reference_policy = _selected_policy(reference)
    values = _finite_reference_utilities(reference)
    scale = max((abs(value) for value in values), default=0.0)
    reference_utility = _selection_utility(reference, reference_policy)
    candidate_utility = _selection_utility(reference, candidate_policy)
    regret_evaluable = (
        agreement
        or (reference_utility is not None and candidate_utility is not None)
    )
    if agreement:
        regret = 0.0
    elif not regret_evaluable:
        regret = None
    elif scale == 0.0:
        # This is the protocol's one zero-regret exception: the relevant
        # reference utilities exist and all finite reference utilities are zero.
        regret = 0.0
    else:
        regret = max(0.0, reference_utility - candidate_utility) / scale
    candidate_eligibility = candidate.get("eligibility", {})
    reference_eligibility = reference.get("eligibility", {})
    eligibility_changed = candidate_eligibility != reference_eligibility
    selected_action_eligibility_changed = any(
        _action_is_eligible(candidate_eligibility, policy_name)
        != _action_is_eligible(reference_eligibility, policy_name)
        for policy_name in {candidate_policy, reference_policy}
        if policy_name not in (None, STOP_NOW)
    )
    return {
        "candidate_draw_count": candidate["draw_count"],
        "reference_draw_count": reference["draw_count"],
        "candidate_choice": candidate_token,
        "reference_choice": reference_token,
        "agreement": agreement,
        "normalized_utility_regret": regret,
        "regret_evaluable": regret_evaluable,
        "utility_normalization": scale,
        "eligibility_changed": eligibility_changed,
        "choice_change_with_selected_action_eligibility_difference": (
            not agreement and selected_action_eligibility_changed
        ),
    }


def _summarize_comparisons(
    comparisons: Sequence[Mapping[str, object]],
    rule: PilotAcceptanceRule,
) -> dict:
    comparisons = tuple(comparisons)
    agreements = sum(bool(item["agreement"]) for item in comparisons)
    regrets = tuple(
        float(item["normalized_utility_regret"])
        for item in comparisons
        if item["normalized_utility_regret"] is not None
    )
    unevaluable_changed = sum(
        not bool(item["agreement"]) and not bool(item["regret_evaluable"])
        for item in comparisons
    )
    changed_regrets = tuple(
        float(item["normalized_utility_regret"])
        for item in comparisons
        if not bool(item["agreement"])
        and item["normalized_utility_regret"] is not None
    )
    agreement_rate = agreements / len(comparisons) if comparisons else 0.0
    maximum_changed_regret = max(changed_regrets, default=0.0)
    return {
        "draw_count": (
            comparisons[0]["candidate_draw_count"] if comparisons else None
        ),
        "case_count": len(comparisons),
        "agreement_count": agreements,
        "agreement_rate": agreement_rate,
        "changed_choice_count": sum(
            not bool(item["agreement"]) for item in comparisons
        ),
        "unevaluable_changed_choice_count": unevaluable_changed,
        "selected_action_eligibility_difference_count": sum(
            bool(item["choice_change_with_selected_action_eligibility_difference"])
            for item in comparisons
        ),
        "mean_normalized_utility_regret": (
            statistics.fmean(regrets) if regrets else 0.0
        ),
        "maximum_changed_choice_normalized_utility_regret": (
            maximum_changed_regret
        ),
        "meets_agreement_rule": (
            agreement_rate >= rule.minimum_action_agreement
        ),
        "meets_regret_rule": (
            unevaluable_changed == 0
            and maximum_changed_regret
            <= rule.maximum_normalized_utility_regret
        ),
        "meets_acceptance_rule": (
            agreement_rate >= rule.minimum_action_agreement
            and unevaluable_changed == 0
            and maximum_changed_regret
            <= rule.maximum_normalized_utility_regret
        ),
    }


def _aggregate_reference_draw_diagnostics(
    prefixes: Sequence[Mapping[str, object]],
) -> dict:
    diagnostics = tuple(prefix.get("draw_diagnostics") for prefix in prefixes)
    if any(
        not isinstance(item, Mapping) or item.get("available") is not True
        for item in diagnostics
    ):
        return {
            "available": False,
            "retained_draw_count": None,
            "failed_draw_count": None,
            "candidate_transition_draw_count": None,
            "candidate_loss_draw_count": None,
            "candidate_recovery_draw_count": None,
            "candidate_loss_status_counts": {},
            "by_action": {},
        }

    count_names = (
        "retained_draw_count",
        "failed_draw_count",
        "candidate_transition_draw_count",
        "candidate_loss_draw_count",
        "candidate_recovery_draw_count",
    )
    result = {
        "available": True,
        **{
            name: sum(int(item[name]) for item in diagnostics)  # type: ignore[index]
            for name in count_names
        },
    }
    status_counts: Dict[str, int] = {}
    by_action: Dict[str, dict] = {}
    for item in diagnostics:
        assert isinstance(item, Mapping)
        statuses = item.get("candidate_loss_status_counts", {})
        if isinstance(statuses, Mapping):
            for status, count in statuses.items():
                status_counts[str(status)] = (
                    status_counts.get(str(status), 0) + int(count)
                )
        actions = item.get("by_action", {})
        if not isinstance(actions, Mapping):
            continue
        for policy_name, counts in actions.items():
            if not isinstance(counts, Mapping):
                continue
            aggregate = by_action.setdefault(
                str(policy_name),
                {name: 0 for name in count_names},
            )
            for name in count_names:
                aggregate[name] += int(counts.get(name, 0))
    result["candidate_loss_status_counts"] = dict(sorted(status_counts.items()))
    result["by_action"] = {
        policy_name: by_action[policy_name]
        for policy_name in sorted(by_action)
    }
    return result


def evaluate_pilot_acceptance(
    block_results: Sequence[Mapping[str, object]],
    rule: PilotAcceptanceRule = PilotAcceptanceRule(),
) -> dict:
    """Apply the declared agreement/regret rule to all twelve retained cases."""

    cases = tuple(
        case
        for block in block_results
        for case in block["cases"]  # type: ignore[index]
    )
    if len(cases) != PILOT_BLOCK_COUNT * len(STAGE3_TRUTH_CONDITIONS):
        raise ValueError("pilot acceptance requires all twelve cases")
    detailed = {}
    summaries = []
    for draw_count in PILOT_DRAW_COUNTS:
        comparisons = tuple(
            {
                "block": case["block"],
                "truth_condition": case["truth_condition"],
                **compare_pilot_choices(
                    _prefix_by_count(case, draw_count),
                    _prefix_by_count(case, PILOT_GENERATED_DRAW_COUNT),
                ),
            }
            for case in cases
        )
        detailed[str(draw_count)] = list(comparisons)
        summaries.append(_summarize_comparisons(comparisons, rule))

    sensitivity_comparisons = tuple(
        {
            "block": case["block"],
            "truth_condition": case["truth_condition"],
            **compare_pilot_choices(
                case["n16_max_unstable_1_sensitivity"],
                _prefix_by_count(case, PILOT_GENERATED_DRAW_COUNT),
            ),
        }
        for case in cases
    )
    sensitivity_summary = _summarize_comparisons(
        sensitivity_comparisons,
        rule,
    )
    sensitivity_summary["candidate_rule"] = "n16_max_unstable_1"
    sensitivity_summary["reference_rule"] = "n16_max_unstable_0"

    pipeline_failures = sum(
        len(case.get("pipeline_failures", ())) for case in cases
    )
    reference_prefixes = tuple(
        _prefix_by_count(case, PILOT_GENERATED_DRAW_COUNT) for case in cases
    )
    reference_draw_diagnostics = _aggregate_reference_draw_diagnostics(
        reference_prefixes
    )
    reference_selection_failures = sum(
        str(prefix.get("choice_token", "")).startswith(
            _SELECTION_FAILURE_PREFIX
        )
        for prefix in reference_prefixes
    )
    reference_ineligible_action_counts = []
    for prefix in reference_prefixes:
        eligibility = prefix.get("eligibility", {})
        reference_ineligible_action_counts.append(
            sum(
                isinstance(item, Mapping) and item.get("eligible") is False
                for item in (
                    eligibility.values()
                    if isinstance(eligibility, Mapping)
                    else ()
                )
            )
        )
    qualifying = tuple(
        item
        for item in summaries
        if bool(item["meets_acceptance_rule"])
    )
    stability_recommendation = (
        min(int(item["draw_count"]) for item in qualifying)
        if qualifying
        else None
    )
    feasibility_passed = (
        pipeline_failures == 0 and reference_selection_failures == 0
    )
    n32_followup_required = (
        feasibility_passed
        and stability_recommendation
        == rule.n32_followup_trigger_draw_count
    )
    pilot_engineering_gate_passed = (
        feasibility_passed
        and stability_recommendation is not None
        and not n32_followup_required
    )
    recommendation = (
        stability_recommendation if pilot_engineering_gate_passed else None
    )
    return {
        "rule": asdict(rule),
        "strict_comparisons": detailed,
        "strict_summaries": summaries,
        "n16_max_unstable_1_sensitivity": {
            "comparisons": list(sensitivity_comparisons),
            "summary": sensitivity_summary,
        },
        "pipeline_failure_count": pipeline_failures,
        "n16_selection_failure_count": reference_selection_failures,
        "n16_ineligible_action_count": sum(reference_ineligible_action_counts),
        "n16_cases_with_ineligible_actions": sum(
            count > 0 for count in reference_ineligible_action_counts
        ),
        "n16_draw_diagnostics": reference_draw_diagnostics,
        "stability_recommended_draw_count": stability_recommendation,
        "feasibility_gate_passed": feasibility_passed,
        "n32_followup_required": n32_followup_required,
        "pilot_engineering_gate_passed": pilot_engineering_gate_passed,
        "phase_d_entry_authorized": False,
        "recommended_draw_count": recommendation,
        "accepted": pilot_engineering_gate_passed,
        "next_required_record": (
            "freeze_draw_count_and_compute_budget_before_phase_d"
            if pilot_engineering_gate_passed
            else "complete_required_n32_followup_or_stop"
            if n32_followup_required
            else "pilot_failed"
        ),
        "interpretation": (
            "engineering_draw_count_heuristic_not_accuracy_or_coverage_evidence"
        ),
    }


def evaluate_n32_followup_acceptance(
    parent_payload: Mapping[str, object],
    block_results: Sequence[Mapping[str, object]],
    rule: PilotAcceptanceRule = PilotAcceptanceRule(),
    *,
    physical_config: Optional[OperatingDecisionRealismConfig] = None,
    selector_rule: ProspectiveSelectorRule = ProspectiveSelectorRule(),
    cost_scenario: ProspectiveCostScenario = PRIMARY_PROSPECTIVE_COST_SCENARIO,
) -> dict:
    """Recompute the N=16-to-N=32 gate from complete authenticated evidence."""

    if not isinstance(parent_payload, Mapping):
        raise ValueError("N=32 acceptance needs the complete parent pilot payload")
    if not isinstance(rule, PilotAcceptanceRule):
        raise ValueError("N=32 acceptance needs the frozen pilot rule")
    if physical_config is None:
        physical_config = corrected_partition_config(
            CorrectedPartition(
                name=PROSPECTIVE_PILOT_PARTITION,
                block_count=PILOT_BLOCK_COUNT,
                campaign=PROSPECTIVE_CAMPAIGN,
            ),
            CORRECTED_REPLICATION_CONFIG,
        )
    if not isinstance(selector_rule, ProspectiveSelectorRule):
        raise ValueError("N=32 acceptance needs a prospective selector rule")
    if not isinstance(cost_scenario, ProspectiveCostScenario):
        raise ValueError("N=32 acceptance needs a prospective cost scenario")
    source_revision = parent_payload.get("source_revision")
    if not isinstance(source_revision, str):
        raise ValueError("parent pilot source revision is missing")
    parent_cases = _validate_parent_pilot_payload(
        parent_payload,
        source_revision=source_revision,
        physical_config=physical_config,
        selector_rule=selector_rule,
        cost_scenario=cost_scenario,
    )

    followup_cases = _validate_followup_case_matrix(block_results)
    parent_by_identity = {
        (case["block"], case["truth_condition"]): case for case in parent_cases
    }
    comparisons = []
    validated_references = []
    pipeline_failures = 0
    selection_failures = 0
    unavailable_diagnostics = 0
    whole_draw_failures = 0
    n16_prefix_mismatches = []
    for block_record in sorted(block_results, key=lambda item: int(item["block"])):
        _exact_mapping(
            block_record,
            {
                "block",
                "cases",
                "corrected_random_stream_manifest",
                "corrected_random_stream_audit",
                "timing",
            },
            "complete N=32 block",
        )
        block_index = int(block_record["block"])
        manifest = block_record["corrected_random_stream_manifest"]
        corrected_audit = block_record["corrected_random_stream_audit"]
        block_failure = None
        if (
            isinstance(corrected_audit, Mapping)
            and set(corrected_audit)
            == {"clean", "not_run_due_to_block_failure"}
        ):
            if corrected_audit["clean"] is not False or manifest != []:
                raise ValueError("canonical failed N=32 block is inconsistent")
            block_failure = _validate_pipeline_failure(
                corrected_audit["not_run_due_to_block_failure"],
                "canonical failed N=32 block failure",
            )
            if block_failure["stage"] != "n32_paired_block_generation":
                raise ValueError("canonical failed N=32 block stage is invalid")
        else:
            _validate_corrected_stream_records(
                manifest,
                corrected_audit,
                block=block_index,
                label="N=32",
                cases=block_record["cases"],
            )
        for case in block_record["cases"]:
            _exact_mapping(
                case,
                {
                    "block",
                    "truth_condition",
                    "device_token",
                    "admissible_source_models",
                    "admissible_source_model_count",
                    "n32_complete_uncertainty_result",
                    "strict_prefixes",
                    "pipeline_failures",
                    "timing",
                },
                "complete N=32 case",
            )
            if case["block"] != block_index:
                raise ValueError("N=32 follow-up case is nested under the wrong block")
            identity = (case["block"], case["truth_condition"])
            parent_case = parent_by_identity[identity]
            raw_failures = case.get("pipeline_failures")
            if not isinstance(raw_failures, list):
                raise ValueError("N=32 pipeline failures are malformed")
            failures = tuple(
                _validate_pipeline_failure(
                    failure,
                    "canonical N=32 pipeline failure",
                )
                for failure in raw_failures
            )
            pipeline_failures += len(failures)
            if block_failure is not None and (
                failures != (block_failure,)
                or case.get("device_token") is not None
                or case.get("admissible_source_models") is not None
                or case.get("admissible_source_model_count") is not None
                or case.get("n32_complete_uncertainty_result") is not None
            ):
                raise ValueError("canonical failed N=32 block case is inconsistent")
            if block_failure is None and case.get("device_token") not in (
                None,
                parent_case.get("device_token"),
            ):
                raise ValueError("N=32 device identity does not match parent pilot")
            if (
                block_failure is None
                and case.get("device_token") is None
                and not failures
            ):
                raise ValueError("N=32 device identity is missing")
            prefixes = case.get("strict_prefixes")
            if not isinstance(prefixes, list) or [
                value.get("draw_count")
                for value in prefixes
                if isinstance(value, Mapping)
            ] != [PILOT_GENERATED_DRAW_COUNT, PILOT_N32_FOLLOWUP_DRAW_COUNT]:
                raise ValueError("N=32 strict prefixes are incomplete")
            complete_payload = case.get("n32_complete_uncertainty_result")
            if complete_payload is None:
                if not failures:
                    raise ValueError("complete prospective uncertainty payload is missing")
                for prefix, draw_count in zip(
                    prefixes,
                    (PILOT_GENERATED_DRAW_COUNT, PILOT_N32_FOLLOWUP_DRAW_COUNT),
                ):
                    failed = _validate_failed_prefix_record(
                        prefix,
                        draw_count=draw_count,
                        max_unstable_draws=PILOT_STRICT_MAX_UNSTABLE,
                    )
                    prefix_failure = failed["pipeline_failure"]
                    if not any(
                        prefix_failure["error_type"] == failure["error_type"]
                        and prefix_failure["message"] == failure["message"]
                        and (
                            prefix_failure["stage"] == failure["stage"]
                            or str(failure["stage"]).startswith(
                                f"{prefix_failure['stage']}_n"
                            )
                        )
                        for failure in failures
                    ):
                        raise ValueError(
                            "failed N=32 prefix does not match its pipeline failure"
                        )
                unavailable_diagnostics += 1
                n16_prefix_mismatches.append(
                    {"block": identity[0], "truth_condition": identity[1]}
                )
                continue
            complete = _validate_complete_uncertainty_payload(
                complete_payload,
                physical_config=physical_config,
                block=block_index,
                draw_count=PILOT_N32_FOLLOWUP_DRAW_COUNT,
                max_unstable_draws=PILOT_STRICT_MAX_UNSTABLE,
            )
            snapshot = complete["acquisition_evidence"]["snapshot"]
            source_models = list(snapshot["admissible_candidate_models"])
            if (
                case.get("admissible_source_models") != source_models
                or case.get("admissible_source_model_count")
                != len(source_models)
            ):
                raise ValueError("N=32 source-model record is inconsistent")
            derived_n16 = _derive_uncertainty_prefix_payload(
                complete,
                physical_config=physical_config,
                block=block_index,
                draw_count=PILOT_GENERATED_DRAW_COUNT,
                max_unstable_draws=PILOT_STRICT_MAX_UNSTABLE,
            )
            validated_prefixes = []
            for prefix, expected_complete, draw_count in (
                (prefixes[0], derived_n16, PILOT_GENERATED_DRAW_COUNT),
                (prefixes[1], complete, PILOT_N32_FOLLOWUP_DRAW_COUNT),
            ):
                if isinstance(prefix, Mapping) and "pipeline_failure" in prefix:
                    if not failures:
                        raise ValueError(
                            "failed N=32 prefix has no recorded pipeline failure"
                        )
                    failed = _validate_failed_prefix_record(
                        prefix,
                        draw_count=draw_count,
                        max_unstable_draws=PILOT_STRICT_MAX_UNSTABLE,
                    )
                    prefix_failure = failed["pipeline_failure"]
                    if not any(
                        prefix_failure["error_type"] == failure["error_type"]
                        and prefix_failure["message"] == failure["message"]
                        and str(failure["stage"]).startswith(
                            str(prefix_failure["stage"])
                        )
                        for failure in failures
                    ):
                        raise ValueError(
                            "failed N=32 prefix does not match its pipeline failure"
                        )
                    validated_prefixes.append(None)
                else:
                    validated_prefixes.append(
                        _validate_authenticated_prefix(
                            prefix,
                            complete=expected_complete,
                            physical_config=physical_config,
                            selector_rule=selector_rule,
                            cost_scenario=cost_scenario,
                        )
                    )
            candidate, reference = validated_prefixes
            parent_prefix = _prefix_by_count(
                parent_case,
                PILOT_GENERATED_DRAW_COUNT,
            )
            if candidate is None:
                unavailable_diagnostics += 1
                n16_prefix_mismatches.append(
                    {"block": identity[0], "truth_condition": identity[1]}
                )
            elif candidate != parent_prefix:
                raise ValueError(
                    "authenticated N=16 prefix does not match parent pilot"
                )
            if reference is None:
                unavailable_diagnostics += 1
                continue
            if str(reference.get("choice_token", "")).startswith(
                _SELECTION_FAILURE_PREFIX
            ):
                selection_failures += 1
            validated_references.append(reference)
            diagnostics = reference.get("draw_diagnostics")
            if not isinstance(diagnostics, Mapping) or (
                diagnostics.get("available") is not True
            ):
                unavailable_diagnostics += 1
            else:
                whole_draw_failures += int(
                    diagnostics.get("failed_draw_count", 0)
                )
            if candidate is not None:
                comparisons.append(
                    {
                        "block": identity[0],
                        "truth_condition": identity[1],
                        **compare_pilot_choices(candidate, reference),
                    }
                )
    summary = _summarize_comparisons(comparisons, rule)
    parent_trigger_passed = True
    pilot_gate = (
        parent_trigger_passed
        and pipeline_failures == 0
        and selection_failures == 0
        and unavailable_diagnostics == 0
        and whole_draw_failures == 0
        and len(comparisons) == PILOT_BLOCK_COUNT * len(STAGE3_TRUTH_CONDITIONS)
        and bool(summary["meets_acceptance_rule"])
    )
    return {
        "rule": asdict(rule),
        "parent_trigger_passed": parent_trigger_passed,
        "authenticated_n16_prefix_match": not n16_prefix_mismatches,
        "n16_prefix_mismatches": n16_prefix_mismatches,
        "case_identity_match": True,
        "case_identity_mismatches": [],
        "comparisons": comparisons,
        "summary": summary,
        "n32_pipeline_failure_count": pipeline_failures,
        "n32_selection_failure_count": selection_failures,
        "n32_unavailable_draw_diagnostic_count": unavailable_diagnostics,
        "n32_whole_draw_failure_count": whole_draw_failures,
        "pilot_engineering_gate_passed": pilot_gate,
        "phase_d_entry_authorized": False,
        "recommended_draw_count": (
            PILOT_GENERATED_DRAW_COUNT if pilot_gate else None
        ),
        "accepted": pilot_gate,
        "next_required_record": (
            "freeze_draw_count_and_compute_budget_before_phase_d"
            if pilot_gate
            else "pilot_failed"
        ),
        "interpretation": (
            "engineering_draw_count_heuristic_not_accuracy_or_coverage_evidence"
        ),
    }


_MEASUREMENT_KEYS = frozenset(
    {
        "timing",
        "wall_seconds",
        "cpu_seconds",
        "peak_rss_bytes",
        "peak_rss_before_bytes",
        "peak_rss_after_bytes",
        "decision_computation_seconds",
        "archive_size_bytes",
        "archive_benchmark",
    }
)


def _without_measurements(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _without_measurements(item)
            for key, item in value.items()
            if str(key) not in _MEASUREMENT_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_without_measurements(item) for item in value]
    return value


def prospective_pilot_scientific_payload(
    *,
    source_revision: str,
    protocol_digest: str,
    block_results: Sequence[Mapping[str, object]],
    acceptance: Mapping[str, object],
) -> dict:
    """Return worker-count-independent scientific material for authentication."""

    _validate_revision(source_revision)
    _validate_sha256("pilot protocol digest", protocol_digest)
    ordered = tuple(sorted(block_results, key=lambda item: int(item["block"])))
    if tuple(int(item["block"]) for item in ordered) != tuple(
        range(PILOT_BLOCK_COUNT)
    ):
        raise ValueError("scientific pilot payload needs all four ordered blocks")
    return {
        "source_revision": source_revision,
        "protocol_digest": protocol_digest,
        "blocks": _without_measurements(ordered),
        "acceptance": _without_measurements(acceptance),
    }


def prospective_n32_followup_scientific_payload(
    *,
    source_revision: str,
    protocol_digest: str,
    parent_payload_digest: str,
    parent_protocol_digest: str,
    parent_scientific_result_digest: str,
    block_results: Sequence[Mapping[str, object]],
    acceptance: Mapping[str, object],
) -> dict:
    """Return timing-independent scientific material for the N=32 artifact."""

    _validate_revision(source_revision)
    for name, value in (
        ("N=32 protocol digest", protocol_digest),
        ("parent payload digest", parent_payload_digest),
        ("parent protocol digest", parent_protocol_digest),
        ("parent scientific result digest", parent_scientific_result_digest),
    ):
        _validate_sha256(name, value)
    ordered = tuple(sorted(block_results, key=lambda item: int(item["block"])))
    _validate_followup_case_matrix(ordered)
    return {
        "source_revision": source_revision,
        "protocol_digest": protocol_digest,
        "parent_payload_digest": parent_payload_digest,
        "parent_protocol_digest": parent_protocol_digest,
        "parent_scientific_result_digest": parent_scientific_result_digest,
        "blocks": _without_measurements(ordered),
        "acceptance": _without_measurements(acceptance),
    }


def _percentile_nearest_rank(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


def _compute_budget_inputs(
    blocks: Sequence[Mapping[str, object]],
    total_wall_seconds: float,
    worker_count: int,
) -> dict:
    cases = tuple(case for block in blocks for case in block["cases"])  # type: ignore[index]
    block_wall = tuple(
        float(block["timing"]["block_wall_seconds"])  # type: ignore[index]
        for block in blocks
    )
    block_cpu = tuple(
        float(block["timing"]["block_cpu_seconds"])  # type: ignore[index]
        for block in blocks
    )
    fixed_policy_seconds = {name: 0.0 for name in POLICY_NAMES}
    predictive_n16_wall = 0.0
    predictive_n16_cpu = 0.0
    prefix_study_wall = 0.0
    prefix_study_cpu = 0.0
    retained_final_scoring_wall = 0.0
    retained_final_scoring_cpu = 0.0
    for case in cases:
        timing = case["timing"]["decision_and_verification_by_policy"]
        reveal = case["timing"]["post_save_reveal_and_scoring_by_policy"]
        for name in POLICY_NAMES:
            fixed_policy_seconds[name] += float(
                timing.get(name, {}).get("wall_seconds", 0.0)
            )
            fixed_policy_seconds[name] += float(
                reveal.get(name, {}).get("wall_seconds", 0.0)
            )
        n16_timing = case["timing"].get("n16_generation")
        if n16_timing is not None:
            predictive_n16_wall += float(n16_timing["wall_seconds"])
            predictive_n16_cpu += float(n16_timing["cpu_seconds"])
        for item in case["timing"]["prefix_scoring"].values():
            prefix_study_wall += float(item["wall_seconds"])
            prefix_study_cpu += float(item["cpu_seconds"])
        final_scoring = case["timing"]["prefix_scoring"].get(
            str(PILOT_GENERATED_DRAW_COUNT)
        )
        if final_scoring is not None:
            retained_final_scoring_wall += float(final_scoring["wall_seconds"])
            retained_final_scoring_cpu += float(final_scoring["cpu_seconds"])
    throughput = PILOT_BLOCK_COUNT / total_wall_seconds if total_wall_seconds else 0.0
    phase_estimates = []
    for spec in PROSPECTIVE_PARTITION_PLAN[1:]:
        phase_estimates.append(
            {
                "partition": spec.name,
                "block_count": spec.block_count,
                "case_count": spec.block_count * len(STAGE3_TRUTH_CONDITIONS),
                "estimated_wall_seconds_at_measured_worker_count": (
                    spec.block_count / throughput if throughput else None
                ),
                "estimated_cpu_seconds": (
                    statistics.fmean(block_cpu) * spec.block_count
                ),
            }
        )
    total_block_wall = sum(block_wall)
    total_block_cpu = sum(block_cpu)
    fixed_worker_wall = max(
        0.0,
        total_block_wall
        - predictive_n16_wall
        - prefix_study_wall
        + retained_final_scoring_wall,
    )
    fixed_worker_cpu = max(
        0.0,
        total_block_cpu
        - predictive_n16_cpu
        - prefix_study_cpu
        + retained_final_scoring_cpu,
    )
    wall_concurrency_factor = (
        total_wall_seconds / total_block_wall if total_block_wall else 1.0
    )
    draw_count_projections = []
    for draw_count in PILOT_DRAW_COUNTS:
        projected_worker_wall_per_block = (
            fixed_worker_wall
            + predictive_n16_wall * draw_count / PILOT_GENERATED_DRAW_COUNT
        ) / PILOT_BLOCK_COUNT
        projected_cpu_per_block = (
            fixed_worker_cpu
            + predictive_n16_cpu * draw_count / PILOT_GENERATED_DRAW_COUNT
        ) / PILOT_BLOCK_COUNT
        draw_count_projections.append(
            {
                "draw_count": draw_count,
                "estimated_worker_wall_seconds_per_block": (
                    projected_worker_wall_per_block
                ),
                "estimated_cpu_seconds_per_block": projected_cpu_per_block,
                "estimated_full_campaign_wall_seconds_at_measured_concurrency": (
                    projected_worker_wall_per_block
                    * sum(item.block_count for item in PROSPECTIVE_PARTITION_PLAN[1:])
                    * wall_concurrency_factor
                ),
                "estimated_full_campaign_cpu_seconds": (
                    projected_cpu_per_block
                    * sum(item.block_count for item in PROSPECTIVE_PARTITION_PLAN[1:])
                ),
            }
        )
    return {
        "pilot_block_wall_seconds": list(block_wall),
        "pilot_block_cpu_seconds": list(block_cpu),
        "mean_block_wall_seconds": statistics.fmean(block_wall),
        "p90_block_wall_seconds_nearest_rank": _percentile_nearest_rank(
            block_wall,
            0.90,
        ),
        "mean_block_cpu_seconds": statistics.fmean(block_cpu),
        "measured_block_throughput_per_second": throughput,
        "measured_worker_count": worker_count,
        "maximum_recorded_worker_peak_rss_bytes": max(
            int(block["timing"]["peak_rss_bytes"]) for block in blocks
        ),
        "conservative_concurrent_peak_rss_bytes": (
            max(int(block["timing"]["peak_rss_bytes"]) for block in blocks)
            * worker_count
        ),
        "measured_pilot_throughput_scope": (
            "conservative_n16_plus_all_prefix_and_eligibility_sensitivity_work"
        ),
        "predictive_n16_wall_seconds": predictive_n16_wall,
        "predictive_n16_cpu_seconds": predictive_n16_cpu,
        "prefix_study_overhead_wall_seconds": prefix_study_wall,
        "prefix_study_overhead_cpu_seconds": prefix_study_cpu,
        "retained_final_scoring_wall_seconds": retained_final_scoring_wall,
        "retained_final_scoring_cpu_seconds": retained_final_scoring_cpu,
        "draw_count_linear_projections": draw_count_projections,
        "one_source_case_count": sum(
            case.get("admissible_source_model_count") == 1 for case in cases
        ),
        "two_source_case_count": sum(
            case.get("admissible_source_model_count") == 2 for case in cases
        ),
        "zero_source_case_count": sum(
            case.get("admissible_source_model_count") == 0 for case in cases
        ),
        "unknown_source_case_count": sum(
            case.get("admissible_source_model_count") is None for case in cases
        ),
        "fixed_policy_wall_seconds_including_verification_and_reveal": (
            fixed_policy_seconds
        ),
        "fixed_policy_comparators_included": list(POLICY_NAMES),
        "verification_included": True,
        "runtime_host_manifest": _runtime_host_manifest(),
        "conservative_measured_throughput_phase_estimates": phase_estimates,
        "assumptions": [
            "same_per_block_case_mix_as_the_disposable_pilot",
            "same_worker_count_and_host",
            "all_fixed_policy_comparators_and_verification_retained",
            "raw_predictions_reused_for_cost_only_scenarios",
            "artifact_writing_added_after_archive_benchmark",
        ],
    }


def run_prospective_draw_count_pilot(
    *,
    source_revision: str,
    workers: int = 1,
    physical_config: Optional[OperatingDecisionRealismConfig] = None,
    selector_rule: ProspectiveSelectorRule = ProspectiveSelectorRule(),
    cost_scenario: ProspectiveCostScenario = PRIMARY_PROSPECTIVE_COST_SCENARIO,
    progress: Optional[Callable[[str], None]] = None,
) -> ProspectivePilotResult:
    """Run only the four-block disposable pilot; no other partition is accepted."""

    _validate_revision(source_revision)
    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise ValueError("pilot workers must be a positive integer")
    partition = CorrectedPartition(
        name=PROSPECTIVE_PILOT_PARTITION,
        block_count=PILOT_BLOCK_COUNT,
        campaign=PROSPECTIVE_CAMPAIGN,
    )
    if physical_config is None:
        physical_config = corrected_partition_config(
            partition,
            CORRECTED_REPLICATION_CONFIG,
        )
    if physical_config.sensor.trial_count != PILOT_BLOCK_COUNT:
        raise ValueError("pilot physical configuration must contain four blocks")
    if not isinstance(selector_rule, ProspectiveSelectorRule):
        raise ValueError("pilot needs a prospective selector rule")
    if not isinstance(cost_scenario, ProspectiveCostScenario):
        raise ValueError("pilot needs a prospective cost scenario")

    protocol_digest = prospective_pilot_protocol_digest(
        physical_config,
        selector_rule,
        cost_scenario,
        source_revision,
    )
    wall_started = perf_counter()
    completed: Dict[int, dict] = {}
    arguments = tuple(
        (block, physical_config, selector_rule, cost_scenario)
        for block in range(PILOT_BLOCK_COUNT)
    )
    if workers == 1:
        for argument in arguments:
            try:
                block_result = _run_pilot_block(*argument)
            except Exception as error:
                block_result = _failed_block_result(argument[0], error)
            completed[block_result["block"]] = block_result
            if progress is not None:
                progress(
                    f"{PROSPECTIVE_PILOT_PARTITION}: "
                    f"block {len(completed)}/{PILOT_BLOCK_COUNT}"
                )
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, PILOT_BLOCK_COUNT)
        ) as executor:
            futures = {
                executor.submit(_run_pilot_block_worker, argument): argument[0]
                for argument in arguments
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as error:
                    result = _failed_block_result(futures[future], error)
                completed[result["block"]] = result
                if progress is not None:
                    progress(
                        f"{PROSPECTIVE_PILOT_PARTITION}: "
                        f"block {len(completed)}/{PILOT_BLOCK_COUNT}"
                    )
    wall_seconds = perf_counter() - wall_started
    blocks = tuple(completed[index] for index in range(PILOT_BLOCK_COUNT))
    acceptance = evaluate_pilot_acceptance(blocks)
    cpu_seconds = sum(
        float(item["timing"]["block_cpu_seconds"]) for item in blocks
    )
    peak_rss = max(int(item["timing"]["peak_rss_bytes"]) for item in blocks)
    scientific_material = prospective_pilot_scientific_payload(
        source_revision=source_revision,
        protocol_digest=protocol_digest,
        block_results=blocks,
        acceptance=acceptance,
    )
    scientific_digest = _digest(
        f"{_HASH_DOMAIN}.scientific_result",
        scientific_material,
    )
    return ProspectivePilotResult(
        source_revision=source_revision,
        worker_count=min(workers, PILOT_BLOCK_COUNT),
        protocol_digest=protocol_digest,
        scientific_result_digest=scientific_digest,
        block_results=blocks,
        acceptance=acceptance,
        compute_budget_inputs=_compute_budget_inputs(
            blocks,
            wall_seconds,
            min(workers, PILOT_BLOCK_COUNT),
        ),
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
        peak_rss_bytes=peak_rss,
        selector_rule=selector_rule,
        cost_scenario=cost_scenario,
    )


def run_prospective_n32_followup(
    parent_payload: Mapping[str, object],
    *,
    source_revision: str,
    workers: int = 1,
    physical_config: Optional[OperatingDecisionRealismConfig] = None,
    selector_rule: ProspectiveSelectorRule = ProspectiveSelectorRule(),
    cost_scenario: ProspectiveCostScenario = PRIMARY_PROSPECTIVE_COST_SCENARIO,
    progress: Optional[Callable[[str], None]] = None,
) -> ProspectiveN32FollowupResult:
    """Run only the predeclared conditional all-case N=32 continuation."""

    _validate_revision(source_revision)
    if not isinstance(parent_payload, Mapping):
        raise ValueError("N=32 follow-up needs the complete parent pilot payload")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise ValueError("N=32 workers must be a positive integer")
    partition = CorrectedPartition(
        name=PROSPECTIVE_PILOT_PARTITION,
        block_count=PILOT_BLOCK_COUNT,
        campaign=PROSPECTIVE_CAMPAIGN,
    )
    if physical_config is None:
        physical_config = corrected_partition_config(
            partition,
            CORRECTED_REPLICATION_CONFIG,
        )
    if physical_config.sensor.trial_count != PILOT_BLOCK_COUNT:
        raise ValueError("N=32 physical configuration must contain four blocks")
    if not isinstance(selector_rule, ProspectiveSelectorRule):
        raise ValueError("N=32 follow-up needs a prospective selector rule")
    if not isinstance(cost_scenario, ProspectiveCostScenario):
        raise ValueError("N=32 follow-up needs a prospective cost scenario")
    _validate_parent_pilot_payload(
        parent_payload,
        source_revision=source_revision,
        physical_config=physical_config,
        selector_rule=selector_rule,
        cost_scenario=cost_scenario,
    )
    protocol_digest = prospective_n32_followup_protocol_digest(
        parent_payload,
        physical_config,
        selector_rule,
        cost_scenario,
        source_revision,
    )
    wall_started = perf_counter()
    completed: Dict[int, dict] = {}
    arguments = tuple(
        (block, physical_config, selector_rule, cost_scenario)
        for block in range(PILOT_BLOCK_COUNT)
    )
    if workers == 1:
        for argument in arguments:
            try:
                block_result = _run_n32_followup_block(*argument)
            except Exception as error:
                block_result = _failed_n32_followup_block(argument[0], error)
            completed[block_result["block"]] = block_result
            if progress is not None:
                progress(
                    f"{PILOT_N32_FOLLOWUP_ARTIFACT_ID}: "
                    f"block {len(completed)}/{PILOT_BLOCK_COUNT}"
                )
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, PILOT_BLOCK_COUNT)
        ) as executor:
            futures = {
                executor.submit(
                    _run_n32_followup_block_worker,
                    argument,
                ): argument[0]
                for argument in arguments
            }
            for future in as_completed(futures):
                try:
                    block_result = future.result()
                except Exception as error:
                    block_result = _failed_n32_followup_block(
                        futures[future],
                        error,
                    )
                completed[block_result["block"]] = block_result
                if progress is not None:
                    progress(
                        f"{PILOT_N32_FOLLOWUP_ARTIFACT_ID}: "
                        f"block {len(completed)}/{PILOT_BLOCK_COUNT}"
                    )
    wall_seconds = perf_counter() - wall_started
    blocks = tuple(completed[index] for index in range(PILOT_BLOCK_COUNT))
    acceptance = evaluate_n32_followup_acceptance(
        parent_payload,
        blocks,
        physical_config=physical_config,
        selector_rule=selector_rule,
        cost_scenario=cost_scenario,
    )
    parent_payload_digest = _digest(
        f"{_HASH_DOMAIN}.n32.parent_payload",
        parent_payload,
    )
    parent_protocol_digest = str(parent_payload["protocol_digest"])
    parent_scientific_result_digest = str(
        parent_payload["scientific_result_digest"]
    )
    scientific = prospective_n32_followup_scientific_payload(
        source_revision=source_revision,
        protocol_digest=protocol_digest,
        parent_payload_digest=parent_payload_digest,
        parent_protocol_digest=parent_protocol_digest,
        parent_scientific_result_digest=parent_scientific_result_digest,
        block_results=blocks,
        acceptance=acceptance,
    )
    scientific_digest = _digest(
        f"{_HASH_DOMAIN}.n32.scientific_result",
        scientific,
    )
    return ProspectiveN32FollowupResult(
        source_revision=source_revision,
        worker_count=min(workers, PILOT_BLOCK_COUNT),
        protocol_digest=protocol_digest,
        parent_payload=dict(parent_payload),
        parent_payload_digest=parent_payload_digest,
        parent_protocol_digest=parent_protocol_digest,
        parent_scientific_result_digest=parent_scientific_result_digest,
        scientific_result_digest=scientific_digest,
        block_results=blocks,
        acceptance=acceptance,
        wall_seconds=wall_seconds,
        cpu_seconds=sum(
            float(item["timing"]["block_cpu_seconds"]) for item in blocks
        ),
        peak_rss_bytes=max(
            int(item["timing"]["peak_rss_bytes"]) for item in blocks
        ),
    )


def prospective_pilot_result_payload(result: ProspectivePilotResult) -> dict:
    if not isinstance(result, ProspectivePilotResult):
        raise ValueError("prospective pilot payload needs a pilot result")
    expected_scientific_digest = _digest(
        f"{_HASH_DOMAIN}.scientific_result",
        prospective_pilot_scientific_payload(
            source_revision=result.source_revision,
            protocol_digest=result.protocol_digest,
            block_results=result.block_results,
            acceptance=result.acceptance,
        ),
    )
    if result.scientific_result_digest != expected_scientific_digest:
        raise ValueError("pilot scientific result digest is invalid")
    return {
        "schema_version": PROSPECTIVE_PILOT_SCHEMA_VERSION,
        "protocol_version": PROSPECTIVE_PILOT_PROTOCOL_VERSION,
        "campaign": PROSPECTIVE_CAMPAIGN,
        "partition": PROSPECTIVE_PILOT_PARTITION,
        "source_revision": result.source_revision,
        "protocol_digest": result.protocol_digest,
        "selector_rule": prospective_selector_rule_payload(result.selector_rule),
        "cost_scenario": _scenario_payload(result.cost_scenario),
        "scientific_result_digest": result.scientific_result_digest,
        "partition_plan": prospective_partition_plan_payload(),
        "worker_count": result.worker_count,
        "case_count": PILOT_BLOCK_COUNT * len(STAGE3_TRUTH_CONDITIONS),
        "block_results": list(result.block_results),
        "acceptance": result.acceptance,
        "compute_budget_inputs": result.compute_budget_inputs,
        "performance": {
            "wall_seconds": result.wall_seconds,
            "cpu_seconds": result.cpu_seconds,
            "peak_rss_bytes": result.peak_rss_bytes,
        },
        "scientific_use": "disposable_engineering_evidence_only",
    }


def prospective_n32_followup_result_payload(
    result: ProspectiveN32FollowupResult,
) -> dict:
    if not isinstance(result, ProspectiveN32FollowupResult):
        raise ValueError("N=32 payload needs an N=32 follow-up result")
    partition = CorrectedPartition(
        name=PROSPECTIVE_PILOT_PARTITION,
        block_count=PILOT_BLOCK_COUNT,
        campaign=PROSPECTIVE_CAMPAIGN,
    )
    physical_config = corrected_partition_config(
        partition,
        CORRECTED_REPLICATION_CONFIG,
    )
    selector_rule, cost_scenario = _parent_selector_and_cost(
        result.parent_payload
    )
    recomputed_acceptance = evaluate_n32_followup_acceptance(
        result.parent_payload,
        result.block_results,
        physical_config=physical_config,
        selector_rule=selector_rule,
        cost_scenario=cost_scenario,
    )
    if result.acceptance != recomputed_acceptance:
        raise ValueError("N=32 acceptance does not recompute from raw evidence")
    expected = _digest(
        f"{_HASH_DOMAIN}.n32.scientific_result",
        prospective_n32_followup_scientific_payload(
            source_revision=result.source_revision,
            protocol_digest=result.protocol_digest,
            parent_payload_digest=result.parent_payload_digest,
            parent_protocol_digest=result.parent_protocol_digest,
            parent_scientific_result_digest=(
                result.parent_scientific_result_digest
            ),
            block_results=result.block_results,
            acceptance=result.acceptance,
        ),
    )
    if result.scientific_result_digest != expected:
        raise ValueError("N=32 scientific result digest is invalid")
    return {
        "schema_version": PROSPECTIVE_N32_FOLLOWUP_SCHEMA_VERSION,
        "protocol_version": PROSPECTIVE_N32_FOLLOWUP_PROTOCOL_VERSION,
        "artifact_id": PILOT_N32_FOLLOWUP_ARTIFACT_ID,
        "campaign": PROSPECTIVE_CAMPAIGN,
        "partition": PROSPECTIVE_PILOT_PARTITION,
        "source_revision": result.source_revision,
        "protocol_digest": result.protocol_digest,
        "parent_payload_digest": result.parent_payload_digest,
        "parent_pilot_archive": result.parent_payload,
        "parent_protocol_digest": result.parent_protocol_digest,
        "parent_scientific_result_digest": (
            result.parent_scientific_result_digest
        ),
        "scientific_result_digest": result.scientific_result_digest,
        "design": prospective_n32_followup_design_payload(),
        "worker_count": result.worker_count,
        "case_count": PILOT_BLOCK_COUNT * len(STAGE3_TRUTH_CONDITIONS),
        "block_results": list(result.block_results),
        "acceptance": result.acceptance,
        "performance": {
            "wall_seconds": result.wall_seconds,
            "cpu_seconds": result.cpu_seconds,
            "peak_rss_bytes": result.peak_rss_bytes,
        },
        "scientific_use": "disposable_engineering_evidence_only",
    }


def validate_prospective_n32_followup_archive(
    payload: Mapping[str, object],
) -> dict:
    """Replay a saved N=32 archive from its complete embedded evidence."""

    item = _exact_mapping(
        payload,
        {
            "schema_version",
            "protocol_version",
            "artifact_id",
            "campaign",
            "partition",
            "source_revision",
            "protocol_digest",
            "parent_payload_digest",
            "parent_pilot_archive",
            "parent_protocol_digest",
            "parent_scientific_result_digest",
            "scientific_result_digest",
            "design",
            "worker_count",
            "case_count",
            "block_results",
            "acceptance",
            "performance",
            "scientific_use",
            "archive_content_digest",
            "archive_size_bytes",
        },
        "complete N=32 archive",
    )
    expected_header = {
        "schema_version": PROSPECTIVE_N32_FOLLOWUP_SCHEMA_VERSION,
        "protocol_version": PROSPECTIVE_N32_FOLLOWUP_PROTOCOL_VERSION,
        "artifact_id": PILOT_N32_FOLLOWUP_ARTIFACT_ID,
        "campaign": PROSPECTIVE_CAMPAIGN,
        "partition": PROSPECTIVE_PILOT_PARTITION,
        "design": prospective_n32_followup_design_payload(),
        "case_count": PILOT_BLOCK_COUNT * len(STAGE3_TRUTH_CONDITIONS),
        "scientific_use": "disposable_engineering_evidence_only",
    }
    if any(item[name] != expected for name, expected in expected_header.items()):
        raise ValueError("complete N=32 archive header is invalid")
    if (
        not isinstance(item["worker_count"], int)
        or isinstance(item["worker_count"], bool)
        or item["worker_count"] <= 0
    ):
        raise ValueError("complete N=32 archive worker count is invalid")
    _validate_revision(item["source_revision"])
    _validate_sha256("N=32 archive content digest", item["archive_content_digest"])
    if not isinstance(item["archive_size_bytes"], int) or isinstance(
        item["archive_size_bytes"], bool
    ):
        raise ValueError("complete N=32 archive size is invalid")
    archive_material = dict(item)
    archive_material.pop("archive_content_digest")
    archive_material.pop("archive_size_bytes")
    if item["archive_content_digest"] != _digest(
        f"{_HASH_DOMAIN}.n32.archive_content",
        archive_material,
    ):
        raise ValueError("complete N=32 archive content digest is invalid")
    if item["archive_size_bytes"] != len(_canonical_bytes(item) + b"\n"):
        raise ValueError("complete N=32 archive size is invalid")
    parent = item["parent_pilot_archive"]
    if not isinstance(parent, Mapping):
        raise ValueError("complete N=32 archive parent is missing")
    partition = CorrectedPartition(
        name=PROSPECTIVE_PILOT_PARTITION,
        block_count=PILOT_BLOCK_COUNT,
        campaign=PROSPECTIVE_CAMPAIGN,
    )
    physical_config = corrected_partition_config(
        partition,
        CORRECTED_REPLICATION_CONFIG,
    )
    selector_rule, cost_scenario = _parent_selector_and_cost(parent)
    _validate_parent_pilot_payload(
        parent,
        source_revision=item["source_revision"],
        physical_config=physical_config,
        selector_rule=selector_rule,
        cost_scenario=cost_scenario,
    )
    parent_digest = _digest(f"{_HASH_DOMAIN}.n32.parent_payload", parent)
    if (
        item["parent_payload_digest"] != parent_digest
        or item["parent_protocol_digest"] != parent["protocol_digest"]
        or item["parent_scientific_result_digest"]
        != parent["scientific_result_digest"]
    ):
        raise ValueError("complete N=32 archive parent references are invalid")
    expected_protocol = prospective_n32_followup_protocol_digest(
        parent,
        physical_config,
        selector_rule,
        cost_scenario,
        item["source_revision"],
    )
    if item["protocol_digest"] != expected_protocol:
        raise ValueError("complete N=32 archive protocol digest is invalid")
    blocks = item["block_results"]
    if not isinstance(blocks, list):
        raise ValueError("complete N=32 archive blocks are missing")
    acceptance = evaluate_n32_followup_acceptance(
        parent,
        blocks,
        physical_config=physical_config,
        selector_rule=selector_rule,
        cost_scenario=cost_scenario,
    )
    if item["acceptance"] != acceptance:
        raise ValueError("complete N=32 archive acceptance does not recompute")
    expected_scientific = _digest(
        f"{_HASH_DOMAIN}.n32.scientific_result",
        prospective_n32_followup_scientific_payload(
            source_revision=item["source_revision"],
            protocol_digest=expected_protocol,
            parent_payload_digest=parent_digest,
            parent_protocol_digest=parent["protocol_digest"],
            parent_scientific_result_digest=parent[
                "scientific_result_digest"
            ],
            block_results=blocks,
            acceptance=acceptance,
        ),
    )
    if item["scientific_result_digest"] != expected_scientific:
        raise ValueError("complete N=32 archive scientific digest is invalid")
    performance = _exact_mapping(
        item["performance"],
        {"wall_seconds", "cpu_seconds", "peak_rss_bytes"},
        "complete N=32 performance record",
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in performance.values()
    ):
        raise ValueError("complete N=32 performance record is invalid")
    return dict(item)


def format_prospective_n32_followup_report(
    payload: Mapping[str, object],
) -> str:
    acceptance = payload["acceptance"]  # type: ignore[index]
    summary = acceptance["summary"]
    performance = payload["performance"]  # type: ignore[index]
    return "\n".join(
        (
            "Prospective operating-decision conditional N=32 follow-up",
            "==========================================================",
            "",
            f"Artifact: {payload['artifact_id']}",
            f"Campaign: {payload['campaign']}",
            f"Partition: {payload['partition']}",
            f"Source revision: {payload['source_revision']}",
            f"Protocol digest: {payload['protocol_digest']}",
            f"Parent payload digest: {payload['parent_payload_digest']}",
            f"Scientific result digest: {payload['scientific_result_digest']}",
            "Cases: 12 in four paired blocks; all parent-pilot cases retained.",
            "Generated N=32 once per case; N=16 is an authenticated prefix.",
            "",
            "Parent trigger valid: "
            + ("yes" if acceptance["parent_trigger_passed"] else "no"),
            "N=16 prefixes match parent pilot: "
            + (
                "yes"
                if acceptance["authenticated_n16_prefix_match"]
                else "no"
            ),
            "Case identities match parent pilot: "
            + ("yes" if acceptance["case_identity_match"] else "no"),
            f"Agreement: {summary['agreement_count']}/{summary['case_count']} "
            f"({summary['agreement_rate']:.1%})",
            "Maximum changed-choice normalized utility regret: "
            f"{summary['maximum_changed_choice_normalized_utility_regret']:.1%}",
            f"N=32 pipeline failures: {acceptance['n32_pipeline_failure_count']}",
            f"N=32 selection failures: {acceptance['n32_selection_failure_count']}",
            f"N=32 whole-draw failures: {acceptance['n32_whole_draw_failure_count']}",
            "Recommended draw count: "
            + (
                str(acceptance["recommended_draw_count"])
                if acceptance["recommended_draw_count"] is not None
                else "none"
            ),
            "Final pilot engineering gate: "
            + ("PASS" if acceptance["pilot_engineering_gate_passed"] else "FAIL"),
            "Phase D authorized by this artifact: no; freeze draw count and "
            "compute budget first.",
            "",
            f"Wall time: {performance['wall_seconds']:.2f} s",
            f"Aggregate block CPU time: {performance['cpu_seconds']:.2f} s",
            f"Peak worker RSS: {performance['peak_rss_bytes']} bytes",
            "",
            "Archive replay authenticates evidence structure, fit invariants, and "
            "deterministic parameter/probe draws; it does not rerun predictive "
            "simulations or candidate refits.",
            "Boundary: this is disposable engineering evidence, not development, "
            "calibration, or confirmation.",
        )
    )


def save_prospective_n32_followup_artifacts(
    result: ProspectiveN32FollowupResult,
    *,
    json_path: Path | str,
    report_path: Path | str,
    hash_path: Path | str,
) -> SavedProspectivePilotArtifacts:
    """Save the source-bound N=32 JSON, report, and detached hashes once."""

    destinations = tuple(
        Path(item).expanduser().resolve()
        for item in (json_path, report_path, hash_path)
    )
    if len(set(destinations)) != 3:
        raise ValueError("N=32 output paths must be distinct")
    existing = tuple(path for path in destinations if path.exists())
    if existing:
        raise FileExistsError(
            "N=32 follow-up refuses to overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
    payload = prospective_n32_followup_result_payload(result)
    payload["archive_content_digest"] = _digest(
        f"{_HASH_DOMAIN}.n32.archive_content",
        payload,
    )
    payload["archive_size_bytes"] = 0
    archive_bytes = b""
    for _ in range(16):
        archive_bytes = _canonical_bytes(payload) + b"\n"
        observed = len(archive_bytes)
        if payload["archive_size_bytes"] == observed:
            break
        payload["archive_size_bytes"] = observed
    else:
        raise RuntimeError("N=32 archive-size fixed point did not converge")
    serialized_payload = _decoded_canonical_archive(
        archive_bytes,
        "N=32 archive",
    )
    validate_prospective_n32_followup_archive(serialized_payload)
    report_bytes = (
        format_prospective_n32_followup_report(serialized_payload) + "\n"
    ).encode("utf-8")
    json_digest = hashlib.sha256(archive_bytes).hexdigest()
    report_digest = hashlib.sha256(report_bytes).hexdigest()
    destinations[0].write_bytes(archive_bytes)
    destinations[1].write_bytes(report_bytes)
    destinations[2].write_text(
        f"{json_digest}  {destinations[0].name}\n"
        f"{report_digest}  {destinations[1].name}\n",
        encoding="utf-8",
    )
    return SavedProspectivePilotArtifacts(
        json_path=destinations[0],
        report_path=destinations[1],
        hash_path=destinations[2],
        json_sha256=json_digest,
        report_sha256=report_digest,
        archive_size_bytes=len(archive_bytes),
    )


def _write_benchmark(data: bytes, directory: Path) -> dict:
    wall_started = perf_counter()
    cpu_started = process_time()
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=".thermotwin-pilot-benchmark-",
        dir=directory,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
    wall_seconds = perf_counter() - wall_started
    cpu_seconds = process_time() - cpu_started
    temporary.unlink()
    return {
        "benchmark_bytes": len(data),
        "write_wall_seconds": wall_seconds,
        "write_cpu_seconds": cpu_seconds,
        "bytes_per_write_wall_second": (
            len(data) / wall_seconds if wall_seconds else None
        ),
    }


def _archive_payload_and_bytes(
    result: ProspectivePilotResult,
    directory: Path,
) -> Tuple[dict, bytes]:
    base = prospective_pilot_result_payload(result)
    serialization_wall_started = perf_counter()
    serialization_cpu_started = process_time()
    benchmark_bytes = _canonical_bytes(base) + b"\n"
    serialization = {
        "benchmark_bytes": len(benchmark_bytes),
        "serialization_wall_seconds": perf_counter() - serialization_wall_started,
        "serialization_cpu_seconds": process_time() - serialization_cpu_started,
    }
    benchmark = {**serialization, **_write_benchmark(benchmark_bytes, directory)}
    bytes_per_block = len(benchmark_bytes) / PILOT_BLOCK_COUNT
    bytes_per_serialization_wall_second = (
        len(benchmark_bytes) / benchmark["serialization_wall_seconds"]
        if benchmark["serialization_wall_seconds"]
        else None
    )
    compute_by_partition = {
        item["partition"]: item["estimated_wall_seconds_at_measured_worker_count"]
        for item in result.compute_budget_inputs[
            "conservative_measured_throughput_phase_estimates"
        ]
    }
    phase_archive_estimates = []
    for spec in PROSPECTIVE_PARTITION_PLAN[1:]:
        estimated_bytes = math.ceil(bytes_per_block * spec.block_count)
        serialization_seconds = (
            estimated_bytes / bytes_per_serialization_wall_second
            if bytes_per_serialization_wall_second
            else None
        )
        write_seconds = (
            estimated_bytes / benchmark["bytes_per_write_wall_second"]
            if benchmark["bytes_per_write_wall_second"]
            else None
        )
        compute_seconds = compute_by_partition[spec.name]
        total_seconds = (
            compute_seconds + serialization_seconds + write_seconds
            if compute_seconds is not None
            and serialization_seconds is not None
            and write_seconds is not None
            else None
        )
        phase_archive_estimates.append(
            {
                "partition": spec.name,
                "block_count": spec.block_count,
                "estimated_archive_bytes": estimated_bytes,
                "estimated_compute_wall_seconds": compute_seconds,
                "estimated_serialization_wall_seconds": serialization_seconds,
                "estimated_write_wall_seconds": write_seconds,
                "estimated_total_wall_seconds": total_seconds,
            }
        )
    post_pilot_blocks = sum(
        item.block_count for item in PROSPECTIVE_PARTITION_PLAN[1:]
    )
    post_pilot_bytes = math.ceil(bytes_per_block * post_pilot_blocks)
    post_pilot_serialization = (
        post_pilot_bytes / bytes_per_serialization_wall_second
        if bytes_per_serialization_wall_second
        else None
    )
    post_pilot_write = (
        post_pilot_bytes / benchmark["bytes_per_write_wall_second"]
        if benchmark["bytes_per_write_wall_second"]
        else None
    )
    draw_count_totals = []
    for item in result.compute_budget_inputs["draw_count_linear_projections"]:
        compute_seconds = item[
            "estimated_full_campaign_wall_seconds_at_measured_concurrency"
        ]
        total_seconds = (
            compute_seconds + post_pilot_serialization + post_pilot_write
            if post_pilot_serialization is not None and post_pilot_write is not None
            else None
        )
        draw_count_totals.append(
            {
                "draw_count": item["draw_count"],
                "estimated_compute_wall_seconds": compute_seconds,
                "estimated_serialization_wall_seconds": post_pilot_serialization,
                "estimated_write_wall_seconds": post_pilot_write,
                "estimated_total_wall_seconds": total_seconds,
            }
        )
    payload = dict(base)
    payload["archive_benchmark"] = {
        **benchmark,
        "bytes_per_serialization_wall_second": bytes_per_serialization_wall_second,
        "estimated_bytes_per_paired_block": bytes_per_block,
        "full_campaign_partition_estimates": phase_archive_estimates,
        "full_campaign_draw_count_estimates": draw_count_totals,
        "note": "measured_on_a_throwaway_write_before_the_final_archive",
    }
    payload["archive_content_digest"] = _digest(
        f"{_HASH_DOMAIN}.archive_content",
        payload,
    )
    payload["archive_size_bytes"] = 0
    for _ in range(16):
        encoded = _canonical_bytes(payload) + b"\n"
        observed = len(encoded)
        if payload["archive_size_bytes"] == observed:
            return payload, encoded
        payload["archive_size_bytes"] = observed
    raise RuntimeError("pilot archive-size fixed point did not converge")


def format_prospective_pilot_report(payload: Mapping[str, object]) -> str:
    """Return a compact human-readable report from the complete archive."""

    acceptance = payload["acceptance"]  # type: ignore[index]
    performance = payload["performance"]  # type: ignore[index]
    budget = payload["compute_budget_inputs"]  # type: ignore[index]
    archive = payload.get("archive_benchmark", {})
    lines = [
        "Prospective operating-decision disposable pilot",
        "===============================================",
        "",
        f"Campaign: {payload['campaign']}",
        f"Partition: {payload['partition']}",
        f"Source revision: {payload['source_revision']}",
        f"Protocol digest: {payload['protocol_digest']}",
        f"Scientific result digest: {payload['scientific_result_digest']}",
        f"Cases: {payload['case_count']} in {PILOT_BLOCK_COUNT} paired blocks",
        "Generated N=16 once per case; N=4/8/16 are authenticated prefixes.",
        "Primary eligibility comparison: maximum whole-draw failures = 0.",
        "Completed candidate transitions are baseline-floored and counted separately.",
        "Sensitivity only: N=16 with maximum whole-draw failures = 1.",
        "",
        "Draw-count acceptance relative to strict N=16:",
    ]
    for summary in acceptance["strict_summaries"]:
        lines.append(
            f"  N={summary['draw_count']}: agreement "
            f"{summary['agreement_count']}/{summary['case_count']} "
            f"({summary['agreement_rate']:.1%}); max changed-choice regret "
            f"{summary['maximum_changed_choice_normalized_utility_regret']:.1%}; "
            f"unevaluable changes {summary['unevaluable_changed_choice_count']}; "
            f"selected-action eligibility differences "
            f"{summary['selected_action_eligibility_difference_count']}; "
            f"{'PASS' if summary['meets_acceptance_rule'] else 'FAIL'}"
        )
    sensitivity = acceptance["n16_max_unstable_1_sensitivity"]["summary"]
    draw_diagnostics = acceptance["n16_draw_diagnostics"]
    lines.extend(
        (
            "",
            "Eligibility sensitivity:",
            f"  strict versus one allowed whole-draw failure: agreement "
            f"{sensitivity['agreement_count']}/{sensitivity['case_count']}; "
            f"selected-action eligibility differences "
            f"{sensitivity['selected_action_eligibility_difference_count']}",
            "",
            f"Pipeline failures: {acceptance['pipeline_failure_count']}",
            f"Strict N=16 selection failures: "
            f"{acceptance['n16_selection_failure_count']}",
            f"Strict N=16 ineligible action records: "
            f"{acceptance['n16_ineligible_action_count']}",
            f"Strict N=16 retained predictive draws: "
            f"{draw_diagnostics['retained_draw_count']}",
            f"Strict N=16 whole-draw failures: "
            f"{draw_diagnostics['failed_draw_count']}",
            f"Strict N=16 completed candidate-transition draws: "
            f"{draw_diagnostics['candidate_transition_draw_count']}",
            f"  candidate-loss draws: "
            f"{draw_diagnostics['candidate_loss_draw_count']}",
            f"  candidate-recovery draws: "
            f"{draw_diagnostics['candidate_recovery_draw_count']}",
            f"  candidate-loss statuses: "
            f"{json.dumps(draw_diagnostics['candidate_loss_status_counts'], sort_keys=True)}",
            "Tested-prefix draw-count recommendation: "
            + (
                str(acceptance["stability_recommended_draw_count"])
                if acceptance["stability_recommended_draw_count"] is not None
                else "none"
            ),
            f"N=32 follow-up required before freeze: "
            f"{'yes' if acceptance['n32_followup_required'] else 'no'}",
            "Frozen draw count authorized by this pilot: "
            + (
                str(acceptance["recommended_draw_count"])
                if acceptance["recommended_draw_count"] is not None
                else "none"
            ),
            f"Pilot engineering gate: "
            f"{'PASS' if acceptance['accepted'] else 'FAIL'}",
            "Phase D authorized by this artifact: no; freeze draw count and "
            "compute budget first.",
            "",
            "Measured resources:",
            f"  wall time: {performance['wall_seconds']:.2f} s",
            f"  aggregate block CPU time: {performance['cpu_seconds']:.2f} s",
            f"  peak worker RSS: {performance['peak_rss_bytes']} bytes",
            f"  one/two/zero/unknown-source cases: "
            f"{budget['one_source_case_count']}/"
            f"{budget['two_source_case_count']}/{budget['zero_source_case_count']}/"
            f"{budget['unknown_source_case_count']}",
            f"  runtime: Python {budget['runtime_host_manifest']['python_version']} "
            f"on {budget['runtime_host_manifest']['operating_system']} "
            f"{budget['runtime_host_manifest']['machine']}; "
            f"{budget['measured_worker_count']} workers",
            f"  archive size: {payload.get('archive_size_bytes', 'pending')} bytes",
            f"  benchmark write rate: "
            f"{archive.get('bytes_per_write_wall_second', 'pending')} bytes/s",
            "",
            "Projected campaign resources at measured throughput:",
        )
    )
    cpu_by_partition = {
        item["partition"]: item["estimated_cpu_seconds"]
        for item in budget["conservative_measured_throughput_phase_estimates"]
    }
    for item in archive["full_campaign_partition_estimates"]:
        compute_hours = item["estimated_compute_wall_seconds"] / 3600.0
        artifact_hours = (
            item["estimated_serialization_wall_seconds"]
            + item["estimated_write_wall_seconds"]
        ) / 3600.0
        total_hours = item["estimated_total_wall_seconds"] / 3600.0
        cpu_hours = cpu_by_partition[item["partition"]] / 3600.0
        lines.append(
            f"  {item['partition']}: {total_hours:.2f} total wall h "
            f"({compute_hours:.2f} compute + {artifact_hours:.2f} artifact); "
            f"{cpu_hours:.2f} CPU h"
        )
    lines.append("")
    lines.append("Draw-count projections for all 230 planned post-pilot blocks:")
    cpu_by_draw = {
        item["draw_count"]: item["estimated_full_campaign_cpu_seconds"]
        for item in budget["draw_count_linear_projections"]
    }
    for item in archive["full_campaign_draw_count_estimates"]:
        lines.append(
            f"  N={item['draw_count']}: "
            f"{item['estimated_total_wall_seconds'] / 3600.0:.2f} total wall h "
            f"({item['estimated_compute_wall_seconds'] / 3600.0:.2f} compute + "
            f"{(item['estimated_serialization_wall_seconds'] + item['estimated_write_wall_seconds']) / 3600.0:.2f} artifact); "
            f"{cpu_by_draw[item['draw_count']] / 3600.0:.2f} CPU h"
        )
    lines.extend(
        (
            "",
            "Boundary:",
            "  Archive replay authenticates structure and deterministic cross-record",
            "  invariants; it does not rerun acquisition/predictive simulations,",
            "  candidate refits, or post-reveal scoring.",
            "  This pilot may choose compute settings only. It is not development,",
            "  calibration, interval-coverage, or confirmatory performance evidence.",
        )
    )
    return "\n".join(lines)


def save_prospective_pilot_artifacts(
    result: ProspectivePilotResult,
    *,
    json_path: Path | str,
    report_path: Path | str,
    hash_path: Path | str,
) -> SavedProspectivePilotArtifacts:
    """Save complete JSON, compact report, and detached SHA-256 manifest once."""

    destinations = tuple(
        Path(item).expanduser().resolve()
        for item in (json_path, report_path, hash_path)
    )
    if len(set(destinations)) != 3:
        raise ValueError("pilot output paths must be distinct")
    existing = tuple(path for path in destinations if path.exists())
    if existing:
        raise FileExistsError(
            "prospective pilot refuses to overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)

    payload, archive_bytes = _archive_payload_and_bytes(result, destinations[0].parent)
    serialized_payload = _decoded_canonical_archive(archive_bytes, "pilot archive")
    validate_prospective_pilot_archive(serialized_payload)
    report = format_prospective_pilot_report(serialized_payload) + "\n"
    report_bytes = report.encode("utf-8")
    json_digest = hashlib.sha256(archive_bytes).hexdigest()
    report_digest = hashlib.sha256(report_bytes).hexdigest()
    destinations[0].write_bytes(archive_bytes)
    destinations[1].write_bytes(report_bytes)
    manifest = (
        f"{json_digest}  {destinations[0].name}\n"
        f"{report_digest}  {destinations[1].name}\n"
    )
    destinations[2].write_text(manifest, encoding="utf-8")
    if destinations[0].stat().st_size != serialized_payload["archive_size_bytes"]:
        raise RuntimeError("saved pilot archive size differs from its record")
    return SavedProspectivePilotArtifacts(
        json_path=destinations[0],
        report_path=destinations[1],
        hash_path=destinations[2],
        json_sha256=json_digest,
        report_sha256=report_digest,
        archive_size_bytes=len(archive_bytes),
    )


__all__ = [
    "CALIBRATION_BLOCK_COUNT",
    "DEVELOPMENT_CHECK_BLOCK_COUNT",
    "DEVELOPMENT_TUNING_BLOCK_COUNT",
    "PILOT_BLOCK_COUNT",
    "PILOT_DRAW_COUNTS",
    "PILOT_GENERATED_DRAW_COUNT",
    "PILOT_N32_FOLLOWUP_ARTIFACT_ID",
    "PILOT_N32_FOLLOWUP_CASE_SELECTION",
    "PILOT_N32_FOLLOWUP_DRAW_COUNT",
    "PILOT_SENSITIVITY_MAX_UNSTABLE",
    "PILOT_STRICT_MAX_UNSTABLE",
    "PROSPECTIVE_CALIBRATION_PARTITION",
    "PROSPECTIVE_CAMPAIGN",
    "PROSPECTIVE_DEVELOPMENT_CHECK_PARTITION",
    "PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION",
    "PROSPECTIVE_PARTITION_PLAN",
    "PROSPECTIVE_PILOT_PARTITION",
    "PROSPECTIVE_PILOT_PROTOCOL_VERSION",
    "PROSPECTIVE_PILOT_SCHEMA_VERSION",
    "PROSPECTIVE_N32_FOLLOWUP_PROTOCOL_VERSION",
    "PROSPECTIVE_N32_FOLLOWUP_SCHEMA_VERSION",
    "PROSPECTIVE_RESERVED_PARTITION",
    "PROSPECTIVE_SUPERSEDED_PILOT_PARTITIONS",
    "RESERVED_BLOCK_COUNT",
    "PilotAcceptanceRule",
    "ProspectivePartitionSpec",
    "ProspectiveN32FollowupResult",
    "ProspectivePilotResult",
    "SavedProspectivePilotArtifacts",
    "compare_pilot_choices",
    "evaluate_pilot_acceptance",
    "evaluate_n32_followup_acceptance",
    "format_prospective_n32_followup_report",
    "format_prospective_pilot_report",
    "prospective_partition_plan_payload",
    "prospective_n32_followup_design_payload",
    "prospective_n32_followup_protocol_digest",
    "prospective_n32_followup_result_payload",
    "prospective_n32_followup_scientific_payload",
    "prospective_pilot_protocol_digest",
    "prospective_pilot_result_payload",
    "prospective_pilot_scientific_payload",
    "run_prospective_draw_count_pilot",
    "run_prospective_n32_followup",
    "save_prospective_n32_followup_artifacts",
    "save_prospective_pilot_artifacts",
    "validate_prospective_pilot_archive",
]
