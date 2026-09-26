"""Pre-generation protocol and disposable rehearsal for prospective Phase D.

This module freezes the development-only choices declared in the Phase D plan.
It can execute one disposable paired block to prove that the full numerical
record survives canonical JSON serialization and strict replay.  It cannot
open the tuning, internal-check, calibration, or reserved partitions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter, process_time
from typing import Mapping, Tuple

from .operating_decision import (
    POLICY_NAMES,
    STOP_NOW,
)
from .operating_decision_calibration import DEFAULT_COST_SCENARIOS
from .operating_decision_prospective import (
    ProspectiveSelectorRule,
    prospective_action_catalog_digest,
    prospective_selector_rule_payload,
)
from .operating_decision_prospective_costs import (
    PRIMARY_PROSPECTIVE_COST_SCENARIO,
    PROSPECTIVE_COST_PRIMARY_SCENARIO_NAME,
    PROSPECTIVE_COST_SENSITIVITY_SCENARIOS,
    _scenario_payload,
)
from .operating_decision_prospective_phase_c_freeze import (
    FROZEN_MAX_UNSTABLE_DRAWS_PER_SOURCE_ACTION,
    FROZEN_PROSPECTIVE_DRAW_COUNT,
    FROZEN_PROSPECTIVE_WORKER_COUNT,
    PHASE_C_FREEZE_PAYLOAD_DIGEST,
    prospective_phase_c_freeze_payload,
)
from .operating_decision_prospective_pilot import (
    DEVELOPMENT_CHECK_BLOCK_COUNT,
    DEVELOPMENT_TUNING_BLOCK_COUNT,
    PILOT_DRAW_COUNTS,
    PILOT_GENERATED_DRAW_COUNT,
    PILOT_SENSITIVITY_MAX_UNSTABLE,
    PROSPECTIVE_CAMPAIGN,
    PROSPECTIVE_DEVELOPMENT_CHECK_PARTITION,
    PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION,
    _SELECTION_INFORMATION_BOUNDARY,
    _canonical_bytes,
    _derive_uncertainty_prefix_payload,
    _digest,
    _exact_mapping,
    _physical_config_payload,
    _prefix_by_count,
    _run_pilot_family,
    _stream_manifest_payload,
    _strict_json_value,
    _validate_authenticated_prefix,
    _validate_complete_uncertainty_payload,
    _validate_corrected_stream_records,
    _validate_fixed_policy_record,
    _validate_revision,
    _validate_sha256,
    pilot_max_unstable_draws,
)
from .operating_decision_provenance import (
    create_source_manifest,
    source_manifest_from_payload,
    source_manifest_payload,
    verify_source_manifest,
)
from .operating_decision_random_streams import RandomStreamRegistry
from .operating_decision_realism import (
    OperatingDecisionRealismConfig,
    STAGE3_TRUTH_CONDITIONS,
)
from .operating_decision_replication import (
    CORRECTED_REPLICATION_CONFIG,
    CorrectedPartition,
    corrected_partition_config,
    corrected_truth_for_block,
)
from .operating_decision_replication_protocol import (
    CORRECTED_NUMERICAL_SOURCE_PATHS,
)


PHASE_D_SCHEMA_VERSION = 1
PHASE_D_PROTOCOL_VERSION = "operating_decision_prospective_phase_d_v1"
PHASE_D_REHEARSAL_SCHEMA_VERSION = 1
PHASE_D_REHEARSAL_PROTOCOL_VERSION = (
    "operating_decision_prospective_phase_d_rehearsal_v1"
)
PHASE_D_REHEARSAL_PARTITION = "p0_disposable_phase_d_archive_roundtrip_v1"
PHASE_D_REHEARSAL_BLOCK_COUNT = 1
PHASE_D_OFFSET_QUANTILE = 0.90
PHASE_D_OFFSET_ORDER_STATISTIC = 18
PHASE_D_OFFSET_ROUNDING_KELVIN = 0.001
PHASE_D_MINIMUM_SEPARATE_SUPPORT_BLOCKS = 20
PHASE_D_GENERAL_CLEARANCE_GRID = (0.0, 0.05, 0.10)
PHASE_D_SINGLE_CANDIDATE_CLEARANCE_GRID = (0.0, 0.05, 0.10)
PHASE_D_MINIMUM_REDUCTION_GRID = (0.0, 0.01, 0.025)
PHASE_D_MINIMUM_UTILITY_GRID = (0.0, 0.01, 0.025)
PHASE_D_RULE_GRID_SIZE = 81
PHASE_D_DRAW_SENSITIVITY_BLOCKS = (0, 5, 10, 15)
PHASE_D_DRAW_SENSITIVITY_REFERENCE = 32
PHASE_D_DRAW_SENSITIVITY_CONTINGENCY = 64
PHASE_D_MINIMUM_ACTION_AGREEMENT = 0.90
PHASE_D_MAXIMUM_NORMALIZED_UTILITY_REGRET = 0.05
PHASE_D_HASH_DOMAIN = "thermotwin.prospective_phase_d"

PHASE_D_NUMERICAL_SOURCE_PATHS = tuple(
    sorted(
        set(CORRECTED_NUMERICAL_SOURCE_PATHS).union(
            {
                "thermotwin/reports/operating_decision_prospective_development.py",
                "thermotwin/studies/operating_decision_prospective.py",
                "thermotwin/studies/operating_decision_prospective_costs.py",
                "thermotwin/studies/operating_decision_prospective_phase_c_freeze.py",
                "thermotwin/studies/operating_decision_prospective_phase_d.py",
                "thermotwin/studies/operating_decision_prospective_pilot.py",
                "thermotwin/studies/operating_decision_prospective_random_streams.py",
                "thermotwin/studies/operating_decision_prospective_uncertainty.py",
            }
        )
    )
)


@dataclass(frozen=True)
class PhaseDSensorScenario:
    """One predeclared sensing or loading scenario."""

    name: str
    voltage_noise: float = 0.002
    face_temperature_noise: float = 0.02
    probe_capacitance: float = 5.0
    probe_response_time: float = 2.5
    run_bias_noise_ratio: float = 2.5
    primary: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Phase D sensor scenario needs a name")
        for label, value in (
            ("voltage noise", self.voltage_noise),
            ("face-temperature noise", self.face_temperature_noise),
            ("probe capacitance", self.probe_capacitance),
            ("probe response time", self.probe_response_time),
            ("run-bias ratio", self.run_bias_noise_ratio),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"Phase D {label} must be finite and positive")
        if not isinstance(self.primary, bool):
            raise ValueError("Phase D primary-scenario flag must be boolean")


PHASE_D_SENSOR_SCENARIOS = (
    PhaseDSensorScenario("nominal", primary=True),
    PhaseDSensorScenario("high_voltage_noise", voltage_noise=0.004),
    PhaseDSensorScenario("high_face_noise", face_temperature_noise=0.04),
    PhaseDSensorScenario("high_probe_loading", probe_capacitance=10.0),
)


@dataclass(frozen=True)
class SavedPhaseDRehearsalArtifacts:
    json_path: Path
    report_path: Path
    hash_path: Path
    json_sha256: str
    report_sha256: str
    archive_size_bytes: int


def _sensor_scenario_payload(value: PhaseDSensorScenario) -> dict:
    return {
        "name": value.name,
        "voltage_noise": value.voltage_noise,
        "face_temperature_noise": value.face_temperature_noise,
        "probe_capacitance": value.probe_capacitance,
        "probe_response_time": value.probe_response_time,
        "run_bias_noise_ratio": value.run_bias_noise_ratio,
        "primary": value.primary,
    }


def _balanced_loss_payload() -> dict:
    balanced = next(item for item in DEFAULT_COST_SCENARIOS if item.name == "balanced")
    return {
        "name": balanced.name,
        "false_approval_weight": balanced.false_approval_weight,
        "false_rejection_weight": balanced.false_rejection_weight,
        "abstention_weight": balanced.abstention_weight,
        "added_run_weight": balanced.added_run_weight,
        "added_sensor_weight": balanced.added_sensor_weight,
        "incremental_energy_weight": balanced.incremental_energy_weight,
    }


def prospective_phase_d_protocol_payload(
    *,
    source_revision: str,
    source_manifest_digest: str,
) -> dict:
    """Return the complete pre-generation Phase D design."""

    _validate_revision(source_revision)
    _validate_sha256("Phase D source-manifest digest", source_manifest_digest)
    cost_scenarios = [
        _scenario_payload(item) for item in PROSPECTIVE_COST_SENSITIVITY_SCENARIOS
    ]
    if len(cost_scenarios) != 12:
        raise RuntimeError("Phase D requires the frozen 12-cell cost grid")
    return {
        "schema_version": PHASE_D_SCHEMA_VERSION,
        "protocol_version": PHASE_D_PROTOCOL_VERSION,
        "campaign": PROSPECTIVE_CAMPAIGN,
        "source_revision": source_revision,
        "source_manifest_digest": source_manifest_digest,
        "phase_c": {
            "protocol_version": prospective_phase_c_freeze_payload()[
                "protocol_version"
            ],
            "payload_digest": PHASE_C_FREEZE_PAYLOAD_DIGEST,
            "draw_count": FROZEN_PROSPECTIVE_DRAW_COUNT,
            "max_unstable_draws_per_source_action": (
                FROZEN_MAX_UNSTABLE_DRAWS_PER_SOURCE_ACTION
            ),
            "worker_count": FROZEN_PROSPECTIVE_WORKER_COUNT,
            "complete_denominator_required": True,
            "failed_draw_scoring": "pre_action_width_no_gain",
        },
        "partitions": [
            {
                "name": PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION,
                "block_count": DEVELOPMENT_TUNING_BLOCK_COUNT,
                "case_count": DEVELOPMENT_TUNING_BLOCK_COUNT
                * len(STAGE3_TRUTH_CONDITIONS),
                "role": "development_tuning",
                "authorized_by_this_protocol": True,
            },
            {
                "name": PROSPECTIVE_DEVELOPMENT_CHECK_PARTITION,
                "block_count": DEVELOPMENT_CHECK_BLOCK_COUNT,
                "case_count": DEVELOPMENT_CHECK_BLOCK_COUNT
                * len(STAGE3_TRUTH_CONDITIONS),
                "role": "development_internal_check",
                "authorized_by_this_protocol": False,
            },
        ],
        "information_boundary": {
            "online_selection": _strict_json_value(_SELECTION_INFORMATION_BOUNDARY),
            "development_truth_use": (
                "offline_offsets_thresholds_diagnostics_and_sensor_sensitivity_only"
            ),
            "decisions_saved_before_truth_or_target_reveal": True,
            "counterfactual_fixed_policies": list(POLICY_NAMES),
        },
        "action_catalog": {
            "policies": list(POLICY_NAMES),
            "digest": prospective_action_catalog_digest(),
            "candidate_menu_change_permitted": False,
        },
        "offset_estimator": {
            "case_nonconformity": "max(0,L-m,m-U)",
            "missing_or_invalid_interval_score": "positive_infinity",
            "within_block_aggregation": "maximum_across_three_truth_families",
            "block_score_count": DEVELOPMENT_TUNING_BLOCK_COUNT,
            "quantile": PHASE_D_OFFSET_QUANTILE,
            "nearest_rank": PHASE_D_OFFSET_ORDER_STATISTIC,
            "round_up_to_kelvin": PHASE_D_OFFSET_ROUNDING_KELVIN,
            "minimum_separate_support_blocks": (
                PHASE_D_MINIMUM_SEPARATE_SUPPORT_BLOCKS
            ),
            "actions": list(POLICY_NAMES),
            "infinite_stop_fallback": "disable_early_resolved_stopping",
            "infinite_measurement_fallback": (
                "action_unselectable_but_fixed_comparator_retained"
            ),
            "family_or_candidate_count_specific_offsets": False,
            "distinct_from_phase_e_calibration": True,
        },
        "rule_grid": {
            "general_stopping_clearance_kelvin": list(
                PHASE_D_GENERAL_CLEARANCE_GRID
            ),
            "single_candidate_stopping_clearance_kelvin": list(
                PHASE_D_SINGLE_CANDIDATE_CLEARANCE_GRID
            ),
            "minimum_expected_reduction_kelvin": list(
                PHASE_D_MINIMUM_REDUCTION_GRID
            ),
            "minimum_utility_per_normalized_cost": list(
                PHASE_D_MINIMUM_UTILITY_GRID
            ),
            "cartesian_rule_count": PHASE_D_RULE_GRID_SIZE,
            "strict_threshold_comparison": True,
            "utility_decimal_places": 12,
            "selection_objective": {
                "aggregation": "mean_of_three_family_losses_then_mean_blocks",
                "loss": _balanced_loss_payload(),
                "tie_breakers": [
                    "fewer_false_approvals",
                    "fewer_false_rejections",
                    "more_definitive_decisions",
                    "lower_mean_realized_diagnostic_energy",
                    "lexicographically_smaller_rule_tuple",
                ],
            },
            "retain_every_grid_row": True,
        },
        "draw_sensitivity": {
            "tuning_block_indices": list(PHASE_D_DRAW_SENSITIVITY_BLOCKS),
            "truth_families": list(STAGE3_TRUTH_CONDITIONS),
            "case_count": len(PHASE_D_DRAW_SENSITIVITY_BLOCKS)
            * len(STAGE3_TRUTH_CONDITIONS),
            "candidate_draw_count": FROZEN_PROSPECTIVE_DRAW_COUNT,
            "reference_draw_count": PHASE_D_DRAW_SENSITIVITY_REFERENCE,
            "minimum_action_agreement": PHASE_D_MINIMUM_ACTION_AGREEMENT,
            "minimum_agreement_count": 11,
            "maximum_normalized_utility_regret": (
                PHASE_D_MAXIMUM_NORMALIZED_UTILITY_REGRET
            ),
            "changed_choices_must_have_evaluable_regret": True,
            "all_measurement_actions_must_be_eligible": True,
            "contingency_draw_count": PHASE_D_DRAW_SENSITIVITY_CONTINGENCY,
            "contingency_rule": (
                "if_n16_to_n32_fails_extend_all_tuning_to_n32_and_compare_"
                "the_same_subset_n32_to_n64"
            ),
            "maximum_draw_count": PHASE_D_DRAW_SENSITIVITY_CONTINGENCY,
        },
        "measurement_maps": {
            "primary_cost_scenario": PROSPECTIVE_COST_PRIMARY_SCENARIO_NAME,
            "cost_scenarios": cost_scenarios,
            "cost_cells_reuse_authenticated_nominal_predictions": True,
            "sensor_scenarios": [
                _sensor_scenario_payload(item) for item in PHASE_D_SENSOR_SCENARIOS
            ],
            "sensor_stress_scope": "twenty_block_tuning_partition_only",
            "changed_sensor_physics_requires_affected_simulation_and_refit": True,
            "required_cell_outputs": [
                "selected_action_counts_and_rates",
                "definitive_decision_counts_and_rates",
                "action_ineligibility",
                "selection_failure",
                "verification_failure",
                "no_useful_measurement",
            ],
        },
        "internal_check": {
            "rule_locked_before_open": True,
            "grid_reselection_forbidden": True,
            "poor_benefit_or_low_action_diversity_is_not_a_redesign_trigger": True,
            "descriptive_targets": {
                "maximum_false_approval_fraction": 0.10,
                "minimum_decision_coverage": 0.70,
                "failure_alone_reopens_tuning": False,
            },
            "redesign_triggers": [
                "material_information_physics_provenance_or_implementation_defect",
                "two_false_approvals_in_distinct_blocks_within_the_same_"
                "selected_action_and_candidate_count_stratum",
                "any_measurement_action_ineligible_in_more_than_20_percent_"
                "of_cases",
            ],
            "maximum_redesign_count": 1,
            "used_check_becomes_tuning_if_redesign_occurs": True,
            "new_check_namespace_required_before_replacement": True,
        },
        "compute_budget": {
            "nominal_tuning_wall_seconds": 35876.15561791521,
            "nominal_tuning_cpu_seconds": 131802.28392,
            "nominal_tuning_archive_bytes": 76965610,
            "nominal_internal_check_wall_seconds": 17938.077808957605,
            "nominal_internal_check_cpu_seconds": 65901.14196,
            "nominal_internal_check_archive_bytes": 38482805,
            "three_sensor_stress_wall_seconds": 107628.46685374563,
            "three_sensor_stress_cpu_seconds": 395406.85176,
            "three_sensor_stress_archive_bytes": 230896830,
            "planned_wall_seconds_before_draw_continuation": 161442.70028061845,
            "planned_cpu_seconds_before_draw_continuation": 593110.27764,
            "planned_archive_bytes_before_draw_continuation": 346345245,
            "n32_subset_wall_seconds": 14612.75,
            "n32_subset_cpu_seconds": 51370.77,
            "n32_subset_archive_bytes": 34992556,
            "concurrent_peak_rss_bytes": 185204736,
            "sensor_scenarios_run_sequentially": True,
            "contingency_budget_must_be_committed_before_execution": True,
        },
        "required_outputs": [
            "source_and_runtime_bound_protocol",
            "tuning_and_internal_check_artifact_hashes",
            "four_development_offsets",
            "complete_81_row_tuning_table",
            "one_selected_primary_rule",
            "draw_count_sensitivity_result",
            "twelve_cell_cost_map",
            "four_scenario_sensor_map",
            "complete_decision_error_coverage_resource_and_failure_metrics",
            "family_c_and_candidate_count_diagnostics",
            "attempted_variant_chronology",
            "phase_d_freeze_or_feasibility_closeout",
        ],
        "authorization": {
            "next_partition": PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION,
            "development_tuning_authorized_after_rehearsal_and_exact_head_ci": True,
            "development_internal_check_authorized_now": False,
            "independent_calibration_authorized_now": False,
            "reserved_evaluation_authorized_now": False,
        },
    }


def prospective_phase_d_protocol_digest(
    *,
    source_revision: str,
    source_manifest_digest: str,
) -> str:
    return _digest(
        f"{PHASE_D_HASH_DOMAIN}.protocol",
        prospective_phase_d_protocol_payload(
            source_revision=source_revision,
            source_manifest_digest=source_manifest_digest,
        ),
    )


def validate_prospective_phase_d_protocol(payload: Mapping[str, object]) -> dict:
    """Reject any drift from the committed Phase D design."""

    if not isinstance(payload, Mapping):
        raise ValueError("Phase D protocol must be a mapping")
    source_revision = payload.get("source_revision")
    manifest_digest = payload.get("source_manifest_digest")
    if not isinstance(source_revision, str) or not isinstance(manifest_digest, str):
        raise ValueError("Phase D protocol source binding is missing")
    expected = prospective_phase_d_protocol_payload(
        source_revision=source_revision,
        source_manifest_digest=manifest_digest,
    )
    if dict(payload) != expected:
        raise ValueError("Phase D protocol does not match the committed design")
    return expected


def _runtime_performance_payload(wall: float, cpu: float, peak_rss: int) -> dict:
    if (
        not math.isfinite(wall)
        or wall < 0.0
        or not math.isfinite(cpu)
        or cpu < 0.0
        or not isinstance(peak_rss, int)
        or peak_rss < 0
    ):
        raise ValueError("Phase D rehearsal performance record is invalid")
    return {"wall_seconds": wall, "cpu_seconds": cpu, "peak_rss_bytes": peak_rss}


def _peak_rss_bytes() -> int:
    from .operating_decision_prospective_pilot import _peak_rss_bytes as value

    return value()


def _run_rehearsal_block(
    physical_config: OperatingDecisionRealismConfig,
    selector_rule: ProspectiveSelectorRule,
) -> dict:
    block = 0
    partition = CorrectedPartition(
        name=PHASE_D_REHEARSAL_PARTITION,
        block_count=PHASE_D_REHEARSAL_BLOCK_COUNT,
        campaign=PROSPECTIVE_CAMPAIGN,
    )
    wall_started = perf_counter()
    cpu_started = process_time()
    registry = RandomStreamRegistry()
    truth_started = perf_counter()
    truth_cpu_started = process_time()
    truth = corrected_truth_for_block(partition, block, physical_config, registry)
    truth_timing = _runtime_performance_payload(
        perf_counter() - truth_started,
        process_time() - truth_cpu_started,
        _peak_rss_bytes(),
    )
    cases = [
        _run_pilot_family(
            truth_condition=truth_condition,
            block=block,
            truth=truth,
            partition=partition,
            physical_config=physical_config,
            registry=registry,
            selector_rule=selector_rule,
            cost_scenario=PRIMARY_PROSPECTIVE_COST_SCENARIO,
        )
        for truth_condition in STAGE3_TRUTH_CONDITIONS
    ]
    audit = registry.audit()
    audit.assert_clean()
    return {
        "block": block,
        "cases": cases,
        "corrected_random_stream_manifest": _stream_manifest_payload(registry.uses),
        "corrected_random_stream_audit": _strict_json_value(audit),
        "timing": {
            "truth_generation": truth_timing,
            "block_wall_seconds": perf_counter() - wall_started,
            "block_cpu_seconds": process_time() - cpu_started,
            "peak_rss_bytes": _peak_rss_bytes(),
        },
    }


def _replay_summary(block_result: Mapping[str, object]) -> dict:
    cases = block_result["cases"]
    choices = {}
    ineligible = 0
    failed_draws = 0
    for case in cases:  # type: ignore[union-attr]
        prefix = _prefix_by_count(case, PILOT_GENERATED_DRAW_COUNT)
        choices[case["truth_condition"]] = prefix["choice_token"]
        ineligible += sum(
            value["eligible"] is not True
            for policy, value in prefix["eligibility"].items()
            if policy != STOP_NOW
        )
        diagnostics = prefix["draw_diagnostics"]
        failed_draws += int(diagnostics["failed_draw_count"])
    return {
        "case_count": len(cases),  # type: ignore[arg-type]
        "truth_families": [case["truth_condition"] for case in cases],
        "n16_choices": choices,
        "pipeline_failure_count": sum(
            len(case["pipeline_failures"]) for case in cases
        ),
        "n16_ineligible_measurement_action_count": ineligible,
        "n16_whole_draw_failure_count": failed_draws,
        "block_evidence_digest": _digest(
            f"{PHASE_D_HASH_DOMAIN}.rehearsal_block",
            block_result,
        ),
    }


def _validate_rehearsal_case(
    case: Mapping[str, object],
    *,
    physical_config: OperatingDecisionRealismConfig,
) -> None:
    required = {
        "block",
        "truth_condition",
        "device_token",
        "truth_revealed_only_after_saves",
        "selection_information_boundary",
        "admissible_source_models",
        "admissible_source_model_count",
        "n16_complete_uncertainty_result",
        "primary_prefixes",
        "n16_max_unstable_0_sensitivity",
        "selected_policy_at_n16_primary",
        "selected_policy_outcome",
        "fixed_policy_results",
        "pipeline_failures",
        "timing",
    }
    _exact_mapping(case, required, "Phase D rehearsal case")
    if (
        case["block"] != 0
        or case["truth_condition"] not in STAGE3_TRUTH_CONDITIONS
        or not isinstance(case["device_token"], str)
        or not case["device_token"]
        or case["truth_revealed_only_after_saves"] is None
        or case["selection_information_boundary"] != _SELECTION_INFORMATION_BOUNDARY
        or case["pipeline_failures"] != []
    ):
        raise ValueError("Phase D rehearsal case evidence is incomplete")
    fixed = case["fixed_policy_results"]
    if not isinstance(fixed, Mapping) or set(fixed) != set(POLICY_NAMES):
        raise ValueError("Phase D rehearsal fixed-policy evidence is incomplete")
    for policy_name, record in fixed.items():
        if not isinstance(record, Mapping) or set(record) != {
            "case",
            "saved_before_reveal",
            "post_reveal_score",
        }:
            raise ValueError("Phase D rehearsal fixed-policy result is incomplete")
        _validate_fixed_policy_record(
            record,
            policy_name=policy_name,
            truth_condition=case["truth_condition"],
            block=0,
            device_token=case["device_token"],
            physical_config=physical_config,
            campaign=PROSPECTIVE_CAMPAIGN,
            partition_name=PHASE_D_REHEARSAL_PARTITION,
            partition_block_count=PHASE_D_REHEARSAL_BLOCK_COUNT,
        )
    complete = _validate_complete_uncertainty_payload(
        case["n16_complete_uncertainty_result"],
        physical_config=physical_config,
        block=0,
        draw_count=FROZEN_PROSPECTIVE_DRAW_COUNT,
        max_unstable_draws=FROZEN_MAX_UNSTABLE_DRAWS_PER_SOURCE_ACTION,
        campaign=PROSPECTIVE_CAMPAIGN,
        partition=PHASE_D_REHEARSAL_PARTITION,
    )
    sources = list(
        complete["acquisition_evidence"]["snapshot"]["admissible_candidate_models"]
    )
    if (
        case["admissible_source_models"] != sources
        or case["admissible_source_model_count"] != len(sources)
    ):
        raise ValueError("Phase D rehearsal source-model record is inconsistent")
    prefixes = case["primary_prefixes"]
    if not isinstance(prefixes, list) or [
        value.get("draw_count") for value in prefixes
    ] != list(PILOT_DRAW_COUNTS):
        raise ValueError("Phase D rehearsal prefix evidence is incomplete")
    selector_rule = ProspectiveSelectorRule()
    for prefix, draw_count in zip(prefixes, PILOT_DRAW_COUNTS):
        expected = _derive_uncertainty_prefix_payload(
            complete,
            physical_config=physical_config,
            block=0,
            draw_count=draw_count,
            max_unstable_draws=pilot_max_unstable_draws(draw_count),
            campaign=PROSPECTIVE_CAMPAIGN,
            partition=PHASE_D_REHEARSAL_PARTITION,
        )
        _validate_authenticated_prefix(
            prefix,
            complete=expected,
            physical_config=physical_config,
            selector_rule=selector_rule,
            cost_scenario=PRIMARY_PROSPECTIVE_COST_SCENARIO,
        )
    sensitivity_complete = _derive_uncertainty_prefix_payload(
        complete,
        physical_config=physical_config,
        block=0,
        draw_count=PILOT_GENERATED_DRAW_COUNT,
        max_unstable_draws=PILOT_SENSITIVITY_MAX_UNSTABLE,
        campaign=PROSPECTIVE_CAMPAIGN,
        partition=PHASE_D_REHEARSAL_PARTITION,
    )
    _validate_authenticated_prefix(
        case["n16_max_unstable_0_sensitivity"],
        complete=sensitivity_complete,
        physical_config=physical_config,
        selector_rule=selector_rule,
        cost_scenario=PRIMARY_PROSPECTIVE_COST_SCENARIO,
    )
    reference = _prefix_by_count(case, FROZEN_PROSPECTIVE_DRAW_COUNT)
    selected = reference["selection"]["selected_policy"]
    if (
        case["selected_policy_at_n16_primary"] != selected
        or case["selected_policy_outcome"] != fixed[selected]
    ):
        raise ValueError("Phase D rehearsal selected outcome is inconsistent")


def _validate_rehearsal_block(
    block_result: Mapping[str, object],
    *,
    physical_config: OperatingDecisionRealismConfig,
) -> None:
    _exact_mapping(
        block_result,
        {
            "block",
            "cases",
            "corrected_random_stream_manifest",
            "corrected_random_stream_audit",
            "timing",
        },
        "Phase D rehearsal block",
    )
    cases = block_result["cases"]
    if (
        block_result["block"] != 0
        or not isinstance(cases, list)
        or len(cases) != len(STAGE3_TRUTH_CONDITIONS)
        or [case.get("truth_condition") for case in cases]
        != list(STAGE3_TRUTH_CONDITIONS)
    ):
        raise ValueError("Phase D rehearsal case matrix is incomplete")
    _validate_corrected_stream_records(
        block_result["corrected_random_stream_manifest"],
        block_result["corrected_random_stream_audit"],
        block=0,
        label="parent pilot",
        campaign=PROSPECTIVE_CAMPAIGN,
        partition=PHASE_D_REHEARSAL_PARTITION,
    )
    for case in cases:
        _validate_rehearsal_case(case, physical_config=physical_config)


def run_phase_d_disposable_rehearsal(
    *,
    repository_root: Path | str,
    source_revision: str,
) -> dict:
    """Run exactly one disposable paired block; never a scientific partition."""

    _validate_revision(source_revision)
    root = Path(repository_root).expanduser().resolve(strict=True)
    manifest = create_source_manifest(root, PHASE_D_NUMERICAL_SOURCE_PATHS)
    protocol = prospective_phase_d_protocol_payload(
        source_revision=source_revision,
        source_manifest_digest=manifest.digest,
    )
    protocol_digest = prospective_phase_d_protocol_digest(
        source_revision=source_revision,
        source_manifest_digest=manifest.digest,
    )
    partition = CorrectedPartition(
        name=PHASE_D_REHEARSAL_PARTITION,
        block_count=PHASE_D_REHEARSAL_BLOCK_COUNT,
        campaign=PROSPECTIVE_CAMPAIGN,
    )
    physical_config = corrected_partition_config(
        partition,
        CORRECTED_REPLICATION_CONFIG,
    )
    wall_started = perf_counter()
    cpu_started = process_time()
    block = _run_rehearsal_block(physical_config, ProspectiveSelectorRule())
    performance = _runtime_performance_payload(
        perf_counter() - wall_started,
        process_time() - cpu_started,
        _peak_rss_bytes(),
    )
    _validate_rehearsal_block(block, physical_config=physical_config)
    replay = _replay_summary(block)
    scientific = {
        "source_revision": source_revision,
        "protocol_digest": protocol_digest,
        "block_result": block,
        "replay_summary": replay,
    }
    return {
        "schema_version": PHASE_D_REHEARSAL_SCHEMA_VERSION,
        "protocol_version": PHASE_D_REHEARSAL_PROTOCOL_VERSION,
        "campaign": PROSPECTIVE_CAMPAIGN,
        "partition": PHASE_D_REHEARSAL_PARTITION,
        "source_revision": source_revision,
        "source_manifest": source_manifest_payload(manifest),
        "phase_d_protocol": protocol,
        "protocol_digest": protocol_digest,
        "selector_rule": prospective_selector_rule_payload(
            ProspectiveSelectorRule()
        ),
        "cost_scenario": _scenario_payload(PRIMARY_PROSPECTIVE_COST_SCENARIO),
        "physical_config": _physical_config_payload(physical_config),
        "case_count": len(STAGE3_TRUTH_CONDITIONS),
        "block_result": block,
        "replay_summary": replay,
        "scientific_result_digest": _digest(
            f"{PHASE_D_HASH_DOMAIN}.rehearsal_scientific_result",
            scientific,
        ),
        "performance": performance,
        "scientific_use": "disposable_archive_roundtrip_only",
    }


def _archive_with_content_seal(payload: Mapping[str, object]) -> Tuple[dict, bytes]:
    sealed = dict(payload)
    sealed["archive_content_digest"] = _digest(
        f"{PHASE_D_HASH_DOMAIN}.rehearsal_archive_content",
        payload,
    )
    sealed["archive_size_bytes"] = 0
    archive = b""
    for _ in range(16):
        archive = _canonical_bytes(sealed) + b"\n"
        observed = len(archive)
        if sealed["archive_size_bytes"] == observed:
            return sealed, archive
        sealed["archive_size_bytes"] = observed
    raise RuntimeError("Phase D rehearsal archive-size fixed point did not converge")


def validate_phase_d_rehearsal_archive(
    payload: Mapping[str, object],
    *,
    repository_root: Path | str | None = None,
) -> dict:
    """Replay a serialized rehearsal and optionally verify executing source."""

    item = _exact_mapping(
        payload,
        {
            "schema_version",
            "protocol_version",
            "campaign",
            "partition",
            "source_revision",
            "source_manifest",
            "phase_d_protocol",
            "protocol_digest",
            "selector_rule",
            "cost_scenario",
            "physical_config",
            "case_count",
            "block_result",
            "replay_summary",
            "scientific_result_digest",
            "performance",
            "scientific_use",
            "archive_content_digest",
            "archive_size_bytes",
        },
        "Phase D rehearsal archive",
    )
    expected_header = {
        "schema_version": PHASE_D_REHEARSAL_SCHEMA_VERSION,
        "protocol_version": PHASE_D_REHEARSAL_PROTOCOL_VERSION,
        "campaign": PROSPECTIVE_CAMPAIGN,
        "partition": PHASE_D_REHEARSAL_PARTITION,
        "case_count": len(STAGE3_TRUTH_CONDITIONS),
        "selector_rule": prospective_selector_rule_payload(
            ProspectiveSelectorRule()
        ),
        "cost_scenario": _scenario_payload(PRIMARY_PROSPECTIVE_COST_SCENARIO),
        "scientific_use": "disposable_archive_roundtrip_only",
    }
    if any(item[name] != value for name, value in expected_header.items()):
        raise ValueError("Phase D rehearsal archive header is invalid")
    _validate_revision(item["source_revision"])
    manifest = source_manifest_from_payload(item["source_manifest"])
    if repository_root is not None:
        verify_source_manifest(
            manifest,
            repository_root,
            PHASE_D_NUMERICAL_SOURCE_PATHS,
        ).assert_valid()
    protocol = validate_prospective_phase_d_protocol(item["phase_d_protocol"])
    if (
        protocol["source_revision"] != item["source_revision"]
        or protocol["source_manifest_digest"] != manifest.digest
    ):
        raise ValueError("Phase D rehearsal protocol/source binding is invalid")
    expected_protocol_digest = prospective_phase_d_protocol_digest(
        source_revision=item["source_revision"],
        source_manifest_digest=manifest.digest,
    )
    if item["protocol_digest"] != expected_protocol_digest:
        raise ValueError("Phase D rehearsal protocol digest is invalid")
    partition = CorrectedPartition(
        name=PHASE_D_REHEARSAL_PARTITION,
        block_count=PHASE_D_REHEARSAL_BLOCK_COUNT,
        campaign=PROSPECTIVE_CAMPAIGN,
    )
    physical_config = corrected_partition_config(
        partition,
        CORRECTED_REPLICATION_CONFIG,
    )
    if item["physical_config"] != _physical_config_payload(physical_config):
        raise ValueError("Phase D rehearsal physical configuration is invalid")
    block = item["block_result"]
    if not isinstance(block, Mapping):
        raise ValueError("Phase D rehearsal block evidence is missing")
    _validate_rehearsal_block(block, physical_config=physical_config)
    replay = _replay_summary(block)
    if item["replay_summary"] != replay:
        raise ValueError("Phase D rehearsal replay summary is invalid")
    scientific = {
        "source_revision": item["source_revision"],
        "protocol_digest": item["protocol_digest"],
        "block_result": block,
        "replay_summary": replay,
    }
    if item["scientific_result_digest"] != _digest(
        f"{PHASE_D_HASH_DOMAIN}.rehearsal_scientific_result",
        scientific,
    ):
        raise ValueError("Phase D rehearsal scientific digest is invalid")
    performance = _exact_mapping(
        item["performance"],
        {"wall_seconds", "cpu_seconds", "peak_rss_bytes"},
        "Phase D rehearsal performance",
    )
    _runtime_performance_payload(
        float(performance["wall_seconds"]),
        float(performance["cpu_seconds"]),
        performance["peak_rss_bytes"],
    )
    _validate_sha256(
        "Phase D rehearsal archive digest",
        item["archive_content_digest"],
    )
    material = dict(item)
    material.pop("archive_content_digest")
    material.pop("archive_size_bytes")
    if item["archive_content_digest"] != _digest(
        f"{PHASE_D_HASH_DOMAIN}.rehearsal_archive_content",
        material,
    ):
        raise ValueError("Phase D rehearsal archive content digest is invalid")
    if (
        not isinstance(item["archive_size_bytes"], int)
        or isinstance(item["archive_size_bytes"], bool)
        or item["archive_size_bytes"] != len(_canonical_bytes(item) + b"\n")
    ):
        raise ValueError("Phase D rehearsal archive size is invalid")
    return dict(item)


def format_phase_d_rehearsal_report(payload: Mapping[str, object]) -> str:
    replay = payload["replay_summary"]
    performance = payload["performance"]
    lines = [
        "Prospective operating-decision Phase D disposable rehearsal",
        "===========================================================",
        "",
        f"Campaign: {payload['campaign']}",
        f"Partition: {payload['partition']}",
        f"Source revision: {payload['source_revision']}",
        f"Protocol digest: {payload['protocol_digest']}",
        f"Scientific result digest: {payload['scientific_result_digest']}",
        f"Cases: {replay['case_count']} in one paired disposable block",
        f"Pipeline failures: {replay['pipeline_failure_count']}",
        "N=16 ineligible measurement actions: "
        f"{replay['n16_ineligible_measurement_action_count']}",
        f"N=16 whole-draw failures: {replay['n16_whole_draw_failure_count']}",
        "",
        "N=16 choices:",
    ]
    for family in STAGE3_TRUTH_CONDITIONS:
        lines.append(f"  {family}: {replay['n16_choices'][family]}")
    lines.extend(
        (
            "",
            f"Wall time: {performance['wall_seconds']:.2f} s",
            f"CPU time: {performance['cpu_seconds']:.2f} s",
            f"Peak RSS: {performance['peak_rss_bytes']} bytes",
            "",
            "PASS: canonical save/load validation reproduced stream namespaces, "
            "fit invariants, draw prefixes, cost scorecards, selections, fixed-"
            "policy identities, and saved-before-reveal chronology.",
            "Boundary: disposable transport evidence only. No development, "
            "calibration, or reserved partition was opened.",
        )
    )
    return "\n".join(lines)


def save_phase_d_rehearsal_artifacts(
    result_payload: Mapping[str, object],
    *,
    json_path: Path | str,
    report_path: Path | str,
    hash_path: Path | str,
    repository_root: Path | str,
) -> SavedPhaseDRehearsalArtifacts:
    """Save a rehearsal archive, compact report, and detached hashes once."""

    destinations = tuple(
        Path(value).expanduser().resolve()
        for value in (json_path, report_path, hash_path)
    )
    if len(set(destinations)) != 3:
        raise ValueError("Phase D rehearsal output paths must be distinct")
    existing = tuple(path for path in destinations if path.exists())
    if existing:
        raise FileExistsError(
            "Phase D rehearsal refuses to overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
    sealed, archive = _archive_with_content_seal(result_payload)
    decoded = json.loads(archive)
    validate_phase_d_rehearsal_archive(
        decoded,
        repository_root=repository_root,
    )
    report_bytes = (format_phase_d_rehearsal_report(decoded) + "\n").encode(
        "utf-8"
    )
    json_digest = hashlib.sha256(archive).hexdigest()
    report_digest = hashlib.sha256(report_bytes).hexdigest()
    destinations[0].write_bytes(archive)
    destinations[1].write_bytes(report_bytes)
    destinations[2].write_text(
        f"{json_digest}  {destinations[0].name}\n"
        f"{report_digest}  {destinations[1].name}\n",
        encoding="utf-8",
    )
    return SavedPhaseDRehearsalArtifacts(
        json_path=destinations[0],
        report_path=destinations[1],
        hash_path=destinations[2],
        json_sha256=json_digest,
        report_sha256=report_digest,
        archive_size_bytes=len(archive),
    )


__all__ = [
    "PHASE_D_DRAW_SENSITIVITY_BLOCKS",
    "PHASE_D_GENERAL_CLEARANCE_GRID",
    "PHASE_D_MINIMUM_REDUCTION_GRID",
    "PHASE_D_MINIMUM_UTILITY_GRID",
    "PHASE_D_NUMERICAL_SOURCE_PATHS",
    "PHASE_D_PROTOCOL_VERSION",
    "PHASE_D_REHEARSAL_PARTITION",
    "PHASE_D_RULE_GRID_SIZE",
    "PHASE_D_SENSOR_SCENARIOS",
    "PHASE_D_SINGLE_CANDIDATE_CLEARANCE_GRID",
    "PhaseDSensorScenario",
    "SavedPhaseDRehearsalArtifacts",
    "format_phase_d_rehearsal_report",
    "prospective_phase_d_protocol_digest",
    "prospective_phase_d_protocol_payload",
    "run_phase_d_disposable_rehearsal",
    "save_phase_d_rehearsal_artifacts",
    "validate_phase_d_rehearsal_archive",
    "validate_prospective_phase_d_protocol",
]
