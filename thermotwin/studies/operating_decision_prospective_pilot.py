"""Disposable draw-count pilot for the prospective operating-decision study.

The runner in this module can open only the predeclared disposable pilot.  It
generates sixteen predictive draws once for every case and derives the matched
4-, 8-, and 16-draw analyses from that authenticated result.  Development,
calibration, and reserved namespaces are recorded here for budget planning but
cannot be executed through this interface.

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

from .operating_decision import POLICY_NAMES, STOP_NOW, default_fixed_policies
from .operating_decision_prospective import (
    ProspectiveSelection,
    ProspectiveSelectorRule,
    prospective_selector_rule_payload,
)
from .operating_decision_prospective_costs import (
    PRIMARY_PROSPECTIVE_COST_SCENARIO,
    ProspectiveCostScenario,
    cost_prospective_uncertainty,
    prospective_cost_protocol_digest,
    prospective_costed_scorecard_payload,
    select_costed_prospective_action,
)
from .operating_decision_prospective_random_streams import (
    ProspectiveRandomStreamNamespace,
)
from .operating_decision_prospective_uncertainty import (
    ProspectiveUncertaintyConfig,
    estimate_prospective_action_uncertainty,
    prefix_prospective_uncertainty_result,
    prepare_prospective_acquisition_evidence,
    prospective_uncertainty_protocol_digest,
    prospective_uncertainty_result_payload,
    prospective_uncertainty_summary_payload,
)
from .operating_decision_random_streams import RandomStreamRegistry
from .operating_decision_realism import (
    STAGE3_TRUTH_CONDITIONS,
    OperatingDecisionRealismConfig,
)
from .operating_decision_replication import (
    CORRECTED_REPLICATION_CONFIG,
    CorrectedPartition,
    build_corrected_blinded_case,
    corrected_partition_config,
    corrected_physical_protocol_digest,
    corrected_truth_for_block,
    decide_corrected_blinded_case,
    score_corrected_saved_decision,
)


PROSPECTIVE_PILOT_SCHEMA_VERSION = 1
PROSPECTIVE_PILOT_PROTOCOL_VERSION = (
    "operating_decision_prospective_draw_count_pilot_v1"
)
PROSPECTIVE_CAMPAIGN = "operating_decision_prospective_v1_2026_09"
PROSPECTIVE_PILOT_PARTITION = "p1_disposable_draw_count_pilot"
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

_PIPELINE_FAILURE_PREFIX = "pipeline_failure:"
_SELECTION_FAILURE_PREFIX = "selection_failure:"
_ACTION_PREFIX = "action:"
_HASH_DOMAIN = "thermotwin.prospective_pilot"


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

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_action_agreement <= 1.0:
            raise ValueError("pilot action agreement must lie in (0, 1]")
        if not 0.0 <= self.maximum_normalized_utility_regret <= 1.0:
            raise ValueError("pilot utility regret must lie in [0, 1]")
        if self.reference_draw_count != PILOT_GENERATED_DRAW_COUNT:
            raise ValueError("pilot reference must use the generated 16 draws")


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
        "draw_counts": list(PILOT_DRAW_COUNTS),
        "generated_draw_count": PILOT_GENERATED_DRAW_COUNT,
        "strict_max_unstable_draws_per_source_action": (
            PILOT_STRICT_MAX_UNSTABLE
        ),
        "sensitivity_max_unstable_draws_per_source_action": (
            PILOT_SENSITIVITY_MAX_UNSTABLE
        ),
        "acceptance_rule": asdict(PilotAcceptanceRule()),
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
        "selection": None,
        "uncertainty_summary": None,
        "costed_scorecard": None,
        "pipeline_failure": {
            "stage": stage,
            "error_type": type(error).__name__,
            "message": str(error),
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
        "selection_information_boundary": {
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
        },
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
    """Compare one prefix with N=16 using only the N=16 action utilities."""

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
    recommendation = (
        stability_recommendation if feasibility_passed else None
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
        "stability_recommended_draw_count": stability_recommendation,
        "feasibility_gate_passed": feasibility_passed,
        "recommended_draw_count": recommendation,
        "accepted": recommendation is not None,
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
        "Primary eligibility comparison: maximum unstable draws = 0.",
        "Sensitivity only: N=16 with maximum unstable draws = 1 (15/16 stable).",
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
    lines.extend(
        (
            "",
            "Eligibility sensitivity:",
            f"  strict versus 15/16: agreement "
            f"{sensitivity['agreement_count']}/{sensitivity['case_count']}; "
            f"selected-action eligibility differences "
            f"{sensitivity['selected_action_eligibility_difference_count']}",
            "",
            f"Pipeline failures: {acceptance['pipeline_failure_count']}",
            f"Strict N=16 selection failures: "
            f"{acceptance['n16_selection_failure_count']}",
            f"Strict N=16 ineligible action records: "
            f"{acceptance['n16_ineligible_action_count']}",
            "Stability-only draw-count recommendation: "
            + (
                str(acceptance["stability_recommended_draw_count"])
                if acceptance["stability_recommended_draw_count"] is not None
                else "none"
            ),
            "Recommended draw count: "
            + (
                str(acceptance["recommended_draw_count"])
                if acceptance["recommended_draw_count"] is not None
                else "none"
            ),
            f"Pilot engineering gate: "
            f"{'PASS' if acceptance['accepted'] else 'FAIL'}",
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
    report = format_prospective_pilot_report(payload) + "\n"
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
    if destinations[0].stat().st_size != payload["archive_size_bytes"]:
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
    "PILOT_SENSITIVITY_MAX_UNSTABLE",
    "PILOT_STRICT_MAX_UNSTABLE",
    "PROSPECTIVE_CALIBRATION_PARTITION",
    "PROSPECTIVE_CAMPAIGN",
    "PROSPECTIVE_DEVELOPMENT_CHECK_PARTITION",
    "PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION",
    "PROSPECTIVE_PARTITION_PLAN",
    "PROSPECTIVE_PILOT_PARTITION",
    "PROSPECTIVE_PILOT_PROTOCOL_VERSION",
    "PROSPECTIVE_RESERVED_PARTITION",
    "RESERVED_BLOCK_COUNT",
    "PilotAcceptanceRule",
    "ProspectivePartitionSpec",
    "ProspectivePilotResult",
    "SavedProspectivePilotArtifacts",
    "compare_pilot_choices",
    "evaluate_pilot_acceptance",
    "format_prospective_pilot_report",
    "prospective_partition_plan_payload",
    "prospective_pilot_protocol_digest",
    "prospective_pilot_result_payload",
    "prospective_pilot_scientific_payload",
    "run_prospective_draw_count_pilot",
    "save_prospective_pilot_artifacts",
]
