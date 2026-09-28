"""Authenticated Phase D4 N=16-to-N=32 draw-count sensitivity.

Only the predeclared tuning blocks 0, 5, 10, and 15 are generated.  Each N=32
case is required to reproduce the preserved Phase D1 N=16 evidence as its exact
prefix before the frozen Phase D3 rule is compared at the two draw counts.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter, process_time
from typing import Callable, Dict, Mapping, Optional, Sequence

from .operating_decision import POLICY_NAMES, STOP_NOW
from .operating_decision_prospective import (
    ProspectiveSelectorRule,
    prospective_selector_rule_from_payload,
    prospective_selector_rule_payload,
)
from .operating_decision_prospective_costs import (
    PRIMARY_PROSPECTIVE_COST_SCENARIO,
    _scenario_payload,
)
from .operating_decision_prospective_phase_d import (
    PHASE_D_HASH_DOMAIN,
    phase_d_runtime_identity,
    validate_executing_phase_d_runtime,
)
from .operating_decision_prospective_phase_d3_grid import (
    PHASE_D3_ANALYSIS_SOURCE_PATHS,
    validate_phase_d3_grid_analysis,
)
from .operating_decision_prospective_phase_d_tuning import (
    phase_d_tuning_partition,
    validate_phase_d_tuning_archive,
)
from .operating_decision_prospective_pilot import (
    PILOT_GENERATED_DRAW_COUNT,
    PILOT_N32_FOLLOWUP_DRAW_COUNT,
    PROSPECTIVE_CAMPAIGN,
    PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION,
    PilotAcceptanceRule,
    _SELECTION_FAILURE_PREFIX,
    _canonical_bytes,
    _derive_uncertainty_prefix_payload,
    _digest,
    _exact_mapping,
    _failed_n32_followup_block,
    _peak_rss_bytes,
    _run_n32_followup_family,
    _stream_manifest_payload,
    _strict_json_value,
    _summarize_comparisons,
    _validate_authenticated_prefix,
    _validate_complete_uncertainty_payload,
    _validate_corrected_stream_records,
    _validate_revision,
    _validate_sha256,
    compare_pilot_choices,
    pilot_max_unstable_draws,
)
from .operating_decision_provenance import (
    create_source_manifest,
    source_manifest_from_payload,
    source_manifest_payload,
    verify_source_manifest,
)
from .operating_decision_random_streams import RandomStreamRegistry
from .operating_decision_realism import STAGE3_TRUTH_CONDITIONS
from .operating_decision_replication import (
    CORRECTED_REPLICATION_CONFIG,
    corrected_partition_config,
    corrected_truth_for_block,
)


PHASE_D4_SCHEMA_VERSION = 1
PHASE_D4_PROTOCOL_VERSION = "operating_decision_phase_d4_sensitivity_v1"
PHASE_D4_BLOCK_SCHEMA_VERSION = 1
PHASE_D4_BLOCK_PROTOCOL_VERSION = "operating_decision_phase_d4_block_v1"
PHASE_D4_MONITOR_PROTOCOL_VERSION = "phase_d4_process_tree_monitor_preflight_v1"
PHASE_D4_BLOCK_INDICES = (0, 5, 10, 15)
PHASE_D4_CASE_COUNT = len(PHASE_D4_BLOCK_INDICES) * len(STAGE3_TRUTH_CONDITIONS)
PHASE_D4_MINIMUM_AGREEMENT_COUNT = 11
PHASE_D4_MAXIMUM_NORMALIZED_REGRET = 0.05
PHASE_D4_WORKER_COUNT = 4
PHASE_D4_JSON_NAME = "p1_development_draw_sensitivity.json"
PHASE_D4_REPORT_NAME = "p1_development_draw_sensitivity.txt"
PHASE_D4_HASH_NAME = "p1_development_draw_sensitivity.sha256"
PHASE_D4_RESOURCE_NAME = "p1_development_draw_sensitivity.resources.json"
PHASE_D4_BLOCK_DIRECTORY = "blocks"
PHASE_D4_MONITOR_RECORD_PATH = (
    "thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_D4_MONITOR_PREFLIGHT.json"
)
PHASE_D4_RULE_FREEZE_PATH = (
    "thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_D3_RULE.json"
)
PHASE_D4_SOURCE_PATHS = tuple(
    sorted(
        set(PHASE_D3_ANALYSIS_SOURCE_PATHS).union(
            {
                PHASE_D4_MONITOR_RECORD_PATH,
                PHASE_D4_RULE_FREEZE_PATH,
                "thermotwin/reports/operating_decision_prospective_phase_d4.py",
                "thermotwin/studies/operating_decision_prospective_phase_d4_sensitivity.py",
            }
        )
    )
)


@dataclass(frozen=True)
class SavedPhaseD4Artifacts:
    json_path: Path
    report_path: Path
    hash_path: Path
    json_sha256: str
    report_sha256: str
    archive_size_bytes: int


def phase_d4_output_paths(output_directory: Path | str) -> dict:
    root = Path(output_directory).expanduser().resolve()
    return {
        "root": root,
        "blocks": root / PHASE_D4_BLOCK_DIRECTORY,
        "json": root / PHASE_D4_JSON_NAME,
        "report": root / PHASE_D4_REPORT_NAME,
        "hashes": root / PHASE_D4_HASH_NAME,
        "resources": root / PHASE_D4_RESOURCE_NAME,
    }


def phase_d4_block_path(output_directory: Path | str, block: int) -> Path:
    if block not in PHASE_D4_BLOCK_INDICES:
        raise ValueError("Phase D4 block is outside the predeclared subset")
    return phase_d4_output_paths(output_directory)["blocks"] / f"block-{block:03d}.json"


def validate_phase_d4_monitor_preflight(payload: object) -> dict:
    item = _exact_mapping(
        payload,
        {
            "schema_version",
            "protocol_version",
            "date",
            "preflight_source_revision",
            "scientific_use",
            "scientific_evidence_generated",
            "status",
            "attempts",
            "authorization",
        },
        "Phase D4 monitor preflight",
    )
    _validate_revision(item["preflight_source_revision"])
    attempts = item["attempts"]
    authorization = item["authorization"]
    if (
        item["schema_version"] != 1
        or item["protocol_version"] != PHASE_D4_MONITOR_PROTOCOL_VERSION
        or item["scientific_use"] != "permission_and_observability_preflight_only"
        or item["scientific_evidence_generated"] is not False
        or item["status"] != "passed_after_required_process_table_permission"
        or not isinstance(attempts, list)
        or len(attempts) != 2
        or attempts[0].get("result") != "failed_before_scientific_work"
        or attempts[0].get("error_type") != "PermissionError"
        or attempts[1].get("result") != "passed"
        or attempts[1].get("sampling_error") is not None
        or not isinstance(attempts[1].get("process_tree_sample_count"), int)
        or attempts[1]["process_tree_sample_count"] <= 0
        or not isinstance(attempts[1].get("process_tree_peak_rss_bytes"), int)
        or attempts[1]["process_tree_peak_rss_bytes"] <= 0
        or not isinstance(authorization, Mapping)
        or authorization.get("phase_d4_may_run_with_process_table_access") is not True
        or authorization.get("synchronous_recheck_required_immediately_before_computation") is not True
    ):
        raise ValueError("Phase D4 monitor preflight did not pass its declared gate")
    return dict(item)


def _validate_rule_freeze(payload: object, d3: Mapping[str, object]) -> dict:
    if not isinstance(payload, Mapping):
        raise ValueError("Phase D4 rule freeze is malformed")
    item = dict(payload)
    claimed = item.pop("artifact_digest", None)
    _validate_sha256("Phase D3 rule-freeze digest", claimed)
    actual = hashlib.sha256(
        b"thermotwin.phase_d3.provisional_rule_freeze_v1\0"
        + json.dumps(item, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if actual != claimed:
        raise ValueError("Phase D3 rule-freeze digest is invalid")
    if (
        item.get("status") != "provisional_rule_selected_d4_authorized"
        or item.get("selector_rule") != d3["winning_selector_rule"]
        or item.get("winning_grid_values") != d3["winning_grid_values"]
        or item.get("development_objective") != d3["winning_objective"]
        or item.get("phase_d3_result", {}).get("scientific_result_digest")
        != d3["scientific_result_digest"]
        or item.get("authorization", {}).get("phase_d4_authorized") is not True
        or item.get("authorization", {}).get("phase_d4_draw_count")
        != PILOT_N32_FOLLOWUP_DRAW_COUNT
        or item.get("authorization", {}).get("phase_d4_tuning_block_indices")
        != list(PHASE_D4_BLOCK_INDICES)
        or item.get("authorization", {}).get("phase_d5_sensor_generation_authorized")
        is not False
    ):
        raise ValueError("Phase D3 rule freeze does not authorize this D4 design")
    return dict(payload)


def _load_inputs(
    *,
    phase_d1_archive_path: Path | str,
    phase_d3_result_path: Path | str,
    phase_d2_result_path: Path | str,
    rule_freeze_path: Path | str,
    monitor_preflight_path: Path | str,
    repository_root: Path | str,
) -> tuple[dict, dict, dict, dict, bytes, bytes, bytes, bytes]:
    root = Path(repository_root).expanduser().resolve(strict=True)
    d1_path = Path(phase_d1_archive_path).expanduser().resolve(strict=True)
    d3_path = Path(phase_d3_result_path).expanduser().resolve(strict=True)
    d2_path = Path(phase_d2_result_path).expanduser().resolve(strict=True)
    rule_path = Path(rule_freeze_path).expanduser().resolve(strict=True)
    monitor_path = Path(monitor_preflight_path).expanduser().resolve(strict=True)
    d1_raw, d3_raw, rule_raw, monitor_raw = (
        d1_path.read_bytes(),
        d3_path.read_bytes(),
        rule_path.read_bytes(),
        monitor_path.read_bytes(),
    )
    d1 = validate_phase_d_tuning_archive(
        json.loads(d1_raw),
        repository_root=root,
        block_directory=d1_path.parent / "blocks",
    )
    d3 = validate_phase_d3_grid_analysis(
        json.loads(d3_raw),
        phase_d1_archive_path=d1_path,
        phase_d2_result_path=d2_path,
        repository_root=root,
    )
    rule = _validate_rule_freeze(json.loads(rule_raw), d3)
    monitor = validate_phase_d4_monitor_preflight(json.loads(monitor_raw))
    if hashlib.sha256(d1_raw).hexdigest() != d3["input_phase_d1"]["json_sha256"]:
        raise ValueError("Phase D4 D1 input differs from the D3 input")
    return d1, d3, rule, monitor, d1_raw, d3_raw, rule_raw, monitor_raw


def phase_d4_protocol_payload(
    *,
    source_revision: str,
    source_manifest_digest: str,
    input_hashes: Mapping[str, str],
    selector_rule: ProspectiveSelectorRule,
) -> dict:
    _validate_revision(source_revision)
    for name, value in input_hashes.items():
        _validate_sha256(f"Phase D4 {name} hash", value)
    return {
        "schema_version": 1,
        "protocol_version": PHASE_D4_PROTOCOL_VERSION,
        "source_revision": source_revision,
        "source_manifest_digest": source_manifest_digest,
        "campaign": PROSPECTIVE_CAMPAIGN,
        "partition": PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION,
        "block_indices": list(PHASE_D4_BLOCK_INDICES),
        "truth_families": list(STAGE3_TRUTH_CONDITIONS),
        "case_count": PHASE_D4_CASE_COUNT,
        "candidate_draw_count": PILOT_GENERATED_DRAW_COUNT,
        "reference_draw_count": PILOT_N32_FOLLOWUP_DRAW_COUNT,
        "max_unstable_draws_per_source_action": 1,
        "minimum_action_agreement_count": PHASE_D4_MINIMUM_AGREEMENT_COUNT,
        "maximum_changed_choice_normalized_utility_regret": (
            PHASE_D4_MAXIMUM_NORMALIZED_REGRET
        ),
        "all_measurement_actions_required_eligible_at_both_counts": True,
        "zero_pipeline_selection_provenance_or_diagnostic_failures": True,
        "selector_rule": prospective_selector_rule_payload(selector_rule),
        "cost_scenario": _scenario_payload(PRIMARY_PROSPECTIVE_COST_SCENARIO),
        "input_hashes": dict(input_hashes),
        "failure_path": (
            "budget_and_commit_all_remaining_n32_plus_subset_n64_before_execution"
        ),
        "pass_path": "authorize_phase_d5_maps_only",
    }


def _run_phase_d4_block(
    block: int,
    physical_config,
    selector_rule: ProspectiveSelectorRule,
) -> dict:
    wall_started, cpu_started = perf_counter(), process_time()
    partition = phase_d_tuning_partition()
    registry = RandomStreamRegistry()
    truth = corrected_truth_for_block(partition, block, physical_config, registry)
    cases = [
        _run_n32_followup_family(
            truth_condition=family,
            block=block,
            truth=truth,
            partition=partition,
            physical_config=physical_config,
            registry=registry,
            selector_rule=selector_rule,
            cost_scenario=PRIMARY_PROSPECTIVE_COST_SCENARIO,
        )
        for family in STAGE3_TRUTH_CONDITIONS
    ]
    audit = registry.audit()
    audit.assert_clean()
    return {
        "block": block,
        "cases": cases,
        "corrected_random_stream_manifest": _stream_manifest_payload(registry.uses),
        "corrected_random_stream_audit": _strict_json_value(audit),
        "timing": {
            "block_wall_seconds": perf_counter() - wall_started,
            "block_cpu_seconds": process_time() - cpu_started,
            "peak_rss_bytes": _peak_rss_bytes(),
        },
    }


def _run_phase_d4_block_worker(arguments: tuple) -> dict:
    return _run_phase_d4_block(*arguments)


def _d1_cases(d1: Mapping[str, object]) -> dict:
    return {
        (block["block"], case["truth_condition"]): case
        for block in d1["block_results"]
        for case in block["cases"]
        if block["block"] in PHASE_D4_BLOCK_INDICES
    }


def _prefix_by_count(case: Mapping[str, object], draw_count: int) -> Mapping[str, object]:
    matches = tuple(
        value for value in case["primary_prefixes"] if value["draw_count"] == draw_count
    )
    if len(matches) != 1:
        raise ValueError("Phase D4 case lacks a unique draw-count prefix")
    return matches[0]


def _validate_generated_block(
    block_result: Mapping[str, object],
    *,
    d1: Mapping[str, object],
    selector_rule: ProspectiveSelectorRule,
) -> dict:
    item = _exact_mapping(
        block_result,
        {"block", "cases", "corrected_random_stream_manifest", "corrected_random_stream_audit", "timing"},
        "Phase D4 block result",
    )
    block = item["block"]
    if block not in PHASE_D4_BLOCK_INDICES:
        raise ValueError("Phase D4 generated an undeclared block")
    cases = item["cases"]
    if (
        not isinstance(cases, list)
        or [case.get("truth_condition") for case in cases]
        != list(STAGE3_TRUTH_CONDITIONS)
        or any(case.get("block") != block for case in cases)
    ):
        raise ValueError("Phase D4 block lacks its three ordered cases")
    audit = item["corrected_random_stream_audit"]
    if isinstance(audit, Mapping) and audit.get("clean") is False:
        if item["corrected_random_stream_manifest"] != [] or any(
            not case.get("pipeline_failures")
            or case.get("n32_complete_uncertainty_result") is not None
            for case in cases
        ):
            raise ValueError("Phase D4 failed block is not canonical")
        return dict(item)
    _validate_corrected_stream_records(
        item["corrected_random_stream_manifest"],
        audit,
        block=block,
        label="N=32",
        cases=cases,
        campaign=PROSPECTIVE_CAMPAIGN,
        partition=PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION,
    )
    physical_config = corrected_partition_config(
        phase_d_tuning_partition(), CORRECTED_REPLICATION_CONFIG
    )
    parent_cases = _d1_cases(d1)
    for case in cases:
        _exact_mapping(
            case,
            {
                "block",
                "truth_condition",
                "device_token",
                "admissible_source_models",
                "admissible_source_model_count",
                "n32_complete_uncertainty_result",
                "primary_prefixes",
                "pipeline_failures",
                "timing",
            },
            "Phase D4 case",
        )
        parent = parent_cases[(block, case["truth_condition"])]
        failures = case.get("pipeline_failures")
        prefixes = case.get("primary_prefixes")
        if (
            not isinstance(failures, list)
            or not isinstance(prefixes, list)
            or [prefix.get("draw_count") for prefix in prefixes]
            != [PILOT_GENERATED_DRAW_COUNT, PILOT_N32_FOLLOWUP_DRAW_COUNT]
            or any(
                not isinstance(failure, Mapping)
                or set(failure) != {"stage", "error_type", "message"}
                for failure in failures
            )
        ):
            raise ValueError("Phase D4 case failure or prefix record is malformed")
        if failures:
            continue
        if case.get("device_token") != parent["device_token"]:
            raise ValueError("Phase D4 device identity differs from Phase D1")
        complete = _validate_complete_uncertainty_payload(
            case["n32_complete_uncertainty_result"],
            physical_config=physical_config,
            block=block,
            draw_count=PILOT_N32_FOLLOWUP_DRAW_COUNT,
            max_unstable_draws=1,
            campaign=PROSPECTIVE_CAMPAIGN,
            partition=PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION,
        )
        derived_n16 = _derive_uncertainty_prefix_payload(
            complete,
            physical_config=physical_config,
            block=block,
            draw_count=PILOT_GENERATED_DRAW_COUNT,
            max_unstable_draws=1,
            campaign=PROSPECTIVE_CAMPAIGN,
            partition=PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION,
        )
        if derived_n16 != parent["n16_complete_uncertainty_result"]:
            raise ValueError("Phase D4 N=16 prefix differs from preserved Phase D1")
        _validate_authenticated_prefix(
            _prefix_by_count(case, PILOT_GENERATED_DRAW_COUNT),
            complete=derived_n16,
            physical_config=physical_config,
            selector_rule=selector_rule,
            cost_scenario=PRIMARY_PROSPECTIVE_COST_SCENARIO,
        )
        _validate_authenticated_prefix(
            _prefix_by_count(case, PILOT_N32_FOLLOWUP_DRAW_COUNT),
            complete=complete,
            physical_config=physical_config,
            selector_rule=selector_rule,
            cost_scenario=PRIMARY_PROSPECTIVE_COST_SCENARIO,
        )
    return dict(item)


def evaluate_phase_d4_gate(block_results: Sequence[Mapping[str, object]]) -> dict:
    ordered = tuple(sorted(block_results, key=lambda value: int(value["block"])))
    if tuple(item["block"] for item in ordered) != PHASE_D4_BLOCK_INDICES:
        raise ValueError("Phase D4 gate requires the four predeclared blocks")
    comparisons = []
    pipeline_failures = 0
    selection_failures = 0
    unavailable_diagnostics = 0
    ineligible_actions = 0
    whole_draw_failures = 0
    for block in ordered:
        for case in block["cases"]:
            failures = case.get("pipeline_failures", [])
            pipeline_failures += len(failures)
            if failures or case.get("n32_complete_uncertainty_result") is None:
                unavailable_diagnostics += 1
                continue
            n16 = _prefix_by_count(case, PILOT_GENERATED_DRAW_COUNT)
            n32 = _prefix_by_count(case, PILOT_N32_FOLLOWUP_DRAW_COUNT)
            for prefix in (n16, n32):
                if str(prefix.get("choice_token", "")).startswith(_SELECTION_FAILURE_PREFIX):
                    selection_failures += 1
                diagnostics = prefix.get("draw_diagnostics")
                if not isinstance(diagnostics, Mapping) or diagnostics.get("available") is not True:
                    unavailable_diagnostics += 1
                else:
                    whole_draw_failures += int(diagnostics.get("failed_draw_count", 0))
                eligibility = prefix.get("eligibility")
                if not isinstance(eligibility, Mapping):
                    ineligible_actions += len(POLICY_NAMES) - 1
                else:
                    ineligible_actions += sum(
                        eligibility.get(policy, {}).get("eligible") is not True
                        for policy in POLICY_NAMES
                        if policy != STOP_NOW
                    )
            comparisons.append(
                {
                    "block": block["block"],
                    "truth_condition": case["truth_condition"],
                    **compare_pilot_choices(n16, n32),
                }
            )
    rule = PilotAcceptanceRule(
        minimum_action_agreement=PHASE_D4_MINIMUM_AGREEMENT_COUNT / PHASE_D4_CASE_COUNT,
        maximum_normalized_utility_regret=PHASE_D4_MAXIMUM_NORMALIZED_REGRET,
    )
    summary = _summarize_comparisons(comparisons, rule)
    gate_passed = (
        len(comparisons) == PHASE_D4_CASE_COUNT
        and summary["agreement_count"] >= PHASE_D4_MINIMUM_AGREEMENT_COUNT
        and summary["unevaluable_changed_choice_count"] == 0
        and summary["maximum_changed_choice_normalized_utility_regret"]
        <= PHASE_D4_MAXIMUM_NORMALIZED_REGRET
        and pipeline_failures == 0
        and selection_failures == 0
        and unavailable_diagnostics == 0
        and ineligible_actions == 0
    )
    return {
        "rule": {
            "minimum_action_agreement_count": PHASE_D4_MINIMUM_AGREEMENT_COUNT,
            "case_count": PHASE_D4_CASE_COUNT,
            "maximum_changed_choice_normalized_utility_regret": PHASE_D4_MAXIMUM_NORMALIZED_REGRET,
            "all_measurement_actions_required_eligible_at_both_counts": True,
            "zero_pipeline_selection_provenance_or_diagnostic_failures": True,
        },
        "comparisons": comparisons,
        "summary": summary,
        "pipeline_failure_count": pipeline_failures,
        "selection_failure_count": selection_failures,
        "unavailable_draw_diagnostic_count": unavailable_diagnostics,
        "ineligible_action_count": ineligible_actions,
        "whole_draw_failure_count": whole_draw_failures,
        "gate_passed": gate_passed,
        "accepted_draw_count": PILOT_GENERATED_DRAW_COUNT if gate_passed else None,
        "phase_d5_authorized": gate_passed,
        "next_required_action": (
            "execute_phase_d5_maps"
            if gate_passed
            else "commit_budget_for_remaining_n32_and_subset_n64_before_execution"
        ),
        "interpretation": "development_draw_count_stability_only",
    }


def _without_measurements(value: object) -> object:
    measurement_keys = {
        "timing",
        "wall_seconds",
        "cpu_seconds",
        "peak_rss_bytes",
        "case_wall_seconds",
        "case_cpu_seconds",
    }
    if isinstance(value, Mapping):
        return {
            str(key): _without_measurements(item)
            for key, item in value.items()
            if str(key) not in measurement_keys
        }
    if isinstance(value, (list, tuple)):
        return [_without_measurements(item) for item in value]
    return value


def _block_scientific_digest(block: Mapping[str, object]) -> str:
    return _digest(
        f"{PHASE_D_HASH_DOMAIN}.phase_d4_block_v1", _without_measurements(block)
    )


def _block_envelope(
    block: Mapping[str, object],
    *,
    source_revision: str,
    protocol_digest: str,
) -> dict:
    payload = {
        "schema_version": PHASE_D4_BLOCK_SCHEMA_VERSION,
        "protocol_version": PHASE_D4_BLOCK_PROTOCOL_VERSION,
        "source_revision": source_revision,
        "protocol_digest": protocol_digest,
        "block": block["block"],
        "scientific_block_digest": _block_scientific_digest(block),
        "block_result": dict(block),
    }
    payload["archive_content_digest"] = _digest(
        f"{PHASE_D_HASH_DOMAIN}.phase_d4_block_archive_v1", payload
    )
    return payload


def _save_block(path: Path, envelope: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError(f"Phase D4 block output already exists: {path}")
    partial.write_bytes(_canonical_bytes(envelope) + b"\n")
    partial.replace(path)


def _load_block(
    path: Path,
    *,
    source_revision: str,
    protocol_digest: str,
    d1: Mapping[str, object],
    selector_rule: ProspectiveSelectorRule,
) -> dict:
    envelope = _exact_mapping(
        json.loads(path.read_bytes()),
        {"schema_version", "protocol_version", "source_revision", "protocol_digest", "block", "scientific_block_digest", "block_result", "archive_content_digest"},
        "Phase D4 block archive",
    )
    if (
        envelope["schema_version"] != PHASE_D4_BLOCK_SCHEMA_VERSION
        or envelope["protocol_version"] != PHASE_D4_BLOCK_PROTOCOL_VERSION
        or envelope["source_revision"] != source_revision
        or envelope["protocol_digest"] != protocol_digest
        or envelope["block"] not in PHASE_D4_BLOCK_INDICES
    ):
        raise ValueError("Phase D4 block archive header is invalid")
    material = dict(envelope)
    claimed = material.pop("archive_content_digest")
    if claimed != _digest(f"{PHASE_D_HASH_DOMAIN}.phase_d4_block_archive_v1", material):
        raise ValueError("Phase D4 block archive digest is invalid")
    block = _validate_generated_block(
        envelope["block_result"], d1=d1, selector_rule=selector_rule
    )
    if (
        block["block"] != envelope["block"]
        or envelope["scientific_block_digest"] != _block_scientific_digest(block)
    ):
        raise ValueError("Phase D4 block scientific identity is invalid")
    return dict(envelope)


def run_phase_d4_sensitivity(
    *,
    phase_d1_archive_path: Path | str,
    phase_d3_result_path: Path | str,
    phase_d2_result_path: Path | str,
    rule_freeze_path: Path | str,
    monitor_preflight_path: Path | str,
    repository_root: Path | str,
    source_revision: str,
    output_directory: Path | str,
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    validate_executing_phase_d_runtime()
    _validate_revision(source_revision)
    root = Path(repository_root).expanduser().resolve(strict=True)
    d1, d3, rule, monitor, d1_raw, d3_raw, rule_raw, monitor_raw = _load_inputs(
        phase_d1_archive_path=phase_d1_archive_path,
        phase_d3_result_path=phase_d3_result_path,
        phase_d2_result_path=phase_d2_result_path,
        rule_freeze_path=rule_freeze_path,
        monitor_preflight_path=monitor_preflight_path,
        repository_root=root,
    )
    selector_rule = prospective_selector_rule_from_payload(d3["winning_selector_rule"])
    manifest = create_source_manifest(root, PHASE_D4_SOURCE_PATHS)
    input_hashes = {
        "phase_d1_json_sha256": hashlib.sha256(d1_raw).hexdigest(),
        "phase_d3_json_sha256": hashlib.sha256(d3_raw).hexdigest(),
        "rule_freeze_sha256": hashlib.sha256(rule_raw).hexdigest(),
        "monitor_preflight_sha256": hashlib.sha256(monitor_raw).hexdigest(),
    }
    protocol = phase_d4_protocol_payload(
        source_revision=source_revision,
        source_manifest_digest=manifest.digest,
        input_hashes=input_hashes,
        selector_rule=selector_rule,
    )
    protocol_digest = _digest(f"{PHASE_D_HASH_DOMAIN}.phase_d4_protocol_v1", protocol)
    paths = phase_d4_output_paths(output_directory)
    if any(paths[name].exists() for name in ("json", "report", "hashes", "resources")):
        raise FileExistsError("Phase D4 final output already exists")
    physical_config = corrected_partition_config(
        phase_d_tuning_partition(), CORRECTED_REPLICATION_CONFIG
    )
    completed: Dict[int, dict] = {}
    envelopes: Dict[int, dict] = {}
    for block in PHASE_D4_BLOCK_INDICES:
        path = phase_d4_block_path(paths["root"], block)
        if path.exists():
            envelope = _load_block(
                path,
                source_revision=source_revision,
                protocol_digest=protocol_digest,
                d1=d1,
                selector_rule=selector_rule,
            )
            completed[block] = envelope["block_result"]
            envelopes[block] = envelope
    resumed = len(completed)
    missing = tuple(block for block in PHASE_D4_BLOCK_INDICES if block not in completed)
    wall_started, cpu_started = perf_counter(), process_time()
    arguments = tuple((block, physical_config, selector_rule) for block in missing)
    if arguments:
        with ProcessPoolExecutor(max_workers=min(PHASE_D4_WORKER_COUNT, len(arguments))) as executor:
            futures = {
                executor.submit(_run_phase_d4_block_worker, argument): argument[0]
                for argument in arguments
            }
            for future in as_completed(futures):
                block = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = _failed_n32_followup_block(block, error)
                result = _validate_generated_block(
                    result, d1=d1, selector_rule=selector_rule
                )
                envelope = _block_envelope(
                    result,
                    source_revision=source_revision,
                    protocol_digest=protocol_digest,
                )
                _save_block(phase_d4_block_path(paths["root"], block), envelope)
                completed[block] = result
                envelopes[block] = envelope
                if progress is not None:
                    progress(f"Phase D4: validated block {len(completed)}/4")
    ordered = [completed[block] for block in PHASE_D4_BLOCK_INDICES]
    gate = evaluate_phase_d4_gate(ordered)
    records = []
    for block in PHASE_D4_BLOCK_INDICES:
        path = phase_d4_block_path(paths["root"], block)
        raw = path.read_bytes()
        envelope = envelopes[block]
        records.append(
            {
                "block": block,
                "path": f"{PHASE_D4_BLOCK_DIRECTORY}/{path.name}",
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "archive_content_digest": envelope["archive_content_digest"],
                "scientific_block_digest": envelope["scientific_block_digest"],
            }
        )
    payload = {
        "schema_version": PHASE_D4_SCHEMA_VERSION,
        "protocol_version": PHASE_D4_PROTOCOL_VERSION,
        "analysis_source_revision": source_revision,
        "analysis_source_manifest": source_manifest_payload(manifest),
        "runtime_identity": phase_d_runtime_identity(),
        "protocol": protocol,
        "protocol_digest": protocol_digest,
        "input_phase_d1": {
            "path_name": Path(phase_d1_archive_path).name,
            "json_sha256": input_hashes["phase_d1_json_sha256"],
            "scientific_result_digest": d1["scientific_result_digest"],
        },
        "input_phase_d3": {
            "path_name": Path(phase_d3_result_path).name,
            "json_sha256": input_hashes["phase_d3_json_sha256"],
            "scientific_result_digest": d3["scientific_result_digest"],
        },
        "rule_freeze": {
            "json_sha256": input_hashes["rule_freeze_sha256"],
            "artifact_digest": rule["artifact_digest"],
        },
        "monitor_preflight": {
            "json_sha256": input_hashes["monitor_preflight_sha256"],
            "status": monitor["status"],
            "synchronous_recheck_required": True,
        },
        "selector_rule": prospective_selector_rule_payload(selector_rule),
        "cost_scenario": _scenario_payload(PRIMARY_PROSPECTIVE_COST_SCENARIO),
        "block_results": ordered,
        "block_file_records": records,
        "acceptance": gate,
        "performance": {
            "execution_wall_seconds_this_invocation": perf_counter() - wall_started,
            "coordinator_cpu_seconds_this_invocation": process_time() - cpu_started,
            "aggregate_block_cpu_seconds": sum(
                float(block["timing"]["block_cpu_seconds"]) for block in ordered
            ),
            "maximum_recorded_worker_peak_rss_bytes": max(
                int(block["timing"]["peak_rss_bytes"]) for block in ordered
            ),
            "resumed_block_count": resumed,
            "computed_block_count_this_invocation": len(missing),
        },
        "scientific_use": "phase_d4_development_draw_count_stability_only",
    }
    payload["scientific_result_digest"] = _digest(
        f"{PHASE_D_HASH_DOMAIN}.phase_d4_scientific_result_v1",
        {
            "protocol_digest": protocol_digest,
            "blocks": _without_measurements(ordered),
            "acceptance": gate,
        },
    )
    return payload


def validate_phase_d4_archive(
    payload: Mapping[str, object],
    *,
    phase_d1_archive_path: Path | str,
    phase_d3_result_path: Path | str,
    phase_d2_result_path: Path | str,
    rule_freeze_path: Path | str,
    monitor_preflight_path: Path | str,
    repository_root: Path | str,
    block_directory: Path | str,
) -> dict:
    item = _exact_mapping(
        payload,
        {
            "schema_version", "protocol_version", "analysis_source_revision",
            "analysis_source_manifest", "runtime_identity", "protocol",
            "protocol_digest", "input_phase_d1", "input_phase_d3", "rule_freeze",
            "monitor_preflight", "selector_rule", "cost_scenario", "block_results",
            "block_file_records", "acceptance", "performance", "scientific_use",
            "scientific_result_digest", "archive_content_digest", "archive_size_bytes",
        },
        "Phase D4 archive",
    )
    if (
        item["schema_version"] != PHASE_D4_SCHEMA_VERSION
        or item["protocol_version"] != PHASE_D4_PROTOCOL_VERSION
        or item["runtime_identity"] != phase_d_runtime_identity()
        or item["scientific_use"] != "phase_d4_development_draw_count_stability_only"
    ):
        raise ValueError("Phase D4 archive header is invalid")
    d1, d3, rule, monitor, d1_raw, d3_raw, rule_raw, monitor_raw = _load_inputs(
        phase_d1_archive_path=phase_d1_archive_path,
        phase_d3_result_path=phase_d3_result_path,
        phase_d2_result_path=phase_d2_result_path,
        rule_freeze_path=rule_freeze_path,
        monitor_preflight_path=monitor_preflight_path,
        repository_root=repository_root,
    )
    selector_rule = prospective_selector_rule_from_payload(d3["winning_selector_rule"])
    manifest = source_manifest_from_payload(item["analysis_source_manifest"])
    verify_source_manifest(manifest, repository_root, PHASE_D4_SOURCE_PATHS).assert_valid()
    input_hashes = {
        "phase_d1_json_sha256": hashlib.sha256(d1_raw).hexdigest(),
        "phase_d3_json_sha256": hashlib.sha256(d3_raw).hexdigest(),
        "rule_freeze_sha256": hashlib.sha256(rule_raw).hexdigest(),
        "monitor_preflight_sha256": hashlib.sha256(monitor_raw).hexdigest(),
    }
    expected_protocol = phase_d4_protocol_payload(
        source_revision=item["analysis_source_revision"],
        source_manifest_digest=manifest.digest,
        input_hashes=input_hashes,
        selector_rule=selector_rule,
    )
    if (
        item["protocol"] != expected_protocol
        or item["protocol_digest"]
        != _digest(f"{PHASE_D_HASH_DOMAIN}.phase_d4_protocol_v1", expected_protocol)
        or item["selector_rule"] != prospective_selector_rule_payload(selector_rule)
        or item["cost_scenario"] != _scenario_payload(PRIMARY_PROSPECTIVE_COST_SCENARIO)
        or item["rule_freeze"]["artifact_digest"] != rule["artifact_digest"]
        or item["monitor_preflight"]["status"] != monitor["status"]
    ):
        raise ValueError("Phase D4 protocol or input binding is invalid")
    if item["input_phase_d1"] != {
        "path_name": Path(phase_d1_archive_path).name,
        "json_sha256": input_hashes["phase_d1_json_sha256"],
        "scientific_result_digest": d1["scientific_result_digest"],
    } or item["input_phase_d3"] != {
        "path_name": Path(phase_d3_result_path).name,
        "json_sha256": input_hashes["phase_d3_json_sha256"],
        "scientific_result_digest": d3["scientific_result_digest"],
    } or item["rule_freeze"] != {
        "json_sha256": input_hashes["rule_freeze_sha256"],
        "artifact_digest": rule["artifact_digest"],
    } or item["monitor_preflight"] != {
        "json_sha256": input_hashes["monitor_preflight_sha256"],
        "status": monitor["status"],
        "synchronous_recheck_required": True,
    }:
        raise ValueError("Phase D4 retained input identity is invalid")
    blocks = item["block_results"]
    if [block.get("block") for block in blocks] != list(PHASE_D4_BLOCK_INDICES):
        raise ValueError("Phase D4 archive blocks are incomplete")
    for block in blocks:
        _validate_generated_block(block, d1=d1, selector_rule=selector_rule)
    if item["acceptance"] != evaluate_phase_d4_gate(blocks):
        raise ValueError("Phase D4 acceptance does not recompute")
    records = item["block_file_records"]
    if [record.get("block") for record in records] != list(PHASE_D4_BLOCK_INDICES):
        raise ValueError("Phase D4 block-file inventory is incomplete")
    for block, record in zip(PHASE_D4_BLOCK_INDICES, records):
        path = Path(block_directory).expanduser().resolve() / f"block-{block:03d}.json"
        raw = path.read_bytes()
        if (
            record["path"] != f"{PHASE_D4_BLOCK_DIRECTORY}/{path.name}"
            or record["bytes"] != len(raw)
            or record["sha256"] != hashlib.sha256(raw).hexdigest()
        ):
            raise ValueError("Phase D4 block file differs from inventory")
        envelope = _load_block(
            path,
            source_revision=item["analysis_source_revision"],
            protocol_digest=item["protocol_digest"],
            d1=d1,
            selector_rule=selector_rule,
        )
        if (
            envelope["archive_content_digest"] != record["archive_content_digest"]
            or envelope["scientific_block_digest"] != record["scientific_block_digest"]
            or envelope["block_result"] != blocks[list(PHASE_D4_BLOCK_INDICES).index(block)]
        ):
            raise ValueError("Phase D4 block-file content is inconsistent")
    expected_scientific = _digest(
        f"{PHASE_D_HASH_DOMAIN}.phase_d4_scientific_result_v1",
        {
            "protocol_digest": item["protocol_digest"],
            "blocks": _without_measurements(blocks),
            "acceptance": item["acceptance"],
        },
    )
    if item["scientific_result_digest"] != expected_scientific:
        raise ValueError("Phase D4 scientific digest is invalid")
    performance = item["performance"]
    if (
        not isinstance(performance, Mapping)
        or set(performance)
        != {
            "execution_wall_seconds_this_invocation",
            "coordinator_cpu_seconds_this_invocation",
            "aggregate_block_cpu_seconds",
            "maximum_recorded_worker_peak_rss_bytes",
            "resumed_block_count",
            "computed_block_count_this_invocation",
        }
        or any(
            isinstance(performance[name], bool)
            or not isinstance(performance[name], (int, float))
            or float(performance[name]) < 0.0
            for name in (
                "execution_wall_seconds_this_invocation",
                "coordinator_cpu_seconds_this_invocation",
                "aggregate_block_cpu_seconds",
                "maximum_recorded_worker_peak_rss_bytes",
            )
        )
        or performance["resumed_block_count"]
        + performance["computed_block_count_this_invocation"]
        != len(PHASE_D4_BLOCK_INDICES)
    ):
        raise ValueError("Phase D4 performance record is invalid")
    material = dict(item)
    material.pop("archive_content_digest")
    material.pop("archive_size_bytes")
    if item["archive_content_digest"] != _digest(
        f"{PHASE_D_HASH_DOMAIN}.phase_d4_archive_v1", material
    ):
        raise ValueError("Phase D4 archive content digest is invalid")
    if item["archive_size_bytes"] != len(_canonical_bytes(item) + b"\n"):
        raise ValueError("Phase D4 archive size is invalid")
    return dict(item)


def format_phase_d4_report(payload: Mapping[str, object]) -> str:
    acceptance = payload["acceptance"]
    summary = acceptance["summary"]
    return "\n".join(
        (
            "Prospective operating-decision Phase D4 draw sensitivity",
            "=========================================================",
            "",
            f"Analysis source revision: {payload['analysis_source_revision']}",
            f"D4 scientific digest: {payload['scientific_result_digest']}",
            f"Blocks: {list(PHASE_D4_BLOCK_INDICES)}",
            f"Cases: {PHASE_D4_CASE_COUNT}",
            "Comparison: authenticated N=16 prefix versus N=32",
            f"Agreement: {summary['agreement_count']}/{PHASE_D4_CASE_COUNT}",
            "Maximum changed-choice normalized regret: "
            f"{summary['maximum_changed_choice_normalized_utility_regret']}",
            f"Pipeline failures: {acceptance['pipeline_failure_count']}",
            f"Selection failures: {acceptance['selection_failure_count']}",
            f"Unavailable diagnostics: {acceptance['unavailable_draw_diagnostic_count']}",
            f"Ineligible action records: {acceptance['ineligible_action_count']}",
            f"Whole-draw failures: {acceptance['whole_draw_failure_count']}",
            f"Gate passed: {'yes' if acceptance['gate_passed'] else 'no'}",
            f"Next required action: {acceptance['next_required_action']}",
            "",
        )
    )


def save_phase_d4_artifacts(
    payload: Mapping[str, object], output_directory: Path | str
) -> SavedPhaseD4Artifacts:
    paths = phase_d4_output_paths(output_directory)
    for name in ("json", "report", "hashes"):
        if paths[name].exists():
            raise FileExistsError(f"Phase D4 output already exists: {paths[name]}")
    material = dict(payload)
    material["archive_content_digest"] = _digest(
        f"{PHASE_D_HASH_DOMAIN}.phase_d4_archive_v1", material
    )
    material["archive_size_bytes"] = 0
    while True:
        size = len(_canonical_bytes(material) + b"\n")
        if material["archive_size_bytes"] == size:
            break
        material["archive_size_bytes"] = size
    json_bytes = _canonical_bytes(material) + b"\n"
    report_bytes = (format_phase_d4_report(material)).encode()
    json_sha = hashlib.sha256(json_bytes).hexdigest()
    report_sha = hashlib.sha256(report_bytes).hexdigest()
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["json"].write_bytes(json_bytes)
    paths["report"].write_bytes(report_bytes)
    paths["hashes"].write_text(
        f"{json_sha}  {paths['json'].name}\n{report_sha}  {paths['report'].name}\n",
        encoding="utf-8",
    )
    return SavedPhaseD4Artifacts(
        json_path=paths["json"], report_path=paths["report"], hash_path=paths["hashes"],
        json_sha256=json_sha, report_sha256=report_sha, archive_size_bytes=len(json_bytes)
    )


__all__ = [
    "PHASE_D4_BLOCK_INDICES",
    "PHASE_D4_CASE_COUNT",
    "PHASE_D4_SOURCE_PATHS",
    "evaluate_phase_d4_gate",
    "format_phase_d4_report",
    "phase_d4_output_paths",
    "run_phase_d4_sensitivity",
    "save_phase_d4_artifacts",
    "validate_phase_d4_archive",
    "validate_phase_d4_monitor_preflight",
]
