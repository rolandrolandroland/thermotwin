"""Source-bound execution and archival for Phase D1 development tuning.

This module opens only ``p1_development_tuning``.  It records the complete
nominal N=16 evidence and all four fixed-policy outcomes, but it performs no
offset estimation, rule selection, truth-stratified summary, or other Phase D2+
analysis.  Complete blocks are written atomically and may be resumed only when
their source, protocol, runtime, partition, and content seals all validate.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter, process_time
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

from .operating_decision import POLICY_NAMES, STOP_NOW
from .operating_decision_prospective import (
    ProspectiveSelectorRule,
    prospective_selector_rule_payload,
)
from .operating_decision_prospective_costs import (
    PRIMARY_PROSPECTIVE_COST_SCENARIO,
    ProspectiveCostScenario,
    _scenario_payload,
)
from .operating_decision_prospective_phase_c_freeze import (
    FROZEN_MAX_UNSTABLE_DRAWS_PER_SOURCE_ACTION,
    FROZEN_PROSPECTIVE_DRAW_COUNT,
    FROZEN_PROSPECTIVE_WORKER_COUNT,
)
from .operating_decision_prospective_phase_d import (
    PHASE_D_HASH_DOMAIN,
    PHASE_D_NUMERICAL_SOURCE_PATHS,
    _runtime_performance_payload,
    phase_d_runtime_identity,
    phase_d_scientific_block_payload,
    prospective_phase_d_protocol_digest,
    prospective_phase_d_protocol_payload,
    validate_executing_phase_d_runtime,
    validate_prospective_phase_d_protocol,
)
from .operating_decision_prospective_pilot import (
    DEVELOPMENT_TUNING_BLOCK_COUNT,
    PILOT_DRAW_COUNTS,
    PILOT_GENERATED_DRAW_COUNT,
    PILOT_SENSITIVITY_MAX_UNSTABLE,
    PROSPECTIVE_CAMPAIGN,
    PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION,
    _SELECTION_INFORMATION_BOUNDARY,
    _canonical_bytes,
    _derive_uncertainty_prefix_payload,
    _digest,
    _exact_mapping,
    _failed_block_result,
    _physical_config_payload,
    _prefix_by_count,
    _run_pilot_family,
    _stream_manifest_payload,
    _strict_json_value,
    _validate_authenticated_prefix,
    _validate_canonical_failed_pilot_block,
    _validate_complete_uncertainty_payload,
    _validate_corrected_stream_records,
    _validate_failed_prefix_record,
    _validate_fixed_policy_record,
    _validate_pipeline_failure,
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


PHASE_D_TUNING_SCHEMA_VERSION = 1
PHASE_D_TUNING_PROTOCOL_VERSION = "operating_decision_phase_d1_tuning_v1"
PHASE_D_TUNING_BLOCK_SCHEMA_VERSION = 1
PHASE_D_TUNING_BLOCK_PROTOCOL_VERSION = (
    "operating_decision_phase_d1_tuning_block_v1"
)
PHASE_D_TUNING_JSON_NAME = "p1_development_tuning.json"
PHASE_D_TUNING_REPORT_NAME = "p1_development_tuning.txt"
PHASE_D_TUNING_HASH_NAME = "p1_development_tuning.sha256"
PHASE_D_TUNING_RESOURCE_NAME = "p1_development_tuning.resources.json"
PHASE_D_TUNING_BLOCK_DIRECTORY = "blocks"


@dataclass(frozen=True)
class SavedPhaseDTuningArtifacts:
    json_path: Path
    report_path: Path
    hash_path: Path
    json_sha256: str
    report_sha256: str
    archive_size_bytes: int


def phase_d_tuning_partition() -> CorrectedPartition:
    return CorrectedPartition(
        name=PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION,
        block_count=DEVELOPMENT_TUNING_BLOCK_COUNT,
        campaign=PROSPECTIVE_CAMPAIGN,
    )


def phase_d_tuning_output_paths(output_directory: Path | str) -> dict:
    root = Path(output_directory).expanduser().resolve()
    return {
        "root": root,
        "blocks": root / PHASE_D_TUNING_BLOCK_DIRECTORY,
        "json": root / PHASE_D_TUNING_JSON_NAME,
        "report": root / PHASE_D_TUNING_REPORT_NAME,
        "hashes": root / PHASE_D_TUNING_HASH_NAME,
        "resources": root / PHASE_D_TUNING_RESOURCE_NAME,
    }


def phase_d_tuning_block_path(output_directory: Path | str, block: int) -> Path:
    if not isinstance(block, int) or isinstance(block, bool) or not (
        0 <= block < DEVELOPMENT_TUNING_BLOCK_COUNT
    ):
        raise ValueError("Phase D1 block index is outside the frozen partition")
    return (
        phase_d_tuning_output_paths(output_directory)["blocks"]
        / f"block-{block:03d}.json"
    )


def phase_d_tuning_preflight_payload(
    *,
    repository_root: Path | str,
    source_revision: str,
    output_directory: Path | str,
) -> dict:
    """Return the exact execution binding without creating evidence."""

    _validate_revision(source_revision)
    runtime = validate_executing_phase_d_runtime()
    root = Path(repository_root).expanduser().resolve(strict=True)
    manifest = create_source_manifest(root, PHASE_D_NUMERICAL_SOURCE_PATHS)
    protocol = prospective_phase_d_protocol_payload(
        source_revision=source_revision,
        source_manifest_digest=manifest.digest,
    )
    validated = validate_prospective_phase_d_protocol(protocol)
    if validated["authorization"]["next_partition"] != (
        PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION
    ):
        raise ValueError("Phase D protocol does not authorize the tuning partition")
    paths = phase_d_tuning_output_paths(output_directory)
    if any(paths[name].exists() for name in ("json", "report", "hashes", "resources")):
        raise FileExistsError("Phase D1 final output already exists")
    partials = tuple(paths["blocks"].glob("*.partial")) if paths["blocks"].is_dir() else ()
    if partials:
        raise FileExistsError(
            "Phase D1 has unresolved partial block files: "
            + ", ".join(str(item) for item in partials)
        )
    if paths["blocks"].is_dir():
        expected_names = {
            f"block-{block:03d}.json"
            for block in range(DEVELOPMENT_TUNING_BLOCK_COUNT)
        }
        unexpected = tuple(
            path
            for path in paths["blocks"].iterdir()
            if path.name not in expected_names
        )
        if unexpected:
            raise ValueError(
                "Phase D1 block directory contains unexpected entries: "
                + ", ".join(str(item) for item in unexpected)
            )
    existing_blocks = tuple(
        path
        for block in range(DEVELOPMENT_TUNING_BLOCK_COUNT)
        if (path := phase_d_tuning_block_path(paths["root"], block)).exists()
    )
    if existing_blocks:
        partition = phase_d_tuning_partition()
        physical_config = corrected_partition_config(
            partition,
            CORRECTED_REPLICATION_CONFIG,
        )
        protocol_digest = prospective_phase_d_protocol_digest(
            source_revision=source_revision,
            source_manifest_digest=manifest.digest,
        )
        for path in existing_blocks:
            _load_valid_block(
                path,
                source_revision=source_revision,
                source_manifest_digest=manifest.digest,
                protocol_digest=protocol_digest,
                physical_config=physical_config,
            )
    return {
        "schema_version": 1,
        "protocol_version": "operating_decision_phase_d1_preflight_v1",
        "campaign": PROSPECTIVE_CAMPAIGN,
        "partition": PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION,
        "block_count": DEVELOPMENT_TUNING_BLOCK_COUNT,
        "case_count": DEVELOPMENT_TUNING_BLOCK_COUNT
        * len(STAGE3_TRUTH_CONDITIONS),
        "worker_count": FROZEN_PROSPECTIVE_WORKER_COUNT,
        "draw_count": FROZEN_PROSPECTIVE_DRAW_COUNT,
        "max_unstable_draws_per_source_action": (
            FROZEN_MAX_UNSTABLE_DRAWS_PER_SOURCE_ACTION
        ),
        "source_revision": source_revision,
        "source_manifest_digest": manifest.digest,
        "protocol_digest": prospective_phase_d_protocol_digest(
            source_revision=source_revision,
            source_manifest_digest=manifest.digest,
        ),
        "runtime_identity": runtime,
        "output_directory": str(paths["root"]),
        "validated_existing_block_count": len(existing_blocks),
        "truth_analysis_authorized": False,
        "next_stop": "complete_archive_validated_before_truth_analysis",
    }


def _run_tuning_block(
    block: int,
    physical_config: OperatingDecisionRealismConfig,
    selector_rule: ProspectiveSelectorRule,
    cost_scenario: ProspectiveCostScenario,
) -> dict:
    wall_started = perf_counter()
    cpu_started = process_time()
    partition = phase_d_tuning_partition()
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
            cost_scenario=cost_scenario,
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


def _run_tuning_block_worker(arguments: tuple) -> dict:
    return _run_tuning_block(*arguments)


def _peak_rss_bytes() -> int:
    from .operating_decision_prospective_pilot import _peak_rss_bytes as value

    return value()


def _validate_tuning_case(
    case: Mapping[str, object],
    *,
    block: int,
    physical_config: OperatingDecisionRealismConfig,
    selector_rule: ProspectiveSelectorRule,
    cost_scenario: ProspectiveCostScenario,
) -> None:
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
            "primary_prefixes",
            "n16_max_unstable_0_sensitivity",
            "selected_policy_at_n16_primary",
            "selected_policy_outcome",
            "fixed_policy_results",
            "pipeline_failures",
            "timing",
        },
        "Phase D1 tuning case",
    )
    if (
        case["block"] != block
        or case["truth_condition"] not in STAGE3_TRUTH_CONDITIONS
        or not isinstance(case.get("device_token"), str)
        or not case.get("device_token")
        or case.get("truth_revealed_only_after_saves") is None
        or case.get("selection_information_boundary")
        != _SELECTION_INFORMATION_BOUNDARY
        or not isinstance(case.get("pipeline_failures"), list)
    ):
        raise ValueError("Phase D1 tuning case evidence is incomplete")
    failures = tuple(
        _validate_pipeline_failure(value, "Phase D1 tuning case pipeline failure")
        for value in case["pipeline_failures"]
    )
    fixed = case["fixed_policy_results"]
    if not isinstance(fixed, Mapping) or set(fixed) != set(POLICY_NAMES):
        raise ValueError("Phase D1 tuning fixed-policy evidence is incomplete")
    for policy_name, record in fixed.items():
        if not isinstance(record, Mapping) or set(record) not in (
            {"case", "saved_before_reveal", "post_reveal_score"},
            {"case", "saved_before_reveal", "post_reveal_score", "failure"},
        ):
            raise ValueError("Phase D1 tuning fixed-policy result is malformed")
        _validate_fixed_policy_record(
            record,
            policy_name=policy_name,
            truth_condition=str(case["truth_condition"]),
            block=block,
            device_token=str(case["device_token"]),
            physical_config=physical_config,
            campaign=PROSPECTIVE_CAMPAIGN,
            partition_name=PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION,
            partition_block_count=DEVELOPMENT_TUNING_BLOCK_COUNT,
        )
        fixed_failure = record.get("failure")
        if fixed_failure == "decision_not_saved" and not any(
            failure["stage"] == f"fixed_policy_save:{policy_name}"
            for failure in failures
        ):
            raise ValueError("Phase D1 failed fixed-policy save is unauthenticated")
        if isinstance(fixed_failure, Mapping) and not any(
            failure["stage"] == f"fixed_policy_reveal:{policy_name}"
            and failure["error_type"] == fixed_failure.get("error_type")
            and failure["message"] == fixed_failure.get("message")
            for failure in failures
        ):
            raise ValueError("Phase D1 failed fixed-policy reveal is unauthenticated")

    complete_payload = case["n16_complete_uncertainty_result"]
    if complete_payload is None:
        matching = tuple(
            failure
            for failure in failures
            if failure["stage"] == "n16_acquisition_or_scoring"
        )
        if len(matching) != 1:
            raise ValueError("Phase D1 missing N=16 evidence is not canonical")
        for prefix, draw_count in zip(case["primary_prefixes"], PILOT_DRAW_COUNTS):
            failed = _validate_failed_prefix_record(
                prefix,
                draw_count=draw_count,
                max_unstable_draws=pilot_max_unstable_draws(draw_count),
            )
            if failed["pipeline_failure"] != matching[0]:
                raise ValueError("Phase D1 failed prefix does not match its failure")
        sensitivity = _validate_failed_prefix_record(
            case["n16_max_unstable_0_sensitivity"],
            draw_count=PILOT_GENERATED_DRAW_COUNT,
            max_unstable_draws=PILOT_SENSITIVITY_MAX_UNSTABLE,
        )
        if sensitivity["pipeline_failure"] != matching[0]:
            raise ValueError("Phase D1 failed sensitivity does not match its failure")
        if any(
            case[name] is not None
            for name in (
                "admissible_source_models",
                "admissible_source_model_count",
                "selected_policy_at_n16_primary",
                "selected_policy_outcome",
            )
        ):
            raise ValueError("Phase D1 failed N=16 case is inconsistent")
        return

    complete = _validate_complete_uncertainty_payload(
        complete_payload,
        physical_config=physical_config,
        block=block,
        draw_count=FROZEN_PROSPECTIVE_DRAW_COUNT,
        max_unstable_draws=FROZEN_MAX_UNSTABLE_DRAWS_PER_SOURCE_ACTION,
        campaign=PROSPECTIVE_CAMPAIGN,
        partition=PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION,
    )
    sources = list(
        complete["acquisition_evidence"]["snapshot"]["admissible_candidate_models"]
    )
    if (
        case["admissible_source_models"] != sources
        or case["admissible_source_model_count"] != len(sources)
    ):
        raise ValueError("Phase D1 source-model record is inconsistent")
    prefixes = case["primary_prefixes"]
    if not isinstance(prefixes, list) or [
        value.get("draw_count") for value in prefixes
    ] != list(PILOT_DRAW_COUNTS):
        raise ValueError("Phase D1 tuning prefix evidence is incomplete")
    for prefix, draw_count in zip(prefixes, PILOT_DRAW_COUNTS):
        expected = _derive_uncertainty_prefix_payload(
            complete,
            physical_config=physical_config,
            block=block,
            draw_count=draw_count,
            max_unstable_draws=pilot_max_unstable_draws(draw_count),
            campaign=PROSPECTIVE_CAMPAIGN,
            partition=PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION,
        )
        if isinstance(prefix, Mapping) and "pipeline_failure" in prefix:
            failed = _validate_failed_prefix_record(
                prefix,
                draw_count=draw_count,
                max_unstable_draws=pilot_max_unstable_draws(draw_count),
            )
            failure = failed["pipeline_failure"]
            if not any(
                item["stage"] == f"{failure['stage']}_n{draw_count}"
                and item["error_type"] == failure["error_type"]
                and item["message"] == failure["message"]
                for item in failures
            ):
                raise ValueError("Phase D1 failed prefix has no pipeline failure")
        else:
            _validate_authenticated_prefix(
                prefix,
                complete=expected,
                physical_config=physical_config,
                selector_rule=selector_rule,
                cost_scenario=cost_scenario,
            )
    sensitivity_complete = _derive_uncertainty_prefix_payload(
        complete,
        physical_config=physical_config,
        block=block,
        draw_count=PILOT_GENERATED_DRAW_COUNT,
        max_unstable_draws=PILOT_SENSITIVITY_MAX_UNSTABLE,
        campaign=PROSPECTIVE_CAMPAIGN,
        partition=PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION,
    )
    sensitivity = case["n16_max_unstable_0_sensitivity"]
    if isinstance(sensitivity, Mapping) and "pipeline_failure" in sensitivity:
        failed = _validate_failed_prefix_record(
            sensitivity,
            draw_count=PILOT_GENERATED_DRAW_COUNT,
            max_unstable_draws=PILOT_SENSITIVITY_MAX_UNSTABLE,
        )
        failure = failed["pipeline_failure"]
        if not any(
            item["stage"] == f"{failure['stage']}_n16_max_unstable_0"
            and item["error_type"] == failure["error_type"]
            and item["message"] == failure["message"]
            for item in failures
        ):
            raise ValueError("Phase D1 failed sensitivity has no pipeline failure")
    else:
        _validate_authenticated_prefix(
            sensitivity,
            complete=sensitivity_complete,
            physical_config=physical_config,
            selector_rule=selector_rule,
            cost_scenario=cost_scenario,
        )
    reference = _prefix_by_count(case, FROZEN_PROSPECTIVE_DRAW_COUNT)
    selected = (
        None
        if "pipeline_failure" in reference
        else reference["selection"]["selected_policy"]
    )
    if (
        case["selected_policy_at_n16_primary"] != selected
        or case["selected_policy_outcome"]
        != (None if selected is None else fixed[selected])
    ):
        raise ValueError("Phase D1 selected outcome is inconsistent")


def validate_phase_d_tuning_block(
    block_result: Mapping[str, object],
    *,
    block: int,
    physical_config: OperatingDecisionRealismConfig,
    selector_rule: ProspectiveSelectorRule = ProspectiveSelectorRule(),
    cost_scenario: ProspectiveCostScenario = PRIMARY_PROSPECTIVE_COST_SCENARIO,
) -> dict:
    item = _exact_mapping(
        block_result,
        {
            "block",
            "cases",
            "corrected_random_stream_manifest",
            "corrected_random_stream_audit",
            "timing",
        },
        "Phase D1 tuning block",
    )
    cases = item["cases"]
    if (
        item["block"] != block
        or not isinstance(cases, list)
        or len(cases) != len(STAGE3_TRUTH_CONDITIONS)
        or [case.get("truth_condition") for case in cases]
        != list(STAGE3_TRUTH_CONDITIONS)
    ):
        raise ValueError("Phase D1 tuning case matrix is incomplete")
    audit = item["corrected_random_stream_audit"]
    if isinstance(audit, Mapping) and set(audit) == {
        "clean",
        "not_run_due_to_block_failure",
    }:
        _validate_canonical_failed_pilot_block(item, block)
        return dict(item)
    _validate_corrected_stream_records(
        item["corrected_random_stream_manifest"],
        audit,
        block=block,
        label="parent pilot",
        campaign=PROSPECTIVE_CAMPAIGN,
        partition=PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION,
    )
    for case in cases:
        _validate_tuning_case(
            case,
            block=block,
            physical_config=physical_config,
            selector_rule=selector_rule,
            cost_scenario=cost_scenario,
        )
    return dict(item)


def _sealed_payload(
    payload: Mapping[str, object],
    *,
    domain: str,
) -> Tuple[dict, bytes]:
    sealed = dict(payload)
    sealed["archive_content_digest"] = _digest(domain, payload)
    sealed["archive_size_bytes"] = 0
    for _ in range(16):
        archive = _canonical_bytes(sealed) + b"\n"
        size = len(archive)
        if sealed["archive_size_bytes"] == size:
            return sealed, archive
        sealed["archive_size_bytes"] = size
    raise RuntimeError("Phase D1 archive-size fixed point did not converge")


def phase_d_tuning_block_archive(
    block_result: Mapping[str, object],
    *,
    source_revision: str,
    source_manifest_digest: str,
    protocol_digest: str,
    physical_config: OperatingDecisionRealismConfig,
    selector_rule: ProspectiveSelectorRule = ProspectiveSelectorRule(),
    cost_scenario: ProspectiveCostScenario = PRIMARY_PROSPECTIVE_COST_SCENARIO,
) -> Tuple[dict, bytes]:
    block = int(block_result["block"])
    validate_phase_d_tuning_block(
        block_result,
        block=block,
        physical_config=physical_config,
        selector_rule=selector_rule,
        cost_scenario=cost_scenario,
    )
    payload = {
        "schema_version": PHASE_D_TUNING_BLOCK_SCHEMA_VERSION,
        "protocol_version": PHASE_D_TUNING_BLOCK_PROTOCOL_VERSION,
        "campaign": PROSPECTIVE_CAMPAIGN,
        "partition": PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION,
        "source_revision": source_revision,
        "source_manifest_digest": source_manifest_digest,
        "protocol_digest": protocol_digest,
        "runtime_identity": phase_d_runtime_identity(),
        "selector_rule": prospective_selector_rule_payload(selector_rule),
        "cost_scenario": _scenario_payload(cost_scenario),
        "physical_config": _physical_config_payload(physical_config),
        "block": block,
        "block_result": dict(block_result),
        "scientific_block_digest": _digest(
            f"{PHASE_D_HASH_DOMAIN}.tuning_scientific_block_v1",
            phase_d_scientific_block_payload(block_result),
        ),
        "scientific_use": "phase_d1_nominal_tuning_evidence_no_truth_analysis",
    }
    return _sealed_payload(
        payload,
        domain=f"{PHASE_D_HASH_DOMAIN}.tuning_block_archive_content_v1",
    )


def validate_phase_d_tuning_block_archive(
    payload: Mapping[str, object],
    *,
    source_revision: str,
    source_manifest_digest: str,
    protocol_digest: str,
    physical_config: OperatingDecisionRealismConfig,
) -> dict:
    item = _exact_mapping(
        payload,
        {
            "schema_version",
            "protocol_version",
            "campaign",
            "partition",
            "source_revision",
            "source_manifest_digest",
            "protocol_digest",
            "runtime_identity",
            "selector_rule",
            "cost_scenario",
            "physical_config",
            "block",
            "block_result",
            "scientific_block_digest",
            "scientific_use",
            "archive_content_digest",
            "archive_size_bytes",
        },
        "Phase D1 tuning block archive",
    )
    expected = {
        "schema_version": PHASE_D_TUNING_BLOCK_SCHEMA_VERSION,
        "protocol_version": PHASE_D_TUNING_BLOCK_PROTOCOL_VERSION,
        "campaign": PROSPECTIVE_CAMPAIGN,
        "partition": PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION,
        "source_revision": source_revision,
        "source_manifest_digest": source_manifest_digest,
        "protocol_digest": protocol_digest,
        "runtime_identity": phase_d_runtime_identity(),
        "selector_rule": prospective_selector_rule_payload(
            ProspectiveSelectorRule()
        ),
        "cost_scenario": _scenario_payload(PRIMARY_PROSPECTIVE_COST_SCENARIO),
        "physical_config": _physical_config_payload(physical_config),
        "scientific_use": "phase_d1_nominal_tuning_evidence_no_truth_analysis",
    }
    if any(item[name] != value for name, value in expected.items()):
        raise ValueError("Phase D1 tuning block header is invalid")
    block = item["block"]
    if not isinstance(block, int) or isinstance(block, bool):
        raise ValueError("Phase D1 tuning block index is invalid")
    validate_phase_d_tuning_block(
        item["block_result"],
        block=block,
        physical_config=physical_config,
    )
    expected_scientific = _digest(
        f"{PHASE_D_HASH_DOMAIN}.tuning_scientific_block_v1",
        phase_d_scientific_block_payload(item["block_result"]),
    )
    if item["scientific_block_digest"] != expected_scientific:
        raise ValueError("Phase D1 tuning scientific block digest is invalid")
    _validate_sha256("Phase D1 block archive digest", item["archive_content_digest"])
    material = dict(item)
    material.pop("archive_content_digest")
    material.pop("archive_size_bytes")
    if item["archive_content_digest"] != _digest(
        f"{PHASE_D_HASH_DOMAIN}.tuning_block_archive_content_v1",
        material,
    ):
        raise ValueError("Phase D1 tuning block content digest is invalid")
    if (
        not isinstance(item["archive_size_bytes"], int)
        or isinstance(item["archive_size_bytes"], bool)
        or item["archive_size_bytes"] != len(_canonical_bytes(item) + b"\n")
    ):
        raise ValueError("Phase D1 tuning block archive size is invalid")
    return dict(item)


def _atomic_save_block(
    path: Path,
    block_result: Mapping[str, object],
    *,
    source_revision: str,
    source_manifest_digest: str,
    protocol_digest: str,
    physical_config: OperatingDecisionRealismConfig,
) -> dict:
    if path.exists():
        raise FileExistsError(f"Phase D1 refuses to overwrite block: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"Phase D1 unresolved partial block exists: {partial}")
    _, archive = phase_d_tuning_block_archive(
        block_result,
        source_revision=source_revision,
        source_manifest_digest=source_manifest_digest,
        protocol_digest=protocol_digest,
        physical_config=physical_config,
    )
    decoded = json.loads(archive)
    validate_phase_d_tuning_block_archive(
        decoded,
        source_revision=source_revision,
        source_manifest_digest=source_manifest_digest,
        protocol_digest=protocol_digest,
        physical_config=physical_config,
    )
    try:
        with partial.open("xb") as stream:
            stream.write(archive)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    return decoded


def _load_valid_block(
    path: Path,
    *,
    source_revision: str,
    source_manifest_digest: str,
    protocol_digest: str,
    physical_config: OperatingDecisionRealismConfig,
) -> dict:
    raw = path.read_bytes()
    decoded = json.loads(raw)
    validated = validate_phase_d_tuning_block_archive(
        decoded,
        source_revision=source_revision,
        source_manifest_digest=source_manifest_digest,
        protocol_digest=protocol_digest,
        physical_config=physical_config,
    )
    if raw != _canonical_bytes(validated) + b"\n":
        raise ValueError(f"Phase D1 block file is not canonical: {path}")
    return validated


def phase_d_tuning_scientific_payload(payload: Mapping[str, object]) -> dict:
    return {
        "schema_version": PHASE_D_TUNING_SCHEMA_VERSION,
        "source_revision": payload["source_revision"],
        "source_manifest_digest": payload["source_manifest"]["digest"],
        "protocol_digest": payload["protocol_digest"],
        "runtime_identity": payload["runtime_identity"],
        "physical_config": payload["physical_config"],
        "selector_rule": payload["selector_rule"],
        "cost_scenario": payload["cost_scenario"],
        "block_results": [
            phase_d_scientific_block_payload(item)
            for item in payload["block_results"]
        ],
        "truth_analysis_status": payload["truth_analysis_status"],
    }


def run_phase_d_development_tuning(
    *,
    repository_root: Path | str,
    source_revision: str,
    output_directory: Path | str,
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """Generate or resume all twenty nominal blocks without analysing truth."""

    preflight = phase_d_tuning_preflight_payload(
        repository_root=repository_root,
        source_revision=source_revision,
        output_directory=output_directory,
    )
    root = Path(repository_root).expanduser().resolve(strict=True)
    paths = phase_d_tuning_output_paths(output_directory)
    manifest = create_source_manifest(root, PHASE_D_NUMERICAL_SOURCE_PATHS)
    protocol = prospective_phase_d_protocol_payload(
        source_revision=source_revision,
        source_manifest_digest=manifest.digest,
    )
    protocol_digest = prospective_phase_d_protocol_digest(
        source_revision=source_revision,
        source_manifest_digest=manifest.digest,
    )
    partition = phase_d_tuning_partition()
    physical_config = corrected_partition_config(
        partition,
        CORRECTED_REPLICATION_CONFIG,
    )
    selector_rule = ProspectiveSelectorRule()
    cost_scenario = PRIMARY_PROSPECTIVE_COST_SCENARIO
    completed: Dict[int, dict] = {}
    block_files: Dict[int, dict] = {}
    for block in range(DEVELOPMENT_TUNING_BLOCK_COUNT):
        path = phase_d_tuning_block_path(paths["root"], block)
        if not path.exists():
            continue
        envelope = _load_valid_block(
            path,
            source_revision=source_revision,
            source_manifest_digest=manifest.digest,
            protocol_digest=protocol_digest,
            physical_config=physical_config,
        )
        completed[block] = envelope["block_result"]
        block_files[block] = envelope
    resumed_count = len(completed)
    missing = tuple(
        block for block in range(DEVELOPMENT_TUNING_BLOCK_COUNT) if block not in completed
    )
    wall_started = perf_counter()
    cpu_started = process_time()
    arguments = tuple(
        (block, physical_config, selector_rule, cost_scenario) for block in missing
    )
    if arguments:
        with ProcessPoolExecutor(
            max_workers=min(FROZEN_PROSPECTIVE_WORKER_COUNT, len(arguments))
        ) as executor:
            futures = {
                executor.submit(_run_tuning_block_worker, argument): argument[0]
                for argument in arguments
            }
            for future in as_completed(futures):
                block = futures[future]
                try:
                    block_result = future.result()
                except Exception as error:
                    block_result = _failed_block_result(block, error)
                path = phase_d_tuning_block_path(paths["root"], block)
                envelope = _atomic_save_block(
                    path,
                    block_result,
                    source_revision=source_revision,
                    source_manifest_digest=manifest.digest,
                    protocol_digest=protocol_digest,
                    physical_config=physical_config,
                )
                completed[block] = envelope["block_result"]
                block_files[block] = envelope
                if progress is not None:
                    progress(
                        f"{PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION}: "
                        f"validated block {len(completed)}/"
                        f"{DEVELOPMENT_TUNING_BLOCK_COUNT}"
                    )
    ordered = [completed[block] for block in range(DEVELOPMENT_TUNING_BLOCK_COUNT)]
    block_file_records = []
    for block in range(DEVELOPMENT_TUNING_BLOCK_COUNT):
        path = phase_d_tuning_block_path(paths["root"], block)
        raw = path.read_bytes()
        envelope = block_files[block]
        block_file_records.append(
            {
                "block": block,
                "path": f"{PHASE_D_TUNING_BLOCK_DIRECTORY}/{path.name}",
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "archive_content_digest": envelope["archive_content_digest"],
                "scientific_block_digest": envelope["scientific_block_digest"],
            }
        )
    performance = {
        "execution_wall_seconds_this_invocation": perf_counter() - wall_started,
        "coordinator_cpu_seconds_this_invocation": process_time() - cpu_started,
        "aggregate_block_cpu_seconds": sum(
            float(item["timing"]["block_cpu_seconds"]) for item in ordered
        ),
        "maximum_recorded_worker_peak_rss_bytes": max(
            int(item["timing"]["peak_rss_bytes"]) for item in ordered
        ),
        "resumed_block_count": resumed_count,
        "computed_block_count_this_invocation": len(missing),
    }
    payload = {
        "schema_version": PHASE_D_TUNING_SCHEMA_VERSION,
        "protocol_version": PHASE_D_TUNING_PROTOCOL_VERSION,
        "campaign": PROSPECTIVE_CAMPAIGN,
        "partition": PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION,
        "source_revision": source_revision,
        "runtime_identity": phase_d_runtime_identity(),
        "source_manifest": source_manifest_payload(manifest),
        "phase_d_protocol": protocol,
        "protocol_digest": protocol_digest,
        "selector_rule": prospective_selector_rule_payload(selector_rule),
        "cost_scenario": _scenario_payload(cost_scenario),
        "physical_config": _physical_config_payload(physical_config),
        "worker_count": FROZEN_PROSPECTIVE_WORKER_COUNT,
        "block_count": DEVELOPMENT_TUNING_BLOCK_COUNT,
        "case_count": DEVELOPMENT_TUNING_BLOCK_COUNT
        * len(STAGE3_TRUTH_CONDITIONS),
        "block_results": ordered,
        "block_file_records": block_file_records,
        "performance": performance,
        "scientific_use": "phase_d1_nominal_development_tuning_evidence",
        "truth_analysis_status": "not_performed_stop_before_phase_d2",
        "preflight": preflight,
    }
    payload["scientific_result_digest"] = _digest(
        f"{PHASE_D_HASH_DOMAIN}.tuning_scientific_result_v1",
        phase_d_tuning_scientific_payload(payload),
    )
    return payload


def validate_phase_d_tuning_archive(
    payload: Mapping[str, object],
    *,
    repository_root: Path | str | None = None,
    block_directory: Path | str | None = None,
) -> dict:
    item = _exact_mapping(
        payload,
        {
            "schema_version",
            "protocol_version",
            "campaign",
            "partition",
            "source_revision",
            "runtime_identity",
            "source_manifest",
            "phase_d_protocol",
            "protocol_digest",
            "selector_rule",
            "cost_scenario",
            "physical_config",
            "worker_count",
            "block_count",
            "case_count",
            "block_results",
            "block_file_records",
            "performance",
            "scientific_use",
            "truth_analysis_status",
            "preflight",
            "scientific_result_digest",
            "archive_content_digest",
            "archive_size_bytes",
        },
        "Phase D1 tuning archive",
    )
    expected_header = {
        "schema_version": PHASE_D_TUNING_SCHEMA_VERSION,
        "protocol_version": PHASE_D_TUNING_PROTOCOL_VERSION,
        "campaign": PROSPECTIVE_CAMPAIGN,
        "partition": PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION,
        "runtime_identity": phase_d_runtime_identity(),
        "selector_rule": prospective_selector_rule_payload(
            ProspectiveSelectorRule()
        ),
        "cost_scenario": _scenario_payload(PRIMARY_PROSPECTIVE_COST_SCENARIO),
        "worker_count": FROZEN_PROSPECTIVE_WORKER_COUNT,
        "block_count": DEVELOPMENT_TUNING_BLOCK_COUNT,
        "case_count": DEVELOPMENT_TUNING_BLOCK_COUNT
        * len(STAGE3_TRUTH_CONDITIONS),
        "scientific_use": "phase_d1_nominal_development_tuning_evidence",
        "truth_analysis_status": "not_performed_stop_before_phase_d2",
    }
    if any(item[name] != value for name, value in expected_header.items()):
        raise ValueError("Phase D1 tuning archive header is invalid")
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
        raise ValueError("Phase D1 protocol/source binding is invalid")
    expected_protocol_digest = prospective_phase_d_protocol_digest(
        source_revision=item["source_revision"],
        source_manifest_digest=manifest.digest,
    )
    if item["protocol_digest"] != expected_protocol_digest:
        raise ValueError("Phase D1 protocol digest is invalid")
    partition = phase_d_tuning_partition()
    physical_config = corrected_partition_config(
        partition,
        CORRECTED_REPLICATION_CONFIG,
    )
    if item["physical_config"] != _physical_config_payload(physical_config):
        raise ValueError("Phase D1 physical configuration is invalid")
    preflight = _exact_mapping(
        item["preflight"],
        {
            "schema_version",
            "protocol_version",
            "campaign",
            "partition",
            "block_count",
            "case_count",
            "worker_count",
            "draw_count",
            "max_unstable_draws_per_source_action",
            "source_revision",
            "source_manifest_digest",
            "protocol_digest",
            "runtime_identity",
            "output_directory",
            "validated_existing_block_count",
            "truth_analysis_authorized",
            "next_stop",
        },
        "Phase D1 preflight record",
    )
    if (
        preflight["schema_version"] != 1
        or preflight["protocol_version"]
        != "operating_decision_phase_d1_preflight_v1"
        or preflight["campaign"] != PROSPECTIVE_CAMPAIGN
        or preflight["partition"] != PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION
        or preflight["block_count"] != DEVELOPMENT_TUNING_BLOCK_COUNT
        or preflight["case_count"]
        != DEVELOPMENT_TUNING_BLOCK_COUNT * len(STAGE3_TRUTH_CONDITIONS)
        or preflight["worker_count"] != FROZEN_PROSPECTIVE_WORKER_COUNT
        or preflight["draw_count"] != FROZEN_PROSPECTIVE_DRAW_COUNT
        or preflight["max_unstable_draws_per_source_action"]
        != FROZEN_MAX_UNSTABLE_DRAWS_PER_SOURCE_ACTION
        or preflight["source_revision"] != item["source_revision"]
        or preflight["source_manifest_digest"] != manifest.digest
        or preflight["protocol_digest"] != item["protocol_digest"]
        or preflight["runtime_identity"] != item["runtime_identity"]
        or not isinstance(preflight["output_directory"], str)
        or not preflight["output_directory"]
        or not isinstance(preflight["validated_existing_block_count"], int)
        or isinstance(preflight["validated_existing_block_count"], bool)
        or not 0
        <= preflight["validated_existing_block_count"]
        <= DEVELOPMENT_TUNING_BLOCK_COUNT
        or preflight["truth_analysis_authorized"] is not False
        or preflight["next_stop"]
        != "complete_archive_validated_before_truth_analysis"
    ):
        raise ValueError("Phase D1 preflight record is inconsistent")
    blocks = item["block_results"]
    if (
        not isinstance(blocks, list)
        or [block.get("block") for block in blocks]
        != list(range(DEVELOPMENT_TUNING_BLOCK_COUNT))
    ):
        raise ValueError("Phase D1 archive does not contain all ordered blocks")
    for block_index, block in enumerate(blocks):
        validate_phase_d_tuning_block(
            block,
            block=block_index,
            physical_config=physical_config,
        )
    records = item["block_file_records"]
    if (
        not isinstance(records, list)
        or [record.get("block") for record in records]
        != list(range(DEVELOPMENT_TUNING_BLOCK_COUNT))
    ):
        raise ValueError("Phase D1 block-file inventory is incomplete")
    for block, record in enumerate(records):
        _exact_mapping(
            record,
            {
                "block",
                "path",
                "bytes",
                "sha256",
                "archive_content_digest",
                "scientific_block_digest",
            },
            "Phase D1 block-file record",
        )
        _validate_sha256("Phase D1 block file SHA-256", record["sha256"])
        _validate_sha256(
            "Phase D1 block file content digest",
            record["archive_content_digest"],
        )
        _validate_sha256(
            "Phase D1 block file scientific digest",
            record["scientific_block_digest"],
        )
        expected_path = f"{PHASE_D_TUNING_BLOCK_DIRECTORY}/block-{block:03d}.json"
        if (
            record["path"] != expected_path
            or not isinstance(record["bytes"], int)
            or isinstance(record["bytes"], bool)
            or record["bytes"] <= 0
        ):
            raise ValueError("Phase D1 block-file path is invalid")
        expected_block_scientific = _digest(
            f"{PHASE_D_HASH_DOMAIN}.tuning_scientific_block_v1",
            phase_d_scientific_block_payload(blocks[block]),
        )
        if record["scientific_block_digest"] != expected_block_scientific:
            raise ValueError("Phase D1 block-file scientific identity is invalid")
        if block_directory is not None:
            path = Path(block_directory).expanduser().resolve() / f"block-{block:03d}.json"
            raw = path.read_bytes()
            if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise ValueError("Phase D1 block file does not match its inventory")
            envelope = _load_valid_block(
                path,
                source_revision=str(item["source_revision"]),
                source_manifest_digest=manifest.digest,
                protocol_digest=str(item["protocol_digest"]),
                physical_config=physical_config,
            )
            if (
                envelope["archive_content_digest"]
                != record["archive_content_digest"]
                or envelope["scientific_block_digest"]
                != record["scientific_block_digest"]
                or envelope["block_result"] != blocks[block]
            ):
                raise ValueError("Phase D1 block inventory identity is inconsistent")
    performance = _exact_mapping(
        item["performance"],
        {
            "execution_wall_seconds_this_invocation",
            "coordinator_cpu_seconds_this_invocation",
            "aggregate_block_cpu_seconds",
            "maximum_recorded_worker_peak_rss_bytes",
            "resumed_block_count",
            "computed_block_count_this_invocation",
        },
        "Phase D1 tuning performance",
    )
    for name in (
        "execution_wall_seconds_this_invocation",
        "coordinator_cpu_seconds_this_invocation",
        "aggregate_block_cpu_seconds",
        "maximum_recorded_worker_peak_rss_bytes",
    ):
        value = performance[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError("Phase D1 tuning performance is invalid")
    if (
        not isinstance(performance["resumed_block_count"], int)
        or isinstance(performance["resumed_block_count"], bool)
        or not isinstance(performance["computed_block_count_this_invocation"], int)
        or isinstance(performance["computed_block_count_this_invocation"], bool)
        or performance["resumed_block_count"] < 0
        or performance["computed_block_count_this_invocation"] < 0
        or performance["resumed_block_count"]
        + performance["computed_block_count_this_invocation"]
        != DEVELOPMENT_TUNING_BLOCK_COUNT
        or performance["resumed_block_count"]
        != preflight["validated_existing_block_count"]
    ):
        raise ValueError("Phase D1 tuning resume accounting is invalid")
    expected_scientific = _digest(
        f"{PHASE_D_HASH_DOMAIN}.tuning_scientific_result_v1",
        phase_d_tuning_scientific_payload(item),
    )
    if item["scientific_result_digest"] != expected_scientific:
        raise ValueError("Phase D1 tuning scientific result digest is invalid")
    _validate_sha256("Phase D1 archive digest", item["archive_content_digest"])
    material = dict(item)
    material.pop("archive_content_digest")
    material.pop("archive_size_bytes")
    if item["archive_content_digest"] != _digest(
        f"{PHASE_D_HASH_DOMAIN}.tuning_archive_content_v1",
        material,
    ):
        raise ValueError("Phase D1 archive content digest is invalid")
    if (
        not isinstance(item["archive_size_bytes"], int)
        or isinstance(item["archive_size_bytes"], bool)
        or item["archive_size_bytes"] != len(_canonical_bytes(item) + b"\n")
    ):
        raise ValueError("Phase D1 archive size is invalid")
    return dict(item)


def format_phase_d_tuning_report(payload: Mapping[str, object]) -> str:
    blocks = payload["block_results"]
    pipeline_failures = sum(
        len(case["pipeline_failures"])
        for block in blocks
        for case in block["cases"]
    )
    block_failures = sum(
        isinstance(block["corrected_random_stream_audit"], Mapping)
        and block["corrected_random_stream_audit"].get("clean") is False
        for block in blocks
    )
    performance = payload["performance"]
    return "\n".join(
        (
            "Prospective operating-decision Phase D1 nominal tuning archive",
            "================================================================",
            "",
            f"Campaign: {payload['campaign']}",
            f"Partition: {payload['partition']}",
            f"Source revision: {payload['source_revision']}",
            f"Protocol digest: {payload['protocol_digest']}",
            f"Scientific result digest: {payload['scientific_result_digest']}",
            f"Blocks: {payload['block_count']}",
            f"Cases: {payload['case_count']}",
            f"Validated block files: {len(payload['block_file_records'])}",
            f"Canonical block failures: {block_failures}",
            f"Recorded case pipeline failures: {pipeline_failures}",
            "Runtime: "
            f"{payload['runtime_identity']['python_implementation']} "
            f"{payload['runtime_identity']['python_version']} on "
            f"{payload['runtime_identity']['operating_system']} "
            f"{payload['runtime_identity']['machine']}",
            "Workers: " + str(payload["worker_count"]),
            "Invocation wall time: "
            f"{performance['execution_wall_seconds_this_invocation']:.2f} s",
            "Aggregate block CPU time: "
            f"{performance['aggregate_block_cpu_seconds']:.2f} s",
            "",
            "PASS: all twenty block envelopes and the complete archive validated.",
            "STOP: truth analysis, offset estimation, and rule selection were not run.",
            "Next permitted step: Phase D2 only after this archive is preserved.",
        )
    )


def save_phase_d_tuning_artifacts(
    payload: Mapping[str, object],
    *,
    output_directory: Path | str,
    repository_root: Path | str,
) -> SavedPhaseDTuningArtifacts:
    paths = phase_d_tuning_output_paths(output_directory)
    destinations = (paths["json"], paths["report"], paths["hashes"])
    existing = tuple(path for path in destinations if path.exists())
    if existing:
        raise FileExistsError(
            "Phase D1 refuses to overwrite final artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    sealed, archive = _sealed_payload(
        payload,
        domain=f"{PHASE_D_HASH_DOMAIN}.tuning_archive_content_v1",
    )
    decoded = json.loads(archive)
    validate_phase_d_tuning_archive(
        decoded,
        repository_root=repository_root,
        block_directory=paths["blocks"],
    )
    report_bytes = (format_phase_d_tuning_report(decoded) + "\n").encode("utf-8")
    json_digest = hashlib.sha256(archive).hexdigest()
    report_digest = hashlib.sha256(report_bytes).hexdigest()
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["json"].write_bytes(archive)
    paths["report"].write_bytes(report_bytes)
    paths["hashes"].write_text(
        f"{json_digest}  {paths['json'].name}\n"
        f"{report_digest}  {paths['report'].name}\n",
        encoding="utf-8",
    )
    return SavedPhaseDTuningArtifacts(
        json_path=paths["json"],
        report_path=paths["report"],
        hash_path=paths["hashes"],
        json_sha256=json_digest,
        report_sha256=report_digest,
        archive_size_bytes=len(archive),
    )


__all__ = [
    "PHASE_D_TUNING_BLOCK_DIRECTORY",
    "PHASE_D_TUNING_JSON_NAME",
    "PHASE_D_TUNING_RESOURCE_NAME",
    "SavedPhaseDTuningArtifacts",
    "format_phase_d_tuning_report",
    "phase_d_tuning_block_archive",
    "phase_d_tuning_block_path",
    "phase_d_tuning_output_paths",
    "phase_d_tuning_partition",
    "phase_d_tuning_preflight_payload",
    "phase_d_tuning_scientific_payload",
    "run_phase_d_development_tuning",
    "save_phase_d_tuning_artifacts",
    "validate_phase_d_tuning_archive",
    "validate_phase_d_tuning_block",
    "validate_phase_d_tuning_block_archive",
]
