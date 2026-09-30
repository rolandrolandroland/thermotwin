"""Source-bound Phase D5 cost and sensor-quality measurement maps.

The nominal cost grid is rescored from the immutable Phase D1 archive.  The
three non-nominal sensor scenarios replay the same twenty development blocks
with paired semantic random streams and a scenario-specific physical
configuration.  Complete stress blocks are saved atomically and can be
resumed only under the same source, protocol, scenario, and input identities.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter, process_time
from typing import Callable, Dict, Mapping, Optional, Sequence

from .operating_decision import INSUFFICIENT_EVIDENCE, POLICY_NAMES, STOP_NOW
from .operating_decision_prospective import (
    ProspectiveDevelopmentOffsets,
    ProspectiveSelectorRule,
    prospective_development_offsets_from_payload,
    prospective_selector_rule_from_payload,
    prospective_selector_rule_payload,
    select_prospective_action,
)
from .operating_decision_prospective_costs import (
    PRIMARY_PROSPECTIVE_COST_SCENARIO,
    PROSPECTIVE_COST_SENSITIVITY_SCENARIOS,
    ProspectiveCostScenario,
    _action_cost_payload,
    _action_resources_payload,
    _build_action_costs,
    _build_action_resources,
    _scenario_payload,
)
from .operating_decision_prospective_phase_d import (
    PHASE_D_HASH_DOMAIN,
    PHASE_D_SENSOR_SCENARIOS,
    PhaseDSensorScenario,
    _sensor_scenario_payload,
    phase_d_runtime_identity,
    validate_executing_phase_d_runtime,
)
from .operating_decision_prospective_phase_d3_grid import (
    _padded_action_evaluations,
    _selected_outcome_record,
    _selection_payload,
    _snapshot_from_payload,
)
from .operating_decision_prospective_phase_d4_sensitivity import (
    PHASE_D4_SOURCE_PATHS,
    validate_phase_d4_archive,
)
from .operating_decision_prospective_phase_d_tuning import (
    _run_tuning_block_worker,
    validate_phase_d_tuning_block,
)
from .operating_decision_prospective_pilot import (
    DEVELOPMENT_TUNING_BLOCK_COUNT,
    PROSPECTIVE_CAMPAIGN,
    PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION,
    _canonical_bytes,
    _digest,
    _exact_mapping,
    _failed_block_result,
    _physical_config_payload,
    _validate_revision,
    _validate_sha256,
)
from .operating_decision_provenance import (
    create_source_manifest,
    source_manifest_from_payload,
    source_manifest_payload,
    verify_source_manifest,
)
from .operating_decision_realism import OperatingDecisionRealismConfig, STAGE3_TRUTH_CONDITIONS
from .operating_decision_replication import (
    CORRECTED_REPLICATION_CONFIG,
    corrected_partition_config,
)
from .sensor_model_discrimination import COLD_FACE, VOLTAGE


PHASE_D5_SCHEMA_VERSION = 1
PHASE_D5_PROTOCOL_VERSION = "operating_decision_phase_d5_maps_v1"
PHASE_D5_STRESS_SCHEMA_VERSION = 1
PHASE_D5_STRESS_PROTOCOL_VERSION = "operating_decision_phase_d5_sensor_stress_v1"
PHASE_D5_BLOCK_SCHEMA_VERSION = 1
PHASE_D5_BLOCK_PROTOCOL_VERSION = "operating_decision_phase_d5_sensor_stress_block_v1"
PHASE_D5_WORKER_COUNT = 4
PHASE_D5_JSON_NAME = "p1_development_measurement_maps.json"
PHASE_D5_REPORT_NAME = "p1_development_measurement_maps.txt"
PHASE_D5_HASH_NAME = "p1_development_measurement_maps.sha256"
PHASE_D5_RESOURCE_NAME = "p1_development_measurement_maps.resources.json"
PHASE_D5_COST_MAP_NAME = "nominal_cost_map.json"
PHASE_D5_STRESS_DIRECTORY = "sensor-stress"
PHASE_D5_STRESS_JSON_NAME = "scenario.json"
PHASE_D5_BLOCK_DIRECTORY = "blocks"
PHASE_D5_SOURCE_PATHS = tuple(
    sorted(
        set(PHASE_D4_SOURCE_PATHS).union(
            {
                "thermotwin/reports/operating_decision_prospective_phase_d5.py",
                "thermotwin/studies/operating_decision_prospective_phase_d5_maps.py",
            }
        )
    )
)


@dataclass(frozen=True)
class SavedPhaseD5Artifacts:
    json_path: Path
    report_path: Path
    hash_path: Path
    json_sha256: str
    report_sha256: str
    archive_size_bytes: int


def phase_d5_output_paths(output_directory: Path | str) -> dict:
    root = Path(output_directory).expanduser().resolve()
    return {
        "root": root,
        "json": root / PHASE_D5_JSON_NAME,
        "report": root / PHASE_D5_REPORT_NAME,
        "hashes": root / PHASE_D5_HASH_NAME,
        "resources": root / PHASE_D5_RESOURCE_NAME,
        "cost_map": root / PHASE_D5_COST_MAP_NAME,
        "stress": root / PHASE_D5_STRESS_DIRECTORY,
    }


def phase_d5_stress_paths(output_directory: Path | str, scenario_name: str) -> dict:
    if scenario_name not in {item.name for item in PHASE_D_SENSOR_SCENARIOS[1:]}:
        raise ValueError("Phase D5 stress scenario is not predeclared")
    root = phase_d5_output_paths(output_directory)["stress"] / scenario_name
    return {
        "root": root,
        "blocks": root / PHASE_D5_BLOCK_DIRECTORY,
        "json": root / PHASE_D5_STRESS_JSON_NAME,
    }


def phase_d5_block_path(
    output_directory: Path | str,
    scenario_name: str,
    block: int,
) -> Path:
    if not isinstance(block, int) or isinstance(block, bool) or not (
        0 <= block < DEVELOPMENT_TUNING_BLOCK_COUNT
    ):
        raise ValueError("Phase D5 block index is outside the tuning partition")
    return phase_d5_stress_paths(output_directory, scenario_name)["blocks"] / (
        f"block-{block:03d}.json"
    )


def _channel_noise_with_scenario(
    physical_config: OperatingDecisionRealismConfig,
    scenario: PhaseDSensorScenario,
) -> tuple[tuple[str, float], ...]:
    values = dict(physical_config.sensor.channel_noise)
    if VOLTAGE not in values or COLD_FACE not in values:
        raise ValueError("Phase D5 needs voltage and face-temperature noise scales")
    values[VOLTAGE] = scenario.voltage_noise
    values[COLD_FACE] = scenario.face_temperature_noise
    return tuple((name, values[name]) for name, _ in physical_config.sensor.channel_noise)


def phase_d5_physical_config(scenario: PhaseDSensorScenario) -> OperatingDecisionRealismConfig:
    """Apply one frozen sensor scenario to the paired tuning configuration."""

    if not isinstance(scenario, PhaseDSensorScenario):
        raise ValueError("Phase D5 needs a frozen sensor scenario")
    from .operating_decision_prospective_phase_d_tuning import phase_d_tuning_partition

    nominal = corrected_partition_config(
        phase_d_tuning_partition(),
        CORRECTED_REPLICATION_CONFIG,
    )
    sensor = replace(
        nominal.sensor,
        channel_noise=_channel_noise_with_scenario(nominal, scenario),
        run_bias_noise_ratio=scenario.run_bias_noise_ratio,
    )
    return replace(
        nominal,
        sensor=sensor,
        face_sensor_capacitance_nominal=scenario.probe_capacitance,
        face_sensor_response_nominal=scenario.probe_response_time,
    )


def phase_d5_protocol_payload(
    *, source_revision: str, source_manifest_digest: str
) -> dict:
    _validate_revision(source_revision)
    _validate_sha256("Phase D5 source-manifest digest", source_manifest_digest)
    return {
        "schema_version": PHASE_D5_SCHEMA_VERSION,
        "protocol_version": PHASE_D5_PROTOCOL_VERSION,
        "campaign": PROSPECTIVE_CAMPAIGN,
        "partition": PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION,
        "source_revision": source_revision,
        "source_manifest_digest": source_manifest_digest,
        "runtime_identity": phase_d_runtime_identity(),
        "block_count": DEVELOPMENT_TUNING_BLOCK_COUNT,
        "case_count": DEVELOPMENT_TUNING_BLOCK_COUNT * len(STAGE3_TRUTH_CONDITIONS),
        "worker_count": PHASE_D5_WORKER_COUNT,
        "cost_scenarios": [
            _scenario_payload(item) for item in PROSPECTIVE_COST_SENSITIVITY_SCENARIOS
        ],
        "sensor_scenarios": [
            _sensor_scenario_payload(item) for item in PHASE_D_SENSOR_SCENARIOS
        ],
        "nominal_evidence": "reuse_authenticated_phase_d1_without_refitting",
        "stress_evidence": "complete_paired_replay_with_affected_physics_and_refit",
        "cross_scenario_stream_rule": (
            "same_campaign_partition_block_and_semantic_streams_for_paired_sensitivity"
        ),
        "map_cells": len(PROSPECTIVE_COST_SENSITIVITY_SCENARIOS)
        * len(PHASE_D_SENSOR_SCENARIOS),
        "no_useful_measurement_definition": (
            "selection_reason_equals_no_valuable_acquisition_stop_then_verify"
        ),
        "truth_use": "offline_development_map_only_after_policy_records_saved",
        "internal_check_opened": False,
    }


def phase_d5_protocol_digest(*, source_revision: str, source_manifest_digest: str) -> str:
    return _digest(
        f"{PHASE_D_HASH_DOMAIN}.phase_d5_protocol_v1",
        phase_d5_protocol_payload(
            source_revision=source_revision,
            source_manifest_digest=source_manifest_digest,
        ),
    )


def _file_identity(path: Path) -> dict:
    raw = path.read_bytes()
    return {"path_name": path.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _validated_inputs(
    *,
    phase_d1_archive_path: Path | str,
    phase_d2_result_path: Path | str,
    phase_d3_result_path: Path | str,
    phase_d4_result_path: Path | str,
    rule_freeze_path: Path | str,
    monitor_preflight_path: Path | str,
    repository_root: Path | str,
) -> dict:
    root = Path(repository_root).expanduser().resolve(strict=True)
    paths = {
        "phase_d1": Path(phase_d1_archive_path).expanduser().resolve(strict=True),
        "phase_d2": Path(phase_d2_result_path).expanduser().resolve(strict=True),
        "phase_d3": Path(phase_d3_result_path).expanduser().resolve(strict=True),
        "phase_d4": Path(phase_d4_result_path).expanduser().resolve(strict=True),
        "rule": Path(rule_freeze_path).expanduser().resolve(strict=True),
        "monitor": Path(monitor_preflight_path).expanduser().resolve(strict=True),
    }
    d4 = validate_phase_d4_archive(
        json.loads(paths["phase_d4"].read_bytes()),
        phase_d1_archive_path=paths["phase_d1"],
        phase_d2_result_path=paths["phase_d2"],
        phase_d3_result_path=paths["phase_d3"],
        rule_freeze_path=paths["rule"],
        monitor_preflight_path=paths["monitor"],
        repository_root=root,
        block_directory=paths["phase_d4"].parent / "blocks",
    )
    if d4["acceptance"]["gate_passed"] is not True or d4["acceptance"]["phase_d5_authorized"] is not True:
        raise ValueError("Phase D5 is not authorized by the validated D4 result")
    d1 = json.loads(paths["phase_d1"].read_bytes())
    # The D4 validator above strictly rebuilds and validates the complete D3
    # result before checking its own archive, so parsing it again here does not
    # weaken the chain or repeat the expensive D1/D3 replay.
    d3 = json.loads(paths["phase_d3"].read_bytes())
    rule_freeze = json.loads(paths["rule"].read_bytes())
    rule = prospective_selector_rule_from_payload(rule_freeze["selector_rule"])
    if prospective_selector_rule_payload(rule) != d3["winning_selector_rule"]:
        raise ValueError("Phase D5 rule freeze differs from the validated D3 winner")
    offsets = prospective_development_offsets_from_payload(d3["development_offsets"])
    if rule.development_offsets != offsets:
        raise ValueError("Phase D5 rule offsets differ from the D3 offsets")
    return {
        "paths": paths,
        "identities": {name: _file_identity(path) for name, path in paths.items()},
        "d1": d1,
        "d3": d3,
        "d4": d4,
        "rule": rule,
        "offsets": offsets,
    }


def _score_case(
    case: Mapping[str, object],
    *,
    block: int,
    resource_payloads: Mapping[str, Mapping[str, object]],
    cost_payloads: Mapping[str, Mapping[str, object]],
    rule: ProspectiveSelectorRule,
    offsets: ProspectiveDevelopmentOffsets,
) -> dict:
    complete = case["n16_complete_uncertainty_result"]
    if not isinstance(complete, Mapping):
        return {
            "block": block,
            "truth_family": case["truth_condition"],
            "device_token": case["device_token"],
            "selected_policy": None,
            "selection_reason": "missing_n16_evidence",
            "eventual_decision": INSUFFICIENT_EVIDENCE,
            "definitive_decision": False,
            "ineligible_actions": list(POLICY_NAMES[1:]),
            "selection_failure": True,
            "verification_failure": False,
            "pipeline_failure": True,
            "no_useful_measurement": False,
        }
    scorecard = {"action_costs": list(cost_payloads.values())}
    evaluations = _padded_action_evaluations(complete, scorecard, offsets)
    snapshot = _snapshot_from_payload(complete["acquisition_evidence"]["snapshot"])
    selection = select_prospective_action(snapshot, evaluations, rule)
    selection_payload = _selection_payload(selection)
    outcome = _selected_outcome_record(
        case,
        block=block,
        policy_name=(selection.selected_policy if selection.selection_succeeded else None),
        selection_payload=selection_payload,
        offsets=offsets,
        resources=resource_payloads,
        costs=cost_payloads,
    )
    decision = outcome["adjusted_decision"]["development_adjusted_decision"]
    return {
        "block": block,
        "truth_family": case["truth_condition"],
        "device_token": case["device_token"],
        "selected_policy": outcome["selected_policy"],
        "selection_reason": selection.reason,
        "eventual_decision": decision,
        "definitive_decision": decision != INSUFFICIENT_EVIDENCE,
        "ineligible_actions": outcome["ineligible_actions"],
        "selection_failure": outcome["selected_policy"] is None,
        "verification_failure": outcome["verification_failure"],
        "pipeline_failure": outcome["pipeline_failure"],
        "no_useful_measurement": selection.reason == "no_valuable_acquisition_stop_then_verify",
    }


def _rate(count: int, denominator: int) -> Optional[float]:
    return None if denominator == 0 else count / denominator


def summarize_measurement_map(records: Sequence[Mapping[str, object]]) -> dict:
    values = tuple(records)
    if not values:
        raise ValueError("Phase D5 measurement map cannot be empty")
    actions = [*POLICY_NAMES, "selection_failure"]
    action_rows = []
    for action in actions:
        selected = [
            item
            for item in values
            if (item["selected_policy"] if item["selected_policy"] is not None else "selection_failure") == action
        ]
        definitive = sum(bool(item["definitive_decision"]) for item in selected)
        action_rows.append(
            {
                "action": action,
                "selected_count": len(selected),
                "selection_rate": _rate(len(selected), len(values)),
                "definitive_count": definitive,
                "definitive_rate_all_cases": _rate(definitive, len(values)),
                "definitive_rate_within_action": _rate(definitive, len(selected)),
            }
        )
    return {
        "case_count": len(values),
        "actions": action_rows,
        "definitive_decision_count": sum(bool(item["definitive_decision"]) for item in values),
        "definitive_decision_rate": _rate(
            sum(bool(item["definitive_decision"]) for item in values), len(values)
        ),
        "ineligible_action_counts": {
            action: sum(action in item["ineligible_actions"] for item in values)
            for action in POLICY_NAMES[1:]
        },
        "selection_failure_count": sum(bool(item["selection_failure"]) for item in values),
        "verification_failure_count": sum(bool(item["verification_failure"]) for item in values),
        "pipeline_failure_count": sum(bool(item["pipeline_failure"]) for item in values),
        "no_useful_measurement_count": sum(bool(item["no_useful_measurement"]) for item in values),
    }


def build_measurement_maps(
    blocks: Sequence[Mapping[str, object]],
    *,
    sensor_scenario: PhaseDSensorScenario,
    rule: ProspectiveSelectorRule,
    offsets: ProspectiveDevelopmentOffsets,
) -> list[dict]:
    if [item.get("block") for item in blocks] != list(range(DEVELOPMENT_TUNING_BLOCK_COUNT)):
        raise ValueError("Phase D5 maps need all twenty ordered tuning blocks")
    physical_config = phase_d5_physical_config(sensor_scenario)
    rows = []
    for cost_scenario in PROSPECTIVE_COST_SENSITIVITY_SCENARIOS:
        resources = _build_action_resources(physical_config, cost_scenario)
        costs, _, _ = _build_action_costs(resources, cost_scenario)
        resource_payloads = {
            item.policy_name: _action_resources_payload(item) for item in resources
        }
        cost_payloads = {
            item.policy_name: _action_cost_payload(item) for item in costs
        }
        records = [
            _score_case(
                case,
                block=block_index,
                resource_payloads=resource_payloads,
                cost_payloads=cost_payloads,
                rule=rule,
                offsets=offsets,
            )
            for block_index, block in enumerate(blocks)
            for case in block["cases"]
        ]
        rows.append(
            {
                "sensor_scenario": sensor_scenario.name,
                "cost_scenario": cost_scenario.name,
                "primary": sensor_scenario.primary
                and cost_scenario.name == PRIMARY_PROSPECTIVE_COST_SCENARIO.name,
                "summary": summarize_measurement_map(records),
                "case_records": records,
            }
        )
    return rows


def _scientific_block_payload(block_result: Mapping[str, object]) -> dict:
    return {key: value for key, value in block_result.items() if key != "timing"}


def _sealed_payload(payload: Mapping[str, object], *, domain: str) -> tuple[dict, bytes]:
    sealed = dict(payload)
    sealed["archive_content_digest"] = _digest(domain, payload)
    sealed["archive_size_bytes"] = 0
    for _ in range(16):
        raw = _canonical_bytes(sealed) + b"\n"
        if sealed["archive_size_bytes"] == len(raw):
            return sealed, raw
        sealed["archive_size_bytes"] = len(raw)
    raise RuntimeError("Phase D5 archive-size fixed point did not converge")


def _block_envelope(
    block_result: Mapping[str, object],
    *,
    block: int,
    scenario: PhaseDSensorScenario,
    physical_config: OperatingDecisionRealismConfig,
    source_revision: str,
    source_manifest_digest: str,
    protocol_digest: str,
) -> tuple[dict, bytes]:
    validate_phase_d_tuning_block(
        block_result,
        block=block,
        physical_config=physical_config,
    )
    payload = {
        "schema_version": PHASE_D5_BLOCK_SCHEMA_VERSION,
        "protocol_version": PHASE_D5_BLOCK_PROTOCOL_VERSION,
        "source_revision": source_revision,
        "source_manifest_digest": source_manifest_digest,
        "phase_d5_protocol_digest": protocol_digest,
        "sensor_scenario": _sensor_scenario_payload(scenario),
        "physical_config": _physical_config_payload(physical_config),
        "block": block,
        "block_result": dict(block_result),
        "scientific_block_digest": _digest(
            f"{PHASE_D_HASH_DOMAIN}.phase_d5_stress_block_v1",
            {
                "sensor_scenario": _sensor_scenario_payload(scenario),
                "physical_config": _physical_config_payload(physical_config),
                "block_result": _scientific_block_payload(block_result),
            },
        ),
    }
    return _sealed_payload(
        payload,
        domain=f"{PHASE_D_HASH_DOMAIN}.phase_d5_stress_block_archive_v1",
    )


def _validate_block_envelope(
    payload: Mapping[str, object],
    *,
    block: int,
    scenario: PhaseDSensorScenario,
    physical_config: OperatingDecisionRealismConfig,
    source_revision: str,
    source_manifest_digest: str,
    protocol_digest: str,
) -> dict:
    item = _exact_mapping(
        payload,
        {
            "schema_version", "protocol_version", "source_revision",
            "source_manifest_digest", "phase_d5_protocol_digest",
            "sensor_scenario", "physical_config", "block", "block_result",
            "scientific_block_digest", "archive_content_digest", "archive_size_bytes",
        },
        "Phase D5 stress block archive",
    )
    if (
        item["schema_version"] != PHASE_D5_BLOCK_SCHEMA_VERSION
        or item["protocol_version"] != PHASE_D5_BLOCK_PROTOCOL_VERSION
        or item["source_revision"] != source_revision
        or item["source_manifest_digest"] != source_manifest_digest
        or item["phase_d5_protocol_digest"] != protocol_digest
        or item["sensor_scenario"] != _sensor_scenario_payload(scenario)
        or item["physical_config"] != _physical_config_payload(physical_config)
        or item["block"] != block
    ):
        raise ValueError("Phase D5 stress block header is invalid")
    validate_phase_d_tuning_block(item["block_result"], block=block, physical_config=physical_config)
    expected_scientific = _digest(
        f"{PHASE_D_HASH_DOMAIN}.phase_d5_stress_block_v1",
        {
            "sensor_scenario": _sensor_scenario_payload(scenario),
            "physical_config": _physical_config_payload(physical_config),
            "block_result": _scientific_block_payload(item["block_result"]),
        },
    )
    unsealed = {key: value for key, value in item.items() if key not in {"archive_content_digest", "archive_size_bytes"}}
    sealed, raw = _sealed_payload(unsealed, domain=f"{PHASE_D_HASH_DOMAIN}.phase_d5_stress_block_archive_v1")
    if (
        item["scientific_block_digest"] != expected_scientific
        or item["archive_content_digest"] != sealed["archive_content_digest"]
        or item["archive_size_bytes"] != len(raw)
    ):
        raise ValueError("Phase D5 stress block identity is invalid")
    return dict(item)


def _atomic_save_block(path: Path, envelope: Mapping[str, object], raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(f"Phase D5 unresolved partial block exists: {partial}")
    partial.write_bytes(raw)
    loaded = json.loads(partial.read_bytes())
    if loaded != envelope:
        partial.unlink(missing_ok=True)
        raise ValueError("Phase D5 block changed during JSON round trip")
    os.replace(partial, path)


def _run_stress_scenario(
    *,
    output_directory: Path | str,
    scenario: PhaseDSensorScenario,
    source_revision: str,
    source_manifest_digest: str,
    protocol_digest: str,
    rule: ProspectiveSelectorRule,
    offsets: ProspectiveDevelopmentOffsets,
    progress: Optional[Callable[[str], None]],
) -> dict:
    if scenario.primary:
        raise ValueError("nominal Phase D5 evidence must reuse Phase D1")
    paths = phase_d5_stress_paths(output_directory, scenario.name)
    physical_config = phase_d5_physical_config(scenario)
    completed: Dict[int, dict] = {}
    records: Dict[int, dict] = {}
    for block in range(DEVELOPMENT_TUNING_BLOCK_COUNT):
        path = phase_d5_block_path(output_directory, scenario.name, block)
        if not path.exists():
            continue
        envelope = _validate_block_envelope(
            json.loads(path.read_bytes()), block=block, scenario=scenario,
            physical_config=physical_config, source_revision=source_revision,
            source_manifest_digest=source_manifest_digest, protocol_digest=protocol_digest,
        )
        completed[block] = envelope["block_result"]
        records[block] = envelope
    resumed_count = len(completed)
    missing = tuple(block for block in range(DEVELOPMENT_TUNING_BLOCK_COUNT) if block not in completed)
    wall_started, cpu_started = perf_counter(), process_time()
    arguments = tuple(
        (block, physical_config, ProspectiveSelectorRule(), PRIMARY_PROSPECTIVE_COST_SCENARIO)
        for block in missing
    )
    if arguments:
        with ProcessPoolExecutor(max_workers=min(PHASE_D5_WORKER_COUNT, len(arguments))) as executor:
            futures = {executor.submit(_run_tuning_block_worker, argument): argument[0] for argument in arguments}
            for future in as_completed(futures):
                block = futures[future]
                try:
                    block_result = future.result()
                except Exception as error:
                    block_result = _failed_block_result(block, error)
                envelope, raw = _block_envelope(
                    block_result, block=block, scenario=scenario,
                    physical_config=physical_config, source_revision=source_revision,
                    source_manifest_digest=source_manifest_digest, protocol_digest=protocol_digest,
                )
                path = phase_d5_block_path(output_directory, scenario.name, block)
                _atomic_save_block(path, envelope, raw)
                completed[block] = envelope["block_result"]
                records[block] = envelope
                if progress is not None:
                    progress(f"Phase D5 {scenario.name}: validated block {len(completed)}/20")
    ordered = [completed[block] for block in range(DEVELOPMENT_TUNING_BLOCK_COUNT)]
    block_files = []
    for block in range(DEVELOPMENT_TUNING_BLOCK_COUNT):
        path = phase_d5_block_path(output_directory, scenario.name, block)
        raw = path.read_bytes()
        block_files.append({
            "block": block,
            "path": f"{PHASE_D5_BLOCK_DIRECTORY}/{path.name}",
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "scientific_block_digest": records[block]["scientific_block_digest"],
            "archive_content_digest": records[block]["archive_content_digest"],
        })
    maps = build_measurement_maps(
        ordered, sensor_scenario=scenario, rule=rule, offsets=offsets
    )
    payload = {
        "schema_version": PHASE_D5_STRESS_SCHEMA_VERSION,
        "protocol_version": PHASE_D5_STRESS_PROTOCOL_VERSION,
        "source_revision": source_revision,
        "source_manifest_digest": source_manifest_digest,
        "phase_d5_protocol_digest": protocol_digest,
        "sensor_scenario": _sensor_scenario_payload(scenario),
        "physical_config": _physical_config_payload(physical_config),
        "block_count": DEVELOPMENT_TUNING_BLOCK_COUNT,
        "case_count": DEVELOPMENT_TUNING_BLOCK_COUNT * len(STAGE3_TRUTH_CONDITIONS),
        "block_results": ordered,
        "block_file_records": block_files,
        "measurement_maps": maps,
        "performance": {
            "execution_wall_seconds_this_invocation": perf_counter() - wall_started,
            "coordinator_cpu_seconds_this_invocation": process_time() - cpu_started,
            "aggregate_block_cpu_seconds": sum(float(item["timing"]["block_cpu_seconds"]) for item in ordered),
            "resumed_block_count": resumed_count,
            "computed_block_count_this_invocation": len(missing),
        },
    }
    payload["scientific_result_digest"] = _digest(
        f"{PHASE_D_HASH_DOMAIN}.phase_d5_stress_result_v1",
        {key: value for key, value in payload.items() if key not in {"performance", "block_results"}} | {
            "scientific_blocks": [_scientific_block_payload(item) for item in ordered]
        },
    )
    sealed, raw = _sealed_payload(payload, domain=f"{PHASE_D_HASH_DOMAIN}.phase_d5_stress_archive_v1")
    paths["root"].mkdir(parents=True, exist_ok=True)
    if paths["json"].exists():
        existing = validate_phase_d5_stress_archive(
            json.loads(paths["json"].read_bytes()), scenario=scenario,
            repository_root=None, block_directory=paths["blocks"],
            source_revision=source_revision, source_manifest_digest=source_manifest_digest,
            protocol_digest=protocol_digest, rule=rule, offsets=offsets,
        )
        if existing["scientific_result_digest"] != sealed["scientific_result_digest"]:
            raise ValueError("Phase D5 existing stress result differs from recomputation")
        return existing
    paths["json"].write_bytes(raw)
    return validate_phase_d5_stress_archive(
        json.loads(paths["json"].read_bytes()), scenario=scenario,
        repository_root=None, block_directory=paths["blocks"],
        source_revision=source_revision, source_manifest_digest=source_manifest_digest,
        protocol_digest=protocol_digest, rule=rule, offsets=offsets,
    )


def validate_phase_d5_stress_archive(
    payload: Mapping[str, object], *, scenario: PhaseDSensorScenario,
    repository_root: Path | str | None, block_directory: Path | str,
    source_revision: str, source_manifest_digest: str, protocol_digest: str,
    rule: ProspectiveSelectorRule, offsets: ProspectiveDevelopmentOffsets,
) -> dict:
    item = _exact_mapping(payload, {
        "schema_version", "protocol_version", "source_revision", "source_manifest_digest",
        "phase_d5_protocol_digest", "sensor_scenario", "physical_config", "block_count",
        "case_count", "block_results", "block_file_records", "measurement_maps",
        "performance", "scientific_result_digest", "archive_content_digest", "archive_size_bytes",
    }, "Phase D5 stress archive")
    physical_config = phase_d5_physical_config(scenario)
    if (
        item["schema_version"] != PHASE_D5_STRESS_SCHEMA_VERSION
        or item["protocol_version"] != PHASE_D5_STRESS_PROTOCOL_VERSION
        or item["source_revision"] != source_revision
        or item["source_manifest_digest"] != source_manifest_digest
        or item["phase_d5_protocol_digest"] != protocol_digest
        or item["sensor_scenario"] != _sensor_scenario_payload(scenario)
        or item["physical_config"] != _physical_config_payload(physical_config)
        or item["block_count"] != DEVELOPMENT_TUNING_BLOCK_COUNT
        or item["case_count"] != DEVELOPMENT_TUNING_BLOCK_COUNT * len(STAGE3_TRUTH_CONDITIONS)
    ):
        raise ValueError("Phase D5 stress archive header is invalid")
    if repository_root is not None:
        manifest = create_source_manifest(repository_root, PHASE_D5_SOURCE_PATHS)
        if manifest.digest != source_manifest_digest:
            raise ValueError("Phase D5 stress archive source manifest differs")
    blocks = item["block_results"]
    if [value.get("block") for value in blocks] != list(range(DEVELOPMENT_TUNING_BLOCK_COUNT)):
        raise ValueError("Phase D5 stress archive blocks are incomplete")
    records = item["block_file_records"]
    if [value.get("block") for value in records] != list(range(DEVELOPMENT_TUNING_BLOCK_COUNT)):
        raise ValueError("Phase D5 stress block inventory is incomplete")
    block_root = Path(block_directory).expanduser().resolve(strict=True)
    for block, record in enumerate(records):
        path = block_root / f"block-{block:03d}.json"
        raw = path.read_bytes()
        if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("Phase D5 stress block file differs from inventory")
        envelope = _validate_block_envelope(
            json.loads(raw), block=block, scenario=scenario, physical_config=physical_config,
            source_revision=source_revision, source_manifest_digest=source_manifest_digest,
            protocol_digest=protocol_digest,
        )
        if envelope["block_result"] != blocks[block]:
            raise ValueError("Phase D5 stress block result differs from its file")
    expected_maps = build_measurement_maps(blocks, sensor_scenario=scenario, rule=rule, offsets=offsets)
    if item["measurement_maps"] != expected_maps:
        raise ValueError("Phase D5 stress measurement maps do not recompute")
    expected_scientific = _digest(
        f"{PHASE_D_HASH_DOMAIN}.phase_d5_stress_result_v1",
        {key: value for key, value in item.items() if key not in {"performance", "block_results", "scientific_result_digest", "archive_content_digest", "archive_size_bytes"}} | {
            "scientific_blocks": [_scientific_block_payload(value) for value in blocks]
        },
    )
    unsealed = {key: value for key, value in item.items() if key not in {"archive_content_digest", "archive_size_bytes"}}
    sealed, raw = _sealed_payload(unsealed, domain=f"{PHASE_D_HASH_DOMAIN}.phase_d5_stress_archive_v1")
    if (
        item["scientific_result_digest"] != expected_scientific
        or item["archive_content_digest"] != sealed["archive_content_digest"]
        or item["archive_size_bytes"] != len(raw)
    ):
        raise ValueError("Phase D5 stress archive identity is invalid")
    return dict(item)


def run_phase_d5_maps(
    *, phase_d1_archive_path: Path | str, phase_d2_result_path: Path | str,
    phase_d3_result_path: Path | str, phase_d4_result_path: Path | str,
    rule_freeze_path: Path | str, monitor_preflight_path: Path | str,
    repository_root: Path | str, source_revision: str, output_directory: Path | str,
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    _validate_revision(source_revision)
    validate_executing_phase_d_runtime()
    root = Path(repository_root).expanduser().resolve(strict=True)
    inputs = _validated_inputs(
        phase_d1_archive_path=phase_d1_archive_path, phase_d2_result_path=phase_d2_result_path,
        phase_d3_result_path=phase_d3_result_path, phase_d4_result_path=phase_d4_result_path,
        rule_freeze_path=rule_freeze_path, monitor_preflight_path=monitor_preflight_path,
        repository_root=root,
    )
    manifest = create_source_manifest(root, PHASE_D5_SOURCE_PATHS)
    protocol = phase_d5_protocol_payload(source_revision=source_revision, source_manifest_digest=manifest.digest)
    protocol_digest = phase_d5_protocol_digest(source_revision=source_revision, source_manifest_digest=manifest.digest)
    paths = phase_d5_output_paths(output_directory)
    if any(paths[name].exists() for name in ("json", "report", "hashes", "resources")):
        raise FileExistsError("Phase D5 final output already exists")
    nominal = PHASE_D_SENSOR_SCENARIOS[0]
    nominal_maps = build_measurement_maps(
        inputs["d1"]["block_results"], sensor_scenario=nominal,
        rule=inputs["rule"], offsets=inputs["offsets"],
    )
    cost_payload = {
        "schema_version": 1,
        "protocol_version": "operating_decision_phase_d5_nominal_cost_map_v1",
        "source_revision": source_revision,
        "phase_d5_protocol_digest": protocol_digest,
        "input_phase_d1": inputs["identities"]["phase_d1"],
        "measurement_maps": nominal_maps,
    }
    cost_payload["scientific_result_digest"] = _digest(
        f"{PHASE_D_HASH_DOMAIN}.phase_d5_nominal_cost_map_v1", cost_payload
    )
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["cost_map"].write_bytes(_canonical_bytes(cost_payload) + b"\n")
    stress_results = []
    for scenario in PHASE_D_SENSOR_SCENARIOS[1:]:
        stress_results.append(_run_stress_scenario(
            output_directory=output_directory, scenario=scenario,
            source_revision=source_revision, source_manifest_digest=manifest.digest,
            protocol_digest=protocol_digest, rule=inputs["rule"], offsets=inputs["offsets"],
            progress=progress,
        ))
    maps = nominal_maps + [row for result in stress_results for row in result["measurement_maps"]]
    payload = {
        "schema_version": PHASE_D5_SCHEMA_VERSION,
        "protocol_version": PHASE_D5_PROTOCOL_VERSION,
        "analysis_source_revision": source_revision,
        "analysis_source_manifest": source_manifest_payload(manifest),
        "runtime_identity": phase_d_runtime_identity(),
        "phase_d5_protocol": protocol,
        "phase_d5_protocol_digest": protocol_digest,
        "inputs": inputs["identities"],
        "selector_rule": prospective_selector_rule_payload(inputs["rule"]),
        "measurement_map_count": len(maps),
        "measurement_maps": maps,
        "stress_scenario_results": [
            {
                "sensor_scenario": item["sensor_scenario"]["name"],
                "scientific_result_digest": item["scientific_result_digest"],
                "archive_content_digest": item["archive_content_digest"],
                "archive_size_bytes": item["archive_size_bytes"],
                "json_sha256": hashlib.sha256(
                    phase_d5_stress_paths(output_directory, item["sensor_scenario"]["name"])["json"].read_bytes()
                ).hexdigest(),
            }
            for item in stress_results
        ],
        "next_required_action": "freeze_provisional_design_then_execute_phase_d6_internal_check",
        "internal_check_opened": False,
    }
    payload["scientific_result_digest"] = _digest(
        f"{PHASE_D_HASH_DOMAIN}.phase_d5_maps_result_v1", payload
    )
    return payload


def validate_phase_d5_archive(
    payload: Mapping[str, object], *, phase_d1_archive_path: Path | str,
    phase_d2_result_path: Path | str, phase_d3_result_path: Path | str,
    phase_d4_result_path: Path | str, rule_freeze_path: Path | str,
    monitor_preflight_path: Path | str, repository_root: Path | str,
    output_directory: Path | str,
) -> dict:
    item = _exact_mapping(payload, {
        "schema_version", "protocol_version", "analysis_source_revision",
        "analysis_source_manifest", "runtime_identity", "phase_d5_protocol",
        "phase_d5_protocol_digest", "inputs", "selector_rule", "measurement_map_count",
        "measurement_maps", "stress_scenario_results", "next_required_action",
        "internal_check_opened", "scientific_result_digest", "archive_content_digest",
        "archive_size_bytes",
    }, "Phase D5 measurement-map archive")
    root = Path(repository_root).expanduser().resolve(strict=True)
    inputs = _validated_inputs(
        phase_d1_archive_path=phase_d1_archive_path, phase_d2_result_path=phase_d2_result_path,
        phase_d3_result_path=phase_d3_result_path, phase_d4_result_path=phase_d4_result_path,
        rule_freeze_path=rule_freeze_path, monitor_preflight_path=monitor_preflight_path,
        repository_root=root,
    )
    manifest = source_manifest_from_payload(item["analysis_source_manifest"])
    verify_source_manifest(manifest, root, PHASE_D5_SOURCE_PATHS).assert_valid()
    expected_protocol = phase_d5_protocol_payload(
        source_revision=item["analysis_source_revision"], source_manifest_digest=manifest.digest
    )
    expected_digest = phase_d5_protocol_digest(
        source_revision=item["analysis_source_revision"], source_manifest_digest=manifest.digest
    )
    if (
        item["schema_version"] != PHASE_D5_SCHEMA_VERSION
        or item["protocol_version"] != PHASE_D5_PROTOCOL_VERSION
        or item["runtime_identity"] != phase_d_runtime_identity()
        or item["phase_d5_protocol"] != expected_protocol
        or item["phase_d5_protocol_digest"] != expected_digest
        or item["inputs"] != inputs["identities"]
        or item["selector_rule"] != prospective_selector_rule_payload(inputs["rule"])
        or item["next_required_action"] != "freeze_provisional_design_then_execute_phase_d6_internal_check"
        or item["internal_check_opened"] is not False
    ):
        raise ValueError("Phase D5 archive header or input binding is invalid")
    nominal_maps = build_measurement_maps(
        inputs["d1"]["block_results"], sensor_scenario=PHASE_D_SENSOR_SCENARIOS[0],
        rule=inputs["rule"], offsets=inputs["offsets"],
    )
    stress_maps = []
    expected_records = []
    for scenario in PHASE_D_SENSOR_SCENARIOS[1:]:
        paths = phase_d5_stress_paths(output_directory, scenario.name)
        raw = paths["json"].read_bytes()
        stress = validate_phase_d5_stress_archive(
            json.loads(raw), scenario=scenario, repository_root=root,
            block_directory=paths["blocks"], source_revision=item["analysis_source_revision"],
            source_manifest_digest=manifest.digest, protocol_digest=expected_digest,
            rule=inputs["rule"], offsets=inputs["offsets"],
        )
        stress_maps.extend(stress["measurement_maps"])
        expected_records.append({
            "sensor_scenario": scenario.name,
            "scientific_result_digest": stress["scientific_result_digest"],
            "archive_content_digest": stress["archive_content_digest"],
            "archive_size_bytes": stress["archive_size_bytes"],
            "json_sha256": hashlib.sha256(raw).hexdigest(),
        })
    expected_maps = nominal_maps + stress_maps
    if (
        item["measurement_map_count"] != 48
        or item["measurement_maps"] != expected_maps
        or item["stress_scenario_results"] != expected_records
    ):
        raise ValueError("Phase D5 measurement maps or stress identities do not recompute")
    unsealed = {key: value for key, value in item.items() if key not in {"archive_content_digest", "archive_size_bytes"}}
    expected_scientific = _digest(f"{PHASE_D_HASH_DOMAIN}.phase_d5_maps_result_v1", {
        key: value for key, value in unsealed.items() if key != "scientific_result_digest"
    })
    sealed, raw = _sealed_payload(unsealed, domain=f"{PHASE_D_HASH_DOMAIN}.phase_d5_maps_archive_v1")
    if (
        item["scientific_result_digest"] != expected_scientific
        or item["archive_content_digest"] != sealed["archive_content_digest"]
        or item["archive_size_bytes"] != len(raw)
    ):
        raise ValueError("Phase D5 archive identity is invalid")
    return dict(item)


def format_phase_d5_report(payload: Mapping[str, object]) -> str:
    primary = next(item for item in payload["measurement_maps"] if item["primary"])
    summary = primary["summary"]
    actions = ", ".join(
        f"{item['action']}={item['selected_count']}" for item in summary["actions"]
    )
    return "\n".join([
        "Prospective operating-decision Phase D5 measurement maps",
        "===========================================================",
        "",
        f"Analysis source revision: {payload['analysis_source_revision']}",
        f"Scientific result digest: {payload['scientific_result_digest']}",
        f"Map cells: {payload['measurement_map_count']}",
        f"Primary action counts: {actions}",
        f"Primary definitive decisions: {summary['definitive_decision_count']}/{summary['case_count']}",
        f"Primary selection failures: {summary['selection_failure_count']}",
        f"Primary verification failures: {summary['verification_failure_count']}",
        f"Next required action: {payload['next_required_action']}",
        "",
    ])


def save_phase_d5_artifacts(
    payload: Mapping[str, object], output_directory: Path | str
) -> SavedPhaseD5Artifacts:
    paths = phase_d5_output_paths(output_directory)
    for name in ("json", "report", "hashes"):
        if paths[name].exists():
            raise FileExistsError(f"Phase D5 output already exists: {paths[name]}")
    sealed, raw = _sealed_payload(payload, domain=f"{PHASE_D_HASH_DOMAIN}.phase_d5_maps_archive_v1")
    report = format_phase_d5_report(sealed).encode("utf-8")
    json_sha = hashlib.sha256(raw).hexdigest()
    report_sha = hashlib.sha256(report).hexdigest()
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["json"].write_bytes(raw)
    paths["report"].write_bytes(report)
    paths["hashes"].write_text(
        f"{json_sha}  {paths['json'].name}\n{report_sha}  {paths['report'].name}\n",
        encoding="utf-8",
    )
    return SavedPhaseD5Artifacts(
        json_path=paths["json"], report_path=paths["report"], hash_path=paths["hashes"],
        json_sha256=json_sha, report_sha256=report_sha, archive_size_bytes=len(raw),
    )


__all__ = [
    "PHASE_D5_SOURCE_PATHS", "PHASE_D5_JSON_NAME", "PHASE_D5_REPORT_NAME",
    "PHASE_D5_HASH_NAME", "PHASE_D5_RESOURCE_NAME", "build_measurement_maps",
    "format_phase_d5_report", "phase_d5_output_paths", "phase_d5_physical_config",
    "phase_d5_protocol_digest", "phase_d5_protocol_payload", "run_phase_d5_maps",
    "save_phase_d5_artifacts", "summarize_measurement_map", "validate_phase_d5_archive",
    "validate_phase_d5_stress_archive",
]
