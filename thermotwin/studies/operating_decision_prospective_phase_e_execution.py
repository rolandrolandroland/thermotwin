"""Checkpointed Phase E calibration and disposable rehearsal execution.

The numerical family worker is reused byte-for-byte from frozen D6. Only its
explicit partition argument changes. New validators bind Phase E identities;
no historical sealed source is edited. This module cannot execute reserved.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from time import perf_counter, process_time, time_ns
from typing import Callable, Mapping, Sequence

from . import operating_decision_prospective_phase_d6_internal_check as d6
from .operating_decision import POLICY_NAMES, STOP_NOW
from .operating_decision_prospective import ProspectiveSelectorRule
from .operating_decision_prospective_costs import ProspectiveCostScenario
from .operating_decision_prospective_phase_e import (
    PHASE_E_DOMAIN, PHASE_E_REHEARSAL_PARTITION, PHASE_E_SOURCE_PATHS,
    PHASE_E_VERSION, phase_e_protocol_digest, phase_e_protocol_payload,
)
from .operating_decision_prospective_phase_e_calibration import (
    CALIBRATION_BLOCKS, PHASE_E_PROCEDURES, SELECTOR_PROCEDURE,
    calibrate_complete_procedures, case_nonconformity, _finite,
)
from .operating_decision_prospective_pilot import (
    PROSPECTIVE_CAMPAIGN, PROSPECTIVE_CALIBRATION_PARTITION,
    _expected_corrected_stream_manifest,
)
from .operating_decision_random_streams import RandomStream, RandomStreamKey
from .operating_decision_provenance import (
    create_source_manifest, source_manifest_from_payload, source_manifest_payload,
    verify_source_manifest, verify_committed_paths,
)
from .operating_decision_realism import OperatingDecisionRealismConfig, STAGE3_TRUTH_CONDITIONS
from .operating_decision_replication import CorrectedPartition


def phase_e_partition(name: str) -> CorrectedPartition:
    sizes = {PHASE_E_REHEARSAL_PARTITION: 1,
             PROSPECTIVE_CALIBRATION_PARTITION: CALIBRATION_BLOCKS}
    if name not in sizes:
        raise ValueError("Phase E can open only disposable rehearsal or independent calibration")
    return CorrectedPartition(name=name, block_count=sizes[name], campaign=PROSPECTIVE_CAMPAIGN)


def _physical_config(partition: CorrectedPartition) -> OperatingDecisionRealismConfig:
    return d6.corrected_partition_config(partition, d6.CORRECTED_REPLICATION_CONFIG)


def calibration_case_records(case: Mapping[str, object]) -> list:
    """After-save offline records; failures retained, shared truth authenticated."""
    fixed = case["fixed_policy_results"]
    margins = []
    for record in fixed.values():
        post = record.get("post_reveal_score")
        if isinstance(post, Mapping):
            margin = post.get("revealed", {}).get("true_margin")
            if not _finite(margin):
                raise ValueError("material incident: nonfinite final truth")
            margins.append(float(margin))
    if not margins or any(margin != margins[0] for margin in margins):
        raise ValueError("material incident: paired procedures do not share one target truth")
    offsets = d6.phase_d6_frozen_selector_rule().development_offsets
    rows = []
    for procedure in PHASE_E_PROCEDURES:
        selected = case["selected_policy_at_n16_primary"] if procedure == SELECTOR_PROCEDURE else procedure
        record = fixed.get(selected) if selected is not None else None
        saved = record.get("saved_before_reveal") if isinstance(record, Mapping) else None
        post = record.get("post_reveal_score") if isinstance(record, Mapping) else None
        verified = bool(isinstance(saved, Mapping) and any(
            v.get("passed") is True for v in saved.get("verifications", [])
            if isinstance(v, Mapping)
        ))
        failed = selected is None or not isinstance(saved, Mapping) or "failure" in record
        scored = post.get("scored") if isinstance(post, Mapping) else None
        resources = {
            name: scored.get(name) if isinstance(scored, Mapping) else None
            for name in ("total_diagnostic_energy", "diagnostic_run_count",
                         "extra_sensor_count", "energized_schedule_time_seconds")
        }
        row = {
            "block": case["block"], "truth_family": case["truth_condition"],
            "device_token": case["device_token"], "procedure": procedure,
            "selected_action": selected,
            "raw_interval": deepcopy(saved.get("margin_envelope")) if isinstance(saved, Mapping) else None,
            "development_offset_kelvin": offsets.for_policy(selected) if selected is not None else 0.,
            "verification_succeeded": verified, "pipeline_failure": failed,
            "true_margin": margins[0], "realized_resources": resources,
        }
        case_nonconformity(row)
        rows.append(row)
    return rows


def _run_block(arguments: tuple) -> dict:
    block, partition_name = arguments
    partition = phase_e_partition(partition_name)
    config = _physical_config(partition)
    registry = d6.RandomStreamRegistry()
    started, cpu = perf_counter(), process_time()
    truth = d6.corrected_truth_for_block(partition, block, config, registry)
    truth_timing = d6._runtime_performance_payload(
        perf_counter() - started, process_time() - cpu, d6._peak_rss_bytes(),
    )
    cases = []
    for family in STAGE3_TRUTH_CONDITIONS:
        case = d6._run_internal_check_family(
            truth_condition=family, block=block, truth=truth, partition=partition,
            physical_config=config, registry=registry,
            selector_rule=ProspectiveSelectorRule(),
            frozen_rule=d6.phase_d6_frozen_selector_rule(),
            cost_scenario=d6.PRIMARY_PROSPECTIVE_COST_SCENARIO,
        )
        # Unknown or failed synthetic truth is an incident, not a fit attrition.
        if any(f["stage"].startswith("fixed_policy_reveal:") for f in case["pipeline_failures"]):
            raise RuntimeError("material incident: target reveal or physical energy computation failed")
        calibration_case_records(case)
        cases.append(case)
    audit = registry.audit()
    audit.assert_clean()
    return {
        "block": block, "cases": cases,
        "corrected_random_stream_manifest": d6._stream_manifest_payload(registry.uses),
        "corrected_random_stream_audit": d6._strict_json_value(audit),
        "timing": {"truth_generation": truth_timing,
                   "block_wall_seconds": perf_counter() - started,
                   "block_cpu_seconds": process_time() - cpu,
                   "peak_rss_bytes": d6._peak_rss_bytes()},
    }


def _validate_phase_e_case(
    case: Mapping[str, object],
    *,
    block: int,
    partition: CorrectedPartition,
    physical_config: OperatingDecisionRealismConfig,
    selector_rule: ProspectiveSelectorRule,
    cost_scenario: ProspectiveCostScenario,
) -> None:
    d6._exact_mapping(
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
            "frozen_selection_before_reveal",
            "selection_saved_before_target_reveal",
            "selected_policy_outcome",
            "fixed_policy_results",
            "pipeline_failures",
            "timing",
        },
        "Phase E calibration case",
    )
    if (
        case["block"] != block
        or case["truth_condition"] not in STAGE3_TRUTH_CONDITIONS
        or not isinstance(case.get("device_token"), str)
        or not case.get("device_token")
        or case.get("truth_revealed_only_after_saves") is None
        or case.get("selection_information_boundary")
        != d6._SELECTION_INFORMATION_BOUNDARY
        or not isinstance(case.get("pipeline_failures"), list)
        or case.get("selection_saved_before_target_reveal") is not True
    ):
        raise ValueError("Phase E calibration case evidence is incomplete")
    failures = tuple(
        d6._validate_pipeline_failure(value, "Phase E calibration case pipeline failure")
        for value in case["pipeline_failures"]
    )
    fixed = case["fixed_policy_results"]
    if not isinstance(fixed, Mapping) or set(fixed) != set(POLICY_NAMES):
        raise ValueError("Phase E calibration fixed-policy evidence is incomplete")
    for policy_name, record in fixed.items():
        if not isinstance(record, Mapping) or set(record) not in (
            {"case", "saved_before_reveal", "post_reveal_score"},
            {"case", "saved_before_reveal", "post_reveal_score", "failure"},
        ):
            raise ValueError("Phase E calibration fixed-policy result is malformed")
        d6._validate_fixed_policy_record(
            record,
            policy_name=policy_name,
            truth_condition=str(case["truth_condition"]),
            block=block,
            device_token=str(case["device_token"]),
            physical_config=physical_config,
            campaign=partition.campaign,
            partition_name=partition.name,
            partition_block_count=partition.block_count,
        )
        fixed_failure = record.get("failure")
        if fixed_failure == "decision_not_saved" and not any(
            failure["stage"] == f"fixed_policy_save:{policy_name}"
            for failure in failures
        ):
            raise ValueError("Phase E failed fixed-policy save is unauthenticated")
        if isinstance(fixed_failure, Mapping) and not any(
            failure["stage"] == f"fixed_policy_reveal:{policy_name}"
            and failure["error_type"] == fixed_failure.get("error_type")
            and failure["message"] == fixed_failure.get("message")
            for failure in failures
        ):
            raise ValueError("Phase E failed fixed-policy reveal is unauthenticated")

    complete_payload = case["n16_complete_uncertainty_result"]
    if complete_payload is None:
        matching = tuple(
            failure
            for failure in failures
            if failure["stage"] == "n16_acquisition_or_scoring"
        )
        if len(matching) != 1:
            raise ValueError("Phase E missing N=16 evidence is not canonical")
        for prefix, draw_count in zip(case["primary_prefixes"], d6.PILOT_DRAW_COUNTS):
            failed = d6._validate_failed_prefix_record(
                prefix,
                draw_count=draw_count,
                max_unstable_draws=d6.pilot_max_unstable_draws(draw_count),
            )
            if failed["pipeline_failure"] != matching[0]:
                raise ValueError("Phase E failed prefix does not match its failure")
        sensitivity = d6._validate_failed_prefix_record(
            case["n16_max_unstable_0_sensitivity"],
            draw_count=d6.PILOT_GENERATED_DRAW_COUNT,
            max_unstable_draws=d6.PILOT_SENSITIVITY_MAX_UNSTABLE,
        )
        if sensitivity["pipeline_failure"] != matching[0]:
            raise ValueError("Phase E failed sensitivity does not match its failure")
        if any(
            case[name] is not None
            for name in (
                "admissible_source_models",
                "admissible_source_model_count",
                "selected_policy_at_n16_primary",
                "frozen_selection_before_reveal",
                "selected_policy_outcome",
            )
        ):
            raise ValueError("Phase E failed N=16 case is inconsistent")
        return

    complete = d6._validate_complete_uncertainty_payload(
        complete_payload,
        physical_config=physical_config,
        block=block,
        draw_count=d6.FROZEN_PROSPECTIVE_DRAW_COUNT,
        max_unstable_draws=d6.FROZEN_MAX_UNSTABLE_DRAWS_PER_SOURCE_ACTION,
        campaign=partition.campaign,
        partition=partition.name,
    )
    sources = list(
        complete["acquisition_evidence"]["snapshot"]["admissible_candidate_models"]
    )
    if (
        case["admissible_source_models"] != sources
        or case["admissible_source_model_count"] != len(sources)
    ):
        raise ValueError("Phase E source-model record is inconsistent")
    prefixes = case["primary_prefixes"]
    if not isinstance(prefixes, list) or [
        value.get("draw_count") for value in prefixes
    ] != list(d6.PILOT_DRAW_COUNTS):
        raise ValueError("Phase E calibration prefix evidence is incomplete")
    for prefix, draw_count in zip(prefixes, d6.PILOT_DRAW_COUNTS):
        expected = d6._derive_uncertainty_prefix_payload(
            complete,
            physical_config=physical_config,
            block=block,
            draw_count=draw_count,
            max_unstable_draws=d6.pilot_max_unstable_draws(draw_count),
            campaign=partition.campaign,
            partition=partition.name,
        )
        if isinstance(prefix, Mapping) and "pipeline_failure" in prefix:
            failed = d6._validate_failed_prefix_record(
                prefix,
                draw_count=draw_count,
                max_unstable_draws=d6.pilot_max_unstable_draws(draw_count),
            )
            failure = failed["pipeline_failure"]
            if not any(
                item["stage"] == f"{failure['stage']}_n{draw_count}"
                and item["error_type"] == failure["error_type"]
                and item["message"] == failure["message"]
                for item in failures
            ):
                raise ValueError("Phase E failed prefix has no pipeline failure")
        else:
            d6._validate_authenticated_prefix(
                prefix,
                complete=expected,
                physical_config=physical_config,
                selector_rule=selector_rule,
                cost_scenario=cost_scenario,
            )
    sensitivity_complete = d6._derive_uncertainty_prefix_payload(
        complete,
        physical_config=physical_config,
        block=block,
        draw_count=d6.PILOT_GENERATED_DRAW_COUNT,
        max_unstable_draws=d6.PILOT_SENSITIVITY_MAX_UNSTABLE,
        campaign=partition.campaign,
        partition=partition.name,
    )
    sensitivity = case["n16_max_unstable_0_sensitivity"]
    if isinstance(sensitivity, Mapping) and "pipeline_failure" in sensitivity:
        failed = d6._validate_failed_prefix_record(
            sensitivity,
            draw_count=d6.PILOT_GENERATED_DRAW_COUNT,
            max_unstable_draws=d6.PILOT_SENSITIVITY_MAX_UNSTABLE,
        )
        failure = failed["pipeline_failure"]
        if not any(
            item["stage"] == f"{failure['stage']}_n16_max_unstable_0"
            and item["error_type"] == failure["error_type"]
            and item["message"] == failure["message"]
            for item in failures
        ):
            raise ValueError("Phase E failed sensitivity has no pipeline failure")
    else:
        d6._validate_authenticated_prefix(
            sensitivity,
            complete=sensitivity_complete,
            physical_config=physical_config,
            selector_rule=selector_rule,
            cost_scenario=cost_scenario,
        )
    reference = d6._prefix_by_count(case, d6.FROZEN_PROSPECTIVE_DRAW_COUNT)
    scorecard = reference["costed_scorecard"]
    snapshot = d6._snapshot_from_payload(
        complete["acquisition_evidence"]["snapshot"]
    )
    frozen_rule = d6.phase_d6_frozen_selector_rule()
    evaluations = d6._padded_action_evaluations(
        complete,
        scorecard,
        frozen_rule.development_offsets,
    )
    expected_selection = d6._frozen_selection_payload(
        d6.select_prospective_action(snapshot, evaluations, frozen_rule)
    )
    selected = (
        expected_selection["selected_policy"]
        if expected_selection["selection_succeeded"]
        else None
    )
    if (
        case["frozen_selection_before_reveal"] != expected_selection
        or case["selection_saved_before_target_reveal"] is not True
        or case["selected_policy_at_n16_primary"] != selected
        or case["selected_policy_outcome"]
        != (None if selected is None else fixed[selected])
    ):
        raise ValueError("Phase E selected outcome is inconsistent")


def _validate_phase_e_corrected_stream_records(
    manifest: object, audit: object, *, block: int, partition: CorrectedPartition,
) -> None:
    # This label selects all four fixed-policy stream inventories. It is a
    # validator mode, not the campaign/partition identity, which stays Phase E.
    d6._validate_corrected_stream_records(
        manifest, audit, block=block, label="parent pilot",
        campaign=partition.campaign, partition=partition.name,
    )


def _check_phase_e_corrected_stream_contract(partition: CorrectedPartition) -> None:
    """Check the inventory/validator interface without generating observations."""
    expected = _expected_corrected_stream_manifest(
        0, "parent pilot", campaign=partition.campaign, partition=partition.name,
    )
    registry = d6.RandomStreamRegistry()
    for use in expected:
        registry.register(
            RandomStream(RandomStreamKey(**use["key"])),
            consumer=use["consumer"], pairing_member=use["pairing_member"],
            pairing_id=use["pairing_id"],
        )
    _validate_phase_e_corrected_stream_records(
        d6._stream_manifest_payload(registry.uses),
        d6._strict_json_value(registry.audit()), block=0, partition=partition,
    )


def validate_phase_e_block(block_result: Mapping[str, object], *, partition_name: str) -> dict:
    partition = phase_e_partition(partition_name)
    item = d6._exact_mapping(block_result, {
        "block", "cases", "corrected_random_stream_manifest",
        "corrected_random_stream_audit", "timing",
    }, "Phase E block")
    block = item["block"]
    if not isinstance(block, int) or isinstance(block, bool) or not 0 <= block < partition.block_count:
        raise ValueError("Phase E block index is outside the frozen partition")
    cases = item["cases"]
    if not isinstance(cases, list) or [c.get("truth_condition") for c in cases] != list(STAGE3_TRUTH_CONDITIONS):
        raise ValueError("Phase E requires the complete three-family block")
    _validate_phase_e_corrected_stream_records(
        item["corrected_random_stream_manifest"], item["corrected_random_stream_audit"],
        block=block, partition=partition,
    )
    config = _physical_config(partition)
    for case in cases:
        _validate_phase_e_case(
            case, block=block, partition=partition, physical_config=config,
            selector_rule=ProspectiveSelectorRule(),
            cost_scenario=d6.PRIMARY_PROSPECTIVE_COST_SCENARIO,
        )
        if any(f["stage"].startswith("fixed_policy_reveal:") for f in case["pipeline_failures"]):
            raise ValueError("Phase E material truth incident cannot be scored as ordinary attrition")
        calibration_case_records(case)
    return deepcopy(dict(item))


def _context(repository_root: Path | str, source_revision: str) -> dict:
    root = Path(repository_root).resolve(strict=True)
    d6._validate_revision(source_revision)
    manifest = create_source_manifest(root, PHASE_E_SOURCE_PATHS)
    verify_source_manifest(manifest, root, PHASE_E_SOURCE_PATHS).assert_valid()
    if verify_committed_paths(root, PHASE_E_SOURCE_PATHS).commit != source_revision:
        raise ValueError("Phase E source revision must equal the clean executing HEAD")
    return {
        "source_revision": source_revision,
        "source_manifest": source_manifest_payload(manifest),
        "source_manifest_digest": manifest.digest,
        "protocol_digest": phase_e_protocol_digest(
            source_revision=source_revision, source_manifest_digest=manifest.digest,
        ),
        "protocol": phase_e_protocol_payload(
            source_revision=source_revision, source_manifest_digest=manifest.digest,
        ),
        "runtime_identity": d6.phase_d_runtime_identity(),
    }


def seal_phase_e_block(block_result: Mapping[str, object], *, partition_name: str, context: Mapping[str, object]) -> tuple:
    validated = validate_phase_e_block(block_result, partition_name=partition_name)
    payload = {
        "schema_version": 1, "protocol_version": "phase_e_block_archive_v1",
        "campaign": PROSPECTIVE_CAMPAIGN, "partition": partition_name,
        "source_revision": context["source_revision"],
        "source_manifest_digest": context["source_manifest_digest"],
        "protocol_digest": context["protocol_digest"],
        "runtime_identity": context["runtime_identity"],
        "block": validated["block"], "block_result": validated,
        "scientific_block_digest": d6._digest(
            f"{PHASE_E_DOMAIN}.scientific_block_v1", d6.phase_d_scientific_block_payload(validated),
        ),
    }
    return d6._sealed_payload(payload, domain=f"{PHASE_E_DOMAIN}.block_content_v1")


def load_phase_e_block(path: Path | str, *, partition_name: str, context: Mapping[str, object]) -> dict:
    raw = Path(path).read_bytes()
    payload = json.loads(raw)
    expected, expected_bytes = seal_phase_e_block(
        payload["block_result"], partition_name=partition_name, context=context,
    )
    if raw != expected_bytes or payload != expected:
        raise ValueError("Phase E block is noncanonical, tampered, or bound to another source/partition")
    return expected


def _save_block(path: Path, block_result: Mapping[str, object], *, partition_name: str, context: Mapping[str, object]) -> dict:
    if path.exists():
        raise FileExistsError("Phase E refuses to overwrite a completed block")
    payload, raw = seal_phase_e_block(block_result, partition_name=partition_name, context=context)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise FileExistsError("Phase E unresolved partial block requires incident review")
    with partial.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    # Reload the exact bytes before publishing the completed checkpoint.
    load_phase_e_block(partial, partition_name=partition_name, context=context)
    os.replace(partial, path)
    return payload


def phase_e_preflight(*, repository_root: Path | str, source_revision: str,
                      output_directory: Path | str, partition_name: str) -> dict:
    partition = phase_e_partition(partition_name)
    runtime = d6.validate_executing_phase_d_runtime()
    context = _context(repository_root, source_revision)
    _check_phase_e_corrected_stream_contract(partition)
    root = Path(output_directory).resolve()
    if any((root / name).exists() for name in ("calibration.json", "calibration.txt", "rehearsal.json", "incident.json")):
        raise FileExistsError("Phase E final output or unresolved incident already exists")
    blocks = root / "blocks"
    expected = {f"block-{block:03d}.json" for block in range(partition.block_count)}
    if blocks.is_dir() and any(path.name not in expected for path in blocks.iterdir()):
        raise ValueError("Phase E unexpected or partial checkpoint files require review")
    existing = []
    for block in range(partition.block_count):
        path = blocks / f"block-{block:03d}.json"
        if path.exists():
            loaded = load_phase_e_block(path, partition_name=partition_name, context=context)
            if loaded["block"] != block:
                raise ValueError("Phase E block file name disagrees with block identity")
            existing.append(block)
    return {"schema_version": 1, "protocol_version": "phase_e_preflight_v2",
            "partition": partition.name, "block_count": partition.block_count,
            "output_directory": str(root), "validated_existing_blocks": existing,
            "runtime_identity": runtime, "context": context,
            "corrected_stream_inventory_preflight_passed": True,
            "reserved_generation_authorized": False}


def load_committed_generation_gate(
    path: Path | str, *, gate_repository_root: Path | str, context: Mapping[str, object],
) -> dict:
    """Require a committed gate at a descendant of the tested numerical source.

Run science from the original clean, tested source clone; the later gate commit
documents its CI and disposable results without changing that numerical source.
"""
    root = Path(gate_repository_root).resolve(strict=True)
    gate_path = Path(path).resolve(strict=True)
    relative = gate_path.relative_to(root).as_posix()
    raw = gate_path.read_bytes()
    committed = subprocess.run(("git", "show", f"HEAD:{relative}"), cwd=root,
                               check=True, capture_output=True).stdout
    if raw != committed:
        raise ValueError("Phase E generation gate must match its committed bytes")
    subprocess.run(("git", "merge-base", "--is-ancestor", context["source_revision"], "HEAD"),
                   cwd=root, check=True, capture_output=True)
    return validate_generation_gate(json.loads(raw), context=context)


def validate_generation_gate(gate: Mapping[str, object], *, context: Mapping[str, object]) -> dict:
    if not isinstance(gate, Mapping):
        raise ValueError("Phase E generation gate must be a mapping")
    expected = {
        "protocol_version": "phase_e_generation_gate_v1",
        "source_revision": context["source_revision"],
        "source_manifest_digest": context["source_manifest_digest"],
        "protocol_digest": context["protocol_digest"],
        "runtime_identity": context["runtime_identity"],
        "calibration_generation_authorized": True,
        "reserved_generation_authorized": False,
    }
    if any(gate.get(key) != value for key, value in expected.items()):
        raise ValueError("Phase E generation gate does not bind the executing source")
    checks = gate.get("checks")
    required = {"protocol_tests", "exact_source_ci", "disposable_roundtrip",
                "exact_scientific_replay", "constructed_100_block_roundtrip",
                "process_tree_monitor_preflight"}
    if not isinstance(checks, Mapping) or set(checks) != required or any(checks[k] is not True for k in required):
        raise ValueError("Phase E generation gate is incomplete")
    ci = gate.get("ci", {})
    if ci.get("head_sha") != context["source_revision"] or ci.get("conclusion") != "success" or ci.get("status") != "completed":
        raise ValueError("Phase E generation gate does not record passing exact-source CI")
    return deepcopy(dict(gate))


def _claim_execution_lock(path: Path, *, source_revision: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"schema_version": 1, "pid": os.getpid(),
                   "source_revision": source_revision}, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def recover_interrupted_execution_lock(output_directory: Path | str, *, source_revision: str) -> list:
    """Preserve a stale lock after checking that its recorded process is gone."""
    root = Path(output_directory).resolve()
    recovered = []
    for name in ("execution.lock", "rehearsal.execution.lock"):
        path = root / name
        if not path.exists():
            continue
        record = json.loads(path.read_bytes())
        pid = record.get("pid")
        if (record.get("source_revision") != source_revision or not isinstance(pid, int)
            or isinstance(pid, bool) or pid < 1):
            raise ValueError("interrupted lock does not bind a valid process and the executing source")
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError("recorded coordinator is still alive; cannot recover its lock")
        preserved = root / f"{name}.interrupted-{pid}-{time_ns()}"
        os.rename(path, preserved)
        recovered.append(str(preserved))
    return recovered


def run_phase_e_calibration(
    *, repository_root: Path | str, source_revision: str, output_directory: Path | str,
    generation_gate_path: Path | str, gate_repository_root: Path | str,
    progress: Callable[[str], None] | None = None,
) -> dict:
    preflight = phase_e_preflight(
        repository_root=repository_root, source_revision=source_revision,
        output_directory=output_directory, partition_name=PROSPECTIVE_CALIBRATION_PARTITION,
    )
    context = preflight["context"]
    gate = load_committed_generation_gate(generation_gate_path, gate_repository_root=gate_repository_root, context=context)
    root = Path(output_directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Exclusive lock survives a killed coordinator. An interrupted lock needs
    # deliberate PID/recovery review; parallel invocations may not race a cohort.
    lock = root / "execution.lock"
    _claim_execution_lock(lock, source_revision=source_revision)
    started, cpu = perf_counter(), process_time()
    try:
        completed = {}
        rows_by_block = {}
        block_cpu_seconds = {}
        def retain_checkpoint(block: int, checkpoint: Mapping[str, object]) -> None:
            result = checkpoint["block_result"]
            path = root / "blocks" / f"block-{block:03d}.json"
            completed[block] = {
                "block": block, "path": f"blocks/{path.name}",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "scientific_block_digest": checkpoint["scientific_block_digest"],
            }
            rows_by_block[block] = [row for case in result["cases"] for row in calibration_case_records(case)]
            block_cpu_seconds[block] = result["timing"]["block_cpu_seconds"]
        for block in preflight["validated_existing_blocks"]:
            checkpoint = load_phase_e_block(root / "blocks" / f"block-{block:03d}.json",
                                           partition_name=PROSPECTIVE_CALIBRATION_PARTITION, context=context)
            retain_checkpoint(block, checkpoint)
            del checkpoint
        missing = [b for b in range(CALIBRATION_BLOCKS) if b not in completed]
        if progress:
            progress(f"Phase E: {len(completed)}/100 authenticated blocks resumed; {len(missing)} pending")
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_run_block, (b, PROSPECTIVE_CALIBRATION_PARTITION)): b for b in missing}
            for future in as_completed(futures):
                block = futures.pop(future)
                try:
                    result = future.result()
                    checkpoint = _save_block(root / "blocks" / f"block-{block:03d}.json", result,
                                             partition_name=PROSPECTIVE_CALIBRATION_PARTITION, context=context)
                    retain_checkpoint(block, checkpoint)
                    del result, checkpoint
                except Exception as error:
                    incident = {"schema_version": 1, "source_revision": source_revision,
                                "partition": PROSPECTIVE_CALIBRATION_PARTITION, "block": block,
                                "error_type": type(error).__name__, "message": str(error),
                                "scientific_interpretation_stopped": True,
                                "completed_blocks_retained": sorted(completed)}
                    with (root / "incident.json").open("x", encoding="utf-8") as stream:
                        json.dump(incident, stream, sort_keys=True, indent=2, allow_nan=False)
                    for pending in futures:
                        pending.cancel()
                    raise
                if progress:
                    progress(f"{PROSPECTIVE_CALIBRATION_PARTITION}: validated block {len(completed)}/100")
        rows = [row for block in range(CALIBRATION_BLOCKS) for row in rows_by_block[block]]
        calibration = calibrate_complete_procedures(rows)
        inventory = [completed[b] for b in range(CALIBRATION_BLOCKS)]
        payload = {"schema_version": 1, "protocol_version": "phase_e_calibration_archive_v1",
                   "campaign": PROSPECTIVE_CAMPAIGN, "partition": PROSPECTIVE_CALIBRATION_PARTITION,
                   "context": context, "generation_gate": gate,
                   "block_file_records": inventory,
                   "calibration_case_records": rows, "calibration": calibration,
                   "performance": {"wall_seconds_this_invocation": perf_counter()-started,
                                   "coordinator_cpu_seconds_this_invocation": process_time()-cpu,
                                   "resumed_blocks": len(preflight["validated_existing_blocks"]),
                                   "aggregate_block_cpu_seconds": sum(block_cpu_seconds.values())}}
        payload["scientific_result_digest"] = d6._digest(
            f"{PHASE_E_DOMAIN}.calibration_scientific_v1", {
                "context": context, "scientific_block_digests": [r["scientific_block_digest"] for r in inventory],
                "calibration_case_records": rows, "calibration": calibration,
            },
        )
        sealed, raw = d6._sealed_payload(payload, domain=f"{PHASE_E_DOMAIN}.calibration_content_v1")
        target = root / "calibration.json"
        with target.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        validate_phase_e_calibration_archive(target, repository_root=repository_root)
        report = "Phase E independent calibration\n" + "\n".join(
            f"{p}: q={calibration['procedures'][p]['correction_kelvin']} K; infinite blocks={calibration['procedures'][p]['infinite_block_count']}/100"
            for p in PHASE_E_PROCEDURES
        ) + f"\nStatus: {calibration['scientific_status']}\nReserved generation: NOT AUTHORIZED\n"
        (root / "calibration.txt").write_text(report, encoding="utf-8")
        (root / "calibration.sha256").write_text(
            hashlib.sha256(raw).hexdigest() + "  calibration.json\n" +
            hashlib.sha256(report.encode()).hexdigest() + "  calibration.txt\n", encoding="utf-8",
        )
        return sealed
    finally:
        lock.unlink(missing_ok=True)


def validate_phase_e_calibration_archive(path: Path | str, *, repository_root: Path | str) -> dict:
    target = Path(path).resolve(strict=True)
    raw, item = target.read_bytes(), json.loads(target.read_bytes())
    d6._exact_mapping(item, {
        "schema_version", "protocol_version", "campaign", "partition", "context",
        "generation_gate", "block_file_records", "calibration_case_records",
        "calibration", "performance", "scientific_result_digest",
        "archive_content_digest", "archive_size_bytes",
    }, "Phase E calibration archive")
    context = _context(repository_root, item["context"]["source_revision"])
    if item["context"] != context or item["partition"] != PROSPECTIVE_CALIBRATION_PARTITION or item["campaign"] != PROSPECTIVE_CAMPAIGN:
        raise ValueError("Phase E calibration source/partition header is invalid")
    validate_generation_gate(item["generation_gate"], context=context)
    if item["schema_version"] != 1 or item["protocol_version"] != "phase_e_calibration_archive_v1":
        raise ValueError("Phase E calibration schema is invalid")
    records = item["block_file_records"]
    if not isinstance(records, list) or [r.get("block") for r in records] != list(range(CALIBRATION_BLOCKS)):
        raise ValueError("Phase E calibration checkpoint inventory is incomplete")
    expected_files = {f"block-{b:03d}.json" for b in range(CALIBRATION_BLOCKS)}
    if {p.name for p in (target.parent / "blocks").iterdir()} != expected_files:
        raise ValueError("Phase E calibration has missing or unexpected checkpoint files")
    rows = []
    for record in records:
        d6._exact_mapping(record, {"block", "path", "sha256", "scientific_block_digest"}, "Phase E checkpoint inventory row")
        b = record["block"]
        if not isinstance(b, int) or isinstance(b, bool):
            raise ValueError("Phase E checkpoint inventory index must be an integer")
        expected_path = f"blocks/block-{b:03d}.json"
        if record["path"] != expected_path:
            raise ValueError("Phase E inventory path escapes its prescribed block")
        checkpoint_path = target.parent / expected_path
        checkpoint = load_phase_e_block(checkpoint_path, partition_name=PROSPECTIVE_CALIBRATION_PARTITION, context=context)
        if (checkpoint["block"] != b or record["scientific_block_digest"] != checkpoint["scientific_block_digest"]
            or record["sha256"] != hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()):
            raise ValueError("Phase E calibration checkpoint binding is invalid")
        rows.extend(row for c in checkpoint["block_result"]["cases"] for row in calibration_case_records(c))
        del checkpoint
    calibration = calibrate_complete_procedures(rows)
    if item["calibration_case_records"] != rows or item["calibration"] != calibration:
        raise ValueError("Phase E calibration does not recompute from its retained cases")
    scientific = d6._digest(f"{PHASE_E_DOMAIN}.calibration_scientific_v1", {
        "context": context, "scientific_block_digests": [r["scientific_block_digest"] for r in records],
        "calibration_case_records": rows, "calibration": calibration,
    })
    if item["scientific_result_digest"] != scientific:
        raise ValueError("Phase E calibration scientific digest is invalid")
    material = dict(item)
    material.pop("archive_content_digest", None)
    material.pop("archive_size_bytes", None)
    sealed, expected_raw = d6._sealed_payload(material, domain=f"{PHASE_E_DOMAIN}.calibration_content_v1")
    if raw != expected_raw or item != sealed:
        raise ValueError("Phase E calibration bytes are noncanonical or tampered")
    return item


def _constructed_score_records(n: int = 100) -> list:
    """Synthetic score fixture, explicitly inadmissible as scientific evidence."""
    return [{"block": b, "truth_family": f, "procedure": p,
             "selected_action": STOP_NOW, "raw_interval": {"lower": -1., "upper": 1.},
             "development_offset_kelvin": 0., "true_margin": 1. + b / 128.,
             "verification_succeeded": True, "pipeline_failure": False}
            for b in range(n) for f in STAGE3_TRUTH_CONDITIONS for p in PHASE_E_PROCEDURES]


def constructed_phase_e_archive_probe(seed: Mapping[str, object], *, output_directory: Path | str) -> dict:
    """Exercise a 100-block-sized raw archive without creating scientific cases."""
    directory = Path(output_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "constructed-100-block-archive.json"
    if path.exists():
        raise FileExistsError("Phase E constructed archive already exists")
    seed_bytes = d6._canonical_bytes(seed["block_result"])
    seed_hash = hashlib.sha256(seed_bytes).hexdigest()
    started, cpu = perf_counter(), process_time()
    # Streaming the write avoids making a second full-size byte string.
    with path.open("xb") as stream:
        stream.write(b'{"scientific_use":"prohibited_constructed_resource_evidence_only","constructed_blocks":[')
        for block in range(100):
            if block:
                stream.write(b",")
            stream.write(seed_bytes)
        stream.write(b"]}\n")
        stream.flush()
        os.fsync(stream.fileno())
    restored = json.loads(path.read_bytes())
    if restored.get("scientific_use") != "prohibited_constructed_resource_evidence_only" or len(restored["constructed_blocks"]) != 100:
        raise ValueError("Phase E constructed archive marker/size is invalid")
    for block in restored["constructed_blocks"]:
        if hashlib.sha256(d6._canonical_bytes(block)).hexdigest() != seed_hash:
            raise ValueError("Phase E constructed archive failed byte roundtrip")
    del restored
    fixture = _constructed_score_records()
    calibration = calibrate_complete_procedures(fixture)
    recalculated = calibrate_complete_procedures(json.loads(d6._canonical_bytes(fixture)))
    if recalculated != calibration:
        raise ValueError("Phase E constructed calibration failed roundtrip")
    file_hash = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            file_hash.update(chunk)
    return {"scientific_use": "prohibited_constructed_resource_evidence_only",
            "constructed_block_count": 100, "constructed_calibration_record_count": len(fixture),
            "archive_size_bytes": path.stat().st_size, "sha256": file_hash.hexdigest(),
            "seed_block_sha256": seed_hash, "roundtrip_passed": True,
            "calibration_order_statistic_rank": calibration["order_statistic_rank"],
            "wall_seconds": perf_counter()-started, "cpu_seconds": process_time()-cpu,
            "coordinator_peak_rss_bytes": d6._peak_rss_bytes()}


def _record_phase_e_rehearsal_incident(
    directory: Path, *, context: Mapping[str, object], repeat: int,
    error: Exception, block_result: Mapping[str, object] | None,
) -> None:
    """Retain rejected computations for diagnosis, never as accepted evidence."""
    failed_result = None
    if block_result is not None:
        name = f"incident-repeat-{repeat+1}.unvalidated.json"
        raw = d6._canonical_bytes({
            "scientific_use": "prohibited_unvalidated_incident_evidence_only",
            "context": context, "partition": PHASE_E_REHEARSAL_PARTITION,
            "block_result": block_result,
        })
        with (directory / name).open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        failed_result = {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}
    incident = {
        "schema_version": 1, "source_revision": context["source_revision"],
        "partition": PHASE_E_REHEARSAL_PARTITION, "repeat": repeat+1,
        "error_type": type(error).__name__, "message": str(error),
        "failed_result": failed_result, "scientific_interpretation_stopped": True,
        "calibration_partition_opened": False, "reserved_partition_opened": False,
    }
    with (directory / "incident.json").open("x", encoding="utf-8") as stream:
        json.dump(incident, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _run_phase_e_rehearsal_unlocked(
    *, repository_root: Path | str, source_revision: str, output_directory: Path | str,
    progress: Callable[[str], None] | None = None,
) -> dict:
    preflight = phase_e_preflight(
        repository_root=repository_root, source_revision=source_revision,
        output_directory=output_directory, partition_name=PHASE_E_REHEARSAL_PARTITION,
    )
    context = preflight["context"]
    directory = Path(output_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    originals = [directory / "blocks" / "block-000.json", directory / "replay-block-000.json"]
    completed = {}
    for index, path in enumerate(originals):
        if path.exists():
            completed[index] = load_phase_e_block(path, partition_name=PHASE_E_REHEARSAL_PARTITION, context=context)
    # Four workers exist, matching the frozen pool size. Two compute the same
    # disposable block independently; only this declared replay reuses its keys.
    with ProcessPoolExecutor(max_workers=4) as executor:
        pending = {executor.submit(_run_block, (0, PHASE_E_REHEARSAL_PARTITION)): i
                   for i in range(2) if i not in completed}
        for future in as_completed(pending):
            index = pending[future]
            result = None
            try:
                result = future.result()
                completed[index] = _save_block(originals[index], result,
                                               partition_name=PHASE_E_REHEARSAL_PARTITION, context=context)
            except Exception as error:
                _record_phase_e_rehearsal_incident(
                    directory, context=context, repeat=index,
                    error=error, block_result=result,
                )
                for other in pending:
                    other.cancel()
                raise
            if progress:
                progress(f"Phase E disposable computation {index+1}/2 saved and validated")
    scientific_equal = completed[0]["scientific_block_digest"] == completed[1]["scientific_block_digest"]
    if not scientific_equal:
        raise ValueError("Phase E disposable exact scientific replay failed")
    if progress:
        progress("Phase E disposable scientific replay matches; exercising constructed 100-block storage")
    probe = constructed_phase_e_archive_probe(completed[0], output_directory=directory)
    record = {"schema_version": 1, "protocol_version": "phase_e_rehearsal_v1",
              "context": context, "partition": PHASE_E_REHEARSAL_PARTITION,
              "scientific_use": "disposable_engineering_only",
              "block_roundtrip_passed": True, "exact_scientific_replay_passed": True,
              "scientific_block_digest": completed[0]["scientific_block_digest"],
              "original_sha256": hashlib.sha256(originals[0].read_bytes()).hexdigest(),
              "replay_sha256": hashlib.sha256(originals[1].read_bytes()).hexdigest(),
              "constructed_archive_probe": probe,
              "calibration_partition_opened": False, "reserved_partition_opened": False}
    with (directory / "rehearsal.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return record


def run_phase_e_rehearsal(
    *, repository_root: Path | str, source_revision: str, output_directory: Path | str,
    progress: Callable[[str], None] | None = None,
) -> dict:
    directory = Path(output_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "rehearsal.execution.lock"
    _claim_execution_lock(lock, source_revision=source_revision)
    try:
        return _run_phase_e_rehearsal_unlocked(
            repository_root=repository_root, source_revision=source_revision,
            output_directory=directory, progress=progress,
        )
    finally:
        lock.unlink(missing_ok=True)
