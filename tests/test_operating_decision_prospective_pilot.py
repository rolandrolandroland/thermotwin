from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from thermotwin.studies.operating_decision import (
    FIXED_FACE_TEMPERATURE,
    FIXED_THERMAL,
    FIXED_VOLTAGE,
    POLICY_NAMES,
    STOP_NOW,
    default_fixed_policies,
)
from thermotwin.studies.operating_decision_prospective import ProspectiveSelectorRule
from thermotwin.studies.operating_decision_prospective_costs import (
    PRIMARY_PROSPECTIVE_COST_SCENARIO,
)
from thermotwin.studies.operating_decision_random_streams import RandomStreamRegistry
from thermotwin.studies.operating_decision_realism import (
    OperatingDecisionRealismConfig,
)
from thermotwin.studies.operating_decision_replication import CorrectedPartition
import thermotwin.studies.operating_decision_prospective_pilot as pilot
import thermotwin.reports.operating_decision_prospective_pilot as pilot_cli
from thermotwin.reports.operating_decision_prospective_pilot import (
    _committed_source_revision,
    _require_clean_head,
    _require_project_root,
)


def _prefix(
    draw_count,
    selected,
    *,
    voltage_utility=0.96,
    eligibility=None,
    max_unstable=None,
):
    if max_unstable is None:
        max_unstable = pilot.pilot_max_unstable_draws(draw_count)
    if eligibility is None:
        eligibility = {
            name: {"eligible": True, "failure_reason": None}
            for name in POLICY_NAMES
        }
    evaluations = [
        {
            "policy_name": name,
            "eligible": eligibility[name]["eligible"],
            "utility_per_cost": (
                1.0
                if name == FIXED_THERMAL
                else voltage_utility
                if name == FIXED_VOLTAGE
                else 0.2
            ),
        }
        for name in POLICY_NAMES
    ]
    return {
        "draw_count": draw_count,
        "max_unstable_draws_per_source_action": max_unstable,
        "choice_token": f"action:{selected}",
        "eligibility": eligibility,
        "draw_diagnostics": {
            "available": True,
            "retained_draw_count": 6 * draw_count,
            "failed_draw_count": 0,
            "candidate_transition_draw_count": 0,
            "candidate_loss_draw_count": 0,
            "candidate_recovery_draw_count": 0,
            "candidate_loss_status_counts": {},
            "by_action": {},
        },
        "selection": {
            "selected_policy": selected,
            "action_evaluations": evaluations,
        },
        "uncertainty_summary": {
            "config": {
                "draw_count": draw_count,
                "max_unstable_draws_per_source_action": max_unstable,
            }
        },
    }


def _blocks(*, changed_regret=0.04, pipeline_failure=False):
    voltage_utility = 1.0 - changed_regret
    blocks = []
    for block in range(4):
        cases = []
        for family_index, family in enumerate(pilot.STAGE3_TRUTH_CONDITIONS):
            changed = block == 3 and family_index == 2
            reference = _prefix(16, FIXED_THERMAL, voltage_utility=voltage_utility)
            n4 = _prefix(
                4,
                FIXED_VOLTAGE if changed else FIXED_THERMAL,
                voltage_utility=voltage_utility,
            )
            n8 = _prefix(8, FIXED_THERMAL, voltage_utility=voltage_utility)
            sensitivity = deepcopy(reference)
            sensitivity["max_unstable_draws_per_source_action"] = 0
            sensitivity["uncertainty_summary"]["config"][
                "max_unstable_draws_per_source_action"
            ] = 0
            cases.append(
                {
                    "block": block,
                    "truth_condition": family,
                    "primary_prefixes": [n4, n8, reference],
                    "n16_max_unstable_0_sensitivity": sensitivity,
                    "pipeline_failures": (
                        [{"stage": "test"}]
                        if pipeline_failure and block == 0 and family_index == 0
                        else []
                    ),
                    "timing": {"case_wall_seconds": 999.0},
                }
            )
        blocks.append(
            {
                "block": block,
                "cases": cases,
                "timing": {
                    "block_wall_seconds": float(block + 1),
                    "block_cpu_seconds": float(block + 2),
                    "peak_rss_bytes": 1000 + block,
                },
            }
        )
    return blocks


def _pilot_physical_config():
    partition = CorrectedPartition(
        pilot.PROSPECTIVE_PILOT_PARTITION,
        pilot.PILOT_BLOCK_COUNT,
        pilot.PROSPECTIVE_CAMPAIGN,
    )
    return pilot.corrected_partition_config(
        partition,
        pilot.CORRECTED_REPLICATION_CONFIG,
    )


def _corrected_stream_records(block, *, followup=False):
    label = "N=32" if followup else "parent pilot"
    manifest = pilot._expected_corrected_stream_manifest(block, label)
    return manifest, _corrected_audit_from_manifest(manifest)


def _corrected_audit_from_manifest(manifest):
    registry = pilot.RandomStreamRegistry()
    for item in manifest:
        stream = pilot.RandomStream(pilot.RandomStreamKey(**item["key"]))
        registry.register(
            stream,
            consumer=item["consumer"],
            pairing_member=item["pairing_member"],
            pairing_id=item["pairing_id"],
        )
    return pilot._strict_json_value(registry.audit())


def _raw_width(policy_name, draw_index):
    if policy_name == FIXED_THERMAL:
        return 0.5
    if policy_name == FIXED_VOLTAGE:
        return 0.5 if draw_index < 8 else 2.0
    if policy_name == FIXED_FACE_TEMPERATURE:
        return 1.8
    raise AssertionError(policy_name)


def _complete_uncertainty(block, draw_count):
    physical_config = _pilot_physical_config()
    config = pilot.ProspectiveUncertaintyConfig(
        draw_count=draw_count,
        max_unstable_draws_per_source_action=(
            pilot.pilot_max_unstable_draws(draw_count)
        ),
    )
    source_model = "four_state"
    excluded_model = "five_state"
    snapshot = {
        "candidate_models": [excluded_model, source_model],
        "admissible_candidate_models": [source_model],
        "excluded_candidates": [
            {"model_name": excluded_model, "reason": "fit_reached_bound"}
        ],
        "failed_candidate_models": [],
        "model_intervals": [
            {
                "model_name": source_model,
                "estimate": 0.0,
                "local_standard_error": 0.5,
                "lower": -1.0,
                "upper": 1.0,
            }
        ],
        "provisional_margin_envelope": {"lower": -1.0, "upper": 1.0},
        "selection_failure_reason": None,
    }
    fits = []
    for model_name in (excluded_model, source_model):
        spec = pilot._parameter_spec(model_name, False, physical_config)
        size = len(spec.names)
        offsets = [0.0] * size
        if model_name == excluded_model:
            offsets[0] = spec.log_bounds[0][0]
        decoded = pilot._decoded_parameters(model_name, tuple(offsets), spec)
        fits.append(
            {
                "model_name": model_name,
                "log_multipliers": offsets,
                "parameter_names": list(spec.names),
                "physical_values": list(decoded[0]),
                "interface_mass": decoded[1],
                "series_resistance": decoded[2],
                "face_sensor": None,
                "objective": 1.0,
                "covariance": [
                    [1.0e-4 if row == column else 0.0 for column in range(size)]
                    for row in range(size)
                ],
                "reached_bound": model_name == excluded_model,
                "evaluation_count": 1,
                "converged": model_name == source_model,
                "termination_reason": (
                    "scaled_projected_gradient_tolerance"
                    if model_name == source_model
                    else "fixed_iteration_limit"
                ),
                "completed_iterations": physical_config.sensor.fit_iterations,
                "accepted_iterations": 1,
                "scaled_gradient_infinity_norm": (
                    0.0 if model_name == source_model else 1.0
                ),
                "last_step_infinity_norm": (
                    0.0 if model_name == source_model else 1.0
                ),
                "last_relative_objective_reduction": (
                    0.0 if model_name == source_model else 1.0
                ),
            }
        )
    initial_regime = pilot.initial_acquisition_regime()
    common_initial_run = {
        "regime": {
            "name": initial_regime.name,
            "phase": initial_regime.phase,
            "transition_times": list(initial_regime.current.transition_times),
            "current_values": list(initial_regime.current.values),
            "channels": list(initial_regime.channels),
        },
        "temporary_face_sensor": False,
        "observations": [
            {
                "channel": channel,
                "time": time,
                "value": 300.0,
            }
            for time in pilot.regular_measurement_times(
                pilot.RUN_DURATION_SECONDS,
                physical_config.sensor.sampling_interval,
            )
            for channel in (pilot.COLD_EXCHANGER, pilot.HOT_EXCHANGER)
        ],
    }
    fit_set = {"fits": fits, "failures": []}
    final_regime = {
        "name": "untouched_final_operating_schedule",
        "phase": "final_evaluation",
        "transition_times": list(physical_config.final_current.transition_times),
        "current_values": list(physical_config.final_current.values),
        "channels": [],
    }
    evidence_material = {
        "domain": pilot._ACQUISITION_EVIDENCE_DOMAIN,
        "common_initial_run": common_initial_run,
        "fit_set": fit_set,
        "snapshot": snapshot,
        "final_regime": final_regime,
        "physical_protocol_digest": pilot.corrected_physical_protocol_digest(
            physical_config
        ),
    }
    evidence_digest = pilot._uncertainty_canonical_digest(evidence_material)
    evidence = {
        "schema_version": pilot.PROSPECTIVE_UNCERTAINTY_SCHEMA_VERSION,
        "protocol_version": pilot.PROSPECTIVE_UNCERTAINTY_PROTOCOL_VERSION,
        "physical_config": pilot._physical_config_payload(physical_config),
        "physical_protocol_digest": evidence_material[
            "physical_protocol_digest"
        ],
        "evidence_digest": evidence_digest,
        "common_initial_run": evidence_material["common_initial_run"],
        "fit_set": evidence_material["fit_set"],
        "snapshot": snapshot,
        "final_regime": evidence_material["final_regime"],
    }
    source_fit_payload = next(
        fit for fit in fits if fit["model_name"] == source_model
    )
    source_fit = pilot._validated_candidate_fit_from_payload(
        source_fit_payload,
        physical_config,
    )
    namespace = pilot.ProspectiveRandomStreamNamespace(
        campaign=pilot.PROSPECTIVE_CAMPAIGN,
        partition=pilot.PROSPECTIVE_PILOT_PARTITION,
        block=block,
        acquisition_evidence_digest=evidence_digest,
    )
    parameter_draws = {}
    probe_draws = {}
    for draw_index in range(draw_count):
        registry = pilot.ProspectiveRandomStreamRegistry()
        parameter_draws[draw_index] = list(
            pilot._generator_parameter_draw(
                source_fit,
                namespace,
                draw_index,
                physical_config,
                config,
                registry,
            )
        )
        probe_draws[draw_index] = list(
            pilot._face_probe_draw(
                namespace,
                source_model,
                draw_index,
                physical_config,
                config,
                registry,
            )
        )
    draws = []
    for policy_name in sorted(name for name in POLICY_NAMES if name != STOP_NOW):
        for draw_index in range(draw_count):
            width = _raw_width(policy_name, draw_index)
            interval = {
                "model_name": source_model,
                "estimate": 0.0,
                "local_standard_error": width / 4.0,
                "lower": -width / 2.0,
                "upper": width / 2.0,
            }
            draws.append(
                {
                    "policy_name": policy_name,
                    "generator_model": source_model,
                    "draw_index": draw_index,
                    "generator_log_offsets": parameter_draws[draw_index],
                    "face_probe_log_offsets": (
                        probe_draws[draw_index]
                        if policy_name == FIXED_FACE_TEMPERATURE
                        else None
                    ),
                    "candidate_outcomes": [
                        {
                            "model_name": excluded_model,
                            "status": "fit_reached_bound",
                            "objective": 1.0,
                            "interval": None,
                        },
                        {
                            "model_name": source_model,
                            "status": "admissible",
                            "objective": 1.0,
                            "interval": interval,
                        },
                    ],
                    "initially_admissible_became_inadmissible": [],
                    "initially_excluded_became_admissible": [],
                    "stable": True,
                    "failed": False,
                    "failure_stage": None,
                    "failure_model": None,
                    "failure_type": None,
                    "raw_after_width": width,
                    "scored_after_width": width,
                    "synthetic_observation_digest": hashlib.sha256(
                        f"{block}/{policy_name}/{draw_index}".encode()
                    ).hexdigest(),
                }
            )
    namespace = pilot.ProspectiveRandomStreamNamespace(
        campaign=pilot.PROSPECTIVE_CAMPAIGN,
        partition=pilot.PROSPECTIVE_PILOT_PARTITION,
        block=block,
        acquisition_evidence_digest=evidence_digest,
    )
    stream_uses = []
    policies = {
        policy.name: policy
        for policy in default_fixed_policies()
        if policy.name != STOP_NOW
    }
    for draw_index in range(draw_count):
        stream = pilot.ProspectiveRandomStream(
            pilot.ProspectiveRandomStreamKey.from_namespace(
                namespace,
                generator_model=source_model,
                draw_index=draw_index,
                purpose="parameter_draw",
            )
        )
        key = {"domain": pilot.PROSPECTIVE_RANDOM_STREAM_DOMAIN, **stream.key.__dict__}
        for policy_name in POLICY_NAMES:
            if policy_name == STOP_NOW:
                continue
            stream_uses.append(
                {
                    "key": key,
                    "seed": str(stream.seed),
                    "consumer": (
                        f"{source_model}/{draw_index}/{policy_name}/"
                        "physical_parameters"
                    ),
                    "shared_for_action": policy_name,
                }
            )
        probe = pilot.ProspectiveRandomStream(
            pilot.ProspectiveRandomStreamKey.from_namespace(
                namespace,
                generator_model=source_model,
                draw_index=draw_index,
                purpose="probe_draw",
                action=FIXED_FACE_TEMPERATURE,
            )
        )
        stream_uses.append(
            {
                "key": {
                    "domain": pilot.PROSPECTIVE_RANDOM_STREAM_DOMAIN,
                    **probe.key.__dict__,
                },
                "seed": str(probe.seed),
                "consumer": (
                    f"{source_model}/{draw_index}/"
                    f"{FIXED_FACE_TEMPERATURE}/probe"
                ),
                "shared_for_action": None,
            }
        )
        for policy_name, policy in policies.items():
            for regime in policy.additional_regimes:
                for channel in regime.channels:
                    for purpose in ("run_bias", "white_noise"):
                        observation = pilot.ProspectiveRandomStream(
                            pilot.ProspectiveRandomStreamKey.from_namespace(
                                namespace,
                                generator_model=source_model,
                                draw_index=draw_index,
                                purpose=purpose,
                                action=policy_name,
                                run=regime.name,
                                channel=channel,
                            )
                        )
                        stream_uses.append(
                            {
                                "key": {
                                    "domain": pilot.PROSPECTIVE_RANDOM_STREAM_DOMAIN,
                                    **observation.key.__dict__,
                                },
                                "seed": str(observation.seed),
                                "consumer": (
                                    f"{source_model}/{draw_index}/{policy_name}/"
                                    f"{regime.name}/{channel}/{purpose}"
                                ),
                                "shared_for_action": None,
                            }
                        )
    actions = pilot._recomputed_action_uncertainties(
        draws,
        source_models=[source_model],
        baseline_width=2.0,
        config=config,
    )
    audit = pilot._stream_audit_from_payload(
        stream_uses,
        draws=draws,
        source_models=[source_model],
        draw_count=draw_count,
        acquisition_evidence_digest=evidence_digest,
        block=block,
    )
    protocol_digest = pilot.prospective_uncertainty_protocol_digest(
        physical_config,
        config,
    )
    payload = {
        "schema_version": pilot.PROSPECTIVE_UNCERTAINTY_SCHEMA_VERSION,
        "protocol_version": pilot.PROSPECTIVE_UNCERTAINTY_PROTOCOL_VERSION,
        "protocol_digest": protocol_digest,
        "result_digest": None,
        "physical_protocol_digest": pilot.corrected_physical_protocol_digest(
            physical_config
        ),
        "acquisition_evidence_digest": evidence_digest,
        "physical_config": pilot._physical_config_payload(physical_config),
        "acquisition_evidence": evidence,
        "config": pilot._uncertainty_config_payload_for(config),
        "action_uncertainties": actions,
        "draw_outcomes": draws,
        "stream_uses": stream_uses,
        "stream_audit": audit,
    }
    payload["result_digest"] = pilot._uncertainty_canonical_digest(
        {
            "domain": "thermotwin.prospective_uncertainty_result",
            "config": payload["config"],
            "physical_config": payload["physical_config"],
            "acquisition_evidence_digest": evidence_digest,
            "protocol_digest": protocol_digest,
            "action_uncertainties": actions,
            "draw_outcomes": draws,
            "stream_uses": stream_uses,
            "stream_audit": audit,
        }
    )
    return payload


def _scorecard(complete):
    physical_config = _pilot_physical_config()
    selector_rule = ProspectiveSelectorRule()
    scenario = PRIMARY_PROSPECTIVE_COST_SCENARIO
    resources = pilot._build_action_resources(physical_config, scenario)
    costs, energy_reference, time_reference = pilot._build_action_costs(
        resources,
        scenario,
    )
    evaluations = []
    for action, cost in zip(complete["action_uncertainties"], costs):
        if action["policy_name"] == STOP_NOW:
            evaluations.append(pilot.ProspectiveActionEvaluation(STOP_NOW, True))
        elif not action["eligible"]:
            evaluations.append(
                pilot.ProspectiveActionEvaluation(
                    action["policy_name"],
                    False,
                    failure_reason=action["failure_reason"],
                )
            )
        else:
            evaluations.append(
                pilot.ProspectiveActionEvaluation(
                    action["policy_name"],
                    action["eligible"],
                    failure_reason=action["failure_reason"],
                    uncertainty_before=action["uncertainty_before"],
                    expected_uncertainty_after=action[
                        "expected_uncertainty_after"
                    ],
                    declared_cost=cost.declared_cost,
                    prospective_draw_count=action["prospective_draw_count"],
                    raw_uncertainty_before=action["uncertainty_before"],
                    raw_expected_uncertainty_after=action[
                        "expected_uncertainty_after"
                    ],
                    development_offset=0.0,
                )
            )
    protocol_digest = pilot.prospective_cost_protocol_digest(
        physical_config,
        pilot.ProspectiveUncertaintyConfig(
            draw_count=complete["config"]["draw_count"],
            max_unstable_draws_per_source_action=complete["config"][
                "max_unstable_draws_per_source_action"
            ],
        ),
        scenario,
        selector_rule,
    )
    payload = {
        "schema_version": pilot.PROSPECTIVE_COST_SCHEMA_VERSION,
        "protocol_version": pilot.PROSPECTIVE_COST_PROTOCOL_VERSION,
        "formula": pilot.PROSPECTIVE_COST_FORMULA,
        "padded_scoring_protocol": pilot.PROSPECTIVE_PADDED_SCORING_PROTOCOL,
        "energy_convention": pilot.PROSPECTIVE_COST_ENERGY_CONVENTION,
        "time_convention": pilot.PROSPECTIVE_COST_TIME_CONVENTION,
        "instrumentation_convention": pilot.PROSPECTIVE_COST_INSTRUMENTATION_CONVENTION,
        "nominal_energy_protocol": pilot.NOMINAL_SELECTION_COST_PROTOCOL,
        "physical_protocol_digest": complete["physical_protocol_digest"],
        "uncertainty_protocol_digest": complete["protocol_digest"],
        "uncertainty_result_digest": complete["result_digest"],
        "action_catalog_digest": pilot.prospective_action_catalog_digest(),
        "selector_protocol_digest": pilot.prospective_selector_protocol_digest(
            selector_rule
        ),
        "selector_rule": pilot.prospective_selector_rule_payload(selector_rule),
        "acquisition_evidence_digest": complete["acquisition_evidence_digest"],
        "protocol_digest": protocol_digest,
        "result_digest": None,
        "scenario": pilot._scenario_payload(scenario),
        "energy_reference_joules": energy_reference,
        "bench_time_reference_seconds": time_reference,
        "action_resources": [pilot._action_resources_payload(x) for x in resources],
        "action_costs": [pilot._action_cost_payload(x) for x in costs],
        "action_evaluations": [
            pilot._cost_action_evaluation_payload(x) for x in evaluations
        ],
    }
    payload["result_digest"] = pilot._uncertainty_canonical_digest(
        {
            "domain": "thermotwin.prospective_costed_scorecard",
            "uncertainty_result_digest": complete["result_digest"],
            "acquisition_evidence_digest": complete["acquisition_evidence_digest"],
            "scenario": payload["scenario"],
            "selector_rule": payload["selector_rule"],
            "protocol_digest": protocol_digest,
            "energy_reference_joules": energy_reference,
            "bench_time_reference_seconds": time_reference,
            "action_resources": payload["action_resources"],
            "action_costs": payload["action_costs"],
            "action_evaluations": payload["action_evaluations"],
        }
    )
    return payload, tuple(evaluations)


def _authenticated_prefix(complete):
    scorecard, evaluations = _scorecard(complete)
    selection = pilot.select_prospective_action(
        pilot._snapshot_from_complete(complete),
        evaluations,
        ProspectiveSelectorRule(),
    )
    return {
        "draw_count": complete["config"]["draw_count"],
        "max_unstable_draws_per_source_action": complete["config"][
            "max_unstable_draws_per_source_action"
        ],
        "choice_token": pilot._choice_token(selection, None),
        "eligibility": {
            action["policy_name"]: {
                "eligible": action["eligible"],
                "failure_reason": action["failure_reason"],
            }
            for action in complete["action_uncertainties"]
        },
        "draw_diagnostics": pilot._draw_diagnostics_from_payload(
            complete["draw_outcomes"]
        ),
        "selection": pilot._selection_payload(selection),
        "uncertainty_summary": pilot._uncertainty_summary_from_complete(complete),
        "costed_scorecard": scorecard,
    }


def _reseal_uncertainty(complete, block):
    snapshot = complete["acquisition_evidence"]["snapshot"]
    complete["stream_audit"] = pilot._stream_audit_from_payload(
        complete["stream_uses"],
        source_models=snapshot["admissible_candidate_models"],
        draw_count=complete["config"]["draw_count"],
        acquisition_evidence_digest=complete["acquisition_evidence_digest"],
        block=block,
    )
    complete["result_digest"] = pilot._uncertainty_canonical_digest(
        {
            "domain": "thermotwin.prospective_uncertainty_result",
            "config": complete["config"],
            "physical_config": complete["physical_config"],
            "acquisition_evidence_digest": complete[
                "acquisition_evidence_digest"
            ],
            "protocol_digest": complete["protocol_digest"],
            "action_uncertainties": complete["action_uncertainties"],
            "draw_outcomes": complete["draw_outcomes"],
            "stream_uses": complete["stream_uses"],
            "stream_audit": complete["stream_audit"],
        }
    )
    return complete


def _recompute_uncertainty_actions(complete, block):
    physical_config = _pilot_physical_config()
    snapshot = complete["acquisition_evidence"]["snapshot"]
    config = pilot.ProspectiveUncertaintyConfig(
        draw_count=complete["config"]["draw_count"],
        max_unstable_draws_per_source_action=complete["config"][
            "max_unstable_draws_per_source_action"
        ],
    )
    envelope = snapshot["provisional_margin_envelope"]
    complete["action_uncertainties"] = pilot._recomputed_action_uncertainties(
        complete["draw_outcomes"],
        source_models=snapshot["admissible_candidate_models"],
        baseline_width=envelope["upper"] - envelope["lower"],
        config=config,
        initially_admissible=snapshot["admissible_candidate_models"],
        initially_excluded=[
            item["model_name"] for item in snapshot["excluded_candidates"]
        ],
        physical_config=physical_config,
    )
    return _reseal_uncertainty(complete, block)


def _build_n32_parent_payload():
    source_revision = "a" * 40
    physical_config = _pilot_physical_config()
    blocks = []
    case_index = 0
    for block in range(pilot.PILOT_BLOCK_COUNT):
        complete = _complete_uncertainty(block, 16)
        primary = []
        for draw_count in pilot.PILOT_DRAW_COUNTS:
            derived = pilot._derive_uncertainty_prefix_payload(
                complete,
                physical_config=physical_config,
                block=block,
                draw_count=draw_count,
                max_unstable_draws=pilot.pilot_max_unstable_draws(draw_count),
            )
            primary.append(_authenticated_prefix(derived))
        zero_failure = pilot._derive_uncertainty_prefix_payload(
            complete,
            physical_config=physical_config,
            block=block,
            draw_count=16,
            max_unstable_draws=0,
        )
        cases = []
        for family in pilot.STAGE3_TRUTH_CONDITIONS:
            partition = CorrectedPartition(
                pilot.PROSPECTIVE_PILOT_PARTITION,
                pilot.PILOT_BLOCK_COUNT,
                pilot.PROSPECTIVE_CAMPAIGN,
            )
            fixed = {}
            for name in POLICY_NAMES:
                case_id = pilot._strict_json_value(
                    pilot.corrected_case_id(
                        partition,
                        family,
                        block,
                        name,
                        physical_config,
                    )
                )
                saved = {
                    "case_id": case_id,
                    "decision": "approve",
                    "decision_computation_seconds": 0.0,
                    "decision_reason": "test_fixture",
                    "failures": [],
                    "margin_envelope": None,
                    "model_intervals": [],
                    "verifications": [],
                }
                fixed[name] = {
                    "case": {
                        "acquisition_runs": [],
                        "case_id": case_id,
                        "final_instrumentation": {
                            "temporary_face_sensor": False
                        },
                        "final_regime": {},
                        "policy": {"name": name},
                        "verification_run": {},
                    },
                    "saved_before_reveal": saved,
                    "post_reveal_score": {
                        "nominal_selection_energy": 0.0,
                        "realized_energy": {
                            "case_id": case_id,
                            "truth_condition": family,
                            "trial_index": block,
                        },
                        "revealed": {
                            "case_id": case_id,
                            "truth_condition": family,
                        },
                        "scored": {
                            "saved": saved,
                            "truth_condition": family,
                        },
                    },
                }
            selected = primary[-1]["selection"]["selected_policy"]
            cases.append(
                {
                    "block": block,
                    "truth_condition": family,
                    "device_token": next(iter(fixed.values()))["case"][
                        "case_id"
                    ]["device_token"],
                    "truth_revealed_only_after_saves": True,
                    "selection_information_boundary": deepcopy(
                        pilot._SELECTION_INFORMATION_BOUNDARY
                    ),
                    "admissible_source_models": ["four_state"],
                    "admissible_source_model_count": 1,
                    "n16_complete_uncertainty_result": deepcopy(complete),
                    "primary_prefixes": deepcopy(primary),
                    "n16_max_unstable_0_sensitivity": _authenticated_prefix(
                        zero_failure
                    ),
                    "selected_policy_at_n16_primary": selected,
                    "selected_policy_outcome": deepcopy(fixed[selected]),
                    "fixed_policy_results": fixed,
                    "pipeline_failures": [],
                    "timing": {"test_fixture": True},
                }
            )
            case_index += 1
        blocks.append(
            {
                "block": block,
                "cases": cases,
                "corrected_random_stream_manifest": _corrected_stream_records(
                    block
                )[0],
                "corrected_random_stream_audit": _corrected_stream_records(
                    block
                )[1],
                "timing": {"test_fixture": True},
            }
        )
    acceptance = pilot.evaluate_pilot_acceptance(blocks)
    protocol_digest = pilot.prospective_pilot_protocol_digest(
        physical_config,
        ProspectiveSelectorRule(),
        PRIMARY_PROSPECTIVE_COST_SCENARIO,
        source_revision,
    )
    scientific = pilot.prospective_pilot_scientific_payload(
        source_revision=source_revision,
        protocol_digest=protocol_digest,
        block_results=blocks,
        acceptance=acceptance,
    )
    payload = {
        "schema_version": pilot.PROSPECTIVE_PILOT_SCHEMA_VERSION,
        "protocol_version": pilot.PROSPECTIVE_PILOT_PROTOCOL_VERSION,
        "campaign": pilot.PROSPECTIVE_CAMPAIGN,
        "partition": pilot.PROSPECTIVE_PILOT_PARTITION,
        "source_revision": source_revision,
        "protocol_digest": protocol_digest,
        "selector_rule": pilot.prospective_selector_rule_payload(
            ProspectiveSelectorRule()
        ),
        "cost_scenario": pilot._scenario_payload(
            PRIMARY_PROSPECTIVE_COST_SCENARIO
        ),
        "scientific_result_digest": pilot._digest(
            "thermotwin.prospective_pilot.scientific_result",
            scientific,
        ),
        "partition_plan": pilot.prospective_partition_plan_payload(),
        "worker_count": 1,
        "case_count": 12,
        "block_results": blocks,
        "acceptance": acceptance,
        "compute_budget_inputs": {
            "pilot_block_wall_seconds": [1.0] * 4,
            "pilot_block_cpu_seconds": [1.0] * 4,
            "mean_block_wall_seconds": 1.0,
            "p90_block_wall_seconds_nearest_rank": 1.0,
            "mean_block_cpu_seconds": 1.0,
            "measured_block_throughput_per_second": 1.0,
            "measured_worker_count": 1,
            "maximum_recorded_worker_peak_rss_bytes": 1,
            "conservative_concurrent_peak_rss_bytes": 1,
            "measured_pilot_throughput_scope": "test_fixture",
            "predictive_n16_wall_seconds": 1.0,
            "predictive_n16_cpu_seconds": 1.0,
            "prefix_study_overhead_wall_seconds": 1.0,
            "prefix_study_overhead_cpu_seconds": 1.0,
            "retained_final_scoring_wall_seconds": 1.0,
            "retained_final_scoring_cpu_seconds": 1.0,
            "draw_count_linear_projections": [
                {
                    "draw_count": count,
                    "estimated_full_campaign_wall_seconds_at_measured_concurrency": (
                        100.0
                    ),
                    "estimated_full_campaign_cpu_seconds": 200.0,
                }
                for count in pilot.PILOT_DRAW_COUNTS
            ],
            "one_source_case_count": 12,
            "two_source_case_count": 0,
            "zero_source_case_count": 0,
            "unknown_source_case_count": 0,
            "fixed_policy_wall_seconds_including_verification_and_reveal": {
                name: 1.0 for name in POLICY_NAMES
            },
            "fixed_policy_comparators_included": list(POLICY_NAMES),
            "verification_included": True,
            "runtime_host_manifest": {
                "python_version": "test",
                "python_implementation": "test",
                "operating_system": "test",
                "operating_system_release": "test",
                "machine": "test",
                "processor": "test",
                "logical_cpu_count": 1,
                "host_node_sha256": "0" * 64,
            },
            "conservative_measured_throughput_phase_estimates": [
                {
                    "partition": item.name,
                    "estimated_wall_seconds_at_measured_worker_count": 10.0,
                    "estimated_cpu_seconds": 20.0,
                }
                for item in pilot.PROSPECTIVE_PARTITION_PLAN[1:]
            ],
            "assumptions": ["test_fixture"],
        },
        "performance": {
            "wall_seconds": 1.0,
            "cpu_seconds": 1.0,
            "peak_rss_bytes": 1,
        },
        "scientific_use": "disposable_engineering_evidence_only",
        "archive_benchmark": {
            "benchmark_bytes": 1,
            "serialization_wall_seconds": 1.0,
            "serialization_cpu_seconds": 1.0,
            "write_wall_seconds": 1.0,
            "write_cpu_seconds": 1.0,
            "bytes_per_write_wall_second": 1.0,
            "bytes_per_serialization_wall_second": 1.0,
            "estimated_bytes_per_paired_block": 1,
            "full_campaign_partition_estimates": [
                {"partition": item.name}
                for item in pilot.PROSPECTIVE_PARTITION_PLAN[1:]
            ],
            "full_campaign_draw_count_estimates": [
                {"draw_count": count} for count in pilot.PILOT_DRAW_COUNTS
            ],
            "note": "test_fixture",
        },
    }
    payload["archive_content_digest"] = pilot._digest(
        "thermotwin.prospective_pilot.archive_content",
        payload,
    )
    payload["archive_size_bytes"] = 0
    for _ in range(16):
        size = len(pilot._canonical_bytes(payload) + b"\n")
        if payload["archive_size_bytes"] == size:
            break
        payload["archive_size_bytes"] = size
    return payload


_N32_PARENT_FIXTURE = None


def _n32_parent_payload():
    global _N32_PARENT_FIXTURE
    if _N32_PARENT_FIXTURE is None:
        _N32_PARENT_FIXTURE = _build_n32_parent_payload()
    return deepcopy(_N32_PARENT_FIXTURE)


def _reseal_parent_payload(payload):
    payload.pop("archive_content_digest", None)
    payload.pop("archive_size_bytes", None)
    payload["archive_content_digest"] = pilot._digest(
        "thermotwin.prospective_pilot.archive_content",
        payload,
    )
    payload["archive_size_bytes"] = 0
    for _ in range(16):
        size = len(pilot._canonical_bytes(payload) + b"\n")
        if payload["archive_size_bytes"] == size:
            break
        payload["archive_size_bytes"] = size
    return payload


def _reseal_parent_scientific(payload):
    payload["acceptance"] = pilot.evaluate_pilot_acceptance(
        payload["block_results"]
    )
    scientific = pilot.prospective_pilot_scientific_payload(
        source_revision=payload["source_revision"],
        protocol_digest=payload["protocol_digest"],
        block_results=payload["block_results"],
        acceptance=payload["acceptance"],
    )
    payload["scientific_result_digest"] = pilot._digest(
        "thermotwin.prospective_pilot.scientific_result",
        scientific,
    )
    return _reseal_parent_payload(payload)


def _reseal_n32_payload(payload):
    payload.pop("archive_content_digest", None)
    payload.pop("archive_size_bytes", None)
    payload["archive_content_digest"] = pilot._digest(
        "thermotwin.prospective_pilot.n32.archive_content",
        payload,
    )
    payload["archive_size_bytes"] = 0
    for _ in range(16):
        size = len(pilot._canonical_bytes(payload) + b"\n")
        if payload["archive_size_bytes"] == size:
            break
        payload["archive_size_bytes"] = size
    return payload


def _build_n32_blocks(parent_payload):
    physical_config = _pilot_physical_config()
    blocks = []
    for parent_block in parent_payload["block_results"]:
        complete = _complete_uncertainty(parent_block["block"], 32)
        derived_n16 = pilot._derive_uncertainty_prefix_payload(
            complete,
            physical_config=physical_config,
            block=parent_block["block"],
            draw_count=16,
            max_unstable_draws=pilot.pilot_max_unstable_draws(16),
        )
        n16 = _authenticated_prefix(derived_n16)
        n32 = _authenticated_prefix(complete)
        cases = []
        for parent_case in parent_block["cases"]:
            cases.append(
                {
                    "block": parent_case["block"],
                    "truth_condition": parent_case["truth_condition"],
                    "device_token": parent_case["device_token"],
                    "admissible_source_models": ["four_state"],
                    "admissible_source_model_count": 1,
                    "n32_complete_uncertainty_result": deepcopy(complete),
                    "primary_prefixes": [deepcopy(n16), deepcopy(n32)],
                    "pipeline_failures": [],
                    "timing": {"case_wall_seconds": 1.0},
                }
            )
        blocks.append(
            {
                "block": parent_block["block"],
                "cases": cases,
                "corrected_random_stream_manifest": _corrected_stream_records(
                    parent_block["block"],
                    followup=True,
                )[0],
                "corrected_random_stream_audit": _corrected_stream_records(
                    parent_block["block"],
                    followup=True,
                )[1],
                "timing": {
                    "block_wall_seconds": 1.0,
                    "block_cpu_seconds": 2.0,
                    "peak_rss_bytes": 3,
                },
            }
        )
    return blocks


_N32_BLOCK_FIXTURES = {}


def _n32_blocks(parent_payload, *, failed_draws=0):
    if failed_draws:
        raise ValueError("authenticated N=32 fixtures cannot fabricate failures")
    identity = parent_payload["archive_content_digest"]
    if identity not in _N32_BLOCK_FIXTURES:
        _N32_BLOCK_FIXTURES[identity] = _build_n32_blocks(parent_payload)
    return deepcopy(_N32_BLOCK_FIXTURES[identity])


class ProspectivePilotProtocolTests(unittest.TestCase):
    def test_partition_names_and_sizes_are_exact_and_disjoint(self):
        self.assertEqual(
            pilot.PROSPECTIVE_CAMPAIGN,
            "operating_decision_prospective_v1_2026_09",
        )
        self.assertEqual(
            tuple((item.name, item.block_count) for item in pilot.PROSPECTIVE_PARTITION_PLAN),
            (
                ("p4_disposable_bounded_instability_pilot", 4),
                ("p1_development_tuning", 20),
                ("p1_development_internal_check", 10),
                ("p1_independent_calibration", 100),
                ("p1_reserved_evaluation", 100),
            ),
        )
        self.assertEqual(
            len({item.name for item in pilot.PROSPECTIVE_PARTITION_PLAN}),
            5,
        )
        self.assertEqual(pilot.PILOT_DRAW_COUNTS, (4, 8, 16))
        self.assertEqual(pilot.PILOT_N32_FOLLOWUP_DRAW_COUNT, 32)
        self.assertEqual(pilot.PROSPECTIVE_PILOT_SCHEMA_VERSION, 4)
        self.assertEqual(
            pilot.PROSPECTIVE_PILOT_PROTOCOL_VERSION,
            "operating_decision_prospective_draw_count_pilot_v5",
        )
        self.assertEqual(
            pilot.PROSPECTIVE_SUPERSEDED_PILOT_PARTITIONS,
            (
                "p1_disposable_draw_count_pilot",
                "p2_disposable_candidate_exclusion_pilot",
                "p3_disposable_archive_roundtrip_replacement_pilot",
            ),
        )
        self.assertEqual(pilot.PROSPECTIVE_N32_FOLLOWUP_SCHEMA_VERSION, 2)
        self.assertEqual(
            pilot.PROSPECTIVE_N32_FOLLOWUP_PROTOCOL_VERSION,
            "operating_decision_prospective_n32_followup_v3",
        )
        self.assertEqual(pilot.PILOT_PRIMARY_MAX_UNSTABLE, 1)
        self.assertEqual(pilot.PILOT_SENSITIVITY_MAX_UNSTABLE, 0)
        self.assertEqual(
            pilot.PILOT_MAX_UNSTABLE_BY_DRAW_COUNT,
            {4: 0, 8: 0, 16: 1, 32: 1},
        )
        self.assertEqual(
            [pilot.pilot_max_unstable_draws(count) for count in (4, 8, 16, 32)],
            [0, 0, 1, 1],
        )
        with self.assertRaisesRegex(ValueError, "no instability allowance"):
            pilot.pilot_max_unstable_draws(64)
        followup = pilot.prospective_n32_followup_design_payload()
        self.assertEqual(
            followup["artifact_id"],
            "p4_disposable_bounded_instability_pilot_n32_all_cases_v1",
        )
        self.assertEqual(followup["partition"], pilot.PROSPECTIVE_PILOT_PARTITION)
        self.assertEqual(followup["blocks"], [0, 1, 2, 3])
        self.assertEqual(
            followup["truth_families"],
            list(pilot.STAGE3_TRUTH_CONDITIONS),
        )
        self.assertEqual(followup["case_count"], 12)
        self.assertEqual(followup["candidate_prefix_draw_count"], 16)
        self.assertEqual(followup["reference_draw_count"], 32)
        self.assertTrue(
            followup["trigger"]["requires_zero_parent_pilot_pipeline_failures"]
        )
        self.assertTrue(
            followup["trigger"][
                "requires_zero_parent_pilot_n16_selection_failures"
            ]
        )
        self.assertTrue(
            followup["trigger"][
                "requires_zero_parent_pilot_n16_ineligible_actions"
            ]
        )
        self.assertNotIn("requires_zero_p2_pipeline_failures", followup["trigger"])
        self.assertTrue(followup["trigger"]["required_before_phase_d"])

    def test_pilot_protocol_digest_binds_both_superseded_partitions(self):
        config = OperatingDecisionRealismConfig()
        selector_rule = ProspectiveSelectorRule()
        revision = "a" * 40
        baseline = pilot.prospective_pilot_protocol_digest(
            config,
            selector_rule,
            PRIMARY_PROSPECTIVE_COST_SCENARIO,
            revision,
        )
        for superseded in (
            ("p1_disposable_draw_count_pilot",),
            ("p2_disposable_candidate_exclusion_pilot",),
        ):
            with self.subTest(superseded=superseded), patch.object(
                pilot,
                "PROSPECTIVE_SUPERSEDED_PILOT_PARTITIONS",
                superseded,
            ):
                self.assertNotEqual(
                    pilot.prospective_pilot_protocol_digest(
                        config,
                        selector_rule,
                        PRIMARY_PROSPECTIVE_COST_SCENARIO,
                        revision,
                    ),
                    baseline,
                )

    def test_acceptance_uses_all_cases_and_declared_agreement_and_regret(self):
        result = pilot.evaluate_pilot_acceptance(_blocks(changed_regret=0.04))
        n4, n8, n16 = result["primary_summaries"]
        self.assertEqual(n4["agreement_count"], 11)
        self.assertAlmostEqual(n4["agreement_rate"], 11 / 12)
        self.assertAlmostEqual(
            n4["maximum_changed_choice_normalized_utility_regret"],
            0.04,
        )
        self.assertTrue(n4["meets_acceptance_rule"])
        self.assertEqual(n8["agreement_count"], 12)
        self.assertEqual(n16["agreement_count"], 12)
        self.assertEqual(result["recommended_draw_count"], 4)
        self.assertTrue(result["accepted"])

    def test_excess_regret_moves_recommendation_to_next_draw_count(self):
        result = pilot.evaluate_pilot_acceptance(_blocks(changed_regret=0.06))
        self.assertFalse(result["primary_summaries"][0]["meets_regret_rule"])
        self.assertEqual(result["recommended_draw_count"], 8)

    def test_n16_recommendation_requires_n32_before_phase_d(self):
        blocks = _blocks()
        cases = [case for block in blocks for case in block["cases"]]
        for case in cases[:2]:
            for prefix in case["primary_prefixes"][:2]:
                prefix["choice_token"] = "action:fixed_voltage"
                prefix["selection"]["selected_policy"] = FIXED_VOLTAGE
        result = pilot.evaluate_pilot_acceptance(blocks)
        self.assertFalse(result["primary_summaries"][0]["meets_agreement_rule"])
        self.assertFalse(result["primary_summaries"][1]["meets_agreement_rule"])
        self.assertEqual(result["stability_recommended_draw_count"], 16)
        self.assertTrue(result["feasibility_gate_passed"])
        self.assertTrue(result["n32_followup_required"])
        self.assertFalse(result["pilot_engineering_gate_passed"])
        self.assertFalse(result["phase_d_entry_authorized"])
        self.assertIsNone(result["recommended_draw_count"])
        self.assertFalse(result["accepted"])

    def test_n32_followup_passes_only_after_authentic_all_case_comparison(self):
        parent = _n32_parent_payload()
        result = pilot.evaluate_n32_followup_acceptance(
            parent,
            _n32_blocks(parent),
        )
        self.assertTrue(result["parent_trigger_passed"])
        self.assertTrue(result["authenticated_n16_prefix_match"])
        self.assertTrue(result["case_identity_match"])
        self.assertEqual(result["summary"]["agreement_count"], 12)
        self.assertEqual(result["n32_whole_draw_failure_count"], 0)
        self.assertTrue(result["pilot_engineering_gate_passed"])
        self.assertFalse(result["phase_d_entry_authorized"])
        self.assertEqual(result["recommended_draw_count"], 16)

    def test_n32_followup_rejects_prefix_and_case_identity_mismatches(self):
        parent = _n32_parent_payload()
        blocks = _n32_blocks(parent)
        blocks[0]["cases"][0]["device_token"] = "different-device"
        with self.assertRaisesRegex(ValueError, "device identity"):
            pilot.evaluate_n32_followup_acceptance(parent, blocks)

        blocks = _n32_blocks(parent)
        blocks[0]["cases"][1]["primary_prefixes"][0]["choice_token"] = (
            "action:fixed_voltage"
        )
        with self.assertRaisesRegex(ValueError, "choice token"):
            pilot.evaluate_n32_followup_acceptance(parent, blocks)

    def test_n32_followup_rejects_pipeline_failure(self):
        parent = _n32_parent_payload()
        blocks = _n32_blocks(parent)
        blocks[0] = pilot._failed_n32_followup_block(
            0,
            RuntimeError("deliberate block failure"),
        )
        acceptance = pilot.evaluate_n32_followup_acceptance(parent, blocks)
        self.assertEqual(acceptance["n32_pipeline_failure_count"], 3)
        self.assertFalse(acceptance["pilot_engineering_gate_passed"])
        self.assertFalse(acceptance["accepted"])

    def test_n32_followup_retains_one_whole_draw_failure_per_action(self):
        parent = _n32_parent_payload()
        blocks = _n32_blocks(parent)
        case = blocks[0]["cases"][0]
        complete = case["n32_complete_uncertainty_result"]
        for draw in complete["draw_outcomes"]:
            if draw["draw_index"] != 20:
                continue
            draw.update(
                {
                    "candidate_outcomes": [],
                    "initially_admissible_became_inadmissible": [],
                    "initially_excluded_became_admissible": [],
                    "stable": False,
                    "failed": True,
                    "failure_stage": "prospective_simulation",
                    "failure_model": None,
                    "failure_type": "ValueError",
                    "raw_after_width": None,
                    "scored_after_width": 2.0,
                    "synthetic_observation_digest": None,
                }
            )
        complete["stream_uses"] = [
            use
            for use in complete["stream_uses"]
            if not (
                use["key"]["draw_index"] == 20
                and use["key"]["purpose"] in ("run_bias", "white_noise")
            )
        ]
        _recompute_uncertainty_actions(complete, 0)
        case["primary_prefixes"][1] = _authenticated_prefix(complete)
        acceptance = pilot.evaluate_n32_followup_acceptance(parent, blocks)
        self.assertEqual(acceptance["n32_whole_draw_failure_count"], 3)
        self.assertEqual(acceptance["n32_selection_failure_count"], 0)
        self.assertEqual(acceptance["n32_ineligible_action_count"], 0)
        self.assertTrue(acceptance["pilot_engineering_gate_passed"])

    def test_n32_followup_rejects_action_that_exceeds_failure_allowance(self):
        parent = _n32_parent_payload()
        blocks = _n32_blocks(parent)
        case = blocks[0]["cases"][0]
        complete = case["n32_complete_uncertainty_result"]
        for draw in complete["draw_outcomes"]:
            if (
                draw["draw_index"] not in (20, 21)
                or draw["policy_name"] != FIXED_VOLTAGE
            ):
                continue
            draw.update(
                {
                    "candidate_outcomes": [],
                    "initially_admissible_became_inadmissible": [],
                    "initially_excluded_became_admissible": [],
                    "stable": False,
                    "failed": True,
                    "failure_stage": "prospective_simulation",
                    "failure_model": None,
                    "failure_type": "ValueError",
                    "raw_after_width": None,
                    "scored_after_width": 2.0,
                    "synthetic_observation_digest": None,
                }
            )
        complete["stream_uses"] = [
            use
            for use in complete["stream_uses"]
            if not (
                use["key"]["draw_index"] in (20, 21)
                and use["key"]["action"] == FIXED_VOLTAGE
                and use["key"]["purpose"] in ("run_bias", "white_noise")
            )
        ]
        _recompute_uncertainty_actions(complete, 0)
        case["primary_prefixes"][1] = _authenticated_prefix(complete)
        acceptance = pilot.evaluate_n32_followup_acceptance(parent, blocks)
        self.assertEqual(acceptance["n32_whole_draw_failure_count"], 2)
        self.assertEqual(acceptance["n32_ineligible_action_count"], 1)
        self.assertEqual(acceptance["n32_cases_with_ineligible_actions"], 1)
        self.assertFalse(acceptance["pilot_engineering_gate_passed"])

    def test_n32_rejects_missing_complete_payload_and_minimal_prefix(self):
        parent = _n32_parent_payload()
        blocks = _n32_blocks(parent)
        blocks[0]["cases"][0]["n32_complete_uncertainty_result"] = None
        with self.assertRaisesRegex(ValueError, "complete prospective uncertainty"):
            pilot.evaluate_n32_followup_acceptance(parent, blocks)

        blocks = _n32_blocks(parent)
        blocks[0]["cases"][0]["primary_prefixes"][0] = _prefix(
            16,
            FIXED_THERMAL,
        )
        with self.assertRaisesRegex(ValueError, "authenticated pilot prefix"):
            pilot.evaluate_n32_followup_acceptance(parent, blocks)

    def test_n32_rejects_dirty_or_missing_random_stream_audits(self):
        parent = _n32_parent_payload()
        blocks = _n32_blocks(parent)
        blocks[0]["corrected_random_stream_audit"]["use_count"] += 1
        with self.assertRaisesRegex(ValueError, "corrected random-stream audit"):
            pilot.evaluate_n32_followup_acceptance(parent, blocks)

        blocks = _n32_blocks(parent)
        complete = blocks[0]["cases"][0]["n32_complete_uncertainty_result"]
        complete.pop("stream_audit")
        with self.assertRaisesRegex(ValueError, "complete prospective uncertainty"):
            pilot.evaluate_n32_followup_acceptance(parent, blocks)

        incomplete_parent = _n32_parent_payload()
        incomplete_parent.pop("archive_benchmark")
        with self.assertRaisesRegex(ValueError, "complete parent pilot archive"):
            pilot.evaluate_n32_followup_acceptance(
                incomplete_parent,
                _n32_blocks(parent),
            )

    def test_n32_rejects_cases_swapped_between_blocks(self):
        parent = _n32_parent_payload()
        blocks = _n32_blocks(parent)
        blocks[0]["cases"][0], blocks[1]["cases"][0] = (
            blocks[1]["cases"][0],
            blocks[0]["cases"][0],
        )
        with self.assertRaisesRegex(ValueError, "nested under the wrong block"):
            pilot.evaluate_n32_followup_acceptance(parent, blocks)

    def test_complete_evidence_rejects_resealed_missing_observation_stream(self):
        parent = _n32_parent_payload()
        blocks = _n32_blocks(parent)
        complete = blocks[0]["cases"][0]["n32_complete_uncertainty_result"]
        complete["stream_uses"] = [
            use
            for use in complete["stream_uses"]
            if not (
                use["key"]["purpose"] == "run_bias"
                and use["key"]["draw_index"] == 0
                and use["key"]["action"] == FIXED_THERMAL
            )
        ]
        _reseal_uncertainty(complete, 0)
        with self.assertRaisesRegex(ValueError, "observation stream inventory"):
            pilot.evaluate_n32_followup_acceptance(parent, blocks)

    def test_complete_evidence_rejects_simulation_failure_stream_suffix(self):
        physical_config = _pilot_physical_config()
        complete = _complete_uncertainty(0, 32)
        draw_index = 20
        draw = next(
            item
            for item in complete["draw_outcomes"]
            if item["draw_index"] == draw_index
            and item["policy_name"] == FIXED_THERMAL
        )
        draw.update(
            {
                "candidate_outcomes": [],
                "initially_admissible_became_inadmissible": [],
                "initially_excluded_became_admissible": [],
                "stable": False,
                "failed": True,
                "failure_stage": "prospective_simulation",
                "failure_model": None,
                "failure_type": "ValueError",
                "raw_after_width": None,
                "scored_after_width": 2.0,
                "synthetic_observation_digest": None,
            }
        )
        thermal = next(
            policy
            for policy in default_fixed_policies()
            if policy.name == FIXED_THERMAL
        )
        final_run = thermal.additional_regimes[-1].name
        complete["stream_uses"] = [
            use
            for use in complete["stream_uses"]
            if not (
                use["key"]["generator_model"] == "four_state"
                and use["key"]["draw_index"] == draw_index
                and use["key"]["action"] == FIXED_THERMAL
                and use["key"]["purpose"] in ("run_bias", "white_noise")
                and use["key"]["run"] != final_run
            )
        ]
        _recompute_uncertainty_actions(complete, 0)
        with self.assertRaisesRegex(ValueError, "observation stream inventory"):
            pilot._validate_complete_uncertainty_payload(
                complete,
                physical_config=physical_config,
                block=0,
                draw_count=32,
                max_unstable_draws=pilot.pilot_max_unstable_draws(32),
            )

    def test_complete_evidence_rejects_forged_failed_draw_shape(self):
        physical_config = _pilot_physical_config()
        complete = _complete_uncertainty(0, 32)
        draw = next(
            item
            for item in complete["draw_outcomes"]
            if item["draw_index"] == 20
            and item["policy_name"] == FIXED_VOLTAGE
        )
        draw.update(
            {
                "candidate_outcomes": [],
                "initially_admissible_became_inadmissible": [],
                "initially_excluded_became_admissible": [],
                "stable": False,
                "failed": True,
                "failure_stage": "prospective_refit",
                "failure_model": "four_state",
                "failure_type": "ValueError",
                "raw_after_width": None,
                "scored_after_width": 2.0,
            }
        )
        _reseal_uncertainty(complete, 0)
        with self.assertRaisesRegex(ValueError, "candidate evidence"):
            pilot._validate_complete_uncertainty_payload(
                complete,
                physical_config=physical_config,
                block=0,
                draw_count=32,
                max_unstable_draws=pilot.pilot_max_unstable_draws(32),
            )

    def test_complete_evidence_replays_in_bounds_rng_values_after_n16(self):
        physical_config = _pilot_physical_config()
        complete = _complete_uncertainty(0, 32)
        selected = [
            draw
            for draw in complete["draw_outcomes"]
            if draw["generator_model"] == "four_state"
            and draw["draw_index"] == 20
        ]
        self.assertEqual(len(selected), 3)
        changed = selected[0]["generator_log_offsets"][0] + 1.0e-6
        for draw in selected:
            draw["generator_log_offsets"][0] = changed
        _reseal_uncertainty(complete, 0)
        with self.assertRaisesRegex(ValueError, "parameter draw.*deterministic"):
            pilot._validate_complete_uncertainty_payload(
                complete,
                physical_config=physical_config,
                block=0,
                draw_count=32,
                max_unstable_draws=pilot.pilot_max_unstable_draws(32),
            )

    def test_complete_evidence_rejects_impossible_partial_sampling_failures(self):
        physical_config = _pilot_physical_config()

        def mark_failed(draw, stage):
            draw.update(
                {
                    "candidate_outcomes": [],
                    "initially_admissible_became_inadmissible": [],
                    "initially_excluded_became_admissible": [],
                    "stable": False,
                    "failed": True,
                    "failure_stage": stage,
                    "failure_model": None,
                    "failure_type": "_ProspectiveDrawSamplingError",
                    "raw_after_width": None,
                    "scored_after_width": 2.0,
                    "synthetic_observation_digest": None,
                }
            )

        complete = _complete_uncertainty(0, 32)
        partial = next(
            draw
            for draw in complete["draw_outcomes"]
            if draw["draw_index"] == 20 and draw["policy_name"] == FIXED_THERMAL
        )
        mark_failed(partial, "prospective_parameter_draw")
        _reseal_uncertainty(complete, 0)
        with self.assertRaisesRegex(ValueError, "parameter failure"):
            pilot._validate_complete_uncertainty_payload(
                complete,
                physical_config=physical_config,
                block=0,
                draw_count=32,
                max_unstable_draws=pilot.pilot_max_unstable_draws(32),
            )

        complete = _complete_uncertainty(0, 32)
        wrong_action = next(
            draw
            for draw in complete["draw_outcomes"]
            if draw["draw_index"] == 20 and draw["policy_name"] == FIXED_VOLTAGE
        )
        mark_failed(wrong_action, "prospective_probe_draw")
        _reseal_uncertainty(complete, 0)
        with self.assertRaisesRegex(ValueError, "face action"):
            pilot._validate_complete_uncertainty_payload(
                complete,
                physical_config=physical_config,
                block=0,
                draw_count=32,
                max_unstable_draws=pilot.pilot_max_unstable_draws(32),
            )

    def test_complete_evidence_validates_fit_and_interval_structure(self):
        physical_config = _pilot_physical_config()
        complete = _complete_uncertainty(0, 32)
        complete["acquisition_evidence"]["fit_set"]["fits"][0][
            "objective"
        ] = float("nan")
        with self.assertRaisesRegex(ValueError, "fit objective"):
            pilot._validate_complete_uncertainty_payload(
                complete,
                physical_config=physical_config,
                block=0,
                draw_count=32,
                max_unstable_draws=pilot.pilot_max_unstable_draws(32),
            )

        complete = _complete_uncertainty(0, 32)
        complete["acquisition_evidence"]["fit_set"]["fits"][0][
            "objective"
        ] = -1.0
        with self.assertRaisesRegex(ValueError, "fit objective"):
            pilot._validate_complete_uncertainty_payload(
                complete,
                physical_config=physical_config,
                block=0,
                draw_count=32,
                max_unstable_draws=pilot.pilot_max_unstable_draws(32),
            )

        complete = _complete_uncertainty(0, 32)
        complete["acquisition_evidence"]["fit_set"]["fits"][0][
            "physical_values"
        ][0] += 1.0
        with self.assertRaisesRegex(ValueError, "do not decode"):
            pilot._validate_complete_uncertainty_payload(
                complete,
                physical_config=physical_config,
                block=0,
                draw_count=32,
                max_unstable_draws=pilot.pilot_max_unstable_draws(32),
            )

        complete = _complete_uncertainty(0, 32)
        complete["acquisition_evidence"]["fit_set"]["fits"][0][
            "covariance"
        ][0][0] = -1.0
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            pilot._validate_complete_uncertainty_payload(
                complete,
                physical_config=physical_config,
                block=0,
                draw_count=32,
                max_unstable_draws=pilot.pilot_max_unstable_draws(32),
            )

        complete = _complete_uncertainty(0, 32)
        complete["draw_outcomes"][0]["candidate_outcomes"][1]["interval"][
            "upper"
        ] += 0.01
        with self.assertRaisesRegex(ValueError, "multiplier.*SE"):
            pilot._validate_complete_uncertainty_payload(
                complete,
                physical_config=physical_config,
                block=0,
                draw_count=32,
                max_unstable_draws=pilot.pilot_max_unstable_draws(32),
            )

        complete = _complete_uncertainty(0, 32)
        complete["acquisition_evidence"]["fit_set"]["fits"][0][
            "accepted_iterations"
        ] = physical_config.sensor.fit_iterations + 1
        with self.assertRaisesRegex(ValueError, "fit diagnostics"):
            pilot._validate_complete_uncertainty_payload(
                complete,
                physical_config=physical_config,
                block=0,
                draw_count=32,
                max_unstable_draws=pilot.pilot_max_unstable_draws(32),
            )

        complete = _complete_uncertainty(0, 32)
        complete["acquisition_evidence"]["fit_set"]["fits"][0][
            "reached_bound"
        ] = False
        with self.assertRaisesRegex(ValueError, "fit status.*diagnostics"):
            pilot._validate_complete_uncertainty_payload(
                complete,
                physical_config=physical_config,
                block=0,
                draw_count=32,
                max_unstable_draws=pilot.pilot_max_unstable_draws(32),
            )

        complete = _complete_uncertainty(0, 32)
        complete["acquisition_evidence"]["fit_set"]["fits"][1][
            "scaled_gradient_infinity_norm"
        ] = 1.0
        with self.assertRaisesRegex(ValueError, "fit status.*diagnostics"):
            pilot._validate_complete_uncertainty_payload(
                complete,
                physical_config=physical_config,
                block=0,
                draw_count=32,
                max_unstable_draws=pilot.pilot_max_unstable_draws(32),
            )

        complete = _complete_uncertainty(0, 32)
        excluded_fit = complete["acquisition_evidence"]["fit_set"]["fits"][0]
        excluded_fit["last_step_infinity_norm"] = 0.0
        excluded_fit["last_relative_objective_reduction"] = 0.0
        with self.assertRaisesRegex(ValueError, "fit status.*diagnostics"):
            pilot._validate_complete_uncertainty_payload(
                complete,
                physical_config=physical_config,
                block=0,
                draw_count=32,
                max_unstable_draws=pilot.pilot_max_unstable_draws(32),
            )

        complete = _complete_uncertainty(0, 32)
        complete["acquisition_evidence"]["snapshot"]["excluded_candidates"][0][
            "reason"
        ] = "optimizer_not_converged"
        with self.assertRaisesRegex(ValueError, "snapshot.*fit set"):
            pilot._validate_complete_uncertainty_payload(
                complete,
                physical_config=physical_config,
                block=0,
                draw_count=32,
                max_unstable_draws=pilot.pilot_max_unstable_draws(32),
            )

        complete = _complete_uncertainty(0, 32)
        fit_set = complete["acquisition_evidence"]["fit_set"]
        failed_fit = fit_set["fits"].pop(0)
        fit_set["failures"].append(
            {
                "model_name": failed_fit["model_name"],
                "stage": "wrong_stage",
                "error_type": "RuntimeError",
            }
        )
        snapshot = complete["acquisition_evidence"]["snapshot"]
        snapshot["excluded_candidates"] = []
        snapshot["failed_candidate_models"] = [failed_fit["model_name"]]
        with self.assertRaisesRegex(ValueError, "fit failure"):
            pilot._validate_complete_uncertainty_payload(
                complete,
                physical_config=physical_config,
                block=0,
                draw_count=32,
                max_unstable_draws=pilot.pilot_max_unstable_draws(32),
            )

    def test_complete_evidence_rejects_invalid_draw_state_and_attrition_floor(self):
        physical_config = _pilot_physical_config()
        complete = _complete_uncertainty(0, 32)
        complete["draw_outcomes"][0]["stable"] = False
        with self.assertRaisesRegex(ValueError, "completed draw.*stable"):
            pilot._validate_complete_uncertainty_payload(
                complete,
                physical_config=physical_config,
                block=0,
                draw_count=32,
                max_unstable_draws=pilot.pilot_max_unstable_draws(32),
            )

        complete = _complete_uncertainty(0, 32)
        draw = complete["draw_outcomes"][0]
        draw["initially_admissible_became_inadmissible"] = ["four_state"]
        draw["initially_excluded_became_admissible"] = ["five_state"]
        draw["candidate_outcomes"][0] = {
            "model_name": "five_state",
            "status": "admissible",
            "objective": 1.0,
            "interval": {
                "model_name": "five_state",
                "estimate": 0.0,
                "local_standard_error": 0.45,
                "lower": -0.9,
                "upper": 0.9,
            },
        }
        draw["candidate_outcomes"][1] = {
            "model_name": "four_state",
            "status": "fit_reached_bound",
            "objective": 1.0,
            "interval": None,
        }
        with self.assertRaisesRegex(ValueError, "attrition floor"):
            pilot._validate_complete_uncertainty_payload(
                complete,
                physical_config=physical_config,
                block=0,
                draw_count=32,
                max_unstable_draws=pilot.pilot_max_unstable_draws(32),
            )

    def test_complete_evidence_rejects_parameter_probe_and_fit_tampering(self):
        physical_config = _pilot_physical_config()
        complete = _complete_uncertainty(0, 32)
        complete["draw_outcomes"][0]["generator_log_offsets"][0] = 99.0
        with self.assertRaisesRegex(ValueError, "shared across actions|fit bounds"):
            pilot._validate_complete_uncertainty_payload(
                complete,
                physical_config=physical_config,
                block=0,
                draw_count=32,
                max_unstable_draws=pilot.pilot_max_unstable_draws(32),
            )

        complete = _complete_uncertainty(0, 32)
        face = next(
            draw
            for draw in complete["draw_outcomes"]
            if draw["policy_name"] == FIXED_FACE_TEMPERATURE
        )
        face["face_probe_log_offsets"] = [99.0, 0.0]
        with self.assertRaisesRegex(ValueError, "probe draw.*bounds"):
            pilot._validate_complete_uncertainty_payload(
                complete,
                physical_config=physical_config,
                block=0,
                draw_count=32,
                max_unstable_draws=pilot.pilot_max_unstable_draws(32),
            )

        complete = _complete_uncertainty(0, 32)
        complete["acquisition_evidence"]["fit_set"]["fits"][0][
            "parameter_names"
        ] = ["forged"]
        with self.assertRaisesRegex(ValueError, "fit parameters"):
            pilot._validate_complete_uncertainty_payload(
                complete,
                physical_config=physical_config,
                block=0,
                draw_count=32,
                max_unstable_draws=pilot.pilot_max_unstable_draws(32),
            )

    def test_corrected_stream_inventory_rejects_cleanly_resealed_deletion(self):
        manifest, _ = _corrected_stream_records(0, followup=True)
        manifest.pop()
        audit = _corrected_audit_from_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "inventory|audit failed"):
            pilot._validate_corrected_stream_records(
                manifest,
                audit,
                block=0,
                label="N=32",
            )

    def test_corrected_stream_inventory_rejects_resealed_identity_tampering(self):
        for field, value in (
            ("family", "bogus_family"),
            ("run", "bogus_run"),
            ("channel", "bogus_channel"),
        ):
            with self.subTest(field=field):
                manifest, _ = _corrected_stream_records(0, followup=True)
                use = next(
                    item
                    for item in manifest
                    if item["key"]["stream_kind"] == "observation"
                )
                use["key"][field] = value
                use["seed"] = pilot.RandomStream(
                    pilot.RandomStreamKey(**use["key"])
                ).seed
                audit = _corrected_audit_from_manifest(manifest)
                with self.assertRaisesRegex(ValueError, "inventory"):
                    pilot._validate_corrected_stream_records(
                        manifest,
                        audit,
                        block=0,
                        label="N=32",
                    )

        manifest, _ = _corrected_stream_records(0, followup=False)
        use = next(item for item in manifest if item["pairing_id"] is not None)
        use["pairing_id"] += "/forged"
        audit = _corrected_audit_from_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "inventory|audit failed"):
            pilot._validate_corrected_stream_records(
                manifest,
                audit,
                block=0,
                label="parent pilot",
            )

    def test_n32_failed_family_streams_must_be_an_ordered_prefix(self):
        manifest, _ = _corrected_stream_records(0, followup=True)
        failed_family = pilot.STAGE3_TRUTH_CONDITIONS[0]
        family_indexes = [
            index
            for index, item in enumerate(manifest)
            if item["key"]["stream_kind"] == "observation"
            and item["key"]["family"] == failed_family
        ]
        manifest.pop(family_indexes[1])
        cases = [
            {
                "truth_condition": family,
                "device_token": None if family == failed_family else "complete",
            }
            for family in pilot.STAGE3_TRUTH_CONDITIONS
        ]
        audit = _corrected_audit_from_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "inventory"):
            pilot._validate_corrected_stream_records(
                manifest,
                audit,
                block=0,
                label="N=32",
                cases=cases,
            )

    def test_n32_protocol_rejects_source_protocol_and_trigger_mismatches(self):
        parent = _n32_parent_payload()
        partition = CorrectedPartition(
            pilot.PROSPECTIVE_PILOT_PARTITION,
            pilot.PILOT_BLOCK_COUNT,
            pilot.PROSPECTIVE_CAMPAIGN,
        )
        physical_config = pilot.corrected_partition_config(
            partition,
            pilot.CORRECTED_REPLICATION_CONFIG,
        )
        digest = pilot.prospective_n32_followup_protocol_digest(
            parent,
            physical_config,
            ProspectiveSelectorRule(),
            PRIMARY_PROSPECTIVE_COST_SCENARIO,
            "a" * 40,
        )
        self.assertEqual(len(digest), 64)
        with self.assertRaisesRegex(ValueError, "source_revision"):
            pilot.prospective_n32_followup_protocol_digest(
                parent,
                physical_config,
                ProspectiveSelectorRule(),
                PRIMARY_PROSPECTIVE_COST_SCENARIO,
                "b" * 40,
            )
        forged = deepcopy(parent)
        forged["protocol_digest"] = "c" * 64
        _reseal_parent_payload(forged)
        with self.assertRaisesRegex(ValueError, "protocol digest"):
            pilot.prospective_n32_followup_protocol_digest(
                forged,
                physical_config,
                ProspectiveSelectorRule(),
                PRIMARY_PROSPECTIVE_COST_SCENARIO,
                "a" * 40,
            )
        with self.assertRaisesRegex(ValueError, "protocol digest"):
            pilot.prospective_n32_followup_protocol_digest(
                parent,
                OperatingDecisionRealismConfig(),
                ProspectiveSelectorRule(),
                PRIMARY_PROSPECTIVE_COST_SCENARIO,
                "a" * 40,
            )
        missing_case = deepcopy(parent)
        missing_case["block_results"][0]["cases"].pop()
        _reseal_parent_payload(missing_case)
        with self.assertRaisesRegex(ValueError, "case identities"):
            pilot.prospective_n32_followup_protocol_digest(
                missing_case,
                physical_config,
                ProspectiveSelectorRule(),
                PRIMARY_PROSPECTIVE_COST_SCENARIO,
                "a" * 40,
            )
        no_trigger = deepcopy(parent)
        no_trigger["acceptance"]["n32_followup_required"] = False
        _reseal_parent_payload(no_trigger)
        with self.assertRaisesRegex(ValueError, "does not recompute"):
            pilot.prospective_n32_followup_protocol_digest(
                no_trigger,
                physical_config,
                ProspectiveSelectorRule(),
                PRIMARY_PROSPECTIVE_COST_SCENARIO,
                "a" * 40,
            )
        wrapper_tamper = deepcopy(parent)
        wrapper_tamper["archive_size_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "archive size"):
            pilot.prospective_n32_followup_protocol_digest(
                wrapper_tamper,
                physical_config,
                ProspectiveSelectorRule(),
                PRIMARY_PROSPECTIVE_COST_SCENARIO,
                "a" * 40,
            )

    def test_n32_runner_uses_all_four_blocks_and_exposes_final_pilot_gate(self):
        parent = json.loads(pilot._canonical_bytes(_n32_parent_payload()))
        followup_blocks = _n32_blocks(parent)
        with patch.object(
            pilot,
            "_run_n32_followup_block",
            side_effect=lambda block, *_args: followup_blocks[block],
        ) as run_block:
            result = pilot.run_prospective_n32_followup(
                parent,
                source_revision="a" * 40,
                workers=1,
            )
        self.assertEqual(run_block.call_count, 4)
        self.assertEqual(
            tuple(item["block"] for item in result.block_results),
            (0, 1, 2, 3),
        )
        self.assertTrue(result.acceptance["pilot_engineering_gate_passed"])
        self.assertFalse(result.acceptance["phase_d_entry_authorized"])
        payload = pilot.prospective_n32_followup_result_payload(result)
        self.assertEqual(
            payload["artifact_id"],
            pilot.PILOT_N32_FOLLOWUP_ARTIFACT_ID,
        )
        self.assertEqual(payload["parent_payload_digest"], result.parent_payload_digest)
        archived = json.loads(
            pilot._canonical_bytes(_reseal_n32_payload(deepcopy(payload)))
        )
        replayed = pilot.validate_prospective_n32_followup_archive(archived)
        self.assertEqual(replayed["acceptance"], result.acceptance)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validator = pilot.validate_prospective_n32_followup_archive
            with patch.object(
                pilot,
                "validate_prospective_n32_followup_archive",
                wraps=validator,
            ) as validate:
                saved = pilot.save_prospective_n32_followup_artifacts(
                    result,
                    json_path=root / "n32.json",
                    report_path=root / "n32.txt",
                    hash_path=root / "n32.sha256",
                )
            saved_payload = json.loads(
                saved.json_path.read_text(encoding="utf-8")
            )
            self.assertEqual(validate.call_args.args[0], saved_payload)
            self.assertEqual(validator(saved_payload), saved_payload)
        tampered = deepcopy(archived)
        tampered["acceptance"]["pilot_engineering_gate_passed"] = False
        _reseal_n32_payload(tampered)
        with self.assertRaisesRegex(ValueError, "acceptance does not recompute"):
            pilot.validate_prospective_n32_followup_archive(tampered)
        result.acceptance["pilot_engineering_gate_passed"] = False
        with self.assertRaisesRegex(ValueError, "acceptance does not recompute"):
            pilot.prospective_n32_followup_result_payload(result)

    def test_n32_runner_saves_and_replays_canonical_negative_artifact(self):
        parent = _n32_parent_payload()
        followup_blocks = _n32_blocks(parent)
        followup_blocks[0] = pilot._failed_n32_followup_block(
            0,
            RuntimeError("deliberate block failure"),
        )
        with patch.object(
            pilot,
            "_run_n32_followup_block",
            side_effect=lambda block, *_args: followup_blocks[block],
        ):
            result = pilot.run_prospective_n32_followup(
                parent,
                source_revision="a" * 40,
                workers=1,
            )
        self.assertEqual(result.acceptance["n32_pipeline_failure_count"], 3)
        self.assertFalse(result.acceptance["accepted"])
        archived = _reseal_n32_payload(
            pilot.prospective_n32_followup_result_payload(result)
        )
        replayed = pilot.validate_prospective_n32_followup_archive(archived)
        self.assertEqual(replayed["acceptance"], result.acceptance)

    def test_n32_runner_replays_one_failed_family_with_stream_prefix(self):
        parent = _n32_parent_payload()
        followup_blocks = _n32_blocks(parent)
        block = followup_blocks[0]
        failed_case = block["cases"][0]
        failed_family = failed_case["truth_condition"]
        error = RuntimeError("deliberate family acquisition failure")
        failure = {
            "stage": "n32_acquisition_or_scoring",
            "error_type": type(error).__name__,
            "message": str(error),
        }
        failed_case.update(
            {
                "device_token": None,
                "admissible_source_models": None,
                "admissible_source_model_count": None,
                "n32_complete_uncertainty_result": None,
                "primary_prefixes": [
                    pilot._failed_prefix_record(
                        count,
                        pilot.pilot_max_unstable_draws(count),
                        error,
                        failure["stage"],
                    )
                    for count in (16, 32)
                ],
                "pipeline_failures": [failure],
            }
        )
        block["corrected_random_stream_manifest"] = [
            item
            for item in block["corrected_random_stream_manifest"]
            if item["key"]["stream_kind"] == "device_truth"
            or item["key"]["family"] != failed_family
        ]
        block["corrected_random_stream_audit"] = _corrected_audit_from_manifest(
            block["corrected_random_stream_manifest"]
        )
        with patch.object(
            pilot,
            "_run_n32_followup_block",
            side_effect=lambda block_index, *_args: followup_blocks[block_index],
        ):
            result = pilot.run_prospective_n32_followup(
                parent,
                source_revision="a" * 40,
                workers=1,
            )
        self.assertEqual(result.acceptance["n32_pipeline_failure_count"], 1)
        self.assertFalse(result.acceptance["accepted"])
        archived = _reseal_n32_payload(
            pilot.prospective_n32_followup_result_payload(result)
        )
        replayed = pilot.validate_prospective_n32_followup_archive(archived)
        self.assertEqual(replayed["acceptance"], result.acceptance)

    def test_pipeline_failure_blocks_acceptance_even_when_choices_agree(self):
        result = pilot.evaluate_pilot_acceptance(
            _blocks(pipeline_failure=True)
        )
        self.assertEqual(result["pipeline_failure_count"], 1)
        self.assertIsNone(result["recommended_draw_count"])
        self.assertFalse(result["accepted"])

    def test_selected_action_eligibility_differences_are_reported_separately(self):
        candidate = _prefix(4, FIXED_VOLTAGE)
        reference = _prefix(16, FIXED_THERMAL)
        candidate["eligibility"][FIXED_THERMAL] = {
            "eligible": False,
            "failure_reason": "insufficient_stable_prospective_draws",
        }
        comparison = pilot.compare_pilot_choices(candidate, reference)
        self.assertFalse(comparison["agreement"])
        self.assertTrue(comparison["eligibility_changed"])
        self.assertTrue(
            comparison[
                "choice_change_with_selected_action_eligibility_difference"
            ]
        )

    def test_changed_choice_without_reference_utility_fails_regret_gate(self):
        candidate = _prefix(4, FIXED_VOLTAGE)
        reference = _prefix(16, FIXED_THERMAL)
        reference["eligibility"][FIXED_VOLTAGE] = {
            "eligible": False,
            "failure_reason": "insufficient_stable_prospective_draws",
        }
        voltage = next(
            item
            for item in reference["selection"]["action_evaluations"]
            if item["policy_name"] == FIXED_VOLTAGE
        )
        voltage["eligible"] = False
        voltage["utility_per_cost"] = None
        comparison = pilot.compare_pilot_choices(candidate, reference)
        self.assertFalse(comparison["regret_evaluable"])
        self.assertIsNone(comparison["normalized_utility_regret"])
        summary = pilot._summarize_comparisons(
            (comparison,),
            pilot.PilotAcceptanceRule(),
        )
        self.assertFalse(summary["meets_regret_rule"])
        self.assertEqual(summary["changed_choice_count"], 1)
        self.assertEqual(summary["unevaluable_changed_choice_count"], 1)

    def test_reference_selection_failures_block_overall_pilot_gate(self):
        blocks = _blocks()
        for block in blocks:
            for case in block["cases"]:
                for prefix in case["primary_prefixes"]:
                    prefix["choice_token"] = (
                        "selection_failure:no_scored_acquisition_action"
                    )
                    prefix["selection"] = None
                    prefix["eligibility"] = {}
        result = pilot.evaluate_pilot_acceptance(blocks)
        self.assertEqual(result["stability_recommended_draw_count"], 4)
        self.assertEqual(result["n16_selection_failure_count"], 12)
        self.assertFalse(result["feasibility_gate_passed"])
        self.assertIsNone(result["recommended_draw_count"])
        self.assertFalse(result["accepted"])

    def test_reference_ineligible_action_blocks_overall_pilot_gate(self):
        blocks = _blocks()
        reference = blocks[0]["cases"][0]["primary_prefixes"][-1]
        reference["eligibility"][FIXED_FACE_TEMPERATURE] = {
            "eligible": False,
            "failure_reason": "insufficient_stable_prospective_draws",
        }
        result = pilot.evaluate_pilot_acceptance(blocks)
        self.assertEqual(result["n16_ineligible_action_count"], 1)
        self.assertEqual(result["n16_cases_with_ineligible_actions"], 1)
        self.assertFalse(result["feasibility_gate_passed"])
        self.assertFalse(result["n32_followup_required"])
        self.assertIsNone(result["recommended_draw_count"])
        self.assertFalse(result["accepted"])

    def test_draw_diagnostics_separate_failures_from_candidate_transitions(self):
        draws = (
            SimpleNamespace(
                policy_name="fixed_voltage",
                failed=False,
                initially_admissible_became_inadmissible=("five_state",),
                initially_excluded_became_admissible=(),
                candidate_outcomes=(
                    SimpleNamespace(
                        model_name="five_state",
                        status="fit_reached_bound",
                    ),
                    SimpleNamespace(model_name="four_state", status="admissible"),
                ),
            ),
            SimpleNamespace(
                policy_name="fixed_voltage",
                failed=True,
                initially_admissible_became_inadmissible=(),
                initially_excluded_became_admissible=(),
                candidate_outcomes=(),
            ),
        )
        result = pilot._draw_diagnostics(draws)
        self.assertEqual(result["retained_draw_count"], 2)
        self.assertEqual(result["failed_draw_count"], 1)
        self.assertEqual(result["candidate_transition_draw_count"], 1)
        self.assertEqual(result["candidate_loss_draw_count"], 1)
        self.assertEqual(result["candidate_recovery_draw_count"], 0)
        self.assertEqual(
            result["candidate_loss_status_counts"],
            {"fit_reached_bound": 1},
        )

    def test_acceptance_reports_n16_draw_diagnostics_separately(self):
        blocks = _blocks()
        for block in blocks:
            for case in block["cases"]:
                reference = case["primary_prefixes"][-1]
                reference["draw_diagnostics"].update(
                    {
                        "failed_draw_count": 1,
                        "candidate_transition_draw_count": 3,
                        "candidate_loss_draw_count": 2,
                        "candidate_recovery_draw_count": 1,
                        "candidate_loss_status_counts": {
                            "fit_reached_bound": 2,
                        },
                    }
                )
        result = pilot.evaluate_pilot_acceptance(blocks)
        diagnostics = result["n16_draw_diagnostics"]
        self.assertEqual(diagnostics["retained_draw_count"], 12 * 96)
        self.assertEqual(diagnostics["failed_draw_count"], 12)
        self.assertEqual(diagnostics["candidate_transition_draw_count"], 36)
        self.assertEqual(diagnostics["candidate_loss_draw_count"], 24)
        self.assertEqual(diagnostics["candidate_recovery_draw_count"], 12)
        self.assertEqual(
            diagnostics["candidate_loss_status_counts"],
            {"fit_reached_bound": 24},
        )

    def test_scientific_payload_is_independent_of_worker_order_and_timings(self):
        blocks = _blocks()
        acceptance = pilot.evaluate_pilot_acceptance(blocks)
        first = pilot.prospective_pilot_scientific_payload(
            source_revision="a" * 40,
            protocol_digest="b" * 64,
            block_results=blocks,
            acceptance=acceptance,
        )
        changed = deepcopy(list(reversed(blocks)))
        for block in changed:
            block["timing"]["block_wall_seconds"] += 10_000.0
            for case in block["cases"]:
                case["timing"]["case_wall_seconds"] += 10_000.0
        second = pilot.prospective_pilot_scientific_payload(
            source_revision="a" * 40,
            protocol_digest="b" * 64,
            block_results=changed,
            acceptance=acceptance,
        )
        self.assertEqual(first, second)
        self.assertNotIn("timing", json.dumps(first))

    def test_family_runner_generates_once_prefixes_authentically_and_saves_before_reveal(self):
        events = []
        prefix_calls = []
        policies = default_fixed_policies()

        def build_case(_family, _block, policy, *_args):
            return SimpleNamespace(
                case_id=SimpleNamespace(device_token="device-token"),
                acquisition_runs=(f"initial-{policy.name}",),
                final_regime="final-regime",
                policy=policy,
            )

        evidence = SimpleNamespace(
            evidence_digest="e" * 64,
            snapshot=SimpleNamespace(
                admissible_candidate_models=("four_state",)
            ),
        )
        n16 = SimpleNamespace(name="n16")

        def prefix_result(_result, *, draw_count, max_unstable_draws_per_source_action):
            prefix_calls.append((draw_count, max_unstable_draws_per_source_action))
            return SimpleNamespace(
                config=SimpleNamespace(
                    draw_count=draw_count,
                    max_unstable_draws_per_source_action=(
                        max_unstable_draws_per_source_action
                    ),
                )
            )

        def prefix_record(result, **_kwargs):
            selected = FIXED_THERMAL
            return {
                "draw_count": result.config.draw_count,
                "max_unstable_draws_per_source_action": (
                    result.config.max_unstable_draws_per_source_action
                ),
                "choice_token": f"action:{selected}",
                "eligibility": {},
                "selection": {"selected_policy": selected},
                "uncertainty_summary": {"authenticated": True},
                "costed_scorecard": {"authenticated": True},
            }

        def save_decision(case, **_kwargs):
            events.append(("save", case.policy.name))
            return {"saved": case.policy.name}

        def reveal(saved, **_kwargs):
            events.append(("reveal", saved["saved"]))
            return SimpleNamespace(
                revealed={"true_margin": 1.0},
                scored={"decision": "approve"},
                nominal_selection_energy=1.0,
                realized_energy={"total": 1.0},
            )

        complete_n16 = {
            "acquisition_evidence": {
                "snapshot": {"admissible_candidate_models": ["four_state"]}
            },
            "draw_outcomes": [{"draw_index": index} for index in range(16)],
        }
        partition = CorrectedPartition(
            pilot.PROSPECTIVE_PILOT_PARTITION,
            pilot.PILOT_BLOCK_COUNT,
            pilot.PROSPECTIVE_CAMPAIGN,
        )
        with (
            patch.object(pilot, "build_corrected_blinded_case", side_effect=build_case),
            patch.object(
                pilot,
                "prepare_prospective_acquisition_evidence",
                return_value=evidence,
            ),
            patch.object(
                pilot,
                "estimate_prospective_action_uncertainty",
                return_value=n16,
            ) as estimate,
            patch.object(
                pilot,
                "prospective_uncertainty_result_payload",
                return_value=complete_n16,
            ),
            patch.object(
                pilot,
                "prefix_prospective_uncertainty_result",
                side_effect=prefix_result,
            ),
            patch.object(pilot, "_prefix_record", side_effect=prefix_record),
            patch.object(
                pilot,
                "decide_corrected_blinded_case",
                side_effect=save_decision,
            ),
            patch.object(
                pilot,
                "score_corrected_saved_decision",
                side_effect=reveal,
            ),
            patch.object(pilot, "_case_payload", return_value={"case": True}),
            patch.object(pilot, "_saved_payload", side_effect=lambda value: value),
        ):
            result = pilot._run_pilot_family(
                truth_condition=pilot.STAGE3_TRUTH_CONDITIONS[0],
                block=0,
                truth={"hidden": "truth"},
                partition=partition,
                physical_config=OperatingDecisionRealismConfig(),
                registry=RandomStreamRegistry(),
                selector_rule=ProspectiveSelectorRule(),
                cost_scenario=PRIMARY_PROSPECTIVE_COST_SCENARIO,
            )

        estimate.assert_called_once()
        config = estimate.call_args.args[3]
        self.assertEqual(config.draw_count, 16)
        self.assertEqual(config.max_unstable_draws_per_source_action, 1)
        self.assertEqual(prefix_calls, [(4, 0), (8, 0), (16, 1), (16, 0)])
        first_reveal = next(index for index, item in enumerate(events) if item[0] == "reveal")
        self.assertEqual(first_reveal, len(policies))
        self.assertTrue(all(item[0] == "save" for item in events[:first_reveal]))
        self.assertIs(result["n16_complete_uncertainty_result"], complete_n16)
        self.assertTrue(
            all("uncertainty_result" not in item for item in result["primary_prefixes"])
        )

    def test_n32_family_generates_once_and_derives_primary_n16_prefix(self):
        prefix_calls = []
        case = SimpleNamespace(
            case_id=SimpleNamespace(device_token="device-token"),
            acquisition_runs=("initial",),
            final_regime="final-regime",
        )
        evidence = SimpleNamespace(
            evidence_digest="e" * 64,
            snapshot=SimpleNamespace(
                admissible_candidate_models=("four_state",)
            ),
        )
        n32 = SimpleNamespace(name="n32")

        def prefix_result(_result, *, draw_count, max_unstable_draws_per_source_action):
            prefix_calls.append((draw_count, max_unstable_draws_per_source_action))
            return SimpleNamespace(
                config=SimpleNamespace(
                    draw_count=draw_count,
                    max_unstable_draws_per_source_action=(
                        max_unstable_draws_per_source_action
                    ),
                )
            )

        def prefix_record(result, **_kwargs):
            return _prefix(result.config.draw_count, FIXED_THERMAL)

        partition = CorrectedPartition(
            pilot.PROSPECTIVE_PILOT_PARTITION,
            pilot.PILOT_BLOCK_COUNT,
            pilot.PROSPECTIVE_CAMPAIGN,
        )
        with (
            patch.object(pilot, "build_corrected_blinded_case", return_value=case),
            patch.object(
                pilot,
                "prepare_prospective_acquisition_evidence",
                return_value=evidence,
            ),
            patch.object(
                pilot,
                "estimate_prospective_action_uncertainty",
                return_value=n32,
            ) as estimate,
            patch.object(
                pilot,
                "prospective_uncertainty_result_payload",
                return_value={"draw_count": 32},
            ),
            patch.object(
                pilot,
                "prefix_prospective_uncertainty_result",
                side_effect=prefix_result,
            ),
            patch.object(pilot, "_prefix_record", side_effect=prefix_record),
        ):
            result = pilot._run_n32_followup_family(
                truth_condition=pilot.STAGE3_TRUTH_CONDITIONS[0],
                block=0,
                truth={"hidden": "truth"},
                partition=partition,
                physical_config=OperatingDecisionRealismConfig(),
                registry=RandomStreamRegistry(),
                selector_rule=ProspectiveSelectorRule(),
                cost_scenario=PRIMARY_PROSPECTIVE_COST_SCENARIO,
            )
        config = estimate.call_args.args[3]
        self.assertEqual(config.draw_count, 32)
        self.assertEqual(config.max_unstable_draws_per_source_action, 1)
        self.assertEqual(prefix_calls, [(16, 1), (32, 1)])
        self.assertEqual(
            [item["draw_count"] for item in result["primary_prefixes"]],
            [16, 32],
        )

    def test_result_rejects_a_forged_scientific_digest(self):
        blocks = _blocks()
        acceptance = pilot.evaluate_pilot_acceptance(blocks)
        with self.assertRaisesRegex(ValueError, "scientific result digest"):
            pilot.ProspectivePilotResult(
                source_revision="a" * 40,
                worker_count=4,
                protocol_digest="b" * 64,
                scientific_result_digest="c" * 64,
                block_results=tuple(blocks),
                acceptance=acceptance,
                compute_budget_inputs={},
                wall_seconds=1.0,
                cpu_seconds=2.0,
                peak_rss_bytes=3,
            )

    def test_cli_source_preflight_rejects_dirty_head(self):
        dirty = Mock(stdout=" M scientific.py\n")
        with patch("subprocess.run", return_value=dirty):
            with self.assertRaisesRegex(ValueError, "clean committed"):
                _committed_source_revision(Path("."))

    def test_cli_source_preflight_rejects_head_revision_mismatch(self):
        responses = (Mock(stdout=""), Mock(stdout="b" * 40 + "\n"))
        with patch("subprocess.run", side_effect=responses):
            with self.assertRaisesRegex(ValueError, "does not equal clean HEAD"):
                _require_clean_head(Path("."), "a" * 40)

    def test_cli_rejects_an_unrelated_repository_root(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "imported source tree"):
                _require_project_root(Path(directory))

    def test_cli_rechecks_clean_head_after_run_before_writing(self):
        revision = "a" * 40
        saved = SimpleNamespace(
            json_path=Path("pilot.json"),
            report_path=Path("pilot.txt"),
            hash_path=Path("pilot.sha256"),
            archive_size_bytes=1,
            json_sha256="b" * 64,
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                pilot_cli,
                "_committed_source_revision",
                return_value=revision,
            ),
            patch.object(pilot_cli, "_require_clean_head") as clean,
            patch.object(
                pilot_cli,
                "run_prospective_draw_count_pilot",
                return_value=object(),
            ),
            patch.object(
                pilot_cli,
                "save_prospective_pilot_artifacts",
                return_value=saved,
            ) as save,
        ):
            pilot_cli.main(
                (
                    "--execute-disposable-pilot",
                    "--json",
                    str(Path(directory) / "pilot.json"),
                )
            )
        self.assertEqual(clean.call_count, 2)
        self.assertEqual(clean.call_args_list[-1].args[1], revision)
        save.assert_called_once()

    def test_cli_n32_mode_requires_parent_and_uses_separate_runner(self):
        revision = "a" * 40
        saved = SimpleNamespace(
            json_path=Path("n32.json"),
            report_path=Path("n32.txt"),
            hash_path=Path("n32.sha256"),
            archive_size_bytes=1,
            json_sha256="b" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent_path = root / "parent-pilot.json"
            parent_path.write_text('{"parent": true}', encoding="utf-8")
            with (
                patch.object(
                    pilot_cli,
                    "_committed_source_revision",
                    return_value=revision,
                ),
                patch.object(pilot_cli, "_require_clean_head"),
                patch.object(
                    pilot_cli,
                    "run_prospective_n32_followup",
                    return_value=object(),
                ) as run,
                patch.object(
                    pilot_cli,
                    "save_prospective_n32_followup_artifacts",
                    return_value=saved,
                ) as save,
            ):
                pilot_cli.main(
                    (
                        "--execute-n32-followup",
                        "--parent-pilot-json",
                        str(parent_path),
                        "--json",
                        str(root / "n32.json"),
                    )
                )
        self.assertEqual(run.call_args.args[0], {"parent": True})
        self.assertEqual(run.call_args.kwargs["source_revision"], revision)
        save.assert_called_once()


class ProspectivePilotArchiveTests(unittest.TestCase):
    def _result(self):
        blocks = _blocks()
        acceptance = pilot.evaluate_pilot_acceptance(blocks)
        scientific = pilot.prospective_pilot_scientific_payload(
            source_revision="a" * 40,
            protocol_digest="b" * 64,
            block_results=blocks,
            acceptance=acceptance,
        )
        digest = pilot._digest(
            "thermotwin.prospective_pilot.scientific_result",
            scientific,
        )
        budget = {
            "one_source_case_count": 6,
            "two_source_case_count": 6,
            "zero_source_case_count": 0,
            "unknown_source_case_count": 0,
            "measured_worker_count": 4,
            "runtime_host_manifest": {
                "python_version": "3.10.12",
                "operating_system": "Darwin",
                "machine": "arm64",
            },
            "conservative_measured_throughput_phase_estimates": [
                {
                    "partition": spec.name,
                    "estimated_wall_seconds_at_measured_worker_count": 10.0,
                    "estimated_cpu_seconds": 20.0,
                }
                for spec in pilot.PROSPECTIVE_PARTITION_PLAN[1:]
            ],
            "draw_count_linear_projections": [
                {
                    "draw_count": count,
                    "estimated_full_campaign_wall_seconds_at_measured_concurrency": 100.0,
                    "estimated_full_campaign_cpu_seconds": 200.0,
                }
                for count in (4, 8, 16)
            ],
        }
        return pilot.ProspectivePilotResult(
            source_revision="a" * 40,
            worker_count=4,
            protocol_digest="b" * 64,
            scientific_result_digest=digest,
            block_results=tuple(blocks),
            acceptance=acceptance,
            compute_budget_inputs=budget,
            wall_seconds=1.0,
            cpu_seconds=2.0,
            peak_rss_bytes=3,
        )

    def test_complete_archive_records_exact_size_and_detached_hashes(self):
        result = self._result()
        result.compute_budget_inputs["assumptions"] = (
            "json",
            "roundtrip",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(
                pilot,
                "validate_prospective_pilot_archive",
                side_effect=lambda payload: payload,
            ) as validate:
                saved = pilot.save_prospective_pilot_artifacts(
                    result,
                    json_path=root / "pilot.json",
                    report_path=root / "pilot.txt",
                    hash_path=root / "pilot.sha256",
                )
            validate.assert_called_once()
            payload = json.loads(saved.json_path.read_text(encoding="utf-8"))
            self.assertEqual(validate.call_args.args[0], payload)
            self.assertEqual(payload["archive_size_bytes"], saved.archive_size_bytes)
            self.assertEqual(saved.archive_size_bytes, saved.json_path.stat().st_size)
            self.assertEqual(
                saved.json_sha256,
                hashlib.sha256(saved.json_path.read_bytes()).hexdigest(),
            )
            manifest = saved.hash_path.read_text(encoding="utf-8")
            self.assertIn(saved.json_sha256, manifest)
            self.assertIn(saved.report_sha256, manifest)

    def test_complete_p3_archive_round_trip_and_tamper_detection(self):
        payload = json.loads(pilot._canonical_bytes(_n32_parent_payload()))
        self.assertEqual(
            pilot.validate_prospective_pilot_archive(payload),
            payload,
        )
        result = pilot.ProspectivePilotResult(
            source_revision=payload["source_revision"],
            worker_count=payload["worker_count"],
            protocol_digest=payload["protocol_digest"],
            scientific_result_digest=payload["scientific_result_digest"],
            block_results=tuple(payload["block_results"]),
            acceptance=payload["acceptance"],
            compute_budget_inputs=payload["compute_budget_inputs"],
            wall_seconds=payload["performance"]["wall_seconds"],
            cpu_seconds=payload["performance"]["cpu_seconds"],
            peak_rss_bytes=payload["performance"]["peak_rss_bytes"],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved = pilot.save_prospective_pilot_artifacts(
                result,
                json_path=root / "pilot.json",
                report_path=root / "pilot.txt",
                hash_path=root / "pilot.sha256",
            )
            saved_payload = json.loads(
                saved.json_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                pilot.validate_prospective_pilot_archive(saved_payload),
                saved_payload,
            )
        tampered = deepcopy(payload)
        tampered["block_results"][0]["cases"][0]["primary_prefixes"][0][
            "choice_token"
        ] = "selection_failure:forged"
        _reseal_parent_payload(tampered)
        with self.assertRaisesRegex(ValueError, "choice token"):
            pilot.validate_prospective_pilot_archive(tampered)

        tampered = _n32_parent_payload()
        tampered["block_results"][0]["cases"][0][
            "selection_information_boundary"
        ]["inputs"].append("truth")
        _reseal_parent_payload(tampered)
        with self.assertRaisesRegex(ValueError, "case evidence"):
            pilot.validate_prospective_pilot_archive(tampered)

        tampered = _n32_parent_payload()
        tampered["block_results"][0]["cases"][0]["fixed_policy_results"][
            STOP_NOW
        ]["case"]["policy"]["name"] = FIXED_VOLTAGE
        _reseal_parent_payload(tampered)
        with self.assertRaisesRegex(ValueError, "case identity"):
            pilot.validate_prospective_pilot_archive(tampered)

    def test_complete_pilot_archive_replays_canonical_failed_blocks(self):
        for failed_block in (0, 2):
            with self.subTest(failed_block=failed_block):
                payload = _n32_parent_payload()
                payload["block_results"][failed_block] = pilot._failed_block_result(
                    failed_block,
                    RuntimeError("deliberate pilot block failure"),
                )
                _reseal_parent_scientific(payload)
                replayed = pilot.validate_prospective_pilot_archive(payload)
                self.assertFalse(replayed["acceptance"]["accepted"])

        payload = _n32_parent_payload()
        payload["block_results"] = [
            pilot._failed_block_result(
                block,
                RuntimeError("deliberate all-block pilot failure"),
            )
            for block in range(pilot.PILOT_BLOCK_COUNT)
        ]
        _reseal_parent_scientific(payload)
        replayed = pilot.validate_prospective_pilot_archive(payload)
        self.assertFalse(replayed["acceptance"]["accepted"])

    def test_failed_pilot_prefixes_must_match_their_owning_failure(self):
        forged = {
            "stage": "forged_stage",
            "error_type": "ForgedError",
            "message": "forged message",
        }
        payload = _n32_parent_payload()
        payload["block_results"][0] = pilot._failed_block_result(
            0,
            RuntimeError("deliberate pilot block failure"),
        )
        prefix = payload["block_results"][0]["cases"][0]["primary_prefixes"][0]
        prefix["pipeline_failure"] = forged
        prefix["choice_token"] = "pipeline_failure:forged_stage:ForgedError"
        _reseal_parent_scientific(payload)
        with self.assertRaisesRegex(ValueError, "match its block failure"):
            pilot.validate_prospective_pilot_archive(payload)

        payload = _n32_parent_payload()
        case = payload["block_results"][0]["cases"][0]
        error = RuntimeError("deliberate N16 acquisition failure")
        owner = {
            "stage": "n16_acquisition_or_scoring",
            "error_type": type(error).__name__,
            "message": str(error),
        }
        case.update(
            {
                "admissible_source_models": None,
                "admissible_source_model_count": None,
                "n16_complete_uncertainty_result": None,
                "primary_prefixes": [
                    pilot._failed_prefix_record(
                        count,
                        pilot.pilot_max_unstable_draws(count),
                        error,
                        owner["stage"],
                    )
                    for count in pilot.PILOT_DRAW_COUNTS
                ],
                "n16_max_unstable_0_sensitivity": pilot._failed_prefix_record(
                    16, 0, error, owner["stage"]
                ),
                "selected_policy_at_n16_primary": None,
                "selected_policy_outcome": None,
                "pipeline_failures": [owner],
            }
        )
        prefix = case["primary_prefixes"][0]
        prefix["pipeline_failure"] = forged
        prefix["choice_token"] = "pipeline_failure:forged_stage:ForgedError"
        _reseal_parent_scientific(payload)
        with self.assertRaisesRegex(ValueError, "match its N=16 failure"):
            pilot.validate_prospective_pilot_archive(payload)

    def test_complete_pilot_archive_replays_case_level_failures(self):
        payload = _n32_parent_payload()
        case = payload["block_results"][0]["cases"][0]
        error = RuntimeError("deliberate N16 acquisition failure")
        failure = {
            "stage": "n16_acquisition_or_scoring",
            "error_type": type(error).__name__,
            "message": str(error),
        }
        case.update(
            {
                "admissible_source_models": None,
                "admissible_source_model_count": None,
                "n16_complete_uncertainty_result": None,
                "primary_prefixes": [
                    pilot._failed_prefix_record(
                        count,
                        pilot.pilot_max_unstable_draws(count),
                        error,
                        failure["stage"],
                    )
                    for count in pilot.PILOT_DRAW_COUNTS
                ],
                "n16_max_unstable_0_sensitivity": pilot._failed_prefix_record(
                    16, 0, error, failure["stage"]
                ),
                "selected_policy_at_n16_primary": None,
                "selected_policy_outcome": None,
                "pipeline_failures": [failure],
            }
        )
        _reseal_parent_scientific(payload)
        self.assertFalse(
            pilot.validate_prospective_pilot_archive(payload)["acceptance"][
                "accepted"
            ]
        )

        payload = _n32_parent_payload()
        case = payload["block_results"][0]["cases"][0]
        fixed = case["fixed_policy_results"][STOP_NOW]
        fixed.update(
            {
                "saved_before_reveal": None,
                "post_reveal_score": None,
                "failure": "decision_not_saved",
            }
        )
        case["pipeline_failures"] = [
            {
                "stage": f"fixed_policy_save:{STOP_NOW}",
                "error_type": "RuntimeError",
                "message": "deliberate save failure",
            }
        ]
        _reseal_parent_scientific(payload)
        self.assertFalse(
            pilot.validate_prospective_pilot_archive(payload)["acceptance"][
                "accepted"
            ]
        )

        payload = _n32_parent_payload()
        case = payload["block_results"][0]["cases"][0]
        fixed = case["fixed_policy_results"][STOP_NOW]
        fixed["post_reveal_score"] = None
        fixed["failure"] = {
            "error_type": "RuntimeError",
            "message": "deliberate reveal failure",
        }
        case["pipeline_failures"] = [
            {
                "stage": f"fixed_policy_reveal:{STOP_NOW}",
                "error_type": "RuntimeError",
                "message": "deliberate reveal failure",
            }
        ]
        _reseal_parent_scientific(payload)
        self.assertFalse(
            pilot.validate_prospective_pilot_archive(payload)["acceptance"][
                "accepted"
            ]
        )

    def test_all_failed_pilot_run_is_saved_and_replayed(self):
        with patch.object(
            pilot,
            "_run_pilot_block",
            side_effect=RuntimeError("deliberate all-block failure"),
        ):
            result = pilot.run_prospective_draw_count_pilot(
                source_revision="a" * 40,
                workers=1,
            )
        self.assertEqual(result.acceptance["pipeline_failure_count"], 12)
        self.assertFalse(result.acceptance["accepted"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved = pilot.save_prospective_pilot_artifacts(
                result,
                json_path=root / "pilot.json",
                report_path=root / "pilot.txt",
                hash_path=root / "pilot.sha256",
            )
            payload = json.loads(saved.json_path.read_text(encoding="utf-8"))
            replayed = pilot.validate_prospective_pilot_archive(payload)
        self.assertFalse(replayed["acceptance"]["accepted"])

        payload = _n32_parent_payload()
        case = payload["block_results"][0]["cases"][0]
        error = RuntimeError("deliberate prefix failure")
        case["primary_prefixes"][0] = pilot._failed_prefix_record(
            4, 0, error, "prefix_score"
        )
        case["pipeline_failures"] = [
            {
                "stage": "prefix_score_n4",
                "error_type": type(error).__name__,
                "message": str(error),
            }
        ]
        _reseal_parent_scientific(payload)
        self.assertFalse(
            pilot.validate_prospective_pilot_archive(payload)["acceptance"][
                "accepted"
            ]
        )

    def test_archive_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "pilot.json"
            json_path.write_text("existing", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                pilot.save_prospective_pilot_artifacts(
                    self._result(),
                    json_path=json_path,
                    report_path=root / "pilot.txt",
                    hash_path=root / "pilot.sha256",
                )


if __name__ == "__main__":
    unittest.main()
