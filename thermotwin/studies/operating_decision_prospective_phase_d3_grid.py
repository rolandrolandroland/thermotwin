"""Source-bound Phase D3 grid analysis of preserved D1/D2 evidence.

The analyzer recomputes development-padded action values from retained raw
N=16 predictive draws, selects an action for every frozen rule, routes that
choice to the already-saved fixed-policy outcome, and applies the corresponding
D2 offset to the final interval.  It performs no simulation or refitting.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Mapping, Optional, Sequence

from .operating_decision import (
    APPROVE,
    INSUFFICIENT_EVIDENCE,
    POLICY_NAMES,
    REJECT,
    STOP_NOW,
    MarginEnvelope,
    MarginInterval,
)
from .operating_decision_prospective import (
    CandidateExclusion,
    ProspectiveAcquisitionSnapshot,
    ProspectiveActionEvaluation,
    ProspectiveDevelopmentOffsets,
    ProspectiveSelectorRule,
    prospective_development_offsets_from_payload,
    prospective_selector_rule_payload,
    select_prospective_action,
)
from .operating_decision_prospective_phase_d import (
    PHASE_D_HASH_DOMAIN,
    PHASE_D_RULE_GRID_SIZE,
    evaluate_phase_d_rule_objective,
    phase_d_development_adjusted_decision,
    phase_d_rule_grid,
    phase_d_runtime_identity,
    validate_executing_phase_d_runtime,
)
from .operating_decision_prospective_phase_d2_offsets import (
    PHASE_D2_ANALYSIS_SOURCE_PATHS,
    validate_phase_d2_offset_analysis,
)
from .operating_decision_prospective_pilot import (
    DEVELOPMENT_TUNING_BLOCK_COUNT,
    PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION,
    _canonical_bytes,
    _digest,
    _exact_mapping,
    _validate_revision,
    _validate_sha256,
)
from .operating_decision_provenance import (
    create_source_manifest,
    source_manifest_from_payload,
    source_manifest_payload,
    verify_source_manifest,
)
from .operating_decision_realism import STAGE3_TRUTH_CONDITIONS


PHASE_D3_SCHEMA_VERSION = 1
PHASE_D3_PROTOCOL_VERSION = "operating_decision_phase_d3_grid_v1"
PHASE_D3_JSON_NAME = "p1_development_grid.json"
PHASE_D3_REPORT_NAME = "p1_development_grid.txt"
PHASE_D3_HASH_NAME = "p1_development_grid.sha256"
PHASE_D3_ANALYSIS_SOURCE_PATHS = tuple(
    sorted(
        set(PHASE_D2_ANALYSIS_SOURCE_PATHS).union(
            {
                "thermotwin/reports/operating_decision_prospective_phase_d3.py",
                "thermotwin/studies/operating_decision_prospective_phase_d3_grid.py",
            }
        )
    )
)


@dataclass(frozen=True)
class SavedPhaseD3Artifacts:
    json_path: Path
    report_path: Path
    hash_path: Path
    json_sha256: str
    report_sha256: str
    archive_size_bytes: int


def _snapshot_from_payload(payload: Mapping[str, object]) -> ProspectiveAcquisitionSnapshot:
    envelope = payload["provisional_margin_envelope"]
    return ProspectiveAcquisitionSnapshot(
        candidate_models=tuple(payload["candidate_models"]),
        admissible_candidate_models=tuple(payload["admissible_candidate_models"]),
        excluded_candidates=tuple(
            CandidateExclusion(item["model_name"], item["reason"])
            for item in payload["excluded_candidates"]
        ),
        failed_candidate_models=tuple(payload["failed_candidate_models"]),
        model_intervals=tuple(MarginInterval(**item) for item in payload["model_intervals"]),
        provisional_margin_envelope=(
            None if envelope is None else MarginEnvelope(**envelope)
        ),
        selection_failure_reason=payload["selection_failure_reason"],
    )


def _primary_n16_prefix(case: Mapping[str, object]) -> Mapping[str, object]:
    matches = tuple(
        item for item in case["primary_prefixes"] if item["draw_count"] == 16
    )
    if len(matches) != 1:
        raise ValueError("Phase D3 needs one primary N=16 prefix per case")
    if matches[0]["max_unstable_draws_per_source_action"] != 1:
        raise ValueError("Phase D3 primary N=16 prefix lost its frozen allowance")
    return matches[0]


def _padded_action_evaluations(
    complete: Mapping[str, object],
    scorecard: Mapping[str, object],
    offsets: ProspectiveDevelopmentOffsets,
) -> tuple[ProspectiveActionEvaluation, ...]:
    """Recompute padded evaluations directly from authenticated raw draws."""

    uncertainties = {item["policy_name"]: item for item in complete["action_uncertainties"]}
    costs = {item["policy_name"]: item for item in scorecard["action_costs"]}
    if set(uncertainties) != set(POLICY_NAMES) or set(costs) != set(POLICY_NAMES):
        raise ValueError("Phase D3 action evidence does not cover the frozen catalog")
    snapshot = complete["acquisition_evidence"]["snapshot"]
    envelope = snapshot["provisional_margin_envelope"]
    if not isinstance(envelope, Mapping):
        raise ValueError("Phase D3 needs a finite provisional envelope")
    raw_baseline = float(envelope["upper"]) - float(envelope["lower"])
    padded_baseline = raw_baseline + 2.0 * offsets.stop_now
    draws = tuple(complete["draw_outcomes"])
    evaluations = [ProspectiveActionEvaluation(STOP_NOW, True)]
    for policy_name in POLICY_NAMES[1:]:
        raw = uncertainties[policy_name]
        if not raw["eligible"]:
            evaluations.append(
                ProspectiveActionEvaluation(
                    policy_name=policy_name,
                    eligible=False,
                    failure_reason=raw["failure_reason"],
                )
            )
            continue
        action_offset = offsets.for_policy(policy_name)
        source_means = []
        for summary in raw["source_summaries"]:
            selected = tuple(
                draw
                for draw in draws
                if draw["policy_name"] == policy_name
                and draw["generator_model"] == summary["generator_model"]
            )
            if len(selected) != summary["draw_count"]:
                raise ValueError("Phase D3 draw records do not match source summaries")
            scored = []
            for draw in selected:
                if draw["failed"]:
                    width = padded_baseline
                else:
                    raw_width = draw["raw_after_width"]
                    if (
                        isinstance(raw_width, bool)
                        or not isinstance(raw_width, (int, float))
                        or not math.isfinite(float(raw_width))
                        or float(raw_width) < 0.0
                    ):
                        raise ValueError("Phase D3 retained raw draw width is invalid")
                    width = float(raw_width) + 2.0 * action_offset
                    if draw["initially_admissible_became_inadmissible"]:
                        width = max(padded_baseline, width)
                scored.append(width)
            source_means.append(fmean(scored))
        if not source_means:
            raise ValueError("Phase D3 eligible action has no source summaries")
        evaluations.append(
            ProspectiveActionEvaluation(
                policy_name=policy_name,
                eligible=True,
                uncertainty_before=padded_baseline,
                expected_uncertainty_after=max(source_means),
                declared_cost=float(costs[policy_name]["declared_cost"]),
                prospective_draw_count=int(raw["prospective_draw_count"]),
                raw_uncertainty_before=float(raw["uncertainty_before"]),
                raw_expected_uncertainty_after=float(
                    raw["expected_uncertainty_after"]
                ),
                development_offset=action_offset,
            )
        )
    return tuple(evaluations)


def _action_evaluation_payload(value: ProspectiveActionEvaluation) -> dict:
    return {
        "policy_name": value.policy_name,
        "eligible": value.eligible,
        "failure_reason": value.failure_reason,
        "uncertainty_before": value.uncertainty_before,
        "expected_uncertainty_after": value.expected_uncertainty_after,
        "expected_uncertainty_reduction": value.expected_uncertainty_reduction,
        "declared_cost": value.declared_cost,
        "utility_per_cost": value.utility_per_cost,
        "prospective_draw_count": value.prospective_draw_count,
        "raw_uncertainty_before": value.raw_uncertainty_before,
        "raw_expected_uncertainty_after": value.raw_expected_uncertainty_after,
        "raw_expected_uncertainty_reduction": (
            value.raw_expected_uncertainty_reduction
        ),
        "development_offset": value.development_offset,
    }


def _selection_payload(selection) -> dict:
    padded = selection.padded_initial_margin_envelope
    return {
        "selected_policy": selection.selected_policy,
        "selection_succeeded": selection.selection_succeeded,
        "requires_additional_acquisition": selection.requires_additional_acquisition,
        "verification_required": selection.verification_required,
        "reason": selection.reason,
        "provisional_decision": selection.provisional_decision,
        "ranked_policies": list(selection.ranked_policies),
        "tied_policies": list(selection.tied_policies),
        "admissible_candidate_models": list(selection.admissible_candidate_models),
        "candidate_reliability_stratum": selection.candidate_reliability_stratum,
        "padded_initial_margin_envelope": (
            None if padded is None else {"lower": padded.lower, "upper": padded.upper}
        ),
        "effective_stopping_clearance": selection.effective_stopping_clearance,
        "selector_protocol_digest": selection.selector_protocol_digest,
        "action_evaluations": [
            _action_evaluation_payload(item) for item in selection.action_evaluations
        ],
    }


def _record_failure_reason(record: Mapping[str, object]) -> Optional[str]:
    failure = record.get("failure")
    if failure is None:
        return None
    if isinstance(failure, str):
        return failure
    if isinstance(failure, Mapping):
        return f"{failure.get('error_type', 'pipeline_failure')}:{failure.get('message', '')}"
    return "malformed_fixed_policy_failure"


def _selected_outcome_record(
    case: Mapping[str, object],
    *,
    block: int,
    policy_name: Optional[str],
    selection_payload: Mapping[str, object],
    offsets: ProspectiveDevelopmentOffsets,
    resources: Mapping[str, Mapping[str, object]],
    costs: Mapping[str, Mapping[str, object]],
) -> dict:
    if policy_name is None:
        reference = case["fixed_policy_results"][STOP_NOW]["post_reveal_score"]
        true_margin = float(reference["revealed"]["true_margin"])
        adjusted = phase_d_development_adjusted_decision(
            None,
            0.0,
            failure_reason=f"selection_failure:{selection_payload['reason']}",
        )
        proxy = {
            "added_run_count": 0,
            "added_sensor_count": 0,
            "normalized_incremental_energy": 0.0,
        }
        realized = {
            "diagnostic_run_count": None,
            "energized_schedule_time_seconds": None,
            "extra_sensor_count": None,
            "total_diagnostic_energy": None,
        }
        verification_failure = False
        pipeline_failure = True
        raw_interval_covered = None
        adjusted_interval_covered = None
    else:
        fixed = case["fixed_policy_results"][policy_name]
        saved = fixed.get("saved_before_reveal")
        post = fixed.get("post_reveal_score")
        revealed = post.get("revealed") if isinstance(post, Mapping) else None
        true_margin = float(revealed["true_margin"])
        raw_interval = saved.get("margin_envelope") if isinstance(saved, Mapping) else None
        verifications = saved.get("verifications", []) if isinstance(saved, Mapping) else []
        verification_succeeded = any(
            item.get("passed") is True for item in verifications if isinstance(item, Mapping)
        )
        fixed_failure = _record_failure_reason(fixed)
        failure_reason = fixed_failure
        if not verification_succeeded and failure_reason is None:
            failure_reason = "no_successful_verification_candidate"
        adjusted = phase_d_development_adjusted_decision(
            (
                None
                if not isinstance(raw_interval, Mapping)
                else [raw_interval.get("lower"), raw_interval.get("upper")]
            ),
            offsets.for_policy(policy_name),
            verification_succeeded=verification_succeeded,
            failure_reason=failure_reason,
        )
        resource = resources[policy_name]
        cost = costs[policy_name]
        proxy = {
            "added_run_count": int(resource["added_run_count"]),
            "added_sensor_count": len(resource["added_instruments"]),
            "normalized_incremental_energy": float(cost["normalized_energy"]),
        }
        scored = post.get("scored") if isinstance(post, Mapping) else None
        realized = {
            name: (scored.get(name) if isinstance(scored, Mapping) else None)
            for name in (
                "diagnostic_run_count",
                "energized_schedule_time_seconds",
                "extra_sensor_count",
                "total_diagnostic_energy",
            )
        }
        verification_failure = not verification_succeeded
        pipeline_failure = fixed_failure is not None
        raw = adjusted["raw_interval"]
        final = adjusted["development_adjusted_interval"]
        raw_interval_covered = (
            None if raw is None else raw["lower"] <= true_margin <= raw["upper"]
        )
        adjusted_interval_covered = (
            None
            if final is None
            else final["lower"] <= true_margin <= final["upper"]
        )
    objective = {
        "block": block,
        "truth_family": case["truth_condition"],
        "adjusted_decision": adjusted,
        "true_margin": true_margin,
        **proxy,
    }
    return {
        "block": block,
        "truth_family": case["truth_condition"],
        "device_token": case["device_token"],
        "candidate_reliability_stratum": selection_payload[
            "candidate_reliability_stratum"
        ],
        "selection": dict(selection_payload),
        "selected_policy": policy_name,
        "ineligible_actions": [
            item["policy_name"]
            for item in selection_payload["action_evaluations"]
            if item["policy_name"] != STOP_NOW and not item["eligible"]
        ],
        "adjusted_decision": adjusted,
        "true_margin": true_margin,
        "raw_interval_covered": raw_interval_covered,
        "development_adjusted_interval_covered": adjusted_interval_covered,
        "verification_failure": verification_failure,
        "pipeline_failure": pipeline_failure,
        "resource_proxy": proxy,
        "realized_resources": realized,
        "objective_input": objective,
    }


def _rule_for_grid(
    values: Sequence[float],
    offsets: ProspectiveDevelopmentOffsets,
) -> ProspectiveSelectorRule:
    general, single, reduction, utility = values
    return ProspectiveSelectorRule(
        development_offsets=offsets,
        stopping_clearance=general,
        single_candidate_stopping_clearance=single,
        minimum_expected_reduction=reduction,
        minimum_utility_per_cost=utility,
    )


def _score_selector_case(
    case: Mapping[str, object],
    *,
    block: int,
    rule: ProspectiveSelectorRule,
    offsets: ProspectiveDevelopmentOffsets,
) -> dict:
    complete = case["n16_complete_uncertainty_result"]
    if not isinstance(complete, Mapping):
        raise ValueError("Phase D3 cannot score a case without N=16 evidence")
    prefix = _primary_n16_prefix(case)
    scorecard = prefix["costed_scorecard"]
    snapshot = _snapshot_from_payload(complete["acquisition_evidence"]["snapshot"])
    evaluations = _padded_action_evaluations(complete, scorecard, offsets)
    selection = select_prospective_action(snapshot, evaluations, rule)
    selection_record = _selection_payload(selection)
    resources = {item["policy_name"]: item for item in scorecard["action_resources"]}
    costs = {item["policy_name"]: item for item in scorecard["action_costs"]}
    return _selected_outcome_record(
        case,
        block=block,
        policy_name=(selection.selected_policy if selection.selection_succeeded else None),
        selection_payload=selection_record,
        offsets=offsets,
        resources=resources,
        costs=costs,
    )


def _fixed_policy_case(
    case: Mapping[str, object],
    *,
    block: int,
    policy_name: str,
    offsets: ProspectiveDevelopmentOffsets,
) -> dict:
    prefix = _primary_n16_prefix(case)
    scorecard = prefix["costed_scorecard"]
    complete = case["n16_complete_uncertainty_result"]
    snapshot = _snapshot_from_payload(complete["acquisition_evidence"]["snapshot"])
    dummy = {
        "reason": "fixed_policy_comparator",
        "candidate_reliability_stratum": (
            "one_admissible_candidate"
            if len(snapshot.admissible_candidate_models) == 1
            else "two_admissible_candidates"
            if len(snapshot.admissible_candidate_models) == 2
            else "no_admissible_candidate"
        ),
        "action_evaluations": [],
    }
    resources = {item["policy_name"]: item for item in scorecard["action_resources"]}
    costs = {item["policy_name"]: item for item in scorecard["action_costs"]}
    return _selected_outcome_record(
        case,
        block=block,
        policy_name=policy_name,
        selection_payload=dummy,
        offsets=offsets,
        resources=resources,
        costs=costs,
    )


def _mean(values: Sequence[float]) -> Optional[float]:
    return None if not values else fmean(values)


def _summary(records: Sequence[Mapping[str, object]]) -> dict:
    values = tuple(records)
    losses = [float(item["loss"]) for item in values]
    decisions = [item["decision"] for item in values]
    raw_coverage = [
        item["raw_interval_covered"]
        for item in values
        if item["raw_interval_covered"] is not None
    ]
    adjusted_coverage = [
        item["development_adjusted_interval_covered"]
        for item in values
        if item["development_adjusted_interval_covered"] is not None
    ]
    energy = [
        float(item["realized_resources"]["total_diagnostic_energy"])
        for item in values
        if item["realized_resources"]["total_diagnostic_energy"] is not None
    ]
    elapsed = [
        float(item["realized_resources"]["energized_schedule_time_seconds"])
        for item in values
        if item["realized_resources"]["energized_schedule_time_seconds"] is not None
    ]
    runs = [
        float(item["realized_resources"]["diagnostic_run_count"])
        for item in values
        if item["realized_resources"]["diagnostic_run_count"] is not None
    ]
    instruments = [
        float(item["realized_resources"]["extra_sensor_count"])
        for item in values
        if item["realized_resources"]["extra_sensor_count"] is not None
    ]
    return {
        "case_count": len(values),
        "mean_loss": _mean(losses),
        "false_approvals": sum(item["false_approval"] for item in values),
        "false_rejections": sum(item["false_rejection"] for item in values),
        "approvals": sum(value == APPROVE for value in decisions),
        "rejections": sum(value == REJECT for value in decisions),
        "abstentions": sum(value == INSUFFICIENT_EVIDENCE for value in decisions),
        "definitive_decisions": sum(value != INSUFFICIENT_EVIDENCE for value in decisions),
        "decision_coverage": (
            None
            if not values
            else sum(value != INSUFFICIENT_EVIDENCE for value in decisions)
            / len(values)
        ),
        "raw_interval_covered": sum(value is True for value in raw_coverage),
        "raw_interval_evaluable": len(raw_coverage),
        "adjusted_interval_covered": sum(
            value is True for value in adjusted_coverage
        ),
        "adjusted_interval_evaluable": len(adjusted_coverage),
        "selection_failures": sum(item["selected_policy"] is None for item in values),
        "verification_failures": sum(item["verification_failure"] for item in values),
        "pipeline_failures": sum(item["pipeline_failure"] for item in values),
        "cases_with_ineligible_actions": sum(bool(item["ineligible_actions"]) for item in values),
        "ineligible_action_count": sum(len(item["ineligible_actions"]) for item in values),
        "mean_realized_diagnostic_energy_joules": _mean(energy),
        "mean_realized_schedule_seconds": _mean(elapsed),
        "mean_realized_diagnostic_run_count": _mean(runs),
        "mean_realized_extra_sensor_count": _mean(instruments),
        "total_added_run_count": sum(
            item["resource_proxy"]["added_run_count"] for item in values
        ),
        "total_added_sensor_count": sum(
            item["resource_proxy"]["added_sensor_count"] for item in values
        ),
        "mean_normalized_incremental_energy": _mean(
            [
                float(item["resource_proxy"]["normalized_incremental_energy"])
                for item in values
            ]
        ),
    }


def _case_loss_records(
    case_records: Sequence[Mapping[str, object]],
    objective: Mapping[str, object],
) -> list:
    by_identity = {
        (item["block"], item["truth_family"]): item
        for item in objective["case_results"]
    }
    result = []
    for case in case_records:
        loss = by_identity[(case["block"], case["truth_family"])]
        result.append(
            {
                **case,
                "decision": loss["decision"],
                "true_pass": loss["true_pass"],
                "false_approval": bool(loss["components"]["false_approval"]),
                "false_rejection": bool(loss["components"]["false_rejection"]),
                "abstention": bool(loss["components"]["abstention"]),
                "loss_components": loss["components"],
                "loss": loss["loss"],
            }
        )
    return result


def _group_summaries(records: Sequence[Mapping[str, object]]) -> dict:
    dimensions = {
        "truth_family": list(STAGE3_TRUTH_CONDITIONS),
        "selected_action": [*POLICY_NAMES, "selection_failure"],
        "candidate_reliability_stratum": [
            "no_admissible_candidate",
            "one_admissible_candidate",
            "two_admissible_candidates",
        ],
    }
    grouped = {}
    for dimension, labels in dimensions.items():
        grouped[dimension] = {}
        for label in labels:
            selected = [
                item
                for item in records
                if (
                    (item["selected_policy"] or "selection_failure") == label
                    if dimension == "selected_action"
                    else item[dimension] == label
                )
            ]
            grouped[dimension][label] = _summary(selected)
    return {"overall": _summary(records), "by": grouped}


def _objective_for_records(
    grid_values: Sequence[float],
    records: Sequence[Mapping[str, object]],
) -> dict:
    return evaluate_phase_d_rule_objective(
        grid_values,
        [item["objective_input"] for item in records],
    )


def analyze_phase_d3_evidence(
    phase_d1: Mapping[str, object],
    offsets: ProspectiveDevelopmentOffsets,
) -> dict:
    """Evaluate every frozen grid row from already-validated D1 evidence."""

    if phase_d1["partition"] != PROSPECTIVE_DEVELOPMENT_TUNING_PARTITION:
        raise ValueError("Phase D3 requires p1 development-tuning evidence")
    blocks = phase_d1["block_results"]
    if (
        len(blocks) != DEVELOPMENT_TUNING_BLOCK_COUNT
        or [item["block"] for item in blocks]
        != list(range(DEVELOPMENT_TUNING_BLOCK_COUNT))
    ):
        raise ValueError("Phase D3 needs all twenty ordered tuning blocks")
    cases = []
    for block_index, block in enumerate(blocks):
        if [item["truth_condition"] for item in block["cases"]] != list(
            STAGE3_TRUTH_CONDITIONS
        ):
            raise ValueError("Phase D3 needs all three ordered families per block")
        cases.extend((block_index, item) for item in block["cases"])

    rows = []
    for grid_values in phase_d_rule_grid():
        rule = _rule_for_grid(grid_values, offsets)
        raw_records = [
            _score_selector_case(
                case,
                block=block,
                rule=rule,
                offsets=offsets,
            )
            for block, case in cases
        ]
        objective = _objective_for_records(grid_values, raw_records)
        records = _case_loss_records(raw_records, objective)
        overall = _summary(records)
        rows.append(
            {
                "grid_values": list(grid_values),
                "selector_rule": prospective_selector_rule_payload(rule),
                "objective": {
                    key: objective[key]
                    for key in (
                        "case_count",
                        "block_count",
                        "block_losses",
                        "mean_block_loss",
                        "false_approvals",
                        "false_rejections",
                        "definitive_decisions",
                    )
                },
                "metrics": _group_summaries(records),
                "case_results": records,
                "tie_break_values": {
                    "mean_block_loss": objective["mean_block_loss"],
                    "false_approvals": objective["false_approvals"],
                    "false_rejections": objective["false_rejections"],
                    "definitive_decisions": objective["definitive_decisions"],
                    "mean_realized_diagnostic_energy_joules": overall[
                        "mean_realized_diagnostic_energy_joules"
                    ],
                    "lexicographic_grid_values": list(grid_values),
                },
            }
        )
    if len(rows) != PHASE_D_RULE_GRID_SIZE:
        raise RuntimeError("Phase D3 did not produce all 81 frozen grid rows")

    def ranking_key(row):
        tie = row["tie_break_values"]
        energy = tie["mean_realized_diagnostic_energy_joules"]
        if energy is None:
            energy = math.inf
        return (
            tie["mean_block_loss"],
            tie["false_approvals"],
            tie["false_rejections"],
            -tie["definitive_decisions"],
            energy,
            tuple(tie["lexicographic_grid_values"]),
        )

    ranked = sorted(rows, key=ranking_key)
    ranking = [
        {
            "rank": index,
            "grid_values": row["grid_values"],
            "tie_break_values": row["tie_break_values"],
        }
        for index, row in enumerate(ranked, 1)
    ]
    winner = ranked[0]
    winner_nonlexicographic_key = ranking_key(winner)[:-1]
    winner_ties = [
        row
        for row in ranked
        if ranking_key(row)[:-1] == winner_nonlexicographic_key
    ]

    fixed = {}
    for policy_name in POLICY_NAMES:
        raw_records = [
            _fixed_policy_case(
                case,
                block=block,
                policy_name=policy_name,
                offsets=offsets,
            )
            for block, case in cases
        ]
        objective = _objective_for_records(phase_d_rule_grid()[0], raw_records)
        records = _case_loss_records(raw_records, objective)
        fixed[policy_name] = {
            "mean_block_loss": objective["mean_block_loss"],
            "block_losses": objective["block_losses"],
            "metrics": _group_summaries(records),
            "case_results": records,
        }

    return {
        "grid_row_count": len(rows),
        "grid_rows": rows,
        "ranking": ranking,
        "winning_grid_values": winner["grid_values"],
        "winning_pre_lexicographic_tie_count": len(winner_ties),
        "winning_pre_lexicographic_tied_grid_values": [
            row["grid_values"] for row in winner_ties
        ],
        "winning_selector_rule": winner["selector_rule"],
        "winning_objective": winner["objective"],
        "winning_metrics": winner["metrics"],
        "fixed_policy_comparators": fixed,
        "truth_use": "offline_development_loss_after_all_policy_records_were_saved",
        "refitting_performed": False,
        "new_partition_opened": False,
    }


def build_phase_d3_grid_analysis(
    *,
    phase_d1_archive_path: Path | str,
    phase_d2_result_path: Path | str,
    repository_root: Path | str,
    analysis_source_revision: str,
) -> dict:
    _validate_revision(analysis_source_revision)
    validate_executing_phase_d_runtime()
    root = Path(repository_root).expanduser().resolve(strict=True)
    d1_path = Path(phase_d1_archive_path).expanduser().resolve(strict=True)
    d2_path = Path(phase_d2_result_path).expanduser().resolve(strict=True)
    d1_raw = d1_path.read_bytes()
    d2 = validate_phase_d2_offset_analysis(
        json.loads(d2_path.read_bytes()),
        phase_d1_archive_path=d1_path,
        repository_root=root,
    )
    if d2["offset_estimation"]["status"] != "finite_offsets_ready":
        raise ValueError("Phase D3 is forbidden without four finite D2 offsets")
    if hashlib.sha256(d1_raw).hexdigest() != d2["input_phase_d1"]["json_sha256"]:
        raise ValueError("Phase D3 D1 input does not match the validated D2 input")
    phase_d1 = json.loads(d1_raw)
    offsets = prospective_development_offsets_from_payload(
        d2["offset_estimation"]["development_offsets"]
    )
    manifest = create_source_manifest(root, PHASE_D3_ANALYSIS_SOURCE_PATHS)
    analysis = analyze_phase_d3_evidence(phase_d1, offsets)
    payload = {
        "schema_version": PHASE_D3_SCHEMA_VERSION,
        "protocol_version": PHASE_D3_PROTOCOL_VERSION,
        "analysis_source_revision": analysis_source_revision,
        "analysis_source_manifest": source_manifest_payload(manifest),
        "runtime_identity": phase_d_runtime_identity(),
        "input_phase_d1": {
            "path_name": d1_path.name,
            "json_sha256": hashlib.sha256(d1_raw).hexdigest(),
            "scientific_result_digest": phase_d1["scientific_result_digest"],
            "source_revision": phase_d1["source_revision"],
            "protocol_digest": phase_d1["protocol_digest"],
        },
        "input_phase_d2": {
            "path_name": d2_path.name,
            "json_sha256": hashlib.sha256(d2_path.read_bytes()).hexdigest(),
            "scientific_result_digest": d2["scientific_result_digest"],
            "analysis_source_revision": d2["analysis_source_revision"],
        },
        "development_offsets": d2["offset_estimation"]["development_offsets"],
        "method": {
            "grid_row_count": PHASE_D_RULE_GRID_SIZE,
            "primary_draw_count": 16,
            "max_unstable_draws_per_source_action": 1,
            "cost_scenario": "balanced_face_equal",
            "paired_block_objective": "mean_family_loss_then_mean_twenty_blocks",
            "tie_breakers": [
                "fewer_false_approvals",
                "fewer_false_rejections",
                "more_definitive_decisions",
                "lower_mean_realized_diagnostic_energy",
                "lexicographically_smaller_grid_values",
            ],
            "refitting_performed": False,
            "new_partition_opened": False,
        },
        **analysis,
    }
    payload["scientific_result_digest"] = _digest(
        f"{PHASE_D_HASH_DOMAIN}.phase_d3_grid_result_v1",
        payload,
    )
    return payload


def validate_phase_d3_grid_analysis(
    payload: Mapping[str, object],
    *,
    phase_d1_archive_path: Path | str,
    phase_d2_result_path: Path | str,
    repository_root: Path | str,
) -> dict:
    expected_keys = {
        "schema_version",
        "protocol_version",
        "analysis_source_revision",
        "analysis_source_manifest",
        "runtime_identity",
        "input_phase_d1",
        "input_phase_d2",
        "development_offsets",
        "method",
        "grid_row_count",
        "grid_rows",
        "ranking",
        "winning_grid_values",
        "winning_pre_lexicographic_tie_count",
        "winning_pre_lexicographic_tied_grid_values",
        "winning_selector_rule",
        "winning_objective",
        "winning_metrics",
        "fixed_policy_comparators",
        "truth_use",
        "refitting_performed",
        "new_partition_opened",
        "scientific_result_digest",
    }
    item = _exact_mapping(payload, expected_keys, "Phase D3 grid analysis")
    if (
        item["schema_version"] != PHASE_D3_SCHEMA_VERSION
        or item["protocol_version"] != PHASE_D3_PROTOCOL_VERSION
        or item["runtime_identity"] != phase_d_runtime_identity()
    ):
        raise ValueError("Phase D3 result header is invalid")
    _validate_revision(item["analysis_source_revision"])
    manifest = source_manifest_from_payload(item["analysis_source_manifest"])
    verify_source_manifest(
        manifest,
        repository_root,
        PHASE_D3_ANALYSIS_SOURCE_PATHS,
    ).assert_valid()
    rebuilt = build_phase_d3_grid_analysis(
        phase_d1_archive_path=phase_d1_archive_path,
        phase_d2_result_path=phase_d2_result_path,
        repository_root=repository_root,
        analysis_source_revision=item["analysis_source_revision"],
    )
    if dict(item) != rebuilt:
        raise ValueError("Phase D3 result does not match strict replay")
    _validate_sha256("Phase D3 scientific digest", item["scientific_result_digest"])
    return dict(item)


def format_phase_d3_grid_report(payload: Mapping[str, object]) -> str:
    values = payload["winning_grid_values"]
    objective = payload["winning_objective"]
    overall = payload["winning_metrics"]["overall"]
    return "\n".join(
        (
            "Prospective operating-decision Phase D3 grid analysis",
            "=======================================================",
            "",
            f"Analysis source revision: {payload['analysis_source_revision']}",
            f"Input D1 SHA-256: {payload['input_phase_d1']['json_sha256']}",
            f"Input D2 SHA-256: {payload['input_phase_d2']['json_sha256']}",
            f"D3 scientific digest: {payload['scientific_result_digest']}",
            "Refitting performed: no",
            "New partition opened: no",
            "",
            f"Grid rows: {payload['grid_row_count']}",
            "Winning grid values (general, single, reduction, utility): "
            + str(values),
            "Rows tied before the final lexicographic tie breaker: "
            + str(payload["winning_pre_lexicographic_tie_count"]),
            f"Winning mean paired-block loss: {objective['mean_block_loss']}",
            f"False approvals: {objective['false_approvals']}",
            f"False rejections: {objective['false_rejections']}",
            f"Definitive decisions: {objective['definitive_decisions']}/60",
            f"Abstentions: {overall['abstentions']}/60",
            "",
            "PASS: all 81 frozen rules were scored and ranked.",
            "Next permitted step: commit this provisional rule, then execute D4.",
        )
    )


def save_phase_d3_grid_artifacts(
    payload: Mapping[str, object],
    *,
    output_directory: Path | str,
) -> SavedPhaseD3Artifacts:
    root = Path(output_directory).expanduser().resolve()
    json_path = root / PHASE_D3_JSON_NAME
    report_path = root / PHASE_D3_REPORT_NAME
    hash_path = root / PHASE_D3_HASH_NAME
    existing = [path for path in (json_path, report_path, hash_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "Phase D3 refuses to overwrite artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    archive = _canonical_bytes(dict(payload)) + b"\n"
    report = (format_phase_d3_grid_report(payload) + "\n").encode("utf-8")
    json_sha = hashlib.sha256(archive).hexdigest()
    report_sha = hashlib.sha256(report).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    json_path.write_bytes(archive)
    report_path.write_bytes(report)
    hash_path.write_text(
        f"{json_sha}  {json_path.name}\n{report_sha}  {report_path.name}\n",
        encoding="utf-8",
    )
    return SavedPhaseD3Artifacts(
        json_path=json_path,
        report_path=report_path,
        hash_path=hash_path,
        json_sha256=json_sha,
        report_sha256=report_sha,
        archive_size_bytes=len(archive),
    )


__all__ = [
    "PHASE_D3_ANALYSIS_SOURCE_PATHS",
    "PHASE_D3_HASH_NAME",
    "PHASE_D3_JSON_NAME",
    "PHASE_D3_REPORT_NAME",
    "SavedPhaseD3Artifacts",
    "analyze_phase_d3_evidence",
    "build_phase_d3_grid_analysis",
    "format_phase_d3_grid_report",
    "save_phase_d3_grid_artifacts",
    "validate_phase_d3_grid_analysis",
]
