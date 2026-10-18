"""Frozen D6 orchestration with one additional durable before-reveal boundary.

The acquisition, selection, fitting, verification and scoring statements are a
static copy of the sealed D6 family worker, qualified through its module. The
only behavioral insertion saves the fixed calibrated decisions after raw
verification and before the first target reveal. A structural regression test
checks that removing this insertion recovers the original worker exactly.
"""

from __future__ import annotations

from . import operating_decision_prospective_phase_d6_internal_check as d6
from .operating_decision_prospective_phase_f_decisions import (
    blinded_inputs_from_saved, calibrated_saves,
)


def run_calibrated_family(
    *,
    truth_condition: str,
    block: int,
    truth,
    partition: d6.CorrectedPartition,
    physical_config: d6.OperatingDecisionRealismConfig,
    registry: d6.RandomStreamRegistry,
    selector_rule: d6.ProspectiveSelectorRule,
    frozen_rule: d6.ProspectiveSelectorRule,
    cost_scenario: d6.ProspectiveCostScenario,
    before_reveal_sink,
) -> tuple:
    case_wall_started = d6.perf_counter()
    case_cpu_started = d6.process_time()
    case_rss_started = d6._peak_rss_bytes()

    policies = d6.default_fixed_policies()
    cases = {}
    generation_timings = {}
    for policy in policies:
        case, timing = d6._timed(
            lambda policy=policy: d6.build_corrected_blinded_case(
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

    common_case = cases[d6.STOP_NOW]
    prefix_records: d6.Dict[int, dict] = {}
    sensitivity_record = None
    pipeline_failures = []
    evidence_timing = None
    n16_timing = None
    prefix_timings = {}
    n16_complete_payload = None
    admissible_source_models = None

    try:
        evidence, evidence_timing = d6._timed(
            lambda: d6.prepare_prospective_acquisition_evidence(
                common_case.acquisition_runs[0],
                common_case.final_regime,
                physical_config,
            )
        )
        admissible_source_models = tuple(
            evidence.snapshot.admissible_candidate_models
        )
        namespace = d6.ProspectiveRandomStreamNamespace(
            campaign=partition.campaign,
            partition=partition.name,
            block=block,
            acquisition_evidence_digest=evidence.evidence_digest,
        )
        n16, n16_timing = d6._timed(
            lambda: d6.estimate_prospective_action_uncertainty(
                evidence,
                namespace,
                physical_config,
                d6.ProspectiveUncertaintyConfig(
                    draw_count=d6.PILOT_GENERATED_DRAW_COUNT,
                    max_unstable_draws_per_source_action=(
                        d6.pilot_max_unstable_draws(d6.PILOT_GENERATED_DRAW_COUNT)
                    ),
                ),
            )
        )
        n16_complete_payload = d6.prospective_uncertainty_result_payload(n16)
        for draw_count in d6.PILOT_DRAW_COUNTS:
            try:
                def derive_and_score(draw_count=draw_count):
                    prefix = d6.prefix_prospective_uncertainty_result(
                        n16,
                        draw_count=draw_count,
                        max_unstable_draws_per_source_action=(
                            d6.pilot_max_unstable_draws(draw_count)
                        ),
                    )
                    return d6._prefix_record(
                        prefix,
                        selector_rule=selector_rule,
                        cost_scenario=cost_scenario,
                    )

                prefix_records[draw_count], prefix_timings[str(draw_count)] = d6._timed(
                    derive_and_score
                )
            except Exception as error:  # retained as a failed pilot case
                prefix_records[draw_count] = d6._failed_prefix_record(
                    draw_count,
                    d6.pilot_max_unstable_draws(draw_count),
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
                relaxed = d6.prefix_prospective_uncertainty_result(
                    n16,
                    draw_count=d6.PILOT_GENERATED_DRAW_COUNT,
                    max_unstable_draws_per_source_action=(
                        d6.PILOT_SENSITIVITY_MAX_UNSTABLE
                    ),
                )
                return d6._prefix_record(
                    relaxed,
                    selector_rule=selector_rule,
                    cost_scenario=cost_scenario,
                )

            sensitivity_record, prefix_timings["16_max_unstable_0"] = d6._timed(
                derive_sensitivity
            )
        except Exception as error:  # retained as a failed sensitivity result
            sensitivity_record = d6._failed_prefix_record(
                d6.PILOT_GENERATED_DRAW_COUNT,
                d6.PILOT_SENSITIVITY_MAX_UNSTABLE,
                error,
                "sensitivity_score",
            )
            pipeline_failures.append(
                {
                    "stage": "sensitivity_score_n16_max_unstable_0",
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
        for draw_count in d6.PILOT_DRAW_COUNTS:
            prefix_records[draw_count] = d6._failed_prefix_record(
                draw_count,
                d6.pilot_max_unstable_draws(draw_count),
                error,
                failure["stage"],
            )
        sensitivity_record = d6._failed_prefix_record(
            d6.PILOT_GENERATED_DRAW_COUNT,
            d6.PILOT_SENSITIVITY_MAX_UNSTABLE,
            error,
            failure["stage"],
        )

    frozen_selection_record = None
    if n16_complete_payload is not None:
        primary = prefix_records[d6.PILOT_GENERATED_DRAW_COUNT]
        if primary.get("costed_scorecard") is not None:
            snapshot = d6._snapshot_from_payload(
                n16_complete_payload["acquisition_evidence"]["snapshot"]
            )
            evaluations = d6._padded_action_evaluations(
                n16_complete_payload,
                primary["costed_scorecard"],
                frozen_rule.development_offsets,
            )
            frozen_selection_record = d6._frozen_selection_payload(
                d6.select_prospective_action(snapshot, evaluations, frozen_rule)
            )

    # The frozen selection and all four fixed-policy decisions are saved before
    # any target response is revealed.
    saved = {}
    decision_timings = {}
    for policy in policies:
        try:
            decision, timing = d6._timed(
                lambda policy=policy: d6.decide_corrected_blinded_case(
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
                "peak_rss_bytes": d6._peak_rss_bytes(),
                "failed": True,
            }
            pipeline_failures.append(
                {
                    "stage": f"fixed_policy_save:{policy.name}",
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )

    # Commit every calibrated decision before the first target-scoring call.
    selected_before_reveal = None
    if frozen_selection_record is not None and frozen_selection_record['selection_succeeded']:
        selected_before_reveal = frozen_selection_record['selected_policy']
    blinded_inputs, saved_actions = blinded_inputs_from_saved(
        {name: d6._saved_payload(value) for name, value in saved.items()},
        selected_before_reveal,
    )
    calibrated_before_reveal = before_reveal_sink(calibrated_saves(blinded_inputs, saved_actions))

    fixed_results = {}
    reveal_timings = {}
    for policy in policies:
        if policy.name not in saved:
            reveal_timings[policy.name] = {
                "wall_seconds": 0.0,
                "cpu_seconds": 0.0,
                "peak_rss_bytes": d6._peak_rss_bytes(),
                "skipped": "decision_not_saved",
            }
            fixed_results[policy.name] = {
                "case": d6._case_payload(cases[policy.name]),
                "saved_before_reveal": None,
                "post_reveal_score": None,
                "failure": "decision_not_saved",
            }
            continue
        try:
            record, timing = d6._timed(
                lambda policy=policy: d6.score_corrected_saved_decision(
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
                "case": d6._case_payload(cases[policy.name]),
                "saved_before_reveal": d6._saved_payload(saved[policy.name]),
                "post_reveal_score": d6._scored_record_payload(record),
            }
        except Exception as error:
            reveal_timings[policy.name] = {
                "wall_seconds": 0.0,
                "cpu_seconds": 0.0,
                "peak_rss_bytes": d6._peak_rss_bytes(),
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
                "case": d6._case_payload(cases[policy.name]),
                "saved_before_reveal": d6._saved_payload(saved[policy.name]),
                "post_reveal_score": None,
                "failure": {
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            }

    selected_policy = None
    if frozen_selection_record is not None and frozen_selection_record["selection_succeeded"]:
        selected_policy = frozen_selection_record["selected_policy"]
    selected_outcome = (
        fixed_results.get(selected_policy) if selected_policy is not None else None
    )

    return ({
        "block": block,
        "truth_condition": truth_condition,
        "device_token": common_case.case_id.device_token,
        "truth_revealed_only_after_saves": d6._strict_json_value(truth),
        "selection_information_boundary": d6._strict_json_value(
            d6._SELECTION_INFORMATION_BOUNDARY
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
        "primary_prefixes": [prefix_records[item] for item in d6.PILOT_DRAW_COUNTS],
        "n16_max_unstable_0_sensitivity": sensitivity_record,
        "selected_policy_at_n16_primary": selected_policy,
        "frozen_selection_before_reveal": frozen_selection_record,
        "selection_saved_before_target_reveal": True,
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
            "case_wall_seconds": d6.perf_counter() - case_wall_started,
            "case_cpu_seconds": d6.process_time() - case_cpu_started,
            "peak_rss_before_bytes": case_rss_started,
            "peak_rss_after_bytes": d6._peak_rss_bytes(),
        },
    }, calibrated_before_reveal)
