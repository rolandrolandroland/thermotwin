"""Predeclared paired-block comparisons for the primary prospective experiment.

No outcome is used to choose an endpoint, threshold, size, or comparator.
Binomial bounds are computed within families (one case per independent block),
then combined conservatively. Energy uses a paired block Student t bound with
its explicitly approximate sampling interpretation.
"""

from __future__ import annotations

import math
from statistics import mean, stdev
from typing import Mapping, Sequence

from scipy.stats import beta, binom, t

from .operating_decision import APPROVE, INSUFFICIENT_EVIDENCE, POLICY_NAMES, REJECT
from .operating_decision_realism import STAGE3_TRUTH_CONDITIONS
from .operating_decision_prospective_phase_e_calibration import (
    PHASE_E_PROCEDURES, SELECTOR_PROCEDURE, _complete_matrix, _finite,
)


ANALYSIS_VERSION = "prospective_phase_e_paired_analysis_v1"
FAMILYWISE_ALPHA = 0.05
# 5 procedures * 3 families * (risk upper, risk lower, coverage lower),
# 4 comparisons * 3 families * 2 discordance bounds, 4 energy lower bounds.
PRIMARY_BOUND_COUNT = 73
BOUND_ALPHA = FAMILYWISE_ALPHA / PRIMARY_BOUND_COUNT
MAX_FALSE_APPROVAL_RISK = 0.10
MIN_DECISION_COVERAGE = 0.70
MAX_COVERAGE_LOSS = 0.05
MAX_RISK_INCREASE = 0.02


def binomial_bounds(successes: int, trials: int, *, alpha: float = BOUND_ALPHA) -> dict:
    if any(not isinstance(v, int) or isinstance(v, bool) for v in (successes, trials)):
        raise ValueError("binomial counts must be integers")
    if trials < 0 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial counts")
    if not _finite(alpha) or not 0 < alpha < 1:
        raise ValueError("invalid one-sided alpha")
    if trials == 0:
        return {"estimate": None, "lower": None, "upper": None,
                "numerator": successes, "denominator": trials}
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha, successes, trials-successes+1))
    upper = 1.0 if successes == trials else float(beta.ppf(1-alpha, successes+1, trials-successes))
    return {"estimate": successes / trials, "lower": lower, "upper": upper,
            "numerator": successes, "denominator": trials}


def zero_error_approval_requirement(risk: float, *, alpha: float = BOUND_ALPHA) -> int:
    if not _finite(risk) or not 0 < risk < 1 or not _finite(alpha) or not 0 < alpha < 1:
        raise ValueError("risk and alpha must be in (0,1)")
    return math.ceil(math.log(alpha) / math.log1p(-risk))


def _rate(numerator: int, denominator: int) -> dict:
    return {"numerator": numerator, "denominator": denominator,
            "estimate": numerator / denominator if denominator else None}


def _metrics(rows: Sequence[Mapping[str, object]]) -> dict:
    count = len(rows)
    decisions = [row["decision"] for row in rows]
    if any(value not in (APPROVE, REJECT, INSUFFICIENT_EVIDENCE) for value in decisions):
        raise ValueError("analysis requires saved final decisions")
    approvals = decisions.count(APPROVE)
    rejections = decisions.count(REJECT)
    false_approvals = sum(bool(r["false_approval"]) for r in rows)
    false_rejections = sum(bool(r["false_rejection"]) for r in rows)
    passes = sum(bool(r["true_pass"]) for r in rows)
    violations = count - passes
    covered = sum(bool(r["interval_covered"]) for r in rows)
    resource_names = (
        "total_diagnostic_energy", "diagnostic_run_count", "extra_sensor_count",
        "energized_schedule_time_seconds",
    )
    resources = {}
    for name in resource_names:
        values = [r.get("realized_resources", {}).get(name) for r in rows]
        complete = all(_finite(v) and v >= 0 for v in values)
        resources[name] = {"mean": mean(values) if values and complete else None,
                           "recorded_count": sum(_finite(v) and v >= 0 for v in values),
                           "required_count": count}
    return {
        "case_count": count, "approvals": approvals, "rejections": rejections,
        "definitive_decisions": approvals + rejections,
        "abstentions": count - approvals - rejections,
        "true_passes": passes, "true_violations": violations,
        "false_approvals": false_approvals, "false_rejections": false_rejections,
        "false_approval_risk": _rate(false_approvals, approvals),
        "false_rejection_rate_given_pass": _rate(false_rejections, passes),
        "missed_violation_rate": _rate(sum(bool(r["missed_violation"]) for r in rows), violations),
        "decision_error_rate": _rate(false_approvals + false_rejections, approvals + rejections),
        "decision_coverage": _rate(approvals + rejections, count),
        "case_interval_coverage": _rate(covered, count),
        "pipeline_failures": sum(bool(r["pipeline_failure"]) for r in rows),
        "verification_failures": sum(not r["verification_succeeded"] for r in rows),
        "missing_or_invalid_final_intervals": sum(r["final_interval"] is None for r in rows),
        "realized_resources": resources,
    }


def _procedure_summary(rows: Sequence[Mapping[str, object]], n: int) -> dict:
    groups = {}
    for family in STAGE3_TRUTH_CONDITIONS:
        family_rows = [r for r in rows if r["truth_family"] == family]
        metrics = _metrics(family_rows)
        coverage_bounds = binomial_bounds(metrics["definitive_decisions"], n)
        coverage_bounds.pop("upper")  # Only this lower direction belongs to the 73-bound family.
        groups[family] = {
            **metrics,
            "false_approval_risk_bounds": binomial_bounds(metrics["false_approvals"], metrics["approvals"]),
            "decision_coverage_bounds": coverage_bounds,
        }
    risks = [g["false_approval_risk_bounds"] for g in groups.values()]
    all_risks_defined = all(r["estimate"] is not None for r in risks)
    risk_upper = max(r["upper"] for r in risks) if all_risks_defined else None
    risk_lower = min(r["lower"] for r in risks) if all_risks_defined else None
    coverage_lower = mean(g["decision_coverage_bounds"]["lower"] for g in groups.values())
    block_covered = sum(
        all(r["interval_covered"] for r in rows if r["block"] == block)
        for block in range(n)
    )
    return {
        "overall": _metrics(rows), "by_truth_family": groups,
        "simultaneous_block_interval_coverage": _rate(block_covered, n),
        "population_mixture_risk_bounds": {
            "lower": risk_lower, "upper": risk_upper,
            "method": "family extrema; unknown population approval weights",
        },
        "population_equal_family_decision_coverage_lower": coverage_lower,
        "qualifies_for_primary_comparison": bool(
            risk_upper is not None and risk_upper <= MAX_FALSE_APPROVAL_RISK
            and coverage_lower >= MIN_DECISION_COVERAGE
        ),
    }


def _energy_contrast(differences: Sequence[float | None]) -> dict:
    n = len(differences)
    if n < 2 or any(not _finite(v) for v in differences):
        return {"paired_blocks": n, "complete": False,
                "mean_saving_joules": None, "lower_saving_joules": None}
    average = mean(differences)
    standard_error = stdev(differences) / math.sqrt(n)
    lower = average - float(t.ppf(1 - BOUND_ALPHA, n - 1)) * standard_error
    return {"paired_blocks": n, "complete": True,
            "mean_saving_joules": average, "lower_saving_joules": lower,
            "standard_error_joules": standard_error,
            "method": "one-sided paired-block Student t; approximate"}


def _frontier(procedures: Mapping[str, object], qualifying: Sequence[str]) -> list:
    def point(policy):
        metrics = procedures[policy]["overall"]
        return (metrics["realized_resources"]["total_diagnostic_energy"]["mean"],
                metrics["decision_coverage"]["estimate"],
                metrics["false_approval_risk"]["estimate"])
    frontier = []
    for policy in qualifying:
        energy, coverage, risk = point(policy)
        if any(v is None for v in (energy, coverage, risk)):
            frontier.append(policy)
            continue
        dominated = False
        for other in qualifying:
            if other == policy:
                continue
            oe, oc, orisk = point(other)
            if any(v is None for v in (oe, oc, orisk)):
                continue
            if oe <= energy and oc >= coverage and orisk <= risk and (oe < energy or oc > coverage or orisk < risk):
                dominated = True
        if not dominated:
            frontier.append(policy)
    return frontier


def analyze_primary_comparison(
    rows: Sequence[Mapping[str, object]], *, block_count: int,
) -> dict:
    keyed = _complete_matrix(rows, block_count)
    from .operating_decision_prospective_phase_e_calibration import calibrated_outcome
    for row in rows:
        reproduced = calibrated_outcome(
            row, row["correction_kelvin"],
            decision_clearance_kelvin=row["decision_clearance_kelvin"],
        )
        if any(row.get(name) != reproduced[name] for name in (
            "final_interval", "decision", "interval_covered", "true_pass",
            "false_approval", "false_rejection", "missed_violation",
        )):
            raise ValueError("analysis final outcome does not reproduce")
        # Recalculate truth-dependent error fields; reject mislabeled stored results.
        margin, decision = row["true_margin"], row.get("decision")
        expected = {
            "true_pass": margin >= 0, "false_approval": decision == APPROVE and margin < 0,
            "false_rejection": decision == REJECT and margin >= 0,
            "missed_violation": margin < 0 and decision != REJECT,
        }
        if any(row.get(k) is not v for k, v in expected.items()):
            raise ValueError("analysis error labels do not reproduce")
        interval = row.get("final_interval")
        covered = bool(interval is not None and interval["lower"] <= margin <= interval["upper"])
        if row.get("interval_covered") is not covered:
            raise ValueError("analysis interval coverage does not reproduce")
    summaries = {
        procedure: _procedure_summary([r for r in rows if r["procedure"] == procedure], block_count)
        for procedure in PHASE_E_PROCEDURES
    }
    selected = summaries[SELECTOR_PROCEDURE]
    comparisons = {}
    for policy in POLICY_NAMES:
        coverage_pairs = {}
        for family in STAGE3_TRUTH_CONDITIONS:
            losses = gains = 0
            for block in range(block_count):
                a = keyed[block, family, SELECTOR_PROCEDURE]["decision"] != INSUFFICIENT_EVIDENCE
                b = keyed[block, family, policy]["decision"] != INSUFFICIENT_EVIDENCE
                losses += b and not a
                gains += a and not b
            loss = binomial_bounds(losses, block_count)
            gain = binomial_bounds(gains, block_count)
            coverage_pairs[family] = {"fixed_only_decisions": losses, "selector_only_decisions": gains,
                                     "coverage_loss_estimate": (losses - gains) / block_count,
                                     "coverage_loss_upper": loss["upper"] - gain["lower"]}
        coverage_loss = mean(v["coverage_loss_upper"] for v in coverage_pairs.values())
        differences = []
        for block in range(block_count):
            values = []
            for family in STAGE3_TRUTH_CONDITIONS:
                selector_energy = keyed[block, family, SELECTOR_PROCEDURE].get("realized_resources", {}).get("total_diagnostic_energy")
                fixed_energy = keyed[block, family, policy].get("realized_resources", {}).get("total_diagnostic_energy")
                values.append(fixed_energy - selector_energy if _finite(fixed_energy) and _finite(selector_energy) else None)
            differences.append(mean(values) if all(v is not None for v in values) else None)
        energy = _energy_contrast(differences)
        selector_risk = selected["population_mixture_risk_bounds"]["upper"]
        fixed_risk = summaries[policy]["population_mixture_risk_bounds"]["lower"]
        risk_increase = selector_risk - fixed_risk if selector_risk is not None and fixed_risk is not None else None
        checks = {
            "energy_saving": energy["lower_saving_joules"] is not None and energy["lower_saving_joules"] > 0,
            "coverage_noninferiority": coverage_loss <= MAX_COVERAGE_LOSS,
            "risk_noninferiority": risk_increase is not None and risk_increase <= MAX_RISK_INCREASE,
        }
        comparisons[policy] = {"energy": energy, "coverage_pairs_by_family": coverage_pairs,
                               "coverage_loss_upper": coverage_loss, "risk_increase_upper": risk_increase,
                               "checks": checks, "all_comparison_checks_pass": all(checks.values())}
    qualifying = [p for p in POLICY_NAMES if summaries[p]["qualifies_for_primary_comparison"]]
    primary_success = bool(
        selected["qualifies_for_primary_comparison"] and qualifying
        and all(comparisons[p]["all_comparison_checks_pass"] for p in qualifying)
    )
    return {
        "protocol_version": ANALYSIS_VERSION, "independent_block_count": block_count,
        "cases_are_not_independent_replicates": True,
        "familywise_alpha": FAMILYWISE_ALPHA, "primary_bound_count": PRIMARY_BOUND_COUNT,
        "one_sided_alpha_per_bound": BOUND_ALPHA,
        "energy_inference_is_approximate": True,
        "procedures": summaries, "comparisons": comparisons,
        "qualifying_fixed_policies": qualifying,
        "qualifying_fixed_frontier": _frontier(summaries, qualifying),
        "primary_success": primary_success,
        "named_comparisons_passing_all_checks": [p for p in POLICY_NAMES if comparisons[p]["all_comparison_checks_pass"]],
        "conclusion_rule": "success requires selector qualification, at least one qualifying fixed policy, and all comparisons against every qualifying fixed policy",
    }


def precision_projection(development_by_procedure: Mapping[str, object], *, reserved_blocks: int = 100) -> dict:
    """Optimistic pre-q precision projection, using development evidence only."""
    requirement_10 = zero_error_approval_requirement(.10)
    requirement_02 = zero_error_approval_requirement(.02)
    critical = next((k for k in range(reserved_blocks + 1)
                     if binomial_bounds(k, reserved_blocks)["lower"] >= .7),
                    reserved_blocks + 1)
    rows = {}
    for procedure in PHASE_E_PROCEDURES:
        family_rows = {}
        for family in STAGE3_TRUTH_CONDITIONS:
            source = development_by_procedure[procedure][family]
            n = source["case_count"]
            approval_rate = source["approvals"] / n
            decision_rate = source["definitive_decisions"] / n
            approvals = int(math.floor(reserved_blocks * approval_rate))
            decisions = int(math.floor(reserved_blocks * decision_rate))
            lower = binomial_bounds(decisions, reserved_blocks)["lower"]
            family_rows[family] = {
                "development_case_count": n, "development_approvals": source["approvals"],
                "development_decisions": source["definitive_decisions"],
                "projected_approvals_before_independent_padding": reserved_blocks * approval_rate,
                "optimistic_zero_error_risk_upper": binomial_bounds(0, approvals)["upper"],
                "projected_decision_coverage_lower_before_padding": lower,
                "zero_error_approvals_needed_for_10pct_bound": requirement_10,
                "zero_error_approvals_needed_for_2pct_bound": requirement_02,
                "blocks_needed_for_10pct_zero_error_bound_at_development_rate": math.ceil(requirement_10 / approval_rate) if approval_rate else None,
                "blocks_needed_for_2pct_zero_error_bound_at_development_rate": math.ceil(requirement_02 / approval_rate) if approval_rate else None,
                "decision_target_power_at_development_rate": float(binom.sf(
                    critical - 1,
                    reserved_blocks, decision_rate,
                )),
            }
        rows[procedure] = family_rows
    return {"planned_reserved_blocks": reserved_blocks,
            "projection_uses_only_closed_development_evidence": True,
            "independent_padding_can_only_reduce_decisions_at_the_fixed_boundary": True,
            "risk_comparison_not_promised_to_be_powered": True,
            "sample_size_will_not_be_extended_after_outcomes": True,
            "procedures": rows}
