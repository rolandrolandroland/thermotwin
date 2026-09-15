"""Run the disposable candidate-exclusion/KKT recovery probe.

The probe deliberately uses a non-scientific campaign and partition.  It runs
the public corrected-partition entry point, which generates and audits every
truth family, but reports only the 12 paired Family A blocks requested by the
decision-rule audit.  The strict-rule counterfactual is computed from the fits
already saved in each current-pipeline decision; it never refits a case.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

from thermotwin.studies.operating_decision import (
    DECISIONS,
    FIXED_VOLTAGE,
    INSUFFICIENT_EVIDENCE,
    POLICY_NAMES,
)
from thermotwin.studies.operating_decision_random_streams import (
    RANDOM_STREAM_PROTOCOL_VERSION,
)
from thermotwin.studies.operating_decision_realism import (
    FIT_BOUND_TOLERANCE,
    FIT_SCALED_GRADIENT_TOLERANCE,
    RealisticCandidateFit,
)
from thermotwin.studies.operating_decision_replication import (
    CORRECTED_GENERATOR_VERSION,
    CorrectedPartition,
    CorrectedTrialRecord,
    corrected_physical_protocol_digest,
    run_corrected_partition,
)
from thermotwin.studies.sensor_model_discrimination import (
    FIVE_STATE_MODEL,
    FOUR_STATE_MODEL,
    TRUTH_CONDITIONS,
)


SCHEMA_VERSION = 2
DISPOSABLE_CAMPAIGN = (
    "operating_decision_disposable_candidate_exclusion_kkt_2026_09_14"
)
DISPOSABLE_PARTITION_NAME = "disposable_family_a_candidate_exclusion_probe"
BLOCK_COUNT = 12
FAMILY_A = TRUTH_CONDITIONS[0]
PROJECTED_KKT_TERMINATION = "scaled_projected_gradient_tolerance"


def _decision_counts(records: Sequence[CorrectedTrialRecord]) -> dict[str, int]:
    return {
        decision: sum(record.scored.saved.decision == decision for record in records)
        for decision in DECISIONS
    }


def _fits(record: CorrectedTrialRecord) -> tuple[RealisticCandidateFit, ...]:
    """Return the exact candidate fits retained with one saved decision."""

    return tuple(item.fit for item in record.scored.saved.verifications)


def _strict_status_trigger(record: CorrectedTrialRecord) -> bool:
    """Reproduce the superseded whole-case reliability gate without refitting."""

    return any(not fit.converged or fit.reached_bound for fit in _fits(record))


def _strict_decision(record: CorrectedTrialRecord) -> str:
    if _strict_status_trigger(record):
        return INSUFFICIENT_EVIDENCE
    return record.scored.saved.decision


def _is_five_state_interface_lower_bound_hit(
    fit: RealisticCandidateFit,
    *,
    lower_bound: float,
) -> bool:
    if fit.model_name != FIVE_STATE_MODEL or fit.interface_mass is None:
        return False
    return abs(math.log(fit.interface_mass / lower_bound)) <= FIT_BOUND_TOLERANCE


def _fit_status(
    fit: RealisticCandidateFit,
    *,
    interface_mass_lower_bound: float,
) -> dict[str, object]:
    return {
        "converged": fit.converged,
        "five_state_interface_mass_lower_bound_hit": (
            _is_five_state_interface_lower_bound_hit(
                fit,
                lower_bound=interface_mass_lower_bound,
            )
        ),
        "model": fit.model_name,
        "projected_converged_bound_hit": (
            fit.reached_bound
            and fit.converged
            and fit.termination_reason == PROJECTED_KKT_TERMINATION
        ),
        "reached_bound": fit.reached_bound,
        "scaled_projected_gradient_infinity_norm": (
            fit.scaled_gradient_infinity_norm
        ),
        "termination_reason": fit.termination_reason,
    }


def _fit_diagnostics(
    records: Sequence[CorrectedTrialRecord],
    *,
    interface_mass_lower_bound: float,
) -> dict[str, object]:
    fits = tuple(fit for record in records for fit in _fits(record))
    by_model = {}
    for model_name in (FOUR_STATE_MODEL, FIVE_STATE_MODEL):
        selected = tuple(fit for fit in fits if fit.model_name == model_name)
        by_model[model_name] = {
            "bound_hits": sum(fit.reached_bound for fit in selected),
            "fits": len(selected),
            "nonconverged": sum(not fit.converged for fit in selected),
            "projected_converged_bound_hits": sum(
                fit.reached_bound
                and fit.converged
                and fit.termination_reason == PROJECTED_KKT_TERMINATION
                for fit in selected
            ),
        }
    return {
        "bound_hit_cases": sum(
            any(fit.reached_bound for fit in _fits(record)) for record in records
        ),
        "bound_hits": sum(fit.reached_bound for fit in fits),
        "by_model": by_model,
        "candidate_fits": len(fits),
        "five_state_interface_mass_lower_bound_hits": sum(
            _is_five_state_interface_lower_bound_hit(
                fit,
                lower_bound=interface_mass_lower_bound,
            )
            for fit in fits
        ),
        "interface_mass_lower_bound_j_per_k": interface_mass_lower_bound,
        "nonconverged_fits": sum(not fit.converged for fit in fits),
        "projected_converged_bound_hits": sum(
            fit.reached_bound
            and fit.converged
            and fit.termination_reason == PROJECTED_KKT_TERMINATION
            for fit in fits
        ),
        "projected_kkt_tolerance": FIT_SCALED_GRADIENT_TOLERANCE,
    }


def _errors(records: Sequence[CorrectedTrialRecord]) -> dict[str, int]:
    false_approvals = sum(record.scored.false_approval for record in records)
    false_rejections = sum(record.scored.false_rejection for record in records)
    failures = tuple(
        failure
        for record in records
        for failure in record.scored.saved.failures
    )
    return {
        "cases_with_numerical_failures": sum(
            bool(record.scored.saved.failures) for record in records
        ),
        "false_approvals": false_approvals,
        "false_rejections": false_rejections,
        "numerical_failures": len(failures),
        "wrong_decisions": false_approvals + false_rejections,
    }


def _counterfactual(records: Sequence[CorrectedTrialRecord]) -> dict[str, object]:
    decisions = tuple(_strict_decision(record) for record in records)
    lost = tuple(
        record
        for record, strict in zip(records, decisions)
        if record.scored.saved.decision != INSUFFICIENT_EVIDENCE
        and strict == INSUFFICIENT_EVIDENCE
    )
    wrong_lost = sum(
        record.scored.false_approval or record.scored.false_rejection
        for record in lost
    )
    return {
        "additional_abstentions": len(lost),
        "correct_decisions_lost": len(lost) - wrong_lost,
        "decisive": sum(item != INSUFFICIENT_EVIDENCE for item in decisions),
        "decision_counts": {
            decision: sum(item == decision for item in decisions)
            for decision in DECISIONS
        },
        "fit_status_triggered_cases": sum(
            _strict_status_trigger(record) for record in records
        ),
        "refits_performed": 0,
        "rule": (
            "abstain if either saved candidate fit is nonconverged or reached a bound"
        ),
        "total_abstentions": sum(
            item == INSUFFICIENT_EVIDENCE for item in decisions
        ),
        "wrong_decisions_lost": wrong_lost,
        "wrong_decisions_remaining": sum(
            (record.scored.false_approval or record.scored.false_rejection)
            and strict != INSUFFICIENT_EVIDENCE
            for record, strict in zip(records, decisions)
        ),
    }


def _case_payload(
    record: CorrectedTrialRecord,
    *,
    interface_mass_lower_bound: float,
) -> dict[str, object]:
    saved = record.scored.saved
    return {
        "current_decision": saved.decision,
        "current_decisive": saved.decision != INSUFFICIENT_EVIDENCE,
        "false_approval": record.scored.false_approval,
        "false_rejection": record.scored.false_rejection,
        "fit_status": [
            _fit_status(
                fit,
                interface_mass_lower_bound=interface_mass_lower_bound,
            )
            for fit in _fits(record)
        ],
        "numerical_failures": [
            {
                "error_type": failure.error_type,
                "model": failure.model_name,
                "stage": failure.stage,
            }
            for failure in saved.failures
        ],
        "policy": saved.policy_name,
        "strict_counterfactual_decision": _strict_decision(record),
        "strict_fit_status_trigger": _strict_status_trigger(record),
    }


def _block_payload(
    block: int,
    records: Sequence[CorrectedTrialRecord],
    *,
    interface_mass_lower_bound: float,
) -> dict[str, object]:
    selected = tuple(record for record in records if record.block == block)
    if tuple(record.scored.saved.policy_name for record in selected) != POLICY_NAMES:
        raise ValueError(f"Family A block {block} does not contain each fixed policy")
    fixed_voltage = next(
        record
        for record in selected
        if record.scored.saved.policy_name == FIXED_VOLTAGE
    )
    return {
        "block": block,
        "cases": [
            _case_payload(
                record,
                interface_mass_lower_bound=interface_mass_lower_bound,
            )
            for record in selected
        ],
        "current_decision_counts": _decision_counts(selected),
        "current_decisive": sum(
            record.scored.saved.decision != INSUFFICIENT_EVIDENCE
            for record in selected
        ),
        "current_total": len(selected),
        "fixed_voltage": {
            "current_decision": fixed_voltage.scored.saved.decision,
            "current_decisive": (
                fixed_voltage.scored.saved.decision != INSUFFICIENT_EVIDENCE
            ),
            "strict_counterfactual_decision": _strict_decision(fixed_voltage),
        },
        "strict_counterfactual_abstentions": sum(
            _strict_decision(record) == INSUFFICIENT_EVIDENCE
            for record in selected
        ),
    }


def build_probe_payload(*, workers: int) -> dict[str, object]:
    partition = CorrectedPartition(
        DISPOSABLE_PARTITION_NAME,
        BLOCK_COUNT,
        DISPOSABLE_CAMPAIGN,
    )
    result = run_corrected_partition(
        partition,
        workers=workers,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    family_a = tuple(
        record for record in result.records if record.truth_condition == FAMILY_A
    )
    expected_cases = BLOCK_COUNT * len(POLICY_NAMES)
    if len(family_a) != expected_cases:
        raise ValueError(
            f"Family A probe produced {len(family_a)} cases; expected {expected_cases}"
        )
    family_a = tuple(
        sorted(
            family_a,
            key=lambda record: (
                record.block,
                POLICY_NAMES.index(record.scored.saved.policy_name),
            ),
        )
    )
    interface_mass_lower_bound = result.config.sensor.interface_mass_bounds[0]
    fixed_voltage = tuple(
        record
        for record in family_a
        if record.scored.saved.policy_name == FIXED_VOLTAGE
    )
    current_decisive = sum(
        record.scored.saved.decision != INSUFFICIENT_EVIDENCE
        for record in family_a
    )
    counterfactual = _counterfactual(family_a)
    counterfactual["fixed_voltage"] = _counterfactual(fixed_voltage)
    return {
        "blocks": [
            _block_payload(
                block,
                family_a,
                interface_mass_lower_bound=interface_mass_lower_bound,
            )
            for block in range(BLOCK_COUNT)
        ],
        "probe": {
            "block_count": BLOCK_COUNT,
            "campaign": partition.campaign,
            "corrected_generator_version": CORRECTED_GENERATOR_VERSION,
            "family_label": "Family A",
            "family_truth_condition": FAMILY_A,
            "partition": partition.name,
            "physical_protocol_digest": corrected_physical_protocol_digest(
                result.config
            ),
            "random_stream_protocol_version": RANDOM_STREAM_PROTOCOL_VERSION,
            "scientific_partition": False,
        },
        "random_stream_audit": {
            "declared_pairings": len(result.random_stream_audit.declared_pairings),
            "ok": result.random_stream_audit.ok,
            "unique_key_count": result.random_stream_audit.unique_key_count,
            "unique_seed_count": result.random_stream_audit.unique_seed_count,
            "unintended_reuses": len(result.random_stream_audit.unintended_reuses),
            "use_count": result.random_stream_audit.use_count,
        },
        "schema_version": SCHEMA_VERSION,
        "summary": {
            "counterfactual_strict_whole_case_abstention": counterfactual,
            "current": {
                "decision_counts": _decision_counts(family_a),
                "decisive": current_decisive,
                "decisive_fraction": f"{current_decisive}/{len(family_a)}",
                "fixed_voltage": {
                    "decision_counts": _decision_counts(fixed_voltage),
                    "decisive": sum(
                        record.scored.saved.decision != INSUFFICIENT_EVIDENCE
                        for record in fixed_voltage
                    ),
                    "total": len(fixed_voltage),
                },
                "total": len(family_a),
            },
            "errors": _errors(family_a),
            "fit_diagnostics": _fit_diagnostics(
                family_a,
                interface_mass_lower_bound=interface_mass_lower_bound,
            ),
        },
    }


def _write_json(payload: Mapping[str, object], output: Path | None) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(encoded)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a disposable 12-block corrected-pipeline probe and report "
            "Family A candidate-exclusion/KKT coverage."
        )
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="parallel worker processes (default: 4)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write stable JSON to this path instead of standard output",
    )
    arguments = parser.parse_args(tuple(argv) if argv is not None else None)
    if arguments.workers <= 0:
        parser.error("--workers must be a positive integer")
    _write_json(
        build_probe_payload(workers=arguments.workers),
        arguments.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
