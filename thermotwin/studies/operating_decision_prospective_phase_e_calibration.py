"""Independent block calibration; no action selection or numerical fitting.

Infinity is an explicit JSON tag. Failed cases remain in their block maxima
and in the order-statistic denominator. The correction is procedure-specific,
not action-specific, and cannot be fed back to the frozen selector.
"""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import math
from typing import Mapping, Sequence

from scipy.stats import binom

from .operating_decision import (
    APPROVE, INSUFFICIENT_EVIDENCE, POLICY_NAMES, REJECT,
)
from .operating_decision_realism import STAGE3_TRUTH_CONDITIONS


SELECTOR_PROCEDURE = "prospective_selector"
PHASE_E_PROCEDURES = (SELECTOR_PROCEDURE, *POLICY_NAMES)
CALIBRATION_BLOCKS = 100
RESERVED_BLOCKS = 100
BLOCK_COVERAGE_TARGET = 0.90
CALIBRATION_CONFIDENCE = 0.95
POSITIVE_INFINITY = "+infinity"
CALIBRATION_VERSION = "prospective_phase_e_block_tolerance_v1"
SECONDARY_CLEARANCES_KELVIN = (0.0, 0.05, 0.10, 0.20)


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _nonnegative(value: object, name: str) -> float:
    if not _finite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def encode_score(score: float) -> float | str:
    if score == math.inf:
        return POSITIVE_INFINITY
    return _nonnegative(score, "calibration score")


def decode_score(score: object) -> float:
    if score == POSITIVE_INFINITY:
        return math.inf
    return _nonnegative(score, "encoded calibration score")


def tolerance_rank(
    n: int, coverage: float = BLOCK_COVERAGE_TARGET,
    confidence: float = CALIBRATION_CONFIDENCE,
) -> int:
    """Smallest k with Binomial(n, coverage).cdf(k - 1) >= confidence.

The n+1 sentinel requires infinite padding, rather than pretending that a
small calibration sample supports the declared conditional guarantee.
"""
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("calibration size must be a positive integer")
    if any(not _finite(v) or not 0 < v < 1 for v in (coverage, confidence)):
        raise ValueError("coverage and confidence must be between zero and one")
    for rank in range(1, n + 1):
        if float(binom.cdf(rank - 1, n, coverage)) >= confidence:
            return rank
    return n + 1


def valid_raw_interval(value: object) -> dict | None:
    if not isinstance(value, Mapping):
        return None
    lower, upper = value.get("lower"), value.get("upper")
    if not _finite(lower) or not _finite(upper) or lower > upper:
        return None
    return {"lower": float(lower), "upper": float(upper)}


def _fraction(value: float) -> Fraction:
    return Fraction.from_float(float(value))


def _round_score_up(value: Fraction) -> float:
    if value <= 0:
        return 0.0
    try:
        rounded = float(value)
    except OverflowError:
        return math.inf
    if _fraction(rounded) < value:
        rounded = math.nextafter(rounded, math.inf)
    return rounded


def _round_bound_outward(value: Fraction, *, lower: bool) -> float:
    try:
        rounded = float(value)
    except OverflowError as error:
        raise ValueError("finite calibration expansion overflowed") from error
    exact_rounded = _fraction(rounded)
    if lower and exact_rounded > value:
        rounded = math.nextafter(rounded, -math.inf)
    elif not lower and exact_rounded < value:
        rounded = math.nextafter(rounded, math.inf)
    if not math.isfinite(rounded):
        raise ValueError("finite calibration expansion overflowed")
    return rounded


def case_nonconformity(record: Mapping[str, object]) -> float:
    """Score the complete procedure, including verification and failure."""
    margin = record.get("true_margin")
    if not _finite(margin):
        # An unknown truth is a material incident, not an ordinary fit failure.
        raise ValueError("calibration needs an authenticated finite truth margin")
    offset = _nonnegative(record.get("development_offset_kelvin"), "offset")
    if not isinstance(record.get("verification_succeeded"), bool):
        raise ValueError("calibration verification status must be boolean")
    if not isinstance(record.get("pipeline_failure"), bool):
        raise ValueError("calibration pipeline failure status must be boolean")
    raw = valid_raw_interval(record.get("raw_interval"))
    if (
        raw is None or not record["verification_succeeded"]
        or record["pipeline_failure"]
    ):
        return math.inf
    # Exact arithmetic on the stored binary inputs prevents cancellation from
    # rounding a required correction down. This is numerical protection, not
    # another learned offset or a tolerance on a coverage/error decision.
    score = max(Fraction(0), _fraction(raw["lower"]) - _fraction(offset) - _fraction(margin),
                _fraction(margin) - _fraction(raw["upper"]) - _fraction(offset))
    return _round_score_up(score)


def apply_calibrated_decision(
    record: Mapping[str, object], correction: object,
    *, decision_clearance_kelvin: float = 0.0,
) -> dict:
    """Truth-free final decision; save this before revealing target outcomes."""
    if set(record) != {
        "raw_interval", "development_offset_kelvin", "verification_succeeded",
        "pipeline_failure",
    }:
        raise ValueError("final decision input must contain only declared blinded fields")
    if not isinstance(record.get("verification_succeeded"), bool):
        raise ValueError("verification status must be boolean")
    if not isinstance(record.get("pipeline_failure"), bool):
        raise ValueError("pipeline failure status must be boolean")
    q = decode_score(correction)
    clearance = _nonnegative(decision_clearance_kelvin, "decision clearance")
    raw = valid_raw_interval(record.get("raw_interval"))
    offset = _nonnegative(record["development_offset_kelvin"], "offset")
    final = None
    decision = INSUFFICIENT_EVIDENCE
    valid = (
        raw is not None and record["verification_succeeded"]
        and not record["pipeline_failure"] and q != math.inf
    )
    if valid:
        final = {
            "lower": _round_bound_outward(_fraction(raw["lower"]) - _fraction(offset) - _fraction(q), lower=True),
            "upper": _round_bound_outward(_fraction(raw["upper"]) + _fraction(offset) + _fraction(q), lower=False),
        }
        if final["lower"] >= clearance:
            decision = APPROVE
        elif final["upper"] < -clearance:
            decision = REJECT
    return {
        "correction_kelvin": encode_score(q),
        "decision_clearance_kelvin": clearance,
        "final_interval": final,
        "decision": decision,
    }


def calibrated_outcome(
    record: Mapping[str, object], correction: object,
    *, decision_clearance_kelvin: float = 0.0,
) -> dict:
    """Offline scoring of the truth-free saved decision, never action selection."""
    case_nonconformity(record)
    decision_input = {name: record[name] for name in (
        "raw_interval", "development_offset_kelvin", "verification_succeeded",
        "pipeline_failure",
    )}
    saved = apply_calibrated_decision(
        decision_input, correction, decision_clearance_kelvin=decision_clearance_kelvin,
    )
    margin = float(record["true_margin"])
    final, decision = saved["final_interval"], saved["decision"]
    return {
        **deepcopy(dict(record)), **saved,
        "interval_covered": bool(
            final is not None and final["lower"] <= margin <= final["upper"]
        ),
        "true_pass": margin >= 0.0,
        "false_approval": decision == APPROVE and margin < 0.0,
        "false_rejection": decision == REJECT and margin >= 0.0,
        "missed_violation": margin < 0.0 and decision != REJECT,
    }


def _complete_matrix(
    records: Sequence[Mapping[str, object]], n: int,
) -> dict:
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("block count must be a positive integer")
    expected = {
        (block, family, procedure)
        for block in range(n) for family in STAGE3_TRUTH_CONDITIONS
        for procedure in PHASE_E_PROCEDURES
    }
    keyed = {}
    for row in records:
        if not isinstance(row, Mapping):
            raise ValueError("calibration records must be mappings")
        block = row.get("block")
        if not isinstance(block, int) or isinstance(block, bool):
            raise ValueError("invalid calibration block index")
        if not isinstance(row.get("truth_family"), str) or not isinstance(row.get("procedure"), str):
            raise ValueError("calibration family and procedure identities must be strings")
        key = (block, row.get("truth_family"), row.get("procedure"))
        if key not in expected or key in keyed:
            raise ValueError("duplicate or unexpected calibration cell")
        case_nonconformity(row)
        keyed[key] = row
    if set(keyed) != expected:
        raise ValueError("calibration must retain every procedure/family/block cell")
    return keyed


def calibrate_complete_procedures(
    records: Sequence[Mapping[str, object]], *, block_count: int = CALIBRATION_BLOCKS,
) -> dict:
    keyed = _complete_matrix(records, block_count)
    rank = tolerance_rank(block_count)
    procedures = {}
    for procedure in PHASE_E_PROCEDURES:
        blocks = []
        for block in range(block_count):
            case_scores = {
                family: encode_score(case_nonconformity(keyed[block, family, procedure]))
                for family in STAGE3_TRUTH_CONDITIONS
            }
            block_score = max(decode_score(value) for value in case_scores.values())
            blocks.append({"block": block, "case_scores_kelvin": case_scores,
                           "block_score_kelvin": encode_score(block_score)})
        ordered = sorted(decode_score(b["block_score_kelvin"]) for b in blocks)
        correction = math.inf if rank > block_count else ordered[rank - 1]
        procedures[procedure] = {
            "block_scores": blocks,
            "sorted_block_scores_kelvin": [encode_score(s) for s in ordered],
            "infinite_block_count": sum(s == math.inf for s in ordered),
            "correction_kelvin": encode_score(correction),
            "finite_correction": math.isfinite(correction),
        }
    finite = all(value["finite_correction"] for value in procedures.values())
    return {
        "schema_version": 1,
        "protocol_version": CALIBRATION_VERSION,
        "block_count": block_count,
        "case_count_per_procedure": 3 * block_count,
        "block_coverage_target": BLOCK_COVERAGE_TARGET,
        "calibration_confidence_target": CALIBRATION_CONFIDENCE,
        "order_statistic_rank": rank,
        "achieved_confidence_per_procedure": float(binom.cdf(rank - 1, block_count, .9)),
        "confidence_is_joint_over_procedures": False,
        "score_denominators_include_all_failures": True,
        "procedures": procedures,
        "all_required_corrections_finite": finite,
        "scientific_status": (
            "finite_calibration_requires_phase_f_freeze" if finite
            else "calibration_infeasible_no_reserved_generation"
        ),
        "reserved_generation_authorized": False,
    }


def validate_calibration_result(
    payload: Mapping[str, object], records: Sequence[Mapping[str, object]],
    *, block_count: int = CALIBRATION_BLOCKS,
) -> dict:
    expected = calibrate_complete_procedures(records, block_count=block_count)
    if payload != expected:
        raise ValueError("calibration result does not reproduce from retained scores")
    return deepcopy(expected)
