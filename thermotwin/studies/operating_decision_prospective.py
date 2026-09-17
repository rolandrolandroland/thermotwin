"""Acquisition-only action selection for the prospective decision experiment.

This module is the first step of the post-replication experiment.  It defines
the frozen four-action menu and a pure selector over action evaluations supplied
by later uncertainty and cost calculations.  It does not open a device
partition, use revealed truth, or alter the completed corrected replication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
import hashlib
import json
import math
from typing import Optional, Sequence, Tuple

from ..numerics.integration import IntegrationDivergenceError
from .operating_decision import (
    APPROVE,
    FINAL_EVALUATION,
    FIXED_FACE_TEMPERATURE,
    FIXED_THERMAL,
    FIXED_VOLTAGE,
    INSUFFICIENT_EVIDENCE,
    POLICY_NAMES,
    REJECT,
    STOP_NOW,
    MarginEnvelope,
    MarginInterval,
    OperatingRegime,
    default_fixed_policies,
    envelope_margin_intervals,
)
from .operating_decision_realism import (
    OperatingDecisionRealismConfig,
    RealisticAcquisitionFitSet,
    forecast_realistic_margin_interval,
)
from .sensor_model_discrimination import MODEL_NAMES


PROSPECTIVE_SELECTOR_PROTOCOL_VERSION = (
    "operating_decision_prospective_selector_v2"
)
PROSPECTIVE_SELECTOR_ALGORITHM_VERSION = (
    "development-padded-stop-and-four-key-ranking-v2"
)
PROSPECTIVE_SELECTOR_SCHEMA_VERSION = 2
PROSPECTIVE_FOUR_ACTION_SELECTOR = "prospective_four_action_selector_v2"
PROSPECTIVE_DEVELOPMENT_OFFSETS_SCHEMA_VERSION = 1
PROSPECTIVE_DEVELOPMENT_OFFSETS_PROTOCOL_VERSION = (
    "operating_decision_development_action_offsets_v1"
)
PROSPECTIVE_UNFITTED_ZERO_OFFSETS_VERSION = "unfitted_zero_offsets_v1"
PROSPECTIVE_ACTIONS = POLICY_NAMES
PROSPECTIVE_ACQUISITION_ACTIONS = (
    FIXED_THERMAL,
    FIXED_VOLTAGE,
    FIXED_FACE_TEMPERATURE,
)
_WIDTH_CONSISTENCY_TOLERANCE = 1.0e-12


def _label(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{name} must be a nonempty, trimmed label")


def _finite_number(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    return converted


def _sha256(name: str, value: str) -> None:
    _label(name, value)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class CandidateExclusion:
    """One acquisition candidate excluded before prospective scoring."""

    model_name: str
    reason: str

    def __post_init__(self) -> None:
        if self.model_name not in MODEL_NAMES:
            raise ValueError("candidate exclusion has an unknown model")
        if self.reason not in ("fit_reached_bound", "optimizer_not_converged"):
            raise ValueError("candidate exclusion has an unknown reason")


@dataclass(frozen=True)
class ProspectiveDevelopmentOffsets:
    """Versioned development-only half-width offsets for action selection.

    These offsets are heuristics learned from development evidence.  They are
    deliberately distinct from the later independent calibration correction.
    The default is named as unfitted so it cannot be mistaken for a calibrated
    scientific rule in serialized provenance.
    """

    version: str = PROSPECTIVE_UNFITTED_ZERO_OFFSETS_VERSION
    stop_now: float = 0.0
    fixed_thermal: float = 0.0
    fixed_voltage: float = 0.0
    fixed_face_temperature: float = 0.0

    def __post_init__(self) -> None:
        _label("development offsets version", self.version)
        for attribute, label in (
            ("stop_now", "stop development offset"),
            ("fixed_thermal", "thermal development offset"),
            ("fixed_voltage", "voltage development offset"),
            (
                "fixed_face_temperature",
                "face-temperature development offset",
            ),
        ):
            converted = _finite_number(label, getattr(self, attribute))
            if converted < 0.0:
                raise ValueError(f"{label} must be nonnegative")
            object.__setattr__(
                self,
                attribute,
                0.0 if converted == 0.0 else converted,
            )
        if self.version == PROSPECTIVE_UNFITTED_ZERO_OFFSETS_VERSION and any(
            self.for_policy(policy_name) != 0.0
            for policy_name in PROSPECTIVE_ACTIONS
        ):
            raise ValueError(
                "the reserved unfitted-zero offset version requires zero values"
            )

    def for_policy(self, policy_name: str) -> float:
        if policy_name == STOP_NOW:
            return self.stop_now
        if policy_name == FIXED_THERMAL:
            return self.fixed_thermal
        if policy_name == FIXED_VOLTAGE:
            return self.fixed_voltage
        if policy_name == FIXED_FACE_TEMPERATURE:
            return self.fixed_face_temperature
        raise ValueError("development offset requested for an unknown policy")


def prospective_development_offsets_payload(
    offsets: ProspectiveDevelopmentOffsets,
) -> dict:
    """Return a strict JSON-ready development-offset record."""

    if not isinstance(offsets, ProspectiveDevelopmentOffsets):
        raise ValueError("prospective offsets need a development-offset record")
    return {
        "schema_version": PROSPECTIVE_DEVELOPMENT_OFFSETS_SCHEMA_VERSION,
        "protocol_version": PROSPECTIVE_DEVELOPMENT_OFFSETS_PROTOCOL_VERSION,
        "version": offsets.version,
        "values": {
            STOP_NOW: offsets.stop_now,
            FIXED_THERMAL: offsets.fixed_thermal,
            FIXED_VOLTAGE: offsets.fixed_voltage,
            FIXED_FACE_TEMPERATURE: offsets.fixed_face_temperature,
        },
    }


def prospective_development_offsets_from_payload(
    payload: dict,
) -> ProspectiveDevelopmentOffsets:
    """Load development offsets while rejecting schema and catalog drift."""

    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "protocol_version",
        "version",
        "values",
    }:
        raise ValueError("prospective development offsets payload is malformed")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"]
        != PROSPECTIVE_DEVELOPMENT_OFFSETS_SCHEMA_VERSION
        or payload["protocol_version"]
        != PROSPECTIVE_DEVELOPMENT_OFFSETS_PROTOCOL_VERSION
    ):
        raise ValueError("prospective development offsets protocol is unsupported")
    values = payload["values"]
    if not isinstance(values, dict) or set(values) != set(PROSPECTIVE_ACTIONS):
        raise ValueError("prospective development offsets action catalog changed")
    return ProspectiveDevelopmentOffsets(
        version=payload["version"],
        stop_now=values[STOP_NOW],
        fixed_thermal=values[FIXED_THERMAL],
        fixed_voltage=values[FIXED_VOLTAGE],
        fixed_face_temperature=values[FIXED_FACE_TEMPERATURE],
    )


@dataclass(frozen=True)
class ProspectiveActionEvaluation:
    """Precomputed padded uncertainty-and-cost evidence for one action.

    The existing uncertainty fields are the development-padded baseline and
    post-action width used for ranking.  The authenticated raw counterparts are
    retained so changing offsets or costs never requires predictive refitting.
    """

    policy_name: str
    eligible: bool
    failure_reason: Optional[str] = None
    uncertainty_before: Optional[float] = None
    expected_uncertainty_after: Optional[float] = None
    declared_cost: Optional[float] = None
    prospective_draw_count: int = 0
    raw_uncertainty_before: Optional[float] = None
    raw_expected_uncertainty_after: Optional[float] = None
    development_offset: Optional[float] = None

    def __post_init__(self) -> None:
        if self.policy_name not in PROSPECTIVE_ACTIONS:
            raise ValueError("prospective evaluation has an unknown policy")
        if not isinstance(self.eligible, bool):
            raise ValueError("prospective eligibility must be boolean")
        if (
            not isinstance(self.prospective_draw_count, int)
            or isinstance(self.prospective_draw_count, bool)
            or self.prospective_draw_count < 0
        ):
            raise ValueError("prospective draw count must be a nonnegative integer")

        numerical = (
            self.uncertainty_before,
            self.expected_uncertainty_after,
            self.declared_cost,
            self.raw_uncertainty_before,
            self.raw_expected_uncertainty_after,
            self.development_offset,
        )
        if self.policy_name == STOP_NOW:
            if not self.eligible or self.failure_reason is not None:
                raise ValueError("stop-now must remain available to the stop gate")
            if any(value is not None for value in numerical):
                raise ValueError("stop-now does not compete through a cost ratio")
            if self.prospective_draw_count != 0:
                raise ValueError("stop-now cannot have prospective draws")
            return

        if not self.eligible:
            if self.failure_reason is None:
                raise ValueError("an ineligible action needs a failure reason")
            _label("action failure reason", self.failure_reason)
            if any(value is not None for value in numerical):
                raise ValueError("an ineligible action cannot retain partial scores")
            if self.prospective_draw_count != 0:
                raise ValueError("an ineligible action cannot retain prospective draws")
            return

        if self.failure_reason is not None:
            raise ValueError("an eligible action cannot have a failure reason")
        if any(
            value is None
            for value in (
                self.uncertainty_before,
                self.expected_uncertainty_after,
                self.declared_cost,
            )
        ):
            raise ValueError("an eligible acquisition action needs complete scores")
        before = _finite_number(
            "uncertainty before",
            self.uncertainty_before,  # type: ignore[arg-type]
        )
        after = _finite_number(
            "expected uncertainty after",
            self.expected_uncertainty_after,  # type: ignore[arg-type]
        )
        cost = _finite_number("declared cost", self.declared_cost)  # type: ignore[arg-type]
        raw_before = _finite_number(
            "raw uncertainty before",
            before
            if self.raw_uncertainty_before is None
            else self.raw_uncertainty_before,
        )
        raw_after = _finite_number(
            "raw expected uncertainty after",
            after
            if self.raw_expected_uncertainty_after is None
            else self.raw_expected_uncertainty_after,
        )
        offset = _finite_number(
            "development action offset",
            0.0 if self.development_offset is None else self.development_offset,
        )
        if before < 0.0 or after < 0.0:
            raise ValueError("prospective uncertainty must be nonnegative")
        if raw_before < 0.0 or raw_after < 0.0 or offset < 0.0:
            raise ValueError("raw uncertainty and development offsets must be nonnegative")
        if before < raw_before or after < raw_after:
            raise ValueError("development padding cannot reduce an uncertainty width")
        if cost <= 0.0:
            raise ValueError("an acquisition action needs positive declared cost")
        if self.prospective_draw_count == 0:
            raise ValueError("an eligible action needs prospective draws")
        if not math.isfinite((before - after) / cost):
            raise ValueError("derived utility per cost must be finite")
        object.__setattr__(
            self,
            "uncertainty_before",
            0.0 if before == 0.0 else before,
        )
        object.__setattr__(
            self,
            "expected_uncertainty_after",
            0.0 if after == 0.0 else after,
        )
        object.__setattr__(self, "declared_cost", cost)
        object.__setattr__(
            self,
            "raw_uncertainty_before",
            0.0 if raw_before == 0.0 else raw_before,
        )
        object.__setattr__(
            self,
            "raw_expected_uncertainty_after",
            0.0 if raw_after == 0.0 else raw_after,
        )
        object.__setattr__(
            self,
            "development_offset",
            0.0 if offset == 0.0 else offset,
        )

    @property
    def expected_uncertainty_reduction(self) -> Optional[float]:
        if self.uncertainty_before is None or self.expected_uncertainty_after is None:
            return None
        return self.uncertainty_before - self.expected_uncertainty_after

    @property
    def utility_per_cost(self) -> Optional[float]:
        reduction = self.expected_uncertainty_reduction
        if reduction is None or self.declared_cost is None:
            return None
        return reduction / self.declared_cost

    @property
    def raw_expected_uncertainty_reduction(self) -> Optional[float]:
        if (
            self.raw_uncertainty_before is None
            or self.raw_expected_uncertainty_after is None
        ):
            return None
        return self.raw_uncertainty_before - self.raw_expected_uncertainty_after


@dataclass(frozen=True)
class ProspectiveAcquisitionSnapshot:
    """Sanitized snapshot available immediately after common acquisition."""

    candidate_models: Tuple[str, ...]
    admissible_candidate_models: Tuple[str, ...]
    excluded_candidates: Tuple[CandidateExclusion, ...]
    failed_candidate_models: Tuple[str, ...]
    model_intervals: Tuple[MarginInterval, ...]
    provisional_margin_envelope: Optional[MarginEnvelope]
    selection_failure_reason: Optional[str]

    def __post_init__(self) -> None:
        candidate_models = tuple(self.candidate_models)
        admissible = tuple(self.admissible_candidate_models)
        exclusions = tuple(self.excluded_candidates)
        failed = tuple(self.failed_candidate_models)
        intervals = tuple(self.model_intervals)
        object.__setattr__(self, "candidate_models", candidate_models)
        object.__setattr__(self, "admissible_candidate_models", admissible)
        object.__setattr__(self, "excluded_candidates", exclusions)
        object.__setattr__(self, "failed_candidate_models", failed)
        object.__setattr__(self, "model_intervals", intervals)

        if candidate_models != tuple(sorted(MODEL_NAMES)):
            raise ValueError("prospective input must account for every candidate model")
        if admissible != tuple(sorted(set(admissible))):
            raise ValueError("admissible candidates must be sorted and unique")
        if not set(admissible).issubset(candidate_models):
            raise ValueError("admissible candidates must come from acquisition fits")
        excluded_models = tuple(item.model_name for item in exclusions)
        if excluded_models != tuple(sorted(set(excluded_models))):
            raise ValueError("candidate exclusions must be sorted and unique")
        if set(admissible).intersection(excluded_models):
            raise ValueError("a candidate cannot be admissible and excluded")
        if failed != tuple(sorted(set(failed))) or any(
            model not in candidate_models for model in failed
        ):
            raise ValueError("failed candidates must be sorted, unique, and known")
        if set(failed).intersection(admissible) or set(failed).intersection(
            excluded_models
        ):
            raise ValueError("failed candidates cannot have another status")
        if set(admissible).union(excluded_models, failed) != set(candidate_models):
            raise ValueError("every acquisition candidate needs an admissibility status")

        if self.selection_failure_reason is not None:
            _label("selection failure reason", self.selection_failure_reason)
            if self.provisional_margin_envelope is not None or intervals:
                raise ValueError("a failed acquisition snapshot cannot expose intervals")
            uncertainty_prefix = "uncertainty_failure:"
            valid_uncertainty_failure = (
                self.selection_failure_reason.startswith(uncertainty_prefix)
                and self.selection_failure_reason[len(uncertainty_prefix) :]
                in admissible
            )
            if self.selection_failure_reason not in (
                "acquisition_fit_failure",
                "no_admissible_candidate",
            ) and not valid_uncertainty_failure:
                raise ValueError("acquisition snapshot has an unknown failure reason")
            if (
                self.selection_failure_reason == "acquisition_fit_failure"
                and not failed
            ):
                raise ValueError("an acquisition fit failure needs a failed candidate")
            if (
                self.selection_failure_reason != "acquisition_fit_failure"
                and failed
            ):
                raise ValueError("failed candidates require acquisition_fit_failure")
            if (
                self.selection_failure_reason == "no_admissible_candidate"
                and admissible
            ):
                raise ValueError("no-admissible failure cannot retain a candidate")
            return

        if failed:
            raise ValueError("a usable acquisition snapshot cannot contain failed fits")
        if not admissible:
            raise ValueError("a usable prospective input needs an admissible candidate")
        if any(
            item.model_name not in candidate_models
            or not all(
                math.isfinite(value)
                for value in (
                    item.estimate,
                    item.local_standard_error,
                    item.lower,
                    item.upper,
                )
            )
            or item.local_standard_error < 0.0
            or not item.lower <= item.estimate <= item.upper
            for item in intervals
        ):
            raise ValueError("prospective margin intervals must be finite and ordered")
        interval_models = tuple(sorted(item.model_name for item in intervals))
        if interval_models != admissible:
            raise ValueError("prospective intervals must cover admissible candidates")
        expected_envelope = envelope_margin_intervals(intervals)
        if expected_envelope != self.provisional_margin_envelope:
            raise ValueError("provisional envelope must contain all admissible intervals")
        if (
            self.provisional_margin_envelope is None
            or not math.isfinite(self.provisional_margin_envelope.lower)
            or not math.isfinite(self.provisional_margin_envelope.upper)
            or self.provisional_margin_envelope.upper
            < self.provisional_margin_envelope.lower
        ):
            raise ValueError("provisional margin envelope must be finite and ordered")


@dataclass(frozen=True)
class ProspectiveSelectorRule:
    """Pure development-padded stopping, ranking, and tie-breaking rule."""

    action_order: Tuple[str, ...] = PROSPECTIVE_ACTIONS
    development_offsets: ProspectiveDevelopmentOffsets = field(
        default_factory=ProspectiveDevelopmentOffsets
    )
    stopping_clearance: float = 0.0
    single_candidate_stopping_clearance: float = 0.0
    minimum_expected_reduction: float = 0.0
    minimum_utility_per_cost: float = 0.0
    utility_decimal_places: int = 12

    def __post_init__(self) -> None:
        action_order = tuple(self.action_order)
        object.__setattr__(self, "action_order", action_order)
        if action_order != PROSPECTIVE_ACTIONS:
            raise ValueError("prospective selector action order is frozen")
        if not isinstance(self.development_offsets, ProspectiveDevelopmentOffsets):
            raise ValueError("prospective selector needs development action offsets")
        for attribute, name, value in (
            ("stopping_clearance", "stopping clearance", self.stopping_clearance),
            (
                "single_candidate_stopping_clearance",
                "single-candidate stopping clearance",
                self.single_candidate_stopping_clearance,
            ),
            (
                "minimum_expected_reduction",
                "minimum expected reduction",
                self.minimum_expected_reduction,
            ),
            (
                "minimum_utility_per_cost",
                "minimum utility per cost",
                self.minimum_utility_per_cost,
            ),
        ):
            converted = _finite_number(name, value)
            if converted < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(
                self,
                attribute,
                0.0 if converted == 0.0 else converted,
            )
        if (
            not isinstance(self.utility_decimal_places, int)
            or isinstance(self.utility_decimal_places, bool)
            or not 0 <= self.utility_decimal_places <= 15
        ):
            raise ValueError("utility precision must be an integer from zero to 15")


@dataclass(frozen=True)
class ProspectiveSelection:
    """Auditable output of the pure four-action selector."""

    selected_policy: Optional[str]
    selection_succeeded: bool
    requires_additional_acquisition: bool
    verification_required: bool
    reason: str
    provisional_decision: str
    ranked_policies: Tuple[str, ...]
    tied_policies: Tuple[str, ...]
    action_evaluations: Tuple[ProspectiveActionEvaluation, ...]
    admissible_candidate_models: Tuple[str, ...]
    candidate_reliability_stratum: str
    padded_initial_margin_envelope: Optional[MarginEnvelope]
    effective_stopping_clearance: float
    selector_protocol_digest: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, bool)
            for value in (
                self.selection_succeeded,
                self.requires_additional_acquisition,
                self.verification_required,
            )
        ):
            raise ValueError("prospective selection state flags must be boolean")
        ranked = tuple(self.ranked_policies)
        tied = tuple(self.tied_policies)
        evaluations = _validated_action_evaluations(self.action_evaluations)
        admissible = tuple(self.admissible_candidate_models)
        object.__setattr__(self, "ranked_policies", ranked)
        object.__setattr__(self, "tied_policies", tied)
        object.__setattr__(self, "action_evaluations", evaluations)
        object.__setattr__(self, "admissible_candidate_models", admissible)
        _label("selection reason", self.reason)
        if admissible != tuple(sorted(set(admissible))) or any(
            model not in MODEL_NAMES for model in admissible
        ):
            raise ValueError("selection candidate models must be sorted and unique")
        expected_stratum = _candidate_reliability_stratum(len(admissible))
        if self.candidate_reliability_stratum != expected_stratum:
            raise ValueError("selection candidate reliability stratum is inconsistent")
        clearance = _finite_number(
            "effective stopping clearance",
            self.effective_stopping_clearance,
        )
        if clearance < 0.0:
            raise ValueError("effective stopping clearance must be nonnegative")
        object.__setattr__(
            self,
            "effective_stopping_clearance",
            0.0 if clearance == 0.0 else clearance,
        )
        envelope = self.padded_initial_margin_envelope
        if envelope is not None and (
            not math.isfinite(envelope.lower)
            or not math.isfinite(envelope.upper)
            or envelope.upper < envelope.lower
        ):
            raise ValueError("padded initial margin envelope must be finite and ordered")
        _sha256("selector protocol digest", self.selector_protocol_digest)
        if self.provisional_decision not in (
            APPROVE,
            REJECT,
            INSUFFICIENT_EVIDENCE,
        ):
            raise ValueError("prospective selection has an unknown provisional decision")
        if self.selection_succeeded != (self.selected_policy is not None):
            raise ValueError("selection success and selected policy disagree")
        if self.verification_required != self.selection_succeeded:
            raise ValueError("every successfully selected action requires verification")
        if self.requires_additional_acquisition != (
            self.selected_policy in PROSPECTIVE_ACQUISITION_ACTIONS
        ):
            raise ValueError("additional-acquisition state disagrees with selected action")
        if (
            self.selected_policy is not None
            and self.selected_policy not in PROSPECTIVE_ACTIONS
        ):
            raise ValueError("prospective selection chose an unknown action")
        if ranked != tuple(dict.fromkeys(ranked)) or any(
            policy not in PROSPECTIVE_ACQUISITION_ACTIONS for policy in ranked
        ):
            raise ValueError("only acquisition actions may enter the utility ranking")
        eligible = {
            item.policy_name
            for item in evaluations
            if item.eligible and item.policy_name in PROSPECTIVE_ACQUISITION_ACTIONS
        }
        if not set(ranked).issubset(eligible):
            raise ValueError("ranked policies must have eligible action scores")
        if tied != tuple(dict.fromkeys(tied)) or not set(tied).issubset(ranked):
            raise ValueError("tied policies must come from the utility ranking")
        if (
            self.selected_policy in PROSPECTIVE_ACQUISITION_ACTIONS
            and self.selected_policy not in ranked
        ):
            raise ValueError("the selected acquisition action must occur in the ranking")

    @property
    def admissible_candidate_count(self) -> int:
        return len(self.admissible_candidate_models)


def _validated_action_evaluations(
    evaluations: Sequence[ProspectiveActionEvaluation],
) -> Tuple[ProspectiveActionEvaluation, ...]:
    scorecard = tuple(evaluations)
    if any(not isinstance(item, ProspectiveActionEvaluation) for item in scorecard):
        raise ValueError("prospective scorecard entries must be action evaluations")
    if tuple(item.policy_name for item in scorecard) != PROSPECTIVE_ACTIONS:
        raise ValueError("prospective action evaluations must use canonical order")
    baselines = tuple(
        item.uncertainty_before
        for item in scorecard
        if item.policy_name in PROSPECTIVE_ACQUISITION_ACTIONS and item.eligible
    )
    if baselines and any(value != baselines[0] for value in baselines[1:]):
        raise ValueError("eligible actions must share one uncertainty baseline")
    return scorecard


def uncomputed_action_evaluations() -> Tuple[ProspectiveActionEvaluation, ...]:
    """Return a safe scorecard until Steps 2 and 3 calculate action values."""

    return (
        ProspectiveActionEvaluation(STOP_NOW, True),
        ProspectiveActionEvaluation(
            FIXED_THERMAL,
            False,
            "expected_uncertainty_not_computed",
        ),
        ProspectiveActionEvaluation(
            FIXED_VOLTAGE,
            False,
            "expected_uncertainty_not_computed",
        ),
        ProspectiveActionEvaluation(
            FIXED_FACE_TEMPERATURE,
            False,
            "expected_uncertainty_not_computed",
        ),
    )


def build_prospective_acquisition_snapshot(
    fit_set: RealisticAcquisitionFitSet,
    final_regime: OperatingRegime,
    physical_config: OperatingDecisionRealismConfig,
) -> ProspectiveAcquisitionSnapshot:
    """Build the acquisition-only snapshot and exclude unreliable fits individually."""

    if not isinstance(physical_config, OperatingDecisionRealismConfig):
        raise ValueError("prospective selection needs a realism configuration")
    if (
        not isinstance(final_regime, OperatingRegime)
        or final_regime.phase != FINAL_EVALUATION
        or final_regime.channels
        or final_regime.current != physical_config.final_current
    ):
        raise ValueError("prospective selection needs the frozen untouched final regime")

    fits = tuple(fit_set.fits)
    fit_names = tuple(sorted(fit.model_name for fit in fits))
    failures = tuple(fit_set.failures)
    failed_names = tuple(sorted(item.model_name for item in failures))
    if len(fit_names) != len(set(fit_names)) or any(
        name not in MODEL_NAMES for name in fit_names
    ):
        raise ValueError("acquisition fit set must contain unique known candidates")
    if len(failed_names) != len(set(failed_names)) or any(
        name not in MODEL_NAMES for name in failed_names
    ):
        raise ValueError("acquisition failures must identify unique known candidates")
    if any(item.stage != "acquisition_fit" for item in failures):
        raise ValueError("prospective selection accepts acquisition failures only")
    if set(fit_names).intersection(failed_names):
        raise ValueError("a candidate cannot both fit and fail acquisition")
    if set(fit_names).union(failed_names) != set(MODEL_NAMES):
        raise ValueError("acquisition evidence must account for every candidate")

    exclusions = []
    admissible_fits = []
    for fit in sorted(fits, key=lambda item: item.model_name):
        if fit.reached_bound:
            exclusions.append(CandidateExclusion(fit.model_name, "fit_reached_bound"))
        elif not fit.converged:
            exclusions.append(
                CandidateExclusion(fit.model_name, "optimizer_not_converged")
            )
        else:
            admissible_fits.append(fit)

    failure_reason = None
    intervals = []
    if failures:
        failure_reason = "acquisition_fit_failure"
    elif not admissible_fits:
        failure_reason = "no_admissible_candidate"
    else:
        for fit in admissible_fits:
            try:
                intervals.append(
                    forecast_realistic_margin_interval(
                        fit,
                        final_regime,
                        physical_config,
                    )
                )
            except (ArithmeticError, IntegrationDivergenceError, ValueError):
                failure_reason = f"uncertainty_failure:{fit.model_name}"
                intervals = []
                break

    envelope = None
    if failure_reason is None:
        envelope = envelope_margin_intervals(intervals)
        if envelope is None:
            failure_reason = "no_admissible_candidate"

    return ProspectiveAcquisitionSnapshot(
        candidate_models=tuple(sorted(MODEL_NAMES)),
        admissible_candidate_models=tuple(
            sorted(fit.model_name for fit in admissible_fits)
        ),
        excluded_candidates=tuple(exclusions),
        failed_candidate_models=failed_names,
        model_intervals=tuple(intervals),
        provisional_margin_envelope=envelope,
        selection_failure_reason=failure_reason,
    )


def _provisional_decision(
    envelope: Optional[MarginEnvelope],
    clearance: float,
) -> str:
    if envelope is None:
        return INSUFFICIENT_EVIDENCE
    if envelope.lower >= clearance:
        return APPROVE
    if envelope.upper < -clearance:
        return REJECT
    return INSUFFICIENT_EVIDENCE


def _candidate_reliability_stratum(candidate_count: int) -> str:
    if candidate_count == 0:
        return "no_admissible_candidate"
    if candidate_count == 1:
        return "one_admissible_candidate"
    if candidate_count == 2:
        return "two_admissible_candidates"
    raise ValueError("prospective selection supports at most two candidate models")


def prospective_padded_initial_envelope(
    acquisition_snapshot: ProspectiveAcquisitionSnapshot,
    offsets: ProspectiveDevelopmentOffsets,
) -> Optional[MarginEnvelope]:
    """Apply the development stop offset to the acquisition-only envelope."""

    if not isinstance(acquisition_snapshot, ProspectiveAcquisitionSnapshot):
        raise ValueError("padded stop gate needs a prospective acquisition snapshot")
    if not isinstance(offsets, ProspectiveDevelopmentOffsets):
        raise ValueError("padded stop gate needs development offsets")
    envelope = acquisition_snapshot.provisional_margin_envelope
    if envelope is None:
        return None
    return MarginEnvelope(
        envelope.lower - offsets.stop_now,
        envelope.upper + offsets.stop_now,
    )


def _effective_stopping_clearance(
    rule: ProspectiveSelectorRule,
    admissible_candidate_count: int,
) -> float:
    return rule.stopping_clearance + (
        rule.single_candidate_stopping_clearance
        if admissible_candidate_count == 1
        else 0.0
    )


def _validate_evaluations_for_rule(
    acquisition_snapshot: ProspectiveAcquisitionSnapshot,
    evaluations: Sequence[ProspectiveActionEvaluation],
    rule: ProspectiveSelectorRule,
) -> None:
    envelope = acquisition_snapshot.provisional_margin_envelope
    raw_baseline = None if envelope is None else envelope.upper - envelope.lower
    padded_baseline = (
        None
        if raw_baseline is None
        else raw_baseline + 2.0 * rule.development_offsets.stop_now
    )
    for evaluation in evaluations:
        if (
            evaluation.policy_name not in PROSPECTIVE_ACQUISITION_ACTIONS
            or not evaluation.eligible
        ):
            continue
        if evaluation.development_offset != rule.development_offsets.for_policy(
            evaluation.policy_name
        ):
            raise ValueError("action evaluation uses different development offsets")
        if raw_baseline is None or padded_baseline is None:
            raise ValueError("eligible action scores need an acquisition envelope")
        if not math.isclose(
            evaluation.raw_uncertainty_before,  # type: ignore[arg-type]
            raw_baseline,
            rel_tol=_WIDTH_CONSISTENCY_TOLERANCE,
            abs_tol=_WIDTH_CONSISTENCY_TOLERANCE,
        ):
            raise ValueError("raw uncertainty baseline does not match the acquisition envelope")
        if not math.isclose(
            evaluation.uncertainty_before,  # type: ignore[arg-type]
            padded_baseline,
            rel_tol=_WIDTH_CONSISTENCY_TOLERANCE,
            abs_tol=_WIDTH_CONSISTENCY_TOLERANCE,
        ):
            raise ValueError("padded uncertainty baseline does not match the stop offset")


def _quantized(value: float, decimal_places: int) -> Decimal:
    quantum = Decimal(1).scaleb(-decimal_places)
    decimal_value = Decimal(str(value))
    integral_digits = max(1, decimal_value.adjusted() + 1)
    with localcontext() as context:
        context.prec = max(28, integral_digits + decimal_places + 4)
        return decimal_value.quantize(quantum, rounding=ROUND_HALF_EVEN)


def _quantized_merit(
    evaluation: ProspectiveActionEvaluation,
    rule: ProspectiveSelectorRule,
) -> Tuple[Decimal, Decimal, Decimal]:
    return (
        _quantized(
            evaluation.utility_per_cost,  # type: ignore[arg-type]
            rule.utility_decimal_places,
        ),
        _quantized(
            evaluation.expected_uncertainty_reduction,  # type: ignore[arg-type]
            rule.utility_decimal_places,
        ),
        _quantized(
            evaluation.declared_cost,  # type: ignore[arg-type]
            rule.utility_decimal_places,
        ),
    )


def _ranked_acquisition_actions(
    evaluations: Sequence[ProspectiveActionEvaluation],
    rule: ProspectiveSelectorRule,
) -> Tuple[ProspectiveActionEvaluation, ...]:
    order = {policy: index for index, policy in enumerate(rule.action_order)}
    eligible = tuple(
        item
        for item in evaluations
        if item.policy_name in PROSPECTIVE_ACQUISITION_ACTIONS and item.eligible
    )
    return tuple(
        sorted(
            eligible,
            key=lambda item: (
                -_quantized_merit(item, rule)[0],
                -_quantized_merit(item, rule)[1],
                _quantized_merit(item, rule)[2],
                order[item.policy_name],
            ),
        )
    )


def select_prospective_action(
    acquisition_snapshot: ProspectiveAcquisitionSnapshot,
    action_evaluations: Sequence[ProspectiveActionEvaluation],
    rule: ProspectiveSelectorRule = ProspectiveSelectorRule(),
) -> ProspectiveSelection:
    """Select a frozen action without using truth or future observations."""

    if not isinstance(acquisition_snapshot, ProspectiveAcquisitionSnapshot):
        raise ValueError("prospective selection needs an acquisition snapshot")
    if not isinstance(rule, ProspectiveSelectorRule):
        raise ValueError("prospective selection needs a selector rule")
    scorecard = _validated_action_evaluations(action_evaluations)
    _validate_evaluations_for_rule(acquisition_snapshot, scorecard, rule)
    admissible = acquisition_snapshot.admissible_candidate_models
    padded_envelope = prospective_padded_initial_envelope(
        acquisition_snapshot,
        rule.development_offsets,
    )
    effective_clearance = _effective_stopping_clearance(rule, len(admissible))
    selection_context = {
        "admissible_candidate_models": admissible,
        "candidate_reliability_stratum": _candidate_reliability_stratum(
            len(admissible)
        ),
        "padded_initial_margin_envelope": padded_envelope,
        "effective_stopping_clearance": effective_clearance,
        "selector_protocol_digest": prospective_selector_protocol_digest(rule),
    }
    provisional = _provisional_decision(
        padded_envelope,
        effective_clearance,
    )
    ranked = _ranked_acquisition_actions(scorecard, rule)
    ranked_names = tuple(item.policy_name for item in ranked)

    if acquisition_snapshot.selection_failure_reason is not None:
        return ProspectiveSelection(
            selected_policy=None,
            selection_succeeded=False,
            requires_additional_acquisition=False,
            verification_required=False,
            reason=(
                "selection_input_failure:"
                f"{acquisition_snapshot.selection_failure_reason}"
            ),
            provisional_decision=INSUFFICIENT_EVIDENCE,
            ranked_policies=ranked_names,
            tied_policies=(),
            action_evaluations=scorecard,
            **selection_context,
        )

    if provisional != INSUFFICIENT_EVIDENCE:
        return ProspectiveSelection(
            selected_policy=STOP_NOW,
            selection_succeeded=True,
            requires_additional_acquisition=False,
            verification_required=True,
            reason="provisionally_resolved_stop_then_verify",
            provisional_decision=provisional,
            ranked_policies=ranked_names,
            tied_policies=(),
            action_evaluations=scorecard,
            **selection_context,
        )

    if not ranked:
        return ProspectiveSelection(
            selected_policy=None,
            selection_succeeded=False,
            requires_additional_acquisition=False,
            verification_required=False,
            reason="no_scored_acquisition_action",
            provisional_decision=provisional,
            ranked_policies=(),
            tied_policies=(),
            action_evaluations=scorecard,
            **selection_context,
        )

    minimum_reduction = _quantized(
        rule.minimum_expected_reduction,
        rule.utility_decimal_places,
    )
    minimum_utility = _quantized(
        rule.minimum_utility_per_cost,
        rule.utility_decimal_places,
    )
    qualifying = tuple(
        item
        for item in ranked
        if _quantized(
            item.expected_uncertainty_reduction,  # type: ignore[arg-type]
            rule.utility_decimal_places,
        )
        > minimum_reduction
        and _quantized(
            item.utility_per_cost,  # type: ignore[arg-type]
            rule.utility_decimal_places,
        )
        > minimum_utility
    )
    if not qualifying:
        return ProspectiveSelection(
            selected_policy=STOP_NOW,
            selection_succeeded=True,
            requires_additional_acquisition=False,
            verification_required=True,
            reason="no_valuable_acquisition_stop_then_verify",
            provisional_decision=provisional,
            ranked_policies=ranked_names,
            tied_policies=(),
            action_evaluations=scorecard,
            **selection_context,
        )

    selected = qualifying[0]
    top_merit = _quantized_merit(selected, rule)
    tied = tuple(
        item.policy_name
        for item in qualifying
        if _quantized_merit(item, rule) == top_merit
    )
    reason = (
        "maximum_value_per_cost_with_frozen_tie_break"
        if len(tied) > 1
        else "maximum_expected_uncertainty_reduction_per_cost"
    )
    return ProspectiveSelection(
        selected_policy=selected.policy_name,
        selection_succeeded=True,
        requires_additional_acquisition=True,
        verification_required=True,
        reason=reason,
        provisional_decision=provisional,
        ranked_policies=ranked_names,
        tied_policies=tied,
        action_evaluations=scorecard,
        **selection_context,
    )


def prospective_action_catalog_payload() -> dict:
    """Return the complete acquisition catalog independently of any rule."""

    policies = []
    for policy in default_fixed_policies():
        regimes = []
        for regime in policy.additional_regimes:
            regimes.append(
                {
                    "name": regime.name,
                    "phase": regime.phase,
                    "transition_times": [
                        float(value) for value in regime.current.transition_times
                    ],
                    "current_values": [
                        float(value) for value in regime.current.values
                    ],
                    "channels": list(regime.channels),
                }
            )
        policies.append(
            {
                "name": policy.name,
                "additional_regimes": regimes,
                "extra_sensor_count": policy.extra_sensor_count,
            }
        )
    return {
        "actions": list(PROSPECTIVE_ACTIONS),
        "acquisition_actions": list(PROSPECTIVE_ACQUISITION_ACTIONS),
        "policies": policies,
    }


def prospective_action_catalog_digest() -> str:
    """Return the content digest of the actual four-action catalog."""

    material = json.dumps(
        prospective_action_catalog_payload(),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def prospective_selector_protocol_digest(
    rule: ProspectiveSelectorRule = ProspectiveSelectorRule(),
) -> str:
    """Bind the actual action catalog and supplied selection rule."""

    if not isinstance(rule, ProspectiveSelectorRule):
        raise ValueError("prospective selector digest needs a selector rule")
    material = json.dumps(
        {
            "protocol_version": PROSPECTIVE_SELECTOR_PROTOCOL_VERSION,
            "algorithm_version": PROSPECTIVE_SELECTOR_ALGORITHM_VERSION,
            "procedure_name": PROSPECTIVE_FOUR_ACTION_SELECTOR,
            "action_catalog_digest": prospective_action_catalog_digest(),
            "action_catalog": prospective_action_catalog_payload(),
            "rule": {
                "action_order": list(rule.action_order),
                "development_offsets": prospective_development_offsets_payload(
                    rule.development_offsets
                ),
                "stopping_clearance": rule.stopping_clearance,
                "single_candidate_stopping_clearance": (
                    rule.single_candidate_stopping_clearance
                ),
                "minimum_expected_reduction": rule.minimum_expected_reduction,
                "minimum_utility_per_cost": rule.minimum_utility_per_cost,
                "utility_decimal_places": rule.utility_decimal_places,
            },
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def prospective_selector_rule_payload(
    rule: ProspectiveSelectorRule = ProspectiveSelectorRule(),
) -> dict:
    """Return a strict, finite JSON-ready Step-1 rule record."""

    return {
        "schema_version": PROSPECTIVE_SELECTOR_SCHEMA_VERSION,
        "protocol_version": PROSPECTIVE_SELECTOR_PROTOCOL_VERSION,
        "algorithm_version": PROSPECTIVE_SELECTOR_ALGORITHM_VERSION,
        "protocol_digest": prospective_selector_protocol_digest(rule),
        "procedure_name": PROSPECTIVE_FOUR_ACTION_SELECTOR,
        "action_catalog_digest": prospective_action_catalog_digest(),
        "action_order": list(rule.action_order),
        "development_offsets": prospective_development_offsets_payload(
            rule.development_offsets
        ),
        "stopping_clearance": rule.stopping_clearance,
        "single_candidate_stopping_clearance": (
            rule.single_candidate_stopping_clearance
        ),
        "minimum_expected_reduction": rule.minimum_expected_reduction,
        "minimum_utility_per_cost": rule.minimum_utility_per_cost,
        "utility_decimal_places": rule.utility_decimal_places,
    }


def prospective_selector_rule_from_payload(payload: dict) -> ProspectiveSelectorRule:
    """Load a Step-1 rule while rejecting schema or protocol drift."""

    expected = {
        "schema_version",
        "protocol_version",
        "algorithm_version",
        "protocol_digest",
        "procedure_name",
        "action_catalog_digest",
        "action_order",
        "development_offsets",
        "stopping_clearance",
        "single_candidate_stopping_clearance",
        "minimum_expected_reduction",
        "minimum_utility_per_cost",
        "utility_decimal_places",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("prospective selector payload is malformed")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != PROSPECTIVE_SELECTOR_SCHEMA_VERSION
    ):
        raise ValueError("prospective selector schema version is unsupported")
    if payload["protocol_version"] != PROSPECTIVE_SELECTOR_PROTOCOL_VERSION:
        raise ValueError("prospective selector protocol version is unsupported")
    if payload["algorithm_version"] != PROSPECTIVE_SELECTOR_ALGORITHM_VERSION:
        raise ValueError("prospective selector algorithm version is unsupported")
    if payload["procedure_name"] != PROSPECTIVE_FOUR_ACTION_SELECTOR:
        raise ValueError("prospective selector procedure name is unsupported")
    if payload["action_catalog_digest"] != prospective_action_catalog_digest():
        raise ValueError("prospective selector action catalog digest is invalid")
    action_order = payload["action_order"]
    if not isinstance(action_order, list):
        raise ValueError("prospective selector action order must be a list")
    rule = ProspectiveSelectorRule(
        action_order=tuple(action_order),
        development_offsets=prospective_development_offsets_from_payload(
            payload["development_offsets"]
        ),
        stopping_clearance=payload["stopping_clearance"],
        single_candidate_stopping_clearance=(
            payload["single_candidate_stopping_clearance"]
        ),
        minimum_expected_reduction=payload["minimum_expected_reduction"],
        minimum_utility_per_cost=payload["minimum_utility_per_cost"],
        utility_decimal_places=payload["utility_decimal_places"],
    )
    if payload["protocol_digest"] != prospective_selector_protocol_digest(rule):
        raise ValueError("prospective selector protocol digest is invalid")
    return rule


__all__ = [
    "CandidateExclusion",
    "PROSPECTIVE_ACQUISITION_ACTIONS",
    "PROSPECTIVE_ACTIONS",
    "PROSPECTIVE_DEVELOPMENT_OFFSETS_PROTOCOL_VERSION",
    "PROSPECTIVE_DEVELOPMENT_OFFSETS_SCHEMA_VERSION",
    "PROSPECTIVE_FOUR_ACTION_SELECTOR",
    "PROSPECTIVE_SELECTOR_PROTOCOL_VERSION",
    "PROSPECTIVE_SELECTOR_ALGORITHM_VERSION",
    "PROSPECTIVE_SELECTOR_SCHEMA_VERSION",
    "PROSPECTIVE_UNFITTED_ZERO_OFFSETS_VERSION",
    "ProspectiveAcquisitionSnapshot",
    "ProspectiveActionEvaluation",
    "ProspectiveDevelopmentOffsets",
    "ProspectiveSelection",
    "ProspectiveSelectorRule",
    "build_prospective_acquisition_snapshot",
    "prospective_action_catalog_digest",
    "prospective_action_catalog_payload",
    "prospective_development_offsets_from_payload",
    "prospective_development_offsets_payload",
    "prospective_padded_initial_envelope",
    "prospective_selector_protocol_digest",
    "prospective_selector_rule_from_payload",
    "prospective_selector_rule_payload",
    "select_prospective_action",
    "uncomputed_action_evaluations",
]
