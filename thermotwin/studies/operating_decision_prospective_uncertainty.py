"""Posterior-predictive uncertainty scoring for prospective measurements.

This module implements Step 2 of the prospective operating-decision
experiment.  It uses only the exact common initial acquisition, fitted
candidate models, and synthetic future observations.  It never accepts hidden
truth, verification observations, realized action data, or final responses.

The completed corrected-replication modules are imported but never modified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from statistics import fmean
from typing import Optional, Sequence, Tuple

from ..numerics.integration import IntegrationDivergenceError
from ..observations.test_stand import regular_measurement_times
from ..simulation.temporary_face_sensor import TemporaryFaceSensor
from .operating_decision import (
    FINAL_EVALUATION,
    FIXED_FACE_TEMPERATURE,
    POLICY_NAMES,
    RUN_DURATION_SECONDS,
    STOP_NOW,
    FixedPolicy,
    MarginInterval,
    NumericalFailure,
    OperatingRegime,
    default_fixed_policies,
    envelope_margin_intervals,
    initial_acquisition_regime,
)
from .operating_decision_prospective import (
    PROSPECTIVE_ACQUISITION_ACTIONS,
    ProspectiveDevelopmentOffsets,
    ProspectiveAcquisitionSnapshot,
    build_prospective_acquisition_snapshot,
    prospective_action_catalog_digest,
)
from .operating_decision_prospective_random_streams import (
    PARAMETER_DRAW,
    PROBE_DRAW,
    PROSPECTIVE_RANDOM_STREAM_PROTOCOL_VERSION,
    ProspectiveRandomStream,
    ProspectiveRandomStreamAudit,
    ProspectiveRandomStreamNamespace,
    ProspectiveRandomStreamRegistry,
    ProspectiveStreamUse,
    RUN_BIAS,
    WHITE_NOISE,
    prospective_observation_stream,
    prospective_parameter_stream,
    prospective_probe_stream,
)
from .operating_decision_realism import (
    OperatingDecisionRealismConfig,
    RealisticAcquisitionFitSet,
    RealisticCandidateFit,
    RealisticOperatingRun,
    RunInstrumentation,
    _decoded_parameters,
    _fit_start_vectors,
    _instrumentation_for_regime,
    _parameter_spec,
    _simulate_realistic_observables,
    fit_realistic_candidate,
    forecast_realistic_margin_interval,
)
from .operating_decision_replication import corrected_physical_protocol_digest
from .sensor_model_discrimination import (
    ALL_CHANNELS,
    COLD_EXCHANGER,
    COLD_FACE,
    HOT_EXCHANGER,
    MODEL_NAMES,
    ObservableRun,
    ObservableValue,
)


PROSPECTIVE_UNCERTAINTY_SCHEMA_VERSION = 3
PROSPECTIVE_UNCERTAINTY_PROTOCOL_VERSION = (
    "operating_decision_prospective_uncertainty_v3"
)
PROSPECTIVE_UNCERTAINTY_ESTIMATOR = "bounded_posterior_predictive_refit_v3"
PROSPECTIVE_UNCERTAINTY_METRIC = "candidate_envelope_full_width_kelvin"
PROSPECTIVE_WITHIN_GENERATOR_AGGREGATION = "arithmetic_mean"
PROSPECTIVE_ACROSS_GENERATOR_AGGREGATION = "worst_case_maximum"
PROSPECTIVE_DRAW_FAILURE_POLICY = "baseline_width_imputation"
PROSPECTIVE_CANDIDATE_ATTRITION_POLICY = (
    "no_positive_gain_candidate_exclusion_retained_as_usable"
)
PROSPECTIVE_REFIT_START_PROTOCOL = "frozen_three_start_multistart"
PROSPECTIVE_ELIGIBILITY_PROTOCOL = (
    "explicit_draw_count_and_max_whole_draw_failures_per_source_action_v2"
)
PROSPECTIVE_PADDED_SCORING_PROTOCOL = (
    "development_offsets_with_padded_failure_floor_v1"
)
_FINAL_REGIME_NAME = "untouched_final_operating_schedule"
_ACQUISITION_EVIDENCE_DOMAIN = "thermotwin.prospective_acquisition_evidence"
_RESULT_DIGEST_DOMAIN = "thermotwin.prospective_uncertainty_result"
_EVIDENCE_CONSTRUCTION_SEAL = object()


class _ProspectiveDrawSamplingError(ValueError):
    """A retained Monte Carlo draw failed after structural validation."""


def _label(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{name} must be a nonempty, trimmed label")


def _finite(name: str, value: float, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted) or (nonnegative and converted < 0.0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return 0.0 if converted == 0.0 else converted


def _sha256(name: str, value: str) -> None:
    _label(name, value)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class ProspectiveUncertaintyConfig:
    """Frozen Monte Carlo and aggregation choices for Step 2 development.

    In protocol v3, the retained ``max_unstable`` field name means the maximum
    number of true whole-draw failures per source/action.  Candidate-level
    bound hits and nonconvergence remain recorded and conservatively scored,
    but a surviving admissible candidate makes the completed draw usable.
    """

    draw_count: int = 4
    max_parameter_draw_attempts: int = 256
    max_unstable_draws_per_source_action: int = 0
    estimator: str = PROSPECTIVE_UNCERTAINTY_ESTIMATOR
    uncertainty_metric: str = PROSPECTIVE_UNCERTAINTY_METRIC
    within_generator_aggregation: str = PROSPECTIVE_WITHIN_GENERATOR_AGGREGATION
    across_generator_aggregation: str = PROSPECTIVE_ACROSS_GENERATOR_AGGREGATION
    draw_failure_policy: str = PROSPECTIVE_DRAW_FAILURE_POLICY
    candidate_attrition_policy: str = PROSPECTIVE_CANDIDATE_ATTRITION_POLICY
    refit_start_protocol: str = PROSPECTIVE_REFIT_START_PROTOCOL

    def __post_init__(self) -> None:
        if (
            not isinstance(self.draw_count, int)
            or isinstance(self.draw_count, bool)
            or self.draw_count <= 0
        ):
            raise ValueError("prospective draw count must be a positive integer")
        if (
            not isinstance(self.max_parameter_draw_attempts, int)
            or isinstance(self.max_parameter_draw_attempts, bool)
            or self.max_parameter_draw_attempts <= 0
        ):
            raise ValueError("parameter draw attempts must be a positive integer")
        if (
            not isinstance(self.max_unstable_draws_per_source_action, int)
            or isinstance(self.max_unstable_draws_per_source_action, bool)
            or self.max_unstable_draws_per_source_action < 0
            or self.max_unstable_draws_per_source_action >= self.draw_count
        ):
            raise ValueError(
                "maximum unstable draws must be a nonnegative integer below draw count"
            )
        expected = {
            "estimator": PROSPECTIVE_UNCERTAINTY_ESTIMATOR,
            "uncertainty_metric": PROSPECTIVE_UNCERTAINTY_METRIC,
            "within_generator_aggregation": (
                PROSPECTIVE_WITHIN_GENERATOR_AGGREGATION
            ),
            "across_generator_aggregation": (
                PROSPECTIVE_ACROSS_GENERATOR_AGGREGATION
            ),
            "draw_failure_policy": PROSPECTIVE_DRAW_FAILURE_POLICY,
            "candidate_attrition_policy": (
                PROSPECTIVE_CANDIDATE_ATTRITION_POLICY
            ),
            "refit_start_protocol": PROSPECTIVE_REFIT_START_PROTOCOL,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"unsupported prospective {name.replace('_', ' ')}")


@dataclass(frozen=True)
class ProspectiveCandidateOutcome:
    """One candidate's numerical status after a hypothetical action."""

    model_name: str
    status: str
    objective: Optional[float]
    interval: Optional[MarginInterval]

    def __post_init__(self) -> None:
        if self.model_name not in MODEL_NAMES:
            raise ValueError("prospective candidate outcome has an unknown model")
        if self.status not in (
            "admissible",
            "fit_reached_bound",
            "optimizer_not_converged",
            "acquisition_fit_failure",
            "uncertainty_failure",
        ):
            raise ValueError("prospective candidate outcome has an unknown status")
        if self.objective is not None:
            object.__setattr__(
                self,
                "objective",
                _finite("fit objective", self.objective, nonnegative=True),
            )
        if self.status == "admissible":
            if self.objective is None or self.interval is None:
                raise ValueError("an admissible refit needs an objective and interval")
            if self.interval.model_name != self.model_name:
                raise ValueError("candidate outcome and interval model disagree")
            values = (
                self.interval.estimate,
                self.interval.local_standard_error,
                self.interval.lower,
                self.interval.upper,
            )
            if (
                any(not math.isfinite(value) for value in values)
                or self.interval.local_standard_error < 0.0
                or not self.interval.lower
                <= self.interval.estimate
                <= self.interval.upper
            ):
                raise ValueError("candidate outcome interval must be finite and ordered")
        elif self.interval is not None:
            raise ValueError("an inadmissible candidate cannot retain an interval")


@dataclass(frozen=True)
class ProspectiveDrawOutcome:
    """Retained result for one source-model, action, and draw index.

    ``stable`` denotes a usable completed draw, not preservation of the exact
    initial candidate set.  Candidate transitions are carried separately and
    trigger the conservative baseline-width floor.
    """

    policy_name: str
    generator_model: str
    draw_index: int
    generator_log_offsets: Tuple[float, ...]
    face_probe_log_offsets: Optional[Tuple[float, float]]
    candidate_outcomes: Tuple[ProspectiveCandidateOutcome, ...]
    initially_admissible_became_inadmissible: Tuple[str, ...]
    initially_excluded_became_admissible: Tuple[str, ...]
    stable: bool
    failed: bool
    failure_stage: Optional[str]
    failure_model: Optional[str]
    failure_type: Optional[str]
    raw_after_width: Optional[float]
    scored_after_width: float
    synthetic_observation_digest: Optional[str]

    def __post_init__(self) -> None:
        if self.policy_name not in PROSPECTIVE_ACQUISITION_ACTIONS:
            raise ValueError("prospective draw has an unknown acquisition action")
        if self.generator_model not in MODEL_NAMES:
            raise ValueError("prospective draw has an unknown generator model")
        if (
            not isinstance(self.draw_index, int)
            or isinstance(self.draw_index, bool)
            or self.draw_index < 0
        ):
            raise ValueError("prospective draw index must be nonnegative")
        offsets = tuple(
            _finite("generator log offset", item)
            for item in self.generator_log_offsets
        )
        outcomes = tuple(self.candidate_outcomes)
        newly_excluded = tuple(self.initially_admissible_became_inadmissible)
        recovered = tuple(self.initially_excluded_became_admissible)
        object.__setattr__(self, "generator_log_offsets", offsets)
        object.__setattr__(self, "candidate_outcomes", outcomes)
        object.__setattr__(self, "initially_admissible_became_inadmissible", newly_excluded)
        object.__setattr__(self, "initially_excluded_became_admissible", recovered)
        if self.face_probe_log_offsets is not None:
            probe = tuple(
                _finite("face probe log offset", item)
                for item in self.face_probe_log_offsets
            )
            if len(probe) != 2 or self.policy_name != FIXED_FACE_TEMPERATURE:
                raise ValueError("face-probe offsets belong only to the face action")
            object.__setattr__(self, "face_probe_log_offsets", probe)
        elif self.policy_name == FIXED_FACE_TEMPERATURE and not self.failed:
            raise ValueError("a completed face draw needs probe offsets")
        if tuple(item.model_name for item in outcomes) != tuple(sorted(
            item.model_name for item in outcomes
        )) or len({item.model_name for item in outcomes}) != len(outcomes):
            raise ValueError("candidate outcomes must be sorted and unique")
        for models in (newly_excluded, recovered):
            if models != tuple(sorted(set(models))) or any(
                model not in MODEL_NAMES for model in models
            ):
                raise ValueError("candidate transitions must be sorted and unique")
        if set(newly_excluded).intersection(recovered):
            raise ValueError("candidate transitions cannot overlap")
        if not isinstance(self.stable, bool) or not isinstance(self.failed, bool):
            raise ValueError("prospective draw state flags must be boolean")
        scored = _finite(
            "scored after width",
            self.scored_after_width,
            nonnegative=True,
        )
        object.__setattr__(self, "scored_after_width", scored)
        if self.failed:
            if self.stable or self.raw_after_width is not None:
                raise ValueError("a failed draw cannot be stable or retain a raw width")
            if any(item.status == "admissible" for item in outcomes):
                raise ValueError(
                    "a failed draw cannot retain an admissible candidate"
                )
            for name, value in (
                ("failure stage", self.failure_stage),
                ("failure type", self.failure_type),
            ):
                if value is None:
                    raise ValueError(f"a failed draw needs {name}")
                _label(name, value)
            if self.synthetic_observation_digest is not None:
                _sha256(
                    "synthetic observation digest",
                    self.synthetic_observation_digest,
                )
            if self.failure_model is not None and self.failure_model not in MODEL_NAMES:
                raise ValueError("a failed draw has an unknown failure model")
        else:
            if tuple(item.model_name for item in outcomes) != tuple(
                sorted(MODEL_NAMES)
            ):
                raise ValueError("a completed draw must account for every candidate")
            if any(
                value is not None
                for value in (self.failure_stage, self.failure_model, self.failure_type)
            ):
                raise ValueError("a completed draw cannot retain a failure")
            raw = _finite(
                "raw after width",
                self.raw_after_width,  # type: ignore[arg-type]
                nonnegative=True,
            )
            object.__setattr__(self, "raw_after_width", raw)
            if self.synthetic_observation_digest is None:
                raise ValueError("a completed draw needs its observation digest")
            _sha256(
                "synthetic observation digest",
                self.synthetic_observation_digest,
            )
            status_by_model = {item.model_name: item.status for item in outcomes}
            if not any(status == "admissible" for status in status_by_model.values()):
                raise ValueError(
                    "a completed draw needs at least one admissible candidate"
                )
            if not self.stable:
                raise ValueError(
                    "a completed draw with an admissible candidate must be stable"
                )
            if any(status_by_model[model] == "admissible" for model in newly_excluded):
                raise ValueError("newly excluded candidates cannot remain admissible")
            if any(status_by_model[model] != "admissible" for model in recovered):
                raise ValueError("recovered candidates must be admissible")


@dataclass(frozen=True)
class ProspectiveModelActionSummary:
    """Mean retained width and whole-draw usability for one source/action."""

    policy_name: str
    generator_model: str
    draw_count: int
    stable_draw_count: int
    failed_draw_count: int
    mean_after_width: float
    eligible: bool
    max_unstable_draws: int = 0

    def __post_init__(self) -> None:
        if self.policy_name not in PROSPECTIVE_ACQUISITION_ACTIONS:
            raise ValueError("model-action summary has an unknown policy")
        if self.generator_model not in MODEL_NAMES:
            raise ValueError("model-action summary has an unknown generator")
        for name, value in (
            ("draw count", self.draw_count),
            ("stable draw count", self.stable_draw_count),
            ("failed draw count", self.failed_draw_count),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.draw_count == 0 or self.stable_draw_count > self.draw_count:
            raise ValueError("model-action draw counts are inconsistent")
        if self.failed_draw_count > self.draw_count:
            raise ValueError("model-action failure count is inconsistent")
        if self.stable_draw_count + self.failed_draw_count != self.draw_count:
            raise ValueError(
                "every model-action draw must be stable or a whole-draw failure"
            )
        object.__setattr__(
            self,
            "mean_after_width",
            _finite("mean after width", self.mean_after_width, nonnegative=True),
        )
        if not isinstance(self.eligible, bool):
            raise ValueError("model-action eligibility must be boolean")
        if (
            not isinstance(self.max_unstable_draws, int)
            or isinstance(self.max_unstable_draws, bool)
            or self.max_unstable_draws < 0
            or self.max_unstable_draws >= self.draw_count
        ):
            raise ValueError("summary maximum unstable draws is inconsistent")
        if self.eligible != (
            self.unstable_draw_count <= self.max_unstable_draws
        ):
            raise ValueError("model-action eligibility does not match unstable draws")

    @property
    def unstable_draw_count(self) -> int:
        return self.draw_count - self.stable_draw_count


@dataclass(frozen=True)
class ProspectiveActionUncertainty:
    """Cost-free expected uncertainty result consumed later by Step 3."""

    policy_name: str
    eligible: bool
    failure_reason: Optional[str]
    uncertainty_before: float
    expected_uncertainty_after: float
    prospective_draw_count: int
    source_summaries: Tuple[ProspectiveModelActionSummary, ...]

    def __post_init__(self) -> None:
        if self.policy_name not in POLICY_NAMES:
            raise ValueError("action uncertainty has an unknown policy")
        if not isinstance(self.eligible, bool):
            raise ValueError("action uncertainty eligibility must be boolean")
        before = _finite(
            "uncertainty before",
            self.uncertainty_before,
            nonnegative=True,
        )
        after = _finite(
            "expected uncertainty after",
            self.expected_uncertainty_after,
            nonnegative=True,
        )
        object.__setattr__(self, "uncertainty_before", before)
        object.__setattr__(self, "expected_uncertainty_after", after)
        summaries = tuple(self.source_summaries)
        object.__setattr__(self, "source_summaries", summaries)
        if (
            not isinstance(self.prospective_draw_count, int)
            or isinstance(self.prospective_draw_count, bool)
            or self.prospective_draw_count < 0
        ):
            raise ValueError("prospective draw count must be nonnegative")
        if self.policy_name == STOP_NOW:
            if not self.eligible or self.failure_reason is not None:
                raise ValueError("stop reference must remain eligible")
            if after != before or self.prospective_draw_count != 0 or summaries:
                raise ValueError("stop reference must preserve baseline uncertainty")
            return
        if tuple(item.generator_model for item in summaries) != tuple(sorted(
            item.generator_model for item in summaries
        )) or len({item.generator_model for item in summaries}) != len(summaries):
            raise ValueError("source summaries must use sorted generator order")
        if not summaries:
            raise ValueError("an acquisition action needs source summaries")
        if any(item.policy_name != self.policy_name for item in summaries):
            raise ValueError("source summaries must match their action")
        if len({item.draw_count for item in summaries}) != 1:
            raise ValueError("source summaries must use a common draw count")
        if self.prospective_draw_count != sum(item.draw_count for item in summaries):
            raise ValueError("action uncertainty draw count is inconsistent")
        if after != max(item.mean_after_width for item in summaries):
            raise ValueError("expected uncertainty must be the worst source mean")
        if self.eligible:
            if self.failure_reason is not None or not all(
                item.eligible for item in summaries
            ):
                raise ValueError("eligible action uncertainty has inconsistent sources")
        else:
            if self.failure_reason is None or all(item.eligible for item in summaries):
                raise ValueError("ineligible action uncertainty needs a reason")
            _label("action uncertainty failure reason", self.failure_reason)

    @property
    def expected_uncertainty_reduction(self) -> float:
        return self.uncertainty_before - self.expected_uncertainty_after


@dataclass(frozen=True)
class ProspectivePaddedModelActionSummary:
    """A raw source summary plus its cheaply recomputed padded mean width."""

    raw_summary: ProspectiveModelActionSummary
    padded_mean_after_width: float

    def __post_init__(self) -> None:
        if not isinstance(self.raw_summary, ProspectiveModelActionSummary):
            raise ValueError("padded source summary needs its raw summary")
        padded = _finite(
            "padded mean after width",
            self.padded_mean_after_width,
            nonnegative=True,
        )
        if padded < self.raw_summary.mean_after_width:
            raise ValueError("development padding cannot reduce a source mean width")
        object.__setattr__(self, "padded_mean_after_width", padded)


@dataclass(frozen=True)
class ProspectivePaddedActionUncertainty:
    """Development-padded widths derived without repeating predictive fits."""

    raw_action: ProspectiveActionUncertainty
    development_offset: float
    padded_uncertainty_before: float
    padded_expected_uncertainty_after: float
    source_summaries: Tuple[ProspectivePaddedModelActionSummary, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.raw_action, ProspectiveActionUncertainty):
            raise ValueError("padded action uncertainty needs its raw action")
        offset = _finite(
            "development action offset",
            self.development_offset,
            nonnegative=True,
        )
        before = _finite(
            "padded uncertainty before",
            self.padded_uncertainty_before,
            nonnegative=True,
        )
        after = _finite(
            "padded expected uncertainty after",
            self.padded_expected_uncertainty_after,
            nonnegative=True,
        )
        summaries = tuple(self.source_summaries)
        object.__setattr__(self, "development_offset", offset)
        object.__setattr__(self, "padded_uncertainty_before", before)
        object.__setattr__(self, "padded_expected_uncertainty_after", after)
        object.__setattr__(self, "source_summaries", summaries)
        if before < self.raw_action.uncertainty_before:
            raise ValueError("padded baseline cannot be narrower than the raw baseline")
        if after < self.raw_action.expected_uncertainty_after:
            raise ValueError("padded action width cannot be narrower than its raw width")
        if self.policy_name == STOP_NOW:
            if summaries or before != after:
                raise ValueError("padded stop action must preserve its padded baseline")
            return
        if tuple(item.raw_summary for item in summaries) != (
            self.raw_action.source_summaries
        ):
            raise ValueError("padded source summaries do not match raw evidence")
        if after != max(item.padded_mean_after_width for item in summaries):
            raise ValueError("padded action width must be the worst source mean")

    @property
    def policy_name(self) -> str:
        return self.raw_action.policy_name

    @property
    def eligible(self) -> bool:
        return self.raw_action.eligible

    @property
    def failure_reason(self) -> Optional[str]:
        return self.raw_action.failure_reason

    @property
    def expected_uncertainty_reduction(self) -> float:
        return (
            self.padded_uncertainty_before
            - self.padded_expected_uncertainty_after
        )


@dataclass(frozen=True)
class ProspectiveAcquisitionEvidence:
    """Authenticated common-acquisition evidence prepared by this module."""

    common_initial_run: RealisticOperatingRun
    fit_set: RealisticAcquisitionFitSet
    snapshot: ProspectiveAcquisitionSnapshot
    final_regime: OperatingRegime
    physical_protocol_digest: str
    evidence_digest: str
    _construction_seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._construction_seal is not _EVIDENCE_CONSTRUCTION_SEAL:
            raise ValueError("prospective evidence must come from its acquisition factory")
        for name, value in (
            ("physical protocol digest", self.physical_protocol_digest),
            ("acquisition evidence digest", self.evidence_digest),
        ):
            _sha256(name, value)


def _audit_stream_uses(
    uses: Sequence[ProspectiveStreamUse],
) -> ProspectiveRandomStreamAudit:
    registry = ProspectiveRandomStreamRegistry()
    for use in uses:
        if not isinstance(use, ProspectiveStreamUse):
            raise ValueError("prospective result contains an invalid stream use")
        registry.register(
            use.stream,
            consumer=use.consumer,
            shared_for_action=use.shared_for_action,
        )
    return registry.audit()


def _validate_draw_against_acquisition(
    draw: ProspectiveDrawOutcome,
    snapshot: ProspectiveAcquisitionSnapshot,
    baseline_width: float,
) -> None:
    if draw.generator_model not in snapshot.admissible_candidate_models:
        raise ValueError("prospective draw uses an inadmissible source model")
    if draw.failed:
        if (
            draw.scored_after_width != baseline_width
            or draw.initially_admissible_became_inadmissible
            or draw.initially_excluded_became_admissible
        ):
            raise ValueError("failed prospective draw must retain the baseline width")
        return

    intervals = tuple(
        outcome.interval
        for outcome in draw.candidate_outcomes
        if outcome.status == "admissible" and outcome.interval is not None
    )
    envelope = envelope_margin_intervals(intervals)
    if envelope is None:
        raise ValueError("completed prospective draw needs an admissible candidate")
    raw_width = envelope.upper - envelope.lower
    initially_admissible = set(snapshot.admissible_candidate_models)
    initially_excluded = {
        exclusion.model_name for exclusion in snapshot.excluded_candidates
    }
    now_admissible = {interval.model_name for interval in intervals}
    newly_excluded = tuple(sorted(initially_admissible - now_admissible))
    recovered = tuple(sorted(initially_excluded.intersection(now_admissible)))
    scored_width = max(raw_width, baseline_width) if newly_excluded else raw_width
    if (
        draw.raw_after_width != raw_width
        or draw.scored_after_width != scored_width
        or draw.initially_admissible_became_inadmissible != newly_excluded
        or draw.initially_excluded_became_admissible != recovered
        or not draw.stable
    ):
        raise ValueError("prospective draw does not match its candidate outcomes")


def _validate_draw_parameters(
    draws: Sequence[ProspectiveDrawOutcome],
    evidence: ProspectiveAcquisitionEvidence,
    physical_config: OperatingDecisionRealismConfig,
    config: ProspectiveUncertaintyConfig,
) -> None:
    fit_by_model = {fit.model_name: fit for fit in evidence.fit_set.fits}
    by_source_draw = {}
    for draw in draws:
        by_source_draw.setdefault(
            (draw.generator_model, draw.draw_index),
            [],
        ).append(draw)
    for generator_model in evidence.snapshot.admissible_candidate_models:
        fit = fit_by_model[generator_model]
        spec = _parameter_spec(generator_model, False, physical_config)
        for draw_index in range(config.draw_count):
            selected = tuple(
                sorted(
                    by_source_draw[(generator_model, draw_index)],
                    key=lambda item: PROSPECTIVE_ACQUISITION_ACTIONS.index(
                        item.policy_name
                    ),
                )
            )
            parameter_failed = all(
                item.failed
                and item.failure_stage == "prospective_parameter_draw"
                for item in selected
            )
            if parameter_failed:
                if any(item.generator_log_offsets for item in selected):
                    raise ValueError("failed parameter draw cannot retain offsets")
            else:
                offsets = {item.generator_log_offsets for item in selected}
                if len(offsets) != 1:
                    raise ValueError("physical draw must be shared across actions")
                values = next(iter(offsets))
                if len(values) != len(spec.names) or any(
                    value < lower or value > upper
                    for value, (lower, upper) in zip(values, spec.log_bounds)
                ):
                    raise ValueError("prospective physical draw is outside fit bounds")

            face = next(
                item
                for item in selected
                if item.policy_name == FIXED_FACE_TEMPERATURE
            )
            if parameter_failed or face.failure_stage == "prospective_probe_draw":
                if face.face_probe_log_offsets is not None:
                    raise ValueError("failed probe draw cannot retain probe offsets")
            else:
                probe = face.face_probe_log_offsets
                if probe is None:
                    raise ValueError("face action must retain its probe draw")
                probe_bounds = tuple(
                    (math.log(lower / nominal), math.log(upper / nominal))
                    for nominal, (lower, upper) in (
                        (
                            physical_config.face_sensor_capacitance_nominal,
                            physical_config.face_sensor_capacitance_bounds,
                        ),
                        (
                            physical_config.face_sensor_response_nominal,
                            physical_config.face_sensor_response_bounds,
                        ),
                    )
                )
                if any(
                    value < lower or value > upper
                    for value, (lower, upper) in zip(probe, probe_bounds)
                ):
                    raise ValueError("prospective probe draw is outside prior bounds")


def _validate_stream_manifest(
    uses: Sequence[ProspectiveStreamUse],
    draws: Sequence[ProspectiveDrawOutcome],
    evidence: ProspectiveAcquisitionEvidence,
    config: ProspectiveUncertaintyConfig,
) -> None:
    if not uses:
        raise ValueError("scored prospective result needs random-stream uses")
    source_models = set(evidence.snapshot.admissible_candidate_models)
    namespaces = {
        (
            use.stream.key.protocol_version,
            use.stream.key.campaign,
            use.stream.key.partition,
            use.stream.key.block,
            use.stream.key.acquisition_evidence_digest,
        )
        for use in uses
    }
    if len(namespaces) != 1:
        raise ValueError("prospective stream uses must share one namespace")
    namespace = next(iter(namespaces))
    if namespace[-1] != evidence.evidence_digest:
        raise ValueError("prospective stream namespace does not match acquisition")
    if any(
        use.stream.key.generator_model not in source_models
        or use.stream.key.draw_index >= config.draw_count
        for use in uses
    ):
        raise ValueError("prospective stream use lies outside the draw matrix")

    policies = {
        policy.name: policy
        for policy in default_fixed_policies()
        if policy.name in PROSPECTIVE_ACQUISITION_ACTIONS
    }
    draw_by_id = {
        (draw.generator_model, draw.draw_index, draw.policy_name): draw
        for draw in draws
    }
    for generator_model in sorted(source_models):
        for draw_index in range(config.draw_count):
            selected_uses = tuple(
                use
                for use in uses
                if use.stream.key.generator_model == generator_model
                and use.stream.key.draw_index == draw_index
            )
            parameter_uses = tuple(
                use
                for use in selected_uses
                if use.stream.key.purpose == PARAMETER_DRAW
            )
            if (
                len(parameter_uses) != len(PROSPECTIVE_ACQUISITION_ACTIONS)
                or {use.shared_for_action for use in parameter_uses}
                != set(PROSPECTIVE_ACQUISITION_ACTIONS)
                or len({use.stream.key for use in parameter_uses}) != 1
            ):
                raise ValueError("prospective parameter stream sharing is incomplete")

            action_draws = tuple(
                draw_by_id[(generator_model, draw_index, policy_name)]
                for policy_name in PROSPECTIVE_ACQUISITION_ACTIONS
            )
            parameter_failed = all(
                draw.failed
                and draw.failure_stage == "prospective_parameter_draw"
                for draw in action_draws
            )
            probe_uses = tuple(
                use
                for use in selected_uses
                if use.stream.key.purpose == PROBE_DRAW
            )
            if len(probe_uses) != (0 if parameter_failed else 1):
                raise ValueError("prospective probe stream inventory is incomplete")

            observation_uses = tuple(
                use
                for use in selected_uses
                if use.stream.key.purpose in (RUN_BIAS, WHITE_NOISE)
            )
            for policy_name in PROSPECTIVE_ACQUISITION_ACTIONS:
                draw = draw_by_id[(generator_model, draw_index, policy_name)]
                actual = {
                    (
                        use.stream.key.purpose,
                        use.stream.key.action,
                        use.stream.key.run,
                        use.stream.key.channel,
                    )
                    for use in observation_uses
                    if use.stream.key.action == policy_name
                }
                expected = {
                    (purpose, policy_name, regime.name, channel)
                    for regime in policies[policy_name].additional_regimes
                    for channel in regime.channels
                    for purpose in (RUN_BIAS, WHITE_NOISE)
                }
                if draw.failure_stage in (
                    "prospective_parameter_draw",
                    "prospective_probe_draw",
                ):
                    valid = not actual
                elif draw.failure_stage == "prospective_simulation":
                    valid = actual.issubset(expected)
                else:
                    valid = actual == expected
                if not valid:
                    raise ValueError(
                        "prospective observation stream inventory is incomplete"
                    )


@dataclass(frozen=True)
class ProspectiveUncertaintyResult:
    """Complete cost-free Step-2 evidence for one common acquisition."""

    config: ProspectiveUncertaintyConfig
    physical_config: OperatingDecisionRealismConfig
    acquisition_evidence: ProspectiveAcquisitionEvidence
    protocol_digest: str
    action_uncertainties: Tuple[ProspectiveActionUncertainty, ...]
    draw_outcomes: Tuple[ProspectiveDrawOutcome, ...]
    stream_uses: Tuple[ProspectiveStreamUse, ...]
    stream_audit: ProspectiveRandomStreamAudit
    result_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.config, ProspectiveUncertaintyConfig):
            raise ValueError("prospective result needs an uncertainty configuration")
        if not isinstance(self.physical_config, OperatingDecisionRealismConfig):
            raise ValueError("prospective result needs a physical configuration")
        normalized_physical_config = _normalized_physical_config(
            self.physical_config
        )
        object.__setattr__(self, "physical_config", normalized_physical_config)
        if not isinstance(self.acquisition_evidence, ProspectiveAcquisitionEvidence):
            raise ValueError("prospective result needs acquisition evidence")
        if (
            corrected_physical_protocol_digest(normalized_physical_config)
            != self.acquisition_evidence.physical_protocol_digest
        ):
            raise ValueError("prospective result physical protocol is inconsistent")
        actions = tuple(self.action_uncertainties)
        draws = tuple(self.draw_outcomes)
        uses = tuple(self.stream_uses)
        object.__setattr__(self, "action_uncertainties", actions)
        object.__setattr__(self, "draw_outcomes", draws)
        object.__setattr__(self, "stream_uses", uses)
        if tuple(item.policy_name for item in actions) != POLICY_NAMES:
            raise ValueError("uncertainty results must use canonical action order")
        draw_ids = tuple(
            (item.generator_model, item.policy_name, item.draw_index) for item in draws
        )
        if draw_ids != tuple(sorted(draw_ids)) or len(set(draw_ids)) != len(draw_ids):
            raise ValueError("prospective draw outcomes must be sorted and unique")
        envelope = self.acquisition_evidence.snapshot.provisional_margin_envelope
        if envelope is None:
            raise ValueError("a scored prospective result needs a provisional envelope")
        baseline_width = envelope.upper - envelope.lower
        expected_actions = _summarize_uncertainty_draws(
            draws,
            baseline_width,
            self.acquisition_evidence.snapshot.admissible_candidate_models,
            self.config,
        )
        for draw in draws:
            _validate_draw_against_acquisition(
                draw,
                self.acquisition_evidence.snapshot,
                baseline_width,
            )
        _validate_draw_parameters(
            draws,
            self.acquisition_evidence,
            normalized_physical_config,
            self.config,
        )
        if actions != expected_actions:
            raise ValueError("prospective action summaries do not match their draws")
        if self.stream_audit != _audit_stream_uses(uses):
            raise ValueError("prospective random-stream audit does not match its uses")
        if not self.stream_audit.clean:
            raise ValueError("prospective random-stream audit must be clean")
        _validate_stream_manifest(
            uses,
            draws,
            self.acquisition_evidence,
            self.config,
        )
        _sha256("prospective uncertainty protocol digest", self.protocol_digest)
        expected_protocol_digest = (
            _prospective_uncertainty_protocol_digest_from_physical_digest(
                self.acquisition_evidence.physical_protocol_digest,
                self.config,
            )
        )
        if self.protocol_digest != expected_protocol_digest:
            raise ValueError("prospective uncertainty protocol digest is invalid")
        _sha256("prospective uncertainty result digest", self.result_digest)
        expected_result_digest = _result_digest(
            config=self.config,
            physical_config=normalized_physical_config,
            acquisition_evidence_digest=self.acquisition_evidence.evidence_digest,
            protocol_digest=self.protocol_digest,
            actions=actions,
            draws=draws,
            stream_uses=uses,
            stream_audit=self.stream_audit,
        )
        if self.result_digest != expected_result_digest:
            raise ValueError("prospective uncertainty result digest is invalid")


def _canonical_digest(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_physical_config(
    physical_config: OperatingDecisionRealismConfig,
) -> OperatingDecisionRealismConfig:
    return replace(
        physical_config,
        sensor=replace(physical_config.sensor, trial_count=1, first_seed=0),
    )


def _physical_config_payload(
    physical_config: OperatingDecisionRealismConfig,
) -> dict:
    normalized = _normalized_physical_config(physical_config)
    return {
        "type": (
            f"{type(normalized).__module__}.{type(normalized).__qualname__}"
        ),
        "values": asdict(normalized),
    }


def _current_payload(regime: OperatingRegime) -> dict:
    return {
        "name": regime.name,
        "phase": regime.phase,
        "transition_times": [float(item) for item in regime.current.transition_times],
        "current_values": [float(item) for item in regime.current.values],
        "channels": list(regime.channels),
    }


def _run_payload(run: RealisticOperatingRun) -> dict:
    return {
        "regime": _current_payload(run.regime),
        "temporary_face_sensor": run.instrumentation.temporary_face_sensor,
        "observations": [
            {
                "channel": item.channel,
                "time": float(item.time),
                "value": float(item.value),
            }
            for item in run.observations.values
        ],
    }


def _interval_payload(interval: Optional[MarginInterval]) -> Optional[dict]:
    if interval is None:
        return None
    return {
        "model_name": interval.model_name,
        "estimate": interval.estimate,
        "local_standard_error": interval.local_standard_error,
        "lower": interval.lower,
        "upper": interval.upper,
    }


def _fit_payload(fit: RealisticCandidateFit) -> dict:
    return {
        "model_name": fit.model_name,
        "log_multipliers": list(fit.log_multipliers),
        "parameter_names": list(fit.parameter_names),
        "physical_values": list(fit.physical_values),
        "interface_mass": fit.interface_mass,
        "series_resistance": fit.series_resistance,
        "face_sensor": (
            None
            if fit.face_sensor is None
            else {
                "thermal_capacitance": fit.face_sensor.thermal_capacitance,
                "response_time_constant": fit.face_sensor.response_time_constant,
            }
        ),
        "objective": fit.objective,
        "covariance": [list(row) for row in fit.covariance],
        "reached_bound": fit.reached_bound,
        "evaluation_count": fit.evaluation_count,
        "converged": fit.converged,
        "termination_reason": fit.termination_reason,
        "completed_iterations": fit.completed_iterations,
        "accepted_iterations": fit.accepted_iterations,
        "scaled_gradient_infinity_norm": fit.scaled_gradient_infinity_norm,
        "last_step_infinity_norm": fit.last_step_infinity_norm,
        "last_relative_objective_reduction": (
            fit.last_relative_objective_reduction
        ),
    }


def _fit_set_payload(fit_set: RealisticAcquisitionFitSet) -> dict:
    return {
        "fits": [
            _fit_payload(fit) for fit in sorted(fit_set.fits, key=lambda item: item.model_name)
        ],
        "failures": [
            {
                "model_name": failure.model_name,
                "stage": failure.stage,
                "error_type": failure.error_type,
            }
            for failure in sorted(
                fit_set.failures,
                key=lambda item: (item.model_name, item.stage, item.error_type),
            )
        ],
    }


def _snapshot_payload(snapshot: ProspectiveAcquisitionSnapshot) -> dict:
    envelope = snapshot.provisional_margin_envelope
    return {
        "candidate_models": list(snapshot.candidate_models),
        "admissible_candidate_models": list(snapshot.admissible_candidate_models),
        "excluded_candidates": [
            {"model_name": item.model_name, "reason": item.reason}
            for item in snapshot.excluded_candidates
        ],
        "failed_candidate_models": list(snapshot.failed_candidate_models),
        "model_intervals": [_interval_payload(item) for item in snapshot.model_intervals],
        "provisional_margin_envelope": (
            None
            if envelope is None
            else {"lower": envelope.lower, "upper": envelope.upper}
        ),
        "selection_failure_reason": snapshot.selection_failure_reason,
    }


def _uncertainty_config_payload(config: ProspectiveUncertaintyConfig) -> dict:
    return {
        "schema_version": PROSPECTIVE_UNCERTAINTY_SCHEMA_VERSION,
        "protocol_version": PROSPECTIVE_UNCERTAINTY_PROTOCOL_VERSION,
        "estimator": config.estimator,
        "uncertainty_metric": config.uncertainty_metric,
        "draw_count": config.draw_count,
        "max_parameter_draw_attempts": config.max_parameter_draw_attempts,
        "max_unstable_draws_per_source_action": (
            config.max_unstable_draws_per_source_action
        ),
        "eligibility_protocol": PROSPECTIVE_ELIGIBILITY_PROTOCOL,
        "within_generator_aggregation": config.within_generator_aggregation,
        "across_generator_aggregation": config.across_generator_aggregation,
        "draw_failure_policy": config.draw_failure_policy,
        "candidate_attrition_policy": config.candidate_attrition_policy,
        "refit_start_protocol": config.refit_start_protocol,
    }


def _policy_payload(policy: FixedPolicy) -> dict:
    return {
        "name": policy.name,
        "additional_regimes": [
            _current_payload(regime) for regime in policy.additional_regimes
        ],
        "extra_sensor_count": policy.extra_sensor_count,
    }


def prospective_uncertainty_protocol_digest(
    physical_config: OperatingDecisionRealismConfig,
    config: ProspectiveUncertaintyConfig = ProspectiveUncertaintyConfig(),
) -> str:
    """Bind the complete Step-2 estimator before it generates evidence."""

    if not isinstance(physical_config, OperatingDecisionRealismConfig):
        raise ValueError("prospective uncertainty needs a realism configuration")
    if not isinstance(config, ProspectiveUncertaintyConfig):
        raise ValueError("prospective uncertainty needs a scoring configuration")
    return _prospective_uncertainty_protocol_digest_from_physical_digest(
        corrected_physical_protocol_digest(physical_config),
        config,
    )


def _prospective_uncertainty_protocol_digest_from_physical_digest(
    physical_protocol_digest: str,
    config: ProspectiveUncertaintyConfig,
) -> str:
    _sha256("physical protocol digest", physical_protocol_digest)
    return _canonical_digest(
        {
            "domain": "thermotwin.prospective_uncertainty_protocol",
            "config": _uncertainty_config_payload(config),
            "action_catalog_digest": prospective_action_catalog_digest(),
            "physical_protocol_digest": physical_protocol_digest,
            "random_stream_protocol": PROSPECTIVE_RANDOM_STREAM_PROTOCOL_VERSION,
            "actions": [_policy_payload(item) for item in default_fixed_policies()],
            "candidate_models": list(MODEL_NAMES),
            "candidate_exclusion": "bound_or_nonconverged_individual_exclusion",
            "new_candidate_attrition": (
                "max_actual_or_baseline_candidate_exclusion_retained_as_usable"
            ),
            "failed_draw": "raw_baseline_width_imputation_retained_denominator",
        }
    )


def _validate_common_initial_run(
    run: RealisticOperatingRun,
    physical_config: OperatingDecisionRealismConfig,
) -> None:
    if not isinstance(run, RealisticOperatingRun):
        raise ValueError("prospective scoring needs one realistic initial run")
    if run.regime != initial_acquisition_regime():
        raise ValueError("prospective scoring needs the exact common initial regime")
    if run.instrumentation != RunInstrumentation(False):
        raise ValueError("the common initial run cannot carry the temporary probe")
    sample_times = regular_measurement_times(
        RUN_DURATION_SECONDS,
        physical_config.sensor.sampling_interval,
    )
    expected_pairs = tuple(
        (channel, time)
        for time in sample_times
        for channel in (COLD_EXCHANGER, HOT_EXCHANGER)
    )
    observed_pairs = tuple(
        (item.channel, item.time) for item in run.observations.values
    )
    if observed_pairs != expected_pairs:
        raise ValueError("common initial observations need the exact regular grid")
    if any(not math.isfinite(item.value) for item in run.observations.values):
        raise ValueError("common initial observations must be finite")


def _validate_final_regime(
    final_regime: OperatingRegime,
    physical_config: OperatingDecisionRealismConfig,
) -> None:
    if (
        not isinstance(final_regime, OperatingRegime)
        or final_regime.name != _FINAL_REGIME_NAME
        or final_regime.phase != FINAL_EVALUATION
        or final_regime.channels
        or final_regime.current != physical_config.final_current
    ):
        raise ValueError("prospective scoring needs the frozen untouched final regime")


def _multistart_fit_candidates(
    runs: Sequence[RealisticOperatingRun],
    physical_config: OperatingDecisionRealismConfig,
) -> RealisticAcquisitionFitSet:
    runs = tuple(runs)
    if not runs:
        raise ValueError("prospective fitting needs acquisition runs")
    instrumented = any(run.instrumentation.temporary_face_sensor for run in runs)
    fits = []
    failures = []
    for model_name in MODEL_NAMES:
        spec = _parameter_spec(model_name, instrumented, physical_config)
        candidates = []
        last_error: Optional[BaseException] = None
        for start in _fit_start_vectors(spec):
            try:
                candidates.append(
                    fit_realistic_candidate(
                        model_name,
                        runs,
                        physical_config,
                        initial_log_multipliers=start,
                    )
                )
            except (ArithmeticError, IntegrationDivergenceError, ValueError) as error:
                last_error = error
        if candidates:
            fits.append(min(candidates, key=lambda item: item.objective))
        else:
            failures.append(
                NumericalFailure(
                    model_name,
                    "acquisition_fit",
                    type(last_error).__name__ if last_error is not None else "unknown",
                )
            )
    return RealisticAcquisitionFitSet(
        tuple(sorted(fits, key=lambda item: item.model_name)),
        tuple(sorted(failures, key=lambda item: item.model_name)),
    )


def _acquisition_evidence_digest(
    common_initial_run: RealisticOperatingRun,
    fit_set: RealisticAcquisitionFitSet,
    snapshot: ProspectiveAcquisitionSnapshot,
    final_regime: OperatingRegime,
    physical_config: OperatingDecisionRealismConfig,
) -> str:
    return _canonical_digest(
        {
            "domain": _ACQUISITION_EVIDENCE_DOMAIN,
            "common_initial_run": _run_payload(common_initial_run),
            "fit_set": _fit_set_payload(fit_set),
            "snapshot": _snapshot_payload(snapshot),
            "final_regime": _current_payload(final_regime),
            "physical_protocol_digest": corrected_physical_protocol_digest(
                physical_config
            ),
        }
    )


def prepare_prospective_acquisition_evidence(
    common_initial_run: RealisticOperatingRun,
    final_regime: OperatingRegime,
    physical_config: OperatingDecisionRealismConfig,
) -> ProspectiveAcquisitionEvidence:
    """Fit the exact common acquisition and authenticate the Step-1 snapshot."""

    if not isinstance(physical_config, OperatingDecisionRealismConfig):
        raise ValueError("prospective scoring needs a realism configuration")
    _validate_common_initial_run(common_initial_run, physical_config)
    _validate_final_regime(final_regime, physical_config)
    fit_set = _multistart_fit_candidates((common_initial_run,), physical_config)
    snapshot = build_prospective_acquisition_snapshot(
        fit_set,
        final_regime,
        physical_config,
    )
    physical_digest = corrected_physical_protocol_digest(physical_config)
    evidence_digest = _acquisition_evidence_digest(
        common_initial_run,
        fit_set,
        snapshot,
        final_regime,
        physical_config,
    )
    return ProspectiveAcquisitionEvidence(
        common_initial_run=common_initial_run,
        fit_set=fit_set,
        snapshot=snapshot,
        final_regime=final_regime,
        physical_protocol_digest=physical_digest,
        evidence_digest=evidence_digest,
        _construction_seal=_EVIDENCE_CONSTRUCTION_SEAL,
    )


def _validate_acquisition_evidence(
    evidence: ProspectiveAcquisitionEvidence,
    physical_config: OperatingDecisionRealismConfig,
) -> None:
    if not isinstance(evidence, ProspectiveAcquisitionEvidence):
        raise ValueError("prospective uncertainty needs acquisition evidence")
    _validate_common_initial_run(evidence.common_initial_run, physical_config)
    _validate_final_regime(evidence.final_regime, physical_config)
    physical_digest = corrected_physical_protocol_digest(physical_config)
    if evidence.physical_protocol_digest != physical_digest:
        raise ValueError("acquisition evidence physical protocol has changed")
    reproduced_fit_set = _multistart_fit_candidates(
        (evidence.common_initial_run,),
        physical_config,
    )
    if reproduced_fit_set != evidence.fit_set:
        raise ValueError("acquisition fit set does not reproduce from the common run")
    rebuilt = build_prospective_acquisition_snapshot(
        evidence.fit_set,
        evidence.final_regime,
        physical_config,
    )
    if rebuilt != evidence.snapshot:
        raise ValueError("acquisition evidence snapshot does not match its fits")
    expected_digest = _acquisition_evidence_digest(
        evidence.common_initial_run,
        evidence.fit_set,
        evidence.snapshot,
        evidence.final_regime,
        physical_config,
    )
    if evidence.evidence_digest != expected_digest:
        raise ValueError("acquisition evidence digest is invalid")


def prospective_acquisition_evidence_payload(
    evidence: ProspectiveAcquisitionEvidence,
    physical_config: OperatingDecisionRealismConfig,
) -> dict:
    """Return a fully reauthenticated, strict JSON-ready acquisition record."""

    if not isinstance(physical_config, OperatingDecisionRealismConfig):
        raise ValueError("prospective evidence payload needs a physical configuration")
    _validate_acquisition_evidence(evidence, physical_config)
    return {
        "schema_version": PROSPECTIVE_UNCERTAINTY_SCHEMA_VERSION,
        "protocol_version": PROSPECTIVE_UNCERTAINTY_PROTOCOL_VERSION,
        "physical_config": _physical_config_payload(physical_config),
        "physical_protocol_digest": evidence.physical_protocol_digest,
        "evidence_digest": evidence.evidence_digest,
        "common_initial_run": _run_payload(evidence.common_initial_run),
        "fit_set": _fit_set_payload(evidence.fit_set),
        "snapshot": _snapshot_payload(evidence.snapshot),
        "final_regime": _current_payload(evidence.final_regime),
    }


def _positive_semidefinite_cholesky(
    covariance: Sequence[Sequence[float]],
) -> Tuple[Tuple[float, ...], ...]:
    matrix = tuple(tuple(float(value) for value in row) for row in covariance)
    size = len(matrix)
    if not matrix or any(len(row) != size for row in matrix):
        raise ValueError("prospective covariance must be square and nonempty")
    if any(not math.isfinite(value) for row in matrix for value in row):
        raise ValueError("prospective covariance must be finite")
    scale = max(abs(value) for row in matrix for value in row)
    tolerance = max(1.0e-15, 1.0e-10 * scale)
    for row in range(size):
        for column in range(row):
            if abs(matrix[row][column] - matrix[column][row]) > tolerance:
                raise ValueError("prospective covariance must be symmetric")

    lower = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            residual = matrix[row][column] - sum(
                lower[row][index] * lower[column][index]
                for index in range(column)
            )
            if row == column:
                if residual < -tolerance:
                    raise ValueError("prospective covariance must be positive semidefinite")
                lower[row][column] = math.sqrt(max(0.0, residual))
            elif lower[column][column] > 0.0:
                lower[row][column] = residual / lower[column][column]
            elif abs(residual) > tolerance:
                raise ValueError("prospective covariance is structurally singular")
    return tuple(tuple(row) for row in lower)


def _validate_generator_fit(
    fit: RealisticCandidateFit,
    physical_config: OperatingDecisionRealismConfig,
) -> Tuple[object, Tuple[Tuple[float, ...], ...]]:
    if fit.model_name not in MODEL_NAMES:
        raise ValueError("prospective generator fit has an unknown model")
    if fit.reached_bound or not fit.converged:
        raise ValueError("prospective generator must be initially admissible")
    spec = _parameter_spec(fit.model_name, False, physical_config)
    if fit.parameter_names != spec.names or len(fit.log_multipliers) != len(spec.names):
        raise ValueError("prospective generator parameters do not match its model")
    if any(
        value < lower or value > upper
        for value, (lower, upper) in zip(fit.log_multipliers, spec.log_bounds)
    ):
        raise ValueError("prospective generator lies outside frozen fit bounds")
    decoded = _decoded_parameters(fit.model_name, fit.log_multipliers, spec)
    if decoded != (
        fit.physical_values,
        fit.interface_mass,
        fit.series_resistance,
        fit.face_sensor,
    ):
        raise ValueError("prospective generator values do not decode from offsets")
    cholesky = _positive_semidefinite_cholesky(fit.covariance)
    if len(cholesky) != len(spec.names):
        raise ValueError("prospective generator covariance has the wrong dimension")
    return spec, cholesky


def _bounded_gaussian_draw(
    mean: Sequence[float],
    cholesky: Sequence[Sequence[float]],
    bounds: Sequence[Tuple[float, float]],
    stream: ProspectiveRandomStream,
    max_attempts: int,
) -> Tuple[float, ...]:
    center = tuple(float(value) for value in mean)
    lower = tuple(tuple(float(value) for value in row) for row in cholesky)
    limits = tuple((float(left), float(right)) for left, right in bounds)
    size = len(center)
    if (
        len(lower) != size
        or any(len(row) != size for row in lower)
        or len(limits) != size
    ):
        raise ValueError("bounded Gaussian inputs have inconsistent dimensions")
    generator = stream.new_generator()
    for _ in range(max_attempts):
        standard = tuple(generator.gauss(0.0, 1.0) for _ in range(size))
        candidate = tuple(
            center[row]
            + sum(lower[row][column] * standard[column] for column in range(size))
            for row in range(size)
        )
        if all(
            left <= value <= right
            for value, (left, right) in zip(candidate, limits)
        ):
            return candidate
    raise _ProspectiveDrawSamplingError(
        "bounded prospective parameter sampler exhausted its attempts"
    )


def _generator_parameter_draw(
    fit: RealisticCandidateFit,
    namespace: ProspectiveRandomStreamNamespace,
    draw_index: int,
    physical_config: OperatingDecisionRealismConfig,
    uncertainty_config: ProspectiveUncertaintyConfig,
    registry: ProspectiveRandomStreamRegistry,
) -> Tuple[float, ...]:
    spec, cholesky = _validate_generator_fit(fit, physical_config)
    stream = prospective_parameter_stream(
        namespace,
        generator_model=fit.model_name,
        draw_index=draw_index,
    )
    for action in PROSPECTIVE_ACQUISITION_ACTIONS:
        registry.register(
            stream,
            consumer=f"{fit.model_name}/{draw_index}/{action}/physical_parameters",
            shared_for_action=action,
        )
    return _bounded_gaussian_draw(
        fit.log_multipliers,
        cholesky,
        spec.log_bounds,  # type: ignore[attr-defined]
        stream,
        uncertainty_config.max_parameter_draw_attempts,
    )


def _face_probe_draw(
    namespace: ProspectiveRandomStreamNamespace,
    generator_model: str,
    draw_index: int,
    physical_config: OperatingDecisionRealismConfig,
    uncertainty_config: ProspectiveUncertaintyConfig,
    registry: ProspectiveRandomStreamRegistry,
) -> Tuple[float, float]:
    stream = prospective_probe_stream(
        namespace,
        generator_model=generator_model,
        draw_index=draw_index,
    )
    registry.register(
        stream,
        consumer=f"{generator_model}/{draw_index}/{FIXED_FACE_TEMPERATURE}/probe",
    )
    standard_deviation = physical_config.face_sensor_prior_log_standard_deviation
    bounds = tuple(
        (
            math.log(lower / nominal),
            math.log(upper / nominal),
        )
        for nominal, (lower, upper) in (
            (
                physical_config.face_sensor_capacitance_nominal,
                physical_config.face_sensor_capacitance_bounds,
            ),
            (
                physical_config.face_sensor_response_nominal,
                physical_config.face_sensor_response_bounds,
            ),
        )
    )
    draw = _bounded_gaussian_draw(
        (0.0, 0.0),
        ((standard_deviation, 0.0), (0.0, standard_deviation)),
        bounds,
        stream,
        uncertainty_config.max_parameter_draw_attempts,
    )
    return draw  # type: ignore[return-value]


def _synthetic_observed_run(
    *,
    policy_name: str,
    generator_model: str,
    draw_index: int,
    regime: OperatingRegime,
    prediction,
    namespace: ProspectiveRandomStreamNamespace,
    physical_config: OperatingDecisionRealismConfig,
    registry: ProspectiveRandomStreamRegistry,
) -> RealisticOperatingRun:
    noise_by_channel = dict(physical_config.sensor.channel_noise)
    values = []
    for channel in regime.channels:
        selected = tuple(
            item for item in prediction.values if item.channel == channel
        )
        bias_stream = prospective_observation_stream(
            namespace,
            generator_model=generator_model,
            draw_index=draw_index,
            purpose=RUN_BIAS,
            action=policy_name,
            run=regime.name,
            channel=channel,
        )
        noise_stream = prospective_observation_stream(
            namespace,
            generator_model=generator_model,
            draw_index=draw_index,
            purpose=WHITE_NOISE,
            action=policy_name,
            run=regime.name,
            channel=channel,
        )
        registry.register(
            bias_stream,
            consumer=(
                f"{generator_model}/{draw_index}/{policy_name}/"
                f"{regime.name}/{channel}/run_bias"
            ),
        )
        registry.register(
            noise_stream,
            consumer=(
                f"{generator_model}/{draw_index}/{policy_name}/"
                f"{regime.name}/{channel}/white_noise"
            ),
        )
        noise = noise_by_channel[channel]
        bias = bias_stream.new_generator().gauss(
            0.0,
            physical_config.sensor.run_bias_noise_ratio * noise,
        )
        noise_generator = noise_stream.new_generator()
        values.extend(
            ObservableValue(
                item.channel,
                item.time,
                item.value + bias + noise_generator.gauss(0.0, noise),
            )
            for item in selected
        )
    values.sort(key=lambda item: (item.time, ALL_CHANNELS.index(item.channel)))
    observations = ObservableRun(regime.name, regime.current, tuple(values))
    return RealisticOperatingRun(
        regime,
        observations,
        _instrumentation_for_regime(regime),
    )


def _simulate_action_runs(
    policy: FixedPolicy,
    generator_fit: RealisticCandidateFit,
    generator_offsets: Sequence[float],
    face_probe_offsets: Optional[Tuple[float, float]],
    draw_index: int,
    namespace: ProspectiveRandomStreamNamespace,
    physical_config: OperatingDecisionRealismConfig,
    registry: ProspectiveRandomStreamRegistry,
) -> Tuple[RealisticOperatingRun, ...]:
    base_spec = _parameter_spec(generator_fit.model_name, False, physical_config)
    physical, interface_mass, series_resistance, no_probe = _decoded_parameters(
        generator_fit.model_name,
        generator_offsets,
        base_spec,
    )
    if no_probe is not None:
        raise ValueError("common-acquisition generator cannot contain a face probe")
    face_sensor = None
    if policy.name == FIXED_FACE_TEMPERATURE:
        if face_probe_offsets is None:
            raise ValueError("face action needs prospective probe parameters")
        face_sensor = TemporaryFaceSensor(
            physical_config.face_sensor_capacitance_nominal
            * math.exp(face_probe_offsets[0]),
            physical_config.face_sensor_response_nominal
            * math.exp(face_probe_offsets[1]),
        )
    elif face_probe_offsets is not None:
        raise ValueError("only the face action may receive probe parameters")

    runs = []
    for regime in policy.additional_regimes:
        prediction = _simulate_realistic_observables(
            generator_fit.model_name,
            regime.current,
            regime.channels,
            physical,
            interface_mass,
            series_resistance,
            face_sensor if COLD_FACE in regime.channels else None,
            physical_config,
        )
        runs.append(
            _synthetic_observed_run(
                policy_name=policy.name,
                generator_model=generator_fit.model_name,
                draw_index=draw_index,
                regime=regime,
                prediction=prediction,
                namespace=namespace,
                physical_config=physical_config,
                registry=registry,
            )
        )
    return tuple(runs)


def _synthetic_observation_digest(
    runs: Sequence[RealisticOperatingRun],
) -> str:
    return _canonical_digest(
        {
            "domain": "thermotwin.prospective_synthetic_observations",
            "runs": [_run_payload(run) for run in runs],
        }
    )


def _failed_draw(
    *,
    policy_name: str,
    generator_model: str,
    draw_index: int,
    generator_offsets: Sequence[float],
    face_probe_offsets: Optional[Tuple[float, float]],
    baseline_width: float,
    stage: str,
    error_type: str,
    failure_model: Optional[str] = None,
    candidate_outcomes: Sequence[ProspectiveCandidateOutcome] = (),
    observation_digest: Optional[str] = None,
) -> ProspectiveDrawOutcome:
    return ProspectiveDrawOutcome(
        policy_name=policy_name,
        generator_model=generator_model,
        draw_index=draw_index,
        generator_log_offsets=tuple(generator_offsets),
        face_probe_log_offsets=face_probe_offsets,
        candidate_outcomes=tuple(
            sorted(candidate_outcomes, key=lambda item: item.model_name)
        ),
        initially_admissible_became_inadmissible=(),
        initially_excluded_became_admissible=(),
        stable=False,
        failed=True,
        failure_stage=stage,
        failure_model=failure_model,
        failure_type=error_type,
        raw_after_width=None,
        scored_after_width=baseline_width,
        synthetic_observation_digest=observation_digest,
    )


def _evaluate_hypothetical_draw(
    *,
    policy: FixedPolicy,
    generator_fit: RealisticCandidateFit,
    generator_offsets: Sequence[float],
    face_probe_offsets: Optional[Tuple[float, float]],
    draw_index: int,
    evidence: ProspectiveAcquisitionEvidence,
    namespace: ProspectiveRandomStreamNamespace,
    physical_config: OperatingDecisionRealismConfig,
    registry: ProspectiveRandomStreamRegistry,
    baseline_width: float,
) -> ProspectiveDrawOutcome:
    try:
        synthetic_runs = _simulate_action_runs(
            policy,
            generator_fit,
            generator_offsets,
            face_probe_offsets,
            draw_index,
            namespace,
            physical_config,
            registry,
        )
    except (ArithmeticError, IntegrationDivergenceError, ValueError) as error:
        return _failed_draw(
            policy_name=policy.name,
            generator_model=generator_fit.model_name,
            draw_index=draw_index,
            generator_offsets=generator_offsets,
            face_probe_offsets=face_probe_offsets,
            baseline_width=baseline_width,
            stage="prospective_simulation",
            error_type=type(error).__name__,
        )

    observation_digest = _synthetic_observation_digest(synthetic_runs)
    refit_set = _multistart_fit_candidates(
        (evidence.common_initial_run, *synthetic_runs),
        physical_config,
    )
    outcomes = []
    failures_by_model = {item.model_name: item for item in refit_set.failures}
    fits_by_model = {item.model_name: item for item in refit_set.fits}
    intervals = []
    forecast_failures: dict[str, BaseException] = {}
    for model_name in MODEL_NAMES:
        if model_name in failures_by_model:
            outcomes.append(
                ProspectiveCandidateOutcome(
                    model_name,
                    "acquisition_fit_failure",
                    None,
                    None,
                )
            )
            continue
        fit = fits_by_model[model_name]
        if fit.reached_bound:
            outcomes.append(
                ProspectiveCandidateOutcome(
                    model_name,
                    "fit_reached_bound",
                    fit.objective,
                    None,
                )
            )
            continue
        if not fit.converged:
            outcomes.append(
                ProspectiveCandidateOutcome(
                    model_name,
                    "optimizer_not_converged",
                    fit.objective,
                    None,
                )
            )
            continue
        try:
            interval = forecast_realistic_margin_interval(
                fit,
                evidence.final_regime,
                physical_config,
            )
        except (ArithmeticError, IntegrationDivergenceError, ValueError) as error:
            outcomes.append(
                ProspectiveCandidateOutcome(
                    model_name,
                    "uncertainty_failure",
                    fit.objective,
                    None,
                )
            )
            forecast_failures[model_name] = error
            continue
        outcomes.append(
            ProspectiveCandidateOutcome(
                model_name,
                "admissible",
                fit.objective,
                interval,
            )
        )
        intervals.append(interval)

    envelope = envelope_margin_intervals(intervals)
    if envelope is None:
        if failures_by_model:
            failed_model = sorted(failures_by_model)[0]
            failure = failures_by_model[failed_model]
            failure_stage = "prospective_refit"
            failure_type = failure.error_type
        elif forecast_failures:
            failed_model = sorted(forecast_failures)[0]
            failure_stage = "prospective_uncertainty"
            failure_type = type(forecast_failures[failed_model]).__name__
        else:
            failed_model = None
            failure_stage = "prospective_uncertainty"
            failure_type = "NoAdmissibleCandidate"
        return _failed_draw(
            policy_name=policy.name,
            generator_model=generator_fit.model_name,
            draw_index=draw_index,
            generator_offsets=generator_offsets,
            face_probe_offsets=face_probe_offsets,
            baseline_width=baseline_width,
            stage=failure_stage,
            failure_model=failed_model,
            error_type=failure_type,
            candidate_outcomes=outcomes,
            observation_digest=observation_digest,
        )

    initially_admissible = set(evidence.snapshot.admissible_candidate_models)
    initially_excluded = {
        item.model_name for item in evidence.snapshot.excluded_candidates
    }
    now_admissible = {item.model_name for item in intervals}
    newly_excluded = tuple(sorted(initially_admissible - now_admissible))
    recovered = tuple(sorted(initially_excluded.intersection(now_admissible)))
    raw_width = envelope.upper - envelope.lower
    scored_width = max(raw_width, baseline_width) if newly_excluded else raw_width
    return ProspectiveDrawOutcome(
        policy_name=policy.name,
        generator_model=generator_fit.model_name,
        draw_index=draw_index,
        generator_log_offsets=tuple(generator_offsets),
        face_probe_log_offsets=face_probe_offsets,
        candidate_outcomes=tuple(sorted(outcomes, key=lambda item: item.model_name)),
        initially_admissible_became_inadmissible=newly_excluded,
        initially_excluded_became_admissible=recovered,
        stable=True,
        failed=False,
        failure_stage=None,
        failure_model=None,
        failure_type=None,
        raw_after_width=raw_width,
        scored_after_width=scored_width,
        synthetic_observation_digest=observation_digest,
    )


def _summarize_uncertainty_draws(
    draws: Sequence[ProspectiveDrawOutcome],
    baseline_width: float,
    source_models: Sequence[str],
    config: ProspectiveUncertaintyConfig,
) -> Tuple[ProspectiveActionUncertainty, ...]:
    retained = tuple(draws)
    source_models = tuple(sorted(source_models))
    observed_ids = tuple(
        (item.generator_model, item.policy_name, item.draw_index)
        for item in retained
    )
    expected_ids = tuple(
        sorted(
            (generator_model, policy_name, draw_index)
            for generator_model in source_models
            for policy_name in PROSPECTIVE_ACQUISITION_ACTIONS
            for draw_index in range(config.draw_count)
        )
    )
    if observed_ids != expected_ids:
        raise ValueError("prospective draw matrix is incomplete, unordered, or extended")
    actions = [
        ProspectiveActionUncertainty(
            policy_name=STOP_NOW,
            eligible=True,
            failure_reason=None,
            uncertainty_before=baseline_width,
            expected_uncertainty_after=baseline_width,
            prospective_draw_count=0,
            source_summaries=(),
        )
    ]
    for policy_name in PROSPECTIVE_ACQUISITION_ACTIONS:
        summaries = []
        for generator_model in source_models:
            selected = tuple(
                item
                for item in retained
                if item.policy_name == policy_name
                and item.generator_model == generator_model
            )
            if tuple(item.draw_index for item in selected) != tuple(
                range(config.draw_count)
            ):
                raise ValueError("prospective draw matrix is incomplete or unordered")
            stable_count = sum(item.stable for item in selected)
            failed_count = sum(item.failed for item in selected)
            summaries.append(
                ProspectiveModelActionSummary(
                    policy_name=policy_name,
                    generator_model=generator_model,
                    draw_count=len(selected),
                    stable_draw_count=stable_count,
                    failed_draw_count=failed_count,
                    mean_after_width=fmean(
                        item.scored_after_width for item in selected
                    ),
                    eligible=(
                        len(selected) - stable_count
                        <= config.max_unstable_draws_per_source_action
                    ),
                    max_unstable_draws=(
                        config.max_unstable_draws_per_source_action
                    ),
                )
            )
        expected_after = max(item.mean_after_width for item in summaries)
        eligible = all(item.eligible for item in summaries)
        actions.append(
            ProspectiveActionUncertainty(
                policy_name=policy_name,
                eligible=eligible,
                failure_reason=(
                    None
                    if eligible
                    else "insufficient_stable_prospective_draws"
                ),
                uncertainty_before=baseline_width,
                expected_uncertainty_after=expected_after,
                prospective_draw_count=sum(item.draw_count for item in summaries),
                source_summaries=tuple(summaries),
            )
        )
    return tuple(actions)


def _padded_scored_draw_width(
    draw: ProspectiveDrawOutcome,
    *,
    padded_baseline_width: float,
    action_offset: float,
) -> float:
    """Score one authenticated raw draw under development padding."""

    if draw.failed:
        return padded_baseline_width
    if draw.raw_after_width is None:
        raise ValueError("completed prospective draw is missing its raw width")
    padded_width = draw.raw_after_width + 2.0 * action_offset
    if draw.initially_admissible_became_inadmissible:
        return max(padded_baseline_width, padded_width)
    return padded_width


def score_prospective_uncertainty_with_offsets(
    result: ProspectiveUncertaintyResult,
    offsets: ProspectiveDevelopmentOffsets,
) -> Tuple[ProspectivePaddedActionUncertainty, ...]:
    """Apply versioned offsets to reusable raw predictive evidence.

    Failed draws receive exactly the padded stop baseline.  A completed draw
    that loses an initially admissible candidate receives at least that same
    baseline.  Stable completed draws receive the selected action's offset.
    No simulation, fit, or random draw is repeated by this operation.
    """

    if not isinstance(result, ProspectiveUncertaintyResult):
        raise ValueError("padded scoring needs a prospective uncertainty result")
    if not isinstance(offsets, ProspectiveDevelopmentOffsets):
        raise ValueError("padded scoring needs development action offsets")
    raw_by_policy = {
        item.policy_name: item for item in result.action_uncertainties
    }
    raw_baseline = raw_by_policy[STOP_NOW].uncertainty_before
    padded_baseline = raw_baseline + 2.0 * offsets.stop_now
    scored = [
        ProspectivePaddedActionUncertainty(
            raw_action=raw_by_policy[STOP_NOW],
            development_offset=offsets.stop_now,
            padded_uncertainty_before=padded_baseline,
            padded_expected_uncertainty_after=padded_baseline,
            source_summaries=(),
        )
    ]
    for policy_name in PROSPECTIVE_ACQUISITION_ACTIONS:
        raw_action = raw_by_policy[policy_name]
        action_offset = offsets.for_policy(policy_name)
        padded_summaries = []
        for raw_summary in raw_action.source_summaries:
            selected = tuple(
                draw
                for draw in result.draw_outcomes
                if draw.policy_name == policy_name
                and draw.generator_model == raw_summary.generator_model
            )
            if not selected and not result.draw_outcomes:
                if any(
                    offsets.for_policy(policy) != 0.0
                    for policy in POLICY_NAMES
                ):
                    raise ValueError(
                        "nonzero development offsets need retained predictive draws"
                    )
                padded_mean = raw_summary.mean_after_width
            elif len(selected) != raw_summary.draw_count:
                raise ValueError("raw action summary does not match predictive draws")
            else:
                padded_mean = fmean(
                    _padded_scored_draw_width(
                        draw,
                        padded_baseline_width=padded_baseline,
                        action_offset=action_offset,
                    )
                    for draw in selected
                )
            padded_summaries.append(
                ProspectivePaddedModelActionSummary(
                    raw_summary=raw_summary,
                    padded_mean_after_width=padded_mean,
                )
            )
        scored.append(
            ProspectivePaddedActionUncertainty(
                raw_action=raw_action,
                development_offset=action_offset,
                padded_uncertainty_before=padded_baseline,
                padded_expected_uncertainty_after=max(
                    item.padded_mean_after_width for item in padded_summaries
                ),
                source_summaries=tuple(padded_summaries),
            )
        )
    return tuple(scored)


def _candidate_outcome_payload(outcome: ProspectiveCandidateOutcome) -> dict:
    return {
        "model_name": outcome.model_name,
        "status": outcome.status,
        "objective": outcome.objective,
        "interval": _interval_payload(outcome.interval),
    }


def _draw_payload(draw: ProspectiveDrawOutcome) -> dict:
    return {
        "policy_name": draw.policy_name,
        "generator_model": draw.generator_model,
        "draw_index": draw.draw_index,
        "generator_log_offsets": list(draw.generator_log_offsets),
        "face_probe_log_offsets": (
            None
            if draw.face_probe_log_offsets is None
            else list(draw.face_probe_log_offsets)
        ),
        "candidate_outcomes": [
            _candidate_outcome_payload(item) for item in draw.candidate_outcomes
        ],
        "initially_admissible_became_inadmissible": list(
            draw.initially_admissible_became_inadmissible
        ),
        "initially_excluded_became_admissible": list(
            draw.initially_excluded_became_admissible
        ),
        "stable": draw.stable,
        "failed": draw.failed,
        "failure_stage": draw.failure_stage,
        "failure_model": draw.failure_model,
        "failure_type": draw.failure_type,
        "raw_after_width": draw.raw_after_width,
        "scored_after_width": draw.scored_after_width,
        "synthetic_observation_digest": draw.synthetic_observation_digest,
    }


def _action_uncertainty_payload(action: ProspectiveActionUncertainty) -> dict:
    return {
        "policy_name": action.policy_name,
        "eligible": action.eligible,
        "failure_reason": action.failure_reason,
        "uncertainty_before": action.uncertainty_before,
        "expected_uncertainty_after": action.expected_uncertainty_after,
        "expected_uncertainty_reduction": (
            action.expected_uncertainty_reduction
        ),
        "prospective_draw_count": action.prospective_draw_count,
        "source_summaries": [
            {
                "policy_name": item.policy_name,
                "generator_model": item.generator_model,
                "draw_count": item.draw_count,
                "stable_draw_count": item.stable_draw_count,
                "failed_draw_count": item.failed_draw_count,
                "unstable_draw_count": item.unstable_draw_count,
                "max_unstable_draws": item.max_unstable_draws,
                "mean_after_width": item.mean_after_width,
                "eligible": item.eligible,
            }
            for item in action.source_summaries
        ],
    }


def _stream_use_payload(use: ProspectiveStreamUse) -> dict:
    return {
        "key": json.loads(use.stream.key.canonical_bytes().decode("utf-8")),
        "seed": str(use.stream.seed),
        "consumer": use.consumer,
        "shared_for_action": use.shared_for_action,
    }


def _stream_audit_payload(audit: ProspectiveRandomStreamAudit) -> dict:
    return {
        "use_count": audit.use_count,
        "unique_key_count": audit.unique_key_count,
        "unique_seed_count": audit.unique_seed_count,
        "declared_parameter_sharing": [
            {
                "key": json.loads(item.key.canonical_bytes().decode("utf-8")),
                "actions": list(item.actions),
                "consumers": list(item.consumers),
            }
            for item in audit.declared_parameter_sharing
        ],
        "unintended_reuse": [
            {
                "seed": str(item.seed),
                "keys": [
                    json.loads(key.canonical_bytes().decode("utf-8"))
                    for key in item.keys
                ],
                "consumers": list(item.consumers),
                "reason": item.reason,
            }
            for item in audit.unintended_reuse
        ],
    }


def _result_digest(
    *,
    config: ProspectiveUncertaintyConfig,
    physical_config: OperatingDecisionRealismConfig,
    acquisition_evidence_digest: str,
    protocol_digest: str,
    actions: Sequence[ProspectiveActionUncertainty],
    draws: Sequence[ProspectiveDrawOutcome],
    stream_uses: Sequence[ProspectiveStreamUse],
    stream_audit: ProspectiveRandomStreamAudit,
) -> str:
    return _canonical_digest(
        {
            "domain": _RESULT_DIGEST_DOMAIN,
            "config": _uncertainty_config_payload(config),
            "physical_config": _physical_config_payload(physical_config),
            "acquisition_evidence_digest": acquisition_evidence_digest,
            "protocol_digest": protocol_digest,
            "action_uncertainties": [
                _action_uncertainty_payload(item) for item in actions
            ],
            "draw_outcomes": [_draw_payload(item) for item in draws],
            "stream_uses": [_stream_use_payload(item) for item in stream_uses],
            "stream_audit": _stream_audit_payload(stream_audit),
        }
    )


def validate_prospective_uncertainty_result_integrity(
    result: ProspectiveUncertaintyResult,
) -> None:
    """Check in-memory provenance without repeating the acquisition fit.

    This is the inexpensive boundary used when the same immutable result is
    rescored for prefixes, offsets, or costs.  Full archive serialization still
    calls :func:`prospective_uncertainty_result_payload`, which independently
    refits the common acquisition before emitting evidence.
    """

    if not isinstance(result, ProspectiveUncertaintyResult):
        raise ValueError("prospective integrity check needs an uncertainty result")
    evidence = result.acquisition_evidence
    expected_physical = corrected_physical_protocol_digest(result.physical_config)
    if evidence.physical_protocol_digest != expected_physical:
        raise ValueError("prospective result physical protocol is inconsistent")
    expected_evidence = _acquisition_evidence_digest(
        evidence.common_initial_run,
        evidence.fit_set,
        evidence.snapshot,
        evidence.final_regime,
        result.physical_config,
    )
    if evidence.evidence_digest != expected_evidence:
        raise ValueError("prospective acquisition evidence digest is invalid")
    expected_protocol = prospective_uncertainty_protocol_digest(
        result.physical_config,
        result.config,
    )
    if result.protocol_digest != expected_protocol:
        raise ValueError("prospective uncertainty protocol digest is invalid")
    expected_result = _result_digest(
        config=result.config,
        physical_config=result.physical_config,
        acquisition_evidence_digest=evidence.evidence_digest,
        protocol_digest=result.protocol_digest,
        actions=result.action_uncertainties,
        draws=result.draw_outcomes,
        stream_uses=result.stream_uses,
        stream_audit=result.stream_audit,
    )
    if result.result_digest != expected_result:
        raise ValueError("prospective uncertainty result digest is invalid")


def prospective_uncertainty_summary_payload(
    result: ProspectiveUncertaintyResult,
) -> dict:
    """Return a compact authenticated summary without acquisition refitting."""

    validate_prospective_uncertainty_result_integrity(result)
    return {
        "schema_version": PROSPECTIVE_UNCERTAINTY_SCHEMA_VERSION,
        "protocol_version": PROSPECTIVE_UNCERTAINTY_PROTOCOL_VERSION,
        "protocol_digest": result.protocol_digest,
        "result_digest": result.result_digest,
        "physical_protocol_digest": result.acquisition_evidence.physical_protocol_digest,
        "acquisition_evidence_digest": result.acquisition_evidence.evidence_digest,
        "config": _uncertainty_config_payload(result.config),
        "action_uncertainties": [
            _action_uncertainty_payload(item) for item in result.action_uncertainties
        ],
        "stream_audit": _stream_audit_payload(result.stream_audit),
        "retained_draw_outcome_count": len(result.draw_outcomes),
        "retained_stream_use_count": len(result.stream_uses),
    }


def estimate_prospective_action_uncertainty(
    acquisition_evidence: ProspectiveAcquisitionEvidence,
    namespace: ProspectiveRandomStreamNamespace,
    physical_config: OperatingDecisionRealismConfig,
    config: ProspectiveUncertaintyConfig = ProspectiveUncertaintyConfig(),
) -> ProspectiveUncertaintyResult:
    """Estimate cost-free action value from acquisition-only predictive draws."""

    if not isinstance(physical_config, OperatingDecisionRealismConfig):
        raise ValueError("prospective uncertainty needs a realism configuration")
    if not isinstance(namespace, ProspectiveRandomStreamNamespace):
        raise ValueError("prospective uncertainty needs a random-stream namespace")
    if not isinstance(config, ProspectiveUncertaintyConfig):
        raise ValueError("prospective uncertainty needs a scoring configuration")
    _validate_acquisition_evidence(acquisition_evidence, physical_config)
    if namespace.acquisition_evidence_digest != acquisition_evidence.evidence_digest:
        raise ValueError("prospective namespace does not match acquisition evidence")

    envelope = acquisition_evidence.snapshot.provisional_margin_envelope
    if envelope is None:
        raise ValueError("prospective uncertainty needs a provisional envelope")
    baseline_width = envelope.upper - envelope.lower
    fit_by_model = {
        fit.model_name: fit for fit in acquisition_evidence.fit_set.fits
    }
    source_models = acquisition_evidence.snapshot.admissible_candidate_models
    policies = {
        item.name: item
        for item in default_fixed_policies()
        if item.name in PROSPECTIVE_ACQUISITION_ACTIONS
    }
    registry = ProspectiveRandomStreamRegistry()
    draws = []
    for generator_model in source_models:
        generator_fit = fit_by_model[generator_model]
        for draw_index in range(config.draw_count):
            try:
                generator_offsets = _generator_parameter_draw(
                    generator_fit,
                    namespace,
                    draw_index,
                    physical_config,
                    config,
                    registry,
                )
            except _ProspectiveDrawSamplingError as error:
                draws.extend(
                    _failed_draw(
                        policy_name=policy_name,
                        generator_model=generator_model,
                        draw_index=draw_index,
                        generator_offsets=(),
                        face_probe_offsets=None,
                        baseline_width=baseline_width,
                        stage="prospective_parameter_draw",
                        error_type=type(error).__name__,
                    )
                    for policy_name in PROSPECTIVE_ACQUISITION_ACTIONS
                )
                continue

            probe_offsets = None
            probe_error = None
            try:
                probe_offsets = _face_probe_draw(
                    namespace,
                    generator_model,
                    draw_index,
                    physical_config,
                    config,
                    registry,
                )
            except _ProspectiveDrawSamplingError as error:
                probe_error = error
            for policy_name in PROSPECTIVE_ACQUISITION_ACTIONS:
                if (
                    policy_name == FIXED_FACE_TEMPERATURE
                    and probe_error is not None
                ):
                    draws.append(
                        _failed_draw(
                            policy_name=policy_name,
                            generator_model=generator_model,
                            draw_index=draw_index,
                            generator_offsets=generator_offsets,
                            face_probe_offsets=None,
                            baseline_width=baseline_width,
                            stage="prospective_probe_draw",
                            error_type=type(probe_error).__name__,
                        )
                    )
                    continue
                draws.append(
                    _evaluate_hypothetical_draw(
                        policy=policies[policy_name],
                        generator_fit=generator_fit,
                        generator_offsets=generator_offsets,
                        face_probe_offsets=(
                            probe_offsets
                            if policy_name == FIXED_FACE_TEMPERATURE
                            else None
                        ),
                        draw_index=draw_index,
                        evidence=acquisition_evidence,
                        namespace=namespace,
                        physical_config=physical_config,
                        registry=registry,
                        baseline_width=baseline_width,
                    )
                )
    draws.sort(
        key=lambda item: (
            item.generator_model,
            item.policy_name,
            item.draw_index,
        )
    )
    actions = _summarize_uncertainty_draws(
        draws,
        baseline_width,
        source_models,
        config,
    )
    audit = registry.audit()
    audit.assert_clean()
    stream_uses = registry.uses
    protocol_digest = prospective_uncertainty_protocol_digest(
        physical_config,
        config,
    )
    result_digest = _result_digest(
        config=config,
        physical_config=physical_config,
        acquisition_evidence_digest=acquisition_evidence.evidence_digest,
        protocol_digest=protocol_digest,
        actions=actions,
        draws=draws,
        stream_uses=stream_uses,
        stream_audit=audit,
    )
    return ProspectiveUncertaintyResult(
        config=config,
        physical_config=physical_config,
        acquisition_evidence=acquisition_evidence,
        protocol_digest=protocol_digest,
        action_uncertainties=actions,
        draw_outcomes=tuple(draws),
        stream_uses=stream_uses,
        stream_audit=audit,
        result_digest=result_digest,
    )


def prefix_prospective_uncertainty_result(
    result: ProspectiveUncertaintyResult,
    *,
    draw_count: int,
    max_unstable_draws_per_source_action: int,
) -> ProspectiveUncertaintyResult:
    """Derive an authenticated matched prefix without repeating any fit.

    Prefixing may reduce the retained draw count or change its explicit
    instability allowance.  It can never expand the raw result.
    """

    if not isinstance(result, ProspectiveUncertaintyResult):
        raise ValueError("prospective prefix needs an uncertainty result")
    validate_prospective_uncertainty_result_integrity(result)
    if (
        not isinstance(draw_count, int)
        or isinstance(draw_count, bool)
        or draw_count <= 0
    ):
        raise ValueError("prospective prefix draw count must be positive")
    if draw_count > result.config.draw_count:
        raise ValueError("prospective prefix cannot expand its source result")
    config = replace(
        result.config,
        draw_count=draw_count,
        max_unstable_draws_per_source_action=(
            max_unstable_draws_per_source_action
        ),
    )
    draws = tuple(
        item for item in result.draw_outcomes if item.draw_index < draw_count
    )
    uses = tuple(
        item
        for item in result.stream_uses
        if item.stream.key.draw_index < draw_count
    )
    envelope = result.acquisition_evidence.snapshot.provisional_margin_envelope
    if envelope is None:
        raise ValueError("prospective prefix needs a provisional envelope")
    baseline_width = envelope.upper - envelope.lower
    actions = _summarize_uncertainty_draws(
        draws,
        baseline_width,
        result.acquisition_evidence.snapshot.admissible_candidate_models,
        config,
    )
    audit = _audit_stream_uses(uses)
    audit.assert_clean()
    protocol_digest = prospective_uncertainty_protocol_digest(
        result.physical_config,
        config,
    )
    result_digest = _result_digest(
        config=config,
        physical_config=result.physical_config,
        acquisition_evidence_digest=result.acquisition_evidence.evidence_digest,
        protocol_digest=protocol_digest,
        actions=actions,
        draws=draws,
        stream_uses=uses,
        stream_audit=audit,
    )
    return ProspectiveUncertaintyResult(
        config=config,
        physical_config=result.physical_config,
        acquisition_evidence=result.acquisition_evidence,
        protocol_digest=protocol_digest,
        action_uncertainties=actions,
        draw_outcomes=draws,
        stream_uses=uses,
        stream_audit=audit,
        result_digest=result_digest,
    )


def prospective_uncertainty_result_payload(
    result: ProspectiveUncertaintyResult,
) -> dict:
    """Return a strict finite JSON-ready Step-2 evidence record."""

    if not isinstance(result, ProspectiveUncertaintyResult):
        raise ValueError("prospective uncertainty payload needs a result")
    evidence_payload = prospective_acquisition_evidence_payload(
        result.acquisition_evidence,
        result.physical_config,
    )
    expected_result_digest = _result_digest(
        config=result.config,
        physical_config=result.physical_config,
        acquisition_evidence_digest=(
            result.acquisition_evidence.evidence_digest
        ),
        protocol_digest=result.protocol_digest,
        actions=result.action_uncertainties,
        draws=result.draw_outcomes,
        stream_uses=result.stream_uses,
        stream_audit=result.stream_audit,
    )
    if result.result_digest != expected_result_digest:
        raise ValueError("prospective uncertainty result digest is invalid")
    return {
        "schema_version": PROSPECTIVE_UNCERTAINTY_SCHEMA_VERSION,
        "protocol_version": PROSPECTIVE_UNCERTAINTY_PROTOCOL_VERSION,
        "protocol_digest": result.protocol_digest,
        "result_digest": result.result_digest,
        "physical_protocol_digest": (
            result.acquisition_evidence.physical_protocol_digest
        ),
        "acquisition_evidence_digest": (
            result.acquisition_evidence.evidence_digest
        ),
        "physical_config": _physical_config_payload(result.physical_config),
        "acquisition_evidence": evidence_payload,
        "config": _uncertainty_config_payload(result.config),
        "action_uncertainties": [
            _action_uncertainty_payload(item)
            for item in result.action_uncertainties
        ],
        "draw_outcomes": [
            _draw_payload(item) for item in result.draw_outcomes
        ],
        "stream_uses": [
            _stream_use_payload(item) for item in result.stream_uses
        ],
        "stream_audit": _stream_audit_payload(result.stream_audit),
    }


__all__ = [
    "PROSPECTIVE_ACROSS_GENERATOR_AGGREGATION",
    "PROSPECTIVE_CANDIDATE_ATTRITION_POLICY",
    "PROSPECTIVE_DRAW_FAILURE_POLICY",
    "PROSPECTIVE_ELIGIBILITY_PROTOCOL",
    "PROSPECTIVE_PADDED_SCORING_PROTOCOL",
    "PROSPECTIVE_REFIT_START_PROTOCOL",
    "PROSPECTIVE_UNCERTAINTY_ESTIMATOR",
    "PROSPECTIVE_UNCERTAINTY_METRIC",
    "PROSPECTIVE_UNCERTAINTY_PROTOCOL_VERSION",
    "PROSPECTIVE_UNCERTAINTY_SCHEMA_VERSION",
    "PROSPECTIVE_WITHIN_GENERATOR_AGGREGATION",
    "ProspectiveAcquisitionEvidence",
    "ProspectiveActionUncertainty",
    "ProspectiveCandidateOutcome",
    "ProspectiveDrawOutcome",
    "ProspectiveModelActionSummary",
    "ProspectivePaddedActionUncertainty",
    "ProspectivePaddedModelActionSummary",
    "ProspectiveUncertaintyConfig",
    "ProspectiveUncertaintyResult",
    "estimate_prospective_action_uncertainty",
    "prefix_prospective_uncertainty_result",
    "prepare_prospective_acquisition_evidence",
    "prospective_acquisition_evidence_payload",
    "prospective_uncertainty_protocol_digest",
    "prospective_uncertainty_result_payload",
    "prospective_uncertainty_summary_payload",
    "score_prospective_uncertainty_with_offsets",
    "validate_prospective_uncertainty_result_integrity",
]
