"""Stage 4 calibration and selection for the operating-decision benchmark.

Stage 3 is retained as an uncalibrated development stress test.  This module
uses a separate synthetic-device partition to calibrate the verification gate
and additive operating-margin envelopes, freezes a low-capacity selector, and
then rehearses every procedure on fresh development seeds.  The final Stage 5
evaluation seed namespace is reserved but never instantiated here.

The selector sees only fits from the shared initial acquisition.  It stops when
both candidate-model intervals form a finite, one-sided envelope and otherwise
requests the voltage package.  Its conformal padding is calibrated for that
entire selection rule, so selecting between stop and voltage does not invalidate
the interval calibration under the declared block-exchangeability assumption.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import hashlib
import math
import random
from statistics import fmean
from typing import Callable, Dict, NamedTuple, Optional, Sequence, Tuple

from ..observations.test_stand import regular_measurement_times
from ..numerics.integration import IntegrationDivergenceError
from .operating_decision import (
    APPROVE,
    FINAL_EVALUATION,
    FIXED_FACE_TEMPERATURE,
    FIXED_THERMAL,
    FIXED_VOLTAGE,
    INSUFFICIENT_EVIDENCE,
    REJECT,
    RUN_DURATION_SECONDS,
    STOP_NOW,
    CandidateVerification,
    MarginEnvelope,
    NumericalFailure,
    OperatingRegime,
    RateEstimate,
    SavedOperatingDecision,
    ScoredOperatingDecision,
    classify_margin_envelope,
    default_fixed_policies,
    envelope_margin_intervals,
    rate_estimate,
)
from .operating_decision_realism import (
    STAGE3_TRUTH_CONDITIONS,
    OperatingDecisionRealismConfig,
    RealisticAcquisitionFitSet,
    forecast_realistic_margin_interval,
    run_operating_decision_realism_truth,
)
from .sensor_model_discrimination import (
    FIVE_STATE_MODEL,
    FOUR_STATE_MODEL,
    MODEL_NAMES,
)


STAGE4_PROTOCOL_VERSION = "operating_decision_stage4_v1"
DECISION_DIRECTED_SELECTOR = "decision_directed_selector"
STAGE4_PROCEDURES = (
    STOP_NOW,
    FIXED_THERMAL,
    FIXED_VOLTAGE,
    FIXED_FACE_TEMPERATURE,
    DECISION_DIRECTED_SELECTOR,
)
CALIBRATION_SPLIT = "calibration"
REHEARSAL_SPLIT = "rehearsal"
ALL_FAMILIES = "all_families"


def _realism_partition(
    trial_count: int,
    first_seed: int,
) -> OperatingDecisionRealismConfig:
    base = OperatingDecisionRealismConfig()
    return replace(
        base,
        sensor=replace(
            base.sensor,
            trial_count=trial_count,
            first_seed=first_seed,
        ),
    )


@dataclass(frozen=True)
class SelectorRule:
    """Frozen action rule evaluated from the common initial acquisition."""

    resolved_policy: str = STOP_NOW
    fallback_policy: str = FIXED_VOLTAGE

    def __post_init__(self) -> None:
        if self.resolved_policy != STOP_NOW:
            raise ValueError("the Stage 4 resolved action must be stop-now")
        if self.fallback_policy != FIXED_VOLTAGE:
            raise ValueError("the frozen Stage 4 fallback must be voltage")


@dataclass(frozen=True)
class CostScenario:
    """Declared dimensionless weights used only for expected-loss sensitivity."""

    name: str
    false_approval_weight: float = 100.0
    false_rejection_weight: float = 20.0
    abstention_weight: float = 1.0
    added_run_weight: float = 0.10
    added_sensor_weight: float = 0.10
    incremental_energy_weight: float = 0.10

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("cost scenario needs a name")
        values = (
            self.false_approval_weight,
            self.false_rejection_weight,
            self.abstention_weight,
            self.added_run_weight,
            self.added_sensor_weight,
            self.incremental_energy_weight,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("cost weights must be finite and nonnegative")


DEFAULT_COST_SCENARIOS = (
    CostScenario("bench_time_dominant", added_sensor_weight=0.02),
    CostScenario("balanced"),
    CostScenario("instrumentation_expensive", added_sensor_weight=0.50),
)


@dataclass(frozen=True)
class OperatingDecisionCalibrationConfig:
    """Frozen Stage 4 partitions, calibration targets, and selector."""

    gate_development: OperatingDecisionRealismConfig = _realism_partition(
        10,
        191_001,
    )
    calibration: OperatingDecisionRealismConfig = _realism_partition(
        20,
        10_191_001,
    )
    rehearsal: OperatingDecisionRealismConfig = _realism_partition(
        10,
        20_191_001,
    )
    reserved_evaluation_first_seed: int = 30_191_001
    reserved_evaluation_trial_count: int = 50
    target_block_coverage: float = 0.90
    gate_target_block_retention: float = 0.90
    gate_null_quantile: float = 0.99
    gate_monte_carlo_draws: int = 50_000
    gate_first_seed: int = 40_191_001
    selector: SelectorRule = SelectorRule()
    cost_scenarios: Tuple[CostScenario, ...] = DEFAULT_COST_SCENARIOS

    def __post_init__(self) -> None:
        if not 0.0 < self.target_block_coverage < 1.0:
            raise ValueError("target block coverage must lie strictly between zero and one")
        if not 0.0 < self.gate_target_block_retention < 1.0:
            raise ValueError("gate target retention must lie strictly between zero and one")
        if not 0.5 < self.gate_null_quantile < 1.0:
            raise ValueError("gate null quantile must lie between 0.5 and one")
        if (
            not isinstance(self.gate_monte_carlo_draws, int)
            or isinstance(self.gate_monte_carlo_draws, bool)
            or self.gate_monte_carlo_draws < 1_000
        ):
            raise ValueError("gate calibration needs at least 1,000 Monte Carlo draws")
        for name, value in (
            ("gate seed", self.gate_first_seed),
            ("reserved evaluation seed", self.reserved_evaluation_first_seed),
            ("reserved evaluation count", self.reserved_evaluation_trial_count),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        scenarios = tuple(self.cost_scenarios)
        object.__setattr__(self, "cost_scenarios", scenarios)
        if not scenarios or len({item.name for item in scenarios}) != len(scenarios):
            raise ValueError("cost scenarios must be nonempty and uniquely named")
        _validate_matching_physics(self.gate_development, self.calibration)
        _validate_matching_physics(self.calibration, self.rehearsal)
        conformal_rank(
            self.gate_development.sensor.trial_count,
            self.gate_target_block_retention,
        )
        if self.calibration.sensor.trial_count < 2:
            raise ValueError("calibration needs at least two paired device blocks")
        conformal_rank(
            self.calibration.sensor.trial_count,
            self.target_block_coverage,
        )
        reserved = replace(
            self.calibration,
            sensor=replace(
                self.calibration.sensor,
                trial_count=self.reserved_evaluation_trial_count,
                first_seed=self.reserved_evaluation_first_seed,
            ),
        )
        _validate_matching_physics(self.calibration, reserved)
        namespaces = (
            _seed_namespace(self.gate_development),
            _seed_namespace(self.calibration),
            _seed_namespace(self.rehearsal),
            _seed_namespace(reserved),
        )
        if any(
            namespaces[left].intersection(namespaces[right])
            for left in range(len(namespaces))
            for right in range(left)
        ):
            raise ValueError("Stage 4 and reserved Stage 5 seed namespaces overlap")


class VerificationGate(NamedTuple):
    policy_name: str
    observation_count: int
    matched_development_block_count: int
    matched_development_rank: int
    target_block_retention: float
    matched_block_scores: Tuple[float, ...]
    noise_reference_quantile: float
    noise_reference_threshold: float
    threshold: float
    monte_carlo_draws: int
    seed: int


class InitialDecisionSignal(NamedTuple):
    """Acquisition-only selector input; it intentionally contains no truth label."""

    candidate_models: Tuple[str, ...]
    failed_fit_count: int
    bound_hit_count: int
    margin_envelope: Optional[MarginEnvelope]


class ProcedureCalibration(NamedTuple):
    procedure_name: str
    block_count: int
    conformal_rank: int
    target_block_coverage: float
    additive_margin_padding: float
    block_nonconformity: Tuple[float, ...]
    emitted_interval_count: int
    raw_interval_covered_count: int
    calibrated_interval_covered_count: int


class CalibrationArtifact(NamedTuple):
    protocol_version: str
    protocol_digest: str
    realism_protocol_digest: str
    gate_development_first_seed: int
    gate_development_block_count: int
    calibration_first_seed: int
    calibration_block_count: int
    rehearsal_first_seed: int
    rehearsal_block_count: int
    reserved_evaluation_first_seed: int
    reserved_evaluation_block_count: int
    verification_gates: Tuple[VerificationGate, ...]
    selector: SelectorRule
    procedure_calibrations: Tuple[ProcedureCalibration, ...]

    def gate_for(self, policy_name: str) -> VerificationGate:
        matches = tuple(
            item for item in self.verification_gates if item.policy_name == policy_name
        )
        if len(matches) != 1:
            raise ValueError("calibration artifact has no unique policy gate")
        return matches[0]

    def procedure_for(self, procedure_name: str) -> ProcedureCalibration:
        matches = tuple(
            item
            for item in self.procedure_calibrations
            if item.procedure_name == procedure_name
        )
        if len(matches) != 1:
            raise ValueError("calibration artifact has no unique procedure")
        return matches[0]


class ProcedureOutcome(NamedTuple):
    split: str
    truth_condition: str
    trial_index: int
    procedure_name: str
    selected_policy: str
    decision: str
    true_margin: float
    true_pass: bool
    false_approval: bool
    false_rejection: bool
    raw_interval_lower: Optional[float]
    raw_interval_upper: Optional[float]
    calibrated_interval_lower: Optional[float]
    calibrated_interval_upper: Optional[float]
    raw_interval_covered: Optional[bool]
    calibrated_interval_covered: Optional[bool]
    additive_margin_padding: float
    verified_candidate_count: int
    score_rejected_candidate_count: int
    numerical_failure_count: int
    acquisition_run_count: int
    diagnostic_run_count: int
    energized_schedule_time_seconds: float
    total_diagnostic_energy: float
    extra_sensor_count: int
    decision_computation_seconds: float


class ScenarioLoss(NamedTuple):
    scenario_name: str
    mean_loss: float


class ProcedureSummary(NamedTuple):
    split: str
    truth_condition: str
    procedure_name: str
    trial_count: int
    approvals: int
    rejections: int
    insufficient_evidence: int
    true_passing: int
    true_violating: int
    false_approvals: RateEstimate
    missed_violations: RateEstimate
    false_rejections: RateEstimate
    decision_coverage: RateEstimate
    abstentions: RateEstimate
    raw_interval_coverage: RateEstimate
    calibrated_interval_coverage: RateEstimate
    simultaneous_block_coverage: RateEstimate
    mean_diagnostic_run_count: float
    mean_energized_schedule_time_seconds: float
    mean_total_diagnostic_energy: float
    mean_extra_sensor_count: float
    mean_decision_computation_seconds: float
    numerical_failures: int
    selected_policy_counts: Tuple[Tuple[str, int], ...]
    expected_losses: Tuple[ScenarioLoss, ...]


class OperatingDecisionCalibrationResult(NamedTuple):
    config: OperatingDecisionCalibrationConfig
    artifact: CalibrationArtifact
    calibration_outcomes: Tuple[ProcedureOutcome, ...]
    rehearsal_outcomes: Tuple[ProcedureOutcome, ...]
    summaries: Tuple[ProcedureSummary, ...]


def _normalized_realism_config(
    config: OperatingDecisionRealismConfig,
) -> OperatingDecisionRealismConfig:
    return replace(
        config,
        sensor=replace(config.sensor, trial_count=1, first_seed=0),
    )


def _validate_matching_physics(
    left: OperatingDecisionRealismConfig,
    right: OperatingDecisionRealismConfig,
) -> None:
    if _normalized_realism_config(left) != _normalized_realism_config(right):
        raise ValueError("Stage 4 partitions must share one physical protocol")


def _seed_namespace(config: OperatingDecisionRealismConfig) -> frozenset[int]:
    """Enumerate the observation and truth seeds consumed by one partition."""

    seeds = set()
    run_offsets = (100, 200, 300, 400, 500, 9_900)
    for trial_index in range(config.sensor.trial_count):
        # Base device-parameter draw used by sensor_model_discrimination._truth_for_trial.
        seeds.add(config.sensor.first_seed + 10_000 * trial_index)
        # Stage 3 series-resistance, probe, and contact-beta draw.
        seeds.add(config.sensor.first_seed + 7_000_000 + 10_000 * trial_index)
        for truth_index in range(len(STAGE3_TRUTH_CONDITIONS)):
            for offset in run_offsets:
                seeds.add(
                    config.sensor.first_seed
                    + truth_index * 1_000_000
                    + trial_index * 10_000
                    + offset
                )
    return frozenset(seeds)


def conformal_rank(block_count: int, target_coverage: float) -> int:
    """Return the one-based finite-sample split-conformal order statistic."""

    if (
        not isinstance(block_count, int)
        or isinstance(block_count, bool)
        or block_count <= 0
        or not math.isfinite(target_coverage)
        or not 0.0 < target_coverage < 1.0
    ):
        raise ValueError("conformal rank needs a positive count and valid coverage")
    rank = math.ceil((block_count + 1) * target_coverage)
    if rank > block_count:
        raise ValueError("too few calibration blocks for a finite conformal quantile")
    return rank


def block_conformal_padding(
    block_nonconformity: Sequence[float],
    target_coverage: float,
) -> Tuple[int, float]:
    """Return the frozen rank and additive padding for block scores."""

    scores = tuple(float(value) for value in block_nonconformity)
    if not scores or any(not math.isfinite(value) or value < 0.0 for value in scores):
        raise ValueError("block nonconformity must be finite and nonnegative")
    rank = conformal_rank(len(scores), target_coverage)
    return rank, sorted(scores)[rank - 1]


def _empirical_quantile(values: Sequence[float], quantile: float) -> float:
    values = tuple(float(value) for value in values)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("empirical quantile needs finite values")
    if not 0.0 < quantile < 1.0:
        raise ValueError("empirical quantile must lie strictly between zero and one")
    rank = min(len(values), max(1, math.ceil(len(values) * quantile)))
    return sorted(values)[rank - 1]


def _noise_reference_threshold(
    observation_count: int,
    config: OperatingDecisionCalibrationConfig,
    *,
    seed: int,
) -> float:
    """Return a conditional exact-prediction reference for the quadratic score."""

    random_source = random.Random(seed)
    normalized_scores = tuple(
        random_source.gammavariate(observation_count / 2.0, 2.0)
        / observation_count
        for _ in range(config.gate_monte_carlo_draws)
    )
    return _empirical_quantile(normalized_scores, config.gate_null_quantile)


def matched_pipeline_verification_gates(
    development_trials: Sequence[ScoredOperatingDecision],
    config: OperatingDecisionCalibrationConfig,
) -> Tuple[VerificationGate, ...]:
    """Calibrate score cutoffs on correct candidates from the Stage 3 cohort.

    Each development block contributes the larger verification score from the
    correct four-state candidate under matched four-state truth and the correct
    five-state candidate under extra-interface-mass truth.  Family C is excluded
    from gate tuning and remains a pure missing-family diagnostic.
    """

    indexed = _validate_paired_trials(development_trials, config.gate_development)
    sample_count = len(
        regular_measurement_times(
            RUN_DURATION_SECONDS,
            config.gate_development.sensor.sampling_interval,
        )
    )
    gates = []
    for policy_index, policy in enumerate(default_fixed_policies()):
        channel_count = 2 + int(policy.extra_sensor_count > 0)
        observation_count = sample_count * channel_count
        seed = config.gate_first_seed + policy_index * 10_000
        block_scores = []
        for trial_index in range(config.gate_development.sensor.trial_count):
            correct_scores = []
            for truth, model_name in (
                (STAGE3_TRUTH_CONDITIONS[0], FOUR_STATE_MODEL),
                (STAGE3_TRUTH_CONDITIONS[1], FIVE_STATE_MODEL),
            ):
                trial = indexed[(truth, trial_index, policy.name)]
                matches = tuple(
                    verification
                    for verification in trial.saved.verifications
                    if verification.fit.model_name == model_name
                )
                if len(matches) != 1:
                    raise ValueError("matched gate development candidate is missing")
                verification = matches[0]
                if verification.fit.reached_bound or not math.isfinite(
                    verification.normalized_score
                ):
                    raise ValueError(
                        "matched gate development candidate is nonfinite or bound-hit"
                    )
                correct_scores.append(verification.normalized_score)
            block_scores.append(max(correct_scores))
        rank, matched_threshold = block_conformal_padding(
            block_scores,
            config.gate_target_block_retention,
        )
        noise_reference = _noise_reference_threshold(
            observation_count,
            config,
            seed=seed,
        )
        gates.append(
            VerificationGate(
                policy_name=policy.name,
                observation_count=observation_count,
                matched_development_block_count=len(block_scores),
                matched_development_rank=rank,
                target_block_retention=config.gate_target_block_retention,
                matched_block_scores=tuple(block_scores),
                noise_reference_quantile=config.gate_null_quantile,
                noise_reference_threshold=noise_reference,
                threshold=max(matched_threshold, noise_reference),
                monte_carlo_draws=config.gate_monte_carlo_draws,
                seed=seed,
            )
        )
    return tuple(gates)


def initial_decision_signal(
    fit_set: RealisticAcquisitionFitSet,
    final_regime: OperatingRegime,
    config: OperatingDecisionRealismConfig,
) -> InitialDecisionSignal:
    """Construct the selector input from acquisition fits only."""

    fits = tuple(fit_set.fits)
    candidate_models = tuple(sorted(fit.model_name for fit in fits))
    bound_hits = sum(fit.reached_bound for fit in fits)
    envelope = None
    if (
        not fit_set.failures
        and not bound_hits
        and set(candidate_models) == set(MODEL_NAMES)
        and len(candidate_models) == len(MODEL_NAMES)
    ):
        try:
            intervals = tuple(
                forecast_realistic_margin_interval(fit, final_regime, config)
                for fit in fits
            )
            envelope = envelope_margin_intervals(intervals)
        except (ArithmeticError, IntegrationDivergenceError, ValueError):
            envelope = None
    return InitialDecisionSignal(
        candidate_models=candidate_models,
        failed_fit_count=len(fit_set.failures),
        bound_hit_count=bound_hits,
        margin_envelope=envelope,
    )


def select_fixed_policy(
    signal: InitialDecisionSignal,
    rule: SelectorRule = SelectorRule(),
) -> str:
    """Choose stop or voltage without verification, action data, or final truth."""

    if (
        signal.margin_envelope is not None
        and classify_margin_envelope(signal.margin_envelope) != INSUFFICIENT_EVIDENCE
    ):
        return rule.resolved_policy
    return rule.fallback_policy


def _fit_set_from_stop_trial(
    trial: ScoredOperatingDecision,
) -> RealisticAcquisitionFitSet:
    if trial.saved.policy_name != STOP_NOW:
        raise ValueError("the selector signal must come from the stop-now acquisition")
    acquisition_failures = tuple(
        item for item in trial.saved.failures if item.stage == "acquisition_fit"
    )
    fits = tuple(
        verification.fit for verification in trial.saved.verifications
    )
    return RealisticAcquisitionFitSet(fits, acquisition_failures)  # type: ignore[arg-type]


def _gate_for_policy(
    gates: Sequence[VerificationGate],
    policy_name: str,
) -> VerificationGate:
    matches = tuple(item for item in gates if item.policy_name == policy_name)
    if len(matches) != 1:
        raise ValueError("Stage 4 needs one verification gate per fixed policy")
    return matches[0]


def rebuild_with_calibrated_gate(
    trial: ScoredOperatingDecision,
    gate: VerificationGate,
    config: OperatingDecisionRealismConfig,
) -> ScoredOperatingDecision:
    """Rebuild one saved decision with the development-calibrated gate."""

    if trial.saved.policy_name != gate.policy_name:
        raise ValueError("verification gate and trial policy do not match")
    acquisition_failures = tuple(
        item for item in trial.saved.failures if item.stage == "acquisition_fit"
    )
    failures = list(acquisition_failures)
    verifications = []
    intervals = []
    for original in trial.saved.verifications:
        fit = original.fit
        if fit.reached_bound:
            reason = "fit_reached_bound"
        elif not math.isfinite(original.normalized_score):
            reason = "nonfinite_score"
            failures.append(
                NumericalFailure(fit.model_name, "verification", reason)
            )
        elif original.normalized_score > gate.threshold:
            reason = "inadequate_development_calibrated_verification"
        else:
            reason = None
        verification = CandidateVerification(
            fit,
            original.normalized_score,
            reason is None,
            reason,
        )
        verifications.append(verification)
        if reason is not None:
            continue
        try:
            intervals.append(
                forecast_realistic_margin_interval(
                    fit,  # type: ignore[arg-type]
                    _final_regime_for(config),
                    config,
                )
            )
        except (ArithmeticError, IntegrationDivergenceError, ValueError) as error:
            failures.append(
                NumericalFailure(
                    fit.model_name,
                    "uncertainty",
                    type(error).__name__,
                )
            )
    candidate_envelope = envelope_margin_intervals(intervals)
    envelope = None if failures else candidate_envelope
    if acquisition_failures:
        decision = INSUFFICIENT_EVIDENCE
        reason = "acquisition_fit_failure"
    elif any(item.stage == "verification" for item in failures):
        decision = INSUFFICIENT_EVIDENCE
        reason = "verification_failure"
    elif any(item.stage == "uncertainty" for item in failures):
        decision = INSUFFICIENT_EVIDENCE
        reason = "uncertainty_failure"
    elif envelope is None:
        decision = INSUFFICIENT_EVIDENCE
        reason = "no_development_calibrated_candidate"
    else:
        decision = classify_margin_envelope(envelope)
        reason = {
            APPROVE: "development_calibrated_envelope_inside_band",
            REJECT: "development_calibrated_envelope_below_zero",
            INSUFFICIENT_EVIDENCE: "development_calibrated_envelope_crosses_zero",
        }[decision]
    saved = SavedOperatingDecision(
        case_id=trial.saved.case_id,
        decision=decision,
        decision_reason=reason,
        margin_envelope=envelope,
        model_intervals=tuple(intervals),
        verifications=tuple(verifications),
        failures=tuple(failures),
        decision_computation_seconds=trial.saved.decision_computation_seconds,
    )
    return _replace_scored_decision(trial, saved)


def _final_regime_for(config: OperatingDecisionRealismConfig) -> OperatingRegime:
    return OperatingRegime(
        name="untouched_final_operating_schedule",
        phase=FINAL_EVALUATION,
        current=config.final_current,
        channels=(),
    )


def _replace_scored_decision(
    trial: ScoredOperatingDecision,
    saved: SavedOperatingDecision,
) -> ScoredOperatingDecision:
    envelope = saved.margin_envelope
    true_pass = trial.true_margin >= 0.0
    return trial._replace(
        saved=saved,
        true_pass=true_pass,
        false_approval=saved.decision == APPROVE and not true_pass,
        false_rejection=saved.decision == REJECT and true_pass,
        interval_covered=(
            None
            if envelope is None
            else envelope.lower <= trial.true_margin <= envelope.upper
        ),
    )


def apply_margin_padding(
    trial: ScoredOperatingDecision,
    padding: float,
) -> ScoredOperatingDecision:
    """Widen an emitted margin envelope and reclassify the saved decision."""

    if not math.isfinite(padding) or padding < 0.0:
        raise ValueError("margin padding must be finite and nonnegative")
    envelope = trial.saved.margin_envelope
    if envelope is None:
        if trial.saved.decision != INSUFFICIENT_EVIDENCE:
            raise ValueError("a missing envelope must already abstain")
        return trial
    calibrated = MarginEnvelope(envelope.lower - padding, envelope.upper + padding)
    decision = classify_margin_envelope(calibrated)
    reason = {
        APPROVE: "conformal_envelope_inside_band",
        REJECT: "conformal_envelope_below_zero",
        INSUFFICIENT_EVIDENCE: "conformal_envelope_crosses_zero",
    }[decision]
    saved = trial.saved._replace(
        decision=decision,
        decision_reason=reason,
        margin_envelope=calibrated,
    )
    return _replace_scored_decision(trial, saved)


def _outside_interval_miss(trial: ScoredOperatingDecision) -> float:
    envelope = trial.saved.margin_envelope
    if envelope is None:
        return 0.0
    return max(
        envelope.lower - trial.true_margin,
        trial.true_margin - envelope.upper,
        0.0,
    )


def _validate_paired_trials(
    trials: Sequence[ScoredOperatingDecision],
    config: OperatingDecisionRealismConfig,
) -> Dict[Tuple[str, int, str], ScoredOperatingDecision]:
    trials = tuple(trials)
    policies = default_fixed_policies()
    expected = {
        (truth, trial_index, policy.name)
        for truth in STAGE3_TRUTH_CONDITIONS
        for trial_index in range(config.sensor.trial_count)
        for policy in policies
    }
    indexed = {
        (item.truth_condition, item.saved.trial_index, item.saved.policy_name): item
        for item in trials
    }
    if len(indexed) != len(trials) or set(indexed) != expected:
        raise ValueError("Stage 4 trials are missing, duplicated, or unknown")
    for truth in STAGE3_TRUTH_CONDITIONS:
        for trial_index in range(config.sensor.trial_count):
            margins = tuple(
                indexed[(truth, trial_index, policy.name)].true_margin
                for policy in policies
            )
            if any(
                not math.isclose(value, margins[0], rel_tol=0.0, abs_tol=1.0e-12)
                for value in margins[1:]
            ):
                raise ValueError("paired policies do not share the unloaded final truth")
    return indexed


def _gate_trials(
    trials: Sequence[ScoredOperatingDecision],
    gates: Sequence[VerificationGate],
    config: OperatingDecisionRealismConfig,
) -> Dict[Tuple[str, int, str], ScoredOperatingDecision]:
    indexed = _validate_paired_trials(trials, config)
    return {
        key: rebuild_with_calibrated_gate(
            trial,
            _gate_for_policy(gates, key[2]),
            config,
        )
        for key, trial in indexed.items()
    }


def _selector_choices(
    indexed_raw: Dict[Tuple[str, int, str], ScoredOperatingDecision],
    config: OperatingDecisionRealismConfig,
    rule: SelectorRule,
) -> Dict[Tuple[str, int], str]:
    choices = {}
    final_regime = _final_regime_for(config)
    for truth in STAGE3_TRUTH_CONDITIONS:
        for trial_index in range(config.sensor.trial_count):
            stop_trial = indexed_raw[(truth, trial_index, STOP_NOW)]
            signal = initial_decision_signal(
                _fit_set_from_stop_trial(stop_trial),
                final_regime,
                config,
            )
            choices[(truth, trial_index)] = select_fixed_policy(signal, rule)
    return choices


def _procedure_trial(
    procedure_name: str,
    truth: str,
    trial_index: int,
    gated: Dict[Tuple[str, int, str], ScoredOperatingDecision],
    selector_choices: Dict[Tuple[str, int], str],
) -> ScoredOperatingDecision:
    policy_name = (
        selector_choices[(truth, trial_index)]
        if procedure_name == DECISION_DIRECTED_SELECTOR
        else procedure_name
    )
    return gated[(truth, trial_index, policy_name)]


def calibrate_operating_decisions(
    trials: Sequence[ScoredOperatingDecision],
    gates: Sequence[VerificationGate],
    config: OperatingDecisionCalibrationConfig,
) -> Tuple[CalibrationArtifact, Dict[Tuple[str, int, str], ScoredOperatingDecision]]:
    """Fit blockwise conformal paddings on the dedicated calibration split."""

    gated = _gate_trials(trials, gates, config.calibration)
    raw_index = _validate_paired_trials(trials, config.calibration)
    choices = _selector_choices(raw_index, config.calibration, config.selector)
    block_count = config.calibration.sensor.trial_count
    rank = conformal_rank(block_count, config.target_block_coverage)
    calibrations = []
    for procedure_name in STAGE4_PROCEDURES:
        block_scores = []
        emitted = 0
        raw_covered = 0
        for trial_index in range(block_count):
            family_scores = []
            for truth in STAGE3_TRUTH_CONDITIONS:
                trial = _procedure_trial(
                    procedure_name,
                    truth,
                    trial_index,
                    gated,
                    choices,
                )
                family_scores.append(_outside_interval_miss(trial))
                if trial.saved.margin_envelope is not None:
                    emitted += 1
                    raw_covered += int(trial.interval_covered is True)
            block_scores.append(max(family_scores))
        computed_rank, padding = block_conformal_padding(
            block_scores,
            config.target_block_coverage,
        )
        if computed_rank != rank:
            raise RuntimeError("conformal rank changed within one calibration run")
        calibrated_covered = 0
        for trial_index in range(block_count):
            for truth in STAGE3_TRUTH_CONDITIONS:
                trial = _procedure_trial(
                    procedure_name,
                    truth,
                    trial_index,
                    gated,
                    choices,
                )
                calibrated = apply_margin_padding(trial, padding)
                calibrated_covered += int(calibrated.interval_covered is True)
        calibrations.append(
            ProcedureCalibration(
                procedure_name=procedure_name,
                block_count=block_count,
                conformal_rank=rank,
                target_block_coverage=config.target_block_coverage,
                additive_margin_padding=padding,
                block_nonconformity=tuple(block_scores),
                emitted_interval_count=emitted,
                raw_interval_covered_count=raw_covered,
                calibrated_interval_covered_count=calibrated_covered,
            )
        )
    digest = _protocol_digest(config, gates, tuple(calibrations))
    artifact = CalibrationArtifact(
        protocol_version=STAGE4_PROTOCOL_VERSION,
        protocol_digest=digest,
        realism_protocol_digest=_realism_protocol_digest(config.calibration),
        gate_development_first_seed=config.gate_development.sensor.first_seed,
        gate_development_block_count=config.gate_development.sensor.trial_count,
        calibration_first_seed=config.calibration.sensor.first_seed,
        calibration_block_count=block_count,
        rehearsal_first_seed=config.rehearsal.sensor.first_seed,
        rehearsal_block_count=config.rehearsal.sensor.trial_count,
        reserved_evaluation_first_seed=config.reserved_evaluation_first_seed,
        reserved_evaluation_block_count=config.reserved_evaluation_trial_count,
        verification_gates=tuple(gates),
        selector=config.selector,
        procedure_calibrations=tuple(calibrations),
    )
    return artifact, gated


def _protocol_digest(
    config: OperatingDecisionCalibrationConfig,
    gates: Sequence[VerificationGate],
    calibrations: Sequence[ProcedureCalibration],
) -> str:
    normalized = _normalized_realism_config(config.calibration)
    material = repr(
        (
            STAGE4_PROTOCOL_VERSION,
            normalized,
            (
                config.gate_development.sensor.first_seed,
                config.gate_development.sensor.trial_count,
                config.calibration.sensor.first_seed,
                config.calibration.sensor.trial_count,
                config.rehearsal.sensor.first_seed,
                config.rehearsal.sensor.trial_count,
                config.reserved_evaluation_first_seed,
                config.reserved_evaluation_trial_count,
            ),
            config.target_block_coverage,
            config.gate_target_block_retention,
            config.gate_null_quantile,
            config.gate_monte_carlo_draws,
            config.gate_first_seed,
            config.selector,
            config.cost_scenarios,
            tuple(gates),
            tuple(calibrations),
        )
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _realism_protocol_digest(config: OperatingDecisionRealismConfig) -> str:
    material = repr(_normalized_realism_config(config)).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _compact_outcome(
    split: str,
    procedure_name: str,
    selected_policy: str,
    raw: ScoredOperatingDecision,
    calibrated: ScoredOperatingDecision,
    padding: float,
) -> ProcedureOutcome:
    raw_envelope = raw.saved.margin_envelope
    calibrated_envelope = calibrated.saved.margin_envelope
    return ProcedureOutcome(
        split=split,
        truth_condition=calibrated.truth_condition,
        trial_index=calibrated.saved.trial_index,
        procedure_name=procedure_name,
        selected_policy=selected_policy,
        decision=calibrated.saved.decision,
        true_margin=calibrated.true_margin,
        true_pass=calibrated.true_pass,
        false_approval=calibrated.false_approval,
        false_rejection=calibrated.false_rejection,
        raw_interval_lower=None if raw_envelope is None else raw_envelope.lower,
        raw_interval_upper=None if raw_envelope is None else raw_envelope.upper,
        calibrated_interval_lower=(
            None if calibrated_envelope is None else calibrated_envelope.lower
        ),
        calibrated_interval_upper=(
            None if calibrated_envelope is None else calibrated_envelope.upper
        ),
        raw_interval_covered=raw.interval_covered,
        calibrated_interval_covered=calibrated.interval_covered,
        additive_margin_padding=padding,
        verified_candidate_count=sum(item.passed for item in raw.saved.verifications),
        score_rejected_candidate_count=sum(
            item.failure_reason == "inadequate_development_calibrated_verification"
            for item in raw.saved.verifications
        ),
        numerical_failure_count=len(raw.saved.failures),
        acquisition_run_count=calibrated.acquisition_run_count,
        diagnostic_run_count=calibrated.diagnostic_run_count,
        energized_schedule_time_seconds=calibrated.energized_schedule_time_seconds,
        total_diagnostic_energy=calibrated.total_diagnostic_energy,
        extra_sensor_count=calibrated.extra_sensor_count,
        decision_computation_seconds=calibrated.saved.decision_computation_seconds,
    )


def apply_calibration_artifact(
    trials: Sequence[ScoredOperatingDecision],
    artifact: CalibrationArtifact,
    config: OperatingDecisionRealismConfig,
    *,
    split: str,
) -> Tuple[ProcedureOutcome, ...]:
    """Apply the frozen gate, selector, and procedure-specific padding."""

    if split not in (CALIBRATION_SPLIT, REHEARSAL_SPLIT):
        raise ValueError("unknown Stage 4 split")
    if _realism_protocol_digest(config) != artifact.realism_protocol_digest:
        raise ValueError("calibration artifact does not match the physical protocol")
    expected_partition = {
        CALIBRATION_SPLIT: (
            artifact.calibration_first_seed,
            artifact.calibration_block_count,
        ),
        REHEARSAL_SPLIT: (
            artifact.rehearsal_first_seed,
            artifact.rehearsal_block_count,
        ),
    }[split]
    if (
        config.sensor.first_seed,
        config.sensor.trial_count,
    ) != expected_partition:
        raise ValueError("calibration artifact does not match the requested split")
    raw_index = _validate_paired_trials(trials, config)
    gated = _gate_trials(trials, artifact.verification_gates, config)
    choices = _selector_choices(raw_index, config, artifact.selector)
    outcomes = []
    for truth in STAGE3_TRUTH_CONDITIONS:
        for trial_index in range(config.sensor.trial_count):
            for procedure_name in STAGE4_PROCEDURES:
                selected_policy = (
                    choices[(truth, trial_index)]
                    if procedure_name == DECISION_DIRECTED_SELECTOR
                    else procedure_name
                )
                raw = gated[(truth, trial_index, selected_policy)]
                calibration = artifact.procedure_for(procedure_name)
                calibrated = apply_margin_padding(
                    raw,
                    calibration.additive_margin_padding,
                )
                outcomes.append(
                    _compact_outcome(
                        split,
                        procedure_name,
                        selected_policy,
                        raw,
                        calibrated,
                        calibration.additive_margin_padding,
                    )
                )
    return tuple(outcomes)


def expected_loss(
    outcome: ProcedureOutcome,
    scenario: CostScenario,
    *,
    stop_energy: float,
) -> float:
    """Return the declared decision-and-resource loss for one outcome."""

    if not math.isfinite(stop_energy) or stop_energy <= 0.0:
        raise ValueError("stop energy must be finite and positive")
    incremental_energy = max(
        0.0,
        (outcome.total_diagnostic_energy - stop_energy) / stop_energy,
    )
    return (
        scenario.false_approval_weight * int(outcome.false_approval)
        + scenario.false_rejection_weight * int(outcome.false_rejection)
        + scenario.abstention_weight
        * int(outcome.decision == INSUFFICIENT_EVIDENCE)
        + scenario.added_run_weight * max(0, outcome.diagnostic_run_count - 2)
        + scenario.added_sensor_weight * outcome.extra_sensor_count
        + scenario.incremental_energy_weight * incremental_energy
    )


def _summary_rate(
    numerator: int,
    denominator: int,
    *,
    descriptive_only: bool,
) -> RateEstimate:
    """Suppress Wilson intervals for fitted or clustered descriptive rates."""

    ordinary = rate_estimate(numerator, denominator)
    if not descriptive_only or ordinary.rate is None:
        return ordinary
    return RateEstimate(
        ordinary.numerator,
        ordinary.denominator,
        ordinary.rate,
        None,
        None,
    )


def summarize_calibrated_outcomes(
    outcomes: Sequence[ProcedureOutcome],
    config: OperatingDecisionCalibrationConfig,
) -> Tuple[ProcedureSummary, ...]:
    outcomes = tuple(outcomes)
    if not outcomes:
        raise ValueError("Stage 4 summary needs outcomes")
    splits = tuple(sorted({item.split for item in outcomes}))
    procedures = STAGE4_PROCEDURES
    summaries = []
    for split in splits:
        split_rows = tuple(item for item in outcomes if item.split == split)
        stop_energy = {
            (item.truth_condition, item.trial_index): item.total_diagnostic_energy
            for item in split_rows
            if item.procedure_name == STOP_NOW
        }
        expected_stop_keys = {
            (item.truth_condition, item.trial_index) for item in split_rows
        }
        if set(stop_energy) != expected_stop_keys:
            raise ValueError("Stage 4 expected loss needs the stop baseline")
        for truth in (*STAGE3_TRUTH_CONDITIONS, ALL_FAMILIES):
            for procedure in procedures:
                selected = tuple(
                    item
                    for item in split_rows
                    if item.procedure_name == procedure
                    and (truth == ALL_FAMILIES or item.truth_condition == truth)
                )
                if not selected:
                    raise ValueError("Stage 4 summary cell is empty")
                approvals = sum(item.decision == APPROVE for item in selected)
                rejections = sum(item.decision == REJECT for item in selected)
                insufficient = len(selected) - approvals - rejections
                true_passing = sum(item.true_pass for item in selected)
                true_violating = len(selected) - true_passing
                raw_intervals = tuple(
                    item for item in selected if item.raw_interval_covered is not None
                )
                calibrated_intervals = tuple(
                    item
                    for item in selected
                    if item.calibrated_interval_covered is not None
                )
                if truth == ALL_FAMILIES:
                    block_ids = sorted({item.trial_index for item in selected})
                    simultaneous = tuple(
                        all(
                            item.calibrated_interval_covered is not False
                            for item in selected
                            if item.trial_index == block_id
                        )
                        for block_id in block_ids
                    )
                else:
                    simultaneous = tuple(
                        item.calibrated_interval_covered is not False
                        for item in selected
                    )
                policy_counts = tuple(
                    (policy.name, sum(item.selected_policy == policy.name for item in selected))
                    for policy in default_fixed_policies()
                    if any(item.selected_policy == policy.name for item in selected)
                )
                summaries.append(
                    ProcedureSummary(
                        split=split,
                        truth_condition=truth,
                        procedure_name=procedure,
                        trial_count=len(selected),
                        approvals=approvals,
                        rejections=rejections,
                        insufficient_evidence=insufficient,
                        true_passing=true_passing,
                        true_violating=true_violating,
                        false_approvals=_summary_rate(
                            sum(item.false_approval for item in selected),
                            approvals,
                            descriptive_only=(
                                split == CALIBRATION_SPLIT or truth == ALL_FAMILIES
                            ),
                        ),
                        missed_violations=_summary_rate(
                            sum(item.false_approval for item in selected),
                            true_violating,
                            descriptive_only=(
                                split == CALIBRATION_SPLIT or truth == ALL_FAMILIES
                            ),
                        ),
                        false_rejections=_summary_rate(
                            sum(item.false_rejection for item in selected),
                            rejections,
                            descriptive_only=(
                                split == CALIBRATION_SPLIT or truth == ALL_FAMILIES
                            ),
                        ),
                        decision_coverage=_summary_rate(
                            approvals + rejections,
                            len(selected),
                            descriptive_only=(
                                split == CALIBRATION_SPLIT or truth == ALL_FAMILIES
                            ),
                        ),
                        abstentions=_summary_rate(
                            insufficient,
                            len(selected),
                            descriptive_only=(
                                split == CALIBRATION_SPLIT or truth == ALL_FAMILIES
                            ),
                        ),
                        raw_interval_coverage=_summary_rate(
                            sum(item.raw_interval_covered is True for item in raw_intervals),
                            len(raw_intervals),
                            descriptive_only=(
                                split == CALIBRATION_SPLIT or truth == ALL_FAMILIES
                            ),
                        ),
                        calibrated_interval_coverage=_summary_rate(
                            sum(
                                item.calibrated_interval_covered is True
                                for item in calibrated_intervals
                            ),
                            len(calibrated_intervals),
                            descriptive_only=(
                                split == CALIBRATION_SPLIT or truth == ALL_FAMILIES
                            ),
                        ),
                        simultaneous_block_coverage=_summary_rate(
                            sum(simultaneous),
                            len(simultaneous),
                            descriptive_only=split == CALIBRATION_SPLIT,
                        ),
                        mean_diagnostic_run_count=fmean(
                            item.diagnostic_run_count for item in selected
                        ),
                        mean_energized_schedule_time_seconds=fmean(
                            item.energized_schedule_time_seconds for item in selected
                        ),
                        mean_total_diagnostic_energy=fmean(
                            item.total_diagnostic_energy for item in selected
                        ),
                        mean_extra_sensor_count=fmean(
                            item.extra_sensor_count for item in selected
                        ),
                        mean_decision_computation_seconds=fmean(
                            item.decision_computation_seconds for item in selected
                        ),
                        numerical_failures=sum(
                            item.numerical_failure_count > 0 for item in selected
                        ),
                        selected_policy_counts=policy_counts,
                        expected_losses=tuple(
                            ScenarioLoss(
                                scenario.name,
                                fmean(
                                    expected_loss(
                                        item,
                                        scenario,
                                        stop_energy=stop_energy[
                                            (
                                                item.truth_condition,
                                                item.trial_index,
                                            )
                                        ],
                                    )
                                    for item in selected
                                ),
                            )
                            for scenario in config.cost_scenarios
                        ),
                    )
                )
    return tuple(summaries)


def _run_truth_worker(args):
    truth_condition, config = args
    return truth_condition, run_operating_decision_realism_truth(
        truth_condition,
        config,
    )


def _run_partition(
    split: str,
    partition: OperatingDecisionRealismConfig,
    *,
    workers: int,
    progress: Optional[Callable[[str], None]],
) -> Tuple[ScoredOperatingDecision, ...]:
    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise ValueError("worker count must be a positive integer")
    if split not in ("gate_development", CALIBRATION_SPLIT, REHEARSAL_SPLIT):
        raise ValueError("unknown Stage 4 generation split")
    completed: Dict[str, Tuple[ScoredOperatingDecision, ...]] = {}
    if workers == 1:
        for truth in STAGE3_TRUTH_CONDITIONS:
            if progress is not None:
                progress(f"{split}: starting {truth}")
            completed[truth] = run_operating_decision_realism_truth(
                truth,
                partition,
                progress=(
                    (lambda message, split=split: progress(f"{split}: {message}"))
                    if progress is not None
                    else None
                ),
            )
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(STAGE3_TRUTH_CONDITIONS))
        ) as executor:
            futures = {
                executor.submit(_run_truth_worker, (truth, partition)): truth
                for truth in STAGE3_TRUTH_CONDITIONS
            }
            for future in as_completed(futures):
                truth = futures[future]
                _, trials = future.result()
                completed[truth] = trials
                if progress is not None:
                    progress(f"{split}: completed {truth}")
    return tuple(
        trial
        for truth in STAGE3_TRUTH_CONDITIONS
        for trial in completed[truth]
    )


def run_operating_decision_calibration(
    config: OperatingDecisionCalibrationConfig = OperatingDecisionCalibrationConfig(),
    *,
    workers: int = 1,
    progress: Optional[Callable[[str], None]] = None,
) -> OperatingDecisionCalibrationResult:
    """Run Stage 4 calibration and the fresh-seed development rehearsal."""

    gate_development_trials = _run_partition(
        "gate_development",
        config.gate_development,
        workers=workers,
        progress=progress,
    )
    gates = matched_pipeline_verification_gates(
        gate_development_trials,
        config,
    )
    if progress is not None:
        progress("matched-pipeline verification gates frozen")
    calibration_trials = _run_partition(
        CALIBRATION_SPLIT,
        config.calibration,
        workers=workers,
        progress=progress,
    )
    artifact, _ = calibrate_operating_decisions(
        calibration_trials,
        gates,
        config,
    )
    if progress is not None:
        progress(f"calibration artifact frozen: {artifact.protocol_digest[:12]}")
    calibration_outcomes = apply_calibration_artifact(
        calibration_trials,
        artifact,
        config.calibration,
        split=CALIBRATION_SPLIT,
    )
    # This call is intentionally after artifact construction.  Rehearsal data
    # cannot exist in this process before the gate, selector, and paddings freeze.
    rehearsal_trials = _run_partition(
        REHEARSAL_SPLIT,
        config.rehearsal,
        workers=workers,
        progress=progress,
    )
    rehearsal_outcomes = apply_calibration_artifact(
        rehearsal_trials,
        artifact,
        config.rehearsal,
        split=REHEARSAL_SPLIT,
    )
    summaries = summarize_calibrated_outcomes(
        (*calibration_outcomes, *rehearsal_outcomes),
        config,
    )
    return OperatingDecisionCalibrationResult(
        config=config,
        artifact=artifact,
        calibration_outcomes=calibration_outcomes,
        rehearsal_outcomes=rehearsal_outcomes,
        summaries=summaries,
    )


__all__ = [
    "ALL_FAMILIES",
    "CALIBRATION_SPLIT",
    "DECISION_DIRECTED_SELECTOR",
    "DEFAULT_COST_SCENARIOS",
    "CalibrationArtifact",
    "CostScenario",
    "InitialDecisionSignal",
    "OperatingDecisionCalibrationConfig",
    "OperatingDecisionCalibrationResult",
    "ProcedureCalibration",
    "ProcedureOutcome",
    "ProcedureSummary",
    "REHEARSAL_SPLIT",
    "STAGE4_PROCEDURES",
    "ScenarioLoss",
    "SelectorRule",
    "VerificationGate",
    "apply_calibration_artifact",
    "apply_margin_padding",
    "block_conformal_padding",
    "calibrate_operating_decisions",
    "conformal_rank",
    "expected_loss",
    "initial_decision_signal",
    "matched_pipeline_verification_gates",
    "rebuild_with_calibrated_gate",
    "run_operating_decision_calibration",
    "select_fixed_policy",
    "summarize_calibrated_outcomes",
]
