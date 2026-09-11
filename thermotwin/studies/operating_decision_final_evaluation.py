"""Mismatch-guarded selector calibration and reserved Stage 5 evaluation.

Stage 4 showed that selecting voltage from the common acquisition could spend a
diagnostic run and still abstain when neither constant-contact candidate was an
adequate description of the device.  This module makes one low-capacity
revision: a calibrated acquisition lack-of-fit guard can abstain immediately;
otherwise the original stop-or-voltage rule is unchanged.

The guard is calibrated on matched four-state and extra-interface-mass
development devices.  The complete revised procedure is then conformally
calibrated on a disjoint three-family partition.  Reserved Stage 5 data can be
generated only by a separate evaluator that requires the serialized artifact.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import fmean
from typing import Callable, Dict, NamedTuple, Optional, Sequence, Tuple

from ..observations.test_stand import regular_measurement_times
from .operating_decision import (
    APPROVE,
    FIXED_FACE_TEMPERATURE,
    FIXED_THERMAL,
    FIXED_VOLTAGE,
    INSUFFICIENT_EVIDENCE,
    REJECT,
    RUN_DURATION_SECONDS,
    STOP_NOW,
    MarginEnvelope,
    RateEstimate,
    SavedOperatingDecision,
    ScoredOperatingDecision,
    classify_margin_envelope,
    default_fixed_policies,
    rate_estimate,
)
from .operating_decision_calibration import (
    ALL_FAMILIES,
    DECISION_DIRECTED_SELECTOR,
    DEFAULT_COST_SCENARIOS,
    CalibrationArtifact,
    CostScenario,
    InitialDecisionSignal,
    OperatingDecisionCalibrationConfig,
    ProcedureCalibration,
    ProcedureOutcome,
    ProcedureSummary,
    ScenarioLoss,
    STAGE4_PROCEDURES,
    SelectorRule,
    VerificationGate,
    _compact_outcome,
    _final_regime_for,
    _fit_set_from_stop_trial,
    _gate_trials,
    _outside_interval_miss,
    _protocol_digest as _stage4_protocol_digest,
    _realism_protocol_digest,
    _seed_namespace,
    _validate_matching_physics,
    _validate_paired_trials,
    _selector_choices,
    apply_margin_padding,
    block_conformal_padding,
    conformal_rank,
    expected_loss,
    initial_decision_signal,
)
from .operating_decision_realism import (
    STAGE3_TRUTH_CONDITIONS,
    OperatingDecisionRealismConfig,
    RealisticAcquisitionFitSet,
    RealisticCandidateFit,
    _parameter_spec,
    run_operating_decision_realism_truth,
)
from .sensor_model_discrimination import (
    FIVE_STATE_MODEL,
    FOUR_STATE_MODEL,
    MODEL_NAMES,
)


STAGE5_PROTOCOL_VERSION = "operating_decision_stage5_mismatch_guard_v1"
EXPECTED_STAGE4_PROTOCOL_VERSION = "operating_decision_stage4_v1"
EXPECTED_STAGE4_PROTOCOL_DIGEST = (
    "0c87ccfdbec5b1047490a3e2408b2a599cfa71687f18ef55076724108e5b3b68"
)
EXPECTED_STAGE4_SOURCE_REVISION = "86205dacbfecec694e4ace233aa974ca739041d1"
MISMATCH_GUARDED_SELECTOR = "mismatch_guarded_selector_v2"
EARLY_MISMATCH_ABSTENTION = "early_mismatch_abstention"
GUARD_DEVELOPMENT_SPLIT = "guard_development"
RECALIBRATION_SPLIT = "selector_recalibration"
FINAL_EVALUATION_SPLIT = "reserved_stage5_evaluation"
FINAL_PROCEDURES = (
    STOP_NOW,
    FIXED_THERMAL,
    FIXED_VOLTAGE,
    FIXED_FACE_TEMPERATURE,
    DECISION_DIRECTED_SELECTOR,
    MISMATCH_GUARDED_SELECTOR,
)


def _partition(trial_count: int, first_seed: int) -> OperatingDecisionRealismConfig:
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
class RevisedSelectorRule:
    """The one frozen Stage 5 revision to the Stage 4 selector."""

    mismatch_policy: str = EARLY_MISMATCH_ABSTENTION
    resolved_policy: str = STOP_NOW
    fallback_policy: str = FIXED_VOLTAGE

    def __post_init__(self) -> None:
        if self.mismatch_policy != EARLY_MISMATCH_ABSTENTION:
            raise ValueError("the mismatch action must be immediate abstention")
        if self.resolved_policy != STOP_NOW:
            raise ValueError("the resolved action must remain stop-now")
        if self.fallback_policy != FIXED_VOLTAGE:
            raise ValueError("the fallback action must remain voltage")


@dataclass(frozen=True)
class OperatingDecisionFinalConfig:
    """Frozen partitions and targets for the single selector revision."""

    guard_development: OperatingDecisionRealismConfig = _partition(20, 50_191_001)
    recalibration: OperatingDecisionRealismConfig = _partition(20, 60_191_001)
    evaluation: OperatingDecisionRealismConfig = _partition(50, 30_191_001)
    guard_target_block_retention: float = 0.90
    target_block_coverage: float = 0.90
    selector: RevisedSelectorRule = RevisedSelectorRule()
    cost_scenarios: Tuple[CostScenario, ...] = DEFAULT_COST_SCENARIOS
    bootstrap_draws: int = 20_000
    bootstrap_seed: int = 70_191_001

    def __post_init__(self) -> None:
        for name, value in (
            ("guard target", self.guard_target_block_retention),
            ("coverage target", self.target_block_coverage),
        ):
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between zero and one")
        if (
            not isinstance(self.bootstrap_draws, int)
            or isinstance(self.bootstrap_draws, bool)
            or self.bootstrap_draws < 1_000
        ):
            raise ValueError("the paired bootstrap needs at least 1,000 draws")
        if (
            not isinstance(self.bootstrap_seed, int)
            or isinstance(self.bootstrap_seed, bool)
            or self.bootstrap_seed <= 0
        ):
            raise ValueError("the bootstrap seed must be a positive integer")
        _validate_matching_physics(self.guard_development, self.recalibration)
        _validate_matching_physics(self.recalibration, self.evaluation)
        conformal_rank(
            self.guard_development.sensor.trial_count,
            self.guard_target_block_retention,
        )
        conformal_rank(
            self.recalibration.sensor.trial_count,
            self.target_block_coverage,
        )
        old_partitions = (
            _partition(10, 191_001),
            _partition(20, 10_191_001),
            _partition(10, 20_191_001),
        )
        partitions = (
            *old_partitions,
            self.guard_development,
            self.recalibration,
            self.evaluation,
        )
        namespaces = tuple(_seed_namespace(item) for item in partitions)
        if any(
            namespaces[left].intersection(namespaces[right])
            for left in range(len(namespaces))
            for right in range(left)
        ):
            raise ValueError("development, calibration, and Stage 5 seeds overlap")
        if self.bootstrap_seed in set().union(*namespaces):
            raise ValueError("bootstrap and physical random-number seeds overlap")
        scenarios = tuple(self.cost_scenarios)
        object.__setattr__(self, "cost_scenarios", scenarios)
        if not scenarios or len({item.name for item in scenarios}) != len(scenarios):
            raise ValueError("cost scenarios must be nonempty and uniquely named")


class AcquisitionAdequacySignal(NamedTuple):
    """Selector input derived only from fits to the common acquisition."""

    candidate_models: Tuple[str, ...]
    candidate_data_scores: Tuple[Tuple[str, float], ...]
    best_candidate_data_score: float
    failed_fit_count: int
    bound_hit_count: int
    margin_envelope: Optional[MarginEnvelope]


class AcquisitionGuardCalibration(NamedTuple):
    target_matched_block_retention: float
    block_count: int
    conformal_rank: int
    matched_block_scores: Tuple[float, ...]
    threshold: float


class RevisedCalibrationArtifact(NamedTuple):
    protocol_version: str
    protocol_digest: str
    implementation_revision: str
    realism_protocol_digest: str
    parent_stage4_protocol_version: str
    parent_stage4_protocol_digest: str
    guard_development_first_seed: int
    guard_development_block_count: int
    recalibration_first_seed: int
    recalibration_block_count: int
    evaluation_first_seed: int
    evaluation_block_count: int
    cost_scenarios: Tuple[CostScenario, ...]
    bootstrap_draws: int
    bootstrap_seed: int
    guard: AcquisitionGuardCalibration
    selector: RevisedSelectorRule
    procedure_calibration: ProcedureCalibration
    parent_selector_padding_floor: float


class GuardDevelopmentSummary(NamedTuple):
    truth_condition: str
    row_count: int
    triggered_count: int
    selected_voltage_count: int
    triggered_then_parent_insufficient_count: int


class RevisedCalibrationResult(NamedTuple):
    config: OperatingDecisionFinalConfig
    artifact: RevisedCalibrationArtifact
    guard_development: Tuple[GuardDevelopmentSummary, ...]
    recalibration_outcomes: Tuple[ProcedureOutcome, ...]
    summaries: Tuple[ProcedureSummary, ...]


class PairedLossComparison(NamedTuple):
    scenario_name: str
    revised_minus_parent_mean: float
    lower_95: float
    upper_95: float
    block_count: int


class FinalEvaluationResult(NamedTuple):
    config: OperatingDecisionFinalConfig
    artifact: RevisedCalibrationArtifact
    outcomes: Tuple[ProcedureOutcome, ...]
    summaries: Tuple[ProcedureSummary, ...]
    paired_loss_comparisons: Tuple[PairedLossComparison, ...]
    guard_summaries: Tuple[GuardDevelopmentSummary, ...]


def acquisition_data_score(
    fit: RealisticCandidateFit,
    config: OperatingDecisionRealismConfig,
) -> float:
    """Recover the data-only normalized lack-of-fit score from a fitted objective.

    The fit objective includes parameter-prior residuals and divides by every
    residual.  The guard removes those prior terms and divides by the actual
    observation count while retaining the two profiled run-bias penalties.
    """

    if fit.model_name not in MODEL_NAMES or fit.face_sensor is not None:
        raise ValueError("the acquisition guard needs an uninstrumented Stage 4 fit")
    spec = _parameter_spec(fit.model_name, False, config)
    if fit.parameter_names != spec.names:
        raise ValueError("fit parameters do not match the acquisition score")
    sample_count = len(
        regular_measurement_times(
            RUN_DURATION_SECONDS,
            config.sensor.sampling_interval,
        )
    )
    observation_count = 2 * sample_count
    profiled_bias_count = 2
    full_residual_count = observation_count + profiled_bias_count + len(spec.names)
    prior_sum = sum(
        (value / scale) ** 2
        for value, scale in zip(
            fit.log_multipliers,
            spec.prior_log_standard_deviations,
        )
    )
    data_sum = fit.objective * full_residual_count - prior_sum
    tolerance = 1.0e-10 * max(1.0, abs(fit.objective * full_residual_count))
    if not math.isfinite(data_sum) or data_sum < -tolerance:
        raise ValueError("fit objective cannot produce a finite acquisition score")
    return max(0.0, data_sum) / observation_count


def acquisition_adequacy_signal(
    fit_set: RealisticAcquisitionFitSet,
    config: OperatingDecisionRealismConfig,
) -> AcquisitionAdequacySignal:
    """Build the revised selector signal without later observations or truth."""

    initial = initial_decision_signal(
        fit_set,
        _final_regime_for(config),
        config,
    )
    scores_by_model = {}
    for fit in fit_set.fits:
        try:
            score = acquisition_data_score(fit, config)
        except (ArithmeticError, ValueError):
            score = math.inf
        if fit.reached_bound or not math.isfinite(score):
            score = math.inf
        scores_by_model[fit.model_name] = score
    scores = tuple(
        (model_name, scores_by_model.get(model_name, math.inf))
        for model_name in sorted(MODEL_NAMES)
    )
    best = min((value for _, value in scores), default=math.inf)
    return AcquisitionAdequacySignal(
        candidate_models=initial.candidate_models,
        candidate_data_scores=scores,
        best_candidate_data_score=best,
        failed_fit_count=initial.failed_fit_count,
        bound_hit_count=initial.bound_hit_count,
        margin_envelope=initial.margin_envelope,
    )


def select_revised_action(
    signal: AcquisitionAdequacySignal,
    guard: AcquisitionGuardCalibration,
    rule: RevisedSelectorRule = RevisedSelectorRule(),
) -> str:
    """Choose immediate abstention, stop, or voltage from acquisition fits."""

    if (
        not math.isfinite(signal.best_candidate_data_score)
        or signal.best_candidate_data_score > guard.threshold
    ):
        return rule.mismatch_policy
    stage4_signal = InitialDecisionSignal(
        candidate_models=signal.candidate_models,
        failed_fit_count=signal.failed_fit_count,
        bound_hit_count=signal.bound_hit_count,
        margin_envelope=signal.margin_envelope,
    )
    if (
        stage4_signal.margin_envelope is not None
        and classify_margin_envelope(stage4_signal.margin_envelope)
        != INSUFFICIENT_EVIDENCE
    ):
        return rule.resolved_policy
    return rule.fallback_policy


def load_stage4_calibration_artifact(
    path: Path | str,
    *,
    expected_digest: Optional[str] = EXPECTED_STAGE4_PROTOCOL_DIGEST,
) -> CalibrationArtifact:
    """Load and validate the committed Stage 4 JSON artifact."""

    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported Stage 4 artifact schema")
    if payload.get("protocol_version") != EXPECTED_STAGE4_PROTOCOL_VERSION:
        raise ValueError("the parent artifact is not the frozen Stage 4 protocol")
    if expected_digest is not None and payload.get("protocol_digest") != expected_digest:
        raise ValueError("the parent Stage 4 protocol digest changed")
    if payload.get("source_revision") != EXPECTED_STAGE4_SOURCE_REVISION:
        raise ValueError("the parent Stage 4 implementation revision changed")
    partitions = payload["partitions"]
    gates = tuple(
        VerificationGate(
            **{
                **item,
                "matched_block_scores": tuple(item["matched_block_scores"]),
            }
        )
        for item in payload["verification_gates"]
    )
    calibrations = tuple(
        ProcedureCalibration(
            **{
                **item,
                "block_nonconformity": tuple(item["block_nonconformity"]),
            }
        )
        for item in payload["procedure_calibrations"]
    )
    artifact = CalibrationArtifact(
        protocol_version=payload["protocol_version"],
        protocol_digest=payload["protocol_digest"],
        realism_protocol_digest=payload["realism_protocol_digest"],
        gate_development_first_seed=partitions["gate_development"]["first_seed"],
        gate_development_block_count=partitions["gate_development"][
            "paired_blocks_per_family"
        ],
        calibration_first_seed=partitions["calibration"]["first_seed"],
        calibration_block_count=partitions["calibration"]["paired_blocks_per_family"],
        rehearsal_first_seed=partitions["rehearsal"]["first_seed"],
        rehearsal_block_count=partitions["rehearsal"]["paired_blocks_per_family"],
        reserved_evaluation_first_seed=partitions["reserved_evaluation"]["first_seed"],
        reserved_evaluation_block_count=partitions["reserved_evaluation"][
            "paired_blocks_per_family"
        ],
        verification_gates=gates,
        selector=SelectorRule(**payload["selector"]),
        procedure_calibrations=calibrations,
    )
    recomputed_digest = _stage4_protocol_digest(
        OperatingDecisionCalibrationConfig(),
        gates,
        calibrations,
    )
    if recomputed_digest != artifact.protocol_digest:
        raise ValueError("the parent Stage 4 artifact contents do not match its digest")
    if (
        artifact.reserved_evaluation_first_seed != 30_191_001
        or artifact.reserved_evaluation_block_count != 50
        or partitions["reserved_evaluation"].get("instantiated_in_stage4") is not False
    ):
        raise ValueError("the parent artifact does not preserve the reserved Stage 5 cohort")
    return artifact


def _truth_worker(args):
    truth, config = args
    return truth, run_operating_decision_realism_truth(truth, config)


def _run_truths(
    split: str,
    truths: Sequence[str],
    config: OperatingDecisionRealismConfig,
    *,
    workers: int,
    progress: Optional[Callable[[str], None]],
) -> Tuple[ScoredOperatingDecision, ...]:
    truths = tuple(truths)
    if (
        not truths
        or len(set(truths)) != len(truths)
        or any(item not in STAGE3_TRUTH_CONDITIONS for item in truths)
    ):
        raise ValueError("generation needs distinct declared truth families")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise ValueError("worker count must be a positive integer")
    completed: Dict[str, Tuple[ScoredOperatingDecision, ...]] = {}
    if workers == 1:
        for truth in truths:
            if progress is not None:
                progress(f"{split}: starting {truth}")
            completed[truth] = run_operating_decision_realism_truth(
                truth,
                config,
                progress=(
                    (lambda message, split=split: progress(f"{split}: {message}"))
                    if progress is not None
                    else None
                ),
            )
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(truths))) as executor:
            futures = {
                executor.submit(_truth_worker, (truth, config)): truth
                for truth in truths
            }
            for future in as_completed(futures):
                truth = futures[future]
                _, trials = future.result()
                completed[truth] = trials
                if progress is not None:
                    progress(f"{split}: completed {truth}")
    return tuple(trial for truth in truths for trial in completed[truth])


def _partial_index(
    trials: Sequence[ScoredOperatingDecision],
    truths: Sequence[str],
    config: OperatingDecisionRealismConfig,
) -> Dict[Tuple[str, int, str], ScoredOperatingDecision]:
    truths = tuple(truths)
    policies = default_fixed_policies()
    expected = {
        (truth, trial_index, policy.name)
        for truth in truths
        for trial_index in range(config.sensor.trial_count)
        for policy in policies
    }
    indexed = {
        (item.truth_condition, item.saved.trial_index, item.saved.policy_name): item
        for item in trials
    }
    if len(indexed) != len(tuple(trials)) or set(indexed) != expected:
        raise ValueError("guard-development trials are missing, duplicated, or unknown")
    return indexed


def calibrate_acquisition_guard(
    matched_trials: Sequence[ScoredOperatingDecision],
    config: OperatingDecisionFinalConfig,
) -> AcquisitionGuardCalibration:
    """Calibrate best-candidate acquisition fit on Families A and B only."""

    matched_truths = STAGE3_TRUTH_CONDITIONS[:2]
    indexed = _partial_index(
        matched_trials,
        matched_truths,
        config.guard_development,
    )
    block_scores = []
    for trial_index in range(config.guard_development.sensor.trial_count):
        family_scores = []
        for truth in matched_truths:
            fit_set = _fit_set_from_stop_trial(indexed[(truth, trial_index, STOP_NOW)])
            signal = acquisition_adequacy_signal(
                fit_set,
                config.guard_development,
            )
            if not math.isfinite(signal.best_candidate_data_score):
                raise ValueError("matched guard development produced no finite candidate")
            family_scores.append(signal.best_candidate_data_score)
        block_scores.append(max(family_scores))
    rank, threshold = block_conformal_padding(
        block_scores,
        config.guard_target_block_retention,
    )
    return AcquisitionGuardCalibration(
        target_matched_block_retention=config.guard_target_block_retention,
        block_count=len(block_scores),
        conformal_rank=rank,
        matched_block_scores=tuple(block_scores),
        threshold=threshold,
    )


def _revised_choices(
    raw_index: Dict[Tuple[str, int, str], ScoredOperatingDecision],
    config: OperatingDecisionRealismConfig,
    guard: AcquisitionGuardCalibration,
    rule: RevisedSelectorRule,
) -> Dict[Tuple[str, int], str]:
    choices = {}
    for truth in STAGE3_TRUTH_CONDITIONS:
        for trial_index in range(config.sensor.trial_count):
            fit_set = _fit_set_from_stop_trial(
                raw_index[(truth, trial_index, STOP_NOW)]
            )
            signal = acquisition_adequacy_signal(fit_set, config)
            choices[(truth, trial_index)] = select_revised_action(
                signal,
                guard,
                rule,
            )
    return choices


def _early_abstention_trial(trial: ScoredOperatingDecision) -> ScoredOperatingDecision:
    """Represent the saved one-acquisition abstention without later diagnostics."""

    if trial.saved.policy_name != STOP_NOW or trial.acquisition_run_count != 1:
        raise ValueError("early abstention must start from the common acquisition")
    acquisition_failures = tuple(
        item for item in trial.saved.failures if item.stage == "acquisition_fit"
    )
    # The simulator times the complete decision as one span, so acquisition-only
    # compute time is unavailable.  Retaining the full stop-trial time is a
    # conservative upper bound; reports do not use it for the resource claim.
    saved = SavedOperatingDecision(
        case_id=trial.saved.case_id,
        decision=INSUFFICIENT_EVIDENCE,
        decision_reason="acquisition_mismatch_guard",
        margin_envelope=None,
        model_intervals=(),
        verifications=(),
        failures=acquisition_failures,
        decision_computation_seconds=trial.saved.decision_computation_seconds,
    )
    return trial._replace(
        saved=saved,
        false_approval=False,
        false_rejection=False,
        interval_covered=None,
        acquisition_run_count=1,
        diagnostic_run_count=1,
        energized_schedule_time_seconds=RUN_DURATION_SECONDS,
        verification_energy=0.0,
        total_diagnostic_energy=trial.acquisition_energy,
        extra_sensor_count=0,
    )


def _revised_raw_trial(
    truth: str,
    trial_index: int,
    choice: str,
    raw_index: Dict[Tuple[str, int, str], ScoredOperatingDecision],
    gated: Dict[Tuple[str, int, str], ScoredOperatingDecision],
) -> ScoredOperatingDecision:
    if choice == EARLY_MISMATCH_ABSTENTION:
        return _early_abstention_trial(raw_index[(truth, trial_index, STOP_NOW)])
    if choice not in (STOP_NOW, FIXED_VOLTAGE):
        raise ValueError("the revised selector chose an unknown action")
    return gated[(truth, trial_index, choice)]


def _reference_outcomes(
    trials: Sequence[ScoredOperatingDecision],
    parent: CalibrationArtifact,
    config: OperatingDecisionRealismConfig,
    *,
    split: str,
) -> Tuple[ProcedureOutcome, ...]:
    raw_index = _validate_paired_trials(trials, config)
    gated = _gate_trials(trials, parent.verification_gates, config)
    choices = _selector_choices(raw_index, config, parent.selector)
    outcomes = []
    for truth in STAGE3_TRUTH_CONDITIONS:
        for trial_index in range(config.sensor.trial_count):
            for procedure in STAGE4_PROCEDURES:
                selected = (
                    choices[(truth, trial_index)]
                    if procedure == DECISION_DIRECTED_SELECTOR
                    else procedure
                )
                raw = gated[(truth, trial_index, selected)]
                padding = parent.procedure_for(procedure).additive_margin_padding
                calibrated = apply_margin_padding(raw, padding)
                outcomes.append(
                    _compact_outcome(
                        split,
                        procedure,
                        selected,
                        raw,
                        calibrated,
                        padding,
                    )
                )
    return tuple(outcomes)


def _calibrate_revised_procedure(
    trials: Sequence[ScoredOperatingDecision],
    parent: CalibrationArtifact,
    config: OperatingDecisionFinalConfig,
    guard: AcquisitionGuardCalibration,
) -> ProcedureCalibration:
    raw_index = _validate_paired_trials(trials, config.recalibration)
    gated = _gate_trials(trials, parent.verification_gates, config.recalibration)
    choices = _revised_choices(
        raw_index,
        config.recalibration,
        guard,
        config.selector,
    )
    block_scores = []
    emitted = 0
    raw_covered = 0
    for trial_index in range(config.recalibration.sensor.trial_count):
        family_scores = []
        for truth in STAGE3_TRUTH_CONDITIONS:
            raw = _revised_raw_trial(
                truth,
                trial_index,
                choices[(truth, trial_index)],
                raw_index,
                gated,
            )
            family_scores.append(_outside_interval_miss(raw))
            if raw.saved.margin_envelope is not None:
                emitted += 1
                raw_covered += int(raw.interval_covered is True)
        block_scores.append(max(family_scores))
    rank, ranked_padding = block_conformal_padding(
        block_scores,
        config.target_block_coverage,
    )
    parent_floor = parent.procedure_for(
        DECISION_DIRECTED_SELECTOR
    ).additive_margin_padding
    padding = max(ranked_padding, parent_floor)
    calibrated_covered = 0
    for trial_index in range(config.recalibration.sensor.trial_count):
        for truth in STAGE3_TRUTH_CONDITIONS:
            raw = _revised_raw_trial(
                truth,
                trial_index,
                choices[(truth, trial_index)],
                raw_index,
                gated,
            )
            calibrated = apply_margin_padding(raw, padding)
            calibrated_covered += int(calibrated.interval_covered is True)
    return ProcedureCalibration(
        procedure_name=MISMATCH_GUARDED_SELECTOR,
        block_count=len(block_scores),
        conformal_rank=rank,
        target_block_coverage=config.target_block_coverage,
        additive_margin_padding=padding,
        block_nonconformity=tuple(block_scores),
        emitted_interval_count=emitted,
        raw_interval_covered_count=raw_covered,
        calibrated_interval_covered_count=calibrated_covered,
    )


def _revised_outcomes(
    trials: Sequence[ScoredOperatingDecision],
    parent: CalibrationArtifact,
    config: OperatingDecisionRealismConfig,
    guard: AcquisitionGuardCalibration,
    rule: RevisedSelectorRule,
    calibration: ProcedureCalibration,
    *,
    split: str,
) -> Tuple[ProcedureOutcome, ...]:
    raw_index = _validate_paired_trials(trials, config)
    gated = _gate_trials(trials, parent.verification_gates, config)
    choices = _revised_choices(raw_index, config, guard, rule)
    outcomes = []
    for truth in STAGE3_TRUTH_CONDITIONS:
        for trial_index in range(config.sensor.trial_count):
            selected = choices[(truth, trial_index)]
            raw = _revised_raw_trial(
                truth,
                trial_index,
                selected,
                raw_index,
                gated,
            )
            calibrated = apply_margin_padding(
                raw,
                calibration.additive_margin_padding,
            )
            outcomes.append(
                _compact_outcome(
                    split,
                    MISMATCH_GUARDED_SELECTOR,
                    selected,
                    raw,
                    calibrated,
                    calibration.additive_margin_padding,
                )
            )
    return tuple(outcomes)


def _artifact_digest(
    *,
    source_revision: str,
    parent: CalibrationArtifact,
    config: OperatingDecisionFinalConfig,
    guard: AcquisitionGuardCalibration,
    calibration: ProcedureCalibration,
) -> str:
    material = repr(
        (
            STAGE5_PROTOCOL_VERSION,
            source_revision,
            parent.protocol_version,
            parent.protocol_digest,
            _realism_protocol_digest(config.recalibration),
            (
                config.guard_development.sensor.first_seed,
                config.guard_development.sensor.trial_count,
                config.recalibration.sensor.first_seed,
                config.recalibration.sensor.trial_count,
                config.evaluation.sensor.first_seed,
                config.evaluation.sensor.trial_count,
            ),
            config.guard_target_block_retention,
            config.target_block_coverage,
            config.selector,
            config.cost_scenarios,
            (config.bootstrap_draws, config.bootstrap_seed),
            guard,
            calibration,
        )
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _build_artifact(
    *,
    source_revision: str,
    parent: CalibrationArtifact,
    config: OperatingDecisionFinalConfig,
    guard: AcquisitionGuardCalibration,
    calibration: ProcedureCalibration,
) -> RevisedCalibrationArtifact:
    if not isinstance(source_revision, str) or not source_revision.strip():
        raise ValueError("the frozen artifact needs its implementation revision")
    _validate_frozen_components(parent, config, guard, calibration)
    digest = _artifact_digest(
        source_revision=source_revision,
        parent=parent,
        config=config,
        guard=guard,
        calibration=calibration,
    )
    return RevisedCalibrationArtifact(
        protocol_version=STAGE5_PROTOCOL_VERSION,
        protocol_digest=digest,
        implementation_revision=source_revision,
        realism_protocol_digest=_realism_protocol_digest(config.recalibration),
        parent_stage4_protocol_version=parent.protocol_version,
        parent_stage4_protocol_digest=parent.protocol_digest,
        guard_development_first_seed=config.guard_development.sensor.first_seed,
        guard_development_block_count=config.guard_development.sensor.trial_count,
        recalibration_first_seed=config.recalibration.sensor.first_seed,
        recalibration_block_count=config.recalibration.sensor.trial_count,
        evaluation_first_seed=config.evaluation.sensor.first_seed,
        evaluation_block_count=config.evaluation.sensor.trial_count,
        cost_scenarios=config.cost_scenarios,
        bootstrap_draws=config.bootstrap_draws,
        bootstrap_seed=config.bootstrap_seed,
        guard=guard,
        selector=config.selector,
        procedure_calibration=calibration,
        parent_selector_padding_floor=parent.procedure_for(
            DECISION_DIRECTED_SELECTOR
        ).additive_margin_padding,
    )


def _validate_frozen_components(
    parent: CalibrationArtifact,
    config: OperatingDecisionFinalConfig,
    guard: AcquisitionGuardCalibration,
    calibration: ProcedureCalibration,
) -> None:
    """Reject a self-consistent digest whose calibrated values break the protocol."""

    expected_guard_count = config.guard_development.sensor.trial_count
    if (
        guard.target_matched_block_retention != config.guard_target_block_retention
        or guard.block_count != expected_guard_count
        or len(guard.matched_block_scores) != expected_guard_count
    ):
        raise ValueError("guard calibration count or target changed")
    guard_rank, guard_threshold = block_conformal_padding(
        guard.matched_block_scores,
        config.guard_target_block_retention,
    )
    if guard.conformal_rank != guard_rank or guard.threshold != guard_threshold:
        raise ValueError("guard threshold is not its declared block order statistic")

    expected_calibration_count = config.recalibration.sensor.trial_count
    if (
        calibration.procedure_name != MISMATCH_GUARDED_SELECTOR
        or calibration.target_block_coverage != config.target_block_coverage
        or calibration.block_count != expected_calibration_count
        or len(calibration.block_nonconformity) != expected_calibration_count
    ):
        raise ValueError("procedure calibration identity, count, or target changed")
    calibration_rank, ranked_padding = block_conformal_padding(
        calibration.block_nonconformity,
        config.target_block_coverage,
    )
    parent_floor = parent.procedure_for(
        DECISION_DIRECTED_SELECTOR
    ).additive_margin_padding
    if not math.isfinite(parent_floor) or parent_floor < 0.0:
        raise ValueError("the parent selector padding floor is invalid")
    if calibration.conformal_rank != calibration_rank:
        raise ValueError("procedure calibration rank changed")
    if calibration.additive_margin_padding != max(ranked_padding, parent_floor):
        raise ValueError("procedure padding is not its order statistic with parent floor")

    emitted = calibration.emitted_interval_count
    raw_covered = calibration.raw_interval_covered_count
    calibrated_covered = calibration.calibrated_interval_covered_count
    maximum_rows = len(STAGE3_TRUTH_CONDITIONS) * expected_calibration_count
    counts = (emitted, raw_covered, calibrated_covered)
    if any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > maximum_rows
        for value in counts
    ):
        raise ValueError("procedure calibration coverage counts are invalid")
    if raw_covered > emitted or calibrated_covered > emitted:
        raise ValueError("covered intervals cannot exceed emitted intervals")
    if calibrated_covered < raw_covered:
        raise ValueError("widening cannot reduce calibration coverage")


def _guard_development_summaries(
    trials: Sequence[ScoredOperatingDecision],
    parent: CalibrationArtifact,
    config: OperatingDecisionRealismConfig,
    guard: AcquisitionGuardCalibration,
    rule: RevisedSelectorRule,
    *,
    split: str,
) -> Tuple[GuardDevelopmentSummary, ...]:
    indexed = _validate_paired_trials(trials, config)
    revised = _revised_choices(indexed, config, guard, rule)
    parent_choices = _selector_choices(indexed, config, parent.selector)
    parent_outcomes = _reference_outcomes(trials, parent, config, split=split)
    parent_index = {
        (item.truth_condition, item.trial_index): item
        for item in parent_outcomes
        if item.procedure_name == DECISION_DIRECTED_SELECTOR
    }
    summaries = []
    for truth in STAGE3_TRUTH_CONDITIONS:
        keys = tuple((truth, index) for index in range(config.sensor.trial_count))
        triggered = tuple(
            key
            for key in keys
            if revised[key] == EARLY_MISMATCH_ABSTENTION
        )
        triggered_voltage = tuple(
            key for key in triggered if parent_choices[key] == FIXED_VOLTAGE
        )
        summaries.append(
            GuardDevelopmentSummary(
                truth_condition=truth,
                row_count=len(keys),
                triggered_count=len(triggered),
                selected_voltage_count=len(triggered_voltage),
                triggered_then_parent_insufficient_count=sum(
                    parent_index[key].decision == INSUFFICIENT_EVIDENCE
                    for key in triggered_voltage
                ),
            )
        )
    return tuple(summaries)


def _summary_rate(
    numerator: int,
    denominator: int,
    *,
    descriptive_only: bool,
) -> RateEstimate:
    ordinary = rate_estimate(numerator, denominator)
    if not descriptive_only or ordinary.rate is None:
        return ordinary
    return RateEstimate(
        numerator=ordinary.numerator,
        denominator=ordinary.denominator,
        rate=ordinary.rate,
        lower_95=None,
        upper_95=None,
    )


def summarize_final_outcomes(
    outcomes: Sequence[ProcedureOutcome],
    config: OperatingDecisionFinalConfig,
) -> Tuple[ProcedureSummary, ...]:
    """Summarize reference and revised procedures without independent-row claims."""

    outcomes = tuple(outcomes)
    if not outcomes:
        raise ValueError("final summaries need outcomes")
    splits = tuple(sorted({item.split for item in outcomes}))
    summaries = []
    for split in splits:
        split_rows = tuple(item for item in outcomes if item.split == split)
        present_procedures = tuple(
            item for item in FINAL_PROCEDURES if any(
                row.procedure_name == item for row in split_rows
            )
        )
        if set(present_procedures) != set(FINAL_PROCEDURES):
            raise ValueError("final comparison is missing a declared procedure")
        stop_rows = tuple(
            item for item in split_rows if item.procedure_name == STOP_NOW
        )
        stop_energy = fmean(item.total_diagnostic_energy for item in stop_rows)
        for truth in (*STAGE3_TRUTH_CONDITIONS, ALL_FAMILIES):
            for procedure in present_procedures:
                selected = tuple(
                    item
                    for item in split_rows
                    if item.procedure_name == procedure
                    and (truth == ALL_FAMILIES or item.truth_condition == truth)
                )
                if not selected:
                    raise ValueError("final summary cell is empty")
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
                descriptive = split == RECALIBRATION_SPLIT or truth == ALL_FAMILIES
                selected_actions = tuple(sorted({item.selected_policy for item in selected}))
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
                            descriptive_only=descriptive,
                        ),
                        missed_violations=_summary_rate(
                            sum(item.false_approval for item in selected),
                            true_violating,
                            descriptive_only=descriptive,
                        ),
                        false_rejections=_summary_rate(
                            sum(item.false_rejection for item in selected),
                            rejections,
                            descriptive_only=descriptive,
                        ),
                        decision_coverage=_summary_rate(
                            approvals + rejections,
                            len(selected),
                            descriptive_only=descriptive,
                        ),
                        abstentions=_summary_rate(
                            insufficient,
                            len(selected),
                            descriptive_only=descriptive,
                        ),
                        raw_interval_coverage=_summary_rate(
                            sum(item.raw_interval_covered is True for item in raw_intervals),
                            len(raw_intervals),
                            descriptive_only=descriptive,
                        ),
                        calibrated_interval_coverage=_summary_rate(
                            sum(
                                item.calibrated_interval_covered is True
                                for item in calibrated_intervals
                            ),
                            len(calibrated_intervals),
                            descriptive_only=descriptive,
                        ),
                        simultaneous_block_coverage=_summary_rate(
                            sum(simultaneous),
                            len(simultaneous),
                            descriptive_only=split == RECALIBRATION_SPLIT,
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
                        selected_policy_counts=tuple(
                            (
                                action,
                                sum(item.selected_policy == action for item in selected),
                            )
                            for action in selected_actions
                        ),
                        expected_losses=tuple(
                            ScenarioLoss(
                                scenario.name,
                                fmean(
                                    expected_loss(
                                        item,
                                        scenario,
                                        stop_energy=stop_energy,
                                    )
                                    for item in selected
                                ),
                            )
                            for scenario in config.cost_scenarios
                        ),
                    )
                )
    return tuple(summaries)


def _validate_monotone_revision(outcomes: Sequence[ProcedureOutcome]) -> None:
    """Ensure v2 can only remove, never add or reverse, a Stage 4 decision."""

    rows = tuple(outcomes)
    parent = {
        (item.truth_condition, item.trial_index): item
        for item in rows
        if item.procedure_name == DECISION_DIRECTED_SELECTOR
    }
    revised = {
        (item.truth_condition, item.trial_index): item
        for item in rows
        if item.procedure_name == MISMATCH_GUARDED_SELECTOR
    }
    if not parent or set(parent) != set(revised):
        raise ValueError("monotone audit needs paired parent and revised outcomes")
    for key, item in revised.items():
        reference = parent[key]
        if item.decision != INSUFFICIENT_EVIDENCE and item.decision != reference.decision:
            raise ValueError("the revised selector introduced or reversed a decision")
        if item.selected_policy != EARLY_MISMATCH_ABSTENTION and (
            item.selected_policy != reference.selected_policy
        ):
            raise ValueError("an unguarded revised action differs from Stage 4")
        if item.selected_policy == EARLY_MISMATCH_ABSTENTION and (
            item.decision != INSUFFICIENT_EVIDENCE
            or item.diagnostic_run_count != 1
            or item.extra_sensor_count != 0
        ):
            raise ValueError("guarded rows must abstain after one acquisition")


def freeze_revised_selector(
    parent: CalibrationArtifact,
    *,
    source_revision: str,
    config: OperatingDecisionFinalConfig = OperatingDecisionFinalConfig(),
    workers: int = 1,
    progress: Optional[Callable[[str], None]] = None,
) -> RevisedCalibrationResult:
    """Freeze the guard and revised padding without generating Stage 5 data."""

    if parent.protocol_digest != EXPECTED_STAGE4_PROTOCOL_DIGEST:
        raise ValueError("selector revision requires the frozen Stage 4 artifact")
    if parent.realism_protocol_digest != _realism_protocol_digest(config.recalibration):
        raise ValueError("Stage 4 and revised calibration physics differ")
    matched_truths = STAGE3_TRUTH_CONDITIONS[:2]
    matched_development = _run_truths(
        GUARD_DEVELOPMENT_SPLIT,
        matched_truths,
        config.guard_development,
        workers=workers,
        progress=progress,
    )
    guard = calibrate_acquisition_guard(matched_development, config)
    if progress is not None:
        progress(
            f"acquisition mismatch guard frozen at {guard.threshold:.6f} "
            f"(rank {guard.conformal_rank}/{guard.block_count})"
        )
    # Family C is deliberately generated only after the matched-family guard
    # threshold has frozen.  It can diagnose guard power but cannot tune tau.
    family_c_development = _run_truths(
        GUARD_DEVELOPMENT_SPLIT,
        STAGE3_TRUTH_CONDITIONS[2:],
        config.guard_development,
        workers=workers,
        progress=progress,
    )
    development_trials = (*matched_development, *family_c_development)
    guard_summaries = _guard_development_summaries(
        development_trials,
        parent,
        config.guard_development,
        guard,
        config.selector,
        split=GUARD_DEVELOPMENT_SPLIT,
    )
    recalibration_trials = _run_truths(
        RECALIBRATION_SPLIT,
        STAGE3_TRUTH_CONDITIONS,
        config.recalibration,
        workers=workers,
        progress=progress,
    )
    calibration = _calibrate_revised_procedure(
        recalibration_trials,
        parent,
        config,
        guard,
    )
    artifact = _build_artifact(
        source_revision=source_revision,
        parent=parent,
        config=config,
        guard=guard,
        calibration=calibration,
    )
    if progress is not None:
        progress(f"revised calibration artifact frozen: {artifact.protocol_digest[:12]}")
    reference = _reference_outcomes(
        recalibration_trials,
        parent,
        config.recalibration,
        split=RECALIBRATION_SPLIT,
    )
    revised = _revised_outcomes(
        recalibration_trials,
        parent,
        config.recalibration,
        guard,
        config.selector,
        calibration,
        split=RECALIBRATION_SPLIT,
    )
    outcomes = (*reference, *revised)
    _validate_monotone_revision(outcomes)
    return RevisedCalibrationResult(
        config=config,
        artifact=artifact,
        guard_development=guard_summaries,
        recalibration_outcomes=outcomes,
        summaries=summarize_final_outcomes(outcomes, config),
    )


def _validate_revised_artifact(
    artifact: RevisedCalibrationArtifact,
    parent: CalibrationArtifact,
    config: OperatingDecisionFinalConfig,
) -> None:
    expected = _build_artifact(
        source_revision=artifact.implementation_revision,
        parent=parent,
        config=config,
        guard=artifact.guard,
        calibration=artifact.procedure_calibration,
    )
    if artifact != expected:
        raise ValueError("revised artifact digest or protocol fields changed")
    if (
        artifact.evaluation_first_seed != 30_191_001
        or artifact.evaluation_block_count != 50
        or config.evaluation.sensor.first_seed != artifact.evaluation_first_seed
        or config.evaluation.sensor.trial_count != artifact.evaluation_block_count
    ):
        raise ValueError("evaluation is not the long-reserved Stage 5 cohort")


def _paired_loss_comparisons(
    outcomes: Sequence[ProcedureOutcome],
    config: OperatingDecisionFinalConfig,
) -> Tuple[PairedLossComparison, ...]:
    rows = tuple(outcomes)
    stop_energy = fmean(
        item.total_diagnostic_energy
        for item in rows
        if item.procedure_name == STOP_NOW
    )
    indexed = {
        (item.procedure_name, item.truth_condition, item.trial_index): item
        for item in rows
    }
    comparisons = []
    for scenario_index, scenario in enumerate(config.cost_scenarios):
        differences = []
        for block in range(config.evaluation.sensor.trial_count):
            revised = fmean(
                expected_loss(
                    indexed[(MISMATCH_GUARDED_SELECTOR, truth, block)],
                    scenario,
                    stop_energy=stop_energy,
                )
                for truth in STAGE3_TRUTH_CONDITIONS
            )
            parent = fmean(
                expected_loss(
                    indexed[(DECISION_DIRECTED_SELECTOR, truth, block)],
                    scenario,
                    stop_energy=stop_energy,
                )
                for truth in STAGE3_TRUTH_CONDITIONS
            )
            differences.append(revised - parent)
        random_source = random.Random(config.bootstrap_seed + scenario_index * 10_000)
        bootstrap = sorted(
            fmean(
                differences[random_source.randrange(len(differences))]
                for _ in differences
            )
            for _ in range(config.bootstrap_draws)
        )
        lower_index = max(0, math.ceil(0.025 * len(bootstrap)) - 1)
        upper_index = min(len(bootstrap) - 1, math.ceil(0.975 * len(bootstrap)) - 1)
        comparisons.append(
            PairedLossComparison(
                scenario_name=scenario.name,
                revised_minus_parent_mean=fmean(differences),
                lower_95=bootstrap[lower_index],
                upper_95=bootstrap[upper_index],
                block_count=len(differences),
            )
        )
    return tuple(comparisons)


def evaluate_reserved_stage5(
    artifact_path: Path | str,
    parent: CalibrationArtifact,
    *,
    config: OperatingDecisionFinalConfig = OperatingDecisionFinalConfig(),
    workers: int = 1,
    progress: Optional[Callable[[str], None]] = None,
) -> FinalEvaluationResult:
    """Load a serialized artifact, then run the predeclared reserved cohort."""

    artifact = load_revised_artifact(artifact_path)
    _validate_revised_artifact(artifact, parent, config)
    trials = _run_truths(
        FINAL_EVALUATION_SPLIT,
        STAGE3_TRUTH_CONDITIONS,
        config.evaluation,
        workers=workers,
        progress=progress,
    )
    reference = _reference_outcomes(
        trials,
        parent,
        config.evaluation,
        split=FINAL_EVALUATION_SPLIT,
    )
    revised = _revised_outcomes(
        trials,
        parent,
        config.evaluation,
        artifact.guard,
        artifact.selector,
        artifact.procedure_calibration,
        split=FINAL_EVALUATION_SPLIT,
    )
    outcomes = (*reference, *revised)
    _validate_monotone_revision(outcomes)
    summaries = summarize_final_outcomes(outcomes, config)
    guard_summaries = _guard_development_summaries(
        trials,
        parent,
        config.evaluation,
        artifact.guard,
        artifact.selector,
        split=FINAL_EVALUATION_SPLIT,
    )
    return FinalEvaluationResult(
        config=config,
        artifact=artifact,
        outcomes=outcomes,
        summaries=summaries,
        paired_loss_comparisons=_paired_loss_comparisons(outcomes, config),
        guard_summaries=guard_summaries,
    )


def revised_artifact_payload(artifact: RevisedCalibrationArtifact) -> dict:
    return {
        "schema_version": 1,
        "protocol_version": artifact.protocol_version,
        "protocol_digest": artifact.protocol_digest,
        "implementation_revision": artifact.implementation_revision,
        "realism_protocol_digest": artifact.realism_protocol_digest,
        "parent_stage4": {
            "protocol_version": artifact.parent_stage4_protocol_version,
            "protocol_digest": artifact.parent_stage4_protocol_digest,
        },
        "partitions": {
            "guard_development": {
                "first_seed": artifact.guard_development_first_seed,
                "paired_blocks_per_family": artifact.guard_development_block_count,
            },
            "recalibration": {
                "first_seed": artifact.recalibration_first_seed,
                "paired_blocks_per_family": artifact.recalibration_block_count,
            },
            "reserved_stage5_evaluation": {
                "first_seed": artifact.evaluation_first_seed,
                "paired_blocks_per_family": artifact.evaluation_block_count,
                "instantiated_when_artifact_frozen": False,
            },
        },
        "analysis": {
            "cost_scenarios": [
                scenario.__dict__ for scenario in artifact.cost_scenarios
            ],
            "paired_block_bootstrap": {
                "draws": artifact.bootstrap_draws,
                "seed": artifact.bootstrap_seed,
            },
        },
        "guard": artifact.guard._asdict(),
        "selector": artifact.selector.__dict__,
        "procedure_calibration": artifact.procedure_calibration._asdict(),
        "parent_selector_padding_floor": artifact.parent_selector_padding_floor,
    }


def save_revised_artifact(
    artifact: RevisedCalibrationArtifact,
    path: Path | str,
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            revised_artifact_payload(artifact),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    return destination


def load_revised_artifact(path: Path | str) -> RevisedCalibrationArtifact:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported revised artifact schema")
    partitions = payload["partitions"]
    analysis = payload["analysis"]
    guard_payload = payload["guard"]
    calibration_payload = payload["procedure_calibration"]
    if partitions["reserved_stage5_evaluation"].get(
        "instantiated_when_artifact_frozen"
    ) is not False:
        raise ValueError("Stage 5 was not reserved when the artifact froze")
    return RevisedCalibrationArtifact(
        protocol_version=payload["protocol_version"],
        protocol_digest=payload["protocol_digest"],
        implementation_revision=payload["implementation_revision"],
        realism_protocol_digest=payload["realism_protocol_digest"],
        parent_stage4_protocol_version=payload["parent_stage4"]["protocol_version"],
        parent_stage4_protocol_digest=payload["parent_stage4"]["protocol_digest"],
        guard_development_first_seed=partitions["guard_development"]["first_seed"],
        guard_development_block_count=partitions["guard_development"][
            "paired_blocks_per_family"
        ],
        recalibration_first_seed=partitions["recalibration"]["first_seed"],
        recalibration_block_count=partitions["recalibration"][
            "paired_blocks_per_family"
        ],
        evaluation_first_seed=partitions["reserved_stage5_evaluation"]["first_seed"],
        evaluation_block_count=partitions["reserved_stage5_evaluation"][
            "paired_blocks_per_family"
        ],
        cost_scenarios=tuple(
            CostScenario(**item) for item in analysis["cost_scenarios"]
        ),
        bootstrap_draws=analysis["paired_block_bootstrap"]["draws"],
        bootstrap_seed=analysis["paired_block_bootstrap"]["seed"],
        guard=AcquisitionGuardCalibration(
            **{
                **guard_payload,
                "matched_block_scores": tuple(guard_payload["matched_block_scores"]),
            }
        ),
        selector=RevisedSelectorRule(**payload["selector"]),
        procedure_calibration=ProcedureCalibration(
            **{
                **calibration_payload,
                "block_nonconformity": tuple(
                    calibration_payload["block_nonconformity"]
                ),
            }
        ),
        parent_selector_padding_floor=payload["parent_selector_padding_floor"],
    )


__all__ = [
    "EARLY_MISMATCH_ABSTENTION",
    "FINAL_EVALUATION_SPLIT",
    "FINAL_PROCEDURES",
    "GUARD_DEVELOPMENT_SPLIT",
    "MISMATCH_GUARDED_SELECTOR",
    "RECALIBRATION_SPLIT",
    "AcquisitionAdequacySignal",
    "AcquisitionGuardCalibration",
    "FinalEvaluationResult",
    "GuardDevelopmentSummary",
    "OperatingDecisionFinalConfig",
    "PairedLossComparison",
    "RevisedCalibrationArtifact",
    "RevisedCalibrationResult",
    "RevisedSelectorRule",
    "acquisition_adequacy_signal",
    "acquisition_data_score",
    "calibrate_acquisition_guard",
    "evaluate_reserved_stage5",
    "freeze_revised_selector",
    "load_revised_artifact",
    "load_stage4_calibration_artifact",
    "revised_artifact_payload",
    "save_revised_artifact",
    "select_revised_action",
    "summarize_final_outcomes",
]
