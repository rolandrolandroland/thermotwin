"""Mismatch guard and sealed evaluation for the corrected replication.

This module implements the single selector revision chosen after the original
operating-decision audit.  A low-capacity acquisition lack-of-fit guard is
calibrated on matched Families A and B.  It may stop an experiment early with
insufficient evidence; otherwise the frozen parent stop-or-voltage selector is
unchanged.  The complete guarded procedure receives a fresh block-conformal
padding on all three families before the reserved cohort can be instantiated.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
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
    rate_estimate,
)
from .operating_decision_calibration import (
    ALL_FAMILIES,
    DECISION_DIRECTED_SELECTOR,
    STAGE4_PROCEDURES,
    CostScenario,
    InitialDecisionSignal,
    ProcedureCalibration,
    ProcedureOutcome,
    ProcedureSummary,
    ScenarioLoss,
    _fit_set_from_stop_trial,
    _outside_interval_miss,
    _selector_choices,
    _validate_paired_trials,
    apply_margin_padding,
    block_conformal_padding,
    expected_loss,
    initial_decision_signal,
)
from .operating_decision_diagnostics import corrected_partition_evidence_digest
from .operating_decision_random_streams import (
    CALIBRATION_STREAM,
    RANDOM_STREAM_PROTOCOL_VERSION,
    RandomStreamKey,
    calibration_stream,
)
from .operating_decision_provenance import verify_committed_artifact_chain
from .operating_decision_realism import (
    STAGE3_TRUTH_CONDITIONS,
    OperatingDecisionRealismConfig,
    RealisticAcquisitionFitSet,
    RealisticCandidateFit,
    _parameter_spec,
)
from .operating_decision_replication import (
    CORRECTED_REPLICATION_CONFIG,
    CorrectedPartition,
    CorrectedPartitionResult,
    _authorize_reserved_evaluation,
    _run_corrected_partition,
    corrected_partition_config,
    corrected_physical_protocol_digest,
)
from .operating_decision_replication_calibration import (
    CorrectedParentCalibrationArtifact,
    CorrectedParentCalibrationConfig,
    CorrectedPartitionIdentity,
    _compact_corrected_outcome,
    _gate_trials,
    _trials,
    corrected_parent_artifact_from_payload,
    corrected_parent_artifact_payload,
    load_corrected_parent_artifact,
    validate_corrected_parent_artifact,
)
from .operating_decision_replication_protocol import (
    CORRECTED_NUMERICAL_SOURCE_PATHS,
    CorrectedGeneratorFreeze,
    verify_corrected_generator_freeze,
)
from .sensor_model_discrimination import MODEL_NAMES


CORRECTED_GUARD_PROTOCOL_VERSION = "operating_decision_corrected_guard_v1"
MISMATCH_GUARDED_SELECTOR = "mismatch_guarded_selector_v2"
EARLY_MISMATCH_ABSTENTION = "early_mismatch_abstention"
GUARD_DEVELOPMENT_SPLIT = "r2_guard_development"
GUARD_CALIBRATION_SPLIT = "r2_guard_calibration"
FINAL_EVALUATION_SPLIT = "r2_reserved_evaluation"
CORRECTED_FINAL_PROCEDURES = (
    STOP_NOW,
    FIXED_THERMAL,
    FIXED_VOLTAGE,
    FIXED_FACE_TEMPERATURE,
    DECISION_DIRECTED_SELECTOR,
    MISMATCH_GUARDED_SELECTOR,
)


@dataclass(frozen=True)
class CorrectedSelectorRule:
    """The one predeclared revision to the frozen parent selector."""

    mismatch_policy: str = EARLY_MISMATCH_ABSTENTION
    resolved_policy: str = STOP_NOW
    fallback_policy: str = FIXED_VOLTAGE

    def __post_init__(self) -> None:
        if self.mismatch_policy != EARLY_MISMATCH_ABSTENTION:
            raise ValueError("corrected mismatch action must be immediate abstention")
        if self.resolved_policy != STOP_NOW:
            raise ValueError("corrected resolved action must remain stop-now")
        if self.fallback_policy != FIXED_VOLTAGE:
            raise ValueError("corrected fallback action must remain voltage")


@dataclass(frozen=True)
class CorrectedGuardCalibrationConfig:
    """Frozen statistical choices for guard calibration and final analysis."""

    guard_target_block_retention: float = 0.90
    target_block_coverage: float = 0.90
    confidence_level: float = 0.95
    primary_scenario_name: str = "balanced"
    selector: CorrectedSelectorRule = CorrectedSelectorRule()

    def __post_init__(self) -> None:
        for name, value in (
            ("guard target", self.guard_target_block_retention),
            ("coverage target", self.target_block_coverage),
            ("confidence level", self.confidence_level),
        ):
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between zero and one")
        if self.confidence_level <= 0.5:
            raise ValueError("confidence level must exceed one half")
        if (
            not isinstance(self.primary_scenario_name, str)
            or not self.primary_scenario_name
            or self.primary_scenario_name != self.primary_scenario_name.strip()
        ):
            raise ValueError("primary scenario name must be a nonempty label")


class AcquisitionAdequacySignal(NamedTuple):
    candidate_models: Tuple[str, ...]
    candidate_data_scores: Tuple[Tuple[str, float], ...]
    best_candidate_data_score: float
    failed_fit_count: int
    bound_hit_count: int
    nonconverged_fit_count: int
    margin_envelope: Optional[MarginEnvelope]


class AcquisitionGuardCalibration(NamedTuple):
    target_matched_block_retention: float
    block_count: int
    conformal_rank: int
    matched_block_scores: Tuple[float, ...]
    threshold: float


class CorrectedGuardCalibrationArtifact(NamedTuple):
    protocol_version: str
    protocol_digest: str
    generator_freeze_digest: str
    source_manifest_digest: str
    parent_protocol_digest: str
    physical_protocol_digest: str
    guard_development_evidence_digest: str
    guard_calibration_evidence_digest: str
    campaign: str
    guard_development: CorrectedPartitionIdentity
    guard_calibration: CorrectedPartitionIdentity
    reserved_evaluation: CorrectedPartitionIdentity
    guard_target_block_retention: float
    target_block_coverage: float
    confidence_level: float
    primary_scenario_name: str
    cost_scenarios: Tuple[CostScenario, ...]
    bootstrap_partition_name: str
    bootstrap_draws: int
    bootstrap_random_stream_keys: Tuple[RandomStreamKey, ...]
    guard: AcquisitionGuardCalibration
    selector: CorrectedSelectorRule
    procedure_calibration: ProcedureCalibration
    parent_selector_padding_floor: float


class GuardTriggerSummary(NamedTuple):
    split: str
    truth_condition: str
    row_count: int
    triggered_count: int
    triggered_from_parent_voltage_count: int
    triggered_then_parent_insufficient_count: int


class PairedLossComparison(NamedTuple):
    scenario_name: str
    revised_minus_parent_mean: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    independent_block_count: int
    bootstrap_draws: int
    random_stream_key: RandomStreamKey


class CorrectedGuardFreezeResult(NamedTuple):
    generator_freeze: CorrectedGeneratorFreeze
    parent_config: CorrectedParentCalibrationConfig
    parent: CorrectedParentCalibrationArtifact
    config: CorrectedGuardCalibrationConfig
    artifact: CorrectedGuardCalibrationArtifact
    guard_development: CorrectedPartitionResult
    guard_calibration: CorrectedPartitionResult
    calibration_outcomes: Tuple[ProcedureOutcome, ...]
    summaries: Tuple[ProcedureSummary, ...]
    guard_summaries: Tuple[GuardTriggerSummary, ...]


class CorrectedFinalEvaluationResult(NamedTuple):
    generator_freeze: CorrectedGeneratorFreeze
    parent_config: CorrectedParentCalibrationConfig
    parent: CorrectedParentCalibrationArtifact
    config: CorrectedGuardCalibrationConfig
    artifact: CorrectedGuardCalibrationArtifact
    source_commit: str
    evaluation: CorrectedPartitionResult
    outcomes: Tuple[ProcedureOutcome, ...]
    summaries: Tuple[ProcedureSummary, ...]
    guard_summaries: Tuple[GuardTriggerSummary, ...]
    paired_loss_comparisons: Tuple[PairedLossComparison, ...]
    primary_success: bool


class LoadedCorrectedGuardArtifact(NamedTuple):
    generator_freeze: CorrectedGeneratorFreeze
    parent_config: CorrectedParentCalibrationConfig
    parent: CorrectedParentCalibrationArtifact
    config: CorrectedGuardCalibrationConfig
    artifact: CorrectedGuardCalibrationArtifact


def acquisition_data_score(
    fit: RealisticCandidateFit,
    config: OperatingDecisionRealismConfig,
) -> float:
    """Recover normalized acquisition-data lack of fit without prior terms."""

    if fit.model_name not in MODEL_NAMES or fit.face_sensor is not None:
        raise ValueError("acquisition guard needs an uninstrumented candidate fit")
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
    residual_count = observation_count + profiled_bias_count + len(spec.names)
    prior_sum = sum(
        (value / scale) ** 2
        for value, scale in zip(
            fit.log_multipliers,
            spec.prior_log_standard_deviations,
        )
    )
    data_sum = fit.objective * residual_count - prior_sum
    tolerance = 1.0e-10 * max(1.0, abs(fit.objective * residual_count))
    if not math.isfinite(data_sum) or data_sum < -tolerance:
        raise ValueError("fit objective cannot produce a finite acquisition score")
    return max(0.0, data_sum) / observation_count


def acquisition_adequacy_signal(
    fit_set: RealisticAcquisitionFitSet,
    config: OperatingDecisionRealismConfig,
) -> AcquisitionAdequacySignal:
    """Build a guard signal from common acquisition fits only."""

    from .operating_decision_replication_calibration import _final_regime

    initial = initial_decision_signal(fit_set, _final_regime(config), config)
    scores_by_model = {}
    nonconverged = 0
    for fit in fit_set.fits:
        if not fit.converged:
            nonconverged += 1
        try:
            score = acquisition_data_score(fit, config)
        except (ArithmeticError, ValueError):
            score = math.inf
        if not fit.converged or fit.reached_bound or not math.isfinite(score):
            score = math.inf
        scores_by_model[fit.model_name] = score
    scores = tuple(
        (model_name, scores_by_model.get(model_name, math.inf))
        for model_name in sorted(MODEL_NAMES)
    )
    return AcquisitionAdequacySignal(
        candidate_models=initial.candidate_models,
        candidate_data_scores=scores,
        best_candidate_data_score=min(
            (value for _, value in scores),
            default=math.inf,
        ),
        failed_fit_count=initial.failed_fit_count,
        bound_hit_count=initial.bound_hit_count,
        nonconverged_fit_count=nonconverged,
        margin_envelope=initial.margin_envelope,
    )


def select_corrected_action(
    signal: AcquisitionAdequacySignal,
    guard: AcquisitionGuardCalibration,
    rule: CorrectedSelectorRule = CorrectedSelectorRule(),
) -> str:
    """Choose early abstention, stop, or voltage from acquisition evidence."""

    if (
        not math.isfinite(signal.best_candidate_data_score)
        or signal.best_candidate_data_score > guard.threshold
    ):
        return rule.mismatch_policy
    parent_signal = InitialDecisionSignal(
        candidate_models=signal.candidate_models,
        failed_fit_count=signal.failed_fit_count,
        bound_hit_count=signal.bound_hit_count,
        margin_envelope=signal.margin_envelope,
    )
    if (
        parent_signal.margin_envelope is not None
        and classify_margin_envelope(parent_signal.margin_envelope)
        != INSUFFICIENT_EVIDENCE
    ):
        return rule.resolved_policy
    return rule.fallback_policy


def calibrate_corrected_acquisition_guard(
    trials: Sequence[ScoredOperatingDecision],
    physical_config: OperatingDecisionRealismConfig,
    partition: CorrectedPartition,
    config: CorrectedGuardCalibrationConfig = CorrectedGuardCalibrationConfig(),
) -> AcquisitionGuardCalibration:
    """Calibrate the guard only on the two represented truth families."""

    if partition.name != GUARD_DEVELOPMENT_SPLIT:
        raise ValueError("guard calibration requires its named development partition")
    if physical_config.sensor.trial_count != partition.block_count:
        raise ValueError("guard physical config and semantic partition differ")
    indexed = _validate_paired_trials(trials, physical_config)
    block_scores = []
    for block in range(partition.block_count):
        matched_scores = []
        for truth in STAGE3_TRUTH_CONDITIONS[:2]:
            fit_set = _fit_set_from_stop_trial(indexed[(truth, block, STOP_NOW)])
            signal = acquisition_adequacy_signal(fit_set, physical_config)
            if not math.isfinite(signal.best_candidate_data_score):
                raise ValueError(
                    "matched guard development produced no reliable candidate"
                )
            matched_scores.append(signal.best_candidate_data_score)
        block_scores.append(max(matched_scores))
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
    physical_config: OperatingDecisionRealismConfig,
    guard: AcquisitionGuardCalibration,
    rule: CorrectedSelectorRule,
) -> Dict[Tuple[str, int], str]:
    choices = {}
    for truth in STAGE3_TRUTH_CONDITIONS:
        for block in range(physical_config.sensor.trial_count):
            fit_set = _fit_set_from_stop_trial(raw_index[(truth, block, STOP_NOW)])
            choices[(truth, block)] = select_corrected_action(
                acquisition_adequacy_signal(fit_set, physical_config),
                guard,
                rule,
            )
    return choices


def _early_abstention_trial(
    trial: ScoredOperatingDecision,
) -> ScoredOperatingDecision:
    if trial.saved.policy_name != STOP_NOW or trial.acquisition_run_count != 1:
        raise ValueError("early abstention must use the common acquisition")
    acquisition_failures = tuple(
        item for item in trial.saved.failures if item.stage == "acquisition_fit"
    )
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
    block: int,
    choice: str,
    raw_index: Dict[Tuple[str, int, str], ScoredOperatingDecision],
    gated: Dict[Tuple[str, int, str], ScoredOperatingDecision],
) -> ScoredOperatingDecision:
    if choice == EARLY_MISMATCH_ABSTENTION:
        return _early_abstention_trial(raw_index[(truth, block, STOP_NOW)])
    if choice not in (STOP_NOW, FIXED_VOLTAGE):
        raise ValueError("corrected selector chose an unknown action")
    return gated[(truth, block, choice)]


def _reference_outcomes(
    trials: Sequence[ScoredOperatingDecision],
    parent: CorrectedParentCalibrationArtifact,
    physical_config: OperatingDecisionRealismConfig,
    *,
    split: str,
) -> Tuple[ProcedureOutcome, ...]:
    raw_index = _validate_paired_trials(trials, physical_config)
    gated = _gate_trials(trials, parent.verification_gates, physical_config)
    choices = _selector_choices(raw_index, physical_config, parent.selector)
    outcomes = []
    for truth in STAGE3_TRUTH_CONDITIONS:
        for block in range(physical_config.sensor.trial_count):
            for procedure in STAGE4_PROCEDURES:
                selected = (
                    choices[(truth, block)]
                    if procedure == DECISION_DIRECTED_SELECTOR
                    else procedure
                )
                raw = gated[(truth, block, selected)]
                padding = parent.procedure_for(procedure).additive_margin_padding
                calibrated = apply_margin_padding(raw, padding)
                outcomes.append(
                    _compact_corrected_outcome(
                        split,
                        procedure,
                        selected,
                        raw,
                        calibrated,
                        padding,
                    )
                )
    return tuple(outcomes)


def calibrate_corrected_guarded_procedure(
    trials: Sequence[ScoredOperatingDecision],
    parent: CorrectedParentCalibrationArtifact,
    physical_config: OperatingDecisionRealismConfig,
    partition: CorrectedPartition,
    guard: AcquisitionGuardCalibration,
    config: CorrectedGuardCalibrationConfig = CorrectedGuardCalibrationConfig(),
) -> ProcedureCalibration:
    """Calibrate the full guarded selector on a disjoint three-family split."""

    if partition.name != GUARD_CALIBRATION_SPLIT:
        raise ValueError("guarded procedure requires its named calibration partition")
    if physical_config.sensor.trial_count != partition.block_count:
        raise ValueError("guard calibration config and semantic partition differ")
    raw_index = _validate_paired_trials(trials, physical_config)
    gated = _gate_trials(trials, parent.verification_gates, physical_config)
    choices = _revised_choices(raw_index, physical_config, guard, config.selector)
    block_scores = []
    emitted = 0
    raw_covered = 0
    for block in range(partition.block_count):
        family_scores = []
        for truth in STAGE3_TRUTH_CONDITIONS:
            raw = _revised_raw_trial(
                truth,
                block,
                choices[(truth, block)],
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
    for block in range(partition.block_count):
        for truth in STAGE3_TRUTH_CONDITIONS:
            raw = _revised_raw_trial(
                truth,
                block,
                choices[(truth, block)],
                raw_index,
                gated,
            )
            calibrated_covered += int(
                apply_margin_padding(raw, padding).interval_covered is True
            )
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
    parent: CorrectedParentCalibrationArtifact,
    physical_config: OperatingDecisionRealismConfig,
    guard: AcquisitionGuardCalibration,
    rule: CorrectedSelectorRule,
    calibration: ProcedureCalibration,
    *,
    split: str,
) -> Tuple[ProcedureOutcome, ...]:
    raw_index = _validate_paired_trials(trials, physical_config)
    gated = _gate_trials(trials, parent.verification_gates, physical_config)
    choices = _revised_choices(raw_index, physical_config, guard, rule)
    outcomes = []
    for truth in STAGE3_TRUTH_CONDITIONS:
        for block in range(physical_config.sensor.trial_count):
            selected = choices[(truth, block)]
            raw = _revised_raw_trial(
                truth,
                block,
                selected,
                raw_index,
                gated,
            )
            calibrated = apply_margin_padding(
                raw,
                calibration.additive_margin_padding,
            )
            outcomes.append(
                _compact_corrected_outcome(
                    split,
                    MISMATCH_GUARDED_SELECTOR,
                    selected,
                    raw,
                    calibrated,
                    calibration.additive_margin_padding,
                )
            )
    return tuple(outcomes)


def _combined_outcomes(
    trials: Sequence[ScoredOperatingDecision],
    parent: CorrectedParentCalibrationArtifact,
    physical_config: OperatingDecisionRealismConfig,
    artifact: CorrectedGuardCalibrationArtifact,
    *,
    split: str,
) -> Tuple[ProcedureOutcome, ...]:
    outcomes = (
        *_reference_outcomes(trials, parent, physical_config, split=split),
        *_revised_outcomes(
            trials,
            parent,
            physical_config,
            artifact.guard,
            artifact.selector,
            artifact.procedure_calibration,
            split=split,
        ),
    )
    _validate_monotone_revision(outcomes)
    return outcomes


def _validate_monotone_revision(outcomes: Sequence[ProcedureOutcome]) -> None:
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
        if item.decision != INSUFFICIENT_EVIDENCE and (
            item.decision != reference.decision
        ):
            raise ValueError("corrected revision introduced or reversed a decision")
        if item.selected_policy != EARLY_MISMATCH_ABSTENTION and (
            item.selected_policy != reference.selected_policy
        ):
            raise ValueError("unguarded revised action differs from the parent")
        if item.selected_policy == EARLY_MISMATCH_ABSTENTION and (
            item.decision != INSUFFICIENT_EVIDENCE
            or item.diagnostic_run_count != 1
            or item.extra_sensor_count != 0
        ):
            raise ValueError("guarded row must abstain after one acquisition")


def _guard_trigger_summaries(
    trials: Sequence[ScoredOperatingDecision],
    parent: CorrectedParentCalibrationArtifact,
    physical_config: OperatingDecisionRealismConfig,
    guard: AcquisitionGuardCalibration,
    rule: CorrectedSelectorRule,
    *,
    split: str,
) -> Tuple[GuardTriggerSummary, ...]:
    indexed = _validate_paired_trials(trials, physical_config)
    revised = _revised_choices(indexed, physical_config, guard, rule)
    parent_choices = _selector_choices(indexed, physical_config, parent.selector)
    parent_outcomes = _reference_outcomes(
        trials,
        parent,
        physical_config,
        split=split,
    )
    parent_index = {
        (item.truth_condition, item.trial_index): item
        for item in parent_outcomes
        if item.procedure_name == DECISION_DIRECTED_SELECTOR
    }
    summaries = []
    for truth in STAGE3_TRUTH_CONDITIONS:
        keys = tuple(
            (truth, block) for block in range(physical_config.sensor.trial_count)
        )
        triggered = tuple(
            key for key in keys if revised[key] == EARLY_MISMATCH_ABSTENTION
        )
        from_voltage = tuple(
            key for key in triggered if parent_choices[key] == FIXED_VOLTAGE
        )
        summaries.append(
            GuardTriggerSummary(
                split=split,
                truth_condition=truth,
                row_count=len(keys),
                triggered_count=len(triggered),
                triggered_from_parent_voltage_count=len(from_voltage),
                triggered_then_parent_insufficient_count=sum(
                    parent_index[key].decision == INSUFFICIENT_EVIDENCE
                    for key in from_voltage
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
    return ordinary._replace(lower_95=None, upper_95=None)


def summarize_corrected_final_outcomes(
    outcomes: Sequence[ProcedureOutcome],
    cost_scenarios: Sequence[CostScenario],
) -> Tuple[ProcedureSummary, ...]:
    """Summarize rows while respecting the independent paired-block unit."""

    rows = tuple(outcomes)
    scenarios = tuple(cost_scenarios)
    if not rows or not scenarios:
        raise ValueError("corrected final summary needs outcomes and cost scenarios")
    summaries = []
    for split in sorted({item.split for item in rows}):
        split_rows = tuple(item for item in rows if item.split == split)
        procedures = tuple(
            procedure
            for procedure in CORRECTED_FINAL_PROCEDURES
            if any(item.procedure_name == procedure for item in split_rows)
        )
        if set(procedures) != set(CORRECTED_FINAL_PROCEDURES):
            raise ValueError("corrected final comparison is missing a procedure")
        stop_index = {
            (item.truth_condition, item.trial_index): item.total_diagnostic_energy
            for item in split_rows
            if item.procedure_name == STOP_NOW
        }
        expected_stop_keys = {
            (item.truth_condition, item.trial_index) for item in split_rows
        }
        if set(stop_index) != expected_stop_keys:
            raise ValueError("corrected final loss has no paired stop baseline")
        for truth in (*STAGE3_TRUTH_CONDITIONS, ALL_FAMILIES):
            for procedure in procedures:
                selected = tuple(
                    item
                    for item in split_rows
                    if item.procedure_name == procedure
                    and (truth == ALL_FAMILIES or item.truth_condition == truth)
                )
                if not selected:
                    raise ValueError("corrected final summary cell is empty")
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
                            if item.trial_index == block
                        )
                        for block in block_ids
                    )
                else:
                    simultaneous = tuple(
                        item.calibrated_interval_covered is not False
                        for item in selected
                    )
                fitted_split = split == GUARD_CALIBRATION_SPLIT
                clustered_cell = truth == ALL_FAMILIES
                descriptive = fitted_split or clustered_cell
                selected_actions = tuple(
                    sorted({item.selected_policy for item in selected})
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
                            descriptive_only=fitted_split,
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
                                        stop_energy=stop_index[
                                            (
                                                item.truth_condition,
                                                item.trial_index,
                                            )
                                        ],
                                    )
                                    for item in selected
                                ),
                            )
                            for scenario in scenarios
                        ),
                    )
                )
    return tuple(summaries)


def _identity(partition: CorrectedPartition) -> CorrectedPartitionIdentity:
    return CorrectedPartitionIdentity(partition.name, partition.block_count)


def _bootstrap_keys(
    generator_freeze: CorrectedGeneratorFreeze,
    cost_scenarios: Sequence[CostScenario],
) -> Tuple[RandomStreamKey, ...]:
    plan = generator_freeze.plan
    return tuple(
        calibration_stream(
            campaign=plan.campaign,
            partition=plan.bootstrap_partition_name,
            purpose=f"paired_loss_bootstrap/{scenario.name}",
        ).key
        for scenario in cost_scenarios
    )


def _guard_digest_material(
    generator_freeze: CorrectedGeneratorFreeze,
    parent: CorrectedParentCalibrationArtifact,
    config: CorrectedGuardCalibrationConfig,
    guard_development_evidence_digest: str,
    guard_calibration_evidence_digest: str,
    guard: AcquisitionGuardCalibration,
    calibration: ProcedureCalibration,
    bootstrap_keys: Sequence[RandomStreamKey],
) -> tuple:
    plan = generator_freeze.plan
    return (
        CORRECTED_GUARD_PROTOCOL_VERSION,
        generator_freeze.artifact_digest,
        generator_freeze.source_manifest.digest,
        parent.protocol_digest,
        generator_freeze.physical_protocol_digest,
        guard_development_evidence_digest,
        guard_calibration_evidence_digest,
        plan.campaign,
        _identity(plan.partition(GUARD_DEVELOPMENT_SPLIT)),
        _identity(plan.partition(GUARD_CALIBRATION_SPLIT)),
        _identity(plan.partition(FINAL_EVALUATION_SPLIT)),
        config,
        parent.cost_scenarios,
        plan.bootstrap_partition_name,
        plan.bootstrap_draws,
        tuple(bootstrap_keys),
        guard,
        calibration,
        parent.procedure_for(
            DECISION_DIRECTED_SELECTOR
        ).additive_margin_padding,
    )


def _validate_guard_components(
    generator_freeze: CorrectedGeneratorFreeze,
    parent: CorrectedParentCalibrationArtifact,
    config: CorrectedGuardCalibrationConfig,
    guard_development_evidence_digest: str,
    guard_calibration_evidence_digest: str,
    guard: AcquisitionGuardCalibration,
    calibration: ProcedureCalibration,
    bootstrap_keys: Sequence[RandomStreamKey],
) -> None:
    plan = generator_freeze.plan
    guard_partition = plan.partition(GUARD_DEVELOPMENT_SPLIT)
    calibration_partition = plan.partition(GUARD_CALIBRATION_SPLIT)
    for name, digest in (
        ("guard-development evidence", guard_development_evidence_digest),
        ("guard-calibration evidence", guard_calibration_evidence_digest),
    ):
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{name} digest must be lowercase SHA-256")
    if parent.generator_freeze_digest != generator_freeze.artifact_digest:
        raise ValueError("guard parent and generator freeze differ")
    if parent.physical_protocol_digest != generator_freeze.physical_protocol_digest:
        raise ValueError("guard parent and physical protocol differ")
    if parent.campaign != plan.campaign:
        raise ValueError("guard parent and campaign differ")
    if config.primary_scenario_name not in {
        item.name for item in parent.cost_scenarios
    }:
        raise ValueError("primary scenario is absent from parent cost scenarios")
    if (
        parent.selector.resolved_policy != config.selector.resolved_policy
        or parent.selector.fallback_policy != config.selector.fallback_policy
    ):
        raise ValueError("guard would alter the untriggered parent action")
    if (
        guard.target_matched_block_retention
        != config.guard_target_block_retention
        or guard.block_count != guard_partition.block_count
        or len(guard.matched_block_scores) != guard_partition.block_count
        or any(not math.isfinite(value) for value in guard.matched_block_scores)
    ):
        raise ValueError("guard calibration metadata or scores changed")
    guard_rank, guard_threshold = block_conformal_padding(
        guard.matched_block_scores,
        config.guard_target_block_retention,
    )
    if guard.conformal_rank != guard_rank or guard.threshold != guard_threshold:
        raise ValueError("guard threshold is not its declared order statistic")
    if (
        calibration.procedure_name != MISMATCH_GUARDED_SELECTOR
        or calibration.block_count != calibration_partition.block_count
        or len(calibration.block_nonconformity) != calibration_partition.block_count
        or calibration.target_block_coverage != config.target_block_coverage
    ):
        raise ValueError("guarded procedure calibration metadata changed")
    rank, ranked_padding = block_conformal_padding(
        calibration.block_nonconformity,
        config.target_block_coverage,
    )
    parent_floor = parent.procedure_for(
        DECISION_DIRECTED_SELECTOR
    ).additive_margin_padding
    if calibration.conformal_rank != rank or (
        calibration.additive_margin_padding != max(ranked_padding, parent_floor)
    ):
        raise ValueError("guarded padding is not its order statistic with parent floor")
    maximum_rows = len(STAGE3_TRUTH_CONDITIONS) * calibration_partition.block_count
    counts = (
        calibration.emitted_interval_count,
        calibration.raw_interval_covered_count,
        calibration.calibrated_interval_covered_count,
    )
    if any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > maximum_rows
        for value in counts
    ):
        raise ValueError("guarded calibration coverage counts are invalid")
    if calibration.raw_interval_covered_count > calibration.emitted_interval_count:
        raise ValueError("raw coverage exceeds emitted guarded intervals")
    if (
        calibration.calibrated_interval_covered_count
        > calibration.emitted_interval_count
    ):
        raise ValueError("calibrated coverage exceeds emitted guarded intervals")
    if (
        calibration.calibrated_interval_covered_count
        < calibration.raw_interval_covered_count
    ):
        raise ValueError("widening reduced guarded calibration coverage")
    keys = tuple(bootstrap_keys)
    scenarios = tuple(parent.cost_scenarios)
    if len(keys) != len(scenarios) or len(set(keys)) != len(keys):
        raise ValueError("guard artifact needs one unique bootstrap stream per scenario")
    for key, scenario in zip(keys, scenarios):
        if (
            key.protocol_version != RANDOM_STREAM_PROTOCOL_VERSION
            or key.campaign != plan.campaign
            or key.partition != plan.bootstrap_partition_name
            or key.stream_kind != CALIBRATION_STREAM
            or key.block != 0
            or key.family is not None
            or key.run is not None
            or key.channel is not None
            or key.purpose != f"paired_loss_bootstrap/{scenario.name}"
        ):
            raise ValueError("bootstrap stream key differs from its semantic namespace")


def build_corrected_guard_artifact(
    generator_freeze: CorrectedGeneratorFreeze,
    parent: CorrectedParentCalibrationArtifact,
    config: CorrectedGuardCalibrationConfig,
    guard_development_evidence_digest: str,
    guard_calibration_evidence_digest: str,
    guard: AcquisitionGuardCalibration,
    calibration: ProcedureCalibration,
) -> CorrectedGuardCalibrationArtifact:
    keys = _bootstrap_keys(generator_freeze, parent.cost_scenarios)
    _validate_guard_components(
        generator_freeze,
        parent,
        config,
        guard_development_evidence_digest,
        guard_calibration_evidence_digest,
        guard,
        calibration,
        keys,
    )
    digest = hashlib.sha256(
        repr(
            _guard_digest_material(
                generator_freeze,
                parent,
                config,
                guard_development_evidence_digest,
                guard_calibration_evidence_digest,
                guard,
                calibration,
                keys,
            )
        ).encode("utf-8")
    ).hexdigest()
    plan = generator_freeze.plan
    return CorrectedGuardCalibrationArtifact(
        protocol_version=CORRECTED_GUARD_PROTOCOL_VERSION,
        protocol_digest=digest,
        generator_freeze_digest=generator_freeze.artifact_digest,
        source_manifest_digest=generator_freeze.source_manifest.digest,
        parent_protocol_digest=parent.protocol_digest,
        physical_protocol_digest=generator_freeze.physical_protocol_digest,
        guard_development_evidence_digest=guard_development_evidence_digest,
        guard_calibration_evidence_digest=guard_calibration_evidence_digest,
        campaign=plan.campaign,
        guard_development=_identity(plan.partition(GUARD_DEVELOPMENT_SPLIT)),
        guard_calibration=_identity(plan.partition(GUARD_CALIBRATION_SPLIT)),
        reserved_evaluation=_identity(plan.partition(FINAL_EVALUATION_SPLIT)),
        guard_target_block_retention=config.guard_target_block_retention,
        target_block_coverage=config.target_block_coverage,
        confidence_level=config.confidence_level,
        primary_scenario_name=config.primary_scenario_name,
        cost_scenarios=parent.cost_scenarios,
        bootstrap_partition_name=plan.bootstrap_partition_name,
        bootstrap_draws=plan.bootstrap_draws,
        bootstrap_random_stream_keys=keys,
        guard=guard,
        selector=config.selector,
        procedure_calibration=calibration,
        parent_selector_padding_floor=parent.procedure_for(
            DECISION_DIRECTED_SELECTOR
        ).additive_margin_padding,
    )


def validate_corrected_guard_artifact(
    artifact: CorrectedGuardCalibrationArtifact,
    generator_freeze: CorrectedGeneratorFreeze,
    parent: CorrectedParentCalibrationArtifact,
    config: CorrectedGuardCalibrationConfig,
) -> None:
    if artifact.protocol_version != CORRECTED_GUARD_PROTOCOL_VERSION:
        raise ValueError("corrected guard protocol version changed")
    expected = build_corrected_guard_artifact(
        generator_freeze,
        parent,
        config,
        artifact.guard_development_evidence_digest,
        artifact.guard_calibration_evidence_digest,
        artifact.guard,
        artifact.procedure_calibration,
    )
    if artifact != expected:
        raise ValueError("corrected guard artifact digest or fields changed")


def _key_payload(key: RandomStreamKey) -> dict:
    return {
        "protocol_version": key.protocol_version,
        "campaign": key.campaign,
        "partition": key.partition,
        "block": key.block,
        "stream_kind": key.stream_kind,
        "purpose": key.purpose,
        "family": key.family,
        "run": key.run,
        "channel": key.channel,
    }


def corrected_guard_artifact_payload(
    artifact: CorrectedGuardCalibrationArtifact,
    generator_freeze: CorrectedGeneratorFreeze,
    parent_config: CorrectedParentCalibrationConfig,
    parent: CorrectedParentCalibrationArtifact,
    config: CorrectedGuardCalibrationConfig,
) -> dict:
    validate_corrected_parent_artifact(parent, generator_freeze, parent_config)
    validate_corrected_guard_artifact(artifact, generator_freeze, parent, config)
    return {
        "schema_version": 1,
        "parent_artifact": corrected_parent_artifact_payload(
            parent,
            generator_freeze,
            parent_config,
        ),
        "guard_config": {
            "guard_target_block_retention": config.guard_target_block_retention,
            "target_block_coverage": config.target_block_coverage,
            "confidence_level": config.confidence_level,
            "primary_scenario_name": config.primary_scenario_name,
            "selector": config.selector.__dict__,
        },
        "guard_artifact": {
            "protocol_version": artifact.protocol_version,
            "protocol_digest": artifact.protocol_digest,
            "generator_freeze_digest": artifact.generator_freeze_digest,
            "source_manifest_digest": artifact.source_manifest_digest,
            "parent_protocol_digest": artifact.parent_protocol_digest,
            "physical_protocol_digest": artifact.physical_protocol_digest,
            "guard_development_evidence_digest": (
                artifact.guard_development_evidence_digest
            ),
            "guard_calibration_evidence_digest": (
                artifact.guard_calibration_evidence_digest
            ),
            "campaign": artifact.campaign,
            "partitions": {
                "guard_development": artifact.guard_development._asdict(),
                "guard_calibration": artifact.guard_calibration._asdict(),
                "reserved_evaluation": artifact.reserved_evaluation._asdict(),
            },
            "bootstrap": {
                "partition": artifact.bootstrap_partition_name,
                "draws": artifact.bootstrap_draws,
                "random_stream_keys": [
                    _key_payload(item)
                    for item in artifact.bootstrap_random_stream_keys
                ],
            },
            "cost_scenarios": [item.__dict__ for item in artifact.cost_scenarios],
            "guard": artifact.guard._asdict(),
            "selector": artifact.selector.__dict__,
            "procedure_calibration": artifact.procedure_calibration._asdict(),
            "parent_selector_padding_floor": artifact.parent_selector_padding_floor,
        },
        "chronology": {
            "guard_artifact_precedes_reserved_evaluation": True,
            "reserved_evaluation_instantiated": False,
        },
    }


def _identity_from_payload(value: object) -> CorrectedPartitionIdentity:
    if not isinstance(value, dict) or set(value) != {
        "name",
        "paired_blocks_per_family",
    }:
        raise ValueError("corrected guard partition identity is malformed")
    return CorrectedPartitionIdentity(**value)


def corrected_guard_artifact_from_payload(
    payload: dict,
) -> LoadedCorrectedGuardArtifact:
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "parent_artifact",
        "guard_config",
        "guard_artifact",
        "chronology",
    }:
        raise ValueError("corrected guard payload has unexpected fields")
    if payload["schema_version"] != 1:
        raise ValueError("corrected guard artifact schema changed")
    if payload["chronology"] != {
        "guard_artifact_precedes_reserved_evaluation": True,
        "reserved_evaluation_instantiated": False,
    }:
        raise ValueError("corrected guard chronology declaration changed")
    loaded_parent = corrected_parent_artifact_from_payload(
        payload["parent_artifact"]
    )
    config_payload = payload["guard_config"]
    artifact_payload = payload["guard_artifact"]
    if not isinstance(config_payload, dict) or set(config_payload) != {
        "guard_target_block_retention",
        "target_block_coverage",
        "confidence_level",
        "primary_scenario_name",
        "selector",
    }:
        raise ValueError("corrected guard config payload is malformed")
    if not isinstance(artifact_payload, dict) or set(artifact_payload) != {
        "protocol_version",
        "protocol_digest",
        "generator_freeze_digest",
        "source_manifest_digest",
        "parent_protocol_digest",
        "physical_protocol_digest",
        "guard_development_evidence_digest",
        "guard_calibration_evidence_digest",
        "campaign",
        "partitions",
        "bootstrap",
        "cost_scenarios",
        "guard",
        "selector",
        "procedure_calibration",
        "parent_selector_padding_floor",
    }:
        raise ValueError("corrected guard nested artifact is malformed")
    config = CorrectedGuardCalibrationConfig(
        guard_target_block_retention=config_payload[
            "guard_target_block_retention"
        ],
        target_block_coverage=config_payload["target_block_coverage"],
        confidence_level=config_payload["confidence_level"],
        primary_scenario_name=config_payload["primary_scenario_name"],
        selector=CorrectedSelectorRule(**config_payload["selector"]),
    )
    partitions = artifact_payload["partitions"]
    bootstrap = artifact_payload["bootstrap"]
    guard_payload = artifact_payload["guard"]
    calibration_payload = artifact_payload["procedure_calibration"]
    if not isinstance(partitions, dict) or set(partitions) != {
        "guard_development",
        "guard_calibration",
        "reserved_evaluation",
    }:
        raise ValueError("corrected guard partitions are malformed")
    if not isinstance(bootstrap, dict) or set(bootstrap) != {
        "partition",
        "draws",
        "random_stream_keys",
    }:
        raise ValueError("corrected guard bootstrap is malformed")
    artifact = CorrectedGuardCalibrationArtifact(
        protocol_version=artifact_payload["protocol_version"],
        protocol_digest=artifact_payload["protocol_digest"],
        generator_freeze_digest=artifact_payload["generator_freeze_digest"],
        source_manifest_digest=artifact_payload["source_manifest_digest"],
        parent_protocol_digest=artifact_payload["parent_protocol_digest"],
        physical_protocol_digest=artifact_payload["physical_protocol_digest"],
        guard_development_evidence_digest=artifact_payload[
            "guard_development_evidence_digest"
        ],
        guard_calibration_evidence_digest=artifact_payload[
            "guard_calibration_evidence_digest"
        ],
        campaign=artifact_payload["campaign"],
        guard_development=_identity_from_payload(
            partitions["guard_development"]
        ),
        guard_calibration=_identity_from_payload(partitions["guard_calibration"]),
        reserved_evaluation=_identity_from_payload(
            partitions["reserved_evaluation"]
        ),
        guard_target_block_retention=config.guard_target_block_retention,
        target_block_coverage=config.target_block_coverage,
        confidence_level=config.confidence_level,
        primary_scenario_name=config.primary_scenario_name,
        cost_scenarios=tuple(
            CostScenario(**item) for item in artifact_payload["cost_scenarios"]
        ),
        bootstrap_partition_name=bootstrap["partition"],
        bootstrap_draws=bootstrap["draws"],
        bootstrap_random_stream_keys=tuple(
            RandomStreamKey(**item) for item in bootstrap["random_stream_keys"]
        ),
        guard=AcquisitionGuardCalibration(
            **{
                **guard_payload,
                "matched_block_scores": tuple(
                    guard_payload["matched_block_scores"]
                ),
            }
        ),
        selector=CorrectedSelectorRule(**artifact_payload["selector"]),
        procedure_calibration=ProcedureCalibration(
            **{
                **calibration_payload,
                "block_nonconformity": tuple(
                    calibration_payload["block_nonconformity"]
                ),
            }
        ),
        parent_selector_padding_floor=artifact_payload[
            "parent_selector_padding_floor"
        ],
    )
    validate_corrected_guard_artifact(
        artifact,
        loaded_parent.generator_freeze,
        loaded_parent.artifact,
        config,
    )
    return LoadedCorrectedGuardArtifact(
        loaded_parent.generator_freeze,
        loaded_parent.config,
        loaded_parent.artifact,
        config,
        artifact,
    )


def save_corrected_guard_artifact(
    artifact: CorrectedGuardCalibrationArtifact,
    generator_freeze: CorrectedGeneratorFreeze,
    parent_config: CorrectedParentCalibrationConfig,
    parent: CorrectedParentCalibrationArtifact,
    config: CorrectedGuardCalibrationConfig,
    path: Path | str,
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        corrected_guard_artifact_payload(
            artifact,
            generator_freeze,
            parent_config,
            parent,
            config,
        ),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    return destination


def load_corrected_guard_artifact(
    path: Path | str,
) -> LoadedCorrectedGuardArtifact:
    source = Path(path).expanduser().resolve(strict=True)
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return corrected_guard_artifact_from_payload(payload)


def freeze_corrected_guard_calibration(
    parent_artifact_path: Path | str,
    repository_root: Path | str,
    *,
    physical_config: OperatingDecisionRealismConfig = CORRECTED_REPLICATION_CONFIG,
    config: CorrectedGuardCalibrationConfig = CorrectedGuardCalibrationConfig(),
    workers: int = 1,
    progress: Optional[Callable[[str], None]] = None,
) -> CorrectedGuardFreezeResult:
    """Freeze guard and padding without instantiating the reserved cohort."""

    loaded = load_corrected_parent_artifact(parent_artifact_path)
    verify_committed_artifact_chain(
        repository_root,
        parent_artifact_path,
        loaded.generator_freeze.source_manifest,
    )
    verify_corrected_generator_freeze(
        loaded.generator_freeze,
        repository_root,
        physical_config,
        loaded.generator_freeze.plan,
        expected_source_paths=CORRECTED_NUMERICAL_SOURCE_PATHS,
    )
    validate_corrected_parent_artifact(
        loaded.artifact,
        loaded.generator_freeze,
        loaded.config,
    )
    plan = loaded.generator_freeze.plan
    guard_partition = plan.partition(GUARD_DEVELOPMENT_SPLIT)
    guard_physical = corrected_partition_config(guard_partition, physical_config)
    if progress is not None:
        progress("corrected acquisition-guard development starting")
    guard_result = _run_corrected_partition(
        guard_partition,
        guard_physical,
        workers=workers,
        progress=progress,
    )
    guard = calibrate_corrected_acquisition_guard(
        _trials(guard_result),
        guard_physical,
        guard_partition,
        config,
    )
    if progress is not None:
        progress(
            f"corrected acquisition guard frozen at {guard.threshold:.6f} "
            f"(rank {guard.conformal_rank}/{guard.block_count})"
        )
    calibration_partition = plan.partition(GUARD_CALIBRATION_SPLIT)
    calibration_physical = corrected_partition_config(
        calibration_partition,
        physical_config,
    )
    if progress is not None:
        progress("corrected guarded-procedure calibration starting")
    calibration_result = _run_corrected_partition(
        calibration_partition,
        calibration_physical,
        workers=workers,
        progress=progress,
    )
    procedure_calibration = calibrate_corrected_guarded_procedure(
        _trials(calibration_result),
        loaded.artifact,
        calibration_physical,
        calibration_partition,
        guard,
        config,
    )
    artifact = build_corrected_guard_artifact(
        loaded.generator_freeze,
        loaded.artifact,
        config,
        corrected_partition_evidence_digest(guard_result),
        corrected_partition_evidence_digest(calibration_result),
        guard,
        procedure_calibration,
    )
    outcomes = _combined_outcomes(
        _trials(calibration_result),
        loaded.artifact,
        calibration_physical,
        artifact,
        split=GUARD_CALIBRATION_SPLIT,
    )
    summaries = summarize_corrected_final_outcomes(
        outcomes,
        artifact.cost_scenarios,
    )
    guard_summaries = _guard_trigger_summaries(
        _trials(guard_result),
        loaded.artifact,
        guard_physical,
        guard,
        config.selector,
        split=GUARD_DEVELOPMENT_SPLIT,
    )
    if progress is not None:
        progress(f"corrected guard artifact frozen: {artifact.protocol_digest[:12]}")
    return CorrectedGuardFreezeResult(
        loaded.generator_freeze,
        loaded.config,
        loaded.artifact,
        config,
        artifact,
        guard_result,
        calibration_result,
        outcomes,
        summaries,
        guard_summaries,
    )


def _paired_loss_comparisons(
    outcomes: Sequence[ProcedureOutcome],
    artifact: CorrectedGuardCalibrationArtifact,
) -> Tuple[PairedLossComparison, ...]:
    rows = tuple(outcomes)
    indexed = {
        (item.procedure_name, item.truth_condition, item.trial_index): item
        for item in rows
    }
    expected_keys = {
        (procedure, truth, block)
        for procedure in CORRECTED_FINAL_PROCEDURES
        for truth in STAGE3_TRUTH_CONDITIONS
        for block in range(artifact.reserved_evaluation.paired_blocks_per_family)
    }
    if len(indexed) != len(rows) or set(indexed) != expected_keys:
        raise ValueError("paired loss comparison needs the complete reserved cohort")
    comparisons = []
    alpha = 1.0 - artifact.confidence_level
    for scenario, key in zip(
        artifact.cost_scenarios,
        artifact.bootstrap_random_stream_keys,
    ):
        differences = []
        for block in range(artifact.reserved_evaluation.paired_blocks_per_family):
            family_differences = []
            for truth in STAGE3_TRUTH_CONDITIONS:
                stop_energy = indexed[
                    (STOP_NOW, truth, block)
                ].total_diagnostic_energy
                revised = indexed[(MISMATCH_GUARDED_SELECTOR, truth, block)]
                parent = indexed[(DECISION_DIRECTED_SELECTOR, truth, block)]
                family_differences.append(
                    expected_loss(revised, scenario, stop_energy=stop_energy)
                    - expected_loss(parent, scenario, stop_energy=stop_energy)
                )
            differences.append(fmean(family_differences))
        random_source = calibration_stream(
            campaign=key.campaign,
            partition=key.partition,
            purpose=key.purpose,
            block=key.block,
            family=key.family,
            protocol_version=key.protocol_version,
        ).new_generator()
        bootstrap = sorted(
            fmean(
                differences[random_source.randrange(len(differences))]
                for _ in differences
            )
            for _ in range(artifact.bootstrap_draws)
        )
        lower_index = max(0, math.ceil((alpha / 2.0) * len(bootstrap)) - 1)
        upper_index = min(
            len(bootstrap) - 1,
            math.ceil((1.0 - alpha / 2.0) * len(bootstrap)) - 1,
        )
        comparisons.append(
            PairedLossComparison(
                scenario_name=scenario.name,
                revised_minus_parent_mean=fmean(differences),
                lower_bound=bootstrap[lower_index],
                upper_bound=bootstrap[upper_index],
                confidence_level=artifact.confidence_level,
                independent_block_count=len(differences),
                bootstrap_draws=artifact.bootstrap_draws,
                random_stream_key=key,
            )
        )
    return tuple(comparisons)


def evaluate_corrected_reserved_cohort(
    artifact_path: Path | str,
    repository_root: Path | str,
    *,
    physical_config: OperatingDecisionRealismConfig = CORRECTED_REPLICATION_CONFIG,
    workers: int = 1,
    progress: Optional[Callable[[str], None]] = None,
) -> CorrectedFinalEvaluationResult:
    """Verify a serialized guard artifact, then instantiate the reserved cohort."""

    loaded = load_corrected_guard_artifact(artifact_path)
    partition = loaded.generator_freeze.plan.partition(FINAL_EVALUATION_SPLIT)
    authorization = _authorize_reserved_evaluation(
        partition,
        repository_root,
        artifact_path,
        physical_config,
    )
    verify_corrected_generator_freeze(
        loaded.generator_freeze,
        repository_root,
        physical_config,
        loaded.generator_freeze.plan,
        expected_source_paths=CORRECTED_NUMERICAL_SOURCE_PATHS,
    )
    validate_corrected_parent_artifact(
        loaded.parent,
        loaded.generator_freeze,
        loaded.parent_config,
    )
    validate_corrected_guard_artifact(
        loaded.artifact,
        loaded.generator_freeze,
        loaded.parent,
        loaded.config,
    )
    if _identity(partition) != loaded.artifact.reserved_evaluation:
        raise ValueError("reserved evaluation partition differs from its artifact")
    evaluation_physical = corrected_partition_config(partition, physical_config)
    if corrected_physical_protocol_digest(
        evaluation_physical
    ) != loaded.artifact.physical_protocol_digest:
        raise ValueError("reserved evaluation physics differs from its artifact")
    if progress is not None:
        progress("serialized corrected guard verified; reserved evaluation starting")
    evaluation = _run_corrected_partition(
        partition,
        evaluation_physical,
        workers=workers,
        progress=progress,
        _reserved_authorization=authorization,
    )
    outcomes = _combined_outcomes(
        _trials(evaluation),
        loaded.parent,
        evaluation_physical,
        loaded.artifact,
        split=FINAL_EVALUATION_SPLIT,
    )
    summaries = summarize_corrected_final_outcomes(
        outcomes,
        loaded.artifact.cost_scenarios,
    )
    comparisons = _paired_loss_comparisons(outcomes, loaded.artifact)
    primary = tuple(
        item
        for item in comparisons
        if item.scenario_name == loaded.artifact.primary_scenario_name
    )
    if len(primary) != 1:
        raise ValueError("reserved result has no unique primary comparison")
    guard_summaries = _guard_trigger_summaries(
        _trials(evaluation),
        loaded.parent,
        evaluation_physical,
        loaded.artifact.guard,
        loaded.artifact.selector,
        split=FINAL_EVALUATION_SPLIT,
    )
    return CorrectedFinalEvaluationResult(
        loaded.generator_freeze,
        loaded.parent_config,
        loaded.parent,
        loaded.config,
        loaded.artifact,
        authorization.commit,
        evaluation,
        outcomes,
        summaries,
        guard_summaries,
        comparisons,
        primary[0].upper_bound < 0.0,
    )


__all__ = [
    "CORRECTED_FINAL_PROCEDURES",
    "CORRECTED_GUARD_PROTOCOL_VERSION",
    "EARLY_MISMATCH_ABSTENTION",
    "FINAL_EVALUATION_SPLIT",
    "GUARD_CALIBRATION_SPLIT",
    "GUARD_DEVELOPMENT_SPLIT",
    "MISMATCH_GUARDED_SELECTOR",
    "AcquisitionAdequacySignal",
    "AcquisitionGuardCalibration",
    "CorrectedFinalEvaluationResult",
    "CorrectedGuardCalibrationArtifact",
    "CorrectedGuardCalibrationConfig",
    "CorrectedGuardFreezeResult",
    "CorrectedSelectorRule",
    "GuardTriggerSummary",
    "LoadedCorrectedGuardArtifact",
    "PairedLossComparison",
    "acquisition_adequacy_signal",
    "acquisition_data_score",
    "build_corrected_guard_artifact",
    "calibrate_corrected_acquisition_guard",
    "calibrate_corrected_guarded_procedure",
    "corrected_guard_artifact_from_payload",
    "corrected_guard_artifact_payload",
    "evaluate_corrected_reserved_cohort",
    "freeze_corrected_guard_calibration",
    "load_corrected_guard_artifact",
    "save_corrected_guard_artifact",
    "select_corrected_action",
    "summarize_corrected_final_outcomes",
    "validate_corrected_guard_artifact",
]
