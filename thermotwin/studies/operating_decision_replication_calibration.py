"""Corrected parent calibration for the audited operating-decision replication."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Dict, NamedTuple, Optional, Sequence, Tuple

from ..numerics.integration import IntegrationDivergenceError
from ..observations.test_stand import regular_measurement_times
from .operating_decision import (
    APPROVE,
    FINAL_EVALUATION,
    INSUFFICIENT_EVIDENCE,
    REJECT,
    RUN_DURATION_SECONDS,
    CandidateVerification,
    NumericalFailure,
    OperatingRegime,
    SavedOperatingDecision,
    ScoredOperatingDecision,
    classify_margin_envelope,
    default_fixed_policies,
    envelope_margin_intervals,
)
from .operating_decision_calibration import (
    CALIBRATION_SPLIT,
    DECISION_DIRECTED_SELECTOR,
    DEFAULT_COST_SCENARIOS,
    REHEARSAL_SPLIT,
    STAGE4_PROCEDURES,
    CostScenario,
    ProcedureCalibration,
    ProcedureOutcome,
    ProcedureSummary,
    SelectorRule,
    VerificationGate,
    _compact_outcome,
    _outside_interval_miss,
    _selector_choices,
    _validate_paired_trials,
    apply_margin_padding,
    block_conformal_padding,
    summarize_calibrated_outcomes,
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
    forecast_realistic_margin_interval,
)
from .operating_decision_replication import (
    CORRECTED_REPLICATION_CONFIG,
    CorrectedPartition,
    CorrectedPartitionResult,
    _run_corrected_partition,
    corrected_partition_config,
    corrected_physical_protocol_digest,
)
from .operating_decision_replication_protocol import (
    CORRECTED_NUMERICAL_SOURCE_PATHS,
    CorrectedGeneratorFreeze,
    corrected_generator_freeze_from_payload,
    corrected_generator_freeze_payload,
    load_corrected_generator_freeze,
    verify_corrected_generator_freeze,
)
from .sensor_model_discrimination import FIVE_STATE_MODEL, FOUR_STATE_MODEL


CORRECTED_PARENT_PROTOCOL_VERSION = "operating_decision_corrected_parent_v1"


@dataclass(frozen=True)
class CorrectedParentCalibrationConfig:
    """Frozen statistical choices for the corrected Stage 4 replication."""

    target_block_coverage: float = 0.90
    gate_target_block_retention: float = 0.90
    gate_null_quantile: float = 0.99
    gate_monte_carlo_draws: int = 50_000
    selector: SelectorRule = SelectorRule()
    cost_scenarios: Tuple[CostScenario, ...] = DEFAULT_COST_SCENARIOS

    def __post_init__(self) -> None:
        for name, value in (
            ("target block coverage", self.target_block_coverage),
            ("gate target block retention", self.gate_target_block_retention),
            ("gate null quantile", self.gate_null_quantile),
        ):
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between zero and one")
        if self.gate_null_quantile <= 0.5:
            raise ValueError("gate null quantile must exceed one half")
        if (
            not isinstance(self.gate_monte_carlo_draws, int)
            or isinstance(self.gate_monte_carlo_draws, bool)
            or self.gate_monte_carlo_draws < 1_000
        ):
            raise ValueError("gate calibration needs at least 1,000 Monte Carlo draws")
        scenarios = tuple(self.cost_scenarios)
        object.__setattr__(self, "cost_scenarios", scenarios)
        if not scenarios or len({item.name for item in scenarios}) != len(scenarios):
            raise ValueError("corrected cost scenarios must be unique and nonempty")


class CorrectedPartitionIdentity(NamedTuple):
    name: str
    paired_blocks_per_family: int


class CorrectedParentCalibrationArtifact(NamedTuple):
    protocol_version: str
    protocol_digest: str
    generator_freeze_digest: str
    physical_protocol_digest: str
    gate_development_evidence_digest: str
    calibration_evidence_digest: str
    campaign: str
    gate_development: CorrectedPartitionIdentity
    calibration: CorrectedPartitionIdentity
    rehearsal: CorrectedPartitionIdentity
    reserved_evaluation: CorrectedPartitionIdentity
    target_block_coverage: float
    gate_target_block_retention: float
    gate_null_quantile: float
    gate_monte_carlo_draws: int
    gate_random_stream_keys: Tuple[RandomStreamKey, ...]
    verification_gates: Tuple[VerificationGate, ...]
    selector: SelectorRule
    procedure_calibrations: Tuple[ProcedureCalibration, ...]
    cost_scenarios: Tuple[CostScenario, ...]

    def gate_for(self, policy_name: str) -> VerificationGate:
        matches = tuple(
            item for item in self.verification_gates if item.policy_name == policy_name
        )
        if len(matches) != 1:
            raise ValueError("corrected parent artifact has no unique policy gate")
        return matches[0]

    def procedure_for(self, procedure_name: str) -> ProcedureCalibration:
        matches = tuple(
            item
            for item in self.procedure_calibrations
            if item.procedure_name == procedure_name
        )
        if len(matches) != 1:
            raise ValueError("corrected parent artifact has no unique procedure")
        return matches[0]


class CorrectedParentFreezeResult(NamedTuple):
    generator_freeze: CorrectedGeneratorFreeze
    config: CorrectedParentCalibrationConfig
    artifact: CorrectedParentCalibrationArtifact
    gate_development: CorrectedPartitionResult
    calibration: CorrectedPartitionResult
    calibration_outcomes: Tuple[ProcedureOutcome, ...]


class CorrectedParentRehearsalResult(NamedTuple):
    generator_freeze: CorrectedGeneratorFreeze
    config: CorrectedParentCalibrationConfig
    artifact: CorrectedParentCalibrationArtifact
    rehearsal: CorrectedPartitionResult
    rehearsal_outcomes: Tuple[ProcedureOutcome, ...]
    summaries: Tuple[ProcedureSummary, ...]


def _trials(result: CorrectedPartitionResult) -> Tuple[ScoredOperatingDecision, ...]:
    return tuple(item.scored for item in result.records)


def _empirical_quantile(values: Sequence[float], quantile: float) -> float:
    values = tuple(float(value) for value in values)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("corrected empirical quantile needs finite values")
    rank = min(len(values), max(1, math.ceil(len(values) * quantile)))
    return sorted(values)[rank - 1]


def _gate_noise_reference(
    observation_count: int,
    policy_name: str,
    partition: CorrectedPartition,
    config: CorrectedParentCalibrationConfig,
) -> Tuple[float, RandomStreamKey, int]:
    stream = calibration_stream(
        campaign=partition.campaign,
        partition=partition.name,
        purpose=f"verification_gate_noise_reference/{policy_name}",
    )
    random_source = stream.new_generator()
    scores = tuple(
        random_source.gammavariate(observation_count / 2.0, 2.0)
        / observation_count
        for _ in range(config.gate_monte_carlo_draws)
    )
    return _empirical_quantile(scores, config.gate_null_quantile), stream.key, stream.seed


def calibrate_corrected_verification_gates(
    trials: Sequence[ScoredOperatingDecision],
    physical_config: OperatingDecisionRealismConfig,
    partition: CorrectedPartition,
    config: CorrectedParentCalibrationConfig = CorrectedParentCalibrationConfig(),
) -> Tuple[Tuple[VerificationGate, ...], Tuple[RandomStreamKey, ...]]:
    """Calibrate matched A/B verification gates with semantic Monte Carlo keys."""

    if physical_config.sensor.trial_count != partition.block_count:
        raise ValueError("gate physical config and semantic partition differ")
    indexed = _validate_paired_trials(trials, physical_config)
    sample_count = len(
        regular_measurement_times(
            RUN_DURATION_SECONDS,
            physical_config.sensor.sampling_interval,
        )
    )
    gates = []
    stream_keys = []
    for policy in default_fixed_policies():
        observation_count = sample_count * (2 + int(policy.extra_sensor_count > 0))
        block_scores = []
        for block in range(partition.block_count):
            correct_scores = []
            for truth_condition, model_name in (
                (STAGE3_TRUTH_CONDITIONS[0], FOUR_STATE_MODEL),
                (STAGE3_TRUTH_CONDITIONS[1], FIVE_STATE_MODEL),
            ):
                trial = indexed[(truth_condition, block, policy.name)]
                matches = tuple(
                    item
                    for item in trial.saved.verifications
                    if item.fit.model_name == model_name
                )
                if len(matches) != 1:
                    raise ValueError("corrected matched gate candidate is missing")
                verification = matches[0]
                if (
                    not verification.fit.converged
                    or verification.fit.reached_bound
                    or not math.isfinite(verification.normalized_score)
                ):
                    raise ValueError("corrected matched gate candidate is unreliable")
                correct_scores.append(verification.normalized_score)
            block_scores.append(max(correct_scores))
        rank, matched_threshold = block_conformal_padding(
            block_scores,
            config.gate_target_block_retention,
        )
        noise_threshold, key, seed = _gate_noise_reference(
            observation_count,
            policy.name,
            partition,
            config,
        )
        stream_keys.append(key)
        gates.append(
            VerificationGate(
                policy_name=policy.name,
                observation_count=observation_count,
                matched_development_block_count=len(block_scores),
                matched_development_rank=rank,
                target_block_retention=config.gate_target_block_retention,
                matched_block_scores=tuple(block_scores),
                noise_reference_quantile=config.gate_null_quantile,
                noise_reference_threshold=noise_threshold,
                threshold=max(matched_threshold, noise_threshold),
                monte_carlo_draws=config.gate_monte_carlo_draws,
                seed=seed,
            )
        )
    return tuple(gates), tuple(stream_keys)


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


def _compact_corrected_outcome(
    split: str,
    procedure_name: str,
    selected_policy: str,
    raw: ScoredOperatingDecision,
    calibrated: ScoredOperatingDecision,
    padding: float,
) -> ProcedureOutcome:
    """Compact a row while retaining corrected-gate rejection counts."""

    outcome = _compact_outcome(
        split,
        procedure_name,
        selected_policy,
        raw,
        calibrated,
        padding,
    )
    return outcome._replace(
        score_rejected_candidate_count=sum(
            item.failure_reason == "inadequate_corrected_verification"
            for item in raw.saved.verifications
        )
    )


def rebuild_with_corrected_gate(
    trial: ScoredOperatingDecision,
    gate: VerificationGate,
    physical_config: OperatingDecisionRealismConfig,
) -> ScoredOperatingDecision:
    """Apply a corrected gate after excluding inadmissible candidates."""

    if trial.saved.policy_name != gate.policy_name:
        raise ValueError("corrected verification gate and trial policy differ")
    acquisition_failures = tuple(
        item for item in trial.saved.failures if item.stage == "acquisition_fit"
    )
    failures = list(acquisition_failures)
    verifications = []
    intervals = []
    for original in trial.saved.verifications:
        fit = original.fit
        if original.failure_reason == "verification_failure":
            reason = "verification_failure"
            failures.append(
                NumericalFailure(fit.model_name, "verification", reason)
            )
        elif fit.reached_bound:
            reason = "fit_reached_bound"
        elif not math.isfinite(original.normalized_score):
            reason = "nonfinite_score"
            failures.append(
                NumericalFailure(fit.model_name, "verification", reason)
            )
        elif not fit.converged:
            reason = "optimizer_not_converged"
        elif original.normalized_score > gate.threshold:
            reason = "inadequate_corrected_verification"
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
                    fit,
                    _final_regime(physical_config),
                    physical_config,
                )
            )
        except (ArithmeticError, IntegrationDivergenceError, ValueError) as error:
            failures.append(
                NumericalFailure(fit.model_name, "uncertainty", type(error).__name__)
            )
    candidate_envelope = envelope_margin_intervals(intervals)
    envelope = None if failures else candidate_envelope
    if acquisition_failures:
        decision = INSUFFICIENT_EVIDENCE
        decision_reason = "acquisition_fit_failure"
    elif any(item.stage == "verification" for item in failures):
        decision = INSUFFICIENT_EVIDENCE
        decision_reason = "verification_failure"
    elif any(item.stage == "uncertainty" for item in failures):
        decision = INSUFFICIENT_EVIDENCE
        decision_reason = "uncertainty_failure"
    elif envelope is None:
        decision = INSUFFICIENT_EVIDENCE
        decision_reason = "no_corrected_verified_candidate"
    else:
        decision = classify_margin_envelope(envelope)
        decision_reason = {
            APPROVE: "corrected_verified_envelope_inside_band",
            REJECT: "corrected_verified_envelope_below_zero",
            INSUFFICIENT_EVIDENCE: "corrected_verified_envelope_crosses_zero",
        }[decision]
    saved = SavedOperatingDecision(
        case_id=trial.saved.case_id,
        decision=decision,
        decision_reason=decision_reason,
        margin_envelope=envelope,
        model_intervals=() if failures else tuple(intervals),
        verifications=tuple(verifications),
        failures=tuple(failures),
        decision_computation_seconds=trial.saved.decision_computation_seconds,
    )
    return _replace_scored_decision(trial, saved)


def _final_regime(config: OperatingDecisionRealismConfig) -> OperatingRegime:
    return OperatingRegime(
        name="untouched_final_operating_schedule",
        phase=FINAL_EVALUATION,
        current=config.final_current,
        channels=(),
    )


def _gate_trials(
    trials: Sequence[ScoredOperatingDecision],
    gates: Sequence[VerificationGate],
    physical_config: OperatingDecisionRealismConfig,
) -> Dict[Tuple[str, int, str], ScoredOperatingDecision]:
    indexed = _validate_paired_trials(trials, physical_config)
    by_policy = {item.policy_name: item for item in gates}
    if set(by_policy) != {policy.name for policy in default_fixed_policies()}:
        raise ValueError("corrected calibration needs one gate per fixed policy")
    return {
        key: rebuild_with_corrected_gate(trial, by_policy[key[2]], physical_config)
        for key, trial in indexed.items()
    }


def calibrate_corrected_parent_procedures(
    trials: Sequence[ScoredOperatingDecision],
    gates: Sequence[VerificationGate],
    physical_config: OperatingDecisionRealismConfig,
    partition: CorrectedPartition,
    config: CorrectedParentCalibrationConfig = CorrectedParentCalibrationConfig(),
) -> Tuple[ProcedureCalibration, ...]:
    """Fit one blockwise padding for every corrected parent procedure."""

    if physical_config.sensor.trial_count != partition.block_count:
        raise ValueError("parent calibration config and semantic partition differ")
    gated = _gate_trials(trials, gates, physical_config)
    raw_index = _validate_paired_trials(trials, physical_config)
    choices = _selector_choices(raw_index, physical_config, config.selector)
    calibrations = []
    for procedure_name in STAGE4_PROCEDURES:
        block_scores = []
        emitted = 0
        raw_covered = 0
        for block in range(partition.block_count):
            family_scores = []
            for truth_condition in STAGE3_TRUTH_CONDITIONS:
                selected_policy = (
                    choices[(truth_condition, block)]
                    if procedure_name == DECISION_DIRECTED_SELECTOR
                    else procedure_name
                )
                trial = gated[(truth_condition, block, selected_policy)]
                family_scores.append(_outside_interval_miss(trial))
                if trial.saved.margin_envelope is not None:
                    emitted += 1
                    raw_covered += int(trial.interval_covered is True)
            block_scores.append(max(family_scores))
        rank, padding = block_conformal_padding(
            block_scores,
            config.target_block_coverage,
        )
        calibrated_covered = 0
        for block in range(partition.block_count):
            for truth_condition in STAGE3_TRUTH_CONDITIONS:
                selected_policy = (
                    choices[(truth_condition, block)]
                    if procedure_name == DECISION_DIRECTED_SELECTOR
                    else procedure_name
                )
                trial = gated[(truth_condition, block, selected_policy)]
                calibrated_covered += int(
                    apply_margin_padding(trial, padding).interval_covered is True
                )
        calibrations.append(
            ProcedureCalibration(
                procedure_name=procedure_name,
                block_count=partition.block_count,
                conformal_rank=rank,
                target_block_coverage=config.target_block_coverage,
                additive_margin_padding=padding,
                block_nonconformity=tuple(block_scores),
                emitted_interval_count=emitted,
                raw_interval_covered_count=raw_covered,
                calibrated_interval_covered_count=calibrated_covered,
            )
        )
    return tuple(calibrations)


def _partition_identity(partition: CorrectedPartition) -> CorrectedPartitionIdentity:
    return CorrectedPartitionIdentity(partition.name, partition.block_count)


def _parent_digest_material(
    *,
    generator_freeze: CorrectedGeneratorFreeze,
    config: CorrectedParentCalibrationConfig,
    gate_development_evidence_digest: str,
    calibration_evidence_digest: str,
    gate_keys: Sequence[RandomStreamKey],
    gates: Sequence[VerificationGate],
    calibrations: Sequence[ProcedureCalibration],
) -> tuple:
    plan = generator_freeze.plan
    return (
        CORRECTED_PARENT_PROTOCOL_VERSION,
        generator_freeze.artifact_digest,
        generator_freeze.physical_protocol_digest,
        gate_development_evidence_digest,
        calibration_evidence_digest,
        plan.campaign,
        tuple(_partition_identity(item) for item in plan.partitions),
        config,
        tuple(gate_keys),
        tuple(gates),
        tuple(calibrations),
    )


def _parent_digest(**arguments) -> str:
    return hashlib.sha256(repr(_parent_digest_material(**arguments)).encode("utf-8")).hexdigest()


def _validate_parent_components(
    generator_freeze: CorrectedGeneratorFreeze,
    config: CorrectedParentCalibrationConfig,
    gate_development_evidence_digest: str,
    calibration_evidence_digest: str,
    gate_keys: Sequence[RandomStreamKey],
    gates: Sequence[VerificationGate],
    calibrations: Sequence[ProcedureCalibration],
) -> None:
    plan = generator_freeze.plan
    gate_partition = plan.partition("r2_gate_development")
    calibration_partition = plan.partition("r2_parent_calibration")
    gate_keys = tuple(gate_keys)
    gates = tuple(gates)
    calibrations = tuple(calibrations)
    policies = default_fixed_policies()
    for name, digest in (
        ("gate-development evidence", gate_development_evidence_digest),
        ("calibration evidence", calibration_evidence_digest),
    ):
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{name} digest must be lowercase SHA-256")
    sample_count = len(
        regular_measurement_times(
            RUN_DURATION_SECONDS,
            corrected_partition_config(gate_partition).sensor.sampling_interval,
        )
    )
    if len(gates) != len(policies) or {item.policy_name for item in gates} != {
        item.name for item in policies
    }:
        raise ValueError("corrected parent needs one verification gate per policy")
    if len(gate_keys) != len(policies) or len(set(gate_keys)) != len(gate_keys):
        raise ValueError("corrected parent needs one unique gate stream per policy")
    for key in gate_keys:
        if (
            key.protocol_version != RANDOM_STREAM_PROTOCOL_VERSION
            or key.stream_kind != CALIBRATION_STREAM
            or key.campaign != plan.campaign
            or key.partition != gate_partition.name
            or key.block != 0
            or key.family is not None
            or key.run is not None
            or key.channel is not None
        ):
            raise ValueError("corrected gate stream key has the wrong namespace")
    for gate in gates:
        policy_matches = tuple(
            policy for policy in policies if policy.name == gate.policy_name
        )
        if len(policy_matches) != 1:
            raise ValueError("corrected gate has no unique fixed policy")
        expected_observation_count = sample_count * (
            2 + int(policy_matches[0].extra_sensor_count > 0)
        )
        if (
            gate.observation_count != expected_observation_count
            or gate.matched_development_block_count != gate_partition.block_count
            or len(gate.matched_block_scores) != gate_partition.block_count
            or any(
                not math.isfinite(value) or value < 0.0
                for value in gate.matched_block_scores
            )
            or gate.target_block_retention != config.gate_target_block_retention
            or gate.noise_reference_quantile != config.gate_null_quantile
            or gate.monte_carlo_draws != config.gate_monte_carlo_draws
        ):
            raise ValueError("corrected gate metadata differs from its protocol")
        rank, matched_threshold = block_conformal_padding(
            gate.matched_block_scores,
            config.gate_target_block_retention,
        )
        if gate.matched_development_rank != rank or gate.threshold != max(
            matched_threshold,
            gate.noise_reference_threshold,
        ):
            raise ValueError("corrected verification gate is not reproducible")
        expected_purpose = f"verification_gate_noise_reference/{gate.policy_name}"
        matching_keys = tuple(
            key
            for key in gate_keys
            if key.purpose == expected_purpose
        )
        if len(matching_keys) != 1:
            raise ValueError("corrected gate has no unique calibration stream key")
        expected_key = matching_keys[0]
        noise_threshold, reproduced_key, reproduced_seed = _gate_noise_reference(
            expected_observation_count,
            gate.policy_name,
            gate_partition,
            config,
        )
        if (
            reproduced_key != expected_key
            or reproduced_seed != gate.seed
            or noise_threshold != gate.noise_reference_threshold
        ):
            raise ValueError("corrected verification gate seed does not match its key")
    if len(calibrations) != len(STAGE4_PROCEDURES) or {
        item.procedure_name for item in calibrations
    } != set(STAGE4_PROCEDURES):
        raise ValueError("corrected parent needs one calibration per procedure")
    for calibration in calibrations:
        if (
            calibration.block_count != calibration_partition.block_count
            or len(calibration.block_nonconformity)
            != calibration_partition.block_count
            or any(
                not math.isfinite(value) or value < 0.0
                for value in calibration.block_nonconformity
            )
            or calibration.target_block_coverage != config.target_block_coverage
        ):
            raise ValueError("corrected procedure calibration metadata changed")
        rank, padding = block_conformal_padding(
            calibration.block_nonconformity,
            config.target_block_coverage,
        )
        if calibration.conformal_rank != rank or (
            calibration.additive_margin_padding != padding
        ):
            raise ValueError("corrected procedure padding is not its order statistic")
        maximum_rows = (
            len(STAGE3_TRUTH_CONDITIONS) * calibration_partition.block_count
        )
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
            raise ValueError("corrected procedure coverage counts are invalid")
        if calibration.raw_interval_covered_count > calibration.emitted_interval_count:
            raise ValueError("raw coverage exceeds emitted corrected intervals")
        if (
            calibration.calibrated_interval_covered_count
            > calibration.emitted_interval_count
        ):
            raise ValueError("calibrated coverage exceeds emitted corrected intervals")
        if (
            calibration.calibrated_interval_covered_count
            < calibration.raw_interval_covered_count
        ):
            raise ValueError("widening reduced corrected calibration coverage")


def build_corrected_parent_artifact(
    generator_freeze: CorrectedGeneratorFreeze,
    config: CorrectedParentCalibrationConfig,
    gate_development_evidence_digest: str,
    calibration_evidence_digest: str,
    gate_keys: Sequence[RandomStreamKey],
    gates: Sequence[VerificationGate],
    calibrations: Sequence[ProcedureCalibration],
) -> CorrectedParentCalibrationArtifact:
    """Freeze corrected gates and paddings before rehearsal or evaluation."""

    _validate_parent_components(
        generator_freeze,
        config,
        gate_development_evidence_digest,
        calibration_evidence_digest,
        gate_keys,
        gates,
        calibrations,
    )
    plan = generator_freeze.plan
    digest = _parent_digest(
        generator_freeze=generator_freeze,
        config=config,
        gate_development_evidence_digest=gate_development_evidence_digest,
        calibration_evidence_digest=calibration_evidence_digest,
        gate_keys=gate_keys,
        gates=gates,
        calibrations=calibrations,
    )
    return CorrectedParentCalibrationArtifact(
        protocol_version=CORRECTED_PARENT_PROTOCOL_VERSION,
        protocol_digest=digest,
        generator_freeze_digest=generator_freeze.artifact_digest,
        physical_protocol_digest=generator_freeze.physical_protocol_digest,
        gate_development_evidence_digest=gate_development_evidence_digest,
        calibration_evidence_digest=calibration_evidence_digest,
        campaign=plan.campaign,
        gate_development=_partition_identity(plan.partition("r2_gate_development")),
        calibration=_partition_identity(plan.partition("r2_parent_calibration")),
        rehearsal=_partition_identity(plan.partition("r2_parent_rehearsal")),
        reserved_evaluation=_partition_identity(
            plan.partition("r2_reserved_evaluation")
        ),
        target_block_coverage=config.target_block_coverage,
        gate_target_block_retention=config.gate_target_block_retention,
        gate_null_quantile=config.gate_null_quantile,
        gate_monte_carlo_draws=config.gate_monte_carlo_draws,
        gate_random_stream_keys=tuple(gate_keys),
        verification_gates=tuple(gates),
        selector=config.selector,
        procedure_calibrations=tuple(calibrations),
        cost_scenarios=config.cost_scenarios,
    )


def validate_corrected_parent_artifact(
    artifact: CorrectedParentCalibrationArtifact,
    generator_freeze: CorrectedGeneratorFreeze,
    config: CorrectedParentCalibrationConfig,
) -> None:
    if artifact.protocol_version != CORRECTED_PARENT_PROTOCOL_VERSION:
        raise ValueError("corrected parent protocol version changed")
    expected = build_corrected_parent_artifact(
        generator_freeze,
        config,
        artifact.gate_development_evidence_digest,
        artifact.calibration_evidence_digest,
        artifact.gate_random_stream_keys,
        artifact.verification_gates,
        artifact.procedure_calibrations,
    )
    if artifact != expected:
        raise ValueError("corrected parent artifact digest or fields changed")


def apply_corrected_parent_artifact(
    trials: Sequence[ScoredOperatingDecision],
    artifact: CorrectedParentCalibrationArtifact,
    generator_freeze: CorrectedGeneratorFreeze,
    config: CorrectedParentCalibrationConfig,
    physical_config: OperatingDecisionRealismConfig,
    partition: CorrectedPartition,
    *,
    split: str,
) -> Tuple[ProcedureOutcome, ...]:
    """Apply a frozen corrected parent rule to calibration or rehearsal rows."""

    validate_corrected_parent_artifact(artifact, generator_freeze, config)
    expected_identity = {
        CALIBRATION_SPLIT: artifact.calibration,
        REHEARSAL_SPLIT: artifact.rehearsal,
    }.get(split)
    if expected_identity is None:
        raise ValueError("corrected parent split must be calibration or rehearsal")
    if _partition_identity(partition) != expected_identity:
        raise ValueError("semantic partition does not match the corrected parent split")
    if partition.campaign != artifact.campaign:
        raise ValueError("corrected parent campaign changed")
    if (
        corrected_physical_protocol_digest(physical_config)
        != artifact.physical_protocol_digest
    ):
        raise ValueError("corrected parent physical protocol changed")
    raw_index = _validate_paired_trials(trials, physical_config)
    gated = _gate_trials(trials, artifact.verification_gates, physical_config)
    choices = _selector_choices(raw_index, physical_config, artifact.selector)
    outcomes = []
    for truth_condition in STAGE3_TRUTH_CONDITIONS:
        for block in range(partition.block_count):
            for procedure_name in STAGE4_PROCEDURES:
                selected_policy = (
                    choices[(truth_condition, block)]
                    if procedure_name == DECISION_DIRECTED_SELECTOR
                    else procedure_name
                )
                raw = gated[(truth_condition, block, selected_policy)]
                calibration = artifact.procedure_for(procedure_name)
                calibrated = apply_margin_padding(
                    raw,
                    calibration.additive_margin_padding,
                )
                outcomes.append(
                    _compact_corrected_outcome(
                        split,
                        procedure_name,
                        selected_policy,
                        raw,
                        calibrated,
                        calibration.additive_margin_padding,
                    )
                )
    return tuple(outcomes)


def freeze_corrected_parent_calibration(
    generator_freeze_path: Path | str,
    repository_root: Path | str,
    *,
    physical_config: OperatingDecisionRealismConfig = CORRECTED_REPLICATION_CONFIG,
    config: CorrectedParentCalibrationConfig = CorrectedParentCalibrationConfig(),
    workers: int = 1,
    progress: Optional[Callable[[str], None]] = None,
) -> CorrectedParentFreezeResult:
    """Load a committed generator freeze, then calibrate without rehearsal."""

    generator_freeze = load_corrected_generator_freeze(generator_freeze_path)
    verify_committed_artifact_chain(
        repository_root,
        generator_freeze_path,
        generator_freeze.source_manifest,
    )
    verify_corrected_generator_freeze(
        generator_freeze,
        repository_root,
        physical_config,
        generator_freeze.plan,
        expected_source_paths=CORRECTED_NUMERICAL_SOURCE_PATHS,
    )
    plan = generator_freeze.plan
    gate_partition = plan.partition("r2_gate_development")
    gate_config = corrected_partition_config(gate_partition, physical_config)
    if progress is not None:
        progress("corrected gate development starting")
    gate_result = _run_corrected_partition(
        gate_partition,
        gate_config,
        workers=workers,
        progress=progress,
    )
    gates, gate_keys = calibrate_corrected_verification_gates(
        _trials(gate_result),
        gate_config,
        gate_partition,
        config,
    )
    if progress is not None:
        progress("corrected matched-model gates frozen")

    calibration_partition = plan.partition("r2_parent_calibration")
    calibration_config = corrected_partition_config(
        calibration_partition,
        physical_config,
    )
    if progress is not None:
        progress("corrected parent calibration starting")
    calibration_result = _run_corrected_partition(
        calibration_partition,
        calibration_config,
        workers=workers,
        progress=progress,
    )
    calibrations = calibrate_corrected_parent_procedures(
        _trials(calibration_result),
        gates,
        calibration_config,
        calibration_partition,
        config,
    )
    artifact = build_corrected_parent_artifact(
        generator_freeze,
        config,
        corrected_partition_evidence_digest(gate_result),
        corrected_partition_evidence_digest(calibration_result),
        gate_keys,
        gates,
        calibrations,
    )
    calibration_outcomes = apply_corrected_parent_artifact(
        _trials(calibration_result),
        artifact,
        generator_freeze,
        config,
        calibration_config,
        calibration_partition,
        split=CALIBRATION_SPLIT,
    )
    if progress is not None:
        progress(f"corrected parent artifact frozen: {artifact.protocol_digest[:12]}")
    return CorrectedParentFreezeResult(
        generator_freeze=generator_freeze,
        config=config,
        artifact=artifact,
        gate_development=gate_result,
        calibration=calibration_result,
        calibration_outcomes=calibration_outcomes,
    )


class LoadedCorrectedParentArtifact(NamedTuple):
    generator_freeze: CorrectedGeneratorFreeze
    config: CorrectedParentCalibrationConfig
    artifact: CorrectedParentCalibrationArtifact


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


def corrected_parent_artifact_payload(
    artifact: CorrectedParentCalibrationArtifact,
    generator_freeze: CorrectedGeneratorFreeze,
    config: CorrectedParentCalibrationConfig,
) -> dict:
    """Serialize the parent artifact with its content-addressed generator."""

    validate_corrected_parent_artifact(artifact, generator_freeze, config)
    return {
        "schema_version": 1,
        "generator_freeze": corrected_generator_freeze_payload(generator_freeze),
        "parent": {
            "protocol_version": artifact.protocol_version,
            "protocol_digest": artifact.protocol_digest,
            "generator_freeze_digest": artifact.generator_freeze_digest,
            "physical_protocol_digest": artifact.physical_protocol_digest,
            "gate_development_evidence_digest": (
                artifact.gate_development_evidence_digest
            ),
            "calibration_evidence_digest": artifact.calibration_evidence_digest,
            "campaign": artifact.campaign,
            "partitions": {
                "gate_development": artifact.gate_development._asdict(),
                "calibration": artifact.calibration._asdict(),
                "rehearsal": artifact.rehearsal._asdict(),
                "reserved_evaluation": artifact.reserved_evaluation._asdict(),
            },
            "statistical_config": {
                "target_block_coverage": artifact.target_block_coverage,
                "gate_target_block_retention": artifact.gate_target_block_retention,
                "gate_null_quantile": artifact.gate_null_quantile,
                "gate_monte_carlo_draws": artifact.gate_monte_carlo_draws,
                "selector": artifact.selector.__dict__,
                "cost_scenarios": [item.__dict__ for item in artifact.cost_scenarios],
            },
            "gate_random_stream_keys": [
                _key_payload(item) for item in artifact.gate_random_stream_keys
            ],
            "verification_gates": [item._asdict() for item in artifact.verification_gates],
            "procedure_calibrations": [
                item._asdict() for item in artifact.procedure_calibrations
            ],
        },
        "chronology": {
            "parent_artifact_precedes_rehearsal": True,
            "reserved_evaluation_instantiated": False,
        },
    }


def corrected_parent_artifact_from_payload(
    payload: dict,
) -> LoadedCorrectedParentArtifact:
    """Strictly parse and internally validate one serialized parent artifact."""

    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "generator_freeze",
        "parent",
        "chronology",
    }:
        raise ValueError("corrected parent payload has unexpected fields")
    if payload["schema_version"] != 1:
        raise ValueError("corrected parent artifact schema changed")
    if payload["chronology"] != {
        "parent_artifact_precedes_rehearsal": True,
        "reserved_evaluation_instantiated": False,
    }:
        raise ValueError("corrected parent chronology declaration changed")
    freeze = corrected_generator_freeze_from_payload(payload["generator_freeze"])
    parent = payload["parent"]
    if not isinstance(parent, dict) or set(parent) != {
        "protocol_version",
        "protocol_digest",
        "generator_freeze_digest",
        "physical_protocol_digest",
        "gate_development_evidence_digest",
        "calibration_evidence_digest",
        "campaign",
        "partitions",
        "statistical_config",
        "gate_random_stream_keys",
        "verification_gates",
        "procedure_calibrations",
    }:
        raise ValueError("corrected parent nested payload is malformed")
    partitions = parent["partitions"]
    statistics = parent["statistical_config"]
    if not isinstance(partitions, dict) or set(partitions) != {
        "gate_development",
        "calibration",
        "rehearsal",
        "reserved_evaluation",
    }:
        raise ValueError("corrected parent partitions are malformed")
    if not isinstance(statistics, dict) or set(statistics) != {
        "target_block_coverage",
        "gate_target_block_retention",
        "gate_null_quantile",
        "gate_monte_carlo_draws",
        "selector",
        "cost_scenarios",
    }:
        raise ValueError("corrected parent statistical config is malformed")
    config = CorrectedParentCalibrationConfig(
        target_block_coverage=statistics["target_block_coverage"],
        gate_target_block_retention=statistics["gate_target_block_retention"],
        gate_null_quantile=statistics["gate_null_quantile"],
        gate_monte_carlo_draws=statistics["gate_monte_carlo_draws"],
        selector=SelectorRule(**statistics["selector"]),
        cost_scenarios=tuple(
            CostScenario(**item) for item in statistics["cost_scenarios"]
        ),
    )
    keys = tuple(RandomStreamKey(**item) for item in parent["gate_random_stream_keys"])
    gates = tuple(
        VerificationGate(
            **{
                **item,
                "matched_block_scores": tuple(item["matched_block_scores"]),
            }
        )
        for item in parent["verification_gates"]
    )
    calibrations = tuple(
        ProcedureCalibration(
            **{
                **item,
                "block_nonconformity": tuple(item["block_nonconformity"]),
            }
        )
        for item in parent["procedure_calibrations"]
    )
    def identity(name: str) -> CorrectedPartitionIdentity:
        value = partitions[name]
        if not isinstance(value, dict) or set(value) != {
            "name",
            "paired_blocks_per_family",
        }:
            raise ValueError("corrected partition identity is malformed")
        return CorrectedPartitionIdentity(**value)

    artifact = CorrectedParentCalibrationArtifact(
        protocol_version=parent["protocol_version"],
        protocol_digest=parent["protocol_digest"],
        generator_freeze_digest=parent["generator_freeze_digest"],
        physical_protocol_digest=parent["physical_protocol_digest"],
        gate_development_evidence_digest=parent[
            "gate_development_evidence_digest"
        ],
        calibration_evidence_digest=parent["calibration_evidence_digest"],
        campaign=parent["campaign"],
        gate_development=identity("gate_development"),
        calibration=identity("calibration"),
        rehearsal=identity("rehearsal"),
        reserved_evaluation=identity("reserved_evaluation"),
        target_block_coverage=statistics["target_block_coverage"],
        gate_target_block_retention=statistics["gate_target_block_retention"],
        gate_null_quantile=statistics["gate_null_quantile"],
        gate_monte_carlo_draws=statistics["gate_monte_carlo_draws"],
        gate_random_stream_keys=keys,
        verification_gates=gates,
        selector=config.selector,
        procedure_calibrations=calibrations,
        cost_scenarios=config.cost_scenarios,
    )
    validate_corrected_parent_artifact(artifact, freeze, config)
    return LoadedCorrectedParentArtifact(freeze, config, artifact)


def save_corrected_parent_artifact(
    artifact: CorrectedParentCalibrationArtifact,
    generator_freeze: CorrectedGeneratorFreeze,
    config: CorrectedParentCalibrationConfig,
    path: Path | str,
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        corrected_parent_artifact_payload(artifact, generator_freeze, config),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    return destination


def load_corrected_parent_artifact(path: Path | str) -> LoadedCorrectedParentArtifact:
    source = Path(path).expanduser().resolve(strict=True)
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return corrected_parent_artifact_from_payload(payload)


def run_corrected_parent_rehearsal(
    artifact_path: Path | str,
    repository_root: Path | str,
    *,
    physical_config: OperatingDecisionRealismConfig = CORRECTED_REPLICATION_CONFIG,
    workers: int = 1,
    progress: Optional[Callable[[str], None]] = None,
) -> CorrectedParentRehearsalResult:
    """Load a serialized parent artifact, verify source, then open rehearsal."""

    loaded = load_corrected_parent_artifact(artifact_path)
    verify_committed_artifact_chain(
        repository_root,
        artifact_path,
        loaded.generator_freeze.source_manifest,
    )
    verify_corrected_generator_freeze(
        loaded.generator_freeze,
        repository_root,
        physical_config,
        loaded.generator_freeze.plan,
        expected_source_paths=CORRECTED_NUMERICAL_SOURCE_PATHS,
    )
    rehearsal_partition = loaded.generator_freeze.plan.partition(
        "r2_parent_rehearsal"
    )
    rehearsal_config = corrected_partition_config(
        rehearsal_partition,
        physical_config,
    )
    if progress is not None:
        progress("serialized parent verified; corrected rehearsal starting")
    rehearsal = _run_corrected_partition(
        rehearsal_partition,
        rehearsal_config,
        workers=workers,
        progress=progress,
    )
    outcomes = apply_corrected_parent_artifact(
        _trials(rehearsal),
        loaded.artifact,
        loaded.generator_freeze,
        loaded.config,
        rehearsal_config,
        rehearsal_partition,
        split=REHEARSAL_SPLIT,
    )
    summaries = summarize_calibrated_outcomes(
        outcomes,
        loaded.config,  # type: ignore[arg-type]
    )
    return CorrectedParentRehearsalResult(
        generator_freeze=loaded.generator_freeze,
        config=loaded.config,
        artifact=loaded.artifact,
        rehearsal=rehearsal,
        rehearsal_outcomes=outcomes,
        summaries=summaries,
    )


__all__ = [
    "CORRECTED_PARENT_PROTOCOL_VERSION",
    "CorrectedParentCalibrationArtifact",
    "CorrectedParentCalibrationConfig",
    "CorrectedParentFreezeResult",
    "CorrectedParentRehearsalResult",
    "CorrectedPartitionIdentity",
    "LoadedCorrectedParentArtifact",
    "apply_corrected_parent_artifact",
    "build_corrected_parent_artifact",
    "calibrate_corrected_parent_procedures",
    "calibrate_corrected_verification_gates",
    "corrected_parent_artifact_from_payload",
    "corrected_parent_artifact_payload",
    "freeze_corrected_parent_calibration",
    "load_corrected_parent_artifact",
    "rebuild_with_corrected_gate",
    "run_corrected_parent_rehearsal",
    "save_corrected_parent_artifact",
    "validate_corrected_parent_artifact",
]
