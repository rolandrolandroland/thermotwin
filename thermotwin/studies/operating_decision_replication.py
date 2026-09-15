"""Collision-free generator for the audited operating-decision replication.

This module does not alter the legacy Stage 3--5 generator. It rebuilds the
same physical experiment with semantic random-stream keys, explicit paired
reuse, and realized terminal-energy scoring. Reserved-evaluation chronology is
owned by the later calibration artifact; this foundation can only construct a
named partition supplied by its caller.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
from typing import Callable, Optional, Tuple

from ..simulation.temporary_face_sensor import TemporaryFaceSensor
from .operating_decision import (
    APPROVE,
    FINAL_EVALUATION,
    INSUFFICIENT_EVIDENCE,
    REJECT,
    VERIFICATION,
    CandidateVerification,
    FixedPolicy,
    NumericalFailure,
    OperatingRegime,
    RevealedOperatingOutcome,
    SavedOperatingDecision,
    ScoredOperatingDecision,
    classify_margin_envelope,
    default_fixed_policies,
    envelope_margin_intervals,
    initial_acquisition_regime,
    operating_margin,
)
from .operating_decision_random_streams import (
    RANDOM_STREAM_PROTOCOL_VERSION,
    RandomStream,
    RandomStreamAudit,
    RandomStreamRegistry,
    StreamUse,
    audit_stream_uses,
    device_truth_stream,
    observation_stream,
)
from .operating_decision_provenance import (
    verify_committed_artifact_chain,
)
from .operating_decision_realism import (
    CORRECTED_FINAL_SCHEDULE_ID,
    STAGE3_TRUTH_CONDITIONS,
    OperatingDecisionRealismConfig,
    RealismTruth,
    RealisticBlindedOperatingCase,
    RealisticObservablePrediction,
    RealisticOperatingRun,
    RunInstrumentation,
    Stage3CaseId,
    _instrumentation_for_regime,
    _truth_model_name,
    _truth_prediction,
    decide_realistic_blinded_case,
    score_realistic_saved_decision,
)
from .operating_decision_resources import (
    RealizedCaseEnergy,
    realized_blinded_case_energy_from_truth,
)
from .sensor_model_discrimination import (
    ALL_CHANNELS,
    ObservableRun,
    ObservableValue,
)


CORRECTED_GENERATOR_VERSION = "operating_decision_generator_v4"
CORRECTED_REPLICATION_CAMPAIGN = (
    "operating_decision_audit_replication_corrected_v2_2026_09"
)
RESERVED_EVALUATION_PARTITION_NAME = "r2_reserved_evaluation"
CORRECTED_FIT_ITERATIONS = 12


@dataclass(frozen=True)
class CorrectedOperatingDecisionRealismConfig(OperatingDecisionRealismConfig):
    """Corrected inference settings without changing the legacy Stage 3 schema."""

    series_resistance_fit_bounds: Tuple[float, float] = (0.005, 1.0)
    face_sensor_capacitance_fit_bounds: Tuple[float, float] = (0.5, 24.0)
    face_sensor_response_fit_bounds: Tuple[float, float] = (0.25, 12.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        for name, truth_bounds, fit_bounds, nominal in (
            (
                "series-resistance",
                self.series_resistance_bounds,
                self.series_resistance_fit_bounds,
                self.series_resistance_nominal,
            ),
            (
                "face-sensor capacitance",
                self.face_sensor_capacitance_bounds,
                self.face_sensor_capacitance_fit_bounds,
                self.face_sensor_capacitance_nominal,
            ),
            (
                "face-sensor response",
                self.face_sensor_response_bounds,
                self.face_sensor_response_fit_bounds,
                self.face_sensor_response_nominal,
            ),
        ):
            if (
                len(fit_bounds) != 2
                or not all(math.isfinite(value) for value in fit_bounds)
                or fit_bounds[0] <= 0.0
                or fit_bounds[1] <= fit_bounds[0]
                or not fit_bounds[0] < nominal < fit_bounds[1]
                or fit_bounds[0] > truth_bounds[0]
                or fit_bounds[1] < truth_bounds[1]
            ):
                raise ValueError(
                    f"corrected {name} fit bounds must contain truth and nominal"
                )


CORRECTED_REPLICATION_CONFIG = CorrectedOperatingDecisionRealismConfig(
    sensor=replace(
        OperatingDecisionRealismConfig().sensor,
        fit_iterations=CORRECTED_FIT_ITERATIONS,
    ),
)


def _label(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{name} must be a nonempty, trimmed label")


@dataclass(frozen=True)
class CorrectedPartition:
    """A semantic namespace and paired-block count for corrected generation."""

    name: str
    block_count: int
    campaign: str = CORRECTED_REPLICATION_CAMPAIGN

    def __post_init__(self) -> None:
        _label("partition name", self.name)
        _label("campaign", self.campaign)
        if (
            not isinstance(self.block_count, int)
            or isinstance(self.block_count, bool)
            or self.block_count <= 0
        ):
            raise ValueError("corrected partition needs a positive block count")


_RESERVED_AUTHORIZATION_SEAL = object()


@dataclass(frozen=True)
class _ReservedEvaluationAuthorization:
    repository_root: str
    artifact_path: str
    campaign: str
    partition_name: str
    block_count: int
    physical_protocol_digest: str
    commit: str
    paths: Tuple[str, ...]
    seal: object


@dataclass(frozen=True)
class _ReservedEvaluationRequest:
    repository_root: str
    artifact_path: str
    expected_commit: str
    expected_paths: Tuple[str, ...]


def _authorize_reserved_evaluation(
    partition: CorrectedPartition,
    repository_root: Path | str,
    artifact_path: Path | str,
    physical_config: OperatingDecisionRealismConfig,
) -> _ReservedEvaluationAuthorization:
    """Validate the committed guard chain before issuing a reserved-run seal."""

    if partition.name != RESERVED_EVALUATION_PARTITION_NAME:
        raise ValueError("reserved authorization requires the reserved partition")
    from .operating_decision_replication_calibration import (
        validate_corrected_parent_artifact,
    )
    from .operating_decision_replication_guard import (
        validate_corrected_guard_artifact,
        load_corrected_guard_artifact,
    )
    from .operating_decision_replication_protocol import (
        CORRECTED_NUMERICAL_SOURCE_PATHS,
        verify_corrected_generator_freeze,
    )

    loaded = load_corrected_guard_artifact(artifact_path)
    manifest = loaded.generator_freeze.source_manifest
    verification = verify_committed_artifact_chain(
        repository_root,
        artifact_path,
        manifest,
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
    embedded_partition = loaded.generator_freeze.plan.partition(
        RESERVED_EVALUATION_PARTITION_NAME
    )
    if embedded_partition != partition:
        raise ValueError("reserved partition differs from the committed guard artifact")
    evaluation_config = corrected_partition_config(partition, physical_config)
    if (
        corrected_physical_protocol_digest(evaluation_config)
        != loaded.artifact.physical_protocol_digest
    ):
        raise ValueError("reserved evaluation physics differs from its artifact")
    source_paths = {item.path for item in manifest.files}
    artifact_paths = tuple(
        item for item in verification.paths if item not in source_paths
    )
    if len(artifact_paths) != 1:
        raise ValueError("reserved authorization needs one committed artifact")
    return _ReservedEvaluationAuthorization(
        verification.repository_root,
        str(Path(verification.repository_root, artifact_paths[0])),
        partition.campaign,
        partition.name,
        partition.block_count,
        corrected_physical_protocol_digest(evaluation_config),
        verification.commit,
        verification.paths,
        _RESERVED_AUTHORIZATION_SEAL,
    )


def _validated_reserved_authorization(
    partition: CorrectedPartition,
    authorization: Optional[_ReservedEvaluationAuthorization],
    config: Optional[OperatingDecisionRealismConfig] = None,
) -> Optional[_ReservedEvaluationAuthorization]:
    if partition.name != RESERVED_EVALUATION_PARTITION_NAME:
        return None
    if not (
        isinstance(authorization, _ReservedEvaluationAuthorization)
        and authorization.seal is _RESERVED_AUTHORIZATION_SEAL
        and authorization.campaign == partition.campaign
        and authorization.partition_name == partition.name
        and authorization.block_count == partition.block_count
        and (
            config is None
            or authorization.physical_protocol_digest
            == corrected_physical_protocol_digest(config)
        )
    ):
        raise ValueError(
            "reserved evaluation requires committed-artifact authorization"
        )
    return authorization


@dataclass(frozen=True)
class CorrectedTrialRecord:
    """Complete post-reveal record for one policy/family/block."""

    partition: CorrectedPartition
    truth_condition: str
    block: int
    truth: RealismTruth
    case: RealisticBlindedOperatingCase
    revealed: RevealedOperatingOutcome
    scored: ScoredOperatingDecision
    nominal_selection_energy: float
    realized_energy: RealizedCaseEnergy

    def __post_init__(self) -> None:
        if not isinstance(self.partition, CorrectedPartition):
            raise TypeError("corrected trial needs a semantic partition")
        if self.truth_condition not in STAGE3_TRUTH_CONDITIONS:
            raise ValueError("corrected trial has an unknown truth family")
        if (
            not isinstance(self.block, int)
            or isinstance(self.block, bool)
            or not 0 <= self.block < self.partition.block_count
        ):
            raise ValueError("corrected trial block lies outside its partition")
        if self.block != self.case.case_id.trial_index:
            raise ValueError("corrected trial block and case identity differ")
        if (
            self.revealed.case_id != self.case.case_id
            or self.scored.saved.case_id != self.case.case_id
            or self.scored.truth_condition != self.truth_condition
            or self.revealed.truth_condition != self.truth_condition
            or self.realized_energy.case_id != self.case.case_id
        ):
            raise ValueError("corrected trial records do not share one case identity")
        if not math.isfinite(self.nominal_selection_energy) or (
            self.nominal_selection_energy < 0.0
        ):
            raise ValueError("nominal selection energy must be finite and nonnegative")
        if not math.isclose(
            self.scored.total_diagnostic_energy,
            self.realized_energy.total_diagnostic_energy,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("corrected score must contain realized terminal energy")


@dataclass(frozen=True)
class CorrectedPartitionResult:
    partition: CorrectedPartition
    config: OperatingDecisionRealismConfig
    records: Tuple[CorrectedTrialRecord, ...]
    random_stream_uses: Tuple[StreamUse, ...]
    random_stream_audit: RandomStreamAudit

    def __post_init__(self) -> None:
        if not isinstance(self.partition, CorrectedPartition):
            raise TypeError("corrected result needs a semantic partition")
        records = tuple(self.records)
        object.__setattr__(self, "records", records)
        uses = tuple(self.random_stream_uses)
        object.__setattr__(self, "random_stream_uses", uses)
        if any(not isinstance(item, CorrectedTrialRecord) for item in records):
            raise TypeError("corrected result records must be CorrectedTrialRecord values")
        if any(not isinstance(item, StreamUse) for item in uses):
            raise TypeError("corrected result stream manifest contains an invalid use")
        if any(item.partition != self.partition for item in records):
            raise ValueError("corrected record belongs to a different partition")
        for item in records:
            expected_case_id = corrected_case_id(
                self.partition,
                item.truth_condition,
                item.block,
                item.case.policy.name,
                self.config,
            )
            if item.case.case_id != expected_case_id:
                raise ValueError("corrected case identity is not reproducible")
        expected = {
            (truth, block, policy.name)
            for block in range(self.partition.block_count)
            for truth in STAGE3_TRUTH_CONDITIONS
            for policy in default_fixed_policies()
        }
        actual = {
            (
                item.truth_condition,
                item.block,
                item.case.policy.name,
            )
            for item in records
        }
        if len(actual) != len(records) or actual != expected:
            raise ValueError("corrected partition records are incomplete or duplicated")
        if self.config.sensor.trial_count != self.partition.block_count:
            raise ValueError("corrected physical config and partition counts differ")
        for use in uses:
            key = use.stream.key
            if (
                key.protocol_version != RANDOM_STREAM_PROTOCOL_VERSION
                or key.campaign != self.partition.campaign
                or key.partition != self.partition.name
                or not 0 <= key.block < self.partition.block_count
            ):
                raise ValueError("corrected stream belongs to a different namespace")
        recomputed_audit = audit_stream_uses(uses)
        if self.random_stream_audit != recomputed_audit:
            raise ValueError("corrected random-stream audit was not recomputed faithfully")
        recomputed_audit.assert_clean()


def corrected_partition_config(
    partition: CorrectedPartition,
    base: OperatingDecisionRealismConfig = CORRECTED_REPLICATION_CONFIG,
) -> OperatingDecisionRealismConfig:
    """Return the physical config with only its block count adapted.

    ``first_seed`` is zeroed because corrected generation never consumes the
    arithmetic legacy namespace.
    """

    return replace(
        base,
        sensor=replace(
            base.sensor,
            trial_count=partition.block_count,
            first_seed=0,
        ),
    )


def _validate_block(
    partition: CorrectedPartition,
    block: int,
    config: OperatingDecisionRealismConfig,
) -> None:
    if (
        not isinstance(block, int)
        or isinstance(block, bool)
        or not 0 <= block < partition.block_count
    ):
        raise ValueError("block must lie inside the corrected partition")
    if config.sensor.trial_count != partition.block_count:
        raise ValueError("corrected config trial count must match its partition")


def _truth_pairing_id(partition: CorrectedPartition, block: int, purpose: str) -> str:
    return f"{partition.campaign}/{partition.name}/{block}/shared-truth/{purpose}"


def _register_shared_truth(
    registry: RandomStreamRegistry,
    stream: RandomStream,
    partition: CorrectedPartition,
    block: int,
    purpose: str,
) -> RandomStream:
    pairing_id = _truth_pairing_id(partition, block, purpose)
    for family in STAGE3_TRUTH_CONDITIONS:
        registry.register(
            stream,
            consumer=f"{family}/block-{block}/truth/{purpose}",
            pairing_member=family,
            pairing_id=pairing_id,
        )
    return stream


def _shared_truth_generator(
    registry: RandomStreamRegistry,
    partition: CorrectedPartition,
    block: int,
    purpose: str,
):
    stream = device_truth_stream(
        campaign=partition.campaign,
        partition=partition.name,
        block=block,
        purpose=purpose,
    )
    return _register_shared_truth(
        registry,
        stream,
        partition,
        block,
        purpose,
    ).new_generator()


def _truncated_lognormal(
    random_source,
    nominal: float,
    spread: float,
    bounds: Tuple[float, float],
) -> float:
    for _ in range(10_000):
        value = nominal * math.exp(random_source.gauss(0.0, spread))
        if bounds[0] <= value <= bounds[1]:
            return value
    raise RuntimeError("corrected lognormal draw did not enter its calibration range")


def corrected_truth_for_block(
    partition: CorrectedPartition,
    block: int,
    config: OperatingDecisionRealismConfig,
    registry: RandomStreamRegistry,
    *,
    _reserved_authorization: Optional[_ReservedEvaluationAuthorization] = None,
) -> RealismTruth:
    """Generate one device shared deliberately by all three truth variants."""

    _validated_reserved_authorization(partition, _reserved_authorization, config)
    _validate_block(partition, block, config)
    if not isinstance(registry, RandomStreamRegistry):
        raise TypeError("corrected truth generation needs a stream registry")
    physical = tuple(
        nominal
        * math.exp(
            _shared_truth_generator(
                registry,
                partition,
                block,
                f"physical_parameter_{index}",
            ).gauss(0.0, spread)
        )
        for index, (nominal, spread) in enumerate(
            zip(
                config.sensor.fit.nominal_values,
                config.sensor.truth_log_standard_deviations,
            )
        )
    )
    interface_mass = config.sensor.interface_mass_nominal * math.exp(
        _shared_truth_generator(
            registry,
            partition,
            block,
            "interface_mass",
        ).gauss(0.0, config.sensor.interface_mass_log_standard_deviation)
    )
    series_resistance = _truncated_lognormal(
        _shared_truth_generator(
            registry,
            partition,
            block,
            "series_resistance",
        ),
        config.series_resistance_nominal,
        config.series_resistance_truth_log_standard_deviation,
        config.series_resistance_bounds,
    )
    sensor_capacitance = _truncated_lognormal(
        _shared_truth_generator(
            registry,
            partition,
            block,
            "face_sensor_capacitance",
        ),
        config.face_sensor_capacitance_nominal,
        config.face_sensor_truth_log_standard_deviation,
        config.face_sensor_capacitance_bounds,
    )
    sensor_response = _truncated_lognormal(
        _shared_truth_generator(
            registry,
            partition,
            block,
            "face_sensor_response",
        ),
        config.face_sensor_response_nominal,
        config.face_sensor_truth_log_standard_deviation,
        config.face_sensor_response_bounds,
    )
    beta_stream = device_truth_stream(
        campaign=partition.campaign,
        partition=partition.name,
        block=block,
        purpose="contact_beta",
        family=STAGE3_TRUTH_CONDITIONS[2],
    )
    registry.register(
        beta_stream,
        consumer=f"{STAGE3_TRUTH_CONDITIONS[2]}/block-{block}/truth/contact_beta",
    )
    beta = beta_stream.new_generator().uniform(*config.contact_beta_bounds)
    return RealismTruth(
        physical_values=physical,  # type: ignore[arg-type]
        interface_mass=interface_mass,
        series_resistance=series_resistance,
        face_sensor=TemporaryFaceSensor(sensor_capacitance, sensor_response),
        contact_beta=beta,
    )


def _instrumented_run_name(regime: OperatingRegime, instrumentation: RunInstrumentation) -> str:
    return f"{regime.name}|temporary_face_sensor={int(instrumentation.temporary_face_sensor)}"


def _registered_observation_generator(
    *,
    registry: RandomStreamRegistry,
    partition: CorrectedPartition,
    block: int,
    truth_condition: str,
    policy_name: str,
    regime: OperatingRegime,
    instrumentation: RunInstrumentation,
    channel: str,
    purpose: str,
):
    run_name = _instrumented_run_name(regime, instrumentation)
    stream = observation_stream(
        campaign=partition.campaign,
        partition=partition.name,
        block=block,
        purpose=purpose,
        family=truth_condition,
        run=run_name,
        channel=channel,
    )
    registry.register(
        stream,
        consumer=(
            f"{truth_condition}/block-{block}/{policy_name}/"
            f"{run_name}/{channel}/{purpose}"
        ),
        pairing_member=policy_name,
        pairing_id=(
            f"{partition.campaign}/{partition.name}/{truth_condition}/{block}/"
            f"{run_name}/{channel}/{purpose}"
        ),
    )
    return stream.new_generator()


def _corrected_observed_run(
    prediction: RealisticObservablePrediction,
    regime: OperatingRegime,
    instrumentation: RunInstrumentation,
    *,
    truth_condition: str,
    block: int,
    policy_name: str,
    partition: CorrectedPartition,
    config: OperatingDecisionRealismConfig,
    registry: RandomStreamRegistry,
) -> RealisticOperatingRun:
    noise_by_channel = dict(config.sensor.channel_noise)
    values = []
    for channel in ALL_CHANNELS:
        selected = tuple(item for item in prediction.values if item.channel == channel)
        if not selected:
            continue
        bias_random = _registered_observation_generator(
            registry=registry,
            partition=partition,
            block=block,
            truth_condition=truth_condition,
            policy_name=policy_name,
            regime=regime,
            instrumentation=instrumentation,
            channel=channel,
            purpose="run_bias",
        )
        noise_random = _registered_observation_generator(
            registry=registry,
            partition=partition,
            block=block,
            truth_condition=truth_condition,
            policy_name=policy_name,
            regime=regime,
            instrumentation=instrumentation,
            channel=channel,
            purpose="white_noise",
        )
        noise = noise_by_channel[channel]
        bias = bias_random.gauss(0.0, config.sensor.run_bias_noise_ratio * noise)
        values.extend(
            ObservableValue(
                channel=item.channel,
                time=item.time,
                value=item.value + bias + noise_random.gauss(0.0, noise),
            )
            for item in selected
        )
    values.sort(key=lambda item: (item.time, ALL_CHANNELS.index(item.channel)))
    observations = ObservableRun(
        name=regime.name,
        current=regime.current,
        values=tuple(values),
    )
    return RealisticOperatingRun(regime, observations, instrumentation)


def corrected_physical_protocol_digest(config: OperatingDecisionRealismConfig) -> str:
    normalized = replace(
        config,
        sensor=replace(config.sensor, trial_count=1, first_seed=0),
    )
    return hashlib.sha256(repr(normalized).encode("utf-8")).hexdigest()


def corrected_case_id(
    partition: CorrectedPartition,
    truth_condition: str,
    block: int,
    policy_name: str,
    config: OperatingDecisionRealismConfig,
) -> Stage3CaseId:
    _truth_model_name(truth_condition)
    material = "|".join(
        (
            CORRECTED_GENERATOR_VERSION,
            RANDOM_STREAM_PROTOCOL_VERSION,
            partition.campaign,
            partition.name,
            str(block),
            truth_condition,
            corrected_physical_protocol_digest(config),
        )
    ).encode("utf-8")
    return Stage3CaseId(
        device_token=hashlib.sha256(material).hexdigest()[:20],
        policy_name=policy_name,
        trial_index=block,
        final_schedule_id=CORRECTED_FINAL_SCHEDULE_ID,
    )


def build_corrected_blinded_case(
    truth_condition: str,
    block: int,
    policy: FixedPolicy,
    truth: RealismTruth,
    partition: CorrectedPartition,
    config: OperatingDecisionRealismConfig,
    registry: RandomStreamRegistry,
    *,
    _reserved_authorization: Optional[_ReservedEvaluationAuthorization] = None,
) -> RealisticBlindedOperatingCase:
    """Build acquisition and verification data without exposing final response."""

    _validated_reserved_authorization(partition, _reserved_authorization, config)
    _truth_model_name(truth_condition)
    _validate_block(partition, block, config)
    if not isinstance(policy, FixedPolicy):
        raise ValueError("corrected operating decision needs a fixed policy")
    if not isinstance(truth, RealismTruth):
        raise ValueError("corrected operating decision needs a versioned truth")
    acquisition_regimes = (initial_acquisition_regime(), *policy.additional_regimes)
    acquisition_runs = []
    for regime in acquisition_regimes:
        instrumentation = _instrumentation_for_regime(regime)
        prediction = _truth_prediction(
            truth_condition,
            truth,
            regime,
            instrumentation,
            config,
        )
        acquisition_runs.append(
            _corrected_observed_run(
                prediction,
                regime,
                instrumentation,
                truth_condition=truth_condition,
                block=block,
                policy_name=policy.name,
                partition=partition,
                config=config,
                registry=registry,
            )
        )
    installed = {
        channel for regime in acquisition_regimes for channel in regime.channels
    }
    verification_regime = OperatingRegime(
        name="fixed_verification",
        phase=VERIFICATION,
        current=config.verification_current,
        channels=tuple(channel for channel in ALL_CHANNELS if channel in installed),
    )
    verification_instrumentation = _instrumentation_for_regime(verification_regime)
    verification_prediction = _truth_prediction(
        truth_condition,
        truth,
        verification_regime,
        verification_instrumentation,
        config,
    )
    verification_run = _corrected_observed_run(
        verification_prediction,
        verification_regime,
        verification_instrumentation,
        truth_condition=truth_condition,
        block=block,
        policy_name=policy.name,
        partition=partition,
        config=config,
        registry=registry,
    )
    final_regime = OperatingRegime(
        name="untouched_final_operating_schedule",
        phase=FINAL_EVALUATION,
        current=config.final_current,
        channels=(),
    )
    return RealisticBlindedOperatingCase(
        case_id=corrected_case_id(
            partition,
            truth_condition,
            block,
            policy.name,
            config,
        ),
        policy=policy,
        acquisition_runs=tuple(acquisition_runs),
        verification_run=verification_run,
        final_regime=final_regime,
        final_instrumentation=RunInstrumentation(False),
    )


def reveal_corrected_operating_outcome(
    truth_condition: str,
    block: int,
    case: RealisticBlindedOperatingCase,
    truth: RealismTruth,
    partition: CorrectedPartition,
    config: OperatingDecisionRealismConfig,
) -> RevealedOperatingOutcome:
    """Reveal the unloaded final response after a decision has been saved."""

    expected = corrected_case_id(
        partition,
        truth_condition,
        block,
        case.policy.name,
        config,
    )
    if case.case_id != expected:
        raise ValueError("corrected case identity does not match the reveal")
    prediction = _truth_prediction(
        truth_condition,
        truth,
        case.final_regime,
        case.final_instrumentation,
        config,
    )
    return RevealedOperatingOutcome(
        case_id=case.case_id,  # type: ignore[arg-type]
        truth_condition=truth_condition,
        cold_face_temperature=prediction.cold_face,
        true_margin=operating_margin(prediction.cold_face, config.band),
    )


def score_corrected_saved_decision(
    saved: SavedOperatingDecision,
    *,
    truth_condition: str,
    block: int,
    case: RealisticBlindedOperatingCase,
    truth: RealismTruth,
    partition: CorrectedPartition,
    config: OperatingDecisionRealismConfig,
) -> CorrectedTrialRecord:
    """Reveal and score a previously saved decision with realized resources."""

    if not isinstance(saved, SavedOperatingDecision) or saved.case_id != case.case_id:
        raise ValueError("corrected scoring requires a saved decision for this case")
    revealed = reveal_corrected_operating_outcome(
        truth_condition,
        block,
        case,
        truth,
        partition,
        config,
    )
    nominal = score_realistic_saved_decision(saved, revealed, case, config)
    realized = realized_blinded_case_energy_from_truth(
        truth_condition,
        block,
        case,
        truth,
        config,
        expected_case_id=case.case_id,
    )
    scored = nominal._replace(
        acquisition_energy=realized.acquisition_energy,
        verification_energy=realized.verification_energy,
        total_diagnostic_energy=realized.total_diagnostic_energy,
    )
    return CorrectedTrialRecord(
        partition=partition,
        truth_condition=truth_condition,
        block=block,
        truth=truth,
        case=case,
        revealed=revealed,
        scored=scored,
        nominal_selection_energy=nominal.total_diagnostic_energy,
        realized_energy=realized,
    )


def decide_corrected_blinded_case(
    case: RealisticBlindedOperatingCase,
    *,
    config: OperatingDecisionRealismConfig,
) -> SavedOperatingDecision:
    """Save a decision after excluding numerically inadmissible candidates."""

    saved = decide_realistic_blinded_case(case, config=config)
    excluded_models = {
        item.fit.model_name
        for item in saved.verifications
        if not item.fit.converged or item.fit.reached_bound
    }
    if not excluded_models:
        return saved
    verifications = tuple(
        CandidateVerification(
            item.fit,
            item.normalized_score,
            False,
            (
                "fit_reached_bound"
                if item.fit.reached_bound
                else "optimizer_not_converged"
            ),
        )
        if (
            item.fit.model_name in excluded_models
            and item.failure_reason not in (
                "verification_failure",
                "nonfinite_score",
            )
        )
        else item
        for item in saved.verifications
    )
    intervals = tuple(
        item for item in saved.model_intervals if item.model_name not in excluded_models
    )
    failures = tuple(
        item
        for item in saved.failures
        if not (
            item.model_name in excluded_models
            and item.stage == "uncertainty"
        )
    )
    candidate_envelope = envelope_margin_intervals(intervals)
    envelope = None if failures else candidate_envelope
    if any(item.stage == "acquisition_fit" for item in failures):
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
    return saved._replace(
        decision=decision,
        decision_reason=decision_reason,
        margin_envelope=envelope,
        model_intervals=intervals if not failures else (),
        verifications=verifications,
        failures=failures,
    )


def _run_corrected_block(
    partition: CorrectedPartition,
    block: int,
    config: OperatingDecisionRealismConfig,
    _reserved_authorization: Optional[_ReservedEvaluationAuthorization] = None,
    _reserved_request: Optional[_ReservedEvaluationRequest] = None,
) -> Tuple[int, Tuple[CorrectedTrialRecord, ...], Tuple[StreamUse, ...]]:
    if partition.name == RESERVED_EVALUATION_PARTITION_NAME:
        if _reserved_authorization is None and isinstance(
            _reserved_request,
            _ReservedEvaluationRequest,
        ):
            _reserved_authorization = _authorize_reserved_evaluation(
                partition,
                _reserved_request.repository_root,
                _reserved_request.artifact_path,
                config,
            )
            if (
                _reserved_authorization.commit
                != _reserved_request.expected_commit
                or _reserved_authorization.paths
                != _reserved_request.expected_paths
            ):
                raise ValueError(
                    "reserved artifact chain changed after authorization"
                )
        _reserved_authorization = _validated_reserved_authorization(
            partition,
            _reserved_authorization,
            config,
        )
    registry = RandomStreamRegistry()
    truth = corrected_truth_for_block(
        partition,
        block,
        config,
        registry,
        _reserved_authorization=_reserved_authorization,
    )
    records = []
    for truth_condition in STAGE3_TRUTH_CONDITIONS:
        for policy in default_fixed_policies():
            case = build_corrected_blinded_case(
                truth_condition,
                block,
                policy,
                truth,
                partition,
                config,
                registry,
                _reserved_authorization=_reserved_authorization,
            )
            saved = decide_corrected_blinded_case(case, config=config)
            records.append(
                score_corrected_saved_decision(
                    saved,
                    truth_condition=truth_condition,
                    block=block,
                    case=case,
                    truth=truth,
                    partition=partition,
                    config=config,
                )
            )
    registry.audit().assert_clean()
    return block, tuple(records), registry.uses


def _run_corrected_block_worker(arguments):
    return _run_corrected_block(*arguments)


def _run_corrected_partition(
    partition: CorrectedPartition,
    config: Optional[OperatingDecisionRealismConfig] = None,
    *,
    workers: int = 1,
    progress: Optional[Callable[[str], None]] = None,
    _reserved_authorization: Optional[_ReservedEvaluationAuthorization] = None,
) -> CorrectedPartitionResult:
    """Generate, decide, reveal, and audit one corrected partition."""

    if config is None:
        config = corrected_partition_config(partition)
    _reserved_authorization = _validated_reserved_authorization(
        partition,
        _reserved_authorization,
        config,
    )
    _validate_block(partition, 0, config)
    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise ValueError("corrected worker count must be a positive integer")
    completed = {}
    reserved_request = (
            _ReservedEvaluationRequest(
                _reserved_authorization.repository_root,
                _reserved_authorization.artifact_path,
                _reserved_authorization.commit,
            _reserved_authorization.paths,
        )
        if _reserved_authorization is not None
        else None
    )
    if workers == 1:
        for block in range(partition.block_count):
            completed[block] = _run_corrected_block(
                partition,
                block,
                config,
                None,
                reserved_request,
            )[1:]
            if progress is not None:
                progress(f"{partition.name}: block {block + 1}/{partition.block_count}")
    else:
        with ProcessPoolExecutor(max_workers=min(workers, partition.block_count)) as pool:
            futures = {
                pool.submit(
                    _run_corrected_block_worker,
                    (partition, block, config, None, reserved_request),
                ): block
                for block in range(partition.block_count)
            }
            for future in as_completed(futures):
                block, records, uses = future.result()
                completed[block] = (records, uses)
                if progress is not None:
                    progress(
                        f"{partition.name}: block {len(completed)}/{partition.block_count}"
                    )
    records = tuple(
        record
        for block in range(partition.block_count)
        for record in completed[block][0]
    )
    uses = tuple(
        use
        for block in range(partition.block_count)
        for use in completed[block][1]
    )
    audit = audit_stream_uses(uses)
    audit.assert_clean()
    return CorrectedPartitionResult(
        partition,
        config,
        records,
        uses,
        audit,
    )


def run_corrected_partition(
    partition: CorrectedPartition,
    config: Optional[OperatingDecisionRealismConfig] = None,
    *,
    workers: int = 1,
    progress: Optional[Callable[[str], None]] = None,
) -> CorrectedPartitionResult:
    """Run a development partition while keeping the reserved cohort sealed."""

    if partition.name == RESERVED_EVALUATION_PARTITION_NAME:
        raise ValueError(
            "reserved evaluation requires a committed, manifest-bound artifact"
        )
    return _run_corrected_partition(
        partition,
        config,
        workers=workers,
        progress=progress,
    )


__all__ = [
    "CORRECTED_GENERATOR_VERSION",
    "CORRECTED_FIT_ITERATIONS",
    "CORRECTED_REPLICATION_CONFIG",
    "CORRECTED_REPLICATION_CAMPAIGN",
    "RESERVED_EVALUATION_PARTITION_NAME",
    "CorrectedPartition",
    "CorrectedOperatingDecisionRealismConfig",
    "CorrectedPartitionResult",
    "CorrectedTrialRecord",
    "build_corrected_blinded_case",
    "corrected_case_id",
    "corrected_partition_config",
    "corrected_physical_protocol_digest",
    "corrected_truth_for_block",
    "decide_corrected_blinded_case",
    "reveal_corrected_operating_outcome",
    "run_corrected_partition",
    "score_corrected_saved_decision",
]
