"""Blinded fixed-policy benchmark for an untouched operating decision.

The benchmark separates acquisition, verification, and final evaluation in
the type system.  Physical parameters are fitted only on acquisition runs.
Verification checks candidate adequacy without refitting physical parameters
or a fresh offset, and the final response is generated only after a decision
has been saved.
"""

from dataclasses import dataclass
import hashlib
import math
from statistics import NormalDist, fmean
from time import perf_counter
from typing import Callable, Dict, NamedTuple, Optional, Sequence, Tuple

from ..core.controls import PiecewiseConstantCurrent
from ..design.control_comparison import piecewise_electrical_energy
from ..inference.experiment_selection import candidate_current
from ..inference.sparse_sensors import (
    simulate_accessible_observations,
    sparse_withheld_current,
)
from ..numerics.integration import IntegrationDivergenceError
from ..simulation.four_node_experiments import (
    constant_current_contact_reference_experiment,
)
from .sensor_model_discrimination import (
    ALL_CHANNELS,
    COLD_EXCHANGER,
    COLD_FACE,
    FIVE_STATE_MODEL,
    FOUR_STATE_MODEL,
    HOT_EXCHANGER,
    MODEL_NAMES,
    TRUTH_CONDITIONS,
    VOLTAGE,
    CandidateModelFit,
    DiscriminationTruth,
    ObservablePrediction,
    ObservableRun,
    SensorDiscriminationConfig,
    _observed_run,
    _model_parameter_spec,
    _profiled_normalized_residuals,
    _simulate_observables,
    _truth_for_trial,
    fit_candidate_model_by_run,
)


ACQUISITION = "acquisition"
VERIFICATION = "verification"
FINAL_EVALUATION = "final_evaluation"
PHASES = (ACQUISITION, VERIFICATION, FINAL_EVALUATION)

APPROVE = "approve"
REJECT = "reject"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
DECISIONS = (APPROVE, REJECT, INSUFFICIENT_EVIDENCE)

STOP_NOW = "stop_now"
FIXED_THERMAL = "fixed_thermal"
FIXED_VOLTAGE = "fixed_voltage"
FIXED_FACE_TEMPERATURE = "fixed_face_temperature"
POLICY_NAMES = (
    STOP_NOW,
    FIXED_THERMAL,
    FIXED_VOLTAGE,
    FIXED_FACE_TEMPERATURE,
)

RUN_DURATION_SECONDS = 80.0


@dataclass(frozen=True)
class OperatingBand:
    lower_temperature: float = 285.0
    upper_temperature: float = 301.7

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.lower_temperature)
            or not math.isfinite(self.upper_temperature)
            or self.upper_temperature <= self.lower_temperature
        ):
            raise ValueError("operating temperatures must be finite and ordered")


@dataclass(frozen=True)
class OperatingRegime:
    name: str
    phase: str
    current: PiecewiseConstantCurrent
    channels: Tuple[str, ...]

    def __post_init__(self) -> None:
        channels = tuple(self.channels)
        object.__setattr__(self, "channels", channels)
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("operating-regime name must be nonempty")
        if not isinstance(self.current, PiecewiseConstantCurrent):
            raise ValueError("operating regime needs a piecewise-constant current")
        if self.phase not in PHASES:
            raise ValueError("operating regime has an unknown phase")
        if len(set(channels)) != len(channels) or any(
            channel not in ALL_CHANNELS for channel in channels
        ):
            raise ValueError("operating-regime channels must be known and distinct")
        if self.phase == FINAL_EVALUATION and channels:
            raise ValueError("final operating regime cannot expose response channels")
        if self.phase != FINAL_EVALUATION and not channels:
            raise ValueError("diagnostic regimes need at least one response channel")
        if any(time >= RUN_DURATION_SECONDS for time in self.current.transition_times):
            raise ValueError("current transition lies outside the fixed run duration")


@dataclass(frozen=True)
class OperatingRun:
    regime: OperatingRegime
    observations: ObservableRun

    def __post_init__(self) -> None:
        if self.regime.phase == FINAL_EVALUATION:
            raise ValueError("a final-evaluation response cannot enter OperatingRun")
        if self.observations.name != self.regime.name:
            raise ValueError("regime and observation names must agree")
        if self.observations.current != self.regime.current:
            raise ValueError("regime and observation currents must agree")
        if not self.observations.values:
            raise ValueError("operating run needs observations")
        keys = tuple(
            (item.channel, item.time) for item in self.observations.values
        )
        if len(set(keys)) != len(keys):
            raise ValueError("operating observations must have unique channel-times")
        if any(
            item.channel not in ALL_CHANNELS
            or not math.isfinite(item.time)
            or item.time < 0.0
            or item.time > RUN_DURATION_SECONDS
            or not math.isfinite(item.value)
            for item in self.observations.values
        ):
            raise ValueError("operating observations must be finite and in range")
        observed_channels = {item.channel for item in self.observations.values}
        if observed_channels != set(self.regime.channels):
            raise ValueError("observations must contain exactly the declared channels")
        time_grids = tuple(
            tuple(
                item.time
                for item in self.observations.values
                if item.channel == channel
            )
            for channel in self.regime.channels
        )
        if any(grid != time_grids[0] for grid in time_grids[1:]):
            raise ValueError("operating observation channels must share a time grid")


@dataclass(frozen=True)
class FixedPolicy:
    name: str
    additional_regimes: Tuple[OperatingRegime, ...]
    extra_sensor_count: int

    def __post_init__(self) -> None:
        regimes = tuple(self.additional_regimes)
        object.__setattr__(self, "additional_regimes", regimes)
        if self.name not in POLICY_NAMES:
            raise ValueError("fixed policy has an unknown name")
        if any(regime.phase != ACQUISITION for regime in regimes):
            raise ValueError("fixed-policy additions must be acquisition regimes")
        if len({regime.name for regime in regimes}) != len(regimes):
            raise ValueError("fixed-policy regime names must be distinct")
        if (
            not isinstance(self.extra_sensor_count, int)
            or isinstance(self.extra_sensor_count, bool)
            or self.extra_sensor_count < 0
        ):
            raise ValueError("extra sensor count must be a nonnegative integer")
        base_channels = {COLD_EXCHANGER, HOT_EXCHANGER}
        added_channels = {
            channel
            for regime in regimes
            for channel in regime.channels
            if channel not in base_channels
        }
        if self.extra_sensor_count != len(added_channels):
            raise ValueError("extra sensor count must match the added channels")
        expected = {
            STOP_NOW: (0, frozenset()),
            FIXED_THERMAL: (3, frozenset()),
            FIXED_VOLTAGE: (1, frozenset((VOLTAGE,))),
            FIXED_FACE_TEMPERATURE: (1, frozenset((COLD_FACE,))),
        }[self.name]
        if len(regimes) != expected[0] or frozenset(added_channels) != expected[1]:
            raise ValueError("fixed policy does not match its frozen protocol")
        exchangers = (COLD_EXCHANGER, HOT_EXCHANGER)
        expected_regime_specs = {
            STOP_NOW: (),
            FIXED_THERMAL: (
                ("thermal_0.6A_30s", candidate_current(0.6, 30.0), exchangers),
                ("thermal_0.6A_15s", candidate_current(0.6, 15.0), exchangers),
                ("thermal_0.4A_5s", candidate_current(0.4, 5.0), exchangers),
            ),
            FIXED_VOLTAGE: (
                (
                    "repeat_0.8A_20s_voltage",
                    candidate_current(0.8, 20.0),
                    (*exchangers, VOLTAGE),
                ),
            ),
            FIXED_FACE_TEMPERATURE: (
                (
                    "repeat_0.8A_20s_face_temperature",
                    candidate_current(0.8, 20.0),
                    (*exchangers, COLD_FACE),
                ),
            ),
        }[self.name]
        actual_regime_specs = tuple(
            (regime.name, regime.current, regime.channels) for regime in regimes
        )
        if actual_regime_specs != expected_regime_specs:
            raise ValueError("fixed policy does not match its frozen regimes")


class OperatingCaseId(NamedTuple):
    device_token: str
    policy_name: str
    trial_index: int
    final_schedule_id: str


@dataclass(frozen=True)
class BlindedOperatingCase:
    case_id: OperatingCaseId
    policy: FixedPolicy
    acquisition_runs: Tuple[OperatingRun, ...]
    verification_run: OperatingRun
    final_regime: OperatingRegime

    def __post_init__(self) -> None:
        acquisition_runs = tuple(self.acquisition_runs)
        object.__setattr__(self, "acquisition_runs", acquisition_runs)
        if not acquisition_runs or any(
            run.regime.phase != ACQUISITION for run in acquisition_runs
        ):
            raise ValueError("blinded case needs acquisition-phase fitting runs")
        if self.verification_run.regime.phase != VERIFICATION:
            raise ValueError("blinded case needs one verification-phase run")
        if self.final_regime.phase != FINAL_EVALUATION:
            raise ValueError("blinded case needs one untouched final regime")
        if (
            not self.case_id.device_token
            or self.case_id.policy_name not in POLICY_NAMES
            or not isinstance(self.case_id.trial_index, int)
            or isinstance(self.case_id.trial_index, bool)
            or self.case_id.trial_index < 0
            or self.case_id.final_schedule_id != "stage2_final_v1"
        ):
            raise ValueError("blinded case has an invalid identity")
        if self.policy.name != self.case_id.policy_name:
            raise ValueError("blinded case policy does not match its identity")
        expected_regimes = (
            initial_acquisition_regime(),
            *self.policy.additional_regimes,
        )
        if tuple(run.regime for run in acquisition_runs) != expected_regimes:
            raise ValueError("blinded case acquisition runs do not match its policy")
        installed = {
            channel for regime in expected_regimes for channel in regime.channels
        }
        if set(self.verification_run.regime.channels) != installed:
            raise ValueError("verification channels do not match installed channels")
        names = (
            *(run.regime.name for run in acquisition_runs),
            self.verification_run.regime.name,
            self.final_regime.name,
        )
        if len(set(names)) != len(names):
            raise ValueError("blinded-case regime names must be distinct")


class RevealedOperatingOutcome(NamedTuple):
    case_id: OperatingCaseId
    truth_condition: str
    cold_face_temperature: Tuple[float, ...]
    true_margin: float


class NumericalFailure(NamedTuple):
    model_name: str
    stage: str
    error_type: str


class AcquisitionFitSet(NamedTuple):
    fits: Tuple[CandidateModelFit, ...]
    failures: Tuple[NumericalFailure, ...]

    @property
    def failed_models(self) -> Tuple[str, ...]:
        return tuple(item.model_name for item in self.failures)


class CandidateVerification(NamedTuple):
    fit: CandidateModelFit
    normalized_score: float
    passed: bool
    failure_reason: Optional[str]


class MarginInterval(NamedTuple):
    model_name: str
    estimate: float
    local_standard_error: float
    lower: float
    upper: float


class MarginEnvelope(NamedTuple):
    lower: float
    upper: float


class SavedOperatingDecision(NamedTuple):
    case_id: OperatingCaseId
    decision: str
    decision_reason: str
    margin_envelope: Optional[MarginEnvelope]
    model_intervals: Tuple[MarginInterval, ...]
    verifications: Tuple[CandidateVerification, ...]
    failures: Tuple[NumericalFailure, ...]
    decision_computation_seconds: float

    @property
    def policy_name(self) -> str:
        return self.case_id.policy_name

    @property
    def trial_index(self) -> int:
        return self.case_id.trial_index

    @property
    def failed_models(self) -> Tuple[str, ...]:
        return tuple(item.model_name for item in self.failures)


class ScoredOperatingDecision(NamedTuple):
    truth_condition: str
    saved: SavedOperatingDecision
    true_margin: float
    true_pass: bool
    false_approval: bool
    false_rejection: bool
    interval_covered: Optional[bool]
    acquisition_run_count: int
    diagnostic_run_count: int
    energized_schedule_time_seconds: float
    acquisition_energy: float
    verification_energy: float
    total_diagnostic_energy: float
    extra_sensor_count: int


class RateEstimate(NamedTuple):
    numerator: int
    denominator: int
    rate: Optional[float]
    lower_95: Optional[float]
    upper_95: Optional[float]


class OperatingDecisionSummary(NamedTuple):
    truth_condition: str
    policy_name: str
    trial_count: int
    approvals: int
    rejections: int
    insufficient_evidence: int
    true_passing: int
    true_violating: int
    numerical_failures: int
    false_approvals: RateEstimate
    missed_violations: RateEstimate
    false_rejections: RateEstimate
    decision_coverage: RateEstimate
    abstentions: RateEstimate
    interval_coverage: RateEstimate
    acquisition_run_count: int
    diagnostic_run_count: int
    mean_energized_schedule_time_seconds: float
    mean_acquisition_energy: float
    mean_verification_energy: float
    mean_total_diagnostic_energy: float
    extra_sensor_count: int
    mean_decision_computation_seconds: float


class OperatingDecisionResult(NamedTuple):
    config: "OperatingDecisionConfig"
    policies: Tuple[FixedPolicy, ...]
    trials: Tuple[ScoredOperatingDecision, ...]
    summaries: Tuple[OperatingDecisionSummary, ...]


def final_operating_current() -> PiecewiseConstantCurrent:
    """Return the frozen topology-sensitive final schedule."""

    return PiecewiseConstantCurrent(
        transition_times=(5.0, 25.0, 38.0, 58.0),
        values=(0.0, 1.0, 0.0, -0.8, 0.0),
    )


@dataclass(frozen=True)
class OperatingDecisionConfig:
    sensor: SensorDiscriminationConfig = SensorDiscriminationConfig(
        trial_count=10,
        first_seed=191_001,
    )
    band: OperatingBand = OperatingBand()
    final_current: PiecewiseConstantCurrent = final_operating_current()
    verification_current: PiecewiseConstantCurrent = sparse_withheld_current()
    verification_score_threshold: float = 2.0
    local_interval_multiplier: float = 2.0
    margin_finite_difference_step: float = 0.01
    fit_initial_log_multipliers: Tuple[Tuple[float, float, float, float], ...] = (
        (0.0, 0.0, 0.0, 0.0),
        (-0.18, 0.18, 0.12, -0.25),
        (0.18, -0.18, -0.12, 0.25),
    )

    def __post_init__(self) -> None:
        for name, value in (
            ("verification threshold", self.verification_score_threshold),
            ("interval multiplier", self.local_interval_multiplier),
            ("margin finite-difference step", self.margin_finite_difference_step),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.final_current != final_operating_current():
            raise ValueError("Stage 2 uses the frozen final operating current")
        if self.verification_current != sparse_withheld_current():
            raise ValueError("Stage 2 uses the frozen verification current")
        if (
            len(self.fit_initial_log_multipliers) < 2
            or any(len(values) != 4 for values in self.fit_initial_log_multipliers)
            or any(
                not math.isfinite(value)
                for values in self.fit_initial_log_multipliers
                for value in values
            )
        ):
            raise ValueError("Stage 2 needs at least two finite four-parameter starts")
        selection = self.sensor.selection
        if (
            (
                selection.nominal_cold_contact_resistance,
                selection.nominal_cold_face_capacitance,
                selection.nominal_sensor_lag,
            )
            != self.sensor.fit.nominal_values
            or selection.sampling_interval != self.sensor.sampling_interval
            or selection.dense_time_step != self.sensor.dense_time_step
        ):
            raise ValueError(
                "Stage 2 resource and inference nominal settings must agree"
            )


def _acquisition_regime(
    name: str,
    amplitude: float,
    duration: float,
    channels: Sequence[str],
) -> OperatingRegime:
    return OperatingRegime(
        name=name,
        phase=ACQUISITION,
        current=candidate_current(amplitude, duration),
        channels=tuple(channels),
    )


def initial_acquisition_regime() -> OperatingRegime:
    exchangers = (COLD_EXCHANGER, HOT_EXCHANGER)
    return _acquisition_regime("initial_0.8A_20s", 0.8, 20.0, exchangers)


def default_fixed_policies() -> Tuple[FixedPolicy, ...]:
    """Return the four Stage 2 policies frozen before the pilot."""

    exchangers = (COLD_EXCHANGER, HOT_EXCHANGER)
    return (
        FixedPolicy(STOP_NOW, (), 0),
        FixedPolicy(
            FIXED_THERMAL,
            (
                _acquisition_regime("thermal_0.6A_30s", 0.6, 30.0, exchangers),
                _acquisition_regime("thermal_0.6A_15s", 0.6, 15.0, exchangers),
                _acquisition_regime("thermal_0.4A_5s", 0.4, 5.0, exchangers),
            ),
            0,
        ),
        FixedPolicy(
            FIXED_VOLTAGE,
            (
                _acquisition_regime(
                    "repeat_0.8A_20s_voltage",
                    0.8,
                    20.0,
                    (*exchangers, VOLTAGE),
                ),
            ),
            1,
        ),
        FixedPolicy(
            FIXED_FACE_TEMPERATURE,
            (
                _acquisition_regime(
                    "repeat_0.8A_20s_face_temperature",
                    0.8,
                    20.0,
                    (*exchangers, COLD_FACE),
                ),
            ),
            1,
        ),
    )


def operating_margin(
    cold_face_temperature: Sequence[float],
    band: OperatingBand,
) -> float:
    """Return the minimum distance to either edge of the operating band."""

    values = tuple(float(value) for value in cold_face_temperature)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("operating margin needs finite temperatures")
    return min(
        min(value - band.lower_temperature, band.upper_temperature - value)
        for value in values
    )


def classify_margin_envelope(
    envelope: Optional[MarginEnvelope],
) -> str:
    """Apply the frozen approve/reject/abstain boundary."""

    if envelope is None:
        return INSUFFICIENT_EVIDENCE
    if envelope.lower > envelope.upper or any(
        not math.isfinite(value) for value in envelope
    ):
        raise ValueError("margin envelope must be finite and ordered")
    if envelope.lower >= 0.0:
        return APPROVE
    if envelope.upper < 0.0:
        return REJECT
    return INSUFFICIENT_EVIDENCE


def envelope_margin_intervals(
    intervals: Sequence[MarginInterval],
) -> Optional[MarginEnvelope]:
    intervals = tuple(intervals)
    if not intervals:
        return None
    if len({item.model_name for item in intervals}) != len(intervals) or any(
        item.model_name not in MODEL_NAMES
        or item.local_standard_error < 0.0
        or item.lower > item.upper
        or item.estimate < item.lower
        or item.estimate > item.upper
        or any(
            not math.isfinite(value)
            for value in (
                item.estimate,
                item.local_standard_error,
                item.lower,
                item.upper,
            )
        )
        for item in intervals
    ):
        raise ValueError("model margin intervals must be finite, ordered, and unique")
    return MarginEnvelope(
        lower=min(item.lower for item in intervals),
        upper=max(item.upper for item in intervals),
    )


def rate_estimate(numerator: int, denominator: int) -> RateEstimate:
    """Return a 95% Wilson interval, or N/A fields for an empty denominator."""

    if (
        isinstance(numerator, bool)
        or isinstance(denominator, bool)
        or not isinstance(numerator, int)
        or not isinstance(denominator, int)
        or numerator < 0
        or denominator < 0
        or numerator > denominator
    ):
        raise ValueError("rate counts must satisfy 0 <= numerator <= denominator")
    if denominator == 0:
        return RateEstimate(numerator, denominator, None, None, None)
    confidence = 0.95
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    n = float(denominator)
    rate = numerator / n
    denominator_term = 1.0 + z * z / n
    center = (rate + z * z / (2.0 * n)) / denominator_term
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / n + z * z / (4.0 * n * n))
        / denominator_term
    )
    return RateEstimate(
        numerator,
        denominator,
        rate,
        max(0.0, center - radius),
        min(1.0, center + radius),
    )


def _truth_model(truth_condition: str) -> str:
    if truth_condition == TRUTH_CONDITIONS[0]:
        return FOUR_STATE_MODEL
    if truth_condition == TRUTH_CONDITIONS[1]:
        return FIVE_STATE_MODEL
    raise ValueError("operating-decision truth condition is unknown")


def _interface_mass_for_model(
    model_name: str,
    truth: DiscriminationTruth,
) -> Optional[float]:
    return truth.interface_mass if model_name == FIVE_STATE_MODEL else None


def _run_seed(
    config: OperatingDecisionConfig,
    truth_condition: str,
    trial_index: int,
    run_name: str,
) -> int:
    offsets = {
        "initial_0.8A_20s": 100,
        "thermal_0.6A_30s": 200,
        "thermal_0.6A_15s": 300,
        "thermal_0.4A_5s": 400,
        "repeat_0.8A_20s_voltage": 500,
        "repeat_0.8A_20s_face_temperature": 500,
        "fixed_verification": 9_900,
    }
    if run_name not in offsets:
        raise ValueError("operating-decision run has no frozen seed offset")
    return (
        config.sensor.first_seed
        + TRUTH_CONDITIONS.index(truth_condition) * 1_000_000
        + trial_index * 10_000
        + offsets[run_name]
    )


def _case_id(
    config: OperatingDecisionConfig,
    truth_condition: str,
    trial_index: int,
    policy_name: str,
) -> OperatingCaseId:
    material = (
        f"thermotwin-stage2|{config.sensor.first_seed}|"
        f"{truth_condition}|{trial_index}"
    ).encode("utf-8")
    token = hashlib.sha256(material).hexdigest()[:20]
    return OperatingCaseId(
        device_token=token,
        policy_name=policy_name,
        trial_index=trial_index,
        final_schedule_id="stage2_final_v1",
    )


def _observed_operating_run(
    regime: OperatingRegime,
    *,
    truth_model: str,
    truth: DiscriminationTruth,
    truth_condition: str,
    trial_index: int,
    config: OperatingDecisionConfig,
) -> OperatingRun:
    prediction = _simulate_observables(
        truth_model,
        regime.current,
        regime.channels,
        truth.physical_values,
        _interface_mass_for_model(truth_model, truth),
        config.sensor,
    )
    observations = _observed_run(
        prediction,
        name=regime.name,
        current=regime.current,
        seed=_run_seed(config, truth_condition, trial_index, regime.name),
        config=config.sensor,
    )
    return OperatingRun(regime, observations)


def build_blinded_operating_case(
    truth_condition: str,
    trial_index: int,
    policy: FixedPolicy,
    config: OperatingDecisionConfig = OperatingDecisionConfig(),
) -> BlindedOperatingCase:
    """Generate visible diagnostic data without generating the final response."""

    if not isinstance(policy, FixedPolicy):
        raise ValueError("operating decision needs a fixed policy")
    if (
        not isinstance(trial_index, int)
        or isinstance(trial_index, bool)
        or trial_index < 0
    ):
        raise ValueError("trial index must be a nonnegative integer")
    truth_model = _truth_model(truth_condition)
    truth = _truth_for_trial(config.sensor, trial_index)
    initial = initial_acquisition_regime()
    acquisition_regimes = (initial, *policy.additional_regimes)
    acquisition_runs = tuple(
        _observed_operating_run(
            regime,
            truth_model=truth_model,
            truth=truth,
            truth_condition=truth_condition,
            trial_index=trial_index,
            config=config,
        )
        for regime in acquisition_regimes
    )
    installed = {
        channel
        for regime in acquisition_regimes
        for channel in regime.channels
    }
    verification_regime = OperatingRegime(
        name="fixed_verification",
        phase=VERIFICATION,
        current=config.verification_current,
        channels=tuple(channel for channel in ALL_CHANNELS if channel in installed),
    )
    verification_run = _observed_operating_run(
        verification_regime,
        truth_model=truth_model,
        truth=truth,
        truth_condition=truth_condition,
        trial_index=trial_index,
        config=config,
    )
    final_regime = OperatingRegime(
        name="untouched_final_operating_schedule",
        phase=FINAL_EVALUATION,
        current=config.final_current,
        channels=(),
    )
    return BlindedOperatingCase(
        case_id=_case_id(
            config,
            truth_condition,
            trial_index,
            policy.name,
        ),
        policy=policy,
        acquisition_runs=acquisition_runs,
        verification_run=verification_run,
        final_regime=final_regime,
    )


def reveal_operating_outcome(
    truth_condition: str,
    trial_index: int,
    case: BlindedOperatingCase,
    config: OperatingDecisionConfig = OperatingDecisionConfig(),
) -> RevealedOperatingOutcome:
    """Generate final truth for scoring after a decision has been saved."""

    final_regime = case.final_regime
    if final_regime.phase != FINAL_EVALUATION or final_regime.channels:
        raise ValueError("only an untouched final regime can be revealed")
    if final_regime.current != config.final_current:
        raise ValueError("final regime does not use the frozen operating current")
    expected_id = _case_id(
        config,
        truth_condition,
        trial_index,
        case.case_id.policy_name,
    )
    if case.case_id != expected_id:
        raise ValueError("case identity does not match the requested reveal")
    truth_model = _truth_model(truth_condition)
    truth = _truth_for_trial(config.sensor, trial_index)
    prediction = _simulate_observables(
        truth_model,
        final_regime.current,
        (),
        truth.physical_values,
        _interface_mass_for_model(truth_model, truth),
        config.sensor,
    )
    return RevealedOperatingOutcome(
        case_id=case.case_id,
        truth_condition=truth_condition,
        cold_face_temperature=prediction.cold_face,
        true_margin=operating_margin(prediction.cold_face, config.band),
    )


def fit_acquisition_models(
    case: BlindedOperatingCase,
    config: OperatingDecisionConfig = OperatingDecisionConfig(),
) -> AcquisitionFitSet:
    """Fit both candidate topologies using acquisition observations only."""

    runs = tuple(run.observations for run in case.acquisition_runs)
    fits = []
    failures = []
    for model_name in MODEL_NAMES:
        parameter_count = 3 if model_name == FOUR_STATE_MODEL else 4
        candidates = []
        for initial in config.fit_initial_log_multipliers:
            try:
                fit = fit_candidate_model_by_run(
                    model_name,
                    runs,
                    config.sensor,
                    initial_log_multipliers=initial[:parameter_count],
                )
                objective = _acquisition_fit_objective(
                    fit,
                    case.acquisition_runs,
                    config,
                )
                candidates.append((objective, fit))
            except (ArithmeticError, IntegrationDivergenceError, ValueError) as error:
                failures.append(
                    NumericalFailure(
                        model_name,
                        "acquisition_fit",
                        type(error).__name__,
                    )
                )
        if candidates:
            fits.append(min(candidates, key=lambda item: item[0])[1])
    return AcquisitionFitSet(tuple(fits), tuple(failures))


def _acquisition_fit_objective(
    fit: CandidateModelFit,
    runs: Sequence[OperatingRun],
    config: OperatingDecisionConfig,
) -> float:
    """Return the full profiled acquisition objective, including one prior."""

    residuals = []
    for run in runs:
        prediction = _simulate_observables(
            fit.model_name,
            run.regime.current,
            run.regime.channels,
            fit.physical_values,
            fit.interface_mass,
            config.sensor,
        )
        residuals.extend(
            _profiled_normalized_residuals(
                run.observations,
                prediction,
                config.sensor,
            )
        )
    _, prior_scales, _ = _model_parameter_spec(fit.model_name, config.sensor)
    residuals.extend(
        value / scale
        for value, scale in zip(fit.log_multipliers, prior_scales)
    )
    if not residuals or any(not math.isfinite(value) for value in residuals):
        raise ValueError("acquisition fit objective is invalid")
    return fmean(value * value for value in residuals)


def verification_score(
    fit: CandidateModelFit,
    verification_run: OperatingRun,
    config: OperatingDecisionConfig = OperatingDecisionConfig(),
) -> Tuple[float, ObservablePrediction]:
    """Score a new run under the declared offset covariance, without refitting."""

    if verification_run.regime.phase != VERIFICATION:
        raise ValueError("candidate adequacy must use the verification phase")
    prediction = _simulate_observables(
        fit.model_name,
        verification_run.regime.current,
        verification_run.regime.channels,
        fit.physical_values,
        fit.interface_mass,
        config.sensor,
    )
    predicted = {
        (item.channel, item.time): item.value for item in prediction.values
    }
    noise_by_channel = dict(config.sensor.channel_noise)
    quadratic = 0.0
    sample_count = 0
    for channel in verification_run.regime.channels:
        residuals = tuple(
            item.value - predicted[(item.channel, item.time)]
            for item in verification_run.observations.values
            if item.channel == channel
        )
        if not residuals:
            raise ValueError("verification channel has no observations")
        mean_residual = fmean(residuals)
        noise = noise_by_channel[channel]
        bias_standard_deviation = config.sensor.run_bias_noise_ratio * noise
        quadratic += sum(
            (value - mean_residual) ** 2 / (noise * noise)
            for value in residuals
        )
        quadratic += mean_residual * mean_residual / (
            bias_standard_deviation * bias_standard_deviation
            + noise * noise / len(residuals)
        )
        sample_count += len(residuals)
    return quadratic / sample_count, prediction


def verify_candidate_models(
    fit_set: AcquisitionFitSet,
    verification_run: OperatingRun,
    config: OperatingDecisionConfig = OperatingDecisionConfig(),
) -> Tuple[CandidateVerification, ...]:
    """Retain every fitted model that passes the fixed adequacy threshold."""

    results = []
    for fit in fit_set.fits:
        try:
            score, _ = verification_score(fit, verification_run, config)
        except (ArithmeticError, IntegrationDivergenceError, ValueError):
            results.append(
                CandidateVerification(fit, math.inf, False, "verification_failure")
            )
            continue
        if fit.reached_bound:
            results.append(
                CandidateVerification(fit, score, False, "fit_reached_bound")
            )
        elif not math.isfinite(score):
            results.append(
                CandidateVerification(fit, score, False, "nonfinite_score")
            )
        elif score > config.verification_score_threshold:
            results.append(
                CandidateVerification(fit, score, False, "inadequate_verification")
            )
        else:
            results.append(CandidateVerification(fit, score, True, None))
    return tuple(results)


def _margin_at_offsets(
    model_name: str,
    offsets: Sequence[float],
    final_regime: OperatingRegime,
    config: OperatingDecisionConfig,
) -> float:
    physical_values = tuple(
        nominal * math.exp(offset)
        for nominal, offset in zip(config.sensor.fit.nominal_values, offsets[:3])
    )
    interface_mass = (
        None
        if model_name == FOUR_STATE_MODEL
        else config.sensor.interface_mass_nominal * math.exp(offsets[3])
    )
    prediction = _simulate_observables(
        model_name,
        final_regime.current,
        (),
        physical_values,  # type: ignore[arg-type]
        interface_mass,
        config.sensor,
    )
    return operating_margin(prediction.cold_face, config.band)


def forecast_margin_interval(
    fit: CandidateModelFit,
    final_regime: OperatingRegime,
    config: OperatingDecisionConfig = OperatingDecisionConfig(),
) -> MarginInterval:
    """Propagate local fit covariance to the scalar final operating margin."""

    if final_regime.phase != FINAL_EVALUATION or final_regime.channels:
        raise ValueError("margin forecast requires the untouched final regime")
    if fit.reached_bound:
        raise ValueError("a bound-hit fit cannot support a local margin interval")
    offsets = tuple(fit.log_multipliers)
    parameter_count = len(offsets)
    expected_parameter_count = 3 if fit.model_name == FOUR_STATE_MODEL else 4
    if fit.model_name not in MODEL_NAMES or parameter_count != expected_parameter_count:
        raise ValueError("fit parameters do not match a candidate model")
    if (
        len(fit.covariance) != parameter_count
        or any(len(row) != parameter_count for row in fit.covariance)
    ):
        raise ValueError("fit covariance has the wrong shape")
    if any(
        not math.isfinite(value)
        for row in fit.covariance
        for value in row
    ):
        raise ValueError("fit covariance must be finite")
    for row in range(parameter_count):
        for column in range(row):
            left_value = fit.covariance[row][column]
            right_value = fit.covariance[column][row]
            scale = max(1.0, abs(left_value), abs(right_value))
            if abs(left_value - right_value) > 1.0e-8 * scale:
                raise ValueError("fit covariance must be symmetric")
    reconstructed = tuple(
        nominal * math.exp(offset)
        for nominal, offset in zip(config.sensor.fit.nominal_values, offsets[:3])
    )
    if any(
        not math.isclose(left, right, rel_tol=1.0e-10, abs_tol=1.0e-12)
        for left, right in zip(reconstructed, fit.physical_values)
    ):
        raise ValueError("fit physical values do not match this configuration")
    if fit.model_name == FIVE_STATE_MODEL:
        reconstructed_mass = config.sensor.interface_mass_nominal * math.exp(
            offsets[3]
        )
        if fit.interface_mass is None or not math.isclose(
            reconstructed_mass,
            fit.interface_mass,
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        ):
            raise ValueError("fit interface mass does not match this configuration")
    bounds, _, _ = _model_parameter_spec(fit.model_name, config.sensor)
    step = config.margin_finite_difference_step
    gradient = []
    for index in range(parameter_count):
        left = list(offsets)
        right = list(offsets)
        left[index] = max(bounds[index][0], left[index] - step)
        right[index] = min(bounds[index][1], right[index] + step)
        denominator = right[index] - left[index]
        if denominator <= 0.0:
            raise ValueError("margin finite-difference interval collapsed")
        gradient.append(
            (
                _margin_at_offsets(fit.model_name, right, final_regime, config)
                - _margin_at_offsets(fit.model_name, left, final_regime, config)
            )
            / denominator
        )
    contributions = tuple(
        gradient[row] * fit.covariance[row][column] * gradient[column]
        for row in range(parameter_count)
        for column in range(parameter_count)
    )
    variance = sum(contributions)
    roundoff_tolerance = 1.0e-10 * max(
        1.0,
        sum(abs(value) for value in contributions),
    )
    if not math.isfinite(variance) or variance < -roundoff_tolerance:
        raise ValueError("local margin variance is invalid")
    standard_error = math.sqrt(max(0.0, variance))
    estimate = _margin_at_offsets(
        fit.model_name,
        offsets,
        final_regime,
        config,
    )
    radius = config.local_interval_multiplier * standard_error
    values = (estimate, standard_error, estimate - radius, estimate + radius)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("local margin interval is nonfinite")
    return MarginInterval(
        fit.model_name,
        estimate,
        standard_error,
        estimate - radius,
        estimate + radius,
    )


def decide_blinded_case(
    case: BlindedOperatingCase,
    *,
    config: OperatingDecisionConfig = OperatingDecisionConfig(),
) -> SavedOperatingDecision:
    """Save a decision without accepting final response or truth as input."""

    started = perf_counter()
    fit_set = fit_acquisition_models(case, config)
    verifications = verify_candidate_models(
        fit_set,
        case.verification_run,
        config,
    )
    intervals = []
    failures = list(fit_set.failures)
    for verification in verifications:
        if verification.failure_reason in ("verification_failure", "nonfinite_score"):
            failures.append(
                NumericalFailure(
                    verification.fit.model_name,
                    "verification",
                    verification.failure_reason,
                )
            )
    for verification in verifications:
        if not verification.passed:
            continue
        try:
            intervals.append(
                forecast_margin_interval(verification.fit, case.final_regime, config)
            )
        except (ArithmeticError, IntegrationDivergenceError, ValueError) as error:
            failures.append(
                NumericalFailure(
                    verification.fit.model_name,
                    "uncertainty",
                    type(error).__name__,
                )
            )

    candidate_envelope = envelope_margin_intervals(intervals)
    envelope = None if failures else candidate_envelope
    if fit_set.failures:
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
        reason = "no_verified_candidate"
    else:
        decision = classify_margin_envelope(envelope)
        if decision == APPROVE:
            reason = "verified_envelope_inside_band"
        elif decision == REJECT:
            reason = "verified_envelope_below_zero"
        else:
            reason = "verified_envelope_crosses_zero"
    return SavedOperatingDecision(
        case_id=case.case_id,
        decision=decision,
        decision_reason=reason,
        margin_envelope=envelope,
        model_intervals=tuple(intervals),
        verifications=verifications,
        failures=tuple(failures),
        decision_computation_seconds=perf_counter() - started,
    )


def nominal_schedule_energy(
    current: PiecewiseConstantCurrent,
    config: OperatingDecisionConfig = OperatingDecisionConfig(),
) -> float:
    """Return nominal modeled net terminal energy for one 80-second run."""

    selection = config.sensor.selection
    _, trajectory = simulate_accessible_observations(
        current,
        cold_contact_resistance=selection.nominal_cold_contact_resistance,
        cold_face_thermal_capacitance=selection.nominal_cold_face_capacitance,
        sensor_time_constant=selection.nominal_sensor_lag,
        sampling_interval=selection.sampling_interval,
        dense_time_step=selection.dense_time_step,
    )
    reference = constant_current_contact_reference_experiment()
    return piecewise_electrical_energy(
        trajectory.time,
        trajectory.cold_face,
        trajectory.hot_face,
        reference.thermoelectric_parameters,
        current,
        start_time=trajectory.time[0],
        end_time=trajectory.time[-1],
    )


def _score_saved_decision(
    saved: SavedOperatingDecision,
    revealed: RevealedOperatingOutcome,
    case: BlindedOperatingCase,
    energy_by_current: Dict[PiecewiseConstantCurrent, float],
    config: OperatingDecisionConfig,
) -> ScoredOperatingDecision:
    if saved.case_id != case.case_id or revealed.case_id != case.case_id:
        raise ValueError(
            "case, saved decision, and revealed outcome must share identity"
        )
    if saved.decision not in DECISIONS:
        raise ValueError("saved decision is unknown")
    _truth_model(revealed.truth_condition)
    recomputed_margin = operating_margin(
        revealed.cold_face_temperature,
        config.band,
    )
    if not math.isclose(
        revealed.true_margin,
        recomputed_margin,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise ValueError("revealed margin does not match the revealed trajectory")
    true_pass = recomputed_margin >= 0.0
    envelope = saved.margin_envelope
    interval_covered = (
        None
        if envelope is None
        else envelope.lower <= recomputed_margin <= envelope.upper
    )
    acquisition_energy = sum(
        energy_by_current[run.regime.current] for run in case.acquisition_runs
    )
    verification_energy = energy_by_current[case.verification_run.regime.current]
    diagnostic_run_count = len(case.acquisition_runs) + 1
    return ScoredOperatingDecision(
        truth_condition=revealed.truth_condition,
        saved=saved,
        true_margin=recomputed_margin,
        true_pass=true_pass,
        false_approval=saved.decision == APPROVE and not true_pass,
        false_rejection=saved.decision == REJECT and true_pass,
        interval_covered=interval_covered,
        acquisition_run_count=len(case.acquisition_runs),
        diagnostic_run_count=diagnostic_run_count,
        energized_schedule_time_seconds=RUN_DURATION_SECONDS * diagnostic_run_count,
        acquisition_energy=acquisition_energy,
        verification_energy=verification_energy,
        total_diagnostic_energy=acquisition_energy + verification_energy,
        extra_sensor_count=len(
            {
                channel
                for run in case.acquisition_runs
                for channel in run.regime.channels
                if channel not in (COLD_EXCHANGER, HOT_EXCHANGER)
            }
        ),
    )


def score_saved_decision(
    saved: SavedOperatingDecision,
    revealed: RevealedOperatingOutcome,
    case: BlindedOperatingCase,
    config: OperatingDecisionConfig = OperatingDecisionConfig(),
) -> ScoredOperatingDecision:
    """Score a previously saved decision against newly revealed final truth."""

    currents = {
        *(run.regime.current for run in case.acquisition_runs),
        case.verification_run.regime.current,
    }
    energies = {
        current: nominal_schedule_energy(current, config) for current in currents
    }
    return _score_saved_decision(saved, revealed, case, energies, config)


def summarize_operating_decisions(
    trials: Sequence[ScoredOperatingDecision],
    policies: Sequence[FixedPolicy],
    config: OperatingDecisionConfig = OperatingDecisionConfig(),
) -> Tuple[OperatingDecisionSummary, ...]:
    summaries = []
    trials = tuple(trials)
    policies = tuple(policies)
    if not policies or len({policy.name for policy in policies}) != len(policies):
        raise ValueError("operating-decision summaries need unique policies")
    expected_keys = {
        (condition, policy.name, trial_index)
        for condition in TRUTH_CONDITIONS
        for policy in policies
        for trial_index in range(config.sensor.trial_count)
    }
    actual_keys = {
        (item.truth_condition, item.saved.policy_name, item.saved.trial_index)
        for item in trials
    }
    if len(actual_keys) != len(trials) or actual_keys != expected_keys:
        raise ValueError(
            "operating-decision trials are missing, duplicated, or unknown"
        )
    for truth_condition in TRUTH_CONDITIONS:
        for trial_index in range(config.sensor.trial_count):
            paired = tuple(
                item.true_margin
                for item in trials
                if item.truth_condition == truth_condition
                and item.saved.trial_index == trial_index
            )
            if len(paired) != len(policies) or any(
                not math.isclose(value, paired[0], rel_tol=0.0, abs_tol=1.0e-12)
                for value in paired[1:]
            ):
                raise ValueError("paired policies must share one revealed truth margin")
    for truth_condition in TRUTH_CONDITIONS:
        for policy in policies:
            selected = tuple(
                item
                for item in trials
                if item.truth_condition == truth_condition
                and item.saved.policy_name == policy.name
            )
            if len(selected) != config.sensor.trial_count:
                raise ValueError("each policy summary needs every paired trial")
            resource_fields = {
                (
                    item.acquisition_run_count,
                    item.diagnostic_run_count,
                    item.energized_schedule_time_seconds,
                    item.acquisition_energy,
                    item.verification_energy,
                    item.total_diagnostic_energy,
                    item.extra_sensor_count,
                )
                for item in selected
            }
            if len(resource_fields) != 1:
                raise ValueError("policy resource accounting must be trial invariant")
            approvals = sum(item.saved.decision == APPROVE for item in selected)
            rejections = sum(item.saved.decision == REJECT for item in selected)
            insufficient = sum(
                item.saved.decision == INSUFFICIENT_EVIDENCE for item in selected
            )
            true_passing = sum(item.true_pass for item in selected)
            true_violating = len(selected) - true_passing
            false_approvals = sum(item.false_approval for item in selected)
            false_rejections = sum(item.false_rejection for item in selected)
            interval_trials = tuple(
                item for item in selected if item.interval_covered is not None
            )
            summaries.append(
                OperatingDecisionSummary(
                    truth_condition=truth_condition,
                    policy_name=policy.name,
                    trial_count=len(selected),
                    approvals=approvals,
                    rejections=rejections,
                    insufficient_evidence=insufficient,
                    true_passing=true_passing,
                    true_violating=true_violating,
                    numerical_failures=sum(
                        bool(item.saved.failures) for item in selected
                    ),
                    false_approvals=rate_estimate(false_approvals, approvals),
                    missed_violations=rate_estimate(
                        false_approvals, true_violating
                    ),
                    false_rejections=rate_estimate(
                        false_rejections, rejections
                    ),
                    decision_coverage=rate_estimate(
                        approvals + rejections, len(selected)
                    ),
                    abstentions=rate_estimate(insufficient, len(selected)),
                    interval_coverage=rate_estimate(
                        sum(item.interval_covered is True for item in interval_trials),
                        len(interval_trials),
                    ),
                    acquisition_run_count=selected[0].acquisition_run_count,
                    diagnostic_run_count=selected[0].diagnostic_run_count,
                    mean_energized_schedule_time_seconds=fmean(
                        item.energized_schedule_time_seconds for item in selected
                    ),
                    mean_acquisition_energy=fmean(
                        item.acquisition_energy for item in selected
                    ),
                    mean_verification_energy=fmean(
                        item.verification_energy for item in selected
                    ),
                    mean_total_diagnostic_energy=fmean(
                        item.total_diagnostic_energy for item in selected
                    ),
                    extra_sensor_count=selected[0].extra_sensor_count,
                    mean_decision_computation_seconds=fmean(
                        item.saved.decision_computation_seconds for item in selected
                    ),
                )
            )
    return tuple(summaries)


def nominal_operating_margins(
    config: OperatingDecisionConfig = OperatingDecisionConfig(),
) -> Tuple[Tuple[str, float], ...]:
    """Return final margins at the nominal parameters for both model families."""

    values = []
    for model_name in MODEL_NAMES:
        prediction = _simulate_observables(
            model_name,
            config.final_current,
            (),
            config.sensor.fit.nominal_values,
            (
                config.sensor.interface_mass_nominal
                if model_name == FIVE_STATE_MODEL
                else None
            ),
            config.sensor,
        )
        values.append((model_name, operating_margin(prediction.cold_face, config.band)))
    return tuple(values)


def run_operating_decision(
    config: OperatingDecisionConfig = OperatingDecisionConfig(),
    *,
    progress: Optional[Callable[[str], None]] = None,
) -> OperatingDecisionResult:
    """Run the paired Stage 2 developmental pilot under all fixed policies."""

    policies = default_fixed_policies()
    if len({policy.name for policy in policies}) != len(policies):
        raise ValueError("operating-decision policies must have unique names")
    currents = {config.verification_current}
    for policy in policies:
        currents.add(initial_acquisition_regime().current)
        currents.update(regime.current for regime in policy.additional_regimes)
    energies = {
        current: nominal_schedule_energy(current, config) for current in currents
    }

    trials = []
    for truth_condition in TRUTH_CONDITIONS:
        for trial_index in range(config.sensor.trial_count):
            if progress is not None:
                progress(
                    f"{truth_condition}: paired trial "
                    f"{trial_index + 1}/{config.sensor.trial_count}"
                )
            for policy in policies:
                case = build_blinded_operating_case(
                    truth_condition,
                    trial_index,
                    policy,
                    config,
                )
                saved = decide_blinded_case(
                    case,
                    config=config,
                )
                revealed = reveal_operating_outcome(
                    truth_condition,
                    trial_index,
                    case,
                    config,
                )
                trials.append(
                    _score_saved_decision(saved, revealed, case, energies, config)
                )
    return OperatingDecisionResult(
        config=config,
        policies=policies,
        trials=tuple(trials),
        summaries=summarize_operating_decisions(trials, policies, config),
    )
