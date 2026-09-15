"""Stage 3 stress test for the blinded operating-decision experiment.

The fixed Stage 2 policies are rerun after adding three effects that make the
diagnostic channels physically costly and imperfect: electrical terminal
resistance, a thermally loading temporary face probe, and a truth-only
temperature-dependent cold contact.  The final operating run is still hidden
until a decision has been saved, and the temporary probe is removed from that
target device before the final run.
"""

from dataclasses import dataclass, replace
import hashlib
import math
import random
from statistics import fmean
from time import perf_counter
from typing import Callable, Dict, NamedTuple, Optional, Sequence, Tuple

from ..core.controls import PiecewiseConstantCurrent, current_at
from ..design.control_comparison import piecewise_electrical_energy
from ..inference.sparse_sensors import sparse_withheld_current
from ..numerics.integration import IntegrationDivergenceError
from ..numerics.matrices import inverse_and_determinant
from ..observations.test_stand import regular_measurement_times
from ..physics.thermoelectric import voltage
from ..simulation.four_node_experiments import (
    constant_current_contact_reference_experiment,
    run_four_node_contact_experiment,
)
from ..simulation.interface_mass_mismatch import (
    InterfaceMassMismatch,
    integrate_interface_mass_truth,
)
from ..simulation.operating_realism import (
    integrate_temperature_dependent_contact_temporary_sensor,
)
from ..simulation.temperature_dependent_contact import (
    TemperatureDependentColdContact,
    integrate_temperature_dependent_contact_truth,
)
from ..simulation.temporary_face_sensor import (
    TemporaryFaceSensor,
    integrate_four_node_temporary_face_sensor,
    integrate_interface_mass_temporary_face_sensor,
)
from .operating_decision import (
    ACQUISITION,
    APPROVE,
    DECISIONS,
    FINAL_EVALUATION,
    FIXED_FACE_TEMPERATURE,
    INSUFFICIENT_EVIDENCE,
    REJECT,
    RUN_DURATION_SECONDS,
    VERIFICATION,
    CandidateVerification,
    FixedPolicy,
    MarginInterval,
    NumericalFailure,
    OperatingBand,
    OperatingDecisionSummary,
    OperatingRegime,
    OperatingRun,
    RevealedOperatingOutcome,
    SavedOperatingDecision,
    ScoredOperatingDecision,
    classify_margin_envelope,
    default_fixed_policies,
    envelope_margin_intervals,
    final_operating_current,
    initial_acquisition_regime,
    operating_margin,
    rate_estimate,
)
from .sensor_model_discrimination import (
    ALL_CHANNELS,
    COLD_EXCHANGER,
    COLD_FACE,
    COLD_HEAT_RATE,
    FIVE_STATE_MODEL,
    FOUR_STATE_MODEL,
    HOT_EXCHANGER,
    MODEL_NAMES,
    TRUTH_CONDITIONS,
    VOLTAGE,
    ObservableRun,
    ObservableValue,
    SensorDiscriminationConfig,
    _lagged_temperature_values,
    _observed_run,
    _trajectory_at_times,
    _truth_for_trial,
)


TEMPERATURE_DEPENDENT_CONTACT = "temperature_dependent_contact"
STAGE3_TRUTH_CONDITIONS = (*TRUTH_CONDITIONS, TEMPERATURE_DEPENDENT_CONTACT)
STAGE3_FINAL_SCHEDULE_ID = "stage3_final_v1"
CORRECTED_FINAL_SCHEDULE_ID = "operating_decision_corrected_final_v2"
FIT_SCALED_GRADIENT_TOLERANCE = 1.0e-4
FIT_STEP_TOLERANCE = 1.0e-6
FIT_RELATIVE_OBJECTIVE_TOLERANCE = 1.0e-8


@dataclass(frozen=True)
class RunInstrumentation:
    """Physical instrumentation installed during one diagnostic run."""

    temporary_face_sensor: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.temporary_face_sensor, bool):
            raise ValueError("temporary-face-sensor flag must be boolean")


@dataclass(frozen=True)
class RealisticOperatingRun:
    regime: OperatingRegime
    observations: ObservableRun
    instrumentation: RunInstrumentation = RunInstrumentation()

    def __post_init__(self) -> None:
        if not isinstance(self.regime, OperatingRegime):
            raise ValueError("realistic operating run needs a valid regime")
        OperatingRun(self.regime, self.observations)
        if self.instrumentation.temporary_face_sensor != (
            COLD_FACE in self.regime.channels
        ):
            raise ValueError(
                "the Stage 3 face channel requires its physical probe state"
            )


class Stage3CaseId(NamedTuple):
    device_token: str
    policy_name: str
    trial_index: int
    final_schedule_id: str


@dataclass(frozen=True)
class RealisticBlindedOperatingCase:
    case_id: Stage3CaseId
    policy: FixedPolicy
    acquisition_runs: Tuple[RealisticOperatingRun, ...]
    verification_run: RealisticOperatingRun
    final_regime: OperatingRegime
    final_instrumentation: RunInstrumentation = RunInstrumentation()

    def __post_init__(self) -> None:
        runs = tuple(self.acquisition_runs)
        object.__setattr__(self, "acquisition_runs", runs)
        if not runs or any(run.regime.phase != ACQUISITION for run in runs):
            raise ValueError("Stage 3 case needs acquisition runs")
        if self.verification_run.regime.phase != VERIFICATION:
            raise ValueError("Stage 3 case needs one verification run")
        if self.final_regime.phase != FINAL_EVALUATION or self.final_regime.channels:
            raise ValueError("Stage 3 case needs an untouched final regime")
        if self.final_instrumentation.temporary_face_sensor:
            raise ValueError("temporary probe must be removed before final evaluation")
        if (
            not self.case_id.device_token
            or self.case_id.policy_name != self.policy.name
            or self.case_id.trial_index < 0
            or self.case_id.final_schedule_id
            not in (STAGE3_FINAL_SCHEDULE_ID, CORRECTED_FINAL_SCHEDULE_ID)
        ):
            raise ValueError("Stage 3 case has an invalid identity")
        expected_regimes = (
            initial_acquisition_regime(),
            *self.policy.additional_regimes,
        )
        if tuple(run.regime for run in runs) != expected_regimes:
            raise ValueError("Stage 3 acquisition runs do not match the policy")
        installed = {
            channel for regime in expected_regimes for channel in regime.channels
        }
        if set(self.verification_run.regime.channels) != installed:
            raise ValueError("verification channels do not match installed channels")
        face_policy = self.policy.name == FIXED_FACE_TEMPERATURE
        loaded_acquisition = tuple(
            run.regime.name for run in runs if run.instrumentation.temporary_face_sensor
        )
        expected_loaded = (
            ("repeat_0.8A_20s_face_temperature",) if face_policy else ()
        )
        if loaded_acquisition != expected_loaded:
            raise ValueError("only the prospective face repeat may load the device")
        if self.verification_run.instrumentation.temporary_face_sensor != face_policy:
            raise ValueError("face probe installation must persist through verification")


class RealismTruth(NamedTuple):
    physical_values: Tuple[float, float, float]
    interface_mass: float
    series_resistance: float
    face_sensor: TemporaryFaceSensor
    contact_beta: float


class RealisticObservablePrediction(NamedTuple):
    values: Tuple[ObservableValue, ...]
    time: Tuple[float, ...]
    cold_face: Tuple[float, ...]
    hot_face: Tuple[float, ...]


class RealisticCandidateFit(NamedTuple):
    model_name: str
    log_multipliers: Tuple[float, ...]
    parameter_names: Tuple[str, ...]
    physical_values: Tuple[float, float, float]
    interface_mass: Optional[float]
    series_resistance: float
    face_sensor: Optional[TemporaryFaceSensor]
    objective: float
    covariance: Tuple[Tuple[float, ...], ...]
    reached_bound: bool
    evaluation_count: int
    converged: bool = False
    termination_reason: str = "legacy_status_unavailable"
    completed_iterations: int = 0
    accepted_iterations: int = 0
    scaled_gradient_infinity_norm: float = float("nan")
    last_step_infinity_norm: Optional[float] = None
    last_relative_objective_reduction: Optional[float] = None


class RealisticAcquisitionFitSet(NamedTuple):
    fits: Tuple[RealisticCandidateFit, ...]
    failures: Tuple[NumericalFailure, ...]


class OperatingDecisionRealismResult(NamedTuple):
    config: "OperatingDecisionRealismConfig"
    policies: Tuple[FixedPolicy, ...]
    trials: Tuple[ScoredOperatingDecision, ...]
    summaries: Tuple[OperatingDecisionSummary, ...]


@dataclass(frozen=True)
class _ParameterSpec:
    names: Tuple[str, ...]
    nominal_values: Tuple[float, ...]
    log_bounds: Tuple[Tuple[float, float], ...]
    prior_log_standard_deviations: Tuple[float, ...]


@dataclass(frozen=True)
class OperatingDecisionRealismConfig:
    """Frozen synthetic assumptions for the Stage 3 development stress test."""

    sensor: SensorDiscriminationConfig = SensorDiscriminationConfig(
        trial_count=10,
        first_seed=191_001,
    )
    band: OperatingBand = OperatingBand()
    final_current: PiecewiseConstantCurrent = final_operating_current()
    verification_current: PiecewiseConstantCurrent = sparse_withheld_current()
    series_resistance_nominal: float = 0.10
    series_resistance_truth_log_standard_deviation: float = 0.35
    series_resistance_bounds: Tuple[float, float] = (0.01, 0.50)
    series_resistance_prior_log_standard_deviation: float = 0.50
    face_sensor_capacitance_nominal: float = 5.0
    face_sensor_capacitance_bounds: Tuple[float, float] = (1.0, 12.0)
    face_sensor_response_nominal: float = 2.5
    face_sensor_response_bounds: Tuple[float, float] = (0.5, 6.0)
    face_sensor_truth_log_standard_deviation: float = 0.35
    face_sensor_prior_log_standard_deviation: float = 0.35
    contact_beta_bounds: Tuple[float, float] = (0.08, 0.16)
    verification_score_threshold: float = 2.0
    local_interval_multiplier: float = 2.0
    margin_finite_difference_step: float = 0.01

    def __post_init__(self) -> None:
        if self.final_current != final_operating_current():
            raise ValueError("Stage 3 uses the frozen final operating current")
        if self.verification_current != sparse_withheld_current():
            raise ValueError("Stage 3 uses the frozen verification current")
        for name, value in (
            ("series-resistance nominal", self.series_resistance_nominal),
            (
                "series-resistance truth spread",
                self.series_resistance_truth_log_standard_deviation,
            ),
            (
                "series-resistance prior spread",
                self.series_resistance_prior_log_standard_deviation,
            ),
            ("face-sensor capacitance nominal", self.face_sensor_capacitance_nominal),
            ("face-sensor response nominal", self.face_sensor_response_nominal),
            (
                "face-sensor truth spread",
                self.face_sensor_truth_log_standard_deviation,
            ),
            (
                "face-sensor prior spread",
                self.face_sensor_prior_log_standard_deviation,
            ),
            ("verification threshold", self.verification_score_threshold),
            ("interval multiplier", self.local_interval_multiplier),
            ("margin finite-difference step", self.margin_finite_difference_step),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name, bounds, nominal in (
            (
                "series-resistance",
                self.series_resistance_bounds,
                self.series_resistance_nominal,
            ),
            (
                "face-sensor capacitance",
                self.face_sensor_capacitance_bounds,
                self.face_sensor_capacitance_nominal,
            ),
            (
                "face-sensor response",
                self.face_sensor_response_bounds,
                self.face_sensor_response_nominal,
            ),
        ):
            if (
                len(bounds) != 2
                or not all(math.isfinite(item) for item in bounds)
                or bounds[0] <= 0.0
                or bounds[1] <= bounds[0]
                or not bounds[0] < nominal < bounds[1]
            ):
                raise ValueError(f"{name} bounds must contain its nominal value")
        beta = self.contact_beta_bounds
        if (
            len(beta) != 2
            or not all(math.isfinite(item) for item in beta)
            or beta[0] <= 0.0
            or beta[1] <= beta[0]
        ):
            raise ValueError("contact-beta bounds must be positive and ordered")


def _truncated_lognormal(
    random_source: random.Random,
    nominal: float,
    spread: float,
    bounds: Tuple[float, float],
) -> float:
    for _ in range(10_000):
        value = nominal * math.exp(random_source.gauss(0.0, spread))
        if bounds[0] <= value <= bounds[1]:
            return value
    raise RuntimeError("truncated lognormal draw did not enter its calibration range")


def realism_truth_for_trial(
    config: OperatingDecisionRealismConfig,
    trial_index: int,
) -> RealismTruth:
    """Return one deterministic hidden device and probe realization."""

    if (
        not isinstance(trial_index, int)
        or isinstance(trial_index, bool)
        or trial_index < 0
    ):
        raise ValueError("trial index must be a nonnegative integer")
    base = _truth_for_trial(config.sensor, trial_index)
    random_source = random.Random(
        config.sensor.first_seed + 7_000_000 + 10_000 * trial_index
    )
    series_resistance = _truncated_lognormal(
        random_source,
        config.series_resistance_nominal,
        config.series_resistance_truth_log_standard_deviation,
        config.series_resistance_bounds,
    )
    sensor_capacitance = _truncated_lognormal(
        random_source,
        config.face_sensor_capacitance_nominal,
        config.face_sensor_truth_log_standard_deviation,
        config.face_sensor_capacitance_bounds,
    )
    sensor_response = _truncated_lognormal(
        random_source,
        config.face_sensor_response_nominal,
        config.face_sensor_truth_log_standard_deviation,
        config.face_sensor_response_bounds,
    )
    beta = random_source.uniform(*config.contact_beta_bounds)
    return RealismTruth(
        physical_values=base.physical_values,
        interface_mass=base.interface_mass,
        series_resistance=series_resistance,
        face_sensor=TemporaryFaceSensor(sensor_capacitance, sensor_response),
        contact_beta=beta,
    )


def _simulate_realistic_observables(
    model_name: str,
    current: PiecewiseConstantCurrent,
    channels: Sequence[str],
    physical_values: Tuple[float, float, float],
    interface_mass: Optional[float],
    series_resistance: float,
    face_sensor: Optional[TemporaryFaceSensor],
    config: OperatingDecisionRealismConfig,
    *,
    contact_beta: Optional[float] = None,
) -> RealisticObservablePrediction:
    """Simulate a candidate or truth family with explicit instrumentation."""

    channels = tuple(channels)
    if len(set(channels)) != len(channels) or any(
        channel not in ALL_CHANNELS for channel in channels
    ):
        raise ValueError("realistic simulation channels must be known and distinct")
    if not math.isfinite(series_resistance) or series_resistance < 0.0:
        raise ValueError("series resistance must be finite and nonnegative")
    if len(physical_values) != 3 or any(
        not math.isfinite(value) or value <= 0.0 for value in physical_values
    ):
        raise ValueError("realistic simulation needs three positive physical values")
    if model_name not in (*MODEL_NAMES, TEMPERATURE_DEPENDENT_CONTACT):
        raise ValueError("unknown Stage 3 model or truth family")
    if model_name != FOUR_STATE_MODEL and (
        interface_mass is None
        or not math.isfinite(interface_mass)
        or interface_mass <= 0.0
    ):
        raise ValueError("five-state simulation needs a positive interface mass")
    if model_name == TEMPERATURE_DEPENDENT_CONTACT and (
        contact_beta is None
        or not math.isfinite(contact_beta)
        or contact_beta < 0.0
    ):
        raise ValueError("Family C simulation needs a nonnegative contact beta")

    resistance, capacitance, sensor_lag = physical_values
    reference = constant_current_contact_reference_experiment()
    thermoelectric = replace(
        reference.thermoelectric_parameters,
        electrical_resistance=(
            reference.thermoelectric_parameters.electrical_resistance
            + series_resistance
        ),
    )
    thermal = replace(
        reference.thermal_parameters,
        cold_contact_resistance=resistance,
        cold_face_thermal_capacitance=capacitance,
    )
    mismatch = (
        None
        if model_name == FOUR_STATE_MODEL
        else InterfaceMassMismatch(thermal_capacitance=float(interface_mass))
    )
    integration = dict(
        duration=RUN_DURATION_SECONDS,
        time_step=config.sensor.dense_time_step,
        current=current,
        cold_reservoir_temperature=300.0,
        hot_reservoir_temperature=300.0,
    )
    if model_name == FOUR_STATE_MODEL:
        if face_sensor is None:
            experiment = replace(
                reference,
                thermoelectric_parameters=thermoelectric,
                thermal_parameters=thermal,
                duration=RUN_DURATION_SECONDS,
                time_step=config.sensor.dense_time_step,
                current=current,
            )
            trajectory = run_four_node_contact_experiment(experiment).trajectory
        else:
            trajectory = integrate_four_node_temporary_face_sensor(
                thermoelectric,
                thermal,
                face_sensor,
                initial_cold_face_temperature=300.0,
                initial_hot_face_temperature=300.0,
                initial_cold_exchanger_temperature=300.0,
                initial_hot_exchanger_temperature=300.0,
                initial_face_sensor_temperature=300.0,
                **integration,
            )
    elif model_name == FIVE_STATE_MODEL:
        if face_sensor is None:
            trajectory = integrate_interface_mass_truth(
                thermoelectric,
                thermal,
                mismatch,  # type: ignore[arg-type]
                initial_temperature=300.0,
                **integration,
            )
        else:
            trajectory = integrate_interface_mass_temporary_face_sensor(
                thermoelectric,
                thermal,
                mismatch,  # type: ignore[arg-type]
                face_sensor,
                initial_cold_face_temperature=300.0,
                initial_hot_face_temperature=300.0,
                initial_cold_interface_temperature=300.0,
                initial_cold_exchanger_temperature=300.0,
                initial_hot_exchanger_temperature=300.0,
                initial_face_sensor_temperature=300.0,
                **integration,
            )
    elif face_sensor is None:
        trajectory = integrate_temperature_dependent_contact_truth(
            thermoelectric,
            thermal,
            mismatch,  # type: ignore[arg-type]
            TemperatureDependentColdContact(beta=float(contact_beta)),
            initial_temperature=300.0,
            **integration,
        )
    else:
        trajectory = integrate_temperature_dependent_contact_temporary_sensor(
            thermoelectric,
            thermal,
            mismatch,  # type: ignore[arg-type]
            TemperatureDependentColdContact(beta=float(contact_beta)),
            face_sensor,
            initial_temperature=300.0,
            **integration,
        )

    sample_times = regular_measurement_times(
        RUN_DURATION_SECONDS,
        config.sensor.sampling_interval,
    )
    indices = _trajectory_at_times(trajectory, sample_times)
    lagged_channels = tuple(
        channel
        for channel in channels
        if channel in (COLD_EXCHANGER, HOT_EXCHANGER)
        or (channel == COLD_FACE and face_sensor is None)
    )
    temperatures = _lagged_temperature_values(
        trajectory,
        current=current,
        temperature_channels=lagged_channels,
        sensor_lag=sensor_lag,
        config=config.sensor,
    )
    values = []
    for time, index in zip(sample_times, indices):
        for channel in channels:
            if channel in lagged_channels:
                value = temperatures[(channel, round(time, 9))]
            elif channel == COLD_FACE and face_sensor is not None:
                value = trajectory.face_sensor[index]
            elif channel == COLD_HEAT_RATE:
                if model_name == FOUR_STATE_MODEL:
                    value = (
                        trajectory.cold_exchanger[index]
                        - trajectory.cold_face[index]
                    ) / resistance
                else:
                    actual_resistance = resistance
                    if model_name == TEMPERATURE_DEPENDENT_CONTACT:
                        actual_resistance = TemperatureDependentColdContact(
                            beta=float(contact_beta)
                        ).resistance(
                            resistance,
                            trajectory.cold_interface[index],
                        )
                    value = (
                        trajectory.cold_exchanger[index]
                        - trajectory.cold_interface[index]
                    ) / (0.5 * actual_resistance)
            elif channel == VOLTAGE:
                value = voltage(
                    thermoelectric,
                    current_at(current, time),
                    trajectory.hot_face[index],
                    trajectory.cold_face[index],
                )
            else:
                raise ValueError(f"unsupported Stage 3 channel: {channel}")
            values.append(ObservableValue(channel, time, value))
    return RealisticObservablePrediction(
        values=tuple(values),
        time=tuple(trajectory.time),
        cold_face=tuple(trajectory.cold_face),
        hot_face=tuple(trajectory.hot_face),
    )


_COLD_CONTACT = "cold_contact_resistance"
_COLD_FACE_CAPACITANCE = "cold_face_capacitance"
_SHARED_SENSOR_LAG = "shared_sensor_lag"
_INTERFACE_MASS = "interface_mass"
_SERIES_RESISTANCE = "series_resistance"
_PROBE_CAPACITANCE = "probe_capacitance"
_PROBE_RESPONSE = "probe_response"


def _parameter_spec(
    model_name: str,
    instrumented: bool,
    config: OperatingDecisionRealismConfig,
) -> _ParameterSpec:
    if model_name not in MODEL_NAMES:
        raise ValueError("Stage 3 inference supports only the two candidates")
    names = [_COLD_CONTACT, _COLD_FACE_CAPACITANCE, _SHARED_SENSOR_LAG]
    nominals = list(config.sensor.fit.nominal_values)
    bounds = list(config.sensor.fit.log_bounds)
    priors = list(config.sensor.selection.prior_standard_deviations[:3])
    if model_name == FIVE_STATE_MODEL:
        names.append(_INTERFACE_MASS)
        nominals.append(config.sensor.interface_mass_nominal)
        lower, upper = config.sensor.interface_mass_bounds
        bounds.append(
            (
                math.log(lower / config.sensor.interface_mass_nominal),
                math.log(upper / config.sensor.interface_mass_nominal),
            )
        )
        priors.append(config.sensor.interface_mass_prior_log_standard_deviation)
    names.append(_SERIES_RESISTANCE)
    nominals.append(config.series_resistance_nominal)
    lower, upper = getattr(
        config,
        "series_resistance_fit_bounds",
        config.series_resistance_bounds,
    )
    bounds.append(
        (
            math.log(lower / config.series_resistance_nominal),
            math.log(upper / config.series_resistance_nominal),
        )
    )
    priors.append(config.series_resistance_prior_log_standard_deviation)
    if instrumented:
        names.extend((_PROBE_CAPACITANCE, _PROBE_RESPONSE))
        nominals.extend(
            (
                config.face_sensor_capacitance_nominal,
                config.face_sensor_response_nominal,
            )
        )
        for nominal, value_bounds in (
            (
                config.face_sensor_capacitance_nominal,
                getattr(
                    config,
                    "face_sensor_capacitance_fit_bounds",
                    config.face_sensor_capacitance_bounds,
                ),
            ),
            (
                config.face_sensor_response_nominal,
                getattr(
                    config,
                    "face_sensor_response_fit_bounds",
                    config.face_sensor_response_bounds,
                ),
            ),
        ):
            bounds.append(
                (
                    math.log(value_bounds[0] / nominal),
                    math.log(value_bounds[1] / nominal),
                )
            )
        priors.extend((config.face_sensor_prior_log_standard_deviation,) * 2)
    return _ParameterSpec(
        names=tuple(names),
        nominal_values=tuple(nominals),
        log_bounds=tuple(bounds),
        prior_log_standard_deviations=tuple(priors),
    )


def _decoded_parameters(
    model_name: str,
    offsets: Sequence[float],
    spec: _ParameterSpec,
) -> Tuple[
    Tuple[float, float, float],
    Optional[float],
    float,
    Optional[TemporaryFaceSensor],
]:
    if len(offsets) != len(spec.names):
        raise ValueError("fit offsets do not match the Stage 3 parameter spec")
    values = {
        name: nominal * math.exp(offset)
        for name, nominal, offset in zip(spec.names, spec.nominal_values, offsets)
    }
    physical = (
        values[_COLD_CONTACT],
        values[_COLD_FACE_CAPACITANCE],
        values[_SHARED_SENSOR_LAG],
    )
    interface_mass = (
        values[_INTERFACE_MASS] if model_name == FIVE_STATE_MODEL else None
    )
    probe = (
        TemporaryFaceSensor(
            values[_PROBE_CAPACITANCE],
            values[_PROBE_RESPONSE],
        )
        if _PROBE_CAPACITANCE in values
        else None
    )
    return physical, interface_mass, values[_SERIES_RESISTANCE], probe


def _regularized_bias_residuals(
    observed: ObservableRun,
    predicted: RealisticObservablePrediction,
    config: OperatingDecisionRealismConfig,
) -> Tuple[float, ...]:
    """Profile run offsets under their Gaussian calibration distribution."""

    prediction = {
        (item.channel, item.time): item.value for item in predicted.values
    }
    noise_by_channel = dict(config.sensor.channel_noise)
    residuals = []
    for channel in ALL_CHANNELS:
        selected = tuple(item for item in observed.values if item.channel == channel)
        if not selected:
            continue
        differences = tuple(
            item.value - prediction[(item.channel, item.time)] for item in selected
        )
        noise = noise_by_channel[channel]
        bias_scale = config.sensor.run_bias_noise_ratio * noise
        bias = (
            bias_scale
            * bias_scale
            * sum(differences)
            / (noise * noise + len(differences) * bias_scale * bias_scale)
        )
        residuals.extend(
            (
                prediction[(item.channel, item.time)] + bias - item.value
            )
            / noise
            for item in selected
        )
        residuals.append(bias / bias_scale)
    if not residuals or any(not math.isfinite(value) for value in residuals):
        raise ValueError("regularized acquisition residuals are invalid")
    return tuple(residuals)


def _fit_start_vectors(spec: _ParameterSpec) -> Tuple[Tuple[float, ...], ...]:
    shifts = {
        _COLD_CONTACT: -0.16,
        _COLD_FACE_CAPACITANCE: 0.16,
        _SHARED_SENSOR_LAG: 0.12,
        _INTERFACE_MASS: -0.22,
        _SERIES_RESISTANCE: -0.20,
        _PROBE_CAPACITANCE: 0.15,
        _PROBE_RESPONSE: -0.18,
    }
    offset = tuple(shifts[name] for name in spec.names)
    return (
        (0.0,) * len(spec.names),
        offset,
        tuple(-value for value in offset),
    )


def fit_realistic_candidate(
    model_name: str,
    runs: Sequence[RealisticOperatingRun],
    config: OperatingDecisionRealismConfig,
    *,
    initial_log_multipliers: Optional[Sequence[float]] = None,
) -> RealisticCandidateFit:
    """Fit one candidate with physical priors and calibrated run offsets."""

    runs = tuple(runs)
    if not runs or any(run.regime.phase != ACQUISITION for run in runs):
        raise ValueError("realistic fit needs acquisition runs only")
    instrumented = any(
        run.instrumentation.temporary_face_sensor for run in runs
    )
    spec = _parameter_spec(model_name, instrumented, config)
    if initial_log_multipliers is None:
        initial_log_multipliers = (0.0,) * len(spec.names)
    if len(initial_log_multipliers) != len(spec.names) or any(
        not math.isfinite(value) for value in initial_log_multipliers
    ):
        raise ValueError("initial offsets must match the Stage 3 parameter spec")

    cache: Dict[Tuple[float, ...], Tuple[float, ...]] = {}
    evaluation_count = 0

    def bounded(values: Sequence[float]) -> Tuple[float, ...]:
        return tuple(
            min(upper, max(lower, float(value)))
            for value, (lower, upper) in zip(values, spec.log_bounds)
        )

    def residuals(values: Sequence[float]) -> Tuple[float, ...]:
        nonlocal evaluation_count
        key = bounded(values)
        if key not in cache:
            physical, interface_mass, series_resistance, face_sensor = (
                _decoded_parameters(model_name, key, spec)
            )
            combined = []
            for run in runs:
                prediction = _simulate_realistic_observables(
                    model_name,
                    run.regime.current,
                    run.regime.channels,
                    physical,
                    interface_mass,
                    series_resistance,
                    (
                        face_sensor
                        if run.instrumentation.temporary_face_sensor
                        else None
                    ),
                    config,
                )
                combined.extend(
                    _regularized_bias_residuals(run.observations, prediction, config)
                )
            combined.extend(
                value / scale
                for value, scale in zip(
                    key,
                    spec.prior_log_standard_deviations,
                )
            )
            if any(not math.isfinite(value) for value in combined):
                raise ValueError("Stage 3 fit produced nonfinite residuals")
            cache[key] = tuple(combined)
            evaluation_count += 1
        return cache[key]

    def objective(values: Sequence[float]) -> float:
        errors = residuals(values)
        return sum(value * value for value in errors) / len(errors)

    def jacobian_columns(values: Sequence[float]) -> Tuple[Tuple[float, ...], ...]:
        columns = []
        step = config.sensor.fit.finite_difference_step
        for index, (lower_bound, upper_bound) in enumerate(spec.log_bounds):
            left = list(values)
            right = list(values)
            left[index] = max(lower_bound, left[index] - step)
            right[index] = min(upper_bound, right[index] + step)
            denominator = right[index] - left[index]
            if denominator <= 0.0:
                raise ValueError("Stage 3 finite-difference interval collapsed")
            left_residuals = residuals(left)
            right_residuals = residuals(right)
            columns.append(
                tuple(
                    (right_value - left_value) / denominator
                    for left_value, right_value in zip(
                        left_residuals,
                        right_residuals,
                    )
                )
            )
        return tuple(columns)

    values = list(bounded(initial_log_multipliers))
    damping = config.sensor.fit.initial_damping
    objective(values)
    accepted_iterations = 0
    last_step_infinity_norm: Optional[float] = None
    last_relative_objective_reduction: Optional[float] = None
    for _ in range(config.sensor.fit_iterations):
        errors = residuals(values)
        columns = jacobian_columns(values)
        normal = tuple(
            tuple(
                sum(a * b for a, b in zip(left, right))
                for right in columns
            )
            for left in columns
        )
        gradient = tuple(
            sum(derivative * error for derivative, error in zip(column, errors))
            for column in columns
        )
        damped = tuple(
            tuple(
                item
                + (
                    damping * max(1.0, normal[row][row])
                    if row == column
                    else 0.0
                )
                for column, item in enumerate(matrix_row)
            )
            for row, matrix_row in enumerate(normal)
        )
        try:
            inverse, _ = inverse_and_determinant(damped)
        except ValueError:
            damping *= 10.0
            continue
        update = tuple(
            -sum(item * component for item, component in zip(row, gradient))
            for row in inverse
        )
        starting_loss = objective(values)
        accepted = False
        for fraction in (1.0, 0.5, 0.25, 0.1, 0.03):
            candidate = bounded(
                tuple(
                    value + fraction * delta
                    for value, delta in zip(values, update)
                )
            )
            candidate_loss = objective(candidate)
            if candidate_loss < starting_loss:
                previous = tuple(values)
                values = list(candidate)
                accepted_iterations += 1
                last_step_infinity_norm = max(
                    abs(after - before)
                    for before, after in zip(previous, candidate)
                )
                last_relative_objective_reduction = (
                    starting_loss - candidate_loss
                ) / max(1.0, abs(starting_loss))
                damping = max(
                    config.sensor.fit.initial_damping * 1.0e-3,
                    damping * 0.3,
                )
                accepted = True
                break
        if not accepted:
            damping *= 10.0

    best = bounded(values)
    columns = jacobian_columns(best)
    information = tuple(
        tuple(
            sum(a * b for a, b in zip(left, right))
            for right in columns
        )
        for left in columns
    )
    final_errors = residuals(best)
    final_gradient = tuple(
        sum(derivative * error for derivative, error in zip(column, final_errors))
        for column in columns
    )
    residual_norm = math.sqrt(sum(error * error for error in final_errors))
    scaled_gradient_infinity_norm = max(
        abs(component)
        / (
            max(1.0e-15, math.sqrt(information[index][index]))
            * max(1.0, residual_norm)
        )
        for index, component in enumerate(final_gradient)
    )
    gradient_converged = (
        scaled_gradient_infinity_norm <= FIT_SCALED_GRADIENT_TOLERANCE
    )
    step_converged = (
        last_step_infinity_norm is not None
        and last_relative_objective_reduction is not None
        and last_step_infinity_norm <= FIT_STEP_TOLERANCE
        and last_relative_objective_reduction <= FIT_RELATIVE_OBJECTIVE_TOLERANCE
    )
    converged = gradient_converged or step_converged
    termination_reason = (
        "scaled_gradient_tolerance"
        if gradient_converged
        else (
            "step_and_objective_tolerance"
            if step_converged
            else "fixed_iteration_limit"
        )
    )
    covariance, _ = inverse_and_determinant(information)
    physical, interface_mass, series_resistance, face_sensor = _decoded_parameters(
        model_name,
        best,
        spec,
    )
    return RealisticCandidateFit(
        model_name=model_name,
        log_multipliers=best,
        parameter_names=spec.names,
        physical_values=physical,
        interface_mass=interface_mass,
        series_resistance=series_resistance,
        face_sensor=face_sensor,
        objective=objective(best),
        covariance=covariance,
        reached_bound=any(
            abs(value - lower) <= 1.0e-5 or abs(upper - value) <= 1.0e-5
            for value, (lower, upper) in zip(best, spec.log_bounds)
        ),
        evaluation_count=evaluation_count,
        converged=converged,
        termination_reason=termination_reason,
        completed_iterations=config.sensor.fit_iterations,
        accepted_iterations=accepted_iterations,
        scaled_gradient_infinity_norm=scaled_gradient_infinity_norm,
        last_step_infinity_norm=last_step_infinity_norm,
        last_relative_objective_reduction=last_relative_objective_reduction,
    )


def _truth_model_name(truth_condition: str) -> str:
    if truth_condition == TRUTH_CONDITIONS[0]:
        return FOUR_STATE_MODEL
    if truth_condition == TRUTH_CONDITIONS[1]:
        return FIVE_STATE_MODEL
    if truth_condition == TEMPERATURE_DEPENDENT_CONTACT:
        return TEMPERATURE_DEPENDENT_CONTACT
    raise ValueError("Stage 3 truth condition is unknown")


def _run_seed(
    config: OperatingDecisionRealismConfig,
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
        raise ValueError("Stage 3 run has no frozen seed offset")
    return (
        config.sensor.first_seed
        + STAGE3_TRUTH_CONDITIONS.index(truth_condition) * 1_000_000
        + trial_index * 10_000
        + offsets[run_name]
    )


def _case_id(
    config: OperatingDecisionRealismConfig,
    truth_condition: str,
    trial_index: int,
    policy_name: str,
) -> Stage3CaseId:
    if truth_condition not in STAGE3_TRUTH_CONDITIONS:
        raise ValueError("Stage 3 truth condition is unknown")
    material = (
        f"thermotwin-stage3|{config.sensor.first_seed}|"
        f"{truth_condition}|{trial_index}"
    ).encode("utf-8")
    return Stage3CaseId(
        device_token=hashlib.sha256(material).hexdigest()[:20],
        policy_name=policy_name,
        trial_index=trial_index,
        final_schedule_id=STAGE3_FINAL_SCHEDULE_ID,
    )


def _instrumentation_for_regime(regime: OperatingRegime) -> RunInstrumentation:
    return RunInstrumentation(temporary_face_sensor=COLD_FACE in regime.channels)


def _truth_prediction(
    truth_condition: str,
    truth: RealismTruth,
    regime: OperatingRegime,
    instrumentation: RunInstrumentation,
    config: OperatingDecisionRealismConfig,
) -> RealisticObservablePrediction:
    model_name = _truth_model_name(truth_condition)
    return _simulate_realistic_observables(
        model_name,
        regime.current,
        regime.channels,
        truth.physical_values,
        truth.interface_mass if model_name != FOUR_STATE_MODEL else None,
        truth.series_resistance,
        (
            truth.face_sensor
            if instrumentation.temporary_face_sensor
            else None
        ),
        config,
        contact_beta=(
            truth.contact_beta
            if model_name == TEMPERATURE_DEPENDENT_CONTACT
            else None
        ),
    )


def _observed_realistic_run(
    regime: OperatingRegime,
    *,
    truth_condition: str,
    truth: RealismTruth,
    trial_index: int,
    config: OperatingDecisionRealismConfig,
) -> RealisticOperatingRun:
    instrumentation = _instrumentation_for_regime(regime)
    prediction = _truth_prediction(
        truth_condition,
        truth,
        regime,
        instrumentation,
        config,
    )
    observations = _observed_run(
        prediction,  # type: ignore[arg-type]
        name=regime.name,
        current=regime.current,
        seed=_run_seed(config, truth_condition, trial_index, regime.name),
        config=config.sensor,
    )
    return RealisticOperatingRun(regime, observations, instrumentation)


def build_realistic_blinded_case(
    truth_condition: str,
    trial_index: int,
    policy: FixedPolicy,
    config: OperatingDecisionRealismConfig = OperatingDecisionRealismConfig(),
) -> RealisticBlindedOperatingCase:
    """Generate diagnostics while keeping truth and final response private."""

    _truth_model_name(truth_condition)
    if not isinstance(policy, FixedPolicy):
        raise ValueError("Stage 3 operating decision needs a fixed policy")
    if (
        not isinstance(trial_index, int)
        or isinstance(trial_index, bool)
        or trial_index < 0
    ):
        raise ValueError("trial index must be a nonnegative integer")
    truth = realism_truth_for_trial(config, trial_index)
    acquisition_regimes = (
        initial_acquisition_regime(),
        *policy.additional_regimes,
    )
    acquisition_runs = tuple(
        _observed_realistic_run(
            regime,
            truth_condition=truth_condition,
            truth=truth,
            trial_index=trial_index,
            config=config,
        )
        for regime in acquisition_regimes
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
    verification_run = _observed_realistic_run(
        verification_regime,
        truth_condition=truth_condition,
        truth=truth,
        trial_index=trial_index,
        config=config,
    )
    final_regime = OperatingRegime(
        name="untouched_final_operating_schedule",
        phase=FINAL_EVALUATION,
        current=config.final_current,
        channels=(),
    )
    return RealisticBlindedOperatingCase(
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
        final_instrumentation=RunInstrumentation(False),
    )


def reveal_realistic_operating_outcome(
    truth_condition: str,
    trial_index: int,
    case: RealisticBlindedOperatingCase,
    config: OperatingDecisionRealismConfig = OperatingDecisionRealismConfig(),
) -> RevealedOperatingOutcome:
    """Generate the unloaded final truth after the decision boundary."""

    expected = _case_id(
        config,
        truth_condition,
        trial_index,
        case.case_id.policy_name,
    )
    if truth_condition not in STAGE3_TRUTH_CONDITIONS or case.case_id != expected:
        raise ValueError("case identity does not match the requested Stage 3 reveal")
    if (
        case.final_regime.phase != FINAL_EVALUATION
        or case.final_regime.channels
        or case.final_regime.current != config.final_current
        or case.final_instrumentation.temporary_face_sensor
    ):
        raise ValueError("only the untouched unloaded final regime can be revealed")
    truth = realism_truth_for_trial(config, trial_index)
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


def fit_realistic_acquisition_models(
    case: RealisticBlindedOperatingCase,
    config: OperatingDecisionRealismConfig = OperatingDecisionRealismConfig(),
) -> RealisticAcquisitionFitSet:
    """Fit both constant-contact candidates to acquisition data only."""

    fits = []
    failures = []
    instrumented = any(
        run.instrumentation.temporary_face_sensor for run in case.acquisition_runs
    )
    for model_name in MODEL_NAMES:
        spec = _parameter_spec(model_name, instrumented, config)
        candidates = []
        last_error: Optional[BaseException] = None
        for start in _fit_start_vectors(spec):
            try:
                fit = fit_realistic_candidate(
                    model_name,
                    case.acquisition_runs,
                    config,
                    initial_log_multipliers=start,
                )
                candidates.append(fit)
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
    return RealisticAcquisitionFitSet(tuple(fits), tuple(failures))


def realistic_verification_score(
    fit: RealisticCandidateFit,
    verification_run: RealisticOperatingRun,
    config: OperatingDecisionRealismConfig = OperatingDecisionRealismConfig(),
) -> Tuple[float, RealisticObservablePrediction]:
    """Score verification under marginalized, unrefitted run offsets."""

    if verification_run.regime.phase != VERIFICATION:
        raise ValueError("candidate adequacy must use the verification phase")
    if verification_run.instrumentation.temporary_face_sensor and fit.face_sensor is None:
        raise ValueError("instrumented verification needs fitted probe parameters")
    prediction = _simulate_realistic_observables(
        fit.model_name,
        verification_run.regime.current,
        verification_run.regime.channels,
        fit.physical_values,
        fit.interface_mass,
        fit.series_resistance,
        (
            fit.face_sensor
            if verification_run.instrumentation.temporary_face_sensor
            else None
        ),
        config,
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
        bias_scale = config.sensor.run_bias_noise_ratio * noise
        quadratic += sum(
            (value - mean_residual) ** 2 / (noise * noise)
            for value in residuals
        )
        quadratic += mean_residual * mean_residual / (
            bias_scale * bias_scale + noise * noise / len(residuals)
        )
        sample_count += len(residuals)
    if sample_count <= 0:
        raise ValueError("verification run contains no scored samples")
    return quadratic / sample_count, prediction


def verify_realistic_candidate_models(
    fit_set: RealisticAcquisitionFitSet,
    verification_run: RealisticOperatingRun,
    config: OperatingDecisionRealismConfig = OperatingDecisionRealismConfig(),
) -> Tuple[CandidateVerification, ...]:
    results = []
    for fit in fit_set.fits:
        try:
            score, _ = realistic_verification_score(
                fit,
                verification_run,
                config,
            )
        except (ArithmeticError, IntegrationDivergenceError, ValueError):
            results.append(
                CandidateVerification(
                    fit,  # type: ignore[arg-type]
                    math.inf,
                    False,
                    "verification_failure",
                )
            )
            continue
        if fit.reached_bound:
            reason = "fit_reached_bound"
        elif not math.isfinite(score):
            reason = "nonfinite_score"
        elif score > config.verification_score_threshold:
            reason = "inadequate_verification"
        else:
            reason = None
        results.append(
            CandidateVerification(
                fit,  # type: ignore[arg-type]
                score,
                reason is None,
                reason,
            )
        )
    return tuple(results)


def _margin_at_offsets(
    fit: RealisticCandidateFit,
    offsets: Sequence[float],
    final_regime: OperatingRegime,
    config: OperatingDecisionRealismConfig,
) -> float:
    spec = _parameter_spec(
        fit.model_name,
        fit.face_sensor is not None,
        config,
    )
    physical, interface_mass, series_resistance, _ = _decoded_parameters(
        fit.model_name,
        offsets,
        spec,
    )
    prediction = _simulate_realistic_observables(
        fit.model_name,
        final_regime.current,
        (),
        physical,
        interface_mass,
        series_resistance,
        None,
        config,
    )
    return operating_margin(prediction.cold_face, config.band)


def forecast_realistic_margin_interval(
    fit: RealisticCandidateFit,
    final_regime: OperatingRegime,
    config: OperatingDecisionRealismConfig = OperatingDecisionRealismConfig(),
) -> MarginInterval:
    """Propagate the full fit covariance to the unloaded final-device margin."""

    if final_regime.phase != FINAL_EVALUATION or final_regime.channels:
        raise ValueError("margin forecast requires the untouched final regime")
    if fit.reached_bound:
        raise ValueError("a bound-hit fit cannot support a local margin interval")
    spec = _parameter_spec(fit.model_name, fit.face_sensor is not None, config)
    if fit.parameter_names != spec.names or len(fit.log_multipliers) != len(spec.names):
        raise ValueError("fit parameters do not match the Stage 3 candidate spec")
    parameter_count = len(spec.names)
    if (
        len(fit.covariance) != parameter_count
        or any(len(row) != parameter_count for row in fit.covariance)
        or any(
            not math.isfinite(value)
            for row in fit.covariance
            for value in row
        )
    ):
        raise ValueError("fit covariance has an invalid shape or value")
    for row in range(parameter_count):
        for column in range(row):
            left_value = fit.covariance[row][column]
            right_value = fit.covariance[column][row]
            scale = max(1.0, abs(left_value), abs(right_value))
            if abs(left_value - right_value) > 1.0e-8 * scale:
                raise ValueError("fit covariance must be symmetric")
    decoded = _decoded_parameters(fit.model_name, fit.log_multipliers, spec)
    expected = (
        fit.physical_values,
        fit.interface_mass,
        fit.series_resistance,
        fit.face_sensor,
    )
    if decoded != expected:
        raise ValueError("fit values do not reconstruct from their log offsets")

    gradient = []
    step = config.margin_finite_difference_step
    for index, (lower_bound, upper_bound) in enumerate(spec.log_bounds):
        left = list(fit.log_multipliers)
        right = list(fit.log_multipliers)
        left[index] = max(lower_bound, left[index] - step)
        right[index] = min(upper_bound, right[index] + step)
        denominator = right[index] - left[index]
        if denominator <= 0.0:
            raise ValueError("margin finite-difference interval collapsed")
        gradient.append(
            (
                _margin_at_offsets(fit, right, final_regime, config)
                - _margin_at_offsets(fit, left, final_regime, config)
            )
            / denominator
        )
    contributions = tuple(
        gradient[row] * fit.covariance[row][column] * gradient[column]
        for row in range(parameter_count)
        for column in range(parameter_count)
    )
    variance = sum(contributions)
    tolerance = 1.0e-10 * max(
        1.0,
        sum(abs(value) for value in contributions),
    )
    if not math.isfinite(variance) or variance < -tolerance:
        raise ValueError("local margin variance is invalid")
    standard_error = math.sqrt(max(0.0, variance))
    estimate = _margin_at_offsets(
        fit,
        fit.log_multipliers,
        final_regime,
        config,
    )
    radius = config.local_interval_multiplier * standard_error
    interval = MarginInterval(
        fit.model_name,
        estimate,
        standard_error,
        estimate - radius,
        estimate + radius,
    )
    if any(not math.isfinite(value) for value in interval[1:]):
        raise ValueError("local margin interval is nonfinite")
    return interval


def decide_realistic_blinded_case(
    case: RealisticBlindedOperatingCase,
    *,
    config: OperatingDecisionRealismConfig = OperatingDecisionRealismConfig(),
) -> SavedOperatingDecision:
    """Save the Stage 3 decision before accepting any final response."""

    started = perf_counter()
    fit_set = fit_realistic_acquisition_models(case, config)
    verifications = verify_realistic_candidate_models(
        fit_set,
        case.verification_run,
        config,
    )
    failures = list(fit_set.failures)
    intervals = []
    for verification in verifications:
        if verification.failure_reason in (
            "verification_failure",
            "nonfinite_score",
        ):
            failures.append(
                NumericalFailure(
                    verification.fit.model_name,
                    "verification",
                    verification.failure_reason,
                )
            )
        if not verification.passed:
            continue
        try:
            intervals.append(
                forecast_realistic_margin_interval(
                    verification.fit,  # type: ignore[arg-type]
                    case.final_regime,
                    config,
                )
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
        reason = {
            APPROVE: "verified_envelope_inside_band",
            REJECT: "verified_envelope_below_zero",
            INSUFFICIENT_EVIDENCE: "verified_envelope_crosses_zero",
        }[decision]
    return SavedOperatingDecision(
        case_id=case.case_id,  # type: ignore[arg-type]
        decision=decision,
        decision_reason=reason,
        margin_envelope=envelope,
        model_intervals=tuple(intervals),
        verifications=verifications,
        failures=tuple(failures),
        decision_computation_seconds=perf_counter() - started,
    )


def nominal_realistic_schedule_energy(
    current: PiecewiseConstantCurrent,
    config: OperatingDecisionRealismConfig = OperatingDecisionRealismConfig(),
    *,
    temporary_face_sensor: bool = False,
) -> float:
    """Return nominal terminal energy with intrinsic contact loss and probe load."""

    prediction = _simulate_realistic_observables(
        FOUR_STATE_MODEL,
        current,
        (),
        config.sensor.fit.nominal_values,
        None,
        config.series_resistance_nominal,
        (
            TemporaryFaceSensor(
                config.face_sensor_capacitance_nominal,
                config.face_sensor_response_nominal,
            )
            if temporary_face_sensor
            else None
        ),
        config,
    )
    reference = constant_current_contact_reference_experiment()
    effective = replace(
        reference.thermoelectric_parameters,
        electrical_resistance=(
            reference.thermoelectric_parameters.electrical_resistance
            + config.series_resistance_nominal
        ),
    )
    return piecewise_electrical_energy(
        prediction.time,
        prediction.cold_face,
        prediction.hot_face,
        effective,
        current,
        start_time=prediction.time[0],
        end_time=prediction.time[-1],
    )


def _score_realistic_decision(
    saved: SavedOperatingDecision,
    revealed: RevealedOperatingOutcome,
    case: RealisticBlindedOperatingCase,
    energy_by_run: Dict[Tuple[PiecewiseConstantCurrent, bool], float],
    config: OperatingDecisionRealismConfig,
) -> ScoredOperatingDecision:
    if saved.case_id != case.case_id or revealed.case_id != case.case_id:
        raise ValueError("case, saved decision, and reveal must share identity")
    if saved.decision not in DECISIONS:
        raise ValueError("saved Stage 3 decision is unknown")
    _truth_model_name(revealed.truth_condition)
    recomputed_margin = operating_margin(
        revealed.cold_face_temperature,
        config.band,
    )
    if not math.isclose(
        recomputed_margin,
        revealed.true_margin,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise ValueError("revealed margin does not match the revealed trajectory")
    acquisition_energy = sum(
        energy_by_run[
            (
                run.regime.current,
                run.instrumentation.temporary_face_sensor,
            )
        ]
        for run in case.acquisition_runs
    )
    verification_energy = energy_by_run[
        (
            case.verification_run.regime.current,
            case.verification_run.instrumentation.temporary_face_sensor,
        )
    ]
    envelope = saved.margin_envelope
    true_pass = recomputed_margin >= 0.0
    return ScoredOperatingDecision(
        truth_condition=revealed.truth_condition,
        saved=saved,
        true_margin=recomputed_margin,
        true_pass=true_pass,
        false_approval=saved.decision == APPROVE and not true_pass,
        false_rejection=saved.decision == REJECT and true_pass,
        interval_covered=(
            None
            if envelope is None
            else envelope.lower <= recomputed_margin <= envelope.upper
        ),
        acquisition_run_count=len(case.acquisition_runs),
        diagnostic_run_count=len(case.acquisition_runs) + 1,
        energized_schedule_time_seconds=(
            RUN_DURATION_SECONDS * (len(case.acquisition_runs) + 1)
        ),
        acquisition_energy=acquisition_energy,
        verification_energy=verification_energy,
        total_diagnostic_energy=acquisition_energy + verification_energy,
        extra_sensor_count=case.policy.extra_sensor_count,
    )


def score_realistic_saved_decision(
    saved: SavedOperatingDecision,
    revealed: RevealedOperatingOutcome,
    case: RealisticBlindedOperatingCase,
    config: OperatingDecisionRealismConfig = OperatingDecisionRealismConfig(),
) -> ScoredOperatingDecision:
    keys = {
        *(
            (run.regime.current, run.instrumentation.temporary_face_sensor)
            for run in case.acquisition_runs
        ),
        (
            case.verification_run.regime.current,
            case.verification_run.instrumentation.temporary_face_sensor,
        ),
    }
    energies = {
        key: nominal_realistic_schedule_energy(
            key[0],
            config,
            temporary_face_sensor=key[1],
        )
        for key in keys
    }
    return _score_realistic_decision(saved, revealed, case, energies, config)


def summarize_operating_decision_realism(
    trials: Sequence[ScoredOperatingDecision],
    policies: Sequence[FixedPolicy],
    config: OperatingDecisionRealismConfig = OperatingDecisionRealismConfig(),
) -> Tuple[OperatingDecisionSummary, ...]:
    """Summarize every fixed policy while retaining abstentions and failures."""

    trials = tuple(trials)
    policies = tuple(policies)
    if not policies or len({policy.name for policy in policies}) != len(policies):
        raise ValueError("Stage 3 summaries need unique fixed policies")
    expected = {
        (truth_condition, policy.name, trial_index)
        for truth_condition in STAGE3_TRUTH_CONDITIONS
        for policy in policies
        for trial_index in range(config.sensor.trial_count)
    }
    actual = {
        (item.truth_condition, item.saved.policy_name, item.saved.trial_index)
        for item in trials
    }
    if len(actual) != len(trials) or actual != expected:
        raise ValueError("Stage 3 trials are missing, duplicated, or unknown")
    for truth_condition in STAGE3_TRUTH_CONDITIONS:
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
                raise ValueError("paired policies must share one unloaded final truth")

    summaries = []
    for truth_condition in STAGE3_TRUTH_CONDITIONS:
        for policy in policies:
            selected = tuple(
                item
                for item in trials
                if item.truth_condition == truth_condition
                and item.saved.policy_name == policy.name
            )
            if len(selected) != config.sensor.trial_count:
                raise ValueError("each Stage 3 summary needs every paired trial")
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
                raise ValueError("Stage 3 policy resources must be trial invariant")
            approvals = sum(item.saved.decision == APPROVE for item in selected)
            rejections = sum(item.saved.decision == REJECT for item in selected)
            insufficient = len(selected) - approvals - rejections
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
                        false_approvals,
                        true_violating,
                    ),
                    false_rejections=rate_estimate(
                        false_rejections,
                        rejections,
                    ),
                    decision_coverage=rate_estimate(
                        approvals + rejections,
                        len(selected),
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


def nominal_realistic_margins(
    config: OperatingDecisionRealismConfig = OperatingDecisionRealismConfig(),
) -> Tuple[Tuple[str, float], ...]:
    """Return unloaded nominal margins for the two candidates and Family C."""

    values = []
    for model_name in (*MODEL_NAMES, TEMPERATURE_DEPENDENT_CONTACT):
        prediction = _simulate_realistic_observables(
            model_name,
            config.final_current,
            (),
            config.sensor.fit.nominal_values,
            (
                None
                if model_name == FOUR_STATE_MODEL
                else config.sensor.interface_mass_nominal
            ),
            config.series_resistance_nominal,
            None,
            config,
            contact_beta=(
                fmean(config.contact_beta_bounds)
                if model_name == TEMPERATURE_DEPENDENT_CONTACT
                else None
            ),
        )
        values.append((model_name, operating_margin(prediction.cold_face, config.band)))
    return tuple(values)


def run_operating_decision_realism(
    config: OperatingDecisionRealismConfig = OperatingDecisionRealismConfig(),
    *,
    progress: Optional[Callable[[str], None]] = None,
) -> OperatingDecisionRealismResult:
    """Run the paired Stage 3 fixed-policy realism stress test."""

    policies = default_fixed_policies()
    trials = []
    for truth_condition in STAGE3_TRUTH_CONDITIONS:
        trials.extend(
            run_operating_decision_realism_truth(
                truth_condition,
                config,
                progress=progress,
            )
        )
    return OperatingDecisionRealismResult(
        config=config,
        policies=policies,
        trials=tuple(trials),
        summaries=summarize_operating_decision_realism(
            trials,
            policies,
            config,
        ),
    )


def _nominal_energy_by_run(
    policies: Sequence[FixedPolicy],
    config: OperatingDecisionRealismConfig,
) -> Dict[Tuple[PiecewiseConstantCurrent, bool], float]:
    run_keys = {(config.verification_current, False)}
    for policy in policies:
        regimes = (initial_acquisition_regime(), *policy.additional_regimes)
        run_keys.update(
            (
                regime.current,
                _instrumentation_for_regime(regime).temporary_face_sensor,
            )
            for regime in regimes
        )
        run_keys.add(
            (
                config.verification_current,
                policy.name == FIXED_FACE_TEMPERATURE,
            )
        )
    return {
        key: nominal_realistic_schedule_energy(
            key[0],
            config,
            temporary_face_sensor=key[1],
        )
        for key in run_keys
    }


def run_operating_decision_realism_truth(
    truth_condition: str,
    config: OperatingDecisionRealismConfig = OperatingDecisionRealismConfig(),
    *,
    progress: Optional[Callable[[str], None]] = None,
) -> Tuple[ScoredOperatingDecision, ...]:
    """Run one truth-family slice, useful for parallel development evaluation."""

    _truth_model_name(truth_condition)
    policies = default_fixed_policies()
    energies = _nominal_energy_by_run(policies, config)
    trials = []
    for trial_index in range(config.sensor.trial_count):
        if progress is not None:
            progress(
                f"{truth_condition}: paired trial "
                f"{trial_index + 1}/{config.sensor.trial_count}"
            )
        for policy in policies:
            case = build_realistic_blinded_case(
                truth_condition,
                trial_index,
                policy,
                config,
            )
            saved = decide_realistic_blinded_case(case, config=config)
            revealed = reveal_realistic_operating_outcome(
                truth_condition,
                trial_index,
                case,
                config,
            )
            trials.append(
                _score_realistic_decision(
                    saved,
                    revealed,
                    case,
                    energies,
                    config,
                )
            )
    return tuple(trials)


__all__ = [
    "CORRECTED_FINAL_SCHEDULE_ID",
    "FIT_RELATIVE_OBJECTIVE_TOLERANCE",
    "FIT_SCALED_GRADIENT_TOLERANCE",
    "FIT_STEP_TOLERANCE",
    "OperatingDecisionRealismConfig",
    "OperatingDecisionRealismResult",
    "RealismTruth",
    "RealisticAcquisitionFitSet",
    "RealisticBlindedOperatingCase",
    "RealisticCandidateFit",
    "RealisticOperatingRun",
    "RunInstrumentation",
    "STAGE3_TRUTH_CONDITIONS",
    "Stage3CaseId",
    "TEMPERATURE_DEPENDENT_CONTACT",
    "build_realistic_blinded_case",
    "decide_realistic_blinded_case",
    "fit_realistic_candidate",
    "fit_realistic_acquisition_models",
    "forecast_realistic_margin_interval",
    "nominal_realistic_margins",
    "nominal_realistic_schedule_energy",
    "realism_truth_for_trial",
    "realistic_verification_score",
    "reveal_realistic_operating_outcome",
    "run_operating_decision_realism",
    "run_operating_decision_realism_truth",
    "score_realistic_saved_decision",
    "summarize_operating_decision_realism",
    "verify_realistic_candidate_models",
]
