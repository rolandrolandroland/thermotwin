"""Test whether one added observable beats collecting more terminal data.

Four- and five-state thermal models are fitted to the same virtual campaigns.
Model choice is made with a complete noisy schedule withheld from parameter
fitting, then audited against synthetic physical parameters and hidden states.
"""

from dataclasses import dataclass, replace
import math
import random
from statistics import fmean
from typing import Callable, Dict, NamedTuple, Optional, Sequence, Tuple

from ..core.controls import PiecewiseConstantCurrent, current_at
from ..inference.experiment_selection import (
    ExperimentCandidateScore,
    ExperimentSelectionConfig,
    candidate_current,
    score_experiment_candidate,
)
from ..inference.joint_thermal_parameters import JointThermalFitConfig
from ..inference.sparse_sensors import sparse_withheld_current
from ..numerics.matrices import inverse_and_determinant
from ..observations.lag import FirstOrderTemperatureLag, apply_first_order_temperature_lag
from ..observations.test_stand import (
    IdealTemperatureSensor,
    IdealVirtualTestStand,
    TemperatureSensorLocation,
    observe_contact_trajectory,
    regular_measurement_times,
)
from ..physics.thermoelectric import voltage
from ..simulation.four_node_experiments import (
    constant_current_contact_reference_experiment,
    run_four_node_contact_experiment,
)
from ..simulation.interface_mass_mismatch import (
    InterfaceMassMismatch,
    InterfaceMassTrajectory,
    integrate_interface_mass_truth,
)


FOUR_STATE_MODEL = "four_state"
FIVE_STATE_MODEL = "five_state"
MODEL_NAMES = (FOUR_STATE_MODEL, FIVE_STATE_MODEL)
TRUTH_CONDITIONS = ("matched_four_state", "extra_interface_mass")

COLD_EXCHANGER = "cold_exchanger_temperature"
HOT_EXCHANGER = "hot_exchanger_temperature"
COLD_FACE = "cold_face_temperature"
COLD_HEAT_RATE = "cold_interface_heat_rate"
VOLTAGE = "terminal_voltage"
ALL_CHANNELS = (
    COLD_EXCHANGER,
    HOT_EXCHANGER,
    COLD_FACE,
    COLD_HEAT_RATE,
    VOLTAGE,
)


@dataclass(frozen=True)
class SensorPackage:
    name: str
    candidate_names: Tuple[str, ...]
    channels: Tuple[str, ...]
    extra_sensor_count: int

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("sensor-package name must be nonempty")
        if not self.candidate_names or len(set(self.candidate_names)) != len(
            self.candidate_names
        ):
            raise ValueError("sensor-package candidates must be nonempty and distinct")
        if COLD_EXCHANGER not in self.channels or HOT_EXCHANGER not in self.channels:
            raise ValueError("both exchanger temperatures are required")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("sensor-package channels must be distinct")
        if any(channel not in ALL_CHANNELS for channel in self.channels):
            raise ValueError("sensor package contains an unknown channel")
        if not isinstance(self.extra_sensor_count, int) or self.extra_sensor_count < 0:
            raise ValueError("extra sensor count must be a nonnegative integer")


def default_sensor_packages() -> Tuple[SensorPackage, ...]:
    selected = ("0.8A_20s",)
    exchangers = (COLD_EXCHANGER, HOT_EXCHANGER)
    return (
        SensorPackage("baseline_one_pulse", selected, exchangers, 0),
        SensorPackage(
            "more_exchanger_tests",
            ("0.8A_20s", "0.6A_30s", "0.6A_15s", "0.4A_5s"),
            exchangers,
            0,
        ),
        SensorPackage(
            "add_cold_face_temperature",
            selected,
            (*exchangers, COLD_FACE),
            1,
        ),
        SensorPackage(
            "add_cold_heat_rate",
            selected,
            (*exchangers, COLD_HEAT_RATE),
            1,
        ),
        SensorPackage(
            "add_voltage",
            selected,
            (*exchangers, VOLTAGE),
            1,
        ),
    )


@dataclass(frozen=True)
class SensorDiscriminationConfig:
    trial_count: int = 20
    first_seed: int = 91_001
    truth_log_standard_deviations: Tuple[float, float, float] = (
        0.18,
        0.18,
        0.18,
    )
    interface_mass_nominal: float = 20.0
    interface_mass_log_standard_deviation: float = 0.25
    interface_mass_bounds: Tuple[float, float] = (8.0, 80.0)
    interface_mass_prior_log_standard_deviation: float = 0.60
    run_bias_noise_ratio: float = 2.5
    hidden_face_rmse_threshold: float = 0.05
    parameter_log_rmse_threshold: float = 0.10
    confidence_log_standard_error: float = 0.10
    model_selection_rate_threshold: float = 0.90
    false_confidence_rate_threshold: float = 0.10
    fit_iterations: int = 6
    sampling_interval: float = 1.0
    dense_time_step: float = 0.25
    channel_noise: Tuple[Tuple[str, float], ...] = (
        (COLD_EXCHANGER, 0.02),
        (HOT_EXCHANGER, 0.02),
        (COLD_FACE, 0.02),
        (COLD_HEAT_RATE, 0.05),
        (VOLTAGE, 0.002),
    )
    fit: JointThermalFitConfig = JointThermalFitConfig(
        dense_time_step=0.25,
        gauss_newton_iterations=6,
        initial_log_multipliers=((0.0, 0.0, 0.0),),
    )
    selection: ExperimentSelectionConfig = ExperimentSelectionConfig(
        dense_time_step=0.25,
        monte_carlo_trials=1,
    )
    packages: Tuple[SensorPackage, ...] = default_sensor_packages()

    def __post_init__(self) -> None:
        if not isinstance(self.trial_count, int) or self.trial_count <= 0:
            raise ValueError("trial count must be a positive integer")
        if not isinstance(self.first_seed, int) or self.first_seed < 0:
            raise ValueError("first seed must be a nonnegative integer")
        if len(self.truth_log_standard_deviations) != 3 or any(
            not math.isfinite(value) or value <= 0.0
            for value in self.truth_log_standard_deviations
        ):
            raise ValueError("three positive truth spreads are required")
        positive = (
            ("interface mass nominal", self.interface_mass_nominal),
            (
                "interface mass truth spread",
                self.interface_mass_log_standard_deviation,
            ),
            (
                "interface mass prior spread",
                self.interface_mass_prior_log_standard_deviation,
            ),
            ("run bias ratio", self.run_bias_noise_ratio),
            ("hidden-face threshold", self.hidden_face_rmse_threshold),
            ("parameter threshold", self.parameter_log_rmse_threshold),
            ("confidence threshold", self.confidence_log_standard_error),
            ("sampling interval", self.sampling_interval),
            ("dense time step", self.dense_time_step),
        )
        for name, value in positive:
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            len(self.interface_mass_bounds) != 2
            or self.interface_mass_bounds[0] <= 0.0
            or self.interface_mass_bounds[1] <= self.interface_mass_bounds[0]
        ):
            raise ValueError("interface mass bounds must be positive and ordered")
        if not isinstance(self.fit_iterations, int) or self.fit_iterations <= 0:
            raise ValueError("fit iterations must be a positive integer")
        for name, value in (
            ("model-selection target", self.model_selection_rate_threshold),
            ("false-confidence target", self.false_confidence_rate_threshold),
        ):
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must lie between zero and one")
        noise = dict(self.channel_noise)
        if set(noise) != set(ALL_CHANNELS) or any(
            not math.isfinite(value) or value <= 0.0 for value in noise.values()
        ):
            raise ValueError("every channel needs one positive noise scale")
        if not self.packages or len({item.name for item in self.packages}) != len(
            self.packages
        ):
            raise ValueError("sensor packages must be nonempty and uniquely named")


class DiscriminationTruth(NamedTuple):
    physical_values: Tuple[float, float, float]
    interface_mass: float


class ObservableValue(NamedTuple):
    channel: str
    time: float
    value: float


class ObservablePrediction(NamedTuple):
    values: Tuple[ObservableValue, ...]
    time: Tuple[float, ...]
    cold_face: Tuple[float, ...]


class ObservableRun(NamedTuple):
    name: str
    current: PiecewiseConstantCurrent
    values: Tuple[ObservableValue, ...]


class CandidateModelFit(NamedTuple):
    model_name: str
    log_multipliers: Tuple[float, ...]
    physical_values: Tuple[float, float, float]
    interface_mass: Optional[float]
    observation_rmse_normalized: float
    covariance: Tuple[Tuple[float, ...], ...]
    physical_log_standard_errors: Tuple[float, float, float]
    reached_bound: bool
    evaluation_count: int


class SensorDiscriminationTrial(NamedTuple):
    truth_condition: str
    package_name: str
    trial_index: int
    training_experiment_count: int
    training_energy: float
    extra_sensor_count: int
    four_state_validation_mse: float
    five_state_validation_mse: float
    selected_model: str
    correct_model: str
    model_selected_correctly: bool
    selected_parameter_log_rmse: float
    selected_hidden_face_rmse: float
    selected_fit_confident: bool
    false_confidence: bool
    decision_passed: bool
    selected_reached_bound: bool


class SensorDiscriminationSummary(NamedTuple):
    truth_condition: str
    package_name: str
    trial_count: int
    training_experiment_count: int
    mean_training_energy: float
    extra_sensor_count: int
    correct_model_selection_rate: float
    mean_parameter_log_rmse: float
    mean_hidden_face_rmse: float
    confidence_rate: float
    false_confidence_rate: float
    decision_pass_rate: float
    bound_hit_rate: float
    mean_validation_mse_margin: float


class SensorModelDiscriminationResult(NamedTuple):
    config: SensorDiscriminationConfig
    candidates: Tuple[ExperimentCandidateScore, ...]
    trials: Tuple[SensorDiscriminationTrial, ...]
    summaries: Tuple[SensorDiscriminationSummary, ...]


_TEMPERATURE_LOCATIONS = {
    COLD_EXCHANGER: TemperatureSensorLocation.COLD_EXCHANGER,
    HOT_EXCHANGER: TemperatureSensorLocation.HOT_EXCHANGER,
    COLD_FACE: TemperatureSensorLocation.COLD_FACE,
}


def _candidate_lookup(
    config: SensorDiscriminationConfig,
) -> Dict[str, ExperimentCandidateScore]:
    candidates = {}
    for amplitude in config.selection.current_amplitudes:
        for duration in config.selection.pulse_durations:
            score, _ = score_experiment_candidate(
                amplitude, duration, config.selection
            )
            if score.feasible:
                candidates[score.name] = score
    requested = {
        name for package in config.packages for name in package.candidate_names
    }
    missing = requested - set(candidates)
    if missing:
        raise ValueError(
            "sensor packages reference unavailable candidates: "
            + ", ".join(sorted(missing))
        )
    return candidates


def _trajectory_at_times(trajectory, times: Sequence[float]):
    by_time = {round(time, 9): index for index, time in enumerate(trajectory.time)}
    indices = []
    for time in times:
        key = round(time, 9)
        if key not in by_time:
            raise ValueError("trajectory does not contain a requested sample time")
        indices.append(by_time[key])
    return tuple(indices)


def _lagged_temperature_values(
    trajectory,
    *,
    current: PiecewiseConstantCurrent,
    temperature_channels: Sequence[str],
    sensor_lag: float,
    config: SensorDiscriminationConfig,
) -> Dict[Tuple[str, float], float]:
    sensors = tuple(
        IdealTemperatureSensor(channel, _TEMPERATURE_LOCATIONS[channel])
        for channel in temperature_channels
    )
    if not sensors:
        return {}
    dense = observe_contact_trajectory(
        trajectory.four_node_projection
        if isinstance(trajectory, InterfaceMassTrajectory)
        else trajectory,
        current=current,
        test_stand=IdealVirtualTestStand(
            sensors=sensors,
            sampling_interval=config.dense_time_step,
        ),
    )
    lagged = apply_first_order_temperature_lag(
        dense,
        FirstOrderTemperatureLag(default_time_constant=sensor_lag),
    ).dataset
    requested = {
        round(time, 9)
        for time in regular_measurement_times(trajectory.time[-1], config.sampling_interval)
    }
    return {
        (item.sensor_name, round(item.time, 9)): item.temperature
        for item in lagged.observations
        if round(item.time, 9) in requested
    }


def _simulate_observables(
    model_name: str,
    current: PiecewiseConstantCurrent,
    channels: Sequence[str],
    physical_values: Tuple[float, float, float],
    interface_mass: Optional[float],
    config: SensorDiscriminationConfig,
) -> ObservablePrediction:
    resistance, capacitance, sensor_lag = physical_values
    reference = constant_current_contact_reference_experiment()
    thermal = replace(
        reference.thermal_parameters,
        cold_contact_resistance=resistance,
        cold_face_thermal_capacitance=capacitance,
    )
    if model_name == FOUR_STATE_MODEL:
        experiment = replace(
            reference,
            thermal_parameters=thermal,
            duration=80.0,
            time_step=config.dense_time_step,
            current=current,
        )
        trajectory = run_four_node_contact_experiment(experiment).trajectory
    elif model_name == FIVE_STATE_MODEL:
        if interface_mass is None:
            raise ValueError("five-state simulation needs an interface mass")
        trajectory = integrate_interface_mass_truth(
            reference.thermoelectric_parameters,
            thermal,
            InterfaceMassMismatch(thermal_capacitance=interface_mass),
            initial_temperature=300.0,
            duration=80.0,
            time_step=config.dense_time_step,
            current=current,
            cold_reservoir_temperature=300.0,
            hot_reservoir_temperature=300.0,
        )
    else:
        raise ValueError(f"unknown candidate model: {model_name}")

    sample_times = regular_measurement_times(80.0, config.sampling_interval)
    indices = _trajectory_at_times(trajectory, sample_times)
    temperature_channels = tuple(
        channel for channel in channels if channel in _TEMPERATURE_LOCATIONS
    )
    temperatures = _lagged_temperature_values(
        trajectory,
        current=current,
        temperature_channels=temperature_channels,
        sensor_lag=sensor_lag,
        config=config,
    )
    values = []
    for time, index in zip(sample_times, indices):
        for channel in channels:
            if channel in _TEMPERATURE_LOCATIONS:
                value = temperatures[(channel, round(time, 9))]
            elif channel == COLD_HEAT_RATE:
                if model_name == FOUR_STATE_MODEL:
                    value = (
                        trajectory.cold_exchanger[index]
                        - trajectory.cold_face[index]
                    ) / resistance
                else:
                    value = (
                        trajectory.cold_exchanger[index]
                        - trajectory.cold_interface[index]
                    ) / (0.5 * resistance)
            elif channel == VOLTAGE:
                value = voltage(
                    reference.thermoelectric_parameters,
                    current_at(current, time),
                    trajectory.hot_face[index],
                    trajectory.cold_face[index],
                )
            else:
                raise ValueError(f"unknown observable channel: {channel}")
            values.append(ObservableValue(channel, time, value))
    return ObservablePrediction(
        values=tuple(values),
        time=trajectory.time,
        cold_face=trajectory.cold_face,
    )


def _truth_for_trial(
    config: SensorDiscriminationConfig,
    trial_index: int,
) -> DiscriminationTruth:
    random_source = random.Random(config.first_seed + 10_000 * trial_index)
    physical_values = tuple(
        nominal * math.exp(random_source.gauss(0.0, spread))
        for nominal, spread in zip(
            config.fit.nominal_values,
            config.truth_log_standard_deviations,
        )
    )
    interface_mass = config.interface_mass_nominal * math.exp(
        random_source.gauss(0.0, config.interface_mass_log_standard_deviation)
    )
    return DiscriminationTruth(
        physical_values=physical_values,  # type: ignore[arg-type]
        interface_mass=interface_mass,
    )


def _observed_run(
    prediction: ObservablePrediction,
    *,
    name: str,
    current: PiecewiseConstantCurrent,
    seed: int,
    config: SensorDiscriminationConfig,
) -> ObservableRun:
    noise_by_channel = dict(config.channel_noise)
    values = []
    for channel_index, channel in enumerate(ALL_CHANNELS):
        selected = tuple(item for item in prediction.values if item.channel == channel)
        if not selected:
            continue
        random_source = random.Random(seed + 100 * channel_index)
        noise = noise_by_channel[channel]
        bias = random_source.gauss(0.0, config.run_bias_noise_ratio * noise)
        values.extend(
            ObservableValue(
                channel=item.channel,
                time=item.time,
                value=item.value + bias + random_source.gauss(0.0, noise),
            )
            for item in selected
        )
    values.sort(key=lambda item: (item.time, ALL_CHANNELS.index(item.channel)))
    return ObservableRun(name=name, current=current, values=tuple(values))


def _prediction_map(
    values: Sequence[ObservableValue],
) -> Dict[Tuple[str, float], float]:
    return {(item.channel, item.time): item.value for item in values}


def _profiled_normalized_residuals(
    observed: ObservableRun,
    predicted: ObservablePrediction,
    config: SensorDiscriminationConfig,
) -> Tuple[float, ...]:
    prediction = _prediction_map(predicted.values)
    noise = dict(config.channel_noise)
    differences = {channel: [] for channel in {item.channel for item in observed.values}}
    for item in observed.values:
        differences[item.channel].append(
            item.value - prediction[(item.channel, item.time)]
        )
    biases = {channel: fmean(values) for channel, values in differences.items()}
    return tuple(
        (
            prediction[(item.channel, item.time)]
            + biases[item.channel]
            - item.value
        )
        / noise[item.channel]
        for item in observed.values
    )


def _model_parameter_spec(
    model_name: str,
    config: SensorDiscriminationConfig,
):
    physical_bounds = config.fit.log_bounds
    physical_prior = config.fit.nominal_values
    if model_name == FOUR_STATE_MODEL:
        return (
            tuple(physical_bounds),
            tuple(config.selection.prior_standard_deviations[:3]),
            tuple(physical_prior),
        )
    if model_name == FIVE_STATE_MODEL:
        lower, upper = config.interface_mass_bounds
        return (
            (
                *physical_bounds,
                (
                    math.log(lower / config.interface_mass_nominal),
                    math.log(upper / config.interface_mass_nominal),
                ),
            ),
            (
                *config.selection.prior_standard_deviations[:3],
                config.interface_mass_prior_log_standard_deviation,
            ),
            (*physical_prior, config.interface_mass_nominal),
        )
    raise ValueError(f"unknown candidate model: {model_name}")


def fit_candidate_model(
    model_name: str,
    runs: Sequence[ObservableRun],
    channels: Sequence[str],
    config: SensorDiscriminationConfig,
) -> CandidateModelFit:
    """Fit one candidate topology with priors counted once across all runs."""

    runs = tuple(runs)
    if not runs:
        raise ValueError("candidate-model fit needs at least one run")
    bounds, prior_scales, nominal_values = _model_parameter_spec(model_name, config)
    parameter_count = len(bounds)
    cache: Dict[Tuple[float, ...], Tuple[float, ...]] = {}
    observation_count = sum(len(run.values) for run in runs)
    evaluation_count = 0

    def bounded(values: Sequence[float]) -> Tuple[float, ...]:
        return tuple(
            min(bounds[index][1], max(bounds[index][0], float(value)))
            for index, value in enumerate(values)
        )

    def physical_values(offsets: Sequence[float]) -> Tuple[float, float, float]:
        values = tuple(
            nominal_values[index] * math.exp(offsets[index])
            for index in range(3)
        )
        return values  # type: ignore[return-value]

    def interface_mass(offsets: Sequence[float]) -> Optional[float]:
        if model_name == FOUR_STATE_MODEL:
            return None
        return config.interface_mass_nominal * math.exp(offsets[3])

    def residuals(offsets: Sequence[float]) -> Tuple[float, ...]:
        nonlocal evaluation_count
        key = bounded(offsets)
        if key not in cache:
            combined = []
            for run in runs:
                prediction = _simulate_observables(
                    model_name,
                    run.current,
                    channels,
                    physical_values(key),
                    interface_mass(key),
                    config,
                )
                combined.extend(
                    _profiled_normalized_residuals(run, prediction, config)
                )
            combined.extend(
                value / scale for value, scale in zip(key, prior_scales)
            )
            cache[key] = tuple(combined)
            evaluation_count += 1
        return cache[key]

    def objective(offsets: Sequence[float]) -> float:
        values = residuals(offsets)
        return sum(value * value for value in values) / len(values)

    values = [0.0] * parameter_count
    damping = config.fit.initial_damping
    objective(values)
    for _ in range(config.fit_iterations):
        error = residuals(values)
        columns = []
        for index in range(parameter_count):
            minus = list(values)
            plus = list(values)
            minus[index] = max(
                bounds[index][0],
                minus[index] - config.fit.finite_difference_step,
            )
            plus[index] = min(
                bounds[index][1],
                plus[index] + config.fit.finite_difference_step,
            )
            denominator = plus[index] - minus[index]
            left = residuals(minus)
            right = residuals(plus)
            columns.append(
                tuple(
                    (right_value - left_value) / denominator
                    for left_value, right_value in zip(left, right)
                )
            )
        normal = tuple(
            tuple(
                sum(a * b for a, b in zip(left, right))
                for right in columns
            )
            for left in columns
        )
        gradient = tuple(
            sum(value * item for value, item in zip(column, error))
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
            candidate = [
                min(
                    bounds[index][1],
                    max(bounds[index][0], value + fraction * update[index]),
                )
                for index, value in enumerate(values)
            ]
            if objective(candidate) < starting_loss:
                values = list(bounded(candidate))
                damping = max(config.fit.initial_damping * 1.0e-3, damping * 0.3)
                accepted = True
                break
        if not accepted:
            damping *= 10.0
    best = bounded(values)
    best_residuals = residuals(best)

    columns = []
    for index in range(parameter_count):
        minus = list(best)
        plus = list(best)
        minus[index] = max(
            bounds[index][0], minus[index] - config.fit.finite_difference_step
        )
        plus[index] = min(
            bounds[index][1], plus[index] + config.fit.finite_difference_step
        )
        denominator = plus[index] - minus[index]
        left = residuals(minus)
        right = residuals(plus)
        columns.append(
            tuple(
                (right_value - left_value) / denominator
                for left_value, right_value in zip(left, right)
            )
        )
    information = tuple(
        tuple(
            sum(a * b for a, b in zip(left, right))
            for right in columns
        )
        for left in columns
    )
    covariance, _ = inverse_and_determinant(information)
    standard_errors = tuple(
        math.sqrt(max(0.0, covariance[index][index])) for index in range(3)
    )
    observed_residuals = best_residuals[:observation_count]
    return CandidateModelFit(
        model_name=model_name,
        log_multipliers=best,
        physical_values=physical_values(best),
        interface_mass=interface_mass(best),
        observation_rmse_normalized=math.sqrt(
            sum(value * value for value in observed_residuals)
            / len(observed_residuals)
        ),
        covariance=covariance,
        physical_log_standard_errors=standard_errors,  # type: ignore[arg-type]
        reached_bound=any(
            abs(best[index] - bounds[index][0]) <= 1.0e-5
            or abs(bounds[index][1] - best[index]) <= 1.0e-5
            for index in range(parameter_count)
        ),
        evaluation_count=evaluation_count,
    )


def _validation_score(
    fit: CandidateModelFit,
    observed: ObservableRun,
    channels: Sequence[str],
    config: SensorDiscriminationConfig,
) -> Tuple[float, ObservablePrediction]:
    prediction = _simulate_observables(
        fit.model_name,
        observed.current,
        channels,
        fit.physical_values,
        fit.interface_mass,
        config,
    )
    residuals = _profiled_normalized_residuals(observed, prediction, config)
    return fmean(value * value for value in residuals), prediction


def _rmse(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("RMSE values must be nonempty and aligned")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)) / len(left))


def _run_trial_package(
    *,
    truth_condition: str,
    trial_index: int,
    truth: DiscriminationTruth,
    package: SensorPackage,
    candidates: Dict[str, ExperimentCandidateScore],
    config: SensorDiscriminationConfig,
) -> SensorDiscriminationTrial:
    truth_model = (
        FOUR_STATE_MODEL
        if truth_condition == "matched_four_state"
        else FIVE_STATE_MODEL
    )
    condition_index = TRUTH_CONDITIONS.index(truth_condition)
    candidate_order = {name: index for index, name in enumerate(sorted(candidates))}
    training_runs = []
    for candidate_name in package.candidate_names:
        candidate = candidates[candidate_name]
        current = candidate_current(
            candidate.current_amplitude, candidate.pulse_duration
        )
        prediction = _simulate_observables(
            truth_model,
            current,
            package.channels,
            truth.physical_values,
            truth.interface_mass if truth_model == FIVE_STATE_MODEL else None,
            config,
        )
        seed = (
            config.first_seed
            + condition_index * 1_000_000
            + trial_index * 10_000
            + candidate_order[candidate_name] * 100
        )
        training_runs.append(
            _observed_run(
                prediction,
                name=candidate_name,
                current=current,
                seed=seed,
                config=config,
            )
        )

    fits = {
        model_name: fit_candidate_model(
            model_name, training_runs, package.channels, config
        )
        for model_name in MODEL_NAMES
    }
    validation_current = sparse_withheld_current()
    validation_truth = _simulate_observables(
        truth_model,
        validation_current,
        package.channels,
        truth.physical_values,
        truth.interface_mass if truth_model == FIVE_STATE_MODEL else None,
        config,
    )
    validation_observed = _observed_run(
        validation_truth,
        name="withheld_bipolar",
        current=validation_current,
        seed=(
            config.first_seed
            + condition_index * 1_000_000
            + trial_index * 10_000
            + 9_900
        ),
        config=config,
    )
    scores = {}
    predictions = {}
    for model_name, fit in fits.items():
        scores[model_name], predictions[model_name] = _validation_score(
            fit, validation_observed, package.channels, config
        )
    selected_model = min(MODEL_NAMES, key=lambda name: scores[name])
    selected_fit = fits[selected_model]
    selected_prediction = predictions[selected_model]
    parameter_log_rmse = math.sqrt(
        sum(
            math.log(estimate / expected) ** 2
            for estimate, expected in zip(
                selected_fit.physical_values, truth.physical_values
            )
        )
        / 3.0
    )
    hidden_face_rmse = _rmse(
        selected_prediction.cold_face,
        validation_truth.cold_face,
    )
    correct_model = truth_model
    correct = selected_model == correct_model
    confident = (
        max(selected_fit.physical_log_standard_errors)
        <= config.confidence_log_standard_error
    )
    decision_passed = (
        correct
        and parameter_log_rmse <= config.parameter_log_rmse_threshold
        and hidden_face_rmse <= config.hidden_face_rmse_threshold
    )
    false_confidence = confident and not decision_passed
    return SensorDiscriminationTrial(
        truth_condition=truth_condition,
        package_name=package.name,
        trial_index=trial_index,
        training_experiment_count=len(package.candidate_names),
        training_energy=sum(
            candidates[name].electrical_energy for name in package.candidate_names
        ),
        extra_sensor_count=package.extra_sensor_count,
        four_state_validation_mse=scores[FOUR_STATE_MODEL],
        five_state_validation_mse=scores[FIVE_STATE_MODEL],
        selected_model=selected_model,
        correct_model=correct_model,
        model_selected_correctly=correct,
        selected_parameter_log_rmse=parameter_log_rmse,
        selected_hidden_face_rmse=hidden_face_rmse,
        selected_fit_confident=confident,
        false_confidence=false_confidence,
        decision_passed=decision_passed,
        selected_reached_bound=selected_fit.reached_bound,
    )


def summarize_sensor_discrimination(
    trials: Sequence[SensorDiscriminationTrial],
    config: SensorDiscriminationConfig,
) -> Tuple[SensorDiscriminationSummary, ...]:
    summaries = []
    for condition in TRUTH_CONDITIONS:
        for package in config.packages:
            selected = tuple(
                item
                for item in trials
                if item.truth_condition == condition
                and item.package_name == package.name
            )
            if len(selected) != config.trial_count:
                raise ValueError("each package summary needs every paired trial")
            correct_model = (
                FOUR_STATE_MODEL
                if condition == "matched_four_state"
                else FIVE_STATE_MODEL
            )
            margins = tuple(
                (
                    item.five_state_validation_mse
                    - item.four_state_validation_mse
                )
                * (1.0 if correct_model == FOUR_STATE_MODEL else -1.0)
                for item in selected
            )
            summaries.append(
                SensorDiscriminationSummary(
                    truth_condition=condition,
                    package_name=package.name,
                    trial_count=len(selected),
                    training_experiment_count=selected[0].training_experiment_count,
                    mean_training_energy=fmean(
                        item.training_energy for item in selected
                    ),
                    extra_sensor_count=selected[0].extra_sensor_count,
                    correct_model_selection_rate=fmean(
                        float(item.model_selected_correctly) for item in selected
                    ),
                    mean_parameter_log_rmse=fmean(
                        item.selected_parameter_log_rmse for item in selected
                    ),
                    mean_hidden_face_rmse=fmean(
                        item.selected_hidden_face_rmse for item in selected
                    ),
                    confidence_rate=fmean(
                        float(item.selected_fit_confident) for item in selected
                    ),
                    false_confidence_rate=fmean(
                        float(item.false_confidence) for item in selected
                    ),
                    decision_pass_rate=fmean(
                        float(item.decision_passed) for item in selected
                    ),
                    bound_hit_rate=fmean(
                        float(item.selected_reached_bound) for item in selected
                    ),
                    mean_validation_mse_margin=fmean(margins),
                )
            )
    return tuple(summaries)


def run_sensor_model_discrimination(
    config: SensorDiscriminationConfig = SensorDiscriminationConfig(),
    *,
    progress: Optional[Callable[[str], None]] = None,
) -> SensorModelDiscriminationResult:
    """Run every sensor package against paired four- and five-state truths."""

    candidates = _candidate_lookup(config)
    trials = []
    for condition in TRUTH_CONDITIONS:
        for trial_index in range(config.trial_count):
            if progress is not None:
                progress(
                    f"{condition}: paired trial {trial_index + 1}/{config.trial_count}"
                )
            truth = _truth_for_trial(config, trial_index)
            for package in config.packages:
                trials.append(
                    _run_trial_package(
                        truth_condition=condition,
                        trial_index=trial_index,
                        truth=truth,
                        package=package,
                        candidates=candidates,
                        config=config,
                    )
                )
    trials = tuple(trials)
    return SensorModelDiscriminationResult(
        config=config,
        candidates=tuple(candidates[name] for name in sorted(candidates)),
        trials=trials,
        summaries=summarize_sensor_discrimination(trials, config),
    )


__all__ = [
    "COLD_EXCHANGER",
    "COLD_FACE",
    "COLD_HEAT_RATE",
    "FIVE_STATE_MODEL",
    "FOUR_STATE_MODEL",
    "HOT_EXCHANGER",
    "SensorDiscriminationConfig",
    "SensorDiscriminationSummary",
    "SensorDiscriminationTrial",
    "SensorModelDiscriminationResult",
    "SensorPackage",
    "VOLTAGE",
    "default_sensor_packages",
    "fit_candidate_model",
    "run_sensor_model_discrimination",
    "summarize_sensor_discrimination",
]
