"""Nonlinear joint inference for contact, capacitance, and sensor lag.

The estimator uses only the two accessible exchanger-temperature histories.
It fits three positive physical quantities in log coordinates and profiles one
constant bias for each sensor analytically.  This keeps the conventional
baseline dependency-free while making the nuisance treatment identical across
experiment-selection candidates.
"""

from dataclasses import dataclass
import math
from typing import NamedTuple, Optional, Sequence, Tuple

from ..core.controls import PiecewiseConstantCurrent
from ..numerics.matrices import gram_matrix, inverse_and_determinant
from ..observations.bias import FixedTemperatureBias, apply_fixed_temperature_bias
from ..observations.noise import GaussianTemperatureNoise, apply_gaussian_temperature_noise
from ..observations.test_stand import ObservationDataset
from .sparse_sensors import (
    ACCESSIBLE_SENSOR_NAMES,
    simulate_accessible_observations,
)


JOINT_PARAMETER_NAMES = (
    "cold_contact_resistance_K_per_W",
    "cold_face_capacitance_J_per_K",
    "shared_sensor_lag_s",
)


@dataclass(frozen=True)
class JointThermalTruth:
    cold_contact_resistance: float = 0.25
    cold_face_thermal_capacitance: float = 50.0
    sensor_time_constant: float = 1.5
    cold_sensor_bias: float = 0.08
    hot_sensor_bias: float = -0.04

    def __post_init__(self) -> None:
        for name, value in zip(
            JOINT_PARAMETER_NAMES,
            self.physical_values,
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not all(
            math.isfinite(value)
            for value in (self.cold_sensor_bias, self.hot_sensor_bias)
        ):
            raise ValueError("sensor biases must be finite")

    @property
    def physical_values(self) -> Tuple[float, float, float]:
        return (
            self.cold_contact_resistance,
            self.cold_face_thermal_capacitance,
            self.sensor_time_constant,
        )


@dataclass(frozen=True)
class JointThermalFitConfig:
    """Bounds, noise scale, simulation resolution, and optimizer settings."""

    nominal_values: Tuple[float, float, float] = (0.25, 50.0, 1.5)
    value_bounds: Tuple[Tuple[float, float], ...] = (
        (0.08, 0.60),
        (20.0, 120.0),
        (0.15, 5.0),
    )
    sampling_interval: float = 1.0
    dense_time_step: float = 0.2
    noise_standard_deviation: float = 0.02
    finite_difference_step: float = 0.01
    gauss_newton_iterations: int = 12
    initial_damping: float = 1.0e-3
    initial_log_multipliers: Tuple[Tuple[float, float, float], ...] = (
        (-0.24, 0.18, -0.16),
        (0.22, -0.16, 0.20),
        (0.08, 0.10, -0.08),
    )
    fixed_log_multipliers: Tuple[Optional[float], ...] = (None, None, None)

    def __post_init__(self) -> None:
        if len(self.nominal_values) != 3 or any(
            not math.isfinite(value) or value <= 0.0
            for value in self.nominal_values
        ):
            raise ValueError("three finite positive nominal values are required")
        if len(self.value_bounds) != 3:
            raise ValueError("three physical-parameter bounds are required")
        for lower, upper in self.value_bounds:
            if (
                not math.isfinite(lower)
                or not math.isfinite(upper)
                or lower <= 0.0
                or upper <= lower
            ):
                raise ValueError("physical-parameter bounds must be positive and ordered")
        for name, value in (
            ("sampling interval", self.sampling_interval),
            ("dense time step", self.dense_time_step),
            ("noise standard deviation", self.noise_standard_deviation),
            ("finite-difference step", self.finite_difference_step),
            ("initial damping", self.initial_damping),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not isinstance(self.gauss_newton_iterations, int)
            or isinstance(self.gauss_newton_iterations, bool)
            or self.gauss_newton_iterations <= 0
        ):
            raise ValueError("Gauss-Newton iterations must be a positive integer")
        if not self.initial_log_multipliers or any(
            len(start) != 3 or any(not math.isfinite(value) for value in start)
            for start in self.initial_log_multipliers
        ):
            raise ValueError("at least one finite three-parameter start is required")
        if len(self.fixed_log_multipliers) != 3 or any(
            value is not None and not math.isfinite(value)
            for value in self.fixed_log_multipliers
        ):
            raise ValueError("fixed log multipliers must contain three finite values or None")
        lower_logs, upper_logs = zip(*self.log_bounds)
        for index, value in enumerate(self.fixed_log_multipliers):
            if value is not None and not lower_logs[index] <= value <= upper_logs[index]:
                raise ValueError("fixed log multiplier lies outside its physical bound")

    @property
    def log_bounds(self) -> Tuple[Tuple[float, float], ...]:
        return tuple(
            (
                math.log(lower / nominal),
                math.log(upper / nominal),
            )
            for nominal, (lower, upper) in zip(
                self.nominal_values,
                self.value_bounds,
            )
        )


class JointThermalFitEvaluation(NamedTuple):
    log_multipliers: Tuple[float, float, float]
    normalized_mean_squared_error: float


class JointThermalInterval(NamedTuple):
    name: str
    estimate: float
    standard_error_log_or_K: float
    lower_95: float
    upper_95: float


class JointThermalIdentifiability(NamedTuple):
    singular_values: Tuple[float, float, float]
    effective_rank: int
    supported_rank: int
    condition_number: float
    maximum_log_displacement: float
    required_noise_normalized_signal: float


class JointThermalFitResult(NamedTuple):
    log_multipliers: Tuple[float, float, float]
    physical_values: Tuple[float, float, float]
    cold_sensor_bias: float
    hot_sensor_bias: float
    observation_rmse: float
    normalized_mean_squared_error: float
    covariance: Tuple[Tuple[float, ...], ...]
    intervals: Tuple[JointThermalInterval, ...]
    correlation: Tuple[Tuple[float, ...], ...]
    identifiability: JointThermalIdentifiability
    reached_bound: bool
    evaluations: Tuple[JointThermalFitEvaluation, ...]


def generate_joint_thermal_observations(
    current: PiecewiseConstantCurrent,
    truth: JointThermalTruth,
    config: JointThermalFitConfig,
    *,
    noise_seed: Optional[int],
) -> ObservationDataset:
    """Generate paired accessible-sensor observations for one current input."""

    dataset, _ = simulate_accessible_observations(
        current,
        cold_contact_resistance=truth.cold_contact_resistance,
        cold_face_thermal_capacitance=truth.cold_face_thermal_capacitance,
        sensor_time_constant=truth.sensor_time_constant,
        sampling_interval=config.sampling_interval,
        dense_time_step=config.dense_time_step,
    )
    dataset = apply_fixed_temperature_bias(
        dataset,
        FixedTemperatureBias(
            sensor_biases=(
                (ACCESSIBLE_SENSOR_NAMES[0], truth.cold_sensor_bias),
                (ACCESSIBLE_SENSOR_NAMES[1], truth.hot_sensor_bias),
            )
        ),
    ).dataset
    if noise_seed is not None:
        dataset = apply_gaussian_temperature_noise(
            dataset,
            GaussianTemperatureNoise(
                default_standard_deviation=config.noise_standard_deviation,
                random_seed=noise_seed,
            ),
        ).dataset
    return dataset


def _prediction_map(dataset: ObservationDataset):
    return {
        (item.sensor_name, item.time): item.temperature
        for item in dataset.observations
    }


def _values_from_log_multipliers(
    log_multipliers: Sequence[float],
    nominal_values: Sequence[float],
) -> Tuple[float, float, float]:
    values = tuple(
        nominal * math.exp(offset)
        for nominal, offset in zip(nominal_values, log_multipliers)
    )
    if len(values) != 3:
        raise ValueError("three joint physical parameters are required")
    return values  # type: ignore[return-value]


def _profiled_residuals(
    observed: ObservationDataset,
    predicted: ObservationDataset,
    *,
    scale: float,
) -> Tuple[Tuple[float, ...], float, float]:
    predicted_map = _prediction_map(predicted)
    differences = {name: [] for name in ACCESSIBLE_SENSOR_NAMES}
    for item in observed.observations:
        key = (item.sensor_name, item.time)
        if key not in predicted_map:
            raise ValueError("prediction is missing an observed sensor-time pair")
        differences[item.sensor_name].append(
            item.temperature - predicted_map[key]
        )
    if any(not values for values in differences.values()):
        raise ValueError("both accessible sensors need at least one observation")
    biases = {
        name: sum(values) / len(values)
        for name, values in differences.items()
    }
    residuals = tuple(
        (
            predicted_map[(item.sensor_name, item.time)]
            + biases[item.sensor_name]
            - item.temperature
        )
        / scale
        for item in observed.observations
    )
    return (
        residuals,
        biases[ACCESSIBLE_SENSOR_NAMES[0]],
        biases[ACCESSIBLE_SENSOR_NAMES[1]],
    )


def _symmetric_eigenvalues(
    matrix: Sequence[Sequence[float]],
    *,
    tolerance: float = 1.0e-12,
) -> Tuple[float, ...]:
    values = [list(map(float, row)) for row in matrix]
    size = len(values)
    if size == 0 or any(len(row) != size for row in values):
        raise ValueError("eigenvalue matrix must be nonempty and square")
    scale = max(1.0, max(abs(value) for row in values for value in row))
    for _ in range(100 * size * size):
        p, q = max(
            (
                (row, column)
                for row in range(size)
                for column in range(row + 1, size)
            ),
            key=lambda pair: abs(values[pair[0]][pair[1]]),
            default=(0, 0),
        )
        if p == q or abs(values[p][q]) <= tolerance * scale:
            break
        app = values[p][p]
        aqq = values[q][q]
        apq = values[p][q]
        angle = 0.5 * math.atan2(2.0 * apq, aqq - app)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        for index in range(size):
            if index in (p, q):
                continue
            aip = values[index][p]
            aiq = values[index][q]
            values[index][p] = values[p][index] = cosine * aip - sine * aiq
            values[index][q] = values[q][index] = sine * aip + cosine * aiq
        values[p][p] = (
            cosine * cosine * app
            - 2.0 * sine * cosine * apq
            + sine * sine * aqq
        )
        values[q][q] = (
            sine * sine * app
            + 2.0 * sine * cosine * apq
            + cosine * cosine * aqq
        )
        values[p][q] = values[q][p] = 0.0
    return tuple(
        sorted((values[index][index] for index in range(size)), reverse=True)
    )


def _identifiability_from_columns(
    columns: Sequence[Sequence[float]],
    *,
    maximum_log_displacement: float = 0.30,
    required_noise_normalized_signal: float = 1.0,
) -> JointThermalIdentifiability:
    information = tuple(
        tuple(
            sum(a * b for a, b in zip(left, right))
            for right in columns
        )
        for left in columns
    )
    eigenvalues = _symmetric_eigenvalues(information)
    singular_values = tuple(math.sqrt(max(0.0, value)) for value in eigenvalues)
    largest = singular_values[0]
    relative_threshold = 1.0e-6 * largest
    effective_rank = sum(value > relative_threshold for value in singular_values)
    practical_threshold = required_noise_normalized_signal / maximum_log_displacement
    supported_rank = sum(value >= practical_threshold for value in singular_values)
    condition = (
        largest / singular_values[-1]
        if singular_values[-1] > relative_threshold
        else math.inf
    )
    return JointThermalIdentifiability(
        singular_values=singular_values,  # type: ignore[arg-type]
        effective_rank=effective_rank,
        supported_rank=supported_rank,
        condition_number=condition,
        maximum_log_displacement=maximum_log_displacement,
        required_noise_normalized_signal=required_noise_normalized_signal,
    )


def fit_joint_thermal_parameters(
    current: PiecewiseConstantCurrent,
    observations: ObservationDataset,
    config: JointThermalFitConfig = JointThermalFitConfig(),
) -> JointThermalFitResult:
    """Fit R_contact, cold-face capacitance, lag, and two profiled biases."""

    lower_logs, upper_logs = zip(*config.log_bounds)
    fixed = config.fixed_log_multipliers
    free_indices = tuple(index for index, value in enumerate(fixed) if value is None)
    evaluations = []
    residual_cache = {}
    bias_cache = {}

    def bounded(values: Sequence[float]) -> Tuple[float, float, float]:
        bounded_values = []
        for index, value in enumerate(values):
            if fixed[index] is not None:
                value = fixed[index]  # type: ignore[assignment]
            bounded_values.append(
                min(upper_logs[index], max(lower_logs[index], float(value)))
            )
        return tuple(bounded_values)  # type: ignore[return-value]

    def normalized_residuals(offsets: Sequence[float]) -> Tuple[float, ...]:
        key = bounded(offsets)
        if key not in residual_cache:
            resistance, capacitance, lag = _values_from_log_multipliers(
                key, config.nominal_values
            )
            prediction, _ = simulate_accessible_observations(
                current,
                cold_contact_resistance=resistance,
                cold_face_thermal_capacitance=capacitance,
                sensor_time_constant=lag,
                sampling_interval=config.sampling_interval,
                dense_time_step=config.dense_time_step,
            )
            residuals, cold_bias, hot_bias = _profiled_residuals(
                observations,
                prediction,
                scale=config.noise_standard_deviation,
            )
            residual_cache[key] = residuals
            bias_cache[key] = (cold_bias, hot_bias)
            evaluations.append(
                JointThermalFitEvaluation(
                    key,
                    sum(value * value for value in residuals) / len(residuals),
                )
            )
        return residual_cache[key]

    def objective(offsets: Sequence[float]) -> float:
        residuals = normalized_residuals(offsets)
        return sum(value * value for value in residuals) / len(residuals)

    def optimize(start: Sequence[float]) -> Tuple[float, float, float]:
        current_values = list(bounded(start))
        damping = config.initial_damping
        objective(current_values)
        for _ in range(config.gauss_newton_iterations):
            if not free_indices:
                break
            residual = normalized_residuals(current_values)
            columns = []
            for index in free_indices:
                minus = list(current_values)
                plus = list(current_values)
                minus[index] = max(
                    lower_logs[index],
                    minus[index] - config.finite_difference_step,
                )
                plus[index] = min(
                    upper_logs[index],
                    plus[index] + config.finite_difference_step,
                )
                denominator = plus[index] - minus[index]
                if denominator <= 0.0:
                    columns.append((0.0,) * len(residual))
                    continue
                left = normalized_residuals(minus)
                right = normalized_residuals(plus)
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
                sum(value * error for value, error in zip(column, residual))
                for column in columns
            )
            damped = tuple(
                tuple(
                    value
                    + (
                        damping * max(1.0, normal[row][row])
                        if row == column
                        else 0.0
                    )
                    for column, value in enumerate(matrix_row)
                )
                for row, matrix_row in enumerate(normal)
            )
            try:
                inverse, _ = inverse_and_determinant(damped)
            except ValueError:
                damping *= 10.0
                continue
            update = tuple(
                -sum(value * component for value, component in zip(row, gradient))
                for row in inverse
            )
            starting_loss = objective(current_values)
            accepted = False
            for fraction in (1.0, 0.5, 0.25, 0.1, 0.03):
                candidate = list(current_values)
                for index, step in zip(free_indices, update):
                    candidate[index] = min(
                        upper_logs[index],
                        max(lower_logs[index], current_values[index] + fraction * step),
                    )
                if objective(candidate) < starting_loss:
                    current_values = list(bounded(candidate))
                    damping = max(config.initial_damping * 1.0e-3, damping * 0.3)
                    accepted = True
                    break
            if not accepted:
                damping *= 10.0
        return bounded(current_values)

    candidates = tuple(optimize(start) for start in config.initial_log_multipliers)
    best = min(candidates, key=objective)
    residuals = normalized_residuals(best)
    cold_bias, hot_bias = bias_cache[best]
    physical_values = _values_from_log_multipliers(best, config.nominal_values)

    # Build the full local Jacobian, retaining both nuisance-bias columns for
    # covariance while using profiled physical columns for the rank decision.
    physical_columns = []
    for index in range(3):
        minus = list(best)
        plus = list(best)
        minus[index] = max(lower_logs[index], minus[index] - config.finite_difference_step)
        plus[index] = min(upper_logs[index], plus[index] + config.finite_difference_step)
        denominator = plus[index] - minus[index]
        if denominator <= 0.0:
            physical_columns.append((0.0,) * len(residuals))
            continue
        left = normalized_residuals(minus)
        right = normalized_residuals(plus)
        physical_columns.append(
            tuple(
                (right_value - left_value) / denominator
                for left_value, right_value in zip(left, right)
            )
        )
    identifiability = _identifiability_from_columns(physical_columns)

    raw_physical_columns = []
    for index in range(3):
        minus = list(best)
        plus = list(best)
        minus[index] = max(lower_logs[index], minus[index] - config.finite_difference_step)
        plus[index] = min(upper_logs[index], plus[index] + config.finite_difference_step)
        denominator = plus[index] - minus[index]
        left_values = _values_from_log_multipliers(minus, config.nominal_values)
        right_values = _values_from_log_multipliers(plus, config.nominal_values)
        left_prediction, _ = simulate_accessible_observations(
            current,
            cold_contact_resistance=left_values[0],
            cold_face_thermal_capacitance=left_values[1],
            sensor_time_constant=left_values[2],
            sampling_interval=config.sampling_interval,
            dense_time_step=config.dense_time_step,
        )
        right_prediction, _ = simulate_accessible_observations(
            current,
            cold_contact_resistance=right_values[0],
            cold_face_thermal_capacitance=right_values[1],
            sensor_time_constant=right_values[2],
            sampling_interval=config.sampling_interval,
            dense_time_step=config.dense_time_step,
        )
        left_map = _prediction_map(left_prediction)
        right_map = _prediction_map(right_prediction)
        raw_physical_columns.append(
            tuple(
                (
                    right_map[(item.sensor_name, item.time)]
                    - left_map[(item.sensor_name, item.time)]
                )
                / denominator
                / config.noise_standard_deviation
                for item in observations.observations
            )
        )
    cold_bias_column = tuple(
        (1.0 if item.sensor_name == ACCESSIBLE_SENSOR_NAMES[0] else 0.0)
        / config.noise_standard_deviation
        for item in observations.observations
    )
    hot_bias_column = tuple(
        (1.0 if item.sensor_name == ACCESSIBLE_SENSOR_NAMES[1] else 0.0)
        / config.noise_standard_deviation
        for item in observations.observations
    )
    full_columns = (*raw_physical_columns, cold_bias_column, hot_bias_column)
    full_jacobian = tuple(
        tuple(column[row] for column in full_columns)
        for row in range(len(observations.observations))
    )
    information = gram_matrix(full_jacobian)
    try:
        covariance, _ = inverse_and_determinant(information)
    except ValueError:
        covariance = tuple(
            tuple(math.inf if row == column else math.nan for column in range(5))
            for row in range(5)
        )
    standard_errors = tuple(
        math.sqrt(max(0.0, covariance[index][index]))
        for index in range(5)
    )
    estimates = (*physical_values, cold_bias, hot_bias)
    names = (*JOINT_PARAMETER_NAMES, "cold_sensor_bias_K", "hot_sensor_bias_K")
    intervals = []
    for index, (name, estimate, standard_error) in enumerate(
        zip(names, estimates, standard_errors)
    ):
        if index < 3:
            lower = estimate * math.exp(-1.96 * standard_error)
            upper = estimate * math.exp(1.96 * standard_error)
        else:
            lower = estimate - 1.96 * standard_error
            upper = estimate + 1.96 * standard_error
        intervals.append(
            JointThermalInterval(name, estimate, standard_error, lower, upper)
        )
    correlation = tuple(
        tuple(
            covariance[row][column]
            / (standard_errors[row] * standard_errors[column])
            if standard_errors[row] > 0.0 and standard_errors[column] > 0.0
            else math.nan
            for column in range(5)
        )
        for row in range(5)
    )
    reached_bound = any(
        abs(best[index] - lower_logs[index]) <= 1.0e-5
        or abs(upper_logs[index] - best[index]) <= 1.0e-5
        for index in range(3)
    )
    return JointThermalFitResult(
        log_multipliers=best,
        physical_values=physical_values,
        cold_sensor_bias=cold_bias,
        hot_sensor_bias=hot_bias,
        observation_rmse=(
            config.noise_standard_deviation
            * math.sqrt(sum(value * value for value in residuals) / len(residuals))
        ),
        normalized_mean_squared_error=objective(best),
        covariance=covariance,
        intervals=tuple(intervals),
        correlation=correlation,
        identifiability=identifiability,
        reached_bound=reached_bound,
        evaluations=tuple(evaluations),
    )


def analyze_joint_thermal_identifiability(
    current: PiecewiseConstantCurrent,
    config: JointThermalFitConfig = JointThermalFitConfig(),
) -> JointThermalIdentifiability:
    """Return the nuisance-profiled local spectrum before collecting data."""

    truth = JointThermalTruth(
        cold_contact_resistance=config.nominal_values[0],
        cold_face_thermal_capacitance=config.nominal_values[1],
        sensor_time_constant=config.nominal_values[2],
        cold_sensor_bias=0.0,
        hot_sensor_bias=0.0,
    )
    observations = generate_joint_thermal_observations(
        current,
        truth,
        config,
        noise_seed=None,
    )
    result = fit_joint_thermal_parameters(
        current,
        observations,
        JointThermalFitConfig(
            nominal_values=config.nominal_values,
            value_bounds=config.value_bounds,
            sampling_interval=config.sampling_interval,
            dense_time_step=config.dense_time_step,
            noise_standard_deviation=config.noise_standard_deviation,
            finite_difference_step=config.finite_difference_step,
            gauss_newton_iterations=1,
            initial_damping=config.initial_damping,
            initial_log_multipliers=((0.0, 0.0, 0.0),),
        ),
    )
    return result.identifiability
