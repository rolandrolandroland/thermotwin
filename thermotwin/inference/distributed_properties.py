"""Conventional spline-coefficient fitting for the distributed reference model."""

from dataclasses import dataclass, replace
import math
from typing import NamedTuple, Optional, Sequence, Tuple

from ..observations.distributed import (
    DistributedObservationSet,
    DistributedObservationChannels,
    run_distributed_virtual_experiment,
)
from ..physics.distributed import PiecewiseLinearProperty
from ..simulation.distributed import DistributedLegExperiment
from ..numerics.matrices import inverse_and_determinant
from .distributed_regularization import (
    mean_square_magnitude,
    second_difference_roughness,
)


@dataclass(frozen=True)
class DistributedPropertyFitConfig:
    """Bounded derivative-free settings for one property function at a time."""

    property_name: str
    observation_interval: float = 0.1
    channels: DistributedObservationChannels = DistributedObservationChannels()
    initial_log_multipliers: Tuple[float, ...] = ()
    fixed_log_multipliers: Tuple[Optional[float], ...] = ()
    log_multiplier_bounds: Tuple[float, float] = (-0.5, 0.5)
    coordinate_passes: int = 3
    golden_section_iterations: int = 10
    gauss_newton_iterations: int = 6
    gauss_newton_step: float = 0.005
    gauss_newton_damping: float = 1.0e-3
    temperature_standard_deviation: float = 0.01
    voltage_standard_deviation: float = 1.0e-5
    heat_rate_standard_deviation: float = 5.0e-4
    smoothness_weight: float = 0.0
    shrinkage_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.property_name not in {
            "seebeck_coefficient",
            "electrical_resistivity",
            "thermal_conductivity",
        }:
            raise ValueError("unknown distributed material property")
        lower, upper = self.log_multiplier_bounds
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("log-multiplier bounds must be finite and ordered")
        for name, value in (
            ("observation interval", self.observation_interval),
            ("temperature standard deviation", self.temperature_standard_deviation),
            ("voltage standard deviation", self.voltage_standard_deviation),
            ("heat-rate standard deviation", self.heat_rate_standard_deviation),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (
            ("coordinate passes", self.coordinate_passes),
            ("golden-section iterations", self.golden_section_iterations),
            ("Gauss-Newton iterations", self.gauss_newton_iterations),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if any(not math.isfinite(value) for value in self.initial_log_multipliers):
            raise ValueError("initial log multipliers must be finite")
        if any(
            value is not None and not math.isfinite(value)
            for value in self.fixed_log_multipliers
        ):
            raise ValueError("fixed log multipliers must be finite or None")
        for name, value in (
            ("Gauss-Newton step", self.gauss_newton_step),
            ("Gauss-Newton damping", self.gauss_newton_damping),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.smoothness_weight) or self.smoothness_weight < 0.0:
            raise ValueError("smoothness weight must be finite and nonnegative")
        if not math.isfinite(self.shrinkage_weight) or self.shrinkage_weight < 0.0:
            raise ValueError("shrinkage weight must be finite and nonnegative")

    def scale(self, channel: str) -> float:
        if channel.endswith("temperature"):
            return self.temperature_standard_deviation
        if channel == "voltage":
            return self.voltage_standard_deviation
        if channel.endswith("heat"):
            return self.heat_rate_standard_deviation
        raise ValueError(f"unknown observation channel: {channel}")


class DistributedPropertyFitEvaluation(NamedTuple):
    log_multipliers: Tuple[float, ...]
    mean_normalized_squared_error: float


class DistributedPropertyFitResult(NamedTuple):
    property_name: str
    temperatures: Tuple[float, ...]
    baseline_values: Tuple[float, ...]
    fitted_values: Tuple[float, ...]
    log_multipliers: Tuple[float, ...]
    mean_normalized_squared_error: float
    evaluations: Tuple[DistributedPropertyFitEvaluation, ...]


def _replace_property(
    experiment: DistributedLegExperiment,
    property_name: str,
    baseline: PiecewiseLinearProperty,
    log_multipliers: Sequence[float],
) -> DistributedLegExperiment:
    values = tuple(
        value * math.exp(offset)
        for value, offset in zip(baseline.values, log_multipliers)
    )
    return replace(
        experiment,
        material=replace(
            experiment.material,
            **{property_name: baseline.with_values(values)},
        ),
    )


def fit_distributed_property(
    experiments: Sequence[DistributedLegExperiment],
    observations: Sequence[DistributedObservationSet],
    config: DistributedPropertyFitConfig,
) -> DistributedPropertyFitResult:
    """Fit one piecewise-linear transport function with coordinate minimization.

    This conventional estimator is intentionally retained as the baseline the
    inverse PINN must beat or justify. It does not use a coefficient grid, so a
    synthetic truth cannot be recovered exactly merely because it is a grid
    node.
    """

    experiments = tuple(experiments)
    observations = tuple(observations)
    if not experiments or len(experiments) != len(observations):
        raise ValueError("experiments and observations must have equal nonzero length")
    baseline = getattr(experiments[0].material, config.property_name)
    if not isinstance(baseline, PiecewiseLinearProperty):
        raise TypeError("fitted property must be PiecewiseLinearProperty")
    for experiment in experiments[1:]:
        if getattr(experiment.material, config.property_name) != baseline:
            raise ValueError("all experiments must share the same baseline property")
    coefficient_count = len(baseline.values)
    if config.initial_log_multipliers:
        if len(config.initial_log_multipliers) != coefficient_count:
            raise ValueError("initial multiplier count must match property coefficients")
        current = list(config.initial_log_multipliers)
    else:
        current = [0.0] * coefficient_count
    lower, upper = config.log_multiplier_bounds
    if config.fixed_log_multipliers:
        if len(config.fixed_log_multipliers) != coefficient_count:
            raise ValueError("fixed multiplier count must match property coefficients")
        fixed = tuple(config.fixed_log_multipliers)
    else:
        fixed = (None,) * coefficient_count
    for index, value in enumerate(fixed):
        if value is not None:
            current[index] = value
    if any(value < lower or value > upper for value in current):
        raise ValueError("initial and fixed multipliers must lie within their bounds")
    free_indices = tuple(
        index for index, value in enumerate(fixed) if value is None
    )

    expected_keys = tuple(dataset.keys() for dataset in observations)
    expected_values = tuple(dataset.values() for dataset in observations)
    evaluations = []
    data_loss_cache = {}
    objective_cache = {}
    residual_cache = {}

    def normalized_residuals(offsets: Sequence[float]) -> Tuple[float, ...]:
        key = tuple(float(value) for value in offsets)
        if key in residual_cache:
            return residual_cache[key]
        squared = []
        residuals = []
        for experiment, target_keys, target_values in zip(
            experiments, expected_keys, expected_values
        ):
            prediction = run_distributed_virtual_experiment(
                _replace_property(
                    experiment, config.property_name, baseline, key
                ),
                observation_interval=config.observation_interval,
                channels=config.channels,
            )
            if prediction.keys() != target_keys:
                raise ValueError("predicted and observed measurement keys differ")
            experiment_residuals = tuple(
                (predicted - observed) / config.scale(channel)
                for predicted, observed, (_, channel) in zip(
                    prediction.values(), target_values, target_keys
                )
            )
            residuals.extend(experiment_residuals)
            squared.extend(value * value for value in experiment_residuals)
        loss = sum(squared) / len(squared)
        data_loss_cache[key] = loss
        residual_cache[key] = tuple(residuals)
        evaluations.append(DistributedPropertyFitEvaluation(key, loss))
        return tuple(residuals)

    def objective(offsets: Sequence[float]) -> float:
        key = tuple(float(value) for value in offsets)
        if key not in data_loss_cache:
            normalized_residuals(key)
        if key not in objective_cache:
            objective_cache[key] = (
                data_loss_cache[key]
                + config.smoothness_weight * second_difference_roughness(key)
                + config.shrinkage_weight * mean_square_magnitude(key)
            )
        return objective_cache[key]

    def optimization_residuals(offsets: Sequence[float]) -> Tuple[float, ...]:
        """Return data residuals plus pseudo-residuals for the shared prior."""

        data = normalized_residuals(offsets)
        values = tuple(float(value) for value in offsets)
        residuals = list(data)
        if config.smoothness_weight > 0.0 and coefficient_count >= 3:
            difference_count = coefficient_count - 2
            scale = math.sqrt(
                len(data) * config.smoothness_weight / difference_count
            )
            residuals.extend(
                scale
                * (values[index + 2] - 2.0 * values[index + 1] + values[index])
                for index in range(difference_count)
            )
        if config.shrinkage_weight > 0.0:
            scale = math.sqrt(
                len(data) * config.shrinkage_weight / coefficient_count
            )
            residuals.extend(scale * value for value in values)
        return tuple(residuals)

    objective(current)
    golden_ratio = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(config.coordinate_passes):
        for index in free_indices:
            left = lower
            right = upper
            x1 = right - golden_ratio * (right - left)
            x2 = left + golden_ratio * (right - left)

            def coordinate_loss(value: float) -> float:
                candidate = list(current)
                candidate[index] = value
                return objective(candidate)

            f1 = coordinate_loss(x1)
            f2 = coordinate_loss(x2)
            for _ in range(config.golden_section_iterations):
                if f1 <= f2:
                    right = x2
                    x2 = x1
                    f2 = f1
                    x1 = right - golden_ratio * (right - left)
                    f1 = coordinate_loss(x1)
                else:
                    left = x1
                    x1 = x2
                    f1 = f2
                    x2 = left + golden_ratio * (right - left)
                    f2 = coordinate_loss(x2)
            current[index] = 0.5 * (left + right)
        objective(current)

    damping = config.gauss_newton_damping
    for _ in range(config.gauss_newton_iterations):
        if not free_indices:
            break
        residual = optimization_residuals(current)
        derivative_columns = []
        for index in free_indices:
            minus = list(current)
            plus = list(current)
            minus[index] = max(lower, minus[index] - config.gauss_newton_step)
            plus[index] = min(upper, plus[index] + config.gauss_newton_step)
            denominator = plus[index] - minus[index]
            if denominator <= 0.0:
                derivative_columns.append((0.0,) * len(residual))
                continue
            minus_residual = optimization_residuals(minus)
            plus_residual = optimization_residuals(plus)
            derivative_columns.append(
                tuple(
                    (right - left) / denominator
                    for left, right in zip(minus_residual, plus_residual)
                )
            )
        free_count = len(free_indices)
        normal = tuple(
            tuple(
                sum(a * b for a, b in zip(derivative_columns[row], derivative_columns[column]))
                for column in range(free_count)
            )
            for row in range(free_count)
        )
        gradient = tuple(
            sum(value * error for value, error in zip(column, residual))
            for column in derivative_columns
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
        starting_loss = objective(current)
        accepted = False
        for fraction in (1.0, 0.5, 0.25, 0.1):
            candidate_values = list(current)
            for index, step in zip(free_indices, update):
                candidate_values[index] = min(
                    upper, max(lower, current[index] + fraction * step)
                )
            candidate = tuple(candidate_values)
            if objective(candidate) < starting_loss:
                current = list(candidate)
                damping = max(config.gauss_newton_damping * 1.0e-3, damping * 0.3)
                accepted = True
                break
        if not accepted:
            damping *= 10.0

    fitted_values = tuple(
        value * math.exp(offset)
        for value, offset in zip(baseline.values, current)
    )
    objective(current)
    final_loss = data_loss_cache[tuple(float(value) for value in current)]
    return DistributedPropertyFitResult(
        property_name=config.property_name,
        temperatures=baseline.temperatures,
        baseline_values=baseline.values,
        fitted_values=fitted_values,
        log_multipliers=tuple(current),
        mean_normalized_squared_error=final_loss,
        evaluations=tuple(evaluations),
    )
