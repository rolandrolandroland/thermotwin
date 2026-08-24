"""Joint inference from realistically accessible exchanger sensors.

The synthetic learning problem exposes only cold- and hot-exchanger
temperatures. Cold contact resistance and a shared first-order sensor lag are
fit jointly; each sensor bias is profiled out analytically. The inferred
physical model then reconstructs the inaccessible thermoelectric-face states
and is transferred to a withheld current schedule.
"""

from dataclasses import dataclass, replace
import math
from typing import Dict, NamedTuple, Optional, Sequence, Tuple

from ..simulation.four_node_experiments import (
    FourNodeContactExperiment,
    constant_current_contact_reference_experiment,
    run_four_node_contact_experiment,
)
from ..physics.four_node import FourNodeContactTemperatureTrajectory
from ..core.controls import PiecewiseConstantCurrent
from ..observations.bias import FixedTemperatureBias, apply_fixed_temperature_bias
from ..observations.lag import FirstOrderTemperatureLag, apply_first_order_temperature_lag
from ..observations.missingness import (
    DeterministicTemperatureMissingness,
    TemperatureSensorOutage,
    apply_deterministic_temperature_missingness,
)
from ..observations.noise import GaussianTemperatureNoise, apply_gaussian_temperature_noise
from ..numerics.matrices import gram_matrix, inverse_and_determinant
from ..observations.test_stand import (
    IdealTemperatureSensor,
    IdealVirtualTestStand,
    ObservationDataset,
    TemperatureObservation,
    TemperatureSensorLocation,
    observe_contact_trajectory,
    regular_measurement_times,
)


ACCESSIBLE_SENSOR_NAMES = (
    "cold_exchanger_sensor",
    "hot_exchanger_sensor",
)


@dataclass(frozen=True)
class SparseSensorInferenceConfig:
    """Truth, observation imperfections, and coarse-to-fine search settings."""

    true_cold_contact_resistance: float = 0.25
    true_sensor_time_constant: float = 1.5
    true_cold_sensor_bias: float = 0.08
    true_hot_sensor_bias: float = -0.04
    noise_standard_deviation: float = 0.02
    random_seed: int = 2026
    sampling_interval: float = 1.0
    dense_lag_time_step: float = 0.2
    resistance_bounds: Tuple[float, float] = (0.10, 0.50)
    lag_bounds: Tuple[float, float] = (0.20, 4.0)
    grid_points_per_axis: int = 9
    refinement_count: int = 3
    local_polish_iterations: int = 18

    def __post_init__(self) -> None:
        positive = (
            ("true cold contact resistance", self.true_cold_contact_resistance),
            ("true sensor time constant", self.true_sensor_time_constant),
            ("sampling interval", self.sampling_interval),
            ("dense lag time step", self.dense_lag_time_step),
        )
        for name, value in positive:
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.noise_standard_deviation)
            or self.noise_standard_deviation <= 0.0
        ):
            raise ValueError("noise standard deviation must be finite and positive")
        if not all(
            math.isfinite(value)
            for value in (self.true_cold_sensor_bias, self.true_hot_sensor_bias)
        ):
            raise ValueError("sensor biases must be finite")
        for name, bounds in (
            ("resistance", self.resistance_bounds),
            ("lag", self.lag_bounds),
        ):
            if (
                len(bounds) != 2
                or not all(math.isfinite(value) and value > 0.0 for value in bounds)
                or bounds[1] <= bounds[0]
            ):
                raise ValueError(f"{name} bounds must be positive and ordered")
        if (
            not isinstance(self.grid_points_per_axis, int)
            or self.grid_points_per_axis < 3
            or not isinstance(self.refinement_count, int)
            or self.refinement_count < 1
            or not isinstance(self.local_polish_iterations, int)
            or self.local_polish_iterations < 1
        ):
            raise ValueError("search grid, refinement, and polish counts are invalid")


class SparseSensorProblem(NamedTuple):
    """Synthetic observations with dense truth kept separate for validation."""

    experiment: FourNodeContactExperiment
    observations: ObservationDataset
    truth_trajectory: FourNodeContactTemperatureTrajectory
    missingness: DeterministicTemperatureMissingness
    config: SparseSensorInferenceConfig


class SparseSensorLossEvaluation(NamedTuple):
    cold_contact_resistance: float
    sensor_time_constant: float
    cold_sensor_bias: float
    hot_sensor_bias: float
    mean_squared_error: float


class SparseSensorFitResult(NamedTuple):
    inferred_cold_contact_resistance: float
    inferred_sensor_time_constant: float
    inferred_cold_sensor_bias: float
    inferred_hot_sensor_bias: float
    observation_rmse: float
    evaluations: Tuple[SparseSensorLossEvaluation, ...]


class ParameterInterval(NamedTuple):
    name: str
    estimate: float
    standard_error: float
    lower_95: float
    upper_95: float
    contains_truth: bool


class SparseSensorUncertainty(NamedTuple):
    intervals: Tuple[ParameterInterval, ...]
    correlation: Tuple[Tuple[float, ...], ...]
    information_determinant: float


class SparseHiddenStateValidation(NamedTuple):
    cold_face_rmse: float
    hot_face_rmse: float
    maximum_cold_face_error: float
    maximum_hot_face_error: float


class SparseWithheldValidation(NamedTuple):
    accessible_sensor_rmse: float
    cold_face_rmse: float
    hot_face_rmse: float


class SparseSensorInferenceResult(NamedTuple):
    problem: SparseSensorProblem
    fit: SparseSensorFitResult
    uncertainty: SparseSensorUncertainty
    hidden_state_validation: SparseHiddenStateValidation
    withheld_validation: SparseWithheldValidation


def accessible_exchanger_test_stand(
    *, sampling_interval: float
) -> IdealVirtualTestStand:
    return IdealVirtualTestStand(
        sensors=(
            IdealTemperatureSensor(
                ACCESSIBLE_SENSOR_NAMES[0],
                TemperatureSensorLocation.COLD_EXCHANGER,
            ),
            IdealTemperatureSensor(
                ACCESSIBLE_SENSOR_NAMES[1],
                TemperatureSensorLocation.HOT_EXCHANGER,
            ),
        ),
        sampling_interval=sampling_interval,
    )


def sparse_training_current() -> PiecewiseConstantCurrent:
    """Two-level excitation with a turn-off interval for joint inference."""

    return PiecewiseConstantCurrent(
        transition_times=(5.0, 20.0, 35.0, 50.0),
        values=(0.0, 1.0, 0.0, 0.55, 0.0),
    )


def sparse_withheld_current() -> PiecewiseConstantCurrent:
    return PiecewiseConstantCurrent(
        transition_times=(8.0, 28.0, 42.0, 58.0),
        values=(0.0, 0.75, 0.0, -0.45, 0.0),
    )


def sparse_sensor_experiment(
    current: PiecewiseConstantCurrent,
    *,
    cold_contact_resistance: float,
    dense_time_step: float,
    cold_face_thermal_capacitance: Optional[float] = None,
) -> FourNodeContactExperiment:
    reference = constant_current_contact_reference_experiment()
    thermal_parameters = replace(
        reference.thermal_parameters,
        cold_contact_resistance=cold_contact_resistance,
    )
    if cold_face_thermal_capacitance is not None:
        thermal_parameters = replace(
            thermal_parameters,
            cold_face_thermal_capacitance=cold_face_thermal_capacitance,
        )
    return replace(
        reference,
        thermal_parameters=thermal_parameters,
        duration=80.0,
        time_step=dense_time_step,
        current=current,
    )


def _resample_dataset(
    dataset: ObservationDataset,
    *,
    sampling_interval: float,
) -> ObservationDataset:
    requested_times = regular_measurement_times(
        dataset.measurement_times[-1], sampling_interval
    )
    observations_by_key = {
        (round(item.time, 10), item.sensor_name): item
        for item in dataset.observations
    }
    observations = []
    for time in requested_times:
        for sensor in dataset.sensors:
            key = (round(time, 10), sensor.name)
            if key not in observations_by_key:
                raise ValueError("dense lag history does not contain requested sample")
            observations.append(observations_by_key[key])
    return ObservationDataset(
        observations=tuple(observations),
        sensors=dataset.sensors,
        sampling_interval=sampling_interval,
        time_unit=dataset.time_unit,
        temperature_unit=dataset.temperature_unit,
        current_unit=dataset.current_unit,
        provenance=dataset.provenance,
    )


def simulate_accessible_observations(
    current: PiecewiseConstantCurrent,
    *,
    cold_contact_resistance: float,
    sensor_time_constant: float,
    sampling_interval: float,
    dense_time_step: float,
    cold_face_thermal_capacitance: Optional[float] = None,
) -> Tuple[ObservationDataset, FourNodeContactTemperatureTrajectory]:
    experiment = sparse_sensor_experiment(
        current,
        cold_contact_resistance=cold_contact_resistance,
        dense_time_step=dense_time_step,
        cold_face_thermal_capacitance=cold_face_thermal_capacitance,
    )
    trajectory = run_four_node_contact_experiment(experiment).trajectory
    dense = observe_contact_trajectory(
        trajectory,
        current=current,
        test_stand=accessible_exchanger_test_stand(
            sampling_interval=dense_time_step
        ),
    )
    lagged = apply_first_order_temperature_lag(
        dense,
        FirstOrderTemperatureLag(default_time_constant=sensor_time_constant),
    ).dataset
    return (
        _resample_dataset(lagged, sampling_interval=sampling_interval),
        trajectory,
    )


def build_sparse_sensor_problem(
    config: SparseSensorInferenceConfig = SparseSensorInferenceConfig(),
) -> SparseSensorProblem:
    """Generate noisy, biased, lagged, and partly missing exchanger readings."""

    current = sparse_training_current()
    ideal_lagged, truth_trajectory = simulate_accessible_observations(
        current,
        cold_contact_resistance=config.true_cold_contact_resistance,
        sensor_time_constant=config.true_sensor_time_constant,
        sampling_interval=config.sampling_interval,
        dense_time_step=config.dense_lag_time_step,
    )
    biased = apply_fixed_temperature_bias(
        ideal_lagged,
        FixedTemperatureBias(
            sensor_biases=(
                (ACCESSIBLE_SENSOR_NAMES[0], config.true_cold_sensor_bias),
                (ACCESSIBLE_SENSOR_NAMES[1], config.true_hot_sensor_bias),
            )
        ),
    ).dataset
    noisy = apply_gaussian_temperature_noise(
        biased,
        GaussianTemperatureNoise(
            default_standard_deviation=config.noise_standard_deviation,
            random_seed=config.random_seed,
        ),
    ).dataset
    missingness = DeterministicTemperatureMissingness(
        outages=(
            TemperatureSensorOutage(
                ACCESSIBLE_SENSOR_NAMES[0],
                18.0,
                24.0,
            ),
        )
    )
    observations = apply_deterministic_temperature_missingness(
        noisy, missingness
    ).dataset
    experiment = sparse_sensor_experiment(
        current,
        cold_contact_resistance=config.true_cold_contact_resistance,
        dense_time_step=config.dense_lag_time_step,
    )
    return SparseSensorProblem(
        experiment=experiment,
        observations=observations,
        truth_trajectory=truth_trajectory,
        missingness=missingness,
        config=config,
    )


def _prediction_map(dataset: ObservationDataset) -> Dict[Tuple[str, float], float]:
    return {
        (item.sensor_name, item.time): item.temperature
        for item in dataset.observations
    }


def _profiled_loss(
    observed: ObservationDataset,
    predicted: ObservationDataset,
) -> Tuple[float, float, float]:
    predicted_map = _prediction_map(predicted)
    differences = {name: [] for name in ACCESSIBLE_SENSOR_NAMES}
    for item in observed.observations:
        key = (item.sensor_name, item.time)
        if key not in predicted_map:
            raise ValueError("prediction is missing an observed sensor-time pair")
        differences[item.sensor_name].append(item.temperature - predicted_map[key])
    biases = {
        name: sum(values) / len(values)
        for name, values in differences.items()
    }
    residuals = tuple(
        predicted_map[(item.sensor_name, item.time)]
        + biases[item.sensor_name]
        - item.temperature
        for item in observed.observations
    )
    mse = sum(value * value for value in residuals) / len(residuals)
    return mse, biases[ACCESSIBLE_SENSOR_NAMES[0]], biases[ACCESSIBLE_SENSOR_NAMES[1]]


def fit_sparse_sensor_parameters(
    problem: SparseSensorProblem,
) -> SparseSensorFitResult:
    """Jointly fit contact resistance and lag; profile two constant biases."""

    config = problem.config
    global_resistance_bounds = config.resistance_bounds
    global_lag_bounds = config.lag_bounds
    resistance_bounds = global_resistance_bounds
    lag_bounds = global_lag_bounds
    evaluations = []
    best: Optional[SparseSensorLossEvaluation] = None

    def evaluate_parameters(
        resistance: float,
        lag: float,
    ) -> SparseSensorLossEvaluation:
        predicted, _ = simulate_accessible_observations(
            problem.experiment.current,
            cold_contact_resistance=resistance,
            sensor_time_constant=lag,
            sampling_interval=config.sampling_interval,
            dense_time_step=config.dense_lag_time_step,
        )
        mse, cold_bias, hot_bias = _profiled_loss(
            problem.observations, predicted
        )
        evaluation = SparseSensorLossEvaluation(
            resistance,
            lag,
            cold_bias,
            hot_bias,
            mse,
        )
        evaluations.append(evaluation)
        return evaluation

    for _ in range(config.refinement_count):
        resistance_step = (
            resistance_bounds[1] - resistance_bounds[0]
        ) / (config.grid_points_per_axis - 1)
        lag_step = (
            lag_bounds[1] - lag_bounds[0]
        ) / (config.grid_points_per_axis - 1)
        for resistance_index in range(config.grid_points_per_axis):
            resistance = resistance_bounds[0] + resistance_index * resistance_step
            for lag_index in range(config.grid_points_per_axis):
                lag = lag_bounds[0] + lag_index * lag_step
                evaluation = evaluate_parameters(resistance, lag)
                if best is None or evaluation.mean_squared_error < best.mean_squared_error:
                    best = evaluation
        if best is None:
            raise RuntimeError("sparse search produced no evaluations")
        resistance_bounds = (
            max(global_resistance_bounds[0], best.cold_contact_resistance - resistance_step),
            min(global_resistance_bounds[1], best.cold_contact_resistance + resistance_step),
        )
        lag_bounds = (
            max(global_lag_bounds[0], best.sensor_time_constant - lag_step),
            min(global_lag_bounds[1], best.sensor_time_constant + lag_step),
        )

    # The grid supplies a global, auditable starting point. A local pattern
    # search then removes grid-node locking without using the hidden truth.
    polish_resistance_step = resistance_step
    polish_lag_step = lag_step
    directions = tuple(
        (resistance_direction, lag_direction)
        for resistance_direction in (-1.0, 0.0, 1.0)
        for lag_direction in (-1.0, 0.0, 1.0)
        if resistance_direction != 0.0 or lag_direction != 0.0
    )
    for _ in range(config.local_polish_iterations):
        candidates = []
        coordinates = set()
        for resistance_direction, lag_direction in directions:
            resistance = min(
                global_resistance_bounds[1],
                max(
                    global_resistance_bounds[0],
                    best.cold_contact_resistance
                    + resistance_direction * polish_resistance_step,
                ),
            )
            lag = min(
                global_lag_bounds[1],
                max(
                    global_lag_bounds[0],
                    best.sensor_time_constant + lag_direction * polish_lag_step,
                ),
            )
            coordinate = (resistance, lag)
            if coordinate in coordinates or coordinate == (
                best.cold_contact_resistance,
                best.sensor_time_constant,
            ):
                continue
            coordinates.add(coordinate)
            candidates.append(evaluate_parameters(resistance, lag))
        candidate_best = min(
            candidates,
            key=lambda item: item.mean_squared_error,
            default=best,
        )
        if candidate_best.mean_squared_error < best.mean_squared_error:
            best = candidate_best
        else:
            polish_resistance_step *= 0.5
            polish_lag_step *= 0.5

    return SparseSensorFitResult(
        inferred_cold_contact_resistance=best.cold_contact_resistance,
        inferred_sensor_time_constant=best.sensor_time_constant,
        inferred_cold_sensor_bias=best.cold_sensor_bias,
        inferred_hot_sensor_bias=best.hot_sensor_bias,
        observation_rmse=math.sqrt(best.mean_squared_error),
        evaluations=tuple(evaluations),
    )


def _aligned_prediction_vector(
    observations: ObservationDataset,
    prediction: ObservationDataset,
) -> Tuple[float, ...]:
    predicted_map = _prediction_map(prediction)
    return tuple(
        predicted_map[(item.sensor_name, item.time)]
        for item in observations.observations
    )


def estimate_sparse_sensor_uncertainty(
    problem: SparseSensorProblem,
    fit: SparseSensorFitResult,
) -> SparseSensorUncertainty:
    """Return local linearized 95% intervals and parameter correlations."""

    config = problem.config
    resistance = fit.inferred_cold_contact_resistance
    lag = fit.inferred_sensor_time_constant
    resistance_step = max(1e-4, 0.01 * resistance)
    lag_step = max(1e-3, 0.01 * lag)

    def vector(candidate_resistance: float, candidate_lag: float) -> Tuple[float, ...]:
        predicted, _ = simulate_accessible_observations(
            problem.experiment.current,
            cold_contact_resistance=candidate_resistance,
            sensor_time_constant=candidate_lag,
            sampling_interval=config.sampling_interval,
            dense_time_step=config.dense_lag_time_step,
        )
        return _aligned_prediction_vector(problem.observations, predicted)

    resistance_minus = vector(resistance - resistance_step, lag)
    resistance_plus = vector(resistance + resistance_step, lag)
    lag_minus = vector(resistance, max(1e-6, lag - lag_step))
    lag_plus = vector(resistance, lag + lag_step)
    lag_denominator = (lag + lag_step) - max(1e-6, lag - lag_step)
    jacobian = []
    for index, observation in enumerate(problem.observations.observations):
        jacobian.append(
            (
                (resistance_plus[index] - resistance_minus[index])
                / (2.0 * resistance_step),
                (lag_plus[index] - lag_minus[index]) / lag_denominator,
                1.0 if observation.sensor_name == ACCESSIBLE_SENSOR_NAMES[0] else 0.0,
                1.0 if observation.sensor_name == ACCESSIBLE_SENSOR_NAMES[1] else 0.0,
            )
        )
    gram = gram_matrix(jacobian)
    information = tuple(
        tuple(
            value / (config.noise_standard_deviation ** 2)
            for value in row
        )
        for row in gram
    )
    covariance, determinant = inverse_and_determinant(information)
    standard_errors = tuple(
        math.sqrt(max(0.0, covariance[index][index]))
        for index in range(4)
    )
    estimates = (
        fit.inferred_cold_contact_resistance,
        fit.inferred_sensor_time_constant,
        fit.inferred_cold_sensor_bias,
        fit.inferred_hot_sensor_bias,
    )
    truths = (
        config.true_cold_contact_resistance,
        config.true_sensor_time_constant,
        config.true_cold_sensor_bias,
        config.true_hot_sensor_bias,
    )
    names = (
        "cold_contact_resistance_K_per_W",
        "shared_sensor_lag_s",
        "cold_sensor_bias_K",
        "hot_sensor_bias_K",
    )
    intervals = []
    for name, estimate, standard_error, truth in zip(
        names, estimates, standard_errors, truths
    ):
        lower = estimate - 1.96 * standard_error
        upper = estimate + 1.96 * standard_error
        intervals.append(
            ParameterInterval(
                name,
                estimate,
                standard_error,
                lower,
                upper,
                lower <= truth <= upper,
            )
        )
    correlation = tuple(
        tuple(
            covariance[row][column]
            / (standard_errors[row] * standard_errors[column])
            for column in range(4)
        )
        for row in range(4)
    )
    return SparseSensorUncertainty(
        intervals=tuple(intervals),
        correlation=correlation,
        information_determinant=determinant,
    )


def _rmse(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("RMSE histories must be nonempty and aligned")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)) / len(left))


def _hidden_validation(
    truth: FourNodeContactTemperatureTrajectory,
    prediction: FourNodeContactTemperatureTrajectory,
) -> SparseHiddenStateValidation:
    cold_errors = tuple(a - b for a, b in zip(prediction.cold_face, truth.cold_face))
    hot_errors = tuple(a - b for a, b in zip(prediction.hot_face, truth.hot_face))
    return SparseHiddenStateValidation(
        cold_face_rmse=_rmse(prediction.cold_face, truth.cold_face),
        hot_face_rmse=_rmse(prediction.hot_face, truth.hot_face),
        maximum_cold_face_error=max(abs(value) for value in cold_errors),
        maximum_hot_face_error=max(abs(value) for value in hot_errors),
    )


def _withheld_validation(
    config: SparseSensorInferenceConfig,
    fit: SparseSensorFitResult,
) -> SparseWithheldValidation:
    current = sparse_withheld_current()
    true_dataset, true_trajectory = simulate_accessible_observations(
        current,
        cold_contact_resistance=config.true_cold_contact_resistance,
        sensor_time_constant=config.true_sensor_time_constant,
        sampling_interval=config.sampling_interval,
        dense_time_step=config.dense_lag_time_step,
    )
    true_dataset = apply_fixed_temperature_bias(
        true_dataset,
        FixedTemperatureBias(
            sensor_biases=(
                (ACCESSIBLE_SENSOR_NAMES[0], config.true_cold_sensor_bias),
                (ACCESSIBLE_SENSOR_NAMES[1], config.true_hot_sensor_bias),
            )
        ),
    ).dataset
    predicted_dataset, predicted_trajectory = simulate_accessible_observations(
        current,
        cold_contact_resistance=fit.inferred_cold_contact_resistance,
        sensor_time_constant=fit.inferred_sensor_time_constant,
        sampling_interval=config.sampling_interval,
        dense_time_step=config.dense_lag_time_step,
    )
    predicted_dataset = apply_fixed_temperature_bias(
        predicted_dataset,
        FixedTemperatureBias(
            sensor_biases=(
                (ACCESSIBLE_SENSOR_NAMES[0], fit.inferred_cold_sensor_bias),
                (ACCESSIBLE_SENSOR_NAMES[1], fit.inferred_hot_sensor_bias),
            )
        ),
    ).dataset
    true_vector = tuple(item.temperature for item in true_dataset.observations)
    predicted_vector = tuple(item.temperature for item in predicted_dataset.observations)
    return SparseWithheldValidation(
        accessible_sensor_rmse=_rmse(predicted_vector, true_vector),
        cold_face_rmse=_rmse(predicted_trajectory.cold_face, true_trajectory.cold_face),
        hot_face_rmse=_rmse(predicted_trajectory.hot_face, true_trajectory.hot_face),
    )


def run_sparse_sensor_inference_experiment(
    config: SparseSensorInferenceConfig = SparseSensorInferenceConfig(),
) -> SparseSensorInferenceResult:
    problem = build_sparse_sensor_problem(config)
    fit = fit_sparse_sensor_parameters(problem)
    uncertainty = estimate_sparse_sensor_uncertainty(problem, fit)
    fitted_experiment = sparse_sensor_experiment(
        problem.experiment.current,
        cold_contact_resistance=fit.inferred_cold_contact_resistance,
        dense_time_step=config.dense_lag_time_step,
    )
    fitted_trajectory = run_four_node_contact_experiment(fitted_experiment).trajectory
    return SparseSensorInferenceResult(
        problem=problem,
        fit=fit,
        uncertainty=uncertainty,
        hidden_state_validation=_hidden_validation(
            problem.truth_trajectory, fitted_trajectory
        ),
        withheld_validation=_withheld_validation(config, fit),
    )


def format_sparse_sensor_inference_report(
    result: SparseSensorInferenceResult,
) -> str:
    config = result.problem.config
    fit = result.fit
    lines = [
        "ThermoTwin sparse accessible-sensor inference",
        "visible sensors: cold and hot exchanger only",
        f"records: {len(result.problem.observations.observations)}",
        (
            f"cold contact resistance: {fit.inferred_cold_contact_resistance:.5f} K/W "
            f"(truth {config.true_cold_contact_resistance:.5f})"
        ),
        (
            f"shared sensor lag: {fit.inferred_sensor_time_constant:.4f} s "
            f"(truth {config.true_sensor_time_constant:.4f})"
        ),
        (
            f"sensor biases: cold {fit.inferred_cold_sensor_bias:+.4f} K, "
            f"hot {fit.inferred_hot_sensor_bias:+.4f} K"
        ),
        f"training observation RMSE: {fit.observation_rmse:.5f} K",
        (
            "hidden face RMSE: "
            f"cold {result.hidden_state_validation.cold_face_rmse:.5f} K, "
            f"hot {result.hidden_state_validation.hot_face_rmse:.5f} K"
        ),
        (
            "withheld schedule RMSE: "
            f"accessible sensors {result.withheld_validation.accessible_sensor_rmse:.5f} K, "
            f"cold face {result.withheld_validation.cold_face_rmse:.5f} K, "
            f"hot face {result.withheld_validation.hot_face_rmse:.5f} K"
        ),
        "local linearized 95% intervals:",
    ]
    for interval in result.uncertainty.intervals:
        lines.append(
            f"  {interval.name}: [{interval.lower_95:.5f}, {interval.upper_95:.5f}], "
            f"truth covered={interval.contains_truth}"
        )
    return "\n".join(lines)


def main() -> None:
    print(
        format_sparse_sensor_inference_report(
            run_sparse_sensor_inference_experiment()
        )
    )


if __name__ == "__main__":
    main()
