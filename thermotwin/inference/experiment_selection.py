"""Select a feasible current pulse expected to reduce parameter uncertainty."""

from dataclasses import dataclass
import math
import random
from typing import NamedTuple, Sequence, Tuple

from ..design.control_comparison import piecewise_electrical_energy
from ..simulation.four_node_experiments import constant_current_contact_reference_experiment
from ..core.controls import PiecewiseConstantCurrent
from ..numerics.matrices import (
    gram_matrix,
    inverse_and_determinant,
    matrix_add,
    matrix_multiply,
    transpose,
)
from .sparse_sensors import simulate_accessible_observations


@dataclass(frozen=True)
class ExperimentSelectionConfig:
    """Candidate grid, nominal parameters, prior, and feasibility limits."""

    current_amplitudes: Tuple[float, ...] = (0.4, 0.6, 0.8, 1.0, 1.2)
    pulse_durations: Tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 30.0)
    maximum_electrical_energy: float = 30.0
    minimum_cold_face_temperature: float = 285.0
    maximum_hot_face_temperature: float = 315.0
    nominal_cold_contact_resistance: float = 0.25
    nominal_cold_face_capacitance: float = 50.0
    nominal_sensor_lag: float = 1.5
    sampling_interval: float = 1.0
    dense_time_step: float = 0.2
    noise_standard_deviation: float = 0.02
    prior_standard_deviations: Tuple[float, ...] = (
        0.20,
        0.50,
        0.30,
        0.10,
        0.10,
    )
    monte_carlo_trials: int = 250
    random_seed: int = 2030

    def __post_init__(self) -> None:
        for name, values in (
            ("current amplitudes", self.current_amplitudes),
            ("pulse durations", self.pulse_durations),
            ("prior standard deviations", self.prior_standard_deviations),
        ):
            if not values or any(
                not math.isfinite(value) or value <= 0.0 for value in values
            ):
                raise ValueError(f"{name} must be finite and positive")
        if len(self.prior_standard_deviations) != 5:
            raise ValueError("five prior standard deviations are required")
        for name, value in (
            ("maximum electrical energy", self.maximum_electrical_energy),
            ("nominal cold contact resistance", self.nominal_cold_contact_resistance),
            ("nominal cold face capacitance", self.nominal_cold_face_capacitance),
            ("nominal sensor lag", self.nominal_sensor_lag),
            ("sampling interval", self.sampling_interval),
            ("dense time step", self.dense_time_step),
            ("noise standard deviation", self.noise_standard_deviation),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.maximum_hot_face_temperature <= self.minimum_cold_face_temperature:
            raise ValueError("temperature limits must be ordered")
        if not isinstance(self.monte_carlo_trials, int) or self.monte_carlo_trials <= 0:
            raise ValueError("Monte Carlo trials must be a positive integer")
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise ValueError("random seed must be an integer")


class ExperimentCandidateScore(NamedTuple):
    """Feasibility and local posterior metric for one pulse."""

    name: str
    current_amplitude: float
    pulse_duration: float
    electrical_energy: float
    minimum_cold_face_temperature: float
    maximum_hot_face_temperature: float
    feasible: bool
    information_gain_nats: float
    resistance_log_standard_error: float
    capacitance_log_standard_error: float
    lag_log_standard_error: float
    resistance_capacitance_correlation: float
    physical_covariance_determinant: float


class ExperimentSelectionValidation(NamedTuple):
    """Linearized repeated-noise comparison against a naive pulse."""

    trial_count: int
    selected_log_parameter_rmse: float
    naive_log_parameter_rmse: float
    rmse_reduction_percent: float
    selected_nominal_95_coverage: float
    naive_nominal_95_coverage: float


class ExperimentSelectionResult(NamedTuple):
    candidates: Tuple[ExperimentCandidateScore, ...]
    selected: ExperimentCandidateScore
    naive: ExperimentCandidateScore
    validation: ExperimentSelectionValidation


def candidate_current(
    current_amplitude: float,
    pulse_duration: float,
) -> PiecewiseConstantCurrent:
    return PiecewiseConstantCurrent.pulse(
        start_time=5.0,
        end_time=5.0 + pulse_duration,
        pulse_current=current_amplitude,
    )


def _prediction_vector(dataset) -> Tuple[float, ...]:
    return tuple(item.temperature for item in dataset.observations)


def _candidate_jacobian(
    current: PiecewiseConstantCurrent,
    config: ExperimentSelectionConfig,
) -> Tuple[Tuple[float, ...], ...]:
    log_step = 0.01
    nominal = (
        config.nominal_cold_contact_resistance,
        config.nominal_cold_face_capacitance,
        config.nominal_sensor_lag,
    )

    def vector(resistance: float, capacitance: float, lag: float) -> Tuple[float, ...]:
        dataset, _ = simulate_accessible_observations(
            current,
            cold_contact_resistance=resistance,
            cold_face_thermal_capacitance=capacitance,
            sensor_time_constant=lag,
            sampling_interval=config.sampling_interval,
            dense_time_step=config.dense_time_step,
        )
        return _prediction_vector(dataset)

    derivatives = []
    for parameter_index in range(3):
        minus = list(nominal)
        plus = list(nominal)
        minus[parameter_index] *= math.exp(-log_step)
        plus[parameter_index] *= math.exp(log_step)
        minus_vector = vector(*minus)
        plus_vector = vector(*plus)
        derivatives.append(
            tuple(
                (right - left) / (2.0 * log_step)
                for left, right in zip(minus_vector, plus_vector)
            )
        )
    baseline, _ = simulate_accessible_observations(
        current,
        cold_contact_resistance=nominal[0],
        cold_face_thermal_capacitance=nominal[1],
        sensor_time_constant=nominal[2],
        sampling_interval=config.sampling_interval,
        dense_time_step=config.dense_time_step,
    )
    jacobian = []
    for index, observation in enumerate(baseline.observations):
        jacobian.append(
            (
                derivatives[0][index],
                derivatives[1][index],
                derivatives[2][index],
                1.0 if observation.sensor_name == "cold_exchanger_sensor" else 0.0,
                1.0 if observation.sensor_name == "hot_exchanger_sensor" else 0.0,
            )
        )
    return tuple(jacobian)


def _information_and_covariance(
    jacobian: Sequence[Sequence[float]],
    config: ExperimentSelectionConfig,
):
    data_gram = gram_matrix(jacobian)
    data_information = tuple(
        tuple(
            value / (config.noise_standard_deviation ** 2)
            for value in row
        )
        for row in data_gram
    )
    prior_information = tuple(
        tuple(
            1.0 / (config.prior_standard_deviations[row] ** 2)
            if row == column
            else 0.0
            for column in range(5)
        )
        for row in range(5)
    )
    posterior_information = matrix_add(data_information, prior_information)
    posterior_covariance, _ = inverse_and_determinant(posterior_information)
    return data_information, posterior_covariance


def _physical_covariance_determinant(covariance) -> float:
    physical = tuple(tuple(row[:3]) for row in covariance[:3])
    _, determinant = inverse_and_determinant(physical)
    return determinant


def _candidate_energy_and_limits(
    current: PiecewiseConstantCurrent,
    config: ExperimentSelectionConfig,
):
    dataset, trajectory = simulate_accessible_observations(
        current,
        cold_contact_resistance=config.nominal_cold_contact_resistance,
        cold_face_thermal_capacitance=config.nominal_cold_face_capacitance,
        sensor_time_constant=config.nominal_sensor_lag,
        sampling_interval=config.sampling_interval,
        dense_time_step=config.dense_time_step,
    )
    del dataset
    reference = constant_current_contact_reference_experiment()
    energy = piecewise_electrical_energy(
        trajectory.time,
        trajectory.cold_face,
        trajectory.hot_face,
        reference.thermoelectric_parameters,
        current,
        start_time=trajectory.time[0],
        end_time=trajectory.time[-1],
    )
    return energy, min(trajectory.cold_face), max(trajectory.hot_face)


def score_experiment_candidate(
    current_amplitude: float,
    pulse_duration: float,
    config: ExperimentSelectionConfig,
) -> Tuple[ExperimentCandidateScore, Tuple[Tuple[float, ...], ...]]:
    current = candidate_current(current_amplitude, pulse_duration)
    jacobian = _candidate_jacobian(current, config)
    _, posterior_covariance = _information_and_covariance(jacobian, config)
    physical_determinant = _physical_covariance_determinant(posterior_covariance)
    prior_determinant = math.prod(
        value * value for value in config.prior_standard_deviations[:3]
    )
    information_gain = 0.5 * math.log(prior_determinant / physical_determinant)
    standard_errors = tuple(
        math.sqrt(posterior_covariance[index][index]) for index in range(3)
    )
    correlation = posterior_covariance[0][1] / (
        standard_errors[0] * standard_errors[1]
    )
    energy, minimum_cold, maximum_hot = _candidate_energy_and_limits(
        current, config
    )
    feasible = (
        energy <= config.maximum_electrical_energy
        and minimum_cold >= config.minimum_cold_face_temperature
        and maximum_hot <= config.maximum_hot_face_temperature
    )
    score = ExperimentCandidateScore(
        name=f"{current_amplitude:.1f}A_{pulse_duration:.0f}s",
        current_amplitude=current_amplitude,
        pulse_duration=pulse_duration,
        electrical_energy=energy,
        minimum_cold_face_temperature=minimum_cold,
        maximum_hot_face_temperature=maximum_hot,
        feasible=feasible,
        information_gain_nats=information_gain,
        resistance_log_standard_error=standard_errors[0],
        capacitance_log_standard_error=standard_errors[1],
        lag_log_standard_error=standard_errors[2],
        resistance_capacitance_correlation=correlation,
        physical_covariance_determinant=physical_determinant,
    )
    return score, jacobian


def _linearized_noise_validation(
    selected_jacobian,
    naive_jacobian,
    config: ExperimentSelectionConfig,
) -> ExperimentSelectionValidation:
    random_source = random.Random(config.random_seed)

    def estimator(jacobian):
        data_information, _ = _information_and_covariance(jacobian, config)
        covariance, _ = inverse_and_determinant(data_information)
        gain = matrix_multiply(
            covariance,
            tuple(
                tuple(
                    value / (config.noise_standard_deviation ** 2)
                    for value in row
                )
                for row in transpose(jacobian)
            ),
        )
        standard_errors = tuple(
            math.sqrt(covariance[index][index]) for index in range(3)
        )
        return gain, standard_errors

    selected_gain, selected_standard_errors = estimator(selected_jacobian)
    naive_gain, naive_standard_errors = estimator(naive_jacobian)
    selected_squared_errors = []
    naive_squared_errors = []
    selected_covered = 0
    naive_covered = 0
    total_coverage_checks = config.monte_carlo_trials * 3
    observation_count = len(selected_jacobian)
    if len(naive_jacobian) != observation_count:
        raise ValueError("candidate observation counts must match")

    for _ in range(config.monte_carlo_trials):
        noise = tuple(
            random_source.gauss(0.0, config.noise_standard_deviation)
            for _ in range(observation_count)
        )
        selected_error = tuple(
            sum(weight * value for weight, value in zip(row, noise))
            for row in selected_gain
        )
        naive_error = tuple(
            sum(weight * value for weight, value in zip(row, noise))
            for row in naive_gain
        )
        selected_squared_errors.extend(value * value for value in selected_error[:3])
        naive_squared_errors.extend(value * value for value in naive_error[:3])
        selected_covered += sum(
            abs(value) <= 1.96 * standard_error
            for value, standard_error in zip(
                selected_error[:3], selected_standard_errors
            )
        )
        naive_covered += sum(
            abs(value) <= 1.96 * standard_error
            for value, standard_error in zip(
                naive_error[:3], naive_standard_errors
            )
        )
    selected_rmse = math.sqrt(
        sum(selected_squared_errors) / len(selected_squared_errors)
    )
    naive_rmse = math.sqrt(sum(naive_squared_errors) / len(naive_squared_errors))
    return ExperimentSelectionValidation(
        trial_count=config.monte_carlo_trials,
        selected_log_parameter_rmse=selected_rmse,
        naive_log_parameter_rmse=naive_rmse,
        rmse_reduction_percent=100.0 * (1.0 - selected_rmse / naive_rmse),
        selected_nominal_95_coverage=selected_covered / total_coverage_checks,
        naive_nominal_95_coverage=naive_covered / total_coverage_checks,
    )


def run_next_experiment_selection(
    config: ExperimentSelectionConfig = ExperimentSelectionConfig(),
) -> ExperimentSelectionResult:
    scores = []
    jacobians = {}
    for current_amplitude in config.current_amplitudes:
        for pulse_duration in config.pulse_durations:
            score, jacobian = score_experiment_candidate(
                current_amplitude, pulse_duration, config
            )
            scores.append(score)
            jacobians[score.name] = jacobian
    feasible = tuple(score for score in scores if score.feasible)
    if not feasible:
        raise ValueError("no candidate satisfies the experiment constraints")
    selected = max(feasible, key=lambda score: score.information_gain_nats)
    naive = min(
        feasible,
        key=lambda score: (score.electrical_energy, score.current_amplitude),
    )
    validation = _linearized_noise_validation(
        jacobians[selected.name], jacobians[naive.name], config
    )
    return ExperimentSelectionResult(
        candidates=tuple(scores),
        selected=selected,
        naive=naive,
        validation=validation,
    )


def format_experiment_selection_report(result: ExperimentSelectionResult) -> str:
    selected = result.selected
    naive = result.naive
    validation = result.validation
    feasible_count = sum(candidate.feasible for candidate in result.candidates)
    lines = [
        "ThermoTwin next-experiment selection",
        "objective: maximum expected joint information under a 30 J budget",
        f"candidates: {len(result.candidates)} total, {feasible_count} feasible",
        (
            f"selected: {selected.current_amplitude:.1f} A for "
            f"{selected.pulse_duration:.0f} s, energy={selected.electrical_energy:.2f} J"
        ),
        f"selected information gain: {selected.information_gain_nats:.3f} nats",
        (
            f"selected log-standard-errors: R_c {selected.resistance_log_standard_error:.4f}, "
            f"C_cf {selected.capacitance_log_standard_error:.4f}, "
            f"lag {selected.lag_log_standard_error:.4f}"
        ),
        (
            f"naive: {naive.current_amplitude:.1f} A for {naive.pulse_duration:.0f} s, "
            f"information gain={naive.information_gain_nats:.3f} nats"
        ),
        (
            f"{validation.trial_count}-trial linearized noise validation: "
            f"log-parameter RMSE reduction={validation.rmse_reduction_percent:.1f}%"
        ),
        (
            f"nominal 95% coverage: selected {validation.selected_nominal_95_coverage:.3f}, "
            f"naive {validation.naive_nominal_95_coverage:.3f}"
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    print(format_experiment_selection_report(run_next_experiment_selection()))


if __name__ == "__main__":
    main()
