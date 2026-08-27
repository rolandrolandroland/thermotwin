"""D-optimal experiment selection for distributed property coefficients."""

from dataclasses import dataclass, replace
import math
from typing import NamedTuple, Sequence, Tuple

from ..core.controls import PiecewiseConstantCurrent
from ..numerics.matrices import inverse_and_determinant, matrix_add
from ..simulation.distributed import (
    DistributedLegExperiment,
    distributed_reference_experiment,
    run_distributed_leg_experiment,
)
from .distributed_identifiability import (
    DistributedIdentifiabilityConfig,
    DistributedPropertyCoefficient,
    analyze_distributed_identifiability,
)


@dataclass(frozen=True)
class DistributedExperimentSelectionConfig:
    current_amplitudes: Tuple[float, ...] = (-0.8, -0.4, 0.4, 0.8)
    pulse_durations: Tuple[float, ...] = (0.2, 0.4, 0.6)
    reservoir_temperature_lifts: Tuple[float, ...] = (0.0, 10.0, 20.0)
    duration: float = 0.8
    pulse_start_time: float = 0.1
    cell_count: int = 6
    time_step: float = 0.0015
    prior_log_standard_deviation: float = 0.20
    maximum_absolute_voltage: float = 0.05
    maximum_absolute_power: float = 0.05
    minimum_temperature: float = 275.0
    maximum_temperature: float = 325.0
    identifiability: DistributedIdentifiabilityConfig = (
        DistributedIdentifiabilityConfig(observation_interval=0.1)
    )

    def __post_init__(self) -> None:
        if not self.current_amplitudes or any(
            not math.isfinite(value) or value == 0.0
            for value in self.current_amplitudes
        ):
            raise ValueError("candidate currents must be finite and nonzero")
        if not self.pulse_durations or any(
            not math.isfinite(value) or value <= 0.0
            for value in self.pulse_durations
        ):
            raise ValueError("pulse durations must be finite and positive")
        if not self.reservoir_temperature_lifts or any(
            not math.isfinite(value) or value < 0.0
            for value in self.reservoir_temperature_lifts
        ):
            raise ValueError("temperature lifts must be finite and nonnegative")
        for name, value in (
            ("duration", self.duration),
            ("time step", self.time_step),
            ("prior log standard deviation", self.prior_log_standard_deviation),
            ("maximum absolute voltage", self.maximum_absolute_voltage),
            ("maximum absolute power", self.maximum_absolute_power),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.pulse_start_time) or self.pulse_start_time < 0.0:
            raise ValueError("pulse start time must be finite and nonnegative")
        if self.maximum_temperature <= self.minimum_temperature:
            raise ValueError("temperature limits must be ordered")
        if (
            not isinstance(self.cell_count, int)
            or isinstance(self.cell_count, bool)
            or self.cell_count < 2
        ):
            raise ValueError("cell count must be an integer of at least two")


class DistributedExperimentCandidateScore(NamedTuple):
    name: str
    current_amplitude: float
    pulse_duration: float
    reservoir_temperature_lift: float
    feasible: bool
    information_gain_nats: float
    effective_rank: int
    condition_number: float
    maximum_absolute_voltage: float
    maximum_absolute_power: float
    minimum_temperature: float
    maximum_temperature: float


class DistributedExperimentSelectionResult(NamedTuple):
    parameter_names: Tuple[str, ...]
    candidates: Tuple[DistributedExperimentCandidateScore, ...]
    selected: DistributedExperimentCandidateScore


class DistributedLinearizedUncertainty(NamedTuple):
    parameter_names: Tuple[str, ...]
    log_standard_errors: Tuple[float, ...]
    multiplier_95_intervals: Tuple[Tuple[float, float], ...]
    correlation_matrix: Tuple[Tuple[float, ...], ...]


def distributed_candidate_experiment(
    *,
    current_amplitude: float,
    pulse_duration: float,
    reservoir_temperature_lift: float,
    config: DistributedExperimentSelectionConfig,
) -> DistributedLegExperiment:
    if config.pulse_start_time + pulse_duration >= config.duration:
        raise ValueError("candidate pulse must end before experiment duration")
    cold_reservoir = 300.0 - 0.5 * reservoir_temperature_lift
    hot_reservoir = 300.0 + 0.5 * reservoir_temperature_lift
    base = distributed_reference_experiment(
        temperature_dependent=True,
        current=PiecewiseConstantCurrent.pulse(
            start_time=config.pulse_start_time,
            end_time=config.pulse_start_time + pulse_duration,
            pulse_current=current_amplitude,
        ),
        cold_reservoir_temperature=cold_reservoir,
        hot_reservoir_temperature=hot_reservoir,
        duration=config.duration,
        cell_count=config.cell_count,
        time_step=config.time_step,
    )
    return replace(
        base,
        initial_cold_face_temperature=cold_reservoir,
        initial_hot_face_temperature=hot_reservoir,
    )


def _posterior_covariance(
    information_matrix: Sequence[Sequence[float]],
    prior_log_standard_deviation: float,
):
    size = len(information_matrix)
    prior_precision = 1.0 / prior_log_standard_deviation**2
    prior = tuple(
        tuple(prior_precision if row == column else 0.0 for column in range(size))
        for row in range(size)
    )
    posterior = matrix_add(information_matrix, prior)
    covariance, determinant = inverse_and_determinant(posterior)
    prior_determinant = prior_precision**size
    information_gain = 0.5 * math.log(determinant / prior_determinant)
    return covariance, information_gain


def linearized_distributed_uncertainty(
    parameters: Sequence[DistributedPropertyCoefficient],
    information_matrix: Sequence[Sequence[float]],
    *,
    prior_log_standard_deviation: float = 0.20,
) -> DistributedLinearizedUncertainty:
    """Return local Gaussian uncertainty in log-coefficient coordinates."""

    parameters = tuple(parameters)
    if len(information_matrix) != len(parameters):
        raise ValueError("information matrix size must match parameter count")
    covariance, _ = _posterior_covariance(
        information_matrix, prior_log_standard_deviation
    )
    standard_errors = tuple(
        math.sqrt(max(0.0, covariance[index][index]))
        for index in range(len(parameters))
    )
    correlations = tuple(
        tuple(
            covariance[row][column]
            / (standard_errors[row] * standard_errors[column])
            for column in range(len(parameters))
        )
        for row in range(len(parameters))
    )
    intervals = tuple(
        (math.exp(-1.96 * value), math.exp(1.96 * value))
        for value in standard_errors
    )
    return DistributedLinearizedUncertainty(
        parameter_names=tuple(parameter.name for parameter in parameters),
        log_standard_errors=standard_errors,
        multiplier_95_intervals=intervals,
        correlation_matrix=correlations,
    )


def select_distributed_experiment(
    parameters: Sequence[DistributedPropertyCoefficient],
    config: DistributedExperimentSelectionConfig = (
        DistributedExperimentSelectionConfig()
    ),
) -> DistributedExperimentSelectionResult:
    """Choose the feasible pulse/lift candidate with maximum information gain."""

    parameters = tuple(parameters)
    if not parameters:
        raise ValueError("at least one target parameter is required")
    scores = []
    for lift in config.reservoir_temperature_lifts:
        for current in config.current_amplitudes:
            for pulse_duration in config.pulse_durations:
                if config.pulse_start_time + pulse_duration >= config.duration:
                    continue
                experiment = distributed_candidate_experiment(
                    current_amplitude=current,
                    pulse_duration=pulse_duration,
                    reservoir_temperature_lift=lift,
                    config=config,
                )
                identifiability = analyze_distributed_identifiability(
                    (experiment,), parameters, config.identifiability
                )
                _, information_gain = _posterior_covariance(
                    identifiability.information_matrix,
                    config.prior_log_standard_deviation,
                )
                result = run_distributed_leg_experiment(experiment)
                temperatures = (
                    *result.trajectory.cold_face,
                    *result.trajectory.hot_face,
                    *(value for row in result.trajectory.cells for value in row),
                )
                maximum_voltage = max(abs(value) for value in result.diagnostics.voltage)
                maximum_power = max(abs(value) for value in result.diagnostics.electrical_power)
                minimum_temperature = min(temperatures)
                maximum_temperature = max(temperatures)
                feasible = (
                    maximum_voltage <= config.maximum_absolute_voltage
                    and maximum_power <= config.maximum_absolute_power
                    and minimum_temperature >= config.minimum_temperature
                    and maximum_temperature <= config.maximum_temperature
                )
                scores.append(
                    DistributedExperimentCandidateScore(
                        name=f"lift{lift:g}K_{current:+g}A_{pulse_duration:g}s",
                        current_amplitude=current,
                        pulse_duration=pulse_duration,
                        reservoir_temperature_lift=lift,
                        feasible=feasible,
                        information_gain_nats=information_gain,
                        effective_rank=identifiability.effective_rank,
                        condition_number=identifiability.condition_number,
                        maximum_absolute_voltage=maximum_voltage,
                        maximum_absolute_power=maximum_power,
                        minimum_temperature=minimum_temperature,
                        maximum_temperature=maximum_temperature,
                    )
                )
    feasible_scores = tuple(score for score in scores if score.feasible)
    if not feasible_scores:
        raise ValueError("no distributed experiment candidate is feasible")
    selected = max(
        feasible_scores,
        key=lambda score: (score.information_gain_nats, score.effective_rank),
    )
    return DistributedExperimentSelectionResult(
        parameter_names=tuple(parameter.name for parameter in parameters),
        candidates=tuple(scores),
        selected=selected,
    )
