"""Conventional local identifiability analysis for distributed properties."""

from dataclasses import dataclass, replace
import math
from typing import NamedTuple, Sequence, Tuple

from ..observations.distributed import (
    DistributedObservationChannels,
    run_distributed_virtual_experiment,
)
from ..physics.distributed import (
    DistributedThermoelectricMaterial,
    PiecewiseLinearProperty,
)
from ..simulation.distributed import (
    DistributedLegExperiment,
    run_distributed_leg_experiment,
)


@dataclass(frozen=True)
class DistributedPropertyCoefficient:
    """One log-magnitude coefficient of alpha(T), rho_e(T), or kappa(T)."""

    property_name: str
    coefficient_index: int

    def __post_init__(self) -> None:
        if self.property_name not in {
            "seebeck_coefficient",
            "electrical_resistivity",
            "thermal_conductivity",
        }:
            raise ValueError("unknown distributed material property")
        if (
            not isinstance(self.coefficient_index, int)
            or isinstance(self.coefficient_index, bool)
            or self.coefficient_index < 0
        ):
            raise ValueError("coefficient index must be a nonnegative integer")

    @property
    def name(self) -> str:
        labels = {
            "seebeck_coefficient": "alpha",
            "electrical_resistivity": "rho_e",
            "thermal_conductivity": "kappa",
        }
        return f"log|{labels[self.property_name]}[{self.coefficient_index}]|"


@dataclass(frozen=True)
class DistributedIdentifiabilityConfig:
    observation_interval: float = 0.1
    channels: DistributedObservationChannels = DistributedObservationChannels()
    temperature_standard_deviation: float = 0.01
    voltage_standard_deviation: float = 1.0e-5
    heat_rate_standard_deviation: float = 5.0e-4
    log_parameter_step: float = 0.01
    relative_rank_tolerance: float = 1.0e-5

    def __post_init__(self) -> None:
        for name, value in (
            ("observation interval", self.observation_interval),
            ("temperature standard deviation", self.temperature_standard_deviation),
            ("voltage standard deviation", self.voltage_standard_deviation),
            ("heat-rate standard deviation", self.heat_rate_standard_deviation),
            ("log-parameter step", self.log_parameter_step),
            ("relative rank tolerance", self.relative_rank_tolerance),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    def channel_scale(self, channel: str) -> float:
        if channel.endswith("temperature"):
            return self.temperature_standard_deviation
        if channel == "voltage":
            return self.voltage_standard_deviation
        if channel.endswith("heat"):
            return self.heat_rate_standard_deviation
        raise ValueError(f"unknown observation channel: {channel}")


class DistributedIdentifiabilityResult(NamedTuple):
    parameter_names: Tuple[str, ...]
    observation_count: int
    jacobian: Tuple[Tuple[float, ...], ...]
    information_matrix: Tuple[Tuple[float, ...], ...]
    singular_values: Tuple[float, ...]
    effective_rank: int
    condition_number: float
    column_norms: Tuple[float, ...]
    temperature_range: Tuple[float, float]


@dataclass(frozen=True)
class DistributedIdentifiabilityGateConfig:
    """Turn a local singular spectrum into a declared inference decision.

    The Jacobian is already normalized by the declared sensor noise.  The
    product ``singular_value * maximum_log_displacement`` therefore estimates
    the largest noise-normalized signal available along one right-singular
    direction inside the allowed coefficient neighborhood.
    """

    maximum_log_displacement: float = 0.3
    required_noise_normalized_signal: float = 1.0
    structural_zero_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        for name, value in (
            ("maximum log displacement", self.maximum_log_displacement),
            (
                "required noise-normalized signal",
                self.required_noise_normalized_signal,
            ),
            ("structural-zero tolerance", self.structural_zero_tolerance),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


class DistributedIdentifiabilityAssessment(NamedTuple):
    """Decision returned before fitting a distributed property curve."""

    status: str
    supported_rank: int
    coefficient_count: int
    minimum_required_singular_value: float
    weakest_resolvable_log_displacement: float
    explanation: str


def assess_distributed_identifiability(
    result: DistributedIdentifiabilityResult,
    config: DistributedIdentifiabilityGateConfig = (
        DistributedIdentifiabilityGateConfig()
    ),
) -> DistributedIdentifiabilityAssessment:
    """Classify structural and practical non-identifiability before fitting.

    ``supported`` does not mean globally identifiable.  It means only that all
    local coefficient directions clear the declared one-sigma signal gate
    within the stated log-coefficient neighborhood.
    """

    coefficient_count = len(result.parameter_names)
    if coefficient_count == 0 or len(result.singular_values) != coefficient_count:
        raise ValueError("identifiability spectrum must match the parameter count")
    minimum_required = (
        config.required_noise_normalized_signal / config.maximum_log_displacement
    )
    largest = result.singular_values[0]
    if largest <= config.structural_zero_tolerance:
        return DistributedIdentifiabilityAssessment(
            status="structurally_non_identifiable",
            supported_rank=0,
            coefficient_count=coefficient_count,
            minimum_required_singular_value=minimum_required,
            weakest_resolvable_log_displacement=math.inf,
            explanation=(
                "the selected observations have zero local sensitivity to every "
                "fitted coefficient"
            ),
        )
    supported_rank = sum(
        value >= minimum_required for value in result.singular_values
    )
    smallest = result.singular_values[-1]
    weakest_displacement = (
        config.required_noise_normalized_signal / smallest
        if smallest > config.structural_zero_tolerance
        else math.inf
    )
    if supported_rank < coefficient_count:
        explanation = (
            f"only {supported_rank} of {coefficient_count} local coefficient "
            "directions clear the declared noise-resolution gate"
        )
        status = "practically_non_identifiable"
    else:
        explanation = (
            "all local coefficient directions clear the declared "
            "noise-resolution gate"
        )
        status = "supported"
    return DistributedIdentifiabilityAssessment(
        status=status,
        supported_rank=supported_rank,
        coefficient_count=coefficient_count,
        minimum_required_singular_value=minimum_required,
        weakest_resolvable_log_displacement=weakest_displacement,
        explanation=explanation,
    )


def _replace_coefficient(
    material: DistributedThermoelectricMaterial,
    parameter: DistributedPropertyCoefficient,
    log_offset: float,
) -> DistributedThermoelectricMaterial:
    prop = getattr(material, parameter.property_name)
    if not isinstance(prop, PiecewiseLinearProperty):
        raise TypeError(
            f"{parameter.property_name} must be PiecewiseLinearProperty for "
            "coefficient identifiability"
        )
    if parameter.coefficient_index >= len(prop.values):
        raise ValueError("property coefficient index is out of range")
    values = list(prop.values)
    values[parameter.coefficient_index] *= math.exp(log_offset)
    return replace(material, **{parameter.property_name: prop.with_values(values)})


def _experiment_vector(
    experiment: DistributedLegExperiment,
    config: DistributedIdentifiabilityConfig,
) -> Tuple[Tuple[Tuple[float, str], ...], Tuple[float, ...]]:
    observations = run_distributed_virtual_experiment(
        experiment,
        observation_interval=config.observation_interval,
        channels=config.channels,
    )
    return observations.keys(), observations.values()


def _jacobi_eigenvalues(
    matrix: Sequence[Sequence[float]],
    *,
    tolerance: float = 1e-12,
    max_sweeps: int = 100,
) -> Tuple[float, ...]:
    """Return eigenvalues of one small real symmetric matrix."""

    values = [list(map(float, row)) for row in matrix]
    size = len(values)
    if size == 0 or any(len(row) != size for row in values):
        raise ValueError("eigenvalue matrix must be nonempty and square")
    if any(not math.isfinite(value) for row in values for value in row):
        raise ValueError("eigenvalue matrix must be finite")
    scale = max(1.0, max(abs(value) for row in values for value in row))
    for _ in range(max_sweeps * max(1, size * size)):
        p, q = max(
            ((row, column) for row in range(size) for column in range(row + 1, size)),
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
    return tuple(sorted((values[index][index] for index in range(size)), reverse=True))


def analyze_distributed_identifiability(
    experiments: Sequence[DistributedLegExperiment],
    parameters: Sequence[DistributedPropertyCoefficient],
    config: DistributedIdentifiabilityConfig = DistributedIdentifiabilityConfig(),
) -> DistributedIdentifiabilityResult:
    """Compute a noise-normalized finite-difference Jacobian and its spectrum."""

    experiments = tuple(experiments)
    parameters = tuple(parameters)
    if not experiments:
        raise ValueError("at least one distributed experiment is required")
    if not parameters:
        raise ValueError("at least one distributed parameter is required")
    baseline_keys = []
    baseline_values = []
    minimum_temperature = math.inf
    maximum_temperature = -math.inf
    for experiment_index, experiment in enumerate(experiments):
        keys, values = _experiment_vector(experiment, config)
        baseline_keys.extend((experiment_index, *key) for key in keys)
        baseline_values.extend(values)
        result = run_distributed_leg_experiment(experiment)
        temperatures = (
            *result.trajectory.cold_face,
            *result.trajectory.hot_face,
            *(value for row in result.trajectory.cells for value in row),
        )
        minimum_temperature = min(minimum_temperature, min(temperatures))
        maximum_temperature = max(maximum_temperature, max(temperatures))

    derivative_columns = []
    for parameter in parameters:
        minus_values = []
        plus_values = []
        minus_keys = []
        plus_keys = []
        for experiment_index, experiment in enumerate(experiments):
            minus_experiment = replace(
                experiment,
                material=_replace_coefficient(
                    experiment.material, parameter, -config.log_parameter_step
                ),
            )
            plus_experiment = replace(
                experiment,
                material=_replace_coefficient(
                    experiment.material, parameter, config.log_parameter_step
                ),
            )
            keys, values = _experiment_vector(minus_experiment, config)
            minus_keys.extend((experiment_index, *key) for key in keys)
            minus_values.extend(values)
            keys, values = _experiment_vector(plus_experiment, config)
            plus_keys.extend((experiment_index, *key) for key in keys)
            plus_values.extend(values)
        if tuple(minus_keys) != tuple(baseline_keys) or tuple(plus_keys) != tuple(baseline_keys):
            raise RuntimeError("perturbed observation keys do not match baseline")
        derivative_columns.append(
            tuple(
                (plus - minus) / (2.0 * config.log_parameter_step)
                for minus, plus in zip(minus_values, plus_values)
            )
        )

    scales = tuple(
        config.channel_scale(channel)
        for _, _, channel in baseline_keys
    )
    jacobian = tuple(
        tuple(column[row] / scales[row] for column in derivative_columns)
        for row in range(len(baseline_values))
    )
    parameter_count = len(parameters)
    information = tuple(
        tuple(
            sum(row[left] * row[right] for row in jacobian)
            for right in range(parameter_count)
        )
        for left in range(parameter_count)
    )
    eigenvalues = _jacobi_eigenvalues(information)
    singular_values = tuple(math.sqrt(max(0.0, value)) for value in eigenvalues)
    largest = singular_values[0]
    threshold = config.relative_rank_tolerance * largest
    effective_rank = sum(value > threshold for value in singular_values)
    condition_number = (
        largest / singular_values[-1]
        if singular_values[-1] > threshold
        else math.inf
    )
    column_norms = tuple(
        math.sqrt(sum(value * value for value in column))
        for column in derivative_columns
    )
    return DistributedIdentifiabilityResult(
        parameter_names=tuple(parameter.name for parameter in parameters),
        observation_count=len(baseline_values),
        jacobian=jacobian,
        information_matrix=information,
        singular_values=singular_values,
        effective_rank=effective_rank,
        condition_number=condition_number,
        column_norms=column_norms,
        temperature_range=(minimum_temperature, maximum_temperature),
    )
