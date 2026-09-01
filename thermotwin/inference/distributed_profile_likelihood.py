"""Nonlinear profile likelihood for one distributed property curve."""

from dataclasses import dataclass, replace
import math
from typing import NamedTuple, Sequence, Tuple

from ..observations.distributed import DistributedObservationSet
from ..physics.distributed import PiecewiseLinearProperty
from ..simulation.distributed import DistributedLegExperiment
from ..numerics.matrices import inverse_and_determinant
from .distributed_identifiability import (
    DistributedIdentifiabilityConfig,
    DistributedPropertyCoefficient,
    analyze_distributed_identifiability,
)
from .distributed_properties import (
    DistributedPropertyFitConfig,
    DistributedPropertyFitResult,
    fit_distributed_property,
)
from .distributed_regularization import (
    mean_square_magnitude,
    second_difference_roughness,
)


@dataclass(frozen=True)
class DistributedProfileLikelihoodConfig:
    """Grid, restart, and interval settings for nonlinear profiles."""

    profile_points: int = 9
    multistart_initial_log_multipliers: Tuple[Tuple[float, ...], ...] = (
        (math.log(0.8),) * 3,
        (0.0,) * 3,
        (math.log(1.2),) * 3,
    )
    profile_restart_count: int = 1
    confidence_levels_and_thresholds: Tuple[Tuple[float, float], ...] = (
        (0.6827, 1.0),
        (0.95, 3.841459),
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.profile_points, int)
            or isinstance(self.profile_points, bool)
            or self.profile_points < 3
        ):
            raise ValueError("profile points must be an integer of at least three")
        if (
            not isinstance(self.profile_restart_count, int)
            or isinstance(self.profile_restart_count, bool)
            or self.profile_restart_count <= 0
        ):
            raise ValueError("profile restart count must be a positive integer")
        if not self.multistart_initial_log_multipliers:
            raise ValueError("at least one profile-likelihood start is required")
        coefficient_count = len(self.multistart_initial_log_multipliers[0])
        if coefficient_count == 0 or any(
            len(values) != coefficient_count
            or any(not math.isfinite(value) for value in values)
            for values in self.multistart_initial_log_multipliers
        ):
            raise ValueError("profile starts must be finite and equal length")
        if self.profile_restart_count > len(self.multistart_initial_log_multipliers) + 1:
            raise ValueError("profile restart count exceeds available starts")
        previous_level = 0.0
        for level, threshold in self.confidence_levels_and_thresholds:
            if (
                not math.isfinite(level)
                or level <= previous_level
                or level >= 1.0
                or not math.isfinite(threshold)
                or threshold <= 0.0
            ):
                raise ValueError("confidence levels and thresholds must be ordered")
            previous_level = level


class DistributedProfilePoint(NamedTuple):
    fixed_log_multiplier: float
    fitted_log_multipliers: Tuple[float, ...]
    data_chi_square: float
    penalized_score: float
    delta_score: float


class DistributedProfileInterval(NamedTuple):
    confidence_level: float
    threshold: float
    lower_log_multiplier: float
    upper_log_multiplier: float
    lower_hits_bound: bool
    upper_hits_bound: bool


class DistributedCoefficientProfile(NamedTuple):
    coefficient_index: int
    optimum_log_multiplier: float
    points: Tuple[DistributedProfilePoint, ...]
    intervals: Tuple[DistributedProfileInterval, ...]


class DistributedProfileLikelihoodResult(NamedTuple):
    property_name: str
    observation_count: int
    best_fit: DistributedPropertyFitResult
    best_data_chi_square: float
    best_penalized_score: float
    smoothness_weight: float
    shrinkage_weight: float
    coefficient_profiles: Tuple[DistributedCoefficientProfile, ...]


class DistributedLocalProfileApproximation(NamedTuple):
    """Quadratic profile approximation centred on one nonlinear best fit."""

    information_matrix: Tuple[Tuple[float, ...], ...]
    penalized_precision_matrix: Tuple[Tuple[float, ...], ...]
    log_standard_errors: Tuple[float, ...]
    intervals_by_coefficient: Tuple[Tuple[DistributedProfileInterval, ...], ...]


def _penalized_score(
    fit: DistributedPropertyFitResult,
    observation_count: int,
    config: DistributedPropertyFitConfig,
) -> Tuple[float, float]:
    data_chi_square = observation_count * fit.mean_normalized_squared_error
    penalty = observation_count * (
        config.smoothness_weight
        * float(second_difference_roughness(fit.log_multipliers))
        + config.shrinkage_weight
        * float(mean_square_magnitude(fit.log_multipliers))
    )
    return data_chi_square, data_chi_square + penalty


def _best_fit(
    experiments: Sequence[DistributedLegExperiment],
    observations: Sequence[DistributedObservationSet],
    fit_config: DistributedPropertyFitConfig,
    starts: Sequence[Sequence[float]],
    *,
    fixed: Tuple[float | None, ...] = (),
) -> Tuple[DistributedPropertyFitResult, float, float]:
    observation_count = sum(len(dataset.observations) for dataset in observations)
    candidates = []
    for start in starts:
        fit = fit_distributed_property(
            experiments,
            observations,
            replace(
                fit_config,
                initial_log_multipliers=tuple(start),
                fixed_log_multipliers=fixed,
            ),
        )
        data_chi_square, score = _penalized_score(
            fit, observation_count, fit_config
        )
        candidates.append((score, data_chi_square, fit))
    score, data_chi_square, fit = min(candidates, key=lambda item: item[0])
    return fit, data_chi_square, score


def _crossing(
    inside_x: float,
    inside_delta: float,
    outside_x: float,
    outside_delta: float,
    threshold: float,
) -> float:
    if outside_delta == inside_delta:
        return 0.5 * (inside_x + outside_x)
    fraction = (threshold - inside_delta) / (outside_delta - inside_delta)
    fraction = min(1.0, max(0.0, fraction))
    return inside_x + fraction * (outside_x - inside_x)


def _profile_interval(
    points: Sequence[DistributedProfilePoint],
    *,
    optimum: float,
    confidence_level: float,
    threshold: float,
    lower_bound: float,
    upper_bound: float,
) -> DistributedProfileInterval:
    ordered = tuple(sorted(points, key=lambda item: item.fixed_log_multiplier))
    anchors = tuple(
        index
        for index, point in enumerate(ordered)
        if point.fixed_log_multiplier == optimum and point.delta_score == 0.0
    )
    if len(anchors) != 1:
        raise RuntimeError("profile grid must contain one exact reported optimum")
    centre = anchors[0]
    left = centre
    while left > 0 and ordered[left - 1].delta_score <= threshold:
        left -= 1
    right = centre
    while right + 1 < len(ordered) and ordered[right + 1].delta_score <= threshold:
        right += 1

    lower_hits_bound = left == 0 and ordered[left].delta_score <= threshold
    upper_hits_bound = right == len(ordered) - 1 and ordered[right].delta_score <= threshold
    if lower_hits_bound:
        lower = lower_bound
    else:
        lower = _crossing(
            ordered[left].fixed_log_multiplier,
            ordered[left].delta_score,
            ordered[left - 1].fixed_log_multiplier,
            ordered[left - 1].delta_score,
            threshold,
        )
    if upper_hits_bound:
        upper = upper_bound
    else:
        upper = _crossing(
            ordered[right].fixed_log_multiplier,
            ordered[right].delta_score,
            ordered[right + 1].fixed_log_multiplier,
            ordered[right + 1].delta_score,
            threshold,
        )
    return DistributedProfileInterval(
        confidence_level=confidence_level,
        threshold=threshold,
        lower_log_multiplier=lower,
        upper_log_multiplier=upper,
        lower_hits_bound=lower_hits_bound,
        upper_hits_bound=upper_hits_bound,
    )


def profile_interval_contains(
    interval: DistributedProfileInterval, value: float
) -> bool:
    """Return whether a log coefficient lies within a reported interval."""

    if not math.isfinite(value):
        raise ValueError("profile target must be finite")
    return interval.lower_log_multiplier <= value <= interval.upper_log_multiplier


def local_profile_approximation(
    experiments: Sequence[DistributedLegExperiment],
    fit: DistributedPropertyFitResult,
    fit_config: DistributedPropertyFitConfig,
    profile_config: DistributedProfileLikelihoodConfig = (
        DistributedProfileLikelihoodConfig()
    ),
) -> DistributedLocalProfileApproximation:
    """Approximate nonlinear profiles locally for repeated coverage audits.

    The representative report still computes full re-optimized profiles. This
    quadratic approximation exists so tens of independent noise trials do not
    require thousands of complete transient refits.
    """

    experiments = tuple(experiments)
    if not experiments:
        raise ValueError("local profile approximation needs experiments")
    baseline = getattr(experiments[0].material, fit.property_name)
    if not isinstance(baseline, PiecewiseLinearProperty):
        raise TypeError("profiled property must be piecewise linear")
    if len(fit.log_multipliers) != len(baseline.values):
        raise ValueError("fit coefficient count does not match the property")
    fitted_property = baseline.with_values(fit.fitted_values)
    fitted_experiments = tuple(
        replace(
            experiment,
            material=replace(
                experiment.material,
                **{fit.property_name: fitted_property},
            ),
        )
        for experiment in experiments
    )
    parameters = tuple(
        DistributedPropertyCoefficient(fit.property_name, index)
        for index in range(len(fit.log_multipliers))
    )
    identifiability = analyze_distributed_identifiability(
        fitted_experiments,
        parameters,
        DistributedIdentifiabilityConfig(
            observation_interval=fit_config.observation_interval,
            channels=fit_config.channels,
            temperature_standard_deviation=(
                fit_config.temperature_standard_deviation
            ),
            voltage_standard_deviation=fit_config.voltage_standard_deviation,
            heat_rate_standard_deviation=fit_config.heat_rate_standard_deviation,
        ),
    )
    size = len(parameters)
    observation_count = identifiability.observation_count
    precision = [list(row) for row in identifiability.information_matrix]
    if fit_config.shrinkage_weight > 0.0:
        diagonal = observation_count * fit_config.shrinkage_weight / size
        for index in range(size):
            precision[index][index] += diagonal
    if fit_config.smoothness_weight > 0.0 and size >= 3:
        vectors = []
        for row in range(size - 2):
            vector = [0.0] * size
            vector[row] = 1.0
            vector[row + 1] = -2.0
            vector[row + 2] = 1.0
            vectors.append(tuple(vector))
        scale = observation_count * fit_config.smoothness_weight / len(vectors)
        for vector in vectors:
            for row in range(size):
                for column in range(size):
                    precision[row][column] += scale * vector[row] * vector[column]
    precision_tuple = tuple(tuple(row) for row in precision)
    covariance, _ = inverse_and_determinant(precision_tuple)
    standard_errors = tuple(
        math.sqrt(max(0.0, covariance[index][index])) for index in range(size)
    )
    lower_bound, upper_bound = fit_config.log_multiplier_bounds
    intervals = []
    for optimum, standard_error in zip(fit.log_multipliers, standard_errors):
        coefficient_intervals = []
        for level, threshold in profile_config.confidence_levels_and_thresholds:
            half_width = math.sqrt(threshold) * standard_error
            raw_lower = optimum - half_width
            raw_upper = optimum + half_width
            coefficient_intervals.append(
                DistributedProfileInterval(
                    confidence_level=level,
                    threshold=threshold,
                    lower_log_multiplier=max(lower_bound, raw_lower),
                    upper_log_multiplier=min(upper_bound, raw_upper),
                    lower_hits_bound=raw_lower <= lower_bound,
                    upper_hits_bound=raw_upper >= upper_bound,
                )
            )
        intervals.append(tuple(coefficient_intervals))
    return DistributedLocalProfileApproximation(
        information_matrix=identifiability.information_matrix,
        penalized_precision_matrix=precision_tuple,
        log_standard_errors=standard_errors,
        intervals_by_coefficient=tuple(intervals),
    )


def fit_distributed_property_profile_likelihood(
    experiments: Sequence[DistributedLegExperiment],
    observations: Sequence[DistributedObservationSet],
    fit_config: DistributedPropertyFitConfig,
    profile_config: DistributedProfileLikelihoodConfig = (
        DistributedProfileLikelihoodConfig()
    ),
) -> DistributedProfileLikelihoodResult:
    """Profile every log coefficient while re-optimizing all remaining ones."""

    experiments = tuple(experiments)
    observations = tuple(observations)
    starts = profile_config.multistart_initial_log_multipliers
    if not experiments or len(experiments) != len(observations):
        raise ValueError("experiments and observations must have equal nonzero length")
    coefficient_count = len(starts[0])
    if fit_config.fixed_log_multipliers:
        raise ValueError("top-level profile fit cannot begin with fixed coefficients")
    lower, upper = fit_config.log_multiplier_bounds
    if any(
        len(start) != coefficient_count
        or any(value < lower or value > upper for value in start)
        for start in starts
    ):
        raise ValueError("profile starts must match coefficient count and bounds")
    best_fit, best_data_chi_square, best_score = _best_fit(
        experiments, observations, fit_config, starts
    )
    observation_count = sum(len(dataset.observations) for dataset in observations)
    grid = tuple(
        lower + index * (upper - lower) / (profile_config.profile_points - 1)
        for index in range(profile_config.profile_points)
    )
    def raw_profiles(current_best):
        result = []
        for coefficient_index in range(coefficient_count):
            values = tuple(
                sorted(
                    set((*grid, current_best.log_multipliers[coefficient_index]))
                )
            )
            points = []
            for fixed_value in values:
                fixed = tuple(
                    fixed_value if index == coefficient_index else None
                    for index in range(coefficient_count)
                )
                adjusted_best = tuple(
                    fixed_value if index == coefficient_index else value
                    for index, value in enumerate(current_best.log_multipliers)
                )
                restarts = [adjusted_best]
                for start in starts:
                    adjusted = tuple(
                        fixed_value if index == coefficient_index else value
                        for index, value in enumerate(start)
                    )
                    if adjusted not in restarts:
                        restarts.append(adjusted)
                    if len(restarts) >= profile_config.profile_restart_count:
                        break
                fit, data_chi_square, score = _best_fit(
                    experiments,
                    observations,
                    fit_config,
                    tuple(restarts[: profile_config.profile_restart_count]),
                    fixed=fixed,
                )
                points.append(
                    DistributedProfilePoint(
                        fixed_log_multiplier=fixed_value,
                        fitted_log_multipliers=fit.log_multipliers,
                        data_chi_square=data_chi_square,
                        penalized_score=score,
                        delta_score=0.0,
                    )
                )
            result.append(tuple(points))
        return tuple(result)

    raw = raw_profiles(best_fit)
    lowest = min(
        (point for points in raw for point in points),
        key=lambda point: point.penalized_score,
    )
    if lowest.penalized_score < best_score:
        fixed_best = tuple(lowest.fitted_log_multipliers)
        best_fit, best_data_chi_square, best_score = _best_fit(
            experiments,
            observations,
            fit_config,
            (fixed_best,),
            fixed=fixed_best,
        )

    profiles = []
    for coefficient_index, raw_points in enumerate(raw):
        candidates = tuple(
            DistributedProfilePoint(
                fixed_log_multiplier=point.fixed_log_multiplier,
                fitted_log_multipliers=point.fitted_log_multipliers,
                data_chi_square=point.data_chi_square,
                penalized_score=point.penalized_score,
                delta_score=max(0.0, point.penalized_score - best_score),
            )
            for point in raw_points
        )
        anchor = DistributedProfilePoint(
            fixed_log_multiplier=best_fit.log_multipliers[coefficient_index],
            fitted_log_multipliers=best_fit.log_multipliers,
            data_chi_square=best_data_chi_square,
            penalized_score=best_score,
            delta_score=0.0,
        )
        by_fixed_value = {}
        for point in (*candidates, anchor):
            existing = by_fixed_value.get(point.fixed_log_multiplier)
            if existing is None or point.penalized_score < existing.penalized_score:
                by_fixed_value[point.fixed_log_multiplier] = point
        points = tuple(
            sorted(by_fixed_value.values(), key=lambda point: point.fixed_log_multiplier)
        )
        intervals = tuple(
            _profile_interval(
                points,
                optimum=best_fit.log_multipliers[coefficient_index],
                confidence_level=level,
                threshold=threshold,
                lower_bound=lower,
                upper_bound=upper,
            )
            for level, threshold in profile_config.confidence_levels_and_thresholds
        )
        profiles.append(
            DistributedCoefficientProfile(
                coefficient_index=coefficient_index,
                optimum_log_multiplier=best_fit.log_multipliers[coefficient_index],
                points=tuple(points),
                intervals=intervals,
            )
        )
    return DistributedProfileLikelihoodResult(
        property_name=fit_config.property_name,
        observation_count=observation_count,
        best_fit=best_fit,
        best_data_chi_square=best_data_chi_square,
        best_penalized_score=best_score,
        smoothness_weight=fit_config.smoothness_weight,
        shrinkage_weight=fit_config.shrinkage_weight,
        coefficient_profiles=tuple(profiles),
    )
