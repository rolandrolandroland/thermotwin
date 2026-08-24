"""Gaussian-process Bayesian optimization and fair random baselines."""

import math
import random
from typing import Sequence, Tuple

from ...numerics.matrices import inverse_and_determinant
from ...numerics.statistics import interpolated_quantile
from .evaluation import optimize_design_current, prototype_cost_index
from .models import (
    ApplicationSpecification,
    BayesianOptimizationResult,
    ModuleAssemblyAssumptions,
    PrototypeDesign,
)
from .sampling import design_features


def _kernel(
    left: Sequence[float],
    right: Sequence[float],
    length_scale: float = 1.4,
) -> float:
    squared_distance = sum((a - b) ** 2 for a, b in zip(left, right))
    return math.exp(-0.5 * squared_distance / length_scale**2)


def gaussian_process_predict(
    observed_features: Sequence[Sequence[float]],
    observed_values: Sequence[float],
    query_features: Sequence[float],
    *,
    nugget: float = 1.0e-5,
) -> Tuple[float, float]:
    """Return GP posterior mean and standard deviation for one query."""

    features = tuple(tuple(row) for row in observed_features)
    values = tuple(float(value) for value in observed_values)
    if not features or len(features) != len(values):
        raise ValueError("GP observations must be nonempty and aligned")
    if any(len(row) != len(features[0]) for row in features):
        raise ValueError("GP feature rows must have equal width")
    if len(query_features) != len(features[0]):
        raise ValueError("query feature width does not match observations")
    if not math.isfinite(nugget) or nugget <= 0.0:
        raise ValueError("GP nugget must be positive and finite")
    mean_value = sum(values) / len(values)
    variance_value = sum((value - mean_value) ** 2 for value in values) / len(values)
    scale_value = max(math.sqrt(variance_value), 1.0e-9)
    standardized = tuple((value - mean_value) / scale_value for value in values)
    covariance = tuple(
        tuple(
            _kernel(left, right) + (nugget if i == j else 0.0)
            for j, right in enumerate(features)
        )
        for i, left in enumerate(features)
    )
    inverse, _ = inverse_and_determinant(covariance)
    alpha = tuple(
        sum(coefficient * value for coefficient, value in zip(row, standardized))
        for row in inverse
    )
    query_covariance = tuple(_kernel(query_features, row) for row in features)
    standardized_mean = sum(a * b for a, b in zip(query_covariance, alpha))
    projected = tuple(
        sum(coefficient * value for coefficient, value in zip(row, query_covariance))
        for row in inverse
    )
    standardized_variance = max(
        0.0,
        1.0 - sum(a * b for a, b in zip(query_covariance, projected)),
    )
    return (
        mean_value + scale_value * standardized_mean,
        scale_value * math.sqrt(standardized_variance),
    )


def expected_improvement(
    predicted_mean: float,
    predicted_standard_deviation: float,
    incumbent: float,
) -> float:
    """Return the maximization expected-improvement acquisition value."""

    if any(
        not math.isfinite(value)
        for value in (predicted_mean, predicted_standard_deviation, incumbent)
    ):
        raise ValueError("expected-improvement arguments must be finite")
    if predicted_standard_deviation < 0.0:
        raise ValueError("predicted standard deviation cannot be negative")
    improvement = predicted_mean - incumbent
    if predicted_standard_deviation == 0.0:
        return max(0.0, improvement)
    z_score = improvement / predicted_standard_deviation
    normal_cdf = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
    normal_pdf = math.exp(-0.5 * z_score**2) / math.sqrt(2.0 * math.pi)
    return improvement * normal_cdf + predicted_standard_deviation * normal_pdf


def run_bayesian_optimization(
    application: ApplicationSpecification,
    initial_designs: Sequence[PrototypeDesign],
    candidate_designs: Sequence[PrototypeDesign],
    *,
    iterations: int = 12,
    random_repetitions: int = 25,
    seed: int = 20260821,
    current_grid_size: int = 28,
    assembly: ModuleAssemblyAssumptions = ModuleAssemblyAssumptions(),
) -> BayesianOptimizationResult:
    """Compare cost-aware expected improvement with random candidate order."""

    initial = tuple(initial_designs)
    candidates = tuple(candidate_designs)
    if not initial or not candidates:
        raise ValueError("initial and candidate design sets must be nonempty")
    if isinstance(iterations, bool) or not isinstance(iterations, int):
        raise ValueError("iterations must be a positive integer")
    if iterations <= 0 or iterations > len(candidates):
        raise ValueError("iterations must fit inside the candidate set")
    if (
        isinstance(random_repetitions, bool)
        or not isinstance(random_repetitions, int)
        or random_repetitions <= 0
    ):
        raise ValueError("random repetitions must be a positive integer")
    all_ids = tuple(design.design_id for design in initial + candidates)
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("design IDs must be unique")
    evaluation_cache = {
        design.design_id: optimize_design_current(
            design,
            application,
            grid_size=current_grid_size,
            assembly=assembly,
        )
        for design in initial + candidates
    }
    observed = [evaluation_cache[design.design_id] for design in initial]
    available = list(candidates)
    acquired = []
    history = [max(point.utility for point in observed)]
    for _ in range(iterations):
        features = tuple(design_features(point.design) for point in observed)
        values = tuple(point.utility for point in observed)
        incumbent = max(values)
        scored = []
        for design in available:
            predicted_mean, predicted_std = gaussian_process_predict(
                features,
                values,
                design_features(design),
            )
            improvement = expected_improvement(
                predicted_mean,
                predicted_std,
                incumbent,
            )
            acquisition = improvement / math.sqrt(prototype_cost_index(design))
            scored.append((acquisition, predicted_mean, predicted_std, design))
        _, _, _, selected_design = max(
            scored,
            key=lambda item: (item[0], item[1] + 0.1 * item[2]),
        )
        selected = evaluation_cache[selected_design.design_id]
        acquired.append(selected)
        observed.append(selected)
        available.remove(selected_design)
        history.append(max(history[-1], selected.utility))

    random_histories = []
    for repetition in range(random_repetitions):
        generator = random.Random(seed + 1009 * (repetition + 1))
        order = list(candidates)
        generator.shuffle(order)
        best = max(point.utility for point in observed[: len(initial)])
        random_history = [best]
        for design in order[:iterations]:
            best = max(best, evaluation_cache[design.design_id].utility)
            random_history.append(best)
        random_histories.append(tuple(random_history))
    random_median = tuple(
        interpolated_quantile(
            tuple(row[index] for row in random_histories),
            0.50,
        )
        for index in range(iterations + 1)
    )
    random_lower = tuple(
        interpolated_quantile(
            tuple(row[index] for row in random_histories),
            0.10,
        )
        for index in range(iterations + 1)
    )
    random_upper = tuple(
        interpolated_quantile(
            tuple(row[index] for row in random_histories),
            0.90,
        )
        for index in range(iterations + 1)
    )
    oracle = max(evaluation_cache.values(), key=lambda point: point.utility)
    selected = max(observed, key=lambda point: point.utility)
    return BayesianOptimizationResult(
        application,
        tuple(observed[: len(initial)]),
        tuple(acquired),
        tuple(history),
        random_median,
        random_lower,
        random_upper,
        selected,
        oracle,
    )
