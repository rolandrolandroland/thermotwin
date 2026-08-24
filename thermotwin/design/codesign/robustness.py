"""Fixed-control robustness evaluation for selected co-design points."""

import math
import random

from ...numerics.statistics import interpolated_quantile
from .evaluation import evaluate_design_current
from .models import (
    DesignOperatingPoint,
    ModuleAssemblyAssumptions,
    PropertyMultipliers,
    RobustnessResult,
)


def _lognormal_unit_mean(generator: random.Random, sigma: float) -> float:
    return math.exp(generator.gauss(-0.5 * sigma**2, sigma))


def run_robustness_study(
    nominal: DesignOperatingPoint,
    *,
    trials: int = 300,
    seed: int = 20260821,
    assembly: ModuleAssemblyAssumptions = ModuleAssemblyAssumptions(),
) -> RobustnessResult:
    """Hold current fixed while perturbing material and interface properties."""

    if isinstance(trials, bool) or trials <= 0:
        raise ValueError("robustness trials must be a positive integer")
    generator = random.Random(seed)
    points = []
    for _ in range(trials):
        multipliers = PropertyMultipliers(
            max(0.80, generator.gauss(1.0, 0.03)),
            max(0.80, generator.gauss(1.0, 0.03)),
            _lognormal_unit_mean(generator, 0.08),
            _lognormal_unit_mean(generator, 0.08),
            _lognormal_unit_mean(generator, 0.08),
            _lognormal_unit_mean(generator, 0.08),
        )
        efficiency = min(0.99, max(0.85, generator.gauss(0.95, 0.01)))
        points.append(
            evaluate_design_current(
                nominal.design,
                nominal.application,
                nominal.mean_current,
                assembly=assembly,
                multipliers=multipliers,
                electrical_contact_resistivity_multiplier=(
                    _lognormal_unit_mean(generator, 0.20)
                ),
                contact_multiplier=_lognormal_unit_mean(generator, 0.15),
                cold_exchanger_multiplier=_lognormal_unit_mean(generator, 0.10),
                hot_exchanger_multiplier=_lognormal_unit_mean(generator, 0.10),
                converter_efficiency=efficiency,
            )
        )
    cooling = tuple(point.delivered_cooling_rate for point in points)
    cop = tuple(point.wall_cooling_cop or 0.0 for point in points)
    hot_face = tuple(point.hot_face_temperature for point in points)
    return RobustnessResult(
        nominal.application,
        nominal,
        trials,
        sum(point.feasible for point in points) / trials,
        (
            interpolated_quantile(cooling, 0.05),
            interpolated_quantile(cooling, 0.50),
            interpolated_quantile(cooling, 0.95),
        ),
        (
            interpolated_quantile(cop, 0.05),
            interpolated_quantile(cop, 0.50),
            interpolated_quantile(cop, 0.95),
        ),
        (
            interpolated_quantile(hot_face, 0.05),
            interpolated_quantile(hot_face, 0.50),
            interpolated_quantile(hot_face, 0.95),
        ),
    )
