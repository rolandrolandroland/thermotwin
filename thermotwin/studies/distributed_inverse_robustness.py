"""Noisy multi-seed recovery for one distributed constitutive curve."""

from dataclasses import dataclass, replace
import math
from statistics import fmean
from typing import NamedTuple, Optional, Sequence, Tuple

from ..inference.distributed_properties import (
    DistributedPropertyFitConfig,
    fit_distributed_property,
)
from ..observations.distributed import (
    DistributedObservationChannels,
    DistributedObservationSet,
    add_distributed_gaussian_noise,
    run_distributed_virtual_experiment,
)
from ..physics.distributed import PiecewiseLinearProperty
from ..simulation.distributed import (
    DistributedLegExperiment,
    distributed_inverse_constant_experiments,
)


RHO_TRUTH_MULTIPLIERS = (1.04, 1.07, 1.03)


@dataclass(frozen=True)
class DistributedRecoveryCriteria:
    """Predeclared criteria applied identically to every completed trial."""

    maximum_absolute_multiplier_error: float = 0.10
    minimum_loss_reduction_fraction: float = 0.90
    maximum_final_normalized_loss: float = 5.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.maximum_absolute_multiplier_error)
            or self.maximum_absolute_multiplier_error <= 0.0
        ):
            raise ValueError("maximum multiplier error must be finite and positive")
        if (
            not math.isfinite(self.minimum_loss_reduction_fraction)
            or not 0.0 <= self.minimum_loss_reduction_fraction < 1.0
        ):
            raise ValueError("minimum loss reduction must lie from zero to one")
        if (
            not math.isfinite(self.maximum_final_normalized_loss)
            or self.maximum_final_normalized_loss <= 0.0
        ):
            raise ValueError("maximum final loss must be finite and positive")


@dataclass(frozen=True)
class DistributedInverseRobustnessConfig:
    """Frozen observation, seed, optimizer, and success settings."""

    trial_count: int = 5
    first_seed: int = 27_001
    inverse_pinn_epochs: int = 600
    observation_interval: float = 0.08
    temperature_standard_deviation: float = 0.01
    voltage_standard_deviation: float = 1.0e-5
    criteria: DistributedRecoveryCriteria = DistributedRecoveryCriteria()

    def __post_init__(self) -> None:
        for name, value in (
            ("trial count", self.trial_count),
            ("first seed", self.first_seed),
            ("inverse PINN epochs", self.inverse_pinn_epochs),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if self.trial_count <= 0:
            raise ValueError("trial count must be positive")
        if self.first_seed < 0:
            raise ValueError("first seed must be nonnegative")
        if self.inverse_pinn_epochs <= 0:
            raise ValueError("inverse PINN epochs must be positive")
        for name, value in (
            ("observation interval", self.observation_interval),
            ("temperature standard deviation", self.temperature_standard_deviation),
            ("voltage standard deviation", self.voltage_standard_deviation),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.criteria, DistributedRecoveryCriteria):
            raise ValueError("criteria must be distributed recovery criteria")


class DistributedInverseRobustnessSeeds(NamedTuple):
    """One neural seed and one independent noise seed per experiment."""

    neural: int
    observations: Tuple[int, ...]


class DistributedInverseRobustnessTrial(NamedTuple):
    """Every result and predeclared pass/fail decision for one trial."""

    trial_index: int
    seeds: DistributedInverseRobustnessSeeds
    conventional_multipliers: Tuple[float, ...]
    conventional_initial_normalized_loss: float
    conventional_final_normalized_loss: float
    conventional_loss_reduction_fraction: float
    conventional_maximum_absolute_multiplier_error: float
    conventional_reached_search_bound: bool
    conventional_success: bool
    conventional_failure_reasons: Tuple[str, ...]
    inverse_pinn_multipliers: Optional[Tuple[float, ...]]
    inverse_pinn_initial_normalized_loss: Optional[float]
    inverse_pinn_final_normalized_loss: Optional[float]
    inverse_pinn_loss_reduction_fraction: Optional[float]
    inverse_pinn_maximum_absolute_multiplier_error: float
    inverse_pinn_success: bool
    inverse_pinn_failure_reasons: Tuple[str, ...]


class DistributedInverseRobustnessSummary(NamedTuple):
    """Aggregate metrics that retain unsuccessful completed trials."""

    trial_count: int
    conventional_success_count: int
    inverse_pinn_success_count: int
    inverse_pinn_completed_count: int
    conventional_search_bound_hits: int
    conventional_mean_multipliers: Tuple[float, ...]
    inverse_pinn_mean_multipliers: Optional[Tuple[float, ...]]
    conventional_multiplier_rmse: float
    inverse_pinn_multiplier_rmse: Optional[float]
    conventional_worst_trial_error: float
    inverse_pinn_worst_trial_error: float


class DistributedInverseRobustnessStudyResult(NamedTuple):
    config: DistributedInverseRobustnessConfig
    truth_multipliers: Tuple[float, ...]
    trials: Tuple[DistributedInverseRobustnessTrial, ...]
    summary: DistributedInverseRobustnessSummary


def distributed_inverse_robustness_seeds(
    first_seed: int,
    trial_index: int,
    experiment_count: int,
) -> DistributedInverseRobustnessSeeds:
    """Allocate collision-free contiguous seed blocks across all trials."""

    for name, value in (
        ("first seed", first_seed),
        ("trial index", trial_index),
        ("experiment count", experiment_count),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
    if first_seed < 0 or trial_index < 0:
        raise ValueError("first seed and trial index must be nonnegative")
    if experiment_count <= 0:
        raise ValueError("experiment count must be positive")
    block_size = experiment_count + 1
    block_start = first_seed + trial_index * block_size
    return DistributedInverseRobustnessSeeds(
        neural=block_start,
        observations=tuple(
            block_start + 1 + index for index in range(experiment_count)
        ),
    )


def distributed_resistivity_truth_experiments() -> Tuple[
    Tuple[DistributedLegExperiment, ...],
    Tuple[DistributedLegExperiment, ...],
    PiecewiseLinearProperty,
]:
    baseline_experiments = distributed_inverse_constant_experiments()
    baseline_property = baseline_experiments[0].material.electrical_resistivity
    if not isinstance(baseline_property, PiecewiseLinearProperty):
        raise TypeError("distributed resistivity baseline must be piecewise linear")
    truth_property = baseline_property.with_values(
        tuple(
            value * multiplier
            for value, multiplier in zip(
                baseline_property.values, RHO_TRUTH_MULTIPLIERS
            )
        )
    )
    truth_experiments = tuple(
        replace(
            experiment,
            material=replace(
                experiment.material,
                electrical_resistivity=truth_property,
            ),
        )
        for experiment in baseline_experiments
    )
    return baseline_experiments, truth_experiments, baseline_property


def noisy_distributed_inverse_observations(
    truth_experiments: Sequence[DistributedLegExperiment],
    *,
    observation_interval: float,
    temperature_standard_deviation: float,
    voltage_standard_deviation: float,
    seeds: Sequence[int],
) -> Tuple[DistributedObservationSet, ...]:
    """Generate independent noisy terminal observations for each regime."""

    truth_experiments = tuple(truth_experiments)
    seeds = tuple(seeds)
    if not truth_experiments or len(truth_experiments) != len(seeds):
        raise ValueError("truth experiments and noise seeds must have equal length")
    channels = DistributedObservationChannels()
    standard_deviations = {
        "cold_face_temperature": temperature_standard_deviation,
        "hot_face_temperature": temperature_standard_deviation,
        "voltage": voltage_standard_deviation,
    }
    return tuple(
        add_distributed_gaussian_noise(
            run_distributed_virtual_experiment(
                experiment,
                observation_interval=observation_interval,
                channels=channels,
            ),
            standard_deviations=standard_deviations,
            seed=seed,
        )
        for experiment, seed in zip(truth_experiments, seeds)
    )


def _loss_reduction_fraction(initial: float, final: float) -> float:
    if not math.isfinite(initial) or not math.isfinite(final) or initial <= 0.0:
        return -math.inf
    return 1.0 - final / initial


def _maximum_multiplier_error(
    multipliers: Sequence[float],
    truth: Sequence[float] = RHO_TRUTH_MULTIPLIERS,
) -> float:
    multipliers = tuple(multipliers)
    truth = tuple(truth)
    if len(multipliers) != len(truth) or not multipliers:
        raise ValueError("multiplier and truth vectors must have equal nonzero length")
    if any(not math.isfinite(value) for value in (*multipliers, *truth)):
        return math.inf
    return max(abs(value - target) for value, target in zip(multipliers, truth))


def recovery_failure_reasons(
    *,
    maximum_multiplier_error: float,
    initial_normalized_loss: float,
    final_normalized_loss: float,
    criteria: DistributedRecoveryCriteria,
) -> Tuple[str, ...]:
    """Apply the predeclared gate without changing it after seeing a trial."""

    reasons = []
    reduction = _loss_reduction_fraction(
        initial_normalized_loss, final_normalized_loss
    )
    if not math.isfinite(maximum_multiplier_error):
        reasons.append("nonfinite_multiplier_error")
    elif maximum_multiplier_error > criteria.maximum_absolute_multiplier_error:
        reasons.append("multiplier_error")
    if not math.isfinite(final_normalized_loss):
        reasons.append("nonfinite_final_loss")
    elif final_normalized_loss > criteria.maximum_final_normalized_loss:
        reasons.append("final_loss")
    if reduction < criteria.minimum_loss_reduction_fraction:
        reasons.append("loss_reduction")
    return tuple(reasons)


def run_distributed_inverse_robustness_trial(
    trial_index: int,
    config: DistributedInverseRobustnessConfig,
) -> DistributedInverseRobustnessTrial:
    """Run one noisy conventional fit and one noisy inverse-PINN fit."""

    baseline_experiments, truth_experiments, baseline_property = (
        distributed_resistivity_truth_experiments()
    )
    seeds = distributed_inverse_robustness_seeds(
        config.first_seed, trial_index, len(truth_experiments)
    )
    observations = noisy_distributed_inverse_observations(
        truth_experiments,
        observation_interval=config.observation_interval,
        temperature_standard_deviation=config.temperature_standard_deviation,
        voltage_standard_deviation=config.voltage_standard_deviation,
        seeds=seeds.observations,
    )
    channels = DistributedObservationChannels()
    conventional = fit_distributed_property(
        baseline_experiments,
        observations,
        DistributedPropertyFitConfig(
            property_name="electrical_resistivity",
            observation_interval=config.observation_interval,
            channels=channels,
            initial_log_multipliers=(math.log(0.9),) * 3,
            log_multiplier_bounds=(-0.2, 0.2),
            coordinate_passes=2,
            golden_section_iterations=8,
            gauss_newton_iterations=6,
            temperature_standard_deviation=config.temperature_standard_deviation,
            voltage_standard_deviation=config.voltage_standard_deviation,
        ),
    )
    conventional_multipliers = tuple(
        value / baseline
        for value, baseline in zip(
            conventional.fitted_values, baseline_property.values
        )
    )
    conventional_initial = conventional.evaluations[0].mean_normalized_squared_error
    conventional_final = conventional.mean_normalized_squared_error
    conventional_reduction = _loss_reduction_fraction(
        conventional_initial, conventional_final
    )
    conventional_error = _maximum_multiplier_error(conventional_multipliers)
    lower_bound, upper_bound = (-0.2, 0.2)
    bound_tolerance = 1.0e-6
    conventional_bound_hit = any(
        abs(value - lower_bound) <= bound_tolerance
        or abs(value - upper_bound) <= bound_tolerance
        for value in conventional.log_multipliers
    )
    conventional_reasons = recovery_failure_reasons(
        maximum_multiplier_error=conventional_error,
        initial_normalized_loss=conventional_initial,
        final_normalized_loss=conventional_final,
        criteria=config.criteria,
    )

    try:
        from ..pinn.distributed_inverse import (
            InverseDistributedPropertyConfig,
            train_multi_experiment_inverse_distributed_property_pinn,
        )

        training = train_multi_experiment_inverse_distributed_property_pinn(
            truth_experiments,
            observations,
            InverseDistributedPropertyConfig(
                property_name="electrical_resistivity",
                hidden_width=20,
                hidden_layers=3,
                interior_space_points=7,
                time_points=18,
                voltage_space_points=16,
                epochs=config.inverse_pinn_epochs,
                network_learning_rate=2.0e-3,
                property_learning_rate=2.0e-3,
                initial_log_multipliers=(math.log(0.9),) * 3,
                smoothness_weight=1.0e-4,
                temperature_standard_deviation=(
                    config.temperature_standard_deviation
                ),
                voltage_standard_deviation=config.voltage_standard_deviation,
                seed=seeds.neural,
                device="cpu",
            ),
            baseline_material=baseline_experiments[0].material,
        )
        pinn_multipliers = tuple(
            value / baseline
            for value, baseline in zip(
                training.history.property_values[-1], baseline_property.values
            )
        )
        pinn_initial = training.history.total_loss[0]
        pinn_final = training.history.total_loss[-1]
        pinn_reduction = _loss_reduction_fraction(pinn_initial, pinn_final)
        pinn_error = _maximum_multiplier_error(pinn_multipliers)
        pinn_reasons = recovery_failure_reasons(
            maximum_multiplier_error=pinn_error,
            initial_normalized_loss=pinn_initial,
            final_normalized_loss=pinn_final,
            criteria=config.criteria,
        )
    except (FloatingPointError, RuntimeError) as error:
        pinn_multipliers = None
        pinn_initial = None
        pinn_final = None
        pinn_reduction = None
        pinn_error = math.inf
        pinn_reasons = (f"training_exception:{type(error).__name__}",)

    return DistributedInverseRobustnessTrial(
        trial_index=trial_index,
        seeds=seeds,
        conventional_multipliers=conventional_multipliers,
        conventional_initial_normalized_loss=conventional_initial,
        conventional_final_normalized_loss=conventional_final,
        conventional_loss_reduction_fraction=conventional_reduction,
        conventional_maximum_absolute_multiplier_error=conventional_error,
        conventional_reached_search_bound=conventional_bound_hit,
        conventional_success=not conventional_reasons,
        conventional_failure_reasons=conventional_reasons,
        inverse_pinn_multipliers=pinn_multipliers,
        inverse_pinn_initial_normalized_loss=pinn_initial,
        inverse_pinn_final_normalized_loss=pinn_final,
        inverse_pinn_loss_reduction_fraction=pinn_reduction,
        inverse_pinn_maximum_absolute_multiplier_error=pinn_error,
        inverse_pinn_success=not pinn_reasons,
        inverse_pinn_failure_reasons=pinn_reasons,
    )


def summarize_distributed_inverse_robustness(
    trials: Sequence[DistributedInverseRobustnessTrial],
) -> DistributedInverseRobustnessSummary:
    """Summarize every trial without discarding completed threshold failures."""

    trials = tuple(trials)
    if not trials:
        raise ValueError("at least one distributed robustness trial is required")
    completed_pinn = tuple(
        trial
        for trial in trials
        if trial.inverse_pinn_multipliers is not None
    )
    conventional_mean = tuple(
        fmean(trial.conventional_multipliers[index] for trial in trials)
        for index in range(3)
    )
    pinn_mean = (
        tuple(
            fmean(trial.inverse_pinn_multipliers[index] for trial in completed_pinn)
            for index in range(3)
        )
        if completed_pinn
        else None
    )
    conventional_rmse = math.sqrt(
        fmean(
            (value - truth) ** 2
            for trial in trials
            for value, truth in zip(
                trial.conventional_multipliers, RHO_TRUTH_MULTIPLIERS
            )
        )
    )
    pinn_rmse = (
        math.sqrt(
            fmean(
                (value - truth) ** 2
                for trial in completed_pinn
                for value, truth in zip(
                    trial.inverse_pinn_multipliers, RHO_TRUTH_MULTIPLIERS
                )
            )
        )
        if completed_pinn
        else None
    )
    return DistributedInverseRobustnessSummary(
        trial_count=len(trials),
        conventional_success_count=sum(trial.conventional_success for trial in trials),
        inverse_pinn_success_count=sum(trial.inverse_pinn_success for trial in trials),
        inverse_pinn_completed_count=len(completed_pinn),
        conventional_search_bound_hits=sum(
            trial.conventional_reached_search_bound for trial in trials
        ),
        conventional_mean_multipliers=conventional_mean,
        inverse_pinn_mean_multipliers=pinn_mean,
        conventional_multiplier_rmse=conventional_rmse,
        inverse_pinn_multiplier_rmse=pinn_rmse,
        conventional_worst_trial_error=max(
            trial.conventional_maximum_absolute_multiplier_error for trial in trials
        ),
        inverse_pinn_worst_trial_error=max(
            trial.inverse_pinn_maximum_absolute_multiplier_error for trial in trials
        ),
    )


def run_distributed_inverse_robustness_study(
    config: DistributedInverseRobustnessConfig = (
        DistributedInverseRobustnessConfig()
    ),
) -> DistributedInverseRobustnessStudyResult:
    """Run all declared trials and retain their original order and status."""

    trials = tuple(
        run_distributed_inverse_robustness_trial(index, config)
        for index in range(config.trial_count)
    )
    return DistributedInverseRobustnessStudyResult(
        config=config,
        truth_multipliers=RHO_TRUTH_MULTIPLIERS,
        trials=trials,
        summary=summarize_distributed_inverse_robustness(trials),
    )
