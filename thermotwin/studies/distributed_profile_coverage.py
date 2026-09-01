"""Nonlinear profiles and repeated local-interval coverage for rho_e(T)."""

from dataclasses import dataclass, replace
import math
from statistics import fmean
from typing import Callable, NamedTuple, Optional, Sequence, Tuple

from ..inference.distributed_profile_likelihood import (
    DistributedProfileInterval,
    DistributedProfileLikelihoodConfig,
    DistributedProfileLikelihoodResult,
    fit_distributed_property_profile_likelihood,
    local_profile_approximation,
    profile_interval_contains,
)
from ..inference.distributed_properties import (
    DistributedPropertyFitConfig,
    DistributedPropertyFitResult,
    fit_distributed_property,
)
from ..inference.distributed_regularization import (
    mean_square_magnitude,
    second_difference_roughness,
)
from ..observations.distributed import (
    DistributedObservationChannels,
    DistributedObservationSet,
    add_distributed_gaussian_noise,
)
from ..physics.distributed import PiecewiseLinearProperty
from ..simulation.distributed import (
    DistributedLegExperiment,
    distributed_inverse_constant_experiments,
)
from ..simulation.distributed_independent import (
    PolynomialTemperatureProperty,
    observe_independent_distributed_result,
    run_independent_distributed_experiment,
)
from .distributed_independent_validation import (
    DistributedMismatchPredictionMetrics,
    evaluate_independent_prediction,
    smooth_resistivity_truth,
)


ESTIMATOR_NAMES = (
    "conventional_unregularized",
    "conventional_regularized",
    "pinn_unregularized",
    "pinn_regularized",
)


@dataclass(frozen=True)
class DistributedProfileCoverageConfig:
    """Frozen settings for nonlinear profiles and repeated interval coverage."""

    trial_count: int = 20
    pinn_trial_count: int = 10
    first_seed: int = 49_001
    inverse_pinn_epochs: int = 400
    truth_node_count: int = 25
    truth_time_step: float = 2.5e-4
    observation_interval: float = 0.08
    temperature_standard_deviation: float = 0.01
    voltage_standard_deviation: float = 1.0e-5
    log_multiplier_bounds: Tuple[float, float] = (-0.3, 0.3)
    smoothness_weight: float = 0.8
    shrinkage_weight: float = 0.9
    representative_profile_points: int = 5
    initial_log_multiplier_sets: Tuple[Tuple[float, ...], ...] = (
        (math.log(0.8), 0.0, math.log(1.2)),
        (0.0, 0.0, 0.0),
        (math.log(1.2), 0.0, math.log(0.8)),
    )

    def __post_init__(self) -> None:
        for name, value in (
            ("trial count", self.trial_count),
            ("PINN trial count", self.pinn_trial_count),
            ("first seed", self.first_seed),
            ("inverse PINN epochs", self.inverse_pinn_epochs),
            ("truth node count", self.truth_node_count),
            ("representative profile points", self.representative_profile_points),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if self.trial_count <= 0 or self.inverse_pinn_epochs <= 0:
            raise ValueError("trial count and PINN epochs must be positive")
        if self.pinn_trial_count < 0 or self.pinn_trial_count > self.trial_count:
            raise ValueError("PINN trial count must lie between zero and all trials")
        if self.first_seed < 0:
            raise ValueError("first seed must be nonnegative")
        if self.truth_node_count < 5 or self.representative_profile_points < 3:
            raise ValueError("truth nodes and representative profile grid are too small")
        for name, value in (
            ("truth time step", self.truth_time_step),
            ("observation interval", self.observation_interval),
            (
                "temperature standard deviation",
                self.temperature_standard_deviation,
            ),
            ("voltage standard deviation", self.voltage_standard_deviation),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (
            ("smoothness weight", self.smoothness_weight),
            ("shrinkage weight", self.shrinkage_weight),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        lower, upper = self.log_multiplier_bounds
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("log-multiplier bounds must be finite and ordered")
        if len(self.initial_log_multiplier_sets) < 2:
            raise ValueError("coverage requires at least two initial curves")
        for values in self.initial_log_multiplier_sets:
            if len(values) != 3 or any(
                not math.isfinite(value) or value < lower or value > upper
                for value in values
            ):
                raise ValueError("initial curves must contain three in-bounds logs")


class DistributedProfileCoverageSeeds(NamedTuple):
    neural: int
    observations: Tuple[int, ...]


class DistributedRepresentativeProfile(NamedTuple):
    case_name: str
    estimator_name: str
    result: DistributedProfileLikelihoodResult


class DistributedProfileCoverageEstimatorResult(NamedTuple):
    name: str
    multipliers: Optional[Tuple[float, ...]]
    property_relative_rmse: float
    property_maximum_relative_error: float
    normalized_observation_loss: float
    intervals_68: Tuple[DistributedProfileInterval, ...]
    intervals_95: Tuple[DistributedProfileInterval, ...]
    coefficient_coverage_68: Tuple[bool, ...]
    coefficient_coverage_95: Tuple[bool, ...]
    holdout_internal_temperature_rmse: float
    holdout_voltage_rmse: float


class DistributedProfileCoverageTrial(NamedTuple):
    trial_index: int
    seeds: DistributedProfileCoverageSeeds
    estimators: Tuple[DistributedProfileCoverageEstimatorResult, ...]


class DistributedProfileCoverageSummary(NamedTuple):
    name: str
    completed_count: int
    interval_trial_count: int
    mean_property_relative_rmse: float
    mean_property_maximum_relative_error: float
    mean_holdout_internal_temperature_rmse: float
    mean_holdout_voltage_rmse: float
    coefficient_coverage_68: float
    coefficient_coverage_95: float
    simultaneous_coverage_68: float
    simultaneous_coverage_95: float
    finite_interval_fraction_95: float
    mean_log_interval_width_95: float


class DistributedProfileCoverageStudyResult(NamedTuple):
    config: DistributedProfileCoverageConfig
    truth_property: PolynomialTemperatureProperty
    truth_knot_multipliers: Tuple[float, ...]
    representative_profiles: Tuple[DistributedRepresentativeProfile, ...]
    trials: Tuple[DistributedProfileCoverageTrial, ...]
    summaries: Tuple[DistributedProfileCoverageSummary, ...]


def distributed_profile_coverage_seeds(
    first_seed: int, trial_index: int, experiment_count: int
) -> DistributedProfileCoverageSeeds:
    """Return disjoint neural/observation seed blocks for every trial."""

    for name, value in (
        ("first seed", first_seed),
        ("trial index", trial_index),
        ("experiment count", experiment_count),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
    if first_seed < 0 or trial_index < 0 or experiment_count <= 0:
        raise ValueError("seed inputs must be nonnegative and count positive")
    block = experiment_count + 1
    start = first_seed + trial_index * block
    return DistributedProfileCoverageSeeds(
        neural=start,
        observations=tuple(start + 1 + index for index in range(experiment_count)),
    )


def _truth_experiment(
    experiment: DistributedLegExperiment,
    truth_property: PolynomialTemperatureProperty,
) -> DistributedLegExperiment:
    return replace(
        experiment,
        material=replace(
            experiment.material,
            electrical_resistivity=truth_property,
        ),
    )


def _fit_config(
    config: DistributedProfileCoverageConfig,
    *,
    regularized: bool,
    channels: DistributedObservationChannels = DistributedObservationChannels(),
) -> DistributedPropertyFitConfig:
    return DistributedPropertyFitConfig(
        property_name="electrical_resistivity",
        observation_interval=config.observation_interval,
        channels=channels,
        log_multiplier_bounds=config.log_multiplier_bounds,
        coordinate_passes=1,
        golden_section_iterations=6,
        gauss_newton_iterations=4,
        temperature_standard_deviation=config.temperature_standard_deviation,
        voltage_standard_deviation=config.voltage_standard_deviation,
        smoothness_weight=config.smoothness_weight if regularized else 0.0,
        shrinkage_weight=config.shrinkage_weight if regularized else 0.0,
    )


def _fit_score(
    fit: DistributedPropertyFitResult,
    observation_count: int,
    fit_config: DistributedPropertyFitConfig,
) -> float:
    return observation_count * (
        fit.mean_normalized_squared_error
        + fit_config.smoothness_weight
        * float(second_difference_roughness(fit.log_multipliers))
        + fit_config.shrinkage_weight
        * float(mean_square_magnitude(fit.log_multipliers))
    )


def _best_conventional_fit(
    experiments: Sequence[DistributedLegExperiment],
    observations: Sequence[DistributedObservationSet],
    fit_config: DistributedPropertyFitConfig,
    starts: Sequence[Sequence[float]],
) -> DistributedPropertyFitResult:
    observation_count = sum(len(dataset.observations) for dataset in observations)
    candidates = tuple(
        fit_distributed_property(
            experiments,
            observations,
            replace(fit_config, initial_log_multipliers=tuple(start)),
        )
        for start in starts
    )
    return min(
        candidates,
        key=lambda fit: _fit_score(fit, observation_count, fit_config),
    )


def _fit_pinn(
    experiments: Sequence[DistributedLegExperiment],
    observations: Sequence[DistributedObservationSet],
    config: DistributedProfileCoverageConfig,
    *,
    regularized: bool,
    seed: int,
) -> Tuple[Optional[Tuple[float, ...]], float]:
    try:
        from ..pinn.distributed_inverse import (
            InverseDistributedPropertyConfig,
            train_multi_experiment_inverse_distributed_property_pinn,
        )

        training = train_multi_experiment_inverse_distributed_property_pinn(
            experiments,
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
                smoothness_weight=(config.smoothness_weight if regularized else 0.0),
                shrinkage_weight=(config.shrinkage_weight if regularized else 0.0),
                temperature_standard_deviation=(
                    config.temperature_standard_deviation
                ),
                voltage_standard_deviation=config.voltage_standard_deviation,
                seed=seed,
                device="cpu",
            ),
            baseline_material=experiments[0].material,
        )
        baseline = experiments[0].material.electrical_resistivity
        if not isinstance(baseline, PiecewiseLinearProperty):
            raise TypeError("PINN baseline resistivity must be piecewise linear")
        return (
            tuple(
                value / base
                for value, base in zip(
                    training.history.property_values[-1], baseline.values
                )
            ),
            training.history.observation_loss[-1],
        )
    except (FloatingPointError, RuntimeError):
        return None, math.inf


def _property_errors(
    baseline: PiecewiseLinearProperty,
    multipliers: Sequence[float],
    truth: PolynomialTemperatureProperty,
) -> Tuple[float, float]:
    fitted = baseline.with_values(
        tuple(value * multiplier for value, multiplier in zip(baseline.values, multipliers))
    )
    temperatures = tuple(285.0 + 0.5 * index for index in range(61))
    errors = tuple(
        (fitted.value(temperature) - truth.value(temperature))
        / truth.value(temperature)
        for temperature in temperatures
    )
    return (
        math.sqrt(fmean(value * value for value in errors)),
        max(abs(value) for value in errors),
    )


def _holdout_metrics(
    holdout: DistributedLegExperiment,
    truth_experiment: DistributedLegExperiment,
    truth_result,
    baseline_property: PiecewiseLinearProperty,
    multipliers: Sequence[float],
) -> DistributedMismatchPredictionMetrics:
    fitted_property = baseline_property.with_values(
        tuple(
            value * multiplier
            for value, multiplier in zip(baseline_property.values, multipliers)
        )
    )
    fitted_experiment = replace(
        holdout,
        material=replace(holdout.material, electrical_resistivity=fitted_property),
    )
    from ..simulation.distributed import run_distributed_leg_experiment

    return evaluate_independent_prediction(
        fitted_experiment,
        run_distributed_leg_experiment(fitted_experiment),
        truth_experiment,
        truth_result,
    )


def _estimator_result(
    *,
    name: str,
    multipliers: Optional[Sequence[float]],
    loss: float,
    intervals: Sequence[Sequence[DistributedProfileInterval]],
    truth_logs: Sequence[float],
    baseline_property: PiecewiseLinearProperty,
    truth_property: PolynomialTemperatureProperty,
    holdout: DistributedLegExperiment,
    holdout_truth_experiment: DistributedLegExperiment,
    holdout_truth_result,
) -> DistributedProfileCoverageEstimatorResult:
    if multipliers is None:
        return DistributedProfileCoverageEstimatorResult(
            name, None, *(math.inf,) * 3, (), (), (), (), math.inf, math.inf
        )
    values = tuple(float(value) for value in multipliers)
    property_rmse, property_maximum = _property_errors(
        baseline_property, values, truth_property
    )
    metrics = _holdout_metrics(
        holdout,
        holdout_truth_experiment,
        holdout_truth_result,
        baseline_property,
        values,
    )
    intervals = tuple(tuple(item) for item in intervals)
    intervals_68 = tuple(items[0] for items in intervals) if intervals else ()
    intervals_95 = tuple(items[-1] for items in intervals) if intervals else ()
    return DistributedProfileCoverageEstimatorResult(
        name=name,
        multipliers=values,
        property_relative_rmse=property_rmse,
        property_maximum_relative_error=property_maximum,
        normalized_observation_loss=float(loss),
        intervals_68=intervals_68,
        intervals_95=intervals_95,
        coefficient_coverage_68=tuple(
            profile_interval_contains(interval, truth)
            for interval, truth in zip(intervals_68, truth_logs)
        ),
        coefficient_coverage_95=tuple(
            profile_interval_contains(interval, truth)
            for interval, truth in zip(intervals_95, truth_logs)
        ),
        holdout_internal_temperature_rmse=metrics.internal_temperature_rmse,
        holdout_voltage_rmse=metrics.voltage_rmse,
    )


def summarize_distributed_profile_coverage(
    trials: Sequence[DistributedProfileCoverageTrial],
) -> Tuple[DistributedProfileCoverageSummary, ...]:
    """Summarize point accuracy, transfer, and empirical interval coverage."""

    summaries = []
    for name in ESTIMATOR_NAMES:
        results = tuple(
            estimator
            for trial in trials
            for estimator in trial.estimators
            if estimator.name == name and estimator.multipliers is not None
        )
        interval_results = tuple(item for item in results if item.intervals_95)
        coverage_68 = tuple(
            value for item in interval_results for value in item.coefficient_coverage_68
        )
        coverage_95 = tuple(
            value for item in interval_results for value in item.coefficient_coverage_95
        )
        intervals_95 = tuple(
            interval for item in interval_results for interval in item.intervals_95
        )
        summaries.append(
            DistributedProfileCoverageSummary(
                name=name,
                completed_count=len(results),
                interval_trial_count=len(interval_results),
                mean_property_relative_rmse=(
                    fmean(item.property_relative_rmse for item in results)
                    if results else math.inf
                ),
                mean_property_maximum_relative_error=(
                    fmean(item.property_maximum_relative_error for item in results)
                    if results else math.inf
                ),
                mean_holdout_internal_temperature_rmse=(
                    fmean(item.holdout_internal_temperature_rmse for item in results)
                    if results else math.inf
                ),
                mean_holdout_voltage_rmse=(
                    fmean(item.holdout_voltage_rmse for item in results)
                    if results else math.inf
                ),
                coefficient_coverage_68=(
                    fmean(coverage_68) if coverage_68 else math.nan
                ),
                coefficient_coverage_95=(
                    fmean(coverage_95) if coverage_95 else math.nan
                ),
                simultaneous_coverage_68=(
                    fmean(all(item.coefficient_coverage_68) for item in interval_results)
                    if interval_results else math.nan
                ),
                simultaneous_coverage_95=(
                    fmean(all(item.coefficient_coverage_95) for item in interval_results)
                    if interval_results else math.nan
                ),
                finite_interval_fraction_95=(
                    fmean(
                        not interval.lower_hits_bound and not interval.upper_hits_bound
                        for interval in intervals_95
                    )
                    if intervals_95 else math.nan
                ),
                mean_log_interval_width_95=(
                    fmean(
                        interval.upper_log_multiplier - interval.lower_log_multiplier
                        for interval in intervals_95
                    )
                    if intervals_95 else math.nan
                ),
            )
        )
    return tuple(summaries)


def run_distributed_profile_coverage_study(
    config: DistributedProfileCoverageConfig = DistributedProfileCoverageConfig(),
    *,
    run_representative_profiles: bool = True,
    progress: Optional[Callable[[str], None]] = None,
) -> DistributedProfileCoverageStudyResult:
    """Run nonlinear example profiles and repeated local coverage checks."""

    notify = progress or (lambda _message: None)
    baselines = distributed_inverse_constant_experiments()[:3]
    baseline_property = baselines[0].material.electrical_resistivity
    if not isinstance(baseline_property, PiecewiseLinearProperty):
        raise TypeError("baseline resistivity must be piecewise linear")
    truth_property = smooth_resistivity_truth()
    truths = tuple(_truth_experiment(item, truth_property) for item in baselines)
    clean_observations = tuple(
        observe_independent_distributed_result(
            truth,
            run_independent_distributed_experiment(
                truth,
                node_count=config.truth_node_count,
                time_step=config.truth_time_step,
            ),
            observation_interval=config.observation_interval,
            channels=DistributedObservationChannels(),
        )
        for truth in truths
    )
    holdout = distributed_inverse_constant_experiments()[3]
    holdout_truth_experiment = _truth_experiment(holdout, truth_property)
    holdout_truth_result = run_independent_distributed_experiment(
        holdout_truth_experiment,
        node_count=config.truth_node_count,
        time_step=config.truth_time_step,
    )
    truth_multipliers = tuple(
        truth_property.value(temperature) / baseline
        for temperature, baseline in zip(
            baseline_property.temperatures, baseline_property.values
        )
    )
    truth_logs = tuple(math.log(value) for value in truth_multipliers)
    first_seeds = distributed_profile_coverage_seeds(
        config.first_seed, 0, len(baselines)
    )
    first_observations = tuple(
        add_distributed_gaussian_noise(
            dataset,
            standard_deviations={
                "cold_face_temperature": config.temperature_standard_deviation,
                "hot_face_temperature": config.temperature_standard_deviation,
                "voltage": config.voltage_standard_deviation,
            },
            seed=seed,
        )
        for dataset, seed in zip(clean_observations, first_seeds.observations)
    )

    representative_profiles = []
    if run_representative_profiles:
        profile_config = DistributedProfileLikelihoodConfig(
            profile_points=config.representative_profile_points,
            multistart_initial_log_multipliers=config.initial_log_multiplier_sets,
            profile_restart_count=1,
        )
        for name, regularized in (
            ("conventional_unregularized", False),
            ("conventional_regularized", True),
        ):
            notify(f"representative full profile: {name}")
            representative_profiles.append(
                DistributedRepresentativeProfile(
                    case_name="full_bidirectional",
                    estimator_name=name,
                    result=fit_distributed_property_profile_likelihood(
                        baselines,
                        first_observations,
                        _fit_config(config, regularized=regularized),
                        profile_config,
                    ),
                )
            )
        notify("representative weak profile: positive_temperature_voltage")
        representative_profiles.append(
            DistributedRepresentativeProfile(
                case_name="positive_temperature_voltage",
                estimator_name="conventional_unregularized",
                result=fit_distributed_property_profile_likelihood(
                    (baselines[1],),
                    (first_observations[1],),
                    _fit_config(config, regularized=False),
                    profile_config,
                ),
            )
        )

    trials = []
    for trial_index in range(config.trial_count):
        seeds = distributed_profile_coverage_seeds(
            config.first_seed, trial_index, len(baselines)
        )
        observations = tuple(
            add_distributed_gaussian_noise(
                dataset,
                standard_deviations={
                    "cold_face_temperature": config.temperature_standard_deviation,
                    "hot_face_temperature": config.temperature_standard_deviation,
                    "voltage": config.voltage_standard_deviation,
                },
                seed=seed,
            )
            for dataset, seed in zip(clean_observations, seeds.observations)
        )
        estimators = []
        for name, regularized in (
            ("conventional_unregularized", False),
            ("conventional_regularized", True),
        ):
            fit_config = _fit_config(config, regularized=regularized)
            fit = _best_conventional_fit(
                baselines,
                observations,
                fit_config,
                config.initial_log_multiplier_sets,
            )
            local = local_profile_approximation(
                baselines,
                fit,
                fit_config,
                DistributedProfileLikelihoodConfig(
                    profile_points=3,
                    multistart_initial_log_multipliers=(
                        config.initial_log_multiplier_sets[0],
                    ),
                ),
            )
            estimators.append(
                _estimator_result(
                    name=name,
                    multipliers=tuple(
                        value / base
                        for value, base in zip(
                            fit.fitted_values, baseline_property.values
                        )
                    ),
                    loss=fit.mean_normalized_squared_error,
                    intervals=local.intervals_by_coefficient,
                    truth_logs=truth_logs,
                    baseline_property=baseline_property,
                    truth_property=truth_property,
                    holdout=holdout,
                    holdout_truth_experiment=holdout_truth_experiment,
                    holdout_truth_result=holdout_truth_result,
                )
            )
        if trial_index < config.pinn_trial_count:
            for name, regularized in (
                ("pinn_unregularized", False),
                ("pinn_regularized", True),
            ):
                multipliers, loss = _fit_pinn(
                    baselines,
                    observations,
                    config,
                    regularized=regularized,
                    seed=seeds.neural,
                )
                estimators.append(
                    _estimator_result(
                        name=name,
                        multipliers=multipliers,
                        loss=loss,
                        intervals=(),
                        truth_logs=truth_logs,
                        baseline_property=baseline_property,
                        truth_property=truth_property,
                        holdout=holdout,
                        holdout_truth_experiment=holdout_truth_experiment,
                        holdout_truth_result=holdout_truth_result,
                    )
                )
        estimators.sort(key=lambda item: ESTIMATOR_NAMES.index(item.name))
        trials.append(
            DistributedProfileCoverageTrial(
                trial_index=trial_index,
                seeds=seeds,
                estimators=tuple(estimators),
            )
        )
        notify(f"coverage trial {trial_index + 1}/{config.trial_count} complete")
    trials_tuple = tuple(trials)
    return DistributedProfileCoverageStudyResult(
        config=config,
        truth_property=truth_property,
        truth_knot_multipliers=truth_multipliers,
        representative_profiles=tuple(representative_profiles),
        trials=trials_tuple,
        summaries=summarize_distributed_profile_coverage(trials_tuple),
    )
