"""Independent-truth and matched-regularization distributed inverse study."""

from dataclasses import dataclass, replace
import math
from statistics import fmean
from typing import NamedTuple, Optional, Sequence, Tuple

from ..core.controls import PiecewiseConstantCurrent
from ..inference.distributed_properties import (
    DistributedPropertyFitConfig,
    fit_distributed_property,
)
from ..inference.distributed_regularization import second_difference_roughness
from ..observations.distributed import (
    DistributedObservationChannels,
    DistributedObservationSet,
    add_distributed_gaussian_noise,
)
from ..physics.distributed import PiecewiseLinearProperty
from ..simulation.distributed import (
    DistributedExperimentResult,
    DistributedLegExperiment,
    distributed_inverse_constant_experiments,
    run_distributed_leg_experiment,
)
from ..simulation.distributed_independent import (
    IndependentDistributedResult,
    PolynomialTemperatureProperty,
    independent_distributed_voltage,
    interpolate_independent_position,
    interpolate_independent_temperature,
    observe_independent_distributed_result,
    run_independent_distributed_experiment,
)


ESTIMATOR_NAMES = (
    "conventional_unregularized",
    "conventional_matched",
    "pinn_unregularized",
    "pinn_matched",
)


@dataclass(frozen=True)
class DistributedIndependentValidationCriteria:
    """Predeclared gates for in-support property and prediction transfer."""

    maximum_property_relative_error: float = 0.10
    maximum_face_temperature_rmse: float = 0.03
    maximum_internal_temperature_rmse: float = 0.03
    maximum_voltage_rmse: float = 5.0e-5
    maximum_absolute_temperature_error: float = 0.08
    maximum_energy_balance_residual: float = 1.0e-10

    def __post_init__(self) -> None:
        for name, value in (
            ("property relative-error limit", self.maximum_property_relative_error),
            ("face-temperature RMSE limit", self.maximum_face_temperature_rmse),
            (
                "internal-temperature RMSE limit",
                self.maximum_internal_temperature_rmse,
            ),
            ("voltage RMSE limit", self.maximum_voltage_rmse),
            (
                "maximum temperature-error limit",
                self.maximum_absolute_temperature_error,
            ),
            ("energy-residual limit", self.maximum_energy_balance_residual),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class DistributedIndependentValidationConfig:
    """Frozen numerical truth, noise, estimator, and gate settings."""

    trial_count: int = 3
    first_seed: int = 47_001
    inverse_pinn_epochs: int = 600
    truth_node_count: int = 25
    truth_time_step: float = 2.5e-4
    observation_interval: float = 0.08
    temperature_standard_deviation: float = 0.01
    voltage_standard_deviation: float = 1.0e-5
    matched_smoothness_weight: float = 25.0
    criteria: DistributedIndependentValidationCriteria = (
        DistributedIndependentValidationCriteria()
    )

    def __post_init__(self) -> None:
        for name, value in (
            ("trial count", self.trial_count),
            ("first seed", self.first_seed),
            ("inverse PINN epochs", self.inverse_pinn_epochs),
            ("truth node count", self.truth_node_count),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if self.trial_count <= 0 or self.inverse_pinn_epochs <= 0:
            raise ValueError("trial count and PINN epochs must be positive")
        if self.first_seed < 0:
            raise ValueError("first seed must be nonnegative")
        if self.truth_node_count < 5:
            raise ValueError("truth node count must be at least five")
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
        if (
            not math.isfinite(self.matched_smoothness_weight)
            or self.matched_smoothness_weight <= 0.0
        ):
            raise ValueError("matched smoothness weight must be finite and positive")
        if not isinstance(self.criteria, DistributedIndependentValidationCriteria):
            raise ValueError("criteria must be independent-validation criteria")


class DistributedIndependentValidationSeeds(NamedTuple):
    neural: int
    observations: Tuple[int, ...]


class DistributedMismatchPredictionMetrics(NamedTuple):
    cold_face_rmse: float
    hot_face_rmse: float
    internal_temperature_rmse: float
    voltage_rmse: float
    maximum_absolute_temperature_error: float
    maximum_prediction_energy_balance_residual: float


class DistributedIndependentHoldout(NamedTuple):
    name: str
    baseline_experiment: DistributedLegExperiment
    truth_experiment: DistributedLegExperiment
    truth_result: IndependentDistributedResult
    counts_toward_gate: bool


class DistributedIndependentEstimatorResult(NamedTuple):
    name: str
    smoothness_weight: float
    multipliers: Optional[Tuple[float, ...]]
    final_normalized_observation_loss: float
    log_multiplier_roughness: float
    in_support_property_relative_rmse: float
    in_support_property_maximum_relative_error: float
    extended_property_relative_rmse: float
    predictions: Tuple[Tuple[str, DistributedMismatchPredictionMetrics], ...]
    success: bool
    failure_reasons: Tuple[str, ...]


class DistributedIndependentValidationTrial(NamedTuple):
    trial_index: int
    seeds: DistributedIndependentValidationSeeds
    estimators: Tuple[DistributedIndependentEstimatorResult, ...]


class DistributedIndependentEstimatorSummary(NamedTuple):
    name: str
    success_count: int
    completed_count: int
    mean_in_support_property_relative_rmse: float
    mean_in_support_property_maximum_relative_error: float
    mean_extended_property_relative_rmse: float
    mean_in_support_internal_temperature_rmse: float
    mean_in_support_voltage_rmse: float


class DistributedIndependentValidationStudyResult(NamedTuple):
    config: DistributedIndependentValidationConfig
    truth_property: PolynomialTemperatureProperty
    truth_knot_multipliers: Tuple[float, ...]
    training_experiment_names: Tuple[str, ...]
    holdout_names: Tuple[str, ...]
    trials: Tuple[DistributedIndependentValidationTrial, ...]
    summaries: Tuple[DistributedIndependentEstimatorSummary, ...]


def smooth_resistivity_truth() -> PolynomialTemperatureProperty:
    """Return a cubic truth that matches the old knot truth but curves between it."""

    # z = (T - 300 K) / 15 K.  This expands
    # 1e-5 * [1.07 - 0.0516 z - 0.0296 z^2 + 0.03 z(z^2 - 1)].
    # The final term is zero at every inference knot and deliberately
    # introduces unresolved between-knot shape.
    return PolynomialTemperatureProperty(
        reference_temperature=300.0,
        temperature_scale=15.0,
        coefficients=(1.0700e-5, -0.0816e-5, -0.0296e-5, 0.0300e-5),
    )


def independent_validation_seeds(
    first_seed: int, trial_index: int, experiment_count: int
) -> DistributedIndependentValidationSeeds:
    """Pair both PINN variants on one neural initialization per trial."""

    for name, value in (
        ("first seed", first_seed),
        ("trial index", trial_index),
        ("experiment count", experiment_count),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
    if first_seed < 0 or trial_index < 0 or experiment_count <= 0:
        raise ValueError("seeds must be nonnegative and experiment count positive")
    block_size = experiment_count + 1
    start = first_seed + trial_index * block_size
    return DistributedIndependentValidationSeeds(
        neural=start,
        observations=tuple(start + 1 + index for index in range(experiment_count)),
    )


def _with_truth_property(
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


def distributed_independent_validation_experiments(
    config: DistributedIndependentValidationConfig,
) -> Tuple[
    Tuple[str, ...],
    Tuple[DistributedLegExperiment, ...],
    Tuple[DistributedLegExperiment, ...],
    Tuple[DistributedIndependentHoldout, ...],
    PiecewiseLinearProperty,
    PolynomialTemperatureProperty,
]:
    """Build three fitting regimes and three distinct transfer challenges."""

    baseline_suite = distributed_inverse_constant_experiments()
    baseline_property = baseline_suite[0].material.electrical_resistivity
    if not isinstance(baseline_property, PiecewiseLinearProperty):
        raise TypeError("baseline resistivity must be piecewise linear")
    truth_property = smooth_resistivity_truth()
    training_names = (
        "zero_current_20K_relaxation",
        "positive_0.8A_10K_lift",
        "negative_0.8A_10K_lift",
    )
    training_baselines = baseline_suite[:3]
    training_truths = tuple(
        _with_truth_property(experiment, truth_property)
        for experiment in training_baselines
    )

    lifted = baseline_suite[3]
    pulse = replace(
        lifted,
        current=PiecewiseConstantCurrent.pulse(
            start_time=0.08,
            end_time=0.28,
            pulse_current=-0.8,
        ),
    )
    outside_support = replace(
        lifted,
        initial_cold_face_temperature=280.0,
        initial_hot_face_temperature=320.0,
        cold_reservoir_temperature=280.0,
        hot_reservoir_temperature=320.0,
    )
    holdout_definitions = (
        ("constant_positive_0.4A_20K", lifted, True),
        ("negative_0.8A_20K_pulse", pulse, True),
        ("outside_support_positive_0.4A_40K", outside_support, False),
    )
    holdouts = tuple(
        DistributedIndependentHoldout(
            name=name,
            baseline_experiment=baseline,
            truth_experiment=_with_truth_property(baseline, truth_property),
            truth_result=run_independent_distributed_experiment(
                _with_truth_property(baseline, truth_property),
                node_count=config.truth_node_count,
                time_step=config.truth_time_step,
            ),
            counts_toward_gate=counts,
        )
        for name, baseline, counts in holdout_definitions
    )
    return (
        training_names,
        training_baselines,
        training_truths,
        holdouts,
        baseline_property,
        truth_property,
    )


def _rmse(errors: Sequence[float]) -> float:
    errors = tuple(errors)
    if not errors:
        raise ValueError("RMSE requires at least one error")
    return math.sqrt(fmean(value * value for value in errors))


def evaluate_independent_prediction(
    prediction_experiment: DistributedLegExperiment,
    prediction: DistributedExperimentResult,
    truth_experiment: DistributedLegExperiment,
    truth: IndependentDistributedResult,
) -> DistributedMismatchPredictionMetrics:
    """Compare finite-volume predictions against independent nodal truth."""

    cell_positions = tuple(
        prediction_experiment.geometry.length
        * (index + 0.5)
        / prediction_experiment.cell_count
        for index in range(prediction_experiment.cell_count)
    )
    cold_errors = []
    hot_errors = []
    internal_errors = []
    voltage_errors = []
    all_temperature_errors = []
    for index, time in enumerate(prediction.trajectory.time):
        truth_state = interpolate_independent_temperature(truth.trajectory, time)
        cold_error = prediction.trajectory.cold_face[index] - truth_state[0]
        hot_error = prediction.trajectory.hot_face[index] - truth_state[-1]
        cold_errors.append(cold_error)
        hot_errors.append(hot_error)
        all_temperature_errors.extend((cold_error, hot_error))
        for position, predicted in zip(
            cell_positions, prediction.trajectory.cells[index]
        ):
            target = interpolate_independent_position(
                truth.trajectory.positions, truth_state, position
            )
            error = predicted - target
            internal_errors.append(error)
            all_temperature_errors.append(error)
        # Current is known exactly and right-continuous, so use the prediction
        # diagnostic's current rather than interpolating a discontinuous truth.
        truth_voltage = independent_distributed_voltage(
            truth_experiment,
            truth_state,
            current=prediction.diagnostics.current[index],
        )
        voltage_errors.append(prediction.diagnostics.voltage[index] - truth_voltage)
    return DistributedMismatchPredictionMetrics(
        cold_face_rmse=_rmse(cold_errors),
        hot_face_rmse=_rmse(hot_errors),
        internal_temperature_rmse=_rmse(internal_errors),
        voltage_rmse=_rmse(voltage_errors),
        maximum_absolute_temperature_error=max(
            abs(value) for value in all_temperature_errors
        ),
        maximum_prediction_energy_balance_residual=max(
            abs(value)
            for value in prediction.diagnostics.instantaneous_energy_balance_residual
        ),
    )


def _property_errors(
    inferred: PiecewiseLinearProperty,
    truth: PolynomialTemperatureProperty,
    temperatures: Sequence[float],
) -> Tuple[float, float]:
    relative = tuple(
        (inferred.value(temperature) - truth.value(temperature))
        / truth.value(temperature)
        for temperature in temperatures
    )
    return _rmse(relative), max(abs(value) for value in relative)


def _prediction_failure_reasons(
    name: str,
    metrics: DistributedMismatchPredictionMetrics,
    criteria: DistributedIndependentValidationCriteria,
) -> Tuple[str, ...]:
    checks = (
        ("cold_face_rmse", metrics.cold_face_rmse, criteria.maximum_face_temperature_rmse),
        ("hot_face_rmse", metrics.hot_face_rmse, criteria.maximum_face_temperature_rmse),
        (
            "internal_temperature_rmse",
            metrics.internal_temperature_rmse,
            criteria.maximum_internal_temperature_rmse,
        ),
        ("voltage_rmse", metrics.voltage_rmse, criteria.maximum_voltage_rmse),
        (
            "maximum_temperature_error",
            metrics.maximum_absolute_temperature_error,
            criteria.maximum_absolute_temperature_error,
        ),
        (
            "energy_balance_residual",
            metrics.maximum_prediction_energy_balance_residual,
            criteria.maximum_energy_balance_residual,
        ),
    )
    return tuple(
        f"{name}:{'nonfinite_' if not math.isfinite(value) else ''}{metric}"
        for metric, value, limit in checks
        if not math.isfinite(value) or value > limit
    )


def _experiment_with_multipliers(
    experiment: DistributedLegExperiment,
    baseline: PiecewiseLinearProperty,
    multipliers: Sequence[float],
) -> DistributedLegExperiment:
    inferred = baseline.with_values(
        tuple(value * multiplier for value, multiplier in zip(baseline.values, multipliers))
    )
    return replace(
        experiment,
        material=replace(experiment.material, electrical_resistivity=inferred),
    )


def _evaluate_estimator(
    *,
    name: str,
    smoothness_weight: float,
    multipliers: Optional[Sequence[float]],
    final_observation_loss: float,
    baseline_property: PiecewiseLinearProperty,
    truth_property: PolynomialTemperatureProperty,
    holdouts: Sequence[DistributedIndependentHoldout],
    criteria: DistributedIndependentValidationCriteria,
) -> DistributedIndependentEstimatorResult:
    if multipliers is None or any(not math.isfinite(value) for value in multipliers):
        return DistributedIndependentEstimatorResult(
            name=name,
            smoothness_weight=smoothness_weight,
            multipliers=None,
            final_normalized_observation_loss=math.inf,
            log_multiplier_roughness=math.inf,
            in_support_property_relative_rmse=math.inf,
            in_support_property_maximum_relative_error=math.inf,
            extended_property_relative_rmse=math.inf,
            predictions=(),
            success=False,
            failure_reasons=("training_failure",),
        )
    multipliers = tuple(float(value) for value in multipliers)
    inferred_property = baseline_property.with_values(
        tuple(
            value * multiplier
            for value, multiplier in zip(baseline_property.values, multipliers)
        )
    )
    support_temperatures = tuple(285.0 + index for index in range(31))
    extended_temperatures = tuple(280.0 + index for index in range(41))
    support_rmse, support_maximum = _property_errors(
        inferred_property, truth_property, support_temperatures
    )
    extended_rmse, _ = _property_errors(
        inferred_property, truth_property, extended_temperatures
    )
    predictions = []
    reasons = []
    if support_maximum > criteria.maximum_property_relative_error:
        reasons.append("in_support_property_error")
    for holdout in holdouts:
        prediction_experiment = _experiment_with_multipliers(
            holdout.baseline_experiment, baseline_property, multipliers
        )
        metrics = evaluate_independent_prediction(
            prediction_experiment,
            run_distributed_leg_experiment(prediction_experiment),
            holdout.truth_experiment,
            holdout.truth_result,
        )
        predictions.append((holdout.name, metrics))
        if holdout.counts_toward_gate:
            reasons.extend(
                _prediction_failure_reasons(holdout.name, metrics, criteria)
            )
    logs = tuple(math.log(value) for value in multipliers)
    return DistributedIndependentEstimatorResult(
        name=name,
        smoothness_weight=smoothness_weight,
        multipliers=multipliers,
        final_normalized_observation_loss=final_observation_loss,
        log_multiplier_roughness=float(second_difference_roughness(logs)),
        in_support_property_relative_rmse=support_rmse,
        in_support_property_maximum_relative_error=support_maximum,
        extended_property_relative_rmse=extended_rmse,
        predictions=tuple(predictions),
        success=not reasons,
        failure_reasons=tuple(reasons),
    )


def _fit_conventional(
    experiments: Sequence[DistributedLegExperiment],
    observations: Sequence[DistributedObservationSet],
    *,
    config: DistributedIndependentValidationConfig,
    smoothness_weight: float,
):
    return fit_distributed_property(
        experiments,
        observations,
        DistributedPropertyFitConfig(
            property_name="electrical_resistivity",
            observation_interval=config.observation_interval,
            channels=DistributedObservationChannels(),
            initial_log_multipliers=(math.log(0.9),) * 3,
            log_multiplier_bounds=(-0.3, 0.3),
            coordinate_passes=2,
            golden_section_iterations=8,
            gauss_newton_iterations=6,
            temperature_standard_deviation=config.temperature_standard_deviation,
            voltage_standard_deviation=config.voltage_standard_deviation,
            smoothness_weight=smoothness_weight,
        ),
    )


def _fit_pinn(
    experiments: Sequence[DistributedLegExperiment],
    observations: Sequence[DistributedObservationSet],
    *,
    config: DistributedIndependentValidationConfig,
    smoothness_weight: float,
    neural_seed: int,
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
                smoothness_weight=smoothness_weight,
                temperature_standard_deviation=(
                    config.temperature_standard_deviation
                ),
                voltage_standard_deviation=config.voltage_standard_deviation,
                seed=neural_seed,
                device="cpu",
            ),
            baseline_material=experiments[0].material,
        )
        baseline = experiments[0].material.electrical_resistivity
        if not isinstance(baseline, PiecewiseLinearProperty):
            raise TypeError("PINN baseline resistivity must be piecewise linear")
        multipliers = tuple(
            value / base
            for value, base in zip(
                training.history.property_values[-1], baseline.values
            )
        )
        return multipliers, training.history.observation_loss[-1]
    except (FloatingPointError, RuntimeError):
        return None, math.inf


def run_distributed_independent_validation_trial(
    trial_index: int,
    config: DistributedIndependentValidationConfig,
    *,
    prepared=None,
) -> DistributedIndependentValidationTrial:
    """Run four paired estimators against one independently generated dataset."""

    if prepared is None:
        prepared = distributed_independent_validation_experiments(config)
    (
        _training_names,
        training_baselines,
        training_truths,
        holdouts,
        baseline_property,
        truth_property,
    ) = prepared
    seeds = independent_validation_seeds(
        config.first_seed, trial_index, len(training_truths)
    )
    clean_observations = tuple(
        observe_independent_distributed_result(
            truth_experiment,
            run_independent_distributed_experiment(
                truth_experiment,
                node_count=config.truth_node_count,
                time_step=config.truth_time_step,
            ),
            observation_interval=config.observation_interval,
            channels=DistributedObservationChannels(),
        )
        for truth_experiment in training_truths
    )
    standard_deviations = {
        "cold_face_temperature": config.temperature_standard_deviation,
        "hot_face_temperature": config.temperature_standard_deviation,
        "voltage": config.voltage_standard_deviation,
    }
    observations = tuple(
        add_distributed_gaussian_noise(
            clean,
            standard_deviations=standard_deviations,
            seed=seed,
        )
        for clean, seed in zip(clean_observations, seeds.observations)
    )
    estimators = []
    for name, weight in (
        ("conventional_unregularized", 0.0),
        ("conventional_matched", config.matched_smoothness_weight),
    ):
        fit = _fit_conventional(
            training_baselines,
            observations,
            config=config,
            smoothness_weight=weight,
        )
        estimators.append(
            _evaluate_estimator(
                name=name,
                smoothness_weight=weight,
                multipliers=tuple(
                    value / baseline
                    for value, baseline in zip(
                        fit.fitted_values, baseline_property.values
                    )
                ),
                final_observation_loss=fit.mean_normalized_squared_error,
                baseline_property=baseline_property,
                truth_property=truth_property,
                holdouts=holdouts,
                criteria=config.criteria,
            )
        )
    for name, weight in (
        ("pinn_unregularized", 0.0),
        ("pinn_matched", config.matched_smoothness_weight),
    ):
        multipliers, loss = _fit_pinn(
            training_baselines,
            observations,
            config=config,
            smoothness_weight=weight,
            neural_seed=seeds.neural,
        )
        estimators.append(
            _evaluate_estimator(
                name=name,
                smoothness_weight=weight,
                multipliers=multipliers,
                final_observation_loss=loss,
                baseline_property=baseline_property,
                truth_property=truth_property,
                holdouts=holdouts,
                criteria=config.criteria,
            )
        )
    estimators.sort(key=lambda item: ESTIMATOR_NAMES.index(item.name))
    return DistributedIndependentValidationTrial(
        trial_index=trial_index,
        seeds=seeds,
        estimators=tuple(estimators),
    )


def summarize_distributed_independent_validation(
    trials: Sequence[DistributedIndependentValidationTrial],
    holdouts: Sequence[DistributedIndependentHoldout],
) -> Tuple[DistributedIndependentEstimatorSummary, ...]:
    """Aggregate paired trials without dropping failures."""

    trials = tuple(trials)
    holdouts = tuple(holdouts)
    if not trials:
        raise ValueError("at least one independent-validation trial is required")
    gated_names = {
        holdout.name for holdout in holdouts if holdout.counts_toward_gate
    }
    summaries = []
    for name in ESTIMATOR_NAMES:
        results = tuple(
            next(item for item in trial.estimators if item.name == name)
            for trial in trials
        )
        completed = tuple(item for item in results if item.multipliers is not None)
        internal = tuple(
            metrics.internal_temperature_rmse
            for item in completed
            for holdout_name, metrics in item.predictions
            if holdout_name in gated_names
        )
        voltage = tuple(
            metrics.voltage_rmse
            for item in completed
            for holdout_name, metrics in item.predictions
            if holdout_name in gated_names
        )
        summaries.append(
            DistributedIndependentEstimatorSummary(
                name=name,
                success_count=sum(item.success for item in results),
                completed_count=len(completed),
                mean_in_support_property_relative_rmse=(
                    fmean(item.in_support_property_relative_rmse for item in completed)
                    if completed
                    else math.inf
                ),
                mean_in_support_property_maximum_relative_error=(
                    fmean(
                        item.in_support_property_maximum_relative_error
                        for item in completed
                    )
                    if completed
                    else math.inf
                ),
                mean_extended_property_relative_rmse=(
                    fmean(item.extended_property_relative_rmse for item in completed)
                    if completed
                    else math.inf
                ),
                mean_in_support_internal_temperature_rmse=(
                    fmean(internal) if internal else math.inf
                ),
                mean_in_support_voltage_rmse=(
                    fmean(voltage) if voltage else math.inf
                ),
            )
        )
    return tuple(summaries)


def run_distributed_independent_validation_study(
    config: DistributedIndependentValidationConfig = (
        DistributedIndependentValidationConfig()
    ),
) -> DistributedIndependentValidationStudyResult:
    """Run the frozen independent-truth and paired-regularization campaign."""

    prepared = distributed_independent_validation_experiments(config)
    training_names, _, _, holdouts, baseline_property, truth_property = prepared
    trials = tuple(
        run_distributed_independent_validation_trial(
            index, config, prepared=prepared
        )
        for index in range(config.trial_count)
    )
    truth_knot_multipliers = tuple(
        truth_property.value(temperature) / baseline
        for temperature, baseline in zip(
            baseline_property.temperatures, baseline_property.values
        )
    )
    return DistributedIndependentValidationStudyResult(
        config=config,
        truth_property=truth_property,
        truth_knot_multipliers=truth_knot_multipliers,
        training_experiment_names=training_names,
        holdout_names=tuple(holdout.name for holdout in holdouts),
        trials=trials,
        summaries=summarize_distributed_independent_validation(trials, holdouts),
    )
