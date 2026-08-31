"""Whole-regime transfer validation for distributed resistivity inference."""

from dataclasses import dataclass, replace
import math
from statistics import fmean
from typing import NamedTuple, Optional, Sequence, Tuple

from ..inference.distributed_properties import (
    DistributedPropertyFitConfig,
    fit_distributed_property,
)
from ..observations.distributed import DistributedObservationChannels
from ..physics.distributed import PiecewiseLinearProperty
from ..simulation.distributed import (
    DistributedExperimentResult,
    DistributedLegExperiment,
    run_distributed_leg_experiment,
)
from .distributed_inverse_robustness import (
    RHO_TRUTH_MULTIPLIERS,
    DistributedInverseRobustnessSeeds,
    distributed_inverse_robustness_seeds,
    distributed_resistivity_truth_experiments,
    noisy_distributed_inverse_observations,
)


@dataclass(frozen=True)
class DistributedWithheldPredictionCriteria:
    """Predeclared limits for a complete noise-free withheld trajectory."""

    maximum_cold_face_rmse: float = 0.03
    maximum_hot_face_rmse: float = 0.03
    maximum_internal_temperature_rmse: float = 0.03
    maximum_voltage_rmse: float = 3.0e-5
    maximum_absolute_temperature_error: float = 0.08
    maximum_energy_balance_residual: float = 1.0e-10

    def __post_init__(self) -> None:
        for name, value in (
            ("cold-face RMSE limit", self.maximum_cold_face_rmse),
            ("hot-face RMSE limit", self.maximum_hot_face_rmse),
            ("internal-temperature RMSE limit", self.maximum_internal_temperature_rmse),
            ("voltage RMSE limit", self.maximum_voltage_rmse),
            ("absolute-temperature-error limit", self.maximum_absolute_temperature_error),
            ("energy-residual limit", self.maximum_energy_balance_residual),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class DistributedWithheldValidationConfig:
    """Frozen training noise, withheld regime, seeds, and prediction gate."""

    trial_count: int = 5
    first_seed: int = 37_001
    inverse_pinn_epochs: int = 600
    withheld_experiment_index: int = 3
    observation_interval: float = 0.08
    temperature_standard_deviation: float = 0.01
    voltage_standard_deviation: float = 1.0e-5
    criteria: DistributedWithheldPredictionCriteria = (
        DistributedWithheldPredictionCriteria()
    )

    def __post_init__(self) -> None:
        for name, value in (
            ("trial count", self.trial_count),
            ("first seed", self.first_seed),
            ("inverse PINN epochs", self.inverse_pinn_epochs),
            ("withheld experiment index", self.withheld_experiment_index),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if self.trial_count <= 0:
            raise ValueError("trial count must be positive")
        if self.first_seed < 0 or self.withheld_experiment_index < 0:
            raise ValueError("first seed and withheld index must be nonnegative")
        if self.inverse_pinn_epochs <= 0:
            raise ValueError("inverse PINN epochs must be positive")
        for name, value in (
            ("observation interval", self.observation_interval),
            ("temperature standard deviation", self.temperature_standard_deviation),
            ("voltage standard deviation", self.voltage_standard_deviation),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.criteria, DistributedWithheldPredictionCriteria):
            raise ValueError("criteria must be withheld-prediction criteria")


class DistributedWithheldPredictionMetrics(NamedTuple):
    cold_face_rmse: float
    hot_face_rmse: float
    internal_temperature_rmse: float
    voltage_rmse: float
    maximum_absolute_temperature_error: float
    maximum_energy_balance_residual: float


class DistributedWithheldValidationTrial(NamedTuple):
    trial_index: int
    seeds: DistributedInverseRobustnessSeeds
    conventional_multipliers: Tuple[float, ...]
    conventional_maximum_absolute_multiplier_error: float
    conventional_reached_search_bound: bool
    conventional_prediction: DistributedWithheldPredictionMetrics
    conventional_prediction_success: bool
    conventional_prediction_failure_reasons: Tuple[str, ...]
    inverse_pinn_multipliers: Optional[Tuple[float, ...]]
    inverse_pinn_maximum_absolute_multiplier_error: float
    inverse_pinn_prediction: Optional[DistributedWithheldPredictionMetrics]
    inverse_pinn_prediction_success: bool
    inverse_pinn_prediction_failure_reasons: Tuple[str, ...]


class DistributedWithheldValidationSummary(NamedTuple):
    trial_count: int
    conventional_success_count: int
    inverse_pinn_success_count: int
    inverse_pinn_completed_count: int
    conventional_search_bound_hits: int
    conventional_mean_prediction: DistributedWithheldPredictionMetrics
    inverse_pinn_mean_prediction: Optional[DistributedWithheldPredictionMetrics]
    conventional_worst_temperature_error: float
    inverse_pinn_worst_temperature_error: float
    conventional_mean_multiplier_error: float
    inverse_pinn_mean_multiplier_error: Optional[float]


class DistributedWithheldValidationStudyResult(NamedTuple):
    config: DistributedWithheldValidationConfig
    withheld_experiment_name: str
    truth_multipliers: Tuple[float, ...]
    trials: Tuple[DistributedWithheldValidationTrial, ...]
    summary: DistributedWithheldValidationSummary


def distributed_inverse_experiment_names() -> Tuple[str, ...]:
    """Stable human-readable names for the four frozen inverse regimes."""

    return (
        "zero_current_20K_relaxation",
        "positive_0.8A_10K_lift",
        "negative_0.8A_10K_lift",
        "positive_0.4A_20K_lift",
    )


def _rmse(predicted: Sequence[float], truth: Sequence[float]) -> float:
    predicted = tuple(predicted)
    truth = tuple(truth)
    if not predicted or len(predicted) != len(truth):
        raise ValueError("prediction and truth histories must have equal nonzero length")
    return math.sqrt(
        fmean((value - target) ** 2 for value, target in zip(predicted, truth))
    )


def evaluate_distributed_withheld_prediction(
    prediction: DistributedExperimentResult,
    truth: DistributedExperimentResult,
) -> DistributedWithheldPredictionMetrics:
    """Compare complete states and diagnostics on one identical time grid."""

    if prediction.trajectory.time != truth.trajectory.time:
        raise ValueError("withheld prediction and truth must share a time grid")
    predicted_internal = tuple(
        value for cells in prediction.trajectory.cells for value in cells
    )
    truth_internal = tuple(value for cells in truth.trajectory.cells for value in cells)
    temperature_errors = tuple(
        predicted - target
        for predicted_history, truth_history in (
            (prediction.trajectory.cold_face, truth.trajectory.cold_face),
            (predicted_internal, truth_internal),
            (prediction.trajectory.hot_face, truth.trajectory.hot_face),
        )
        for predicted, target in zip(predicted_history, truth_history)
    )
    return DistributedWithheldPredictionMetrics(
        cold_face_rmse=_rmse(
            prediction.trajectory.cold_face, truth.trajectory.cold_face
        ),
        hot_face_rmse=_rmse(
            prediction.trajectory.hot_face, truth.trajectory.hot_face
        ),
        internal_temperature_rmse=_rmse(predicted_internal, truth_internal),
        voltage_rmse=_rmse(prediction.diagnostics.voltage, truth.diagnostics.voltage),
        maximum_absolute_temperature_error=max(
            abs(value) for value in temperature_errors
        ),
        maximum_energy_balance_residual=max(
            abs(value)
            for value in prediction.diagnostics.instantaneous_energy_balance_residual
        ),
    )


def withheld_prediction_failure_reasons(
    metrics: DistributedWithheldPredictionMetrics,
    criteria: DistributedWithheldPredictionCriteria,
) -> Tuple[str, ...]:
    """Apply every predeclared transfer threshold, including finite checks."""

    checks = (
        ("cold_face_rmse", metrics.cold_face_rmse, criteria.maximum_cold_face_rmse),
        ("hot_face_rmse", metrics.hot_face_rmse, criteria.maximum_hot_face_rmse),
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
            metrics.maximum_energy_balance_residual,
            criteria.maximum_energy_balance_residual,
        ),
    )
    return tuple(
        f"nonfinite_{name}" if not math.isfinite(value) else name
        for name, value, limit in checks
        if not math.isfinite(value) or value > limit
    )


def _property_with_multipliers(
    baseline: PiecewiseLinearProperty,
    multipliers: Sequence[float],
) -> PiecewiseLinearProperty:
    multipliers = tuple(multipliers)
    if len(multipliers) != len(baseline.values):
        raise ValueError("multiplier count must match resistivity knots")
    return baseline.with_values(
        tuple(value * multiplier for value, multiplier in zip(baseline.values, multipliers))
    )


def _prediction_experiment(
    baseline: DistributedLegExperiment,
    baseline_property: PiecewiseLinearProperty,
    multipliers: Sequence[float],
) -> DistributedLegExperiment:
    return replace(
        baseline,
        material=replace(
            baseline.material,
            electrical_resistivity=_property_with_multipliers(
                baseline_property, multipliers
            ),
        ),
    )


def _maximum_multiplier_error(multipliers: Sequence[float]) -> float:
    values = tuple(multipliers)
    if len(values) != len(RHO_TRUTH_MULTIPLIERS):
        raise ValueError("resistivity multiplier count must match truth")
    if any(not math.isfinite(value) for value in values):
        return math.inf
    return max(
        abs(value - truth)
        for value, truth in zip(values, RHO_TRUTH_MULTIPLIERS)
    )


def run_distributed_withheld_validation_trial(
    trial_index: int,
    config: DistributedWithheldValidationConfig,
) -> DistributedWithheldValidationTrial:
    """Fit three noisy regimes, freeze the curve, and predict the fourth."""

    baseline_experiments, truth_experiments, baseline_property = (
        distributed_resistivity_truth_experiments()
    )
    if config.withheld_experiment_index >= len(baseline_experiments):
        raise ValueError("withheld experiment index is outside the frozen suite")
    training_indices = tuple(
        index
        for index in range(len(baseline_experiments))
        if index != config.withheld_experiment_index
    )
    training_baselines = tuple(baseline_experiments[index] for index in training_indices)
    training_truths = tuple(truth_experiments[index] for index in training_indices)
    heldout_baseline = baseline_experiments[config.withheld_experiment_index]
    heldout_truth = truth_experiments[config.withheld_experiment_index]
    seeds = distributed_inverse_robustness_seeds(
        config.first_seed, trial_index, len(training_indices)
    )
    observations = noisy_distributed_inverse_observations(
        training_truths,
        observation_interval=config.observation_interval,
        temperature_standard_deviation=config.temperature_standard_deviation,
        voltage_standard_deviation=config.voltage_standard_deviation,
        seeds=seeds.observations,
    )
    channels = DistributedObservationChannels()
    conventional = fit_distributed_property(
        training_baselines,
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
    lower_bound, upper_bound = (-0.2, 0.2)
    conventional_bound_hit = any(
        abs(value - lower_bound) <= 1.0e-6
        or abs(value - upper_bound) <= 1.0e-6
        for value in conventional.log_multipliers
    )
    truth_result = run_distributed_leg_experiment(heldout_truth)
    conventional_prediction = evaluate_distributed_withheld_prediction(
        run_distributed_leg_experiment(
            _prediction_experiment(
                heldout_baseline, baseline_property, conventional_multipliers
            )
        ),
        truth_result,
    )
    conventional_reasons = withheld_prediction_failure_reasons(
        conventional_prediction, config.criteria
    )

    try:
        from ..pinn.distributed_inverse import (
            InverseDistributedPropertyConfig,
            train_multi_experiment_inverse_distributed_property_pinn,
        )

        training = train_multi_experiment_inverse_distributed_property_pinn(
            training_truths,
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
            baseline_material=training_baselines[0].material,
        )
        pinn_multipliers = tuple(
            value / baseline
            for value, baseline in zip(
                training.history.property_values[-1], baseline_property.values
            )
        )
        pinn_prediction = evaluate_distributed_withheld_prediction(
            run_distributed_leg_experiment(
                _prediction_experiment(
                    heldout_baseline, baseline_property, pinn_multipliers
                )
            ),
            truth_result,
        )
        pinn_reasons = withheld_prediction_failure_reasons(
            pinn_prediction, config.criteria
        )
        pinn_error = _maximum_multiplier_error(pinn_multipliers)
    except (FloatingPointError, RuntimeError) as error:
        pinn_multipliers = None
        pinn_prediction = None
        pinn_reasons = (f"training_exception:{type(error).__name__}",)
        pinn_error = math.inf

    return DistributedWithheldValidationTrial(
        trial_index=trial_index,
        seeds=seeds,
        conventional_multipliers=conventional_multipliers,
        conventional_maximum_absolute_multiplier_error=(
            _maximum_multiplier_error(conventional_multipliers)
        ),
        conventional_reached_search_bound=conventional_bound_hit,
        conventional_prediction=conventional_prediction,
        conventional_prediction_success=not conventional_reasons,
        conventional_prediction_failure_reasons=conventional_reasons,
        inverse_pinn_multipliers=pinn_multipliers,
        inverse_pinn_maximum_absolute_multiplier_error=pinn_error,
        inverse_pinn_prediction=pinn_prediction,
        inverse_pinn_prediction_success=not pinn_reasons,
        inverse_pinn_prediction_failure_reasons=pinn_reasons,
    )


def _mean_metrics(
    metrics: Sequence[DistributedWithheldPredictionMetrics],
) -> DistributedWithheldPredictionMetrics:
    metrics = tuple(metrics)
    if not metrics:
        raise ValueError("at least one prediction metric set is required")
    return DistributedWithheldPredictionMetrics(
        *(fmean(values) for values in zip(*metrics))
    )


def summarize_distributed_withheld_validation(
    trials: Sequence[DistributedWithheldValidationTrial],
) -> DistributedWithheldValidationSummary:
    """Aggregate every completed prediction, including threshold failures."""

    trials = tuple(trials)
    if not trials:
        raise ValueError("at least one withheld-validation trial is required")
    completed_pinn = tuple(
        trial for trial in trials if trial.inverse_pinn_prediction is not None
    )
    return DistributedWithheldValidationSummary(
        trial_count=len(trials),
        conventional_success_count=sum(
            trial.conventional_prediction_success for trial in trials
        ),
        inverse_pinn_success_count=sum(
            trial.inverse_pinn_prediction_success for trial in trials
        ),
        inverse_pinn_completed_count=len(completed_pinn),
        conventional_search_bound_hits=sum(
            trial.conventional_reached_search_bound for trial in trials
        ),
        conventional_mean_prediction=_mean_metrics(
            tuple(trial.conventional_prediction for trial in trials)
        ),
        inverse_pinn_mean_prediction=(
            _mean_metrics(
                tuple(trial.inverse_pinn_prediction for trial in completed_pinn)
            )
            if completed_pinn
            else None
        ),
        conventional_worst_temperature_error=max(
            trial.conventional_prediction.maximum_absolute_temperature_error
            for trial in trials
        ),
        inverse_pinn_worst_temperature_error=(
            max(
                trial.inverse_pinn_prediction.maximum_absolute_temperature_error
                for trial in completed_pinn
            )
            if completed_pinn
            else math.inf
        ),
        conventional_mean_multiplier_error=fmean(
            trial.conventional_maximum_absolute_multiplier_error for trial in trials
        ),
        inverse_pinn_mean_multiplier_error=(
            fmean(
                trial.inverse_pinn_maximum_absolute_multiplier_error
                for trial in completed_pinn
            )
            if completed_pinn
            else None
        ),
    )


def run_distributed_withheld_validation_study(
    config: DistributedWithheldValidationConfig = DistributedWithheldValidationConfig(),
) -> DistributedWithheldValidationStudyResult:
    """Run all trials with one complete regime excluded from every fit."""

    names = distributed_inverse_experiment_names()
    if config.withheld_experiment_index >= len(names):
        raise ValueError("withheld experiment index is outside the frozen suite")
    trials = tuple(
        run_distributed_withheld_validation_trial(index, config)
        for index in range(config.trial_count)
    )
    return DistributedWithheldValidationStudyResult(
        config=config,
        withheld_experiment_name=names[config.withheld_experiment_index],
        truth_multipliers=RHO_TRUTH_MULTIPLIERS,
        trials=trials,
        summary=summarize_distributed_withheld_validation(trials),
    )
