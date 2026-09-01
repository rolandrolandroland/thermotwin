"""Training-budget and curve-shape audit for the distributed inverse PINN."""

from dataclasses import dataclass
import math
from statistics import fmean
from typing import NamedTuple, Sequence, Tuple

from ..core.controls import current_at
from ..physics.distributed import distributed_leg_rhs
from ..pinn.distributed_inverse import (
    InverseDistributedHistory,
    InverseDistributedPropertyConfig,
    train_multi_experiment_inverse_distributed_property_pinn,
)
from ..simulation.distributed import (
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
class DistributedPINNTrainingAuditConfig:
    """Frozen CPU budget and truth-blind checkpoint acceptance criteria."""

    trial_count: int = 3
    first_seed: int = 63_001
    checkpoint_epochs: Tuple[int, ...] = (600, 1_200, 2_400)
    observation_interval: float = 0.08
    temperature_standard_deviation: float = 0.01
    voltage_standard_deviation: float = 1.0e-5
    physics_weight: float = 10.0
    maximum_normalized_observation_loss: float = 2.0
    maximum_physics_residual_ratio: float = 0.25
    minimum_shape_ratio: float = 0.75
    maximum_shape_ratio: float = 1.50

    def __post_init__(self) -> None:
        for name, value in (
            ("trial count", self.trial_count),
            ("first seed", self.first_seed),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if self.trial_count <= 0 or self.first_seed < 0:
            raise ValueError("trial count must be positive and seed nonnegative")
        checkpoints = tuple(self.checkpoint_epochs)
        object.__setattr__(self, "checkpoint_epochs", checkpoints)
        if (
            not checkpoints
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                for value in checkpoints
            )
            or tuple(sorted(set(checkpoints))) != checkpoints
        ):
            raise ValueError(
                "checkpoint epochs must be unique positive integers in order"
            )
        for name, value in (
            ("observation interval", self.observation_interval),
            (
                "temperature standard deviation",
                self.temperature_standard_deviation,
            ),
            ("voltage standard deviation", self.voltage_standard_deviation),
            ("physics weight", self.physics_weight),
            (
                "maximum normalized observation loss",
                self.maximum_normalized_observation_loss,
            ),
            ("maximum physics residual ratio", self.maximum_physics_residual_ratio),
            ("minimum shape ratio", self.minimum_shape_ratio),
            ("maximum shape ratio", self.maximum_shape_ratio),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_shape_ratio >= self.maximum_shape_ratio:
            raise ValueError("shape-ratio limits must be ordered")


class DistributedCurveShapeMetrics(NamedTuple):
    """Level and non-flat shape diagnostics for a three-knot curve."""

    mean_multiplier: float
    coefficient_rmse: float
    maximum_absolute_multiplier_error: float
    amplitude: float
    amplitude_ratio: float
    center_contrast: float
    center_contrast_ratio: float
    constant_baseline_rmse: float
    improves_over_best_constant: bool


class DistributedPINNTrainingCheckpoint(NamedTuple):
    """Loss components and truth-known diagnostics at one saved epoch."""

    epoch: int
    multipliers: Tuple[float, ...]
    total_loss: float
    physics_loss: float
    observation_loss: float
    smoothness_loss: float
    shrinkage_loss: float
    physics_residual_rms: float
    reference_temperature_rate_rms: float
    physics_residual_ratio: float
    shape: DistributedCurveShapeMetrics
    operationally_acceptable: bool
    operational_failure_reasons: Tuple[str, ...]
    shape_recovered: bool
    shape_failure_reasons: Tuple[str, ...]


class DistributedPINNTrainingAuditTrial(NamedTuple):
    trial_index: int
    seeds: DistributedInverseRobustnessSeeds
    checkpoints: Tuple[DistributedPINNTrainingCheckpoint, ...]
    first_operational_epoch: int | None


class DistributedPINNTrainingEpochSummary(NamedTuple):
    epoch: int
    completed_count: int
    operationally_acceptable_count: int
    shape_recovered_count: int
    mean_observation_loss: float
    mean_physics_residual_ratio: float
    mean_coefficient_rmse: float
    mean_amplitude_ratio: float
    mean_center_contrast_ratio: float


class DistributedPINNTrainingAuditResult(NamedTuple):
    config: DistributedPINNTrainingAuditConfig
    truth_multipliers: Tuple[float, ...]
    reference_temperature_rate_rms: float
    trials: Tuple[DistributedPINNTrainingAuditTrial, ...]
    epoch_summaries: Tuple[DistributedPINNTrainingEpochSummary, ...]


def distributed_curve_shape_metrics(
    multipliers: Sequence[float],
    truth_multipliers: Sequence[float] = RHO_TRUTH_MULTIPLIERS,
) -> DistributedCurveShapeMetrics:
    """Separate average curve level from its recoverable temperature shape."""

    multipliers = tuple(float(value) for value in multipliers)
    truth = tuple(float(value) for value in truth_multipliers)
    if len(multipliers) != 3 or len(truth) != 3:
        raise ValueError("curve-shape audit requires exactly three coefficients")
    if any(not math.isfinite(value) for value in multipliers + truth):
        raise ValueError("curve coefficients must be finite")
    truth_amplitude = max(truth) - min(truth)
    truth_contrast = truth[1] - 0.5 * (truth[0] + truth[2])
    if truth_amplitude <= 0.0 or truth_contrast == 0.0:
        raise ValueError("truth curve must have nonzero amplitude and center contrast")
    amplitude = max(multipliers) - min(multipliers)
    contrast = multipliers[1] - 0.5 * (multipliers[0] + multipliers[2])
    mean_multiplier = fmean(multipliers)
    coefficient_rmse = math.sqrt(
        fmean((value - target) ** 2 for value, target in zip(multipliers, truth))
    )
    constant_multiplier = fmean(truth)
    constant_rmse = math.sqrt(
        fmean((constant_multiplier - target) ** 2 for target in truth)
    )
    return DistributedCurveShapeMetrics(
        mean_multiplier=mean_multiplier,
        coefficient_rmse=coefficient_rmse,
        maximum_absolute_multiplier_error=max(
            abs(value - target) for value, target in zip(multipliers, truth)
        ),
        amplitude=amplitude,
        amplitude_ratio=amplitude / truth_amplitude,
        center_contrast=contrast,
        center_contrast_ratio=contrast / truth_contrast,
        constant_baseline_rmse=constant_rmse,
        improves_over_best_constant=coefficient_rmse < constant_rmse - 1.0e-12,
    )


def reference_temperature_rate_rms(
    experiments: Sequence[DistributedLegExperiment],
) -> float:
    """Return RMS temperature rate over all supplied states and saved times."""

    experiments = tuple(experiments)
    if not experiments:
        raise ValueError("at least one truth experiment is required")
    rates = []
    for experiment in experiments:
        trajectory = run_distributed_leg_experiment(experiment).trajectory
        for time, cold, cells, hot in zip(
            trajectory.time,
            trajectory.cold_face,
            trajectory.cells,
            trajectory.hot_face,
        ):
            result = distributed_leg_rhs(
                experiment.material,
                experiment.geometry,
                experiment.face_parameters,
                cold_face_temperature=cold,
                cell_temperatures=cells,
                hot_face_temperature=hot,
                current=current_at(experiment.current, time),
                cold_reservoir_temperature=experiment.cold_reservoir_temperature,
                hot_reservoir_temperature=experiment.hot_reservoir_temperature,
                cold_external_heat=experiment.cold_external_heat,
                hot_external_heat=experiment.hot_external_heat,
            )
            rates.extend((result.cold_face, *result.cells, result.hot_face))
    return math.sqrt(fmean(value * value for value in rates))


def _operational_failure_reasons(
    *,
    observation_loss: float,
    physics_residual_ratio: float,
    config: DistributedPINNTrainingAuditConfig,
) -> Tuple[str, ...]:
    reasons = []
    if not math.isfinite(observation_loss):
        reasons.append("nonfinite_observation_loss")
    elif observation_loss > config.maximum_normalized_observation_loss:
        reasons.append("observation_loss")
    if not math.isfinite(physics_residual_ratio):
        reasons.append("nonfinite_physics_residual")
    elif physics_residual_ratio > config.maximum_physics_residual_ratio:
        reasons.append("physics_residual")
    return tuple(reasons)


def _shape_failure_reasons(
    shape: DistributedCurveShapeMetrics,
    config: DistributedPINNTrainingAuditConfig,
) -> Tuple[str, ...]:
    reasons = []
    if not shape.improves_over_best_constant:
        reasons.append("does_not_beat_constant_curve")
    if not config.minimum_shape_ratio <= shape.amplitude_ratio <= config.maximum_shape_ratio:
        reasons.append("amplitude_ratio")
    if not config.minimum_shape_ratio <= shape.center_contrast_ratio <= config.maximum_shape_ratio:
        reasons.append("center_contrast_ratio")
    return tuple(reasons)


def distributed_training_checkpoint(
    history: InverseDistributedHistory,
    *,
    epoch: int,
    baseline_values: Sequence[float],
    reference_rate_rms: float,
    residual_rate_scale: float,
    config: DistributedPINNTrainingAuditConfig,
) -> DistributedPINNTrainingCheckpoint:
    """Extract one predeclared checkpoint from a single uninterrupted run."""

    if epoch <= 0 or epoch > len(history.total_loss):
        raise ValueError("checkpoint epoch lies outside the training history")
    baseline_values = tuple(float(value) for value in baseline_values)
    property_values = history.property_values[epoch - 1]
    if len(property_values) != len(baseline_values):
        raise ValueError("property and baseline coefficient counts must match")
    multipliers = tuple(
        value / baseline for value, baseline in zip(property_values, baseline_values)
    )
    index = epoch - 1
    physics_loss = history.physics_loss[index]
    # Each experiment's physics loss is the sum of three mean-square residual
    # families: interior, cold-face balance, and hot-face balance.
    physics_residual_rms = residual_rate_scale * math.sqrt(physics_loss / 3.0)
    ratio = physics_residual_rms / reference_rate_rms
    shape = distributed_curve_shape_metrics(multipliers)
    operational_reasons = _operational_failure_reasons(
        observation_loss=history.observation_loss[index],
        physics_residual_ratio=ratio,
        config=config,
    )
    shape_reasons = _shape_failure_reasons(shape, config)
    return DistributedPINNTrainingCheckpoint(
        epoch=epoch,
        multipliers=multipliers,
        total_loss=history.total_loss[index],
        physics_loss=physics_loss,
        observation_loss=history.observation_loss[index],
        smoothness_loss=history.smoothness_loss[index],
        shrinkage_loss=history.shrinkage_loss[index],
        physics_residual_rms=physics_residual_rms,
        reference_temperature_rate_rms=reference_rate_rms,
        physics_residual_ratio=ratio,
        shape=shape,
        operationally_acceptable=not operational_reasons,
        operational_failure_reasons=operational_reasons,
        shape_recovered=not shape_reasons,
        shape_failure_reasons=shape_reasons,
    )


def run_distributed_pinn_training_audit_trial(
    trial_index: int,
    config: DistributedPINNTrainingAuditConfig,
    *,
    reference_rate_rms: float | None = None,
) -> DistributedPINNTrainingAuditTrial:
    """Train once to the maximum budget and retain declared checkpoints."""

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
    training_config = InverseDistributedPropertyConfig(
        property_name="electrical_resistivity",
        hidden_width=20,
        hidden_layers=3,
        interior_space_points=7,
        time_points=18,
        voltage_space_points=16,
        epochs=config.checkpoint_epochs[-1],
        network_learning_rate=2.0e-3,
        property_learning_rate=2.0e-3,
        initial_log_multipliers=(math.log(0.9),) * 3,
        smoothness_weight=1.0e-4,
        physics_weight=config.physics_weight,
        temperature_standard_deviation=config.temperature_standard_deviation,
        voltage_standard_deviation=config.voltage_standard_deviation,
        seed=seeds.neural,
        device="cpu",
    )
    training = train_multi_experiment_inverse_distributed_property_pinn(
        truth_experiments,
        observations,
        training_config,
        baseline_material=baseline_experiments[0].material,
    )
    reference_rate_rms = (
        reference_rate_rms
        if reference_rate_rms is not None
        else reference_temperature_rate_rms(baseline_experiments)
    )
    checkpoints = tuple(
        distributed_training_checkpoint(
            training.history,
            epoch=epoch,
            baseline_values=baseline_property.values,
            reference_rate_rms=reference_rate_rms,
            residual_rate_scale=training_config.residual_rate_scale,
            config=config,
        )
        for epoch in config.checkpoint_epochs
    )
    first_operational_epoch = next(
        (
            checkpoint.epoch
            for checkpoint in checkpoints
            if checkpoint.operationally_acceptable
        ),
        None,
    )
    return DistributedPINNTrainingAuditTrial(
        trial_index=trial_index,
        seeds=seeds,
        checkpoints=checkpoints,
        first_operational_epoch=first_operational_epoch,
    )


def summarize_distributed_pinn_training_audit(
    trials: Sequence[DistributedPINNTrainingAuditTrial],
) -> Tuple[DistributedPINNTrainingEpochSummary, ...]:
    trials = tuple(trials)
    if not trials:
        raise ValueError("at least one training-audit trial is required")
    epochs = tuple(checkpoint.epoch for checkpoint in trials[0].checkpoints)
    if any(
        tuple(checkpoint.epoch for checkpoint in trial.checkpoints) != epochs
        for trial in trials
    ):
        raise ValueError("all training-audit trials must share checkpoint epochs")
    return tuple(
        DistributedPINNTrainingEpochSummary(
            epoch=epoch,
            completed_count=len(trials),
            operationally_acceptable_count=sum(
                trial.checkpoints[index].operationally_acceptable for trial in trials
            ),
            shape_recovered_count=sum(
                trial.checkpoints[index].shape_recovered for trial in trials
            ),
            mean_observation_loss=fmean(
                trial.checkpoints[index].observation_loss for trial in trials
            ),
            mean_physics_residual_ratio=fmean(
                trial.checkpoints[index].physics_residual_ratio for trial in trials
            ),
            mean_coefficient_rmse=fmean(
                trial.checkpoints[index].shape.coefficient_rmse for trial in trials
            ),
            mean_amplitude_ratio=fmean(
                trial.checkpoints[index].shape.amplitude_ratio for trial in trials
            ),
            mean_center_contrast_ratio=fmean(
                trial.checkpoints[index].shape.center_contrast_ratio for trial in trials
            ),
        )
        for index, epoch in enumerate(epochs)
    )


def run_distributed_pinn_training_audit(
    config: DistributedPINNTrainingAuditConfig = DistributedPINNTrainingAuditConfig(),
) -> DistributedPINNTrainingAuditResult:
    """Run the frozen multi-seed audit without truth-based early stopping."""

    baseline_experiments, _, _ = distributed_resistivity_truth_experiments()
    # Use the released nominal model rather than hidden property truth so this
    # normalization can support a truth-blind checkpoint rule.
    rate_rms = reference_temperature_rate_rms(baseline_experiments)
    trials = tuple(
        run_distributed_pinn_training_audit_trial(
            index, config, reference_rate_rms=rate_rms
        )
        for index in range(config.trial_count)
    )
    return DistributedPINNTrainingAuditResult(
        config=config,
        truth_multipliers=RHO_TRUTH_MULTIPLIERS,
        reference_temperature_rate_rms=rate_rms,
        trials=trials,
        epoch_summaries=summarize_distributed_pinn_training_audit(trials),
    )
