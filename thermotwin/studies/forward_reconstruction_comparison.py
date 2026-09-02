"""Matched physics-informed and observation-only transient reconstruction."""

from copy import deepcopy
from dataclasses import dataclass
import math
from statistics import fmean
from time import perf_counter
from typing import Callable, NamedTuple, Optional, Sequence, Tuple

import torch

from ..observations.missingness import (
    DeterministicTemperatureMissingness,
    TemperatureSensorOutage,
    apply_deterministic_temperature_missingness,
)
from ..observations.noise import (
    GaussianTemperatureNoise,
    apply_gaussian_temperature_noise,
)
from ..observations.test_stand import (
    IdealTemperatureSensor,
    IdealVirtualTestStand,
    ObservationDataset,
    TemperatureSensorLocation,
    observe_contact_trajectory,
)
from ..core.controls import current_at
from ..pinn.energy_closure import (
    ContactPINNEnergyClosureConfig,
    evaluate_piecewise_contact_energy_closure,
)
from ..pinn.forward_four_node import contact_physics_residuals, predict_contact_trajectory
from ..pinn.forward_piecewise import (
    PiecewiseContactForwardPINN,
    current_segment_boundaries,
    piecewise_collocation_times,
    scheduled_current_tensor,
    unipolar_pulse_contact_experiment,
)
from ..pinn.forward_two_node import select_device
from ..simulation.four_node_experiments import (
    FourNodeContactExperiment,
    run_four_node_contact_experiment,
)


@dataclass(frozen=True)
class ForwardReconstructionCriteria:
    maximum_missing_exchanger_rmse: float = 0.02
    maximum_hidden_face_rmse: float = 0.02
    maximum_node_residual_rms: float = 0.005
    maximum_normalized_energy_rate_closure_rms: float = 0.15
    maximum_absolute_cumulative_energy_error: float = 4.0

    def __post_init__(self) -> None:
        for name, value in (
            ("missing-exchanger RMSE", self.maximum_missing_exchanger_rmse),
            ("hidden-face RMSE", self.maximum_hidden_face_rmse),
            ("node residual RMS", self.maximum_node_residual_rms),
            (
                "normalized energy-rate closure RMS",
                self.maximum_normalized_energy_rate_closure_rms,
            ),
            (
                "absolute cumulative energy error",
                self.maximum_absolute_cumulative_energy_error,
            ),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"maximum {name} must be finite and positive")


@dataclass(frozen=True)
class ForwardReconstructionComparisonConfig:
    """Frozen observation design, architecture, optimizer, and trial count."""

    trial_count: int = 5
    first_seed: int = 72_001
    sampling_interval: float = 2.0
    noise_standard_deviation: float = 0.02
    missing_start_time: float = 17.0
    missing_end_time: float = 23.0
    hidden_width: int = 16
    hidden_layers: int = 2
    collocation_points: int = 72
    epochs: int = 5_000
    learning_rate: float = 1.0e-3
    temperature_scale: float = 10.0
    observation_temperature_scale: float = 0.10
    residual_rate_scale: float = 0.10
    physics_weight: float = 100.0
    energy_sampling_interval: float = 0.2
    device: str = "cpu"
    criteria: ForwardReconstructionCriteria = ForwardReconstructionCriteria()

    def __post_init__(self) -> None:
        for name, value in (
            ("trial count", self.trial_count),
            ("hidden width", self.hidden_width),
            ("hidden layers", self.hidden_layers),
            ("collocation points", self.collocation_points),
            ("epochs", self.epochs),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(self.first_seed, int)
            or isinstance(self.first_seed, bool)
            or self.first_seed < 0
        ):
            raise ValueError("first seed must be a nonnegative integer")
        for name, value in (
            ("sampling interval", self.sampling_interval),
            ("noise standard deviation", self.noise_standard_deviation),
            ("learning rate", self.learning_rate),
            ("temperature scale", self.temperature_scale),
            ("observation scale", self.observation_temperature_scale),
            ("residual rate scale", self.residual_rate_scale),
            ("physics weight", self.physics_weight),
            ("energy sampling interval", self.energy_sampling_interval),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.missing_start_time)
            or not math.isfinite(self.missing_end_time)
            or self.missing_start_time < 0.0
            or self.missing_end_time < self.missing_start_time
        ):
            raise ValueError("missing interval must be finite, nonnegative, and ordered")
        if self.device not in {"cpu", "mps", "auto"}:
            raise ValueError("device must be 'cpu', 'mps', or 'auto'")
        if not isinstance(self.criteria, ForwardReconstructionCriteria):
            raise ValueError("criteria must be a forward-reconstruction criteria object")


class ReconstructionLossSummary(NamedTuple):
    initial_observation_loss: float
    final_observation_loss: float
    initial_physics_loss: float
    final_physics_loss: float
    training_seconds: float


class ForwardReconstructionMetrics(NamedTuple):
    method: str
    losses: ReconstructionLossSummary
    retained_noisy_observation_rmse: float
    retained_truth_rmse: float
    missing_exchanger_rmse: float
    hidden_face_rmse: float
    hidden_face_missing_interval_rmse: float
    all_state_rmse: float
    node_residual_rms: float
    energy_rate_closure_rms: float
    normalized_energy_rate_closure_rms: float
    final_cumulative_energy_error: float
    maximum_absolute_cumulative_energy_error: float
    maximum_boundary_temperature_jump: float


class ForwardReconstructionTrial(NamedTuple):
    trial_index: int
    observation_seed: int
    neural_seed: int
    retained_observation_count: int
    removed_observation_count: int
    initialization_maximum_absolute_difference: float
    physics_informed: ForwardReconstructionMetrics
    data_only: ForwardReconstructionMetrics
    physics_completion_gate_passed: bool


class ForwardReconstructionSummary(NamedTuple):
    method: str
    trial_count: int
    mean_retained_noisy_observation_rmse: float
    mean_retained_truth_rmse: float
    mean_missing_exchanger_rmse: float
    mean_hidden_face_rmse: float
    mean_hidden_face_missing_interval_rmse: float
    mean_all_state_rmse: float
    mean_node_residual_rms: float
    mean_energy_rate_closure_rms: float
    mean_normalized_energy_rate_closure_rms: float
    mean_absolute_final_cumulative_energy_error: float
    mean_maximum_absolute_cumulative_energy_error: float
    mean_training_seconds: float


class ForwardReconstructionTrace(NamedTuple):
    time: Tuple[float, ...]
    current: Tuple[float, ...]
    reference: Tuple[Tuple[float, ...], ...]
    physics_informed: Tuple[Tuple[float, ...], ...]
    data_only: Tuple[Tuple[float, ...], ...]
    observation_time: Tuple[float, ...]
    observation_location: Tuple[str, ...]
    observed_temperature: Tuple[float, ...]


class ForwardReconstructionComparisonResult(NamedTuple):
    config: ForwardReconstructionComparisonConfig
    trainable_parameter_count_per_model: int
    complete_observation_count: int
    retained_observation_count: int
    removed_observation_count: int
    trials: Tuple[ForwardReconstructionTrial, ...]
    summaries: Tuple[ForwardReconstructionSummary, ...]
    representative_trace: ForwardReconstructionTrace
    physics_missing_exchanger_rmse_reduction_percent: float
    physics_hidden_face_rmse_reduction_percent: float
    physics_hidden_face_gap_rmse_reduction_percent: float
    physics_energy_rate_error_reduction_percent: float
    physics_all_metric_advantage_count: int
    physics_completion_gate_pass_count: int


class _TrainedReconstruction(NamedTuple):
    model: PiecewiseContactForwardPINN
    losses: ReconstructionLossSummary


_LOCATION_COLUMN = {
    TemperatureSensorLocation.COLD_FACE: 0,
    TemperatureSensorLocation.HOT_FACE: 1,
    TemperatureSensorLocation.COLD_EXCHANGER: 2,
    TemperatureSensorLocation.HOT_EXCHANGER: 3,
}


def forward_reconstruction_seeds(
    first_seed: int,
    trial_index: int,
) -> Tuple[int, int]:
    if first_seed < 0 or trial_index < 0:
        raise ValueError("seeds and trial indices must be nonnegative")
    return first_seed + 2 * trial_index, first_seed + 2 * trial_index + 1


def build_forward_reconstruction_observations(
    experiment: FourNodeContactExperiment,
    config: ForwardReconstructionComparisonConfig,
    *,
    observation_seed: int,
) -> Tuple[ObservationDataset, ObservationDataset]:
    """Return complete ideal exchanger rows and the noisy incomplete subset."""

    trajectory = run_four_node_contact_experiment(experiment).trajectory
    sensors = (
        IdealTemperatureSensor(
            "cold_exchanger_sensor",
            TemperatureSensorLocation.COLD_EXCHANGER,
        ),
        IdealTemperatureSensor(
            "hot_exchanger_sensor",
            TemperatureSensorLocation.HOT_EXCHANGER,
        ),
    )
    ideal = observe_contact_trajectory(
        trajectory,
        current=experiment.current,
        test_stand=IdealVirtualTestStand(
            sensors=sensors,
            sampling_interval=config.sampling_interval,
        ),
    )
    noisy = apply_gaussian_temperature_noise(
        ideal,
        GaussianTemperatureNoise(
            default_standard_deviation=config.noise_standard_deviation,
            random_seed=observation_seed,
        ),
    ).dataset
    incomplete = apply_deterministic_temperature_missingness(
        noisy,
        DeterministicTemperatureMissingness(
            outages=tuple(
                TemperatureSensorOutage(
                    sensor_name=sensor.name,
                    start_time=config.missing_start_time,
                    end_time=config.missing_end_time,
                )
                for sensor in sensors
            )
        ),
    ).dataset
    return ideal, incomplete


def _observation_tensors(dataset: ObservationDataset, device: torch.device):
    time = torch.tensor(
        tuple(item.time for item in dataset.observations),
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 1)
    column = torch.tensor(
        tuple(_LOCATION_COLUMN[item.location] for item in dataset.observations),
        dtype=torch.int64,
        device=device,
    ).reshape(-1, 1)
    temperature = torch.tensor(
        tuple(item.temperature for item in dataset.observations),
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 1)
    return time, column, temperature


def _losses(
    model: PiecewiseContactForwardPINN,
    observation_time: torch.Tensor,
    observation_column: torch.Tensor,
    observed_temperature: torch.Tensor,
    collocation_time: torch.Tensor,
    collocation_current: torch.Tensor,
    experiment: FourNodeContactExperiment,
    config: ForwardReconstructionComparisonConfig,
):
    prediction = model(observation_time).gather(1, observation_column)
    observation_loss = (
        (prediction - observed_temperature) / config.observation_temperature_scale
    ).square().mean()
    residuals = contact_physics_residuals(
        model,
        collocation_time,
        experiment,
        current_values=collocation_current,
    )
    physics_loss = torch.cat(tuple(residuals), dim=1).div(
        config.residual_rate_scale
    ).square().mean()
    return observation_loss, physics_loss


def _train_one(
    model: PiecewiseContactForwardPINN,
    observations: ObservationDataset,
    experiment: FourNodeContactExperiment,
    config: ForwardReconstructionComparisonConfig,
    *,
    use_physics: bool,
) -> _TrainedReconstruction:
    parameter = next(model.parameters())
    device = parameter.device
    boundaries = model.segment_boundaries
    collocation_time = piecewise_collocation_times(
        boundaries,
        config.collocation_points,
        dtype=parameter.dtype,
        device=device,
    )
    collocation_current = scheduled_current_tensor(
        experiment.current,
        collocation_time,
    )
    observation_time, observation_column, observed_temperature = (
        _observation_tensors(observations, device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    model.train()
    initial_observation, initial_physics = _losses(
        model,
        observation_time,
        observation_column,
        observed_temperature,
        collocation_time,
        collocation_current,
        experiment,
        config,
    )
    start = perf_counter()
    for _ in range(config.epochs):
        optimizer.zero_grad()
        if use_physics:
            observation_loss, physics_loss = _losses(
                model,
                observation_time,
                observation_column,
                observed_temperature,
                collocation_time,
                collocation_current,
                experiment,
                config,
            )
            loss = observation_loss + config.physics_weight * physics_loss
        else:
            prediction = model(observation_time).gather(1, observation_column)
            observation_loss = (
                (prediction - observed_temperature)
                / config.observation_temperature_scale
            ).square().mean()
            loss = observation_loss
        loss.backward()
        optimizer.step()
    duration = perf_counter() - start
    final_observation, final_physics = _losses(
        model,
        observation_time,
        observation_column,
        observed_temperature,
        collocation_time,
        collocation_current,
        experiment,
        config,
    )
    return _TrainedReconstruction(
        model=model,
        losses=ReconstructionLossSummary(
            initial_observation_loss=float(initial_observation.detach().cpu()),
            final_observation_loss=float(final_observation.detach().cpu()),
            initial_physics_loss=float(initial_physics.detach().cpu()),
            final_physics_loss=float(final_physics.detach().cpu()),
            training_seconds=duration,
        ),
    )


def train_matched_forward_reconstruction_models(
    experiment: FourNodeContactExperiment,
    observations: ObservationDataset,
    config: ForwardReconstructionComparisonConfig,
    *,
    neural_seed: int,
):
    """Return identically initialized models trained with and without physics."""

    device = select_device(config.device)
    boundaries = current_segment_boundaries(experiment.current, experiment.duration)
    torch.manual_seed(neural_seed)
    template = PiecewiseContactForwardPINN(
        duration=experiment.duration,
        transition_times=boundaries[1:-1],
        initial_temperatures=(
            experiment.initial_cold_face_temperature,
            experiment.initial_hot_face_temperature,
            experiment.initial_cold_exchanger_temperature,
            experiment.initial_hot_exchanger_temperature,
        ),
        hidden_width=config.hidden_width,
        hidden_layers=config.hidden_layers,
        temperature_scale=config.temperature_scale,
    ).to(device)
    physics_model = deepcopy(template)
    data_model = deepcopy(template)
    initialization_difference = max(
        float((left - right).abs().max().detach().cpu())
        for left, right in zip(physics_model.parameters(), data_model.parameters())
    )
    physics = _train_one(
        physics_model,
        observations,
        experiment,
        config,
        use_physics=True,
    )
    data_only = _train_one(
        data_model,
        observations,
        experiment,
        config,
        use_physics=False,
    )
    return physics, data_only, initialization_difference


def _rmse(errors: Sequence[float]) -> float:
    errors = tuple(errors)
    if not errors:
        raise ValueError("RMSE needs at least one error")
    return math.sqrt(sum(value * value for value in errors) / len(errors))


def _node_residual_rms(
    model: PiecewiseContactForwardPINN,
    experiment: FourNodeContactExperiment,
    point_count: int,
) -> float:
    parameter = next(model.parameters())
    time = piecewise_collocation_times(
        model.segment_boundaries,
        point_count,
        dtype=parameter.dtype,
        device=parameter.device,
    )
    residuals = contact_physics_residuals(
        model,
        time,
        experiment,
        current_values=scheduled_current_tensor(experiment.current, time),
    )
    values = torch.cat(tuple(residuals), dim=1).detach().cpu().reshape(-1)
    return float(torch.sqrt(torch.mean(values.square())))


def _evaluate_model(
    method: str,
    trained: _TrainedReconstruction,
    experiment: FourNodeContactExperiment,
    ideal: ObservationDataset,
    incomplete: ObservationDataset,
    config: ForwardReconstructionComparisonConfig,
) -> ForwardReconstructionMetrics:
    reference = run_four_node_contact_experiment(experiment).trajectory
    prediction = predict_contact_trajectory(trained.model, reference.time)
    predicted_histories = (
        prediction.cold_face,
        prediction.hot_face,
        prediction.cold_exchanger,
        prediction.hot_exchanger,
    )
    reference_histories = (
        reference.cold_face,
        reference.hot_face,
        reference.cold_exchanger,
        reference.hot_exchanger,
    )
    state_errors = tuple(
        tuple(left - right for left, right in zip(predicted, expected))
        for predicted, expected in zip(predicted_histories, reference_histories)
    )

    parameter = next(trained.model.parameters())
    observation_time, observation_column, noisy_temperature = _observation_tensors(
        incomplete, parameter.device
    )
    with torch.no_grad():
        retained_prediction = (
            trained.model(observation_time)
            .gather(1, observation_column)
            .detach()
            .cpu()
            .reshape(-1)
        )
    ideal_map = {
        (item.time, item.sensor_name): item.temperature for item in ideal.observations
    }
    retained_truth = tuple(
        ideal_map[(item.time, item.sensor_name)] for item in incomplete.observations
    )
    noisy_values = tuple(float(value) for value in noisy_temperature.cpu().reshape(-1))

    missing_indices = tuple(
        index
        for index, time in enumerate(reference.time)
        if config.missing_start_time <= time <= config.missing_end_time
    )
    missing_exchanger_errors = tuple(
        state_errors[column][index]
        for column in (2, 3)
        for index in missing_indices
    )
    hidden_missing_errors = tuple(
        state_errors[column][index]
        for column in (0, 1)
        for index in missing_indices
    )
    energy = evaluate_piecewise_contact_energy_closure(
        trained.model,
        experiment,
        ContactPINNEnergyClosureConfig(config.energy_sampling_interval),
    )
    jumps = trained.model.boundary_temperature_jumps()
    return ForwardReconstructionMetrics(
        method=method,
        losses=trained.losses,
        retained_noisy_observation_rmse=_rmse(
            float(predicted) - observed
            for predicted, observed in zip(retained_prediction, noisy_values)
        ),
        retained_truth_rmse=_rmse(
            float(predicted) - expected
            for predicted, expected in zip(retained_prediction, retained_truth)
        ),
        missing_exchanger_rmse=_rmse(missing_exchanger_errors),
        hidden_face_rmse=_rmse((*state_errors[0], *state_errors[1])),
        hidden_face_missing_interval_rmse=_rmse(hidden_missing_errors),
        all_state_rmse=_rmse(tuple(value for errors in state_errors for value in errors)),
        node_residual_rms=_node_residual_rms(
            trained.model, experiment, config.collocation_points * 2
        ),
        energy_rate_closure_rms=energy.rate_closure_rms,
        normalized_energy_rate_closure_rms=energy.normalized_rate_closure_rms,
        final_cumulative_energy_error=energy.final_cumulative_closure_error,
        maximum_absolute_cumulative_energy_error=(
            energy.maximum_absolute_cumulative_closure_error
        ),
        maximum_boundary_temperature_jump=(
            0.0
            if jumps.numel() == 0
            else float(jumps.abs().max().detach().cpu())
        ),
    )


def _summarize(
    method: str,
    metrics: Sequence[ForwardReconstructionMetrics],
) -> ForwardReconstructionSummary:
    metrics = tuple(metrics)
    return ForwardReconstructionSummary(
        method=method,
        trial_count=len(metrics),
        mean_retained_noisy_observation_rmse=fmean(
            item.retained_noisy_observation_rmse for item in metrics
        ),
        mean_retained_truth_rmse=fmean(item.retained_truth_rmse for item in metrics),
        mean_missing_exchanger_rmse=fmean(
            item.missing_exchanger_rmse for item in metrics
        ),
        mean_hidden_face_rmse=fmean(item.hidden_face_rmse for item in metrics),
        mean_hidden_face_missing_interval_rmse=fmean(
            item.hidden_face_missing_interval_rmse for item in metrics
        ),
        mean_all_state_rmse=fmean(item.all_state_rmse for item in metrics),
        mean_node_residual_rms=fmean(item.node_residual_rms for item in metrics),
        mean_energy_rate_closure_rms=fmean(
            item.energy_rate_closure_rms for item in metrics
        ),
        mean_normalized_energy_rate_closure_rms=fmean(
            item.normalized_energy_rate_closure_rms for item in metrics
        ),
        mean_absolute_final_cumulative_energy_error=fmean(
            abs(item.final_cumulative_energy_error) for item in metrics
        ),
        mean_maximum_absolute_cumulative_energy_error=fmean(
            item.maximum_absolute_cumulative_energy_error for item in metrics
        ),
        mean_training_seconds=fmean(item.losses.training_seconds for item in metrics),
    )


def _build_trace(
    experiment: FourNodeContactExperiment,
    physics_model: PiecewiseContactForwardPINN,
    data_model: PiecewiseContactForwardPINN,
    observations: ObservationDataset,
) -> ForwardReconstructionTrace:
    reference = run_four_node_contact_experiment(experiment).trajectory
    physics = predict_contact_trajectory(physics_model, reference.time)
    data_only = predict_contact_trajectory(data_model, reference.time)

    def histories(trajectory):
        return (
            trajectory.cold_face,
            trajectory.hot_face,
            trajectory.cold_exchanger,
            trajectory.hot_exchanger,
        )

    return ForwardReconstructionTrace(
        time=reference.time,
        current=tuple(current_at(experiment.current, time) for time in reference.time),
        reference=histories(reference),
        physics_informed=histories(physics),
        data_only=histories(data_only),
        observation_time=tuple(item.time for item in observations.observations),
        observation_location=tuple(
            item.location.value for item in observations.observations
        ),
        observed_temperature=tuple(
            item.temperature for item in observations.observations
        ),
    )


def _reduction(preferred: float, comparison: float) -> float:
    if comparison <= 0.0:
        raise ValueError("comparison metric must be positive")
    return 100.0 * (1.0 - preferred / comparison)


def _passes_completion_gate(
    metrics: ForwardReconstructionMetrics,
    criteria: ForwardReconstructionCriteria,
) -> bool:
    return (
        metrics.missing_exchanger_rmse
        <= criteria.maximum_missing_exchanger_rmse
        and metrics.hidden_face_rmse <= criteria.maximum_hidden_face_rmse
        and metrics.node_residual_rms <= criteria.maximum_node_residual_rms
        and metrics.normalized_energy_rate_closure_rms
        <= criteria.maximum_normalized_energy_rate_closure_rms
        and metrics.maximum_absolute_cumulative_energy_error
        <= criteria.maximum_absolute_cumulative_energy_error
    )


def run_forward_reconstruction_comparison(
    config: ForwardReconstructionComparisonConfig = (
        ForwardReconstructionComparisonConfig()
    ),
    *,
    progress: Optional[Callable[[str], None]] = None,
) -> ForwardReconstructionComparisonResult:
    """Run paired identical-capacity reconstruction trials."""

    experiment = unipolar_pulse_contact_experiment()
    trials = []
    complete_count = retained_count = removed_count = 0
    parameter_count = 0
    representative_trace = None
    for trial_index in range(config.trial_count):
        if progress is not None:
            progress(f"matched reconstruction trial {trial_index + 1}/{config.trial_count}")
        observation_seed, neural_seed = forward_reconstruction_seeds(
            config.first_seed, trial_index
        )
        ideal, incomplete = build_forward_reconstruction_observations(
            experiment,
            config,
            observation_seed=observation_seed,
        )
        physics, data_only, initialization_difference = (
            train_matched_forward_reconstruction_models(
                experiment,
                incomplete,
                config,
                neural_seed=neural_seed,
            )
        )
        if trial_index == 0:
            complete_count = len(ideal.observations)
            retained_count = len(incomplete.observations)
            removed_count = complete_count - retained_count
            parameter_count = sum(
                parameter.numel() for parameter in physics.model.parameters()
            )
            representative_trace = _build_trace(
                experiment,
                physics.model,
                data_only.model,
                incomplete,
            )
        physics_metrics = _evaluate_model(
            "physics_informed", physics, experiment, ideal, incomplete, config
        )
        data_metrics = _evaluate_model(
            "data_only", data_only, experiment, ideal, incomplete, config
        )
        trials.append(
            ForwardReconstructionTrial(
                trial_index=trial_index,
                observation_seed=observation_seed,
                neural_seed=neural_seed,
                retained_observation_count=len(incomplete.observations),
                removed_observation_count=(
                    len(ideal.observations) - len(incomplete.observations)
                ),
                initialization_maximum_absolute_difference=initialization_difference,
                physics_informed=physics_metrics,
                data_only=data_metrics,
                physics_completion_gate_passed=_passes_completion_gate(
                    physics_metrics, config.criteria
                ),
            )
        )
    summaries = (
        _summarize("physics_informed", tuple(item.physics_informed for item in trials)),
        _summarize("data_only", tuple(item.data_only for item in trials)),
    )
    if representative_trace is None:
        raise RuntimeError("comparison produced no representative trajectory")
    physics, data_only = summaries
    advantage_count = sum(
        trial.physics_informed.missing_exchanger_rmse
        < trial.data_only.missing_exchanger_rmse
        and trial.physics_informed.hidden_face_rmse
        < trial.data_only.hidden_face_rmse
        and trial.physics_informed.energy_rate_closure_rms
        < trial.data_only.energy_rate_closure_rms
        for trial in trials
    )
    return ForwardReconstructionComparisonResult(
        config=config,
        trainable_parameter_count_per_model=parameter_count,
        complete_observation_count=complete_count,
        retained_observation_count=retained_count,
        removed_observation_count=removed_count,
        trials=tuple(trials),
        summaries=summaries,
        representative_trace=representative_trace,
        physics_missing_exchanger_rmse_reduction_percent=_reduction(
            physics.mean_missing_exchanger_rmse,
            data_only.mean_missing_exchanger_rmse,
        ),
        physics_hidden_face_rmse_reduction_percent=_reduction(
            physics.mean_hidden_face_rmse,
            data_only.mean_hidden_face_rmse,
        ),
        physics_hidden_face_gap_rmse_reduction_percent=_reduction(
            physics.mean_hidden_face_missing_interval_rmse,
            data_only.mean_hidden_face_missing_interval_rmse,
        ),
        physics_energy_rate_error_reduction_percent=_reduction(
            physics.mean_energy_rate_closure_rms,
            data_only.mean_energy_rate_closure_rms,
        ),
        physics_all_metric_advantage_count=advantage_count,
        physics_completion_gate_pass_count=sum(
            trial.physics_completion_gate_passed for trial in trials
        ),
    )
