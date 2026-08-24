"""Infer cold contact resistance with a switched-current contact PINN."""

import argparse
from dataclasses import dataclass
import math
from typing import NamedTuple, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional

from ..simulation.four_node_experiments import (
    FourNodeContactExperiment,
    run_four_node_contact_experiment,
)
from .forward_four_node import (
    contact_physics_residuals,
    predict_contact_trajectory,
)
from ..inference.contact_resistance import (
    REFERENCE_COLD_CONTACT_RESISTANCE,
    ContactResistanceRegimeDataset,
    ContactResistanceRegimeMetrics,
    contact_resistance_experiment,
    evaluate_contact_resistance_regime,
    fit_cold_contact_resistance,
    reference_contact_resistance_dataset_split,
    reference_contact_resistance_regimes,
    simulate_contact_resistance_observations,
)
from .forward_two_node import select_device
from .inverse_contact_resistance import (
    ColdContactTemperatureObservations,
    InverseContactTrainingHistory,
    cold_contact_observations_from_dataset,
    inverse_softplus,
)
from .forward_piecewise import (
    PiecewiseContactForwardPINN,
    current_segment_boundaries,
    piecewise_collocation_times,
    scheduled_current_tensor,
)


class IdealPiecewiseInverseContactProblem(NamedTuple):
    """One pulse experiment and its sparse ideal cold-pair observations."""

    experiment: FourNodeContactExperiment
    dataset: ContactResistanceRegimeDataset
    observations: ColdContactTemperatureObservations


def ideal_piecewise_inverse_contact_problem(
    *,
    observation_interval: float = 1.0,
) -> IdealPiecewiseInverseContactProblem:
    """Return the frozen 0--1--0 A inverse problem with ideal cold data."""

    regime = reference_contact_resistance_regimes()[0]
    experiment = contact_resistance_experiment(
        regime,
        cold_contact_resistance=REFERENCE_COLD_CONTACT_RESISTANCE,
    )
    dataset = ContactResistanceRegimeDataset(
        regime=regime,
        observations=simulate_contact_resistance_observations(
            regime,
            cold_contact_resistance=REFERENCE_COLD_CONTACT_RESISTANCE,
            sampling_interval=observation_interval,
        ),
    )
    return IdealPiecewiseInverseContactProblem(
        experiment=experiment,
        dataset=dataset,
        observations=cold_contact_observations_from_dataset(dataset),
    )


class PiecewiseInverseContactResistancePINN(nn.Module):
    """Predict piecewise temperatures and learn one shared positive contact."""

    def __init__(
        self,
        *,
        duration: float,
        transition_times: Sequence[float],
        initial_temperatures: Sequence[float],
        initial_cold_contact_resistance: float,
        hidden_width: int = 32,
        hidden_layers: int = 2,
        temperature_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.temperature_model = PiecewiseContactForwardPINN(
            duration=duration,
            transition_times=transition_times,
            initial_temperatures=initial_temperatures,
            hidden_width=hidden_width,
            hidden_layers=hidden_layers,
            temperature_scale=temperature_scale,
        )
        self.raw_cold_contact_resistance = nn.Parameter(
            torch.tensor(
                inverse_softplus(initial_cold_contact_resistance),
                dtype=torch.float32,
            )
        )

    @property
    def cold_contact_resistance(self) -> Tensor:
        """Return the positive shared cold contact resistance in K/W."""

        return functional.softplus(self.raw_cold_contact_resistance)

    @property
    def segment_boundaries(self) -> Tuple[float, ...]:
        """Return the temperature model's interval boundaries."""

        return self.temperature_model.segment_boundaries

    def boundary_temperature_jumps(self) -> Tensor:
        """Return exact right-minus-left temperature jumps at switches."""

        return self.temperature_model.boundary_temperature_jumps()

    def forward(self, time: Tensor) -> Tensor:
        return self.temperature_model(time)


@dataclass(frozen=True)
class PiecewiseInverseContactResistanceConfig:
    """Network, normalized loss, and CPU-first optimization settings."""

    hidden_width: int = 32
    hidden_layers: int = 2
    collocation_points: int = 192
    epochs: int = 8_000
    network_learning_rate: float = 1e-3
    parameter_learning_rate: float = 5e-3
    initial_cold_contact_resistance: float = 0.5
    temperature_scale: float = 10.0
    residual_rate_scale: float = 0.1
    observation_temperature_scale: float = 1.0
    physics_weight: float = 1.0
    observation_weight: float = 20.0
    seed: int = 19
    device: str = "cpu"

    def __post_init__(self) -> None:
        integer_settings = (
            ("hidden width", self.hidden_width, 1),
            ("hidden layer count", self.hidden_layers, 1),
            ("collocation point count", self.collocation_points, 2),
            ("epoch count", self.epochs, 1),
        )
        for name, value, minimum in integer_settings:
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < minimum
            ):
                raise ValueError(f"{name} must be at least {minimum}")
        positive_settings = (
            ("network learning rate", self.network_learning_rate),
            ("parameter learning rate", self.parameter_learning_rate),
            (
                "initial cold contact resistance",
                self.initial_cold_contact_resistance,
            ),
            ("temperature scale", self.temperature_scale),
            ("residual rate scale", self.residual_rate_scale),
            (
                "observation temperature scale",
                self.observation_temperature_scale,
            ),
            ("physics weight", self.physics_weight),
            ("observation weight", self.observation_weight),
        )
        for name, value in positive_settings:
            try:
                value_is_finite = math.isfinite(value)
            except TypeError as error:
                raise ValueError(
                    f"{name} must be finite and positive"
                ) from error
            if not value_is_finite or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if self.device not in {"cpu", "mps", "auto"}:
            raise ValueError("device must be 'cpu', 'mps', or 'auto'")


class PiecewiseInverseContactTrainingResult(NamedTuple):
    """Trained inverse model, histories, device, and collocation times."""

    model: PiecewiseInverseContactResistancePINN
    history: InverseContactTrainingHistory
    device: str
    collocation_time: Tuple[float, ...]


class PiecewiseInverseContactResistanceValidation(NamedTuple):
    """Parameter, trajectory, observation, continuity, and transfer errors."""

    true_cold_contact_resistance: float
    inferred_cold_contact_resistance: float
    conventional_cold_contact_resistance: float
    absolute_parameter_error: float
    relative_parameter_error_percent: float
    cold_face_trajectory_rmse: float
    hot_face_trajectory_rmse: float
    cold_exchanger_trajectory_rmse: float
    hot_exchanger_trajectory_rmse: float
    cold_face_observation_rmse: float
    cold_exchanger_observation_rmse: float
    max_boundary_temperature_jump: float
    training_regime_metrics: ContactResistanceRegimeMetrics
    validation_regime_metrics: ContactResistanceRegimeMetrics
    test_regime_metrics: ContactResistanceRegimeMetrics


def _rmse(errors: Sequence[float]) -> float:
    if not errors:
        raise ValueError("at least one error is required")
    return math.sqrt(sum(error * error for error in errors) / len(errors))


def train_piecewise_inverse_contact_resistance(
    problem: IdealPiecewiseInverseContactProblem,
    config: PiecewiseInverseContactResistanceConfig = (
        PiecewiseInverseContactResistanceConfig()
    ),
) -> PiecewiseInverseContactTrainingResult:
    """Learn one resistance and three exactly joined temperature segments."""

    experiment = problem.experiment
    observations = problem.observations
    boundaries = current_segment_boundaries(
        experiment.current,
        experiment.duration,
    )
    if config.collocation_points < 2 * (len(boundaries) - 1):
        raise ValueError(
            "collocation count must provide at least two points per segment"
        )
    if (
        observations.time[0] < 0.0
        or observations.time[-1] > experiment.duration
    ):
        raise ValueError("observation times must lie within the experiment")

    device = select_device(config.device)
    torch.manual_seed(config.seed)
    model = PiecewiseInverseContactResistancePINN(
        duration=experiment.duration,
        transition_times=boundaries[1:-1],
        initial_temperatures=(
            experiment.initial_cold_face_temperature,
            experiment.initial_hot_face_temperature,
            experiment.initial_cold_exchanger_temperature,
            experiment.initial_hot_exchanger_temperature,
        ),
        initial_cold_contact_resistance=(
            config.initial_cold_contact_resistance
        ),
        hidden_width=config.hidden_width,
        hidden_layers=config.hidden_layers,
        temperature_scale=config.temperature_scale,
    ).to(device)
    optimizer = torch.optim.Adam(
        (
            {
                "params": model.temperature_model.parameters(),
                "lr": config.network_learning_rate,
            },
            {
                "params": (model.raw_cold_contact_resistance,),
                "lr": config.parameter_learning_rate,
            },
        )
    )
    collocation_time = piecewise_collocation_times(
        boundaries,
        config.collocation_points,
        device=device,
    )
    collocation_current = scheduled_current_tensor(
        experiment.current,
        collocation_time,
    )
    observation_time = torch.tensor(
        observations.time,
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 1)
    observed_temperatures = torch.tensor(
        tuple(zip(observations.cold_face, observations.cold_exchanger)),
        dtype=torch.float32,
        device=device,
    )

    total_losses = []
    physics_losses = []
    observation_losses = []
    resistances = []
    model.train()
    for _ in range(config.epochs):
        optimizer.zero_grad()
        residuals = contact_physics_residuals(
            model,
            collocation_time,
            experiment,
            cold_contact_resistance=model.cold_contact_resistance,
            current_values=collocation_current,
        )
        physics_loss = sum(
            (residual / config.residual_rate_scale).square().mean()
            for residual in residuals
        )
        predicted_cold_pair = model(observation_time)[:, (0, 2)]
        observation_loss = (
            (
                predicted_cold_pair - observed_temperatures
            )
            / config.observation_temperature_scale
        ).square().mean()
        total_loss = (
            config.physics_weight * physics_loss
            + config.observation_weight * observation_loss
        )
        total_loss.backward()
        optimizer.step()

        total_losses.append(float(total_loss.detach().cpu()))
        physics_losses.append(float(physics_loss.detach().cpu()))
        observation_losses.append(float(observation_loss.detach().cpu()))
        resistances.append(
            float(model.cold_contact_resistance.detach().cpu())
        )

    return PiecewiseInverseContactTrainingResult(
        model=model,
        history=InverseContactTrainingHistory(
            total_loss=tuple(total_losses),
            physics_loss=tuple(physics_losses),
            observation_loss=tuple(observation_losses),
            cold_contact_resistance=tuple(resistances),
        ),
        device=str(device),
        collocation_time=tuple(
            float(value) for value in collocation_time.detach().cpu().reshape(-1)
        ),
    )


def validate_piecewise_inverse_contact_resistance(
    training: PiecewiseInverseContactTrainingResult,
    problem: IdealPiecewiseInverseContactProblem,
) -> PiecewiseInverseContactResistanceValidation:
    """Compare pulse inference with dense truth and two unseen regimes."""

    experiment = problem.experiment
    observations = problem.observations
    reference = run_four_node_contact_experiment(experiment).trajectory
    prediction = predict_contact_trajectory(training.model, reference.time)
    observation_prediction = predict_contact_trajectory(
        training.model,
        observations.time,
    )
    true_resistance = experiment.thermal_parameters.cold_contact_resistance
    inferred_resistance = float(
        training.model.cold_contact_resistance.detach().cpu()
    )
    conventional_fit = fit_cold_contact_resistance((problem.dataset,))
    transfer_datasets = reference_contact_resistance_dataset_split()
    training_metrics = evaluate_contact_resistance_regime(
        inferred_resistance,
        problem.dataset,
    )
    validation_metrics = evaluate_contact_resistance_regime(
        inferred_resistance,
        transfer_datasets.validation[0],
    )
    test_metrics = evaluate_contact_resistance_regime(
        inferred_resistance,
        transfer_datasets.test[0],
    )
    absolute_error = abs(inferred_resistance - true_resistance)
    trajectory_errors = tuple(
        tuple(left - right for left, right in zip(predicted, expected))
        for predicted, expected in (
            (prediction.cold_face, reference.cold_face),
            (prediction.hot_face, reference.hot_face),
            (prediction.cold_exchanger, reference.cold_exchanger),
            (prediction.hot_exchanger, reference.hot_exchanger),
        )
    )
    jumps = training.model.boundary_temperature_jumps()
    max_jump = (
        0.0
        if jumps.numel() == 0
        else float(jumps.abs().max().detach().cpu())
    )
    return PiecewiseInverseContactResistanceValidation(
        true_cold_contact_resistance=true_resistance,
        inferred_cold_contact_resistance=inferred_resistance,
        conventional_cold_contact_resistance=(
            conventional_fit.inferred_cold_contact_resistance
        ),
        absolute_parameter_error=absolute_error,
        relative_parameter_error_percent=(
            100.0 * absolute_error / true_resistance
        ),
        cold_face_trajectory_rmse=_rmse(trajectory_errors[0]),
        hot_face_trajectory_rmse=_rmse(trajectory_errors[1]),
        cold_exchanger_trajectory_rmse=_rmse(trajectory_errors[2]),
        hot_exchanger_trajectory_rmse=_rmse(trajectory_errors[3]),
        cold_face_observation_rmse=_rmse(
            tuple(
                left - right
                for left, right in zip(
                    observation_prediction.cold_face,
                    observations.cold_face,
                )
            )
        ),
        cold_exchanger_observation_rmse=_rmse(
            tuple(
                left - right
                for left, right in zip(
                    observation_prediction.cold_exchanger,
                    observations.cold_exchanger,
                )
            )
        ),
        max_boundary_temperature_jump=max_jump,
        training_regime_metrics=training_metrics,
        validation_regime_metrics=validation_metrics,
        test_regime_metrics=test_metrics,
    )


def main() -> None:
    """Run the ideal switched-current inverse contact baseline."""

    parser = argparse.ArgumentParser(
        description="Infer cold contact resistance with the piecewise PINN"
    )
    parser.add_argument("--epochs", type=int, default=8_000)
    parser.add_argument(
        "--initial-resistance",
        type=float,
        default=0.5,
        help="initial cold contact resistance in K/W",
    )
    parser.add_argument(
        "--observation-interval",
        type=float,
        default=1.0,
        help="ideal cold-pair observation spacing in seconds",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "auto"),
        default="cpu",
    )
    arguments = parser.parse_args()

    problem = ideal_piecewise_inverse_contact_problem(
        observation_interval=arguments.observation_interval
    )
    training = train_piecewise_inverse_contact_resistance(
        problem,
        PiecewiseInverseContactResistanceConfig(
            epochs=arguments.epochs,
            initial_cold_contact_resistance=arguments.initial_resistance,
            device=arguments.device,
        ),
    )
    validation = validate_piecewise_inverse_contact_resistance(
        training,
        problem,
    )
    print(f"device: {training.device}")
    print(f"segments: {len(training.model.segment_boundaries) - 1}")
    print(f"cold-pair observation times: {len(problem.observations.time)}")
    print(
        "cold contact resistance: "
        f"true {validation.true_cold_contact_resistance:.6f} K/W, "
        f"PINN {validation.inferred_cold_contact_resistance:.6f} K/W, "
        "conventional "
        f"{validation.conventional_cold_contact_resistance:.6f} K/W"
    )
    print(
        "parameter error: "
        f"{validation.relative_parameter_error_percent:.3f}%"
    )
    print(
        "final normalized losses: "
        f"physics {training.history.physics_loss[-1]:.6e}, "
        f"observations {training.history.observation_loss[-1]:.6e}"
    )
    print(
        "maximum boundary temperature jump: "
        f"{validation.max_boundary_temperature_jump:.6e} K"
    )
    print(
        "unseen pulse all-sensor RMSE: "
        f"validation {validation.validation_regime_metrics.all_sensor_rmse:.6f} K, "
        f"test {validation.test_regime_metrics.all_sensor_rmse:.6f} K"
    )


if __name__ == "__main__":
    main()
