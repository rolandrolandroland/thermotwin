"""Infer the module thermal conductance from sparse temperature observations."""

import argparse
from bisect import bisect_left
from dataclasses import dataclass
import math
from typing import NamedTuple, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional

from ..simulation.two_node_experiments import (
    TwoNodeExperiment,
    constant_current_reference_experiment,
    run_two_node_experiment,
)
from .forward_two_node import (
    ForwardPINN,
    physics_residuals,
    predict_trajectory,
    select_device,
)


@dataclass(frozen=True)
class TemperatureObservations:
    """Sparse cold and hot temperature observations in kelvin."""

    time: Tuple[float, ...]
    cold: Tuple[float, ...]
    hot: Tuple[float, ...]

    def __post_init__(self) -> None:
        times = tuple(float(value) for value in self.time)
        cold = tuple(float(value) for value in self.cold)
        hot = tuple(float(value) for value in self.hot)
        object.__setattr__(self, "time", times)
        object.__setattr__(self, "cold", cold)
        object.__setattr__(self, "hot", hot)

        if not times or len(cold) != len(times) or len(hot) != len(times):
            raise ValueError(
                "time, cold, and hot observations must have equal "
                "nonzero lengths"
            )
        if any(not math.isfinite(value) for value in times + cold + hot):
            raise ValueError("all observations must be finite")
        if any(later <= earlier for earlier, later in zip(times, times[1:])):
            raise ValueError("observation times must strictly increase")


def _interpolate(
    times: Sequence[float],
    values: Sequence[float],
    target_time: float,
) -> float:
    index = bisect_left(times, target_time)
    if index == 0:
        return float(values[0])
    if index == len(times):
        return float(values[-1])
    if times[index] == target_time:
        return float(values[index])

    left_time = times[index - 1]
    right_time = times[index]
    fraction = (target_time - left_time) / (right_time - left_time)
    return float(
        values[index - 1]
        + fraction * (values[index] - values[index - 1])
    )


def synthetic_temperature_observations(
    experiment: TwoNodeExperiment,
    *,
    observation_interval: float = 5.0,
) -> TemperatureObservations:
    """Sample a noise-free RK4 trajectory at a regular observation interval."""

    if (
        not math.isfinite(observation_interval)
        or observation_interval <= 0.0
    ):
        raise ValueError("observation interval must be finite and positive")
    if not math.isfinite(experiment.duration) or experiment.duration <= 0.0:
        raise ValueError("experiment duration must be finite and positive")

    reference = run_two_node_experiment(experiment).trajectory
    observation_count = int(experiment.duration // observation_interval)
    observation_times = [
        index * observation_interval
        for index in range(observation_count + 1)
    ]
    if observation_times[-1] < experiment.duration:
        observation_times.append(experiment.duration)
    else:
        observation_times[-1] = experiment.duration

    return TemperatureObservations(
        time=tuple(observation_times),
        cold=tuple(
            _interpolate(reference.time, reference.cold, time)
            for time in observation_times
        ),
        hot=tuple(
            _interpolate(reference.time, reference.hot, time)
            for time in observation_times
        ),
    )


def _inverse_softplus(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("initial thermal conductance must be positive")
    if value > 20.0:
        return value
    return math.log(math.expm1(value))


class InverseThermalConductancePINN(nn.Module):
    """Predict temperatures and learn one positive thermal conductance."""

    def __init__(
        self,
        *,
        duration: float,
        initial_cold_temperature: float,
        initial_hot_temperature: float,
        initial_thermal_conductance: float,
        hidden_width: int = 32,
        hidden_layers: int = 2,
        temperature_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.temperature_model = ForwardPINN(
            duration=duration,
            initial_cold_temperature=initial_cold_temperature,
            initial_hot_temperature=initial_hot_temperature,
            hidden_width=hidden_width,
            hidden_layers=hidden_layers,
            temperature_scale=temperature_scale,
        )
        self.raw_thermal_conductance = nn.Parameter(
            torch.tensor(
                _inverse_softplus(initial_thermal_conductance),
                dtype=torch.float32,
            )
        )

    @property
    def thermal_conductance(self) -> Tensor:
        """Return the positive inferred conductance in W/K."""

        return functional.softplus(self.raw_thermal_conductance)

    def forward(self, time: Tensor) -> Tensor:
        return self.temperature_model(time)


@dataclass(frozen=True)
class InverseThermalConductanceConfig:
    """Network and optimization settings for the first inverse problem."""

    hidden_width: int = 32
    hidden_layers: int = 2
    collocation_points: int = 128
    epochs: int = 4_000
    network_learning_rate: float = 1e-3
    parameter_learning_rate: float = 5e-3
    initial_thermal_conductance: float = 0.2
    temperature_scale: float = 10.0
    residual_rate_scale: float = 0.1
    observation_temperature_scale: float = 1.0
    physics_weight: float = 1.0
    observation_weight: float = 1.0
    seed: int = 7
    device: str = "cpu"

    def __post_init__(self) -> None:
        positive_integer_settings = (
            ("hidden width", self.hidden_width),
            ("hidden layers", self.hidden_layers),
            ("collocation points", self.collocation_points),
            ("epochs", self.epochs),
        )
        for name, value in positive_integer_settings:
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.collocation_points < 2:
            raise ValueError("at least two collocation points are required")

        positive_settings = (
            ("network learning rate", self.network_learning_rate),
            ("parameter learning rate", self.parameter_learning_rate),
            ("initial thermal conductance", self.initial_thermal_conductance),
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
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.device not in {"cpu", "mps", "auto"}:
            raise ValueError("device must be 'cpu', 'mps', or 'auto'")


class InverseTrainingHistory(NamedTuple):
    """Per-epoch dimensionless losses and inferred conductance in W/K."""

    total_loss: Tuple[float, ...]
    physics_loss: Tuple[float, ...]
    observation_loss: Tuple[float, ...]
    thermal_conductance: Tuple[float, ...]


class InverseTrainingResult(NamedTuple):
    """Trained inverse model, optimization history, and selected device."""

    model: InverseThermalConductancePINN
    history: InverseTrainingHistory
    device: str


class InverseThermalConductanceValidation(NamedTuple):
    """Parameter and temperature errors for the synthetic inverse problem."""

    true_thermal_conductance: float
    inferred_thermal_conductance: float
    absolute_parameter_error: float
    relative_parameter_error_percent: float
    cold_trajectory_rmse: float
    hot_trajectory_rmse: float
    cold_observation_rmse: float
    hot_observation_rmse: float


def _rmse(errors: Sequence[float]) -> float:
    if not errors:
        raise ValueError("at least one error is required")
    return math.sqrt(sum(error**2 for error in errors) / len(errors))


def train_inverse_thermal_conductance(
    experiment: TwoNodeExperiment,
    observations: TemperatureObservations,
    config: InverseThermalConductanceConfig = (
        InverseThermalConductanceConfig()
    ),
) -> InverseTrainingResult:
    """Jointly fit temperatures and one positive K from physics plus data."""

    if (
        observations.time[0] < 0.0
        or observations.time[-1] > experiment.duration
    ):
        raise ValueError("observation times must lie within the experiment")

    device = select_device(config.device)
    torch.manual_seed(config.seed)
    model = InverseThermalConductancePINN(
        duration=experiment.duration,
        initial_cold_temperature=experiment.initial_cold_temperature,
        initial_hot_temperature=experiment.initial_hot_temperature,
        initial_thermal_conductance=config.initial_thermal_conductance,
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
                "params": (model.raw_thermal_conductance,),
                "lr": config.parameter_learning_rate,
            },
        )
    )
    collocation_time = torch.linspace(
        0.0,
        experiment.duration,
        config.collocation_points,
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 1)
    observation_time = torch.tensor(
        observations.time,
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 1)
    observed_temperatures = torch.tensor(
        tuple(zip(observations.cold, observations.hot)),
        dtype=torch.float32,
        device=device,
    )

    total_losses = []
    physics_losses = []
    observation_losses = []
    conductances = []
    model.train()
    for _ in range(config.epochs):
        optimizer.zero_grad()
        residuals = physics_residuals(
            model,
            collocation_time,
            experiment,
            thermal_conductance=model.thermal_conductance,
        )
        physics_loss = (
            (residuals.cold / config.residual_rate_scale).square().mean()
            + (residuals.hot / config.residual_rate_scale).square().mean()
        )
        predicted_observations = model(observation_time)
        observation_loss = (
            (
                predicted_observations - observed_temperatures
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
        conductances.append(float(model.thermal_conductance.detach().cpu()))

    return InverseTrainingResult(
        model=model,
        history=InverseTrainingHistory(
            total_loss=tuple(total_losses),
            physics_loss=tuple(physics_losses),
            observation_loss=tuple(observation_losses),
            thermal_conductance=tuple(conductances),
        ),
        device=str(device),
    )


def validate_inverse_thermal_conductance(
    training: InverseTrainingResult,
    experiment: TwoNodeExperiment,
    observations: TemperatureObservations,
) -> InverseThermalConductanceValidation:
    """Validate inferred K and temperatures against synthetic truth."""

    reference = run_two_node_experiment(experiment).trajectory
    prediction = predict_trajectory(training.model, reference.time)
    observation_prediction = predict_trajectory(
        training.model,
        observations.time,
    )
    true_conductance = experiment.thermoelectric_parameters.thermal_conductance
    inferred_conductance = float(
        training.model.thermal_conductance.detach().cpu()
    )
    absolute_parameter_error = abs(inferred_conductance - true_conductance)

    return InverseThermalConductanceValidation(
        true_thermal_conductance=true_conductance,
        inferred_thermal_conductance=inferred_conductance,
        absolute_parameter_error=absolute_parameter_error,
        relative_parameter_error_percent=(
            100.0 * absolute_parameter_error / true_conductance
        ),
        cold_trajectory_rmse=_rmse(
            tuple(
                predicted - expected
                for predicted, expected in zip(
                    prediction.cold,
                    reference.cold,
                )
            )
        ),
        hot_trajectory_rmse=_rmse(
            tuple(
                predicted - expected
                for predicted, expected in zip(
                    prediction.hot,
                    reference.hot,
                )
            )
        ),
        cold_observation_rmse=_rmse(
            tuple(
                predicted - expected
                for predicted, expected in zip(
                    observation_prediction.cold,
                    observations.cold,
                )
            )
        ),
        hot_observation_rmse=_rmse(
            tuple(
                predicted - expected
                for predicted, expected in zip(
                    observation_prediction.hot,
                    observations.hot,
                )
            )
        ),
    )


def main() -> None:
    """Run the first noise-free synthetic K-inference baseline."""

    parser = argparse.ArgumentParser(
        description="Infer ThermoTwin thermal conductance from synthetic data"
    )
    parser.add_argument("--epochs", type=int, default=4_000)
    parser.add_argument(
        "--initial-k",
        type=float,
        default=0.2,
        help="initial thermal conductance guess in W/K",
    )
    parser.add_argument(
        "--observation-interval",
        type=float,
        default=5.0,
        help="synthetic observation spacing in seconds",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "auto"),
        default="cpu",
    )
    arguments = parser.parse_args()

    experiment = constant_current_reference_experiment()
    observations = synthetic_temperature_observations(
        experiment,
        observation_interval=arguments.observation_interval,
    )
    training = train_inverse_thermal_conductance(
        experiment,
        observations,
        InverseThermalConductanceConfig(
            epochs=arguments.epochs,
            initial_thermal_conductance=arguments.initial_k,
            device=arguments.device,
        ),
    )
    validation = validate_inverse_thermal_conductance(
        training,
        experiment,
        observations,
    )
    print(f"device: {training.device}")
    print(f"observations: {len(observations.time)}")
    print(
        "thermal conductance: "
        f"true {validation.true_thermal_conductance:.6f} W/K, "
        f"inferred {validation.inferred_thermal_conductance:.6f} W/K"
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
        "trajectory RMSE: "
        f"cold {validation.cold_trajectory_rmse:.6f} K, "
        f"hot {validation.hot_trajectory_rmse:.6f} K"
    )
    print(
        "observation RMSE: "
        f"cold {validation.cold_observation_rmse:.6f} K, "
        f"hot {validation.hot_observation_rmse:.6f} K"
    )


if __name__ == "__main__":
    main()
