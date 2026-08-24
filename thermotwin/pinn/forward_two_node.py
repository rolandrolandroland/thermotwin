"""Small physics-informed neural network for the two-node forward problem.

PyTorch is intentionally imported only by this optional module. Importing the
core :mod:`thermotwin` package therefore does not require a machine-learning
dependency.
"""

from dataclasses import dataclass
import math
from typing import NamedTuple, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from ..core.controls import PiecewiseConstantCurrent, current_at
from ..simulation.two_node_experiments import (
    TwoNodeExperiment,
    constant_current_reference_experiment,
    run_two_node_experiment,
)
from ..physics.two_node import TemperatureTrajectory


@dataclass(frozen=True)
class ForwardPINNConfig:
    """Training and network settings for the small forward PINN."""

    hidden_width: int = 32
    hidden_layers: int = 2
    collocation_points: int = 128
    epochs: int = 2_000
    learning_rate: float = 1e-3
    temperature_scale: float = 10.0
    seed: int = 7
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.hidden_width <= 0:
            raise ValueError("hidden width must be positive")
        if self.hidden_layers <= 0:
            raise ValueError("hidden layer count must be positive")
        if self.collocation_points < 2:
            raise ValueError("at least two collocation points are required")
        if self.epochs <= 0:
            raise ValueError("epoch count must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning rate must be finite and positive")
        if (
            not math.isfinite(self.temperature_scale)
            or self.temperature_scale <= 0
        ):
            raise ValueError("temperature scale must be finite and positive")
        if self.device not in {"cpu", "mps", "auto"}:
            raise ValueError("device must be 'cpu', 'mps', or 'auto'")


class ForwardPINN(nn.Module):
    """Map time to cold and hot temperatures while enforcing the initial state.

    The raw network output is transformed as

    ``T(t) = T(0) + (t / duration) * temperature_scale * raw(t)``.

    The multiplier makes both predicted temperatures exactly equal to their
    specified initial values at ``t = 0``; no approximate initial-condition
    penalty is needed in the loss.
    """

    def __init__(
        self,
        *,
        duration: float,
        initial_cold_temperature: float,
        initial_hot_temperature: float,
        hidden_width: int = 32,
        hidden_layers: int = 2,
        temperature_scale: float = 10.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("PINN duration must be finite and positive")
        if hidden_width <= 0:
            raise ValueError("hidden width must be positive")
        if hidden_layers <= 0:
            raise ValueError("hidden layer count must be positive")
        if not math.isfinite(temperature_scale) or temperature_scale <= 0:
            raise ValueError("temperature scale must be finite and positive")

        layers = [nn.Linear(1, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Linear(hidden_width, hidden_width), nn.Tanh()))
        layers.append(nn.Linear(hidden_width, 2))
        self.network = nn.Sequential(*layers)

        self.register_buffer(
            "_duration",
            torch.tensor(float(duration), dtype=torch.float32),
        )
        self.register_buffer(
            "_initial_temperatures",
            torch.tensor(
                [initial_cold_temperature, initial_hot_temperature],
                dtype=torch.float32,
            ).reshape(1, 2),
        )
        self.register_buffer(
            "_temperature_scale",
            torch.tensor(float(temperature_scale), dtype=torch.float32),
        )

    def forward(self, time: Tensor) -> Tensor:
        """Return columns ``(T_c, T_h)`` for a column of times in seconds."""

        if time.ndim == 1:
            time = time.reshape(-1, 1)
        if time.ndim != 2 or time.shape[1] != 1:
            raise ValueError("time must have shape (samples,) or (samples, 1)")

        normalized_time = 2.0 * time / self._duration - 1.0
        raw_temperatures = self.network(normalized_time)
        progress = time / self._duration
        return (
            self._initial_temperatures
            + progress * self._temperature_scale * raw_temperatures
        )


class PINNResiduals(NamedTuple):
    """Cold and hot ODE residuals in K/s."""

    cold: Tensor
    hot: Tensor


class PINNTrainingResult(NamedTuple):
    """Trained model, scalar loss history, and selected device."""

    model: ForwardPINN
    loss_history: Tuple[float, ...]
    device: str


class PINNValidation(NamedTuple):
    """Temperature errors against an RK4 reference trajectory, in kelvin."""

    cold_rmse: float
    hot_rmse: float
    cold_max_absolute_error: float
    hot_max_absolute_error: float


def select_device(requested: str = "cpu") -> torch.device:
    """Select CPU by default, with optional explicit or automatic MPS use."""

    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")
    if requested == "auto":
        return torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    raise ValueError("device must be 'cpu', 'mps', or 'auto'")


def _constant_current(experiment: TwoNodeExperiment) -> float:
    if isinstance(experiment.current, PiecewiseConstantCurrent):
        if experiment.current.transition_times:
            raise ValueError(
                "the first forward PINN supports constant current only"
            )
        return experiment.current.values[0]
    return current_at(experiment.current, 0.0)


def physics_residuals(
    model: nn.Module,
    time: Tensor,
    experiment: TwoNodeExperiment,
    *,
    thermal_conductance: Optional[Tensor] = None,
) -> PINNResiduals:
    """Evaluate both agreed two-node ODE residuals at collocation times.

    Each residual is ``predicted temperature rate - physical RHS``. A perfect
    forward solution therefore makes both returned tensors zero. An optional
    differentiable thermal conductance replaces the fixed experiment value for
    inverse parameter inference.
    """

    if time.ndim == 1:
        time = time.reshape(-1, 1)
    if time.ndim != 2 or time.shape[1] != 1:
        raise ValueError("time must have shape (samples,) or (samples, 1)")

    differentiable_time = time.clone().detach().requires_grad_(True)
    temperatures = model(differentiable_time)
    if temperatures.ndim != 2 or temperatures.shape[1] != 2:
        raise ValueError("model output must have shape (samples, 2)")

    cold_temperature = temperatures[:, 0:1]
    hot_temperature = temperatures[:, 1:2]
    cold_temperature_rate = torch.autograd.grad(
        cold_temperature,
        differentiable_time,
        grad_outputs=torch.ones_like(cold_temperature),
        create_graph=True,
        retain_graph=True,
    )[0]
    hot_temperature_rate = torch.autograd.grad(
        hot_temperature,
        differentiable_time,
        grad_outputs=torch.ones_like(hot_temperature),
        create_graph=True,
    )[0]

    thermoelectric = experiment.thermoelectric_parameters
    thermal = experiment.thermal_parameters
    current = _constant_current(experiment)
    effective_thermal_conductance = (
        thermoelectric.thermal_conductance
        if thermal_conductance is None
        else thermal_conductance
    )
    temperature_difference = hot_temperature - cold_temperature
    half_joule_heat = (
        0.5 * current**2 * thermoelectric.electrical_resistance
    )
    cold_heat = (
        thermoelectric.seebeck_coefficient * current * cold_temperature
        - half_joule_heat
        - effective_thermal_conductance * temperature_difference
    )
    hot_heat = (
        thermoelectric.seebeck_coefficient * current * hot_temperature
        + half_joule_heat
        - effective_thermal_conductance * temperature_difference
    )

    cold_rhs = (
        thermal.cold_reservoir_conductance
        * (experiment.cold_reservoir_temperature - cold_temperature)
        + experiment.cold_external_heat
        - cold_heat
    ) / thermal.cold_thermal_capacitance
    hot_rhs = (
        thermal.hot_reservoir_conductance
        * (experiment.hot_reservoir_temperature - hot_temperature)
        + experiment.hot_external_heat
        + hot_heat
    ) / thermal.hot_thermal_capacitance

    return PINNResiduals(
        cold=cold_temperature_rate - cold_rhs,
        hot=hot_temperature_rate - hot_rhs,
    )


def train_forward_pinn(
    experiment: TwoNodeExperiment,
    config: ForwardPINNConfig = ForwardPINNConfig(),
) -> PINNTrainingResult:
    """Train only on the ODE residuals; RK4 data are not used in the loss."""

    _constant_current(experiment)
    if not math.isfinite(experiment.duration) or experiment.duration <= 0:
        raise ValueError("PINN experiment duration must be positive")

    device = select_device(config.device)
    torch.manual_seed(config.seed)
    model = ForwardPINN(
        duration=experiment.duration,
        initial_cold_temperature=experiment.initial_cold_temperature,
        initial_hot_temperature=experiment.initial_hot_temperature,
        hidden_width=config.hidden_width,
        hidden_layers=config.hidden_layers,
        temperature_scale=config.temperature_scale,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )
    collocation_time = torch.linspace(
        0.0,
        experiment.duration,
        config.collocation_points,
        dtype=torch.float32,
        device=device,
    ).reshape(-1, 1)

    losses = []
    model.train()
    for _ in range(config.epochs):
        optimizer.zero_grad()
        residuals = physics_residuals(
            model,
            collocation_time,
            experiment,
        )
        loss = residuals.cold.square().mean() + residuals.hot.square().mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return PINNTrainingResult(
        model=model,
        loss_history=tuple(losses),
        device=str(device),
    )


def predict_trajectory(
    model: ForwardPINN,
    times: Sequence[float],
) -> TemperatureTrajectory:
    """Evaluate a trained PINN and return the conventional trajectory type."""

    time_values = tuple(float(value) for value in times)
    if not time_values:
        raise ValueError("at least one prediction time is required")

    parameter = next(model.parameters())
    time_tensor = torch.tensor(
        time_values,
        dtype=parameter.dtype,
        device=parameter.device,
    ).reshape(-1, 1)
    model.eval()
    with torch.no_grad():
        temperatures = model(time_tensor).detach().cpu()

    return TemperatureTrajectory(
        time=time_values,
        cold=tuple(float(value) for value in temperatures[:, 0]),
        hot=tuple(float(value) for value in temperatures[:, 1]),
    )


def validate_against_rk4(
    model: ForwardPINN,
    experiment: TwoNodeExperiment,
) -> PINNValidation:
    """Compare a trained PINN with RK4 without using RK4 during training."""

    reference = run_two_node_experiment(experiment).trajectory
    prediction = predict_trajectory(model, reference.time)
    cold_errors = tuple(
        predicted - expected
        for predicted, expected in zip(prediction.cold, reference.cold)
    )
    hot_errors = tuple(
        predicted - expected
        for predicted, expected in zip(prediction.hot, reference.hot)
    )

    return PINNValidation(
        cold_rmse=math.sqrt(
            sum(error**2 for error in cold_errors) / len(cold_errors)
        ),
        hot_rmse=math.sqrt(
            sum(error**2 for error in hot_errors) / len(hot_errors)
        ),
        cold_max_absolute_error=max(abs(error) for error in cold_errors),
        hot_max_absolute_error=max(abs(error) for error in hot_errors),
    )


def main() -> None:
    """Train and validate the agreed reference problem from the command line."""

    experiment = constant_current_reference_experiment()
    training = train_forward_pinn(experiment)
    validation = validate_against_rk4(training.model, experiment)
    print(f"device: {training.device}")
    print(f"initial physics loss: {training.loss_history[0]:.6e}")
    print(f"final physics loss: {training.loss_history[-1]:.6e}")
    print(f"cold RMSE: {validation.cold_rmse:.6f} K")
    print(f"hot RMSE: {validation.hot_rmse:.6f} K")


if __name__ == "__main__":
    main()
