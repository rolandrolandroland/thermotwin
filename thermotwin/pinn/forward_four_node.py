"""Four-node contact-aware forward physics-informed neural network.

PyTorch remains an optional dependency. Importing :mod:`thermotwin` does not
import this module or require a machine-learning runtime.
"""

from dataclasses import dataclass
import math
from typing import NamedTuple, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from ..simulation.four_node_experiments import (
    FourNodeContactExperiment,
    constant_current_contact_reference_experiment,
    run_four_node_contact_experiment,
)
from ..physics.four_node import FourNodeContactTemperatureTrajectory
from ..core.controls import PiecewiseConstantCurrent, current_at
from .forward_two_node import select_device


@dataclass(frozen=True)
class ContactForwardPINNConfig:
    """Network and CPU-first training settings for the four-node PINN."""

    hidden_width: int = 32
    hidden_layers: int = 2
    collocation_points: int = 128
    epochs: int = 3_000
    learning_rate: float = 1e-3
    temperature_scale: float = 10.0
    seed: int = 11
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
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning rate must be finite and positive")
        if (
            not math.isfinite(self.temperature_scale)
            or self.temperature_scale <= 0
        ):
            raise ValueError("temperature scale must be finite and positive")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if self.device not in {"cpu", "mps", "auto"}:
            raise ValueError("device must be 'cpu', 'mps', or 'auto'")


class ContactForwardPINN(nn.Module):
    """Map time to four temperatures with exact initial conditions."""

    def __init__(
        self,
        *,
        duration: float,
        initial_temperatures: Sequence[float],
        hidden_width: int = 32,
        hidden_layers: int = 2,
        temperature_scale: float = 10.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("PINN duration must be finite and positive")
        try:
            initial_temperatures = tuple(
                float(value) for value in initial_temperatures
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "exactly four finite initial temperatures are required"
            ) from error
        if (
            len(initial_temperatures) != 4
            or any(not math.isfinite(value) for value in initial_temperatures)
        ):
            raise ValueError(
                "exactly four finite initial temperatures are required"
            )
        if (
            not isinstance(hidden_width, int)
            or isinstance(hidden_width, bool)
            or hidden_width <= 0
        ):
            raise ValueError("hidden width must be positive")
        if (
            not isinstance(hidden_layers, int)
            or isinstance(hidden_layers, bool)
            or hidden_layers <= 0
        ):
            raise ValueError("hidden layer count must be positive")
        if not math.isfinite(temperature_scale) or temperature_scale <= 0:
            raise ValueError("temperature scale must be finite and positive")

        layers = [nn.Linear(1, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Linear(hidden_width, hidden_width), nn.Tanh()))
        layers.append(nn.Linear(hidden_width, 4))
        self.network = nn.Sequential(*layers)

        self.register_buffer(
            "_duration",
            torch.tensor(float(duration), dtype=torch.float32),
        )
        self.register_buffer(
            "_initial_temperatures",
            torch.tensor(
                initial_temperatures,
                dtype=torch.float32,
            ).reshape(1, 4),
        )
        self.register_buffer(
            "_temperature_scale",
            torch.tensor(float(temperature_scale), dtype=torch.float32),
        )

    def forward(self, time: Tensor) -> Tensor:
        """Return ``(T_cf, T_hf, T_cx, T_hx)`` at supplied times."""

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


class ContactPINNResiduals(NamedTuple):
    """Four contact-model ODE residual columns in K/s."""

    cold_face: Tensor
    hot_face: Tensor
    cold_exchanger: Tensor
    hot_exchanger: Tensor


class ContactPINNTrainingResult(NamedTuple):
    """Trained contact PINN, scalar loss history, and selected device."""

    model: ContactForwardPINN
    loss_history: Tuple[float, ...]
    device: str


class ContactPINNValidation(NamedTuple):
    """Four-node temperature errors against RK4, in kelvin."""

    cold_face_rmse: float
    hot_face_rmse: float
    cold_exchanger_rmse: float
    hot_exchanger_rmse: float
    cold_face_max_absolute_error: float
    hot_face_max_absolute_error: float
    cold_exchanger_max_absolute_error: float
    hot_exchanger_max_absolute_error: float


def _constant_current(experiment: FourNodeContactExperiment) -> float:
    if isinstance(experiment.current, PiecewiseConstantCurrent):
        if experiment.current.transition_times:
            raise ValueError(
                "the first contact forward PINN supports constant current only"
            )
        return experiment.current.values[0]
    return current_at(experiment.current, 0.0)


def _temperature_rate(
    temperature: Tensor,
    differentiable_time: Tensor,
    *,
    retain_graph: bool,
) -> Tensor:
    return torch.autograd.grad(
        temperature,
        differentiable_time,
        grad_outputs=torch.ones_like(temperature),
        create_graph=True,
        retain_graph=retain_graph,
    )[0]


def contact_physics_residuals(
    model: nn.Module,
    time: Tensor,
    experiment: FourNodeContactExperiment,
    *,
    cold_contact_resistance: Optional[Tensor] = None,
    current_values: Optional[Tensor] = None,
) -> ContactPINNResiduals:
    """Evaluate all four contact-aware ODE residuals at supplied times.

    A differentiable cold contact resistance may replace the fixed experiment
    value for inverse parameter inference. Supplied current values allow a
    piecewise model to evaluate known scheduled current at each time. The
    original forward model omits both arguments and retains its known constant
    experiment values.
    """

    if time.ndim == 1:
        time = time.reshape(-1, 1)
    if time.ndim != 2 or time.shape[1] != 1:
        raise ValueError("time must have shape (samples,) or (samples, 1)")

    differentiable_time = time.clone().detach().requires_grad_(True)
    temperatures = model(differentiable_time)
    if temperatures.ndim != 2 or temperatures.shape[1] != 4:
        raise ValueError("model output must have shape (samples, 4)")

    cold_face = temperatures[:, 0:1]
    hot_face = temperatures[:, 1:2]
    cold_exchanger = temperatures[:, 2:3]
    hot_exchanger = temperatures[:, 3:4]
    cold_face_rate = _temperature_rate(
        cold_face,
        differentiable_time,
        retain_graph=True,
    )
    hot_face_rate = _temperature_rate(
        hot_face,
        differentiable_time,
        retain_graph=True,
    )
    cold_exchanger_rate = _temperature_rate(
        cold_exchanger,
        differentiable_time,
        retain_graph=True,
    )
    hot_exchanger_rate = _temperature_rate(
        hot_exchanger,
        differentiable_time,
        retain_graph=True,
    )

    thermoelectric = experiment.thermoelectric_parameters
    thermal = experiment.thermal_parameters
    if current_values is None:
        current = _constant_current(experiment)
    else:
        current = current_values
        if current.ndim == 1:
            current = current.reshape(-1, 1)
        if current.ndim != 2 or current.shape != differentiable_time.shape:
            raise ValueError(
                "current values must match the supplied time shape"
            )
    face_difference = hot_face - cold_face
    half_joule_heat = (
        0.5 * current**2 * thermoelectric.electrical_resistance
    )
    cold_module_heat = (
        thermoelectric.seebeck_coefficient * current * cold_face
        - half_joule_heat
        - thermoelectric.thermal_conductance * face_difference
    )
    hot_module_heat = (
        thermoelectric.seebeck_coefficient * current * hot_face
        + half_joule_heat
        - thermoelectric.thermal_conductance * face_difference
    )
    effective_cold_contact_resistance = (
        thermal.cold_contact_resistance
        if cold_contact_resistance is None
        else cold_contact_resistance
    )
    cold_contact_heat = (
        cold_exchanger - cold_face
    ) / effective_cold_contact_resistance
    hot_contact_heat = (
        hot_face - hot_exchanger
    ) / thermal.hot_contact_resistance

    cold_face_rhs = (
        cold_contact_heat - cold_module_heat
    ) / thermal.cold_face_thermal_capacitance
    hot_face_rhs = (
        hot_module_heat - hot_contact_heat
    ) / thermal.hot_face_thermal_capacitance
    cold_exchanger_rhs = (
        thermal.cold_reservoir_conductance
        * (experiment.cold_reservoir_temperature - cold_exchanger)
        + experiment.cold_external_heat
        - cold_contact_heat
    ) / thermal.cold_exchanger_thermal_capacitance
    hot_exchanger_rhs = (
        thermal.hot_reservoir_conductance
        * (experiment.hot_reservoir_temperature - hot_exchanger)
        + experiment.hot_external_heat
        + hot_contact_heat
    ) / thermal.hot_exchanger_thermal_capacitance

    return ContactPINNResiduals(
        cold_face=cold_face_rate - cold_face_rhs,
        hot_face=hot_face_rate - hot_face_rhs,
        cold_exchanger=cold_exchanger_rate - cold_exchanger_rhs,
        hot_exchanger=hot_exchanger_rate - hot_exchanger_rhs,
    )


def contact_physics_loss(residuals: ContactPINNResiduals) -> Tensor:
    """Return the equal-weight sum of four mean squared residuals."""

    return sum(
        residual.square().mean()
        for residual in residuals
    )


def train_contact_forward_pinn(
    experiment: FourNodeContactExperiment,
    config: ContactForwardPINNConfig = ContactForwardPINNConfig(),
) -> ContactPINNTrainingResult:
    """Train on four physics residuals without using RK4 temperatures."""

    _constant_current(experiment)
    if not math.isfinite(experiment.duration) or experiment.duration <= 0.0:
        raise ValueError("PINN experiment duration must be positive")
    device = select_device(config.device)
    torch.manual_seed(config.seed)
    model = ContactForwardPINN(
        duration=experiment.duration,
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
        residuals = contact_physics_residuals(
            model,
            collocation_time,
            experiment,
        )
        loss = contact_physics_loss(residuals)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return ContactPINNTrainingResult(
        model=model,
        loss_history=tuple(losses),
        device=str(device),
    )


def predict_contact_trajectory(
    model: nn.Module,
    times: Sequence[float],
) -> FourNodeContactTemperatureTrajectory:
    """Evaluate a trained contact PINN in the conventional trajectory type."""

    time_values = tuple(float(value) for value in times)
    if not time_values:
        raise ValueError("at least one prediction time is required")
    if any(not math.isfinite(value) for value in time_values):
        raise ValueError("prediction times must be finite")
    parameter = next(model.parameters())
    time_tensor = torch.tensor(
        time_values,
        dtype=parameter.dtype,
        device=parameter.device,
    ).reshape(-1, 1)
    model.eval()
    with torch.no_grad():
        temperatures = model(time_tensor).detach().cpu()
    return FourNodeContactTemperatureTrajectory(
        time=time_values,
        cold_face=tuple(float(value) for value in temperatures[:, 0]),
        hot_face=tuple(float(value) for value in temperatures[:, 1]),
        cold_exchanger=tuple(float(value) for value in temperatures[:, 2]),
        hot_exchanger=tuple(float(value) for value in temperatures[:, 3]),
    )


def _rmse(errors: Sequence[float]) -> float:
    return math.sqrt(sum(error * error for error in errors) / len(errors))


def validate_contact_pinn_against_rk4(
    model: nn.Module,
    experiment: FourNodeContactExperiment,
) -> ContactPINNValidation:
    """Compare all four learned histories with withheld RK4 temperatures."""

    reference = run_four_node_contact_experiment(experiment).trajectory
    prediction = predict_contact_trajectory(model, reference.time)
    errors = tuple(
        tuple(predicted - expected for predicted, expected in zip(left, right))
        for left, right in (
            (prediction.cold_face, reference.cold_face),
            (prediction.hot_face, reference.hot_face),
            (prediction.cold_exchanger, reference.cold_exchanger),
            (prediction.hot_exchanger, reference.hot_exchanger),
        )
    )
    return ContactPINNValidation(
        cold_face_rmse=_rmse(errors[0]),
        hot_face_rmse=_rmse(errors[1]),
        cold_exchanger_rmse=_rmse(errors[2]),
        hot_exchanger_rmse=_rmse(errors[3]),
        cold_face_max_absolute_error=max(abs(value) for value in errors[0]),
        hot_face_max_absolute_error=max(abs(value) for value in errors[1]),
        cold_exchanger_max_absolute_error=max(
            abs(value) for value in errors[2]
        ),
        hot_exchanger_max_absolute_error=max(
            abs(value) for value in errors[3]
        ),
    )


def main() -> None:
    """Train and validate the frozen four-node contact reference."""

    experiment = constant_current_contact_reference_experiment()
    training = train_contact_forward_pinn(experiment)
    validation = validate_contact_pinn_against_rk4(
        training.model,
        experiment,
    )
    print(f"device: {training.device}")
    print(f"initial physics loss: {training.loss_history[0]:.6e}")
    print(f"final physics loss: {training.loss_history[-1]:.6e}")
    print(f"cold face RMSE: {validation.cold_face_rmse:.6f} K")
    print(f"hot face RMSE: {validation.hot_face_rmse:.6f} K")
    print(
        "cold exchanger RMSE: "
        f"{validation.cold_exchanger_rmse:.6f} K"
    )
    print(
        "hot exchanger RMSE: "
        f"{validation.hot_exchanger_rmse:.6f} K"
    )


if __name__ == "__main__":
    main()
