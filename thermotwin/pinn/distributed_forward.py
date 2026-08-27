"""Forward PINN for the coupled 1-D thermoelectric leg and face nodes."""

from dataclasses import dataclass
import math
from typing import NamedTuple, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from ..core.controls import PiecewiseConstantCurrent
from ..physics.distributed import (
    ConstantProperty,
    PiecewiseLinearProperty,
    TemperatureProperty,
)
from ..simulation.distributed import (
    DistributedLegExperiment,
    DistributedTemperatureTrajectory,
    run_distributed_leg_experiment,
)
from .forward_two_node import select_device


def _constant_current(experiment: DistributedLegExperiment) -> float:
    if isinstance(experiment.current, PiecewiseConstantCurrent):
        if experiment.current.transition_times:
            raise ValueError(
                "the first distributed PINN supports constant current only"
            )
        return float(experiment.current.values[0])
    value = float(experiment.current)
    if not math.isfinite(value):
        raise ValueError("distributed PINN current must be finite")
    return value


def _property_tensor(
    prop: TemperatureProperty,
    temperature: Tensor,
    values_override: Optional[Tensor] = None,
) -> Tensor:
    if isinstance(prop, ConstantProperty):
        if values_override is not None:
            if values_override.numel() != 1:
                raise ValueError("constant property override must have one value")
            return torch.ones_like(temperature) * values_override.reshape(())
        return torch.ones_like(temperature) * prop.constant
    if not isinstance(prop, PiecewiseLinearProperty):
        raise TypeError("PINN properties must be constant or piecewise linear")
    knots = torch.tensor(
        prop.temperatures,
        dtype=temperature.dtype,
        device=temperature.device,
    )
    values = (
        torch.tensor(prop.values, dtype=temperature.dtype, device=temperature.device)
        if values_override is None
        else values_override.to(dtype=temperature.dtype, device=temperature.device)
    )
    if values.numel() != len(prop.values):
        raise ValueError("piecewise property override has the wrong length")
    clipped = torch.clamp(temperature, min=knots[0], max=knots[-1])
    indices = torch.bucketize(clipped.detach(), knots[1:-1])
    left_temperature = knots[indices]
    right_temperature = knots[indices + 1]
    fraction = (clipped - left_temperature) / (
        right_temperature - left_temperature
    )
    return values[indices] + fraction * (values[indices + 1] - values[indices])


def _property_slope_tensor(
    prop: TemperatureProperty,
    temperature: Tensor,
    values_override: Optional[Tensor] = None,
) -> Tensor:
    if isinstance(prop, ConstantProperty):
        return torch.zeros_like(temperature)
    if not isinstance(prop, PiecewiseLinearProperty):
        raise TypeError("PINN properties must be constant or piecewise linear")
    knots = torch.tensor(
        prop.temperatures,
        dtype=temperature.dtype,
        device=temperature.device,
    )
    values = (
        torch.tensor(prop.values, dtype=temperature.dtype, device=temperature.device)
        if values_override is None
        else values_override.to(dtype=temperature.dtype, device=temperature.device)
    )
    clipped = torch.clamp(temperature.detach(), min=knots[0], max=knots[-1])
    indices = torch.bucketize(clipped, knots[1:-1])
    slope = (values[indices + 1] - values[indices]) / (
        knots[indices + 1] - knots[indices]
    )
    inside = (temperature >= knots[0]) & (temperature <= knots[-1])
    return torch.where(inside, slope, torch.zeros_like(slope))


class DistributedForwardPINN(nn.Module):
    """Map ``(x, t)`` to T while enforcing the complete initial profile."""

    def __init__(
        self,
        *,
        length: float,
        duration: float,
        initial_cold_temperature: float,
        initial_hot_temperature: float,
        hidden_width: int = 48,
        hidden_layers: int = 3,
        temperature_scale: float = 10.0,
    ) -> None:
        super().__init__()
        for name, value in (
            ("length", length),
            ("duration", duration),
            ("temperature scale", temperature_scale),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if hidden_width <= 0 or hidden_layers <= 0:
            raise ValueError("hidden width and layers must be positive")
        self.length = float(length)
        self.duration = float(duration)
        self.initial_cold_temperature = float(initial_cold_temperature)
        self.initial_hot_temperature = float(initial_hot_temperature)
        self.temperature_scale = float(temperature_scale)
        layers = []
        input_width = 2
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(input_width, hidden_width), nn.Tanh()))
            input_width = hidden_width
        layers.append(nn.Linear(input_width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, position: Tensor, time: Tensor) -> Tensor:
        position = position.reshape(-1, 1)
        time = time.reshape(-1, 1)
        if position.shape != time.shape:
            raise ValueError("position and time tensors must have equal shape")
        normalized_position = 2.0 * position / self.length - 1.0
        normalized_time = 2.0 * time / self.duration - 1.0
        initial_temperature = (
            self.initial_cold_temperature
            + (
                self.initial_hot_temperature
                - self.initial_cold_temperature
            )
            * position
            / self.length
        )
        correction = self.network(
            torch.cat((normalized_position, normalized_time), dim=1)
        )
        return (
            initial_temperature
            + (time / self.duration) * self.temperature_scale * correction
        )


@dataclass(frozen=True)
class DistributedForwardPINNConfig:
    hidden_width: int = 48
    hidden_layers: int = 3
    interior_space_points: int = 12
    time_points: int = 48
    epochs: int = 3_000
    learning_rate: float = 1.0e-3
    temperature_scale: float = 10.0
    residual_rate_scale: float = 1.0
    interior_weight: float = 1.0
    boundary_weight: float = 1.0
    seed: int = 7
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name, value in (
            ("hidden width", self.hidden_width),
            ("hidden layers", self.hidden_layers),
            ("interior space points", self.interior_space_points),
            ("time points", self.time_points),
            ("epochs", self.epochs),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.interior_space_points < 2 or self.time_points < 2:
            raise ValueError("distributed PINN needs at least two space/time points")
        for name, value in (
            ("learning rate", self.learning_rate),
            ("temperature scale", self.temperature_scale),
            ("residual rate scale", self.residual_rate_scale),
            ("interior weight", self.interior_weight),
            ("boundary weight", self.boundary_weight),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.device not in {"cpu", "mps", "auto"}:
            raise ValueError("device must be 'cpu', 'mps', or 'auto'")


class DistributedPINNResiduals(NamedTuple):
    interior: Tensor
    cold_boundary: Tensor
    hot_boundary: Tensor


class DistributedForwardTrainingHistory(NamedTuple):
    total_loss: Tuple[float, ...]
    interior_loss: Tuple[float, ...]
    boundary_loss: Tuple[float, ...]


class DistributedForwardTrainingResult(NamedTuple):
    model: DistributedForwardPINN
    history: DistributedForwardTrainingHistory
    device: str


class DistributedForwardValidation(NamedTuple):
    cold_face_rmse: float
    hot_face_rmse: float
    internal_temperature_rmse: float
    maximum_absolute_temperature_error: float


def _material_overrides(
    material,
    property_values: Optional[dict[str, Tensor]],
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    del material
    overrides = property_values or {}
    return (
        overrides.get("seebeck_coefficient"),
        overrides.get("electrical_resistivity"),
        overrides.get("thermal_conductivity"),
    )


def distributed_physics_residuals(
    model: DistributedForwardPINN,
    interior_position: Tensor,
    interior_time: Tensor,
    boundary_time: Tensor,
    experiment: DistributedLegExperiment,
    *,
    property_values: Optional[dict[str, Tensor]] = None,
) -> DistributedPINNResiduals:
    """Return PDE and dynamic face-node residuals in K/s."""

    current = _constant_current(experiment)
    position = interior_position.detach().clone().requires_grad_(True)
    time = interior_time.detach().clone().requires_grad_(True)
    temperature = model(position, time)
    temperature_time = torch.autograd.grad(
        temperature,
        time,
        grad_outputs=torch.ones_like(temperature),
        create_graph=True,
    )[0]
    temperature_position = torch.autograd.grad(
        temperature,
        position,
        grad_outputs=torch.ones_like(temperature),
        create_graph=True,
    )[0]
    material = experiment.material
    alpha_values, resistivity_values, conductivity_values = _material_overrides(
        material, property_values
    )
    conductivity = _property_tensor(
        material.thermal_conductivity, temperature, conductivity_values
    )
    conductive_divergence = torch.autograd.grad(
        conductivity * temperature_position,
        position,
        grad_outputs=torch.ones_like(temperature),
        create_graph=True,
    )[0]
    resistivity = _property_tensor(
        material.electrical_resistivity, temperature, resistivity_values
    )
    alpha_slope = _property_slope_tensor(
        material.seebeck_coefficient, temperature, alpha_values
    )
    current_density = current / experiment.geometry.area
    volumetric_heat_capacity = material.mass_density * material.specific_heat_capacity
    interior_rhs = (
        conductive_divergence
        + resistivity * current_density**2
        - temperature * alpha_slope * current_density * temperature_position
    ) / volumetric_heat_capacity

    boundary_time_value = boundary_time.detach().clone().requires_grad_(True)
    cold_position = torch.zeros_like(boundary_time_value).requires_grad_(True)
    hot_position = torch.full_like(
        boundary_time_value, experiment.geometry.length
    ).requires_grad_(True)
    cold_temperature = model(cold_position, boundary_time_value)
    hot_temperature = model(hot_position, boundary_time_value)
    cold_temperature_time = torch.autograd.grad(
        cold_temperature,
        boundary_time_value,
        grad_outputs=torch.ones_like(cold_temperature),
        create_graph=True,
        retain_graph=True,
    )[0]
    hot_temperature_time = torch.autograd.grad(
        hot_temperature,
        boundary_time_value,
        grad_outputs=torch.ones_like(hot_temperature),
        create_graph=True,
        retain_graph=True,
    )[0]
    cold_gradient = torch.autograd.grad(
        cold_temperature,
        cold_position,
        grad_outputs=torch.ones_like(cold_temperature),
        create_graph=True,
    )[0]
    hot_gradient = torch.autograd.grad(
        hot_temperature,
        hot_position,
        grad_outputs=torch.ones_like(hot_temperature),
        create_graph=True,
    )[0]
    cold_alpha = _property_tensor(
        material.seebeck_coefficient, cold_temperature, alpha_values
    )
    hot_alpha = _property_tensor(
        material.seebeck_coefficient, hot_temperature, alpha_values
    )
    cold_conductivity = _property_tensor(
        material.thermal_conductivity, cold_temperature, conductivity_values
    )
    hot_conductivity = _property_tensor(
        material.thermal_conductivity, hot_temperature, conductivity_values
    )
    cold_heat = experiment.geometry.area * (
        cold_alpha * cold_temperature * current_density
        - cold_conductivity * cold_gradient
    )
    hot_heat = experiment.geometry.area * (
        hot_alpha * hot_temperature * current_density
        - hot_conductivity * hot_gradient
    )
    faces = experiment.face_parameters
    cold_rhs = (
        faces.cold_reservoir_conductance
        * (experiment.cold_reservoir_temperature - cold_temperature)
        + experiment.cold_external_heat
        - cold_heat
    ) / faces.cold_thermal_capacitance
    hot_rhs = (
        faces.hot_reservoir_conductance
        * (experiment.hot_reservoir_temperature - hot_temperature)
        + experiment.hot_external_heat
        + hot_heat
    ) / faces.hot_thermal_capacitance
    return DistributedPINNResiduals(
        interior=temperature_time - interior_rhs,
        cold_boundary=cold_temperature_time - cold_rhs,
        hot_boundary=hot_temperature_time - hot_rhs,
    )


def _collocation_tensors(
    experiment: DistributedLegExperiment,
    config: DistributedForwardPINNConfig,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    positions = torch.linspace(
        0.0,
        experiment.geometry.length,
        config.interior_space_points + 2,
        device=device,
    )[1:-1]
    times = torch.linspace(0.0, experiment.duration, config.time_points, device=device)
    position_grid, time_grid = torch.meshgrid(positions, times, indexing="ij")
    return (
        position_grid.reshape(-1, 1),
        time_grid.reshape(-1, 1),
        times.reshape(-1, 1),
    )


def train_distributed_forward_pinn(
    experiment: DistributedLegExperiment,
    config: DistributedForwardPINNConfig = DistributedForwardPINNConfig(),
) -> DistributedForwardTrainingResult:
    _constant_current(experiment)
    device = select_device(config.device)
    torch.manual_seed(config.seed)
    model = DistributedForwardPINN(
        length=experiment.geometry.length,
        duration=experiment.duration,
        initial_cold_temperature=experiment.initial_cold_face_temperature,
        initial_hot_temperature=experiment.initial_hot_face_temperature,
        hidden_width=config.hidden_width,
        hidden_layers=config.hidden_layers,
        temperature_scale=config.temperature_scale,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    position, time, boundary_time = _collocation_tensors(experiment, config, device)
    total_history = []
    interior_history = []
    boundary_history = []
    model.train()
    for _ in range(config.epochs):
        optimizer.zero_grad()
        residuals = distributed_physics_residuals(
            model, position, time, boundary_time, experiment
        )
        interior_loss = (
            residuals.interior / config.residual_rate_scale
        ).square().mean()
        boundary_loss = (
            (residuals.cold_boundary / config.residual_rate_scale).square().mean()
            + (residuals.hot_boundary / config.residual_rate_scale).square().mean()
        )
        loss = (
            config.interior_weight * interior_loss
            + config.boundary_weight * boundary_loss
        )
        loss.backward()
        optimizer.step()
        total_history.append(float(loss.detach().cpu()))
        interior_history.append(float(interior_loss.detach().cpu()))
        boundary_history.append(float(boundary_loss.detach().cpu()))
    return DistributedForwardTrainingResult(
        model=model,
        history=DistributedForwardTrainingHistory(
            total_loss=tuple(total_history),
            interior_loss=tuple(interior_history),
            boundary_loss=tuple(boundary_history),
        ),
        device=str(device),
    )


def predict_distributed_temperature(
    model: DistributedForwardPINN,
    *,
    positions: Sequence[float],
    times: Sequence[float],
) -> Tuple[Tuple[float, ...], ...]:
    """Return a time-major rectangular T(t, x) prediction."""

    x_values = tuple(float(value) for value in positions)
    t_values = tuple(float(value) for value in times)
    if not x_values or not t_values:
        raise ValueError("prediction positions and times cannot be empty")
    parameter = next(model.parameters())
    x = torch.tensor(x_values, dtype=parameter.dtype, device=parameter.device)
    t = torch.tensor(t_values, dtype=parameter.dtype, device=parameter.device)
    time_grid, position_grid = torch.meshgrid(t, x, indexing="ij")
    model.eval()
    with torch.no_grad():
        values = model(position_grid.reshape(-1, 1), time_grid.reshape(-1, 1))
    array = values.detach().cpu().reshape(len(t_values), len(x_values))
    return tuple(tuple(float(value) for value in row) for row in array)


def validate_distributed_forward_pinn(
    model: DistributedForwardPINN,
    experiment: DistributedLegExperiment,
) -> DistributedForwardValidation:
    reference = run_distributed_leg_experiment(experiment).trajectory
    positions = (0.0,) + tuple(
        experiment.geometry.length * (index + 0.5) / experiment.cell_count
        for index in range(experiment.cell_count)
    ) + (experiment.geometry.length,)
    prediction = predict_distributed_temperature(
        model, positions=positions, times=reference.time
    )
    cold_errors = tuple(
        row[0] - expected for row, expected in zip(prediction, reference.cold_face)
    )
    hot_errors = tuple(
        row[-1] - expected for row, expected in zip(prediction, reference.hot_face)
    )
    internal_errors = tuple(
        predicted - expected
        for row, expected_row in zip(prediction, reference.cells)
        for predicted, expected in zip(row[1:-1], expected_row)
    )
    all_errors = cold_errors + internal_errors + hot_errors

    def rmse(values: Sequence[float]) -> float:
        return math.sqrt(sum(value * value for value in values) / len(values))

    return DistributedForwardValidation(
        cold_face_rmse=rmse(cold_errors),
        hot_face_rmse=rmse(hot_errors),
        internal_temperature_rmse=rmse(internal_errors),
        maximum_absolute_temperature_error=max(abs(value) for value in all_errors),
    )
