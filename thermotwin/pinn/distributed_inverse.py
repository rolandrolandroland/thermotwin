"""Inverse PINN for one temperature-dependent thermoelectric property curve."""

from dataclasses import dataclass, replace
import math
from typing import NamedTuple, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from ..observations.distributed import DistributedObservationSet
from ..physics.distributed import (
    DistributedThermoelectricMaterial,
    PiecewiseLinearProperty,
)
from ..simulation.distributed import DistributedLegExperiment
from ..inference.distributed_regularization import second_difference_roughness
from .distributed_forward import (
    DistributedForwardPINN,
    DistributedForwardPINNConfig,
    _collocation_tensors,
    _constant_current,
    _property_tensor,
    distributed_physics_residuals,
)
from .forward_two_node import select_device


class InverseDistributedPropertyPINN(nn.Module):
    """Joint hidden-temperature model and trainable property coefficients."""

    def __init__(
        self,
        experiment: DistributedLegExperiment,
        *,
        property_name: str,
        baseline_material: DistributedThermoelectricMaterial,
        initial_log_multipliers: Sequence[float],
        hidden_width: int,
        hidden_layers: int,
        temperature_scale: float,
    ) -> None:
        super().__init__()
        if property_name not in {
            "seebeck_coefficient",
            "electrical_resistivity",
            "thermal_conductivity",
        }:
            raise ValueError("unknown inverse distributed property")
        baseline_property = getattr(baseline_material, property_name)
        if not isinstance(baseline_property, PiecewiseLinearProperty):
            raise TypeError("inverse property must be PiecewiseLinearProperty")
        offsets = tuple(float(value) for value in initial_log_multipliers)
        if len(offsets) != len(baseline_property.values):
            raise ValueError("initial multiplier count must match property coefficients")
        if any(not math.isfinite(value) for value in offsets):
            raise ValueError("initial log multipliers must be finite")
        if property_name == "seebeck_coefficient" and any(
            value == 0.0 for value in baseline_property.values
        ):
            raise ValueError("Seebeck base coefficients cannot be zero")
        self.temperature_model = DistributedForwardPINN(
            length=experiment.geometry.length,
            duration=experiment.duration,
            initial_cold_temperature=experiment.initial_cold_face_temperature,
            initial_hot_temperature=experiment.initial_hot_face_temperature,
            hidden_width=hidden_width,
            hidden_layers=hidden_layers,
            temperature_scale=temperature_scale,
        )
        self.property_name = property_name
        self.baseline_material = baseline_material
        self.register_buffer(
            "baseline_values",
            torch.tensor(baseline_property.values, dtype=torch.float32),
        )
        self.raw_log_multipliers = nn.Parameter(
            torch.tensor(offsets, dtype=torch.float32)
        )

    @property
    def property_values(self) -> Tensor:
        """Return signed alpha values or positive rho/kappa values."""

        return self.baseline_values * torch.exp(self.raw_log_multipliers)

    def property_overrides(self) -> dict[str, Tensor]:
        return {self.property_name: self.property_values}

    def forward(self, position: Tensor, time: Tensor) -> Tensor:
        return self.temperature_model(position, time)


class MultiExperimentInverseDistributedPropertyPINN(nn.Module):
    """One shared property curve with one hidden T(x,t) field per experiment."""

    def __init__(
        self,
        experiments: Sequence[DistributedLegExperiment],
        *,
        property_name: str,
        baseline_material: DistributedThermoelectricMaterial,
        initial_log_multipliers: Sequence[float],
        hidden_width: int,
        hidden_layers: int,
        temperature_scale: float,
    ) -> None:
        super().__init__()
        experiments = tuple(experiments)
        if not experiments:
            raise ValueError("multi-experiment inverse PINN needs experiments")
        baseline_property = getattr(baseline_material, property_name)
        if not isinstance(baseline_property, PiecewiseLinearProperty):
            raise TypeError("inverse property must be PiecewiseLinearProperty")
        offsets = tuple(float(value) for value in initial_log_multipliers)
        if len(offsets) != len(baseline_property.values):
            raise ValueError("initial multiplier count must match property coefficients")
        self.temperature_models = nn.ModuleList(
            DistributedForwardPINN(
                length=experiment.geometry.length,
                duration=experiment.duration,
                initial_cold_temperature=experiment.initial_cold_face_temperature,
                initial_hot_temperature=experiment.initial_hot_face_temperature,
                hidden_width=hidden_width,
                hidden_layers=hidden_layers,
                temperature_scale=temperature_scale,
            )
            for experiment in experiments
        )
        self.property_name = property_name
        self.baseline_material = baseline_material
        self.register_buffer(
            "baseline_values",
            torch.tensor(baseline_property.values, dtype=torch.float32),
        )
        self.raw_log_multipliers = nn.Parameter(
            torch.tensor(offsets, dtype=torch.float32)
        )

    @property
    def property_values(self) -> Tensor:
        return self.baseline_values * torch.exp(self.raw_log_multipliers)

    def property_overrides(self) -> dict[str, Tensor]:
        return {self.property_name: self.property_values}


@dataclass(frozen=True)
class InverseDistributedPropertyConfig:
    property_name: str
    hidden_width: int = 48
    hidden_layers: int = 3
    interior_space_points: int = 12
    time_points: int = 48
    voltage_space_points: int = 24
    epochs: int = 4_000
    network_learning_rate: float = 1.0e-3
    property_learning_rate: float = 3.0e-3
    initial_log_multipliers: Tuple[float, ...] = ()
    temperature_scale: float = 10.0
    residual_rate_scale: float = 1.0
    temperature_standard_deviation: float = 0.01
    voltage_standard_deviation: float = 1.0e-5
    heat_rate_standard_deviation: float = 5.0e-4
    physics_weight: float = 1.0
    observation_weight: float = 1.0
    smoothness_weight: float = 1.0e-3
    seed: int = 7
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.property_name not in {
            "seebeck_coefficient",
            "electrical_resistivity",
            "thermal_conductivity",
        }:
            raise ValueError("unknown inverse distributed property")
        for name, value in (
            ("hidden width", self.hidden_width),
            ("hidden layers", self.hidden_layers),
            ("interior space points", self.interior_space_points),
            ("time points", self.time_points),
            ("voltage space points", self.voltage_space_points),
            ("epochs", self.epochs),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.interior_space_points < 2 or self.time_points < 2:
            raise ValueError("inverse distributed PINN needs at least two collocation points")
        if self.voltage_space_points < 3:
            raise ValueError("voltage quadrature needs at least three space points")
        for name, value in (
            ("network learning rate", self.network_learning_rate),
            ("property learning rate", self.property_learning_rate),
            ("temperature scale", self.temperature_scale),
            ("residual rate scale", self.residual_rate_scale),
            ("temperature standard deviation", self.temperature_standard_deviation),
            ("voltage standard deviation", self.voltage_standard_deviation),
            ("heat-rate standard deviation", self.heat_rate_standard_deviation),
            ("physics weight", self.physics_weight),
            ("observation weight", self.observation_weight),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.smoothness_weight) or self.smoothness_weight < 0.0:
            raise ValueError("smoothness weight must be finite and nonnegative")
        if any(not math.isfinite(value) for value in self.initial_log_multipliers):
            raise ValueError("initial log multipliers must be finite")
        if self.device not in {"cpu", "mps", "auto"}:
            raise ValueError("device must be 'cpu', 'mps', or 'auto'")

    def observation_scale(self, channel: str) -> float:
        if channel.endswith("temperature"):
            return self.temperature_standard_deviation
        if channel == "voltage":
            return self.voltage_standard_deviation
        if channel.endswith("heat"):
            return self.heat_rate_standard_deviation
        raise ValueError(f"unknown distributed observation channel: {channel}")


class InverseDistributedHistory(NamedTuple):
    total_loss: Tuple[float, ...]
    physics_loss: Tuple[float, ...]
    observation_loss: Tuple[float, ...]
    smoothness_loss: Tuple[float, ...]
    property_values: Tuple[Tuple[float, ...], ...]


class InverseDistributedTrainingResult(NamedTuple):
    model: InverseDistributedPropertyPINN
    history: InverseDistributedHistory
    device: str


class MultiExperimentInverseDistributedTrainingResult(NamedTuple):
    model: MultiExperimentInverseDistributedPropertyPINN
    history: InverseDistributedHistory
    device: str


def _temperature_model_observation_predictions(
    temperature_model: DistributedForwardPINN,
    baseline_material: DistributedThermoelectricMaterial,
    property_overrides: dict[str, Tensor],
    experiment: DistributedLegExperiment,
    observations: DistributedObservationSet,
    *,
    voltage_space_points: int,
) -> Tensor:
    """Predict sparse terminal observations in their original order."""

    current = _constant_current(experiment)
    material = baseline_material
    overrides = property_overrides
    alpha_values = overrides.get("seebeck_coefficient")
    resistivity_values = overrides.get("electrical_resistivity")
    conductivity_values = overrides.get("thermal_conductivity")
    parameter = next(temperature_model.parameters())
    unique_times = tuple(dict.fromkeys(item.time for item in observations.observations))
    predictions = {}
    for time_value in unique_times:
        time = torch.tensor(
            [[time_value]], dtype=parameter.dtype, device=parameter.device
        )
        cold_position = torch.zeros_like(time).requires_grad_(True)
        hot_position = torch.full_like(time, experiment.geometry.length).requires_grad_(True)
        cold_temperature = temperature_model(cold_position, time)
        hot_temperature = temperature_model(hot_position, time)
        predictions[(time_value, "cold_face_temperature")] = cold_temperature.reshape(())
        predictions[(time_value, "hot_face_temperature")] = hot_temperature.reshape(())

        requested = {
            item.channel
            for item in observations.observations
            if item.time == time_value
        }
        if "cold_side_heat" in requested:
            cold_gradient = torch.autograd.grad(
                cold_temperature,
                cold_position,
                grad_outputs=torch.ones_like(cold_temperature),
                create_graph=True,
            )[0]
            alpha = _property_tensor(
                material.seebeck_coefficient, cold_temperature, alpha_values
            )
            conductivity = _property_tensor(
                material.thermal_conductivity,
                cold_temperature,
                conductivity_values,
            )
            heat = experiment.geometry.area * (
                alpha * cold_temperature * current / experiment.geometry.area
                - conductivity * cold_gradient
            )
            predictions[(time_value, "cold_side_heat")] = heat.reshape(())
        if "hot_side_heat" in requested:
            hot_gradient = torch.autograd.grad(
                hot_temperature,
                hot_position,
                grad_outputs=torch.ones_like(hot_temperature),
                create_graph=True,
            )[0]
            alpha = _property_tensor(
                material.seebeck_coefficient, hot_temperature, alpha_values
            )
            conductivity = _property_tensor(
                material.thermal_conductivity,
                hot_temperature,
                conductivity_values,
            )
            heat = experiment.geometry.area * (
                alpha * hot_temperature * current / experiment.geometry.area
                - conductivity * hot_gradient
            )
            predictions[(time_value, "hot_side_heat")] = heat.reshape(())
        if "voltage" in requested:
            position = torch.linspace(
                0.0,
                experiment.geometry.length,
                voltage_space_points,
                dtype=parameter.dtype,
                device=parameter.device,
            ).reshape(-1, 1).requires_grad_(True)
            times = torch.full_like(position, time_value)
            temperature = temperature_model(position, times)
            gradient = torch.autograd.grad(
                temperature,
                position,
                grad_outputs=torch.ones_like(temperature),
                create_graph=True,
            )[0]
            resistivity = _property_tensor(
                material.electrical_resistivity,
                temperature,
                resistivity_values,
            )
            alpha = _property_tensor(
                material.seebeck_coefficient, temperature, alpha_values
            )
            electric_field = (
                resistivity * current / experiment.geometry.area
                + alpha * gradient
            )
            voltage = torch.trapezoid(
                electric_field.reshape(-1), position.reshape(-1)
            )
            predictions[(time_value, "voltage")] = voltage
    return torch.stack(
        tuple(predictions[(item.time, item.channel)] for item in observations.observations)
    )


def _observation_predictions(
    model: InverseDistributedPropertyPINN,
    experiment: DistributedLegExperiment,
    observations: DistributedObservationSet,
    *,
    voltage_space_points: int,
) -> Tensor:
    return _temperature_model_observation_predictions(
        model.temperature_model,
        model.baseline_material,
        model.property_overrides(),
        experiment,
        observations,
        voltage_space_points=voltage_space_points,
    )


def train_inverse_distributed_property_pinn(
    experiment: DistributedLegExperiment,
    observations: DistributedObservationSet,
    config: InverseDistributedPropertyConfig,
    *,
    baseline_material: Optional[DistributedThermoelectricMaterial] = None,
) -> InverseDistributedTrainingResult:
    """Jointly infer a hidden temperature field and one property function."""

    _constant_current(experiment)
    if any(
        item.time < 0.0 or item.time > experiment.duration
        for item in observations.observations
    ):
        raise ValueError("inverse observations must lie within the experiment")
    baseline_material = baseline_material or experiment.material
    baseline_property = getattr(baseline_material, config.property_name)
    if not isinstance(baseline_property, PiecewiseLinearProperty):
        raise TypeError("inverse property must be PiecewiseLinearProperty")
    initial_offsets = (
        config.initial_log_multipliers
        if config.initial_log_multipliers
        else (0.0,) * len(baseline_property.values)
    )
    model_experiment = replace(experiment, material=baseline_material)
    device = select_device(config.device)
    torch.manual_seed(config.seed)
    model = InverseDistributedPropertyPINN(
        model_experiment,
        property_name=config.property_name,
        baseline_material=baseline_material,
        initial_log_multipliers=initial_offsets,
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
                "params": (model.raw_log_multipliers,),
                "lr": config.property_learning_rate,
            },
        )
    )
    forward_config = DistributedForwardPINNConfig(
        hidden_width=config.hidden_width,
        hidden_layers=config.hidden_layers,
        interior_space_points=config.interior_space_points,
        time_points=config.time_points,
        epochs=1,
        learning_rate=config.network_learning_rate,
        temperature_scale=config.temperature_scale,
        residual_rate_scale=config.residual_rate_scale,
        device=config.device,
    )
    position, time, boundary_time = _collocation_tensors(
        model_experiment, forward_config, device
    )
    observed = torch.tensor(
        observations.values(), dtype=torch.float32, device=device
    )
    scales = torch.tensor(
        tuple(config.observation_scale(item.channel) for item in observations.observations),
        dtype=torch.float32,
        device=device,
    )
    total_history = []
    physics_history = []
    observation_history = []
    smoothness_history = []
    value_history = []
    model.train()
    for _ in range(config.epochs):
        optimizer.zero_grad()
        residuals = distributed_physics_residuals(
            model.temperature_model,
            position,
            time,
            boundary_time,
            model_experiment,
            property_values=model.property_overrides(),
        )
        physics_loss = (
            (residuals.interior / config.residual_rate_scale).square().mean()
            + (residuals.cold_boundary / config.residual_rate_scale).square().mean()
            + (residuals.hot_boundary / config.residual_rate_scale).square().mean()
        )
        predicted = _observation_predictions(
            model,
            model_experiment,
            observations,
            voltage_space_points=config.voltage_space_points,
        )
        observation_loss = ((predicted - observed) / scales).square().mean()
        smoothness_loss = second_difference_roughness(
            model.raw_log_multipliers
        )
        total_loss = (
            config.physics_weight * physics_loss
            + config.observation_weight * observation_loss
            + config.smoothness_weight * smoothness_loss
        )
        total_loss.backward()
        optimizer.step()
        total_history.append(float(total_loss.detach().cpu()))
        physics_history.append(float(physics_loss.detach().cpu()))
        observation_history.append(float(observation_loss.detach().cpu()))
        smoothness_history.append(float(smoothness_loss.detach().cpu()))
        value_history.append(
            tuple(float(value) for value in model.property_values.detach().cpu())
        )
    return InverseDistributedTrainingResult(
        model=model,
        history=InverseDistributedHistory(
            total_loss=tuple(total_history),
            physics_loss=tuple(physics_history),
            observation_loss=tuple(observation_history),
            smoothness_loss=tuple(smoothness_history),
            property_values=tuple(value_history),
        ),
        device=str(device),
    )


def train_multi_experiment_inverse_distributed_property_pinn(
    experiments: Sequence[DistributedLegExperiment],
    observations: Sequence[DistributedObservationSet],
    config: InverseDistributedPropertyConfig,
    *,
    baseline_material: Optional[DistributedThermoelectricMaterial] = None,
) -> MultiExperimentInverseDistributedTrainingResult:
    """Infer one property curve shared by several independently solved regimes."""

    experiments = tuple(experiments)
    observations = tuple(observations)
    if not experiments or len(experiments) != len(observations):
        raise ValueError("experiments and observations must have equal nonzero length")
    for experiment, dataset in zip(experiments, observations):
        _constant_current(experiment)
        if any(
            item.time < 0.0 or item.time > experiment.duration
            for item in dataset.observations
        ):
            raise ValueError("inverse observations must lie within each experiment")
    baseline_material = baseline_material or experiments[0].material
    baseline_property = getattr(baseline_material, config.property_name)
    if not isinstance(baseline_property, PiecewiseLinearProperty):
        raise TypeError("inverse property must be PiecewiseLinearProperty")
    initial_offsets = (
        config.initial_log_multipliers
        if config.initial_log_multipliers
        else (0.0,) * len(baseline_property.values)
    )
    model_experiments = tuple(
        replace(experiment, material=baseline_material)
        for experiment in experiments
    )
    device = select_device(config.device)
    torch.manual_seed(config.seed)
    model = MultiExperimentInverseDistributedPropertyPINN(
        model_experiments,
        property_name=config.property_name,
        baseline_material=baseline_material,
        initial_log_multipliers=initial_offsets,
        hidden_width=config.hidden_width,
        hidden_layers=config.hidden_layers,
        temperature_scale=config.temperature_scale,
    ).to(device)
    optimizer = torch.optim.Adam(
        (
            {
                "params": model.temperature_models.parameters(),
                "lr": config.network_learning_rate,
            },
            {
                "params": (model.raw_log_multipliers,),
                "lr": config.property_learning_rate,
            },
        )
    )
    forward_config = DistributedForwardPINNConfig(
        hidden_width=config.hidden_width,
        hidden_layers=config.hidden_layers,
        interior_space_points=config.interior_space_points,
        time_points=config.time_points,
        epochs=1,
        learning_rate=config.network_learning_rate,
        temperature_scale=config.temperature_scale,
        residual_rate_scale=config.residual_rate_scale,
        device=config.device,
    )
    collocation = tuple(
        _collocation_tensors(experiment, forward_config, device)
        for experiment in model_experiments
    )
    observed_tensors = tuple(
        torch.tensor(dataset.values(), dtype=torch.float32, device=device)
        for dataset in observations
    )
    scale_tensors = tuple(
        torch.tensor(
            tuple(config.observation_scale(item.channel) for item in dataset.observations),
            dtype=torch.float32,
            device=device,
        )
        for dataset in observations
    )
    total_history = []
    physics_history = []
    observation_history = []
    smoothness_history = []
    value_history = []
    model.train()
    for _ in range(config.epochs):
        optimizer.zero_grad()
        physics_terms = []
        observation_terms = []
        for temperature_model, experiment, dataset, tensors, observed, scales in zip(
            model.temperature_models,
            model_experiments,
            observations,
            collocation,
            observed_tensors,
            scale_tensors,
        ):
            position, time, boundary_time = tensors
            residuals = distributed_physics_residuals(
                temperature_model,
                position,
                time,
                boundary_time,
                experiment,
                property_values=model.property_overrides(),
            )
            physics_terms.append(
                (residuals.interior / config.residual_rate_scale).square().mean()
                + (residuals.cold_boundary / config.residual_rate_scale).square().mean()
                + (residuals.hot_boundary / config.residual_rate_scale).square().mean()
            )
            predicted = _temperature_model_observation_predictions(
                temperature_model,
                model.baseline_material,
                model.property_overrides(),
                experiment,
                dataset,
                voltage_space_points=config.voltage_space_points,
            )
            observation_terms.append(((predicted - observed) / scales).square().mean())
        physics_loss = torch.stack(physics_terms).mean()
        observation_loss = torch.stack(observation_terms).mean()
        smoothness_loss = second_difference_roughness(
            model.raw_log_multipliers
        )
        total_loss = (
            config.physics_weight * physics_loss
            + config.observation_weight * observation_loss
            + config.smoothness_weight * smoothness_loss
        )
        total_loss.backward()
        optimizer.step()
        total_history.append(float(total_loss.detach().cpu()))
        physics_history.append(float(physics_loss.detach().cpu()))
        observation_history.append(float(observation_loss.detach().cpu()))
        smoothness_history.append(float(smoothness_loss.detach().cpu()))
        value_history.append(
            tuple(float(value) for value in model.property_values.detach().cpu())
        )
    return MultiExperimentInverseDistributedTrainingResult(
        model=model,
        history=InverseDistributedHistory(
            total_loss=tuple(total_history),
            physics_loss=tuple(physics_history),
            observation_loss=tuple(observation_history),
            smoothness_loss=tuple(smoothness_history),
            property_values=tuple(value_history),
        ),
        device=str(device),
    )
