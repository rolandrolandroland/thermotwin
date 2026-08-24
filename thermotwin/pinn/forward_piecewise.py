"""Piecewise four-node forward PINN for switched current experiments."""

import argparse
from dataclasses import dataclass
import math
from typing import NamedTuple, Sequence, Tuple

import torch
from torch import Tensor, nn

from ..simulation.four_node_experiments import FourNodeContactExperiment
from .forward_four_node import (
    ContactPINNValidation,
    contact_physics_loss,
    contact_physics_residuals,
    validate_contact_pinn_against_rk4,
)
from ..inference.contact_resistance import (
    REFERENCE_COLD_CONTACT_RESISTANCE,
    contact_resistance_experiment,
    reference_contact_resistance_regimes,
)
from ..core.controls import CurrentInput, PiecewiseConstantCurrent, current_at
from .forward_two_node import select_device


@dataclass(frozen=True)
class PiecewiseContactForwardPINNConfig:
    """Network and CPU-first training settings for switched current."""

    hidden_width: int = 32
    hidden_layers: int = 2
    collocation_points: int = 192
    epochs: int = 5_000
    learning_rate: float = 1e-3
    temperature_scale: float = 10.0
    seed: int = 17
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
        for name, value in (
            ("learning rate", self.learning_rate),
            ("temperature scale", self.temperature_scale),
        ):
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


def current_segment_boundaries(
    current: CurrentInput,
    duration: float,
) -> Tuple[float, ...]:
    """Return positive-duration current interval boundaries."""

    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("experiment duration must be finite and positive")
    if not isinstance(current, PiecewiseConstantCurrent):
        current_at(current, 0.0)
        return (0.0, float(duration))
    interior = tuple(
        float(time)
        for time in current.transition_times
        if 0.0 < time < duration
    )
    return (0.0,) + interior + (float(duration),)


def scheduled_current_tensor(
    current: CurrentInput,
    time: Tensor,
) -> Tensor:
    """Evaluate scalar or right-continuous scheduled current on a tensor."""

    if time.ndim == 1:
        time = time.reshape(-1, 1)
    if time.ndim != 2 or time.shape[1] != 1:
        raise ValueError("time must have shape (samples,) or (samples, 1)")
    if not isinstance(current, PiecewiseConstantCurrent):
        return torch.full_like(time, current_at(current, 0.0))

    values = torch.full_like(time, current.values[0])
    for transition, value in zip(
        current.transition_times,
        current.values[1:],
    ):
        values = torch.where(
            time >= transition,
            torch.as_tensor(value, dtype=time.dtype, device=time.device),
            values,
        )
    return values


def piecewise_collocation_times(
    boundaries: Sequence[float],
    point_count: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Allocate midpoint collocation coordinates by interval duration."""

    boundaries = tuple(float(value) for value in boundaries)
    if len(boundaries) < 2 or any(
        not math.isfinite(value) for value in boundaries
    ):
        raise ValueError("finite segment boundaries are required")
    durations = tuple(
        right - left for left, right in zip(boundaries, boundaries[1:])
    )
    if any(value <= 0.0 for value in durations):
        raise ValueError("segment boundaries must strictly increase")
    segment_count = len(durations)
    if (
        not isinstance(point_count, int)
        or isinstance(point_count, bool)
        or point_count < 2 * segment_count
    ):
        raise ValueError(
            "collocation count must provide at least two points per segment"
        )

    remaining = point_count - 2 * segment_count
    total_duration = sum(durations)
    quotas = tuple(
        remaining * duration / total_duration for duration in durations
    )
    additions = [math.floor(value) for value in quotas]
    leftover = remaining - sum(additions)
    fractional_order = sorted(
        range(segment_count),
        key=lambda index: quotas[index] - additions[index],
        reverse=True,
    )
    for index in fractional_order[:leftover]:
        additions[index] += 1
    allocations = tuple(2 + value for value in additions)

    parts = []
    for left, right, count in zip(
        boundaries,
        boundaries[1:],
        allocations,
    ):
        fractions = (
            torch.arange(count, dtype=dtype, device=device) + 0.5
        ) / count
        parts.append(left + fractions * (right - left))
    return torch.cat(parts).reshape(-1, 1)


def _temperature_network(
    hidden_width: int,
    hidden_layers: int,
) -> nn.Sequential:
    layers = [nn.Linear(1, hidden_width), nn.Tanh()]
    for _ in range(hidden_layers - 1):
        layers.extend((nn.Linear(hidden_width, hidden_width), nn.Tanh()))
    layers.append(nn.Linear(hidden_width, 4))
    return nn.Sequential(*layers)


class PiecewiseContactForwardPINN(nn.Module):
    """Use one smooth subnetwork per constant-current time interval."""

    def __init__(
        self,
        *,
        duration: float,
        transition_times: Sequence[float],
        initial_temperatures: Sequence[float],
        hidden_width: int = 32,
        hidden_layers: int = 2,
        temperature_scale: float = 10.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("PINN duration must be finite and positive")
        try:
            transitions = tuple(float(value) for value in transition_times)
        except (TypeError, ValueError) as error:
            raise ValueError("transition times must be finite") from error
        if any(
            not math.isfinite(value)
            or value <= 0.0
            or value >= duration
            for value in transitions
        ):
            raise ValueError(
                "transition times must lie strictly inside the duration"
            )
        if any(
            later <= earlier
            for earlier, later in zip(transitions, transitions[1:])
        ):
            raise ValueError("transition times must strictly increase")
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

        self._boundaries = (0.0,) + transitions + (float(duration),)
        self.networks = nn.ModuleList(
            _temperature_network(hidden_width, hidden_layers)
            for _ in range(len(self._boundaries) - 1)
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

    @property
    def segment_boundaries(self) -> Tuple[float, ...]:
        """Return time boundaries including zero and final duration."""

        return self._boundaries

    def _temperature_with_start(
        self,
        segment_index: int,
        time: Tensor,
        start_temperature: Tensor,
    ) -> Tensor:
        left = self._boundaries[segment_index]
        right = self._boundaries[segment_index + 1]
        progress = (time - left) / (right - left)
        normalized_time = 2.0 * progress - 1.0
        return (
            start_temperature
            + progress
            * self._temperature_scale
            * self.networks[segment_index](normalized_time)
        )

    def _segment_start_temperatures(self) -> Tuple[Tensor, ...]:
        starts = []
        state = self._initial_temperatures
        for index in range(len(self.networks)):
            starts.append(state)
            endpoint = self._initial_temperatures.new_tensor(
                [[self._boundaries[index + 1]]]
            )
            state = self._temperature_with_start(index, endpoint, state)
        return tuple(starts)

    def predict_in_segment(
        self,
        segment_index: int,
        time: Tensor,
    ) -> Tensor:
        """Evaluate one subnetwork, including either interval endpoint."""

        if (
            not isinstance(segment_index, int)
            or isinstance(segment_index, bool)
            or segment_index < 0
            or segment_index >= len(self.networks)
        ):
            raise ValueError("segment index is outside the model")
        if time.ndim == 1:
            time = time.reshape(-1, 1)
        if time.ndim != 2 or time.shape[1] != 1 or time.shape[0] == 0:
            raise ValueError(
                "time must have nonempty shape (samples,) or (samples, 1)"
            )
        left = self._boundaries[segment_index]
        right = self._boundaries[segment_index + 1]
        if bool(torch.any((time < left) | (time > right)).detach().cpu()):
            raise ValueError("time lies outside the selected segment")
        starts = self._segment_start_temperatures()
        return self._temperature_with_start(
            segment_index,
            time,
            starts[segment_index],
        )

    def boundary_temperature_jumps(self) -> Tensor:
        """Return right-minus-left temperature jumps at all switches."""

        jumps = []
        for index, boundary in enumerate(self._boundaries[1:-1]):
            time = self._initial_temperatures.new_tensor([[boundary]])
            left = self.predict_in_segment(index, time)
            right = self.predict_in_segment(index + 1, time)
            jumps.append(right - left)
        if not jumps:
            return self._initial_temperatures.new_empty((0, 4))
        return torch.cat(jumps, dim=0)

    def forward(self, time: Tensor) -> Tensor:
        """Return right-continuous piecewise temperature predictions."""

        if time.ndim == 1:
            time = time.reshape(-1, 1)
        if time.ndim != 2 or time.shape[1] != 1 or time.shape[0] == 0:
            raise ValueError(
                "time must have nonempty shape (samples,) or (samples, 1)"
            )
        if bool(
            torch.any(
                (time < self._boundaries[0])
                | (time > self._boundaries[-1])
            )
            .detach()
            .cpu()
        ):
            raise ValueError("time must lie inside the modeled duration")

        starts = self._segment_start_temperatures()
        indexed_parts = []
        for index, (left, right) in enumerate(
            zip(self._boundaries, self._boundaries[1:])
        ):
            if index == len(self.networks) - 1:
                mask = (time[:, 0] >= left) & (time[:, 0] <= right)
            else:
                mask = (time[:, 0] >= left) & (time[:, 0] < right)
            indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
            if indices.numel() == 0:
                continue
            segment_time = time.index_select(0, indices)
            temperatures = self._temperature_with_start(
                index,
                segment_time,
                starts[index],
            )
            indexed_parts.append((indices, temperatures))

        indices = torch.cat(tuple(item[0] for item in indexed_parts))
        temperatures = torch.cat(tuple(item[1] for item in indexed_parts))
        order = torch.argsort(indices)
        return temperatures.index_select(0, order)


class PiecewiseContactPINNTrainingResult(NamedTuple):
    """Trained piecewise model, physics loss history, and selected device."""

    model: PiecewiseContactForwardPINN
    loss_history: Tuple[float, ...]
    device: str
    collocation_time: Tuple[float, ...]


def unipolar_pulse_contact_experiment() -> FourNodeContactExperiment:
    """Return the established 5--20 s, 1 A training-pulse experiment."""

    training_regime = reference_contact_resistance_regimes()[0]
    return contact_resistance_experiment(
        training_regime,
        cold_contact_resistance=REFERENCE_COLD_CONTACT_RESISTANCE,
    )


def train_piecewise_contact_forward_pinn(
    experiment: FourNodeContactExperiment,
    config: PiecewiseContactForwardPINNConfig = (
        PiecewiseContactForwardPINNConfig()
    ),
) -> PiecewiseContactPINNTrainingResult:
    """Train smooth subnetworks joined by exact temperature continuity."""

    boundaries = current_segment_boundaries(
        experiment.current,
        experiment.duration,
    )
    if config.collocation_points < 2 * (len(boundaries) - 1):
        raise ValueError(
            "collocation count must provide at least two points per segment"
        )
    device = select_device(config.device)
    torch.manual_seed(config.seed)
    model = PiecewiseContactForwardPINN(
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
    collocation_time = piecewise_collocation_times(
        boundaries,
        config.collocation_points,
        device=device,
    )
    current_values = scheduled_current_tensor(
        experiment.current,
        collocation_time,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    losses = []
    model.train()
    for _ in range(config.epochs):
        optimizer.zero_grad()
        residuals = contact_physics_residuals(
            model,
            collocation_time,
            experiment,
            current_values=current_values,
        )
        loss = contact_physics_loss(residuals)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return PiecewiseContactPINNTrainingResult(
        model=model,
        loss_history=tuple(losses),
        device=str(device),
        collocation_time=tuple(
            float(value) for value in collocation_time.detach().cpu().reshape(-1)
        ),
    )


def validate_piecewise_contact_pinn(
    training: PiecewiseContactPINNTrainingResult,
    experiment: FourNodeContactExperiment,
) -> ContactPINNValidation:
    """Compare the switched-current PINN with transition-splitting RK4."""

    return validate_contact_pinn_against_rk4(training.model, experiment)


def main() -> None:
    """Train and validate the frozen unipolar-pulse forward problem."""

    parser = argparse.ArgumentParser(
        description="Train the ThermoTwin piecewise contact forward PINN"
    )
    parser.add_argument("--epochs", type=int, default=5_000)
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "auto"),
        default="cpu",
    )
    arguments = parser.parse_args()

    experiment = unipolar_pulse_contact_experiment()
    training = train_piecewise_contact_forward_pinn(
        experiment,
        PiecewiseContactForwardPINNConfig(
            epochs=arguments.epochs,
            device=arguments.device,
        ),
    )
    validation = validate_piecewise_contact_pinn(training, experiment)
    max_jump = float(
        training.model.boundary_temperature_jumps().abs().max().detach().cpu()
    )
    print(f"device: {training.device}")
    print(f"segments: {len(training.model.segment_boundaries) - 1}")
    print(f"initial physics loss: {training.loss_history[0]:.6e}")
    print(f"final physics loss: {training.loss_history[-1]:.6e}")
    print(f"maximum boundary temperature jump: {max_jump:.6e} K")
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
