"""Independent whole-system energy diagnostics for a contact PINN.

The four node residuals remain the training constraints.  This module does not
add another loss term.  It independently sums stored-energy rates and compares
them with electrical, reservoir, and declared external heat inputs.  Segment
endpoints are evaluated on both sides of a current switch so discontinuous
electrical power is never trapezoid-interpolated across the switch.
"""

from dataclasses import dataclass
import math
from typing import NamedTuple, Tuple

import torch

from ..core.controls import current_at
from ..simulation.four_node_experiments import FourNodeContactExperiment
from .forward_piecewise import (
    PiecewiseContactForwardPINN,
    current_segment_boundaries,
)


@dataclass(frozen=True)
class ContactPINNEnergyClosureConfig:
    sampling_interval: float = 0.2

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.sampling_interval)
            or self.sampling_interval <= 0.0
        ):
            raise ValueError("energy sampling interval must be finite and positive")


class ContactPINNEnergyClosure(NamedTuple):
    """One-sided rate histories and switch-safe cumulative energy closure."""

    time: Tuple[float, ...]
    segment_index: Tuple[int, ...]
    current: Tuple[float, ...]
    storage_rate: Tuple[float, ...]
    electrical_power: Tuple[float, ...]
    reservoir_heat_input: Tuple[float, ...]
    external_heat_input: Tuple[float, ...]
    net_input_rate: Tuple[float, ...]
    rate_closure_error: Tuple[float, ...]
    stored_energy_change: Tuple[float, ...]
    cumulative_net_input: Tuple[float, ...]
    cumulative_closure_error: Tuple[float, ...]
    rate_closure_rms: float
    maximum_absolute_rate_closure_error: float
    final_cumulative_closure_error: float
    maximum_absolute_cumulative_closure_error: float
    net_input_rate_rms: float
    normalized_rate_closure_rms: float


def _segment_times(left: float, right: float, interval: float, model) -> torch.Tensor:
    count = max(1, math.ceil((right - left) / interval))
    parameter = next(model.parameters())
    return torch.linspace(
        left,
        right,
        count + 1,
        dtype=parameter.dtype,
        device=parameter.device,
    ).reshape(-1, 1)


def _rates(temperatures: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    columns = []
    for index in range(4):
        columns.append(
            torch.autograd.grad(
                temperatures[:, index : index + 1],
                time,
                grad_outputs=torch.ones_like(temperatures[:, index : index + 1]),
                create_graph=False,
                retain_graph=index < 3,
            )[0]
        )
    return torch.cat(columns, dim=1)


def evaluate_piecewise_contact_energy_closure(
    model: PiecewiseContactForwardPINN,
    experiment: FourNodeContactExperiment,
    config: ContactPINNEnergyClosureConfig = ContactPINNEnergyClosureConfig(),
) -> ContactPINNEnergyClosure:
    """Evaluate a whole-system first-law balance outside the training loss."""

    if not isinstance(model, PiecewiseContactForwardPINN):
        raise ValueError("energy closure requires a piecewise contact PINN")
    boundaries = model.segment_boundaries
    expected_boundaries = current_segment_boundaries(
        experiment.current,
        experiment.duration,
    )
    if len(boundaries) != len(expected_boundaries) or any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-9)
        for actual, expected in zip(boundaries, expected_boundaries)
    ):
        raise ValueError(
            "PINN segment boundaries must match the experiment current switches"
        )

    thermal = experiment.thermal_parameters
    thermoelectric = experiment.thermoelectric_parameters
    capacitances = (
        thermal.cold_face_thermal_capacitance,
        thermal.hot_face_thermal_capacitance,
        thermal.cold_exchanger_thermal_capacitance,
        thermal.hot_exchanger_thermal_capacitance,
    )
    initial = (
        experiment.initial_cold_face_temperature,
        experiment.initial_hot_face_temperature,
        experiment.initial_cold_exchanger_temperature,
        experiment.initial_hot_exchanger_temperature,
    )
    external = experiment.cold_external_heat + experiment.hot_external_heat

    time_history = []
    segment_history = []
    current_history = []
    storage_history = []
    electrical_history = []
    reservoir_history = []
    external_history = []
    net_history = []
    rate_error_history = []
    stored_change_history = []
    cumulative_history = []
    cumulative_error_history = []
    cumulative = 0.0

    model.eval()
    for segment_index, (left, right) in enumerate(
        zip(boundaries, boundaries[1:])
    ):
        segment_time = _segment_times(
            left, right, config.sampling_interval, model
        ).requires_grad_(True)
        temperatures = model.predict_in_segment(segment_index, segment_time)
        rates = _rates(temperatures, segment_time)
        temperature_values = temperatures.detach().cpu().tolist()
        rate_values = rates.detach().cpu().tolist()
        time_values = tuple(
            float(value) for value in segment_time.detach().cpu().reshape(-1)
        )
        segment_current = current_at(experiment.current, 0.5 * (left + right))

        segment_net = []
        segment_stored = []
        segment_rate_error = []
        for values, derivatives in zip(temperature_values, rate_values):
            cold_face, hot_face, cold_exchanger, hot_exchanger = values
            storage_rate = sum(
                capacitance * derivative
                for capacitance, derivative in zip(capacitances, derivatives)
            )
            electrical_power = (
                thermoelectric.seebeck_coefficient
                * segment_current
                * (hot_face - cold_face)
                + segment_current**2 * thermoelectric.electrical_resistance
            )
            reservoir_heat = (
                thermal.cold_reservoir_conductance
                * (experiment.cold_reservoir_temperature - cold_exchanger)
                + thermal.hot_reservoir_conductance
                * (experiment.hot_reservoir_temperature - hot_exchanger)
            )
            net_input = electrical_power + reservoir_heat + external
            stored_change = sum(
                capacitance * (temperature - initial_temperature)
                for capacitance, temperature, initial_temperature in zip(
                    capacitances, values, initial
                )
            )
            segment_net.append(net_input)
            segment_stored.append(stored_change)
            segment_rate_error.append(storage_rate - net_input)
            storage_history.append(storage_rate)
            electrical_history.append(electrical_power)
            reservoir_history.append(reservoir_heat)
            external_history.append(external)
            net_history.append(net_input)
            rate_error_history.append(storage_rate - net_input)
            stored_change_history.append(stored_change)

        segment_cumulative = [cumulative]
        for index in range(1, len(time_values)):
            step = time_values[index] - time_values[index - 1]
            cumulative += 0.5 * step * (
                segment_net[index - 1] + segment_net[index]
            )
            segment_cumulative.append(cumulative)

        time_history.extend(time_values)
        segment_history.extend((segment_index,) * len(time_values))
        current_history.extend((segment_current,) * len(time_values))
        cumulative_history.extend(segment_cumulative)
        cumulative_error_history.extend(
            stored - supplied
            for stored, supplied in zip(segment_stored, segment_cumulative)
        )

    rate_rms = math.sqrt(
        sum(value * value for value in rate_error_history)
        / len(rate_error_history)
    )
    input_rms = math.sqrt(
        sum(value * value for value in net_history) / len(net_history)
    )
    return ContactPINNEnergyClosure(
        time=tuple(time_history),
        segment_index=tuple(segment_history),
        current=tuple(current_history),
        storage_rate=tuple(storage_history),
        electrical_power=tuple(electrical_history),
        reservoir_heat_input=tuple(reservoir_history),
        external_heat_input=tuple(external_history),
        net_input_rate=tuple(net_history),
        rate_closure_error=tuple(rate_error_history),
        stored_energy_change=tuple(stored_change_history),
        cumulative_net_input=tuple(cumulative_history),
        cumulative_closure_error=tuple(cumulative_error_history),
        rate_closure_rms=rate_rms,
        maximum_absolute_rate_closure_error=max(
            abs(value) for value in rate_error_history
        ),
        final_cumulative_closure_error=cumulative_error_history[-1],
        maximum_absolute_cumulative_closure_error=max(
            abs(value) for value in cumulative_error_history
        ),
        net_input_rate_rms=input_rms,
        normalized_rate_closure_rms=(
            rate_rms / input_rms
            if input_rms > 0.0
            else (0.0 if rate_rms == 0.0 else math.inf)
        ),
    )
