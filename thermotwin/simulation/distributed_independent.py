"""Independent numerical truth for the distributed thermoelectric leg.

This module is intentionally separate from :mod:`simulation.distributed`.
The established inference reference is cell-centred finite volume advanced by
classical RK4.  The truth model below uses nodal temperatures, independently
assembled fluxes and voltage quadrature, and the third-order SSPRK method.
It exists to measure discretization and constitutive-representation mismatch;
it is not a second claim about hardware fidelity.
"""

from bisect import bisect_left
from dataclasses import dataclass
import math
from typing import NamedTuple, Sequence, Tuple

from ..core.controls import PiecewiseConstantCurrent, current_at
from ..observations.distributed import (
    DistributedObservation,
    DistributedObservationChannels,
    DistributedObservationSet,
    regular_distributed_observation_times,
)
from ..physics.distributed import TemperatureProperty
from .distributed import DistributedLegExperiment


@dataclass(frozen=True)
class PolynomialTemperatureProperty:
    """Smooth polynomial in normalized temperature with an analytic integral."""

    reference_temperature: float
    temperature_scale: float
    coefficients: Tuple[float, ...]

    def __post_init__(self) -> None:
        coefficients = tuple(float(value) for value in self.coefficients)
        object.__setattr__(self, "coefficients", coefficients)
        if (
            not math.isfinite(self.reference_temperature)
            or self.reference_temperature <= 0.0
        ):
            raise ValueError("reference temperature must be finite positive kelvin")
        if not math.isfinite(self.temperature_scale) or self.temperature_scale <= 0.0:
            raise ValueError("temperature scale must be finite and positive")
        if not coefficients or any(not math.isfinite(value) for value in coefficients):
            raise ValueError("polynomial coefficients must be finite and nonempty")

    def _normalized(self, temperature: float) -> float:
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("property temperature must be positive kelvin")
        return (temperature - self.reference_temperature) / self.temperature_scale

    def value(self, temperature: float) -> float:
        coordinate = self._normalized(temperature)
        value = 0.0
        for coefficient in reversed(self.coefficients):
            value = coefficient + coordinate * value
        if not math.isfinite(value):
            raise ValueError("polynomial property became nonfinite")
        return value

    def integral(self, start_temperature: float, end_temperature: float) -> float:
        start = self._normalized(start_temperature)
        end = self._normalized(end_temperature)
        value = self.temperature_scale * sum(
            coefficient
            * (end ** (power + 1) - start ** (power + 1))
            / (power + 1)
            for power, coefficient in enumerate(self.coefficients)
        )
        if not math.isfinite(value):
            raise ValueError("polynomial property integral became nonfinite")
        return value


class IndependentDistributedTrajectory(NamedTuple):
    """Time-major nodal temperatures including both physical faces."""

    time: Tuple[float, ...]
    positions: Tuple[float, ...]
    temperature: Tuple[Tuple[float, ...], ...]


class IndependentDistributedDiagnostics(NamedTuple):
    current: Tuple[float, ...]
    voltage: Tuple[float, ...]
    cold_side_heat: Tuple[float, ...]
    hot_side_heat: Tuple[float, ...]


class IndependentDistributedResult(NamedTuple):
    trajectory: IndependentDistributedTrajectory
    diagnostics: IndependentDistributedDiagnostics


def _validate_state(state: Sequence[float]) -> Tuple[float, ...]:
    values = tuple(float(value) for value in state)
    if len(values) < 5:
        raise ValueError("independent truth requires at least five nodes")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("independent-truth temperature became nonfinite")
    if any(value <= 0.0 for value in values):
        raise ValueError("independent-truth temperature left positive kelvin")
    return values


def _edge_heat_fluxes(
    experiment: DistributedLegExperiment,
    state: Tuple[float, ...],
    current: float,
) -> Tuple[float, ...]:
    spacing = experiment.geometry.length / (len(state) - 1)
    current_density = current / experiment.geometry.area
    material = experiment.material
    fluxes = []
    for left, right in zip(state, state[1:]):
        midpoint = 0.5 * (left + right)
        conductivity = material.thermal_conductivity.value(midpoint)
        resistivity = material.electrical_resistivity.value(midpoint)
        if conductivity <= 0.0 or resistivity <= 0.0:
            raise ValueError("truth transport properties must remain positive")
        fluxes.append(
            material.seebeck_coefficient.value(midpoint)
            * midpoint
            * current_density
            - conductivity * (right - left) / spacing
        )
    return tuple(fluxes)


def _boundary_heat_fluxes(
    experiment: DistributedLegExperiment,
    state: Tuple[float, ...],
    current: float,
) -> Tuple[float, float]:
    spacing = experiment.geometry.length / (len(state) - 1)
    current_density = current / experiment.geometry.area
    cold_gradient = (-3.0 * state[0] + 4.0 * state[1] - state[2]) / (
        2.0 * spacing
    )
    hot_gradient = (3.0 * state[-1] - 4.0 * state[-2] + state[-3]) / (
        2.0 * spacing
    )
    material = experiment.material

    def flux(temperature: float, gradient: float) -> float:
        conductivity = material.thermal_conductivity.value(temperature)
        resistivity = material.electrical_resistivity.value(temperature)
        if conductivity <= 0.0 or resistivity <= 0.0:
            raise ValueError("truth transport properties must remain positive")
        return (
            material.seebeck_coefficient.value(temperature)
            * temperature
            * current_density
            - conductivity * gradient
        )

    return flux(state[0], cold_gradient), flux(state[-1], hot_gradient)


def independent_distributed_rhs(
    experiment: DistributedLegExperiment,
    state: Sequence[float],
    *,
    current: float,
) -> Tuple[float, ...]:
    """Return nodal rates from an independently assembled conservative PDE."""

    state = _validate_state(state)
    if not math.isfinite(current):
        raise ValueError("truth current must be finite")
    spacing = experiment.geometry.length / (len(state) - 1)
    current_density = current / experiment.geometry.area
    material = experiment.material
    edge_fluxes = _edge_heat_fluxes(experiment, state, current)
    cold_flux, hot_flux = _boundary_heat_fluxes(experiment, state, current)
    volumetric_heat_capacity = material.mass_density * material.specific_heat_capacity

    rates = [
        (
            experiment.face_parameters.cold_reservoir_conductance
            * (experiment.cold_reservoir_temperature - state[0])
            + experiment.cold_external_heat
            - experiment.geometry.area * cold_flux
        )
        / experiment.face_parameters.cold_thermal_capacitance
    ]
    for index in range(1, len(state) - 1):
        gradient = (state[index + 1] - state[index - 1]) / (2.0 * spacing)
        electric_field = (
            material.electrical_resistivity.value(state[index]) * current_density
            + material.seebeck_coefficient.value(state[index]) * gradient
        )
        rates.append(
            (
                (edge_fluxes[index - 1] - edge_fluxes[index]) / spacing
                + current_density * electric_field
            )
            / volumetric_heat_capacity
        )
    rates.append(
        (
            experiment.face_parameters.hot_reservoir_conductance
            * (experiment.hot_reservoir_temperature - state[-1])
            + experiment.hot_external_heat
            + experiment.geometry.area * hot_flux
        )
        / experiment.face_parameters.hot_thermal_capacitance
    )
    return _validate_state_rates(rates)


def _validate_state_rates(rates: Sequence[float]) -> Tuple[float, ...]:
    values = tuple(float(value) for value in rates)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("independent-truth temperature rate became nonfinite")
    return values


def independent_distributed_voltage(
    experiment: DistributedLegExperiment,
    state: Sequence[float],
    *,
    current: float,
) -> float:
    """Integrate the nodal electric field with trapezoidal quadrature."""

    state = _validate_state(state)
    spacing = experiment.geometry.length / (len(state) - 1)
    current_density = current / experiment.geometry.area
    gradients = [(-3.0 * state[0] + 4.0 * state[1] - state[2]) / (2.0 * spacing)]
    gradients.extend(
        (state[index + 1] - state[index - 1]) / (2.0 * spacing)
        for index in range(1, len(state) - 1)
    )
    gradients.append(
        (3.0 * state[-1] - 4.0 * state[-2] + state[-3]) / (2.0 * spacing)
    )
    fields = tuple(
        experiment.material.electrical_resistivity.value(temperature)
        * current_density
        + experiment.material.seebeck_coefficient.value(temperature) * gradient
        for temperature, gradient in zip(state, gradients)
    )
    voltage = spacing * (
        0.5 * fields[0] + sum(fields[1:-1]) + 0.5 * fields[-1]
    )
    if not math.isfinite(voltage):
        raise ValueError("independent-truth voltage became nonfinite")
    return voltage


def run_independent_distributed_experiment(
    experiment: DistributedLegExperiment,
    *,
    node_count: int = 25,
    time_step: float = 2.5e-4,
) -> IndependentDistributedResult:
    """Integrate nodal truth with transition-split third-order SSPRK."""

    if not isinstance(node_count, int) or isinstance(node_count, bool) or node_count < 5:
        raise ValueError("node count must be an integer of at least five")
    if not math.isfinite(time_step) or time_step <= 0.0:
        raise ValueError("truth time step must be finite and positive")
    positions = tuple(
        experiment.geometry.length * index / (node_count - 1)
        for index in range(node_count)
    )
    initial = tuple(
        experiment.initial_cold_face_temperature
        + (
            experiment.initial_hot_face_temperature
            - experiment.initial_cold_face_temperature
        )
        * index
        / (node_count - 1)
        for index in range(node_count)
    )
    times = [0.0]
    states = [_validate_state(initial)]
    tolerance = 1.0e-12 * max(1.0, experiment.duration)
    while times[-1] < experiment.duration - tolerance:
        start = times[-1]
        end = min(start + time_step, experiment.duration)
        if isinstance(experiment.current, PiecewiseConstantCurrent):
            transition = experiment.current.next_transition_after(start)
            if transition is not None:
                end = min(end, transition)
        step = end - start
        if step <= 0.0:
            raise RuntimeError("independent truth failed to advance in time")
        current = current_at(experiment.current, start)
        state = states[-1]
        first_rate = independent_distributed_rhs(
            experiment, state, current=current
        )
        first = _validate_state(
            tuple(value + step * rate for value, rate in zip(state, first_rate))
        )
        second_rate = independent_distributed_rhs(
            experiment, first, current=current
        )
        second = _validate_state(
            tuple(
                0.75 * original + 0.25 * (stage + step * rate)
                for original, stage, rate in zip(state, first, second_rate)
            )
        )
        third_rate = independent_distributed_rhs(
            experiment, second, current=current
        )
        final = _validate_state(
            tuple(
                original / 3.0 + 2.0 * (stage + step * rate) / 3.0
                for original, stage, rate in zip(state, second, third_rate)
            )
        )
        times.append(end)
        states.append(final)

    if abs(times[-1] - experiment.duration) <= tolerance:
        times[-1] = experiment.duration

    currents = tuple(current_at(experiment.current, time) for time in times)
    boundary_fluxes = tuple(
        _boundary_heat_fluxes(experiment, state, current)
        for state, current in zip(states, currents)
    )
    return IndependentDistributedResult(
        trajectory=IndependentDistributedTrajectory(
            time=tuple(times), positions=positions, temperature=tuple(states)
        ),
        diagnostics=IndependentDistributedDiagnostics(
            current=currents,
            voltage=tuple(
                independent_distributed_voltage(
                    experiment, state, current=current
                )
                for state, current in zip(states, currents)
            ),
            cold_side_heat=tuple(
                experiment.geometry.area * fluxes[0]
                for fluxes in boundary_fluxes
            ),
            hot_side_heat=tuple(
                experiment.geometry.area * fluxes[1]
                for fluxes in boundary_fluxes
            ),
        ),
    )


def interpolate_independent_temperature(
    trajectory: IndependentDistributedTrajectory,
    time: float,
) -> Tuple[float, ...]:
    """Linearly interpolate continuous nodal temperatures in time."""

    if not math.isfinite(time) or time < trajectory.time[0] or time > trajectory.time[-1]:
        raise ValueError("truth interpolation time lies outside the trajectory")
    right = bisect_left(trajectory.time, time)
    if right == len(trajectory.time):
        return trajectory.temperature[-1]
    if trajectory.time[right] == time or right == 0:
        return trajectory.temperature[right]
    left = right - 1
    fraction = (time - trajectory.time[left]) / (
        trajectory.time[right] - trajectory.time[left]
    )
    return tuple(
        low + fraction * (high - low)
        for low, high in zip(
            trajectory.temperature[left], trajectory.temperature[right]
        )
    )


def interpolate_independent_position(
    positions: Sequence[float],
    values: Sequence[float],
    target: float,
) -> float:
    """Linearly interpolate a nodal profile at one physical position."""

    positions = tuple(positions)
    values = tuple(values)
    if len(positions) != len(values) or len(values) < 2:
        raise ValueError("position and value vectors must have equal length")
    if target < positions[0] or target > positions[-1]:
        raise ValueError("target position lies outside the truth grid")
    right = bisect_left(positions, target)
    if positions[right] == target or right == 0:
        return values[right]
    left = right - 1
    fraction = (target - positions[left]) / (positions[right] - positions[left])
    return values[left] + fraction * (values[right] - values[left])


def observe_independent_distributed_result(
    experiment: DistributedLegExperiment,
    result: IndependentDistributedResult,
    *,
    observation_interval: float,
    channels: DistributedObservationChannels = DistributedObservationChannels(),
) -> DistributedObservationSet:
    """Expose sparse terminal measurements without reusing inference diagnostics."""

    selected = channels.names()
    observations = []
    for time in regular_distributed_observation_times(
        experiment.duration, observation_interval
    ):
        state = interpolate_independent_temperature(result.trajectory, time)
        current = current_at(experiment.current, time)
        cold_flux, hot_flux = _boundary_heat_fluxes(experiment, state, current)
        values = {
            "cold_face_temperature": state[0],
            "hot_face_temperature": state[-1],
            "voltage": independent_distributed_voltage(
                experiment, state, current=current
            ),
            "cold_side_heat": experiment.geometry.area * cold_flux,
            "hot_side_heat": experiment.geometry.area * hot_flux,
        }
        observations.extend(
            DistributedObservation(time, channel, values[channel])
            for channel in selected
        )
    return DistributedObservationSet(tuple(observations))


def property_values(
    prop: TemperatureProperty, temperatures: Sequence[float]
) -> Tuple[float, ...]:
    """Evaluate one truth property on a declared reporting grid."""

    return tuple(prop.value(float(temperature)) for temperature in temperatures)
