"""Independent five-state truth model for interface-mass stress tests.

The production four-node inference model collapses the cold interface into one
resistance.  This module inserts an unobserved thermal mass between the cold
thermoelectric face and cold exchanger while preserving the same steady-state
series resistance.  It is intentionally truth-only: fitting continues to use
the simpler four-node model so withheld prediction can expose model mismatch.
"""

from dataclasses import dataclass
import math
from typing import NamedTuple, Tuple

from ..core.controls import CurrentInput, PiecewiseConstantCurrent, current_at
from ..numerics.integration import IntegrationDivergenceError
from ..observations.lag import FirstOrderTemperatureLag, apply_first_order_temperature_lag
from ..observations.test_stand import (
    IdealTemperatureSensor,
    IdealVirtualTestStand,
    ObservationDataset,
    TemperatureObservation,
    TemperatureSensorLocation,
    observe_contact_trajectory,
    regular_measurement_times,
)
from ..physics.four_node import (
    FourNodeContactTemperatureTrajectory,
    FourNodeContactThermalParameters,
)
from ..physics.thermoelectric import (
    ThermoelectricParameters,
    cold_side_heat,
    hot_side_heat,
)


def _accessible_test_stand(*, sampling_interval: float) -> IdealVirtualTestStand:
    return IdealVirtualTestStand(
        sensors=(
            IdealTemperatureSensor(
                "cold_exchanger_sensor",
                TemperatureSensorLocation.COLD_EXCHANGER,
            ),
            IdealTemperatureSensor(
                "hot_exchanger_sensor",
                TemperatureSensorLocation.HOT_EXCHANGER,
            ),
        ),
        sampling_interval=sampling_interval,
    )


@dataclass(frozen=True)
class InterfaceMassMismatch:
    """Extra cold-interface dynamics omitted by the inference model."""

    thermal_capacitance: float = 20.0
    face_side_resistance_fraction: float = 0.5

    def __post_init__(self) -> None:
        if not math.isfinite(self.thermal_capacitance) or self.thermal_capacitance <= 0.0:
            raise ValueError("interface thermal capacitance must be finite and positive")
        if (
            not math.isfinite(self.face_side_resistance_fraction)
            or not 0.0 < self.face_side_resistance_fraction < 1.0
        ):
            raise ValueError("interface resistance fraction must lie between zero and one")


class InterfaceMassRates(NamedTuple):
    cold_face: float
    hot_face: float
    cold_interface: float
    cold_exchanger: float
    hot_exchanger: float


class InterfaceMassTrajectory(NamedTuple):
    time: Tuple[float, ...]
    cold_face: Tuple[float, ...]
    hot_face: Tuple[float, ...]
    cold_interface: Tuple[float, ...]
    cold_exchanger: Tuple[float, ...]
    hot_exchanger: Tuple[float, ...]

    @property
    def four_node_projection(self) -> FourNodeContactTemperatureTrajectory:
        return FourNodeContactTemperatureTrajectory(
            time=self.time,
            cold_face=self.cold_face,
            hot_face=self.hot_face,
            cold_exchanger=self.cold_exchanger,
            hot_exchanger=self.hot_exchanger,
        )


def interface_mass_rhs(
    thermoelectric_parameters: ThermoelectricParameters,
    thermal_parameters: FourNodeContactThermalParameters,
    mismatch: InterfaceMassMismatch,
    *,
    cold_face_temperature: float,
    hot_face_temperature: float,
    cold_interface_temperature: float,
    cold_exchanger_temperature: float,
    hot_exchanger_temperature: float,
    current: float,
    cold_reservoir_temperature: float,
    hot_reservoir_temperature: float,
    cold_external_heat: float = 0.0,
    hot_external_heat: float = 0.0,
) -> InterfaceMassRates:
    """Return energy-conserving rates for the five-state virtual hardware."""

    total_resistance = thermal_parameters.cold_contact_resistance
    face_resistance = total_resistance * mismatch.face_side_resistance_fraction
    exchanger_resistance = total_resistance - face_resistance
    face_contact_heat = (
        cold_interface_temperature - cold_face_temperature
    ) / face_resistance
    exchanger_contact_heat = (
        cold_exchanger_temperature - cold_interface_temperature
    ) / exchanger_resistance
    cold_heat = cold_side_heat(
        thermoelectric_parameters,
        current,
        hot_face_temperature,
        cold_face_temperature,
    )
    hot_heat = hot_side_heat(
        thermoelectric_parameters,
        current,
        hot_face_temperature,
        cold_face_temperature,
    )
    hot_contact_heat = (
        hot_face_temperature - hot_exchanger_temperature
    ) / thermal_parameters.hot_contact_resistance
    cold_reservoir_heat = thermal_parameters.cold_reservoir_conductance * (
        cold_reservoir_temperature - cold_exchanger_temperature
    )
    hot_reservoir_heat = thermal_parameters.hot_reservoir_conductance * (
        hot_reservoir_temperature - hot_exchanger_temperature
    )
    return InterfaceMassRates(
        cold_face=(face_contact_heat - cold_heat)
        / thermal_parameters.cold_face_thermal_capacitance,
        hot_face=(hot_heat - hot_contact_heat)
        / thermal_parameters.hot_face_thermal_capacitance,
        cold_interface=(exchanger_contact_heat - face_contact_heat)
        / mismatch.thermal_capacitance,
        cold_exchanger=(
            cold_reservoir_heat + cold_external_heat - exchanger_contact_heat
        )
        / thermal_parameters.cold_exchanger_thermal_capacitance,
        hot_exchanger=(
            hot_reservoir_heat + hot_external_heat + hot_contact_heat
        )
        / thermal_parameters.hot_exchanger_thermal_capacitance,
    )


def integrate_interface_mass_truth(
    thermoelectric_parameters: ThermoelectricParameters,
    thermal_parameters: FourNodeContactThermalParameters,
    mismatch: InterfaceMassMismatch,
    *,
    initial_temperature: float,
    duration: float,
    time_step: float,
    current: CurrentInput,
    cold_reservoir_temperature: float,
    hot_reservoir_temperature: float,
    cold_external_heat: float = 0.0,
    hot_external_heat: float = 0.0,
) -> InterfaceMassTrajectory:
    """Integrate the truth model with switch-aligned classical RK4."""

    finite = (
        initial_temperature,
        duration,
        time_step,
        cold_reservoir_temperature,
        hot_reservoir_temperature,
        cold_external_heat,
        hot_external_heat,
    )
    if any(not math.isfinite(value) for value in finite):
        raise ValueError("interface-mass integration inputs must be finite")
    if (
        initial_temperature <= 0.0
        or cold_reservoir_temperature <= 0.0
        or hot_reservoir_temperature <= 0.0
    ):
        raise ValueError("initial and reservoir temperatures must be positive")
    if duration < 0.0 or time_step <= 0.0:
        raise ValueError("duration must be nonnegative and time step positive")
    current_at(current, 0.0)

    times = [0.0]
    states = [[initial_temperature] for _ in range(5)]

    def rates(values, step_current, evaluation_time, stage):
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise IntegrationDivergenceError(
                f"interface-mass truth diverged near t={evaluation_time:.9g} s during {stage}"
            )
        result = interface_mass_rhs(
            thermoelectric_parameters,
            thermal_parameters,
            mismatch,
            cold_face_temperature=values[0],
            hot_face_temperature=values[1],
            cold_interface_temperature=values[2],
            cold_exchanger_temperature=values[3],
            hot_exchanger_temperature=values[4],
            current=step_current,
            cold_reservoir_temperature=cold_reservoir_temperature,
            hot_reservoir_temperature=hot_reservoir_temperature,
            cold_external_heat=cold_external_heat,
            hot_external_heat=hot_external_heat,
        )
        if any(not math.isfinite(value) for value in result):
            raise IntegrationDivergenceError("interface-mass truth produced a nonfinite rate")
        return result

    while times[-1] < duration:
        time = times[-1]
        next_time = min(time + time_step, duration)
        if isinstance(current, PiecewiseConstantCurrent):
            transition = current.next_transition_after(time)
            if transition is not None:
                next_time = min(next_time, transition)
        step = next_time - time
        if step <= 0.0:
            raise RuntimeError("interface-mass integration time failed to advance")
        values = [history[-1] for history in states]
        step_current = current_at(current, time)
        k1 = rates(values, step_current, time, "RK4 k1")
        k2_values = [value + 0.5 * step * rate for value, rate in zip(values, k1)]
        k2 = rates(k2_values, step_current, time + 0.5 * step, "RK4 k2")
        k3_values = [value + 0.5 * step * rate for value, rate in zip(values, k2)]
        k3 = rates(k3_values, step_current, time + 0.5 * step, "RK4 k3")
        k4_values = [value + step * rate for value, rate in zip(values, k3)]
        k4 = rates(k4_values, step_current, next_time, "RK4 k4")
        next_values = [
            value + step * (a + 2.0 * b + 2.0 * c + d) / 6.0
            for value, a, b, c, d in zip(values, k1, k2, k3, k4)
        ]
        if any(not math.isfinite(value) or value <= 0.0 for value in next_values):
            raise IntegrationDivergenceError(
                "interface-mass truth left the positive-kelvin domain"
            )
        for history, value in zip(states, next_values):
            history.append(value)
        times.append(next_time)

    return InterfaceMassTrajectory(
        time=tuple(times),
        cold_face=tuple(states[0]),
        hot_face=tuple(states[1]),
        cold_interface=tuple(states[2]),
        cold_exchanger=tuple(states[3]),
        hot_exchanger=tuple(states[4]),
    )


def _resample_lagged(
    dataset: ObservationDataset,
    *,
    sampling_interval: float,
) -> ObservationDataset:
    requested = regular_measurement_times(dataset.measurement_times[-1], sampling_interval)
    by_key = {
        (round(item.time, 9), item.sensor_name): item
        for item in dataset.observations
    }
    observations = []
    for time in requested:
        for sensor in dataset.sensors:
            item = by_key.get((round(time, 9), sensor.name))
            if item is None:
                raise ValueError("dense mismatch history does not contain requested sample")
            observations.append(
                TemperatureObservation(
                    time=time,
                    sensor_name=item.sensor_name,
                    location=item.location,
                    temperature=item.temperature,
                    current=item.current,
                )
            )
    return ObservationDataset(
        observations=tuple(observations),
        sensors=dataset.sensors,
        sampling_interval=sampling_interval,
    )


def simulate_interface_mass_observations(
    current: PiecewiseConstantCurrent,
    *,
    thermoelectric_parameters: ThermoelectricParameters,
    thermal_parameters: FourNodeContactThermalParameters,
    mismatch: InterfaceMassMismatch,
    sensor_time_constant: float,
    sampling_interval: float,
    dense_time_step: float,
    duration: float = 80.0,
    equilibrium_temperature: float = 300.0,
) -> Tuple[ObservationDataset, InterfaceMassTrajectory]:
    """Return accessible lagged observations from the richer truth model."""

    trajectory = integrate_interface_mass_truth(
        thermoelectric_parameters,
        thermal_parameters,
        mismatch,
        initial_temperature=equilibrium_temperature,
        duration=duration,
        time_step=dense_time_step,
        current=current,
        cold_reservoir_temperature=equilibrium_temperature,
        hot_reservoir_temperature=equilibrium_temperature,
    )
    dense = observe_contact_trajectory(
        trajectory.four_node_projection,
        current=current,
        test_stand=_accessible_test_stand(sampling_interval=dense_time_step),
    )
    lagged = apply_first_order_temperature_lag(
        dense,
        FirstOrderTemperatureLag(default_time_constant=sensor_time_constant),
    ).dataset
    return (
        _resample_lagged(lagged, sampling_interval=sampling_interval),
        trajectory,
    )


__all__ = [
    "InterfaceMassMismatch",
    "InterfaceMassRates",
    "InterfaceMassTrajectory",
    "integrate_interface_mass_truth",
    "interface_mass_rhs",
    "simulate_interface_mass_observations",
]
