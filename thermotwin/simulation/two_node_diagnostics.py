"""Derived histories for interpreting a ThermoTwin temperature trajectory."""

from typing import NamedTuple, Optional, Tuple

from ..core.controls import CurrentInput, current_at
from ..physics.thermoelectric import (
    ThermoelectricParameters,
    cold_side_heat,
    electrical_power,
    hot_side_heat,
    voltage,
)
from ..physics.two_node import TemperatureTrajectory


class TrajectoryDiagnostics(NamedTuple):
    """Pointwise outputs aligned with a ``TemperatureTrajectory``."""

    current: Tuple[float, ...]
    temperature_difference: Tuple[float, ...]
    cold_heat: Tuple[float, ...]
    hot_heat: Tuple[float, ...]
    voltage: Tuple[float, ...]
    electrical_power: Tuple[float, ...]
    cooling_cop: Tuple[Optional[float], ...]


def evaluate_trajectory(
    thermoelectric_parameters: ThermoelectricParameters,
    trajectory: TemperatureTrajectory,
    *,
    current: CurrentInput,
) -> TrajectoryDiagnostics:
    """Evaluate derived thermoelectric quantities at every trajectory sample.

    Heat rates and electrical power are in watts, temperature difference is in
    kelvin, and voltage is in volts. Cooling COP is ``Q_c / (V * I)``. It is
    returned as ``None`` when electrical power is zero and the ratio is
    undefined; a nonpositive value is retained but is not a useful cooling COP.
    """

    sample_count = len(trajectory.time)
    if (
        len(trajectory.cold) != sample_count
        or len(trajectory.hot) != sample_count
    ):
        raise ValueError(
            "trajectory time, cold, and hot histories must have equal lengths"
        )

    currents = []
    temperature_differences = []
    cold_heats = []
    hot_heats = []
    voltages = []
    electrical_powers = []
    cooling_cops = []

    for time, cold_temperature, hot_temperature in zip(
        trajectory.time, trajectory.cold, trajectory.hot
    ):
        sample_current = current_at(current, time)
        cold_heat = cold_side_heat(
            thermoelectric_parameters,
            sample_current,
            hot_temperature,
            cold_temperature,
        )
        hot_heat = hot_side_heat(
            thermoelectric_parameters,
            sample_current,
            hot_temperature,
            cold_temperature,
        )
        terminal_voltage = voltage(
            thermoelectric_parameters,
            sample_current,
            hot_temperature,
            cold_temperature,
        )
        power = electrical_power(
            thermoelectric_parameters,
            sample_current,
            hot_temperature,
            cold_temperature,
        )

        currents.append(sample_current)
        temperature_differences.append(hot_temperature - cold_temperature)
        cold_heats.append(cold_heat)
        hot_heats.append(hot_heat)
        voltages.append(terminal_voltage)
        electrical_powers.append(power)
        cooling_cops.append(None if power == 0.0 else cold_heat / power)

    return TrajectoryDiagnostics(
        current=tuple(currents),
        temperature_difference=tuple(temperature_differences),
        cold_heat=tuple(cold_heats),
        hot_heat=tuple(hot_heats),
        voltage=tuple(voltages),
        electrical_power=tuple(electrical_powers),
        cooling_cop=tuple(cooling_cops),
    )
