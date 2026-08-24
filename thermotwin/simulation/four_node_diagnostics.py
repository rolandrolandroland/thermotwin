"""Derived histories for the contact-aware four-node trajectory."""

from typing import NamedTuple, Optional, Tuple

from ..physics.four_node import (
    FourNodeContactTemperatureTrajectory,
    FourNodeContactThermalParameters,
    four_node_contact_rhs,
    thermal_contact_heat,
)
from ..core.controls import CurrentInput, current_at
from ..physics.thermoelectric import (
    ThermoelectricParameters,
    cold_side_heat,
    electrical_power,
    hot_side_heat,
    voltage,
)


class ContactTrajectoryDiagnostics(NamedTuple):
    """Pointwise contact, thermoelectric, COP, and energy histories."""

    current: Tuple[float, ...]
    face_temperature_difference: Tuple[float, ...]
    exchanger_temperature_difference: Tuple[float, ...]
    cold_contact_temperature_drop: Tuple[float, ...]
    hot_contact_temperature_drop: Tuple[float, ...]
    cold_contact_heat: Tuple[float, ...]
    hot_contact_heat: Tuple[float, ...]
    cold_heat: Tuple[float, ...]
    hot_heat: Tuple[float, ...]
    voltage: Tuple[float, ...]
    electrical_power: Tuple[float, ...]
    module_cooling_cop: Tuple[Optional[float], ...]
    exchanger_cooling_cop: Tuple[Optional[float], ...]
    stored_energy_rate: Tuple[float, ...]
    external_energy_rate: Tuple[float, ...]
    energy_balance_residual: Tuple[float, ...]


def evaluate_contact_trajectory(
    thermoelectric_parameters: ThermoelectricParameters,
    thermal_parameters: FourNodeContactThermalParameters,
    trajectory: FourNodeContactTemperatureTrajectory,
    *,
    current: CurrentInput,
    cold_reservoir_temperature: float,
    hot_reservoir_temperature: float,
    cold_external_heat: float = 0.0,
    hot_external_heat: float = 0.0,
) -> ContactTrajectoryDiagnostics:
    """Evaluate aligned derived quantities for a four-node trajectory."""

    sample_count = len(trajectory.time)
    histories = (
        trajectory.cold_face,
        trajectory.hot_face,
        trajectory.cold_exchanger,
        trajectory.hot_exchanger,
    )
    if any(len(history) != sample_count for history in histories):
        raise ValueError(
            "time and all four temperature histories must have equal lengths"
        )

    currents = []
    face_temperature_differences = []
    exchanger_temperature_differences = []
    cold_contact_temperature_drops = []
    hot_contact_temperature_drops = []
    cold_contact_heats = []
    hot_contact_heats = []
    cold_heats = []
    hot_heats = []
    voltages = []
    electrical_powers = []
    module_cooling_cops = []
    exchanger_cooling_cops = []
    stored_energy_rates = []
    external_energy_rates = []
    energy_balance_residuals = []

    for (
        time,
        cold_face,
        hot_face,
        cold_exchanger,
        hot_exchanger,
    ) in zip(
        trajectory.time,
        trajectory.cold_face,
        trajectory.hot_face,
        trajectory.cold_exchanger,
        trajectory.hot_exchanger,
    ):
        sample_current = current_at(current, time)
        cold_contact_drop = cold_exchanger - cold_face
        hot_contact_drop = hot_face - hot_exchanger
        cold_contact = thermal_contact_heat(
            cold_exchanger,
            cold_face,
            thermal_parameters.cold_contact_resistance,
        )
        hot_contact = thermal_contact_heat(
            hot_face,
            hot_exchanger,
            thermal_parameters.hot_contact_resistance,
        )
        cold_module_heat = cold_side_heat(
            thermoelectric_parameters,
            sample_current,
            hot_face,
            cold_face,
        )
        hot_module_heat = hot_side_heat(
            thermoelectric_parameters,
            sample_current,
            hot_face,
            cold_face,
        )
        terminal_voltage = voltage(
            thermoelectric_parameters,
            sample_current,
            hot_face,
            cold_face,
        )
        power = electrical_power(
            thermoelectric_parameters,
            sample_current,
            hot_face,
            cold_face,
        )
        rates = four_node_contact_rhs(
            thermoelectric_parameters,
            thermal_parameters,
            cold_face_temperature=cold_face,
            hot_face_temperature=hot_face,
            cold_exchanger_temperature=cold_exchanger,
            hot_exchanger_temperature=hot_exchanger,
            current=sample_current,
            cold_reservoir_temperature=cold_reservoir_temperature,
            hot_reservoir_temperature=hot_reservoir_temperature,
            cold_external_heat=cold_external_heat,
            hot_external_heat=hot_external_heat,
        )
        stored_rate = (
            thermal_parameters.cold_face_thermal_capacitance
            * rates.cold_face
            + thermal_parameters.hot_face_thermal_capacitance
            * rates.hot_face
            + thermal_parameters.cold_exchanger_thermal_capacitance
            * rates.cold_exchanger
            + thermal_parameters.hot_exchanger_thermal_capacitance
            * rates.hot_exchanger
        )
        external_rate = (
            thermal_parameters.cold_reservoir_conductance
            * (cold_reservoir_temperature - cold_exchanger)
            + thermal_parameters.hot_reservoir_conductance
            * (hot_reservoir_temperature - hot_exchanger)
            + cold_external_heat
            + hot_external_heat
            + power
        )

        currents.append(sample_current)
        face_temperature_differences.append(hot_face - cold_face)
        exchanger_temperature_differences.append(
            hot_exchanger - cold_exchanger
        )
        cold_contact_temperature_drops.append(cold_contact_drop)
        hot_contact_temperature_drops.append(hot_contact_drop)
        cold_contact_heats.append(cold_contact)
        hot_contact_heats.append(hot_contact)
        cold_heats.append(cold_module_heat)
        hot_heats.append(hot_module_heat)
        voltages.append(terminal_voltage)
        electrical_powers.append(power)
        module_cooling_cops.append(
            None if power == 0.0 else cold_module_heat / power
        )
        exchanger_cooling_cops.append(
            None if power == 0.0 else cold_contact / power
        )
        stored_energy_rates.append(stored_rate)
        external_energy_rates.append(external_rate)
        energy_balance_residuals.append(stored_rate - external_rate)

    return ContactTrajectoryDiagnostics(
        current=tuple(currents),
        face_temperature_difference=tuple(face_temperature_differences),
        exchanger_temperature_difference=tuple(
            exchanger_temperature_differences
        ),
        cold_contact_temperature_drop=tuple(
            cold_contact_temperature_drops
        ),
        hot_contact_temperature_drop=tuple(hot_contact_temperature_drops),
        cold_contact_heat=tuple(cold_contact_heats),
        hot_contact_heat=tuple(hot_contact_heats),
        cold_heat=tuple(cold_heats),
        hot_heat=tuple(hot_heats),
        voltage=tuple(voltages),
        electrical_power=tuple(electrical_powers),
        module_cooling_cop=tuple(module_cooling_cops),
        exchanger_cooling_cop=tuple(exchanger_cooling_cops),
        stored_energy_rate=tuple(stored_energy_rates),
        external_energy_rate=tuple(external_energy_rates),
        energy_balance_residual=tuple(energy_balance_residuals),
    )
