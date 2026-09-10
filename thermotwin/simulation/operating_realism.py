"""Composed virtual hardware used by the operating-decision realism study.

This module combines the truth-only temperature-dependent cold contact with
the removable cold-face probe.  The two primitive models live separately so
their limiting cases remain testable; this composition is the six-state
Family C hardware used only while the temporary probe is installed.
"""

import math
from ..core.controls import CurrentInput
from ..physics.four_node import FourNodeContactThermalParameters
from ..physics.thermoelectric import ThermoelectricParameters
from .interface_mass_mismatch import InterfaceMassMismatch
from .temperature_dependent_contact import (
    TemperatureDependentColdContact,
    temperature_dependent_contact_rhs,
)
from .temporary_face_sensor import (
    InterfaceMassTemporaryFaceSensorRates,
    InterfaceMassTemporaryFaceSensorTrajectory,
    TemporaryFaceSensor,
    _integrate_switch_aligned_rk4,
    temporary_face_sensor_coupling,
)


def temperature_dependent_contact_temporary_sensor_rhs(
    thermoelectric_parameters: ThermoelectricParameters,
    thermal_parameters: FourNodeContactThermalParameters,
    mismatch: InterfaceMassMismatch,
    contact: TemperatureDependentColdContact,
    sensor: TemporaryFaceSensor,
    *,
    cold_face_temperature: float,
    hot_face_temperature: float,
    cold_interface_temperature: float,
    cold_exchanger_temperature: float,
    hot_exchanger_temperature: float,
    face_sensor_temperature: float,
    current: float,
    cold_reservoir_temperature: float,
    hot_reservoir_temperature: float,
    cold_external_heat: float = 0.0,
    hot_external_heat: float = 0.0,
) -> InterfaceMassTemporaryFaceSensorRates:
    """Return Family C rates with an energy-conserving face probe attached."""

    base = temperature_dependent_contact_rhs(
        thermoelectric_parameters,
        thermal_parameters,
        mismatch,
        contact,
        cold_face_temperature=cold_face_temperature,
        hot_face_temperature=hot_face_temperature,
        cold_interface_temperature=cold_interface_temperature,
        cold_exchanger_temperature=cold_exchanger_temperature,
        hot_exchanger_temperature=hot_exchanger_temperature,
        current=current,
        cold_reservoir_temperature=cold_reservoir_temperature,
        hot_reservoir_temperature=hot_reservoir_temperature,
        cold_external_heat=cold_external_heat,
        hot_external_heat=hot_external_heat,
    )
    coupling = temporary_face_sensor_coupling(
        sensor,
        cold_face_temperature=cold_face_temperature,
        face_sensor_temperature=face_sensor_temperature,
        cold_face_thermal_capacitance=(
            thermal_parameters.cold_face_thermal_capacitance
        ),
    )
    return InterfaceMassTemporaryFaceSensorRates(
        cold_face=base.cold_face + coupling.cold_face_rate_adjustment,
        hot_face=base.hot_face,
        cold_interface=base.cold_interface,
        cold_exchanger=base.cold_exchanger,
        hot_exchanger=base.hot_exchanger,
        face_sensor=coupling.face_sensor_rate,
    )


def integrate_temperature_dependent_contact_temporary_sensor(
    thermoelectric_parameters: ThermoelectricParameters,
    thermal_parameters: FourNodeContactThermalParameters,
    mismatch: InterfaceMassMismatch,
    contact: TemperatureDependentColdContact,
    sensor: TemporaryFaceSensor,
    *,
    initial_temperature: float,
    duration: float,
    time_step: float,
    current: CurrentInput,
    cold_reservoir_temperature: float,
    hot_reservoir_temperature: float,
    cold_external_heat: float = 0.0,
    hot_external_heat: float = 0.0,
) -> InterfaceMassTemporaryFaceSensorTrajectory:
    """Integrate the instrumented Family C hardware with aligned RK4 steps."""

    if not isinstance(mismatch, InterfaceMassMismatch):
        raise ValueError("interface-mass mismatch configuration is required")
    if not isinstance(contact, TemperatureDependentColdContact):
        raise ValueError("temperature-dependent contact configuration is required")
    if not isinstance(sensor, TemporaryFaceSensor):
        raise ValueError("temporary face sensor configuration is required")
    environment = (
        initial_temperature,
        cold_reservoir_temperature,
        hot_reservoir_temperature,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in environment):
        raise ValueError("initial and reservoir temperatures must be positive")
    if any(
        not math.isfinite(value)
        for value in (cold_external_heat, hot_external_heat)
    ):
        raise ValueError("external heat inputs must be finite")

    initial = (initial_temperature,) * 6

    def rates(values, step_current, _evaluation_time, _stage):
        return temperature_dependent_contact_temporary_sensor_rhs(
            thermoelectric_parameters,
            thermal_parameters,
            mismatch,
            contact,
            sensor,
            cold_face_temperature=values[0],
            hot_face_temperature=values[1],
            cold_interface_temperature=values[2],
            cold_exchanger_temperature=values[3],
            hot_exchanger_temperature=values[4],
            face_sensor_temperature=values[5],
            current=step_current,
            cold_reservoir_temperature=cold_reservoir_temperature,
            hot_reservoir_temperature=hot_reservoir_temperature,
            cold_external_heat=cold_external_heat,
            hot_external_heat=hot_external_heat,
        )

    time, histories = _integrate_switch_aligned_rk4(
        initial_values=initial,
        duration=duration,
        time_step=time_step,
        current=current,
        rates=rates,
        model_name=(
            "temperature-dependent-contact temporary-face-sensor integration"
        ),
    )
    return InterfaceMassTemporaryFaceSensorTrajectory(time, *histories)


__all__ = [
    "integrate_temperature_dependent_contact_temporary_sensor",
    "temperature_dependent_contact_temporary_sensor_rhs",
]
