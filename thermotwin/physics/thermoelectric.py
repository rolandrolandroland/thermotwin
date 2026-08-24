"""Pure functions for a lumped thermoelectric cooler.

Sign conventions
----------------
* Positive current pumps heat from the cold face toward the hot face.
* ``cold_side_heat > 0`` means heat is removed from the cold node.
* ``hot_side_heat > 0`` means heat is delivered to the hot node.
* Temperatures are absolute temperatures in kelvin.

The model assumes constant properties, equal division of Joule heating
between the two faces, no Thomson effect, and no energy storage inside the
thermoelectric element.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ThermoelectricParameters:
    """Constant single-block thermoelectric properties.

    Attributes:
        seebeck_coefficient: Effective Seebeck coefficient, alpha, in V/K.
        electrical_resistance: Module electrical resistance, R, in ohms.
        thermal_conductance: Internal thermal conductance, K, in W/K.

    Idealized zero-valued properties are allowed for limiting-case tests.
    """

    seebeck_coefficient: float
    electrical_resistance: float
    thermal_conductance: float


def peltier_heat(
    parameters: ThermoelectricParameters,
    current: float,
    temperature: float,
) -> float:
    """Return the signed Peltier heat alpha * I * T in watts."""

    return parameters.seebeck_coefficient * current * temperature


def joule_heating(
    parameters: ThermoelectricParameters,
    current: float,
) -> float:
    """Return total irreversible Joule heating I**2 * R in watts."""

    return current**2 * parameters.electrical_resistance


def conductive_heat_leak(
    parameters: ThermoelectricParameters,
    hot_temperature: float,
    cold_temperature: float,
) -> float:
    """Return K * (T_h - T_c) in watts.

    A positive result denotes the passive heat leak from hot to cold.
    """

    return parameters.thermal_conductance * (
        hot_temperature - cold_temperature
    )


def cold_side_heat(
    parameters: ThermoelectricParameters,
    current: float,
    hot_temperature: float,
    cold_temperature: float,
) -> float:
    """Return heat removed from the cold node, Q_c, in watts."""

    return (
        peltier_heat(parameters, current, cold_temperature)
        - 0.5 * joule_heating(parameters, current)
        - conductive_heat_leak(
            parameters, hot_temperature, cold_temperature
        )
    )


def hot_side_heat(
    parameters: ThermoelectricParameters,
    current: float,
    hot_temperature: float,
    cold_temperature: float,
) -> float:
    """Return heat delivered to the hot node, Q_h, in watts."""

    return (
        peltier_heat(parameters, current, hot_temperature)
        + 0.5 * joule_heating(parameters, current)
        - conductive_heat_leak(
            parameters, hot_temperature, cold_temperature
        )
    )


def voltage(
    parameters: ThermoelectricParameters,
    current: float,
    hot_temperature: float,
    cold_temperature: float,
) -> float:
    """Return terminal voltage V = alpha * (T_h - T_c) + I * R."""

    return (
        parameters.seebeck_coefficient
        * (hot_temperature - cold_temperature)
        + current * parameters.electrical_resistance
    )


def electrical_power(
    parameters: ThermoelectricParameters,
    current: float,
    hot_temperature: float,
    cold_temperature: float,
) -> float:
    """Return signed terminal electrical power V * I in watts."""

    return current * voltage(
        parameters, current, hot_temperature, cold_temperature
    )


def coefficient_of_performance(
    parameters: ThermoelectricParameters,
    current: float,
    hot_temperature: float,
    cold_temperature: float,
) -> float:
    """Return cooling COP = Q_c / (V * I).

    Cooling COP is physically meaningful only when both Q_c and electrical
    input power are positive. At zero electrical power the ratio is undefined
    and this function raises ``ZeroDivisionError``.
    """

    power = electrical_power(
        parameters, current, hot_temperature, cold_temperature
    )
    if power == 0:
        raise ZeroDivisionError(
            "cooling COP is undefined at zero electrical power"
        )
    return cold_side_heat(
        parameters, current, hot_temperature, cold_temperature
    ) / power


def heating_coefficient_of_performance(
    parameters: ThermoelectricParameters,
    current: float,
    hot_temperature: float,
    cold_temperature: float,
) -> float:
    """Return heating COP = Q_h / (V * I).

    Heating COP is physically meaningful only when heat is delivered to the
    hot side and terminal electrical input is positive. At zero electrical
    power the ratio is undefined and this function raises ``ZeroDivisionError``.
    """

    power = electrical_power(
        parameters, current, hot_temperature, cold_temperature
    )
    if power == 0:
        raise ZeroDivisionError(
            "heating COP is undefined at zero electrical power"
        )
    return hot_side_heat(
        parameters, current, hot_temperature, cold_temperature
    ) / power
