"""Four-node transient model with explicit thermal contact resistances.

The existing two-node model remains the reduced model without explicit
face-to-exchanger interfaces. This module adds separate thermoelectric-face
and heat-exchanger temperatures on both sides of the device.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple, Tuple

from ..core.controls import CurrentInput, PiecewiseConstantCurrent, current_at
from ..numerics.integration import IntegrationDivergenceError
from ..numerics.matrices import inverse_and_determinant
from .thermoelectric import (
    ThermoelectricParameters,
    cold_side_heat,
    hot_side_heat,
)


@dataclass(frozen=True)
class FourNodeContactThermalParameters:
    """Thermal properties for the contact-aware four-node network.

    Thermal capacitances are in J/K, contact resistances are in K/W, and
    reservoir conductances are in W/K. The four dynamic nodes are the cold
    thermoelectric face, hot thermoelectric face, cold heat exchanger, and hot
    heat exchanger.
    """

    cold_face_thermal_capacitance: float
    hot_face_thermal_capacitance: float
    cold_exchanger_thermal_capacitance: float
    hot_exchanger_thermal_capacitance: float
    cold_contact_resistance: float
    hot_contact_resistance: float
    cold_reservoir_conductance: float
    hot_reservoir_conductance: float

    def __post_init__(self) -> None:
        positive_parameters = (
            (
                "cold face thermal capacitance",
                self.cold_face_thermal_capacitance,
            ),
            (
                "hot face thermal capacitance",
                self.hot_face_thermal_capacitance,
            ),
            (
                "cold exchanger thermal capacitance",
                self.cold_exchanger_thermal_capacitance,
            ),
            (
                "hot exchanger thermal capacitance",
                self.hot_exchanger_thermal_capacitance,
            ),
            ("cold contact resistance", self.cold_contact_resistance),
            ("hot contact resistance", self.hot_contact_resistance),
        )
        for name, value in positive_parameters:
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

        nonnegative_parameters = (
            (
                "cold reservoir conductance",
                self.cold_reservoir_conductance,
            ),
            (
                "hot reservoir conductance",
                self.hot_reservoir_conductance,
            ),
        )
        for name, value in nonnegative_parameters:
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


class FourNodeContactTemperatureRates(NamedTuple):
    """Face and exchanger temperature rates in K/s."""

    cold_face: float
    hot_face: float
    cold_exchanger: float
    hot_exchanger: float


class FourNodeContactTemperatureTrajectory(NamedTuple):
    """Sampled time and four node-temperature histories."""

    time: Tuple[float, ...]
    cold_face: Tuple[float, ...]
    hot_face: Tuple[float, ...]
    cold_exchanger: Tuple[float, ...]
    hot_exchanger: Tuple[float, ...]


class FourNodeContactSteadyState(NamedTuple):
    """Unique constant-input four-node equilibrium temperatures in kelvin."""

    cold_face: float
    hot_face: float
    cold_exchanger: float
    hot_exchanger: float


def _validate_integration_temperatures(
    temperatures: Tuple[float, ...],
    *,
    time: float,
    stage: str,
) -> None:
    if any(not math.isfinite(value) for value in temperatures):
        reason = "the temperature state became nonfinite"
    elif any(value <= 0.0 for value in temperatures):
        reason = "the temperature state left the positive-kelvin domain"
    else:
        return
    raise IntegrationDivergenceError(
        f"four-node contact integration diverged near t={time:.9g} s during "
        f"{stage}: {reason}; reduce the time step or revise the model inputs"
    )


def _validate_integration_rates(
    rates: FourNodeContactTemperatureRates,
    *,
    time: float,
    stage: str,
) -> None:
    if any(not math.isfinite(value) for value in rates):
        raise IntegrationDivergenceError(
            f"four-node contact integration diverged near t={time:.9g} s "
            f"during {stage}: a temperature rate became nonfinite; reduce "
            "the time step or revise the model inputs"
        )


def thermal_contact_heat(
    from_temperature: float,
    to_temperature: float,
    contact_resistance: float,
) -> float:
    """Return contact heat from the first node toward the second in watts."""

    if (
        not math.isfinite(contact_resistance)
        or contact_resistance <= 0.0
    ):
        raise ValueError("contact resistance must be finite and positive")
    if not math.isfinite(from_temperature) or not math.isfinite(
        to_temperature
    ):
        raise ValueError("contact temperatures must be finite")
    return (from_temperature - to_temperature) / contact_resistance


def four_node_contact_rhs(
    thermoelectric_parameters: ThermoelectricParameters,
    thermal_parameters: FourNodeContactThermalParameters,
    *,
    cold_face_temperature: float,
    hot_face_temperature: float,
    cold_exchanger_temperature: float,
    hot_exchanger_temperature: float,
    current: float,
    cold_reservoir_temperature: float,
    hot_reservoir_temperature: float,
    cold_external_heat: float = 0.0,
    hot_external_heat: float = 0.0,
) -> FourNodeContactTemperatureRates:
    """Return all four temperature rates for the contact-aware model.

    Positive cold contact heat flows from the cold exchanger toward the cold
    thermoelectric face. Positive hot contact heat flows from the hot
    thermoelectric face toward the hot exchanger. Positive external heat
    enters its heat-exchanger node.
    """

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
    cold_contact_heat = thermal_contact_heat(
        cold_exchanger_temperature,
        cold_face_temperature,
        thermal_parameters.cold_contact_resistance,
    )
    hot_contact_heat = thermal_contact_heat(
        hot_face_temperature,
        hot_exchanger_temperature,
        thermal_parameters.hot_contact_resistance,
    )
    cold_reservoir_heat = (
        thermal_parameters.cold_reservoir_conductance
        * (cold_reservoir_temperature - cold_exchanger_temperature)
    )
    hot_reservoir_heat = (
        thermal_parameters.hot_reservoir_conductance
        * (hot_reservoir_temperature - hot_exchanger_temperature)
    )

    cold_face_net_heat = cold_contact_heat - cold_heat
    hot_face_net_heat = hot_heat - hot_contact_heat
    cold_exchanger_net_heat = (
        cold_reservoir_heat + cold_external_heat - cold_contact_heat
    )
    hot_exchanger_net_heat = (
        hot_reservoir_heat + hot_external_heat + hot_contact_heat
    )

    return FourNodeContactTemperatureRates(
        cold_face=(
            cold_face_net_heat
            / thermal_parameters.cold_face_thermal_capacitance
        ),
        hot_face=(
            hot_face_net_heat
            / thermal_parameters.hot_face_thermal_capacitance
        ),
        cold_exchanger=(
            cold_exchanger_net_heat
            / thermal_parameters.cold_exchanger_thermal_capacitance
        ),
        hot_exchanger=(
            hot_exchanger_net_heat
            / thermal_parameters.hot_exchanger_thermal_capacitance
        ),
    )


def four_node_contact_steady_state_from_current_moments(
    thermoelectric_parameters: ThermoelectricParameters,
    thermal_parameters: FourNodeContactThermalParameters,
    *,
    mean_current: float,
    mean_square_current: float,
    cold_reservoir_temperature: float,
    hot_reservoir_temperature: float,
    cold_external_heat: float = 0.0,
    hot_external_heat: float = 0.0,
) -> FourNodeContactSteadyState:
    """Solve steady balances from the first two current moments.

    Peltier heat uses ``mean_current`` and Joule heat uses
    ``mean_square_current``.  This is the shared algebraic kernel for scalar
    DC, direct PWM, smoothed PWM, and material/geometry co-design.  Thermal
    capacitances do not affect the equilibrium.
    """

    finite_inputs = (
        mean_current,
        mean_square_current,
        cold_reservoir_temperature,
        hot_reservoir_temperature,
        cold_external_heat,
        hot_external_heat,
    )
    if any(not math.isfinite(value) for value in finite_inputs):
        raise ValueError("steady-state inputs must be finite")
    if cold_reservoir_temperature <= 0.0 or hot_reservoir_temperature <= 0.0:
        raise ValueError("reservoir temperatures must be positive kelvin")
    if mean_square_current < 0.0:
        raise ValueError("mean-square current must be nonnegative")
    moment_tolerance = 1e-12 * max(1.0, mean_current**2)
    if mean_square_current + moment_tolerance < mean_current**2:
        raise ValueError(
            "mean-square current cannot be smaller than mean current squared"
        )

    alpha_current = (
        thermoelectric_parameters.seebeck_coefficient * mean_current
    )
    module_conductance = thermoelectric_parameters.thermal_conductance
    half_joule_heat = (
        0.5
        * mean_square_current
        * thermoelectric_parameters.electrical_resistance
    )
    cold_contact_conductance = 1.0 / thermal_parameters.cold_contact_resistance
    hot_contact_conductance = 1.0 / thermal_parameters.hot_contact_resistance
    cold_reservoir_conductance = thermal_parameters.cold_reservoir_conductance
    hot_reservoir_conductance = thermal_parameters.hot_reservoir_conductance

    coefficients = (
        (
            alpha_current + module_conductance + cold_contact_conductance,
            -module_conductance,
            -cold_contact_conductance,
            0.0,
        ),
        (
            -module_conductance,
            -alpha_current + module_conductance + hot_contact_conductance,
            0.0,
            -hot_contact_conductance,
        ),
        (
            -cold_contact_conductance,
            0.0,
            cold_reservoir_conductance + cold_contact_conductance,
            0.0,
        ),
        (
            0.0,
            -hot_contact_conductance,
            0.0,
            hot_reservoir_conductance + hot_contact_conductance,
        ),
    )
    source = (
        half_joule_heat,
        half_joule_heat,
        (
            cold_reservoir_conductance * cold_reservoir_temperature
            + cold_external_heat
        ),
        (
            hot_reservoir_conductance * hot_reservoir_temperature
            + hot_external_heat
        ),
    )
    inverse, _ = inverse_and_determinant(coefficients)
    solution = tuple(
        sum(coefficient * value for coefficient, value in zip(row, source))
        for row in inverse
    )
    steady_state = FourNodeContactSteadyState(*solution)
    if any(not math.isfinite(value) for value in steady_state):
        raise ValueError("steady-state temperatures must be finite")
    if any(value <= 0.0 for value in steady_state):
        raise ValueError(
            "steady-state solution lies outside the positive-kelvin model "
            "domain; reduce the current or revise the model inputs"
        )
    return steady_state


def four_node_contact_steady_state(
    thermoelectric_parameters: ThermoelectricParameters,
    thermal_parameters: FourNodeContactThermalParameters,
    *,
    current: float,
    cold_reservoir_temperature: float,
    hot_reservoir_temperature: float,
    cold_external_heat: float = 0.0,
    hot_external_heat: float = 0.0,
) -> FourNodeContactSteadyState:
    """Solve the four constant-current balances with all rates zero."""

    return four_node_contact_steady_state_from_current_moments(
        thermoelectric_parameters,
        thermal_parameters,
        mean_current=current,
        mean_square_current=current**2,
        cold_reservoir_temperature=cold_reservoir_temperature,
        hot_reservoir_temperature=hot_reservoir_temperature,
        cold_external_heat=cold_external_heat,
        hot_external_heat=hot_external_heat,
    )


def integrate_four_node_contact(
    thermoelectric_parameters: ThermoelectricParameters,
    thermal_parameters: FourNodeContactThermalParameters,
    *,
    initial_cold_face_temperature: float,
    initial_hot_face_temperature: float,
    initial_cold_exchanger_temperature: float,
    initial_hot_exchanger_temperature: float,
    duration: float,
    time_step: float,
    current: CurrentInput,
    cold_reservoir_temperature: float,
    hot_reservoir_temperature: float,
    cold_external_heat: float = 0.0,
    hot_external_heat: float = 0.0,
) -> FourNodeContactTemperatureTrajectory:
    """Integrate the four-node contact model with classical fixed-step RK4.

    The output contains the initial state and ends exactly at the requested
    duration. The final step is shortened when necessary. Steps also end
    exactly at piecewise-constant current transitions so a discontinuity is
    not averaged across an RK4 interval.
    """

    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError("duration must be finite and nonnegative")
    if not math.isfinite(time_step) or time_step <= 0.0:
        raise ValueError("time step must be finite and positive")

    finite_inputs = (
        initial_cold_face_temperature,
        initial_hot_face_temperature,
        initial_cold_exchanger_temperature,
        initial_hot_exchanger_temperature,
        cold_reservoir_temperature,
        hot_reservoir_temperature,
        cold_external_heat,
        hot_external_heat,
    )
    if any(not math.isfinite(value) for value in finite_inputs):
        raise ValueError(
            "temperatures and external heat inputs must be finite"
        )
    if any(value <= 0.0 for value in finite_inputs[:6]):
        raise ValueError(
            "initial and reservoir temperatures must be positive kelvin"
        )
    current_at(current, 0.0)

    times = [0.0]
    cold_face_temperatures = [initial_cold_face_temperature]
    hot_face_temperatures = [initial_hot_face_temperature]
    cold_exchanger_temperatures = [initial_cold_exchanger_temperature]
    hot_exchanger_temperatures = [initial_hot_exchanger_temperature]

    def rates_at(
        cold_face_temperature: float,
        hot_face_temperature: float,
        cold_exchanger_temperature: float,
        hot_exchanger_temperature: float,
        step_current: float,
        evaluation_time: float,
        stage: str,
    ) -> FourNodeContactTemperatureRates:
        temperatures = (
            cold_face_temperature,
            hot_face_temperature,
            cold_exchanger_temperature,
            hot_exchanger_temperature,
        )
        _validate_integration_temperatures(
            temperatures,
            time=evaluation_time,
            stage=stage,
        )
        try:
            rates = four_node_contact_rhs(
                thermoelectric_parameters,
                thermal_parameters,
                cold_face_temperature=cold_face_temperature,
                hot_face_temperature=hot_face_temperature,
                cold_exchanger_temperature=cold_exchanger_temperature,
                hot_exchanger_temperature=hot_exchanger_temperature,
                current=step_current,
                cold_reservoir_temperature=cold_reservoir_temperature,
                hot_reservoir_temperature=hot_reservoir_temperature,
                cold_external_heat=cold_external_heat,
                hot_external_heat=hot_external_heat,
            )
        except OverflowError as error:
            raise IntegrationDivergenceError(
                "four-node contact integration overflowed near "
                f"t={evaluation_time:.9g} s during {stage}; reduce the time "
                "step or revise the model inputs"
            ) from error
        _validate_integration_rates(
            rates,
            time=evaluation_time,
            stage=stage,
        )
        return rates

    while times[-1] < duration:
        current_time = times[-1]
        next_time = min(current_time + time_step, duration)
        if isinstance(current, PiecewiseConstantCurrent):
            transition = current.next_transition_after(current_time)
            if transition is not None:
                next_time = min(next_time, transition)
        step = next_time - current_time
        if step <= 0.0:
            raise RuntimeError("integration time failed to advance")

        cold_face = cold_face_temperatures[-1]
        hot_face = hot_face_temperatures[-1]
        cold_exchanger = cold_exchanger_temperatures[-1]
        hot_exchanger = hot_exchanger_temperatures[-1]
        step_current = current_at(current, current_time)

        midpoint_time = current_time + 0.5 * step
        k1 = rates_at(
            cold_face,
            hot_face,
            cold_exchanger,
            hot_exchanger,
            step_current,
            current_time,
            "RK4 k1",
        )
        k2 = rates_at(
            cold_face + 0.5 * step * k1.cold_face,
            hot_face + 0.5 * step * k1.hot_face,
            cold_exchanger + 0.5 * step * k1.cold_exchanger,
            hot_exchanger + 0.5 * step * k1.hot_exchanger,
            step_current,
            midpoint_time,
            "RK4 k2",
        )
        k3 = rates_at(
            cold_face + 0.5 * step * k2.cold_face,
            hot_face + 0.5 * step * k2.hot_face,
            cold_exchanger + 0.5 * step * k2.cold_exchanger,
            hot_exchanger + 0.5 * step * k2.hot_exchanger,
            step_current,
            midpoint_time,
            "RK4 k3",
        )
        k4 = rates_at(
            cold_face + step * k3.cold_face,
            hot_face + step * k3.hot_face,
            cold_exchanger + step * k3.cold_exchanger,
            hot_exchanger + step * k3.hot_exchanger,
            step_current,
            next_time,
            "RK4 k4",
        )

        next_cold_face = (
            cold_face
            + step
            * (
                k1.cold_face
                + 2.0 * k2.cold_face
                + 2.0 * k3.cold_face
                + k4.cold_face
            )
            / 6.0
        )
        next_hot_face = (
            hot_face
            + step
            * (
                k1.hot_face
                + 2.0 * k2.hot_face
                + 2.0 * k3.hot_face
                + k4.hot_face
            )
            / 6.0
        )
        next_cold_exchanger = (
            cold_exchanger
            + step
            * (
                k1.cold_exchanger
                + 2.0 * k2.cold_exchanger
                + 2.0 * k3.cold_exchanger
                + k4.cold_exchanger
            )
            / 6.0
        )
        next_hot_exchanger = (
            hot_exchanger
            + step
            * (
                k1.hot_exchanger
                + 2.0 * k2.hot_exchanger
                + 2.0 * k3.hot_exchanger
                + k4.hot_exchanger
            )
            / 6.0
        )
        _validate_integration_temperatures(
            (
                next_cold_face,
                next_hot_face,
                next_cold_exchanger,
                next_hot_exchanger,
            ),
            time=next_time,
            stage="RK4 step result",
        )
        cold_face_temperatures.append(next_cold_face)
        hot_face_temperatures.append(next_hot_face)
        cold_exchanger_temperatures.append(next_cold_exchanger)
        hot_exchanger_temperatures.append(next_hot_exchanger)
        times.append(next_time)

    return FourNodeContactTemperatureTrajectory(
        time=tuple(times),
        cold_face=tuple(cold_face_temperatures),
        hot_face=tuple(hot_face_temperatures),
        cold_exchanger=tuple(cold_exchanger_temperatures),
        hot_exchanger=tuple(hot_exchanger_temperatures),
    )
