"""Two-node transient thermal model for ThermoTwin.

The thermoelectric element is quasi-steady. Thermal energy is stored only in
the cold and hot nodes, whose temperatures evolve according to the balances
agreed in Milestone 0.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple, Tuple

from ..core.controls import CurrentInput, PiecewiseConstantCurrent, current_at
from ..numerics.integration import IntegrationDivergenceError
from .thermoelectric import (
    ThermoelectricParameters,
    cold_side_heat,
    hot_side_heat,
)


@dataclass(frozen=True)
class TwoNodeThermalParameters:
    """Thermal properties of the cold and hot lumped nodes.

    Attributes:
        cold_thermal_capacitance: C_c in J/K.
        hot_thermal_capacitance: C_h in J/K.
        cold_reservoir_conductance: G_c in W/K.
        hot_reservoir_conductance: G_h in W/K.

    Reservoir conductances may be zero to represent insulated nodes. Thermal
    capacitances must be strictly positive because the RHS divides by them.
    """

    cold_thermal_capacitance: float
    hot_thermal_capacitance: float
    cold_reservoir_conductance: float
    hot_reservoir_conductance: float

    def __post_init__(self) -> None:
        if self.cold_thermal_capacitance <= 0:
            raise ValueError("cold thermal capacitance must be positive")
        if self.hot_thermal_capacitance <= 0:
            raise ValueError("hot thermal capacitance must be positive")
        if self.cold_reservoir_conductance < 0:
            raise ValueError("cold reservoir conductance cannot be negative")
        if self.hot_reservoir_conductance < 0:
            raise ValueError("hot reservoir conductance cannot be negative")


class TemperatureRates(NamedTuple):
    """Cold and hot temperature rates in K/s, in that order."""

    cold: float
    hot: float


class TemperatureTrajectory(NamedTuple):
    """Sampled time and node-temperature histories from an integration."""

    time: Tuple[float, ...]
    cold: Tuple[float, ...]
    hot: Tuple[float, ...]


class SteadyStateTemperatures(NamedTuple):
    """Cold and hot temperatures at a unique constant-input steady state."""

    cold: float
    hot: float


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
        f"two-node integration diverged near t={time:.9g} s during {stage}: "
        f"{reason}; reduce the time step or revise the model inputs"
    )


def _validate_integration_rates(
    rates: TemperatureRates,
    *,
    time: float,
    stage: str,
) -> None:
    if any(not math.isfinite(value) for value in rates):
        raise IntegrationDivergenceError(
            f"two-node integration diverged near t={time:.9g} s during "
            f"{stage}: a temperature rate became nonfinite; reduce the time "
            "step or revise the model inputs"
        )


def two_node_rhs(
    thermoelectric_parameters: ThermoelectricParameters,
    thermal_parameters: TwoNodeThermalParameters,
    *,
    cold_temperature: float,
    hot_temperature: float,
    current: float,
    cold_reservoir_temperature: float,
    hot_reservoir_temperature: float,
    cold_external_heat: float = 0.0,
    hot_external_heat: float = 0.0,
) -> TemperatureRates:
    """Return ``(dT_c/dt, dT_h/dt)`` for the two-node model.

    Temperatures are in kelvin, current is in amperes, and external heat inputs
    are in watts. Positive external heat enters its node.

    The implemented balances are

    ``C_c * dT_c/dt = G_c * (T_c_inf - T_c) + q_c_ext - Q_c``

    ``C_h * dT_h/dt = G_h * (T_h_inf - T_h) + q_h_ext + Q_h``.
    """

    cold_heat = cold_side_heat(
        thermoelectric_parameters,
        current,
        hot_temperature,
        cold_temperature,
    )
    hot_heat = hot_side_heat(
        thermoelectric_parameters,
        current,
        hot_temperature,
        cold_temperature,
    )

    cold_reservoir_heat = (
        thermal_parameters.cold_reservoir_conductance
        * (cold_reservoir_temperature - cold_temperature)
    )
    hot_reservoir_heat = (
        thermal_parameters.hot_reservoir_conductance
        * (hot_reservoir_temperature - hot_temperature)
    )

    cold_net_heat = cold_reservoir_heat + cold_external_heat - cold_heat
    hot_net_heat = hot_reservoir_heat + hot_external_heat + hot_heat

    return TemperatureRates(
        cold=cold_net_heat / thermal_parameters.cold_thermal_capacitance,
        hot=hot_net_heat / thermal_parameters.hot_thermal_capacitance,
    )


def two_node_steady_state(
    thermoelectric_parameters: ThermoelectricParameters,
    thermal_parameters: TwoNodeThermalParameters,
    *,
    current: float,
    cold_reservoir_temperature: float,
    hot_reservoir_temperature: float,
    cold_external_heat: float = 0.0,
    hot_external_heat: float = 0.0,
) -> SteadyStateTemperatures:
    """Solve the two constant-input node balances with both rates set to zero.

    The constant-property model produces a two-by-two linear system in the
    steady cold and hot temperatures. Thermal capacitances do not appear
    because no energy is accumulating at steady state.

    A unique algebraic solution does not by itself guarantee that the
    transient dynamics approach that solution. A singular or numerically
    ill-conditioned balance matrix raises ``ValueError``.
    """

    finite_inputs = (
        current,
        cold_reservoir_temperature,
        hot_reservoir_temperature,
        cold_external_heat,
        hot_external_heat,
    )
    if any(not math.isfinite(value) for value in finite_inputs):
        raise ValueError("steady-state inputs must be finite")
    if cold_reservoir_temperature <= 0.0 or hot_reservoir_temperature <= 0.0:
        raise ValueError("reservoir temperatures must be positive kelvin")

    alpha_current = thermoelectric_parameters.seebeck_coefficient * current
    thermal_conductance = thermoelectric_parameters.thermal_conductance
    half_joule_heat = (
        0.5
        * current**2
        * thermoelectric_parameters.electrical_resistance
    )
    cold_reservoir_conductance = (
        thermal_parameters.cold_reservoir_conductance
    )
    hot_reservoir_conductance = (
        thermal_parameters.hot_reservoir_conductance
    )

    cold_coefficient = (
        cold_reservoir_conductance
        + alpha_current
        + thermal_conductance
    )
    hot_coefficient = (
        hot_reservoir_conductance
        + thermal_conductance
        - alpha_current
    )
    cross_coefficient = -thermal_conductance

    cold_source = (
        cold_reservoir_conductance * cold_reservoir_temperature
        + cold_external_heat
        + half_joule_heat
    )
    hot_source = (
        hot_reservoir_conductance * hot_reservoir_temperature
        + hot_external_heat
        + half_joule_heat
    )

    determinant = (
        cold_coefficient * hot_coefficient - cross_coefficient**2
    )
    determinant_scale = (
        abs(cold_coefficient * hot_coefficient)
        + abs(cross_coefficient**2)
    )
    if determinant == 0.0 or (
        determinant_scale > 0.0
        and abs(determinant) <= 1e-12 * determinant_scale
    ):
        raise ValueError(
            "steady-state balance matrix is singular or ill-conditioned"
        )

    steady_state = SteadyStateTemperatures(
        cold=(
            cold_source * hot_coefficient
            - cross_coefficient * hot_source
        )
        / determinant,
        hot=(
            cold_coefficient * hot_source
            - cross_coefficient * cold_source
        )
        / determinant,
    )
    if any(not math.isfinite(value) for value in steady_state):
        raise ValueError("steady-state temperatures must be finite")
    if any(value <= 0.0 for value in steady_state):
        raise ValueError(
            "steady-state solution lies outside the positive-kelvin model "
            "domain; reduce the current or revise the model inputs"
        )
    return steady_state


def integrate_two_node(
    thermoelectric_parameters: ThermoelectricParameters,
    thermal_parameters: TwoNodeThermalParameters,
    *,
    initial_cold_temperature: float,
    initial_hot_temperature: float,
    duration: float,
    time_step: float,
    current: CurrentInput,
    cold_reservoir_temperature: float,
    hot_reservoir_temperature: float,
    cold_external_heat: float = 0.0,
    hot_external_heat: float = 0.0,
) -> TemperatureTrajectory:
    """Integrate the two-node model with fixed-input classical RK4.

    ``duration`` and ``time_step`` are in seconds. The returned trajectory
    includes the initial state at time zero and the final state at exactly
    ``duration``. If the duration is not an integer multiple of the requested
    step, the last step is shortened to end at the requested duration.

    ``current`` may be a constant scalar or a ``PiecewiseConstantCurrent``.
    Reservoir temperatures and external heat inputs remain constant. A zero
    duration returns only the initial state.

    When current switches, the integrator ends the preceding RK4 step exactly
    at the transition and begins the next step with the new current. This
    prevents a discontinuous step or pulse from being averaged across one
    integration interval.
    """

    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError("duration must be finite and nonnegative")
    if not math.isfinite(time_step) or time_step <= 0.0:
        raise ValueError("time step must be finite and positive")
    temperature_inputs = (
        initial_cold_temperature,
        initial_hot_temperature,
        cold_reservoir_temperature,
        hot_reservoir_temperature,
    )
    if any(not math.isfinite(value) for value in temperature_inputs):
        raise ValueError("initial and reservoir temperatures must be finite")
    if any(value <= 0.0 for value in temperature_inputs):
        raise ValueError(
            "initial and reservoir temperatures must be positive kelvin"
        )
    if any(
        not math.isfinite(value)
        for value in (cold_external_heat, hot_external_heat)
    ):
        raise ValueError("external heat inputs must be finite")
    current_at(current, 0.0)

    times = [0.0]
    cold_temperatures = [initial_cold_temperature]
    hot_temperatures = [initial_hot_temperature]

    def rates_at(
        cold_temperature: float,
        hot_temperature: float,
        step_current: float,
        evaluation_time: float,
        stage: str,
    ) -> TemperatureRates:
        _validate_integration_temperatures(
            (cold_temperature, hot_temperature),
            time=evaluation_time,
            stage=stage,
        )
        try:
            rates = two_node_rhs(
                thermoelectric_parameters,
                thermal_parameters,
                cold_temperature=cold_temperature,
                hot_temperature=hot_temperature,
                current=step_current,
                cold_reservoir_temperature=cold_reservoir_temperature,
                hot_reservoir_temperature=hot_reservoir_temperature,
                cold_external_heat=cold_external_heat,
                hot_external_heat=hot_external_heat,
            )
        except OverflowError as error:
            raise IntegrationDivergenceError(
                f"two-node integration overflowed near t={evaluation_time:.9g} "
                f"s during {stage}; reduce the time step or revise the model "
                "inputs"
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

        cold_temperature = cold_temperatures[-1]
        hot_temperature = hot_temperatures[-1]
        step_current = current_at(current, current_time)

        midpoint_time = current_time + 0.5 * step
        k1 = rates_at(
            cold_temperature,
            hot_temperature,
            step_current,
            current_time,
            "RK4 k1",
        )
        k2 = rates_at(
            cold_temperature + 0.5 * step * k1.cold,
            hot_temperature + 0.5 * step * k1.hot,
            step_current,
            midpoint_time,
            "RK4 k2",
        )
        k3 = rates_at(
            cold_temperature + 0.5 * step * k2.cold,
            hot_temperature + 0.5 * step * k2.hot,
            step_current,
            midpoint_time,
            "RK4 k3",
        )
        k4 = rates_at(
            cold_temperature + step * k3.cold,
            hot_temperature + step * k3.hot,
            step_current,
            next_time,
            "RK4 k4",
        )

        next_cold_temperature = (
            cold_temperature
            + step
            * (k1.cold + 2.0 * k2.cold + 2.0 * k3.cold + k4.cold)
            / 6.0
        )
        next_hot_temperature = (
            hot_temperature
            + step
            * (k1.hot + 2.0 * k2.hot + 2.0 * k3.hot + k4.hot)
            / 6.0
        )
        _validate_integration_temperatures(
            (next_cold_temperature, next_hot_temperature),
            time=next_time,
            stage="RK4 step result",
        )
        cold_temperatures.append(next_cold_temperature)
        hot_temperatures.append(next_hot_temperature)
        times.append(next_time)

    return TemperatureTrajectory(
        time=tuple(times),
        cold=tuple(cold_temperatures),
        hot=tuple(hot_temperatures),
    )
