"""Independent five-state truth with temperature-dependent cold contact.

This truth model retains the interface thermal mass from
``interface_mass_mismatch`` and makes the total cold thermal-contact
resistance an exponential function of the hidden interface temperature.  The
four- and five-state inference models continue to use constant resistance, so
this constitutive law must remain a truth-only stress test.
"""

from dataclasses import dataclass
import math

from ..core.controls import CurrentInput, PiecewiseConstantCurrent, current_at
from ..numerics.integration import IntegrationDivergenceError
from ..physics.four_node import FourNodeContactThermalParameters
from ..physics.thermoelectric import (
    ThermoelectricParameters,
    cold_side_heat,
    hot_side_heat,
)
from .interface_mass_mismatch import (
    InterfaceMassMismatch,
    InterfaceMassRates,
    InterfaceMassTrajectory,
)


@dataclass(frozen=True)
class TemperatureDependentColdContact:
    """Positive exponential law for total cold thermal-contact resistance.

    ``beta`` is the logarithmic resistance slope in inverse kelvin.  The
    reference resistance itself remains the ``cold_contact_resistance`` in the
    shared thermal-parameter record and is interpreted at
    ``reference_temperature``.
    """

    beta: float = 0.12
    reference_temperature: float = 300.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.beta) or self.beta < 0.0:
            raise ValueError("contact beta must be finite and nonnegative")
        if (
            not math.isfinite(self.reference_temperature)
            or self.reference_temperature <= 0.0
        ):
            raise ValueError(
                "contact reference temperature must be finite and positive"
            )

    def resistance(
        self,
        reference_resistance: float,
        cold_interface_temperature: float,
    ) -> float:
        """Return ``R_300 exp(beta * (T_ci - 300 K))`` in K/W."""

        if (
            not math.isfinite(reference_resistance)
            or reference_resistance <= 0.0
        ):
            raise ValueError(
                "reference contact resistance must be finite and positive"
            )
        if (
            not math.isfinite(cold_interface_temperature)
            or cold_interface_temperature <= 0.0
        ):
            raise ValueError("cold interface temperature must be finite and positive")
        try:
            resistance = reference_resistance * math.exp(
                self.beta
                * (cold_interface_temperature - self.reference_temperature)
            )
        except OverflowError as error:
            raise ValueError(
                "temperature-dependent contact resistance overflowed"
            ) from error
        if not math.isfinite(resistance) or resistance <= 0.0:
            raise ValueError(
                "temperature-dependent contact resistance must be finite and positive"
            )
        return resistance


def temperature_dependent_contact_rhs(
    thermoelectric_parameters: ThermoelectricParameters,
    thermal_parameters: FourNodeContactThermalParameters,
    mismatch: InterfaceMassMismatch,
    contact: TemperatureDependentColdContact,
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
    """Return energy-conserving rates for the Family C virtual hardware."""

    total_resistance = contact.resistance(
        thermal_parameters.cold_contact_resistance,
        cold_interface_temperature,
    )
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


def integrate_temperature_dependent_contact_truth(
    thermoelectric_parameters: ThermoelectricParameters,
    thermal_parameters: FourNodeContactThermalParameters,
    mismatch: InterfaceMassMismatch,
    contact: TemperatureDependentColdContact,
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
    """Integrate the Family C truth with switch-aligned classical RK4."""

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
        raise ValueError("temperature-dependent contact inputs must be finite")
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
                "temperature-dependent contact truth diverged near "
                f"t={evaluation_time:.9g} s during {stage}"
            )
        try:
            result = temperature_dependent_contact_rhs(
                thermoelectric_parameters,
                thermal_parameters,
                mismatch,
                contact,
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
        except (OverflowError, ValueError) as error:
            raise IntegrationDivergenceError(
                "temperature-dependent contact truth failed near "
                f"t={evaluation_time:.9g} s during {stage}"
            ) from error
        if any(not math.isfinite(value) for value in result):
            raise IntegrationDivergenceError(
                "temperature-dependent contact truth produced a nonfinite rate"
            )
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
            raise RuntimeError(
                "temperature-dependent contact integration time failed to advance"
            )
        values = [history[-1] for history in states]
        step_current = current_at(current, time)
        k1 = rates(values, step_current, time, "RK4 k1")
        k2_values = [
            value + 0.5 * step * rate for value, rate in zip(values, k1)
        ]
        k2 = rates(k2_values, step_current, time + 0.5 * step, "RK4 k2")
        k3_values = [
            value + 0.5 * step * rate for value, rate in zip(values, k2)
        ]
        k3 = rates(k3_values, step_current, time + 0.5 * step, "RK4 k3")
        k4_values = [value + step * rate for value, rate in zip(values, k3)]
        k4 = rates(k4_values, step_current, next_time, "RK4 k4")
        next_values = [
            value + step * (a + 2.0 * b + 2.0 * c + d) / 6.0
            for value, a, b, c, d in zip(values, k1, k2, k3, k4)
        ]
        if any(not math.isfinite(value) or value <= 0.0 for value in next_values):
            raise IntegrationDivergenceError(
                "temperature-dependent contact truth left the positive-kelvin domain"
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


__all__ = [
    "TemperatureDependentColdContact",
    "integrate_temperature_dependent_contact_truth",
    "temperature_dependent_contact_rhs",
]
