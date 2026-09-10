"""Energy-conserving temporary cold-face temperature sensor dynamics.

The temporary sensor is a lumped thermal mass coupled only to the device's
cold face.  Its response time and thermal capacitance define the contact
conductance, so the measured sensor temperature both lags and perturbs the
face temperature.  Removing the sensor means using the original four- or
five-state model without this additional state.
"""

from dataclasses import dataclass
import math
from typing import Callable, NamedTuple, Sequence, Tuple

from ..core.controls import CurrentInput, PiecewiseConstantCurrent, current_at
from ..numerics.integration import IntegrationDivergenceError
from ..physics.four_node import (
    FourNodeContactTemperatureRates,
    FourNodeContactTemperatureTrajectory,
    FourNodeContactThermalParameters,
    four_node_contact_rhs,
)
from ..physics.thermoelectric import ThermoelectricParameters
from .interface_mass_mismatch import (
    InterfaceMassMismatch,
    InterfaceMassRates,
    InterfaceMassTrajectory,
    interface_mass_rhs,
)


@dataclass(frozen=True)
class TemporaryFaceSensor:
    """A temporary cold-face sensor with thermal loading.

    ``thermal_capacitance`` is the sensor's lumped heat capacity, ``C_s``, in
    J/K.  ``response_time_constant`` is ``tau_s`` in seconds for the sensor
    responding to a prescribed face temperature.  Their ratio is the thermal
    conductance between the device face and the sensor.

    The defaults are synthetic development values, not hardware calibration.
    """

    thermal_capacitance: float = 5.0
    response_time_constant: float = 2.5

    def __post_init__(self) -> None:
        normalized = []
        for name, value in (
            ("sensor thermal capacitance", self.thermal_capacitance),
            ("sensor response time constant", self.response_time_constant),
        ):
            try:
                finite = math.isfinite(value)
            except TypeError as error:
                raise ValueError(f"{name} must be finite and positive") from error
            if not finite or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            normalized.append(float(value))
        object.__setattr__(self, "thermal_capacitance", normalized[0])
        object.__setattr__(self, "response_time_constant", normalized[1])

    @property
    def thermal_conductance(self) -> float:
        """Return ``C_s / tau_s``, the face-to-sensor conductance in W/K."""

        return self.thermal_capacitance / self.response_time_constant


class TemporaryFaceSensorCoupling(NamedTuple):
    """Heat and temperature-rate terms caused by the sensor coupling."""

    heat_into_sensor: float
    cold_face_rate_adjustment: float
    face_sensor_rate: float


class FourNodeTemporaryFaceSensorRates(NamedTuple):
    cold_face: float
    hot_face: float
    cold_exchanger: float
    hot_exchanger: float
    face_sensor: float


class InterfaceMassTemporaryFaceSensorRates(NamedTuple):
    cold_face: float
    hot_face: float
    cold_interface: float
    cold_exchanger: float
    hot_exchanger: float
    face_sensor: float


class FourNodeTemporaryFaceSensorTrajectory(NamedTuple):
    time: Tuple[float, ...]
    cold_face: Tuple[float, ...]
    hot_face: Tuple[float, ...]
    cold_exchanger: Tuple[float, ...]
    hot_exchanger: Tuple[float, ...]
    face_sensor: Tuple[float, ...]

    @property
    def four_node_projection(self) -> FourNodeContactTemperatureTrajectory:
        return FourNodeContactTemperatureTrajectory(
            time=self.time,
            cold_face=self.cold_face,
            hot_face=self.hot_face,
            cold_exchanger=self.cold_exchanger,
            hot_exchanger=self.hot_exchanger,
        )


class InterfaceMassTemporaryFaceSensorTrajectory(NamedTuple):
    time: Tuple[float, ...]
    cold_face: Tuple[float, ...]
    hot_face: Tuple[float, ...]
    cold_interface: Tuple[float, ...]
    cold_exchanger: Tuple[float, ...]
    hot_exchanger: Tuple[float, ...]
    face_sensor: Tuple[float, ...]

    @property
    def interface_mass_projection(self) -> InterfaceMassTrajectory:
        return InterfaceMassTrajectory(
            time=self.time,
            cold_face=self.cold_face,
            hot_face=self.hot_face,
            cold_interface=self.cold_interface,
            cold_exchanger=self.cold_exchanger,
            hot_exchanger=self.hot_exchanger,
        )

    @property
    def four_node_projection(self) -> FourNodeContactTemperatureTrajectory:
        return self.interface_mass_projection.four_node_projection


def temporary_face_sensor_coupling(
    sensor: TemporaryFaceSensor,
    *,
    cold_face_temperature: float,
    face_sensor_temperature: float,
    cold_face_thermal_capacitance: float,
) -> TemporaryFaceSensorCoupling:
    """Return equal-and-opposite face/sensor thermal coupling terms."""

    if not isinstance(sensor, TemporaryFaceSensor):
        raise ValueError("temporary face sensor configuration is required")
    values = (
        cold_face_temperature,
        face_sensor_temperature,
        cold_face_thermal_capacitance,
    )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("face-sensor coupling inputs must be finite")
    if cold_face_temperature <= 0.0 or face_sensor_temperature <= 0.0:
        raise ValueError("face-sensor temperatures must be positive kelvin")
    if cold_face_thermal_capacitance <= 0.0:
        raise ValueError("cold-face thermal capacitance must be positive")

    heat_into_sensor = sensor.thermal_conductance * (
        cold_face_temperature - face_sensor_temperature
    )
    return TemporaryFaceSensorCoupling(
        heat_into_sensor=heat_into_sensor,
        cold_face_rate_adjustment=(
            -heat_into_sensor / cold_face_thermal_capacitance
        ),
        face_sensor_rate=heat_into_sensor / sensor.thermal_capacitance,
    )


def four_node_temporary_face_sensor_rhs(
    thermoelectric_parameters: ThermoelectricParameters,
    thermal_parameters: FourNodeContactThermalParameters,
    sensor: TemporaryFaceSensor,
    *,
    cold_face_temperature: float,
    hot_face_temperature: float,
    cold_exchanger_temperature: float,
    hot_exchanger_temperature: float,
    face_sensor_temperature: float,
    current: float,
    cold_reservoir_temperature: float,
    hot_reservoir_temperature: float,
    cold_external_heat: float = 0.0,
    hot_external_heat: float = 0.0,
) -> FourNodeTemporaryFaceSensorRates:
    """Return four-node device rates augmented by a temporary sensor state."""

    base: FourNodeContactTemperatureRates = four_node_contact_rhs(
        thermoelectric_parameters,
        thermal_parameters,
        cold_face_temperature=cold_face_temperature,
        hot_face_temperature=hot_face_temperature,
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
    return FourNodeTemporaryFaceSensorRates(
        cold_face=base.cold_face + coupling.cold_face_rate_adjustment,
        hot_face=base.hot_face,
        cold_exchanger=base.cold_exchanger,
        hot_exchanger=base.hot_exchanger,
        face_sensor=coupling.face_sensor_rate,
    )


def interface_mass_temporary_face_sensor_rhs(
    thermoelectric_parameters: ThermoelectricParameters,
    thermal_parameters: FourNodeContactThermalParameters,
    mismatch: InterfaceMassMismatch,
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
    """Return five-state device rates augmented by a temporary sensor state."""

    base: InterfaceMassRates = interface_mass_rhs(
        thermoelectric_parameters,
        thermal_parameters,
        mismatch,
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


def _integrate_switch_aligned_rk4(
    *,
    initial_values: Sequence[float],
    duration: float,
    time_step: float,
    current: CurrentInput,
    rates: Callable[
        [Tuple[float, ...], float, float, str], Tuple[float, ...]
    ],
    model_name: str,
) -> Tuple[Tuple[float, ...], Tuple[Tuple[float, ...], ...]]:
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError("duration must be finite and nonnegative")
    if not math.isfinite(time_step) or time_step <= 0.0:
        raise ValueError("time step must be finite and positive")
    values = tuple(float(value) for value in initial_values)
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("initial temperatures must be finite and positive")
    current_at(current, 0.0)

    times = [0.0]
    histories = [[value] for value in values]

    def checked_rates(
        state: Tuple[float, ...],
        step_current: float,
        evaluation_time: float,
        stage: str,
    ) -> Tuple[float, ...]:
        if any(not math.isfinite(value) or value <= 0.0 for value in state):
            raise IntegrationDivergenceError(
                f"{model_name} diverged near t={evaluation_time:.9g} s "
                f"during {stage}"
            )
        try:
            result = tuple(rates(state, step_current, evaluation_time, stage))
        except OverflowError as error:
            raise IntegrationDivergenceError(
                f"{model_name} overflowed near t={evaluation_time:.9g} s "
                f"during {stage}"
            ) from error
        if len(result) != len(state) or any(
            not math.isfinite(value) for value in result
        ):
            raise IntegrationDivergenceError(
                f"{model_name} produced invalid rates near "
                f"t={evaluation_time:.9g} s during {stage}"
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
            raise RuntimeError("temporary-sensor integration failed to advance")
        state = tuple(history[-1] for history in histories)
        step_current = current_at(current, time)
        k1 = checked_rates(state, step_current, time, "RK4 k1")
        k2_state = tuple(
            value + 0.5 * step * rate for value, rate in zip(state, k1)
        )
        k2 = checked_rates(
            k2_state,
            step_current,
            time + 0.5 * step,
            "RK4 k2",
        )
        k3_state = tuple(
            value + 0.5 * step * rate for value, rate in zip(state, k2)
        )
        k3 = checked_rates(
            k3_state,
            step_current,
            time + 0.5 * step,
            "RK4 k3",
        )
        k4_state = tuple(
            value + step * rate for value, rate in zip(state, k3)
        )
        k4 = checked_rates(k4_state, step_current, next_time, "RK4 k4")
        next_values = tuple(
            value + step * (a + 2.0 * b + 2.0 * c + d) / 6.0
            for value, a, b, c, d in zip(state, k1, k2, k3, k4)
        )
        if any(
            not math.isfinite(value) or value <= 0.0 for value in next_values
        ):
            raise IntegrationDivergenceError(
                f"{model_name} left the positive-kelvin domain"
            )
        for history, value in zip(histories, next_values):
            history.append(value)
        times.append(next_time)
    return tuple(times), tuple(tuple(history) for history in histories)


def _validate_environment(
    *,
    sensor: TemporaryFaceSensor,
    initial_temperatures: Sequence[float],
    cold_reservoir_temperature: float,
    hot_reservoir_temperature: float,
    cold_external_heat: float,
    hot_external_heat: float,
) -> None:
    if not isinstance(sensor, TemporaryFaceSensor):
        raise ValueError("temporary face sensor configuration is required")
    temperatures = (
        *initial_temperatures,
        cold_reservoir_temperature,
        hot_reservoir_temperature,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in temperatures):
        raise ValueError("initial and reservoir temperatures must be positive")
    if any(
        not math.isfinite(value)
        for value in (cold_external_heat, hot_external_heat)
    ):
        raise ValueError("external heat inputs must be finite")


def integrate_four_node_temporary_face_sensor(
    thermoelectric_parameters: ThermoelectricParameters,
    thermal_parameters: FourNodeContactThermalParameters,
    sensor: TemporaryFaceSensor,
    *,
    initial_cold_face_temperature: float,
    initial_hot_face_temperature: float,
    initial_cold_exchanger_temperature: float,
    initial_hot_exchanger_temperature: float,
    initial_face_sensor_temperature: float,
    duration: float,
    time_step: float,
    current: CurrentInput,
    cold_reservoir_temperature: float,
    hot_reservoir_temperature: float,
    cold_external_heat: float = 0.0,
    hot_external_heat: float = 0.0,
) -> FourNodeTemporaryFaceSensorTrajectory:
    """Integrate an instrumented four-node device with switch-aligned RK4."""

    initial = (
        initial_cold_face_temperature,
        initial_hot_face_temperature,
        initial_cold_exchanger_temperature,
        initial_hot_exchanger_temperature,
        initial_face_sensor_temperature,
    )
    _validate_environment(
        sensor=sensor,
        initial_temperatures=initial,
        cold_reservoir_temperature=cold_reservoir_temperature,
        hot_reservoir_temperature=hot_reservoir_temperature,
        cold_external_heat=cold_external_heat,
        hot_external_heat=hot_external_heat,
    )

    def rates(
        values: Tuple[float, ...],
        step_current: float,
        _evaluation_time: float,
        _stage: str,
    ) -> Tuple[float, ...]:
        return four_node_temporary_face_sensor_rhs(
            thermoelectric_parameters,
            thermal_parameters,
            sensor,
            cold_face_temperature=values[0],
            hot_face_temperature=values[1],
            cold_exchanger_temperature=values[2],
            hot_exchanger_temperature=values[3],
            face_sensor_temperature=values[4],
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
        model_name="four-node temporary-face-sensor integration",
    )
    return FourNodeTemporaryFaceSensorTrajectory(time, *histories)


def integrate_interface_mass_temporary_face_sensor(
    thermoelectric_parameters: ThermoelectricParameters,
    thermal_parameters: FourNodeContactThermalParameters,
    mismatch: InterfaceMassMismatch,
    sensor: TemporaryFaceSensor,
    *,
    initial_cold_face_temperature: float,
    initial_hot_face_temperature: float,
    initial_cold_interface_temperature: float,
    initial_cold_exchanger_temperature: float,
    initial_hot_exchanger_temperature: float,
    initial_face_sensor_temperature: float,
    duration: float,
    time_step: float,
    current: CurrentInput,
    cold_reservoir_temperature: float,
    hot_reservoir_temperature: float,
    cold_external_heat: float = 0.0,
    hot_external_heat: float = 0.0,
) -> InterfaceMassTemporaryFaceSensorTrajectory:
    """Integrate an instrumented five-state device with switch-aligned RK4."""

    if not isinstance(mismatch, InterfaceMassMismatch):
        raise ValueError("interface-mass mismatch configuration is required")
    initial = (
        initial_cold_face_temperature,
        initial_hot_face_temperature,
        initial_cold_interface_temperature,
        initial_cold_exchanger_temperature,
        initial_hot_exchanger_temperature,
        initial_face_sensor_temperature,
    )
    _validate_environment(
        sensor=sensor,
        initial_temperatures=initial,
        cold_reservoir_temperature=cold_reservoir_temperature,
        hot_reservoir_temperature=hot_reservoir_temperature,
        cold_external_heat=cold_external_heat,
        hot_external_heat=hot_external_heat,
    )

    def rates(
        values: Tuple[float, ...],
        step_current: float,
        _evaluation_time: float,
        _stage: str,
    ) -> Tuple[float, ...]:
        return interface_mass_temporary_face_sensor_rhs(
            thermoelectric_parameters,
            thermal_parameters,
            mismatch,
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
        model_name="interface-mass temporary-face-sensor integration",
    )
    return InterfaceMassTemporaryFaceSensorTrajectory(time, *histories)


__all__ = [
    "FourNodeTemporaryFaceSensorRates",
    "FourNodeTemporaryFaceSensorTrajectory",
    "InterfaceMassTemporaryFaceSensorRates",
    "InterfaceMassTemporaryFaceSensorTrajectory",
    "TemporaryFaceSensor",
    "TemporaryFaceSensorCoupling",
    "four_node_temporary_face_sensor_rhs",
    "integrate_four_node_temporary_face_sensor",
    "integrate_interface_mass_temporary_face_sensor",
    "interface_mass_temporary_face_sensor_rhs",
    "temporary_face_sensor_coupling",
]
