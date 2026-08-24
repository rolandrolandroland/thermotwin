"""Thermally averaged PWM and converter-loss operating experiment.

This module deliberately does not shrink the thermal integrator time step to
the electrical switching period. It averages the current moments that the
thermoelectric equations actually use: Peltier heat depends on mean current,
while Joule heat depends on mean-square current. It also assumes temperature
ripple is negligible over a switching cycle, so current-temperature covariance
in the Peltier terms is neglected.
"""

from dataclasses import dataclass, replace
import math
from typing import NamedTuple, Optional, Tuple

from ..simulation.four_node_experiments import constant_current_contact_reference_experiment
from ..physics.four_node import (
    FourNodeContactSteadyState,
    four_node_contact_steady_state_from_current_moments,
)


@dataclass(frozen=True)
class PWMPowerElectronicsConfig:
    """Generic comparison grid and explicit converter assumptions."""

    mean_reservoir_temperature: float = 300.0
    external_temperature_lifts: Tuple[float, ...] = (0.0, 10.0, 20.0)
    mean_currents: Tuple[float, ...] = (
        0.15,
        0.30,
        0.45,
        0.60,
        0.75,
        0.90,
        1.05,
        1.20,
    )
    symmetric_contact_resistance: float = 0.25
    direct_pwm_peak_current: float = 1.50
    smoothed_ripple_peak_to_peak_fraction: float = 0.10
    converter_efficiency: float = 0.95
    fixed_switching_loss: float = 0.05

    def __post_init__(self) -> None:
        positive = (
            ("mean reservoir temperature", self.mean_reservoir_temperature),
            ("contact resistance", self.symmetric_contact_resistance),
            ("direct PWM peak current", self.direct_pwm_peak_current),
        )
        for name, value in positive:
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.external_temperature_lifts or any(
            not math.isfinite(value) or value < 0.0
            for value in self.external_temperature_lifts
        ):
            raise ValueError("external lifts must be finite and nonnegative")
        if tuple(sorted(set(self.external_temperature_lifts))) != tuple(
            self.external_temperature_lifts
        ):
            raise ValueError("external lifts must be unique and ordered")
        if not self.mean_currents or any(
            not math.isfinite(value)
            or value <= 0.0
            or value > self.direct_pwm_peak_current
            for value in self.mean_currents
        ):
            raise ValueError("mean currents must be positive and no larger than peak current")
        if tuple(sorted(set(self.mean_currents))) != tuple(self.mean_currents):
            raise ValueError("mean currents must be unique and ordered")
        if (
            not math.isfinite(self.smoothed_ripple_peak_to_peak_fraction)
            or self.smoothed_ripple_peak_to_peak_fraction < 0.0
        ):
            raise ValueError("ripple fraction must be finite and nonnegative")
        if (
            not math.isfinite(self.converter_efficiency)
            or self.converter_efficiency <= 0.0
            or self.converter_efficiency > 1.0
        ):
            raise ValueError("converter efficiency must lie in (0, 1]")
        if (
            not math.isfinite(self.fixed_switching_loss)
            or self.fixed_switching_loss < 0.0
        ):
            raise ValueError("switching loss must be finite and nonnegative")


class CurrentMoments(NamedTuple):
    """Electrical statistics retained by the averaged thermal model."""

    mode: str
    mean_current: float
    mean_square_current: float
    rms_current: float
    peak_current: float
    duty_cycle: Optional[float]
    joule_multiplier_over_dc: float


class AveragedThermoelectricRates(NamedTuple):
    cold_heat: float
    hot_heat: float
    module_electrical_power: float


class PWMOperatingPoint(NamedTuple):
    mode: str
    external_temperature_lift: float
    current: CurrentMoments
    cold_face_temperature: float
    hot_face_temperature: float
    cold_exchanger_temperature: float
    hot_exchanger_temperature: float
    face_temperature_lift: float
    delivered_cooling_rate: float
    delivered_heating_rate: float
    module_electrical_power: float
    supply_electrical_power: float
    module_cooling_cop: Optional[float]
    module_heating_cop: Optional[float]
    wall_cooling_cop: Optional[float]
    wall_heating_cop: Optional[float]


class PWMPowerElectronicsResult(NamedTuple):
    config: PWMPowerElectronicsConfig
    points: Tuple[PWMOperatingPoint, ...]


def ideal_dc_current_moments(mean_current: float) -> CurrentMoments:
    if not math.isfinite(mean_current) or mean_current <= 0.0:
        raise ValueError("mean current must be finite and positive")
    return CurrentMoments(
        "ideal_dc",
        mean_current,
        mean_current**2,
        mean_current,
        mean_current,
        1.0,
        1.0,
    )


def direct_pwm_current_moments(
    mean_current: float,
    peak_current: float,
) -> CurrentMoments:
    if (
        not math.isfinite(mean_current)
        or not math.isfinite(peak_current)
        or mean_current <= 0.0
        or peak_current <= 0.0
        or mean_current > peak_current
    ):
        raise ValueError("direct PWM requires 0 < mean current <= peak current")
    duty_cycle = mean_current / peak_current
    mean_square = duty_cycle * peak_current**2
    return CurrentMoments(
        "direct_pwm",
        mean_current,
        mean_square,
        math.sqrt(mean_square),
        peak_current,
        duty_cycle,
        mean_square / mean_current**2,
    )


def smoothed_pwm_current_moments(
    mean_current: float,
    ripple_peak_to_peak_fraction: float,
) -> CurrentMoments:
    if not math.isfinite(mean_current) or mean_current <= 0.0:
        raise ValueError("mean current must be finite and positive")
    if (
        not math.isfinite(ripple_peak_to_peak_fraction)
        or ripple_peak_to_peak_fraction < 0.0
    ):
        raise ValueError("ripple fraction must be finite and nonnegative")
    ripple_peak_to_peak = ripple_peak_to_peak_fraction * mean_current
    mean_square = mean_current**2 + ripple_peak_to_peak**2 / 12.0
    return CurrentMoments(
        "smoothed_pwm",
        mean_current,
        mean_square,
        math.sqrt(mean_square),
        mean_current + 0.5 * ripple_peak_to_peak,
        None,
        mean_square / mean_current**2,
    )


def averaged_thermoelectric_rates(
    *,
    seebeck_coefficient: float,
    electrical_resistance: float,
    thermal_conductance: float,
    current: CurrentMoments,
    hot_temperature: float,
    cold_temperature: float,
) -> AveragedThermoelectricRates:
    """Evaluate heat and power from current moments and mean temperatures.

    This closes ``mean(I*T)`` as ``mean(I)*mean(T)``. The neglected term is
    ``covariance(I, T)`` and is small only when thermal temperature ripple is
    negligible over an electrical switching period.
    """

    temperature_lift = hot_temperature - cold_temperature
    peltier_cold = seebeck_coefficient * current.mean_current * cold_temperature
    peltier_hot = seebeck_coefficient * current.mean_current * hot_temperature
    half_joule = 0.5 * electrical_resistance * current.mean_square_current
    conduction = thermal_conductance * temperature_lift
    cold_heat = peltier_cold - half_joule - conduction
    hot_heat = peltier_hot + half_joule - conduction
    power = (
        seebeck_coefficient * current.mean_current * temperature_lift
        + electrical_resistance * current.mean_square_current
    )
    return AveragedThermoelectricRates(cold_heat, hot_heat, power)


def averaged_contact_steady_state(
    current: CurrentMoments,
    *,
    cold_reservoir_temperature: float,
    hot_reservoir_temperature: float,
    symmetric_contact_resistance: float = 0.25,
) -> FourNodeContactSteadyState:
    """Solve the four zero-storage balances for an averaged current waveform."""

    if (
        not math.isfinite(symmetric_contact_resistance)
        or symmetric_contact_resistance <= 0.0
    ):
        raise ValueError("contact resistance must be positive and finite")
    if not math.isfinite(cold_reservoir_temperature) or not math.isfinite(
        hot_reservoir_temperature
    ):
        raise ValueError("reservoir temperatures must be finite")
    reference = constant_current_contact_reference_experiment()
    te = reference.thermoelectric_parameters
    thermal = replace(
        reference.thermal_parameters,
        cold_contact_resistance=symmetric_contact_resistance,
        hot_contact_resistance=symmetric_contact_resistance,
    )
    return four_node_contact_steady_state_from_current_moments(
        te,
        thermal,
        mean_current=current.mean_current,
        mean_square_current=current.mean_square_current,
        cold_reservoir_temperature=cold_reservoir_temperature,
        hot_reservoir_temperature=hot_reservoir_temperature,
    )


def evaluate_pwm_operating_point(
    current: CurrentMoments,
    external_temperature_lift: float,
    config: PWMPowerElectronicsConfig,
) -> PWMOperatingPoint:
    half_lift = 0.5 * external_temperature_lift
    cold_reservoir = config.mean_reservoir_temperature - half_lift
    hot_reservoir = config.mean_reservoir_temperature + half_lift
    state = averaged_contact_steady_state(
        current,
        cold_reservoir_temperature=cold_reservoir,
        hot_reservoir_temperature=hot_reservoir,
        symmetric_contact_resistance=config.symmetric_contact_resistance,
    )
    reference = constant_current_contact_reference_experiment()
    rates = averaged_thermoelectric_rates(
        seebeck_coefficient=(
            reference.thermoelectric_parameters.seebeck_coefficient
        ),
        electrical_resistance=(
            reference.thermoelectric_parameters.electrical_resistance
        ),
        thermal_conductance=(
            reference.thermoelectric_parameters.thermal_conductance
        ),
        current=current,
        hot_temperature=state.hot_face,
        cold_temperature=state.cold_face,
    )
    thermal = reference.thermal_parameters
    delivered_cooling = thermal.cold_reservoir_conductance * (
        cold_reservoir - state.cold_exchanger
    )
    delivered_heating = thermal.hot_reservoir_conductance * (
        state.hot_exchanger - hot_reservoir
    )
    if not math.isclose(delivered_cooling, rates.cold_heat, abs_tol=1e-10) or not math.isclose(
        delivered_heating, rates.hot_heat, abs_tol=1e-10
    ):
        raise RuntimeError("averaged steady heat rates do not close")
    if current.mode == "ideal_dc":
        supply_power = rates.module_electrical_power
    else:
        supply_power = (
            rates.module_electrical_power / config.converter_efficiency
            + config.fixed_switching_loss
        )
    module_cooling_cop = (
        delivered_cooling / rates.module_electrical_power
        if delivered_cooling > 0.0 and rates.module_electrical_power > 0.0
        else None
    )
    module_heating_cop = (
        delivered_heating / rates.module_electrical_power
        if delivered_heating > 0.0 and rates.module_electrical_power > 0.0
        else None
    )
    return PWMOperatingPoint(
        mode=current.mode,
        external_temperature_lift=external_temperature_lift,
        current=current,
        cold_face_temperature=state.cold_face,
        hot_face_temperature=state.hot_face,
        cold_exchanger_temperature=state.cold_exchanger,
        hot_exchanger_temperature=state.hot_exchanger,
        face_temperature_lift=state.hot_face - state.cold_face,
        delivered_cooling_rate=delivered_cooling,
        delivered_heating_rate=delivered_heating,
        module_electrical_power=rates.module_electrical_power,
        supply_electrical_power=supply_power,
        module_cooling_cop=module_cooling_cop,
        module_heating_cop=module_heating_cop,
        wall_cooling_cop=(
            delivered_cooling / supply_power
            if delivered_cooling > 0.0 and supply_power > 0.0
            else None
        ),
        wall_heating_cop=(
            delivered_heating / supply_power
            if delivered_heating > 0.0 and supply_power > 0.0
            else None
        ),
    )


def run_pwm_power_electronics_experiment(
    config: PWMPowerElectronicsConfig = PWMPowerElectronicsConfig(),
) -> PWMPowerElectronicsResult:
    points = []
    for lift in config.external_temperature_lifts:
        for mean_current in config.mean_currents:
            current_cases = (
                ideal_dc_current_moments(mean_current),
                smoothed_pwm_current_moments(
                    mean_current,
                    config.smoothed_ripple_peak_to_peak_fraction,
                ),
                direct_pwm_current_moments(
                    mean_current,
                    config.direct_pwm_peak_current,
                ),
            )
            points.extend(
                evaluate_pwm_operating_point(current, lift, config)
                for current in current_cases
            )
    return PWMPowerElectronicsResult(config, tuple(points))


def pwm_points_for(
    result: PWMPowerElectronicsResult,
    *,
    mode: str,
    external_temperature_lift: float,
) -> Tuple[PWMOperatingPoint, ...]:
    return tuple(
        point
        for point in result.points
        if point.mode == mode
        and point.external_temperature_lift == external_temperature_lift
    )


def format_pwm_power_electronics_report(
    result: PWMPowerElectronicsResult,
) -> str:
    lines = [
        "ThermoTwin thermally averaged PWM experiment",
        "Peltier term uses mean current; Joule term uses mean-square current",
        (
            f"converter: efficiency={result.config.converter_efficiency:.2f}, "
            f"fixed loss={result.config.fixed_switching_loss:.2f} W, "
            f"direct-PWM peak={result.config.direct_pwm_peak_current:.2f} A"
        ),
    ]
    representative_current = min(
        result.config.mean_currents,
        key=lambda value: abs(value - 0.60),
    )
    for lift in result.config.external_temperature_lifts:
        lines.append(f"external lift {lift:.0f} K at mean I={representative_current:.2f} A:")
        for mode in ("ideal_dc", "smoothed_pwm", "direct_pwm"):
            point = next(
                item
                for item in result.points
                if item.mode == mode
                and item.external_temperature_lift == lift
                and item.current.mean_current == representative_current
            )
            cooling_cop = (
                f"{point.wall_cooling_cop:.3f}"
                if point.wall_cooling_cop is not None
                else "not cooling"
            )
            lines.append(
                f"  {mode}: Qc={point.delivered_cooling_rate:.3f} W, "
                f"wall COPc={cooling_cop}, COPh={point.wall_heating_cop:.3f}, "
                f"Irms={point.current.rms_current:.3f} A, "
                f"Joule multiplier={point.current.joule_multiplier_over_dc:.3f}"
            )
    return "\n".join(lines)


def main() -> None:
    print(
        format_pwm_power_electronics_report(
            run_pwm_power_electronics_experiment()
        )
    )


if __name__ == "__main__":
    main()
