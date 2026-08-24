"""Fair periodic continuous-versus-pulsed control comparison.

The comparison uses heat extracted from the cold reservoir as useful cooling,
not instantaneous module heat. Every operating point is evaluated only after
a warm-up interval, and the mean stored-energy drift is reported so transient
thermal storage cannot silently inflate delivered COP.
"""

from dataclasses import dataclass, replace
import math
from typing import Callable, NamedTuple, Optional, Sequence, Tuple

from ..simulation.four_node_experiments import (
    FourNodeContactExperiment,
    constant_current_contact_reference_experiment,
    run_four_node_contact_experiment,
)
from ..core.controls import CurrentInput, PiecewiseConstantCurrent, current_at
from ..numerics.integration import (
    first_rising_crossing_bracket,
    interpolated_value as _value_at,
    linear_interpolation as _linear_interpolation,
    trapezoidal_integral,
)
from ..physics.thermoelectric import ThermoelectricParameters, electrical_power


@dataclass(frozen=True)
class ControlComparisonConfig:
    """Frozen engineering constraints and search grid."""

    warmup_duration: float = 360.0
    evaluation_duration: float = 120.0
    time_step: float = 0.2
    target_cooling_rates: Tuple[float, ...] = (2.0, 5.0, 8.0)
    pulse_periods: Tuple[float, ...] = (10.0, 20.0, 30.0, 60.0)
    pulse_duty_cycles: Tuple[float, ...] = (
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
    )
    maximum_current: float = 1.5
    minimum_cold_face_temperature: float = 285.0
    maximum_hot_face_temperature: float = 315.0
    cooling_match_tolerance: float = 0.01
    amplitude_tolerance: float = 1e-4
    amplitude_bracket_subdivisions: int = 24
    maximum_storage_drift: float = 0.05

    def __post_init__(self) -> None:
        positive_scalars = (
            ("warm-up duration", self.warmup_duration),
            ("evaluation duration", self.evaluation_duration),
            ("time step", self.time_step),
            ("maximum current", self.maximum_current),
            ("cooling match tolerance", self.cooling_match_tolerance),
            ("amplitude tolerance", self.amplitude_tolerance),
            ("maximum storage drift", self.maximum_storage_drift),
        )
        for name, value in positive_scalars:
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name, values in (
            ("target cooling rates", self.target_cooling_rates),
            ("pulse periods", self.pulse_periods),
        ):
            if not values or any(
                not math.isfinite(value) or value <= 0.0 for value in values
            ):
                raise ValueError(f"{name} must be finite and positive")
        if not self.pulse_duty_cycles or any(
            not math.isfinite(value) or value <= 0.0 or value >= 1.0
            for value in self.pulse_duty_cycles
        ):
            raise ValueError("pulse duty cycles must lie between zero and one")
        if (
            isinstance(self.amplitude_bracket_subdivisions, bool)
            or not isinstance(self.amplitude_bracket_subdivisions, int)
            or self.amplitude_bracket_subdivisions < 2
        ):
            raise ValueError("amplitude bracket subdivisions must be at least two")
        if (
            not math.isfinite(self.minimum_cold_face_temperature)
            or not math.isfinite(self.maximum_hot_face_temperature)
            or self.maximum_hot_face_temperature
            <= self.minimum_cold_face_temperature
        ):
            raise ValueError("temperature limits must be finite and ordered")

    @property
    def total_duration(self) -> float:
        return self.warmup_duration + self.evaluation_duration


class ControlOperatingPoint(NamedTuple):
    """One post-warm-up operating point and its physical checks."""

    control_kind: str
    current_amplitude: float
    period: Optional[float]
    duty_cycle: Optional[float]
    average_cooling_rate: float
    average_electrical_power: float
    delivered_cooling_cop: float
    mean_storage_energy_drift: float
    minimum_cold_face_temperature: float
    maximum_hot_face_temperature: float
    safe: bool
    cyclically_settled: bool


class TargetControlComparison(NamedTuple):
    """Fair equal-cooling comparison at one requested capacity."""

    target_cooling_rate: float
    continuous: ControlOperatingPoint
    pulsed_candidates: Tuple[ControlOperatingPoint, ...]
    best_pulsed: ControlOperatingPoint
    equal_power_continuous: ControlOperatingPoint
    infeasible_pulse_shapes: int
    pulsed_cop_change_percent: float
    pulsed_cooling_change_at_equal_power_percent: float


class ControlUncertaintyCase(NamedTuple):
    """Fixed nominal schedules reevaluated at one contact resistance."""

    cold_contact_resistance: float
    continuous: ControlOperatingPoint
    pulsed: ControlOperatingPoint
    pulsed_cop_change_percent: float


class ControlComparisonResult(NamedTuple):
    """Capacity sweep plus resistance-uncertainty stress test."""

    comparisons: Tuple[TargetControlComparison, ...]
    uncertainty_cases: Tuple[ControlUncertaintyCase, ...]


def piecewise_electrical_energy(
    time: Sequence[float],
    cold_face_temperature: Sequence[float],
    hot_face_temperature: Sequence[float],
    thermoelectric_parameters: ThermoelectricParameters,
    current: CurrentInput,
    *,
    start_time: float,
    end_time: float,
) -> float:
    """Integrate electrical power without smoothing current discontinuities.

    Temperature is interpolated continuously, but each integration subinterval
    is assigned one constant current selected at its midpoint. Transition times
    are inserted as breakpoints even when they are absent from the sampled
    output grid. Both endpoint powers are then evaluated with that interval's
    current, preserving the correct left and right limits at a switch.
    """

    if (
        len(time) != len(cold_face_temperature)
        or len(time) != len(hot_face_temperature)
        or len(time) < 2
    ):
        raise ValueError("power histories must have equal length of at least two")
    if any(right <= left for left, right in zip(time, time[1:])):
        raise ValueError("power-history time must be strictly increasing")
    if end_time <= start_time:
        raise ValueError("integration end time must exceed start time")
    if start_time < time[0] or end_time > time[-1]:
        raise ValueError("integration interval lies outside sampled time")

    breakpoints = {start_time, end_time}
    breakpoints.update(
        sample_time
        for sample_time in time
        if start_time < sample_time < end_time
    )
    if isinstance(current, PiecewiseConstantCurrent):
        breakpoints.update(
            transition
            for transition in current.transition_times
            if start_time < transition < end_time
        )
    ordered = tuple(sorted(breakpoints))

    # Breakpoints are ordered, so advance through the sampled trajectory once.
    # Calling the general-purpose _value_at search for every endpoint would be
    # quadratic in the number of output samples for long control campaigns.
    sample_index = 0

    def temperatures_at(target_time: float) -> Tuple[float, float]:
        nonlocal sample_index
        while (
            sample_index < len(time) - 2
            and time[sample_index + 1] < target_time
        ):
            sample_index += 1
        left_time = time[sample_index]
        right_time = time[sample_index + 1]
        if not left_time <= target_time <= right_time:
            raise ValueError("power breakpoint lies outside sampled time")
        return (
            _linear_interpolation(
                left_time,
                right_time,
                cold_face_temperature[sample_index],
                cold_face_temperature[sample_index + 1],
                target_time,
            ),
            _linear_interpolation(
                left_time,
                right_time,
                hot_face_temperature[sample_index],
                hot_face_temperature[sample_index + 1],
                target_time,
            ),
        )

    energy = 0.0
    left_cold, left_hot = temperatures_at(ordered[0])
    for left_time, right_time in zip(ordered, ordered[1:]):
        interval_current = current_at(current, 0.5 * (left_time + right_time))
        right_cold, right_hot = temperatures_at(right_time)
        left_power = electrical_power(
            thermoelectric_parameters,
            interval_current,
            left_hot,
            left_cold,
        )
        right_power = electrical_power(
            thermoelectric_parameters,
            interval_current,
            right_hot,
            right_cold,
        )
        energy += 0.5 * (left_power + right_power) * (
            right_time - left_time
        )
        left_cold, left_hot = right_cold, right_hot
    return energy


def evaluate_control_schedule(
    experiment: FourNodeContactExperiment,
    *,
    control_kind: str,
    current_amplitude: float,
    period: Optional[float],
    duty_cycle: Optional[float],
    config: ControlComparisonConfig,
) -> ControlOperatingPoint:
    """Evaluate useful cooling, electrical input, storage drift, and safety."""

    result = run_four_node_contact_experiment(experiment)
    trajectory = result.trajectory
    start_time = config.warmup_duration
    end_time = config.total_duration
    duration = config.evaluation_duration
    thermal = experiment.thermal_parameters

    reservoir_cooling = tuple(
        thermal.cold_reservoir_conductance
        * (experiment.cold_reservoir_temperature - temperature)
        for temperature in trajectory.cold_exchanger
    )
    average_cooling = trapezoidal_integral(
        trajectory.time,
        reservoir_cooling,
        start_time=start_time,
        end_time=end_time,
    ) / duration
    average_power = piecewise_electrical_energy(
        trajectory.time,
        trajectory.cold_face,
        trajectory.hot_face,
        experiment.thermoelectric_parameters,
        experiment.current,
        start_time=start_time,
        end_time=end_time,
    ) / duration
    stored_energy = tuple(
        thermal.cold_face_thermal_capacitance * cold_face
        + thermal.hot_face_thermal_capacitance * hot_face
        + thermal.cold_exchanger_thermal_capacitance * cold_exchanger
        + thermal.hot_exchanger_thermal_capacitance * hot_exchanger
        for cold_face, hot_face, cold_exchanger, hot_exchanger in zip(
            trajectory.cold_face,
            trajectory.hot_face,
            trajectory.cold_exchanger,
            trajectory.hot_exchanger,
        )
    )
    storage_drift = (
        _value_at(trajectory.time, stored_energy, end_time)
        - _value_at(trajectory.time, stored_energy, start_time)
    ) / duration
    window_indices = tuple(
        index
        for index, sample_time in enumerate(trajectory.time)
        if start_time <= sample_time <= end_time
    )
    minimum_cold_face = min(
        trajectory.cold_face[index] for index in window_indices
    )
    maximum_hot_face = max(
        trajectory.hot_face[index] for index in window_indices
    )
    safe = (
        current_amplitude <= config.maximum_current
        and minimum_cold_face >= config.minimum_cold_face_temperature
        and maximum_hot_face <= config.maximum_hot_face_temperature
    )
    settled = abs(storage_drift) <= config.maximum_storage_drift
    cop = average_cooling / average_power if average_power > 0.0 else math.nan
    return ControlOperatingPoint(
        control_kind=control_kind,
        current_amplitude=current_amplitude,
        period=period,
        duty_cycle=duty_cycle,
        average_cooling_rate=average_cooling,
        average_electrical_power=average_power,
        delivered_cooling_cop=cop,
        mean_storage_energy_drift=storage_drift,
        minimum_cold_face_temperature=minimum_cold_face,
        maximum_hot_face_temperature=maximum_hot_face,
        safe=safe,
        cyclically_settled=settled,
    )


def _experiment_for_schedule(
    base: FourNodeContactExperiment,
    schedule: PiecewiseConstantCurrent,
    config: ControlComparisonConfig,
) -> FourNodeContactExperiment:
    return replace(
        base,
        duration=config.total_duration,
        time_step=config.time_step,
        current=schedule,
    )


def _match_target(
    base: FourNodeContactExperiment,
    target: float,
    schedule_builder: Callable[[float], PiecewiseConstantCurrent],
    point_builder: Callable[[FourNodeContactExperiment, float], ControlOperatingPoint],
    config: ControlComparisonConfig,
) -> Optional[ControlOperatingPoint]:
    evaluated = {}

    def evaluate(amplitude: float) -> ControlOperatingPoint:
        if amplitude not in evaluated:
            experiment = _experiment_for_schedule(
                base, schedule_builder(amplitude), config
            )
            evaluated[amplitude] = point_builder(experiment, amplitude)
        return evaluated[amplitude]

    # Preserve the inexpensive monotone-path bisection used in the validated
    # current envelope. If the endpoint falls below target, scan the interior
    # before declaring infeasibility because cooling can turn over at high I.
    maximum_point = evaluate(config.maximum_current)
    if maximum_point.average_cooling_rate >= target:
        bracket = (0.0, config.maximum_current)
    else:
        bracket = first_rising_crossing_bracket(
            lambda amplitude: evaluate(amplitude).average_cooling_rate,
            target=target,
            maximum_input=config.maximum_current,
            subdivisions=config.amplitude_bracket_subdivisions,
        )
    if bracket is None:
        return None
    lower, upper = bracket
    if lower == upper:
        return evaluate(upper)
    best = evaluate(upper)
    while upper - lower > config.amplitude_tolerance:
        candidate = 0.5 * (lower + upper)
        point = evaluate(candidate)
        best = point
        if point.average_cooling_rate < target:
            lower = candidate
        else:
            upper = candidate
    if abs(best.average_cooling_rate - target) > config.cooling_match_tolerance:
        best = evaluate(upper)
    return best


def _continuous_point(
    base: FourNodeContactExperiment,
    target: float,
    config: ControlComparisonConfig,
) -> ControlOperatingPoint:
    point = _match_target(
        base,
        target,
        lambda amplitude: PiecewiseConstantCurrent.constant(amplitude),
        lambda experiment, amplitude: evaluate_control_schedule(
            experiment,
            control_kind="continuous",
            current_amplitude=amplitude,
            period=None,
            duty_cycle=None,
            config=config,
        ),
        config,
    )
    if point is None:
        raise ValueError("target cooling exceeds continuous-current capacity")
    return point


def _continuous_point_at_power(
    base: FourNodeContactExperiment,
    target_power: float,
    config: ControlComparisonConfig,
) -> ControlOperatingPoint:
    lower = 0.0
    upper = config.maximum_current

    def evaluate(amplitude: float) -> ControlOperatingPoint:
        schedule = PiecewiseConstantCurrent.constant(amplitude)
        return evaluate_control_schedule(
            _experiment_for_schedule(base, schedule, config),
            control_kind="continuous_equal_power",
            current_amplitude=amplitude,
            period=None,
            duty_cycle=None,
            config=config,
        )

    upper_point = evaluate(upper)
    if upper_point.average_electrical_power < target_power:
        raise ValueError("target power exceeds continuous-current capability")
    best = upper_point
    while upper - lower > config.amplitude_tolerance:
        candidate = 0.5 * (lower + upper)
        best = evaluate(candidate)
        if best.average_electrical_power < target_power:
            lower = candidate
        else:
            upper = candidate
    return evaluate(0.5 * (lower + upper))


def _pulse_point(
    base: FourNodeContactExperiment,
    target: float,
    period: float,
    duty_cycle: float,
    config: ControlComparisonConfig,
) -> Optional[ControlOperatingPoint]:
    return _match_target(
        base,
        target,
        lambda amplitude: PiecewiseConstantCurrent.periodic_pulse(
            duration=config.total_duration,
            period=period,
            duty_cycle=duty_cycle,
            pulse_current=amplitude,
        ),
        lambda experiment, amplitude: evaluate_control_schedule(
            experiment,
            control_kind="pulsed",
            current_amplitude=amplitude,
            period=period,
            duty_cycle=duty_cycle,
            config=config,
        ),
        config,
    )


def run_control_comparison(
    config: ControlComparisonConfig = ControlComparisonConfig(),
    *,
    cold_contact_resistance_samples: Sequence[float] = (0.20, 0.25, 0.30),
) -> ControlComparisonResult:
    """Run equal-cooling control comparisons and a resistance stress test."""

    samples = tuple(float(value) for value in cold_contact_resistance_samples)
    if not samples or any(not math.isfinite(value) or value <= 0.0 for value in samples):
        raise ValueError("contact-resistance samples must be positive and finite")
    base = constant_current_contact_reference_experiment()
    comparisons = []
    for target in config.target_cooling_rates:
        continuous = _continuous_point(base, target, config)
        pulsed = []
        infeasible = 0
        for period in config.pulse_periods:
            for duty_cycle in config.pulse_duty_cycles:
                point = _pulse_point(
                    base, target, period, duty_cycle, config
                )
                if point is None or not point.safe or not point.cyclically_settled:
                    infeasible += 1
                else:
                    pulsed.append(point)
        if not pulsed:
            raise ValueError("no feasible pulsed schedule reaches the target")
        best_pulsed = max(pulsed, key=lambda point: point.delivered_cooling_cop)
        equal_power_continuous = _continuous_point_at_power(
            base,
            best_pulsed.average_electrical_power,
            config,
        )
        comparisons.append(
            TargetControlComparison(
                target_cooling_rate=target,
                continuous=continuous,
                pulsed_candidates=tuple(pulsed),
                best_pulsed=best_pulsed,
                equal_power_continuous=equal_power_continuous,
                infeasible_pulse_shapes=infeasible,
                pulsed_cop_change_percent=(
                    100.0
                    * (
                        best_pulsed.delivered_cooling_cop
                        / continuous.delivered_cooling_cop
                        - 1.0
                    )
                ),
                pulsed_cooling_change_at_equal_power_percent=(
                    100.0
                    * (
                        best_pulsed.average_cooling_rate
                        / equal_power_continuous.average_cooling_rate
                        - 1.0
                    )
                ),
            )
        )

    central_index = min(
        range(len(comparisons)),
        key=lambda index: abs(
            comparisons[index].target_cooling_rate - 5.0
        ),
    )
    nominal = comparisons[central_index]
    uncertainty_cases = []
    for resistance in samples:
        thermal = replace(
            base.thermal_parameters,
            cold_contact_resistance=resistance,
        )
        uncertain_base = replace(base, thermal_parameters=thermal)
        continuous_schedule = PiecewiseConstantCurrent.constant(
            nominal.continuous.current_amplitude
        )
        pulse_schedule = PiecewiseConstantCurrent.periodic_pulse(
            duration=config.total_duration,
            period=nominal.best_pulsed.period,
            duty_cycle=nominal.best_pulsed.duty_cycle,
            pulse_current=nominal.best_pulsed.current_amplitude,
        )
        continuous = evaluate_control_schedule(
            _experiment_for_schedule(uncertain_base, continuous_schedule, config),
            control_kind="continuous",
            current_amplitude=nominal.continuous.current_amplitude,
            period=None,
            duty_cycle=None,
            config=config,
        )
        pulse = evaluate_control_schedule(
            _experiment_for_schedule(uncertain_base, pulse_schedule, config),
            control_kind="pulsed",
            current_amplitude=nominal.best_pulsed.current_amplitude,
            period=nominal.best_pulsed.period,
            duty_cycle=nominal.best_pulsed.duty_cycle,
            config=config,
        )
        uncertainty_cases.append(
            ControlUncertaintyCase(
                cold_contact_resistance=resistance,
                continuous=continuous,
                pulsed=pulse,
                pulsed_cop_change_percent=(
                    100.0
                    * (
                        pulse.delivered_cooling_cop
                        / continuous.delivered_cooling_cop
                        - 1.0
                    )
                ),
            )
        )
    return ControlComparisonResult(
        comparisons=tuple(comparisons),
        uncertainty_cases=tuple(uncertainty_cases),
    )


def format_control_comparison_report(result: ControlComparisonResult) -> str:
    """Format the reproducible engineering conclusion."""

    lines = [
        "ThermoTwin continuous-versus-pulsed control comparison",
        "useful cooling: heat extracted from the cold reservoir",
        "comparison: equal delivered cooling after periodic warm-up",
    ]
    for comparison in result.comparisons:
        continuous = comparison.continuous
        pulsed = comparison.best_pulsed
        lines.extend(
            (
                f"target {comparison.target_cooling_rate:.3f} W:",
                (
                    f"  continuous I={continuous.current_amplitude:.4f} A, "
                    f"COP={continuous.delivered_cooling_cop:.4f}"
                ),
                (
                    f"  highest-COP tested pulse I={pulsed.current_amplitude:.4f} A, "
                    f"period={pulsed.period:.1f} s, "
                    f"duty={pulsed.duty_cycle:.2f}, "
                    f"COP={pulsed.delivered_cooling_cop:.4f}"
                ),
                (
                    f"  pulsed COP change={comparison.pulsed_cop_change_percent:+.2f}% "
                    f"({comparison.infeasible_pulse_shapes} infeasible shapes)"
                ),
                (
                    "  at equal electrical power: "
                    f"continuous cooling={comparison.equal_power_continuous.average_cooling_rate:.4f} W, "
                    f"pulsed cooling change="
                    f"{comparison.pulsed_cooling_change_at_equal_power_percent:+.2f}%"
                ),
            )
        )
        lines.append("  duty sweep (highest COP across tested periods):")
        feasible_duties = sorted(
            {
                point.duty_cycle
                for point in comparison.pulsed_candidates
                if point.duty_cycle is not None
            }
        )
        for duty_cycle in feasible_duties:
            duty_points = tuple(
                point
                for point in comparison.pulsed_candidates
                if point.duty_cycle == duty_cycle
            )
            duty_best = max(
                duty_points,
                key=lambda point: point.delivered_cooling_cop,
            )
            duty_change = 100.0 * (
                duty_best.delivered_cooling_cop
                / continuous.delivered_cooling_cop
                - 1.0
            )
            lines.append(
                f"    duty={duty_cycle:.2f}: "
                f"COP={duty_best.delivered_cooling_cop:.4f}, "
                f"change={duty_change:+.2f}%, "
                f"period={duty_best.period:.1f} s"
            )
    lines.append("fixed-schedule cold-contact-resistance stress test:")
    for case in result.uncertainty_cases:
        lines.append(
            f"  R_c={case.cold_contact_resistance:.3f} K/W: "
            f"pulsed COP change={case.pulsed_cop_change_percent:+.2f}%"
        )
    return "\n".join(lines)


def main() -> None:
    print(format_control_comparison_report(run_control_comparison()))


if __name__ == "__main__":
    main()
