"""Place the existing transient pulse comparison on the steady COP map."""

import math
from typing import NamedTuple, Tuple

from .control_comparison import (
    ControlComparisonConfig,
    ControlComparisonResult,
    run_control_comparison,
)
from .operating_map import (
    COPOperatingMapConfig,
    SteadyOperatingPoint,
    contact_steady_operating_point,
)


class PulseMapComparison(NamedTuple):
    """One equal-cooling pulse result and its steady-envelope reference."""

    target_cooling_rate: float
    continuous_current: float
    continuous_cop: float
    steady_map_cop: float
    steady_map_cop_error_percent: float
    pulse_peak_current: float
    pulse_period: float
    pulse_duty_cycle: float
    pulse_mean_current: float
    pulse_rms_current: float
    pulse_cop: float
    pulse_cop_change_percent: float
    pulse_cooling_change_at_equal_power_percent: float
    continuous_storage_drift: float
    pulse_storage_drift: float


class PulseOperatingMapResult(NamedTuple):
    """Steady baseline curve, tested pulse markers, and full duty candidates."""

    steady_curve: Tuple[SteadyOperatingPoint, ...]
    comparisons: Tuple[PulseMapComparison, ...]
    control_result: ControlComparisonResult


def run_pulse_operating_map(
    control_config: ControlComparisonConfig = ControlComparisonConfig(),
    map_config: COPOperatingMapConfig = COPOperatingMapConfig(),
) -> PulseOperatingMapResult:
    """Run the existing fair pulse study and connect it to the zero-lift map."""

    control_result = run_control_comparison(control_config)
    baseline_resistance = 0.25
    baseline_mean_temperature = 300.0
    steady_curve = tuple(
        contact_steady_operating_point(
            current,
            0.0,
            baseline_resistance,
            mean_reservoir_temperature=baseline_mean_temperature,
        )
        for current in map_config.currents
    )
    comparisons = []
    for comparison in control_result.comparisons:
        continuous = comparison.continuous
        pulse = comparison.best_pulsed
        if pulse.period is None or pulse.duty_cycle is None:
            raise RuntimeError("best pulsed point lacks pulse settings")
        steady_reference = contact_steady_operating_point(
            continuous.current_amplitude,
            0.0,
            baseline_resistance,
            mean_reservoir_temperature=baseline_mean_temperature,
        )
        if steady_reference.cooling_cop is None:
            raise RuntimeError("positive matched cooling must have steady COP")
        comparisons.append(
            PulseMapComparison(
                target_cooling_rate=comparison.target_cooling_rate,
                continuous_current=continuous.current_amplitude,
                continuous_cop=continuous.delivered_cooling_cop,
                steady_map_cop=steady_reference.cooling_cop,
                steady_map_cop_error_percent=100.0
                * (
                    continuous.delivered_cooling_cop
                    / steady_reference.cooling_cop
                    - 1.0
                ),
                pulse_peak_current=pulse.current_amplitude,
                pulse_period=pulse.period,
                pulse_duty_cycle=pulse.duty_cycle,
                pulse_mean_current=pulse.duty_cycle * pulse.current_amplitude,
                pulse_rms_current=(
                    math.sqrt(pulse.duty_cycle) * pulse.current_amplitude
                ),
                pulse_cop=pulse.delivered_cooling_cop,
                pulse_cop_change_percent=comparison.pulsed_cop_change_percent,
                pulse_cooling_change_at_equal_power_percent=(
                    comparison.pulsed_cooling_change_at_equal_power_percent
                ),
                continuous_storage_drift=continuous.mean_storage_energy_drift,
                pulse_storage_drift=pulse.mean_storage_energy_drift,
            )
        )
    return PulseOperatingMapResult(
        steady_curve=steady_curve,
        comparisons=tuple(comparisons),
        control_result=control_result,
    )


def format_pulse_operating_map_report(result: PulseOperatingMapResult) -> str:
    lines = [
        "ThermoTwin transient pulse points on the steady COP map",
        "boundary: equal 300 K reservoirs (external lift = 0 K)",
        "fair comparison: equal post-warm-up delivered cooling",
    ]
    for item in result.comparisons:
        lines.extend(
            (
                f"target {item.target_cooling_rate:.1f} W:",
                (
                    f"  continuous I={item.continuous_current:.4f} A, "
                    f"COP={item.continuous_cop:.4f}, "
                    f"steady-map mismatch={item.steady_map_cop_error_percent:+.3f}%"
                ),
                (
                    f"  pulse peak/mean/rms={item.pulse_peak_current:.4f}/"
                    f"{item.pulse_mean_current:.4f}/{item.pulse_rms_current:.4f} A, "
                    f"period={item.pulse_period:.1f} s, "
                    f"duty={item.pulse_duty_cycle:.2f}"
                ),
                (
                    f"  pulse COP={item.pulse_cop:.4f}, "
                    f"COP change={item.pulse_cop_change_percent:+.2f}%, "
                    f"equal-power cooling change="
                    f"{item.pulse_cooling_change_at_equal_power_percent:+.2f}%"
                ),
            )
        )
    return "\n".join(lines)


def main() -> None:
    print(format_pulse_operating_map_report(run_pulse_operating_map()))


if __name__ == "__main__":
    main()
