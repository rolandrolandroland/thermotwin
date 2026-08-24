"""Generate the pulse-to-steady-operating-map comparison figure."""

import argparse
from pathlib import Path
from typing import Optional, Sequence, Union

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .paths import default_figure_path
from ..design.pulse_map import (
    PulseOperatingMapResult,
    format_pulse_operating_map_report,
    run_pulse_operating_map,
)


DEFAULT_PULSE_OPERATING_MAP_PATH = default_figure_path(
    "pulse_operating_map.png"
)


def save_pulse_operating_map_report(
    result: PulseOperatingMapResult,
    output_path: Union[str, Path],
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(12.0, 8.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)

    steady_cooling = tuple(
        point.delivered_cooling_rate
        for point in result.steady_curve
        if point.cooling_cop is not None
    )
    steady_cop = tuple(
        point.cooling_cop
        for point in result.steady_curve
        if point.cooling_cop is not None
    )
    targets = tuple(item.target_cooling_rate for item in result.comparisons)
    axes[0, 0].plot(
        steady_cooling,
        steady_cop,
        color="0.25",
        label="steady continuous envelope",
    )
    axes[0, 0].scatter(
        tuple(item.target_cooling_rate for item in result.comparisons),
        tuple(item.continuous_cop for item in result.comparisons),
        color="tab:blue",
        marker="o",
        s=55,
        label="transient continuous",
    )
    axes[0, 0].scatter(
        tuple(item.target_cooling_rate for item in result.comparisons),
        tuple(item.pulse_cop for item in result.comparisons),
        color="tab:orange",
        marker="X",
        s=75,
        label="highest-COP tested pulse",
    )
    axes[0, 0].set_title("Pulse points on the 0 K-lift COP envelope")
    axes[0, 0].set_xlabel("Delivered cooling (W)")
    axes[0, 0].set_ylabel("Delivered cooling COP")
    axes[0, 0].legend(fontsize="small")

    current_axis = axes[0, 1]
    current_axis.plot(
        targets,
        tuple(item.continuous_current for item in result.comparisons),
        marker="o",
        label="continuous current",
    )
    current_axis.plot(
        targets,
        tuple(item.pulse_mean_current for item in result.comparisons),
        marker="s",
        label="pulse mean current",
    )
    current_axis.plot(
        targets,
        tuple(item.pulse_rms_current for item in result.comparisons),
        marker="^",
        label="pulse RMS current",
    )
    current_axis.plot(
        targets,
        tuple(item.pulse_peak_current for item in result.comparisons),
        marker="x",
        linestyle="--",
        label="pulse peak current",
    )
    current_axis.set_title("Peltier follows mean I; Joule follows RMS I squared")
    current_axis.set_xlabel("Delivered cooling target (W)")
    current_axis.set_ylabel("Current (A)")
    current_axis.legend(fontsize="small")

    penalty_axis = axes[1, 0]
    for comparison in result.control_result.comparisons:
        best_by_duty = []
        duty_cycles = sorted(
            set(
                point.duty_cycle
                for point in comparison.pulsed_candidates
                if point.duty_cycle is not None
            )
        )
        for duty_cycle in duty_cycles:
            duty_points = tuple(
                point
                for point in comparison.pulsed_candidates
                if point.duty_cycle == duty_cycle
            )
            best = max(
                duty_points,
                key=lambda point: point.delivered_cooling_cop,
            )
            best_by_duty.append(best)
        penalty_axis.plot(
            duty_cycles,
            tuple(
                100.0
                * (
                    point.delivered_cooling_cop
                    / comparison.continuous.delivered_cooling_cop
                    - 1.0
                )
                for point in best_by_duty
            ),
            marker="o",
            label=f"{comparison.target_cooling_rate:.0f} W target",
        )
    penalty_axis.axhline(0.0, color="0.3", linewidth=0.8)
    penalty_axis.set_title("Duty-dependent COP penalty")
    penalty_axis.set_xlabel("Pulse duty cycle")
    penalty_axis.set_ylabel("COP change from continuous (%)")
    penalty_axis.legend(fontsize="small")

    drift_axis = axes[1, 1]
    drift_axis.plot(
        targets,
        tuple(abs(item.continuous_storage_drift) for item in result.comparisons),
        marker="o",
        label="continuous",
    )
    drift_axis.plot(
        targets,
        tuple(abs(item.pulse_storage_drift) for item in result.comparisons),
        marker="s",
        label="pulse",
    )
    drift_axis.axhline(0.05, color="tab:red", linestyle="--", label="settling limit")
    drift_axis.set_title("Stored-energy drift check")
    drift_axis.set_xlabel("Cooling target (W)")
    drift_axis.set_ylabel("Absolute mean drift (W)")
    drift_axis.legend(fontsize="small")

    figure.suptitle(
        "ThermoTwin seconds-scale pulse comparison\n"
        "Joule penalty approaches zero as duty approaches continuous operation",
        fontsize=14,
    )
    figure.savefig(destination, dpi=150)
    return destination


def build_and_save_pulse_operating_map_report(
    output_path: Union[str, Path] = DEFAULT_PULSE_OPERATING_MAP_PATH,
) -> tuple[PulseOperatingMapResult, Path]:
    result = run_pulse_operating_map()
    return result, save_pulse_operating_map_report(result, output_path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_PULSE_OPERATING_MAP_PATH))
    arguments = parser.parse_args(argv)
    result, destination = build_and_save_pulse_operating_map_report(arguments.output)
    print(format_pulse_operating_map_report(result))
    print(f"report: {destination}")


if __name__ == "__main__":
    main()
