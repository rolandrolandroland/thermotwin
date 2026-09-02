"""Dedicated figure report for continuous-versus-pulsed control."""

import argparse
from pathlib import Path
from typing import Union

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..design.control_comparison import (
    ControlComparisonConfig,
    ControlComparisonResult,
    format_control_comparison_report,
    run_control_comparison,
)
from .paths import default_figure_path, save_figure_data


DEFAULT_CONTROL_COMPARISON_PATH = default_figure_path(
    "control_comparison.png", "CONTROL_COMPARISON_EXPERIMENT.md"
)


def save_control_comparison_report(
    result: ControlComparisonResult,
    output_path: Union[str, Path],
) -> Path:
    """Save the capacity, duty, equal-power, and robustness comparisons."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(13.0, 9.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)

    targets = tuple(item.target_cooling_rate for item in result.comparisons)
    continuous_cop = tuple(
        item.continuous.delivered_cooling_cop for item in result.comparisons
    )
    pulsed_cop = tuple(
        item.best_pulsed.delivered_cooling_cop for item in result.comparisons
    )
    axis = axes[0, 0]
    axis.plot(targets, continuous_cop, "o-", linewidth=2.0, label="Continuous")
    axis.plot(targets, pulsed_cop, "s-", linewidth=2.0, label="Best tested pulse")
    axis.set_xlabel("Delivered cooling target (W)")
    axis.set_ylabel("Delivered cooling COP")
    axis.set_title("Matched-cooling performance")
    axis.legend()

    central = min(
        result.comparisons,
        key=lambda item: abs(item.target_cooling_rate - 5.0),
    )
    best_by_duty = {}
    for candidate in central.pulsed_candidates:
        incumbent = best_by_duty.get(candidate.duty_cycle)
        if incumbent is None or (
            candidate.delivered_cooling_cop > incumbent.delivered_cooling_cop
        ):
            best_by_duty[candidate.duty_cycle] = candidate
    duties = tuple(sorted(best_by_duty))
    duty_changes = tuple(
        100.0
        * (
            best_by_duty[duty].delivered_cooling_cop
            / central.continuous.delivered_cooling_cop
            - 1.0
        )
        for duty in duties
    )
    axis = axes[0, 1]
    axis.plot(tuple(100.0 * duty for duty in duties), duty_changes, "o-")
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Pulse duty cycle (%)")
    axis.set_ylabel("COP change from continuous (%)")
    axis.set_title(f"Duty law near {central.target_cooling_rate:g} W")

    axis = axes[1, 0]
    equal_power_changes = tuple(
        item.pulsed_cooling_change_at_equal_power_percent
        for item in result.comparisons
    )
    axis.bar(tuple(str(value) for value in targets), equal_power_changes)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Pulse cooling target (W)")
    axis.set_ylabel("Pulse cooling change at equal power (%)")
    axis.set_title("Reverse fairness check")

    axis = axes[1, 1]
    resistance = tuple(
        item.cold_contact_resistance for item in result.uncertainty_cases
    )
    uncertainty_change = tuple(
        item.pulsed_cop_change_percent for item in result.uncertainty_cases
    )
    axis.plot(resistance, uncertainty_change, "o-", color="tab:purple")
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Cold contact resistance (K/W)")
    axis.set_ylabel("Pulse COP change (%)")
    axis.set_title("Interface-resistance stress test")

    for axis in axes.reshape(-1):
        axis.grid(True, alpha=0.25)
    figure.suptitle(
        "Continuous versus pulsed current at fair matched conditions\n"
        "Synthetic four-node reference model",
        fontsize=15,
    )
    figure.savefig(destination, dpi=170)
    save_figure_data(result, destination)
    return destination


def run_and_save_control_comparison_report(
    output_path: Union[str, Path] = DEFAULT_CONTROL_COMPARISON_PATH,
) -> tuple[ControlComparisonResult, Path]:
    result = run_control_comparison()
    return result, save_control_comparison_report(result, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the continuous-versus-pulsed control figure"
    )
    parser.add_argument("--output", default=str(DEFAULT_CONTROL_COMPARISON_PATH))
    arguments = parser.parse_args()
    result, destination = run_and_save_control_comparison_report(arguments.output)
    print(format_control_comparison_report(result))
    print(f"\nfigure: {destination}")


if __name__ == "__main__":
    main()
