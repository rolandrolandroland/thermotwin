"""Generate the averaged direct-versus-smoothed PWM report figure."""

import argparse
from pathlib import Path
from typing import Optional, Sequence, Union

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .paths import default_figure_path
from ..design.power_electronics import (
    PWMPowerElectronicsResult,
    format_pwm_power_electronics_report,
    pwm_points_for,
    run_pwm_power_electronics_experiment,
)


DEFAULT_PWM_POWER_ELECTRONICS_PATH = default_figure_path(
    "pwm_power_electronics.png"
)


def _defined_xy(points, field: str):
    pairs = tuple(
        (point.current.mean_current, getattr(point, field))
        for point in points
        if getattr(point, field) is not None
    )
    return (
        tuple(pair[0] for pair in pairs),
        tuple(pair[1] for pair in pairs),
    )


def save_pwm_power_electronics_report(
    result: PWMPowerElectronicsResult,
    output_path: Union[str, Path],
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(15.0, 9.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3)
    styles = {
        "ideal_dc": ("tab:blue", "Ideal DC"),
        "smoothed_pwm": ("tab:green", "Smoothed PWM current"),
        "direct_pwm": ("tab:orange", "Direct current PWM"),
    }
    lift_styles = ("-", "--", ":")
    for lift, linestyle in zip(
        result.config.external_temperature_lifts,
        lift_styles,
    ):
        for mode, (color, label) in styles.items():
            points = pwm_points_for(
                result,
                mode=mode,
                external_temperature_lift=lift,
            )
            for axis, field in (
                (axes[0, 0], "wall_cooling_cop"),
                (axes[0, 1], "delivered_cooling_rate"),
                (axes[0, 2], "wall_heating_cop"),
            ):
                x_values, y_values = _defined_xy(points, field)
                axis.plot(
                    x_values,
                    y_values,
                    color=color,
                    linestyle=linestyle,
                    label=f"{label}, {lift:.0f} K" if axis is axes[0, 0] else None,
                )
    axes[0, 0].set_title("Wall-plug cooling COP")
    axes[0, 1].set_title("Delivered cooling")
    axes[0, 2].set_title("Wall-plug heating COP")
    axes[0, 0].set_ylabel("Cooling COP")
    axes[0, 1].set_ylabel("Cooling rate (W)")
    axes[0, 2].set_ylabel("Heating COP")
    for axis in axes[0]:
        axis.set_xlabel("Mean TEC current (A)")
        axis.axhline(0.0, color="0.75", linewidth=0.8)
    axes[0, 0].legend(fontsize="x-small", ncol=2)

    representative_lift = min(
        result.config.external_temperature_lifts,
        key=lambda value: abs(value - 10.0),
    )
    for mode, (color, label) in styles.items():
        points = pwm_points_for(
            result,
            mode=mode,
            external_temperature_lift=representative_lift,
        )
        axes[1, 0].plot(
            tuple(point.current.mean_current for point in points),
            tuple(point.current.joule_multiplier_over_dc for point in points),
            color=color,
            marker="o",
            label=label,
        )
        axes[1, 2].plot(
            tuple(point.module_electrical_power for point in points),
            tuple(point.supply_electrical_power for point in points),
            color=color,
            marker="o",
            label=label,
        )
    axes[1, 0].set_title("Joule heat relative to ideal DC")
    axes[1, 0].set_xlabel("Mean TEC current (A)")
    axes[1, 0].set_ylabel("Mean-square-current multiplier")
    axes[1, 0].legend(fontsize="small")

    ideal_points = pwm_points_for(
        result,
        mode="ideal_dc",
        external_temperature_lift=representative_lift,
    )
    ideal_by_current = {
        point.current.mean_current: point for point in ideal_points
    }
    for mode, color, label in (
        ("smoothed_pwm", "tab:green", "Smoothed PWM"),
        ("direct_pwm", "tab:orange", "Direct PWM"),
    ):
        points = pwm_points_for(
            result,
            mode=mode,
            external_temperature_lift=representative_lift,
        )
        comparable = tuple(
            point
            for point in points
            if point.wall_cooling_cop is not None
            and ideal_by_current[point.current.mean_current].wall_cooling_cop is not None
        )
        axes[1, 1].plot(
            tuple(point.current.mean_current for point in comparable),
            tuple(
                100.0
                * (
                    point.wall_cooling_cop
                    / ideal_by_current[point.current.mean_current].wall_cooling_cop
                    - 1.0
                )
                for point in comparable
            ),
            marker="o",
            color=color,
            label=label,
        )
    axes[1, 1].axhline(0.0, color="0.3", linewidth=0.8)
    axes[1, 1].set_title(f"Cooling COP penalty at {representative_lift:.0f} K lift")
    axes[1, 1].set_xlabel("Mean TEC current (A)")
    axes[1, 1].set_ylabel("Wall COP change from ideal DC (%)")
    axes[1, 1].legend(fontsize="small")

    axes[1, 2].set_title("Converter input versus module terminal power")
    axes[1, 2].set_xlabel("Module terminal power (W)")
    axes[1, 2].set_ylabel("Supply power (W)")
    axes[1, 2].legend(fontsize="small")
    figure.suptitle(
        "ThermoTwin power-electronics-aware PWM layer\n"
        "electrical switching is averaged; thermal integration is not run at switching resolution",
        fontsize=14,
    )
    figure.savefig(destination, dpi=150)
    return destination


def build_and_save_pwm_power_electronics_report(
    output_path: Union[str, Path] = DEFAULT_PWM_POWER_ELECTRONICS_PATH,
) -> tuple[PWMPowerElectronicsResult, Path]:
    result = run_pwm_power_electronics_experiment()
    return result, save_pwm_power_electronics_report(result, output_path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_PWM_POWER_ELECTRONICS_PATH))
    arguments = parser.parse_args(argv)
    result, destination = build_and_save_pwm_power_electronics_report(arguments.output)
    print(format_pwm_power_electronics_report(result))
    print(f"report: {destination}")


if __name__ == "__main__":
    main()
