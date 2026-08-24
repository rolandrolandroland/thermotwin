"""Generate the steady cooling/heating COP operating-map figure."""

import argparse
from pathlib import Path
from typing import Optional, Sequence, Union

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..design.operating_map import (
    COPOperatingMapResult,
    format_cop_operating_map_report,
    points_for,
    run_cop_operating_map,
)
from .paths import default_figure_path


DEFAULT_COP_OPERATING_MAP_PATH = default_figure_path(
    "cop_operating_map.png"
)


def _values(points, field: str):
    return tuple(getattr(point, field) for point in points)


def _plot_defined(axis, x_values, y_values, **kwargs) -> None:
    pairs = tuple(
        (x_value, y_value)
        for x_value, y_value in zip(x_values, y_values)
        if y_value is not None
    )
    if pairs:
        axis.plot(
            tuple(pair[0] for pair in pairs),
            tuple(pair[1] for pair in pairs),
            **kwargs,
        )


def _useful_cop_values(result, points, mode: str):
    heat_field = (
        "delivered_cooling_rate" if mode == "cooling" else "delivered_heating_rate"
    )
    cop_field = "cooling_cop" if mode == "cooling" else "heating_cop"
    return tuple(
        getattr(point, cop_field)
        if getattr(point, heat_field) >= result.config.minimum_useful_heat_rate
        else None
        for point in points
    )


def save_cop_operating_map_report(
    result: COPOperatingMapResult,
    output_path: Union[str, Path],
) -> Path:
    """Save a six-panel decision map and return its resolved path."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(15.0, 10.0), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3)
    baseline_resistance = min(
        result.config.symmetric_contact_resistances,
        key=lambda value: abs(value - 0.25),
    )
    selected_lifts = tuple(
        lift
        for lift in (0.0, 10.0, 20.0, 30.0)
        if lift in result.config.external_temperature_lifts
    )
    colors = ("tab:blue", "tab:green", "tab:orange", "tab:red")

    for lift, color in zip(selected_lifts, colors):
        reduced = points_for(
            result,
            topology="reduced_no_explicit_contact",
            external_temperature_lift=lift,
        )
        contact = points_for(
            result,
            topology="four_node_contact",
            external_temperature_lift=lift,
            symmetric_contact_resistance=baseline_resistance,
        )
        for axis, field, mode in (
            (axes[0, 0], "cooling_cop", "cooling"),
            (axes[0, 1], "delivered_cooling_rate", None),
            (axes[0, 2], "heating_cop", "heating"),
        ):
            reduced_values = (
                _useful_cop_values(result, reduced, mode)
                if mode is not None
                else _values(reduced, field)
            )
            contact_values = (
                _useful_cop_values(result, contact, mode)
                if mode is not None
                else _values(contact, field)
            )
            _plot_defined(
                axis,
                _values(reduced, "current"),
                reduced_values,
                color=color,
                linestyle="--",
                alpha=0.75,
            )
            _plot_defined(
                axis,
                _values(contact, "current"),
                contact_values,
                color=color,
                label=f"{lift:.0f} K",
            )

    axes[0, 0].set_title("Cooling COP: contacts vs reduced model")
    axes[0, 1].set_title("Delivered cooling capacity")
    axes[0, 2].set_title("Heating COP: contacts vs reduced model")
    axes[0, 0].set_ylabel("Cooling COP")
    axes[0, 1].set_ylabel("Cooling rate (W)")
    axes[0, 2].set_ylabel("Heating COP")
    for axis in axes[0]:
        axis.set_xlabel("Current (A)")
        axis.axhline(0.0, color="0.75", linewidth=0.8)
    axes[0, 0].legend(title="External lift", fontsize="small")
    axes[0, 2].text(
        0.98,
        0.95,
        "COP curves require at least 1 W useful heat",
        transform=axes[0, 2].transAxes,
        ha="right",
        va="top",
        fontsize="small",
    )
    axes[0, 1].text(
        0.03,
        0.04,
        "solid: 0.25 K/W per contact\ndashed: no explicit contacts",
        transform=axes[0, 1].transAxes,
        fontsize="small",
    )

    topology_styles = (("reduced_no_explicit_contact", None, "No explicit contacts"),) + tuple(
        ("four_node_contact", resistance, f"{resistance:.2f} K/W each")
        for resistance in result.config.symmetric_contact_resistances
    )
    for topology, resistance, label in topology_styles:
        for mode, axis in (("cooling", axes[1, 0]), ("heating", axes[1, 1])):
            summaries = tuple(
                item
                for item in result.optima
                if item.topology == topology
                and item.symmetric_contact_resistance == resistance
                and item.mode == mode
            )
            axis.plot(
                _values(summaries, "external_temperature_lift"),
                _values(summaries, "maximum_cop"),
                marker="o",
                label=label,
            )
    axes[1, 0].set_title("Best useful cooling COP")
    axes[1, 1].set_title("Best useful heating COP")
    axes[1, 0].set_ylabel("Maximum cooling COP")
    axes[1, 1].set_ylabel("Maximum heating COP")
    for axis in (axes[1, 0], axes[1, 1]):
        axis.set_xlabel("External temperature lift (K)")
        axis.legend(fontsize="small")

    penalty_axis = axes[1, 2]
    for resistance in result.config.symmetric_contact_resistances:
        cooling = tuple(
            item
            for item in result.equal_load_comparisons
            if item.mode == "cooling"
            and item.symmetric_contact_resistance == resistance
            and item.feasible
        )
        heating = tuple(
            item
            for item in result.equal_load_comparisons
            if item.mode == "heating"
            and item.symmetric_contact_resistance == resistance
            and item.feasible
        )
        penalty_axis.plot(
            _values(cooling, "external_temperature_lift"),
            _values(cooling, "contact_cop_penalty_percent"),
            marker="o",
            label=f"cool {resistance:.2f} K/W",
        )
        penalty_axis.plot(
            _values(heating, "external_temperature_lift"),
            _values(heating, "contact_cop_penalty_percent"),
            marker="x",
            linestyle="--",
            label=f"heat {resistance:.2f} K/W",
        )
    penalty_axis.axhline(0.0, color="0.3", linewidth=0.8)
    penalty_axis.set_title("Contact penalty at equal useful heat")
    penalty_axis.set_xlabel("External temperature lift (K)")
    penalty_axis.set_ylabel("COP change from reduced model (%)")
    penalty_axis.legend(fontsize="x-small", ncol=2)

    figure.suptitle(
        "ThermoTwin steady operating map\n"
        "generic parameters; maximum COP requires at least 1 W useful heat",
        fontsize=14,
    )
    figure.savefig(destination, dpi=150)
    return destination


def build_and_save_cop_operating_map_report(
    output_path: Union[str, Path] = DEFAULT_COP_OPERATING_MAP_PATH,
) -> tuple[COPOperatingMapResult, Path]:
    result = run_cop_operating_map()
    return result, save_cop_operating_map_report(result, output_path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_COP_OPERATING_MAP_PATH),
        help="PNG output path",
    )
    arguments = parser.parse_args(argv)
    result, destination = build_and_save_cop_operating_map_report(arguments.output)
    print(format_cop_operating_map_report(result))
    print(f"report: {destination}")


if __name__ == "__main__":
    main()
