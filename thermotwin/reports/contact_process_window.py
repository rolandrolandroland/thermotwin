"""Run and plot the electrical-contact process-window experiment."""

import argparse
import math
from pathlib import Path
from typing import Optional, Sequence, Union

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.colors import BoundaryNorm, ListedColormap

from .paths import default_figure_path
from ..design.contact_process_window import (
    ContactProcessWindowConfig,
    ContactProcessWindowResult,
    format_contact_process_window_report,
    maximum_feasible_contact_resistivity,
    run_contact_process_window,
)
from ..design.literature_materials import PUBLISHED_AG2SE_UNICOUPLE


DEFAULT_CONTACT_PROCESS_WINDOW_PATH = default_figure_path(
    "electrical_contact_process_window.png"
)


def _point_map(result, *, pair_key, application_name, density_limit):
    return {
        (point.leg_length, point.specific_contact_resistivity): point
        for point in result.points
        if point.pair_key == pair_key
        and point.application_name == application_name
        and point.maximum_current_density == density_limit
    }


def save_contact_process_window_report(
    result: ContactProcessWindowResult,
    output_path: Union[str, Path],
) -> Path:
    """Save the paper landmark, contact map, process limits, and COP map."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(19.0, 10.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3)
    paper = PUBLISHED_AG2SE_UNICOUPLE
    nonzero_paper = tuple(
        point
        for point in result.published_electrical_sweep
        if point.specific_contact_resistivity > 0.0
    )

    paper_axis = axes[0, 0]
    paper_axis.plot(
        tuple(point.specific_contact_resistivity for point in nonzero_paper),
        tuple(100.0 * point.electrical_contact_fraction for point in nonzero_paper),
        label="electrical contact share",
        color="tab:red",
    )
    paper_axis.plot(
        tuple(point.specific_contact_resistivity for point in nonzero_paper),
        tuple(100.0 * point.normalized_zt_retention for point in nonzero_paper),
        label="normalized ZT retention",
        color="tab:blue",
    )
    paper_axis.axvline(
        paper.half_contact_share_resistivity,
        color="0.25",
        linestyle="--",
        label="analytical 50% crossover",
    )
    paper_axis.axvline(
        paper.inferred_specific_contact_resistivity,
        color="tab:purple",
        linestyle=":",
        label="approx. RcA from paper",
    )
    paper_axis.set_xscale("log")
    paper_axis.set_ylim(0.0, 100.0)
    paper_axis.set_title("Published 1.5 mm unicouple: electrical-only translation")
    paper_axis.set_xlabel("Specific contact resistivity (ohm m$^2$)")
    paper_axis.set_ylabel("Share or retention (%)")
    paper_axis.legend(fontsize=8)
    paper_axis.grid(True, which="both", alpha=0.25)

    pair_key = "ag2se_p_9107"
    application_name = "high_lift_balanced"
    density_limit = result.config.current_density_limits[0]
    lengths = result.config.leg_lengths
    rhos = tuple(value for value in result.config.specific_contact_resistivities if value > 0)
    point_map = _point_map(
        result,
        pair_key=pair_key,
        application_name=application_name,
        density_limit=density_limit,
    )

    share_axis = axes[0, 1]
    share_values = tuple(
        tuple(
            100.0 * point_map[(length, rho)].electrical_contact_fraction
            for length in lengths
        )
        for rho in rhos
    )
    share_mesh = share_axis.pcolormesh(
        tuple(value * 1e3 for value in lengths),
        rhos,
        share_values,
        shading="auto",
        cmap="magma",
        vmin=0.0,
        vmax=100.0,
    )
    share_axis.set_xscale("log")
    share_axis.set_yscale("log")
    share_axis.axvline(paper.leg_length * 1e3, color="cyan", linestyle="--")
    share_axis.set_title("Ag2Se + p 9107 electrical contact share")
    share_axis.set_xlabel("Leg length (mm)")
    share_axis.set_ylabel("Specific contact resistivity (ohm m$^2$)")
    figure.colorbar(share_mesh, ax=share_axis, label="Contact share (%)")

    zt_axis = axes[0, 2]
    zt_values = tuple(
        tuple(
            point_map[(length, rho)].device_zt_300k
            for length in lengths
        )
        for rho in rhos
    )
    zt_mesh = zt_axis.pcolormesh(
        tuple(value * 1e3 for value in lengths),
        rhos,
        zt_values,
        shading="auto",
        cmap="cividis",
    )
    zt_axis.set_xscale("log")
    zt_axis.set_yscale("log")
    zt_axis.axvline(paper.leg_length * 1e3, color="white", linestyle="--")
    zt_axis.set_title("Ag2Se + p 9107 device ZT at 300 K")
    zt_axis.set_xlabel("Leg length (mm)")
    zt_axis.set_ylabel("Specific contact resistivity (ohm m$^2$)")
    figure.colorbar(zt_mesh, ax=zt_axis, label="Device ZT")

    limit_axis = axes[1, 0]
    colors = ("tab:green", "tab:orange", "tab:blue")
    exploratory_density = max(result.config.current_density_limits)
    for application, color in zip(result.config.applications, colors):
        for density in result.config.current_density_limits:
            style = "--" if density == exploratory_density else "-"
            maximum_values = []
            for length in lengths:
                selected = tuple(
                    point
                    for point in result.points
                    if point.pair_key == pair_key
                    and point.application_name == application.name
                    and point.maximum_current_density == density
                    and point.leg_length == length
                )
                maximum = maximum_feasible_contact_resistivity(selected)
                maximum_values.append(maximum if maximum is not None and maximum > 0 else math.nan)
            limit_axis.plot(
                tuple(value * 1e3 for value in lengths),
                tuple(maximum_values),
                color=color,
                linestyle=style,
                label=(
                    f"{application.label}; {density / 1e6:.0f} A/mm$^2$"
                    + (
                        " exploratory sensitivity"
                        if density == exploratory_density
                        else " existing campaign constraint"
                    )
                ),
            )
    limit_axis.set_xscale("log")
    limit_axis.set_yscale("log")
    limit_axis.axvline(paper.leg_length * 1e3, color="0.25", linestyle=":")
    limit_axis.set_title("Largest rho_c meeting application requirements")
    limit_axis.set_xlabel("Leg length (mm)")
    limit_axis.set_ylabel("Maximum feasible rho_c (ohm m$^2$)")
    limit_axis.legend(fontsize=7)
    limit_axis.grid(True, which="both", alpha=0.25)

    exploratory_point_map = _point_map(
        result,
        pair_key=pair_key,
        application_name=application_name,
        density_limit=exploratory_density,
    )
    cop_axis = axes[1, 1]
    cop_values = tuple(
        tuple(
            (
                exploratory_point_map[(length, rho)].wall_cooling_cop
                if exploratory_point_map[(length, rho)].feasible
                else math.nan
            )
            for length in lengths
        )
        for rho in rhos
    )
    cop_mesh = cop_axis.pcolormesh(
        tuple(value * 1e3 for value in lengths),
        rhos,
        cop_values,
        shading="auto",
        cmap="viridis",
    )
    cop_axis.set_xscale("log")
    cop_axis.set_yscale("log")
    cop_axis.axvline(paper.leg_length * 1e3, color="white", linestyle="--")
    cop_axis.set_title("25 K feasible cooling COP; exploratory 3 A/mm$^2$")
    cop_axis.set_xlabel("Leg length (mm)")
    cop_axis.set_ylabel("Specific contact resistivity (ohm m$^2$)")
    figure.colorbar(cop_mesh, ax=cop_axis, label="Delivered wall COP")

    status_axis = axes[1, 2]
    status_codes = []
    for rho in rhos:
        row = []
        for length in lengths:
            point = exploratory_point_map[(length, rho)]
            if point.feasible:
                code = 0
            elif (
                "cooling_target_current_density_limited"
                in point.infeasibility_reasons
            ):
                code = 1
            elif "cooling_target_physics_limited" in point.infeasibility_reasons:
                code = 2
            elif "cop_target" in point.infeasibility_reasons:
                code = 3
            elif "supply_power_limit" in point.infeasibility_reasons:
                code = 4
            elif "voltage_limit" in point.infeasibility_reasons:
                code = 5
            else:
                code = 6
            row.append(code)
        status_codes.append(tuple(row))
    status_labels = (
        "feasible",
        "cooling target: current-density cap",
        "cooling target: physical maximum",
        "COP limit",
        "power limit",
        "voltage limit",
        "other constraint",
    )
    status_cmap = ListedColormap(
        (
            "#2ca02c",
            "#ff7f0e",
            "#d62728",
            "#ffbf00",
            "#9467bd",
            "#1f77b4",
            "#7f7f7f",
        )
    )
    status_norm = BoundaryNorm(tuple(value - 0.5 for value in range(8)), 7)
    status_mesh = status_axis.pcolormesh(
        tuple(value * 1e3 for value in lengths),
        rhos,
        tuple(status_codes),
        shading="auto",
        cmap=status_cmap,
        norm=status_norm,
    )
    status_axis.set_xscale("log")
    status_axis.set_yscale("log")
    status_axis.axvline(paper.leg_length * 1e3, color="white", linestyle="--")
    status_axis.set_title("25 K limiting status; exploratory 3 A/mm$^2$")
    status_axis.set_xlabel("Leg length (mm)")
    status_axis.set_ylabel("Specific contact resistivity (ohm m$^2$)")
    status_bar = figure.colorbar(
        status_mesh,
        ax=status_axis,
        ticks=tuple(range(len(status_labels))),
    )
    status_bar.ax.set_yticklabels(status_labels)

    figure.suptitle(
        "ThermoTwin electrical-contact process window\n"
        "paper landmark: L=1.5 mm, A=2.25 mm2 unicouple; "
        "system maps: N=120, A=1.6 mm2 cooling assumptions",
        fontsize=14,
    )
    figure.savefig(destination, dpi=150)
    return destination


def build_and_save_contact_process_window_report(
    output_path: Union[str, Path] = DEFAULT_CONTACT_PROCESS_WINDOW_PATH,
    *,
    config: ContactProcessWindowConfig = ContactProcessWindowConfig(),
) -> tuple[ContactProcessWindowResult, Path]:
    result = run_contact_process_window(config)
    return result, save_contact_process_window_report(result, output_path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_CONTACT_PROCESS_WINDOW_PATH))
    arguments = parser.parse_args(argv)
    result, destination = build_and_save_contact_process_window_report(arguments.output)
    print(format_contact_process_window_report(result))
    print(f"report: {destination}")


if __name__ == "__main__":
    main()
