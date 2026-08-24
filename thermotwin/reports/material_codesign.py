"""Run and plot the public-data-seeded material/geometry co-design campaign."""

import argparse
from pathlib import Path
from typing import Optional, Sequence, Union

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .paths import default_figure_path
from ..design.materials import N_TYPE_SAMPLES, P_TYPE_SAMPLES
from ..design.codesign import (
    CodesignCampaignConfig,
    CodesignCampaignResult,
    format_codesign_campaign_report,
    run_codesign_campaign,
)


DEFAULT_MATERIAL_GEOMETRY_CODESIGN_PATH = default_figure_path(
    "material_geometry_bayesian_codesign.png"
)


def save_material_geometry_codesign_report(
    result: CodesignCampaignResult,
    output_path: Union[str, Path],
) -> Path:
    """Save a nine-panel campaign audit figure."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(16.0, 13.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 3)

    material_axis = axes[0, 0]
    for samples, color, marker, label in (
        (P_TYPE_SAMPLES, "tab:red", "o", "p-type"),
        (N_TYPE_SAMPLES, "tab:blue", "s", "n-type"),
    ):
        material_axis.scatter(
            tuple(sample.power_factor * 1e3 for sample in samples),
            tuple(sample.thermal_conductivity for sample in samples),
            color=color,
            marker=marker,
            label=label,
        )
        for sample in samples:
            material_axis.annotate(
                str(sample.sample_id),
                (sample.power_factor * 1e3, sample.thermal_conductivity),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
    material_axis.set_title("Curated same-row records at 300 K")
    material_axis.set_xlabel("Power factor (mW m$^{-1}$ K$^{-2}$)")
    material_axis.set_ylabel("Thermal conductivity (W m$^{-1}$ K$^{-1}$)")
    material_axis.legend(fontsize="small")

    design_axis = axes[0, 1]
    initial = result.initial_designs
    scatter = design_axis.scatter(
        tuple(design.geometry.leg_length * 1e3 for design in initial),
        tuple(design.geometry.leg_area * 1e6 for design in initial),
        s=tuple(design.geometry.couple_count * 0.8 for design in initial),
        c=tuple(design.symmetric_contact_resistance for design in initial),
        cmap="viridis_r",
        edgecolors="0.2",
        linewidths=0.4,
    )
    design_axis.set_title(f"Space-filling initial screen (n={len(initial)})")
    design_axis.set_xlabel("Leg length (mm)")
    design_axis.set_ylabel("Leg area (mm$^2$)")
    figure.colorbar(scatter, ax=design_axis, label="Contact resistance (K/W)")

    screen_axis = axes[0, 2]
    first_summary = result.initial_summaries[0]
    for point in first_summary.evaluations:
        screen_axis.scatter(
            point.prototype_cost_index,
            point.wall_cooling_cop or 0.0,
            color="tab:green" if point.feasible else "0.65",
            marker="o" if point.feasible else "x",
        )
    screen_axis.set_title(
        f"Initial 10 K screen: {first_summary.feasible_count}/"
        f"{len(first_summary.evaluations)} feasible"
    )
    screen_axis.set_xlabel("Relative prototype cost index")
    screen_axis.set_ylabel("Wall-plug cooling COP")

    for axis, optimization in zip(axes[1], result.bayesian_results):
        x_values = tuple(range(len(optimization.best_utility_history)))
        axis.fill_between(
            x_values,
            optimization.random_lower_history,
            optimization.random_upper_history,
            color="0.85",
            label="Random 10--90%",
        )
        axis.plot(
            x_values,
            optimization.random_median_history,
            color="0.45",
            linestyle="--",
            label="Random median",
        )
        axis.plot(
            x_values,
            optimization.best_utility_history,
            color="tab:purple",
            marker="o",
            label="Cost-aware BO",
        )
        axis.axhline(
            optimization.oracle_best.utility,
            color="tab:orange",
            linestyle=":",
            label="Pool oracle",
        )
        axis.set_title(optimization.application.label)
        axis.set_xlabel("Additional virtual prototypes")
        axis.set_ylabel("Best application utility")
        axis.legend(fontsize="x-small")
        selected = optimization.selected
        axis.text(
            0.02,
            0.04,
            (
                f"selected peak J: {selected.peak_current_density / 1.0e6:.3f} A/mm$^2$\n"
                f"{100.0 * selected.current_density_utilization:.1f}% of limit; "
                f"binding: {'yes' if selected.current_density_constraint_binding else 'no'}"
            ),
            transform=axis.transAxes,
            fontsize=8,
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
        )

    labels = tuple(item.application.label.replace(" ", "\n", 1) for item in result.robustness_results)
    x_values = tuple(range(len(labels)))
    cooling_axis = axes[2, 0]
    for x_value, robustness in zip(x_values, result.robustness_results):
        q05, q50, q95 = robustness.cooling_rate_quantiles
        cooling_axis.errorbar(
            x_value,
            q50,
            yerr=[[q50 - q05], [q95 - q50]],
            fmt="o",
            capsize=5,
            color="tab:blue",
        )
        cooling_axis.scatter(
            x_value,
            robustness.nominal.delivered_cooling_rate,
            marker="x",
            color="black",
            zorder=3,
        )
    cooling_axis.set_title("Fixed-current robustness: cooling")
    cooling_axis.set_ylabel("Delivered cooling (W), 5/50/95%")
    cooling_axis.set_xticks(x_values, labels, fontsize=8)

    cop_axis = axes[2, 1]
    for x_value, robustness in zip(x_values, result.robustness_results):
        c05, c50, c95 = robustness.wall_cop_quantiles
        cop_axis.errorbar(
            x_value,
            c50,
            yerr=[[c50 - c05], [c95 - c50]],
            fmt="o",
            capsize=5,
            color="tab:green",
        )
        cop_axis.scatter(
            x_value,
            robustness.nominal.wall_cooling_cop,
            marker="x",
            color="black",
            zorder=3,
        )
    cop_axis.set_title("Fixed-current robustness: wall COP")
    cop_axis.set_ylabel("Cooling COP, 5/50/95%")
    cop_axis.set_xticks(x_values, labels, fontsize=8)

    feasibility_axis = axes[2, 2]
    feasibility_axis.bar(
        x_values,
        tuple(100.0 * item.feasible_fraction for item in result.robustness_results),
        color=("tab:green", "tab:orange", "tab:blue"),
    )
    feasibility_axis.axhline(95.0, color="0.3", linestyle="--", linewidth=0.9)
    feasibility_axis.set_ylim(0.0, 105.0)
    feasibility_axis.set_title("As-built requirement pass rate")
    feasibility_axis.set_ylabel("Feasible trials (%)")
    feasibility_axis.set_xticks(x_values, labels, fontsize=8)
    for x_value, robustness in zip(x_values, result.robustness_results):
        feasibility_axis.text(
            x_value,
            100.0 * robustness.feasible_fraction + 2.0,
            f"{100 * robustness.feasible_fraction:.1f}%",
            ha="center",
            fontsize=8,
        )

    figure.suptitle(
        "ThermoTwin public-data-seeded material/geometry Bayesian co-design\n"
        "real same-row material records; explicit areal electrical contacts; "
        "virtual geometry, cost, and uncertainty",
        fontsize=15,
    )
    figure.savefig(destination, dpi=150)
    return destination


def build_and_save_material_geometry_codesign_report(
    output_path: Union[str, Path] = DEFAULT_MATERIAL_GEOMETRY_CODESIGN_PATH,
    *,
    config: CodesignCampaignConfig = CodesignCampaignConfig(),
) -> tuple[CodesignCampaignResult, Path]:
    result = run_codesign_campaign(config)
    return result, save_material_geometry_codesign_report(result, output_path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_MATERIAL_GEOMETRY_CODESIGN_PATH),
    )
    arguments = parser.parse_args(argv)
    result, destination = build_and_save_material_geometry_codesign_report(
        arguments.output
    )
    print(format_codesign_campaign_report(result))
    print(f"report: {destination}")


if __name__ == "__main__":
    main()
