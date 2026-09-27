"""Report complete nonlinear validation of next-experiment selection."""

import argparse
import math
from pathlib import Path
from statistics import fmean
from typing import Optional, Sequence

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from ..figure_paths import default_figure_path, save_figure_data
from ..inference.joint_thermal_parameters import JOINT_PARAMETER_NAMES
from ..studies.nonlinear_experiment_selection import (
    NonlinearExperimentSelectionConfig,
    NonlinearExperimentSelectionResult,
    run_nonlinear_experiment_selection_study,
)


DEFAULT_NONLINEAR_EXPERIMENT_SELECTION_PATH = default_figure_path(
    "nonlinear_experiment_selection.png", "NONLINEAR_EXPERIMENT_SELECTION.md"
)


def format_nonlinear_experiment_selection_report(
    result: NonlinearExperimentSelectionResult,
) -> str:
    """Return selection, nonlinear recovery, coverage, and profile evidence."""

    selected_spectrum = result.selected_identifiability
    zero_spectrum = result.zero_current_identifiability
    lines = [
        "Nonlinear validation of next-experiment selection",
        "=================================================",
        "",
        "Question:",
        "  Does the locally D-optimal pulse still improve a complete nonlinear,",
        "  bounded, multistart fit when truth and noise vary across paired trials?",
        "",
        "Released physical parameters:",
        *(f"  {name}" for name in JOINT_PARAMETER_NAMES),
        "  two constant exchanger-sensor biases are profiled as nuisance terms",
        "",
        "Candidate controls:",
    ]
    for definition in result.definitions:
        candidate = definition.candidate
        lines.append(
            f"  {definition.role}: {candidate.name}; {candidate.current_amplitude:.1f} A; "
            f"{candidate.pulse_duration:.0f} s; energy={candidate.electrical_energy:.4f} J; "
            f"local information={candidate.information_gain_nats:.4f} nats"
        )
    lines.extend(
        (
            "",
            "Prefit identifiability:",
            f"  selected singular values: "
            f"{tuple(round(value, 5) for value in selected_spectrum.singular_values)}",
            f"  selected supported rank: {selected_spectrum.supported_rank}/3; "
            f"condition number={selected_spectrum.condition_number:.4f}",
            f"  zero-current singular values: "
            f"{tuple(round(value, 5) for value in zero_spectrum.singular_values)}",
            f"  zero-current supported rank: {zero_spectrum.supported_rank}/3",
            "",
            f"Paired nonlinear study: {result.config.trial_count} trials",
        )
    )
    for summary in result.summaries:
        lines.append(
            f"  {summary.role}: mean log-RMSE={summary.mean_physical_log_rmse:.6f}; "
            f"median={summary.median_physical_log_rmse:.6f}; "
            f"worst={summary.worst_physical_log_rmse:.6f}; "
            f"individual 95% coverage={summary.individual_parameter_95_coverage:.1%}; "
            f"simultaneous 95% coverage={summary.simultaneous_physical_95_coverage:.1%}; "
            f"uncertainty volume={summary.mean_physical_uncertainty_volume:.6e}; "
            f"mean |corr(log R, log C)|="
            f"{summary.mean_absolute_resistance_capacitance_correlation:.4f}; "
            f"|corr(log R, log lag)|="
            f"{summary.mean_absolute_resistance_lag_correlation:.4f}; "
            f"|corr(log C, log lag)|="
            f"{summary.mean_absolute_capacitance_lag_correlation:.4f}; "
            f"withheld face RMSE=({summary.mean_withheld_cold_face_rmse:.6f}, "
            f"{summary.mean_withheld_hot_face_rmse:.6f}) K; "
            f"bound hits={summary.search_bound_hits}"
        )
    lines.extend(
        (
            "",
            "Measured improvements:",
            f"  selected mean log-RMSE reduction versus naive: "
            f"{result.selected_rmse_reduction_vs_naive_percent:.2f}%",
            f"  selected mean log-RMSE reduction versus closest-energy grid control: "
            f"{result.selected_rmse_reduction_vs_resource_control_percent:.2f}%",
            f"  selected local uncertainty-volume reduction versus naive: "
            f"{result.selected_interval_volume_reduction_vs_naive_percent:.2f}%",
            f"  selected local uncertainty-volume reduction versus closest-energy grid control: "
            f"{result.selected_interval_volume_reduction_vs_resource_control_percent:.2f}%",
            "",
            "Interpretation boundary:",
            "  The ranking is local at nominal parameters; the validation is nonlinear",
            "  and varies three physical truths, two biases, and observation noise.",
            "  All candidates use paired truths and paired random-noise sequences.",
            "  The closest-energy control is the feasible point nearest the selected",
            "  energy in the existing discrete grid; it is not exactly energy matched.",
            "  Intervals are local quadratic 95% intervals after a nonlinear fit.",
            "  Their repeated synthetic coverage, not the nominal 95% label alone,",
            "  determines whether they are calibrated in this frozen campaign.",
            "  Representative profiles fix one log parameter and nonlinearly re-fit",
            "  the other two; they are not a replacement for global confidence sets.",
            "  Truth and inference share the same lumped equations. This closes the",
            "  current synthetic selection claim, not hardware identifiability.",
        )
    )
    return "\n".join(lines)


def save_nonlinear_experiment_selection_figure(
    result: NonlinearExperimentSelectionResult,
    output: Path | str,
) -> Path:
    """Save per-parameter recovery, interval width, coverage, and joint volume.

    Percentage RMSE is computed across trials for each physical parameter.
    It is distinct from the original mean combined log-RMSE in the text report.
    Interval widths are arithmetic means of (upper - lower) / estimate.
    Nonlinear profiles remain available in the unchanged result sidecar.
    """

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(14.8, 11.5), facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    figure.subplots_adjust(
        left=0.075, right=0.975, top=0.785, bottom=0.195,
        wspace=0.25, hspace=0.68,
    )
    summaries = result.summaries
    styles = {
        "selected": ("Selected", "#087F8C", "o", -0.23),
        "naive": ("Naive", "#D46A34", "^", 0.0),
        "resource_control": ("Control", "#5266A3", "s", 0.23),
    }
    parameter_labels = (
        "Thermal contact\nresistance", "Heat\nstorage", "Sensor response\ntime",
    )
    ink, muted = "#192C40", "#64748B"
    figure.text(
        0.055, 0.955, "Choosing the next experiment",
        fontsize=25, fontweight="bold", color=ink,
    )
    figure.text(
        0.055, 0.919,
        f"{result.config.trial_count} paired synthetic trials  |  "
        "Same sensor setup  |  Three electrical pulse choices",
        fontsize=13, color=muted,
    )
    handles = [
        Line2D(
            [], [], color=styles[item.role][1], marker=styles[item.role][2],
            linestyle="none", markersize=9,
            label=(f"{styles[item.role][0]}: {item.current_amplitude:g} A for "
                   f"{item.pulse_duration:g} s  ({item.electrical_energy:.2f} J)"),
        )
        for item in summaries
    ]
    figure.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.051, 0.89),
        ncol=3, frameon=False, fontsize=11.5, handletextpad=0.4,
        columnspacing=2.0,
    )

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        for side in ("bottom", "left"):
            axis.spines[side].set_color("#CBD5E1")
        axis.tick_params(colors=muted, labelsize=11, length=0, pad=9)
        axis.grid(axis="y", color="#E2E8F0", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.yaxis.label.set_color(muted)
        axis.yaxis.label.set_size(11)

    def heading(axis, title, detail):
        axis.text(0, 1.19, title, transform=axis.transAxes,
                  fontsize=15, fontweight="bold", color=ink)
        axis.text(0, 1.07, detail, transform=axis.transAxes,
                  fontsize=10.5, color=muted)

    heading(axes[0, 0], "How accurate are the estimates?",
            "Error for each parameter across trials  |  Lower is better")
    heading(axes[0, 1], "How wide are the uncertainty ranges?",
            "Mean 95% interval width  |  Narrower is more precise")
    heading(axes[1, 0], "Do the ranges contain the true values?",
            "Labels count trials  |  Dashed line marks the nominal 95% level")
    heading(axes[1, 1], "How much joint uncertainty remains?",
            "All three parameters together, including their correlations")

    parameter_errors = {}
    for summary in summaries:
        _, color, marker, offset = styles[summary.role]
        trials = tuple(trial for trial in result.trials if trial.role == summary.role)
        parameter_errors[summary.role] = []
        for parameter_index in range(3):
            estimates = tuple(trial.fit.physical_values[parameter_index] for trial in trials)
            truths = tuple(trial.truth.physical_values[parameter_index] for trial in trials)
            intervals = tuple(trial.fit.intervals[parameter_index] for trial in trials)
            rmse = 100 * math.sqrt(fmean(
                (estimate / truth - 1) ** 2
                for estimate, truth in zip(estimates, truths)
            ))
            parameter_errors[summary.role].append(rmse)
            width = 100 * fmean(
                (interval.upper_95 - interval.lower_95) / estimate
                for interval, estimate in zip(intervals, estimates)
            )
            covered = sum(
                interval.lower_95 <= truth <= interval.upper_95
                for interval, truth in zip(intervals, truths)
            )
            x = parameter_index + offset
            for axis, value, label in (
                (axes[0, 0], rmse, f"{rmse:.2f}%"),
                (axes[0, 1], width, f"{width:.1f}%"),
                (axes[1, 0], 100 * covered / len(trials), f"{covered}/{len(trials)}"),
            ):
                axis.plot(x, value, marker=marker, color=color, markersize=9,
                          markeredgecolor="white", markeredgewidth=0.7)
                label_offset = 23 if axis is axes[1, 0] and summary.role == "naive" else 10
                axis.annotate(label, (x, value), xytext=(0, label_offset),
                              textcoords="offset points", ha="center",
                              color=color, fontsize=10, fontweight="medium")

    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xticks(range(3), parameter_labels)
        axis.set_xlim(-0.55, 2.55)

    percent = FuncFormatter(lambda value, _: f"{value:g}%")
    axes[0, 0].set_ylim(bottom=0)
    axes[0, 0].margins(y=0.30)
    axes[0, 0].set_ylabel("Root-mean-square percentage error")
    axes[0, 0].yaxis.set_major_formatter(percent)

    axes[0, 1].set_yscale("log")
    axes[0, 1].margins(y=0.28)
    axes[0, 1].set_ylabel("Width / estimate (%) · log scale")
    axes[0, 1].yaxis.set_major_formatter(percent)
    axes[0, 1].tick_params(axis="y", which="minor", left=False)

    axes[1, 0].axhline(95, color="#94A3B8", linestyle="--", linewidth=1.2, zorder=1)
    axes[1, 0].set_ylim(0, 115)
    axes[1, 0].set_yticks((0, 25, 50, 75, 100))
    axes[1, 0].yaxis.set_major_formatter(percent)
    axes[1, 0].set_ylabel("Trials whose interval included the truth")
    simultaneous = "   |   ".join(
        f"{styles[item.role][0]} "
        f"{sum(trial.all_physical_intervals_cover for trial in result.trials if trial.role == item.role)}"
        f"/{item.trial_count}"
        for item in summaries
    )
    axes[1, 0].text(
        0.5, 0.28, "All three ranges included the truth:\n" + simultaneous,
        transform=axes[1, 0].transAxes, ha="center", fontsize=10.5,
        color=muted, linespacing=1.9,
        bbox={"boxstyle": "round,pad=0.7", "facecolor": "#F5F8FB", "edgecolor": "none"},
    )

    naive_volume = next(item.mean_physical_uncertainty_volume
                        for item in summaries if item.role == "naive")
    axis = axes[1, 1]
    for index, summary in enumerate(summaries):
        _, color, marker, _ = styles[summary.role]
        relative_volume = 100 * summary.mean_physical_uncertainty_volume / naive_volume
        axis.plot(index, relative_volume, marker=marker, color=color, markersize=10)
        axis.annotate(f"{relative_volume:.2f}%" if relative_volume < 1 else f"{relative_volume:g}%",
                      (index, relative_volume), xytext=(0, 12),
                      textcoords="offset points", ha="center", fontsize=11, color=color)
    axis.set_yscale("log")
    axis.margins(y=0.24)
    axis.set_xlim(-0.55, len(summaries) - 0.45)
    axis.set_xticks(range(len(summaries)), tuple(styles[item.role][0] for item in summaries))
    axis.yaxis.set_major_formatter(percent)
    axis.set_ylabel("Mean joint volume / naive (%) · log scale")
    axis.tick_params(axis="y", which="minor", left=False)
    axis.text(
        0.5, 0.52,
        f"Selected: {result.selected_interval_volume_reduction_vs_resource_control_percent:.1f}% "
        "smaller joint volume than control",
        transform=axis.transAxes, ha="center", fontsize=10.5, color=styles["selected"][1],
    )

    improves_each_parameter = all(
        selected < other
        for role, errors in parameter_errors.items() if role != "selected"
        for selected, other in zip(parameter_errors["selected"], errors)
    )
    takeaway = (
        "The selected pulse improves all three estimates; wide naive ranges can cover the truth while revealing little."
        if improves_each_parameter else
        "Read estimation error, interval width, and coverage together to judge how much an experiment reveals."
    )
    figure.text(
        0.055, 0.112,
        takeaway,
        fontsize=12, fontweight="medium", color=ink,
    )
    figure.text(
        0.055, 0.078,
        "Widths are arithmetic means and can be inflated by unusually wide intervals. "
        "Intervals use a local approximation.",
        fontsize=9.5, color=muted,
    )
    figure.text(
        0.055, 0.054,
        "Joint volume includes parameter correlations in log coordinates. "
        f"The original {result.selected_rmse_reduction_vs_naive_percent:.1f}% / "
        f"{result.selected_rmse_reduction_vs_resource_control_percent:.1f}% error reductions "
        "use a separate combined log-error score.",
        fontsize=9.5, color=muted,
    )
    figure.text(
        0.055, 0.030,
        "Synthetic validation, not hardware measurements. "
        "The control uses similar, not identical, energy.",
        fontsize=9.5, color=muted,
    )
    figure.savefig(destination, dpi=200)
    save_figure_data(result, destination)
    return destination


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the selected thermal experiment with nonlinear refits."
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--first-seed", type=int, default=52_001)
    parser.add_argument("--skip-profiles", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_NONLINEAR_EXPERIMENT_SELECTION_PATH
    )
    args = parser.parse_args(argv)
    result = run_nonlinear_experiment_selection_study(
        NonlinearExperimentSelectionConfig(
            trial_count=args.trials,
            first_seed=args.first_seed,
        ),
        include_profiles=not args.skip_profiles,
        progress=lambda message: print(message, flush=True),
    )
    destination = save_nonlinear_experiment_selection_figure(result, args.output)
    print(format_nonlinear_experiment_selection_report(result))
    print(f"figure: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
