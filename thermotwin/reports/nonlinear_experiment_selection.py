"""Report complete nonlinear validation of next-experiment selection."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

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
    """Save nonlinear error, uncertainty, coverage, and profile panels."""

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(13.0, 9.0), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    summaries = result.summaries
    labels = tuple(item.role.replace("_", "\n") for item in summaries)
    positions = tuple(range(len(summaries)))

    axis = axes[0, 0]
    axis.bar(positions, tuple(item.mean_physical_log_rmse for item in summaries))
    axis.set_ylabel("Mean physical log-parameter RMSE")
    axis.set_title(f"Complete nonlinear refits ({result.config.trial_count} paired trials)")

    axis = axes[0, 1]
    axis.bar(
        positions,
        tuple(item.mean_physical_uncertainty_volume for item in summaries),
        color=("tab:blue", "tab:orange", "tab:green"),
    )
    axis.set_yscale("log")
    axis.set_ylabel("sqrt(det covariance), log coordinates")
    axis.set_title("Local three-parameter uncertainty volume")

    axis = axes[1, 0]
    width = 0.36
    axis.bar(
        tuple(index - width / 2 for index in positions),
        tuple(item.individual_parameter_95_coverage for item in summaries),
        width,
        label="individual parameters",
    )
    axis.bar(
        tuple(index + width / 2 for index in positions),
        tuple(item.simultaneous_physical_95_coverage for item in summaries),
        width,
        label="all three simultaneously",
    )
    axis.axhline(0.95, color="red", linestyle="--", label="nominal 95%")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Empirical coverage")
    axis.set_title("Repeated local-interval coverage")
    axis.legend(fontsize=8)

    axis = axes[1, 1]
    colors = ("tab:blue", "tab:orange", "tab:green")
    for role, linestyle in (("selected", "-"), ("naive", "--")):
        for parameter_index, color in enumerate(colors):
            points = tuple(
                point
                for point in result.profiles
                if point.role == role and point.parameter_index == parameter_index
            )
            if not points:
                continue
            minimum = min(point.normalized_mean_squared_error for point in points)
            axis.plot(
                tuple(point.fixed_log_multiplier for point in points),
                tuple(point.normalized_mean_squared_error - minimum for point in points),
                linestyle=linestyle,
                marker="o",
                color=color,
                label=f"{role}: {JOINT_PARAMETER_NAMES[parameter_index].split('_')[0]}",
            )
    axis.set_yscale("symlog", linthresh=0.1)
    axis.set_xlabel("Fixed log multiplier from nominal")
    axis.set_ylabel("Profile normalized-MSE increase")
    axis.set_title("Representative nonlinear profiles")
    axis.legend(fontsize=7, ncol=2)

    for axis in axes.flat:
        if axis is not axes[1, 1]:
            axis.set_xticks(positions, labels)
        axis.grid(alpha=0.25)
    figure.suptitle(
        "Next-experiment selection: local recommendation, nonlinear validation",
        fontsize=15,
    )
    figure.savefig(destination, dpi=170)
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
