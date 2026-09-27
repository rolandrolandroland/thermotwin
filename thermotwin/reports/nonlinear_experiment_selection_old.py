"""Archived four-panel layout of the nonlinear next-experiment validation figure.

This renders the figure exactly as it stood before the per-parameter redesign:
mean combined log-parameter RMSE, joint uncertainty volume, repeated interval
coverage, and the representative re-optimized profiles.

It is kept as a separate ``_old`` artifact for comparison. The current layout in
:mod:`thermotwin.reports.nonlinear_experiment_selection` is unaffected, and
neither report overwrites the other's figure or sidecars.
"""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .paths import default_figure_path, save_figure_data, save_figure_explanation
from ..inference.joint_thermal_parameters import JOINT_PARAMETER_NAMES
from ..studies.nonlinear_experiment_selection import (
    NonlinearExperimentSelectionConfig,
    NonlinearExperimentSelectionResult,
    run_nonlinear_experiment_selection_study,
)


WALKTHROUGH = "NONLINEAR_EXPERIMENT_SELECTION.md"

DEFAULT_NONLINEAR_EXPERIMENT_SELECTION_OLD_PATH = default_figure_path(
    "nonlinear_experiment_selection_old.png", WALKTHROUGH
)

FIGURE_EXPLANATION = (
    "This is the archived four-panel layout of the nonlinear next-experiment "
    "validation, kept for comparison with the current per-parameter figure. It "
    "shows mean combined log-parameter RMSE across the selected, naive, and "
    "closest-energy control pulses, the joint three-parameter uncertainty "
    "volume, repeated individual and simultaneous interval coverage, and the "
    "representative profiles from one trial. The underlying campaign is "
    "unchanged; only the presentation differs."
)


def save_nonlinear_experiment_selection_old_figure(
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
        description=(
            "Render the archived four-panel nonlinear next-experiment "
            "selection figure."
        )
    )
    parser.add_argument(
        "--trials", type=int, default=None,
        help="override the paired trial count",
    )
    parser.add_argument(
        "--output", type=Path,
        default=DEFAULT_NONLINEAR_EXPERIMENT_SELECTION_OLD_PATH,
        help="destination figure path",
    )
    arguments = parser.parse_args(argv)
    config = NonlinearExperimentSelectionConfig()
    if arguments.trials is not None:
        config = NonlinearExperimentSelectionConfig(trial_count=arguments.trials)
    result = run_nonlinear_experiment_selection_study(
        config, progress=lambda message: print(message, flush=True)
    )
    figure_path = save_nonlinear_experiment_selection_old_figure(
        result, arguments.output
    )
    save_figure_data(result, figure_path)
    save_figure_explanation(
        figure_path, explanation=FIGURE_EXPLANATION, walkthrough=WALKTHROUGH
    )
    print(f"figure: {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
