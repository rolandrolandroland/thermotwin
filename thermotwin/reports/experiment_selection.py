"""Dedicated figure report for constrained next-experiment selection."""

import argparse
from pathlib import Path
from typing import Union

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..inference.experiment_selection import (
    ExperimentSelectionResult,
    format_experiment_selection_report,
    run_next_experiment_selection,
)
from .paths import default_figure_path, save_figure_data


DEFAULT_EXPERIMENT_SELECTION_PATH = default_figure_path(
    "experiment_selection.png", "NEXT_EXPERIMENT_WALKTHROUGH.md"
)


def save_experiment_selection_report(
    result: ExperimentSelectionResult,
    output_path: Union[str, Path],
) -> Path:
    """Save candidate ranking, uncertainty, and repeated-noise validation."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(13.0, 9.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    feasible = tuple(item for item in result.candidates if item.feasible)
    infeasible = tuple(item for item in result.candidates if not item.feasible)

    axis = axes[0, 0]
    axis.scatter(
        tuple(item.electrical_energy for item in feasible),
        tuple(item.information_gain_nats for item in feasible),
        label="Feasible",
    )
    axis.scatter(
        tuple(item.electrical_energy for item in infeasible),
        tuple(item.information_gain_nats for item in infeasible),
        marker="x",
        color="0.55",
        label="Rejected",
    )
    axis.scatter(
        (result.selected.electrical_energy,),
        (result.selected.information_gain_nats,),
        marker="*",
        s=200,
        color="tab:red",
        label="Selected",
        zorder=4,
    )
    axis.scatter(
        (result.naive.electrical_energy,),
        (result.naive.information_gain_nats,),
        marker="D",
        color="tab:orange",
        label="Naive",
        zorder=4,
    )
    axis.set_xlabel("Modeled electrical energy (J)")
    axis.set_ylabel("Expected information gain (nats)")
    axis.set_title("Constrained candidate frontier")
    axis.legend(fontsize="small")

    currents = tuple(sorted({item.current_amplitude for item in result.candidates}))
    durations = tuple(sorted({item.pulse_duration for item in result.candidates}))
    score_by_pair = {
        (item.current_amplitude, item.pulse_duration): item
        for item in result.candidates
    }
    matrix = [
        [score_by_pair[(current, duration)].information_gain_nats for duration in durations]
        for current in currents
    ]
    axis = axes[0, 1]
    image = axis.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
    for row, current in enumerate(currents):
        for column, duration in enumerate(durations):
            candidate = score_by_pair[(current, duration)]
            if not candidate.feasible:
                axis.text(column, row, "×", ha="center", va="center", color="white", fontsize=15)
            elif candidate.name == result.selected.name:
                axis.text(column, row, "★", ha="center", va="center", color="red", fontsize=15)
    axis.set_xticks(tuple(range(len(durations))), tuple(f"{value:g}" for value in durations))
    axis.set_yticks(tuple(range(len(currents))), tuple(f"{value:g}" for value in currents))
    axis.set_xlabel("Pulse duration (s)")
    axis.set_ylabel("Pulse current (A)")
    axis.set_title("Information across the candidate grid")
    figure.colorbar(image, ax=axis, label="Information gain (nats)")

    selected_errors = (
        result.selected.resistance_log_standard_error,
        result.selected.capacitance_log_standard_error,
        result.selected.lag_log_standard_error,
    )
    naive_errors = (
        result.naive.resistance_log_standard_error,
        result.naive.capacitance_log_standard_error,
        result.naive.lag_log_standard_error,
    )
    positions = (0, 1, 2)
    width = 0.36
    axis = axes[1, 0]
    axis.bar(tuple(x - width / 2 for x in positions), selected_errors, width, label="Selected")
    axis.bar(tuple(x + width / 2 for x in positions), naive_errors, width, label="Naive")
    axis.set_xticks(positions, ("Contact R", "Face C", "Sensor lag"))
    axis.set_ylabel("Local log-parameter standard error")
    axis.set_title("Expected parameter precision")
    axis.legend()

    validation = result.validation
    axis = axes[1, 1]
    labels = ("Selected", "Naive")
    rmse = (validation.selected_log_parameter_rmse, validation.naive_log_parameter_rmse)
    axis.bar(labels, rmse, color=("tab:blue", "tab:orange"))
    axis.set_ylabel("Repeated-noise log-parameter RMSE")
    axis.set_title(f"{validation.trial_count}-trial validation")
    coverage_axis = axis.twinx()
    coverage_axis.plot(
        labels,
        (
            100.0 * validation.selected_nominal_95_coverage,
            100.0 * validation.naive_nominal_95_coverage,
        ),
        "ko--",
        label="95% interval coverage",
    )
    coverage_axis.set_ylabel("Empirical coverage (%)")
    coverage_axis.set_ylim(0.0, 105.0)

    for axis in axes.reshape(-1):
        axis.grid(True, alpha=0.25)
    figure.suptitle(
        "Next-experiment selection under an electrical-energy budget\n"
        "Local D-optimal ranking with nuisance sensor biases",
        fontsize=15,
    )
    figure.savefig(destination, dpi=170)
    save_figure_data(result, destination)
    return destination


def run_and_save_experiment_selection_report(
    output_path: Union[str, Path] = DEFAULT_EXPERIMENT_SELECTION_PATH,
) -> tuple[ExperimentSelectionResult, Path]:
    result = run_next_experiment_selection()
    return result, save_experiment_selection_report(result, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the constrained next-experiment-selection figure"
    )
    parser.add_argument("--output", default=str(DEFAULT_EXPERIMENT_SELECTION_PATH))
    arguments = parser.parse_args()
    result, destination = run_and_save_experiment_selection_report(arguments.output)
    print(format_experiment_selection_report(result))
    print(f"\nfigure: {destination}")


if __name__ == "__main__":
    main()
