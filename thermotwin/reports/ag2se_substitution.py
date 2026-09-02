"""Run and plot the matched Ag2Se material-substitution study."""

import argparse
from pathlib import Path
from typing import Optional, Sequence, Union

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .paths import default_figure_path, save_figure_data
from ..design.ag2se_substitution import (
    Ag2SeSubstitutionConfig,
    Ag2SeSubstitutionResult,
    format_ag2se_substitution_report,
    run_ag2se_substitution_study,
)


DEFAULT_AG2SE_SUBSTITUTION_PATH = default_figure_path(
    "ag2se_matched_substitution.png", "AG2SE_SUBSTITUTION_EXPERIMENT.md"
)


def save_ag2se_substitution_report(
    result: Ag2SeSubstitutionResult,
    output_path: Union[str, Path],
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(13.0, 9.0), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    summaries = result.summaries
    application_labels = tuple(
        item.application_label.replace(" ", "\n", 1) for item in summaries
    )
    labels = tuple(
        f"{application_label}\n"
        f"rho={item.specific_contact_resistivity:.1e}"
        for item, application_label in zip(summaries, application_labels)
    )
    positions = tuple(range(len(summaries)))

    improvement_axis = axes[0, 0]
    width = 0.25
    for offset, field, color, label in (
        (-width, "utility_improved_fraction", "tab:blue", "utility"),
        (0.0, "cooling_improved_fraction", "tab:green", "cooling"),
        (width, "cop_improved_fraction", "tab:orange", "COP"),
    ):
        improvement_axis.bar(
            tuple(value + offset for value in positions),
            tuple(100.0 * getattr(item, field) for item in summaries),
            width=width,
            color=color,
            label=label,
        )
    improvement_axis.set_title("Matched designs improved by Ag2Se")
    improvement_axis.set_ylabel("Designs improved (%)")
    improvement_axis.set_xticks(positions, labels, fontsize=7)
    improvement_axis.legend(fontsize=8)

    feasibility_axis = axes[0, 1]
    feasibility_axis.bar(
        tuple(value - 0.18 for value in positions),
        tuple(item.feasibility_gained_count for item in summaries),
        width=0.36,
        color="tab:green",
        label="gained",
    )
    feasibility_axis.bar(
        tuple(value + 0.18 for value in positions),
        tuple(-item.feasibility_lost_count for item in summaries),
        width=0.36,
        color="tab:red",
        label="lost",
    )
    feasibility_axis.axhline(0.0, color="0.25", linewidth=0.8)
    feasibility_axis.set_title("Requirement feasibility changes")
    feasibility_axis.set_ylabel("Matched design count")
    feasibility_axis.set_xticks(positions, labels, fontsize=7)
    feasibility_axis.legend(fontsize=8)

    delta_axis = axes[1, 0]
    delta_axis.bar(
        tuple(value - 0.18 for value in positions),
        tuple(item.median_cooling_change for item in summaries),
        width=0.36,
        color="tab:blue",
        label="delta cooling (W)",
    )
    delta_axis.bar(
        tuple(value + 0.18 for value in positions),
        tuple(item.median_cop_change or 0.0 for item in summaries),
        width=0.36,
        color="tab:orange",
        label="delta COP",
    )
    delta_axis.axhline(0.0, color="0.25", linewidth=0.8)
    delta_axis.set_title("Median matched changes")
    delta_axis.set_xticks(positions, labels, fontsize=7)
    delta_axis.legend(fontsize=8)

    best_axis = axes[1, 1]
    best_axis.bar(
        tuple(value - 0.18 for value in positions),
        tuple(item.best_original_utility or 0.0 for item in summaries),
        width=0.36,
        color="0.55",
        label="original best",
    )
    best_axis.bar(
        tuple(value + 0.18 for value in positions),
        tuple(item.best_ag2se_utility or 0.0 for item in summaries),
        width=0.36,
        color="tab:purple",
        label="Ag2Se best",
    )
    best_axis.set_title("Best feasible utility in matched pool")
    best_axis.set_xticks(positions, labels, fontsize=7)
    best_axis.legend(fontsize=8)

    figure.suptitle(
        "ThermoTwin matched Ag2Se substitution\n"
        "same p material, geometry, interfaces, exchangers, and current grid",
        fontsize=14,
    )
    figure.savefig(destination, dpi=150)
    save_figure_data(result, destination)
    return destination


def build_and_save_ag2se_substitution_report(
    output_path: Union[str, Path] = DEFAULT_AG2SE_SUBSTITUTION_PATH,
    *,
    config: Ag2SeSubstitutionConfig = Ag2SeSubstitutionConfig(),
) -> tuple[Ag2SeSubstitutionResult, Path]:
    result = run_ag2se_substitution_study(config)
    return result, save_ag2se_substitution_report(result, output_path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_AG2SE_SUBSTITUTION_PATH))
    arguments = parser.parse_args(argv)
    result, destination = build_and_save_ag2se_substitution_report(arguments.output)
    print(format_ag2se_substitution_report(result))
    print(f"report: {destination}")


if __name__ == "__main__":
    main()
