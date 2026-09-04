"""Report which additional observable best exposes a hidden interface state."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .paths import default_figure_path, save_figure_data
from ..studies.sensor_model_discrimination import (
    SensorDiscriminationConfig,
    SensorModelDiscriminationResult,
    TRUTH_CONDITIONS,
    run_sensor_model_discrimination,
)


DEFAULT_SENSOR_DISCRIMINATION_PATH = default_figure_path(
    "sensor_model_discrimination.png",
    "SENSOR_MODEL_DISCRIMINATION.md",
)


_PACKAGE_LABELS = {
    "baseline_one_pulse": "One pulse, exchanger T",
    "more_exchanger_tests": "Four pulses, exchanger T",
    "add_cold_face_temperature": "One pulse + cold-face T",
    "add_cold_heat_rate": "One pulse + cold-side heat rate",
    "add_voltage": "One pulse + voltage",
}


def _package_label(name: str) -> str:
    return _PACKAGE_LABELS.get(name, name.replace("_", " "))


def _passes_predeclared_gates(result, package_name: str) -> bool:
    summaries = tuple(
        item for item in result.summaries if item.package_name == package_name
    )
    return bool(summaries) and all(
        item.correct_model_selection_rate
        >= result.config.model_selection_rate_threshold
        and item.false_confidence_rate
        <= result.config.false_confidence_rate_threshold
        and item.decision_pass_rate
        >= result.config.model_selection_rate_threshold
        for item in summaries
    )


def format_sensor_model_discrimination_report(
    result: SensorModelDiscriminationResult,
) -> str:
    """Return a cost-transparent model-selection and physical-recovery report."""

    config = result.config
    lines = [
        "Sensor model discrimination experiment",
        "======================================",
        "",
        "Decision question:",
        "  Is it more valuable to repeat terminal-temperature tests, or to add one",
        "  observable that can reveal a hidden cold-side interface state?",
        "",
        "Candidate models:",
        "  four_state: the existing contact-aware model",
        "  five_state: the same model plus one cold-interface thermal mass",
        "",
        "Protocol:",
        f"  paired synthetic trials per truth condition: {config.trial_count}",
        "  train both model topologies on the declared package",
        "  choose the lower normalized MSE on one noisy bipolar schedule withheld",
        "  from fitting; the validation schedule uses the same installed channels",
        "  audit the choice against known topology, parameters, and hidden cold face",
        "  profile an independent constant bias for each run and channel",
        "",
        "Resource packages (the common withheld validation run is additional):",
    ]
    for package in config.packages:
        summary = next(
            item
            for item in result.summaries
            if item.truth_condition == TRUTH_CONDITIONS[0]
            and item.package_name == package.name
        )
        lines.append(
            f"  {_package_label(package.name)}: "
            f"{summary.training_experiment_count} training run(s), "
            f"{summary.mean_training_energy:.2f} J modeled training energy, "
            f"{summary.extra_sensor_count} added channel(s)"
        )
    lines.extend(
        (
            "",
            "Predeclared cohort gates:",
            f"  correct topology selection >= "
            f"{config.model_selection_rate_threshold:.0%} in both truth conditions",
            f"  false-confidence rate <= "
            f"{config.false_confidence_rate_threshold:.0%}",
            "  false confidence means tight local physical intervals despite a",
            "  failed topology, physical-parameter, or hidden-state decision",
            f"  physical decision pass requires correct topology, parameter log-RMSE "
            f"<= {config.parameter_log_rmse_threshold:.3f}, and hidden-face RMSE "
            f"<= {config.hidden_face_rmse_threshold:.3f} K",
            "",
            "Results:",
        )
    )
    for summary in result.summaries:
        lines.append(
            f"  {summary.truth_condition} / {_package_label(summary.package_name)}: "
            f"correct={summary.correct_model_selection_rate:.1%}; "
            f"decision pass={summary.decision_pass_rate:.1%}; "
            f"parameter log-RMSE={summary.mean_parameter_log_rmse:.4f}; "
            f"hidden-face RMSE={summary.mean_hidden_face_rmse:.4f} K; "
            f"validation margin={summary.mean_validation_mse_margin:.3f}; "
            f"false confidence={summary.false_confidence_rate:.1%}; "
            f"bound hits={summary.bound_hit_rate:.1%}"
        )
    lines.extend(("", "Gate outcome by package:"))
    for package in config.packages:
        outcome = "PASS" if _passes_predeclared_gates(result, package.name) else "FAIL"
        lines.append(f"  {_package_label(package.name)}: {outcome}")
    lines.extend(
        (
            "",
            "Interpretation boundary:",
            "  This is a synthetic identifiability and model-selection study; it is",
            "  not hardware validation. Noise levels, channel biases, candidate topologies,",
            "  and the 8-80 J/K interface-mass range are assumptions. A real sensor may",
            "  add thermal mass, delay, calibration drift, wiring constraints, and cost",
            "  that are not represented here. The extra-sensor packages are therefore",
            "  evidence about information value, not a purchasing recommendation.",
            "  The held-out run prevents training fit alone from choosing the richer",
            "  model, but it is still generated from one of the two candidate families.",
        )
    )
    return "\n".join(lines)


def save_sensor_model_discrimination_figure(
    result: SensorModelDiscriminationResult,
    output: Path | str,
) -> Path:
    """Save topology selection, recovery, and resource-use panels."""

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(14.0, 9.0), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    package_names = tuple(package.name for package in result.config.packages)
    labels = tuple(_package_label(name) for name in package_names)
    positions = tuple(range(len(package_names)))
    colors = ("tab:blue", "tab:orange")
    widths = 0.36

    for condition_index, condition in enumerate(TRUTH_CONDITIONS):
        selected = tuple(
            next(
                item
                for item in result.summaries
                if item.truth_condition == condition and item.package_name == name
            )
            for name in package_names
        )
        offset = (condition_index - 0.5) * widths
        axes[0, 0].bar(
            tuple(position + offset for position in positions),
            tuple(item.correct_model_selection_rate for item in selected),
            widths,
            color=colors[condition_index],
            label=condition.replace("_", " "),
        )
        axes[0, 1].bar(
            tuple(position + offset for position in positions),
            tuple(item.mean_hidden_face_rmse for item in selected),
            widths,
            color=colors[condition_index],
        )
        axes[1, 0].bar(
            tuple(position + offset for position in positions),
            tuple(item.mean_parameter_log_rmse for item in selected),
            widths,
            color=colors[condition_index],
        )

    axes[0, 0].axhline(
        result.config.model_selection_rate_threshold,
        color="black",
        linestyle="--",
        label="predeclared gate",
    )
    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 0].set_ylabel("Correct topology selection rate")
    axes[0, 0].set_title("Does the package choose the right model?")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].axhline(
        result.config.hidden_face_rmse_threshold,
        color="black",
        linestyle="--",
    )
    axes[0, 1].set_ylabel("Mean withheld cold-face RMSE (K)")
    axes[0, 1].set_title("Hidden-state transfer after model choice")
    axes[1, 0].axhline(
        result.config.parameter_log_rmse_threshold,
        color="black",
        linestyle="--",
    )
    axes[1, 0].set_ylabel("Mean physical log-parameter RMSE")
    axes[1, 0].set_title("Physical-parameter recovery after model choice")

    reference = tuple(
        next(
            item
            for item in result.summaries
            if item.truth_condition == TRUTH_CONDITIONS[0]
            and item.package_name == name
        )
        for name in package_names
    )
    package_colors = {
        "baseline_one_pulse": "tab:gray",
        "more_exchanger_tests": "tab:gray",
        "add_cold_face_temperature": "tab:green",
        "add_cold_heat_rate": "tab:red",
        "add_voltage": "tab:purple",
    }
    axes[1, 1].scatter(
        tuple(item.mean_training_energy for item in reference),
        tuple(item.training_experiment_count for item in reference),
        s=tuple(90 + 90 * item.extra_sensor_count for item in reference),
        color=tuple(package_colors.get(name, "tab:blue") for name in package_names),
    )
    for item, label in zip(reference, labels):
        axes[1, 1].annotate(
            label,
            (item.mean_training_energy, item.training_experiment_count),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
        )
    axes[1, 1].set_xlabel("Modeled training energy (J)")
    axes[1, 1].set_ylabel("Training experiments")
    axes[1, 1].set_title("Resource comparison (larger marker = added channel)")

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xticks(positions, tuple(label.replace(" ", "\n") for label in labels))
        axis.tick_params(axis="x", labelsize=7)
    figure.suptitle(
        "Sensor choice for discriminating a hidden interface state\n"
        "Four-state versus five-state thermal models on withheld excitation",
        fontsize=15,
    )
    figure.savefig(destination, dpi=170)
    save_figure_data(result, destination)
    return destination


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare extra sensing with repeated terminal-temperature tests."
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--first-seed", type=int, default=91_001)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_SENSOR_DISCRIMINATION_PATH
    )
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="print the numerical report without importing optional Matplotlib",
    )
    arguments = parser.parse_args(argv)
    result = run_sensor_model_discrimination(
        SensorDiscriminationConfig(
            trial_count=arguments.trials,
            first_seed=arguments.first_seed,
        ),
        progress=lambda message: print(message, flush=True),
    )
    print(format_sensor_model_discrimination_report(result))
    if not arguments.no_figure:
        destination = save_sensor_model_discrimination_figure(
            result, arguments.output
        )
        print(f"figure: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
