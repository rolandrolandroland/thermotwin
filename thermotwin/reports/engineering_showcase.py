"""One-command engineering showcase for ThermoTwin decision workflows."""

import argparse
from pathlib import Path
from typing import NamedTuple, Sequence, Union

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..inference.assembly_fingerprint import (
    AssemblyFingerprintStudyResult,
    AssemblySpecification,
    format_assembly_fingerprint_report,
    reference_assembly_batch,
    run_assembly_fingerprint_study,
)
from ..design.control_comparison import (
    ControlComparisonConfig,
    ControlComparisonResult,
    format_control_comparison_report,
    run_control_comparison,
)
from ..inference.experiment_selection import (
    ExperimentSelectionConfig,
    ExperimentSelectionResult,
    format_experiment_selection_report,
    run_next_experiment_selection,
)
from .paths import default_figure_path
from ..inference.sparse_sensors import (
    SparseSensorInferenceConfig,
    SparseSensorInferenceResult,
    format_sparse_sensor_inference_report,
    run_sparse_sensor_inference_experiment,
)


DEFAULT_ENGINEERING_SHOWCASE_PATH = default_figure_path(
    "engineering_decision_showcase.png"
)


class EngineeringShowcaseData(NamedTuple):
    sparse_inference: SparseSensorInferenceResult
    control_comparison: ControlComparisonResult
    experiment_selection: ExperimentSelectionResult
    assembly_fingerprints: AssemblyFingerprintStudyResult


def run_engineering_showcase(
    *,
    sparse_config: SparseSensorInferenceConfig = SparseSensorInferenceConfig(),
    control_config: ControlComparisonConfig = ControlComparisonConfig(),
    selection_config: ExperimentSelectionConfig = ExperimentSelectionConfig(),
    assemblies: Sequence[AssemblySpecification] = reference_assembly_batch(),
) -> EngineeringShowcaseData:
    """Run all reproducible synthetic engineering experiments."""

    sparse = run_sparse_sensor_inference_experiment(sparse_config)
    resistance_interval = sparse.uncertainty.intervals[0]
    resistance_samples = (
        max(0.01, resistance_interval.lower_95),
        sparse.fit.inferred_cold_contact_resistance,
        resistance_interval.upper_95,
    )
    return EngineeringShowcaseData(
        sparse_inference=sparse,
        control_comparison=run_control_comparison(
            control_config,
            cold_contact_resistance_samples=resistance_samples,
        ),
        experiment_selection=run_next_experiment_selection(selection_config),
        assembly_fingerprints=run_assembly_fingerprint_study(assemblies),
    )


def save_engineering_showcase(
    result: EngineeringShowcaseData,
    output_path: Union[str, Path],
) -> Path:
    """Save the four-panel decision evidence figure."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(14.0, 10.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)

    control_axis = axes[0, 0]
    targets = tuple(
        item.target_cooling_rate for item in result.control_comparison.comparisons
    )
    continuous_cop = tuple(
        item.continuous.delivered_cooling_cop
        for item in result.control_comparison.comparisons
    )
    pulsed_cop = tuple(
        item.best_pulsed.delivered_cooling_cop
        for item in result.control_comparison.comparisons
    )
    control_axis.plot(
        targets,
        continuous_cop,
        marker="o",
        linewidth=2.0,
        label="Optimized continuous",
    )
    control_axis.plot(
        targets,
        pulsed_cop,
        marker="s",
        linewidth=2.0,
        label="Highest-COP tested pulse",
    )
    control_axis.set_title("Equal delivered-cooling comparison")
    control_axis.set_xlabel("Cold-reservoir cooling rate (W)")
    control_axis.set_ylabel("Delivered cooling COP")
    control_axis.legend()

    inference_axis = axes[0, 1]
    sparse = result.sparse_inference
    interval_by_name = {item.name: item for item in sparse.uncertainty.intervals}
    resistance = interval_by_name["cold_contact_resistance_K_per_W"]
    lag = interval_by_name["shared_sensor_lag_s"]
    truths = (
        sparse.problem.config.true_cold_contact_resistance,
        sparse.problem.config.true_sensor_time_constant,
    )
    intervals = (resistance, lag)
    normalized_estimates = tuple(
        interval.estimate / truth for interval, truth in zip(intervals, truths)
    )
    lower_errors = tuple(
        (interval.estimate - interval.lower_95) / truth
        for interval, truth in zip(intervals, truths)
    )
    upper_errors = tuple(
        (interval.upper_95 - interval.estimate) / truth
        for interval, truth in zip(intervals, truths)
    )
    inference_axis.errorbar(
        (0, 1),
        normalized_estimates,
        yerr=(lower_errors, upper_errors),
        fmt="o",
        capsize=6,
        color="tab:purple",
    )
    inference_axis.axhline(1.0, color="black", linestyle="--", label="Hidden truth")
    inference_axis.set_xticks((0, 1), ("Cold contact R", "Sensor lag"))
    inference_axis.set_ylabel("Estimate / hidden truth")
    inference_axis.set_title("Two exchanger sensors: joint inference")
    inference_axis.text(
        0.02,
        0.04,
        "No face-temperature measurements\n"
        f"Withheld sensor RMSE: {sparse.withheld_validation.accessible_sensor_rmse:.4f} K",
        transform=inference_axis.transAxes,
    )

    selection_axis = axes[1, 0]
    selection = result.experiment_selection
    feasible = tuple(item for item in selection.candidates if item.feasible)
    infeasible = tuple(item for item in selection.candidates if not item.feasible)
    selection_axis.scatter(
        tuple(item.electrical_energy for item in feasible),
        tuple(item.information_gain_nats for item in feasible),
        color="tab:blue",
        label="Feasible",
    )
    selection_axis.scatter(
        tuple(item.electrical_energy for item in infeasible),
        tuple(item.information_gain_nats for item in infeasible),
        color="0.7",
        marker="x",
        label="Over budget / constrained",
    )
    selection_axis.scatter(
        (selection.selected.electrical_energy,),
        (selection.selected.information_gain_nats,),
        color="tab:red",
        marker="*",
        s=180,
        label="Selected",
        zorder=4,
    )
    selection_axis.set_title("Next experiment under a 30 J budget")
    selection_axis.set_xlabel("Electrical energy (J)")
    selection_axis.set_ylabel("Expected information gain (nats)")
    selection_axis.legend(fontsize="small")

    fingerprint_axis = axes[1, 1]
    fingerprints = result.assembly_fingerprints.fingerprints
    truth = tuple(item.true_cold_contact_resistance for item in fingerprints)
    inferred = tuple(item.inferred_cold_contact_resistance for item in fingerprints)
    lower_error = tuple(
        item.inferred_cold_contact_resistance - item.lower_95
        for item in fingerprints
    )
    upper_error = tuple(
        item.upper_95 - item.inferred_cold_contact_resistance
        for item in fingerprints
    )
    fingerprint_axis.errorbar(
        truth,
        inferred,
        yerr=(lower_error, upper_error),
        fmt="o",
        capsize=4,
        color="tab:green",
    )
    lower = min(truth + inferred) - 0.02
    upper = max(truth + inferred) + 0.02
    fingerprint_axis.plot((lower, upper), (lower, upper), "k--", label="Ideal inference")
    fingerprint_axis.axhspan(0.20, 0.30, color="tab:green", alpha=0.10, label="Reference band")
    fingerprint_axis.set_xlim(lower, upper)
    fingerprint_axis.set_ylim(lower, upper)
    fingerprint_axis.set_xlabel("Hidden contact resistance (K/W)")
    fingerprint_axis.set_ylabel("Inferred contact resistance (K/W)")
    fingerprint_axis.set_title("Standardized assembly thermal fingerprint")
    fingerprint_axis.legend(fontsize="small")

    for axis in axes.reshape(-1):
        axis.grid(True, alpha=0.25)
    figure.suptitle(
        "ThermoTwin Engineering Decision Showcase\n"
        "Sparse diagnosis, honest control comparison, experiment design, and assembly screening\n"
        "Synthetic validation — not hardware-calibrated performance",
        fontsize=15,
    )
    figure.savefig(destination, dpi=170)
    return destination


def format_engineering_showcase_report(result: EngineeringShowcaseData) -> str:
    return "\n\n".join(
        (
            format_sparse_sensor_inference_report(result.sparse_inference),
            format_control_comparison_report(result.control_comparison),
            format_experiment_selection_report(result.experiment_selection),
            format_assembly_fingerprint_report(result.assembly_fingerprints),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ThermoTwin's synthetic engineering decision showcase"
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_ENGINEERING_SHOWCASE_PATH,
        help=(
            "destination PNG path "
            "(default: thermotwin/figures/engineering_decision_showcase.png)"
        ),
    )
    arguments = parser.parse_args()
    result = run_engineering_showcase()
    destination = save_engineering_showcase(result, arguments.output)
    print(format_engineering_showcase_report(result))
    print(f"\nfigure: {destination}")


if __name__ == "__main__":
    main()
