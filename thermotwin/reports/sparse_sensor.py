"""Dedicated figure report for sparse accessible-sensor inference."""

import argparse
from pathlib import Path
from typing import Union

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..inference.sparse_sensors import (
    SparseSensorInferenceResult,
    format_sparse_sensor_inference_report,
    run_sparse_sensor_inference_experiment,
)
from .paths import default_figure_path, save_figure_data


DEFAULT_SPARSE_SENSOR_PATH = default_figure_path(
    "sparse_sensor_inference.png", "SPARSE_SENSOR_EXPERIMENT.md"
)


def _interval_by_name(result: SparseSensorInferenceResult):
    return {item.name: item for item in result.uncertainty.intervals}


def _single_parameter_axis(axis, interval, truth, *, title, ylabel):
    axis.errorbar(
        (0,),
        (interval.estimate,),
        yerr=(
            (interval.estimate - interval.lower_95,),
            (interval.upper_95 - interval.estimate,),
        ),
        fmt="o",
        capsize=7,
        color="tab:blue",
        label="Estimate and local 95% interval",
    )
    axis.scatter((0,), (truth,), marker="x", s=90, color="black", label="Hidden truth")
    axis.set_xticks((0,), (title,))
    axis.set_ylabel(ylabel)
    axis.legend(fontsize="small")


def save_sparse_sensor_report(
    result: SparseSensorInferenceResult,
    output_path: Union[str, Path],
) -> Path:
    """Save parameter recovery, correlation, and transfer-error evidence."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(13.0, 9.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    intervals = _interval_by_name(result)
    config = result.problem.config

    _single_parameter_axis(
        axes[0, 0],
        intervals["cold_contact_resistance_K_per_W"],
        config.true_cold_contact_resistance,
        title="Cold contact resistance",
        ylabel="Resistance (K/W)",
    )
    axes[0, 0].set_title("Hidden interface recovery")
    _single_parameter_axis(
        axes[0, 1],
        intervals["shared_sensor_lag_s"],
        config.true_sensor_time_constant,
        title="Shared lag",
        ylabel="Time constant (s)",
    )
    axes[0, 1].set_title("Sensor-dynamics recovery")

    axis = axes[1, 0]
    bias_intervals = (
        intervals["cold_sensor_bias_K"],
        intervals["hot_sensor_bias_K"],
    )
    truths = (config.true_cold_sensor_bias, config.true_hot_sensor_bias)
    positions = (0, 1)
    axis.errorbar(
        positions,
        tuple(item.estimate for item in bias_intervals),
        yerr=(
            tuple(item.estimate - item.lower_95 for item in bias_intervals),
            tuple(item.upper_95 - item.estimate for item in bias_intervals),
        ),
        fmt="o",
        capsize=7,
        label="Estimate and local 95% interval",
    )
    axis.scatter(positions, truths, marker="x", s=90, color="black", label="Hidden truth")
    axis.axhline(0.0, color="0.5", linewidth=1.0)
    axis.set_xticks(positions, ("Cold sensor", "Hot sensor"))
    axis.set_ylabel("Calibration bias (K)")
    axis.set_title("Profiled sensor offsets")
    axis.legend(fontsize="small")

    axis = axes[1, 1]
    errors = (
        result.fit.observation_rmse,
        result.hidden_state_validation.cold_face_rmse,
        result.hidden_state_validation.hot_face_rmse,
        result.withheld_validation.accessible_sensor_rmse,
        result.withheld_validation.cold_face_rmse,
        result.withheld_validation.hot_face_rmse,
    )
    labels = (
        "Train\nsensors",
        "Train hidden\ncold face",
        "Train hidden\nhot face",
        "Withheld\nsensors",
        "Withheld hidden\ncold face",
        "Withheld hidden\nhot face",
    )
    axis.bar(tuple(range(len(errors))), errors, color=("tab:blue",) * 3 + ("tab:orange",) * 3)
    axis.set_yscale("log")
    axis.set_xticks(tuple(range(len(labels))), labels, rotation=20, ha="right")
    axis.set_ylabel("RMSE (K, logarithmic scale)")
    axis.set_title("Fit, hidden-state, and transfer errors")
    correlation = result.uncertainty.correlation[0][1]
    axis.text(
        0.03,
        0.95,
        f"Contact R / lag correlation: {correlation:+.3f}\n"
        f"Available temperature records: {len(result.problem.observations.observations)}",
        transform=axis.transAxes,
        va="top",
    )

    for axis in axes.reshape(-1):
        axis.grid(True, alpha=0.25)
    figure.suptitle(
        "Sparse accessible-sensor inference\n"
        "Noisy exchanger temperatures, sensor lag and bias, missing readings, and withheld transfer",
        fontsize=15,
    )
    figure.savefig(destination, dpi=170)
    save_figure_data(result, destination)
    return destination


def run_and_save_sparse_sensor_report(
    output_path: Union[str, Path] = DEFAULT_SPARSE_SENSOR_PATH,
) -> tuple[SparseSensorInferenceResult, Path]:
    result = run_sparse_sensor_inference_experiment()
    return result, save_sparse_sensor_report(result, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the sparse accessible-sensor inference figure"
    )
    parser.add_argument("--output", default=str(DEFAULT_SPARSE_SENSOR_PATH))
    arguments = parser.parse_args()
    result, destination = run_and_save_sparse_sensor_report(arguments.output)
    print(format_sparse_sensor_inference_report(result))
    print(f"\nfigure: {destination}")


if __name__ == "__main__":
    main()
