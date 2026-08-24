"""Comparison report for the forward PINN and conventional RK4 solution."""

import argparse
from pathlib import Path
from typing import NamedTuple, Sequence, Tuple, Union

import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..simulation.two_node_experiments import (
    TwoNodeExperiment,
    constant_current_reference_experiment,
    run_two_node_experiment,
)
from .paths import default_figure_path
from ..pinn.forward_two_node import (
    ForwardPINNConfig,
    PINNTrainingResult,
    PINNValidation,
    physics_residuals,
    predict_trajectory,
    train_forward_pinn,
)


DEFAULT_FORWARD_PINN_REPORT_PATH = default_figure_path(
    "forward_pinn_comparison.png"
)


class ForwardPINNReportData(NamedTuple):
    """Aligned histories and summary metrics used by the report."""

    time: Tuple[float, ...]
    reference_cold: Tuple[float, ...]
    reference_hot: Tuple[float, ...]
    predicted_cold: Tuple[float, ...]
    predicted_hot: Tuple[float, ...]
    cold_error: Tuple[float, ...]
    hot_error: Tuple[float, ...]
    cold_residual: Tuple[float, ...]
    hot_residual: Tuple[float, ...]
    epoch: Tuple[int, ...]
    loss: Tuple[float, ...]
    validation: PINNValidation


def _validation_from_errors(
    cold_errors: Sequence[float],
    hot_errors: Sequence[float],
) -> PINNValidation:
    sample_count = len(cold_errors)
    if sample_count == 0 or len(hot_errors) != sample_count:
        raise ValueError("cold and hot errors must have equal nonzero lengths")

    return PINNValidation(
        cold_rmse=(
            sum(error**2 for error in cold_errors) / sample_count
        )
        ** 0.5,
        hot_rmse=(
            sum(error**2 for error in hot_errors) / sample_count
        )
        ** 0.5,
        cold_max_absolute_error=max(abs(error) for error in cold_errors),
        hot_max_absolute_error=max(abs(error) for error in hot_errors),
    )


def build_forward_pinn_report_data(
    training: PINNTrainingResult,
    experiment: TwoNodeExperiment,
) -> ForwardPINNReportData:
    """Evaluate RK4, predictions, residuals, errors, and loss on aligned grids."""

    if not training.loss_history:
        raise ValueError("training loss history cannot be empty")

    reference = run_two_node_experiment(experiment).trajectory
    prediction = predict_trajectory(training.model, reference.time)
    cold_errors = tuple(
        predicted - expected
        for predicted, expected in zip(
            prediction.cold,
            reference.cold,
        )
    )
    hot_errors = tuple(
        predicted - expected
        for predicted, expected in zip(
            prediction.hot,
            reference.hot,
        )
    )

    parameter = next(training.model.parameters())
    evaluation_time = torch.tensor(
        reference.time,
        dtype=parameter.dtype,
        device=parameter.device,
    ).reshape(-1, 1)
    training.model.eval()
    residuals = physics_residuals(
        training.model,
        evaluation_time,
        experiment,
    )
    cold_residuals = tuple(
        float(value)
        for value in residuals.cold.detach().cpu().reshape(-1)
    )
    hot_residuals = tuple(
        float(value)
        for value in residuals.hot.detach().cpu().reshape(-1)
    )

    return ForwardPINNReportData(
        time=reference.time,
        reference_cold=reference.cold,
        reference_hot=reference.hot,
        predicted_cold=prediction.cold,
        predicted_hot=prediction.hot,
        cold_error=cold_errors,
        hot_error=hot_errors,
        cold_residual=cold_residuals,
        hot_residual=hot_residuals,
        epoch=tuple(range(1, len(training.loss_history) + 1)),
        loss=training.loss_history,
        validation=_validation_from_errors(cold_errors, hot_errors),
    )


def save_forward_pinn_comparison_report(
    report: ForwardPINNReportData,
    output_path: Union[str, Path],
) -> Path:
    """Save a four-panel PNG report and return its resolved path."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure = Figure(figsize=(12.0, 9.0), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    temperature_axis = axes[0, 0]
    error_axis = axes[0, 1]
    residual_axis = axes[1, 0]
    loss_axis = axes[1, 1]

    temperature_axis.plot(
        report.time,
        report.reference_cold,
        color="tab:blue",
        label="Cold — RK4",
    )
    temperature_axis.plot(
        report.time,
        report.predicted_cold,
        color="tab:blue",
        linestyle="--",
        label="Cold — PINN",
    )
    temperature_axis.plot(
        report.time,
        report.reference_hot,
        color="tab:red",
        label="Hot — RK4",
    )
    temperature_axis.plot(
        report.time,
        report.predicted_hot,
        color="tab:red",
        linestyle="--",
        label="Hot — PINN",
    )
    temperature_axis.set_title("Temperature trajectories")
    temperature_axis.set_xlabel("Time (s)")
    temperature_axis.set_ylabel("Temperature (K)")
    temperature_axis.legend()

    error_axis.plot(
        report.time,
        report.cold_error,
        color="tab:blue",
        label="Cold error",
    )
    error_axis.plot(
        report.time,
        report.hot_error,
        color="tab:red",
        label="Hot error",
    )
    error_axis.axhline(0.0, color="0.3", linewidth=0.8)
    error_axis.set_title("PINN minus RK4")
    error_axis.set_xlabel("Time (s)")
    error_axis.set_ylabel("Temperature error (K)")
    error_axis.legend()

    residual_axis.plot(
        report.time,
        report.cold_residual,
        color="tab:blue",
        label="Cold residual",
    )
    residual_axis.plot(
        report.time,
        report.hot_residual,
        color="tab:red",
        label="Hot residual",
    )
    residual_axis.axhline(0.0, color="0.3", linewidth=0.8)
    residual_axis.set_title("Physics residuals")
    residual_axis.set_xlabel("Time (s)")
    residual_axis.set_ylabel("Residual (K/s)")
    residual_axis.legend()

    loss_axis.semilogy(
        report.epoch,
        report.loss,
        color="tab:purple",
    )
    loss_axis.set_title("Training history")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Physics loss (K²/s²)")

    for axis in axes.reshape(-1):
        axis.grid(True, alpha=0.25)

    validation = report.validation
    figure.suptitle(
        "Forward PINN validation against RK4\n"
        f"Cold RMSE {validation.cold_rmse:.6f} K, "
        f"max {validation.cold_max_absolute_error:.6f} K  |  "
        f"Hot RMSE {validation.hot_rmse:.6f} K, "
        f"max {validation.hot_max_absolute_error:.6f} K"
    )
    figure.savefig(destination, dpi=150)
    return destination


def main() -> None:
    """Train the reference PINN and write its comparison report."""

    parser = argparse.ArgumentParser(
        description="Train and plot the ThermoTwin forward PINN reference case"
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_FORWARD_PINN_REPORT_PATH,
        help=(
            "destination PNG path (default: "
            "thermotwin/figures/forward_pinn_comparison.png)"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "auto"),
        default="cpu",
        help="PyTorch device selection",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2_000,
        help="number of Adam training epochs",
    )
    arguments = parser.parse_args()

    experiment = constant_current_reference_experiment()
    training = train_forward_pinn(
        experiment,
        ForwardPINNConfig(
            epochs=arguments.epochs,
            device=arguments.device,
        ),
    )
    report = build_forward_pinn_report_data(training, experiment)
    output_path = save_forward_pinn_comparison_report(
        report,
        arguments.output,
    )
    print(f"device: {training.device}")
    print(f"final physics loss: {training.loss_history[-1]:.6e}")
    print(f"cold RMSE: {report.validation.cold_rmse:.6f} K")
    print(f"hot RMSE: {report.validation.hot_rmse:.6f} K")
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
