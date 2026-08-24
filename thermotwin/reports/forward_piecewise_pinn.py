"""Switch-aware RK4 comparison report for the piecewise contact PINN."""

import argparse
from pathlib import Path
from typing import NamedTuple, Tuple, Union

import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..simulation.four_node_experiments import (
    FourNodeContactExperiment,
    run_four_node_contact_experiment,
)
from ..pinn.forward_four_node import (
    ContactPINNValidation,
    contact_physics_residuals,
    predict_contact_trajectory,
)
from .paths import default_figure_path
from ..pinn.forward_piecewise import (
    PiecewiseContactForwardPINNConfig,
    PiecewiseContactPINNTrainingResult,
    scheduled_current_tensor,
    train_piecewise_contact_forward_pinn,
    unipolar_pulse_contact_experiment,
    validate_piecewise_contact_pinn,
)


DEFAULT_PIECEWISE_CONTACT_FORWARD_PINN_REPORT_PATH = default_figure_path(
    "piecewise_contact_forward_pinn_comparison.png"
)


class PiecewiseContactForwardPINNReportData(NamedTuple):
    """Aligned switch, trajectory, error, residual, and loss histories."""

    time: Tuple[float, ...]
    current: Tuple[float, ...]
    switch_times: Tuple[float, ...]
    reference_cold_face: Tuple[float, ...]
    reference_hot_face: Tuple[float, ...]
    reference_cold_exchanger: Tuple[float, ...]
    reference_hot_exchanger: Tuple[float, ...]
    predicted_cold_face: Tuple[float, ...]
    predicted_hot_face: Tuple[float, ...]
    predicted_cold_exchanger: Tuple[float, ...]
    predicted_hot_exchanger: Tuple[float, ...]
    cold_face_error: Tuple[float, ...]
    hot_face_error: Tuple[float, ...]
    cold_exchanger_error: Tuple[float, ...]
    hot_exchanger_error: Tuple[float, ...]
    cold_face_residual: Tuple[float, ...]
    hot_face_residual: Tuple[float, ...]
    cold_exchanger_residual: Tuple[float, ...]
    hot_exchanger_residual: Tuple[float, ...]
    epoch: Tuple[int, ...]
    loss: Tuple[float, ...]
    max_boundary_temperature_jump: float
    validation: ContactPINNValidation


def _errors(
    predicted: Tuple[float, ...],
    expected: Tuple[float, ...],
) -> Tuple[float, ...]:
    return tuple(left - right for left, right in zip(predicted, expected))


def build_piecewise_contact_forward_pinn_report_data(
    training: PiecewiseContactPINNTrainingResult,
    experiment: FourNodeContactExperiment,
) -> PiecewiseContactForwardPINNReportData:
    """Evaluate the switched-current PINN and RK4 on aligned coordinates."""

    if not training.loss_history:
        raise ValueError("training loss history cannot be empty")
    reference = run_four_node_contact_experiment(experiment).trajectory
    prediction = predict_contact_trajectory(training.model, reference.time)
    parameter = next(training.model.parameters())
    evaluation_time = torch.tensor(
        reference.time,
        dtype=parameter.dtype,
        device=parameter.device,
    ).reshape(-1, 1)
    current_values = scheduled_current_tensor(
        experiment.current,
        evaluation_time,
    )
    residuals = contact_physics_residuals(
        training.model,
        evaluation_time,
        experiment,
        current_values=current_values,
    )
    residual_histories = tuple(
        tuple(float(value) for value in residual.detach().cpu().reshape(-1))
        for residual in residuals
    )
    jump_tensor = training.model.boundary_temperature_jumps()
    max_jump = (
        0.0
        if jump_tensor.numel() == 0
        else float(jump_tensor.abs().max().detach().cpu())
    )
    validation = validate_piecewise_contact_pinn(training, experiment)

    return PiecewiseContactForwardPINNReportData(
        time=reference.time,
        current=tuple(
            float(value) for value in current_values.detach().cpu().reshape(-1)
        ),
        switch_times=training.model.segment_boundaries[1:-1],
        reference_cold_face=reference.cold_face,
        reference_hot_face=reference.hot_face,
        reference_cold_exchanger=reference.cold_exchanger,
        reference_hot_exchanger=reference.hot_exchanger,
        predicted_cold_face=prediction.cold_face,
        predicted_hot_face=prediction.hot_face,
        predicted_cold_exchanger=prediction.cold_exchanger,
        predicted_hot_exchanger=prediction.hot_exchanger,
        cold_face_error=_errors(prediction.cold_face, reference.cold_face),
        hot_face_error=_errors(prediction.hot_face, reference.hot_face),
        cold_exchanger_error=_errors(
            prediction.cold_exchanger,
            reference.cold_exchanger,
        ),
        hot_exchanger_error=_errors(
            prediction.hot_exchanger,
            reference.hot_exchanger,
        ),
        cold_face_residual=residual_histories[0],
        hot_face_residual=residual_histories[1],
        cold_exchanger_residual=residual_histories[2],
        hot_exchanger_residual=residual_histories[3],
        epoch=tuple(range(1, len(training.loss_history) + 1)),
        loss=training.loss_history,
        max_boundary_temperature_jump=max_jump,
        validation=validation,
    )


def _mark_switches(axis, switch_times: Tuple[float, ...]) -> None:
    for switch_time in switch_times:
        axis.axvline(
            switch_time,
            color="0.35",
            linestyle=":",
            linewidth=1.0,
        )


def save_piecewise_contact_forward_pinn_report(
    report: PiecewiseContactForwardPINNReportData,
    output_path: Union[str, Path],
) -> Path:
    """Save a six-panel switched-current PNG and return its resolved path."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure = Figure(figsize=(13.0, 12.0), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 2)
    face_axis = axes[0, 0]
    exchanger_axis = axes[0, 1]
    current_axis = axes[1, 0]
    error_axis = axes[1, 1]
    residual_axis = axes[2, 0]
    loss_axis = axes[2, 1]

    for reference, predicted, color, label in (
        (
            report.reference_cold_face,
            report.predicted_cold_face,
            "tab:blue",
            "Cold face",
        ),
        (
            report.reference_hot_face,
            report.predicted_hot_face,
            "tab:red",
            "Hot face",
        ),
    ):
        face_axis.plot(report.time, reference, color=color, label=f"{label} RK4")
        face_axis.plot(
            report.time,
            predicted,
            color=color,
            linestyle="--",
            label=f"{label} PINN",
        )
    face_axis.set_title("Thermoelectric face temperatures")
    face_axis.set_ylabel("Temperature (K)")
    face_axis.legend(fontsize="small")

    for reference, predicted, color, label in (
        (
            report.reference_cold_exchanger,
            report.predicted_cold_exchanger,
            "tab:cyan",
            "Cold exchanger",
        ),
        (
            report.reference_hot_exchanger,
            report.predicted_hot_exchanger,
            "tab:orange",
            "Hot exchanger",
        ),
    ):
        exchanger_axis.plot(
            report.time,
            reference,
            color=color,
            label=f"{label} RK4",
        )
        exchanger_axis.plot(
            report.time,
            predicted,
            color=color,
            linestyle="--",
            label=f"{label} PINN",
        )
    exchanger_axis.set_title("Heat-exchanger temperatures")
    exchanger_axis.set_ylabel("Temperature (K)")
    exchanger_axis.legend(fontsize="small")

    current_axis.step(
        report.time,
        report.current,
        where="post",
        color="black",
    )
    current_axis.set_title("Right-continuous current schedule")
    current_axis.set_ylabel("Current (A)")

    for values, color, label in (
        (report.cold_face_error, "tab:blue", "Cold face"),
        (report.hot_face_error, "tab:red", "Hot face"),
        (report.cold_exchanger_error, "tab:cyan", "Cold exchanger"),
        (report.hot_exchanger_error, "tab:orange", "Hot exchanger"),
    ):
        error_axis.plot(report.time, values, color=color, label=label)
    error_axis.axhline(0.0, color="0.3", linewidth=0.8)
    error_axis.set_title("Piecewise PINN minus RK4")
    error_axis.set_ylabel("Temperature error (K)")
    error_axis.legend(fontsize="small")

    for values, color, label in (
        (report.cold_face_residual, "tab:blue", "Cold face"),
        (report.hot_face_residual, "tab:red", "Hot face"),
        (report.cold_exchanger_residual, "tab:cyan", "Cold exchanger"),
        (report.hot_exchanger_residual, "tab:orange", "Hot exchanger"),
    ):
        residual_axis.plot(report.time, values, color=color, label=label)
    residual_axis.axhline(0.0, color="0.3", linewidth=0.8)
    residual_axis.set_title("Right-continuous physics residuals")
    residual_axis.set_ylabel("Residual (K/s)")
    residual_axis.legend(fontsize="small")

    loss_axis.semilogy(report.epoch, report.loss, color="tab:purple")
    loss_axis.set_title("Training history")
    loss_axis.set_ylabel("Physics loss (K²/s²)")
    loss_axis.set_xlabel("Epoch")

    for axis in (
        face_axis,
        exchanger_axis,
        current_axis,
        error_axis,
        residual_axis,
    ):
        _mark_switches(axis, report.switch_times)
        axis.set_xlabel("Time (s)")
    for axis in axes.reshape(-1):
        axis.grid(True, alpha=0.25)

    validation = report.validation
    figure.suptitle(
        "Piecewise contact forward PINN for a switched-current pulse\n"
        f"Boundary jump {report.max_boundary_temperature_jump:.2e} K  |  "
        f"Face RMSE {validation.cold_face_rmse:.5f} K cold, "
        f"{validation.hot_face_rmse:.5f} K hot  |  "
        f"Exchanger RMSE {validation.cold_exchanger_rmse:.5f} K cold, "
        f"{validation.hot_exchanger_rmse:.5f} K hot"
    )
    figure.savefig(destination, dpi=150)
    return destination


def main() -> None:
    """Train the unipolar-pulse PINN and save its comparison report."""

    parser = argparse.ArgumentParser(
        description="Train and plot the piecewise contact forward PINN"
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_PIECEWISE_CONTACT_FORWARD_PINN_REPORT_PATH,
        help=(
            "destination PNG path (default: thermotwin/figures/"
            "piecewise_contact_forward_pinn_comparison.png)"
        ),
    )
    parser.add_argument("--epochs", type=int, default=5_000)
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "auto"),
        default="cpu",
    )
    arguments = parser.parse_args()

    experiment = unipolar_pulse_contact_experiment()
    training = train_piecewise_contact_forward_pinn(
        experiment,
        PiecewiseContactForwardPINNConfig(
            epochs=arguments.epochs,
            device=arguments.device,
        ),
    )
    report = build_piecewise_contact_forward_pinn_report_data(
        training,
        experiment,
    )
    output_path = save_piecewise_contact_forward_pinn_report(
        report,
        arguments.output,
    )
    validation = report.validation
    print(f"device: {training.device}")
    print(f"segments: {len(training.model.segment_boundaries) - 1}")
    print(f"final physics loss: {training.loss_history[-1]:.6e}")
    print(
        "maximum boundary temperature jump: "
        f"{report.max_boundary_temperature_jump:.6e} K"
    )
    print(f"cold face RMSE: {validation.cold_face_rmse:.6f} K")
    print(f"hot face RMSE: {validation.hot_face_rmse:.6f} K")
    print(
        "cold exchanger RMSE: "
        f"{validation.cold_exchanger_rmse:.6f} K"
    )
    print(
        "hot exchanger RMSE: "
        f"{validation.hot_exchanger_rmse:.6f} K"
    )
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
