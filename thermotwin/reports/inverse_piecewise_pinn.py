"""Report for switched-current cold-contact-resistance PINN inference."""

import argparse
from pathlib import Path
from typing import NamedTuple, Tuple, Union

import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..simulation.four_node_experiments import run_four_node_contact_experiment
from ..pinn.forward_four_node import (
    contact_physics_residuals,
    predict_contact_trajectory,
)
from .paths import default_figure_path
from ..pinn.forward_piecewise import scheduled_current_tensor
from ..pinn.inverse_piecewise_contact_resistance import (
    IdealPiecewiseInverseContactProblem,
    PiecewiseInverseContactResistanceConfig,
    PiecewiseInverseContactResistanceValidation,
    PiecewiseInverseContactTrainingResult,
    ideal_piecewise_inverse_contact_problem,
    train_piecewise_inverse_contact_resistance,
    validate_piecewise_inverse_contact_resistance,
)


DEFAULT_PIECEWISE_INVERSE_CONTACT_RESISTANCE_REPORT_PATH = (
    default_figure_path(
        "piecewise_inverse_contact_resistance_comparison.png"
    )
)


class PiecewiseInverseContactResistanceReportData(NamedTuple):
    """Aligned pulse, trajectory, residual, loss, and parameter histories."""

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
    observation_time: Tuple[float, ...]
    observed_cold_face: Tuple[float, ...]
    observed_cold_exchanger: Tuple[float, ...]
    reference_cold_contact_drop: Tuple[float, ...]
    predicted_cold_contact_drop: Tuple[float, ...]
    epoch: Tuple[int, ...]
    total_loss: Tuple[float, ...]
    physics_loss: Tuple[float, ...]
    observation_loss: Tuple[float, ...]
    inferred_resistance: Tuple[float, ...]
    validation: PiecewiseInverseContactResistanceValidation


def _errors(
    predicted: Tuple[float, ...],
    expected: Tuple[float, ...],
) -> Tuple[float, ...]:
    return tuple(left - right for left, right in zip(predicted, expected))


def build_piecewise_inverse_contact_resistance_report_data(
    training: PiecewiseInverseContactTrainingResult,
    problem: IdealPiecewiseInverseContactProblem,
) -> PiecewiseInverseContactResistanceReportData:
    """Build aligned inverse-PINN, RK4, observation, and residual data."""

    history = training.history
    if not history.total_loss:
        raise ValueError("training history cannot be empty")
    history_length = len(history.total_loss)
    if any(
        len(values) != history_length
        for values in (
            history.physics_loss,
            history.observation_loss,
            history.cold_contact_resistance,
        )
    ):
        raise ValueError("all training histories must have equal lengths")

    reference = run_four_node_contact_experiment(
        problem.experiment
    ).trajectory
    prediction = predict_contact_trajectory(training.model, reference.time)
    parameter = next(training.model.parameters())
    evaluation_time = torch.tensor(
        reference.time,
        dtype=parameter.dtype,
        device=parameter.device,
    ).reshape(-1, 1)
    current_values = scheduled_current_tensor(
        problem.experiment.current,
        evaluation_time,
    )
    residuals = contact_physics_residuals(
        training.model,
        evaluation_time,
        problem.experiment,
        cold_contact_resistance=(
            training.model.cold_contact_resistance
        ),
        current_values=current_values,
    )
    residual_histories = tuple(
        tuple(float(value) for value in residual.detach().cpu().reshape(-1))
        for residual in residuals
    )
    reference_cold_drop = tuple(
        exchanger - face
        for exchanger, face in zip(
            reference.cold_exchanger,
            reference.cold_face,
        )
    )
    predicted_cold_drop = tuple(
        exchanger - face
        for exchanger, face in zip(
            prediction.cold_exchanger,
            prediction.cold_face,
        )
    )
    return PiecewiseInverseContactResistanceReportData(
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
        observation_time=problem.observations.time,
        observed_cold_face=problem.observations.cold_face,
        observed_cold_exchanger=problem.observations.cold_exchanger,
        reference_cold_contact_drop=reference_cold_drop,
        predicted_cold_contact_drop=predicted_cold_drop,
        epoch=tuple(range(1, history_length + 1)),
        total_loss=history.total_loss,
        physics_loss=history.physics_loss,
        observation_loss=history.observation_loss,
        inferred_resistance=history.cold_contact_resistance,
        validation=validate_piecewise_inverse_contact_resistance(
            training,
            problem,
        ),
    )


def _mark_switches(axis, switch_times: Tuple[float, ...]) -> None:
    for switch_time in switch_times:
        axis.axvline(
            switch_time,
            color="0.35",
            linestyle=":",
            linewidth=1.0,
        )


def save_piecewise_inverse_contact_resistance_report(
    report: PiecewiseInverseContactResistanceReportData,
    output_path: Union[str, Path],
) -> Path:
    """Save an eight-panel inverse pulse report and return its path."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure = Figure(figsize=(13.0, 16.0), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(4, 2)
    face_axis = axes[0, 0]
    exchanger_axis = axes[0, 1]
    current_axis = axes[1, 0]
    error_axis = axes[1, 1]
    residual_axis = axes[2, 0]
    loss_axis = axes[2, 1]
    resistance_axis = axes[3, 0]
    contact_axis = axes[3, 1]

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
    face_axis.scatter(
        report.observation_time,
        report.observed_cold_face,
        color="black",
        marker="o",
        s=10,
        label="Observed cold face",
        zorder=4,
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
    exchanger_axis.scatter(
        report.observation_time,
        report.observed_cold_exchanger,
        color="black",
        marker="o",
        s=10,
        label="Observed cold exchanger",
        zorder=4,
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
    current_axis.set_title("Right-continuous training current")
    current_axis.set_ylabel("Current (A)")

    for values, color, label in (
        (report.cold_face_error, "tab:blue", "Cold face"),
        (report.hot_face_error, "tab:red", "Hot face"),
        (report.cold_exchanger_error, "tab:cyan", "Cold exchanger"),
        (report.hot_exchanger_error, "tab:orange", "Hot exchanger"),
    ):
        error_axis.plot(report.time, values, color=color, label=label)
    error_axis.axhline(0.0, color="0.3", linewidth=0.8)
    error_axis.set_title("Inverse PINN minus RK4")
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

    for values, color, label in (
        (report.total_loss, "tab:purple", "Total"),
        (report.physics_loss, "tab:green", "Physics"),
        (report.observation_loss, "tab:brown", "Observations"),
    ):
        loss_axis.semilogy(report.epoch, values, color=color, label=label)
    loss_axis.set_title("Normalized training losses")
    loss_axis.set_ylabel("Dimensionless loss")
    loss_axis.legend(fontsize="small")

    validation = report.validation
    resistance_axis.plot(
        report.epoch,
        report.inferred_resistance,
        color="tab:purple",
        label="PINN estimate",
    )
    resistance_axis.axhline(
        validation.true_cold_contact_resistance,
        color="black",
        linestyle="--",
        label="Hidden truth",
    )
    resistance_axis.axhline(
        validation.conventional_cold_contact_resistance,
        color="tab:green",
        linestyle=":",
        label="Conventional fit",
    )
    resistance_axis.set_title("Shared cold contact-resistance recovery")
    resistance_axis.set_ylabel("Resistance (K/W)")
    resistance_axis.legend(fontsize="small")

    contact_axis.plot(
        report.time,
        report.reference_cold_contact_drop,
        color="tab:blue",
        label="RK4",
    )
    contact_axis.plot(
        report.time,
        report.predicted_cold_contact_drop,
        color="tab:blue",
        linestyle="--",
        label="Inverse PINN",
    )
    contact_axis.set_title("Cold contact temperature drop")
    contact_axis.set_ylabel("Temperature difference (K)")
    contact_axis.legend(fontsize="small")

    time_axes = (
        face_axis,
        exchanger_axis,
        current_axis,
        error_axis,
        residual_axis,
        contact_axis,
    )
    for axis in time_axes:
        _mark_switches(axis, report.switch_times)
        axis.set_xlabel("Time (s)")
    loss_axis.set_xlabel("Epoch")
    resistance_axis.set_xlabel("Epoch")
    for axis in axes.reshape(-1):
        axis.grid(True, alpha=0.25)

    figure.suptitle(
        "Piecewise inverse cold-contact-resistance PINN\n"
        f"Truth {validation.true_cold_contact_resistance:.6f} K/W, "
        f"PINN {validation.inferred_cold_contact_resistance:.6f} K/W, "
        "conventional "
        f"{validation.conventional_cold_contact_resistance:.6f} K/W  |  "
        f"parameter error {validation.relative_parameter_error_percent:.3f}%  |  "
        f"boundary jump {validation.max_boundary_temperature_jump:.2e} K"
    )
    figure.savefig(destination, dpi=150)
    return destination


def main() -> None:
    """Train the pulse inverse PINN and save its comparison report."""

    parser = argparse.ArgumentParser(
        description="Train and plot piecewise cold-contact PINN inference"
    )
    parser.add_argument(
        "--output",
        default=(
            DEFAULT_PIECEWISE_INVERSE_CONTACT_RESISTANCE_REPORT_PATH
        ),
        help=(
            "destination PNG path (default: thermotwin/figures/"
            "piecewise_inverse_contact_resistance_comparison.png)"
        ),
    )
    parser.add_argument("--epochs", type=int, default=8_000)
    parser.add_argument(
        "--initial-resistance",
        type=float,
        default=0.5,
        help="initial cold contact resistance in K/W",
    )
    parser.add_argument(
        "--observation-interval",
        type=float,
        default=1.0,
        help="ideal cold-pair observation spacing in seconds",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "auto"),
        default="cpu",
    )
    arguments = parser.parse_args()

    problem = ideal_piecewise_inverse_contact_problem(
        observation_interval=arguments.observation_interval
    )
    training = train_piecewise_inverse_contact_resistance(
        problem,
        PiecewiseInverseContactResistanceConfig(
            epochs=arguments.epochs,
            initial_cold_contact_resistance=arguments.initial_resistance,
            device=arguments.device,
        ),
    )
    report = build_piecewise_inverse_contact_resistance_report_data(
        training,
        problem,
    )
    output_path = save_piecewise_inverse_contact_resistance_report(
        report,
        arguments.output,
    )
    validation = report.validation
    print(f"device: {training.device}")
    print(f"segments: {len(training.model.segment_boundaries) - 1}")
    print(f"cold-pair observation times: {len(problem.observations.time)}")
    print(
        "cold contact resistance: "
        f"true {validation.true_cold_contact_resistance:.6f} K/W, "
        f"PINN {validation.inferred_cold_contact_resistance:.6f} K/W, "
        "conventional "
        f"{validation.conventional_cold_contact_resistance:.6f} K/W"
    )
    print(
        "parameter error: "
        f"{validation.relative_parameter_error_percent:.3f}%"
    )
    print(
        "final normalized losses: "
        f"physics {training.history.physics_loss[-1]:.6e}, "
        f"observations {training.history.observation_loss[-1]:.6e}"
    )
    print(
        "maximum boundary temperature jump: "
        f"{validation.max_boundary_temperature_jump:.6e} K"
    )
    print(
        "unseen pulse all-sensor RMSE: "
        f"validation {validation.validation_regime_metrics.all_sensor_rmse:.6f} K, "
        f"test {validation.test_regime_metrics.all_sensor_rmse:.6f} K"
    )
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
