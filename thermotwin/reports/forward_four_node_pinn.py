"""Visual validation report for the four-node contact-aware forward PINN."""

import argparse
import math
from pathlib import Path
from typing import NamedTuple, Tuple, Union

import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..simulation.four_node_experiments import (
    FourNodeContactExperiment,
    constant_current_contact_reference_experiment,
    run_four_node_contact_experiment,
)
from ..pinn.forward_four_node import (
    ContactForwardPINNConfig,
    ContactPINNTrainingResult,
    ContactPINNValidation,
    contact_physics_residuals,
    predict_contact_trajectory,
    train_contact_forward_pinn,
)
from .paths import default_figure_path, save_figure_data


DEFAULT_CONTACT_FORWARD_PINN_REPORT_PATH = default_figure_path(
    "contact_forward_pinn_comparison.png", "CONTACT_RESISTANCE_EXPERIMENT.md"
)


class ContactForwardPINNReportData(NamedTuple):
    """Aligned reference, prediction, residual, and training histories."""

    time: Tuple[float, ...]
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
    reference_cold_contact_drop: Tuple[float, ...]
    reference_hot_contact_drop: Tuple[float, ...]
    predicted_cold_contact_drop: Tuple[float, ...]
    predicted_hot_contact_drop: Tuple[float, ...]
    epoch: Tuple[int, ...]
    loss: Tuple[float, ...]
    validation: ContactPINNValidation


def _errors(
    predicted: Tuple[float, ...],
    expected: Tuple[float, ...],
) -> Tuple[float, ...]:
    return tuple(left - right for left, right in zip(predicted, expected))


def _rmse(values: Tuple[float, ...]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def build_contact_forward_pinn_report_data(
    training: ContactPINNTrainingResult,
    experiment: FourNodeContactExperiment,
) -> ContactForwardPINNReportData:
    """Evaluate the contact PINN and RK4 on one aligned time grid."""

    if not training.loss_history:
        raise ValueError("training loss history cannot be empty")

    reference = run_four_node_contact_experiment(experiment).trajectory
    prediction = predict_contact_trajectory(training.model, reference.time)
    cold_face_error = _errors(prediction.cold_face, reference.cold_face)
    hot_face_error = _errors(prediction.hot_face, reference.hot_face)
    cold_exchanger_error = _errors(
        prediction.cold_exchanger,
        reference.cold_exchanger,
    )
    hot_exchanger_error = _errors(
        prediction.hot_exchanger,
        reference.hot_exchanger,
    )

    parameter = next(training.model.parameters())
    evaluation_time = torch.tensor(
        reference.time,
        dtype=parameter.dtype,
        device=parameter.device,
    ).reshape(-1, 1)
    training.model.eval()
    residuals = contact_physics_residuals(
        training.model,
        evaluation_time,
        experiment,
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
    reference_hot_drop = tuple(
        face - exchanger
        for face, exchanger in zip(
            reference.hot_face,
            reference.hot_exchanger,
        )
    )
    predicted_cold_drop = tuple(
        exchanger - face
        for exchanger, face in zip(
            prediction.cold_exchanger,
            prediction.cold_face,
        )
    )
    predicted_hot_drop = tuple(
        face - exchanger
        for face, exchanger in zip(
            prediction.hot_face,
            prediction.hot_exchanger,
        )
    )
    validation = ContactPINNValidation(
        cold_face_rmse=_rmse(cold_face_error),
        hot_face_rmse=_rmse(hot_face_error),
        cold_exchanger_rmse=_rmse(cold_exchanger_error),
        hot_exchanger_rmse=_rmse(hot_exchanger_error),
        cold_face_max_absolute_error=max(
            abs(value) for value in cold_face_error
        ),
        hot_face_max_absolute_error=max(
            abs(value) for value in hot_face_error
        ),
        cold_exchanger_max_absolute_error=max(
            abs(value) for value in cold_exchanger_error
        ),
        hot_exchanger_max_absolute_error=max(
            abs(value) for value in hot_exchanger_error
        ),
    )

    return ContactForwardPINNReportData(
        time=reference.time,
        reference_cold_face=reference.cold_face,
        reference_hot_face=reference.hot_face,
        reference_cold_exchanger=reference.cold_exchanger,
        reference_hot_exchanger=reference.hot_exchanger,
        predicted_cold_face=prediction.cold_face,
        predicted_hot_face=prediction.hot_face,
        predicted_cold_exchanger=prediction.cold_exchanger,
        predicted_hot_exchanger=prediction.hot_exchanger,
        cold_face_error=cold_face_error,
        hot_face_error=hot_face_error,
        cold_exchanger_error=cold_exchanger_error,
        hot_exchanger_error=hot_exchanger_error,
        cold_face_residual=residual_histories[0],
        hot_face_residual=residual_histories[1],
        cold_exchanger_residual=residual_histories[2],
        hot_exchanger_residual=residual_histories[3],
        reference_cold_contact_drop=reference_cold_drop,
        reference_hot_contact_drop=reference_hot_drop,
        predicted_cold_contact_drop=predicted_cold_drop,
        predicted_hot_contact_drop=predicted_hot_drop,
        epoch=tuple(range(1, len(training.loss_history) + 1)),
        loss=training.loss_history,
        validation=validation,
    )


def save_contact_forward_pinn_report(
    report: ContactForwardPINNReportData,
    output_path: Union[str, Path],
) -> Path:
    """Save a six-panel PNG comparison and return its resolved path."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure = Figure(figsize=(13.0, 12.0), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 2)
    face_axis = axes[0, 0]
    exchanger_axis = axes[0, 1]
    error_axis = axes[1, 0]
    residual_axis = axes[1, 1]
    contact_axis = axes[2, 0]
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

    for values, color, label in (
        (report.cold_face_error, "tab:blue", "Cold face"),
        (report.hot_face_error, "tab:red", "Hot face"),
        (report.cold_exchanger_error, "tab:cyan", "Cold exchanger"),
        (report.hot_exchanger_error, "tab:orange", "Hot exchanger"),
    ):
        error_axis.plot(report.time, values, color=color, label=label)
    error_axis.axhline(0.0, color="0.3", linewidth=0.8)
    error_axis.set_title("PINN minus RK4")
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
    residual_axis.set_title("Physics residuals")
    residual_axis.set_ylabel("Residual (K/s)")
    residual_axis.legend(fontsize="small")

    for reference, predicted, color, label in (
        (
            report.reference_cold_contact_drop,
            report.predicted_cold_contact_drop,
            "tab:blue",
            "Cold contact",
        ),
        (
            report.reference_hot_contact_drop,
            report.predicted_hot_contact_drop,
            "tab:red",
            "Hot contact",
        ),
    ):
        contact_axis.plot(
            report.time,
            reference,
            color=color,
            label=f"{label} RK4",
        )
        contact_axis.plot(
            report.time,
            predicted,
            color=color,
            linestyle="--",
            label=f"{label} PINN",
        )
    contact_axis.set_title("Contact temperature drops")
    contact_axis.set_ylabel("Temperature difference (K)")
    contact_axis.legend(fontsize="small")

    loss_axis.semilogy(report.epoch, report.loss, color="tab:purple")
    loss_axis.set_title("Training history")
    loss_axis.set_ylabel("Physics loss (K²/s²)")

    for axis in axes.reshape(-1):
        axis.set_xlabel("Time (s)" if axis is not loss_axis else "Epoch")
        axis.grid(True, alpha=0.25)

    validation = report.validation
    figure.suptitle(
        "Four-node contact forward PINN validation against RK4\n"
        f"Face RMSE: {validation.cold_face_rmse:.5f} K cold, "
        f"{validation.hot_face_rmse:.5f} K hot  |  "
        f"Exchanger RMSE: {validation.cold_exchanger_rmse:.5f} K cold, "
        f"{validation.hot_exchanger_rmse:.5f} K hot"
    )
    figure.savefig(destination, dpi=150)
    save_figure_data(report, destination)
    return destination


def main() -> None:
    """Train the fixed-parameter contact PINN and save its report."""

    parser = argparse.ArgumentParser(
        description="Train and plot the ThermoTwin contact forward PINN"
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_CONTACT_FORWARD_PINN_REPORT_PATH,
        help=(
            "destination PNG path (default: "
            "thermotwin/figures/CONTACT_RESISTANCE_EXPERIMENT/"
            "contact_forward_pinn_comparison.png)"
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
        default=3_000,
        help="number of Adam training epochs",
    )
    arguments = parser.parse_args()

    experiment = constant_current_contact_reference_experiment()
    training = train_contact_forward_pinn(
        experiment,
        ContactForwardPINNConfig(
            epochs=arguments.epochs,
            device=arguments.device,
        ),
    )
    report = build_contact_forward_pinn_report_data(training, experiment)
    output_path = save_contact_forward_pinn_report(report, arguments.output)
    print(f"device: {training.device}")
    print(f"initial physics loss: {training.loss_history[0]:.6e}")
    print(f"final physics loss: {training.loss_history[-1]:.6e}")
    print(f"cold face RMSE: {report.validation.cold_face_rmse:.6f} K")
    print(f"hot face RMSE: {report.validation.hot_face_rmse:.6f} K")
    print(
        "cold exchanger RMSE: "
        f"{report.validation.cold_exchanger_rmse:.6f} K"
    )
    print(
        "hot exchanger RMSE: "
        f"{report.validation.hot_exchanger_rmse:.6f} K"
    )
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
