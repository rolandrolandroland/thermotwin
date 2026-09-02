"""One-command showcase for ThermoTwin's switched-current PINNs."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Union

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .paths import default_figure_path, save_figure_data
from ..pinn.forward_piecewise import (
    PiecewiseContactForwardPINNConfig,
    PiecewiseContactPINNTrainingResult,
    train_piecewise_contact_forward_pinn,
    unipolar_pulse_contact_experiment,
)
from .forward_piecewise_pinn import (
    PiecewiseContactForwardPINNReportData,
    build_piecewise_contact_forward_pinn_report_data,
)
from ..pinn.inverse_piecewise_contact_resistance import (
    IdealPiecewiseInverseContactProblem,
    PiecewiseInverseContactResistanceConfig,
    PiecewiseInverseContactTrainingResult,
    ideal_piecewise_inverse_contact_problem,
    train_piecewise_inverse_contact_resistance,
)
from .inverse_piecewise_pinn import (
    PiecewiseInverseContactResistanceReportData,
    build_piecewise_inverse_contact_resistance_report_data,
)


DEFAULT_PINN_SHOWCASE_PATH = default_figure_path(
    "pinn_showcase.png", "PINN_SHOWCASE.md"
)


@dataclass(frozen=True)
class PINNShowcaseConfig:
    """CPU-first epoch and device choices for the reproducible showcase."""

    forward_epochs: int = 5_000
    inverse_epochs: int = 8_000
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name, value in (
            ("forward epoch count", self.forward_epochs),
            ("inverse epoch count", self.inverse_epochs),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if self.device not in {"cpu", "mps", "auto"}:
            raise ValueError("device must be 'cpu', 'mps', or 'auto'")


class PINNShowcaseData(NamedTuple):
    """Validated physics-only and inverse report data for one pulse."""

    forward: PiecewiseContactForwardPINNReportData
    inverse: PiecewiseInverseContactResistanceReportData


def build_pinn_showcase_data(
    forward_training: PiecewiseContactPINNTrainingResult,
    inverse_training: PiecewiseInverseContactTrainingResult,
    inverse_problem: IdealPiecewiseInverseContactProblem,
) -> PINNShowcaseData:
    """Combine the two independently validated switched-current workflows."""

    forward_experiment = unipolar_pulse_contact_experiment()
    forward = build_piecewise_contact_forward_pinn_report_data(
        forward_training,
        forward_experiment,
    )
    inverse = build_piecewise_inverse_contact_resistance_report_data(
        inverse_training,
        inverse_problem,
    )
    if (
        forward.time != inverse.time
        or forward.current != inverse.current
        or forward.switch_times != inverse.switch_times
    ):
        raise ValueError(
            "forward and inverse showcase experiments must be aligned"
        )
    return PINNShowcaseData(forward=forward, inverse=inverse)


def train_pinn_showcase(
    config: PINNShowcaseConfig = PINNShowcaseConfig(),
) -> PINNShowcaseData:
    """Train both frozen pulse PINNs and build their comparison data."""

    forward_experiment = unipolar_pulse_contact_experiment()
    forward_training = train_piecewise_contact_forward_pinn(
        forward_experiment,
        PiecewiseContactForwardPINNConfig(
            epochs=config.forward_epochs,
            device=config.device,
        ),
    )
    inverse_problem = ideal_piecewise_inverse_contact_problem()
    inverse_training = train_piecewise_inverse_contact_resistance(
        inverse_problem,
        PiecewiseInverseContactResistanceConfig(
            epochs=config.inverse_epochs,
            device=config.device,
        ),
    )
    return build_pinn_showcase_data(
        forward_training,
        inverse_training,
        inverse_problem,
    )


def _mark_switches(axis, switch_times) -> None:
    for switch_time in switch_times:
        axis.axvline(
            switch_time,
            color="0.35",
            linestyle=":",
            linewidth=1.0,
        )


def save_pinn_showcase(
    showcase: PINNShowcaseData,
    output_path: Union[str, Path],
) -> Path:
    """Save the focused six-panel showcase and return its resolved path."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    forward = showcase.forward
    inverse = showcase.inverse
    validation = inverse.validation

    figure = Figure(figsize=(14.0, 13.0), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 2)
    current_axis = axes[0, 0]
    forward_axis = axes[0, 1]
    observed_axis = axes[1, 0]
    withheld_axis = axes[1, 1]
    resistance_axis = axes[2, 0]
    evidence_axis = axes[2, 1]

    current_axis.step(
        forward.time,
        forward.current,
        where="post",
        color="black",
        linewidth=2.0,
    )
    current_axis.set_title("Known switched-current experiment")
    current_axis.set_ylabel("Current (A)")
    current_axis.set_ylim(-0.08, 1.18)
    current_axis.text(
        0.98,
        0.88,
        "Forward training: 0 temperature labels\n"
        "Inverse training: 2 of 4 sensors\n"
        "Dense truth withheld until validation",
        transform=current_axis.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )

    forward_histories = (
        (
            forward.reference_cold_face,
            forward.predicted_cold_face,
            "tab:blue",
            "Cold face",
        ),
        (
            forward.reference_hot_face,
            forward.predicted_hot_face,
            "tab:red",
            "Hot face",
        ),
        (
            forward.reference_cold_exchanger,
            forward.predicted_cold_exchanger,
            "tab:cyan",
            "Cold exchanger",
        ),
        (
            forward.reference_hot_exchanger,
            forward.predicted_hot_exchanger,
            "tab:orange",
            "Hot exchanger",
        ),
    )
    for reference, predicted, color, label in forward_histories:
        forward_axis.plot(
            forward.time,
            reference,
            color=color,
            linewidth=1.5,
            label=f"{label} RK4",
        )
        forward_axis.plot(
            forward.time,
            predicted,
            color=color,
            linestyle="--",
            linewidth=1.2,
            label=f"{label} PINN",
        )
    forward_axis.set_title("Physics-only PINN: four states, no labels")
    forward_axis.set_ylabel("Temperature (K)")
    forward_axis.legend(fontsize="x-small", ncol=2)

    for reference, predicted, observed, color, label in (
        (
            inverse.reference_cold_face,
            inverse.predicted_cold_face,
            inverse.observed_cold_face,
            "tab:blue",
            "Cold face",
        ),
        (
            inverse.reference_cold_exchanger,
            inverse.predicted_cold_exchanger,
            inverse.observed_cold_exchanger,
            "tab:cyan",
            "Cold exchanger",
        ),
    ):
        observed_axis.plot(
            inverse.time,
            reference,
            color=color,
            linewidth=1.5,
            label=f"{label} RK4",
        )
        observed_axis.plot(
            inverse.time,
            predicted,
            color=color,
            linestyle="--",
            linewidth=1.2,
            label=f"{label} inverse PINN",
        )
        observed_axis.scatter(
            inverse.observation_time,
            observed,
            color="black",
            s=6,
            alpha=0.55,
            label=("1 s observations" if label == "Cold face" else None),
        )
    observed_axis.set_title("Inverse PINN: observed cold-side states")
    observed_axis.set_ylabel("Temperature (K)")
    observed_axis.legend(fontsize="x-small", ncol=2)

    for reference, predicted, color, label in (
        (
            inverse.reference_hot_face,
            inverse.predicted_hot_face,
            "tab:red",
            "Hot face",
        ),
        (
            inverse.reference_hot_exchanger,
            inverse.predicted_hot_exchanger,
            "tab:orange",
            "Hot exchanger",
        ),
    ):
        withheld_axis.plot(
            inverse.time,
            reference,
            color=color,
            linewidth=1.5,
            label=f"{label} withheld RK4",
        )
        withheld_axis.plot(
            inverse.time,
            predicted,
            color=color,
            linestyle="--",
            linewidth=1.2,
            label=f"{label} inverse PINN",
        )
    withheld_axis.set_title("Physics reconstructs two unobserved hot states")
    withheld_axis.set_ylabel("Temperature (K)")
    withheld_axis.legend(fontsize="small")

    resistance_axis.plot(
        inverse.epoch,
        inverse.inferred_resistance,
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
    resistance_axis.set_title("Inverse calibration from a 100% wrong start")
    resistance_axis.set_xlabel("Epoch")
    resistance_axis.set_ylabel("Cold contact resistance (K/W)")
    resistance_axis.legend(fontsize="small")

    error_values = (
        validation.cold_face_trajectory_rmse,
        validation.hot_face_trajectory_rmse,
        validation.cold_exchanger_trajectory_rmse,
        validation.hot_exchanger_trajectory_rmse,
        validation.validation_regime_metrics.all_sensor_rmse,
        validation.test_regime_metrics.all_sensor_rmse,
    )
    error_labels = ("CF", "HF*", "CX", "HX*", "Validation", "Bipolar test")
    colors = (
        "tab:blue",
        "tab:red",
        "tab:cyan",
        "tab:orange",
        "tab:green",
        "tab:purple",
    )
    evidence_axis.bar(error_labels, error_values, color=colors)
    evidence_axis.set_yscale("log")
    evidence_axis.set_title("Withheld-state and unseen-control evidence")
    evidence_axis.set_ylabel("Temperature RMSE (K, log scale)")
    evidence_axis.tick_params(axis="x", rotation=20)
    evidence_axis.text(
        0.98,
        0.96,
        "* no temperature labels\n"
        f"R: 0.500000 → {validation.inferred_cold_contact_resistance:.6f} K/W\n"
        f"truth: {validation.true_cold_contact_resistance:.6f} K/W  "
        f"({validation.relative_parameter_error_percent:.3f}% error)\n"
        f"maximum switch jump: {validation.max_boundary_temperature_jump:.1e} K",
        transform=evidence_axis.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )

    time_axes = (current_axis, forward_axis, observed_axis, withheld_axis)
    for axis in time_axes:
        _mark_switches(axis, forward.switch_times)
        axis.set_xlabel("Time (s)")
    for axis in axes.reshape(-1):
        axis.grid(True, alpha=0.25)

    figure.suptitle(
        "ThermoTwin PINN Showcase\n"
        "Physics-only prediction, inverse calibration, and transfer to unseen controls",
        fontsize=16,
    )
    figure.savefig(destination, dpi=170)
    save_figure_data(showcase, destination)
    return destination


def main() -> None:
    """Train both showcase PINNs and save the focused evidence figure."""

    parser = argparse.ArgumentParser(
        description="Run the ThermoTwin switched-current PINN showcase"
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_PINN_SHOWCASE_PATH,
        help=(
            "destination PNG path "
            "(default: thermotwin/figures/PINN_SHOWCASE/pinn_showcase.png)"
        ),
    )
    parser.add_argument("--forward-epochs", type=int, default=5_000)
    parser.add_argument("--inverse-epochs", type=int, default=8_000)
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "auto"),
        default="cpu",
    )
    arguments = parser.parse_args()

    showcase = train_pinn_showcase(
        PINNShowcaseConfig(
            forward_epochs=arguments.forward_epochs,
            inverse_epochs=arguments.inverse_epochs,
            device=arguments.device,
        )
    )
    destination = save_pinn_showcase(showcase, arguments.output)
    forward = showcase.forward.validation
    inverse = showcase.inverse.validation
    print("ThermoTwin PINN showcase")
    print("physics-only forward temperature labels: 0")
    print(
        "forward RMSE (K): "
        f"CF {forward.cold_face_rmse:.6f}, "
        f"HF {forward.hot_face_rmse:.6f}, "
        f"CX {forward.cold_exchanger_rmse:.6f}, "
        f"HX {forward.hot_exchanger_rmse:.6f}"
    )
    print("inverse observed sensors: 2 of 4")
    print(
        "cold contact resistance (K/W): "
        f"start 0.500000, PINN {inverse.inferred_cold_contact_resistance:.6f}, "
        f"truth {inverse.true_cold_contact_resistance:.6f}, "
        "conventional "
        f"{inverse.conventional_cold_contact_resistance:.6f}"
    )
    print(
        "unseen-control all-sensor RMSE (K): "
        f"validation {inverse.validation_regime_metrics.all_sensor_rmse:.6f}, "
        f"bipolar test {inverse.test_regime_metrics.all_sensor_rmse:.6f}"
    )
    print(f"report: {destination}")


if __name__ == "__main__":
    main()
