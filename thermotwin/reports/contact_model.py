"""Comparison report for the contact-aware and reduced two-node models."""

import argparse
from dataclasses import replace
import math
from pathlib import Path
from typing import NamedTuple, Sequence, Tuple, Union

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..simulation.four_node_experiments import (
    FourNodeContactExperiment,
    constant_current_contact_reference_experiment,
    run_four_node_contact_experiment,
)
from ..simulation.two_node_experiments import (
    TwoNodeExperiment,
    constant_current_reference_experiment,
    run_two_node_experiment,
)
from .paths import default_figure_path


DEFAULT_CONTACT_REPORT_PATH = default_figure_path(
    "contact_model_comparison.png"
)


class ContactResistanceSweepPoint(NamedTuple):
    """Final contact-model outputs at one symmetric resistance."""

    contact_resistance: float
    cold_face_temperature: float
    hot_face_temperature: float
    cold_exchanger_temperature: float
    hot_exchanger_temperature: float
    cold_contact_temperature_drop: float
    hot_contact_temperature_drop: float
    cold_contact_heat: float
    hot_contact_heat: float


class ContactComparisonReportData(NamedTuple):
    """Aligned histories and sweep outputs used by the contact report."""

    contact_time: Tuple[float, ...]
    two_node_time: Tuple[float, ...]
    cold_face_temperature: Tuple[float, ...]
    hot_face_temperature: Tuple[float, ...]
    cold_exchanger_temperature: Tuple[float, ...]
    hot_exchanger_temperature: Tuple[float, ...]
    two_node_cold_temperature: Tuple[float, ...]
    two_node_hot_temperature: Tuple[float, ...]
    cold_contact_temperature_drop: Tuple[float, ...]
    hot_contact_temperature_drop: Tuple[float, ...]
    cold_contact_heat: Tuple[float, ...]
    hot_contact_heat: Tuple[float, ...]
    cold_module_heat: Tuple[float, ...]
    hot_module_heat: Tuple[float, ...]
    sweep: Tuple[ContactResistanceSweepPoint, ...]
    max_absolute_energy_balance_residual: float


def build_contact_comparison_report_data(
    contact_experiment: FourNodeContactExperiment,
    two_node_experiment: TwoNodeExperiment,
    *,
    sweep_resistances: Sequence[float] = (0.1, 0.25, 0.5, 1.0),
) -> ContactComparisonReportData:
    """Run both topologies and a symmetric contact-resistance sweep."""

    resistances = tuple(float(value) for value in sweep_resistances)
    if not resistances:
        raise ValueError("at least one sweep resistance is required")
    if any(
        not math.isfinite(value) or value <= 0.0 for value in resistances
    ):
        raise ValueError("sweep resistances must be finite and positive")

    contact_result = run_four_node_contact_experiment(contact_experiment)
    two_node_result = run_two_node_experiment(two_node_experiment)
    sweep_points = []
    for resistance in resistances:
        sweep_parameters = replace(
            contact_experiment.thermal_parameters,
            cold_contact_resistance=resistance,
            hot_contact_resistance=resistance,
        )
        sweep_experiment = replace(
            contact_experiment,
            thermal_parameters=sweep_parameters,
        )
        sweep_result = run_four_node_contact_experiment(sweep_experiment)
        trajectory = sweep_result.trajectory
        diagnostics = sweep_result.diagnostics
        sweep_points.append(
            ContactResistanceSweepPoint(
                contact_resistance=resistance,
                cold_face_temperature=trajectory.cold_face[-1],
                hot_face_temperature=trajectory.hot_face[-1],
                cold_exchanger_temperature=trajectory.cold_exchanger[-1],
                hot_exchanger_temperature=trajectory.hot_exchanger[-1],
                cold_contact_temperature_drop=(
                    diagnostics.cold_contact_temperature_drop[-1]
                ),
                hot_contact_temperature_drop=(
                    diagnostics.hot_contact_temperature_drop[-1]
                ),
                cold_contact_heat=diagnostics.cold_contact_heat[-1],
                hot_contact_heat=diagnostics.hot_contact_heat[-1],
            )
        )

    contact_trajectory = contact_result.trajectory
    contact_diagnostics = contact_result.diagnostics
    two_node_trajectory = two_node_result.trajectory
    return ContactComparisonReportData(
        contact_time=contact_trajectory.time,
        two_node_time=two_node_trajectory.time,
        cold_face_temperature=contact_trajectory.cold_face,
        hot_face_temperature=contact_trajectory.hot_face,
        cold_exchanger_temperature=contact_trajectory.cold_exchanger,
        hot_exchanger_temperature=contact_trajectory.hot_exchanger,
        two_node_cold_temperature=two_node_trajectory.cold,
        two_node_hot_temperature=two_node_trajectory.hot,
        cold_contact_temperature_drop=(
            contact_diagnostics.cold_contact_temperature_drop
        ),
        hot_contact_temperature_drop=(
            contact_diagnostics.hot_contact_temperature_drop
        ),
        cold_contact_heat=contact_diagnostics.cold_contact_heat,
        hot_contact_heat=contact_diagnostics.hot_contact_heat,
        cold_module_heat=contact_diagnostics.cold_heat,
        hot_module_heat=contact_diagnostics.hot_heat,
        sweep=tuple(sweep_points),
        max_absolute_energy_balance_residual=max(
            abs(value)
            for value in contact_diagnostics.energy_balance_residual
        ),
    )


def save_contact_comparison_report(
    report: ContactComparisonReportData,
    output_path: Union[str, Path],
) -> Path:
    """Save a four-panel contact-model PNG and return its resolved path."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure = Figure(figsize=(12.0, 9.0), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    temperature_axis = axes[0, 0]
    drop_axis = axes[0, 1]
    heat_axis = axes[1, 0]
    sweep_axis = axes[1, 1]

    temperature_axis.plot(
        report.contact_time,
        report.cold_face_temperature,
        color="tab:blue",
        label="Cold TE face",
    )
    temperature_axis.plot(
        report.contact_time,
        report.cold_exchanger_temperature,
        color="tab:cyan",
        label="Cold exchanger",
    )
    temperature_axis.plot(
        report.contact_time,
        report.hot_face_temperature,
        color="tab:red",
        label="Hot TE face",
    )
    temperature_axis.plot(
        report.contact_time,
        report.hot_exchanger_temperature,
        color="tab:orange",
        label="Hot exchanger",
    )
    temperature_axis.plot(
        report.two_node_time,
        report.two_node_cold_temperature,
        color="tab:blue",
        linestyle="--",
        alpha=0.7,
        label="Two-node cold",
    )
    temperature_axis.plot(
        report.two_node_time,
        report.two_node_hot_temperature,
        color="tab:red",
        linestyle="--",
        alpha=0.7,
        label="Two-node hot",
    )
    temperature_axis.set_title("Temperature trajectories")
    temperature_axis.set_xlabel("Time (s)")
    temperature_axis.set_ylabel("Temperature (K)")
    temperature_axis.legend(fontsize="small")

    drop_axis.plot(
        report.contact_time,
        report.cold_contact_temperature_drop,
        color="tab:blue",
        label="Cold exchanger minus face",
    )
    drop_axis.plot(
        report.contact_time,
        report.hot_contact_temperature_drop,
        color="tab:red",
        label="Hot face minus exchanger",
    )
    drop_axis.axhline(0.0, color="0.3", linewidth=0.8)
    drop_axis.set_title("Contact temperature drops")
    drop_axis.set_xlabel("Time (s)")
    drop_axis.set_ylabel("Temperature drop (K)")
    drop_axis.legend()

    heat_axis.plot(
        report.contact_time,
        report.cold_module_heat,
        color="tab:blue",
        label="Module Qc",
    )
    heat_axis.plot(
        report.contact_time,
        report.cold_contact_heat,
        color="tab:cyan",
        linestyle="--",
        label="Cold contact heat",
    )
    heat_axis.plot(
        report.contact_time,
        report.hot_module_heat,
        color="tab:red",
        label="Module Qh",
    )
    heat_axis.plot(
        report.contact_time,
        report.hot_contact_heat,
        color="tab:orange",
        linestyle="--",
        label="Hot contact heat",
    )
    heat_axis.set_title("Module and delivered heat rates")
    heat_axis.set_xlabel("Time (s)")
    heat_axis.set_ylabel("Heat rate (W)")
    heat_axis.legend(fontsize="small")

    sweep_resistances = tuple(
        point.contact_resistance for point in report.sweep
    )
    sweep_axis.plot(
        sweep_resistances,
        tuple(point.cold_face_temperature for point in report.sweep),
        marker="o",
        color="tab:blue",
        label="Cold TE face",
    )
    sweep_axis.plot(
        sweep_resistances,
        tuple(point.cold_exchanger_temperature for point in report.sweep),
        marker="o",
        color="tab:cyan",
        label="Cold exchanger",
    )
    sweep_axis.plot(
        sweep_resistances,
        tuple(point.hot_face_temperature for point in report.sweep),
        marker="o",
        color="tab:red",
        label="Hot TE face",
    )
    sweep_axis.plot(
        sweep_resistances,
        tuple(point.hot_exchanger_temperature for point in report.sweep),
        marker="o",
        color="tab:orange",
        label="Hot exchanger",
    )
    sweep_axis.set_title("Final temperatures versus symmetric contact R")
    sweep_axis.set_xlabel("Contact resistance on each side (K/W)")
    sweep_axis.set_ylabel("Final temperature (K)")
    sweep_axis.legend(fontsize="small")

    for axis in axes.reshape(-1):
        axis.grid(True, alpha=0.25)

    figure.suptitle(
        "Contact-aware reference model\n"
        "Maximum energy-balance residual "
        f"{report.max_absolute_energy_balance_residual:.3e} W"
    )
    figure.savefig(destination, dpi=150)
    return destination


def main() -> None:
    """Run the contact reference and write its comparison report."""

    parser = argparse.ArgumentParser(
        description="Compare ThermoTwin contact-aware and two-node models"
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_CONTACT_REPORT_PATH,
        help=(
            "destination PNG path (default: "
            "thermotwin/figures/contact_model_comparison.png)"
        ),
    )
    parser.add_argument(
        "--resistances",
        type=float,
        nargs="+",
        default=(0.1, 0.25, 0.5, 1.0),
        help="symmetric contact resistances for the final-state sweep",
    )
    arguments = parser.parse_args()

    report = build_contact_comparison_report_data(
        constant_current_contact_reference_experiment(),
        constant_current_reference_experiment(),
        sweep_resistances=arguments.resistances,
    )
    output_path = save_contact_comparison_report(report, arguments.output)
    print(
        "final contact temperatures: "
        f"cold face {report.cold_face_temperature[-1]:.6f} K, "
        f"cold exchanger {report.cold_exchanger_temperature[-1]:.6f} K, "
        f"hot face {report.hot_face_temperature[-1]:.6f} K, "
        f"hot exchanger {report.hot_exchanger_temperature[-1]:.6f} K"
    )
    print(
        "final contact drops: "
        f"cold {report.cold_contact_temperature_drop[-1]:.6f} K, "
        f"hot {report.hot_contact_temperature_drop[-1]:.6f} K"
    )
    print(
        "maximum energy-balance residual: "
        f"{report.max_absolute_energy_balance_residual:.6e} W"
    )
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
