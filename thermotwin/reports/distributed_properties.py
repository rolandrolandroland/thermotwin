"""Reproducible report for distributed-property identifiability and PINNs."""

import argparse
from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Optional, Sequence, Tuple

from ..figure_paths import default_figure_path
from ..inference.distributed_experiment_selection import (
    DistributedExperimentSelectionConfig,
    DistributedExperimentSelectionResult,
    DistributedLinearizedUncertainty,
    linearized_distributed_uncertainty,
    select_distributed_experiment,
)
from ..inference.distributed_identifiability import (
    DistributedIdentifiabilityConfig,
    DistributedIdentifiabilityResult,
    DistributedPropertyCoefficient,
    analyze_distributed_identifiability,
)
from ..simulation.distributed import (
    DistributedExperimentResult,
    distributed_identifiability_experiments,
    distributed_reference_experiment,
    run_distributed_leg_experiment,
)


@dataclass(frozen=True)
class DistributedPropertyStudyResult:
    reference: DistributedExperimentResult
    family_identifiability: Tuple[Tuple[str, DistributedIdentifiabilityResult], ...]
    joint_identifiability: DistributedIdentifiabilityResult
    joint_uncertainty: DistributedLinearizedUncertainty
    selection: DistributedExperimentSelectionResult
    forward_pinn_validation: Optional[object]
    forward_pinn_initial_loss: Optional[float]
    forward_pinn_final_loss: Optional[float]
    inverse_truth_multipliers: Optional[Tuple[float, ...]]
    conventional_inverse_multipliers: Optional[Tuple[float, ...]]
    inverse_pinn_multipliers: Optional[Tuple[float, ...]]
    inverse_pinn_initial_loss: Optional[float]
    inverse_pinn_final_loss: Optional[float]


def _property_parameters(property_name: str) -> Tuple[DistributedPropertyCoefficient, ...]:
    return tuple(
        DistributedPropertyCoefficient(property_name, index)
        for index in range(3)
    )


def run_distributed_property_study(
    *,
    run_forward_pinn: bool = False,
    pinn_epochs: int = 800,
    run_inverse_pinn: bool = False,
    inverse_pinn_epochs: int = 800,
    quick: bool = False,
) -> DistributedPropertyStudyResult:
    """Run the conventional gate, next-experiment ranking, and optional PINN."""

    reference_experiment = distributed_reference_experiment(
        temperature_dependent=True,
        current=0.8,
        duration=0.8,
        cell_count=8,
        time_step=0.0015,
    )
    reference = run_distributed_leg_experiment(reference_experiment)
    experiments = distributed_identifiability_experiments()
    if quick:
        experiments = experiments[:2]
    identifiability_config = DistributedIdentifiabilityConfig(
        observation_interval=0.2 if quick else 0.1
    )
    family_results = tuple(
        (
            property_name,
            analyze_distributed_identifiability(
                experiments,
                _property_parameters(property_name),
                identifiability_config,
            ),
        )
        for property_name in (
            "seebeck_coefficient",
            "electrical_resistivity",
            "thermal_conductivity",
        )
    )
    joint_parameters = tuple(
        parameter
        for property_name, _ in family_results
        for parameter in _property_parameters(property_name)
    )
    joint = analyze_distributed_identifiability(
        experiments, joint_parameters, identifiability_config
    )
    uncertainty = linearized_distributed_uncertainty(
        joint_parameters,
        joint.information_matrix,
        prior_log_standard_deviation=0.20,
    )
    selection = select_distributed_experiment(
        _property_parameters("electrical_resistivity"),
        DistributedExperimentSelectionConfig(
            current_amplitudes=(0.4, 0.8) if quick else (-0.8, 0.8),
            pulse_durations=(0.2,) if quick else (0.2, 0.5),
            reservoir_temperature_lifts=(0.0,) if quick else (0.0, 10.0, 20.0),
            identifiability=identifiability_config,
        ),
    )

    validation = None
    initial_loss = None
    final_loss = None
    if run_forward_pinn:
        from ..pinn.distributed_forward import (
            DistributedForwardPINNConfig,
            train_distributed_forward_pinn,
            validate_distributed_forward_pinn,
        )

        if pinn_epochs <= 0:
            raise ValueError("PINN epochs must be positive when training is enabled")
        pinn_experiment = distributed_reference_experiment(
            temperature_dependent=True,
            current=0.5,
            duration=0.2,
            cell_count=6,
            time_step=0.001,
        )
        training = train_distributed_forward_pinn(
            pinn_experiment,
            DistributedForwardPINNConfig(
                hidden_width=32,
                hidden_layers=3,
                interior_space_points=10,
                time_points=32,
                epochs=pinn_epochs,
                learning_rate=2.0e-3,
                seed=21,
                device="cpu",
            ),
        )
        validation = validate_distributed_forward_pinn(
            training.model, pinn_experiment
        )
        initial_loss = training.history.total_loss[0]
        final_loss = training.history.total_loss[-1]
    inverse_truth = None
    conventional_inverse = None
    inverse_pinn = None
    inverse_initial_loss = None
    inverse_final_loss = None
    if run_inverse_pinn:
        from ..inference.distributed_properties import (
            DistributedPropertyFitConfig,
            fit_distributed_property,
        )
        from ..observations.distributed import (
            DistributedObservationChannels,
            run_distributed_virtual_experiment,
        )
        from ..pinn.distributed_inverse import (
            InverseDistributedPropertyConfig,
            train_multi_experiment_inverse_distributed_property_pinn,
        )
        from ..simulation.distributed import (
            distributed_inverse_constant_experiments,
        )

        if inverse_pinn_epochs <= 0:
            raise ValueError("inverse PINN epochs must be positive")
        baseline_experiments = distributed_inverse_constant_experiments()
        baseline_property = (
            baseline_experiments[0].material.electrical_resistivity
        )
        inverse_truth = (1.04, 1.07, 1.03)
        truth_property = baseline_property.with_values(
            tuple(
                value * multiplier
                for value, multiplier in zip(
                    baseline_property.values, inverse_truth
                )
            )
        )
        truth_experiments = tuple(
            replace(
                experiment,
                material=replace(
                    experiment.material,
                    electrical_resistivity=truth_property,
                ),
            )
            for experiment in baseline_experiments
        )
        inverse_channels = DistributedObservationChannels()
        inverse_observations = tuple(
            run_distributed_virtual_experiment(
                experiment,
                observation_interval=0.08,
                channels=inverse_channels,
            )
            for experiment in truth_experiments
        )
        conventional_fit = fit_distributed_property(
            baseline_experiments,
            inverse_observations,
            DistributedPropertyFitConfig(
                property_name="electrical_resistivity",
                observation_interval=0.08,
                channels=inverse_channels,
                initial_log_multipliers=(math.log(0.9),) * 3,
                log_multiplier_bounds=(-0.2, 0.2),
                coordinate_passes=2,
                golden_section_iterations=8,
                gauss_newton_iterations=6,
            ),
        )
        conventional_inverse = tuple(
            value / baseline
            for value, baseline in zip(
                conventional_fit.fitted_values, baseline_property.values
            )
        )
        inverse_training = (
            train_multi_experiment_inverse_distributed_property_pinn(
                truth_experiments,
                inverse_observations,
                InverseDistributedPropertyConfig(
                    property_name="electrical_resistivity",
                    hidden_width=20,
                    hidden_layers=3,
                    interior_space_points=7,
                    time_points=18,
                    voltage_space_points=16,
                    epochs=inverse_pinn_epochs,
                    network_learning_rate=2.0e-3,
                    property_learning_rate=2.0e-3,
                    initial_log_multipliers=(math.log(0.9),) * 3,
                    smoothness_weight=1.0e-4,
                    seed=19,
                    device="cpu",
                ),
                baseline_material=baseline_experiments[0].material,
            )
        )
        inverse_pinn = tuple(
            value / baseline
            for value, baseline in zip(
                inverse_training.history.property_values[-1],
                baseline_property.values,
            )
        )
        inverse_initial_loss = inverse_training.history.total_loss[0]
        inverse_final_loss = inverse_training.history.total_loss[-1]
    return DistributedPropertyStudyResult(
        reference=reference,
        family_identifiability=family_results,
        joint_identifiability=joint,
        joint_uncertainty=uncertainty,
        selection=selection,
        forward_pinn_validation=validation,
        forward_pinn_initial_loss=initial_loss,
        forward_pinn_final_loss=final_loss,
        inverse_truth_multipliers=inverse_truth,
        conventional_inverse_multipliers=conventional_inverse,
        inverse_pinn_multipliers=inverse_pinn,
        inverse_pinn_initial_loss=inverse_initial_loss,
        inverse_pinn_final_loss=inverse_final_loss,
    )


def format_distributed_property_study(
    result: DistributedPropertyStudyResult,
) -> str:
    lines = [
        "Distributed constitutive-inference study",
        "========================================",
        "",
        "Local identifiability (declared synthetic sensors/noise):",
    ]
    for name, analysis in result.family_identifiability:
        singular = ", ".join(f"{value:.4g}" for value in analysis.singular_values)
        lines.append(
            f"  {name}: rank {analysis.effective_rank}/3; "
            f"condition {analysis.condition_number:.3g}; singular values {singular}"
        )
    joint = result.joint_identifiability
    lines.extend(
        (
            f"  joint: rank {joint.effective_rank}/9; "
            f"condition {joint.condition_number:.3g}",
            f"  explored temperature range: {joint.temperature_range[0]:.3f} to "
            f"{joint.temperature_range[1]:.3f} K",
            "",
            "Selected rho_e(T) experiment:",
            f"  {result.selection.selected.name}",
            f"  information gain: "
            f"{result.selection.selected.information_gain_nats:.4f} nats",
            f"  candidate count: {len(result.selection.candidates)}",
        )
    )
    if result.forward_pinn_validation is None:
        lines.extend(("", "Forward PINN: not trained (use --train-pinn)."))
    else:
        validation = result.forward_pinn_validation
        lines.extend(
            (
                "",
                "Forward distributed PINN versus withheld finite volume:",
                f"  physics loss: {result.forward_pinn_initial_loss:.6e} -> "
                f"{result.forward_pinn_final_loss:.6e}",
                f"  cold-face RMSE: {validation.cold_face_rmse:.6f} K",
                f"  hot-face RMSE: {validation.hot_face_rmse:.6f} K",
                f"  internal-field RMSE: "
                f"{validation.internal_temperature_rmse:.6f} K",
                f"  maximum absolute error: "
                f"{validation.maximum_absolute_temperature_error:.6f} K",
            )
        )
    if result.inverse_pinn_multipliers is None:
        lines.extend(("", "Inverse distributed PINN: not trained (use --train-inverse-pinn)."))
    else:
        lines.extend(
            (
                "",
                "Shared rho_e(T) inverse across four constant regimes:",
                "  truth multipliers: "
                + ", ".join(
                    f"{value:.6f}" for value in result.inverse_truth_multipliers
                ),
                "  conventional multipliers: "
                + ", ".join(
                    f"{value:.6f}"
                    for value in result.conventional_inverse_multipliers
                ),
                "  inverse-PINN multipliers: "
                + ", ".join(
                    f"{value:.6f}" for value in result.inverse_pinn_multipliers
                ),
                f"  inverse-PINN loss: {result.inverse_pinn_initial_loss:.6e} -> "
                f"{result.inverse_pinn_final_loss:.6e}",
                "  conventional exactness is same-model/noise-free and is not a "
                "hardware claim.",
            )
        )
    lines.extend(
        (
            "",
            "Interpretation boundary:",
            "  These ranks and intervals are local to the synthetic model,",
            "  experiment suite, sensor channels, and assumed noise. They are a",
            "  training gate, not evidence that hardware identifies all curves.",
        )
    )
    return "\n".join(lines)


def save_distributed_property_figure(
    result: DistributedPropertyStudyResult,
    output: Path | str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    reference = result.reference
    profile_axis = axes[0, 0]
    cell_count = len(reference.trajectory.cells[0])
    positions = tuple((index + 0.5) / cell_count for index in range(cell_count))
    profile_indices = (0, len(reference.trajectory.time) // 2, -1)
    for index in profile_indices:
        profile_axis.plot(
            (0.0,) + positions + (1.0,),
            (
                reference.trajectory.cold_face[index],
                *reference.trajectory.cells[index],
                reference.trajectory.hot_face[index],
            ),
            label=f"t={reference.trajectory.time[index]:.3g} s",
        )
    profile_axis.set_title("Hidden 1-D temperature field")
    profile_axis.set_xlabel("Normalized cold-to-hot position")
    profile_axis.set_ylabel("Temperature (K)")
    profile_axis.legend()
    profile_axis.grid(alpha=0.25)

    history_axis = axes[0, 1]
    history_axis.plot(
        reference.trajectory.time,
        reference.diagnostics.cold_side_heat,
        label="Q_c",
    )
    history_axis.plot(
        reference.trajectory.time,
        reference.diagnostics.hot_side_heat,
        label="Q_h",
    )
    history_axis.plot(
        reference.trajectory.time,
        reference.diagnostics.electrical_power,
        label="VI",
    )
    history_axis.set_title("Heat and electrical-power histories")
    history_axis.set_xlabel("Time (s)")
    history_axis.set_ylabel("Power (W)")
    history_axis.legend()
    history_axis.grid(alpha=0.25)

    singular_axis = axes[1, 0]
    for name, analysis in result.family_identifiability:
        singular_axis.plot(
            range(1, len(analysis.singular_values) + 1),
            analysis.singular_values,
            marker="o",
            label=name.replace("_", " "),
        )
    singular_axis.set_yscale("log")
    singular_axis.set_title("Noise-normalized local singular spectrum")
    singular_axis.set_xlabel("Singular-value index")
    singular_axis.set_ylabel("Singular value")
    singular_axis.legend(fontsize=8)
    singular_axis.grid(True, which="both", alpha=0.25)

    selection_axis = axes[1, 1]
    feasible = tuple(score for score in result.selection.candidates if score.feasible)
    selection_axis.scatter(
        tuple(score.reservoir_temperature_lift for score in feasible),
        tuple(score.information_gain_nats for score in feasible),
        c=tuple(abs(score.current_amplitude) for score in feasible),
        s=tuple(100.0 * score.pulse_duration for score in feasible),
        cmap="viridis",
        alpha=0.75,
    )
    chosen = result.selection.selected
    selection_axis.scatter(
        (chosen.reservoir_temperature_lift,),
        (chosen.information_gain_nats,),
        marker="*",
        s=220,
        color="red",
        label="selected",
    )
    selection_axis.set_title("Feasible next-experiment candidates")
    selection_axis.set_xlabel("Reservoir temperature lift (K)")
    selection_axis.set_ylabel("Information gain (nats)")
    selection_axis.legend()
    selection_axis.grid(alpha=0.25)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the distributed thermoelectric property study."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_figure_path("distributed_property_study.png"),
    )
    parser.add_argument(
        "--train-pinn",
        action="store_true",
        help="also train and validate the CPU forward distributed PINN",
    )
    parser.add_argument("--pinn-epochs", type=int, default=800)
    parser.add_argument(
        "--train-inverse-pinn",
        action="store_true",
        help="also compare conventional and shared-property inverse recovery",
    )
    parser.add_argument("--inverse-pinn-epochs", type=int, default=800)
    args = parser.parse_args(argv)
    result = run_distributed_property_study(
        run_forward_pinn=args.train_pinn,
        pinn_epochs=args.pinn_epochs,
        run_inverse_pinn=args.train_inverse_pinn,
        inverse_pinn_epochs=args.inverse_pinn_epochs,
    )
    output = save_distributed_property_figure(result, args.output)
    print(format_distributed_property_study(result))
    print(f"figure: {output}")


if __name__ == "__main__":
    main()
