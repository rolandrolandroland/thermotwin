"""Report the distributed inverse-PINN budget and curve-shape audit."""

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

from ..figure_paths import default_figure_path
from ..studies.distributed_pinn_training_audit import (
    DistributedPINNTrainingAuditConfig,
    DistributedPINNTrainingAuditResult,
    run_distributed_pinn_training_audit,
)


def _multipliers(values: Sequence[float]) -> str:
    return "(" + ", ".join(f"{value:.6f}" for value in values) + ")"


def format_distributed_pinn_training_audit_report(
    result: DistributedPINNTrainingAuditResult,
) -> str:
    """Print truth-blind training checks separately from truth-known recovery."""

    config = result.config
    lines = [
        "Distributed inverse PINN: training and curve-shape audit",
        "=======================================================",
        "",
        "Question:",
        "  Does a small knot error reflect a physics-satisfying recovery of",
        "  rho_e(T), or only an undertrained, nearly constant curve?",
        "",
        "Frozen protocol:",
        f"  trials: {config.trial_count}; checkpoints: {config.checkpoint_epochs}",
        f"  truth multipliers: {_multipliers(result.truth_multipliers)}",
        f"  temperature noise: {config.temperature_standard_deviation:.6f} K",
        f"  voltage noise: {config.voltage_standard_deviation:.6e} V",
        f"  physics-loss weight: {config.physics_weight:.3f}",
        f"  nominal reference temperature-rate RMS: "
        f"{result.reference_temperature_rate_rms:.6f} K/s",
        "  each seed is trained once to the largest budget; earlier entries are",
        "  checkpoints from that uninterrupted deterministic Adam trajectory.",
        "",
        "Truth-blind operational gate:",
        f"  normalized observation loss <= "
        f"{config.maximum_normalized_observation_loss:.3f}",
        f"  PDE/boundary residual RMS <= "
        f"{config.maximum_physics_residual_ratio:.2%} of the nominal rate RMS",
        "  this gate can choose a usable checkpoint without inspecting truth.",
        "",
        "Truth-known benchmark diagnostics:",
        f"  amplitude and center-contrast ratios must each lie in "
        f"[{config.minimum_shape_ratio:.2f}, {config.maximum_shape_ratio:.2f}]",
        "  and coefficient RMSE must beat the best constant curve.",
        "  These diagnostics judge synthetic recovery; they are not stopping rules.",
        "",
        "Trials:",
    ]
    for trial in result.trials:
        lines.append(
            f"  trial {trial.trial_index}: neural seed={trial.seeds.neural}; "
            f"noise seeds={trial.seeds.observations}; "
            f"first operational epoch={trial.first_operational_epoch}"
        )
        for checkpoint in trial.checkpoints:
            lines.append(
                f"    epoch {checkpoint.epoch}: multipliers="
                f"{_multipliers(checkpoint.multipliers)}; observation="
                f"{checkpoint.observation_loss:.6f}; physics RMS="
                f"{checkpoint.physics_residual_rms:.6f} K/s "
                f"({checkpoint.physics_residual_ratio:.2%}); coefficient RMSE="
                f"{checkpoint.shape.coefficient_rmse:.6f}; amplitude ratio="
                f"{checkpoint.shape.amplitude_ratio:.3f}; contrast ratio="
                f"{checkpoint.shape.center_contrast_ratio:.3f}; "
                f"operational={'PASS' if checkpoint.operationally_acceptable else 'FAIL'}"
                f" {checkpoint.operational_failure_reasons or ('none',)}; "
                f"shape={'PASS' if checkpoint.shape_recovered else 'FAIL'} "
                f"{checkpoint.shape_failure_reasons or ('none',)}"
            )
    lines.extend(("", "Checkpoint summary:"))
    for summary in result.epoch_summaries:
        lines.append(
            f"  epoch {summary.epoch}: operational="
            f"{summary.operationally_acceptable_count}/{summary.completed_count}; "
            f"shape={summary.shape_recovered_count}/{summary.completed_count}; "
            f"mean observation={summary.mean_observation_loss:.6f}; "
            f"mean residual ratio={summary.mean_physics_residual_ratio:.2%}; "
            f"mean coefficient RMSE={summary.mean_coefficient_rmse:.6f}; "
            f"mean amplitude ratio={summary.mean_amplitude_ratio:.3f}; "
            f"mean contrast ratio={summary.mean_center_contrast_ratio:.3f}"
        )
    lines.extend(
        (
            "",
            "Interpretation boundary:",
            "  This is an optimization audit under same-equation, same-basis",
            "  synthetic truth. It intentionally isolates training behavior and",
            "  does not replace the independent nodal/SSPRK3 truth study.",
            "  The physical RMS combines the interior PDE and both dynamic face",
            "  balances. Dividing by the nominal reference-rate RMS makes its scale",
            "  interpretable, but the 25% limit is a declared engineering check,",
            "  not a theorem or a hardware-validation threshold.",
            "  Point error, average level, curve amplitude, and center curvature",
            "  are reported separately; no nearly flat curve can pass merely by",
            "  lying inside a broad per-knot tolerance.",
        )
    )
    return "\n".join(lines)


def save_distributed_pinn_training_audit_figure(
    result: DistributedPINNTrainingAuditResult,
    output: Path | str,
) -> Path:
    """Plot loss-scale convergence separately from curve-shape recovery."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    epochs = result.config.checkpoint_epochs

    for trial in result.trials:
        axes[0, 0].plot(
            epochs,
            tuple(item.observation_loss for item in trial.checkpoints),
            marker="o",
            alpha=0.75,
            label=f"seed {trial.seeds.neural}",
        )
        axes[0, 1].plot(
            epochs,
            tuple(item.physics_residual_ratio for item in trial.checkpoints),
            marker="o",
            alpha=0.75,
        )
        axes[1, 0].plot(
            epochs,
            tuple(item.shape.amplitude_ratio for item in trial.checkpoints),
            marker="o",
            alpha=0.75,
        )
        axes[1, 1].plot(
            epochs,
            tuple(item.shape.center_contrast_ratio for item in trial.checkpoints),
            marker="o",
            alpha=0.75,
        )

    axes[0, 0].axhline(
        result.config.maximum_normalized_observation_loss,
        color="red",
        linestyle="--",
        label="operational limit",
    )
    axes[0, 0].set_title("Normalized observation fit")
    axes[0, 0].set_ylabel("Mean normalized squared error")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].axhline(
        result.config.maximum_physics_residual_ratio,
        color="red",
        linestyle="--",
        label="operational limit",
    )
    axes[0, 1].set_title("Physics residual relative to nominal dynamics")
    axes[0, 1].set_ylabel("Residual RMS / nominal rate RMS")
    axes[0, 1].legend(fontsize=8)

    for axis, title in (
        (axes[1, 0], "Recovered curve amplitude"),
        (axes[1, 1], "Recovered center contrast"),
    ):
        axis.axhspan(
            result.config.minimum_shape_ratio,
            result.config.maximum_shape_ratio,
            color="green",
            alpha=0.12,
            label="declared shape band",
        )
        axis.axhline(1.0, color="black", linestyle=":", label="truth")
        axis.set_title(title)
        axis.set_ylabel("Recovered / truth")
        axis.legend(fontsize=8)
    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.set_xscale("log")
        axis.set_xticks(epochs, labels=tuple(str(value) for value in epochs))
        axis.grid(alpha=0.25)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def _checkpoints(value: str) -> Tuple[int, ...]:
    try:
        checkpoints = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("checkpoints must be comma-separated integers") from error
    if not checkpoints:
        raise argparse.ArgumentTypeError("at least one checkpoint is required")
    return checkpoints


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Audit distributed inverse-PINN convergence and curve shape."
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--first-seed", type=int, default=63_001)
    parser.add_argument("--physics-weight", type=float, default=10.0)
    parser.add_argument("--checkpoints", type=_checkpoints, default=(600, 1_200, 2_400))
    parser.add_argument(
        "--output",
        type=Path,
        default=default_figure_path("distributed_pinn_training_audit.png"),
    )
    args = parser.parse_args(argv)
    result = run_distributed_pinn_training_audit(
        DistributedPINNTrainingAuditConfig(
            trial_count=args.trials,
            first_seed=args.first_seed,
            checkpoint_epochs=args.checkpoints,
            physics_weight=args.physics_weight,
        )
    )
    output = save_distributed_pinn_training_audit_figure(result, args.output)
    print(format_distributed_pinn_training_audit_report(result))
    print(f"figure: {output}")


if __name__ == "__main__":
    main()
