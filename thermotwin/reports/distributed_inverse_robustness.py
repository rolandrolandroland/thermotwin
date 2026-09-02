"""Report and figure for noisy multi-seed distributed inverse recovery."""

import argparse
import math
from pathlib import Path
from typing import Optional, Sequence

from ..figure_paths import default_figure_path, save_figure_data
from ..studies.distributed_inverse_robustness import (
    DistributedInverseRobustnessConfig,
    DistributedInverseRobustnessStudyResult,
    run_distributed_inverse_robustness_study,
)


def _multipliers_text(values) -> str:
    if values is None:
        return "training failed"
    return "(" + ", ".join(f"{value:.6f}" for value in values) + ")"


def _optional_float_text(value: Optional[float]) -> str:
    return "not completed" if value is None else f"{value:.6f}"


def format_distributed_inverse_robustness_report(
    result: DistributedInverseRobustnessStudyResult,
) -> str:
    """Report fixed criteria, every trial, and aggregate results."""

    config = result.config
    criteria = config.criteria
    lines = [
        "Distributed resistivity inverse: noisy multi-seed study",
        "========================================================",
        "",
        "Declared problem:",
        "  released curve: rho_e(T) at 285, 300, and 315 K",
        f"  truth multipliers: {_multipliers_text(result.truth_multipliers)}",
        "  visible channels: cold-face temperature, hot-face temperature, voltage",
        f"  temperature noise standard deviation: "
        f"{config.temperature_standard_deviation:.6f} K",
        f"  voltage noise standard deviation: "
        f"{config.voltage_standard_deviation:.6e} V",
        f"  trials: {config.trial_count}; inverse-PINN epochs per trial: "
        f"{config.inverse_pinn_epochs}",
        "",
        "Predeclared success gate:",
        f"  maximum absolute multiplier error <= "
        f"{criteria.maximum_absolute_multiplier_error:.6f}",
        f"  loss reduction fraction >= "
        f"{criteria.minimum_loss_reduction_fraction:.3f}",
        f"  final normalized observation loss <= "
        f"{criteria.maximum_final_normalized_loss:.3f}",
        "  the comparable gate uses observation loss for both estimators;",
        "  PINN physics loss and total objective are reported separately.",
        "  all criteria must pass; no failed trial is dropped.",
        "",
        "Trials:",
    ]
    for trial in result.trials:
        conventional_status = "PASS" if trial.conventional_success else "FAIL"
        pinn_status = "PASS" if trial.inverse_pinn_success else "FAIL"
        lines.extend(
            (
                f"  trial {trial.trial_index}: neural seed={trial.seeds.neural}; "
                f"noise seeds={trial.seeds.observations}",
                f"    conventional: {conventional_status}; multipliers="
                f"{_multipliers_text(trial.conventional_multipliers)}; "
                f"max error={trial.conventional_maximum_absolute_multiplier_error:.6f}; "
                f"loss={trial.conventional_final_normalized_loss:.6f}; "
                f"reduction={trial.conventional_loss_reduction_fraction:.6f}; "
                f"search bound hit={trial.conventional_reached_search_bound}; "
                f"reasons={trial.conventional_failure_reasons or ('none',)}",
                f"    inverse PINN: {pinn_status}; multipliers="
                f"{_multipliers_text(trial.inverse_pinn_multipliers)}; "
                f"max error={trial.inverse_pinn_maximum_absolute_multiplier_error:.6f}; "
                f"observation loss="
                f"{_optional_float_text(trial.inverse_pinn_final_observation_loss)}; "
                f"observation reduction="
                f"{_optional_float_text(trial.inverse_pinn_observation_loss_reduction_fraction)}; "
                f"physics loss="
                f"{_optional_float_text(trial.inverse_pinn_final_physics_loss)}; "
                f"total objective="
                f"{_optional_float_text(trial.inverse_pinn_final_normalized_loss)}; "
                f"reasons={trial.inverse_pinn_failure_reasons or ('none',)}",
            )
        )
    summary = result.summary
    lines.extend(
        (
            "",
            "Summary:",
            f"  conventional successes: {summary.conventional_success_count}/"
            f"{summary.trial_count}",
            f"  inverse-PINN successes: {summary.inverse_pinn_success_count}/"
            f"{summary.trial_count}",
            f"  inverse-PINN completed trials: "
            f"{summary.inverse_pinn_completed_count}/{summary.trial_count}",
            f"  conventional search-bound hits: "
            f"{summary.conventional_search_bound_hits}",
            f"  conventional mean multipliers: "
            f"{_multipliers_text(summary.conventional_mean_multipliers)}",
            f"  inverse-PINN mean multipliers: "
            f"{_multipliers_text(summary.inverse_pinn_mean_multipliers)}",
            f"  conventional coefficient RMSE: "
            f"{summary.conventional_multiplier_rmse:.6f}",
            f"  inverse-PINN coefficient RMSE: "
            f"{_optional_float_text(summary.inverse_pinn_multiplier_rmse)}",
            f"  conventional worst trial error: "
            f"{summary.conventional_worst_trial_error:.6f}",
            f"  inverse-PINN worst trial error: "
            f"{summary.inverse_pinn_worst_trial_error:.6f}",
            "",
            "Interpretation boundary:",
            "  Noise and neural initialization both vary, but truth and inference",
            "  still use the same finite-volume equations and property basis.",
            "  The conventional fit has no explicit smoothness penalty; the PINN",
            "  has explicit and implicit regularization, so this is not a matched-",
            "  prior estimator-superiority experiment.",
            "  This is a repeatability test, not independent-model or hardware",
            f"  validation. {config.trial_count} trial(s) estimate failures poorly;",
            "  quote every trial rather than treating the observed pass fraction",
            "  as a population rate.",
        )
    )
    return "\n".join(lines)


def save_distributed_inverse_robustness_figure(
    result: DistributedInverseRobustnessStudyResult,
    output: Path | str,
) -> Path:
    """Plot trial errors and all completed coefficient estimates."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    indices = tuple(trial.trial_index for trial in result.trials)
    criteria = result.config.criteria

    error_axis = axes[0]
    error_axis.plot(
        indices,
        tuple(
            trial.conventional_maximum_absolute_multiplier_error
            for trial in result.trials
        ),
        marker="o",
        label="conventional",
    )
    error_axis.plot(
        indices,
        tuple(
            trial.inverse_pinn_maximum_absolute_multiplier_error
            if math.isfinite(trial.inverse_pinn_maximum_absolute_multiplier_error)
            else math.nan
            for trial in result.trials
        ),
        marker="o",
        label="inverse PINN",
    )
    error_axis.axhline(
        criteria.maximum_absolute_multiplier_error,
        color="red",
        linestyle="--",
        label="predeclared limit",
    )
    error_axis.set_title("Recovery error by retained trial")
    error_axis.set_xlabel("Trial index")
    error_axis.set_ylabel("Maximum absolute multiplier error")
    error_axis.set_xticks(indices)
    error_axis.legend()
    error_axis.grid(alpha=0.25)

    coefficient_axis = axes[1]
    knots = (285.0, 300.0, 315.0)
    coefficient_axis.plot(
        knots,
        result.truth_multipliers,
        color="black",
        linewidth=2.5,
        marker="o",
        label="truth",
    )
    for trial in result.trials:
        coefficient_axis.plot(
            knots,
            trial.conventional_multipliers,
            color="tab:blue",
            alpha=0.35,
        )
        if trial.inverse_pinn_multipliers is not None:
            coefficient_axis.plot(
                knots,
                trial.inverse_pinn_multipliers,
                color="tab:orange",
                alpha=0.45,
            )
    coefficient_axis.plot((), (), color="tab:blue", label="conventional trials")
    coefficient_axis.plot((), (), color="tab:orange", label="inverse-PINN trials")
    coefficient_axis.set_title("All completed noisy estimates")
    coefficient_axis.set_xlabel("Property-knot temperature (K)")
    coefficient_axis.set_ylabel("rho_e multiplier")
    coefficient_axis.set_xticks(knots)
    coefficient_axis.legend()
    coefficient_axis.grid(alpha=0.25)

    figure.savefig(output_path, dpi=180)
    save_figure_data(result, output_path)
    plt.close(figure)
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run noisy multi-seed distributed resistivity inference."
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--first-seed", type=int, default=27_001)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_figure_path(
            "distributed_inverse_robustness.png",
            "DISTRIBUTED_INVERSE_ROBUSTNESS.md",
        ),
    )
    args = parser.parse_args(argv)
    result = run_distributed_inverse_robustness_study(
        DistributedInverseRobustnessConfig(
            trial_count=args.trials,
            first_seed=args.first_seed,
            inverse_pinn_epochs=args.epochs,
        )
    )
    output = save_distributed_inverse_robustness_figure(result, args.output)
    print(format_distributed_inverse_robustness_report(result))
    print(f"figure: {output}")


if __name__ == "__main__":
    main()
