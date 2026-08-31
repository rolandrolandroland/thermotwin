"""Report for complete-regime distributed resistivity transfer validation."""

import argparse
import math
from pathlib import Path
from typing import Optional, Sequence

from ..figure_paths import default_figure_path
from ..studies.distributed_withheld_validation import (
    DistributedWithheldPredictionMetrics,
    DistributedWithheldValidationConfig,
    DistributedWithheldValidationStudyResult,
    run_distributed_withheld_validation_study,
)


def _multipliers_text(values) -> str:
    if values is None:
        return "training failed"
    return "(" + ", ".join(f"{value:.6f}" for value in values) + ")"


def _metrics_text(metrics: Optional[DistributedWithheldPredictionMetrics]) -> str:
    if metrics is None:
        return "prediction not completed"
    return (
        f"cold={metrics.cold_face_rmse:.6f} K, "
        f"hot={metrics.hot_face_rmse:.6f} K, "
        f"internal={metrics.internal_temperature_rmse:.6f} K, "
        f"voltage={metrics.voltage_rmse:.6e} V, "
        f"maxT={metrics.maximum_absolute_temperature_error:.6f} K, "
        f"energy={metrics.maximum_energy_balance_residual:.3e} W"
    )


def format_distributed_withheld_validation_report(
    result: DistributedWithheldValidationStudyResult,
) -> str:
    """Print the excluded regime, fixed thresholds, every trial, and caveats."""

    config = result.config
    criteria = config.criteria
    lines = [
        "Distributed resistivity inverse: withheld-regime validation",
        "===========================================================",
        "",
        f"withheld complete regime: {result.withheld_experiment_name}",
        "fit regimes: the other three frozen constant-current regimes",
        f"truth multipliers: {_multipliers_text(result.truth_multipliers)}",
        f"training noise: {config.temperature_standard_deviation:.6f} K and "
        f"{config.voltage_standard_deviation:.6e} V",
        "withheld scoring target: noise-free hidden synthetic truth",
        "property curve is frozen before prediction; no withheld refitting",
        "",
        "Predeclared prediction gate:",
        f"  cold-face RMSE <= {criteria.maximum_cold_face_rmse:.6f} K",
        f"  hot-face RMSE <= {criteria.maximum_hot_face_rmse:.6f} K",
        f"  internal-field RMSE <= "
        f"{criteria.maximum_internal_temperature_rmse:.6f} K",
        f"  voltage RMSE <= {criteria.maximum_voltage_rmse:.6e} V",
        f"  maximum absolute temperature error <= "
        f"{criteria.maximum_absolute_temperature_error:.6f} K",
        f"  maximum energy-balance residual <= "
        f"{criteria.maximum_energy_balance_residual:.3e} W",
        "  all criteria must pass; property error is reported but not gated.",
        "",
        "Trials:",
    ]
    for trial in result.trials:
        conventional_status = (
            "PASS" if trial.conventional_prediction_success else "FAIL"
        )
        pinn_status = "PASS" if trial.inverse_pinn_prediction_success else "FAIL"
        lines.extend(
            (
                f"  trial {trial.trial_index}: neural seed={trial.seeds.neural}; "
                f"noise seeds={trial.seeds.observations}",
                f"    conventional: {conventional_status}; multipliers="
                f"{_multipliers_text(trial.conventional_multipliers)}; "
                f"property max error="
                f"{trial.conventional_maximum_absolute_multiplier_error:.6f}; "
                f"search bound hit={trial.conventional_reached_search_bound}; "
                f"prediction: {_metrics_text(trial.conventional_prediction)}; "
                f"reasons={trial.conventional_prediction_failure_reasons or ('none',)}",
                f"    inverse PINN: {pinn_status}; multipliers="
                f"{_multipliers_text(trial.inverse_pinn_multipliers)}; "
                f"property max error="
                f"{trial.inverse_pinn_maximum_absolute_multiplier_error:.6f}; "
                f"prediction: {_metrics_text(trial.inverse_pinn_prediction)}; "
                f"reasons={trial.inverse_pinn_prediction_failure_reasons or ('none',)}",
            )
        )
    summary = result.summary
    lines.extend(
        (
            "",
            "Summary:",
            f"  conventional prediction successes: "
            f"{summary.conventional_success_count}/{summary.trial_count}",
            f"  inverse-PINN prediction successes: "
            f"{summary.inverse_pinn_success_count}/{summary.trial_count}",
            f"  inverse-PINN completed predictions: "
            f"{summary.inverse_pinn_completed_count}/{summary.trial_count}",
            f"  conventional search-bound hits: "
            f"{summary.conventional_search_bound_hits}",
            f"  conventional mean prediction: "
            f"{_metrics_text(summary.conventional_mean_prediction)}",
            f"  inverse-PINN mean prediction: "
            f"{_metrics_text(summary.inverse_pinn_mean_prediction)}",
            f"  conventional worst maximum temperature error: "
            f"{summary.conventional_worst_temperature_error:.6f} K",
            f"  inverse-PINN worst maximum temperature error: "
            f"{summary.inverse_pinn_worst_temperature_error:.6f} K",
            f"  conventional mean property max error: "
            f"{summary.conventional_mean_multiplier_error:.6f}",
            f"  inverse-PINN mean property max error: "
            + (
                f"{summary.inverse_pinn_mean_multiplier_error:.6f}"
                if summary.inverse_pinn_mean_multiplier_error is not None
                else "not completed"
            ),
            "",
            "Interpretation boundary:",
            "  The withheld operating regime is absent from fitting, but the truth",
            "  and prediction solver still share equations, grid, and curve basis.",
            "  The internal field is hidden during fitting and scored only afterward.",
            "  Passing establishes synthetic regime transfer, not extrapolation to",
            "  an unseen material law, independent discretization, or hardware.",
        )
    )
    return "\n".join(lines)


def _maximum_temperature_rmse(metrics: DistributedWithheldPredictionMetrics) -> float:
    return max(
        metrics.cold_face_rmse,
        metrics.hot_face_rmse,
        metrics.internal_temperature_rmse,
    )


def save_distributed_withheld_validation_figure(
    result: DistributedWithheldValidationStudyResult,
    output: Path | str,
) -> Path:
    """Plot coefficient and complete-regime prediction errors by trial."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    indices = tuple(trial.trial_index for trial in result.trials)
    completed_pinn = tuple(
        trial for trial in result.trials if trial.inverse_pinn_prediction is not None
    )

    property_axis = axes[0, 0]
    property_axis.plot(
        indices,
        tuple(
            trial.conventional_maximum_absolute_multiplier_error
            for trial in result.trials
        ),
        marker="o",
        label="conventional",
    )
    property_axis.plot(
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
    property_axis.set_title("Recovered-property error (not transfer-gated)")
    property_axis.set_ylabel("Maximum knot-multiplier error")

    temperature_axis = axes[0, 1]
    temperature_axis.plot(
        indices,
        tuple(
            _maximum_temperature_rmse(trial.conventional_prediction)
            for trial in result.trials
        ),
        marker="o",
        label="conventional",
    )
    temperature_axis.plot(
        tuple(trial.trial_index for trial in completed_pinn),
        tuple(
            _maximum_temperature_rmse(trial.inverse_pinn_prediction)
            for trial in completed_pinn
        ),
        marker="o",
        label="inverse PINN",
    )
    temperature_axis.axhline(
        result.config.criteria.maximum_internal_temperature_rmse,
        color="red",
        linestyle="--",
        label="RMSE limit",
    )
    temperature_axis.set_title("Worst withheld temperature RMSE")
    temperature_axis.set_ylabel("RMSE (K)")

    voltage_axis = axes[1, 0]
    voltage_axis.plot(
        indices,
        tuple(trial.conventional_prediction.voltage_rmse for trial in result.trials),
        marker="o",
        label="conventional",
    )
    voltage_axis.plot(
        tuple(trial.trial_index for trial in completed_pinn),
        tuple(trial.inverse_pinn_prediction.voltage_rmse for trial in completed_pinn),
        marker="o",
        label="inverse PINN",
    )
    voltage_axis.axhline(
        result.config.criteria.maximum_voltage_rmse,
        color="red",
        linestyle="--",
        label="voltage limit",
    )
    voltage_axis.set_title("Withheld voltage prediction")
    voltage_axis.set_ylabel("RMSE (V)")

    maximum_axis = axes[1, 1]
    maximum_axis.plot(
        indices,
        tuple(
            trial.conventional_prediction.maximum_absolute_temperature_error
            for trial in result.trials
        ),
        marker="o",
        label="conventional",
    )
    maximum_axis.plot(
        tuple(trial.trial_index for trial in completed_pinn),
        tuple(
            trial.inverse_pinn_prediction.maximum_absolute_temperature_error
            for trial in completed_pinn
        ),
        marker="o",
        label="inverse PINN",
    )
    maximum_axis.axhline(
        result.config.criteria.maximum_absolute_temperature_error,
        color="red",
        linestyle="--",
        label="maximum-error limit",
    )
    maximum_axis.set_title("Worst point on withheld trajectory")
    maximum_axis.set_ylabel("Absolute temperature error (K)")

    for axis in axes.flat:
        axis.set_xlabel("Trial index")
        axis.set_xticks(indices)
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fit three distributed regimes and predict the fourth."
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--first-seed", type=int, default=37_001)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_figure_path("distributed_withheld_validation.png"),
    )
    args = parser.parse_args(argv)
    result = run_distributed_withheld_validation_study(
        DistributedWithheldValidationConfig(
            trial_count=args.trials,
            first_seed=args.first_seed,
            inverse_pinn_epochs=args.epochs,
        )
    )
    output = save_distributed_withheld_validation_figure(result, args.output)
    print(format_distributed_withheld_validation_report(result))
    print(f"figure: {output}")


if __name__ == "__main__":
    main()
