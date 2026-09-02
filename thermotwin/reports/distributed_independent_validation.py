"""Report for independent-truth and matched-regularization validation."""

import argparse
import math
from pathlib import Path
from statistics import fmean
from typing import Optional, Sequence

from ..figure_paths import default_figure_path, save_figure_data
from ..simulation.distributed import distributed_inverse_constant_experiments
from ..studies.distributed_independent_validation import (
    ESTIMATOR_NAMES,
    DistributedIndependentValidationConfig,
    DistributedIndependentValidationStudyResult,
    DistributedMismatchPredictionMetrics,
    run_distributed_independent_validation_study,
)


def _multipliers(values) -> str:
    if values is None:
        return "training failed"
    return "(" + ", ".join(f"{value:.6f}" for value in values) + ")"


def _mean_holdout_metrics(
    result: DistributedIndependentValidationStudyResult,
    estimator_name: str,
    holdout_name: str,
) -> DistributedMismatchPredictionMetrics:
    metrics = tuple(
        metric
        for trial in result.trials
        for estimator in trial.estimators
        if estimator.name == estimator_name
        for name, metric in estimator.predictions
        if name == holdout_name
    )
    if not metrics:
        return DistributedMismatchPredictionMetrics(*(math.inf,) * 6)
    return DistributedMismatchPredictionMetrics(
        *(fmean(values) for values in zip(*metrics))
    )


def format_distributed_independent_validation_report(
    result: DistributedIndependentValidationStudyResult,
) -> str:
    """Describe numerical independence, paired priors, results, and limits."""

    config = result.config
    criteria = config.criteria
    lines = [
        "Distributed inverse: independent truth and matched regularization",
        "=================================================================",
        "",
        "Truth generator:",
        f"  node-centred grid: {config.truth_node_count} nodes",
        f"  integrator: transition-split SSPRK3, dt={config.truth_time_step:.3e} s",
        "  resistivity: smooth cubic, not representable by the fitted three-knot curve",
        f"  truth values at inference knots: {_multipliers(result.truth_knot_multipliers)}",
        "  inference solver remains the established cell-centred finite-volume/RK4 model",
        "",
        "Training data:",
        f"  regimes: {result.training_experiment_names}",
        f"  noise: {config.temperature_standard_deviation:.6f} K and "
        f"{config.voltage_standard_deviation:.6e} V",
        f"  trials: {config.trial_count}; PINN epochs per fit: {config.inverse_pinn_epochs}",
        "",
        "Matched explicit prior:",
        "  roughness = mean squared second difference of log multipliers",
        f"  matched weight = {config.matched_smoothness_weight:.6g}",
        "  within each estimator family, variants use identical data and starts",
        "  implicit neural regularization is not and cannot be matched by this step",
        "",
        "Predeclared in-support gate:",
        f"  property maximum relative error <= "
        f"{criteria.maximum_property_relative_error:.6f}",
        f"  face/internal RMSE <= {criteria.maximum_face_temperature_rmse:.6f} K",
        f"  voltage RMSE <= {criteria.maximum_voltage_rmse:.6e} V",
        f"  maximum temperature error <= "
        f"{criteria.maximum_absolute_temperature_error:.6f} K",
        f"  prediction energy residual <= "
        f"{criteria.maximum_energy_balance_residual:.3e} W",
        "  the 40 K outside-knot case is diagnostic and is not included in pass/fail.",
        "",
        "Estimator summary:",
    ]
    for summary in result.summaries:
        lines.append(
            f"  {summary.name}: passes={summary.success_count}/{config.trial_count}; "
            f"property relative RMSE={summary.mean_in_support_property_relative_rmse:.6f}; "
            f"property max error={summary.mean_in_support_property_maximum_relative_error:.6f}; "
            f"in-support internal RMSE="
            f"{summary.mean_in_support_internal_temperature_rmse:.6f} K; "
            f"in-support voltage RMSE={summary.mean_in_support_voltage_rmse:.6e} V"
        )
    lines.extend(("", "Mean transfer metrics:"))
    for estimator_name in ESTIMATOR_NAMES:
        lines.append(f"  {estimator_name}:")
        for holdout_name in result.holdout_names:
            metrics = _mean_holdout_metrics(result, estimator_name, holdout_name)
            lines.append(
                f"    {holdout_name}: internal={metrics.internal_temperature_rmse:.6f} K, "
                f"voltage={metrics.voltage_rmse:.6e} V, "
                f"maxT={metrics.maximum_absolute_temperature_error:.6f} K"
            )
    lines.extend(("", "Every trial:"))
    for trial in result.trials:
        lines.append(
            f"  trial {trial.trial_index}: neural seed={trial.seeds.neural}; "
            f"noise seeds={trial.seeds.observations}"
        )
        for estimator in trial.estimators:
            lines.append(
                f"    {estimator.name}: {'PASS' if estimator.success else 'FAIL'}; "
                f"multipliers={_multipliers(estimator.multipliers)}; "
                f"data loss={estimator.final_normalized_observation_loss:.6f}; "
                f"roughness={estimator.log_multiplier_roughness:.6e}; "
                f"property max error="
                f"{estimator.in_support_property_maximum_relative_error:.6f}; "
                f"reasons={estimator.failure_reasons or ('none',)}"
            )
    lines.extend(
        (
            "",
            "Interpretation boundary:",
            "  This removes the exact grid, time-integrator, and property-basis inverse crime.",
            "  Truth remains synthetic and obeys the same continuum thermoelectric equations.",
            "  The explicit coefficient penalty is identical, but the PINN still has PDE loss",
            "  and implicit neural bias; the comparison is fairer, not perfectly equivalent.",
            "  Three trials are a model-mismatch demonstration, not a failure-rate estimate.",
            "  The outside-support result tests declared extrapolation behavior, not recovery",
            "  of material evidence beyond the fitted temperature interval.",
        )
    )
    return "\n".join(lines)


def save_distributed_independent_validation_figure(
    result: DistributedIndependentValidationStudyResult,
    output: Path | str,
) -> Path:
    """Plot truth curves, property errors, and transfer errors."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    temperatures = tuple(280.0 + 0.5 * index for index in range(81))
    baseline = distributed_inverse_constant_experiments()[0].material.electrical_resistivity
    first_trial = result.trials[0]

    curve_axis = axes[0, 0]
    curve_axis.plot(
        temperatures,
        tuple(result.truth_property.value(value) * 1.0e5 for value in temperatures),
        color="black",
        linewidth=2.5,
        label="independent cubic truth",
    )
    if hasattr(baseline, "with_values"):
        for estimator in first_trial.estimators:
            if estimator.multipliers is None:
                continue
            prop = baseline.with_values(
                tuple(
                    value * multiplier
                    for value, multiplier in zip(baseline.values, estimator.multipliers)
                )
            )
            curve_axis.plot(
                temperatures,
                tuple(prop.value(value) * 1.0e5 for value in temperatures),
                label=estimator.name,
                alpha=0.85,
            )
    curve_axis.axvspan(285.0, 315.0, alpha=0.08, color="green", label="fit support")
    curve_axis.set_title("Recovered resistivity curves: trial 0")
    curve_axis.set_xlabel("Temperature (K)")
    curve_axis.set_ylabel("Resistivity (10^-5 ohm m)")

    indices = tuple(trial.trial_index for trial in result.trials)
    property_axis = axes[0, 1]
    for name in ESTIMATOR_NAMES:
        property_axis.plot(
            indices,
            tuple(
                next(
                    estimator.in_support_property_maximum_relative_error
                    for estimator in trial.estimators
                    if estimator.name == name
                )
                for trial in result.trials
            ),
            marker="o",
            label=name,
        )
    property_axis.axhline(
        result.config.criteria.maximum_property_relative_error,
        color="red",
        linestyle="--",
        label="property gate",
    )
    property_axis.set_title("In-support property error")
    property_axis.set_xlabel("Trial")
    property_axis.set_ylabel("Maximum relative error")

    voltage_axis = axes[1, 0]
    gated_holdouts = result.holdout_names[:2]
    for name in ESTIMATOR_NAMES:
        voltage_axis.plot(
            indices,
            tuple(
                fmean(
                    metrics.voltage_rmse
                    for estimator in trial.estimators
                    if estimator.name == name
                    for holdout, metrics in estimator.predictions
                    if holdout in gated_holdouts
                )
                for trial in result.trials
            ),
            marker="o",
            label=name,
        )
    voltage_axis.axhline(
        result.config.criteria.maximum_voltage_rmse,
        color="red",
        linestyle="--",
        label="voltage gate",
    )
    voltage_axis.set_title("In-support heldout voltage")
    voltage_axis.set_xlabel("Trial")
    voltage_axis.set_ylabel("Mean RMSE (V)")

    extrapolation_axis = axes[1, 1]
    outside_name = result.holdout_names[-1]
    for name in ESTIMATOR_NAMES:
        extrapolation_axis.plot(
            indices,
            tuple(
                next(
                    metrics.voltage_rmse
                    for estimator in trial.estimators
                    if estimator.name == name
                    for holdout, metrics in estimator.predictions
                    if holdout == outside_name
                )
                for trial in result.trials
            ),
            marker="o",
            label=name,
        )
    extrapolation_axis.set_title("Outside-support 40 K voltage diagnostic")
    extrapolation_axis.set_xlabel("Trial")
    extrapolation_axis.set_ylabel("RMSE (V)")

    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.savefig(output_path, dpi=180)
    save_figure_data(result, output_path)
    plt.close(figure)
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run independent-truth and matched-prior distributed inversion."
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--first-seed", type=int, default=47_001)
    parser.add_argument("--truth-nodes", type=int, default=25)
    parser.add_argument("--truth-time-step", type=float, default=2.5e-4)
    parser.add_argument("--smoothness-weight", type=float, default=25.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_figure_path(
            "distributed_independent_validation.png",
            "DISTRIBUTED_INDEPENDENT_VALIDATION.md",
        ),
    )
    args = parser.parse_args(argv)
    result = run_distributed_independent_validation_study(
        DistributedIndependentValidationConfig(
            trial_count=args.trials,
            first_seed=args.first_seed,
            inverse_pinn_epochs=args.epochs,
            truth_node_count=args.truth_nodes,
            truth_time_step=args.truth_time_step,
            matched_smoothness_weight=args.smoothness_weight,
        )
    )
    output = save_distributed_independent_validation_figure(result, args.output)
    print(format_distributed_independent_validation_report(result))
    print(f"figure: {output}")


if __name__ == "__main__":
    main()
