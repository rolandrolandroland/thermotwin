"""Report inverse contact-resistance PINN recovery under imperfect data."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..figure_paths import default_figure_path, save_figure_data
from ..studies.imperfect_inverse_pinn import (
    ImperfectInversePINNConfig,
    ImperfectInversePINNStudyResult,
    run_imperfect_inverse_pinn_study,
)


DEFAULT_IMPERFECT_INVERSE_PINN_PATH = default_figure_path(
    "imperfect_inverse_pinn.png", "IMPERFECT_INVERSE_PINN.md"
)


def format_imperfect_inverse_pinn_report(
    result: ImperfectInversePINNStudyResult,
) -> str:
    """Return the complete trial table, gates, and interpretation boundary."""

    config = result.config
    criteria = config.criteria
    lines = [
        "Inverse contact PINN under imperfect observations",
        "=================================================",
        "",
        "Question:",
        "  Can the switched-current inverse PINN recover a hidden cold-contact",
        "  resistance from noisy or structurally missing temperatures, and does",
        "  the recovered parameter transfer to complete unseen current regimes?",
        "",
        "Frozen comparison:",
        f"  trials per case: {config.trial_count_per_case}",
        f"  epochs per PINN: {config.training_epochs}",
        f"  initial resistances: {config.initial_resistances[:config.trial_count_per_case]}",
        "  conventional and PINN estimators receive the identical transformed rows",
        "  validation and bipolar test regimes are complete and withheld from fitting",
        "",
        "Predeclared recovery gate:",
        f"  training-loss reduction >= {criteria.minimum_loss_reduction_fraction:.0%}",
        f"  absolute resistance error <= {criteria.maximum_absolute_parameter_error:.3f} K/W",
        f"  validation and test all-sensor RMSE <= "
        f"{criteria.maximum_withheld_all_sensor_rmse:.3f} K",
        "",
        "Trial results:",
    ]
    for trial in result.trials:
        validation = trial.validation
        lines.append(
            f"  {trial.case_name} / trial {trial.trial_index}: "
            f"PINN={validation.inferred_cold_contact_resistance:.6f} K/W; "
            f"conventional={validation.conventional_cold_contact_resistance:.6f} K/W; "
            f"absolute error={validation.absolute_parameter_error:.6f} K/W; "
            f"validation={validation.validation_regime_metrics.all_sensor_rmse:.6f} K; "
            f"test={validation.test_regime_metrics.all_sensor_rmse:.6f} K; "
            f"loss reduction={trial.loss_reduction_fraction:.2%}; "
            f"recovery={'PASS' if trial.recovery_success else 'FAIL'}; "
            f"reasons={trial.failure_reasons or ('none',)}"
        )
    lines.extend(("", "Case summaries:"))
    for summary in result.summaries:
        expectation = "recovery expected" if summary.expected_recovery else "mismatch expected"
        lines.append(
            f"  {summary.case_name} ({expectation}): "
            f"recovery={summary.recovery_success_count}/{summary.trial_count}; "
            f"operational={summary.operational_success_count}/{summary.trial_count}; "
            f"PINN RMSE={summary.pinn_parameter_rmse:.6f} K/W; "
            f"conventional RMSE={summary.conventional_parameter_rmse:.6f} K/W; "
            f"mean validation={summary.mean_validation_all_sensor_rmse:.6f} K; "
            f"mean test={summary.mean_test_all_sensor_rmse:.6f} K"
        )
    lines.extend(
        (
            "",
            "Interpretation boundary:",
            "  These are synthetic same-model observations, not hardware validation.",
            "  The bias case deliberately withholds a sensor-bias term from the inverse",
            "  model. Its purpose is to test whether a falling optimization loss can",
            "  hide model mismatch; recovery is not assumed for this case.",
            "  Missing rows are removed from both observed cold channels at the same",
            "  turn-off times, so neither estimator receives a hidden dense target.",
            "  The conventional scalar fit is the accuracy reference for this small",
            "  inverse problem; the PINN is valuable for its continuous hidden state",
            "  and simultaneous enforcement of the switched energy balances.",
        )
    )
    return "\n".join(lines)


def save_imperfect_inverse_pinn_figure(
    result: ImperfectInversePINNStudyResult,
    output: Path | str,
) -> Path:
    """Save recovery, transfer, and explicit-failure panels."""

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(13.0, 9.0), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    summaries = result.summaries
    labels = tuple(item.case_name.replace("_", "\n") for item in summaries)
    positions = tuple(range(len(summaries)))
    width = 0.36

    axis = axes[0, 0]
    axis.bar(
        tuple(index - width / 2 for index in positions),
        tuple(item.pinn_parameter_rmse for item in summaries),
        width,
        label="inverse PINN",
    )
    axis.bar(
        tuple(index + width / 2 for index in positions),
        tuple(item.conventional_parameter_rmse for item in summaries),
        width,
        label="conventional",
    )
    axis.axhline(
        result.config.criteria.maximum_absolute_parameter_error,
        color="red",
        linestyle="--",
        label="parameter gate",
    )
    axis.set_ylabel("Resistance RMSE (K/W)")
    axis.set_title("Parameter recovery")
    axis.legend(fontsize=8)

    axis = axes[0, 1]
    axis.bar(
        tuple(index - width / 2 for index in positions),
        tuple(item.mean_validation_all_sensor_rmse for item in summaries),
        width,
        label="validation schedule",
    )
    axis.bar(
        tuple(index + width / 2 for index in positions),
        tuple(item.mean_test_all_sensor_rmse for item in summaries),
        width,
        label="bipolar test schedule",
    )
    axis.axhline(
        result.config.criteria.maximum_withheld_all_sensor_rmse,
        color="red",
        linestyle="--",
        label="transfer gate",
    )
    axis.set_ylabel("All-sensor RMSE (K)")
    axis.set_title("Withheld-regime transfer")
    axis.legend(fontsize=8)

    axis = axes[1, 0]
    for case_index, summary in enumerate(summaries):
        selected = tuple(
            trial for trial in result.trials if trial.case_name == summary.case_name
        )
        axis.scatter(
            (case_index,) * len(selected),
            tuple(
                trial.validation.inferred_cold_contact_resistance
                for trial in selected
            ),
            label="PINN" if case_index == 0 else None,
            color="tab:blue",
        )
        axis.scatter(
            tuple(case_index + 0.12 for _ in selected),
            tuple(
                trial.validation.conventional_cold_contact_resistance
                for trial in selected
            ),
            label="conventional" if case_index == 0 else None,
            marker="x",
            color="tab:orange",
        )
    axis.axhline(0.25, color="black", linestyle=":", label="truth")
    axis.set_ylabel("Inferred cold contact resistance (K/W)")
    axis.set_title("Every retained trial")
    axis.legend(fontsize=8)

    axis = axes[1, 1]
    axis.bar(
        positions,
        tuple(item.recovery_success_count / item.trial_count for item in summaries),
        color=tuple(
            "tab:green" if item.expected_recovery else "tab:gray"
            for item in summaries
        ),
    )
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Recovery success fraction")
    axis.set_title("Predeclared all-criteria gate")
    for index, summary in enumerate(summaries):
        axis.text(
            index,
            0.04,
            "expected\nrecovery" if summary.expected_recovery else "intentional\nmismatch",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for axis in axes.flat:
        axis.set_xticks(positions, labels, rotation=12)
        axis.grid(alpha=0.25)
    figure.suptitle(
        "Inverse contact PINN: imperfect observations and withheld transfer",
        fontsize=15,
    )
    figure.savefig(destination, dpi=170)
    save_figure_data(result, destination)
    return destination


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the imperfect-observation inverse contact-PINN audit."
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2_500)
    parser.add_argument("--first-seed", type=int, default=61_001)
    parser.add_argument("--output", type=Path, default=DEFAULT_IMPERFECT_INVERSE_PINN_PATH)
    args = parser.parse_args(argv)
    result = run_imperfect_inverse_pinn_study(
        ImperfectInversePINNConfig(
            trial_count_per_case=args.trials,
            training_epochs=args.epochs,
            first_seed=args.first_seed,
        ),
        progress=lambda message: print(message, flush=True),
    )
    destination = save_imperfect_inverse_pinn_figure(result, args.output)
    print(format_imperfect_inverse_pinn_report(result))
    print(f"figure: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
