"""Report the sequential experiment campaign and model-mismatch stress test."""

import argparse
from collections import Counter
import math
from pathlib import Path
from typing import Optional, Sequence

from .paths import default_figure_path, save_figure_data
from ..studies.adaptive_experiment_campaign import (
    AdaptiveCampaignConfig,
    AdaptiveExperimentCampaignResult,
    CAMPAIGN_STRATEGIES,
    TRUTH_CONDITIONS,
    run_adaptive_experiment_campaign,
)


DEFAULT_ADAPTIVE_CAMPAIGN_PATH = default_figure_path(
    "adaptive_experiment_campaign.png",
    "ADAPTIVE_EXPERIMENT_CAMPAIGN.md",
)


def _finite_or_never(value: float, *, unit: str = "") -> str:
    if not math.isfinite(value):
        return "never"
    return f"{value:.2f}{unit}"


def format_adaptive_campaign_report(
    result: AdaptiveExperimentCampaignResult,
) -> str:
    """Return resource use, prediction, uncertainty, and false-confidence results."""

    config = result.config
    lines = [
        "Adaptive experiment campaign under model mismatch",
        "=================================================",
        "",
        "Question:",
        "  Does posterior-aware pulse selection reach a predictive gate sooner than",
        "  a precommitted D-optimal batch or an engineer heuristic, and does a",
        "  withheld hidden-state check expose confident but physically wrong fits?",
        "",
        "Campaign:",
        f"  paired trials per truth condition: {config.trial_count}",
        f"  experiments per strategy: {config.experiment_count}",
        f"  total modeled-energy cap: {config.total_energy_budget:.2f} J",
        f"  feasible candidates: {len(result.candidates)}",
        f"  static plan: {', '.join(result.static_plan)}",
        f"  engineer plan: {', '.join(result.heuristic_plan)}",
        "  adaptive policy: re-fit all accumulated data, relinearize at the current",
        "  estimate, then maximize expected log-determinant reduction subject to the",
        "  remaining energy and experiment-count budget",
        "",
        "Adaptive choices by paired trial:",
    ]
    for condition in TRUTH_CONDITIONS:
        for step_index in range(1, config.experiment_count + 1):
            counts = Counter(
                item.candidate_name
                for item in result.steps
                if item.truth_condition == condition
                and item.strategy == "adaptive"
                and item.step_index == step_index
            )
            choices = ", ".join(
                f"{name} ({count}/{config.trial_count})"
                for name, count in sorted(counts.items())
            )
            lines.append(f"  {condition}, step {step_index}: {choices}")
    lines.extend(
        (
        "",
        "Predeclared gates:",
        f"  accessible held-out RMSE <= {config.prediction_rmse_threshold:.3f} K",
        f"  hidden cold-face held-out RMSE <= {config.hidden_face_rmse_threshold:.3f} K",
        f"  maximum switch-window bias <= {config.switch_bias_threshold:.3f} K",
        f"  physical log-parameter RMSE <= {config.parameter_log_rmse_threshold:.3f}",
        f"  confident means maximum local log standard error <= "
        f"{config.confidence_log_standard_error:.3f}",
        "",
        "Strategy outcomes:",
        )
    )
    for outcome in result.outcomes:
        lines.append(
            f"  {outcome.truth_condition} / {outcome.strategy}: "
            f"median first pass={_finite_or_never(outcome.median_first_passing_step)}; "
            f"median energy to pass="
            f"{_finite_or_never(outcome.median_energy_to_prediction_gate, unit=' J')}; "
            f"never passed={outcome.never_passed_rate:.1%}; "
            f"final pass={outcome.final_prediction_pass_rate:.1%}; "
            f"final false confidence={outcome.final_false_confidence_rate:.1%}"
        )
    lines.extend(("", "Final-step means:"))
    final_summaries = tuple(
        item
        for item in result.summaries
        if item.step_index == config.experiment_count
    )
    for summary in final_summaries:
        lines.append(
            f"  {summary.truth_condition} / {summary.strategy}: "
            f"energy={summary.mean_cumulative_energy:.2f} J; "
            f"parameter log-RMSE={summary.mean_parameter_log_rmse:.5f}; "
            f"observation RMSE={summary.mean_observation_rmse:.5f} K; "
            f"uncertainty volume={summary.median_uncertainty_volume:.6e}; "
            f"accessible withheld RMSE={summary.mean_heldout_prediction_rmse:.5f} K; "
            f"hidden-face withheld RMSE={summary.mean_heldout_hidden_face_rmse:.5f} K; "
            f"parameter pass={summary.parameter_recovery_pass_rate:.1%}; "
            f"prediction pass={summary.prediction_pass_rate:.1%}; "
            f"false confidence={summary.false_confidence_rate:.1%}; "
            f"bound hits={summary.bound_hit_rate:.1%}"
        )
    lines.extend(
        (
            "",
            "Interpretation boundary:",
            "  Every result is synthetic. The matched condition is an intentional",
            "  same-equation control. The mismatch condition preserves the declared",
            "  steady cold-contact resistance but inserts an unobserved interface",
            "  thermal mass. The four-node planning and inference model never sees",
            "  that extra state.",
            "  Accessible predictions can remain accurate while inferred physical",
            "  parameters or hidden states are wrong. In that case, this campaign",
            "  establishes predictive equivalence at the measured terminals, not",
            "  physical identification.",
            "  The energy cap uses nominal modeled energy, all runs reset to the same",
            "  equilibrium, and candidate pulses are not hardware safety limits.",
            "  The study quantifies virtual test cycles; it does not count prototypes",
            "  avoided and is not hardware validation.",
        )
    )
    return "\n".join(lines)


def save_adaptive_campaign_figure(
    result: AdaptiveExperimentCampaignResult,
    output: Path | str,
) -> Path:
    """Save prediction, hidden-state, uncertainty, and confidence panels."""

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(13.5, 9.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    colors = {
        "adaptive": "tab:blue",
        "static_d_optimal": "tab:orange",
        "engineer_heuristic": "tab:green",
    }
    labels = {
        "adaptive": "Adaptive",
        "static_d_optimal": "Static D-optimal",
        "engineer_heuristic": "Engineer heuristic",
    }

    for strategy in CAMPAIGN_STRATEGIES:
        selected = tuple(
            item
            for item in result.summaries
            if item.truth_condition == "matched_model" and item.strategy == strategy
        )
        axes[0, 0].plot(
            tuple(item.mean_cumulative_energy for item in selected),
            tuple(item.mean_heldout_hidden_face_rmse for item in selected),
            marker="o",
            color=colors[strategy],
            label=labels[strategy],
        )
    axes[0, 0].axhline(
        result.config.hidden_face_rmse_threshold,
        color="black",
        linestyle="--",
        label="Prediction gate",
    )
    axes[0, 0].set_title("Matched-model hidden-state transfer")
    axes[0, 0].set_xlabel("Mean cumulative modeled energy (J)")
    axes[0, 0].set_ylabel("Mean withheld cold-face RMSE (K)")
    axes[0, 0].legend(fontsize=8)

    for strategy in CAMPAIGN_STRATEGIES:
        selected = tuple(
            item
            for item in result.summaries
            if item.truth_condition == "extra_interface_mass"
            and item.strategy == strategy
        )
        axes[0, 1].plot(
            tuple(item.mean_cumulative_energy for item in selected),
            tuple(item.mean_heldout_hidden_face_rmse for item in selected),
            marker="o",
            color=colors[strategy],
            label=labels[strategy],
        )
    axes[0, 1].axhline(
        result.config.hidden_face_rmse_threshold,
        color="black",
        linestyle="--",
        label="Prediction gate",
    )
    axes[0, 1].set_title("Unmodeled interface mass: hidden-state transfer")
    axes[0, 1].set_xlabel("Mean cumulative modeled energy (J)")
    axes[0, 1].set_ylabel("Mean withheld cold-face RMSE (K)")
    axes[0, 1].legend(fontsize=8)

    for strategy in CAMPAIGN_STRATEGIES:
        selected = tuple(
            item
            for item in result.summaries
            if item.truth_condition == "extra_interface_mass"
            and item.strategy == strategy
        )
        axes[1, 0].plot(
            tuple(item.step_index for item in selected),
            tuple(item.median_uncertainty_volume for item in selected),
            marker="o",
            color=colors[strategy],
            label=labels[strategy],
        )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Reported uncertainty under model mismatch")
    axes[1, 0].set_xlabel("Experiments collected")
    axes[1, 0].set_ylabel("Median sqrt(det covariance))")

    final = tuple(
        item
        for item in result.summaries
        if item.truth_condition == "extra_interface_mass"
        and item.step_index == result.config.experiment_count
    )
    positions = tuple(range(len(final)))
    width = 0.38
    axes[1, 1].bar(
        tuple(position - width / 2 for position in positions),
        tuple(item.prediction_pass_rate for item in final),
        width,
        label="Prediction pass",
    )
    axes[1, 1].bar(
        tuple(position + width / 2 for position in positions),
        tuple(item.false_confidence_rate for item in final),
        width,
        label="False confidence",
    )
    axes[1, 1].set_xticks(
        positions,
        tuple(labels[item.strategy].replace(" ", "\n") for item in final),
    )
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].set_ylabel("Fraction of paired trials")
    axes[1, 1].set_title("Final model-mismatch decision audit")
    axes[1, 1].legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.suptitle(
        "Sequential experiment design: learning speed and model-risk detection\n"
        "Synthetic campaigns under a shared 65 J cap",
        fontsize=15,
    )
    figure.savefig(destination, dpi=170)
    save_figure_data(result, destination)
    return destination


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run adaptive, static, and heuristic virtual test campaigns."
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--first-seed", type=int, default=73_001)
    parser.add_argument("--output", type=Path, default=DEFAULT_ADAPTIVE_CAMPAIGN_PATH)
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="print the numerical report without importing optional Matplotlib",
    )
    arguments = parser.parse_args(argv)
    result = run_adaptive_experiment_campaign(
        AdaptiveCampaignConfig(
            trial_count=arguments.trials,
            first_seed=arguments.first_seed,
        ),
        progress=lambda message: print(message, flush=True),
    )
    print(format_adaptive_campaign_report(result))
    if not arguments.no_figure:
        destination = save_adaptive_campaign_figure(result, arguments.output)
        print(f"figure: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
