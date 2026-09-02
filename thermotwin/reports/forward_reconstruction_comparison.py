"""Report a matched physics-informed versus data-only reconstruction study."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..figure_paths import default_figure_path, save_figure_data
from ..studies.forward_reconstruction_comparison import (
    ForwardReconstructionComparisonConfig,
    ForwardReconstructionComparisonResult,
    run_forward_reconstruction_comparison,
)


DEFAULT_FORWARD_RECONSTRUCTION_COMPARISON_PATH = default_figure_path(
    "forward_reconstruction_comparison.png",
    "FORWARD_RECONSTRUCTION_COMPARISON.md",
)


def format_forward_reconstruction_comparison_report(
    result: ForwardReconstructionComparisonResult,
) -> str:
    """Return the frozen design, all trials, and fairness boundary."""

    config = result.config
    lines = [
        "Physics-informed versus data-only transient reconstruction",
        "==========================================================",
        "",
        "Question:",
        "  When only sparse noisy exchanger temperatures are visible and both",
        "  sensors are missing around current turn-off, what does the known",
        "  thermoelectric physics add to transient reconstruction?",
        "",
        "Matched design:",
        f"  trials: {config.trial_count}",
        f"  architecture: three exactly joined subnetworks; width "
        f"{config.hidden_width}; hidden layers {config.hidden_layers}",
        f"  trainable parameters per model: {result.trainable_parameter_count_per_model}",
        f"  optimizer: Adam; epochs={config.epochs}; learning rate={config.learning_rate:.4g}",
        "  each pair starts from bit-identical weights and receives identical rows",
        "  both models receive the exact initial state and known switch locations",
        "  the only objective difference is the four normalized node residuals",
        f"  physics weight={config.physics_weight:g}; temperature scale="
        f"{config.observation_temperature_scale:.3f} K; rate scale="
        f"{config.residual_rate_scale:.3f} K/s",
        "",
        "Documented regression gate for every physics-informed trial:",
        f"  missing-exchanger RMSE <= "
        f"{config.criteria.maximum_missing_exchanger_rmse:.3f} K",
        f"  hidden-face RMSE <= {config.criteria.maximum_hidden_face_rmse:.3f} K",
        f"  node-residual RMS <= "
        f"{config.criteria.maximum_node_residual_rms:.4f} K/s",
        f"  normalized energy-rate closure RMS <= "
        f"{config.criteria.maximum_normalized_energy_rate_closure_rms:.1%}",
        f"  maximum absolute cumulative closure <= "
        f"{config.criteria.maximum_absolute_cumulative_energy_error:.1f} J",
        "",
        "Observations:",
        "  visible nodes: cold and hot exchangers only; both faces remain hidden",
        f"  sampling interval: {config.sampling_interval:.1f} s",
        f"  Gaussian noise: {config.noise_standard_deviation:.3f} K",
        f"  both sensors absent from {config.missing_start_time:.1f} through "
        f"{config.missing_end_time:.1f} s, including the 20 s turn-off",
        f"  rows: {result.complete_observation_count} complete, "
        f"{result.retained_observation_count} retained, "
        f"{result.removed_observation_count} removed",
        "",
        "Trial results:",
    ]
    for trial in result.trials:
        physics = trial.physics_informed
        data = trial.data_only
        lines.append(
            f"  trial {trial.trial_index}; observation seed={trial.observation_seed}; "
            f"neural seed={trial.neural_seed}; initial difference="
            f"{trial.initialization_maximum_absolute_difference:.1e}"
        )
        for metrics in (physics, data):
            lines.append(
                f"    {metrics.method}: retained noisy={metrics.retained_noisy_observation_rmse:.6f} K; "
                f"missing exchangers={metrics.missing_exchanger_rmse:.6f} K; "
                f"hidden faces={metrics.hidden_face_rmse:.6f} K; "
                f"node residual={metrics.node_residual_rms:.6f} K/s; "
                f"energy rate closure={metrics.energy_rate_closure_rms:.6f} W; "
                f"final cumulative closure={metrics.final_cumulative_energy_error:.6f} J; "
                f"training={metrics.losses.training_seconds:.3f} s"
            )
    lines.extend(("", "Means:"))
    for summary in result.summaries:
        lines.append(
            f"  {summary.method}: retained noisy="
            f"{summary.mean_retained_noisy_observation_rmse:.6f} K; "
            f"retained truth={summary.mean_retained_truth_rmse:.6f} K; "
            f"missing exchangers={summary.mean_missing_exchanger_rmse:.6f} K; "
            f"hidden faces={summary.mean_hidden_face_rmse:.6f} K; "
            f"hidden faces in gap={summary.mean_hidden_face_missing_interval_rmse:.6f} K; "
            f"all states={summary.mean_all_state_rmse:.6f} K; "
            f"node residual={summary.mean_node_residual_rms:.6f} K/s; "
            f"energy closure={summary.mean_energy_rate_closure_rms:.6f} W; "
            f"normalized closure={summary.mean_normalized_energy_rate_closure_rms:.3%}; "
            f"|final cumulative closure|="
            f"{summary.mean_absolute_final_cumulative_energy_error:.6f} J; "
            f"training={summary.mean_training_seconds:.3f} s"
        )
    lines.extend(
        (
            "",
            "Physics-informed reductions:",
            f"  missing-exchanger RMSE: "
            f"{result.physics_missing_exchanger_rmse_reduction_percent:.2f}%",
            f"  hidden-face RMSE: {result.physics_hidden_face_rmse_reduction_percent:.2f}%",
            f"  hidden-face gap RMSE: "
            f"{result.physics_hidden_face_gap_rmse_reduction_percent:.2f}%",
            f"  whole-system rate-closure error: "
            f"{result.physics_energy_rate_error_reduction_percent:.2f}%",
            f"  all three primary advantages in: "
            f"{result.physics_all_metric_advantage_count}/{config.trial_count} trials",
            f"  physics completion gate: "
            f"{result.physics_completion_gate_pass_count}/{config.trial_count} trials",
            "",
            "Independent energy diagnostic:",
            "  storage rate = sum(C_i dT_i/dt)",
            "  net input = electrical power + both reservoir heat inputs + external heat",
            "  closure error = storage rate - net input",
            "  this equation is not added as a fifth training loss; rates are",
            "  recomputed by automatic differentiation after training and cumulative",
            "  input is integrated separately within each constant-current segment",
            "  it is an independent post-training calculation, but not an independent",
            "  physical law: algebraically it follows from the four",
            "  node balances already used during training",
            "",
            "Interpretation boundary:",
            "  The comparison isolates the value of the stated physics under",
            "  same-equation synthetic truth. It is not hardware validation.",
            "  Same architecture and epoch count do not mean equal compute: the",
            "  physics-informed model evaluates derivatives and trains more slowly.",
            "  The data-only model has no direct labels for either face. Its poor",
            "  hidden-face result demonstrates non-identifiability without a physical",
            "  or statistical prior; it is not a universal neural-network baseline.",
            "  Both methods receive true initial temperatures and known switch times.",
            "  RK4 trajectories are used only after training for evaluation.",
            "  The numerical gates are maintained regression criteria, not",
            "  preregistered statistical acceptance thresholds.",
        )
    )
    return "\n".join(lines)


def save_forward_reconstruction_comparison_figure(
    result: ForwardReconstructionComparisonResult,
    output: Path | str,
) -> Path:
    """Save representative trajectories plus repeated error/closure metrics."""

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(13.5, 9.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    trace = result.representative_trace
    labels = ("Cold", "Hot")
    colors = ("tab:blue", "tab:red")

    axis = axes[0, 0]
    for offset, label, color in zip((2, 3), labels, colors):
        axis.plot(trace.time, trace.reference[offset], color=color, label=f"{label} truth")
        axis.plot(
            trace.time,
            trace.physics_informed[offset],
            color=color,
            linestyle="--",
            label=f"{label} physics-informed",
        )
        axis.plot(
            trace.time,
            trace.data_only[offset],
            color=color,
            linestyle=":",
            alpha=0.85,
            label=f"{label} data-only",
        )
        points = tuple(
            (time, temperature)
            for time, location, temperature in zip(
                trace.observation_time,
                trace.observation_location,
                trace.observed_temperature,
            )
            if location == f"{label.lower()}_exchanger"
        )
        axis.scatter(
            tuple(item[0] for item in points),
            tuple(item[1] for item in points),
            color=color,
            s=12,
            alpha=0.6,
        )
    axis.axvspan(
        result.config.missing_start_time,
        result.config.missing_end_time,
        color="0.7",
        alpha=0.25,
        label="missing observations",
    )
    axis.set_title("Visible exchanger reconstruction")
    axis.set_ylabel("Temperature (K)")
    axis.legend(fontsize=7, ncol=2)

    axis = axes[0, 1]
    for offset, label, color in zip((0, 1), labels, colors):
        axis.plot(trace.time, trace.reference[offset], color=color, label=f"{label} truth")
        axis.plot(
            trace.time,
            trace.physics_informed[offset],
            color=color,
            linestyle="--",
            label=f"{label} physics-informed",
        )
        axis.plot(
            trace.time,
            trace.data_only[offset],
            color=color,
            linestyle=":",
            label=f"{label} data-only",
        )
    axis.axvspan(
        result.config.missing_start_time,
        result.config.missing_end_time,
        color="0.7",
        alpha=0.25,
    )
    axis.set_title("Completely hidden face reconstruction")
    axis.set_ylabel("Temperature (K)")
    axis.legend(fontsize=7, ncol=2)

    methods = result.summaries
    positions = tuple(range(3))
    width = 0.36
    axis = axes[1, 0]
    axis.bar(
        tuple(index - width / 2 for index in positions),
        (
            methods[0].mean_retained_truth_rmse,
            methods[0].mean_missing_exchanger_rmse,
            methods[0].mean_hidden_face_rmse,
        ),
        width,
        label="physics-informed",
    )
    axis.bar(
        tuple(index + width / 2 for index in positions),
        (
            methods[1].mean_retained_truth_rmse,
            methods[1].mean_missing_exchanger_rmse,
            methods[1].mean_hidden_face_rmse,
        ),
        width,
        label="data-only",
    )
    axis.set_yscale("log")
    axis.set_xticks(positions, ("Retained\ntruth", "Missing\nexchangers", "Hidden\nfaces"))
    axis.set_ylabel("Mean RMSE (K), log scale")
    axis.set_title(f"Reconstruction across {result.config.trial_count} paired trials")
    axis.legend(fontsize=8)

    axis = axes[1, 1]
    axis.bar(
        tuple(index - width / 2 for index in (0, 1)),
        (
            methods[0].mean_energy_rate_closure_rms,
            methods[0].mean_absolute_final_cumulative_energy_error,
        ),
        width,
        label="physics-informed",
    )
    axis.bar(
        tuple(index + width / 2 for index in (0, 1)),
        (
            methods[1].mean_energy_rate_closure_rms,
            methods[1].mean_absolute_final_cumulative_energy_error,
        ),
        width,
        label="data-only",
    )
    axis.set_yscale("log")
    axis.set_xticks((0, 1), ("Rate closure\n(W)", "Final cumulative\nclosure (J)"))
    axis.set_title("Independent whole-system energy audit")
    axis.legend(fontsize=8)

    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 1].set_xlabel("Time (s)")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.suptitle(
        "Matched physics-informed versus observation-only reconstruction",
        fontsize=15,
    )
    figure.savefig(destination, dpi=170)
    save_figure_data(result, destination)
    return destination


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the matched sparse/missing forward reconstruction study."
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5_000)
    parser.add_argument("--first-seed", type=int, default=72_001)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_FORWARD_RECONSTRUCTION_COMPARISON_PATH,
    )
    args = parser.parse_args(argv)
    result = run_forward_reconstruction_comparison(
        ForwardReconstructionComparisonConfig(
            trial_count=args.trials,
            epochs=args.epochs,
            first_seed=args.first_seed,
        ),
        progress=lambda message: print(message, flush=True),
    )
    destination = save_forward_reconstruction_comparison_figure(result, args.output)
    print(format_forward_reconstruction_comparison_report(result))
    print(f"figure: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
