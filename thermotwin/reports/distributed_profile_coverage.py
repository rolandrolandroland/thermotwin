"""Report nonlinear profiles and repeated interval coverage for rho_e(T)."""

import argparse
import math
from pathlib import Path
from typing import Optional, Sequence

from ..figure_paths import default_figure_path, save_figure_data
from ..studies.distributed_profile_coverage import (
    DistributedProfileCoverageConfig,
    DistributedProfileCoverageStudyResult,
    run_distributed_profile_coverage_study,
)


def _number(value: float, *, digits: int = 6) -> str:
    return "not available" if not math.isfinite(value) else f"{value:.{digits}f}"


def format_distributed_profile_coverage_report(
    result: DistributedProfileCoverageStudyResult,
) -> str:
    """Describe nonlinear profiles, empirical coverage, and limitations."""

    config = result.config
    lines = [
        "Distributed resistivity: nonlinear profiles and interval coverage",
        "================================================================",
        "",
        "Question:",
        "  When ThermoTwin reports rho_e(T), how wide are the supported",
        "  coefficient intervals, and do nominal intervals contain synthetic truth",
        "  at their advertised rate under independent numerical/model mismatch?",
        "",
        "Frozen data and compute budget:",
        f"  independent truth: {config.truth_node_count} nodal points, SSPRK3, "
        f"dt={config.truth_time_step:.3e} s, smooth cubic rho_e(T)",
        f"  noise: {config.temperature_standard_deviation:.6f} K and "
        f"{config.voltage_standard_deviation:.6e} V",
        f"  conventional interval trials: {config.trial_count}",
        f"  paired PINN point-estimate trials: {config.pinn_trial_count}",
        f"  PINN epochs per fit: {config.inverse_pinn_epochs}",
        f"  bounds: {config.log_multiplier_bounds}",
        "",
        "Regularized variant:",
        f"  curvature weight={config.smoothness_weight:.6f}",
        f"  zero-centred shrinkage weight={config.shrinkage_weight:.6f}",
        "  the same explicit weights are used by conventional and PINN variants.",
        "",
        "Representative nonlinear re-optimized profiles:",
    ]
    if not result.representative_profiles:
        lines.append("  not run")
    for representative in result.representative_profiles:
        profile_result = representative.result
        lines.append(
            f"  {representative.case_name} / {representative.estimator_name}: "
            f"best logs={tuple(round(value, 6) for value in profile_result.best_fit.log_multipliers)}; "
            f"data chi-square={profile_result.best_data_chi_square:.6f}; "
            f"score={profile_result.best_penalized_score:.6f}"
        )
        for profile in profile_result.coefficient_profiles:
            interval = profile.intervals[-1]
            lines.append(
                f"    coefficient {profile.coefficient_index}: 95% log interval="
                f"[{interval.lower_log_multiplier:.6f}, "
                f"{interval.upper_log_multiplier:.6f}]; "
                f"bound limited=({interval.lower_hits_bound}, "
                f"{interval.upper_hits_bound})"
            )
    lines.extend(("", "Repeated-study summary:"))
    for summary in result.summaries:
        lines.append(
            f"  {summary.name}: completed={summary.completed_count}; "
            f"property RMSE={_number(summary.mean_property_relative_rmse)}; "
            f"property max error={_number(summary.mean_property_maximum_relative_error)}; "
            f"holdout voltage={_number(summary.mean_holdout_voltage_rmse)} V; "
            f"interval trials={summary.interval_trial_count}; "
            f"68% coefficient coverage={_number(summary.coefficient_coverage_68)}; "
            f"95% coefficient coverage={_number(summary.coefficient_coverage_95)}; "
            f"68% simultaneous coverage={_number(summary.simultaneous_coverage_68)}; "
            f"95% simultaneous coverage={_number(summary.simultaneous_coverage_95)}; "
            f"finite 95% fraction={_number(summary.finite_interval_fraction_95)}; "
            f"mean 95% log width={_number(summary.mean_log_interval_width_95)}"
        )
    lines.extend(("", "Trial-level point estimates:"))
    for trial in result.trials:
        lines.append(
            f"  trial {trial.trial_index}: neural seed={trial.seeds.neural}; "
            f"observation seeds={trial.seeds.observations}"
        )
        for estimator in trial.estimators:
            multipliers = (
                "training failed"
                if estimator.multipliers is None
                else "(" + ", ".join(f"{value:.6f}" for value in estimator.multipliers) + ")"
            )
            lines.append(
                f"    {estimator.name}: multipliers={multipliers}; "
                f"property max error={estimator.property_maximum_relative_error:.6f}; "
                f"holdout voltage={estimator.holdout_voltage_rmse:.6e} V"
            )
    lines.extend(
        (
            "",
            "Interpretation boundary:",
            "  Representative curves are nonlinear re-optimized profiles: one coefficient",
            "  is fixed and the others are re-optimized at every plotted point.",
            "  Repeated coverage uses a quadratic profile approximation centred on",
            "  each nonlinear multistart optimum because brute-force refitting every",
            "  profile at every noise trial is not CPU-practical.",
            "  Unregularized 68%/95% thresholds use one-parameter chi-square values",
            "  under declared independent Gaussian noise. Penalized intervals use the",
            "  same thresholds diagnostically and are not classical confidence",
            "  intervals. Empirical coverage is therefore the deciding check.",
            "  Twenty conventional trials and ten PINN trials are demonstrations with",
            "  wide binomial uncertainty, not precise failure-rate estimates.",
            "  Truth remains synthetic and shares the one-dimensional continuum model.",
        )
    )
    return "\n".join(lines)


def _plot_profiles(axis, representative, title: str) -> None:
    if representative is None:
        axis.text(0.5, 0.5, "profile not run", ha="center", va="center")
        axis.set_title(title)
        return
    for profile in representative.result.coefficient_profiles:
        axis.plot(
            tuple(point.fixed_log_multiplier for point in profile.points),
            tuple(point.delta_score for point in profile.points),
            marker="o",
            label=f"coefficient {profile.coefficient_index}",
        )
    axis.axhline(1.0, color="orange", linestyle="--", label="68% threshold")
    axis.axhline(3.841459, color="red", linestyle="--", label="95% threshold")
    axis.set_title(title)
    axis.set_xlabel("Fixed log multiplier")
    axis.set_ylabel("Profile score increase")
    axis.set_yscale("symlog", linthresh=1.0)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)


def save_distributed_profile_coverage_figure(
    result: DistributedProfileCoverageStudyResult,
    output: Path | str,
) -> Path:
    """Save nonlinear profile and empirical-coverage panels."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    def find(case, estimator):
        return next(
            (
                item
                for item in result.representative_profiles
                if item.case_name == case and item.estimator_name == estimator
            ),
            None,
        )

    _plot_profiles(
        axes[0, 0],
        find("full_bidirectional", "conventional_unregularized"),
        "Full data: unregularized nonlinear profiles",
    )
    _plot_profiles(
        axes[0, 1],
        find("full_bidirectional", "conventional_regularized"),
        "Full data: shrinkage + curvature profiles",
    )
    _plot_profiles(
        axes[1, 0],
        find("positive_temperature_voltage", "conventional_unregularized"),
        "One current direction: nonlinear profiles",
    )

    coverage_axis = axes[1, 1]
    interval_summaries = tuple(
        item for item in result.summaries if item.interval_trial_count > 0
    )
    indices = tuple(range(len(interval_summaries)))
    width = 0.35
    coverage_axis.bar(
        tuple(index - width / 2 for index in indices),
        tuple(item.coefficient_coverage_68 for item in interval_summaries),
        width=width,
        label="68% coefficient coverage",
    )
    coverage_axis.bar(
        tuple(index + width / 2 for index in indices),
        tuple(item.coefficient_coverage_95 for item in interval_summaries),
        width=width,
        label="95% coefficient coverage",
    )
    coverage_axis.axhline(0.6827, color="orange", linestyle="--")
    coverage_axis.axhline(0.95, color="red", linestyle="--")
    coverage_axis.set_xticks(
        indices,
        tuple(item.name.replace("conventional_", "") for item in interval_summaries),
        rotation=15,
    )
    coverage_axis.set_ylim(0.0, 1.05)
    coverage_axis.set_ylabel("Empirical fraction")
    coverage_axis.set_title("Repeated local-interval coverage")
    coverage_axis.grid(axis="y", alpha=0.25)
    coverage_axis.legend(fontsize=8)

    figure.savefig(output_path, dpi=170)
    save_figure_data(result, output_path)
    plt.close(figure)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run distributed nonlinear profiles and coverage audit."
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--pinn-trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--profile-points", type=int, default=5)
    parser.add_argument("--skip-profiles", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=default_figure_path(
            "distributed_profile_coverage.png",
            "DISTRIBUTED_PROFILE_COVERAGE.md",
        ),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        help="optional path for the complete text report",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_distributed_profile_coverage_study(
        DistributedProfileCoverageConfig(
            trial_count=args.trials,
            pinn_trial_count=args.pinn_trials,
            inverse_pinn_epochs=args.epochs,
            representative_profile_points=args.profile_points,
        ),
        run_representative_profiles=not args.skip_profiles,
        progress=lambda message: print(message, flush=True),
    )
    report = format_distributed_profile_coverage_report(result)
    print(report)
    if args.report_output is not None:
        report_output = args.report_output.expanduser().resolve()
        report_output.parent.mkdir(parents=True, exist_ok=True)
        report_output.write_text(report + "\n", encoding="utf-8")
        print(f"text report: {report_output}")
    output = save_distributed_profile_coverage_figure(result, args.output)
    print(f"figure: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
