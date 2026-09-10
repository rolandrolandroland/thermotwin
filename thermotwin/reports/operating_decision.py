"""Report the blinded fixed-policy operating-decision development pilot."""

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

from .paths import default_figure_path, save_figure_data
from ..studies.operating_decision import (
    FIXED_FACE_TEMPERATURE,
    FIXED_THERMAL,
    FIXED_VOLTAGE,
    STOP_NOW,
    OperatingDecisionConfig,
    OperatingDecisionResult,
    RateEstimate,
    nominal_operating_margins,
    run_operating_decision,
)


DEFAULT_OPERATING_DECISION_PATH = default_figure_path(
    "operating_decision.png",
    "OPERATING_DECISION_EXPERIMENT.md",
)


_POLICY_LABELS = {
    STOP_NOW: "Stop now",
    FIXED_THERMAL: "More thermal tests",
    FIXED_VOLTAGE: "Add voltage",
    FIXED_FACE_TEMPERATURE: "Add face temperature",
}

_PLOT_POLICY_LABELS = {
    STOP_NOW: "Stop\nnow",
    FIXED_THERMAL: "More\nthermal",
    FIXED_VOLTAGE: "Add\nvoltage",
    FIXED_FACE_TEMPERATURE: "Add face\ntemperature",
}

_TRUTH_LABELS = {
    "matched_four_state": "Matched four-state truth",
    "extra_interface_mass": "Extra-interface-mass truth",
}


def _policy_label(name: str) -> str:
    return _POLICY_LABELS.get(name, name.replace("_", " "))


def _truth_label(name: str) -> str:
    return _TRUTH_LABELS.get(name, name.replace("_", " "))


def _format_rate(rate: RateEstimate) -> str:
    if rate.rate is None:
        return f"N/A ({rate.numerator}/{rate.denominator})"
    return (
        f"{rate.rate:.1%} ({rate.numerator}/{rate.denominator}; "
        f"95% Wilson {rate.lower_95:.1%}-{rate.upper_95:.1%})"
    )


def format_operating_decision_report(result: OperatingDecisionResult) -> str:
    """Return decision quality and diagnostic effort without hiding abstentions."""

    config = result.config
    nominal_margins = nominal_operating_margins(config)
    lines = [
        "Operating-decision Stage 2 development pilot",
        "==============================================",
        "",
        "Decision question:",
        "  Which fixed diagnostic package supports approve, reject, or insufficient",
        "  evidence for one previously untried operating schedule?",
        "",
        "Blinded protocol:",
        f"  paired synthetic trials per truth condition: {config.sensor.trial_count}",
        f"  acquisition observations fit both candidates from "
        f"{len(config.fit_initial_log_multipliers)} declared starts",
        "  the lowest complete acquisition objective selects each candidate fit",
        "  one distinct verification schedule checks adequacy without physical refit",
        "  candidate margin intervals are enveloped rather than selecting one model",
        "  the final response is generated only after the decision has been saved",
        "  final cold-face observations never enter fitting, verification, or decision",
        "",
        "Frozen operating decision:",
        f"  acceptable cold-face band: {config.band.lower_temperature:.2f} to "
        f"{config.band.upper_temperature:.2f} K",
        f"  verification normalized-score threshold: "
        f"{config.verification_score_threshold:.3f}",
        f"  uncalibrated local delta-margin multiplier: "
        f"{config.local_interval_multiplier:.2f}",
        "  verification and interval thresholds are development choices",
        "  approve requires the complete verified margin envelope to be nonnegative",
        "  reject requires the complete verified margin envelope to be negative",
        "  every other case returns insufficient evidence",
        "",
        "Nominal final-schedule margins:",
    ]
    for model_name, margin in nominal_margins:
        lines.append(f"  {model_name}: {margin:.4f} K")

    lines.extend(("", "Fixed diagnostic packages:"))
    reference_truth = result.summaries[0].truth_condition
    for policy in result.policies:
        summary = next(
            item
            for item in result.summaries
            if item.truth_condition == reference_truth
            and item.policy_name == policy.name
        )
        lines.append(
            f"  {_policy_label(policy.name)}: "
            f"{summary.acquisition_run_count} acquisition run(s), "
            f"{summary.diagnostic_run_count} including verification, "
            f"{summary.mean_total_diagnostic_energy:.2f} J modeled net diagnostic "
            f"electrical energy, {summary.extra_sensor_count} added channel(s)"
        )

    lines.extend(("", "Pilot outcomes:"))
    for summary in result.summaries:
        lines.extend(
            (
                f"  {_truth_label(summary.truth_condition)} / "
                f"{_policy_label(summary.policy_name)}:",
                f"    decisions: approve={summary.approvals}; "
                f"reject={summary.rejections}; "
                f"insufficient={summary.insufficient_evidence}; "
                f"true pass/violate={summary.true_passing}/{summary.true_violating}",
                f"    false approvals among approvals: "
                f"{_format_rate(summary.false_approvals)}",
                f"    missed violations: {_format_rate(summary.missed_violations)}",
                f"    false rejections among rejections: "
                f"{_format_rate(summary.false_rejections)}",
                f"    decision coverage: {_format_rate(summary.decision_coverage)}; "
                f"abstentions: {_format_rate(summary.abstentions)}",
                f"    interval coverage where an interval exists: "
                f"{_format_rate(summary.interval_coverage)}",
                f"    diagnostic effort: {summary.diagnostic_run_count} runs, "
                f"{summary.mean_energized_schedule_time_seconds:.0f} "
                f"energized-schedule "
                f"seconds (resets not modeled), "
                f"{summary.mean_total_diagnostic_energy:.2f} J; "
                f"mean decision computation="
                f"{summary.mean_decision_computation_seconds:.3f} s; "
                f"numerical failures={summary.numerical_failures}",
            )
        )

    lines.extend(
        (
            "",
            "Interpretation boundary:",
            "  This is a synthetic development pilot, not a frozen evaluation result",
            "  and not hardware validation. Its rates are diagnostic rather than claims",
            "  about manufactured devices, prototype savings, or operating safety.",
            "  The untouched-final split prevents direct response leakage, but both truth",
            "  conditions still belong to the fitted candidate families. Voltage",
            "  confounding, face-sensor thermal loading, a missing-physics truth family,",
            "  calibrated uncertainty, and the decision-directed selector remain later",
            "  stages. Fixed-policy comparisons here guide that work; they do not choose",
            "  a sensor for a physical device.",
            "  Local covariance propagation is provisional because the scalar margin is",
            "  a minimum over time; bootstrap calibration is required before evaluation.",
        )
    )
    return "\n".join(lines)


def save_operating_decision_figure(
    result: OperatingDecisionResult,
    output: Path | str,
) -> Path:
    """Save two truth-specific panels of correct, incorrect, and withheld decisions."""

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(13.5, 6.4), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2, sharey=True)

    truth_conditions = tuple(
        dict.fromkeys(summary.truth_condition for summary in result.summaries)
    )
    if len(truth_conditions) != 2:
        raise ValueError("the Stage 2 pilot figure requires two truth conditions")

    components = (
        ("Correct approval", "#2a9d8f"),
        ("False approval", "#d62728"),
        ("Correct rejection", "#457b9d"),
        ("False rejection", "#f4a261"),
        ("Insufficient evidence", "#bdbdbd"),
    )
    policy_names = tuple(policy.name for policy in result.policies)
    positions = tuple(range(len(policy_names)))

    for axis, truth_condition in zip(axes, truth_conditions):
        summaries = tuple(
            next(
                item
                for item in result.summaries
                if item.truth_condition == truth_condition
                and item.policy_name == policy_name
            )
            for policy_name in policy_names
        )
        values_by_component = (
            tuple(
                (item.approvals - item.false_approvals.numerator) / item.trial_count
                for item in summaries
            ),
            tuple(
                item.false_approvals.numerator / item.trial_count
                for item in summaries
            ),
            tuple(
                (item.rejections - item.false_rejections.numerator) / item.trial_count
                for item in summaries
            ),
            tuple(
                item.false_rejections.numerator / item.trial_count
                for item in summaries
            ),
            tuple(item.insufficient_evidence / item.trial_count for item in summaries),
        )
        bottoms = [0.0] * len(summaries)
        for (label, color), values in zip(components, values_by_component):
            axis.bar(
                positions,
                values,
                bottom=bottoms,
                width=0.72,
                label=label,
                color=color,
            )
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
        for position, summary in zip(positions, summaries):
            axis.text(
                position,
                1.035,
                f"{summary.diagnostic_run_count} runs\n"
                f"{summary.mean_total_diagnostic_energy:.1f} J",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axis.set_xticks(
            positions,
            tuple(
                _PLOT_POLICY_LABELS.get(name, _policy_label(name))
                for name in policy_names
            ),
        )
        axis.set_ylim(0.0, 1.20)
        axis.set_title(_truth_label(truth_condition))
        axis.set_xlabel("Fixed diagnostic policy")
        axis.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Fraction of paired virtual devices")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=8)
    figure.suptitle(
        "Stage 2 development pilot: blinded decisions on an untouched final schedule\n"
        "Annotations show total diagnostic runs and modeled net electrical energy",
        fontsize=14,
    )
    figure.savefig(destination, dpi=170)
    save_figure_data(result, destination)
    return destination


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the blinded fixed-policy operating-decision development pilot."
        )
    )
    defaults = OperatingDecisionConfig()
    parser.add_argument(
        "--trials",
        type=int,
        default=defaults.sensor.trial_count,
        help="paired virtual devices per truth condition",
    )
    parser.add_argument(
        "--first-seed",
        type=int,
        default=defaults.sensor.first_seed,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OPERATING_DECISION_PATH,
    )
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="print the numerical report without importing optional Matplotlib",
    )
    arguments = parser.parse_args(argv)
    config = replace(
        defaults,
        sensor=replace(
            defaults.sensor,
            trial_count=arguments.trials,
            first_seed=arguments.first_seed,
        ),
    )
    result = run_operating_decision(
        config,
        progress=lambda message: print(message, flush=True),
    )
    print(format_operating_decision_report(result))
    if not arguments.no_figure:
        destination = save_operating_decision_figure(result, arguments.output)
        print(f"figure: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
