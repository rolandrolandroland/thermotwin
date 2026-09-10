"""Report the Stage 3 operating-decision realism stress test."""

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
    RateEstimate,
)
from ..studies.operating_decision_realism import (
    STAGE3_TRUTH_CONDITIONS,
    OperatingDecisionRealismConfig,
    OperatingDecisionRealismResult,
    nominal_realistic_margins,
    run_operating_decision_realism,
)
from ..studies.sensor_model_discrimination import VOLTAGE


DEFAULT_OPERATING_DECISION_REALISM_PATH = default_figure_path(
    "operating_decision_realism.png",
    "OPERATING_DECISION_REALISM.md",
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
    "temperature_dependent_contact": (
        "Temperature-dependent-contact truth (missing-family stress)"
    ),
}

_PLOT_TRUTH_LABELS = {
    "matched_four_state": "Matched four-state truth",
    "extra_interface_mass": "Extra-interface-mass truth",
    "temperature_dependent_contact": (
        "Temperature-dependent contact\n(missing-family stress)"
    ),
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


def format_operating_decision_realism_report(
    result: OperatingDecisionRealismResult,
) -> str:
    """Return Stage 3 decision quality, abstentions, and diagnostic effort."""

    config = result.config
    nominal_margins = nominal_realistic_margins(config)
    voltage_noise = dict(config.sensor.channel_noise)[VOLTAGE]
    voltage_offset_scale = config.sensor.run_bias_noise_ratio * voltage_noise
    lines = [
        "Operating-decision Stage 3 realism stress test",
        "================================================",
        "",
        "Development question:",
        "  Do the fixed diagnostic packages still support an operating decision",
        "  after voltage confounding, face-probe loading, and missing-family",
        "  thermal-contact physics are introduced?",
        "",
        "Blinded protocol:",
        f"  paired synthetic devices per truth family: {config.sensor.trial_count}",
        "  three truth families share the frozen Stage 2 policies and final schedule",
        "  two constant-contact candidate models are fitted from acquisition data",
        "  each candidate uses three declared starting points",
        f"  each fit start uses {config.sensor.fit_iterations} Gauss-Newton iteration(s)",
        "  a distinct verification schedule checks adequacy without physical refit",
        "  the final response is generated only after the decision has been saved",
        "  the temporary face probe is removed before the final operating run",
        "  every policy therefore forecasts the same unloaded target device",
        "",
        "Realism stressors:",
        f"  terminal series resistance: nominal "
        f"{config.series_resistance_nominal:.3f} ohm; fit bounds "
        f"{config.series_resistance_bounds[0]:.3f}-"
        f"{config.series_resistance_bounds[1]:.3f} ohm",
        f"  voltage white noise / run-offset standard deviation: "
        f"{voltage_noise:.4f} / {voltage_offset_scale:.4f} V",
        f"  temporary face-probe capacitance: nominal "
        f"{config.face_sensor_capacitance_nominal:.2f} J/K; fit bounds "
        f"{config.face_sensor_capacitance_bounds[0]:.2f}-"
        f"{config.face_sensor_capacitance_bounds[1]:.2f} J/K",
        f"  temporary face-probe response time: nominal "
        f"{config.face_sensor_response_nominal:.2f} s; fit bounds "
        f"{config.face_sensor_response_bounds[0]:.2f}-"
        f"{config.face_sensor_response_bounds[1]:.2f} s",
        f"  missing-family contact coefficient range: "
        f"{config.contact_beta_bounds[0]:.3f}-"
        f"{config.contact_beta_bounds[1]:.3f} 1/K",
        "  series-contact Joule heat enters the thermal dynamics; voltage offsets",
        "  remain measurement effects and do not enter physical electrical power",
        "",
        "Frozen operating decision:",
        f"  acceptable cold-face band: {config.band.lower_temperature:.2f} to "
        f"{config.band.upper_temperature:.2f} K",
        f"  verification normalized-score threshold: "
        f"{config.verification_score_threshold:.3f}",
        f"  uncalibrated local delta-margin multiplier: "
        f"{config.local_interval_multiplier:.2f}",
        "  approve requires the complete verified margin envelope to be nonnegative",
        "  reject requires the complete verified margin envelope to be negative",
        "  every other case returns insufficient evidence",
        "",
        "Nominal unloaded final-schedule margins:",
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
        probe_note = (
            "; probe loaded during its prospective acquisition and verification runs"
            if policy.name == FIXED_FACE_TEMPERATURE
            else ""
        )
        lines.append(
            f"  {_policy_label(policy.name)}: "
            f"{summary.acquisition_run_count} acquisition run(s), "
            f"{summary.diagnostic_run_count} including verification, "
            f"{summary.mean_total_diagnostic_energy:.2f} J nominal modeled "
            f"diagnostic electrical energy, "
            f"{summary.extra_sensor_count} added channel(s){probe_note}"
        )

    lines.extend(("", "Development stress-test outcomes:"))
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
                f"energized-schedule seconds (resets not modeled), "
                f"{summary.mean_total_diagnostic_energy:.2f} J; "
                f"mean decision computation="
                f"{summary.mean_decision_computation_seconds:.3f} s; "
                f"numerical failures={summary.numerical_failures}",
            )
        )

    lines.extend(("", "Cross-family fixed-policy comparison:"))
    for policy in result.policies:
        selected = tuple(
            summary
            for summary in result.summaries
            if summary.policy_name == policy.name
        )
        determinate = sum(
            summary.approvals + summary.rejections for summary in selected
        )
        total = sum(summary.trial_count for summary in selected)
        interval_covered = sum(
            summary.interval_coverage.numerator for summary in selected
        )
        interval_total = sum(
            summary.interval_coverage.denominator for summary in selected
        )
        lines.append(
            f"  {_policy_label(policy.name)}: decisions {determinate}/{total}; "
            f"interval coverage {interval_covered}/{interval_total}"
        )

    family_c_verifications = tuple(
        verification
        for trial in result.trials
        if trial.truth_condition == "temperature_dependent_contact"
        for verification in trial.saved.verifications
    )
    passed_family_c = sum(item.passed for item in family_c_verifications)
    inadequate_family_c = sum(
        item.failure_reason == "inadequate_verification"
        for item in family_c_verifications
    )
    bound_family_c = sum(
        item.failure_reason == "fit_reached_bound"
        for item in family_c_verifications
    )
    lines.extend(
        (
            "",
            "Missing-family adequacy diagnostic:",
            f"  Family C candidate-policy fits: {len(family_c_verifications)}",
            f"  passed the fixed verification gate: {passed_family_c}",
            f"  rejected for verification score: {inadequate_family_c}",
            f"  excluded for a parameter-bound hit: {bound_family_c}",
            "  the frozen verification score did not identify the omitted contact law",
        )
    )

    lines.extend(
        (
            "",
            "Interpretation boundary:",
            "  This is a synthetic development stress test. It does not provide",
            "  hardware validation, an operating-safety certification, or a final",
            "  comparison of manufactured-device diagnostic costs.",
            "  The temperature-dependent-contact truth is deliberately outside both",
            "  fitted candidate families. Its outcomes measure missing-family stress;",
            "  abstention is an intended response when verification detects mismatch.",
            "  The face probe changes diagnostic dynamics and is physically absent",
            "  from the final evaluation, so its policy must transfer to an unloaded",
            "  device rather than predict the loaded measurement setup.",
            "  Local covariance propagation and the verification threshold remain",
            "  development choices. Calibration, the decision-directed selector, and",
            "  a frozen fresh-device evaluation remain later stages.",
        )
    )
    return "\n".join(lines)


def save_operating_decision_realism_figure(
    result: OperatingDecisionRealismResult,
    output: Path | str,
) -> Path:
    """Save three truth-specific decision panels and machine-readable sidecars."""

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(19.0, 6.6), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3, sharey=True)

    reported_truths = {
        summary.truth_condition for summary in result.summaries
    }
    if reported_truths != set(STAGE3_TRUTH_CONDITIONS):
        raise ValueError("the Stage 3 figure requires all three truth families")

    components = (
        ("Correct approval", "#2a9d8f"),
        ("False approval", "#d62728"),
        ("Correct rejection", "#457b9d"),
        ("False rejection", "#f4a261"),
        ("Insufficient evidence", "#bdbdbd"),
    )
    policy_names = tuple(policy.name for policy in result.policies)
    positions = tuple(range(len(policy_names)))

    for axis, truth_condition in zip(axes, STAGE3_TRUTH_CONDITIONS):
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
        axis.set_title(_PLOT_TRUTH_LABELS[truth_condition])
        axis.set_xlabel("Fixed diagnostic policy")
        axis.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Fraction of paired virtual devices")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=8)
    figure.suptitle(
        "Stage 3 realism stress test: blinded decisions on an unloaded target\n"
        "Family C tests missing physics; annotations show diagnostic effort",
        fontsize=14,
    )
    figure.savefig(destination, dpi=170)
    save_figure_data(result, destination)
    return destination


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Stage 3 operating-decision realism stress test."
    )
    defaults = OperatingDecisionRealismConfig()
    parser.add_argument(
        "--trials",
        type=int,
        default=defaults.sensor.trial_count,
        help="paired virtual devices per truth family",
    )
    parser.add_argument(
        "--first-seed",
        type=int,
        default=defaults.sensor.first_seed,
    )
    parser.add_argument(
        "--fit-iterations",
        type=int,
        default=defaults.sensor.fit_iterations,
        help="Gauss-Newton iterations for each declared fit start",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OPERATING_DECISION_REALISM_PATH,
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
            fit_iterations=arguments.fit_iterations,
        ),
    )
    result = run_operating_decision_realism(
        config,
        progress=lambda message: print(message, flush=True),
    )
    print(format_operating_decision_realism_report(result))
    if not arguments.no_figure:
        destination = save_operating_decision_realism_figure(
            result,
            arguments.output,
        )
        print(f"figure: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
