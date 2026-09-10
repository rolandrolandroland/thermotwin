"""Report Stage 4 calibration and decision-directed selection."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from .paths import default_figure_path, save_figure_data
from ..studies.operating_decision import (
    APPROVE,
    FIXED_FACE_TEMPERATURE,
    FIXED_THERMAL,
    FIXED_VOLTAGE,
    INSUFFICIENT_EVIDENCE,
    REJECT,
    STOP_NOW,
    RateEstimate,
)
from ..studies.operating_decision_calibration import (
    ALL_FAMILIES,
    CALIBRATION_SPLIT,
    DECISION_DIRECTED_SELECTOR,
    REHEARSAL_SPLIT,
    STAGE4_PROCEDURES,
    CalibrationArtifact,
    OperatingDecisionCalibrationConfig,
    OperatingDecisionCalibrationResult,
    ProcedureSummary,
    run_operating_decision_calibration,
)
from ..studies.operating_decision_realism import STAGE3_TRUTH_CONDITIONS


DEFAULT_OPERATING_DECISION_CALIBRATION_PATH = default_figure_path(
    "operating_decision_calibration.png",
    "OPERATING_DECISION_CALIBRATION.md",
)


_PROCEDURE_LABELS = {
    STOP_NOW: "Stop now",
    FIXED_THERMAL: "More thermal tests",
    FIXED_VOLTAGE: "Add voltage",
    FIXED_FACE_TEMPERATURE: "Add face temperature",
    DECISION_DIRECTED_SELECTOR: "Decision-directed selector",
}

_PLOT_LABELS = {
    STOP_NOW: "Stop\nnow",
    FIXED_THERMAL: "More\nthermal",
    FIXED_VOLTAGE: "Add\nvoltage",
    FIXED_FACE_TEMPERATURE: "Add face\ntemperature",
    DECISION_DIRECTED_SELECTOR: "Decision\nselector",
}

_TRUTH_LABELS = {
    "matched_four_state": "Matched four-state",
    "extra_interface_mass": "Extra interface mass",
    "temperature_dependent_contact": "Temperature-dependent contact",
    ALL_FAMILIES: "All three families",
}


def _procedure_label(name: str) -> str:
    return _PROCEDURE_LABELS.get(name, name.replace("_", " "))


def _truth_label(name: str) -> str:
    return _TRUTH_LABELS.get(name, name.replace("_", " "))


def _format_rate(rate: RateEstimate) -> str:
    if rate.rate is None:
        return f"N/A ({rate.numerator}/{rate.denominator})"
    if rate.lower_95 is None or rate.upper_95 is None:
        return (
            f"{rate.rate:.1%} ({rate.numerator}/{rate.denominator}; "
            "descriptive rate; CI suppressed for fitted calibration data or "
            "clustered family rows)"
        )
    return (
        f"{rate.rate:.1%} ({rate.numerator}/{rate.denominator}; "
        f"95% Wilson {rate.lower_95:.1%}-{rate.upper_95:.1%})"
    )


def _summary(
    result: OperatingDecisionCalibrationResult,
    split: str,
    truth: str,
    procedure: str,
) -> ProcedureSummary:
    matches = tuple(
        item
        for item in result.summaries
        if item.split == split
        and item.truth_condition == truth
        and item.procedure_name == procedure
    )
    if len(matches) != 1:
        raise ValueError("Stage 4 report has no unique summary cell")
    return matches[0]


def format_operating_decision_calibration_report(
    result: OperatingDecisionCalibrationResult,
) -> str:
    """Return the Stage 4 calibration and fresh-seed rehearsal report."""

    config = result.config
    artifact = result.artifact
    lines = [
        "Operating-decision Stage 4 calibration and selector",
        "===================================================",
        "",
        "Purpose:",
        "  Calibrate verification and final-margin uncertainty under all three",
        "  Stage 3 truth families, freeze a selector that sees only the common",
        "  initial acquisition, and rehearse it once on fresh development seeds.",
        "",
        "Partition freeze:",
        f"  matched-pipeline gate development: "
        f"{artifact.gate_development_block_count} paired blocks/family; "
        f"first seed {artifact.gate_development_first_seed}",
        f"  calibration: {artifact.calibration_block_count} paired blocks/family; "
        f"first seed {artifact.calibration_first_seed}",
        f"  fresh-seed rehearsal: {artifact.rehearsal_block_count} paired "
        f"blocks/family; first seed {artifact.rehearsal_first_seed}",
        f"  reserved Stage 5 evaluation: {artifact.reserved_evaluation_block_count} "
        f"blocks/family; first seed {artifact.reserved_evaluation_first_seed}",
        "  the reserved evaluation namespace was checked for collisions and was",
        "  not instantiated by this run",
        f"  protocol digest: {artifact.protocol_digest}",
        "",
        "Frozen selector:",
        "  fit both constant-contact candidates to the initial 0.8 A acquisition",
        "  choose stop-now only when both finite, non-bound candidate intervals",
        "  form an envelope wholly on one side of zero; otherwise choose voltage",
        "  verification data, action-specific observations, truth-family labels,",
        "  trial identifiers, and final responses are absent from the selector input",
        "",
        "Matched-pipeline verification gate:",
        f"  target correct-candidate block retention: "
        f"{config.gate_target_block_retention:.1%}",
        "  each development block takes the larger correct-candidate score from",
        "  matched four-state and extra-interface-mass truth; Family C is excluded",
        f"  conditional exact-prediction noise reference: "
        f"{config.gate_null_quantile:.1%} chi-square quantile from "
        f"{config.gate_monte_carlo_draws} draws/policy",
    ]
    for gate in artifact.verification_gates:
        lines.append(
            f"  {_procedure_label(gate.policy_name)}: Q <= {gate.threshold:.4f} "
            f"(matched rank {gate.matched_development_rank}/"
            f"{gate.matched_development_block_count}; noise reference "
            f"{gate.noise_reference_threshold:.4f}; "
            f"{gate.observation_count} observations)"
        )
    lines.extend(
        (
            "  this calibrates score retention for the two represented matched",
            "  pipelines; bound and numerical failures remain separate exclusions",
            "  it is not claimed to identify every omitted constitutive law",
            "",
            "Blockwise split-conformal margin calibration:",
            f"  target simultaneous coverage across the three family variants: "
            f"{config.target_block_coverage:.1%}",
            "  each fixed policy is calibrated as a complete procedure; the",
            "  selector receives its own post-selection calibration",
            "  a missing envelope from fitting or gate failure is treated as an",
            "  unbounded set; a finite zero-crossing envelope remains emitted",
            "  every insufficient-evidence outcome remains in decision/resource metrics",
        )
    )
    for item in artifact.procedure_calibrations:
        raw_rate = (
            item.raw_interval_covered_count / item.emitted_interval_count
            if item.emitted_interval_count
            else None
        )
        calibrated_rate = (
            item.calibrated_interval_covered_count / item.emitted_interval_count
            if item.emitted_interval_count
            else None
        )
        raw_text = "N/A" if raw_rate is None else f"{raw_rate:.1%}"
        calibrated_text = (
            "N/A" if calibrated_rate is None else f"{calibrated_rate:.1%}"
        )
        lines.append(
            f"  {_procedure_label(item.procedure_name)}: +/−"
            f"{item.additive_margin_padding:.4f} K; rank "
            f"{item.conformal_rank}/{item.block_count}; emitted "
            f"{item.emitted_interval_count}; raw/calibrated row coverage "
            f"{raw_text}/{calibrated_text}"
        )

    for split, heading in (
        (CALIBRATION_SPLIT, "Calibration-cohort diagnostics"),
        (REHEARSAL_SPLIT, "Fresh-seed development rehearsal"),
    ):
        lines.extend(("", f"{heading}:"))
        if split == CALIBRATION_SPLIT:
            lines.append(
                "  rates are descriptive because the gate/paddings were fitted here; CIs are suppressed"
            )
        for truth in STAGE3_TRUTH_CONDITIONS:
            lines.append(f"  {_truth_label(truth)}:")
            for procedure in STAGE4_PROCEDURES:
                summary = _summary(result, split, truth, procedure)
                lines.extend(
                    (
                        f"    {_procedure_label(procedure)}: approve/reject/"
                        f"insufficient={summary.approvals}/{summary.rejections}/"
                        f"{summary.insufficient_evidence}; true pass/violate="
                        f"{summary.true_passing}/{summary.true_violating}",
                        f"      false approvals={_format_rate(summary.false_approvals)}; "
                        f"false rejections={_format_rate(summary.false_rejections)}",
                        f"      decision coverage={_format_rate(summary.decision_coverage)}; "
                        f"interval coverage={_format_rate(summary.calibrated_interval_coverage)}",
                        f"      effort={summary.mean_diagnostic_run_count:.2f} runs, "
                        f"{summary.mean_total_diagnostic_energy:.2f} J, "
                        f"{summary.mean_extra_sensor_count:.2f} added sensor(s)",
                    )
                )
        lines.append("  Cross-family comparison:")
        for procedure in STAGE4_PROCEDURES:
            summary = _summary(result, split, ALL_FAMILIES, procedure)
            losses = ", ".join(
                f"{item.scenario_name}={item.mean_loss:.3f}"
                for item in summary.expected_losses
            )
            lines.extend(
                (
                    f"    {_procedure_label(procedure)}: decision coverage "
                    f"{_format_rate(summary.decision_coverage)}; calibrated interval "
                    f"coverage {_format_rate(summary.calibrated_interval_coverage)}; "
                    f"procedure-set block coverage "
                    f"{_format_rate(summary.simultaneous_block_coverage)}",
                    f"      mean runs/energy/sensors="
                    f"{summary.mean_diagnostic_run_count:.2f}/"
                    f"{summary.mean_total_diagnostic_energy:.2f} J/"
                    f"{summary.mean_extra_sensor_count:.2f}; empirical mean loss: {losses}",
                )
            )

    rehearsal_selector = _summary(
        result,
        REHEARSAL_SPLIT,
        ALL_FAMILIES,
        DECISION_DIRECTED_SELECTOR,
    )
    action_text = ", ".join(
        f"{_procedure_label(name)}={count}"
        for name, count in rehearsal_selector.selected_policy_counts
    )
    family_c_rows = tuple(
        item
        for item in result.rehearsal_outcomes
        if item.truth_condition == "temperature_dependent_contact"
        and item.procedure_name in (
            STOP_NOW,
            FIXED_THERMAL,
            FIXED_VOLTAGE,
            FIXED_FACE_TEMPERATURE,
        )
    )
    lines.extend(
        (
            "",
            "Selector and adequacy diagnostics:",
            f"  rehearsal selector actions across 30 family rows: {action_text}",
            f"  Family C fixed-policy candidate fits rejected by the calibrated gate: "
            f"{sum(item.score_rejected_candidate_count for item in family_c_rows)}/"
            f"{2 * len(family_c_rows)}",
            "  Family C rejection is reported as a diagnostic, not interpreted as",
            "  proof that the scalar verification score detects missing physics",
            "",
            "Interpretation boundary:",
            "  Stage 4 remains synthetic development evidence. The conformal result",
            "  assumes exchangeability with the declared calibration generator and",
            "  gives no hardware guarantee or protection against a new truth family.",
            "  The rehearsal seeds were used once after the artifact was frozen and",
            "  did not update the gate, padding, or selector. The reserved Stage 5",
            "  cohort remains untouched for the final paired comparison.",
            "  Expected-loss weights are declared engineering preferences, not dollar",
            "  costs; the raw error, abstention, run, energy, and sensor counts remain",
            "  the primary evidence.",
            "  Procedure-set block coverage treats only a missing envelope as an",
            "  unbounded set. Finite zero-crossing envelopes remain scored; conditional",
            "  emitted-interval coverage is a separate untargeted diagnostic.",
        )
    )
    return "\n".join(lines)


def _artifact_payload(
    artifact: CalibrationArtifact,
    *,
    source_revision: Optional[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_revision": source_revision,
        "protocol_version": artifact.protocol_version,
        "protocol_digest": artifact.protocol_digest,
        "realism_protocol_digest": artifact.realism_protocol_digest,
        "partitions": {
            "gate_development": {
                "first_seed": artifact.gate_development_first_seed,
                "paired_blocks_per_family": artifact.gate_development_block_count,
            },
            "calibration": {
                "first_seed": artifact.calibration_first_seed,
                "paired_blocks_per_family": artifact.calibration_block_count,
            },
            "rehearsal": {
                "first_seed": artifact.rehearsal_first_seed,
                "paired_blocks_per_family": artifact.rehearsal_block_count,
            },
            "reserved_evaluation": {
                "first_seed": artifact.reserved_evaluation_first_seed,
                "paired_blocks_per_family": artifact.reserved_evaluation_block_count,
                "instantiated_in_stage4": False,
            },
        },
        "verification_gates": [item._asdict() for item in artifact.verification_gates],
        "selector": artifact.selector.__dict__,
        "procedure_calibrations": [
            item._asdict() for item in artifact.procedure_calibrations
        ],
    }


def save_calibration_artifact(
    artifact: CalibrationArtifact,
    output: Path | str,
    *,
    source_revision: Optional[str] = None,
) -> Path:
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(
            _artifact_payload(artifact, source_revision=source_revision),
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
    return destination


def save_operating_decision_calibration_figure(
    result: OperatingDecisionCalibrationResult,
    output: Path | str,
) -> Path:
    """Save the Stage 4 rehearsal comparison and calibration diagnostics."""

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(17.0, 10.0), constrained_layout=True)
    FigureCanvasAgg(figure)
    decision_axis, coverage_axis, action_axis, effort_axis = figure.subplots(2, 2).flat

    combined = tuple(
        _summary(result, REHEARSAL_SPLIT, ALL_FAMILIES, procedure)
        for procedure in STAGE4_PROCEDURES
    )
    positions = tuple(range(len(combined)))
    components = (
        (
            "Correct approval",
            tuple((item.approvals - item.false_approvals.numerator) / item.trial_count for item in combined),
            "#2a9d8f",
        ),
        (
            "False approval",
            tuple(item.false_approvals.numerator / item.trial_count for item in combined),
            "#d62728",
        ),
        (
            "Correct rejection",
            tuple((item.rejections - item.false_rejections.numerator) / item.trial_count for item in combined),
            "#457b9d",
        ),
        (
            "False rejection",
            tuple(item.false_rejections.numerator / item.trial_count for item in combined),
            "#f4a261",
        ),
        (
            "Insufficient",
            tuple(item.insufficient_evidence / item.trial_count for item in combined),
            "#bdbdbd",
        ),
    )
    bottoms = [0.0] * len(combined)
    for label, values, color in components:
        decision_axis.bar(positions, values, bottom=bottoms, color=color, label=label)
        bottoms = [left + right for left, right in zip(bottoms, values)]
    decision_axis.set_xticks(
        positions,
        tuple(_PLOT_LABELS[item.procedure_name] for item in combined),
    )
    decision_axis.set_ylabel("Fraction of 30 rehearsal rows")
    decision_axis.set_title("Saved decisions after calibration")
    decision_axis.grid(axis="y", alpha=0.25)
    decision_axis.legend(fontsize=8, ncol=2)

    coverage_values = tuple(
        0.0
        if item.simultaneous_block_coverage.rate is None
        else item.simultaneous_block_coverage.rate
        for item in combined
    )
    coverage_axis.bar(positions, coverage_values, color="#6a4c93")
    coverage_axis.axhline(
        result.config.target_block_coverage,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"{result.config.target_block_coverage:.0%} calibration target",
    )
    coverage_axis.set_xticks(
        positions,
        tuple(_PLOT_LABELS[item.procedure_name] for item in combined),
    )
    coverage_axis.set_ylim(0.0, 1.08)
    coverage_axis.set_ylabel("Covered paired device blocks")
    coverage_axis.set_title(
        "Procedure-set block coverage\n(missing envelope is an unbounded set)"
    )
    coverage_axis.grid(axis="y", alpha=0.25)
    coverage_axis.legend(fontsize=8)

    selector_by_truth = tuple(
        _summary(
            result,
            REHEARSAL_SPLIT,
            truth,
            DECISION_DIRECTED_SELECTOR,
        )
        for truth in STAGE3_TRUTH_CONDITIONS
    )
    action_names = (STOP_NOW, FIXED_VOLTAGE)
    truth_positions = tuple(range(len(selector_by_truth)))
    bottoms = [0] * len(selector_by_truth)
    for action, color in zip(action_names, ("#8ecae6", "#023047")):
        values = tuple(
            dict(item.selected_policy_counts).get(action, 0)
            for item in selector_by_truth
        )
        action_axis.bar(
            truth_positions,
            values,
            bottom=bottoms,
            color=color,
            label=_procedure_label(action),
        )
        bottoms = [left + right for left, right in zip(bottoms, values)]
    action_axis.set_xticks(
        truth_positions,
        tuple(_truth_label(truth) for truth in STAGE3_TRUTH_CONDITIONS),
        rotation=10,
    )
    action_axis.set_ylabel("Selected devices")
    action_axis.set_title("Acquisition-only selector actions")
    action_axis.legend(fontsize=8)
    action_axis.grid(axis="y", alpha=0.25)

    for summary in combined:
        error_rate = (
            summary.false_approvals.numerator + summary.false_rejections.numerator
        ) / summary.trial_count
        effort_axis.scatter(
            summary.mean_diagnostic_run_count,
            summary.decision_coverage.rate or 0.0,
            s=90 + 700 * error_rate,
            label=_procedure_label(summary.procedure_name),
        )
    effort_axis.set_xlabel("Mean diagnostic runs")
    effort_axis.set_ylabel("Decision coverage")
    effort_axis.set_title("Coverage versus diagnostic effort\n(marker size increases with errors)")
    effort_axis.grid(alpha=0.25)
    effort_axis.legend(fontsize=8)

    figure.suptitle(
        "Stage 4: calibrated decision selection under realistic sensor and model mismatch",
        fontsize=14,
    )
    figure.savefig(destination, dpi=170)
    save_figure_data(result, destination)
    return destination


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Stage 4 operating-decision calibration and selector rehearsal."
    )
    defaults = OperatingDecisionCalibrationConfig()
    parser.add_argument(
        "--gate-blocks",
        type=int,
        default=defaults.gate_development.sensor.trial_count,
    )
    parser.add_argument(
        "--gate-seed",
        type=int,
        default=defaults.gate_development.sensor.first_seed,
    )
    parser.add_argument("--calibration-blocks", type=int, default=defaults.calibration.sensor.trial_count)
    parser.add_argument("--rehearsal-blocks", type=int, default=defaults.rehearsal.sensor.trial_count)
    parser.add_argument("--calibration-seed", type=int, default=defaults.calibration.sensor.first_seed)
    parser.add_argument("--rehearsal-seed", type=int, default=defaults.rehearsal.sensor.first_seed)
    parser.add_argument("--fit-iterations", type=int, default=defaults.calibration.sensor.fit_iterations)
    parser.add_argument(
        "--target-coverage",
        type=float,
        default=defaults.target_block_coverage,
    )
    parser.add_argument(
        "--gate-target-retention",
        type=float,
        default=defaults.gate_target_block_retention,
    )
    parser.add_argument("--gate-draws", type=int, default=defaults.gate_monte_carlo_draws)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OPERATING_DECISION_CALIBRATION_PATH)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--artifact-output", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="print and optionally save text/JSON without importing Matplotlib",
    )
    arguments = parser.parse_args(argv)
    gate_development = replace(
        defaults.gate_development,
        sensor=replace(
            defaults.gate_development.sensor,
            trial_count=arguments.gate_blocks,
            first_seed=arguments.gate_seed,
            fit_iterations=arguments.fit_iterations,
        ),
    )
    calibration = replace(
        defaults.calibration,
        sensor=replace(
            defaults.calibration.sensor,
            trial_count=arguments.calibration_blocks,
            first_seed=arguments.calibration_seed,
            fit_iterations=arguments.fit_iterations,
        ),
    )
    rehearsal = replace(
        defaults.rehearsal,
        sensor=replace(
            defaults.rehearsal.sensor,
            trial_count=arguments.rehearsal_blocks,
            first_seed=arguments.rehearsal_seed,
            fit_iterations=arguments.fit_iterations,
        ),
    )
    config = replace(
        defaults,
        gate_development=gate_development,
        calibration=calibration,
        rehearsal=rehearsal,
        gate_monte_carlo_draws=arguments.gate_draws,
        target_block_coverage=arguments.target_coverage,
        gate_target_block_retention=arguments.gate_target_retention,
    )
    result = run_operating_decision_calibration(
        config,
        workers=arguments.workers,
        progress=lambda message: print(message, flush=True),
    )
    report = format_operating_decision_calibration_report(result)
    print(report)
    if arguments.report_output is not None:
        destination = arguments.report_output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(report + "\n", encoding="utf-8")
    if arguments.artifact_output is not None:
        save_calibration_artifact(
            result.artifact,
            arguments.artifact_output,
            source_revision=arguments.source_revision,
        )
    if not arguments.no_figure:
        save_operating_decision_calibration_figure(result, arguments.output)
    return 0


__all__ = [
    "DEFAULT_OPERATING_DECISION_CALIBRATION_PATH",
    "format_operating_decision_calibration_report",
    "main",
    "save_calibration_artifact",
    "save_operating_decision_calibration_figure",
]


if __name__ == "__main__":
    raise SystemExit(main())
