"""Report the mismatch-guarded selector freeze and reserved Stage 5 result."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Optional, Sequence

from ..studies.operating_decision import (
    FIXED_FACE_TEMPERATURE,
    FIXED_THERMAL,
    FIXED_VOLTAGE,
    STOP_NOW,
    RateEstimate,
)
from ..studies.operating_decision_calibration import (
    ALL_FAMILIES,
    DECISION_DIRECTED_SELECTOR,
    ProcedureSummary,
)
from ..studies.operating_decision_final_evaluation import (
    EARLY_MISMATCH_ABSTENTION,
    FINAL_EVALUATION_SPLIT,
    FINAL_PROCEDURES,
    MISMATCH_GUARDED_SELECTOR,
    RECALIBRATION_SPLIT,
    FinalEvaluationResult,
    OperatingDecisionFinalConfig,
    RevisedCalibrationResult,
    evaluate_reserved_stage5,
    freeze_revised_selector,
    load_stage4_calibration_artifact,
    revised_artifact_payload,
    save_revised_artifact,
)
from ..studies.operating_decision_realism import STAGE3_TRUTH_CONDITIONS


DEFAULT_PARENT_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "OPERATING_DECISION_CALIBRATION_ARTIFACT.json"
)
DEFAULT_REVISED_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "OPERATING_DECISION_FINAL_ARTIFACT.json"
)

_PROCEDURE_LABELS = {
    STOP_NOW: "Stop now",
    FIXED_THERMAL: "More thermal tests",
    FIXED_VOLTAGE: "Add voltage",
    FIXED_FACE_TEMPERATURE: "Add face temperature",
    DECISION_DIRECTED_SELECTOR: "Stage 4 selector",
    MISMATCH_GUARDED_SELECTOR: "Mismatch-guarded selector",
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
        return f"{rate.rate:.1%} ({rate.numerator}/{rate.denominator}; descriptive)"
    return (
        f"{rate.rate:.1%} ({rate.numerator}/{rate.denominator}; "
        f"95% Wilson {rate.lower_95:.1%}-{rate.upper_95:.1%})"
    )


def _summary(
    summaries: Sequence[ProcedureSummary],
    split: str,
    truth: str,
    procedure: str,
) -> ProcedureSummary:
    matches = tuple(
        item
        for item in summaries
        if item.split == split
        and item.truth_condition == truth
        and item.procedure_name == procedure
    )
    if len(matches) != 1:
        raise ValueError("report has no unique summary cell")
    return matches[0]


def format_revised_calibration_report(result: RevisedCalibrationResult) -> str:
    artifact = result.artifact
    calibration = artifact.procedure_calibration
    raw_rate = (
        calibration.raw_interval_covered_count / calibration.emitted_interval_count
        if calibration.emitted_interval_count
        else None
    )
    calibrated_rate = (
        calibration.calibrated_interval_covered_count
        / calibration.emitted_interval_count
        if calibration.emitted_interval_count
        else None
    )
    lines = [
        "Mismatch-guarded selector freeze",
        "=================================",
        "",
        "Chronology:",
        f"  guard development: {artifact.guard_development_block_count} blocks/family "
        f"from seed {artifact.guard_development_first_seed}",
        "  threshold used only matched four-state and extra-interface-mass rows",
        "  Family C was generated only after the threshold froze",
        f"  procedure recalibration: {artifact.recalibration_block_count} blocks/family "
        f"from seed {artifact.recalibration_first_seed}",
        f"  reserved Stage 5: {artifact.evaluation_block_count} blocks/family from "
        f"seed {artifact.evaluation_first_seed}; not instantiated by this command",
        f"  implementation revision: {artifact.implementation_revision}",
        f"  protocol digest: {artifact.protocol_digest}",
        f"  parent Stage 4 digest: {artifact.parent_stage4_protocol_digest}",
        "",
        "Acquisition mismatch guard:",
        "  score each constant-contact candidate on the common fitted acquisition",
        "  remove parameter-prior residuals, retain profiled run-bias penalties,",
        "  and divide by the number of observed temperature samples",
        "  guard score G is the smaller candidate score",
        f"  alarm when G > {artifact.guard.threshold:.6f}; matched-family rank "
        f"{artifact.guard.conformal_rank}/{artifact.guard.block_count}",
        "  an alarm immediately returns insufficient evidence after one acquisition",
        "  unflagged cases use the unchanged Stage 4 stop-or-voltage rule",
        "",
        "Guard-development diagnostic after threshold freeze:",
    ]
    for item in result.guard_development:
        lines.append(
            f"  {_truth_label(item.truth_condition)}: alarms "
            f"{item.triggered_count}/{item.row_count}; alarms that replaced a Stage 4 "
            f"voltage action {item.selected_voltage_count}; those ending insufficient "
            f"after voltage {item.triggered_then_parent_insufficient_count}"
        )
    lines.extend(
        (
            "",
            "Revised procedure calibration:",
            "  ranked padding="
            f"{sorted(calibration.block_nonconformity)[calibration.conformal_rank - 1]:.6f} K",
            f"  Stage 4 padding floor={artifact.parent_selector_padding_floor:.6f} K",
            f"  frozen padding={calibration.additive_margin_padding:.6f} K",
            f"  block rank={calibration.conformal_rank}/{calibration.block_count}",
            f"  emitted finite intervals={calibration.emitted_interval_count}",
            f"  raw finite-interval coverage="
            f"{'N/A' if raw_rate is None else f'{raw_rate:.1%}'}",
            f"  padded finite-interval coverage="
            f"{'N/A' if calibrated_rate is None else f'{calibrated_rate:.1%}'}",
            "",
            "Recalibration-cohort comparison (descriptive):",
        )
    )
    for procedure in FINAL_PROCEDURES:
        summary = _summary(
            result.summaries,
            RECALIBRATION_SPLIT,
            ALL_FAMILIES,
            procedure,
        )
        losses = ", ".join(
            f"{item.scenario_name}={item.mean_loss:.3f}"
            for item in summary.expected_losses
        )
        lines.extend(
            (
                f"  {_procedure_label(procedure)}: approve/reject/insufficient="
                f"{summary.approvals}/{summary.rejections}/"
                f"{summary.insufficient_evidence}; decision coverage "
                f"{_format_rate(summary.decision_coverage)}",
                f"    block procedure-set coverage "
                f"{_format_rate(summary.simultaneous_block_coverage)}; mean "
                f"runs/energy/sensors={summary.mean_diagnostic_run_count:.2f}/"
                f"{summary.mean_total_diagnostic_energy:.2f} J/"
                f"{summary.mean_extra_sensor_count:.2f}; empirical mean loss {losses}",
            )
        )
    lines.extend(
        (
            "",
            "Boundary:",
            "  this command freezes development and calibration choices only",
            "  no Stage 5 observation or final response was generated",
            "  a missing envelope is an unbounded procedure set but remains an abstention",
            "  the guard is a diagnostic for this declared generator, not generic OOD detection",
        )
    )
    return "\n".join(lines)


def format_final_evaluation_report(result: FinalEvaluationResult) -> str:
    artifact = result.artifact
    lines = [
        "Reserved Stage 5 operating-decision evaluation",
        "==============================================",
        "",
        f"Frozen artifact: {artifact.protocol_digest}",
        f"Implementation revision: {artifact.implementation_revision}",
        f"Evaluation: {artifact.evaluation_block_count} paired blocks/family; "
        f"first seed {artifact.evaluation_first_seed}",
        "No threshold, selector branch, padding, or loss weight was updated after reveal.",
    ]
    for truth in STAGE3_TRUTH_CONDITIONS:
        lines.extend(("", f"{_truth_label(truth)}:"))
        for procedure in FINAL_PROCEDURES:
            summary = _summary(
                result.summaries,
                FINAL_EVALUATION_SPLIT,
                truth,
                procedure,
            )
            lines.extend(
                (
                    f"  {_procedure_label(procedure)}: approve/reject/insufficient="
                    f"{summary.approvals}/{summary.rejections}/"
                    f"{summary.insufficient_evidence}; true pass/violate="
                    f"{summary.true_passing}/{summary.true_violating}",
                    f"    false approvals={_format_rate(summary.false_approvals)}; "
                    f"false rejections={_format_rate(summary.false_rejections)}",
                    f"    decision coverage={_format_rate(summary.decision_coverage)}; "
                    f"finite-interval coverage="
                    f"{_format_rate(summary.calibrated_interval_coverage)}",
                    f"    runs/energy/sensors={summary.mean_diagnostic_run_count:.2f}/"
                    f"{summary.mean_total_diagnostic_energy:.2f} J/"
                    f"{summary.mean_extra_sensor_count:.2f}",
                )
            )
    lines.extend(("", "Cross-family paired-block comparison:"))
    for procedure in FINAL_PROCEDURES:
        summary = _summary(
            result.summaries,
            FINAL_EVALUATION_SPLIT,
            ALL_FAMILIES,
            procedure,
        )
        losses = ", ".join(
            f"{item.scenario_name}={item.mean_loss:.3f}"
            for item in summary.expected_losses
        )
        actions = ", ".join(
            f"{name}={count}" for name, count in summary.selected_policy_counts
        )
        lines.extend(
            (
                f"  {_procedure_label(procedure)}: approve/reject/insufficient="
                f"{summary.approvals}/{summary.rejections}/"
                f"{summary.insufficient_evidence}; decision coverage "
                f"{_format_rate(summary.decision_coverage)}",
                f"    block procedure-set coverage "
                f"{_format_rate(summary.simultaneous_block_coverage)}; mean "
                f"runs/energy/sensors={summary.mean_diagnostic_run_count:.2f}/"
                f"{summary.mean_total_diagnostic_energy:.2f} J/"
                f"{summary.mean_extra_sensor_count:.2f}",
                f"    empirical mean loss {losses}; actions {actions}",
            )
        )
    primary = tuple(
        item for item in result.paired_loss_comparisons if item.scenario_name == "balanced"
    )
    if len(primary) != 1:
        raise ValueError("the final report needs one balanced primary loss contrast")
    lines.extend(("", "Primary paired balanced-loss contrast (revised minus Stage 4 selector):"))
    for item in primary:
        lines.append(
            f"  {item.scenario_name}: {item.revised_minus_parent_mean:+.4f} "
            f"(paired-block bootstrap 95% interval {item.lower_95:+.4f} to "
            f"{item.upper_95:+.4f}; {item.block_count} blocks)"
        )
    sensitivity = tuple(
        item for item in result.paired_loss_comparisons if item.scenario_name != "balanced"
    )
    if sensitivity:
        lines.extend(("", "Paired loss sensitivity analyses:"))
        for item in sensitivity:
            lines.append(
                f"  {item.scenario_name}: {item.revised_minus_parent_mean:+.4f} "
                f"(paired-block bootstrap 95% interval {item.lower_95:+.4f} to "
                f"{item.upper_95:+.4f}; {item.block_count} blocks)"
            )
    lines.extend(("", "Mismatch-guard diagnostics:"))
    for item in result.guard_summaries:
        lines.append(
            f"  {_truth_label(item.truth_condition)}: alarms "
            f"{item.triggered_count}/{item.row_count}; alarms replacing voltage "
            f"{item.selected_voltage_count}; voltage-then-insufficient among those "
            f"{item.triggered_then_parent_insufficient_count}"
        )
    lines.extend(
        (
            "",
            "Interpretation boundary:",
            "  the protocol designates this predeclared cohort for one final run",
            "  the command refuses to overwrite an existing artifact, result, or report",
            "  family rows share a device block, so pooled row-level CIs are suppressed",
            "  finite-interval coverage is conditional and is not the conformal target",
            "  early-abstention computation time is a conservative full-trial upper bound",
            "  the result does not establish hardware safety or generic mismatch detection",
        )
    )
    return "\n".join(lines)


def _rate_payload(rate: RateEstimate) -> dict:
    return rate._asdict()


def _summary_payload(summary: ProcedureSummary) -> dict:
    payload = summary._asdict()
    for field in (
        "false_approvals",
        "missed_violations",
        "false_rejections",
        "decision_coverage",
        "abstentions",
        "raw_interval_coverage",
        "calibrated_interval_coverage",
        "simultaneous_block_coverage",
    ):
        payload[field] = _rate_payload(getattr(summary, field))
    payload["selected_policy_counts"] = [
        {"policy": name, "count": count}
        for name, count in summary.selected_policy_counts
    ]
    payload["expected_losses"] = [item._asdict() for item in summary.expected_losses]
    return payload


def final_result_payload(result: FinalEvaluationResult) -> dict:
    return {
        "schema_version": 1,
        "artifact": revised_artifact_payload(result.artifact),
        "evaluation": {
            "split": FINAL_EVALUATION_SPLIT,
            "first_seed": result.artifact.evaluation_first_seed,
            "paired_blocks_per_family": result.artifact.evaluation_block_count,
            "instantiated": True,
            "protocol_designation": "single_final_run",
        },
        "outcomes": [item._asdict() for item in result.outcomes],
        "summaries": [_summary_payload(item) for item in result.summaries],
        "paired_loss_comparisons": [
            item._asdict() for item in result.paired_loss_comparisons
        ],
        "guard_summaries": [item._asdict() for item in result.guard_summaries],
        "measurement_notes": {
            "early_abstention_computation_seconds": (
                "conservative upper bound copied from the full stop-trial timing; "
                "excluded from the resource comparison"
            ),
        },
    }


def save_final_result(result: FinalEvaluationResult, path: Path | str) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(final_result_payload(result), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    return destination


def _save_report(text: str, path: Optional[Path]) -> None:
    if path is None:
        return
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(text + "\n")


def _preflight_output_paths(
    *,
    input_paths: Sequence[Path],
    output_paths: Sequence[Optional[Path]],
) -> None:
    """Refuse aliases or overwrites before any experimental data are generated."""

    inputs = {path.expanduser().resolve() for path in input_paths}
    outputs = tuple(
        path.expanduser().resolve() for path in output_paths if path is not None
    )
    if len(set(outputs)) != len(outputs):
        raise ValueError("experimental output paths must be distinct")
    if inputs.intersection(outputs):
        raise ValueError("an experimental output path aliases an input artifact")
    existing = tuple(path for path in outputs if path.exists())
    if existing:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite experimental output: {rendered}")


def _configured_freeze(arguments) -> OperatingDecisionFinalConfig:
    defaults = OperatingDecisionFinalConfig()
    return replace(
        defaults,
        guard_development=replace(
            defaults.guard_development,
            sensor=replace(
                defaults.guard_development.sensor,
                trial_count=arguments.guard_blocks,
                first_seed=arguments.guard_seed,
                fit_iterations=arguments.fit_iterations,
            ),
        ),
        recalibration=replace(
            defaults.recalibration,
            sensor=replace(
                defaults.recalibration.sensor,
                trial_count=arguments.calibration_blocks,
                first_seed=arguments.calibration_seed,
                fit_iterations=arguments.fit_iterations,
            ),
        ),
        evaluation=replace(
            defaults.evaluation,
            sensor=replace(
                defaults.evaluation.sensor,
                fit_iterations=arguments.fit_iterations,
            ),
        ),
        guard_target_block_retention=arguments.guard_target,
        target_block_coverage=arguments.coverage_target,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the mismatch guard or run the reserved Stage 5 evaluation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    defaults = OperatingDecisionFinalConfig()

    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--parent-artifact", type=Path, default=DEFAULT_PARENT_ARTIFACT)
    freeze_parser.add_argument("--artifact-output", type=Path, default=DEFAULT_REVISED_ARTIFACT)
    freeze_parser.add_argument("--report-output", type=Path)
    freeze_parser.add_argument("--source-revision", required=True)
    freeze_parser.add_argument("--workers", type=int, default=1)
    freeze_parser.add_argument(
        "--guard-blocks",
        type=int,
        default=defaults.guard_development.sensor.trial_count,
    )
    freeze_parser.add_argument(
        "--guard-seed",
        type=int,
        default=defaults.guard_development.sensor.first_seed,
    )
    freeze_parser.add_argument(
        "--calibration-blocks",
        type=int,
        default=defaults.recalibration.sensor.trial_count,
    )
    freeze_parser.add_argument(
        "--calibration-seed",
        type=int,
        default=defaults.recalibration.sensor.first_seed,
    )
    freeze_parser.add_argument(
        "--fit-iterations",
        type=int,
        default=defaults.recalibration.sensor.fit_iterations,
    )
    freeze_parser.add_argument(
        "--guard-target",
        type=float,
        default=defaults.guard_target_block_retention,
    )
    freeze_parser.add_argument(
        "--coverage-target",
        type=float,
        default=defaults.target_block_coverage,
    )

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--parent-artifact", type=Path, default=DEFAULT_PARENT_ARTIFACT)
    evaluate_parser.add_argument("--artifact", type=Path, default=DEFAULT_REVISED_ARTIFACT)
    evaluate_parser.add_argument("--result-output", type=Path, required=True)
    evaluate_parser.add_argument("--report-output", type=Path)
    evaluate_parser.add_argument("--workers", type=int, default=1)

    arguments = parser.parse_args(argv)
    if arguments.command == "freeze":
        _preflight_output_paths(
            input_paths=(arguments.parent_artifact,),
            output_paths=(arguments.artifact_output, arguments.report_output),
        )
    else:
        _preflight_output_paths(
            input_paths=(arguments.parent_artifact, arguments.artifact),
            output_paths=(arguments.result_output, arguments.report_output),
        )
    parent = load_stage4_calibration_artifact(arguments.parent_artifact)
    if arguments.command == "freeze":
        config = _configured_freeze(arguments)
        result = freeze_revised_selector(
            parent,
            source_revision=arguments.source_revision,
            config=config,
            workers=arguments.workers,
            progress=lambda message: print(message, flush=True),
        )
        save_revised_artifact(result.artifact, arguments.artifact_output)
        report = format_revised_calibration_report(result)
    else:
        result = evaluate_reserved_stage5(
            arguments.artifact,
            parent,
            workers=arguments.workers,
            progress=lambda message: print(message, flush=True),
        )
        save_final_result(result, arguments.result_output)
        report = format_final_evaluation_report(result)
    print(report)
    _save_report(report, arguments.report_output)
    return 0


__all__ = [
    "DEFAULT_PARENT_ARTIFACT",
    "DEFAULT_REVISED_ARTIFACT",
    "final_result_payload",
    "format_final_evaluation_report",
    "format_revised_calibration_report",
    "main",
    "save_final_result",
]


if __name__ == "__main__":
    raise SystemExit(main())
