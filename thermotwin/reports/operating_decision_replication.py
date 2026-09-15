"""CLI for the content-addressed corrected operating-decision replication."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from ..studies.operating_decision_calibration import (
    ALL_FAMILIES,
    CALIBRATION_SPLIT,
    DECISION_DIRECTED_SELECTOR,
    REHEARSAL_SPLIT,
    STAGE4_PROCEDURES,
    ProcedureSummary,
    summarize_calibrated_outcomes,
)
from ..studies.operating_decision_diagnostics import (
    corrected_partition_evidence_digest,
    save_corrected_partition_diagnostics,
    save_diagnostic_payload,
)
from ..studies.operating_decision_replication_calibration import (
    CorrectedParentFreezeResult,
    CorrectedParentRehearsalResult,
    freeze_corrected_parent_calibration,
    run_corrected_parent_rehearsal,
    save_corrected_parent_artifact,
)
from ..studies.operating_decision_replication_guard import (
    FINAL_EVALUATION_SPLIT,
    GUARD_CALIBRATION_SPLIT,
    MISMATCH_GUARDED_SELECTOR,
    CorrectedFinalEvaluationResult,
    CorrectedGuardFreezeResult,
    evaluate_corrected_reserved_cohort,
    freeze_corrected_guard_calibration,
    save_corrected_guard_artifact,
)
from ..studies.operating_decision_replication_protocol import (
    build_corrected_generator_freeze,
    save_corrected_generator_freeze,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_GENERATOR_FREEZE = PACKAGE_ROOT / "OPERATING_DECISION_CORRECTED_GENERATOR.json"
DEFAULT_PARENT_ARTIFACT = PACKAGE_ROOT / "OPERATING_DECISION_CORRECTED_PARENT_ARTIFACT.json"
DEFAULT_GATE_DIAGNOSTICS = PACKAGE_ROOT / "OPERATING_DECISION_CORRECTED_GATE_DIAGNOSTICS.json"
DEFAULT_CALIBRATION_DIAGNOSTICS = PACKAGE_ROOT / "OPERATING_DECISION_CORRECTED_CALIBRATION_DIAGNOSTICS.json"
DEFAULT_CALIBRATION_OUTCOMES = PACKAGE_ROOT / "OPERATING_DECISION_CORRECTED_CALIBRATION_OUTCOMES.json"
DEFAULT_REHEARSAL_DIAGNOSTICS = PACKAGE_ROOT / "OPERATING_DECISION_CORRECTED_REHEARSAL_DIAGNOSTICS.json"
DEFAULT_REHEARSAL_OUTCOMES = PACKAGE_ROOT / "OPERATING_DECISION_CORRECTED_REHEARSAL_OUTCOMES.json"
DEFAULT_GUARD_ARTIFACT = PACKAGE_ROOT / "OPERATING_DECISION_CORRECTED_GUARD_ARTIFACT.json"
DEFAULT_GUARD_DEVELOPMENT_DIAGNOSTICS = PACKAGE_ROOT / "OPERATING_DECISION_CORRECTED_GUARD_DEVELOPMENT_DIAGNOSTICS.json"
DEFAULT_GUARD_CALIBRATION_DIAGNOSTICS = PACKAGE_ROOT / "OPERATING_DECISION_CORRECTED_GUARD_CALIBRATION_DIAGNOSTICS.json"
DEFAULT_GUARD_CALIBRATION_OUTCOMES = PACKAGE_ROOT / "OPERATING_DECISION_CORRECTED_GUARD_CALIBRATION_OUTCOMES.json"
DEFAULT_FINAL_DIAGNOSTICS = PACKAGE_ROOT / "OPERATING_DECISION_CORRECTED_FINAL_DIAGNOSTICS.json"
DEFAULT_FINAL_OUTCOMES = PACKAGE_ROOT / "OPERATING_DECISION_CORRECTED_FINAL_OUTCOMES.json"


def _preflight(inputs: Sequence[Path], outputs: Sequence[Optional[Path]]) -> None:
    for path in inputs:
        if not path.expanduser().resolve().is_file():
            raise FileNotFoundError(f"required input does not exist: {path}")
    resolved = [path.expanduser().resolve() for path in outputs if path is not None]
    if len(set(resolved)) != len(resolved):
        raise ValueError("corrected replication output paths must be distinct")
    existing = tuple(path for path in resolved if path.exists())
    if existing:
        raise FileExistsError(
            "corrected replication refuses to overwrite: "
            + ", ".join(str(path) for path in existing)
        )


def _rate_text(rate) -> str:
    if rate.rate is None:
        return f"N/A ({rate.numerator}/{rate.denominator})"
    interval = (
        "descriptive"
        if rate.lower_95 is None or rate.upper_95 is None
        else f"95% Wilson {rate.lower_95:.1%}-{rate.upper_95:.1%}"
    )
    return f"{rate.rate:.1%} ({rate.numerator}/{rate.denominator}; {interval})"


def _summary(
    summaries: Sequence[ProcedureSummary],
    split: str,
    procedure: str,
) -> ProcedureSummary:
    matches = tuple(
        item
        for item in summaries
        if item.split == split
        and item.truth_condition == ALL_FAMILIES
        and item.procedure_name == procedure
    )
    if len(matches) != 1:
        raise ValueError("corrected report has no unique cross-family summary")
    return matches[0]


def format_parent_freeze_report(result: CorrectedParentFreezeResult) -> str:
    summaries = summarize_calibrated_outcomes(
        result.calibration_outcomes,
        result.config,  # type: ignore[arg-type]
    )
    lines = [
        "Corrected operating-decision parent freeze",
        "===========================================",
        "",
        f"Generator freeze: {result.generator_freeze.artifact_digest}",
        f"Source manifest: {result.generator_freeze.source_manifest.digest}",
        f"Parent protocol: {result.artifact.protocol_digest}",
        "Random-stream audits: gate and calibration clean",
        "Realized per-device terminal energy used for outcome scoring",
        "Rehearsal and reserved evaluation remain unopened",
        "",
        "Verification gates:",
    ]
    for gate in result.artifact.verification_gates:
        lines.append(
            f"  {gate.policy_name}: threshold {gate.threshold:.6f}; matched rank "
            f"{gate.matched_development_rank}/{gate.matched_development_block_count}"
        )
    lines.extend(("", "Procedure paddings and calibration outcomes:"))
    for procedure in STAGE4_PROCEDURES:
        calibration = result.artifact.procedure_for(procedure)
        summary = _summary(summaries, CALIBRATION_SPLIT, procedure)
        lines.append(
            f"  {procedure}: padding {calibration.additive_margin_padding:.6f} K; "
            f"decision coverage {_rate_text(summary.decision_coverage)}; block coverage "
            f"{_rate_text(summary.simultaneous_block_coverage)}; mean realized energy "
            f"{summary.mean_total_diagnostic_energy:.2f} J"
        )
    return "\n".join(lines)


def format_parent_rehearsal_report(result: CorrectedParentRehearsalResult) -> str:
    lines = [
        "Corrected operating-decision parent rehearsal",
        "==============================================",
        "",
        f"Loaded parent artifact: {result.artifact.protocol_digest}",
        f"Verified source manifest: {result.generator_freeze.source_manifest.digest}",
        "Reserved evaluation remains unopened",
        "",
        "Fresh rehearsal outcomes:",
    ]
    for procedure in STAGE4_PROCEDURES:
        summary = _summary(result.summaries, REHEARSAL_SPLIT, procedure)
        losses = ", ".join(
            f"{item.scenario_name}={item.mean_loss:.3f}"
            for item in summary.expected_losses
        )
        lines.append(
            f"  {procedure}: approve/reject/insufficient "
            f"{summary.approvals}/{summary.rejections}/{summary.insufficient_evidence}; "
            f"decision coverage {_rate_text(summary.decision_coverage)}; block coverage "
            f"{_rate_text(summary.simultaneous_block_coverage)}; mean runs/realized "
            f"energy/sensors {summary.mean_diagnostic_run_count:.2f}/"
            f"{summary.mean_total_diagnostic_energy:.2f} J/"
            f"{summary.mean_extra_sensor_count:.2f}; empirical loss {losses}"
        )
    return "\n".join(lines)


def format_guard_freeze_report(result: CorrectedGuardFreezeResult) -> str:
    lines = [
        "Corrected operating-decision guard freeze",
        "==========================================",
        "",
        f"Generator freeze: {result.generator_freeze.artifact_digest}",
        f"Parent protocol: {result.parent.protocol_digest}",
        f"Guard protocol: {result.artifact.protocol_digest}",
        f"Guard threshold: {result.artifact.guard.threshold:.6f} "
        f"(rank {result.artifact.guard.conformal_rank}/"
        f"{result.artifact.guard.block_count})",
        "Guard-development and guard-calibration evidence are content-bound",
        "Reserved evaluation remains unopened",
        "",
        "Guard triggers during development:",
    ]
    for item in result.guard_summaries:
        lines.append(
            f"  {item.truth_condition}: {item.triggered_count}/{item.row_count}; "
            f"from parent voltage {item.triggered_from_parent_voltage_count}; "
            f"those ending in parent abstention "
            f"{item.triggered_then_parent_insufficient_count}"
        )
    parent = _summary(
        result.summaries,
        GUARD_CALIBRATION_SPLIT,
        DECISION_DIRECTED_SELECTOR,
    )
    revised = _summary(
        result.summaries,
        GUARD_CALIBRATION_SPLIT,
        MISMATCH_GUARDED_SELECTOR,
    )
    lines.extend(
        (
            "",
            "Guard-calibration comparison (descriptive):",
            f"  Parent selector: decision coverage {_rate_text(parent.decision_coverage)}; "
            f"mean runs/energy {parent.mean_diagnostic_run_count:.2f}/"
            f"{parent.mean_total_diagnostic_energy:.2f} J",
            f"  Guarded selector: decision coverage {_rate_text(revised.decision_coverage)}; "
            f"mean runs/energy {revised.mean_diagnostic_run_count:.2f}/"
            f"{revised.mean_total_diagnostic_energy:.2f} J; padding "
            f"{result.artifact.procedure_calibration.additive_margin_padding:.6f} K",
        )
    )
    return "\n".join(lines)


def format_final_evaluation_report(result: CorrectedFinalEvaluationResult) -> str:
    primary = next(
        item
        for item in result.paired_loss_comparisons
        if item.scenario_name == result.artifact.primary_scenario_name
    )
    lines = [
        "Corrected operating-decision reserved evaluation",
        "=================================================",
        "",
        f"Generator freeze: {result.generator_freeze.artifact_digest}",
        f"Committed source/artifact state: {result.source_commit}",
        f"Parent protocol: {result.parent.protocol_digest}",
        f"Guard protocol: {result.artifact.protocol_digest}",
        f"Reserved evidence: {corrected_partition_evidence_digest(result.evaluation)}",
        f"Primary result ({primary.scenario_name}): "
        f"revised minus parent {primary.revised_minus_parent_mean:+.6f}; "
        f"{primary.confidence_level:.0%} paired-block bootstrap "
        f"[{primary.lower_bound:+.6f}, {primary.upper_bound:+.6f}]",
        "Primary success criterion (upper bound below zero): "
        + ("met" if result.primary_success else "not met"),
        "",
        "All predeclared loss scenarios:",
    ]
    for item in result.paired_loss_comparisons:
        lines.append(
            f"  {item.scenario_name}: mean {item.revised_minus_parent_mean:+.6f}; "
            f"interval [{item.lower_bound:+.6f}, {item.upper_bound:+.6f}]; "
            f"{item.independent_block_count} independent blocks"
        )
    parent = _summary(
        result.summaries,
        FINAL_EVALUATION_SPLIT,
        DECISION_DIRECTED_SELECTOR,
    )
    revised = _summary(
        result.summaries,
        FINAL_EVALUATION_SPLIT,
        MISMATCH_GUARDED_SELECTOR,
    )
    lines.extend(
        (
            "",
            "Reserved-cohort operating outcomes:",
            f"  Parent selector: approve/reject/insufficient "
            f"{parent.approvals}/{parent.rejections}/{parent.insufficient_evidence}; "
            f"mean runs/energy {parent.mean_diagnostic_run_count:.2f}/"
            f"{parent.mean_total_diagnostic_energy:.2f} J",
            f"  Guarded selector: approve/reject/insufficient "
            f"{revised.approvals}/{revised.rejections}/"
            f"{revised.insufficient_evidence}; mean runs/energy "
            f"{revised.mean_diagnostic_run_count:.2f}/"
            f"{revised.mean_total_diagnostic_energy:.2f} J",
        )
    )
    return "\n".join(lines)


def _stream_key_payload(key) -> dict:
    return {
        "protocol_version": key.protocol_version,
        "campaign": key.campaign,
        "partition": key.partition,
        "block": key.block,
        "stream_kind": key.stream_kind,
        "purpose": key.purpose,
        "family": key.family,
        "run": key.run,
        "channel": key.channel,
    }


def _outcome_payload(
    split: str,
    outcomes,
    summaries,
    *,
    provenance: dict,
    guard_summaries=(),
    paired_loss_comparisons=(),
    primary_success=None,
) -> dict:
    payload = {
        "schema_version": 1,
        "split": split,
        "provenance": provenance,
        "outcomes": [item._asdict() for item in outcomes],
        "summaries": [
            {
                **item._asdict(),
                "false_approvals": item.false_approvals._asdict(),
                "missed_violations": item.missed_violations._asdict(),
                "false_rejections": item.false_rejections._asdict(),
                "decision_coverage": item.decision_coverage._asdict(),
                "abstentions": item.abstentions._asdict(),
                "raw_interval_coverage": item.raw_interval_coverage._asdict(),
                "calibrated_interval_coverage": (
                    item.calibrated_interval_coverage._asdict()
                ),
                "simultaneous_block_coverage": item.simultaneous_block_coverage._asdict(),
                "selected_policy_counts": [
                    {"policy": name, "count": count}
                    for name, count in item.selected_policy_counts
                ],
                "expected_losses": [loss._asdict() for loss in item.expected_losses],
            }
            for item in summaries
        ],
        "guard_summaries": [item._asdict() for item in guard_summaries],
        "paired_loss_comparisons": [
            {
                **item._asdict(),
                "random_stream_key": _stream_key_payload(item.random_stream_key),
            }
            for item in paired_loss_comparisons
        ],
    }
    if primary_success is not None:
        payload["primary_success"] = bool(primary_success)
    return payload


def _save_text_once(text: str, path: Optional[Path]) -> None:
    if path is None:
        return
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(text + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the collision-free, content-addressed decision replication."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generator = commands.add_parser("freeze-generator")
    generator.add_argument("--repository-root", type=Path, default=PROJECT_ROOT)
    generator.add_argument("--output", type=Path, default=DEFAULT_GENERATOR_FREEZE)

    parent = commands.add_parser("freeze-parent")
    parent.add_argument("--repository-root", type=Path, default=PROJECT_ROOT)
    parent.add_argument("--generator-freeze", type=Path, default=DEFAULT_GENERATOR_FREEZE)
    parent.add_argument("--artifact-output", type=Path, default=DEFAULT_PARENT_ARTIFACT)
    parent.add_argument("--gate-diagnostics", type=Path, default=DEFAULT_GATE_DIAGNOSTICS)
    parent.add_argument(
        "--calibration-diagnostics",
        type=Path,
        default=DEFAULT_CALIBRATION_DIAGNOSTICS,
    )
    parent.add_argument(
        "--calibration-outcomes",
        type=Path,
        default=DEFAULT_CALIBRATION_OUTCOMES,
    )
    parent.add_argument("--report-output", type=Path)
    parent.add_argument("--workers", type=int, default=1)

    rehearsal = commands.add_parser("rehearse-parent")
    rehearsal.add_argument("--repository-root", type=Path, default=PROJECT_ROOT)
    rehearsal.add_argument("--artifact", type=Path, default=DEFAULT_PARENT_ARTIFACT)
    rehearsal.add_argument(
        "--diagnostics-output",
        type=Path,
        default=DEFAULT_REHEARSAL_DIAGNOSTICS,
    )
    rehearsal.add_argument(
        "--outcomes-output",
        type=Path,
        default=DEFAULT_REHEARSAL_OUTCOMES,
    )
    rehearsal.add_argument("--report-output", type=Path)
    rehearsal.add_argument("--workers", type=int, default=1)

    guard = commands.add_parser("freeze-guard")
    guard.add_argument("--repository-root", type=Path, default=PROJECT_ROOT)
    guard.add_argument("--parent-artifact", type=Path, default=DEFAULT_PARENT_ARTIFACT)
    guard.add_argument("--artifact-output", type=Path, default=DEFAULT_GUARD_ARTIFACT)
    guard.add_argument(
        "--development-diagnostics",
        type=Path,
        default=DEFAULT_GUARD_DEVELOPMENT_DIAGNOSTICS,
    )
    guard.add_argument(
        "--calibration-diagnostics",
        type=Path,
        default=DEFAULT_GUARD_CALIBRATION_DIAGNOSTICS,
    )
    guard.add_argument(
        "--calibration-outcomes",
        type=Path,
        default=DEFAULT_GUARD_CALIBRATION_OUTCOMES,
    )
    guard.add_argument("--report-output", type=Path)
    guard.add_argument("--workers", type=int, default=1)

    final = commands.add_parser("evaluate-reserved")
    final.add_argument("--repository-root", type=Path, default=PROJECT_ROOT)
    final.add_argument("--artifact", type=Path, default=DEFAULT_GUARD_ARTIFACT)
    final.add_argument(
        "--diagnostics-output",
        type=Path,
        default=DEFAULT_FINAL_DIAGNOSTICS,
    )
    final.add_argument(
        "--outcomes-output",
        type=Path,
        default=DEFAULT_FINAL_OUTCOMES,
    )
    final.add_argument("--report-output", type=Path)
    final.add_argument("--workers", type=int, default=1)

    arguments = parser.parse_args(argv)
    if arguments.command == "freeze-generator":
        _preflight((), (arguments.output,))
        artifact = build_corrected_generator_freeze(arguments.repository_root)
        save_corrected_generator_freeze(artifact, arguments.output)
        print(f"Corrected generator frozen: {artifact.artifact_digest}")
        return 0
    if arguments.command == "freeze-parent":
        _preflight(
            (arguments.generator_freeze,),
            (
                arguments.artifact_output,
                arguments.gate_diagnostics,
                arguments.calibration_diagnostics,
                arguments.calibration_outcomes,
                arguments.report_output,
            ),
        )
        result = freeze_corrected_parent_calibration(
            arguments.generator_freeze,
            arguments.repository_root,
            workers=arguments.workers,
            progress=lambda message: print(message, flush=True),
        )
        save_corrected_parent_artifact(
            result.artifact,
            result.generator_freeze,
            result.config,
            arguments.artifact_output,
        )
        save_corrected_partition_diagnostics(
            result.gate_development,
            arguments.gate_diagnostics,
        )
        save_corrected_partition_diagnostics(
            result.calibration,
            arguments.calibration_diagnostics,
        )
        summaries = summarize_calibrated_outcomes(
            result.calibration_outcomes,
            result.config,  # type: ignore[arg-type]
        )
        save_diagnostic_payload(
            _outcome_payload(
                CALIBRATION_SPLIT,
                result.calibration_outcomes,
                summaries,
                provenance={
                    "generator_freeze_digest": (
                        result.generator_freeze.artifact_digest
                    ),
                    "source_manifest_digest": (
                        result.generator_freeze.source_manifest.digest
                    ),
                    "parent_protocol_digest": result.artifact.protocol_digest,
                    "partition_evidence_digest": (
                        result.artifact.calibration_evidence_digest
                    ),
                },
            ),
            arguments.calibration_outcomes,
        )
        report = format_parent_freeze_report(result)
    elif arguments.command == "rehearse-parent":
        _preflight(
            (arguments.artifact,),
            (
                arguments.diagnostics_output,
                arguments.outcomes_output,
                arguments.report_output,
            ),
        )
        result = run_corrected_parent_rehearsal(
            arguments.artifact,
            arguments.repository_root,
            workers=arguments.workers,
            progress=lambda message: print(message, flush=True),
        )
        save_corrected_partition_diagnostics(
            result.rehearsal,
            arguments.diagnostics_output,
        )
        save_diagnostic_payload(
            _outcome_payload(
                REHEARSAL_SPLIT,
                result.rehearsal_outcomes,
                result.summaries,
                provenance={
                    "generator_freeze_digest": (
                        result.generator_freeze.artifact_digest
                    ),
                    "source_manifest_digest": (
                        result.generator_freeze.source_manifest.digest
                    ),
                    "parent_protocol_digest": result.artifact.protocol_digest,
                    "partition_evidence_digest": (
                        corrected_partition_evidence_digest(result.rehearsal)
                    ),
                },
            ),
            arguments.outcomes_output,
        )
        report = format_parent_rehearsal_report(result)
    elif arguments.command == "freeze-guard":
        _preflight(
            (arguments.parent_artifact,),
            (
                arguments.artifact_output,
                arguments.development_diagnostics,
                arguments.calibration_diagnostics,
                arguments.calibration_outcomes,
                arguments.report_output,
            ),
        )
        result = freeze_corrected_guard_calibration(
            arguments.parent_artifact,
            arguments.repository_root,
            workers=arguments.workers,
            progress=lambda message: print(message, flush=True),
        )
        save_corrected_guard_artifact(
            result.artifact,
            result.generator_freeze,
            result.parent_config,
            result.parent,
            result.config,
            arguments.artifact_output,
        )
        save_corrected_partition_diagnostics(
            result.guard_development,
            arguments.development_diagnostics,
        )
        save_corrected_partition_diagnostics(
            result.guard_calibration,
            arguments.calibration_diagnostics,
        )
        save_diagnostic_payload(
            _outcome_payload(
                GUARD_CALIBRATION_SPLIT,
                result.calibration_outcomes,
                result.summaries,
                provenance={
                    "generator_freeze_digest": (
                        result.generator_freeze.artifact_digest
                    ),
                    "source_manifest_digest": (
                        result.generator_freeze.source_manifest.digest
                    ),
                    "parent_protocol_digest": result.parent.protocol_digest,
                    "guard_protocol_digest": result.artifact.protocol_digest,
                    "guard_development_evidence_digest": (
                        result.artifact.guard_development_evidence_digest
                    ),
                    "partition_evidence_digest": (
                        result.artifact.guard_calibration_evidence_digest
                    ),
                },
                guard_summaries=result.guard_summaries,
            ),
            arguments.calibration_outcomes,
        )
        report = format_guard_freeze_report(result)
    elif arguments.command == "evaluate-reserved":
        _preflight(
            (arguments.artifact,),
            (
                arguments.diagnostics_output,
                arguments.outcomes_output,
                arguments.report_output,
            ),
        )
        result = evaluate_corrected_reserved_cohort(
            arguments.artifact,
            arguments.repository_root,
            workers=arguments.workers,
            progress=lambda message: print(message, flush=True),
        )
        evidence_digest = corrected_partition_evidence_digest(result.evaluation)
        save_corrected_partition_diagnostics(
            result.evaluation,
            arguments.diagnostics_output,
        )
        save_diagnostic_payload(
            _outcome_payload(
                FINAL_EVALUATION_SPLIT,
                result.outcomes,
                result.summaries,
                provenance={
                    "generator_freeze_digest": (
                        result.generator_freeze.artifact_digest
                    ),
                    "source_manifest_digest": (
                        result.generator_freeze.source_manifest.digest
                    ),
                    "parent_protocol_digest": result.parent.protocol_digest,
                    "guard_protocol_digest": result.artifact.protocol_digest,
                    "source_commit": result.source_commit,
                    "partition_evidence_digest": evidence_digest,
                },
                guard_summaries=result.guard_summaries,
                paired_loss_comparisons=result.paired_loss_comparisons,
                primary_success=result.primary_success,
            ),
            arguments.outcomes_output,
        )
        report = format_final_evaluation_report(result)
    else:
        raise AssertionError("unhandled corrected replication command")
    print(report)
    _save_text_once(report, arguments.report_output)
    return 0


__all__ = [
    "DEFAULT_FINAL_DIAGNOSTICS",
    "DEFAULT_FINAL_OUTCOMES",
    "DEFAULT_GENERATOR_FREEZE",
    "DEFAULT_GUARD_ARTIFACT",
    "DEFAULT_PARENT_ARTIFACT",
    "format_final_evaluation_report",
    "format_guard_freeze_report",
    "format_parent_freeze_report",
    "format_parent_rehearsal_report",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
