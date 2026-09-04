"""Recompute and verify the principal public ThermoTwin evidence."""

import argparse
from dataclasses import dataclass
import math
from typing import Callable, NamedTuple, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ReleaseEvidenceExpectation:
    name: str
    expected: float
    absolute_tolerance: float
    unit: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("release evidence needs a name")
        if not math.isfinite(self.expected):
            raise ValueError("release evidence expectation must be finite")
        if (
            not math.isfinite(self.absolute_tolerance)
            or self.absolute_tolerance < 0.0
        ):
            raise ValueError("release evidence tolerance must be finite and nonnegative")


class ReleaseEvidenceCheck(NamedTuple):
    name: str
    observed: float
    expected: float
    absolute_tolerance: float
    unit: str
    passed: bool


class ReleaseEvidenceAudit(NamedTuple):
    checks: Tuple[ReleaseEvidenceCheck, ...]
    passed: bool


def _check(
    expectation: ReleaseEvidenceExpectation,
    observed: float,
) -> ReleaseEvidenceCheck:
    observed = float(observed)
    passed = math.isfinite(observed) and math.isclose(
        observed,
        expectation.expected,
        rel_tol=0.0,
        abs_tol=expectation.absolute_tolerance,
    )
    return ReleaseEvidenceCheck(
        name=expectation.name,
        observed=observed,
        expected=expectation.expected,
        absolute_tolerance=expectation.absolute_tolerance,
        unit=expectation.unit,
        passed=passed,
    )


def evaluate_release_evidence(
    engineering_showcase,
    forward_reconstruction,
    nonlinear_selection,
    codesign_campaign,
) -> ReleaseEvidenceAudit:
    """Compare recomputed results with the values used by public artifacts."""

    sparse = engineering_showcase.sparse_inference
    selection = engineering_showcase.experiment_selection
    physics = forward_reconstruction.summaries[0]
    data_only = forward_reconstruction.summaries[1]
    values = (
        (
            ReleaseEvidenceExpectation(
                "sparse inferred cold contact resistance", 0.25103, 5.0e-5, "K/W"
            ),
            sparse.fit.inferred_cold_contact_resistance,
        ),
        (
            ReleaseEvidenceExpectation(
                "sparse withheld exchanger RMSE", 0.00187, 5.0e-5, "K"
            ),
            sparse.withheld_validation.accessible_sensor_rmse,
        ),
        (
            ReleaseEvidenceExpectation("selected pulse amplitude", 0.8, 1.0e-12, "A"),
            selection.selected.current_amplitude,
        ),
        (
            ReleaseEvidenceExpectation("selected pulse duration", 20.0, 1.0e-12, "s"),
            selection.selected.pulse_duration,
        ),
        (
            ReleaseEvidenceExpectation(
                "selected local information gain", 7.198, 5.0e-4, "nats"
            ),
            selection.selected.information_gain_nats,
        ),
        (
            ReleaseEvidenceExpectation(
                "linearized parameter-RMSE reduction", 82.2, 0.05, "%"
            ),
            selection.validation.rmse_reduction_percent,
        ),
        (
            ReleaseEvidenceExpectation(
                "physics retained noisy-row RMSE", 0.019433, 5.0e-5, "K"
            ),
            physics.mean_retained_noisy_observation_rmse,
        ),
        (
            ReleaseEvidenceExpectation(
                "data-only retained noisy-row RMSE", 0.017471, 5.0e-5, "K"
            ),
            data_only.mean_retained_noisy_observation_rmse,
        ),
        (
            ReleaseEvidenceExpectation(
                "physics missing-exchanger RMSE", 0.009696, 5.0e-5, "K"
            ),
            physics.mean_missing_exchanger_rmse,
        ),
        (
            ReleaseEvidenceExpectation(
                "data-only missing-exchanger RMSE", 0.079878, 5.0e-4, "K"
            ),
            data_only.mean_missing_exchanger_rmse,
        ),
        (
            ReleaseEvidenceExpectation(
                "physics hidden-face RMSE", 0.007105, 5.0e-5, "K"
            ),
            physics.mean_hidden_face_rmse,
        ),
        (
            ReleaseEvidenceExpectation(
                "data-only hidden-face RMSE", 2.193724, 5.0e-3, "K"
            ),
            data_only.mean_hidden_face_rmse,
        ),
        (
            ReleaseEvidenceExpectation(
                "physics energy-rate closure RMS", 0.132833, 5.0e-4, "W"
            ),
            physics.mean_energy_rate_closure_rms,
        ),
        (
            ReleaseEvidenceExpectation(
                "data-only energy-rate closure RMS", 16.493587, 5.0e-2, "W"
            ),
            data_only.mean_energy_rate_closure_rms,
        ),
        (
            ReleaseEvidenceExpectation(
                "physics absolute final cumulative closure error",
                1.952169,
                5.0e-3,
                "J",
            ),
            physics.mean_absolute_final_cumulative_energy_error,
        ),
        (
            ReleaseEvidenceExpectation(
                "data-only absolute final cumulative closure error",
                370.392719,
                0.5,
                "J",
            ),
            data_only.mean_absolute_final_cumulative_energy_error,
        ),
        (
            ReleaseEvidenceExpectation(
                "physics missing-exchanger error reduction", 87.86, 0.05, "%"
            ),
            forward_reconstruction.physics_missing_exchanger_rmse_reduction_percent,
        ),
        (
            ReleaseEvidenceExpectation(
                "physics hidden-face error reduction", 99.68, 0.05, "%"
            ),
            forward_reconstruction.physics_hidden_face_rmse_reduction_percent,
        ),
        (
            ReleaseEvidenceExpectation(
                "physics energy-rate error reduction", 99.19, 0.05, "%"
            ),
            forward_reconstruction.physics_energy_rate_error_reduction_percent,
        ),
        (
            ReleaseEvidenceExpectation(
                "physics all-primary-metric advantages", 5.0, 0.0, "trials"
            ),
            forward_reconstruction.physics_all_metric_advantage_count,
        ),
        (
            ReleaseEvidenceExpectation(
                "physics completion-gate passes", 5.0, 0.0, "trials"
            ),
            forward_reconstruction.physics_completion_gate_pass_count,
        ),
        (
            ReleaseEvidenceExpectation(
                "nonlinear RMSE reduction versus naive", 81.46, 0.01, "%"
            ),
            nonlinear_selection.selected_rmse_reduction_vs_naive_percent,
        ),
        (
            ReleaseEvidenceExpectation(
                "nonlinear RMSE reduction versus closest-energy control",
                11.77,
                0.01,
                "%",
            ),
            nonlinear_selection.selected_rmse_reduction_vs_resource_control_percent,
        ),
        (
            ReleaseEvidenceExpectation(
                "nominal efficiency-winner robustness pass rate",
                55.3,
                0.05,
                "%",
            ),
            100.0 * codesign_campaign.robustness_results[0].feasible_fraction,
        ),
    )
    checks = tuple(_check(expectation, observed) for expectation, observed in values)
    return ReleaseEvidenceAudit(checks=checks, passed=all(item.passed for item in checks))


def run_release_evidence_audit(
    *,
    progress: Optional[Callable[[str], None]] = None,
) -> ReleaseEvidenceAudit:
    """Recompute the headline studies and compare them with release values."""

    # Keep these imports local so ``--help`` and the lightweight comparison
    # tests do not initialize PyTorch and Matplotlib.
    from .engineering_showcase import run_engineering_showcase
    from ..design.codesign.campaign import run_codesign_campaign
    from ..studies.forward_reconstruction_comparison import (
        run_forward_reconstruction_comparison,
    )
    from ..studies.nonlinear_experiment_selection import (
        run_nonlinear_experiment_selection_study,
    )

    if progress is not None:
        progress("recomputing engineering decision showcase")
    engineering = run_engineering_showcase()
    if progress is not None:
        progress("recomputing matched PINN reconstruction comparison")
    reconstruction = run_forward_reconstruction_comparison(progress=progress)
    if progress is not None:
        progress("recomputing nonlinear experiment-selection validation")
    nonlinear = run_nonlinear_experiment_selection_study(
        include_profiles=False,
        progress=progress,
    )
    if progress is not None:
        progress("recomputing material/geometry co-design campaign")
    codesign = run_codesign_campaign()
    return evaluate_release_evidence(
        engineering,
        reconstruction,
        nonlinear,
        codesign,
    )


def format_release_evidence_audit(audit: ReleaseEvidenceAudit) -> str:
    lines = [
        "ThermoTwin release evidence audit",
        "=================================",
    ]
    for item in audit.checks:
        suffix = f" {item.unit}" if item.unit else ""
        lines.append(
            f"{'PASS' if item.passed else 'FAIL'}  {item.name}: "
            f"observed={item.observed:.9g}{suffix}; "
            f"expected={item.expected:.9g}{suffix}; "
            f"tolerance={item.absolute_tolerance:.3g}{suffix}"
        )
    lines.extend(
        (
            "",
            f"overall: {'PASS' if audit.passed else 'FAIL'}",
            "scope: deterministic synthetic evidence only; not hardware validation",
        )
    )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recompute and verify ThermoTwin's public headline evidence."
    )
    parser.parse_args(argv)
    audit = run_release_evidence_audit(progress=lambda message: print(message, flush=True))
    print(format_release_evidence_audit(audit))
    return 0 if audit.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
