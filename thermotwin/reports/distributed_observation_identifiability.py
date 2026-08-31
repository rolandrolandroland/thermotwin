"""Report for distributed-property observation sufficiency and fit rejection."""

import argparse
import math
from pathlib import Path
from typing import Optional, Sequence

from ..figure_paths import default_figure_path
from ..physics.distributed import PiecewiseLinearProperty
from ..simulation.distributed import distributed_inverse_constant_experiments
from ..studies.distributed_observation_identifiability import (
    DistributedObservationIdentifiabilityConfig,
    DistributedObservationIdentifiabilityStudyResult,
    estimator_coefficient_spread,
    run_distributed_observation_identifiability_study,
)


def _multipliers(values) -> str:
    if values is None:
        return "training failed"
    return "(" + ", ".join(f"{value:.6f}" for value in values) + ")"


def format_distributed_observation_identifiability_report(
    result: DistributedObservationIdentifiabilityStudyResult,
) -> str:
    """Report the pre-fit decision separately from diagnostic optimizer output."""

    config = result.config
    gate = config.gate
    lines = [
        "Distributed resistivity: observation sufficiency gate",
        "=====================================================",
        "",
        "Question:",
        "  Can the available current regimes and sensors support a unique local",
        "  three-knot rho_e(T) curve, or must inference refuse the estimate?",
        "",
        "Independent synthetic data:",
        f"  truth: {config.truth_node_count}-node SSPRK3 generator, "
        f"dt={config.truth_time_step:.3e} s",
        "  truth curve: smooth cubic outside the fitted three-knot basis",
        f"  truth knot multipliers: {_multipliers(result.truth_knot_multipliers)}",
        f"  noise: {config.temperature_standard_deviation:.6f} K and "
        f"{config.voltage_standard_deviation:.6e} V",
        "",
        "Pre-fit practical gate:",
        f"  allowed local log-coefficient displacement: "
        f"{gate.maximum_log_displacement:.3f}",
        f"  required noise-normalized signal: "
        f"{gate.required_noise_normalized_signal:.3f}",
        "  a supported curve needs every singular value >= "
        f"{gate.required_noise_normalized_signal / gate.maximum_log_displacement:.6f}",
        "  this is a declared local one-sigma resolution rule, not a global proof.",
        "",
        "Observation cases:",
    ]
    for case in result.cases:
        singular = ", ".join(
            f"{value:.6g}" for value in case.identifiability.singular_values
        )
        weakest = case.assessment.weakest_resolvable_log_displacement
        weakest_text = "infinite" if math.isinf(weakest) else f"{weakest:.6f}"
        lines.extend(
            (
                f"  {case.definition.name}:",
                f"    question: {case.definition.scientific_question}",
                f"    observations: {case.identifiability.observation_count}; "
                f"singular values=({singular})",
                f"    decision: {case.assessment.status}; supported rank="
                f"{case.assessment.supported_rank}/"
                f"{case.assessment.coefficient_count}",
                f"    weakest one-sigma log displacement: {weakest_text}",
                f"    fit policy: {case.fit_policy}",
            )
        )
        if not case.fits:
            lines.append(
                "    no curve was fitted because rho_e is absent from the selected physics."
            )
            continue
        for estimator in ("conventional", "pinn"):
            spread = estimator_coefficient_spread(case, estimator)
            lines.append(
                f"    {estimator} maximum multistart coefficient spread: "
                f"{spread:.6f}"
            )
            for fit in case.fits:
                if fit.estimator != estimator:
                    continue
                lines.append(
                    f"      start {fit.start_index}: initial="
                    f"{_multipliers(fit.initial_multipliers)}; fitted="
                    f"{_multipliers(fit.fitted_multipliers)}; "
                    f"data loss={fit.normalized_observation_loss:.6f}; "
                    f"property max error={fit.property_maximum_relative_error:.6f}; "
                    f"holdout internal={fit.holdout_internal_temperature_rmse:.6f} K; "
                    f"holdout voltage={fit.holdout_voltage_rmse:.6e} V; "
                    f"prefit gate permits estimate={fit.permitted_by_prefit_gate}"
                )
    lines.extend(
        (
            "",
            "Interpretation:",
            "  Zero current is structurally blind to electrical resistivity: neither",
            "  Joule heating nor the ohmic voltage term contains rho_e when I=0.",
            "  Positive-current face temperatures carry only weak rho_e information",
            "  through I^2 rho_e heating. Voltage strongly identifies an average",
            "  resistance but one current direction does not resolve every curve-shape",
            "  direction at the declared noise and coefficient range.",
            "  Only the bidirectional, temperature-plus-voltage set clears the frozen",
            "  three-direction gate. That makes its fits eligible for validation; it",
            "  does not guarantee estimator accuracy. Optimizer output from weaker",
            "  cases is retained as a diagnostic and explicitly rejected as a property",
            "  estimate. A stable-looking PINN curve cannot manufacture missing rank.",
            "",
            "Boundary:",
            "  The rank is local to the baseline curve, finite experiment set, sensor",
            "  noise, and +/-0.3 log neighborhood. It is not a global uniqueness proof.",
            "  The truth generator is numerically and constitutively independent but",
            "  still obeys the same one-dimensional continuum equations.",
        )
    )
    return "\n".join(lines)


def save_distributed_observation_identifiability_figure(
    result: DistributedObservationIdentifiabilityStudyResult,
    output: Path | str,
) -> Path:
    """Plot singular spectra, supported ranks, curves, and multistart spread."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    names = tuple(case.definition.name for case in result.cases)

    spectrum_axis = axes[0, 0]
    for case in result.cases:
        spectrum_axis.plot(
            (1, 2, 3),
            tuple(max(1.0e-8, value) for value in case.identifiability.singular_values),
            marker="o",
            label=case.definition.name,
        )
    threshold = (
        result.config.gate.required_noise_normalized_signal
        / result.config.gate.maximum_log_displacement
    )
    spectrum_axis.axhline(threshold, color="red", linestyle="--", label="gate")
    spectrum_axis.set_yscale("log")
    spectrum_axis.set_title("Noise-normalized local singular spectra")
    spectrum_axis.set_xlabel("Singular direction")
    spectrum_axis.set_ylabel("Singular value")
    spectrum_axis.grid(True, which="both", alpha=0.25)
    spectrum_axis.legend(fontsize=8)

    rank_axis = axes[0, 1]
    rank_axis.bar(
        range(len(result.cases)),
        tuple(case.assessment.supported_rank for case in result.cases),
        color=("#2ca02c", "#d62728", "#ff7f0e", "#ff7f0e"),
    )
    rank_axis.axhline(3.0, color="black", linestyle="--", linewidth=1)
    rank_axis.set_xticks(range(len(names)), names, rotation=20, ha="right")
    rank_axis.set_ylim(0.0, 3.3)
    rank_axis.set_ylabel("Supported coefficient directions")
    rank_axis.set_title("Pre-fit decision (three coefficients total)")

    curve_axis = axes[1, 0]
    temperatures = tuple(285.0 + 0.5 * index for index in range(61))
    curve_axis.plot(
        temperatures,
        tuple(result.truth_property.value(value) * 1.0e5 for value in temperatures),
        color="black",
        linewidth=2.5,
        label="independent truth",
    )
    baseline = distributed_inverse_constant_experiments()[0].material.electrical_resistivity
    if not isinstance(baseline, PiecewiseLinearProperty):
        raise TypeError("baseline resistivity must be piecewise linear")
    full = result.cases[0]
    for fit in full.fits:
        if fit.fitted_multipliers is None:
            continue
        prop = baseline.with_values(
            tuple(
                value * multiplier
                for value, multiplier in zip(
                    baseline.values, fit.fitted_multipliers
                )
            )
        )
        curve_axis.plot(
            temperatures,
            tuple(prop.value(value) * 1.0e5 for value in temperatures),
            alpha=0.65,
            label=f"{fit.estimator} start {fit.start_index}",
        )
    curve_axis.set_title("Gate-eligible bidirectional multistart fits")
    curve_axis.set_xlabel("Temperature (K)")
    curve_axis.set_ylabel("Resistivity (10^-5 ohm m)")
    curve_axis.grid(alpha=0.25)
    curve_axis.legend(fontsize=7)

    spread_axis = axes[1, 1]
    width = 0.36
    indices = tuple(range(len(result.cases)))
    for offset, estimator, color in (
        (-width / 2, "conventional", "#1f77b4"),
        (width / 2, "pinn", "#9467bd"),
    ):
        values = tuple(
            estimator_coefficient_spread(case, estimator) for case in result.cases
        )
        finite = tuple(0.0 if math.isinf(value) else value for value in values)
        spread_axis.bar(
            tuple(index + offset for index in indices),
            finite,
            width=width,
            label=estimator,
            color=color,
        )
    spread_axis.set_xticks(indices, names, rotation=20, ha="right")
    spread_axis.set_ylabel("Maximum multiplier range")
    spread_axis.set_title("Dependence on initial property curve")
    spread_axis.legend()
    spread_axis.grid(axis="y", alpha=0.25)

    figure.savefig(output_path, dpi=170)
    plt.close(figure)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the distributed observation-identifiability study."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_figure_path("distributed_observation_identifiability.png"),
    )
    parser.add_argument("--epochs", type=int, default=500)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_distributed_observation_identifiability_study(
        DistributedObservationIdentifiabilityConfig(
            inverse_pinn_epochs=args.epochs
        )
    )
    print(format_distributed_observation_identifiability_report(result))
    output = save_distributed_observation_identifiability_figure(
        result, args.output
    )
    print(f"figure: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
