"""Dedicated figure report for standardized assembly fingerprinting."""

import argparse
from pathlib import Path
from typing import Union

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..inference.assembly_fingerprint import (
    AssemblyFingerprintStudyResult,
    format_assembly_fingerprint_report,
    run_assembly_fingerprint_study,
)
from .paths import default_figure_path, save_figure_data


DEFAULT_ASSEMBLY_FINGERPRINT_PATH = default_figure_path(
    "assembly_fingerprint.png", "ASSEMBLY_FINGERPRINT_EXPERIMENT.md"
)


def save_assembly_fingerprint_report(
    result: AssemblyFingerprintStudyResult,
    output_path: Union[str, Path],
) -> Path:
    """Save inference accuracy and accessible thermal signatures."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(13.0, 5.8), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    fingerprints = result.fingerprints

    truth = tuple(item.true_cold_contact_resistance for item in fingerprints)
    inferred = tuple(item.inferred_cold_contact_resistance for item in fingerprints)
    lower_error = tuple(
        item.inferred_cold_contact_resistance - item.lower_95 for item in fingerprints
    )
    upper_error = tuple(
        item.upper_95 - item.inferred_cold_contact_resistance for item in fingerprints
    )
    lower = min(truth + inferred) - 0.025
    upper = max(truth + inferred) + 0.025
    axis = axes[0]
    axis.axhspan(0.20, 0.30, color="tab:green", alpha=0.12, label="Reference band")
    axis.errorbar(
        truth,
        inferred,
        yerr=(lower_error, upper_error),
        fmt="o",
        capsize=5,
        color="tab:blue",
        label="Estimate and local 95% interval",
    )
    axis.plot((lower, upper), (lower, upper), "k--", label="Ideal inference")
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_xlabel("Hidden cold contact resistance (K/W)")
    axis.set_ylabel("Inferred cold contact resistance (K/W)")
    axis.set_title("Assembly grading accuracy")
    axis.legend(fontsize="small")

    positions = tuple(range(len(fingerprints)))
    width = 0.38
    axis = axes[1]
    axis.bar(
        tuple(position - width / 2 for position in positions),
        tuple(item.cold_exchanger_drop for item in fingerprints),
        width=width,
        label="Cold exchanger drop",
    )
    axis.bar(
        tuple(position + width / 2 for position in positions),
        tuple(item.hot_exchanger_rise for item in fingerprints),
        width=width,
        label="Hot exchanger rise",
    )
    axis.set_xticks(positions, tuple(item.name.replace("_", "\n") for item in fingerprints))
    axis.set_ylabel("Temperature excursion (K)")
    axis.set_title("Accessible standardized-pulse signatures")
    axis.legend(fontsize="small")

    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.suptitle(
        f"Synthetic assembly fingerprint: {result.pulse_current:.1f} A for "
        f"{result.pulse_duration:.0f} s, sensor noise {result.noise_standard_deviation:.2f} K",
        fontsize=14,
    )
    figure.savefig(destination, dpi=170)
    save_figure_data(result, destination)
    return destination


def run_and_save_assembly_fingerprint_report(
    output_path: Union[str, Path] = DEFAULT_ASSEMBLY_FINGERPRINT_PATH,
) -> tuple[AssemblyFingerprintStudyResult, Path]:
    result = run_assembly_fingerprint_study()
    return result, save_assembly_fingerprint_report(result, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the synthetic assembly-fingerprint figure"
    )
    parser.add_argument("--output", default=str(DEFAULT_ASSEMBLY_FINGERPRINT_PATH))
    arguments = parser.parse_args()
    result, destination = run_and_save_assembly_fingerprint_report(arguments.output)
    print(format_assembly_fingerprint_report(result))
    print(f"\nfigure: {destination}")


if __name__ == "__main__":
    main()
