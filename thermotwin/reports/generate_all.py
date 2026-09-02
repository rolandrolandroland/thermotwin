"""Regenerate every completed ThermoTwin experiment figure and sidecar."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


REPORT_MODULES = (
    "thermotwin.ag2se_substitution",
    "thermotwin.assembly_fingerprint_report",
    "thermotwin.contact_report",
    "thermotwin.contact_forward_pinn_report",
    "thermotwin.inverse_contact_resistance_report",
    "thermotwin.piecewise_contact_forward_pinn_report",
    "thermotwin.piecewise_inverse_contact_resistance_report",
    "thermotwin.control_comparison_report",
    "thermotwin.cop_operating_map_report",
    "thermotwin.distributed_property_report",
    "thermotwin.distributed_inverse_robustness",
    "thermotwin.distributed_pinn_training_audit",
    "thermotwin.distributed_withheld_validation",
    "thermotwin.distributed_independent_validation",
    "thermotwin.distributed_observation_identifiability",
    "thermotwin.distributed_profile_coverage",
    "thermotwin.contact_process_window",
    "thermotwin.engineering_showcase",
    "thermotwin.experiment_selection_report",
    "thermotwin.forward_pinn_report",
    "thermotwin.material_geometry_codesign_report",
    "thermotwin.pinn_showcase",
    "thermotwin.pulse_operating_map_report",
    "thermotwin.pwm_power_electronics_report",
    "thermotwin.sparse_sensor_report",
)

REPORT_ARGUMENTS = {
    # The fast default deliberately leaves the inverse-validation panels blank.
    # The catalog should contain the completed experiment rather than a prompt.
    "thermotwin.distributed_property_report": ("--train-inverse-pinn",),
}


def generate_all_figures(
    modules: Sequence[str] = REPORT_MODULES,
) -> tuple[str, ...]:
    """Run each canonical report in an isolated Python process."""

    modules = tuple(modules)
    if not modules:
        raise ValueError("at least one report module is required")
    unknown = tuple(module for module in modules if module not in REPORT_MODULES)
    if unknown:
        raise ValueError(f"unknown report modules: {', '.join(unknown)}")
    environment = dict(os.environ)
    environment.setdefault(
        "MPLCONFIGDIR",
        str(Path(os.environ.get("TMPDIR", "/tmp")) / "thermotwin-matplotlib"),
    )
    for index, module in enumerate(modules, start=1):
        print(f"[{index}/{len(modules)}] {module}", flush=True)
        subprocess.run(
            (sys.executable, "-m", module, *REPORT_ARGUMENTS.get(module, ())),
            check=True,
            env=environment,
        )
    return modules


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate every completed ThermoTwin experiment figure with JSON "
            "data and a plain-text explanation"
        )
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=REPORT_MODULES,
        help="generate only this report module; repeat to select several",
    )
    arguments = parser.parse_args()
    generate_all_figures(arguments.only or REPORT_MODULES)


if __name__ == "__main__":
    main()
