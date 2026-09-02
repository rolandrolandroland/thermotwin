"""Shared locations and human- and machine-readable report sidecars."""

from dataclasses import fields, is_dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Mapping


FIGURES_DIRECTORY = Path(__file__).resolve().parent.parent / "figures"


FIGURE_EXPLANATIONS = {
    "ag2se_matched_substitution.png": (
        "AG2SE_SUBSTITUTION_EXPERIMENT.md",
        "This multi-panel figure compares matched virtual devices before and after "
        "substituting the n-type leg with the published Ag2Se properties. It shows "
        "how often the substitution improves each application objective, how the "
        "effect changes between good and paper-derived electrical contacts, and why "
        "fixed areal contact resistance dilutes a bulk-material advantage.",
    ),
    "assembly_fingerprint.png": (
        "ASSEMBLY_FINGERPRINT_EXPERIMENT.md",
        "This figure tests a standardized pulse as a synthetic assembly-screening "
        "tool. The left panel compares hidden and inferred cold-interface resistance "
        "with local 95% intervals and illustrative classification bands; the right "
        "panel shows the accessible exchanger-temperature signatures used to "
        "distinguish the assemblies.",
    ),
    "contact_model_comparison.png": (
        "CONTACT_RESISTANCE_EXPERIMENT.md",
        "This figure compares the original two-node model with the contact-aware "
        "four-node reference model. It shows how finite thermal contacts create "
        "temperature drops between module faces and exchangers and how those extra "
        "states alter the transient response and delivered heat flow.",
    ),
    "contact_forward_pinn_comparison.png": (
        "CONTACT_RESISTANCE_EXPERIMENT.md",
        "This figure compares a four-node forward PINN with the conventional RK4 "
        "contact-aware trajectory. Temperature histories, residual behavior, and "
        "prediction errors show whether the learned solution satisfies the same "
        "transient energy balances as the numerical reference.",
    ),
    "control_comparison.png": (
        "CONTROL_COMPARISON_EXPERIMENT.md",
        "This figure compares optimized continuous current with matched-cooling "
        "rectangular pulses. It shows COP versus delivered cooling, the duty-cycle "
        "penalty caused by increased mean-square current, equal-power cooling, and "
        "the stability of the conclusion to cold-contact resistance.",
    ),
    "cop_operating_map.png": (
        "COP_OPERATING_MAP_EXPERIMENT.md",
        "This operating map shows cooling and heating COP across current and imposed "
        "temperature lift. It separates module-side performance from exchanger- or "
        "wall-plug-delivered performance so contact and auxiliary losses can be read "
        "as reductions from an otherwise identical module calculation.",
    ),
    "distributed_independent_validation.png": (
        "DISTRIBUTED_INDEPENDENT_VALIDATION.md",
        "This figure evaluates the distributed inverse model against independently "
        "generated constitutive truth rather than the same parameter family used for "
        "fitting. Its panels compare recovered property curves, internal temperature "
        "fields, terminal observables, and residuals to expose transfer error.",
    ),
    "distributed_inverse_robustness.png": (
        "DISTRIBUTED_INVERSE_ROBUSTNESS.md",
        "This figure summarizes repeated noisy distributed inverse-PINN fits across "
        "random initializations. It shows the spread and bias of recovered resistivity "
        "parameters, prediction errors, and convergence diagnostics rather than "
        "presenting a single favorable training run.",
    ),
    "distributed_observation_identifiability.png": (
        "DISTRIBUTED_OBSERVATION_IDENTIFIABILITY.md",
        "This figure asks whether the declared currents and terminal sensors contain "
        "enough independent information to distinguish the distributed resistivity "
        "degrees of freedom. Singular values, parameter correlations, and profile "
        "directions reveal weak or confounded combinations before inverse training.",
    ),
    "distributed_pinn_training_audit.png": (
        "DISTRIBUTED_PINN_TRAINING_AUDIT.md",
        "This training audit separates parameter recovery from actual PINN quality. "
        "It plots loss components, PDE and boundary residuals, constitutive-curve "
        "error, and field error so a small final scalar loss cannot hide an incorrect "
        "temperature field or property curve.",
    ),
    "distributed_profile_coverage.png": (
        "DISTRIBUTED_PROFILE_COVERAGE.md",
        "This figure tests uncertainty intervals on nonlinear resistivity profiles "
        "over repeated synthetic trials. It compares recovered curves with truth and "
        "summarizes empirical interval coverage, showing whether nominal uncertainty "
        "claims remain calibrated away from the simplest profile family.",
    ),
    "distributed_property_study.png": (
        "DISTRIBUTED_CONSTITUTIVE_INFERENCE.md",
        "This figure presents the core distributed constitutive-inference problem. "
        "It compares hidden and recovered temperature-dependent material laws, the "
        "corresponding internal thermal/electrical fields, measured terminal signals, "
        "and the fit residuals used to constrain the inverse PINN.",
    ),
    "distributed_withheld_validation.png": (
        "DISTRIBUTED_WITHHELD_VALIDATION.md",
        "This figure transfers an inferred distributed property model to a complete "
        "current regime withheld from fitting. Agreement in terminal responses and "
        "the hidden spatial field tests predictive transfer, while residual panels "
        "make any regime-specific model discrepancy visible.",
    ),
    "electrical_contact_process_window.png": (
        "ELECTRICAL_CONTACT_PROCESS_WINDOW.md",
        "This process-window figure maps specific electrical contact resistivity and "
        "leg length to contact resistance share, retained device ZT, and application "
        "feasibility. It converts an interface measurement into geometry-dependent "
        "process targets and distinguishes current-cap failures from physical cooling "
        "limits.",
    ),
    "engineering_decision_showcase.png": (
        "ENGINEERING_SHOWCASE.md",
        "This summary combines four established workflows: sparse hidden-parameter "
        "inference, continuous-versus-pulsed control, constrained next-experiment "
        "selection, and standardized assembly fingerprinting. Each panel is a compact "
        "entry point to the dedicated walkthrough and figure for that experiment.",
    ),
    "experiment_selection.png": (
        "NEXT_EXPERIMENT_WALKTHROUGH.md",
        "This figure shows how 25 candidate current pulses are ranked before data are "
        "collected. It identifies feasible candidates under energy and temperature "
        "constraints, highlights the selected and naive pulses, compares their local "
        "parameter uncertainties, and validates the expected RMSE reduction with "
        "repeated noise.",
    ),
    "forward_pinn_comparison.png": (
        "PINN_SHOWCASE.md",
        "This figure compares the simplest two-node forward PINN against the RK4 "
        "reference trajectory. It shows temperature predictions, pointwise error, "
        "physics residuals, and training behavior for a case where a conventional "
        "solver remains the accuracy baseline.",
    ),
    "inverse_contact_resistance_comparison.png": (
        "CONTACT_RESISTANCE_EXPERIMENT.md",
        "This figure compares inverse estimates of hidden contact resistance from "
        "synthetic temperatures. It shows the fitted response and parameter error for "
        "the inverse PINN alongside the conventional reference, clarifying what is "
        "identified by the observations and what remains model-assumed.",
    ),
    "material_geometry_bayesian_codesign.png": (
        "MATERIAL_GEOMETRY_BAYESIAN_CODESIGN.md",
        "This co-design figure connects material properties, geometry, interfaces, "
        "application constraints, cost assumptions, and robustness. It compares "
        "Bayesian optimization with the fixed candidate pool and reports both useful "
        "wins and null results rather than treating material ZT as a device objective.",
    ),
    "piecewise_contact_forward_pinn_comparison.png": (
        "CONTACT_RESISTANCE_EXPERIMENT.md",
        "This figure tests the contact-aware forward PINN under a time-varying, "
        "piecewise-constant current schedule. The comparison around current switches "
        "shows whether one learned trajectory respects state continuity and each "
        "interval's thermal energy balance.",
    ),
    "piecewise_inverse_contact_resistance_comparison.png": (
        "CONTACT_RESISTANCE_EXPERIMENT.md",
        "This figure evaluates inverse contact-resistance recovery using a richer "
        "piecewise-current experiment. It compares fitted and reference histories, "
        "parameter convergence, and residuals to show how additional excitation can "
        "separate interface loss from transient state evolution.",
    ),
    "pinn_showcase.png": (
        "PINN_SHOWCASE.md",
        "This combined PINN showcase contrasts forward state reconstruction, inverse "
        "contact inference, and conventional optimization. It emphasizes both the "
        "physics-constrained learned trajectories and the cases where the transparent "
        "numerical reference is faster or more accurate.",
    ),
    "pulse_operating_map.png": (
        "PULSE_OPERATING_MAP_EXPERIMENT.md",
        "This figure places seconds-scale pulse results on the steady continuous-COP "
        "envelope. Duty-cycle families show how the Joule penalty grows through mean "
        "square current and approaches the continuous limit as duty tends to one.",
    ),
    "pwm_power_electronics.png": (
        "PWM_POWER_ELECTRONICS_EXPERIMENT.md",
        "This figure models high-frequency PWM through thermal averages rather than "
        "millisecond thermal time steps. It shows how duty cycle changes mean current, "
        "mean-square current, Joule heating, converter loss, cooling, and wall-plug "
        "COP relative to ideal continuous drive.",
    ),
    "sparse_sensor_inference.png": (
        "SPARSE_SENSOR_EXPERIMENT.md",
        "This figure summarizes joint inference of cold contact resistance, sensor "
        "lag, and two sensor biases using only noisy exchanger temperatures with a "
        "missing-data interval. Parameter intervals, correlation, training error, "
        "hidden-face reconstruction, and withheld-schedule error show both recovery "
        "quality and remaining ambiguity.",
    ),
}


def experiment_artifact_directory(walkthrough: str) -> Path:
    """Return the figure folder matching one walkthrough Markdown filename."""

    candidate = Path(walkthrough)
    if not walkthrough or candidate.name != walkthrough:
        raise ValueError("walkthrough must be one plain Markdown filename or stem")
    if candidate.suffix not in {"", ".md"}:
        raise ValueError("walkthrough must have no suffix or the .md suffix")
    stem = candidate.stem
    if not stem or stem in {".", ".."}:
        raise ValueError("walkthrough stem cannot be empty")
    return FIGURES_DIRECTORY / stem


def default_figure_path(filename: str, walkthrough: str | None = None) -> Path:
    """Return an absolute figure path, grouped by walkthrough when supplied."""

    candidate = Path(filename)
    if not filename or candidate.name != filename:
        raise ValueError("figure filename must be one plain filename")
    directory = (
        experiment_artifact_directory(walkthrough)
        if walkthrough is not None
        else FIGURES_DIRECTORY
    )
    return directory / filename


def figure_data_path(figure_path: str | Path) -> Path:
    """Return the JSON-sidecar path colocated with a generated figure."""

    candidate = Path(figure_path).expanduser().resolve()
    if not candidate.suffix:
        raise ValueError("figure path must have a filename suffix")
    return candidate.with_suffix(".json")


def figure_explanation_path(figure_path: str | Path) -> Path:
    """Return the plain-text explanation path colocated with a figure."""

    candidate = Path(figure_path).expanduser().resolve()
    if not candidate.suffix:
        raise ValueError("figure path must have a filename suffix")
    return candidate.with_suffix(".txt")


def save_figure_explanation(
    figure_path: str | Path,
    *,
    explanation: str | None = None,
    walkthrough: str | None = None,
) -> Path:
    """Write a human-readable explanation beside one generated figure."""

    figure = Path(figure_path).expanduser().resolve()
    registered = FIGURE_EXPLANATIONS.get(figure.name)
    if registered is not None:
        registered_walkthrough, registered_explanation = registered
        walkthrough = walkthrough or registered_walkthrough
        explanation = explanation or registered_explanation
    if explanation is None:
        explanation = (
            "This generated figure summarizes the numerical result stored in its "
            "same-stem JSON sidecar. Consult the associated experiment walkthrough "
            "for the model assumptions, methods, and interpretation limits."
        )
    output = figure_explanation_path(figure)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"Figure: {figure.name}"]
    if walkthrough is not None:
        lines.extend((f"Walkthrough: thermotwin/{Path(walkthrough).name}", ""))
    else:
        lines.append("")
    lines.extend(
        (
            "What this figure shows",
            explanation.strip(),
            "",
            "Supporting files",
            (
                f"{figure.name} is the rendered figure. "
                f"{figure.with_suffix('.json').name} contains the structured report "
                "result used to generate it."
            ),
            "",
            "Interpretation boundary",
            (
                "This is a reproducible model result, not independent hardware "
                "validation. Read the walkthrough before quoting values because it "
                "states the assumptions, constraints, and validation boundary."
            ),
            "",
        )
    )
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return {
            name: _jsonable(getattr(value, name))
            for name in value._fields
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_jsonable(item) for item in sorted(value, key=repr)]
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        return _jsonable(to_list())
    detach = getattr(value, "detach", None)
    if callable(detach):
        return _jsonable(value.detach().cpu().tolist())
    return {
        "object_type": f"{type(value).__module__}.{type(value).__qualname__}"
    }


def save_figure_data(data: Any, figure_path: str | Path) -> Path:
    """Write deterministic JSON data and a plain-text figure explanation."""

    output = figure_data_path(figure_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "figure": Path(figure_path).name,
        "data": _jsonable(data),
    }
    with output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    save_figure_explanation(figure_path)
    return output
