"""Saved-evidence comparison; method verification is separate from forward gates."""
import argparse
import json
import math
import os
from pathlib import Path
import tempfile

from .pinn_spatial_continuity import canonical_digest, digest_file
from .pinn_spatial_graded import _by_id, _fmt, _gate, _protected_directories, _residuals
from .pinn_spatial_step_control import _case_evidence

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("baseline", "step_control", "regularized")
LABELS = {"baseline": "Original log-off", "step_control": "V7 step controller", "regularized": "Fixed ridge"}


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def _validate_regularized(path):
    from ..studies.pinn_spatial_regularized import validate_regularized
    return validate_regularized(path)


def _validate_step_control(path):
    from ..studies.pinn_spatial_step_control import validate_step_control
    return validate_step_control(path)


def _validate_graded(path):
    from ..studies.pinn_spatial_graded import validate_comparison
    return validate_comparison(path)


def load_comparison(campaign_directory, *, validation_directory=None):
    directory = Path(campaign_directory).resolve()
    validation = Path(validation_directory).resolve() if validation_directory else directory.parent/"regularized-validation"
    # Terminal flags only until both the complete campaign and supervisor stop.
    if any(_read(path).get("status") != "completed" for path in
           (directory/"state.json", directory/"execution.json", validation/"execution-state.json")):
        raise ValueError("campaign execution and supervisor must both be completed before report access")
    plan = _read(directory/"plan.json")
    context = _validate_regularized(directory)
    wrapper = _read(validation/"execution-state.json")
    if wrapper.get("evidence") != context or wrapper.get("plan_sha256") != digest_file(directory/"plan.json"):
        raise ValueError("completed supervisor proof differs from the campaign")
    previous = Path(plan["preserved_step_control"]["directory"]).resolve()
    old_plan = _read(previous/"plan.json")
    baseline = Path(old_plan["step_control_provenance"]["graded_comparison"]["directory"]).resolve()
    baseline_plan = _read(baseline/"plan.json")
    evidence = {"regularized": context, "step_control": _validate_step_control(previous), "baseline": _validate_graded(baseline)}
    if evidence["step_control"] != plan["preserved_step_control"]["evidence"]:
        raise ValueError("preserved step-controller evidence changed")
    baseline_bound = old_plan["step_control_provenance"]["graded_comparison"]
    if evidence["baseline"] != baseline_bound["context"] or digest_file(baseline/"plan.json") != baseline_bound["plan_sha256"]:
        raise ValueError("preserved original baseline changed")
    result = _read(directory/"result.json")
    results = {"regularized": result, "step_control": _read(previous/"result.json"),
               "baseline": _read(baseline/"log_off"/"final-forward-selection.json")}
    bases = {"regularized": directory, "step_control": previous, "baseline": baseline/"log_off"}
    declared = _by_id(plan["cases"], "regularized plan")
    mapped = {method: _by_id(value["cases"], method) for method, value in results.items()}
    if len(declared) != 4 or any(set(value) != set(declared) for value in mapped.values()):
        raise ValueError("all four cases must remain in each of the three method denominators")
    rows = []
    for case_id, case in declared.items():
        methods = {}
        for method in METHODS:
            saved = mapped[method][case_id]
            if any(saved.get(key) != case[key] for key in ("geometry", "known_h")):
                raise ValueError("paired physical case differs: "+case_id)
            value = _case_evidence(bases[method], case_id, saved)
            raw_path = bases[method]/case_id/"training-report.json"
            raw = _read(raw_path) if raw_path.is_file() else {}
            for name in ("final_unpenalized_objective", "final_regularization_penalty", "final_head_stationarity",
                         "initial_calibration_matrix_builds", "initial_calibration_spectral_calculations", "hard_singular_value_truncation"):
                value["training"][name] = raw.get(name)
            calibration = raw.get("readout_calibration") or {}
            value["training"]["readout_calibration_identity"] = {name: calibration.get(name) for name in
                ("sha256", "column_scale_sha256", "fixed_lambda", "initial_matrix_sha256", "initial_sigma_max", "relative_scale")}
            methods[method] = value
        rows.append(dict(case, methods=methods))
    protected = {str(directory), str(previous), str(baseline), str(validation)}
    for value in (plan, old_plan, baseline_plan):
        protected |= _protected_directories(value)
    protected |= set(_read(Path(plan["preserved_cutoff_audit"]["directory"])/"plan.json").get("protected_trees", {}))
    return {"schema": "fixed_ridge_saved_forward_comparison_v1", "development_only": True,
        "plan_sha256": digest_file(directory/"plan.json"), "campaign_status": result["status"],
        "saved_forward_status": result.get("forward_status"), "evidence": evidence,
        "method_verification": plan["method_checks"], "initial_method_checks_passed": result.get("initial_method_checks_passed"),
        "denominator": {"declared_cases_per_method": 4, "planned_case_attempts": 12, "reused_datasets": 4,
                        "new_datasets": 0, "independent_replications": 0},
        "configs": {"regularized": plan["model_config"], "step_control": old_plan["model_config"],
                    "baseline": baseline_plan["model_configs"]["log_off"]},
        "calibration_accounting": result.get("calibration_accounting"), "topology": plan.get("topology"),
        "campaign_costs": {method: {key: value.get(key) for key in ("status", "wall_seconds", "peak_rss_bytes", "peak_worker_rss_bytes", "forward_status")}
                           for method, value in results.items()},
        "supervisor": wrapper, "scope_flags": {key: result.get(key) for key in ("inverse_opened", "fresh_evaluation_opened", "three_d_opened")},
        "protected_directories": sorted(protected), "rows": rows}


def _markdown(summary):
    lines = ["# Continuous regularized readout: saved forward comparison", "",
        f"Saved forward status: **{summary['saved_forward_status']}**. Training-only method checks passed: **{summary['initial_method_checks_passed']}**.", "",
        "Gradient verification establishes the numerical method prerequisite. The unchanged forward gates separately assess the retained final field. Every method retains four cases, including failures and unrun cases. The same four opened noisy datasets are reused; there are no new datasets, independent replications, or uncertainty-coverage claims.", "",
        "The ridge method optimizes J = original weighted observation/physics loss + λ²‖Dc‖². Its fixed initial D and λ remain unchanged throughout each fit. Regularized J is a different objective and must not be ranked directly against unpenalized J from older methods. Both components are reported below.", "",
        "Saved primary limits: observation MSE 2; each applicable physics RMS 0.05; field RMS 0.05 K; maximum error 0.15 K. The reporter copies saved gates and never recomputes them.", "",
        "| Case | Method | Status | Saved gate | PDE RMS | Bottom contact | Bottom insulation | Field RMS K | Field max K | Obs MSE |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["rows"]:
        for method in METHODS:
            saved = row["methods"][method]["saved"]
            residuals, field = _residuals(saved), saved.get("field_metrics") or {}
            insulated = "not applicable" if row["geometry"] == "full_contact" else _fmt(residuals.get("bottom_insulated"))
            lines.append(f"| {row['id']} | {LABELS[method]} | {saved.get('status')} | {_gate((saved.get('gate') or {}).get('all'))} | {_fmt(residuals.get('interior'))} | {_fmt(residuals.get('bottom_contact'))} | {insulated} | {_fmt(field.get('rmse_K'))} | {_fmt(field.get('maximum_error_K'))} | {_fmt(saved.get('normalized_observation_mse'))} |")
            if saved.get("error") or saved.get("reason"):
                lines.extend(["", f"{row['id']} {method}: {saved.get('error') or saved.get('reason')}", ""])
    lines += ["", "## Training and fixed calibration", "",
        "Training endpoints are bounded returns, not convergence certificates. The JSON retains saved controller trajectories, conditioning, audits, costs and source identities.", "",
        "| Case | Method | Actual / max attempts | Stop | Training objective | Original loss | Penalty | Stationarity | Trial head solves |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["rows"]:
        for method in METHODS:
            training = row["methods"][method]["training"]
            original = training.get("final_unpenalized_objective") if method == "regularized" else training.get("final_objective")
            completed = training.get("completed_updates")
            requested = training.get("requested_updates")
            completed = training.get("hidden_update_count") if completed is None else completed
            requested = summary["configs"][method].get("adam_epochs") if requested is None else requested
            lines.append(f"| {row['id']} | {method} | {_fmt(completed)} / {_fmt(requested)} | {training.get('stop_reason') or 'not recorded'} | {_fmt(training.get('final_objective'))} | {_fmt(original)} | {_fmt(training.get('final_regularization_penalty'))} | {_fmt(training.get('final_head_stationarity'))} | {_fmt(training.get('trial_solve_count'))} |")
    lines += ["", "| Ridge case | Frozen λ | Calibration identity | Training s | Worker elapsed s | Worker peak RSS bytes |",
              "| --- | ---: | --- | ---: | ---: | ---: |"]
    for row in summary["rows"]:
        value = row["methods"]["regularized"]
        calibration = value["training"]["readout_calibration_identity"]
        lines.append(f"| {row['id']} | {_fmt(calibration.get('fixed_lambda'))} | {calibration.get('sha256') or 'not recorded'} | {_fmt(value['training'].get('wall_seconds'))} | {_fmt(value['saved'].get('wall_seconds'))} | {_fmt(value['saved'].get('peak_rss_bytes'))} |")
    lines += ["", "Preparation / worker accounting:", "", "```json", json.dumps(summary.get("calibration_accounting"), indent=2), "```", "",
        "The four initial calibration matrices/spectral calculations and worker head solves are separate costs. Threshold lines below are references; saved coordinator gates govern outcomes. Supplemental near-interface and bottom-layer audits remain report-only.", "",
        "No inverse study, reserved cohort, 3D stage, hardware validation, or automatic follow-up is opened by this report.", "",
        f"Plan SHA256: `{summary['plan_sha256']}`.", "", "[Complete saved evidence](result-summary.json).", "",
        "![Saved primary forward metrics](comparison.png)", ""]
    return "\n".join(lines)


def _plots(output, summary):
    with tempfile.TemporaryDirectory(prefix="thermotwin-ridge-plots-") as cache:
        os.environ.setdefault("MPLCONFIGDIR", cache)
        os.environ.setdefault("XDG_CACHE_HOME", cache)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        colors = ("#2563eb", "#6b7280", "#c2410c")
        panels = (("PDE normalized RMS", lambda s: _residuals(s).get("interior"), .05),
            ("Bottom contact normalized RMS", lambda s: _residuals(s).get("bottom_contact"), .05),
            ("Bottom insulation normalized RMS", lambda s: _residuals(s).get("bottom_insulated"), .05),
            ("Field RMS error (K)", lambda s: (s.get("field_metrics") or {}).get("rmse_K"), .05),
            ("Field maximum error (K)", lambda s: (s.get("field_metrics") or {}).get("maximum_error_K"), .15),
            ("Observation normalized MSE", lambda s: s.get("normalized_observation_mse"), 2.))
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
        for axis, (title, getter, limit) in zip(axes.ravel(), panels):
            for index, method in enumerate(METHODS):
                positions = np.arange(4)+(index-1)*.25
                values = [getter(row["methods"][method]["saved"]) for row in summary["rows"]]
                numeric = [type(value) in (int, float) and math.isfinite(value) for value in values]
                axis.bar(positions, [value if good else np.nan for value, good in zip(values, numeric)], width=.23,
                         color=colors[index], label=LABELS[method])
                for position, good, row in zip(positions, numeric, summary["rows"]):
                    if not good:
                        label = "n/a" if "insulation" in title and row["geometry"] == "full_contact" else row["methods"][method]["saved"].get("status", "missing")
                        axis.text(position, .02, label, transform=axis.get_xaxis_transform(), rotation=90, ha="center", fontsize=6, color=colors[index])
            axis.axhline(limit, linestyle="--", color="#991b1b", linewidth=1.)
            axis.set(title=title, xticks=np.arange(4), xticklabels=[row["id"] for row in summary["rows"]], ylim=(0, None), xlim=(-.6, 3.6))
            axis.grid(axis="y", alpha=.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("Saved forward metrics — numerical method checks are a separate prerequisite")
        fig.savefig(output/"comparison.png", dpi=170); fig.savefig(output/"comparison.pdf"); plt.close(fig)
    return ["comparison.png", "comparison.pdf"]


def render_report(campaign_directory, *, output_directory, validation_directory=None):
    source, output = Path(campaign_directory).resolve(), Path(output_directory).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("report output must be separate from campaign evidence")
    for ancestor in (output, *output.parents):
        if any((ancestor/name).is_file() for name in ("artifact-index.json", "evidence-index.json", "report-index.json", "manifest.json")):
            raise ValueError("report output must be separate from sealed evidence")
    summary = load_comparison(source, validation_directory=validation_directory)
    for name in summary["protected_directories"]:
        protected = Path(name).resolve()
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("report output must be separate from protected evidence")
    output.mkdir(parents=True, exist_ok=False)
    _write(output/"result-summary.json", summary)
    (output/"REPORT.md").write_text(_markdown(summary))
    names = ["REPORT.md", "result-summary.json"]+_plots(output, summary)
    records = [{"path": name, "bytes": (output/name).stat().st_size, "sha256": digest_file(output/name)} for name in names]
    index = {"files": records, "digest": canonical_digest(records), "renderer_sha256": digest_file(__file__),
        "plan_sha256": summary["plan_sha256"], "evidence": summary["evidence"]}
    _write(output/"report-index.json", index)
    return {"directory": str(output), "files": len(records), "digest": index["digest"], "index_sha256": digest_file(output/"report-index.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--validation-directory")
    args = parser.parse_args(argv)
    print(json.dumps(render_report(args.campaign_directory, output_directory=args.output_directory,
                                  validation_directory=args.validation_directory), allow_nan=False))


if __name__ == "__main__":
    main()
