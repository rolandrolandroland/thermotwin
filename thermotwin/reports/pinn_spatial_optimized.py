"""Saved-evidence comparison of the probe-selected optimizer and log-off rule.

This renderer validates completed campaigns and copies coordinator decisions.
It cannot fit a model, recompute a gate, select a checkpoint, or open an inverse
or reserved cohort. Report files are written outside all protected evidence.
"""
import argparse
import json
import math
import os
from pathlib import Path
import tempfile

from .pinn_spatial_continuity import canonical_digest, digest_file
from .pinn_spatial_graded import (
    _by_id, _fmt, _gate, _optional, _protected_directories,
    _residuals, _training,
)


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("baseline", "optimized")


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def _validate_optimized(directory):
    from ..studies.pinn_spatial_optimized import validate_optimized
    return validate_optimized(directory)


def _validate_graded(directory):
    from ..studies.pinn_spatial_graded import validate_comparison
    return validate_comparison(directory)


def _check_sources(records):
    for name, expected in records.items():
        path = (ROOT/name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(ROOT.resolve()) or not path.is_file() or digest_file(path) != expected:
            raise ValueError("declared report source changed: "+name)


def _training_summary(raw):
    result = _training(raw)
    for name in ("optimizer_history", "optimizer_instances", "optimizer_state_rollbacks",
                 "optimizer_state_policy", "final_optimizer_step_counts", "training_family_mse"):
        result[name] = raw.get(name)
    history = raw.get("hidden_update_history")
    if history is not None:
        for summary, event in zip(result["trajectory"], history):
            for name in ("optimizer_step_counts_before", "optimizer_step_counts_proposed", "optimizer_step_counts_after",
                         "optimizer_state_committed", "optimizer_state_rolled_back", "optimizer_reset_required",
                         "proposed_hidden_displacement_finite", "proposed_optimizer_state_finite"):
                summary[name] = event.get(name)
    return result


def _probe_evidence(bound_decision):
    decision = bound_decision["decision"]
    records = []
    for saved in decision.get("probe_evidence") or []:
        path = Path(saved["path"])
        path = path.resolve() if path.is_absolute() else (ROOT/path).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or digest_file(path) != saved["sha256"]:
            raise ValueError("bound optimizer probe evidence changed")
        record = dict(saved)
        if path.suffix == ".json":
            value = _read(path)
            if isinstance(value, dict):
                record.update(status=value.get("status"), wall_seconds=value.get("wall_seconds"),
                              peak_rss_bytes=value.get("peak_rss_bytes"))
        records.append(record)
    return records


def load_comparison(campaign_directory):
    directory = Path(campaign_directory).resolve()
    if _read(directory/"state.json").get("status") != "completed":
        raise ValueError("optimizer campaign has not completed; outcomes remain unopened")
    plan = _read(directory/"plan.json")
    provenance = plan["optimization_provenance"]
    old_directory = Path(provenance["graded_comparison"]["directory"]).resolve()
    if _read(old_directory/"state.json").get("status") != "completed":
        raise ValueError("graded baseline has not completed")
    current_evidence = _validate_optimized(directory)
    old_evidence = _validate_graded(old_directory)
    if old_evidence != provenance["graded_comparison"]["context"]:
        raise ValueError("graded baseline evidence differs from the optimizer plan")
    if digest_file(old_directory/"plan.json") != provenance["graded_comparison"]["plan_sha256"]:
        raise ValueError("graded baseline plan differs from the optimizer plan")
    old_plan = _read(old_directory/"plan.json")
    _check_sources(plan.get("source_sha256") or {})
    _check_sources(old_plan.get("source_sha256") or {})
    result = _read(directory/"final-forward-selection.json")
    old_result = _read(old_directory/"log_off"/"final-forward-selection.json")
    old_top = _read(old_directory/"result.json")
    declared = _by_id(plan["cases"], "optimizer plan")
    old_declared = _by_id(old_plan["cases"], "graded plan")
    current = _by_id(result.get("cases") or [], "optimizer outcomes")
    old = _by_id(old_result.get("cases") or [], "baseline outcomes")
    top_rows = _by_id(old_top.get("paired_case_rows") or [], "graded paired outcomes")
    if len(declared) != 4 or any(set(value) != set(declared) for value in (old_declared, current, old, top_rows)):
        raise ValueError("comparison must retain exactly four declared paired datasets")
    if result.get("source_sha256") != plan.get("source_sha256"):
        raise ValueError("optimizer outcome source proof differs from its plan")
    rows = []
    for case_id, case in declared.items():
        methods = {}
        for method, saved, base in (("baseline", old[case_id], old_directory/"log_off"),
                                    ("optimized", current[case_id], directory)):
            if any(saved.get(key) != case[key] or old_declared[case_id].get(key) != case[key]
                   for key in ("geometry", "known_h")):
                raise ValueError("paired physical case changed: "+case_id)
            if method == "baseline" and top_rows[case_id]["methods"]["log_off"] != saved:
                raise ValueError("baseline row differs from its sealed combined result")
            raw_path = base/case_id/"training-report.json"
            raw = _read(raw_path) if raw_path.is_file() else {}
            methods[method] = {"saved": saved, "training": _training_summary(raw),
                "training_report_sha256": digest_file(raw_path) if raw_path.is_file() else None,
                "near_interface_audit": _optional(base/case_id/"near-interface-audit.json"),
                "bottom_layer_audit": _optional(base/case_id/"near-bottom-audit.json")}
        rows.append(dict(case, methods=methods))
    probes = _probe_evidence(provenance["decision"])
    protected = _protected_directories(plan) | _protected_directories(old_plan) | {str(directory), str(old_directory)}
    for record in probes:
        path = Path(record["path"])
        path = path.resolve() if path.is_absolute() else (ROOT/path).resolve()
        if (path.parent/"plan.json").is_file() and (path.parent/"state.json").is_file():
            protected.add(str(path.parent))
    return {"schema": "probe_selected_optimizer_saved_evidence_report", "development_only": True,
        "campaign_status": result.get("status"), "plan_sha256": digest_file(directory/"plan.json"),
        "campaign_evidence": current_evidence, "baseline_evidence": old_evidence,
        "source_sha256": plan.get("source_sha256"), "source_unchanged": result.get("source_unchanged"),
        "denominator": {"declared_cases_per_method": 4, "method_fits": 8,
            "reused_paired_datasets": 4, "independent_noise_replications_claimed": False},
        "configs": {"baseline": old_plan["model_configs"]["log_off"], "optimized": plan["model_config"]},
        "campaign_cost": {"baseline": {key: old_result.get(key) for key in ("wall_seconds", "peak_rss_bytes", "status")},
                          "optimized": {key: result.get(key) for key in ("wall_seconds", "peak_rss_bytes", "status")}},
        "optimizer_decision": provenance["decision"], "probe_evidence": probes,
        "initialization_policy": provenance.get("initialization_policy"),
        "selection_boundary": provenance.get("selection_boundary"),
        "saved_scope_flags": {key: result.get(key) for key in ("inverse_opened", "fresh_evaluation_opened", "three_d_opened")},
        "protected_directories": sorted(protected), "rows": rows}


def _labels(comparison):
    return {"baseline": f"Log-off {comparison['configs']['baseline']['adam_epochs']} updates",
            "optimized": f"Selected optimizer {comparison['configs']['optimized']['adam_epochs']} updates"}


def _markdown(comparison):
    labels = _labels(comparison)
    config = comparison["configs"]["optimized"]
    lines = ["# Probe-selected optimizer: saved forward checks", "",
        f"Saved coordinator status: **{comparison['campaign_status']}**.", "",
        "Four reused development datasets are retained for each method, including failed and unrun rows. These are fresh initializations on previously opened observations, not independent replications or fresh-cohort evaluation. The reporter copies saved coordinator gates; it does not recalculate acceptance, select checkpoints, or certify convergence.", "",
        f"The selected rule uses learning rate **{_fmt(config.get('learning_rate'))}**, optimizer history **{config.get('optimizer_history', 'not recorded')}**, and **{_fmt(config.get('adam_epochs'))}** fixed hidden updates. The saved training-only probe decision binds this rule. Architecture, log-off input, graded grid, observations, objective, readout cutoff and primary gates remain the declared comparison controls.", "",
        "Unchanged primary limits: observation MSE 2; each applicable normalized physics RMS 0.05; field RMS 0.05 K; field maximum 0.15 K. Threshold lines are references; saved coordinator decisions govern every reported outcome.", "",
        "## Saved primary outcomes", "",
        "| Case | Geometry / known h | Method | Status | Saved overall gate | PDE RMS | Bottom contact | Bottom insulated | Field RMS K | Field max K | Obs MSE |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in comparison["rows"]:
        for method in METHODS:
            saved = row["methods"][method]["saved"]
            physics, field = _residuals(saved), saved.get("field_metrics") or {}
            insulated = "not applicable" if row["geometry"] == "full_contact" else _fmt(physics.get("bottom_insulated"))
            lines.append(f"| {row['id']} | {row['geometry']} / {_fmt(row['known_h'])} | {labels[method]} | {saved.get('status', 'not recorded')} | {_gate((saved.get('gate') or {}).get('all'))} | {_fmt(physics.get('interior'))} | {_fmt(physics.get('bottom_contact'))} | {insulated} | {_fmt(field.get('rmse_K'))} | {_fmt(field.get('maximum_error_K'))} | {_fmt(saved.get('normalized_observation_mse'))} |")
            if saved.get("error") or saved.get("reason"):
                lines.extend(["", f"{row['id']} {method}: {saved.get('error') or saved.get('reason')}", ""])
    lines += ["", "## Boundary and continuity diagnostics", "",
        "All values are saved normalized RMS. Full-contact cases have no artificial interface. Exact continuity does not establish heat-equation accuracy.", "",
        "| Case | Method | Top | Left | Right | Interface 0 T / flux | Interface 1 T / flux |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |"]
    for row in comparison["rows"]:
        for method in METHODS:
            values = _residuals(row["methods"][method]["saved"])
            interfaces = ["not applicable" if row["geometry"] == "full_contact" else
                f"{_fmt(values.get(f'interface_{index}_temperature'))} / {_fmt(values.get(f'interface_{index}_flux'))}" for index in (0, 1)]
            lines.append(f"| {row['id']} | {method} | {_fmt(values.get('top'))} | {_fmt(values.get('left'))} | {_fmt(values.get('right'))} | {interfaces[0]} | {interfaces[1]} |")
    lines += ["", "## Training trajectory and numerical conditioning", "",
        "Training objectives and individual family mean squares are retained in result-summary.json. Fixed-readout gradients are proposals; truncated SVD does not establish exact profiled-gradient semantics. Budget endpoints are not convergence certificates.", "",
        "| Case | Method | Accepted / recorded | Rejected | Backtracks | Training s | Final rank / columns | Condition | Coefficient norm | Final objective |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |"]
    for row in comparison["rows"]:
        for method in METHODS:
            training = row["methods"][method]["training"]
            final = training["final_projection"]
            lines.append(f"| {row['id']} | {method} | {_fmt(training['hidden_accepted_updates'])} / {_fmt(training['hidden_update_count'])} | {_fmt(training['hidden_rejected_updates'])} | {_fmt(training['hidden_backtrack_count'])} | {_fmt(training['wall_seconds'])} | {_fmt(final['rank'])} / {_fmt(final['columns'])} | {_fmt(final['resolved_condition'])} | {_fmt(final['coefficient_norm_returned'])} | {_fmt(training['final_objective'])} |")
    lines += ["", "| Case | Selected history | Adam instances | State rollbacks | Final saved moment step counts |",
              "| --- | --- | ---: | ---: | --- |"]
    for row in comparison["rows"]:
        training = row["methods"]["optimized"]["training"]
        counts = training.get("final_optimizer_step_counts")
        lines.append(f"| {row['id']} | {training.get('optimizer_history') or 'not recorded'} | {_fmt(training.get('optimizer_instances'))} | {_fmt(training.get('optimizer_state_rollbacks'))} | {counts if counts is not None else 'not recorded'} |")
    lines += ["", "Saved training-family mean squares (not independent gate values):", "",
        "| Case | Method | Interior MSE | Bottom contact MSE | Bottom insulation MSE |",
        "| --- | --- | ---: | ---: | ---: |"]
    for row in comparison["rows"]:
        for method in METHODS:
            families = row["methods"][method]["training"].get("training_family_mse") or {}
            insulated = "not applicable" if row["geometry"] == "full_contact" else _fmt(families.get("bottom_insulated"))
            lines.append(f"| {row['id']} | {method} | {_fmt(families.get('interior'))} | {_fmt(families.get('bottom_contact'))} | {insulated} |")
    lines += ["", "## Saved computation costs", "",
        "Training times exclude later scoring. Campaign elapsed time and peak process RSS are separately recorded. Probe records may contain nested totals; they are displayed individually and are not summed into CPU time.", "",
        "| Record | Elapsed s | Peak RSS bytes | Saved status |", "| --- | ---: | ---: | --- |"]
    for method in METHODS:
        cost = comparison["campaign_cost"][method]
        lines.append(f"| {labels[method]} campaign | {_fmt(cost.get('wall_seconds'))} | {_fmt(cost.get('peak_rss_bytes'))} | {cost.get('status', 'not recorded')} |")
    for record in comparison["probe_evidence"]:
        lines.append(f"| {record['path']} | {_fmt(record.get('wall_seconds'))} | {_fmt(record.get('peak_rss_bytes'))} | {record.get('status') or 'not recorded'} |")
    lines += ["", "## Unchanged report-only audits", "",
        "Near-interface and whole-plate bottom-layer audits retain their saved values and have no primary acceptance or selection role. Missing diagnostics remain missing.", "",
        "| Case | Method | Edge status / PDE RMS | Bottom status / PDE RMS |", "| --- | --- | --- | --- |"]
    for row in comparison["rows"]:
        for method in METHODS:
            value = row["methods"][method]
            edge, bottom = value["near_interface_audit"], value["bottom_layer_audit"]
            edge_rms = ((edge.get("diagnostics") or {}).get("normalized_rms") or {}).get("interior")
            bottom_rms = ((bottom.get("diagnostics") or {}).get("normalized_rms") or {}).get("interior")
            lines.append(f"| {row['id']} | {method} | {edge.get('status', 'not recorded')} / {_fmt(edge_rms)} | {bottom.get('status', 'not recorded')} / {_fmt(bottom_rms)} |")
    lines += ["", "| Case | Method | z / H | Points | PDE normalized RMS | PDE physical RMS K/s |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in comparison["rows"]:
        for method in METHODS:
            strata = row["methods"][method]["bottom_layer_audit"].get("by_height") or []
            if not strata:
                lines.append(f"| {row['id']} | {method} | not recorded | not recorded | not recorded | not recorded |")
            for stratum in strata:
                diagnostic = stratum.get("diagnostics") or {}
                lines.append(f"| {row['id']} | {method} | {_fmt(stratum.get('z_over_H'))} | {_fmt(stratum.get('point_count'))} | {_fmt((diagnostic.get('normalized_rms') or {}).get('interior'))} | {_fmt((diagnostic.get('physical_rms') or {}).get('interior'))} |")
    lines += ["", "## Provenance and scope", "",
        f"Initialization policy: {comparison.get('initialization_policy') or 'not recorded'}.", "",
        f"Selection boundary: {comparison.get('selection_boundary') or 'not recorded'}.", "",
        "The fixed rule starts from the original initial weights. Probe checkpoints are not campaign warm starts. Four reused datasets do not establish independent replication, inverse recovery, reserved-cohort performance, hardware validity, or 3D validity. The reporter opens none of those stages.", "",
        f"Campaign plan SHA256: `{comparison['plan_sha256']}`. Optimizer decision SHA256: `{comparison['optimizer_decision']['sha256']}`.", "",
        "[Complete saved summary and trajectory identities](result-summary.json).", "",
        "![Saved primary metrics](comparison.png)", "", "![Saved training objectives](trajectories.png)", "",
        "![Saved Adam moment step counters](optimizer-history.png)", ""]
    return "\n".join(lines)


def _plots(output, comparison):
    with tempfile.TemporaryDirectory(prefix="thermotwin-optimized-plots-") as cache:
        os.environ.setdefault("MPLCONFIGDIR", cache)
        os.environ.setdefault("XDG_CACHE_HOME", cache)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        labels, colors, rows = _labels(comparison), ("#2563eb", "#c2410c"), comparison["rows"]
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
        panels = (("PDE normalized RMS", lambda s: _residuals(s).get("interior"), .05),
                  ("Bottom contact normalized RMS", lambda s: _residuals(s).get("bottom_contact"), .05),
                  ("Bottom insulation normalized RMS", lambda s: _residuals(s).get("bottom_insulated"), .05),
                  ("Field RMS error (K)", lambda s: (s.get("field_metrics") or {}).get("rmse_K"), .05),
                  ("Field maximum error (K)", lambda s: (s.get("field_metrics") or {}).get("maximum_error_K"), .15),
                  ("Observation normalized MSE", lambda s: s.get("normalized_observation_mse"), 2.))
        def number(value):
            return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        for axis, (title, getter, threshold) in zip(axes.ravel(), panels):
            for j, method in enumerate(METHODS):
                values = [getter(row["methods"][method]["saved"]) for row in rows]
                positions = np.arange(4)+(-.19 if j == 0 else .19)
                axis.bar(positions, [v if number(v) else np.nan for v in values], width=.36, color=colors[j], label=labels[method])
                for position, value, row in zip(positions, values, rows):
                    if not number(value):
                        status = "n/a" if "insulation" in title and row["geometry"] == "full_contact" else row["methods"][method]["saved"].get("status", "missing")
                        axis.text(position, .02, status, transform=axis.get_xaxis_transform(), rotation=90, ha="center", fontsize=7, color=colors[j])
            axis.axhline(threshold, linestyle="--", color="#991b1b", linewidth=1.)
            axis.set(title=title, xticks=np.arange(4), xticklabels=[row["id"] for row in rows], ylim=(0, None))
            axis.set_xlim(-.65, len(rows)-.35)
            axis.grid(axis="y", alpha=.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("Saved forward metrics — coordinator gates reported separately")
        fig.savefig(output/"comparison.png", dpi=170); fig.savefig(output/"comparison.pdf"); plt.close(fig)
        for filename, moment in (("trajectories", False), ("optimizer-history", True)):
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
            for axis, row in zip(axes.ravel(), rows):
                for color, method in zip(colors, METHODS):
                    training = row["methods"][method]["training"]
                    values = []
                    for event in training.get("trajectory") or []:
                        value = event.get("objective_after")
                        if moment:
                            counts = event.get("optimizer_step_counts_after")
                            value = max(counts) if counts else 0 if counts == [] else None
                        if number(event.get("step")) and number(value) and (moment or value > 0):
                            values.append((event["step"], value))
                    if not moment:
                        initial = training.get("initial_projection_objective")
                        if number(initial) and initial > 0:
                            values.insert(0, (0, initial))
                    if values:
                        axis.plot(*zip(*values), color=color, label=labels[method],
                                  marker="o" if len(values) == 1 else None, markersize=3)
                    else:
                        axis.text(.02, .95 if method == "baseline" else .86, f"{method}: no saved {'moment counters' if moment else 'trajectory'}", transform=axis.transAxes, color=color, va="top", fontsize=8)
                axis.set_title(f"{row['id']}: {row['geometry']}, h={row['known_h']:g}", fontsize=10)
                axis.set_xlabel("Fixed hidden update")
                axis.set_ylabel("Saved Adam step counter" if moment else "Saved complete training objective")
                if not moment:
                    axis.set_yscale("log")
                axis.grid(alpha=.2)
            axes[0, 0].legend(fontsize=8)
            fig.suptitle("Saved optimizer state counters" if moment else "Saved training trajectories — budget endpoints")
            fig.savefig(output/(filename+".png"), dpi=170); fig.savefig(output/(filename+".pdf")); plt.close(fig)
    return [name+extension for name in ("comparison", "trajectories", "optimizer-history") for extension in (".png", ".pdf")]


def render_report(campaign_directory, *, output_directory):
    source, output = Path(campaign_directory).resolve(), Path(output_directory).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("report output must be separate from campaign evidence")
    for ancestor in (output, *output.parents):
        if any((ancestor/name).is_file() for name in ("artifact-index.json", "evidence-index.json", "report-index.json")):
            raise ValueError("report output must be separate from sealed evidence")
    comparison = load_comparison(source)
    for protected in comparison["protected_directories"]:
        root = Path(protected).resolve()
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("report output must be separate from protected evidence")
    output.mkdir(parents=True, exist_ok=False)
    _write(output/"result-summary.json", comparison)
    (output/"REPORT.md").write_text(_markdown(comparison))
    names = ["REPORT.md", "result-summary.json"]+_plots(output, comparison)
    records = [{"path": name, "bytes": (output/name).stat().st_size, "sha256": digest_file(output/name)} for name in names]
    index = {"files": records, "digest": canonical_digest(records), "renderer_sha256": digest_file(__file__),
             "plan_sha256": comparison["plan_sha256"], "campaign_evidence": comparison["campaign_evidence"],
             "baseline_evidence": comparison["baseline_evidence"]}
    _write(output/"report-index.json", index)
    return {"directory": str(output), "files": len(records), "digest": index["digest"],
            "index_sha256": digest_file(output/"report-index.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(render_report(args.campaign_directory, output_directory=args.output_directory), allow_nan=False))


if __name__ == "__main__":
    main()
