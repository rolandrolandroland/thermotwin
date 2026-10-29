"""Render sealed forward evidence, including the interrupted optimizer attempt.

No fitting, residual evaluation, gate calculation, checkpoint selection, inverse
study, or reserved-cohort operation is available in this renderer.
"""
import argparse
import json
import math
import os
from pathlib import Path
import tempfile

from .pinn_spatial_continuity import canonical_digest, digest_file
from .pinn_spatial_graded import _by_id, _fmt, _gate, _optional, _protected_directories, _residuals
from .pinn_spatial_optimized import _training_summary


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("baseline", "prior_attempt", "controlled")


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def _validate_step_control(directory):
    from ..studies.pinn_spatial_step_control import validate_step_control
    return validate_step_control(directory)


def _validate_graded(directory):
    from ..studies.pinn_spatial_graded import validate_comparison
    return validate_comparison(directory)


def _validate_prior(directory):
    from ..studies.pinn_spatial_optimized import validate_optimized
    return validate_optimized(directory)


def _check_sources(records):
    for name, expected in records.items():
        path = (ROOT/name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(ROOT.resolve()) or not path.is_file() or digest_file(path) != expected:
            raise ValueError("declared source changed: "+name)


def _bound_path(name):
    path = Path(name)
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def _probe_evidence(bound):
    records = []
    for saved in bound["decision"].get("probe_evidence") or []:
        path = _bound_path(saved["path"])
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or digest_file(path) != saved["sha256"]:
            raise ValueError("bound repair probe evidence changed")
        record = dict(saved)
        if path.suffix == ".json":
            value = _read(path)
            if isinstance(value, dict):
                record.update({key: value.get(key) for key in ("status", "wall_seconds", "peak_rss_bytes")})
        records.append(record)
    return records


def _training(raw):
    result = _training_summary(raw)
    for name in ("controller_history", "final_controller_state", "completed_updates", "requested_updates",
                 "stop_reason", "termination", "trial_solve_count", "maximum_trial_solves",
                 "additional_initial_final_solves", "total_projection_solve_calls",
                 "controller_moment_restarts", "policy_moment_resets"):
        result[name] = raw.get(name)
    for summary, event in zip(result["trajectory"] or [], raw.get("hidden_update_history") or []):
        for name in ("controller_before", "controller_after", "controller_transitions", "controller_moments_restarted",
                     "controller_optimizer_step_counts_after", "controller_stop", "controller_stop_reason",
                     "trial_solve_count", "next_learning_rate"):
            summary[name] = event.get(name)
    return result


def _progress_provenance(path):
    if not path.is_file():
        return {"status": "not_recorded"}
    rows, partial = [], False
    for line in path.read_text().splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            partial = True
            continue
        if value.get("stage") in ("transactional_profiled_adam", "controlled_transactional_profiled_adam"):
            rows.append(value)
    last = rows[-1] if rows else {}
    return {"status": "saved_progress_only", "sha256": digest_file(path), "bytes": path.stat().st_size,
        "recorded_update_events": len(rows), "incomplete_json_line_present": partial,
        "last_update": {key: last.get(key) for key in ("step", "accepted", "backtracks", "objective_before",
            "objective_after", "gradient_norm_before_clip", "optimizer_step_counts_after", "controller_after")},
        "role": "partial training provenance; not a completed fit or primary gate"}


def _case_evidence(base, case_id, saved):
    directory = base/case_id
    worker_path = directory/"result.json"
    if worker_path.is_file() and _read(worker_path) != saved:
        raise ValueError("saved aggregate row differs from its worker result: "+case_id)
    raw_path = directory/"training-report.json"
    raw = _read(raw_path) if raw_path.is_file() else {}
    return {"saved": saved, "training": _training(raw),
        "training_report_sha256": digest_file(raw_path) if raw_path.is_file() else None,
        "worker_result_sha256": digest_file(worker_path) if worker_path.is_file() else None,
        "progress_provenance": _progress_provenance(directory/"progress.jsonl"),
        "near_interface_audit": _optional(directory/"near-interface-audit.json"),
        "bottom_layer_audit": _optional(directory/"near-bottom-audit.json")}


def load_comparison(campaign_directory):
    directory = Path(campaign_directory).resolve()
    if _read(directory/"state.json").get("status") != "completed":
        raise ValueError("controlled campaign has not completed; outcomes remain unopened")
    plan = _read(directory/"plan.json")
    provenance = plan["step_control_provenance"]
    baseline = Path(provenance["graded_comparison"]["directory"]).resolve()
    prior = Path(provenance["prior_optimized_campaign"]["directory"]).resolve()
    for old in (baseline, prior):
        if _read(old/"state.json").get("status") != "completed":
            raise ValueError("a bound prior campaign has not completed")
    evidence = {"controlled": _validate_step_control(directory), "baseline": _validate_graded(baseline),
                "prior_attempt": _validate_prior(prior)}
    for name, old in (("graded_comparison", baseline), ("prior_optimized_campaign", prior)):
        method = "baseline" if name == "graded_comparison" else "prior_attempt"
        bound = provenance[name]
        if evidence[method] != bound["context"] or digest_file(old/"plan.json") != bound["plan_sha256"]:
            raise ValueError("bound prior campaign evidence differs: "+name)
    plans = {"controlled": plan, "baseline": _read(baseline/"plan.json"), "prior_attempt": _read(prior/"plan.json")}
    results = {"controlled": _read(directory/"result.json"),
        "baseline": _read(baseline/"log_off"/"final-forward-selection.json"),
        "prior_attempt": _read(prior/"final-forward-selection.json")}
    for value in plans.values():
        _check_sources(value.get("source_sha256") or {})
    declared = _by_id(plan["cases"], "controlled plan")
    mapped = {method: _by_id(value.get("cases") or [], method+" saved outcomes") for method, value in results.items()}
    declarations = {method: _by_id(value.get("cases") or [], method+" plan") for method, value in plans.items()}
    paired = _by_id(_read(baseline/"result.json").get("paired_case_rows") or [], "baseline paired outcomes")
    if len(declared) != 4 or any(set(value) != set(declared) for value in (*mapped.values(), *declarations.values(), paired)):
        raise ValueError("all four declared datasets must remain in every attempt denominator")
    if results["controlled"].get("source_sha256") != plan.get("source_sha256"):
        raise ValueError("controlled result source proof differs from its plan")
    rows = []
    for case_id, case in declared.items():
        if paired[case_id]["methods"]["log_off"] != mapped["baseline"][case_id]:
            raise ValueError("baseline outcome differs from its sealed paired result")
        methods = {}
        for method, base in (("baseline", baseline/"log_off"), ("prior_attempt", prior), ("controlled", directory)):
            saved = mapped[method][case_id]
            if any(saved.get(key) != case[key] or declarations[method][case_id].get(key) != case[key]
                   for key in ("geometry", "known_h")):
                raise ValueError("paired physical case changed: "+case_id)
            methods[method] = _case_evidence(base, case_id, saved)
        rows.append(dict(case, methods=methods))
    probes = _probe_evidence(plan["decision"])
    execution = _optional(directory/"execution.json")
    protected = {str(directory), str(baseline), str(prior)}
    for value in plans.values():
        protected |= _protected_directories(value)
    for value in probes:
        path = _bound_path(value["path"])
        if (path.parent/"plan.json").is_file() and (path.parent/"state.json").is_file():
            protected.add(str(path.parent))
    return {"schema": "bounded_step_controller_saved_evidence_report", "development_only": True,
        "campaign_status": results["controlled"].get("status"), "saved_forward_status": results["controlled"].get("forward_status"),
        "plan_sha256": digest_file(directory/"plan.json"),
        "campaign_evidence": evidence["controlled"], "baseline_evidence": evidence["baseline"],
        "prior_attempt_evidence": evidence["prior_attempt"],
        "source_sha256": plan.get("source_sha256"), "source_unchanged": results["controlled"].get("source_unchanged"),
        "denominator": {"declared_cases_per_attempt": 4, "planned_case_attempts": 12,
            "comparison_case_attempts": 8, "prior_failed_attempt_cases_retained": 4,
            "reused_paired_datasets": 4, "independent_noise_replications_claimed": False},
        "configs": {"baseline": plans["baseline"]["model_configs"]["log_off"],
                    "prior_attempt": plans["prior_attempt"]["model_config"], "controlled": plan["model_config"]},
        "campaign_cost": {method: {key: value.get(key) for key in ("wall_seconds", "peak_rss_bytes", "peak_worker_rss_bytes", "status", "error", "reason")}
            for method, value in results.items()},
        "controlled_execution": execution,
        "worker_topology": plan.get("topology"), "repair_decision": plan["decision"], "probe_evidence": probes,
        "initialization_policy": plan.get("initialization_policy"), "selection_boundary": plan.get("selection_boundary"),
        "saved_scope_flags": {key: results["controlled"].get(key) for key in ("inverse_opened", "fresh_evaluation_opened", "three_d_opened")},
        "protected_directories": sorted(protected), "rows": rows}


def _labels(comparison):
    configs = comparison["configs"]
    return {"baseline": f"Original log-off ({configs['baseline']['adam_epochs']} max updates)",
        "prior_attempt": f"Prior V6 attempt ({configs['prior_attempt']['adam_epochs']} max updates)",
        "controlled": f"Step controller ({configs['controlled']['adam_epochs']} max updates)"}


def _markdown(comparison):
    config, labels = comparison["configs"]["controlled"], _labels(comparison)
    lines = ["# Bounded step controller: saved 2D forward evidence", "",
        f"Saved execution status: **{comparison['campaign_status']}**. Saved aggregate forward status: **{comparison.get('saved_forward_status') or 'not recorded'}**.", "",
        "Four reused development datasets remain in every denominator. The original log-off baseline, all four outcomes of the failed V6 attempt, and the new controlled run are retained. Failed, interrupted and unrun cases do not disappear. These are fresh initializations on opened observations, not independent replications.", "",
        "Every primary decision below is copied from the saved coordinator. This renderer does not fit, recalculate a gate, choose a checkpoint, or certify convergence. Controller early stops describe bounded training, not scientific acceptance.", "",
        f"The controlled rule starts at learning rate {_fmt(config.get('learning_rate'))}, allows at most {_fmt(config.get('adam_epochs'))} hidden attempts and {_fmt(config.get('maximum_trial_solves'))} trial readout solves plus two initial/final solves. Failed transactions roll back before the separate controller changes learning rate and restarts moments. The saved plan binds all controller constants and worker caps.", "",
        "Unchanged primary references: observation MSE 2; applicable physics RMS 0.05; field RMS 0.05 K and maximum 0.15 K. Saved gates govern outcomes.", "",
        "## Saved primary outcomes", "",
        "| Case | Method | Saved status | Saved gate | PDE RMS | Bottom contact | Bottom insulation | Field RMS K | Field max K | Obs MSE |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in comparison["rows"]:
        for method in METHODS:
            saved = row["methods"][method]["saved"]
            residual, field = _residuals(saved), saved.get("field_metrics") or {}
            insulated = "not applicable" if row["geometry"] == "full_contact" else _fmt(residual.get("bottom_insulated"))
            lines.append(f"| {row['id']} | {labels[method]} | {saved.get('status', 'not recorded')} | {_gate((saved.get('gate') or {}).get('all'))} | {_fmt(residual.get('interior'))} | {_fmt(residual.get('bottom_contact'))} | {insulated} | {_fmt(field.get('rmse_K'))} | {_fmt(field.get('maximum_error_K'))} | {_fmt(saved.get('normalized_observation_mse'))} |")
            if saved.get("error") or saved.get("reason"):
                lines += ["", f"{row['id']} {method}: {saved.get('error') or saved.get('reason')}", ""]
    lines += ["", "## Controller and training endpoints", "",
        "No endpoint is labeled convergence. A projection-budget stop can occur between attempted updates, after the last recorded controller event. Partial progress from a resource-limited fit is retained as provenance and never substituted for a completed fit.", "",
        "| Case | Method | Accepted / recorded | Rejected | Backtracks | Final objective | Final rank / columns | Condition | Coefficient norm |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |"]
    for row in comparison["rows"]:
        for method in METHODS:
            value = row["methods"][method]
            training, final = value["training"], value["training"]["final_projection"]
            lines.append(f"| {row['id']} | {method} | {_fmt(training['hidden_accepted_updates'])} / {_fmt(training['hidden_update_count'])} | {_fmt(training['hidden_rejected_updates'])} | {_fmt(training['hidden_backtrack_count'])} | {_fmt(training['final_objective'])} | {_fmt(final['rank'])} / {_fmt(final['columns'])} | {_fmt(final['resolved_condition'])} | {_fmt(final['coefficient_norm_returned'])} |")
            if value["progress_provenance"].get("recorded_update_events") is not None and training["hidden_update_count"] is None:
                lines += ["", f"{row['id']} {method}: {value['progress_provenance']['recorded_update_events']} saved partial progress events; no completed training report.", ""]
    lines += ["", "| Case | Actual / maximum attempts | Saved termination | Trial solves / cap | Extra initial/final solves | Moment restarts | Final LR | Rejection streak |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for row in comparison["rows"]:
        training = row["methods"]["controlled"]["training"]
        state = training.get("final_controller_state") or {}
        lines.append(f"| {row['id']} | {_fmt(training.get('completed_updates'))} / {_fmt(training.get('requested_updates'))} | {training.get('stop_reason') or 'not recorded'} | {_fmt(training.get('trial_solve_count'))} / {_fmt(training.get('maximum_trial_solves'))} | {_fmt(training.get('additional_initial_final_solves'))} | {_fmt(training.get('controller_moment_restarts'))} | {_fmt(state.get('learning_rate'))} | {_fmt(state.get('consecutive_failures'))} |")
    lines += ["", "## Continuity, boundary and report-only diagnostics", "",
        "Exact shared traces do not establish heat-equation accuracy. These normalized RMS values and supplemental audit outcomes are copied without changing their acceptance role.", "",
        "| Case | Method | Top | Left | Right | Interface 0 T / flux | Interface 1 T / flux | Edge audit PDE | Bottom-layer audit PDE |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: |"]
    for row in comparison["rows"]:
        for method in METHODS:
            value = row["methods"][method]
            residual = _residuals(value["saved"])
            interfaces = ["not applicable" if row["geometry"] == "full_contact" else
                f"{_fmt(residual.get(f'interface_{j}_temperature'))} / {_fmt(residual.get(f'interface_{j}_flux'))}" for j in (0, 1)]
            audits = [((value[key].get("diagnostics") or {}).get("normalized_rms") or {}).get("interior")
                      for key in ("near_interface_audit", "bottom_layer_audit")]
            lines.append(f"| {row['id']} | {method} | {_fmt(residual.get('top'))} | {_fmt(residual.get('left'))} | {_fmt(residual.get('right'))} | {interfaces[0]} | {interfaces[1]} | {_fmt(audits[0])} | {_fmt(audits[1])} |")
    lines += ["", "## Saved computation and failed-attempt provenance", "",
        "The V6 time/resource failure remains part of the record. Training times, per-worker RSS and campaign elapsed wall time are distinct quantities; parallel worker times are not added to elapsed time.", "",
        "| Attempt | Saved status | Campaign elapsed s | Saved process / worker peak RSS bytes |", "| --- | --- | ---: | ---: |"]
    for method in METHODS:
        cost = comparison["campaign_cost"][method]
        wall = (comparison.get("controlled_execution") or {}).get("wall_seconds") if method == "controlled" else cost.get("wall_seconds")
        rss = cost.get("peak_rss_bytes") if cost.get("peak_rss_bytes") is not None else cost.get("peak_worker_rss_bytes")
        lines.append(f"| {labels[method]} | {cost.get('status') or 'not recorded'} | {_fmt(wall)} | {_fmt(rss)} |")
        if cost.get("error") or cost.get("reason"):
            lines += ["", str(cost.get("error") or cost.get("reason")), ""]
    lines += ["", "| Controlled case | Saved worker status | Worker elapsed s | Worker peak RSS bytes |", "| --- | --- | ---: | ---: |"]
    for row in comparison["rows"]:
        saved = row["methods"]["controlled"]["saved"]
        lines.append(f"| {row['id']} | {saved.get('status') or 'not recorded'} | {_fmt(saved.get('wall_seconds'))} | {_fmt(saved.get('peak_rss_bytes'))} |")
    lines += ["", "Worker topology and caps:", "", "```json", json.dumps(comparison.get("worker_topology"), indent=2, allow_nan=False), "```", "",
        "Training-only repair probe costs:", "", "| Saved record | Status | Elapsed s | Peak RSS bytes |", "| --- | --- | ---: | ---: |"]
    for record in comparison["probe_evidence"]:
        lines.append(f"| {record['path']} | {record.get('status') or 'not recorded'} | {_fmt(record.get('wall_seconds'))} | {_fmt(record.get('peak_rss_bytes'))} |")
    lines += ["", "## Scope and provenance", "",
        f"Initialization: {comparison.get('initialization_policy') or 'not recorded'}.", "",
        f"Selection boundary: {comparison.get('selection_boundary') or 'not recorded'}.", "",
        "All numerical controls and saved observations remain the declared log-off graded problem. No independent replication, inverse recovery, reserved-cohort evaluation, hardware validation or 3D validity is claimed. Missing results remain missing.", "",
        f"Plan SHA256: `{comparison['plan_sha256']}`. Repair decision SHA256: `{comparison['repair_decision']['sha256']}`.", "",
        "[Complete saved rows, controller transitions, costs and evidence identities](result-summary.json).", "",
        "![Saved primary metrics](comparison.png)", "", "![Saved training trajectories](trajectories.png)", "",
        "![Saved controller learning rates and restarts](controller-history.png)", ""]
    return "\n".join(lines)


def _plots(output, comparison):
    with tempfile.TemporaryDirectory(prefix="thermotwin-step-control-plots-") as cache:
        os.environ.setdefault("MPLCONFIGDIR", cache)
        os.environ.setdefault("XDG_CACHE_HOME", cache)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        rows, colors, labels = comparison["rows"], ("#2563eb", "#6b7280", "#c2410c"), _labels(comparison)
        def number(value):
            return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
        panels = (("PDE normalized RMS", lambda s: _residuals(s).get("interior"), .05),
            ("Bottom contact normalized RMS", lambda s: _residuals(s).get("bottom_contact"), .05),
            ("Bottom insulation normalized RMS", lambda s: _residuals(s).get("bottom_insulated"), .05),
            ("Field RMS error (K)", lambda s: (s.get("field_metrics") or {}).get("rmse_K"), .05),
            ("Field maximum error (K)", lambda s: (s.get("field_metrics") or {}).get("maximum_error_K"), .15),
            ("Observation normalized MSE", lambda s: s.get("normalized_observation_mse"), 2.))
        for axis, (title, getter, limit) in zip(axes.ravel(), panels):
            for j, method in enumerate(METHODS):
                positions = np.arange(4)+(j-1)*.25
                values = [getter(row["methods"][method]["saved"]) for row in rows]
                axis.bar(positions, [value if number(value) else np.nan for value in values], width=.23,
                         color=colors[j], label=labels[method])
                for x, value, row in zip(positions, values, rows):
                    if not number(value):
                        status = "n/a" if "insulation" in title and row["geometry"] == "full_contact" else row["methods"][method]["saved"].get("status", "missing")
                        axis.text(x, .02, status, transform=axis.get_xaxis_transform(), rotation=90, ha="center", fontsize=6, color=colors[j])
            axis.axhline(limit, linestyle="--", color="#991b1b", linewidth=1.)
            axis.set(title=title, xticks=np.arange(4), xticklabels=[row["id"] for row in rows], ylim=(0, None), xlim=(-.6, 3.6))
            axis.grid(axis="y", alpha=.2)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle("Saved metrics — failed and unrun attempts retained")
        fig.savefig(output/"comparison.png", dpi=170); fig.savefig(output/"comparison.pdf"); plt.close(fig)
        for filename in ("trajectories", "controller-history"):
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
            for axis, row in zip(axes.ravel(), rows):
                if filename == "trajectories":
                    for method, color in zip(METHODS, colors):
                        training = row["methods"][method]["training"]
                        points = [(event["step"], event["objective_after"]) for event in training.get("trajectory") or []
                                  if number(event.get("step")) and number(event.get("objective_after")) and event["objective_after"] > 0]
                        initial = training.get("initial_projection_objective")
                        if number(initial) and initial > 0:
                            points.insert(0, (0, initial))
                        if points:
                            axis.plot(*zip(*points), color=color, label=labels[method], marker="o" if len(points) == 1 else None, markersize=3)
                        else:
                            axis.text(.02, .96-.07*METHODS.index(method), method+": no completed training report", transform=axis.transAxes, color=color, va="top", fontsize=8)
                    axis.set_ylabel("Saved full training objective")
                else:
                    training = row["methods"]["controlled"]["training"]
                    history = training.get("controller_history") or []
                    points = [(event["step"], event["after"]["learning_rate"]) for event in history
                              if number(event.get("step")) and number((event.get("after") or {}).get("learning_rate"))]
                    if points:
                        axis.step(*zip(*points), where="post", color=colors[2], label="Carried learning rate")
                        restarts = [(event["step"], event["after"]["learning_rate"]) for event in history
                                    if event["after"].get("moment_restarts", 0) > event["before"].get("moment_restarts", 0)]
                        if restarts:
                            axis.scatter(*zip(*restarts), marker="x", color="#991b1b", label="Moment restart")
                    else:
                        axis.text(.02, .95, "No saved controller trajectory", transform=axis.transAxes, va="top")
                    axis.set_ylabel("Saved carried learning rate")
                    axis.text(.02, .02, training.get("stop_reason") or "not recorded", transform=axis.transAxes, fontsize=8)
                axis.set(title=f"{row['id']}: {row['geometry']}, h={row['known_h']:g}", xlabel="Attempted hidden update", yscale="log")
                axis.grid(alpha=.2)
                if axis.get_legend_handles_labels()[0]:
                    axis.legend(fontsize=7)
            fig.suptitle("Saved controlled learning-rate transitions" if filename == "controller-history" else "Saved completed training trajectories")
            fig.savefig(output/(filename+".png"), dpi=170); fig.savefig(output/(filename+".pdf")); plt.close(fig)
    return [name+extension for name in ("comparison", "trajectories", "controller-history") for extension in (".png", ".pdf")]


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
        "baseline_evidence": comparison["baseline_evidence"], "prior_attempt_evidence": comparison["prior_attempt_evidence"]}
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
