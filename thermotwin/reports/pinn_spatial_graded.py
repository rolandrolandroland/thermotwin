"""Saved-evidence report for the paired graded-grid log-coordinate comparison.

The reporter reads coordinator gates.  It performs no numerical fitting,
scientific acceptance calculation, checkpoint selection, or winner selection.
"""
import argparse
import json
import math
import os
from pathlib import Path
import tempfile

from .pinn_spatial_continuity import (
    canonical_digest, digest_file, validate_campaign,
)


ROOT = Path(__file__).resolve().parents[2]
ARMS = ("log_on", "log_off")


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _fmt(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "not recorded"
    return f"{value:.6g}"


def _gate(value):
    return "pass" if value is True else "fail" if value is False else "not recorded"


def _coordinator_validation(directory, plan):
    # Imported only after the top completion flag.  Both functions are
    # read-only preservation/source validators, not numerical executors.
    from ..studies.pinn_spatial_graded import (
        validate_comparison, verify_comparison_plan,
    )
    verify_comparison_plan(plan)
    return validate_comparison(directory)


def _check_sources(records):
    for name, expected in records.items():
        path = (ROOT / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            raise ValueError("invalid declared source path: " + name)
        if digest_file(path) != expected:
            raise ValueError("declared source changed: " + name)


def _by_id(rows, label):
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate case identifier in " + label)
    return result


def _residuals(saved, kind="normalized_rms"):
    return (saved.get("independent_residuals") or {}).get(kind) or {}


def _training(raw):
    projections = raw.get("projections") or []
    final = projections[-1] if projections else {}
    updates = raw.get("hidden_update_history")
    point_metadata = raw.get("graded_point_metadata") or {}
    trajectory = None if updates is None else [{
        key: event.get(key) for key in (
            "step", "objective_before", "objective_after", "accepted",
            "accepted_scale", "backtracks", "gradient_norm_before_clip",
        )
    } for event in updates]
    if updates is not None:
        for summary, event in zip(trajectory, updates):
            summary["trials"] = []
            for trial in event.get("trials") or []:
                record = {key: trial.get(key) for key in (
                    "backtracks", "scale", "objective", "accepted", "rank",
                    "resolved_condition", "rank_rcond", "reason", "error_type", "error",
                )}
                projection = trial.get("projection") or {}
                record["projection"] = {key: projection.get(key) for key in (
                    "rank", "columns", "rank_rcond", "resolved_condition",
                    "coefficient_norm_returned", "objective_before", "objective_after", "accepted",
                )}
                summary["trials"].append(record)
    return {
        "wall_seconds": raw.get("wall_seconds"),
        "final_objective": raw.get("final_objective"),
        "initial_projection_objective": projections[0].get("objective_after") if projections else None,
        "coefficient_count": raw.get("coefficient_count"),
        "final_projection": {key: final.get(key) for key in (
            "rank", "columns", "rows", "rank_rcond", "resolved_condition",
            "smallest_singular_value", "coefficient_norm_returned", "accepted",
        )},
        "projection_count": len(projections) if "projections" in raw else None,
        "hidden_update_count": len(updates) if updates is not None else None,
        "hidden_accepted_updates": sum(event.get("accepted") is True for event in updates) if updates is not None else None,
        "hidden_rejected_updates": raw.get("hidden_rejected_updates"),
        "hidden_backtrack_count": raw.get("hidden_backtrack_count"),
        "hidden_accepted_objective_increases": raw.get("hidden_accepted_objective_increases"),
        "optimizer_resets": raw.get("optimizer_resets"),
        "initialization": raw.get("initialization"),
        "training_point_digest": raw.get("training_point_digest", point_metadata.get("point_digest")),
        "training_point_counts": raw.get("training_point_counts", point_metadata.get("point_counts")),
        "training_strata": raw.get("training_strata", point_metadata.get("strata")),
        "graded_point_metadata": point_metadata or None,
        "trajectory": trajectory,
        "gradient_semantics": raw.get("gradient_semantics"),
    }


def _optional(path):
    return _read(path) if path.is_file() else {"status": "not_recorded", "role": "report only"}


def _protected_directories(value):
    result = set()
    if isinstance(value, dict):
        candidate = value.get("directory")
        if isinstance(candidate, str) and Path(candidate).is_absolute():
            result.add(str(Path(candidate).resolve()))
        for key in ("preserved_campaign", "paired_v2_directory"):
            candidate = value.get(key)
            if isinstance(candidate, str) and Path(candidate).is_absolute():
                result.add(str(Path(candidate).resolve()))
        for child in value.values():
            result.update(_protected_directories(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_protected_directories(child))
    return result


def load_comparison(comparison_directory):
    """Load all declared rows only after both arms have been sealed."""
    directory = Path(comparison_directory).resolve()
    state = _read(directory / "state.json")
    if state.get("status") != "completed":
        raise ValueError("comparison has not completed; outcomes are not opened")
    plan = _read(directory / "plan.json")
    result = _read(directory / "result.json")
    _check_sources(plan.get("source_sha256") or {})
    evidence = _coordinator_validation(directory, plan)
    declared = _by_id(plan["cases"], "top plan")
    if len(declared) != 4:
        raise ValueError("comparison must retain all four declared paired datasets")
    top_rows = _by_id(result["paired_case_rows"], "combined result")
    if set(top_rows) != set(declared):
        raise ValueError("combined result differs from the declared case denominator")
    campaigns, saved = {}, {}
    protected = _protected_directories(plan)
    for arm in ARMS:
        arm_directory = directory / arm
        proof = validate_campaign(arm_directory)
        subplan = _read(arm_directory / "plan.json")
        protected.update(_protected_directories(subplan))
        _check_sources(subplan.get("source_sha256") or {})
        record = plan["subplans"][arm]
        if record["relative_directory"] != arm or record["plan_sha256"] != proof["plan_sha256"]:
            raise ValueError("subcampaign plan is not bound to the top plan: " + arm)
        final = _read(arm_directory / "final-forward-selection.json")
        if final.get("source_sha256") != subplan.get("source_sha256"):
            raise ValueError("arm source proof differs from its plan: " + arm)
        if result["arms"][arm]["artifact_index_sha256"] != proof["artifact_index_sha256"]:
            raise ValueError("combined result refers to a different arm index: " + arm)
        saved[arm] = _by_id(final["cases"], arm + " result")
        if set(saved[arm]) != set(declared):
            raise ValueError("arm result drops or adds a declared case: " + arm)
        campaigns[arm] = {
            "directory": str(arm_directory), "evidence": proof,
            "status": final.get("status"), "source_sha256": final.get("source_sha256"),
            "source_unchanged": final.get("source_unchanged"),
            "wall_seconds": final.get("wall_seconds"), "peak_rss_bytes": final.get("peak_rss_bytes"),
            "model_config": subplan.get("model_config"),
            "inverse_opened": final.get("inverse_opened"),
            "fresh_evaluation_opened": final.get("fresh_evaluation_opened"),
        }
    rows = []
    for case_id, case in declared.items():
        top = top_rows[case_id]
        methods = {}
        for arm in ARMS:
            entry = saved[arm][case_id]
            if any(entry.get(key) != case[key] or top.get(key) != case[key]
                   for key in ("geometry", "known_h")):
                raise ValueError("paired geometry or conductance differs for " + case_id)
            if top["methods"][arm] != entry:
                raise ValueError("combined saved row differs from its sealed arm: " + case_id)
            case_directory = directory / arm / case_id
            training_path = case_directory / "training-report.json"
            training = _read(training_path) if training_path.is_file() else {}
            methods[arm] = {
                "saved": entry, "training": _training(training),
                "training_report_sha256": digest_file(training_path) if training_path.is_file() else None,
                "near_interface_audit": _optional(case_directory / "near-interface-audit.json"),
                "bottom_layer_audit": _optional(case_directory / "near-bottom-audit.json"),
            }
        rows.append(dict(case, methods=methods, pairing=top.get("pairing")))
    return {
        "schema": "graded_log_saved_evidence_report", "development_only": True,
        "campaign_status": result.get("status"), "plan_sha256": digest_file(directory / "plan.json"),
        "comparison_evidence": evidence, "source_sha256": plan.get("source_sha256"),
        "denominator": {"method_fits": 8, "reused_paired_datasets": 4,
                        "independent_noise_replications_claimed": False},
        "saved_no_winner_selected": result.get("no_winner_selected"),
        "protected_directories": sorted(protected),
        "campaigns": campaigns, "rows": rows,
        "saved_top_result": result,
    }


def _markdown(comparison):
    lines = ["# Graded-grid log-coordinate comparison", "",
        f"Saved coordinator status: **{comparison['campaign_status']}**.", "",
        "This development comparison contains eight declared method fits on four reused paired datasets. The reporter displays saved gate decisions; it does not recompute acceptance, choose a checkpoint, or name a winning arm. Missing metrics remain missing.", "",
        "The unchanged primary limits are observation MSE at most 2, every applicable normalized physics RMS at most 0.05, field RMS error at most 0.05 K, and field maximum error at most 0.15 K. The saved coordinator gate remains the source of each reported decision.", "",
        "The declared comparison matches graded PDE points, observed temperatures, and common initial weights. The log input column starts at zero in log-on and is omitted in log-off; saved pairing proofs are retained below. Each declared fit has a budget of 80 fixed variable-projection updates. These are budget endpoints, not certificates of convergence. Saved V4 uniform-grid results are prior development context, not a controlled sampling-only comparison.", "",
        "## Saved primary outcomes", "",
        "| Case | Geometry | Known h | Arm | Status | Saved overall gate | PDE RMS normalized | Field RMS K | Field max K | Observation MSE |",
        "| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for row in comparison["rows"]:
        for arm in ARMS:
            saved = row["methods"][arm]["saved"]
            field = saved.get("field_metrics") or {}
            lines.append(f"| {row['id']} | {row['geometry']} | {_fmt(row['known_h'])} | {arm} | {saved.get('status', 'not_recorded')} | {_gate((saved.get('gate') or {}).get('all'))} | {_fmt(_residuals(saved).get('interior'))} | {_fmt(field.get('rmse_K'))} | {_fmt(field.get('maximum_error_K'))} | {_fmt(saved.get('normalized_observation_mse'))} |")
            if saved.get("error") or saved.get("reason"):
                lines.extend(["", f"{row['id']} {arm}: {saved.get('error') or saved.get('reason')}", ""])
    lines += ["", "## Saved boundary and continuity diagnostics", "",
        "Every value below is a saved normalized RMS. Full-contact cases have no bottom-insulation or artificial-interface family. Exact continuity construction does not establish heat-equation accuracy.", "",
        "| Case | Arm | Bottom contact | Bottom insulated | Top | Left | Right | Interface 0 T | Interface 0 flux | Interface 1 T | Interface 1 flux |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in comparison["rows"]:
        for arm in ARMS:
            values = _residuals(row["methods"][arm]["saved"])
            keys = ("bottom_contact", "bottom_insulated", "top", "left", "right",
                    "interface_0_temperature", "interface_0_flux", "interface_1_temperature", "interface_1_flux")
            cells = ["not applicable" if row["geometry"] == "full_contact" and (key == "bottom_insulated" or key.startswith("interface_")) else _fmt(values.get(key)) for key in keys]
            lines.append(f"| {row['id']} | {arm} | " + " | ".join(cells) + " |")
    lines += ["", "## Solver trajectory, conditioning, and cost", "",
        "Ranks and coefficient norms refer to the returned final projection. Training times exclude subsequent independent scoring. Full supplied training trajectories and stratum weights are retained in result-summary.json. A fixed-readout proposal gradient is not asserted to be an exact profiled gradient under truncated SVD.", "",
        "| Case | Arm | Training s | Accepted / recorded updates | Backtracks | Final rank / columns | Resolved condition | Coefficient norm | Final objective |",
        "| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: |"]
    for row in comparison["rows"]:
        for arm in ARMS:
            train = row["methods"][arm]["training"]
            final = train["final_projection"]
            lines.append(f"| {row['id']} | {arm} | {_fmt(train['wall_seconds'])} | {_fmt(train['hidden_accepted_updates'])} / {_fmt(train['hidden_update_count'])} | {_fmt(train['hidden_backtrack_count'])} | {_fmt(final['rank'])} / {_fmt(final['columns'])} | {_fmt(final['resolved_condition'])} | {_fmt(final['coefficient_norm_returned'])} | {_fmt(train['final_objective'])} |")
    lines += ["", "| Arm | Campaign elapsed s | Peak process RSS bytes | Saved scientific status |", "| --- | ---: | ---: | --- |"]
    for arm in ARMS:
        campaign = comparison["campaigns"][arm]
        lines.append(f"| {arm} | {_fmt(campaign['wall_seconds'])} | {_fmt(campaign['peak_rss_bytes'])} | {campaign['status']} |")
    lines += ["", "Arms run in separate processes. Their individual elapsed times are not the combined wall time; summing them describes process time spent in the campaigns and is not a CPU-time measurement.", "",
        "## Separate supplemental diagnostics", "",
        "Near-interface and bottom-layer probes are report only and never replace primary acceptance. Bottom-layer strata, including probes away from contact edges and in full-contact cases, are shown only when saved by the coordinator. No missing supplemental result is interpreted as success.", "",
        "| Case | Arm | Near-interface status | Near-interface PDE normalized RMS | Bottom-layer status | Bottom-layer PDE normalized RMS |",
        "| --- | --- | --- | ---: | --- | ---: |"]
    for row in comparison["rows"]:
        for arm in ARMS:
            method = row["methods"][arm]
            edge = method["near_interface_audit"]
            pde = ((edge.get("diagnostics") or {}).get("normalized_rms") or {}).get("interior")
            bottom = method["bottom_layer_audit"]
            bottom_pde = ((bottom.get("diagnostics") or {}).get("normalized_rms") or {}).get("interior")
            lines.append(f"| {row['id']} | {arm} | {edge.get('status', 'not_recorded')} | {_fmt(pde)} | {bottom.get('status', 'not_recorded')} | {_fmt(bottom_pde)} |")
    lines += ["", "### Saved bottom-layer strata", "",
        "Physical PDE RMS uses the saved temperature-rate normalization, in K/s. Heights are fractions of plate thickness. These saved strata do not change a primary gate.", "",
        "| Case | Arm | z / H | Point count | PDE normalized RMS | PDE physical RMS K/s |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in comparison["rows"]:
        for arm in ARMS:
            strata = row["methods"][arm]["bottom_layer_audit"].get("by_height") or []
            if not strata:
                lines.append(f"| {row['id']} | {arm} | not recorded | not recorded | not recorded | not recorded |")
            for stratum in strata:
                diagnostic = stratum.get("diagnostics") or {}
                normalized = (diagnostic.get("normalized_rms") or {}).get("interior")
                physical = (diagnostic.get("physical_rms") or {}).get("interior")
                lines.append(f"| {row['id']} | {arm} | {_fmt(stratum.get('z_over_H'))} | {_fmt(stratum.get('point_count'))} | {_fmt(normalized)} | {_fmt(physical)} |")
    lines += ["", "## Pairing and provenance", "",
        "Pairing proofs are saved coordinator outputs. Failed and unrun fits remain in the eight-fit denominator. This comparison uses known contact conductances and previously opened development data; it establishes no inverse recovery, fresh-cohort performance, neural superiority, or 3D validity.", "",
        "| Case | Declared matched initial function | Declared identical graded grid | Executed log-on proof | Executed log-off proof |", "| --- | --- | --- | --- | --- |"]
    for row in comparison["rows"]:
        pairing = row.get("pairing") or {}
        executed = pairing.get("executed_proofs") or {}
        lines.append(f"| {row['id']} | {pairing.get('predeclared_matched_initialization', 'not recorded')} | {pairing.get('predeclared_identical_graded_grid', 'not recorded')} | {(executed.get('log_on') or {}).get('status', 'not recorded')} | {(executed.get('log_off') or {}).get('status', 'not recorded')} |")
    lines += ["",
        f"Top plan SHA256: `{comparison['plan_sha256']}`. Saved no-winner-selected flag: `{comparison['saved_no_winner_selected']}`.", "",
        "Full physical and normalized diagnostics, initialization and grid digests, source/index proofs, failure details, stratum weights, and training trajectories: [result-summary.json](result-summary.json).", "",
        "![Saved primary metrics](comparison.png)", "", "![Saved training objectives](trajectories.png)", ""]
    return "\n".join(lines)


def _plots(output, comparison):
    with tempfile.TemporaryDirectory(prefix="thermotwin-graded-plots-") as cache:
        os.environ.setdefault("MPLCONFIGDIR", cache)
        os.environ.setdefault("XDG_CACHE_HOME", cache)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        rows = comparison["rows"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        panels = (("PDE normalized RMS", lambda s: _residuals(s).get("interior"), .05),
                  ("Field RMS error (K)", lambda s: (s.get("field_metrics") or {}).get("rmse_K"), .05),
                  ("Observation normalized MSE", lambda s: s.get("normalized_observation_mse"), 2.),
                  ("Bottom contact normalized RMS", lambda s: _residuals(s).get("bottom_contact"), .05))
        colors = ("#2563eb", "#c2410c")
        for axis, (title, metric, reference) in zip(axes.ravel(), panels):
            for index, arm in enumerate(ARMS):
                values = [metric(row["methods"][arm]["saved"]) for row in rows]
                values = [value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else np.nan for value in values]
                positions = np.arange(4) + (-.19 if index == 0 else .19)
                axis.bar(positions, values, width=.36, color=colors[index], label=arm)
                for position, value, row in zip(positions, values, rows):
                    if not math.isfinite(value):
                        axis.text(position, .02, row["methods"][arm]["saved"].get("status", "not recorded"),
                                  transform=axis.get_xaxis_transform(), rotation=90,
                                  color=colors[index], ha="center", va="bottom", fontsize=7)
            axis.axhline(reference, color="#991b1b", linestyle="--", linewidth=1.)
            axis.set_title(title, fontsize=10)
            axis.set_xticks(np.arange(4), [row["id"] for row in rows])
            axis.set_xlim(-.65, len(rows) - .35)
            axis.set_ylim(bottom=0)
            axis.grid(axis="y", alpha=.2)
        axes[0, 0].legend()
        fig.suptitle("Saved development metrics — coordinator gates reported separately")
        fig.savefig(output / "comparison.png", dpi=170)
        fig.savefig(output / "comparison.pdf")
        plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for axis, row in zip(axes.ravel(), rows):
            for color, arm in zip(colors, ARMS):
                history = row["methods"][arm]["training"]["trajectory"] or []
                values = [(event["step"], event["objective_after"]) for event in history
                          if isinstance(event.get("step"), (int, float)) and isinstance(event.get("objective_after"), (int, float)) and event["objective_after"] > 0]
                initial = row["methods"][arm]["training"]["initial_projection_objective"]
                if isinstance(initial, (int, float)) and not isinstance(initial, bool) and math.isfinite(initial) and initial > 0:
                    values.insert(0, (0, initial))
                if values:
                    axis.plot(*zip(*values), color=color, label=arm,
                              marker="o" if len(values) == 1 else None, markersize=3)
                else:
                    axis.text(.02, .95 if arm == "log_on" else .85,
                              f"{arm}: no saved trajectory", color=color,
                              transform=axis.transAxes, va="top", fontsize=8)
            axis.set_title(f"{row['id']}: {row['geometry']}, h={row['known_h']:g}", fontsize=10)
            axis.set_yscale("log")
            axis.set_xlabel("Fixed hidden update")
            axis.set_ylabel("Saved complete training objective")
            axis.grid(alpha=.2)
        axes[0, 0].legend()
        fig.suptitle("Returned training trajectories — no convergence or winner claim")
        fig.savefig(output / "trajectories.png", dpi=170)
        fig.savefig(output / "trajectories.pdf")
        plt.close(fig)
    return ["comparison.png", "comparison.pdf", "trajectories.png", "trajectories.pdf"]


def render_report(comparison_directory, *, output_directory):
    source, output = Path(comparison_directory).resolve(), Path(output_directory).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("report output must be separate from campaign evidence")
    for ancestor in (output, *output.parents):
        if any((ancestor / name).is_file() for name in (
                "artifact-index.json", "evidence-index.json", "report-index.json")):
            raise ValueError("report output must be separate from sealed evidence")
    comparison = load_comparison(source)
    for protected in comparison["protected_directories"]:
        protected = Path(protected)
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("report output must be separate from protected campaign evidence")
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "result-summary.json", comparison)
    (output / "REPORT.md").write_text(_markdown(comparison))
    names = ["REPORT.md", "result-summary.json"] + _plots(output, comparison)
    records = [{"path": name, "bytes": (output / name).stat().st_size,
                "sha256": digest_file(output / name)} for name in names]
    index = {"files": records, "digest": canonical_digest(records),
             "comparison_evidence": comparison["comparison_evidence"],
             "plan_sha256": comparison["plan_sha256"],
             "renderer_sha256": digest_file(__file__)}
    _write(output / "report-index.json", index)
    return {"directory": str(output), "files": len(records),
            "digest": index["digest"], "index_sha256": digest_file(output / "report-index.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(render_report(args.comparison_directory,
                                  output_directory=args.output_directory), allow_nan=False))


if __name__ == "__main__":
    main()
