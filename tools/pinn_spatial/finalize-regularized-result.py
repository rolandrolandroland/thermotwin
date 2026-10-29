"""Finalize completed saved evidence only, after canonical full-suite QA passes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BASE = ROOT/"reports/pinn-spatial-validation"
CAMPAIGN = BASE/"regularized-v8-development"
VALIDATION = BASE/"regularized-validation"
REPORT = BASE/"regularized-v8-report"
FINALIZATION = BASE/"regularized-result-finalization"
TRACKED_PLAN = ROOT/"thermotwin/PINN_SPATIAL_REGULARIZED_PLAN.json"
TRACKED_RESULT = ROOT/"thermotwin/PINN_SPATIAL_REGULARIZED_RESULT.json"
REPORT_FILES = {"REPORT.md", "result-summary.json", "comparison.png", "comparison.pdf"}
REQUIRED_QA_SOURCES = {"thermotwin/reports/pinn_spatial_regularized.py",
    "tests/test_pinn_spatial_regularized_report.py", "tools/pinn_spatial/finalize-regularized-result.py",
    "tools/pinn_spatial/run-regularized-validation.py"}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def qa_evidence(validation, *, root=None):
    """Authenticate the terminal log, exact counts and current tested sources."""
    validation, root = Path(validation).resolve(), Path(root or ROOT).resolve()
    state_path, log_path = validation/"full-suite-state.json", validation/"full-suite.log"
    state = read(state_path)
    if (state.get("status") != "completed" or state.get("passed") is not True
        or state.get("source_unchanged") is not True or type(state.get("return_code")) is not int or state["return_code"] != 0):
        raise RuntimeError("full-suite QA must be completed, passed and source-unchanged")
    for key in ("failures", "errors", "expected_failures", "unexpected_successes", "skipped"):
        if type(state.get(key)) is not int or state[key] != 0:
            raise RuntimeError("full-suite QA has missing or nonzero outcome counts")
    counts = [state.get(key) for key in ("tests_run", "discovered_test_count", "expected_test_count")]
    if any(type(value) is not int or value < 1 for value in counts) or len(set(counts)) != 1:
        raise RuntimeError("full-suite run, discovery and expected counts disagree")
    if state.get("log_sha256") != digest(log_path):
        raise RuntimeError("full-suite terminal log hash differs")
    text = log_path.read_text(errors="replace")
    found = re.findall(r"^Ran ([0-9]+) tests? in ([0-9.]+)s\s*$", text, re.MULTILINE)
    last = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), None)
    if not found or int(found[-1][0]) != counts[0] or last != "OK":
        raise RuntimeError("full-suite log lacks the matching terminal count and OK")
    sources = state.get("source_sha256")
    if not isinstance(sources, dict) or not REQUIRED_QA_SOURCES.issubset(sources) or state.get("source_manifest_sha256") != canonical(sources):
        raise RuntimeError("full-suite source manifest is incomplete or changed")
    for name, expected in sources.items():
        path = (root/name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(root) or not path.is_file() or digest(path) != expected:
            raise RuntimeError("a tested source changed: "+name)
    return {"status": "completed", "passed": True, "tests": counts[0],
        "state_path": str(state_path.relative_to(root)), "state_sha256": digest(state_path),
        "log_path": str(log_path.relative_to(root)), "log_sha256": digest(log_path),
        "source_manifest_sha256": state["source_manifest_sha256"], "source_files": len(sources),
        "campaign_plan_sha256": state.get("campaign_plan_sha256"),
        "source_sha256": sources}


def sealed_report(directory):
    directory = Path(directory).resolve()
    index = read(directory/"report-index.json")
    records, seen = index["files"], set()
    if canonical(records) != index["digest"]:
        raise RuntimeError("report index identity differs")
    for record in records:
        name = record["path"]
        path = (directory/name).resolve()
        if (name in seen or not path.is_relative_to(directory) or not path.is_file()
            or path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]):
            raise RuntimeError("a sealed report artifact changed or is duplicated")
        seen.add(name)
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if path.is_file() and path != directory/"report-index.json"}
    if seen != REPORT_FILES or actual != seen:
        raise RuntimeError("the report seal does not cover its exact four artifacts")
    return index


def compact_rows(summary):
    keep = ("final_objective", "final_unpenalized_objective", "final_regularization_penalty",
        "final_head_stationarity", "readout_calibration_identity", "initial_calibration_matrix_builds",
        "initial_calibration_spectral_calculations", "completed_updates", "requested_updates", "stop_reason",
        "trial_solve_count", "total_projection_solve_calls", "controller_moment_restarts", "final_controller_state",
        "hidden_accepted_updates", "hidden_rejected_updates", "hidden_backtrack_count", "wall_seconds")
    rows = []
    for row in summary["rows"]:
        methods = {}
        for method, value in row["methods"].items():
            methods[method] = {"saved": value["saved"], "training": {key: value["training"].get(key) for key in keep},
                "training_report_sha256": value["training_report_sha256"],
                "progress_provenance": value["progress_provenance"]}
        rows.append({key: value for key, value in row.items() if key != "methods"} | {"methods": methods})
    return rows


def finalize():
    if TRACKED_RESULT.exists() or FINALIZATION.exists():
        raise RuntimeError("regularized finalization was already created; no overwrite")
    if any(read(path).get("status") != "completed" for path in
           (CAMPAIGN/"state.json", CAMPAIGN/"execution.json", VALIDATION/"execution-state.json")):
        raise RuntimeError("campaign, execution and supervisor must be completed")
    qa = qa_evidence(VALIDATION)
    plan = read(CAMPAIGN/"plan.json")
    if (qa["campaign_plan_sha256"] != digest(CAMPAIGN/"plan.json")
        or not plan.get("source_sha256") or any(qa["source_sha256"].get(name) != expected
            for name, expected in plan["source_sha256"].items())):
        raise RuntimeError("full-suite QA does not bind this complete frozen campaign source closure")
    from thermotwin.reports import pinn_spatial_regularized as renderer
    summary = renderer.load_comparison(CAMPAIGN, validation_directory=VALIDATION)
    index = sealed_report(REPORT)
    if (read(REPORT/"result-summary.json") != summary or index["plan_sha256"] != summary["plan_sha256"]
        or index.get("evidence") != summary["evidence"] or index["renderer_sha256"] != digest(renderer.__file__)):
        raise RuntimeError("sealed report differs from the exact validated saved coordinator evidence")
    if digest(TRACKED_PLAN) != digest(CAMPAIGN/"plan.json"):
        raise RuntimeError("tracked and executed regularized plans differ byte for byte")
    result = {"schema": "completed_fixed_ridge_forward_development_v1", "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(), "development_only": True,
        "saved_forward_status": summary["saved_forward_status"],
        "initial_method_checks_passed": summary["initial_method_checks_passed"],
        "denominator": summary["denominator"], "scope_flags": summary["scope_flags"],
        "no_independent_replication_or_new_dataset_claimed": True,
        "objective_comparison_rule": "ridge total includes a penalty; compare original components explicitly, never rank differing totals naively",
        "campaign": {"path": str(CAMPAIGN.relative_to(ROOT)), "evidence": summary["evidence"]["regularized"],
            "tracked_plan_sha256": digest(TRACKED_PLAN)},
        "prior_evidence": {key: value for key, value in summary["evidence"].items() if key != "regularized"},
        "method_verification": summary["method_verification"], "calibration_accounting": summary["calibration_accounting"],
        "rows": compact_rows(summary), "campaign_costs": summary["campaign_costs"], "full_suite_qa": qa,
        "report": {"directory": str(REPORT.relative_to(ROOT)), "index_sha256": digest(REPORT/"report-index.json"),
            "digest": index["digest"], "files": len(index["files"]), "summary_sha256": digest(REPORT/"result-summary.json")},
        "finalizer": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": digest(__file__)},
        "new_fits_or_gate_calculations": 0, "checkpoint_selection_performed": False,
        "automatic_followup_launched": False}
    # Reauthenticate the scientific chain after reading; no output touches it.
    if renderer.load_comparison(CAMPAIGN, validation_directory=VALIDATION) != summary:
        raise RuntimeError("scientific evidence changed during finalization")
    FINALIZATION.mkdir(parents=True, exist_ok=False)
    payload = (json.dumps(result, indent=2, allow_nan=False)+"\n").encode()
    (FINALIZATION/"result.json").write_bytes(payload)
    records = [{"path": "result.json", "bytes": len(payload), "sha256": digest(FINALIZATION/"result.json")}]
    (FINALIZATION/"artifact-index.json").write_text(json.dumps({"files": records, "digest": canonical(records)}, indent=2)+"\n")
    with TRACKED_RESULT.open("xb") as handle:
        handle.write(payload)
    return {"status": "completed", "tracked_result": str(TRACKED_RESULT.relative_to(ROOT)),
            "sha256": digest(TRACKED_RESULT), "saved_forward_status": result["saved_forward_status"]}


if __name__ == "__main__":
    print(json.dumps(finalize(), allow_nan=False))
