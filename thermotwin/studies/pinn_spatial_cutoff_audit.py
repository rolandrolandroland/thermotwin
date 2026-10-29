"""Fourteen training-only readout solves around two frozen terminal fields.

This diagnostic has no optimizer, forward gate, field-reference loader, inverse
stage, checkpoint selector, or automatic follow-up campaign.
"""
import argparse
from dataclasses import dataclass, field
import hashlib
import contextlib
import io
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

from . import pinn_spatial_continuity as common
from ..physics.planar_contact import PlanarContactConfig
from ..pinn.planar_contact_continuity import linear_system, objective_continuity
from ..pinn.planar_contact_inverse import PlanarObservations
from ..pinn.planar_contact_graded import graded_point_metadata, graded_training_points
from ..pinn.planar_contact_step_control import StepControlConfig, StepControlPINN


ROOT = common.ROOT
CASES = ("t02", "t04")
RADII = (1.e-4, 1.e-5, 1.e-6)
MAXIMUM_SOLVES = 14
WALL_SECONDS = 900.
RSS_BYTES = 2*1024**3
INPUT_NAMES = ("final_checkpoint.pt", "training-report.json", "observations.npz")


def _read(path):
    return json.loads(Path(path).read_text())


def _record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": common.digest_file(path), "bytes": path.stat().st_size}


def runtime_context():
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        np.show_config()
    return {**common.runtime_context(), "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(), "machine": platform.machine(),
        "numpy_blas_lapack_configuration": stream.getvalue(),
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "blas_thread_environment": {name: os.environ.get(name) for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}}


def _one_thread():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def _parent_commit():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def _validate_campaign(directory):
    # Preservation authentication only; gates never select diagnostic inputs.
    from .pinn_spatial_step_control import validate_step_control
    return validate_step_control(directory)


def _protocol():
    return [{"case": case, "kind": kind, "radius": radius, "sign": sign}
        for case in CASES for kind, radius, sign in
        [("baseline", 0., 0)]+[("perturbation", radius, sign) for radius in RADII for sign in (-1, 1)]]


def _outside(output, protected):
    for name in protected:
        directory = Path(name).resolve()
        if output.is_relative_to(directory) or directory.is_relative_to(output):
            raise ValueError("audit output must be separate from protected scientific evidence")


def prepare_cutoff_audit(output_directory, *, campaign_directory, source_paths=(),
                         protected_files=(), protected_directories=()):
    """Bind inputs and the complete fixed protocol without solving any head."""
    campaign, output = Path(campaign_directory).resolve(), Path(output_directory).resolve()
    _one_thread()
    if _read(campaign/"state.json").get("status") != "completed":
        raise ValueError("terminal input campaign must already be completed")
    campaign_context = _validate_campaign(campaign)
    old_plan = _read(campaign/"plan.json")
    declared = {case["id"]: case for case in old_plan["cases"]}
    if any(declared[case]["geometry"] != "partial_contact" for case in CASES):
        raise ValueError("the two declared partial-contact cases are required")
    trees = sorted({str(campaign), *(str(Path(path).resolve()) for path in protected_directories)})
    _outside(output, trees)
    for ancestor in (output, *output.parents):
        if any((ancestor/name).is_file() for name in ("artifact-index.json", "evidence-index.json", "report-index.json")):
            raise ValueError("audit output must be separate from sealed evidence")
    inputs = {}
    for case in CASES:
        worker = campaign/case
        state, index = _read(worker/"state.json"), _read(worker/"artifact-index.json")
        if state.get("status") != "completed" or state.get("artifact_index_sha256") != common.digest_file(worker/"artifact-index.json"):
            raise ValueError("terminal worker proof changed")
        inputs[case] = {name: _record(worker/name) for name in INPUT_NAMES}
        for name, saved in inputs[case].items():
            if {key: saved[key] for key in ("sha256", "bytes")} != index.get(name):
                raise ValueError("terminal training input is not bound by its sealed worker")
    sources = dict(old_plan["source_sha256"])
    sources.update(common._source_records(set(source_paths) | {
        "thermotwin/studies/pinn_spatial_cutoff_audit.py", "tests/test_pinn_spatial_cutoff_audit.py"}))
    if common._source_records(sources) != sources:
        raise RuntimeError("a frozen numerical source changed")
    plan = {"schema": "terminal_partial_cutoff_directional_audit_v1", "cases": list(CASES),
        "created_utc": datetime.now(timezone.utc).isoformat(), "parent_commit": _parent_commit(),
        "protocol": _protocol(), "radii": list(RADII), "inputs": inputs,
        "campaign_directory": str(campaign), "source_sha256": sources, "runtime": runtime_context(),
        "campaign_context": campaign_context,
        "protected_trees": {name: common._tree_records(name) for name in trees},
        "protected_files": common._source_records(protected_files),
        "maximum_readout_solves": MAXIMUM_SOLVES, "maximum_wall_seconds": WALL_SECONDS,
        "maximum_rss_bytes": RSS_BYTES, "threads": 1, "automatic_retries": 0,
        "neural_training_updates": 0, "new_checkpoints_written": 0,
        "training_only": True, "reference_fields_used": False, "independent_diagnostics_used": False,
        "independent_diagnostics_used_for_selection": False,
        "preservation_validation_role": "saved diagnostics may be authenticated solely for preservation and are never passed to the numerical audit",
        "observation_keys_read": ["coordinates", "temperature"], "noise_standard_deviation_K": .02,
        "direction": "negative original-terminal fixed-head gradient divided by maximum absolute hidden-parameter component",
        "reset_policy": "restore terminal hidden parameters and terminal head independently before every point",
        "readout_policy": "one frozen column-scaled TSVD solve; compare strong-AD candidate loss to same-point original-terminal-head loss",
        "fd_policy": "centered pure-TSVD and monotone-policy differences only when sampled minus/base/plus ranks agree; sampled equality does not exclude intermediate crossings or imply an envelope-theorem identity",
        "interpretation": "any differing solved ranks within a case support projection-policy review only; all stable ranks are inconclusive about hard cutoff; no automatic corner or optimizer revision"}
    output.mkdir(parents=True, exist_ok=False)
    common._write(output/"plan.json", plan)
    for name in sources:
        target = output/"source-snapshot"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    common._write(output/"state.json", {"status": "prepared", "plan_sha256": common.digest_file(output/"plan.json")})
    return plan


def verify_plan(plan, directory=None):
    expected = {"schema": "terminal_partial_cutoff_directional_audit_v1", "cases": list(CASES),
        "protocol": _protocol(), "radii": list(RADII), "maximum_readout_solves": MAXIMUM_SOLVES,
        "maximum_wall_seconds": WALL_SECONDS, "maximum_rss_bytes": RSS_BYTES, "threads": 1,
        "automatic_retries": 0, "neural_training_updates": 0, "new_checkpoints_written": 0,
        "training_only": True, "reference_fields_used": False, "independent_diagnostics_used": False,
        "independent_diagnostics_used_for_selection": False,
        "observation_keys_read": ["coordinates", "temperature"], "noise_standard_deviation_K": .02}
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("the fixed fourteen-solve protocol changed")
    if runtime_context() != plan["runtime"] or common._source_records(plan["source_sha256"]) != plan["source_sha256"]:
        raise RuntimeError("bound audit source or one-thread runtime changed")
    if common._source_records(plan["protected_files"]) != plan["protected_files"]:
        raise RuntimeError("a protected compact scientific record changed")
    if set(plan["inputs"]) != set(CASES):
        raise ValueError("audit input cases changed")
    for case, records in plan["inputs"].items():
        if set(records) != set(INPUT_NAMES):
            raise ValueError("audit may load only the declared training inputs")
        for name, record in records.items():
            if Path(record["path"]).resolve() != Path(plan["campaign_directory"]).resolve()/case/name or _record(record["path"]) != record:
                raise RuntimeError("bound terminal training input changed")
    for name, records in plan["protected_trees"].items():
        if common._tree_records(name) != records:
            raise RuntimeError("a protected scientific experiment tree changed")
    if _validate_campaign(plan["campaign_directory"]) != plan["campaign_context"]:
        raise RuntimeError("the completed terminal campaign proof changed")
    if directory is not None:
        directory = Path(directory).resolve()
        if _read(directory/"plan.json") != plan:
            raise RuntimeError("audit plan changed on disk")
        _outside(directory, plan["protected_trees"])
        for name, expected_hash in plan["source_sha256"].items():
            if common.digest_file(directory/"source-snapshot"/name) != expected_hash:
                raise RuntimeError("bound audit source snapshot changed")
    return True


@dataclass
class SolveBudget:
    attempted: int = 0
    records: dict = field(default_factory=dict)

    @property
    def successful(self):
        return sum(value.get("status") == "solved" for value in self.records.values())

    def consume(self):
        if self.attempted >= MAXIMUM_SOLVES:
            raise RuntimeError("fourteen-solve budget exhausted")
        self.attempted += 1
        self.records[self.attempted] = {"status": "attempting"}


def _finite(value):
    return float(value) if math.isfinite(float(value)) else None


def solve_readout_once(model, points, observations, budget):
    """One TSVD call, raw/policy diagnostics, and exact original-head restoration."""
    original = model.coefficients.detach().clone()
    try:
        before = float(objective_continuity(model, points, observations)[0].detach())
        if not math.isfinite(before):
            raise FloatingPointError("nonfinite same-point terminal-head objective")
        matrix, rhs, metadata = linear_system(model, points, observations)
        scales = np.maximum(np.linalg.norm(matrix, axis=0), 1.e-12)
        normalized = matrix/scales
        budget.consume()
        solution, _, rank, singular = np.linalg.lstsq(normalized, rhs, rcond=model.config.rank_rcond)
        proposed = solution/scales
        if not np.isfinite(proposed).all():
            raise FloatingPointError("nonfinite audit readout")
        with torch.no_grad():
            model.coefficients.copy_(torch.tensor(proposed, dtype=torch.float64))
        try:
            after = float(objective_continuity(model, points, observations)[0].detach())
        except FloatingPointError:
            after = math.inf
        with np.errstate(over="ignore", invalid="ignore"):
            residual = matrix@proposed-rhs
            algebraic = float(np.sum(residual**2))
            normal = matrix.T@residual
            scaled_normal = normalized.T@residual
        cutoff = float(singular[0]*model.config.rank_rcond) if len(singular) else 0.
        smallest = float(singular[rank-1]) if rank else None
        discarded = float(singular[rank]) if rank < len(singular) else None
        accepted = math.isfinite(after) and after <= before
        finite_losses = math.isfinite(after) and math.isfinite(algebraic)
        record = {"status": "solved" if finite_losses else "failed",
            "failure": None if finite_losses else "nonfinite pure-TSVD diagnostic objective",
            "rank": int(rank), "columns": matrix.shape[1], "rows": len(rhs),
            "rank_rcond": model.config.rank_rcond, "singular_values": singular.tolist(), "cutoff": cutoff,
            "smallest_retained_singular": smallest, "largest_discarded_singular": discarded,
            "retained_relative_cutoff_margin": smallest/cutoff-1 if smallest is not None and cutoff else None,
            "discarded_relative_cutoff_margin": 1-discarded/cutoff if discarded is not None and cutoff else None,
            "fixed_terminal_head_objective": before, "pure_tsvd_objective": _finite(after),
            "pure_tsvd_algebraic_objective": _finite(algebraic),
            "algebraic_strong_objective_abs_difference": _finite(abs(algebraic-after)),
            "monotone_policy_accepted": accepted, "monotone_policy_objective": after if accepted else before,
            "coefficient_norm_proposed": _finite(np.linalg.norm(proposed)),
            "head_gradient_norm": _finite(2*np.linalg.norm(normal)),
            "column_scaled_head_gradient_norm": _finite(2*np.linalg.norm(scaled_normal)),
            "head_normal_equation_relative_residual": _finite(np.linalg.norm(normal)/max(1., np.linalg.norm(matrix.T@rhs))),
            "system": metadata, "solve_number": budget.attempted}
        budget.records[budget.attempted] = record
        return record
    finally:
        with torch.no_grad():
            model.coefficients.copy_(original)


def fixed_head_direction(model, points, observations):
    """Direction and AD derivative at the original terminal state, before solves."""
    parameters = list(model.networks.parameters())
    objective = objective_continuity(model, points, observations)[0]
    gradients = torch.autograd.grad(objective, parameters, allow_unused=True)
    gradients = [torch.zeros_like(parameter) if gradient is None else gradient.detach()
                 for parameter, gradient in zip(parameters, gradients)]
    if not bool(torch.isfinite(objective.detach())) or any(not bool(torch.all(torch.isfinite(gradient))) for gradient in gradients):
        raise FloatingPointError("nonfinite terminal hidden gradient or objective")
    maximum = max(float(torch.max(abs(gradient))) for gradient in gradients)
    if not math.isfinite(maximum) or maximum == 0:
        raise FloatingPointError("terminal hidden gradient cannot define the fixed direction")
    direction = [-gradient/maximum for gradient in gradients]
    digest = hashlib.sha256()
    for value in direction:
        digest.update(np.asarray(value.cpu().numpy(), dtype="<f8").tobytes())
    derivative = sum(float(torch.sum(gradient*value)) for gradient, value in zip(gradients, direction))
    squared_norm = sum(float(torch.sum(gradient**2)) for gradient in gradients)
    if not math.isfinite(derivative) or not math.isfinite(squared_norm):
        raise FloatingPointError("nonfinite terminal gradient diagnostic")
    return direction, {"sha256": digest.hexdigest(), "normalization": "maximum absolute hidden component equals one",
        "parameter_shapes": [list(parameter.shape) for parameter in parameters], "gradient_max_abs": maximum,
        "gradient_l2": math.sqrt(squared_norm),
        "fixed_head_ad_directional_derivative": derivative, "terminal_fixed_head_objective": float(objective.detach())}


def directional_differences(samples, derivative):
    baseline = samples[0]
    rows = []
    for radius in RADII:
        minus, plus = [sample for sample in samples if sample["radius"] == radius]
        stable = all(sample.get("status") == "solved" for sample in (minus, baseline, plus)) and len({sample["rank"] for sample in (minus, baseline, plus)}) == 1
        row = {"radius": radius, "sampled_rank_stable": stable,
            "sampled_ranks": [sample.get("rank") for sample in (minus, baseline, plus)],
            "fixed_head_ad_directional_derivative": derivative,
            "pure_tsvd_centered_fd": None, "monotone_policy_centered_fd": None,
            "limitation": "sampled rank equality does not exclude intermediate crossings; no envelope-theorem equality is assumed"}
        if stable:
            for name, key in (("pure_tsvd_centered_fd", "pure_tsvd_objective"), ("monotone_policy_centered_fd", "monotone_policy_objective")):
                if minus.get(key) is not None and plus.get(key) is not None:
                    row[name] = _finite((plus[key]-minus[key])/(2*radius))
        rows.append(row)
    return rows


def _load_case(case, records):
    saved = torch.load(records["final_checkpoint.pt"]["path"], map_location="cpu", weights_only=False)
    raw = _read(records["training-report.json"]["path"])
    expected = next(value for value in common.CASES if value["id"] == case)
    if (saved.get("model_class") != "StepControlPINN" or saved.get("model_module") != "thermotwin.pinn.planar_contact_step_control"
        or saved["config"] != raw["config"] or saved["physics"] != raw["physics"] or saved["known_h"] != raw["known_h"]
        or saved["known_h"] != expected["known_h"]):
        raise ValueError("terminal checkpoint and training metadata differ")
    model = StepControlPINN(PlanarContactConfig(**saved["physics"]), StepControlConfig(**saved["config"]), known_h=saved["known_h"])
    if model.physics != common.physics_for(expected):
        raise ValueError("terminal physical case differs from its declared partial geometry")
    model.load_state_dict(saved["state_dict"])
    points = {name: np.asarray(values, dtype=np.float64) for name, values in raw["training_points"].items()}
    regenerated = graded_training_points(model.physics, model.config, raw["sampling_seed"])
    if set(points) != set(regenerated) or any(not np.array_equal(values, regenerated[name]) for name, values in points.items()):
        raise ValueError("terminal training coordinates differ from the frozen graded grid")
    if raw["graded_point_metadata"] != graded_point_metadata(model.physics, model.config, raw["sampling_seed"]):
        raise ValueError("terminal training grid metadata differs")
    for name, values in points.items():
        if not np.array_equal(np.asarray(raw["training_point_weights"][name], dtype=float), np.full(len(values), 1/len(values))):
            raise ValueError("terminal training row weights differ from the frozen objective")
    with np.load(records["observations.npz"]["path"]) as data:
        observations = PlanarObservations(data["coordinates"].copy(), data["temperature"].copy(), .02)
    return model, points, observations, raw


def audit_case(case, records, budget, progress=None):
    model, points, observations, raw = _load_case(case, records)
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    hidden = [parameter.detach().clone() for parameter in model.networks.parameters()]
    samples = [dict(point, status="not_run", reason="unattempted declared opportunity",
                    readout_attempted=False, attempt_before=None, attempt_after=None)
               for point in _protocol() if point["case"] == case]
    direction_metadata, failure, resource_failure, current_index = None, None, False, None
    try:
        direction, direction_metadata = fixed_head_direction(model, points, observations)
        if not math.isclose(direction_metadata["terminal_fixed_head_objective"], raw["final_objective"], rel_tol=1.e-8, abs_tol=1.e-9):
            raise ValueError("saved terminal training objective did not reproduce")
        for index, point in enumerate([value for value in _protocol() if value["case"] == case]):
            current_index, attempts_before = index, budget.attempted
            model.load_state_dict(original)
            with torch.no_grad():
                for parameter, terminal, vector in zip(model.networks.parameters(), hidden, direction):
                    parameter.copy_(terminal+point["sign"]*point["radius"]*vector)
            try:
                record = dict(point, **solve_readout_once(model, points, observations, budget))
            except common.ResourceLimit as error:
                resource_failure = True
                failure = {"error_type": type(error).__name__, "error": str(error)}
                retained = budget.records.get(budget.attempted, {}) if budget.attempted > attempts_before else {}
                record = dict(point, **retained) if retained.get("status") in ("solved", "failed") else dict(point, status="failed", **failure)
                record["resource_failure"] = failure
            except Exception as error:
                record = dict(point, status="failed", error_type=type(error).__name__, error=str(error),
                              attempted_readout_solves=budget.attempted)
            record.update(attempt_before=attempts_before, attempt_after=budget.attempted,
                          readout_attempted=budget.attempted > attempts_before)
            if record["readout_attempted"]:
                record["solve_number"] = budget.attempted
            samples[index] = record
            if progress:
                progress(record)
            if resource_failure:
                break
    except common.ResourceLimit as error:
        resource_failure, failure = True, {"error_type": type(error).__name__, "error": str(error)}
        if current_index is not None:
            record = samples[current_index]
            if record["status"] == "not_run":
                retained = budget.records.get(budget.attempted, {}) if budget.attempted > attempts_before else {}
                record.update(retained if retained.get("status") in ("solved", "failed") else dict(status="failed", **failure))
                record.update(attempt_before=attempts_before, attempt_after=budget.attempted,
                              readout_attempted=budget.attempted > attempts_before)
                if record["readout_attempted"]:
                    record["solve_number"] = budget.attempted
            record["resource_failure"] = failure
    except Exception as error:
        failure = {"error_type": type(error).__name__, "error": str(error)}
    finally:
        model.load_state_dict(original)
    differences = directional_differences(samples, direction_metadata["fixed_head_ad_directional_derivative"]) if direction_metadata else []
    ranks = sorted({sample["rank"] for sample in samples if sample.get("status") == "solved"})
    return {"case": case, "status": "completed" if failure is None and len(samples) == 7 and all(sample["status"] == "solved" for sample in samples) else "failed",
        "direction": direction_metadata, "samples": samples, "directional_differences": differences,
        "failure": failure, "resource_failure": resource_failure,
        "distinct_solved_ranks": ranks, "sampled_cutoff_sensitivity": len(ranks) > 1,
        "policy_rejections": sum(sample.get("monotone_policy_accepted") is False for sample in samples),
        "original_state_restored_exactly": all(torch.equal(value, original[name]) for name, value in model.state_dict().items())}


def run_cutoff_audit(output_directory):
    directory = Path(output_directory).resolve()
    _one_thread()
    plan = _read(directory/"plan.json")
    plan_hash = common.digest_file(directory/"plan.json")
    if _read(directory/"state.json") != {"status": "prepared", "plan_sha256": plan_hash}:
        raise RuntimeError("audit plan was already consumed or changed")
    verify_plan(plan, directory)
    descriptor = os.open(directory/"run.lock", os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(str(os.getpid())+"\n")
    started, budget, rows, error = time.perf_counter(), SolveBudget(), [], None
    preserved = False
    def guard(signum, frame):
        if time.perf_counter()-started > WALL_SECONDS or common._rss_bytes() > RSS_BYTES:
            signal.setitimer(signal.ITIMER_REAL, 0.)
            raise common.ResourceLimit("cutoff audit resource limit")
    previous = signal.signal(signal.SIGALRM, guard)
    signal.setitimer(signal.ITIMER_REAL, .5, .5)
    common._write(directory/"state.json", {"status": "running", "plan_sha256": plan_hash})
    def progress(record):
        with (directory/"progress.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False)+"\n")
    try:
        for case in CASES:
            try:
                row = audit_case(case, plan["inputs"][case], budget, progress)
            except Exception as caught:
                row = {"case": case, "status": "failed", "error_type": type(caught).__name__, "error": str(caught),
                    "resource_failure": isinstance(caught, common.ResourceLimit), "samples": []}
            rows.append(row)
            common._write(directory/(case+".json"), row)
            if row.get("resource_failure"):
                error = {"error_type": "ResourceLimit", "error": "cutoff audit resource limit"}
                break
    except BaseException as caught:
        error = {"error_type": type(caught).__name__, "error": str(caught)}
    finally:
        try:
            verify_plan(plan, directory)
            preserved = True
        except BaseException as caught:
            preservation_error = {"error_type": type(caught).__name__, "error": str(caught)}
            error = dict(error or {}, preservation_error=preservation_error)
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, previous)
    # Retain both case denominators and every declared opportunity after a cap.
    by_case = {row["case"]: row for row in rows}
    for case in CASES:
        row = by_case.setdefault(case, {"case": case, "status": "not_run", "reason": "audit stopped before this case", "samples": []})
        present = {(sample["kind"], sample["radius"], sample["sign"]): sample for sample in row["samples"]}
        row["samples"] = [present.get((point["kind"], point["radius"], point["sign"]),
            dict(point, status="not_run", reason="unattempted declared opportunity", readout_attempted=False,
                 attempt_before=None, attempt_after=None)) for point in _protocol() if point["case"] == case]
    rows = [by_case[case] for case in CASES]
    # Canonicalize the retained prefix after interruption; no lost/duplicate
    # progress event can disagree with the final per-opportunity accounting.
    progress_rows = [sample for row in rows for sample in row["samples"] if sample["status"] != "not_run"]
    common._atomic(directory/"progress.jsonl", ("".join(json.dumps(value, allow_nan=False)+"\n" for value in progress_rows)).encode())
    for row in rows:
        common._write(directory/(row["case"]+".json"), row)
    completed = error is None and budget.attempted == budget.successful == MAXIMUM_SOLVES and all(row["status"] == "completed" and row["original_state_restored_exactly"] for row in rows)
    sensitivity = any(row.get("sampled_cutoff_sensitivity") for row in rows)
    result = {"status": "completed" if completed else "failed", "plan_sha256": plan_hash,
        "cases": rows, "attempted_readout_solves": budget.attempted, "successful_readout_solves": budget.successful,
        "maximum_readout_solves": MAXIMUM_SOLVES, "training_only": True, "neural_training_updates": 0,
        "inputs_and_sources_preserved": preserved,
        "reference_fields_used": False, "independent_diagnostics_used": False, "new_checkpoints_written": 0,
        "independent_diagnostics_used_for_selection": False,
        "sampled_cutoff_sensitivity_observed": sensitivity if completed else None,
        "interpretation": "projection-policy review supported; this does not explain all forward error" if completed and sensitivity else "inconclusive; no automatic architecture or optimizer choice",
        "automatic_followup_launched": False, "wall_seconds": time.perf_counter()-started,
        "peak_rss_bytes": common._rss_bytes(), "failure": error}
    common._write(directory/"result.json", result)
    index = {str(path.relative_to(directory)): {"sha256": common.digest_file(path), "bytes": path.stat().st_size}
        for path in sorted(directory.rglob("*")) if path.is_file() and path not in (directory/"state.json", directory/"artifact-index.json")}
    common._write(directory/"artifact-index.json", index)
    common._write(directory/"state.json", {"status": result["status"], "plan_sha256": plan_hash,
        "artifact_index_sha256": common.digest_file(directory/"artifact-index.json")})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output-directory", required=True)
    prepare.add_argument("--campaign-directory", required=True)
    prepare.add_argument("--source-path", action="append", default=[])
    prepare.add_argument("--protected-file", action="append", default=[])
    prepare.add_argument("--protected-directory", action="append", default=[])
    run = commands.add_parser("run")
    run.add_argument("--output-directory", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        value = prepare_cutoff_audit(args.output_directory, campaign_directory=args.campaign_directory,
            source_paths=args.source_path, protected_files=args.protected_file, protected_directories=args.protected_directory)
        print(json.dumps({"status": "prepared", "schema": value["schema"], "maximum_readout_solves": MAXIMUM_SOLVES}))
    else:
        value = run_cutoff_audit(args.output_directory)
        print(json.dumps({key: value[key] for key in ("status", "attempted_readout_solves", "successful_readout_solves", "interpretation")}, allow_nan=False))


if __name__ == "__main__":
    main()
