"""Frozen, bounded 2D thermal-plate contact experiment orchestration.

Only ``run_planar_benchmark`` launches scientific fits.  Importing this module
or exercising its pure helpers creates no benchmark partitions.  The model
is a thermal plate with prescribed bath and heat-flux traces, not a complete
two-dimensional electrothermal module.  The known-contact neural method is
an explicitly truth-informed oracle diagnostic.
"""

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter, process_time
import traceback

import numpy as np
from scipy.optimize import least_squares
import torch

from ..physics.planar_contact import (
    PlanarContactConfig, PlanarContactModel, verify_planar_contact_reference,
)
from ..pinn.planar_contact_inverse import (
    PlanarObservations, PlanarPINNConfig, predict_planar_temperature,
    train_planar_contact_pinn,
)


BENCHMARK_SPEC = {
    "campaign": "pinn_spatial_v1_2026_10",
    "model": "2D thermal plate with prescribed bath and top heat flux",
    "physics": asdict(PlanarContactConfig()),
    "duration_seconds": 120.0,
    "initial_temperature_kelvin": 300.0,
    "temperature_scale_kelvin": 5.0,
    "training_trace": {"ramp_seconds": 10.0, "bath_drop_kelvin": 5.0,
                       "top_flux_watts_per_square_metre": 500.0},
    "holdout_trace": {"ramp_seconds": 15.0, "bath_drop_kelvin": 7.0,
                      "top_flux_watts_per_square_metre": 750.0},
    "development_contact_w_m2k": 800.0,
    "evaluation_contact_w_m2k": 1000.0,
    "h_bounds_w_m2k": [400.0, 2000.0],
    "initial_h_w_m2k": math.sqrt(400.0 * 2000.0),
    "evaluation_trials": 3,
    "sensors": {"x_m": [0.004, 0.020, 0.036], "z_m": 0.003,
                "times_s": list(range(0, 121, 10)),
                "missing_sensor_index": 1, "missing_times_s": [40, 50, 60],
                "noise_standard_deviation_kelvin": 0.02,
                "noise_draws": "one independent normal draw per retained point"},
    "field": {"nx": 40, "nz": 20,
              "times_s": [0, 1, 2, 5] + list(range(10, 121, 10)),
              "weighting": "unweighted_fixed_space_time_point_set"},
    "reference": {"nx": 160, "nz": 40, "max_step_s": 0.0625},
    "conventional": {"nx": 80, "nz": 20, "max_step_s": 0.125,
                     "optimizer": "scipy least_squares in log(h)",
                     "max_nfev": 40, "xtol": 1e-8, "ftol": 1e-8, "gtol": 1e-8},
    "neural": {"width": 32, "layers": 3, "interior_points": 384,
               "boundary_points_per_side": 64, "learning_rate": 0.001,
               "dtype": "float64", "device": "cpu", "threads": 1,
               "development_physics_weights": [10.0, 100.0],
               "development_adam_epochs": [1800, 5400], "lbfgs_iterations": 200,
               "checkpoint": "optimizer-returned final state"},
    "development_selection": {
        "uses": ["normalized_observation_mse", "independent_residual_families",
                 "populated_bottom_contact_and_insulated_regions"],
        "passing_order": ["smallest_adam_epochs", "observation_mse_plus_residual_rms_sum",
                          "smallest_physics_weight"],
        "failure_order": ["smallest_max_normalized_gate_violation", "smallest_adam_epochs",
                          "observation_mse_plus_residual_rms_sum", "smallest_physics_weight"],
        "gate_violation": "max(0,max(observation_mse/2,each_residual_rms/.05)-1); missing=infinity",
        "all_candidates_run_once": True,
    },
    "gates": {"normalized_observation_mse": 2.0, "independent_residual_rms": 0.05,
              "hidden_field_rmse_kelvin": 0.05, "hidden_field_max_kelvin": 0.15,
              "holdout_field_rmse_kelvin": 0.05, "holdout_field_max_kelvin": 0.15,
              "relative_contact_error": 0.10,
              "minimum_fraction_absolute_error_reduction_from_h0": 0.50,
              "meaningful_parameter_recovery": "relative error <=.10 AND signed absolute-error reduction from h0 >=.50"},
    "sensitivity": {"contacts_w_m2k": [400.0, 800.0, 1000.0, 2000.0],
                    "log_h_difference_step": 0.001, "normal_95_multiplier": 1.96,
                    "maximum_95_log_h_halfwidth": math.log(1.1),
                    "failure_action": "continue bounded field study; prohibit h-recovery claim",
                    "interpretation": "optimistic local Fisher information with all other quantities known"},
    "development_profile": {"points": 21, "grid": "equally spaced in log(h)",
                            "interpretation": "finite-grid objective diagnostic; no global uniqueness claim"},
    "random_streams": {"derivation": "first 4 SHA256 bytes, big endian, of campaign/partition/trial/purpose",
                       "purposes": ["observation_noise", "neural_initialization", "collocation", "diagnostics"],
                       "development_trial": 0, "evaluation_trial_indices": [0, 1, 2],
                       "paired_neural_methods": True, "paired_development_candidates": True},
    "evaluation_methods": ["inverse_pinn", "data_only", "known_h_oracle", "conventional", "frozen_h0"],
    "frozen_h0_baseline": {"fit": False, "h_w_m2k": math.sqrt(400.0 * 2000.0),
                           "forward_solver": "conventional", "includes": ["sensors", "hidden_field", "holdout", "wall_cost"]},
    "holdout_transfer": "estimated h in conventional forward model; oracle uses known h; data-only unavailable",
    "evaluation_policy": "fixed configuration; no tuning, retries, checkpoint selection, or failure removal",
}

_RESIDUAL_FAMILIES = (
    "interior", "left", "right", "top", "bottom", "bottom_contact", "bottom_insulated",
)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, value):
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _semantic_seed(partition, trial, purpose):
    payload = f"{BENCHMARK_SPEC['campaign']}/{partition}/{trial}/{purpose}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _paired_seeds(partition, trial):
    return {purpose: _semantic_seed(partition, trial, purpose)
            for purpose in BENCHMARK_SPEC["random_streams"]["purposes"]}


def _seed_audit():
    streams = {"development/0": _paired_seeds("development", 0)}
    streams.update({f"evaluation/{trial}": _paired_seeds("evaluation", trial)
                    for trial in range(BENCHMARK_SPEC["evaluation_trials"])})
    flattened = [seed for values in streams.values() for seed in values.values()]
    if len(set(flattened)) != len(flattened):
        raise ValueError("predeclared random-stream hash collision; no data or fits may run")
    return {"streams": streams, "distinct_semantic_streams": len(flattened), "all_unique": True,
            "intentional_reuse": "same trial seeds across neural methods; same development seeds across candidates"}


def _observation_coordinates():
    spec = BENCHMARK_SPEC["sensors"]
    coordinates, retained = [], []
    for time in spec["times_s"]:
        for sensor, x in enumerate(spec["x_m"]):
            coordinates.append((x, spec["z_m"], time))
            retained.append(not (sensor == spec["missing_sensor_index"]
                                and time in spec["missing_times_s"]))
    return np.asarray(coordinates, dtype=float), np.asarray(retained, dtype=bool)


def _field_coordinates():
    spec, physics = BENCHMARK_SPEC["field"], BENCHMARK_SPEC["physics"]
    x = (np.arange(spec["nx"]) + .5) * physics["width"] / spec["nx"]
    z = (np.arange(spec["nz"]) + .5) * physics["thickness"] / spec["nz"]
    tt, zz, xx = np.meshgrid(np.asarray(spec["times_s"], dtype=float), z, x, indexing="ij")
    return np.column_stack((xx.ravel(), zz.ravel(), tt.ravel())), tt.shape


def _schedule(trace):
    def bath(t):
        return BENCHMARK_SPEC["initial_temperature_kelvin"] + trace["bath_drop_kelvin"] * math.expm1(
            -t / trace["ramp_seconds"])

    def flux(t):
        return -trace["top_flux_watts_per_square_metre"] * math.expm1(-t / trace["ramp_seconds"])

    return bath, flux


def _reference_trajectory(h, trace, *, canonical=True):
    numerical = BENCHMARK_SPEC["reference" if canonical else "conventional"]
    bath, flux = _schedule(trace)
    times = np.unique(np.concatenate((
        np.asarray(BENCHMARK_SPEC["field"]["times_s"], dtype=float),
        np.asarray(BENCHMARK_SPEC["sensors"]["times_s"], dtype=float),
    )))
    return PlanarContactModel(
        PlanarContactConfig(**BENCHMARK_SPEC["physics"]), nx=numerical["nx"],
        nz=numerical["nz"], contact_conductance=float(h),
    ).solve(times, initial_temperature=BENCHMARK_SPEC["initial_temperature_kelvin"],
            te_temperature=bath, top_heat_flux=flux, max_step=numerical["max_step_s"])


def _sample_trajectory(trajectory, coordinates):
    return trajectory.sample(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])


def _observations(trajectory, seed):
    full, mask = _observation_coordinates()
    coordinates = full[mask]
    clean = _sample_trajectory(trajectory, coordinates)
    sigma = BENCHMARK_SPEC["sensors"]["noise_standard_deviation_kelvin"]
    noise = np.random.default_rng(seed).normal(0.0, sigma, len(coordinates))
    return PlanarObservations(coordinates, clean + noise, sigma), {
        "full_coordinates": full, "retained_mask": mask, "coordinates": coordinates,
        "clean_temperature": clean, "noise": noise, "temperature": clean + noise,
        "standard_deviation": np.asarray(sigma), "noise_seed": np.asarray(seed),
    }


def _independent_gate(report):
    """Require each family and both mixed-bottom regions independently."""
    summary = report.get("independent_residuals", {})
    values = summary.get("normalized_rms", {})
    counts = summary.get("bottom_region_point_counts", {})
    threshold = BENCHMARK_SPEC["gates"]["independent_residual_rms"]
    populated = all(counts.get(name, 0) > 0 for name in ("bottom_contact", "bottom_insulated"))
    valid = populated and all(
        isinstance(values.get(name), (int, float)) and math.isfinite(values[name])
        for name in _RESIDUAL_FAMILIES
    )
    passed = bool(valid and all(values[name] <= threshold for name in _RESIDUAL_FAMILIES))
    return {"passes": passed, "populated_bottom_regions": populated,
            "required_families": list(_RESIDUAL_FAMILIES),
            "normalized_rms": {name: values.get(name) for name in _RESIDUAL_FAMILIES},
            "threshold": threshold}


def _candidate_quality(record):
    report = record.get("report", {})
    observation = report.get("normalized_observation_mse")
    gate = _independent_gate(report)
    values = gate["normalized_rms"]
    valid = (record.get("status") == "completed" and isinstance(observation, (int, float))
             and math.isfinite(observation) and gate["populated_bottom_regions"]
             and all(isinstance(v, (int, float)) and math.isfinite(v) for v in values.values()))
    if not valid:
        return False, math.inf, math.inf
    ratios = [observation / BENCHMARK_SPEC["gates"]["normalized_observation_mse"]]
    ratios.extend(v / gate["threshold"] for v in values.values())
    violation = max(0.0, max(ratios) - 1.0)
    return bool(gate["passes"] and ratios[0] <= 1.0), violation, observation + sum(values.values())


def _select_development(records):
    if not records:
        raise ValueError("development selection requires the predeclared candidate records")
    passing = [r for r in records if _candidate_quality(r)[0]]
    if passing:
        chosen = min(passing, key=lambda r: (
            r["adam_epochs"], _candidate_quality(r)[2], r["physics_weight"], r["candidate_id"],
        ))
    else:
        chosen = min(records, key=lambda r: (
            _candidate_quality(r)[1], r["adam_epochs"], _candidate_quality(r)[2],
            r["physics_weight"], r["candidate_id"],
        ))
    return {"candidate_id": chosen["candidate_id"], "adam_epochs": chosen["adam_epochs"],
            "physics_weight": chosen["physics_weight"], "development_gate_passed": bool(passing),
            "normalized_gate_violation": _json_safe(_candidate_quality(chosen)[1]),
            "selection_uses_hidden_fields_or_true_h": False,
            "rule": BENCHMARK_SPEC["development_selection"]}


def _config(adam_epochs):
    n, trace = BENCHMARK_SPEC["neural"], BENCHMARK_SPEC["training_trace"]
    return PlanarPINNConfig(
        width=n["width"], layers=n["layers"], adam_epochs=adam_epochs,
        lbfgs_iterations=n["lbfgs_iterations"], interior_points=n["interior_points"],
        boundary_points_per_side=n["boundary_points_per_side"], learning_rate=n["learning_rate"],
        duration=BENCHMARK_SPEC["duration_seconds"],
        initial_temperature=BENCHMARK_SPEC["initial_temperature_kelvin"],
        temperature_scale=BENCHMARK_SPEC["temperature_scale_kelvin"],
        h_bounds=tuple(BENCHMARK_SPEC["h_bounds_w_m2k"]),
        initial_h=BENCHMARK_SPEC["initial_h_w_m2k"], **trace,
    )


def _failure(error):
    return {"exception_type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc()}


def _fit_neural(directory, observations, *, config, mode, weight, seeds, known_h=None, progress=None):
    """The training boundary accepts only observed points and known inputs."""
    directory.mkdir()
    started, cpu = perf_counter(), process_time()
    record = {"mode": mode, "physics_weight": weight, "config": asdict(config),
              "seeds": seeds, "oracle_truth_information": known_h,
              "checkpoint_available": False}
    try:
        model, report = train_planar_contact_pinn(
            observations, physics=PlanarContactConfig(**BENCHMARK_SPEC["physics"]), config=config,
            mode=mode, physics_weight=weight, neural_seed=seeds["neural_initialization"],
            collocation_seed=seeds["collocation"], diagnostic_seed=seeds["diagnostics"],
            known_h=known_h, progress=progress,
        )
        torch.save({"state_dict": model.state_dict(), "config": asdict(config),
                    "mode": mode, "known_h": known_h}, directory / "final_checkpoint.pt")
        record.update(status="completed", report=report, checkpoint_available=True)
        _write_json(directory / "training_report.json", report)
    except Exception as error:
        model = None
        record.update(status="failed", failure=_failure(error))
        _write_json(directory / "failure.json", record["failure"])
    record.update(wall_seconds=perf_counter()-started, cpu_seconds=process_time()-cpu)
    _write_json(directory / "fit_record.json", record)
    return model, record


def _fit_conventional(observations):
    started, cpu = perf_counter(), process_time()
    evaluations = []

    def residual(log_h):
        h = math.exp(float(log_h[0]))
        trajectory = _reference_trajectory(h, BENCHMARK_SPEC["training_trace"], canonical=False)
        prediction = _sample_trajectory(trajectory, observations.coordinates)
        value = (prediction - observations.temperature) / observations.standard_deviation
        evaluations.append({"h_w_m2k": h, "normalized_observation_mse": float(np.mean(value**2))})
        return value

    numerical = BENCHMARK_SPEC["conventional"]
    result = least_squares(
        residual, [math.log(BENCHMARK_SPEC["initial_h_w_m2k"])],
        bounds=np.log(BENCHMARK_SPEC["h_bounds_w_m2k"]), max_nfev=numerical["max_nfev"],
        xtol=numerical["xtol"], ftol=numerical["ftol"], gtol=numerical["gtol"],
    )
    h = math.exp(float(result.x[0]))
    return {"status": "completed" if result.success else "optimizer_failed",
            "contact_estimate": h, "normalized_observation_mse": float(np.mean(result.fun**2)),
            "optimizer_success": bool(result.success), "optimizer_status": int(result.status),
            "optimizer_message": str(result.message), "nfev": int(result.nfev),
            "objective_evaluations": evaluations, "wall_seconds": perf_counter()-started,
            "cpu_seconds": process_time()-cpu, "truth_information_used_in_fit": False}


def _sensitivity():
    coordinates, mask = _observation_coordinates()
    coordinates = coordinates[mask]
    spec = BENCHMARK_SPEC["sensitivity"]
    sigma = BENCHMARK_SPEC["sensors"]["noise_standard_deviation_kelvin"]
    lower, upper = np.log(BENCHMARK_SPEC["h_bounds_w_m2k"])
    points, derivatives = [], []
    for h in spec["contacts_w_m2k"]:
        log_h = math.log(h)
        left = max(lower, log_h - spec["log_h_difference_step"])
        right = min(upper, log_h + spec["log_h_difference_step"])
        predictions = []
        for value in (left, right):
            trajectory = _reference_trajectory(math.exp(value), BENCHMARK_SPEC["training_trace"],
                                               canonical=False)
            predictions.append(_sample_trajectory(trajectory, coordinates))
        derivative = (predictions[1] - predictions[0]) / (right - left)
        information = float(np.sum((derivative / sigma)**2))
        halfwidth = spec["normal_95_multiplier"] / math.sqrt(information) if information > 0 else math.inf
        points.append({"h_w_m2k": h, "log_h_fisher_information": information,
                       "optimistic_95_log_h_halfwidth": _json_safe(halfwidth),
                       "passes": halfwidth <= spec["maximum_95_log_h_halfwidth"],
                       "difference_log_h_endpoints": [float(left), float(right)]})
        derivatives.append(derivative)
    return {"points": points, "passes": all(p["passes"] for p in points),
            "maximum_95_log_h_halfwidth": spec["maximum_95_log_h_halfwidth"],
            "interpretation": spec["interpretation"], "failure_action": spec["failure_action"]}, {
                "coordinates": coordinates, "d_temperature_d_log_h": np.asarray(derivatives),
            }


def _objective_profile(observations):
    log_grid = np.linspace(*np.log(BENCHMARK_SPEC["h_bounds_w_m2k"]),
                           BENCHMARK_SPEC["development_profile"]["points"])
    values = []
    for log_h in log_grid:
        trajectory = _reference_trajectory(math.exp(log_h), BENCHMARK_SPEC["training_trace"], canonical=False)
        residual = (_sample_trajectory(trajectory, observations.coordinates) - observations.temperature) / (
            observations.standard_deviation)
        values.append(float(np.mean(residual**2)))
    return {"h_w_m2k": np.exp(log_grid).tolist(), "normalized_observation_mse": values,
            "interpretation": BENCHMARK_SPEC["development_profile"]["interpretation"],
            "used_to_select_neural_configuration": False}


def _field_metrics(prediction, truth):
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    if prediction.shape != truth.shape or not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("field predictions and truth require equal shapes and finite values")
    error = prediction - truth
    return {"rmse_kelvin": float(np.sqrt(np.mean(error**2))),
            "max_abs_kelvin": float(np.max(np.abs(error))),
            "weighting": BENCHMARK_SPEC["field"]["weighting"]}


def _score(predicted_field, true_field, observation_mse, h, true_h, *, report=None, method):
    metrics = _field_metrics(predicted_field, true_field)
    gate = BENCHMARK_SPEC["gates"]
    h_error = abs(h / true_h - 1.0) if h is not None else None
    initial_error = abs(BENCHMARK_SPEC["initial_h_w_m2k"] - true_h)
    fraction_reduction = ((initial_error - abs(h - true_h)) / initial_error
                          if h is not None and initial_error > 0 else None)
    residual = _independent_gate(report) if report is not None else None
    field_passes = (metrics["rmse_kelvin"] <= gate["hidden_field_rmse_kelvin"]
                    and metrics["max_abs_kelvin"] <= gate["hidden_field_max_kelvin"])
    observation_passes = math.isfinite(observation_mse) and observation_mse <= gate["normalized_observation_mse"]
    h_passes = h_error is not None and h_error <= gate["relative_contact_error"]
    meaningful_recovery = bool(h_passes and fraction_reduction is not None
                              and fraction_reduction >= gate["minimum_fraction_absolute_error_reduction_from_h0"])
    inverse_eligible = method != "data_only"
    joint = field_passes and observation_passes and meaningful_recovery and (
        residual is None or residual["passes"])
    return {"hidden_field": metrics, "normalized_observation_mse": observation_mse,
            "contact_estimate": h, "relative_contact_error": h_error,
            "parameter_initial_relative_error": initial_error / true_h,
            "fraction_absolute_error_reduction": fraction_reduction,
            "independent_residual_gate": residual,
            "gates": {"hidden_field": bool(field_passes), "observations": bool(observation_passes),
                      "contact": bool(h_passes) if inverse_eligible else None,
                      "meaningful_parameter_recovery": meaningful_recovery if inverse_eligible else None,
                      "independent_residuals": residual["passes"] if residual is not None else None,
                      "joint": bool(joint) if inverse_eligible else None},
            "truth_informed_oracle": method == "known_h_oracle",
            "contact_inference_eligible": method in ("inverse_pinn", "conventional")}


def _save_truth(directory, h):
    directory.mkdir()
    coordinates, shape = _field_coordinates()
    train = _reference_trajectory(h, BENCHMARK_SPEC["training_trace"])
    holdout = _reference_trajectory(h, BENCHMARK_SPEC["holdout_trace"])
    training_field = _sample_trajectory(train, coordinates)
    holdout_field = _sample_trajectory(holdout, coordinates)
    np.savez_compressed(directory / "hidden_reference_fields.npz", coordinates=coordinates,
                        shape=np.asarray(shape), training_temperature=training_field.reshape(shape),
                        holdout_temperature=holdout_field.reshape(shape))
    _write_json(directory / "reference_diagnostics.json", {
        "true_contact_w_m2k": h, "numerical_config": BENCHMARK_SPEC["reference"],
        "training_energy_residual_max_j": float(abs(train.energy_balance_residual).max()),
        "holdout_energy_residual_max_j": float(abs(holdout.energy_balance_residual).max()),
    })
    return train, training_field, holdout_field


def _apply_holdout_gate(score, transfer_metrics):
    gate = BENCHMARK_SPEC["gates"]
    if transfer_metrics is None:
        score["gates"]["holdout_transfer"] = None
    else:
        passed = (transfer_metrics["rmse_kelvin"] <= gate["holdout_field_rmse_kelvin"]
                  and transfer_metrics["max_abs_kelvin"] <= gate["holdout_field_max_kelvin"])
        score["gates"]["holdout_transfer"] = bool(passed)
        score["gates"]["joint"] = bool(score["gates"]["joint"] and passed)
    return score


def _score_and_save(directory, model, record, observations, true_field, holdout_truth, true_h, *, method):
    started = perf_counter()
    coordinates, shape = _field_coordinates()
    if model is None and method not in ("conventional", "frozen_h0"):
        score = {"status": "fit_failed", "failure": record.get("failure"), "gates": {"joint": False}}
        _write_json(directory / "scores.json", score)
        return score
    if method in ("conventional", "frozen_h0"):
        h, report = record["contact_estimate"], None
        forward = _reference_trajectory(h, BENCHMARK_SPEC["training_trace"], canonical=False)
        field = _sample_trajectory(forward, coordinates)
        observation_predictions = _sample_trajectory(forward, observations.coordinates)
        mse = record["normalized_observation_mse"]
    else:
        report = record["report"]
        h = report["contact_estimate"]
        field = predict_planar_temperature(model, coordinates)
        observation_predictions = predict_planar_temperature(model, observations.coordinates)
        mse = report["normalized_observation_mse"]
    score = _score(field, true_field, mse, h, true_h, report=report, method=method)
    saved = {"field_coordinates": coordinates, "field_shape": np.asarray(shape),
             "training_field_temperature": field.reshape(shape),
             "observation_coordinates": observations.coordinates,
             "observation_prediction": observation_predictions}
    if h is not None:
        transferred = _reference_trajectory(h, BENCHMARK_SPEC["holdout_trace"], canonical=False)
        transfer_field = _sample_trajectory(transferred, coordinates)
        saved["holdout_forward_temperature"] = transfer_field.reshape(shape)
        score["holdout_transfer"] = {"mechanism": BENCHMARK_SPEC["holdout_transfer"],
                                     "h_used_w_m2k": h,
                                     "field": _field_metrics(transfer_field, holdout_truth)}
        _apply_holdout_gate(score, score["holdout_transfer"]["field"])
    else:
        score["holdout_transfer"] = {"available": False, "reason": "data-only method emits no contact estimate"}
        _apply_holdout_gate(score, None)
    np.savez_compressed(directory / "predictions.npz", **saved)
    score["status"] = "completed" if record["status"] == "completed" else record["status"]
    if score["status"] != "completed":
        score["gates"]["joint"] = False
    score["evaluation_wall_seconds"] = perf_counter() - started
    score["wall_seconds"] = record.get("wall_seconds", 0.0) + score["evaluation_wall_seconds"]
    _write_json(directory / "scores.json", score)
    return score


def _manifest(directory):
    return {str(path.relative_to(directory)): {"sha256": _digest(path), "bytes": path.stat().st_size}
            for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "artifact_manifest.json"}


def _compact_fit(record):
    result = dict(record)
    if "report" in result:
        result["report"] = {key: value for key, value in result["report"].items() if key != "history"}
    return result


def _method_artifacts(root, directory):
    names = {"predictions": "predictions.npz", "checkpoint": "final_checkpoint.pt",
             "fit_record": "fit_record.json", "training_report": "training_report.json",
             "scores": "scores.json", "failure": "failure.json", "scoring_failure": "scoring_failure.json"}
    return {name: str((directory / filename).relative_to(root)) for name, filename in names.items()
            if (directory / filename).is_file()}


def _optimizer_progress(emit, stage, **context):
    # Trainer events contain their own stage/mode keys. Keep that namespace
    # nested so progress reporting cannot interrupt an optimizer attempt.
    return lambda event: emit(stage, "running", optimizer_event=event, **context)


def run_planar_benchmark(output_directory, *, progress=None):
    """Run the fixed campaign once, retaining every attempt and failure.

    The destination must not exist.  No evaluation-dependent retries or
    configuration changes occur.  Scientific failure is returned and saved;
    an existing destination raises ``FileExistsError`` before any mutation.
    """
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=False)
    started, cpu = perf_counter(), process_time()
    _write_json(directory / "benchmark_spec.json", BENCHMARK_SPEC)
    dependencies = [Path(__file__), Path(__file__).parents[1] / "physics/planar_contact.py",
                    Path(__file__).parents[1] / "pinn/planar_contact_inverse.py"]
    source_digests = {str(path.relative_to(Path(__file__).parents[2])): _digest(path) for path in dependencies}
    _write_json(directory / "source_digests.json", source_digests)
    result = {"campaign": BENCHMARK_SPEC["campaign"], "status": "running",
              "specification_sha256": _digest(directory / "benchmark_spec.json"),
              "source_digests": source_digests, "development": {}, "trials": [],
              "artifact_manifest": "artifact_manifest.json", "failures_retained": True,
              "contact_recovery_claim_allowed": False}

    def emit(stage, status, **details):
        if progress:
            progress({"stage": stage, "status": status, **details})

    try:
        seed_audit = _seed_audit()
        _write_json(directory / "seed_audit.json", seed_audit)
        result["seed_audit"] = seed_audit
        emit("reference_verification", "started")
        verification = verify_planar_contact_reference()
        _write_json(directory / "reference_verification.json", verification)
        result["reference_verification_passed"] = verification["passes"]
        emit("reference_verification", "passed" if verification["passes"] else "failed")
        if not verification["passes"]:
            result["status"] = "blocked_reference"
        else:
            sensitivity, arrays = _sensitivity()
            _write_json(directory / "prefit_sensitivity.json", sensitivity)
            np.savez_compressed(directory / "prefit_sensitivity.npz", **arrays)
            result["sensitivity_gate_passed"] = sensitivity["passes"]
            emit("prefit_sensitivity", "passed" if sensitivity["passes"] else "failed_continuing_field_study")
            dev = directory / "development"
            dev.mkdir()
            train, _, _ = _save_truth(dev / "reference", BENCHMARK_SPEC["development_contact_w_m2k"])
            dev_seeds = _paired_seeds("development", 0)
            observations, arrays = _observations(train, dev_seeds["observation_noise"])
            np.savez_compressed(dev / "observations.npz", **arrays)
            _write_json(dev / "objective_profile.json", _objective_profile(observations))
            candidates = []
            for epochs in BENCHMARK_SPEC["neural"]["development_adam_epochs"]:
                for weight in BENCHMARK_SPEC["neural"]["development_physics_weights"]:
                    candidate_id = f"adam{epochs}_weight{weight:g}"
                    emit("development", "started", candidate=candidate_id)
                    _, record = _fit_neural(
                        dev / candidate_id, observations, config=_config(epochs), mode="inverse",
                        weight=weight, seeds=dev_seeds,
                        progress=_optimizer_progress(emit, "development_optimizer", candidate=candidate_id),
                    )
                    record.update(candidate_id=candidate_id, adam_epochs=epochs)
                    candidates.append(_compact_fit(record))
                    emit("development", record["status"], candidate=candidate_id)
            selection = _select_development(candidates)
            _write_json(dev / "selection.json", selection)
            result["development"] = {"selection": selection, "candidates": candidates}
            result["development_selection"] = selection
            config = _config(selection["adam_epochs"])
            eval_directory = directory / "evaluation"
            eval_directory.mkdir()
            true_h = BENCHMARK_SPEC["evaluation_contact_w_m2k"]
            evaluation_truth, hidden_field, holdout_truth = _save_truth(eval_directory / "reference", true_h)
            for trial in range(BENCHMARK_SPEC["evaluation_trials"]):
                trial_dir = eval_directory / f"trial_{trial:02d}"
                trial_dir.mkdir()
                seeds = _paired_seeds("evaluation", trial)
                observations, arrays = _observations(evaluation_truth, seeds["observation_noise"])
                np.savez_compressed(trial_dir / "observations.npz", **arrays)
                trial_result = {"index": trial, "seeds": seeds, "methods": {}}
                for method, mode in (("inverse_pinn", "inverse"), ("data_only", "data_only"),
                                     ("known_h_oracle", "known_contact")):
                    emit("evaluation", "started", trial=trial, method=method)
                    model, record = _fit_neural(
                        trial_dir / method, observations, config=config, mode=mode,
                        weight=selection["physics_weight"], seeds=seeds,
                        known_h=true_h if method == "known_h_oracle" else None,
                        progress=_optimizer_progress(emit, "evaluation_optimizer", trial=trial, method=method),
                    )
                    try:
                        score = _score_and_save(trial_dir / method, model, record, observations,
                                                hidden_field, holdout_truth, true_h, method=method)
                    except Exception as error:
                        score = {"status": "scoring_failed", "failure": _failure(error), "gates": {"joint": False}}
                        _write_json(trial_dir / method / "scoring_failure.json", score)
                    trial_result["methods"][method] = {
                        "fit": _compact_fit(record), "metrics": score,
                        "artifacts": _method_artifacts(directory, trial_dir / method)}
                    emit("evaluation", score["status"], trial=trial, method=method)
                conventional_dir = trial_dir / "conventional"
                conventional_dir.mkdir()
                try:
                    conventional = _fit_conventional(observations)
                    _write_json(conventional_dir / "fit_record.json", conventional)
                    score = _score_and_save(conventional_dir, None, conventional, observations,
                                            hidden_field, holdout_truth, true_h, method="conventional")
                    trial_result["methods"]["conventional"] = {"fit": conventional, "metrics": score}
                except Exception as error:
                    failure = {"status": "failed", "failure": _failure(error), "gates": {"joint": False}}
                    _write_json(conventional_dir / "failure.json", failure)
                    trial_result["methods"]["conventional"] = {"fit": failure, "metrics": failure}
                trial_result["methods"]["conventional"]["artifacts"] = _method_artifacts(directory, conventional_dir)
                frozen_dir = trial_dir / "frozen_h0"
                frozen_dir.mkdir()
                try:
                    frozen_started = perf_counter()
                    frozen_h = BENCHMARK_SPEC["initial_h_w_m2k"]
                    frozen_trajectory = _reference_trajectory(frozen_h, BENCHMARK_SPEC["training_trace"], canonical=False)
                    frozen_prediction = _sample_trajectory(frozen_trajectory, observations.coordinates)
                    frozen = {"status": "completed", "contact_estimate": frozen_h,
                              "normalized_observation_mse": float(np.mean(((frozen_prediction-observations.temperature)
                                                               / observations.standard_deviation)**2)),
                              "fit_performed": False, "wall_seconds": perf_counter()-frozen_started,
                              "truth_information_used_in_fit": False}
                    _write_json(frozen_dir / "fit_record.json", frozen)
                    frozen_score = _score_and_save(frozen_dir, None, frozen, observations, hidden_field,
                                                  holdout_truth, true_h, method="frozen_h0")
                    trial_result["methods"]["frozen_h0"] = {"fit": frozen, "metrics": frozen_score}
                except Exception as error:
                    failure = {"status": "failed", "failure": _failure(error), "gates": {"joint": False}}
                    _write_json(frozen_dir / "failure.json", failure)
                    trial_result["methods"]["frozen_h0"] = {"fit": failure, "metrics": failure}
                trial_result["methods"]["frozen_h0"]["artifacts"] = _method_artifacts(directory, frozen_dir)
                digests = [trial_result["methods"][m]["fit"].get("report", {}).get("initial_field_digest")
                           for m in ("inverse_pinn", "data_only", "known_h_oracle")]
                trial_result["paired_initial_fields_verified"] = bool(all(digests) and len(set(digests)) == 1)
                _write_json(trial_dir / "trial_result.json", trial_result)
                result["trials"].append(trial_result)
            result["status"] = "completed"
            result["summary"] = {}
            for method in BENCHMARK_SPEC["evaluation_methods"]:
                scores = [trial["methods"][method]["metrics"] for trial in result["trials"]]
                result["summary"][method] = {
                    "trials": len(scores), "failed_trials": sum(s.get("status") != "completed" for s in scores),
                    "joint_gate_passes": sum(s.get("gates", {}).get("joint") is True for s in scores),
                    "reconstruction_gate_passes": sum(s.get("status") == "completed"
                        and s.get("gates", {}).get("hidden_field") is True
                        and s.get("gates", {}).get("observations") is True for s in scores),
                    "joint_gate_denominator": BENCHMARK_SPEC["evaluation_trials"],
                    "truth_informed_oracle": method == "known_h_oracle",
                    "contact_inference_eligible": method in ("inverse_pinn", "conventional"),
                }
            result["contact_recovery_claim_allowed"] = bool(
                sensitivity["passes"] and selection["development_gate_passed"]
                and result["summary"]["inverse_pinn"]["joint_gate_passes"] == BENCHMARK_SPEC["evaluation_trials"]
                and all(t["paired_initial_fields_verified"] for t in result["trials"])
            )
    except Exception as error:
        result.update(status="failed", failure=_failure(error), contact_recovery_claim_allowed=False)
        _write_json(directory / "failure.json", result["failure"])
    result.update(wall_seconds=perf_counter()-started, cpu_seconds=process_time()-cpu)
    _write_json(directory / "result.json", result)
    _write_json(directory / "artifact_manifest.json", _manifest(directory))
    emit("benchmark", result["status"])
    return _json_safe(result)
