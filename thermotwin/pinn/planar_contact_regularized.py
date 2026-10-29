"""Continuous stationary readouts for the unchanged graded log-off PINN field.

The diagonal scaling and ridge strength are calibrated once at the original
common initialization. They stay fixed as hidden weights move. Full-rank
augmented QR replaces hard spectral truncation and the former head fallback.
Only the outer regularized objective decides whether a hidden step is accepted.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from time import perf_counter

import numpy as np
from scipy.linalg import qr, solve_triangular
import torch

from ..physics.planar_contact import PlanarContactConfig
from .planar_contact_continuity import _observation_tensors, linear_system, objective_continuity
from .planar_contact_graded import _array_digest, graded_point_metadata, graded_training_points
from .planar_contact_optimized import _finite_optimizer_state, _step_counts
from .planar_contact_step_control import (
    StepControlConfig, StepControlPINN, StepControlState, _restart_moments,
    _set_learning_rate, diagnose_step_control, matched_step_control_model,
    predict_step_control,
)


@dataclass(frozen=True)
class RegularizedConfig(StepControlConfig):
    adam_epochs: int = 320
    learning_rate: float = .0064
    controller_max_failures: int = 8
    regularization_relative_scale: float = 1.e-7
    head_stationarity_tolerance: float = 1.e-8

    def __post_init__(self):
        super().__post_init__()
        if self.regularization_relative_scale != self.rank_rcond:
            raise ValueError("the fixed soft knee must equal the original cutoff ratio")
        if not math.isfinite(self.head_stationarity_tolerance) or self.head_stationarity_tolerance <= 0:
            raise ValueError("head stationarity tolerance must be finite and positive")


class RegularizedPINN(StepControlPINN):
    def __init__(self, physics=PlanarContactConfig(), config=RegularizedConfig(), *, known_h, mode="known_contact"):
        super().__init__(physics, config, known_h=known_h, mode=mode)
        self.register_buffer("readout_scale", torch.ones_like(self.coefficients))
        self.register_buffer("readout_lambda", torch.zeros((), dtype=torch.float64))
        self.register_buffer("readout_calibrated", torch.tensor(False))


_READOUT_BUFFERS = {"readout_scale", "readout_lambda", "readout_calibrated"}


def load_field_state(model, state_dict):
    """Load only the old field, preserving an already bound ridge calibration."""
    if set(state_dict) != set(model.state_dict())-_READOUT_BUFFERS:
        raise ValueError("field state keys differ from the frozen architecture")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if set(missing) != _READOUT_BUFFERS or unexpected:
        raise ValueError("field state did not load exactly")


def matched_regularized_model(physics, config, known_h, neural_seed):
    base = {key: value for key, value in asdict(config).items() if key in StepControlConfig.__dataclass_fields__}
    source, initialization = matched_step_control_model(physics, StepControlConfig(**base), known_h, neural_seed)
    model = RegularizedPINN(physics, config, known_h=known_h)
    load_field_state(model, source.state_dict())
    model.initialization_metadata = initialization
    return model, initialization


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _bindings(model, points, observations):
    return {"initial_field_state": {name: _array_digest(value.detach().cpu().numpy())
            for name, value in model.state_dict().items() if name not in _READOUT_BUFFERS},
        "points": {name: _array_digest(value) for name, value in points.items()},
        "observation_coordinates": _array_digest(observations.coordinates),
        "observation_temperature": _array_digest(observations.temperature),
        "observation_standard_deviation": float(observations.standard_deviation),
        "physics": asdict(model.physics), "config": asdict(model.config), "known_h": float(model.known_h)}


def calibrate_readout(model, points, observations):
    """One initial matrix build and spectral calculation; no head solve."""
    if bool(model.readout_calibrated):
        raise RuntimeError("readout calibration is immutable once installed")
    matrix, rhs, _ = linear_system(model, points, observations)
    scales = np.maximum(np.linalg.norm(matrix, axis=0), 1.e-12)
    singular = np.linalg.svd(matrix/scales, compute_uv=False)
    strength = model.config.regularization_relative_scale*float(singular[0])
    if not np.isfinite(scales).all() or not math.isfinite(strength) or strength <= 0:
        raise FloatingPointError("nonfinite or empty initial ridge calibration")
    metadata = {"schema": "fixed_initial_column_scale_ridge_v1", "column_scale": scales.tolist(),
        "column_scale_sha256": _array_digest(scales), "fixed_lambda": strength,
        "relative_scale": model.config.regularization_relative_scale, "initial_sigma_max": float(singular[0]),
        "initial_matrix_sha256": _array_digest(matrix), "initial_rhs_sha256": _array_digest(rhs),
        "initial_matrix_shape": list(matrix.shape), "bindings": _bindings(model, points, observations),
        "calibration_matrix_builds": 1, "calibration_spectral_calculations": 1,
        "formula": "D_j=max(||A0_j||2,1e-12); lambda=original_cutoff_ratio*smax(A0/D)",
        "frozen_for_entire_fit": True, "hidden_parameters_differentiated_through_calibration": False}
    metadata["sha256"] = _digest(metadata)
    apply_readout_calibration(model, metadata, points, observations)
    return metadata


def apply_readout_calibration(model, metadata, points, observations):
    """Install prepared arrays after fresh-state checks, without building A0."""
    if bool(model.readout_calibrated):
        raise RuntimeError("readout calibration is immutable once installed")
    payload = {key: value for key, value in metadata.items() if key != "sha256"}
    if metadata.get("sha256") != _digest(payload) or metadata.get("bindings") != _bindings(model, points, observations):
        raise ValueError("frozen readout calibration identity differs")
    scales = np.asarray(metadata["column_scale"], dtype=np.float64)
    strength = float(metadata["fixed_lambda"])
    if (metadata.get("schema") != "fixed_initial_column_scale_ridge_v1"
        or scales.shape != (len(model.coefficients),) or not np.isfinite(scales).all() or np.any(scales < 1.e-12)
        or _array_digest(scales) != metadata["column_scale_sha256"] or not math.isfinite(strength) or strength <= 0
        or metadata["relative_scale"] != model.config.regularization_relative_scale
        or strength != metadata["relative_scale"]*metadata["initial_sigma_max"]
        or metadata.get("frozen_for_entire_fit") is not True):
        raise ValueError("invalid frozen ridge arrays or strength")
    with torch.no_grad():
        model.readout_scale.copy_(torch.tensor(scales, dtype=torch.float64))
        model.readout_lambda.fill_(strength)
        model.readout_calibrated.fill_(True)
    model.readout_calibration = deepcopy(metadata)


def objective_regularized(model, points, observations):
    if not bool(model.readout_calibrated):
        raise RuntimeError("a frozen readout calibration is required")
    original, obs, physical, families = objective_continuity(model, points, observations)
    penalty = model.readout_lambda**2*torch.sum((model.readout_scale*model.coefficients)**2)
    total = original+penalty
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("nonfinite regularized objective")
    return total, original, penalty, obs, physical, families


def project_regularized(model, points, observations):
    """Unconditionally install a unique stationary head using full-rank QR."""
    started = perf_counter()
    before = float(objective_regularized(model, points, observations)[0].detach())
    original_head = model.coefficients.detach().clone()
    try:
        matrix, rhs, metadata = linear_system(model, points, observations)
        scales = model.readout_scale.detach().cpu().numpy()
        strength = float(model.readout_lambda)
        columns = len(scales)
        augmented = np.vstack((matrix/scales, strength*np.eye(columns)))
        target = np.concatenate((rhs, np.zeros(columns)))
        if not np.isfinite(augmented).all() or not np.isfinite(target).all():
            raise FloatingPointError("nonfinite augmented readout system")
        orthogonal, triangular, pivots = qr(augmented, mode="economic", pivoting=True, check_finite=False)
        diagonal = abs(np.diag(triangular))
        if len(diagonal) != columns or not np.isfinite(diagonal).all() or np.any(diagonal == 0):
            raise FloatingPointError("augmented QR readout is not full rank")
        solution = np.empty(columns)
        solution[pivots] = solve_triangular(triangular, orthogonal.T@target, check_finite=False)
        proposed = solution/scales
        residual = augmented@solution-target
        normal = augmented.T@residual
        denominator = max(1., float(np.linalg.norm(augmented.T@target)))
        stationarity = float(np.linalg.norm(normal))/denominator
        if not np.isfinite(proposed).all() or not math.isfinite(stationarity) or stationarity > model.config.head_stationarity_tolerance:
            raise FloatingPointError("augmented readout failed its stationarity check")
        with torch.no_grad():
            model.coefficients.copy_(torch.tensor(proposed, dtype=torch.float64))
        total, old, penalty, _, _, _ = objective_regularized(model, points, observations)
        after, unpenalized, regularizer = float(total.detach()), float(old.detach()), float(penalty.detach())
        algebraic = float(np.sum(residual**2))
        if not math.isclose(after, algebraic, rel_tol=1.e-8, abs_tol=1.e-8):
            raise FloatingPointError("strong and augmented algebraic objectives disagree")
        return {"accepted": True, "head_policy": "unconditional unique stationary ridge readout",
            "objective_before": before, "objective_after": after, "objective_proposed": after,
            "unpenalized_objective": unpenalized, "regularization_penalty": regularizer,
            "algebraic_objective": algebraic, "algebraic_objective_abs_difference": abs(after-algebraic),
            "rank": columns, "columns": columns, "rows": len(rhs), "augmented_rows": len(target),
            "hard_singular_value_truncation": False, "rank_rcond": None, "resolved_condition": None,
            "qr_diagonal_ratio": float(diagonal.max()/diagonal.min()),
            "condition_metric": "R diagonal ratio is not a spectral condition estimate",
            "stationarity_relative": stationarity, "stationarity_absolute_normal_residual": float(np.linalg.norm(normal)),
            "stationarity_augmented_gradient_l2": 2*float(np.linalg.norm(normal)),
            "stationarity_denominator": denominator, "augmented_residual_l2": float(np.linalg.norm(residual)),
            "stationarity_definition": "||C.T@(C@z-d)||2/max(1,||C.T@d||2)",
            "stationarity_tolerance": model.config.head_stationarity_tolerance,
            "coefficient_norm_returned": float(np.linalg.norm(proposed)),
            "scaled_coefficient_norm": float(np.linalg.norm(solution)), "fixed_lambda": strength,
            "column_scale_sha256": model.readout_calibration["column_scale_sha256"],
            "solver": "float64 augmented column-scaled pivoted QR and full-rank triangular solve; no normal equations or spectral cutoff",
            "system": metadata, "head_solve_calls": 1, "wall_seconds": perf_counter()-started}
    except BaseException:
        with torch.no_grad():
            model.coefficients.copy_(original_head)
        raise


def transactional_adam_step(model, points, observations, optimizer, parameters=None):
    """A new objective/solve path with the frozen optimizer rollback contract."""
    parameters = list(model.networks.parameters()) if parameters is None else list(parameters)
    hidden = [parameter.detach().clone() for parameter in parameters]
    gradients = [None if parameter.grad is None else parameter.grad.detach().clone() for parameter in parameters]
    head, optimizer_before = model.coefficients.detach().clone(), deepcopy(optimizer.state_dict())
    counts_before = _step_counts(optimizer)
    def restore_model():
        with torch.no_grad():
            for parameter, value in zip(parameters, hidden):
                parameter.copy_(value)
            model.coefficients.copy_(head)
    def restore_transaction():
        restore_model()
        optimizer.load_state_dict(deepcopy(optimizer_before))
        for parameter, value in zip(parameters, gradients):
            parameter.grad = None if value is None else value.clone()
    try:
        optimizer.zero_grad()
        total = objective_regularized(model, points, observations)[0]
        before = float(total.detach())
        total.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, model.config.gradient_clip))
        if not math.isfinite(gradient_norm):
            raise FloatingPointError("nonfinite regularized hidden gradient")
        optimizer.step()
        finite_state = _finite_optimizer_state(optimizer)
        counts_proposed = _step_counts(optimizer) if finite_state else []
        displacement = [parameter.detach()-value for parameter, value in zip(parameters, hidden)]
        finite_direction = all(bool(torch.all(torch.isfinite(value))) for value in displacement)
        accepted, after, accepted_scale, backtracks, trials = False, before, None, 0, []
        if not finite_state or not finite_direction:
            trials.append({"backtracks": 0, "scale": 1., "accepted": False, "objective": None,
                           "rank": None, "resolved_condition": None, "head_solve_attempted": False,
                           "reason": "nonfinite hidden displacement or optimizer state"})
        else:
            for backtracks in range(model.config.max_backtracks+1):
                scale = 2.**(-backtracks)
                restore_model()
                with torch.no_grad():
                    for parameter, value, difference in zip(parameters, hidden, displacement):
                        parameter.copy_(value+scale*difference)
                trial = {"backtracks": backtracks, "scale": scale, "accepted": False,
                         "head_solve_attempted": True, "rank_rcond": None}
                try:
                    projection = project_regularized(model, points, observations)
                    candidate = float(objective_regularized(model, points, observations)[0].detach())
                    trial.update(projection=projection, objective=candidate, rank=projection["rank"], resolved_condition=None)
                    if math.isfinite(candidate) and candidate <= before:
                        accepted, after, accepted_scale = True, candidate, scale
                        trial["accepted"] = True
                        trials.append(trial)
                        break
                    trial["reason"] = "stationary regularized objective did not decrease"
                except (FloatingPointError, np.linalg.LinAlgError) as error:
                    trial.update(objective=None, rank=None, resolved_condition=None,
                                 reason="nonfinite or failed regularized readout", error_type=type(error).__name__, error=str(error))
                trials.append(trial)
        if not accepted:
            restore_transaction()
        return {"objective_before": before, "objective_after": after, "accepted": accepted,
            "accepted_scale": accepted_scale, "backtracks": backtracks, "gradient_norm_before_clip": gradient_norm,
            "trials": trials, "proposed_hidden_displacement_finite": finite_direction,
            "proposed_optimizer_state_finite": finite_state, "optimizer_step_counts_before": counts_before,
            "optimizer_step_counts_proposed": counts_proposed, "optimizer_step_counts_after": _step_counts(optimizer),
            "optimizer_state_committed": accepted, "optimizer_state_rolled_back": not accepted,
            "optimizer_reset_required": False}
    except BaseException:
        restore_transaction()
        raise


def controller_step(model, points, observations, optimizer, state, parameters=None):
    """Same bounded controller transitions, around the new stationary solver."""
    config = model.config
    if state.stopped:
        raise RuntimeError("step controller has stopped: "+str(state.stop_reason))
    if not config.controller_minimum_lr <= state.learning_rate <= config.learning_rate:
        raise ValueError("controller learning rate is outside its declared bounds")
    if any(group["lr"] != state.learning_rate for group in optimizer.param_groups):
        raise ValueError("optimizer and controller learning rates disagree")
    before = asdict(state)
    event = transactional_adam_step(model, points, observations, optimizer, parameters)
    state.attempts += 1
    trial_solves = sum(trial["head_solve_attempted"] for trial in event["trials"])
    state.trial_solves += trial_solves
    transitions = []
    if event["accepted"]:
        state.consecutive_failures = 0
        state.full_scale_successes = state.full_scale_successes+1 if event["backtracks"] == 0 else 0
        if event["backtracks"] >= config.controller_carry_backtracks:
            state.learning_rate = max(config.controller_minimum_lr, state.learning_rate*event["accepted_scale"])
            transitions.append("carry accepted backtracking scale")
        elif state.full_scale_successes >= config.controller_growth_streak:
            state.learning_rate = min(config.learning_rate, state.learning_rate*config.controller_growth_factor)
            state.full_scale_successes = 0
            transitions.append("grow after full-scale success streak")
    else:
        state.consecutive_failures += 1
        state.full_scale_successes = 0
        reduced = state.learning_rate*config.controller_reduction
        state.learning_rate = max(config.controller_minimum_lr, reduced)
        _restart_moments(optimizer)
        state.moment_restarts += 1
        transitions.extend(("reduce after exhausted rejection", "restart moments after transactional rollback"))
        if state.consecutive_failures >= config.controller_max_failures:
            state.stopped, state.stop_reason = True, "consecutive exhausted searches"
        elif before["learning_rate"] == config.controller_minimum_lr:
            state.stopped, state.stop_reason = True, "learning-rate floor reached"
        if not state.stopped and not state.learning_rate < before["learning_rate"]:
            state.stopped, state.stop_reason = True, "controller cannot change the next proposal"
    _set_learning_rate(optimizer, state.learning_rate)
    if state.stopped:
        transitions.append("stop: "+state.stop_reason)
    event.update(controller_before=before, controller_after=asdict(state), controller_transitions=transitions,
        controller_optimizer_step_counts_after=_step_counts(optimizer), controller_moments_restarted=not event["accepted"],
        trial_solve_count=trial_solves, controller_stop=state.stopped, controller_stop_reason=state.stop_reason,
        next_learning_rate=state.learning_rate,
        identical_rejected_next_attempt_precluded=bool(event["accepted"] or state.stopped or state.learning_rate < before["learning_rate"]))
    return event


def fit_regularized(observations, *, physics=PlanarContactConfig(), config=RegularizedConfig(),
                    known_h, neural_seed, sampling_seed, progress=None, readout_calibration=None):
    torch.set_num_threads(1)
    started = perf_counter()
    model, initialization = matched_regularized_model(physics, config, known_h, neural_seed)
    _observation_tensors(model, observations)
    points = graded_training_points(physics, config, sampling_seed)
    metadata = graded_point_metadata(physics, config, sampling_seed)
    locally_calibrated = readout_calibration is None
    if locally_calibrated:
        calibration = calibrate_readout(model, points, observations)
    else:
        apply_readout_calibration(model, readout_calibration, points, observations)
        calibration = deepcopy(readout_calibration)
    initial = project_regularized(model, points, observations)
    initial.update(stage="initial_projection", step=0)
    if progress:
        progress(initial)
    optimizer = torch.optim.Adam(model.networks.parameters(), lr=config.learning_rate)
    state, updates, reset_count = StepControlState(config.learning_rate), [], 0
    for step in range(config.adam_epochs):
        if state.trial_solves+config.max_backtracks+1 > config.maximum_trial_solves:
            state.stopped, state.stop_reason = True, "projection budget"
            break
        if step and config.optimizer_history == "reset":
            _restart_moments(optimizer)
            reset_count += 1
        event = controller_step(model, points, observations, optimizer, state)
        event.update(stage="regularized_controlled_profiled_adam", step=step+1,
                     policy_moments_reset_before_attempt=bool(step and config.optimizer_history == "reset"))
        updates.append(event)
        if progress:
            progress(event)
        if state.stopped:
            break
    final = project_regularized(model, points, observations)
    final.update(stage="final_projection", step=len(updates))
    if progress:
        progress(final)
    total, original, penalty, obs, physical, components = objective_regularized(model, points, observations)
    model.final_optimizer_state, model.final_controller_state = deepcopy(optimizer.state_dict()), asdict(state)
    projections = [initial]+[trial["projection"] for update in updates for trial in update["trials"] if "projection" in trial]+[final]
    report = {"architecture": "graded_hermite_field_with_fixed_ridge_stationary_readout",
        "model_module": __name__, "model_class": "RegularizedPINN", "mode": "known_contact",
        "known_h_oracle": True, "known_h": float(known_h), "config": asdict(config), "physics": asdict(physics),
        "neural_seed": neural_seed, "sampling_seed": sampling_seed, "initialization": initialization,
        "readout_calibration": calibration, "initial_calibration_matrix_builds": int(locally_calibrated),
        "initial_calibration_spectral_calculations": int(locally_calibrated),
        "graded_point_metadata": metadata, "training_points": {key: value.tolist() for key, value in points.items()},
        "training_point_weights": {key: np.full(len(value), 1/len(value)).tolist() for key, value in points.items()},
        "training_point_counts": {key: len(value) for key, value in points.items()},
        "normalized_observation_mse": float(obs.detach()), "final_physics_loss": float(physical.detach()),
        "final_objective": float(total.detach()), "final_unpenalized_objective": float(original.detach()),
        "final_regularization_penalty": float(penalty.detach()), "training_family_mse": {key: float(value.detach()) for key, value in components.items()},
        "final_head_stationarity": final["stationarity_relative"], "projections": projections,
        "hidden_update_history": updates, "hidden_rejected_updates": sum(not event["accepted"] for event in updates),
        "hidden_backtrack_count": sum(event["backtracks"] for event in updates),
        "hidden_accepted_objective_increases": sum(event["objective_after"] > event["objective_before"] for event in updates),
        "optimizer_history": config.optimizer_history, "optimizer_instances": 1,
        "optimizer_resets": reset_count+state.moment_restarts, "policy_moment_resets": reset_count,
        "controller_moment_restarts": state.moment_restarts,
        "optimizer_state_rollbacks": sum(event["optimizer_state_rolled_back"] for event in updates),
        "final_optimizer_step_counts": _step_counts(optimizer), "final_controller_state": asdict(state),
        "controller_history": [{"step": event["step"], "before": event["controller_before"], "after": event["controller_after"],
                                "transitions": event["controller_transitions"]} for event in updates],
        "trial_solve_count": state.trial_solves, "additional_initial_final_solves": 2,
        "total_projection_solve_calls": state.trial_solves+2, "maximum_trial_solves": config.maximum_trial_solves,
        "requested_updates": config.adam_epochs, "completed_updates": len(updates),
        "termination": state.stop_reason if state.stopped else "fixed update budget exhausted",
        "stop_reason": {"consecutive exhausted searches": "consecutive_failure_stop", "learning-rate floor reached": "learning_rate_floor_stop",
            "controller cannot change the next proposal": "unchanged_proposal_stop", "projection budget": "projection_budget"}.get(state.stop_reason, "fixed_budget_exhausted"),
        "optimizer_state_policy": "same transactional rollback and separate bounded controller; acceptance uses regularized objective only",
        "coefficient_count": len(model.coefficients), "base_coefficient_count": model.base_coefficient_count,
        "wall_seconds": perf_counter()-started, "checkpoint": "final stationary-head accepted state at fixed budget or declared controller stop",
        "gradient_semantics": "fixed-head partial derivative equals the regularized profiled derivative at a unique stationary head; frozen D and lambda have no hidden-parameter dependence",
        "head_stationarity_required": True, "hard_singular_value_truncation": False,
        "hidden_truth_labels_used": False, "independent_diagnostics_used_for_training": False}
    return model, report


predict_regularized = predict_step_control
diagnose_regularized = diagnose_step_control


def save_checkpoint(model, destination):
    torch.save({"state_dict": model.state_dict(), "config": asdict(model.config), "physics": asdict(model.physics),
        "known_h": model.known_h, "initialization": getattr(model, "initialization_metadata", None),
        "readout_calibration": getattr(model, "readout_calibration", None),
        "optimizer_state": getattr(model, "final_optimizer_state", None),
        "controller_state": getattr(model, "final_controller_state", None),
        "model_module": __name__, "model_class": "RegularizedPINN",
        "architecture": "graded_hermite_field_with_fixed_ridge_stationary_readout"}, destination)


def checkpoint_bytes(model):
    destination = io.BytesIO()
    save_checkpoint(model, destination)
    return destination.getvalue()
