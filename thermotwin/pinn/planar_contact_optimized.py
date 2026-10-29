"""Transactional Adam proposals for the unchanged graded log-off trial field.

Configuration values are mechanics defaults, not an evidence-selected scientific
rule. A separate training-only decision record must choose the final budget,
learning rate and history policy before a four-case campaign is prepared.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import io
import math
from time import perf_counter

import numpy as np
import torch

from ..physics.planar_contact import PlanarContactConfig
from .planar_contact_continuity import _observation_tensors, objective_continuity, project_output
from .planar_contact_graded import (
    GradedConfig, GradedPINN, diagnose_graded, graded_point_metadata,
    graded_training_points, matched_graded_model, predict_graded,
)


@dataclass(frozen=True)
class OptimizedConfig(GradedConfig):
    bottom_log_feature: bool = False
    optimizer_history: str = "persistent"

    def __post_init__(self):
        super().__post_init__()
        if self.bottom_log_feature:
            raise ValueError("the optimizer experiment retains the log-off field")
        if self.projection_every != 1:
            raise ValueError("every optimizer proposal must re-solve its readout")
        if self.optimizer_history not in ("persistent", "reset"):
            raise ValueError("optimizer_history must be persistent or reset")


class OptimizedPINN(GradedPINN):
    def __init__(self, physics=PlanarContactConfig(), config=OptimizedConfig(),
                 *, known_h, mode="known_contact"):
        super().__init__(physics, config, known_h=known_h, mode=mode)


def matched_optimized_model(physics, config, known_h, neural_seed):
    base_config = asdict(config)
    base_config.pop("optimizer_history")
    source, initialization = matched_graded_model(physics, GradedConfig(**base_config), known_h, neural_seed)
    model = OptimizedPINN(physics, config, known_h=known_h)
    model.load_state_dict(source.state_dict())
    model.initialization_metadata = initialization
    return model, initialization


def _step_counts(optimizer):
    return sorted({int(value["step"].item() if torch.is_tensor(value["step"]) else value["step"])
                   for value in optimizer.state.values() if "step" in value})


def _finite_optimizer_state(optimizer):
    return all(bool(torch.all(torch.isfinite(value))) if torch.is_tensor(value)
               else math.isfinite(value) if isinstance(value, (float, int)) else True
               for state in optimizer.state.values() for value in state.values())


def transactional_adam_step(model, points, observations, optimizer, parameters=None):
    """Commit one Adam state only when a reprojected parameter point is accepted.

    Backtracking scales the proposed parameter displacement. The first/second
    moment estimates describe the one gradient evaluation and are committed
    once on acceptance, including acceptance at a smaller displacement. Failed
    transactions restore parameters, readout, optimizer groups/moments/step
    counters, and pre-existing gradient buffers exactly.
    """
    parameters = list(model.networks.parameters()) if parameters is None else list(parameters)
    hidden_before = [parameter.detach().clone() for parameter in parameters]
    gradients_before = [None if parameter.grad is None else parameter.grad.detach().clone() for parameter in parameters]
    head_before = model.coefficients.detach().clone()
    optimizer_before = deepcopy(optimizer.state_dict())
    counts_before = _step_counts(optimizer)

    def restore_model():
        with torch.no_grad():
            for parameter, value in zip(parameters, hidden_before):
                parameter.copy_(value)
            model.coefficients.copy_(head_before)

    def restore_transaction():
        restore_model()
        optimizer.load_state_dict(deepcopy(optimizer_before))
        for parameter, value in zip(parameters, gradients_before):
            parameter.grad = None if value is None else value.clone()

    try:
        optimizer.zero_grad()
        total = objective_continuity(model, points, observations)[0]
        before = float(total.detach())
        total.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, model.config.gradient_clip))
        if not math.isfinite(gradient_norm):
            raise FloatingPointError("nonfinite optimizer proposal gradient")
        optimizer.step()
        finite_state = _finite_optimizer_state(optimizer)
        counts_proposed = _step_counts(optimizer) if finite_state else []
        displacement = [parameter.detach()-value for parameter, value in zip(parameters, hidden_before)]
        finite_direction = all(bool(torch.all(torch.isfinite(value))) for value in displacement)
        trials = []
        accepted, after, accepted_scale, backtracks = False, before, None, 0
        if not finite_direction or not finite_state:
            trials.append({"backtracks": 0, "scale": 1., "accepted": False,
                           "objective": None, "rank": None, "resolved_condition": None,
                           "rank_rcond": model.config.rank_rcond,
                           "reason": "nonfinite hidden displacement or optimizer state"})
        else:
            for backtracks in range(model.config.max_backtracks+1):
                scale = 2.**(-backtracks)
                restore_model()
                with torch.no_grad():
                    for parameter, value, difference in zip(parameters, hidden_before, displacement):
                        parameter.copy_(value+scale*difference)
                trial = {"backtracks": backtracks, "scale": scale, "accepted": False,
                         "rank_rcond": model.config.rank_rcond}
                try:
                    projection = project_output(model, points, observations)
                    trial["projection"] = projection
                    candidate = float(objective_continuity(model, points, observations)[0].detach())
                    trial.update(objective=candidate if math.isfinite(candidate) else None,
                                 rank=projection["rank"], resolved_condition=projection["resolved_condition"])
                    if math.isfinite(candidate) and candidate <= before:
                        accepted, after, accepted_scale = True, candidate, scale
                        trial["accepted"] = True
                        trials.append(trial)
                        break
                    trial["reason"] = "reprojected complete objective did not decrease"
                except (FloatingPointError, np.linalg.LinAlgError) as error:
                    trial.update(objective=None, rank=None, resolved_condition=None,
                                 reason="nonfinite or failed numerical readout trial",
                                 error_type=type(error).__name__, error=str(error))
                trials.append(trial)
        if not accepted:
            restore_transaction()
        return {"objective_before": before, "objective_after": after, "accepted": accepted,
                "accepted_scale": accepted_scale, "backtracks": backtracks,
                "gradient_norm_before_clip": gradient_norm, "trials": trials,
                "proposed_hidden_displacement_finite": finite_direction,
                "proposed_optimizer_state_finite": finite_state,
                "optimizer_step_counts_before": counts_before,
                "optimizer_step_counts_proposed": counts_proposed,
                "optimizer_step_counts_after": _step_counts(optimizer),
                "optimizer_state_committed": accepted, "optimizer_state_rolled_back": not accepted,
                "optimizer_reset_required": False}
    except BaseException:
        restore_transaction()
        raise


persistent_adam_step = transactional_adam_step


def fit_optimized(observations, *, physics=PlanarContactConfig(), config=OptimizedConfig(),
                  known_h, neural_seed, sampling_seed, progress=None):
    torch.set_num_threads(1)
    started = perf_counter()
    model, initialization = matched_optimized_model(physics, config, known_h, neural_seed)
    _observation_tensors(model, observations)
    points = graded_training_points(physics, config, sampling_seed)
    metadata = graded_point_metadata(physics, config, sampling_seed)
    initial = project_output(model, points, observations)
    initial.update(stage="initial_projection", step=0)
    if progress:
        progress(initial)
    optimizer = torch.optim.Adam(model.networks.parameters(), lr=config.learning_rate)
    instances, updates = 1, []
    for step in range(config.adam_epochs):
        if step and config.optimizer_history == "reset":
            optimizer = torch.optim.Adam(model.networks.parameters(), lr=config.learning_rate)
            instances += 1
        event = transactional_adam_step(model, points, observations, optimizer)
        event.update(stage="transactional_profiled_adam", step=step+1)
        updates.append(event)
        if progress:
            progress(event)
    final = project_output(model, points, observations)
    final.update(stage="final_projection", step=config.adam_epochs)
    if progress:
        progress(final)
    total, obs, physical, components = objective_continuity(model, points, observations)
    model.final_optimizer_state = deepcopy(optimizer.state_dict())
    projections = [initial]+[trial["projection"] for update in updates for trial in update["trials"] if "projection" in trial]+[final]
    report = {"architecture": "graded_hermite_field_with_transactional_profiled_adam",
        "model_module": __name__, "model_class": "OptimizedPINN", "mode": "known_contact",
        "known_h_oracle": True, "known_h": float(known_h), "config": asdict(config), "physics": asdict(physics),
        "neural_seed": neural_seed, "sampling_seed": sampling_seed, "initialization": initialization,
        "graded_point_metadata": metadata, "training_points": {k: v.tolist() for k, v in points.items()},
        "training_point_weights": {k: np.full(len(v), 1/len(v)).tolist() for k, v in points.items()},
        "training_point_counts": {k: len(v) for k, v in points.items()},
        "normalized_observation_mse": float(obs.detach()), "final_physics_loss": float(physical.detach()),
        "final_objective": float(total.detach()), "training_family_mse": {k: float(v.detach()) for k, v in components.items()},
        "projections": projections, "hidden_update_history": updates,
        "hidden_rejected_updates": sum(not event["accepted"] for event in updates),
        "hidden_backtrack_count": sum(event["backtracks"] for event in updates),
        "hidden_accepted_objective_increases": sum(event["objective_after"] > event["objective_before"] for event in updates),
        "optimizer_history": config.optimizer_history, "optimizer_instances": instances,
        "optimizer_resets": instances-1,
        "optimizer_state_rollbacks": sum(event["optimizer_state_rolled_back"] for event in updates),
        "final_optimizer_step_counts": _step_counts(optimizer),
        "optimizer_state_policy": "transactional moments, step counters, groups and gradient buffers; commit once on acceptance; exact rollback on rejection or exception",
        "coefficient_count": len(model.coefficients), "base_coefficient_count": model.base_coefficient_count,
        "wall_seconds": perf_counter()-started, "checkpoint": "fixed-budget final accepted model state",
        "gradient_semantics": "fixed-readout proposal; no exact profiled-gradient claim with truncated SVD",
        "hidden_truth_labels_used": False, "independent_diagnostics_used_for_training": False}
    return model, report


predict_optimized = predict_graded
diagnose_optimized = diagnose_graded


def save_checkpoint(model, destination):
    torch.save({"state_dict": model.state_dict(), "config": asdict(model.config), "physics": asdict(model.physics),
        "known_h": model.known_h, "initialization": getattr(model, "initialization_metadata", None),
        "optimizer_state": getattr(model, "final_optimizer_state", None),
        "model_module": __name__, "model_class": "OptimizedPINN",
        "architecture": "graded_hermite_field_with_transactional_profiled_adam"}, destination)


def checkpoint_bytes(model):
    destination = io.BytesIO()
    save_checkpoint(model, destination)
    return destination.getvalue()
