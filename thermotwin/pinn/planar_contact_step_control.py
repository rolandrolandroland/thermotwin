"""Bounded step control around the frozen transactional graded-field optimizer.

The controller is deliberately outside an Adam proposal's transaction. A failed
proposal restores the field, readout and optimizer first; the controller then
changes the next learning rate and clears moments, or stops. Configuration
defaults describe mechanics, not an evidence-selected scientific run.
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
from .planar_contact_graded import graded_point_metadata, graded_training_points
from .planar_contact_optimized import (
    OptimizedConfig, OptimizedPINN, _step_counts, diagnose_optimized,
    matched_optimized_model, predict_optimized, transactional_adam_step,
)


@dataclass(frozen=True)
class StepControlConfig(OptimizedConfig):
    max_backtracks: int = 4
    maximum_trial_solves: int = 480
    controller_reduction: float = .5
    controller_minimum_lr: float = 1.e-7
    controller_max_failures: int = 3
    controller_carry_backtracks: int = 2
    controller_growth_streak: int = 8
    controller_growth_factor: float = 2.

    def __post_init__(self):
        super().__post_init__()
        if not math.isfinite(self.controller_reduction) or not 0 < self.controller_reduction < 1:
            raise ValueError("controller reduction must lie strictly between zero and one")
        if not math.isfinite(self.learning_rate) or not math.isfinite(self.controller_minimum_lr) or not 0 < self.controller_minimum_lr <= self.learning_rate:
            raise ValueError("controller minimum learning rate must be positive and no greater than the initial rate")
        if not math.isfinite(self.controller_growth_factor) or self.controller_growth_factor <= 1:
            raise ValueError("controller growth factor must be finite and greater than one")
        for name in ("maximum_trial_solves", "controller_max_failures", "controller_carry_backtracks", "controller_growth_streak"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(name+" must be a positive integer")


@dataclass
class StepControlState:
    learning_rate: float
    attempts: int = 0
    consecutive_failures: int = 0
    full_scale_successes: int = 0
    moment_restarts: int = 0
    trial_solves: int = 0
    stopped: bool = False
    stop_reason: str | None = None


class StepControlPINN(OptimizedPINN):
    def __init__(self, physics=PlanarContactConfig(), config=StepControlConfig(),
                 *, known_h, mode="known_contact"):
        super().__init__(physics, config, known_h=known_h, mode=mode)


def matched_step_control_model(physics, config, known_h, neural_seed):
    base = {key: value for key, value in asdict(config).items()
            if key in OptimizedConfig.__dataclass_fields__}
    source, initialization = matched_optimized_model(physics, OptimizedConfig(**base), known_h, neural_seed)
    model = StepControlPINN(physics, config, known_h=known_h)
    model.load_state_dict(source.state_dict())
    model.initialization_metadata = initialization
    return model, initialization


def _set_learning_rate(optimizer, learning_rate):
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _restart_moments(optimizer):
    # Keep the optimizer object, parameter bindings and other group options.
    optimizer.state.clear()


def controller_step(model, points, observations, optimizer, state, parameters=None):
    """Attempt one transaction, then make an explicit bounded controller move.

    A rejected transaction's exact optimizer rollback is observable separately
    from the subsequent, intentional learning-rate/moment change. Exceptions
    propagate after the inherited rollback without advancing controller state.
    """
    config = model.config
    if state.stopped:
        raise RuntimeError("step controller has stopped: "+str(state.stop_reason))
    if not config.controller_minimum_lr <= state.learning_rate <= config.learning_rate:
        raise ValueError("controller learning rate is outside its declared bounds")
    if any(group["lr"] != state.learning_rate for group in optimizer.param_groups):
        raise ValueError("optimizer and controller learning rates disagree")
    before = asdict(state)
    event = transactional_adam_step(model, points, observations, optimizer, parameters)
    # No controller state is changed until the inherited transaction returns.
    state.attempts += 1
    trial_solves = len(event["trials"]) if event["proposed_hidden_displacement_finite"] and event["proposed_optimizer_state_finite"] else 0
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
        if not event["optimizer_state_rolled_back"]:
            raise RuntimeError("rejected transaction did not restore optimizer state")
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
    event.update(controller_before=before, controller_after=asdict(state),
        controller_transitions=transitions,
        controller_optimizer_step_counts_after=_step_counts(optimizer),
        controller_moments_restarted=not event["accepted"],
        trial_solve_count=trial_solves,
        controller_stop=state.stopped, controller_stop_reason=state.stop_reason,
        next_learning_rate=state.learning_rate,
        identical_rejected_next_attempt_precluded=bool(event["accepted"] or state.stopped or state.learning_rate < before["learning_rate"]))
    return event


def fit_step_control(observations, *, physics=PlanarContactConfig(), config=StepControlConfig(),
                     known_h, neural_seed, sampling_seed, progress=None):
    torch.set_num_threads(1)
    started = perf_counter()
    model, initialization = matched_step_control_model(physics, config, known_h, neural_seed)
    _observation_tensors(model, observations)
    points = graded_training_points(physics, config, sampling_seed)
    metadata = graded_point_metadata(physics, config, sampling_seed)
    initial = project_output(model, points, observations)
    initial.update(stage="initial_projection", step=0)
    if progress:
        progress(initial)
    optimizer = torch.optim.Adam(model.networks.parameters(), lr=config.learning_rate)
    state = StepControlState(config.learning_rate)
    updates, reset_count = [], 0
    for step in range(config.adam_epochs):
        if state.trial_solves+config.max_backtracks+1 > config.maximum_trial_solves:
            state.stopped, state.stop_reason = True, "projection budget"
            break
        if step and config.optimizer_history == "reset":
            _restart_moments(optimizer)
            reset_count += 1
        event = controller_step(model, points, observations, optimizer, state)
        event.update(stage="controlled_transactional_profiled_adam", step=step+1,
                     policy_moments_reset_before_attempt=bool(step and config.optimizer_history == "reset"))
        updates.append(event)
        if progress:
            progress(event)
        if state.stopped:
            break
    final = project_output(model, points, observations)
    final.update(stage="final_projection", step=len(updates))
    if progress:
        progress(final)
    total, obs, physical, components = objective_continuity(model, points, observations)
    model.final_optimizer_state = deepcopy(optimizer.state_dict())
    model.final_controller_state = asdict(state)
    projections = [initial]+[trial["projection"] for update in updates for trial in update["trials"] if "projection" in trial]+[final]
    report = {"architecture": "graded_hermite_field_with_controlled_transactional_profiled_adam",
        "model_module": __name__, "model_class": "StepControlPINN", "mode": "known_contact",
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
        "optimizer_history": config.optimizer_history, "optimizer_instances": 1,
        "optimizer_resets": reset_count+state.moment_restarts,
        "policy_moment_resets": reset_count, "controller_moment_restarts": state.moment_restarts,
        "optimizer_state_rollbacks": sum(event["optimizer_state_rolled_back"] for event in updates),
        "final_optimizer_step_counts": _step_counts(optimizer), "final_controller_state": asdict(state),
        "controller_history": [{"step": event["step"], "before": event["controller_before"],
            "after": event["controller_after"], "transitions": event["controller_transitions"]} for event in updates],
        "trial_solve_count": state.trial_solves, "additional_initial_final_solves": 2,
        "total_projection_solve_calls": state.trial_solves+2,
        "maximum_trial_solves": config.maximum_trial_solves,
        "requested_updates": config.adam_epochs, "completed_updates": len(updates),
        "termination": state.stop_reason if state.stopped else "fixed update budget exhausted",
        "stop_reason": {"consecutive exhausted searches": "consecutive_failure_stop",
            "learning-rate floor reached": "learning_rate_floor_stop",
            "controller cannot change the next proposal": "unchanged_proposal_stop",
            "projection budget": "projection_budget"}.get(state.stop_reason, "fixed_budget_exhausted"),
        "optimizer_state_policy": "transactional rollback precedes separate bounded learning-rate control and moment restart",
        "coefficient_count": len(model.coefficients), "base_coefficient_count": model.base_coefficient_count,
        "wall_seconds": perf_counter()-started, "checkpoint": "final accepted state at fixed budget or declared controller stop",
        "gradient_semantics": "fixed-readout proposal; no exact profiled-gradient claim with truncated SVD",
        "hidden_truth_labels_used": False, "independent_diagnostics_used_for_training": False}
    return model, report


predict_step_control = predict_optimized
diagnose_step_control = diagnose_optimized


def save_checkpoint(model, destination):
    torch.save({"state_dict": model.state_dict(), "config": asdict(model.config), "physics": asdict(model.physics),
        "known_h": model.known_h, "initialization": getattr(model, "initialization_metadata", None),
        "optimizer_state": getattr(model, "final_optimizer_state", None),
        "controller_state": getattr(model, "final_controller_state", None),
        "model_module": __name__, "model_class": "StepControlPINN",
        "architecture": "graded_hermite_field_with_controlled_transactional_profiled_adam"}, destination)


def checkpoint_bytes(model):
    destination = io.BytesIO()
    save_checkpoint(model, destination)
    return destination.getvalue()
