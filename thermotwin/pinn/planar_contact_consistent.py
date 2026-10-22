"""Additive v4 correction of the preserved shared-continuity neural method.

The paired readout restores odd vertical derivative freedom at the top while
retaining exact temperature/normal-flux continuity and prescribed side/top
conditions. Hidden proposals are accepted only after a new linear readout
solve, using the same complete training objective. Truncated-SVD readouts do
not justify an exact profiled-gradient claim; acceptance uses actual losses.
"""
from dataclasses import asdict, dataclass, replace
import io
import math
from time import perf_counter

import numpy as np
import torch
from torch import nn

from ..physics.planar_contact import PlanarContactConfig
from .planar_contact_continuity import (
    ContinuityConfig, ContinuityPINN, _observation_tensors,
    diagnose_continuity, objective_continuity, predict_continuity,
    project_output, training_points,
)


@dataclass(frozen=True)
class ConsistentConfig(ContinuityConfig):
    adam_epochs: int = 20
    projection_every: int = 1
    interior_points: int = 8192
    max_backtracks: int = 2


class ConsistentPINN(ContinuityPINN):
    """Shared neural features paired with a top-flat cubic vertical factor."""
    def __init__(self, physics=PlanarContactConfig(), config=ConsistentConfig(),
                 *, known_h, mode="known_contact"):
        super().__init__(physics, config, known_h=known_h, mode=mode)
        self.base_coefficient_count = len(self.coefficients)
        self.coefficients = nn.Parameter(torch.zeros(2*self.base_coefficient_count,
                                                     dtype=torch.float64), requires_grad=False)
        self.paired_slices = (slice(0, self.base_coefficient_count),
                              slice(self.base_coefficient_count, 2*self.base_coefficient_count))

    def _basis_region(self, q, index):
        original = super()._basis_region(q, index)
        z = q[:, 1:2]
        factor = z*z*(3-2*z)
        return torch.cat((original, factor*original), dim=1)


def consistent_training_points(physics, config, seed):
    """Double only the declared vertical grid; retain v3 bottom arrays."""
    if config.interior_points != 8192:
        # Constructed small-grid tests use the inherited generic point builder.
        return training_points(physics, config, seed)
    points = training_points(physics, replace(config, interior_points=4096), seed)
    rng = np.random.default_rng(seed)
    x = (np.arange(32)+rng.uniform(.25, .75))/32
    z = (np.arange(16)+rng.uniform(.25, .75))/16
    times = np.array([.2, .5, 1., 2., 3., 5., 8., 12., 18., 26., 36., 50., 68., 88., 105., 119.])/config.duration
    xx, zz, tt = np.meshgrid(x, z, times, indexing="ij")
    points["interior"] = np.column_stack((xx.ravel(), zz.ravel(), tt.ravel()))
    return points


def variable_projection_step(model, points, observations, optimizer, parameters=None):
    """One bounded proposal with full readout/hidden rollback on rejection.

    The gradient holds the current readout fixed. Each proposed hidden point
    gets its own SVD solve before the complete objective is compared against
    the original point. All trial solver metadata is retained, including for
    proposals rejected after a successful readout solve.
    """
    parameters = list(model.networks.parameters()) if parameters is None else list(parameters)
    optimizer.zero_grad()
    total = objective_continuity(model, points, observations)[0]
    before = float(total.detach())
    total.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, model.config.gradient_clip))
    if not math.isfinite(gradient_norm):
        raise FloatingPointError("nonfinite consistent hidden gradient")
    hidden_before = [parameter.detach().clone() for parameter in parameters]
    head_before = model.coefficients.detach().clone()
    trials = []
    accepted, after, accepted_scale, backtracks = False, before, None, 0

    def restore():
        with torch.no_grad():
            for parameter, previous in zip(parameters, hidden_before):
                parameter.copy_(previous)
            model.coefficients.copy_(head_before)

    try:
        optimizer.step()
        proposal = [parameter.detach()-previous for parameter, previous in zip(parameters, hidden_before)]
        for backtracks in range(model.config.max_backtracks+1):
            scale = 2.**(-backtracks)
            restore()
            with torch.no_grad():
                for parameter, previous, difference in zip(parameters, hidden_before, proposal):
                    parameter.copy_(previous+scale*difference)
            trial = {"backtracks": backtracks, "scale": scale,
                     "rank_rcond": model.config.rank_rcond, "accepted": False}
            try:
                projection = project_output(model, points, observations)
                trial["projection"] = projection
                candidate = float(objective_continuity(model, points, observations)[0].detach())
                trial["objective"] = candidate if math.isfinite(candidate) else None
                trial["rank"] = projection["rank"]
                trial["resolved_condition"] = projection["resolved_condition"]
                if math.isfinite(candidate) and candidate <= before:
                    accepted, after, accepted_scale = True, candidate, scale
                    trial["accepted"] = True
                    trials.append(trial)
                    break
                trial["reason"] = "complete objective did not decrease after readout solve"
            except Exception as error:
                restore()
                if type(error).__name__ == "ResourceLimit":
                    raise
                trial.update(objective=None, rank=None, resolved_condition=None,
                             reason="trial readout or objective failed", error_type=type(error).__name__, error=str(error))
            trials.append(trial)
    except BaseException:
        restore()
        raise
    if not accepted:
        restore()
    return {"objective_before": before, "objective_after": after,
            "accepted": accepted, "accepted_scale": accepted_scale,
            "backtracks": backtracks, "gradient_norm_before_clip": gradient_norm,
            "trials": trials, "optimizer_reset_required": True}


def fit_consistent(observations, *, physics=PlanarContactConfig(), config=ConsistentConfig(),
                   known_h, neural_seed, sampling_seed, progress=None):
    torch.set_num_threads(1)
    torch.manual_seed(neural_seed)
    started = perf_counter()
    model = ConsistentPINN(physics, config, known_h=known_h)
    _observation_tensors(model, observations)
    points = consistent_training_points(physics, config, sampling_seed)
    initial = project_output(model, points, observations)
    initial.update(stage="initial_projection", step=0)
    if progress:
        progress(initial)
    updates = []
    optimizer = torch.optim.Adam(model.networks.parameters(), lr=config.learning_rate)
    resets = 1
    for step in range(config.adam_epochs):
        event = variable_projection_step(model, points, observations, optimizer)
        event.update(stage="hidden_variable_projection", step=step+1)
        updates.append(event)
        if progress:
            progress(event)
        optimizer = torch.optim.Adam(model.networks.parameters(), lr=config.learning_rate)
        resets += 1
    final = project_output(model, points, observations)
    final.update(stage="final_projection", step=config.adam_epochs)
    if progress:
        progress(final)
    total, obs, physics_loss, components = objective_continuity(model, points, observations)
    projections = [initial]+[trial["projection"] for update in updates for trial in update["trials"] if "projection" in trial]+[final]
    report = {"architecture": "consistent_paired_hermite_variable_projection_mlp_v4",
              "model_module": __name__, "model_class": "ConsistentPINN", "mode": "known_contact",
              "known_h_oracle": True, "known_h": float(known_h), "config": asdict(config), "physics": asdict(physics),
              "neural_seed": neural_seed, "sampling_seed": sampling_seed,
              "normalized_observation_mse": float(obs.detach()), "final_physics_loss": float(physics_loss.detach()),
              "final_objective": float(total.detach()), "training_family_mse": {k: float(v.detach()) for k, v in components.items()},
              "training_points": {k: v.tolist() for k, v in points.items()}, "training_point_counts": {k: len(v) for k, v in points.items()},
              "projections": projections, "hidden_update_history": updates,
              "hidden_rejected_updates": sum(not event["accepted"] for event in updates),
              "hidden_backtrack_count": sum(event["backtracks"] for event in updates),
              "hidden_accepted_objective_increases": sum(event["objective_after"] > event["objective_before"] for event in updates),
              "optimizer_resets": resets, "coefficient_count": len(model.coefficients),
              "base_coefficient_count": model.base_coefficient_count, "wall_seconds": perf_counter()-started,
              "checkpoint": "fixed-budget final state with monotone complete-objective variable projection",
              "gradient_semantics": "fixed-readout hidden gradient is a proposal only; truncated SVD precludes exact profiled-gradient claim",
              "hidden_truth_labels_used": False, "independent_diagnostics_used_for_training": False,
              "exact_corner_policy": "continuous field value; singular corner derivatives excluded"}
    return model, report


predict_consistent = predict_continuity
diagnose_consistent = diagnose_continuity


def save_checkpoint(model, destination):
    torch.save({"state_dict": model.state_dict(), "config": asdict(model.config), "physics": asdict(model.physics),
                "known_h": model.known_h, "model_module": __name__, "model_class": "ConsistentPINN",
                "architecture": "consistent_paired_hermite_variable_projection_mlp_v4"}, destination)


def checkpoint_bytes(model):
    destination = io.BytesIO()
    save_checkpoint(model, destination)
    return destination.getvalue()
