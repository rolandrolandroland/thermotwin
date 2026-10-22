"""Matched log-feature comparison with a frozen geometry-graded PDE grid.

Both arms retain the v4 trial field, linear solver, full-objective acceptance,
and optimizer. They differ only in the availability of the extra logarithmic
input after a function-matched initialization. No hidden reference values enter
initialization, point construction, or training.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import io
import json
import math
from time import perf_counter

import numpy as np
import torch

from ..physics.planar_contact import PlanarContactConfig
from .planar_contact_continuity import (
    _observation_tensors, diagnose_continuity, objective_continuity,
    predict_continuity, project_output, training_points,
)
from .planar_contact_consistent import (
    ConsistentConfig, ConsistentPINN, consistent_training_points,
    variable_projection_step,
)


@dataclass(frozen=True)
class GradedConfig(ConsistentConfig):
    adam_epochs: int = 80
    projection_every: int = 1
    interior_points: int = 8192
    bottom_points: int = 512


class GradedPINN(ConsistentPINN):
    def __init__(self, physics=PlanarContactConfig(), config=GradedConfig(),
                 *, known_h, mode="known_contact"):
        super().__init__(physics, config, known_h=known_h, mode=mode)


INITIALIZATION_PROBE_POINTS = np.array([
    [.0731, .0173, .0047], [.2467, .1379, .0713],
    [.3839, .3731, .2339], [.6733, .6137, .5817],
    [.9149, .9473, .9233],
], dtype=np.float64)


def _array_digest(array):
    value = np.ascontiguousarray(array, dtype="<f8")
    digest = hashlib.sha256(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _state_digest(model, *, common=False):
    digest = hashlib.sha256()
    regular_first_weights = {f"networks.{j}.0.weight" for j, (kind, _) in enumerate(model.descriptors)
                             if kind != "corner"}
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().numpy()
        if common and model.config.bottom_log_feature and name in regular_first_weights:
            value = value[:, :-1]
        digest.update(name.encode()+b"\0")
        digest.update(_array_digest(value).encode())
    return digest.hexdigest()


def _probe(model):
    original = model.coefficients.detach().clone()
    try:
        # Nonzero, deterministic coefficients expose every learned basis channel.
        with torch.no_grad():
            model.coefficients.copy_(.01*torch.sin(torch.arange(len(original), dtype=torch.float64)*.173))
        q = torch.tensor(INITIALIZATION_PROBE_POINTS.copy(), dtype=torch.float64, requires_grad=True)
        basis = model.basis(q).detach().numpy()
        theta = model.dimensionless_temperature(q)
        gradient = torch.autograd.grad(theta.sum(), q, create_graph=True)[0]
        hessian = torch.stack([torch.autograd.grad(gradient[:, j].sum(), q, retain_graph=j < 2)[0]
                               for j in range(3)], dim=1)
        scales = np.array([model.physics.width, model.physics.thickness, model.config.duration])
        normalized_gradient = gradient.detach().numpy()
        normalized_hessian = hessian.detach().numpy()
        return {"basis": basis, "temperature": (model.config.initial_temperature+model.config.temperature_scale*theta).detach().numpy(),
                "normalized_gradient": normalized_gradient, "normalized_hessian": normalized_hessian,
                "physical_gradient": normalized_gradient*model.config.temperature_scale/scales,
                "physical_hessian": normalized_hessian*model.config.temperature_scale/(scales[None, :, None]*scales[None, None, :])}
    finally:
        with torch.no_grad():
            model.coefficients.copy_(original)


def _matched_pair(physics, config_on, config_off, known_h, neural_seed):
    if not config_on.bottom_log_feature or config_off.bottom_log_feature:
        raise ValueError("matched pair requires explicit log-on and log-off configurations")
    left, right = asdict(config_on), asdict(config_off)
    left.pop("bottom_log_feature"); right.pop("bottom_log_feature")
    if left != right:
        raise ValueError("matched arms may differ only in the log-feature switch")
    torch.manual_seed(neural_seed)
    on = GradedPINN(physics, config_on, known_h=known_h)
    with torch.no_grad():
        for network, (kind, _) in zip(on.networks, on.descriptors):
            if kind != "corner":
                network[0].weight[:, -1].zero_()
    off = GradedPINN(physics, config_off, known_h=known_h)
    for source, target, (kind, _) in zip(on.networks, off.networks, on.descriptors):
        state = {name: value.detach().clone() for name, value in source.state_dict().items()}
        if kind != "corner":
            state["0.weight"] = state["0.weight"][:, :-1].clone()
        target.load_state_dict(state)
    with torch.no_grad():
        off.coefficients.copy_(on.coefficients)
    common_on, common_off = _state_digest(on, common=True), _state_digest(off, common=True)
    if common_on != common_off:
        raise ValueError("common initial parameter identities differ")
    function_digest = hashlib.sha256(json.dumps({"common_weights": common_on,
        "physics": asdict(physics), "common_config": left, "known_h": float(known_h)}, sort_keys=True).encode()).hexdigest()
    on_values, off_values = _probe(on), _probe(off)
    agreement = {key+"_max_abs": float(np.max(abs(on_values[key]-off_values[key]))) for key in on_values}
    agreement.update(relative_tolerance=1.e-12, absolute_tolerance=1.e-10,
        passed=all(np.allclose(on_values[key], off_values[key], rtol=1.e-12, atol=1.e-10) for key in on_values))
    if not agreement["passed"]:
        raise ValueError("function-matched initial float64 probes disagree")
    metadata = {}
    for arm, model in (("on", on), ("off", off)):
        zero = all(bool(torch.all(network[0].weight[:, -1] == 0))
                   for network, (kind, _) in zip(on.networks, on.descriptors) if kind != "corner")
        metadata[arm] = {"arm": arm, "neural_seed": neural_seed,
            "common_weights_sha256": common_on, "shared_function_sha256": function_digest,
            "complete_state_sha256": _state_digest(model),
            "log_column_zero": zero if arm == "on" else None,
            "on_only_log_column_initially_zero": zero,
            "log_column_present": model.config.bottom_log_feature,
            "probe_agreement": agreement}
        model.initialization_metadata = metadata[arm]
    proof = {"on": metadata["on"], "off": metadata["off"], "probe_agreement": agreement,
             "probe_coordinates_normalized": INITIALIZATION_PROBE_POINTS.tolist(),
             "probe_coordinate_sha256": _array_digest(INITIALIZATION_PROBE_POINTS),
             "comparison": "exact common parameters; float64 function and coordinate-derivative agreement, not bitwise numerical equality"}
    return on, off, proof


def verify_matched_initialization(physics, config_on, config_off, known_h, neural_seed):
    return _matched_pair(physics, config_on, config_off, known_h, neural_seed)[2]


def matched_graded_model(physics, config, known_h, neural_seed):
    on, off, proof = _matched_pair(physics, replace(config, bottom_log_feature=True),
                                  replace(config, bottom_log_feature=False), known_h, neural_seed)
    arm = "on" if config.bottom_log_feature else "off"
    return (on if arm == "on" else off), proof[arm]


def _graded_points(physics, config, seed):
    if config.interior_points != 8192:
        points = consistent_training_points(physics, config, seed)
        strata = [{"name": "constructed_small_grid", "start": 0, "stop": len(points["interior"]),
                   "count": len(points["interior"]), "mean_weight": 1.}]
        return points, strata
    points = training_points(physics, replace(config, interior_points=4096), seed)
    rng = np.random.default_rng(seed)
    shift_x, shift_z = rng.uniform(.25, .75), rng.uniform(.25, .75)
    times = np.array([.2, .5, 1., 2., 3., 5., 8., 12., 18., 26., 36., 50., 68., 88., 105., 119.])/config.duration
    edges = [edge for edge in (physics.contact_start, physics.contact_end) if 0 < edge < physics.width]
    nz = 8 if edges else 12
    xx, zz, tt = np.meshgrid((np.arange(32)+shift_x)/32, (np.arange(nz)+shift_z)/nz, times, indexing="ij")
    blocks = [("bulk", np.column_stack((xx.ravel(), zz.ravel(), tt.ravel())))]
    xx, zz, tt = np.meshgrid((np.arange(16)+shift_x)/16, np.geomspace(.001, .2, 8), times, indexing="ij")
    blocks.append(("bottom_band", np.column_stack((xx.ravel(), zz.ravel(), tt.ravel()))))
    if edges:
        radii = physics.thickness*np.geomspace(.001, 1., 16//len(edges))
        angles = math.pi*(np.arange(8)+.5)/8
        rows = []
        for edge in edges:
            rr, aa, tt = np.meshgrid(radii, angles, times, indexing="ij")
            rows.append(np.column_stack(((edge+rr.ravel()*np.cos(aa.ravel()))/physics.width,
                                         rr.ravel()*np.sin(aa.ravel())/physics.thickness, tt.ravel())))
        blocks.append(("corner_patch", np.concatenate(rows)))
    start, strata = 0, []
    for name, block in blocks:
        stop = start+len(block)
        strata.append({"name": name, "start": start, "stop": stop,
                       "count": len(block), "mean_weight": len(block)/8192})
        start = stop
    points["interior"] = np.concatenate([block for _, block in blocks])
    if len(points["interior"]) != 8192 or np.any(points["interior"][:, :2] <= 0) or np.any(points["interior"][:, :2] >= 1):
        raise ValueError("graded interior construction escaped its declared grid")
    return points, strata


def graded_training_points(physics, config, seed):
    return _graded_points(physics, config, seed)[0]


def graded_point_metadata(physics, config, seed):
    points, strata = _graded_points(physics, config, seed)
    identities = {key: {"shape": list(value.shape), "sha256": _array_digest(value)} for key, value in sorted(points.items())}
    return {"point_digest": hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest(),
            "point_identities": identities, "point_counts": {k: len(v) for k, v in points.items()},
            "strata": strata, "interior_min_height_m": float(points["interior"][:, 1].min()*physics.thickness),
            "interior_max_height_m": float(points["interior"][:, 1].max()*physics.thickness),
            "objective_weights": "uniform mean over all interior rows; each bottom family retains its own uniform mean"}


def fit_graded(observations, *, physics=PlanarContactConfig(), config=GradedConfig(),
               known_h, neural_seed, sampling_seed, progress=None):
    torch.set_num_threads(1)
    started = perf_counter()
    model, initialization = matched_graded_model(physics, config, known_h, neural_seed)
    _observation_tensors(model, observations)
    points = graded_training_points(physics, config, sampling_seed)
    point_metadata = graded_point_metadata(physics, config, sampling_seed)
    initial = project_output(model, points, observations)
    initial.update(stage="initial_projection", step=0)
    if progress:
        progress(initial)
    updates = []
    optimizer = torch.optim.Adam(model.networks.parameters(), lr=config.learning_rate)
    for step in range(config.adam_epochs):
        event = variable_projection_step(model, points, observations, optimizer)
        event.update(stage="hidden_variable_projection", step=step+1)
        updates.append(event)
        if progress:
            progress(event)
        optimizer = torch.optim.Adam(model.networks.parameters(), lr=config.learning_rate)
    final = project_output(model, points, observations)
    final.update(stage="final_projection", step=config.adam_epochs)
    if progress:
        progress(final)
    total, obs, physical, components = objective_continuity(model, points, observations)
    projections = [initial]+[trial["projection"] for update in updates for trial in update["trials"] if "projection" in trial]+[final]
    report = {"architecture": "matched_graded_hermite_variable_projection_mlp",
        "model_module": __name__, "model_class": "GradedPINN", "mode": "known_contact",
        "known_h_oracle": True, "known_h": float(known_h), "config": asdict(config), "physics": asdict(physics),
        "neural_seed": neural_seed, "sampling_seed": sampling_seed, "initialization": initialization,
        "graded_point_metadata": point_metadata, "training_points": {k: v.tolist() for k, v in points.items()},
        "training_point_weights": {k: np.full(len(v), 1/len(v)).tolist() for k, v in points.items()},
        "training_point_counts": {k: len(v) for k, v in points.items()},
        "normalized_observation_mse": float(obs.detach()), "final_physics_loss": float(physical.detach()),
        "final_objective": float(total.detach()), "training_family_mse": {k: float(v.detach()) for k, v in components.items()},
        "projections": projections, "hidden_update_history": updates,
        "hidden_rejected_updates": sum(not event["accepted"] for event in updates),
        "hidden_backtrack_count": sum(event["backtracks"] for event in updates),
        "hidden_accepted_objective_increases": sum(event["objective_after"] > event["objective_before"] for event in updates),
        "optimizer_resets": 1+config.adam_epochs, "coefficient_count": len(model.coefficients),
        "base_coefficient_count": model.base_coefficient_count, "wall_seconds": perf_counter()-started,
        "checkpoint": "fixed-budget final state with inherited monotone complete-objective variable projection",
        "gradient_semantics": "fixed-readout proposal; no exact profiled-gradient claim with truncated SVD",
        "hidden_truth_labels_used": False, "independent_diagnostics_used_for_training": False}
    return model, report


predict_graded = predict_continuity
diagnose_graded = diagnose_continuity


def save_checkpoint(model, destination):
    torch.save({"state_dict": model.state_dict(), "config": asdict(model.config), "physics": asdict(model.physics),
        "known_h": model.known_h, "initialization": getattr(model, "initialization_metadata", None),
        "model_module": __name__, "model_class": "GradedPINN",
        "architecture": "matched_graded_hermite_variable_projection_mlp"}, destination)


def checkpoint_bytes(model):
    destination = io.BytesIO()
    save_checkpoint(model, destination)
    return destination.getvalue()
