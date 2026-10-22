"""Shared-trace neural fields with physics-based linear readout projection.

This separate development method leaves the earlier spatial studies unchanged.
Temperature and normal Fourier flux agree identically across the artificial
vertical interfaces.  The mixed bottom condition is imposed through residuals;
it is not combined with incompatible smooth corner endpoint constraints.
No reference solver, hidden field values, or inverse truth labels train this
field.  The known contact coefficient belongs only to a forward diagnostic.
"""
from dataclasses import asdict, dataclass
import io
import math
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.func import jacfwd, vmap

from ..physics.planar_contact import PlanarContactConfig
from .planar_contact_inverse import PlanarObservations


@dataclass(frozen=True)
class ContinuityConfig:
    width: int = 24
    layers: int = 2
    adam_epochs: int = 120
    projection_every: int = 20
    interior_points: int = 4096
    bottom_points: int = 512
    learning_rate: float = .0001
    gradient_clip: float = 1.
    max_backtracks: int = 12
    physics_weight: float = 100.
    observation_weight: float = 1.
    rank_rcond: float = 1.e-7
    projection_chunk: int = 128
    bottom_log_feature: bool = True
    bottom_log_scale: float = .001
    corner_enrichment: bool = True
    duration: float = 120.
    initial_temperature: float = 300.
    temperature_scale: float = 5.
    ramp_seconds: float = 10.
    bath_drop_kelvin: float = 5.
    top_flux_watts_per_square_metre: float = 500.
    interface_temperature_scale: float = .1

    def __post_init__(self):
        for name in ("width", "layers", "projection_every", "interior_points",
                     "bottom_points", "projection_chunk"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive integer")
        if type(self.adam_epochs) is not int or self.adam_epochs < 0:
            raise ValueError("adam_epochs must be a nonnegative integer")
        if type(self.max_backtracks) is not int or self.max_backtracks < 0:
            raise ValueError("max_backtracks must be a nonnegative integer")
        for name in ("learning_rate", "gradient_clip", "physics_weight", "rank_rcond",
                     "bottom_log_scale", "duration", "initial_temperature",
                     "temperature_scale", "ramp_seconds", "bath_drop_kelvin",
                     "top_flux_watts_per_square_metre", "interface_temperature_scale"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be finite and positive")
        if not math.isfinite(self.observation_weight) or self.observation_weight < 0:
            raise ValueError("observation_weight must be finite and nonnegative")
        if not self.rank_rcond < 1 or not self.bottom_log_scale < 1:
            raise ValueError("rank cutoff and bottom scale must be below one")
        if type(self.bottom_log_feature) is not bool or type(self.corner_enrichment) is not bool:
            raise TypeError("feature switches must be boolean")


def _trunk(inputs, config):
    layers = []
    for index in range(config.layers):
        layer = nn.Linear(inputs if index == 0 else config.width,
                          config.width, dtype=torch.float64)
        nn.init.xavier_normal_(layer.weight)
        nn.init.zeros_(layer.bias)
        layers.extend((layer, nn.Tanh()))
    return nn.Sequential(*layers)


def corner_function(s, y):
    """Continuous Re(w log(w)); derivatives at w=0 are undefined.

    A safe explicit value extension avoids NaN even in branches discarded by
    ``where``. Strong derivative callers must reject the exact corner.
    """
    corner = (s == 0) & (y == 0)
    safe_s = torch.where(corner, torch.ones_like(s), s)
    safe_y = torch.where(corner, torch.zeros_like(y), y)
    value = .5*s*torch.log(safe_s**2+safe_y**2)-y*torch.atan2(safe_y, safe_s)
    return torch.where(corner, torch.zeros_like(value), value)


class ContinuityPINN(nn.Module):
    """Piecewise C2, exactly C1 neural trial field away from bottom corners.

    Shared A traces specify temperature; shared B traces specify H times the
    x derivative.  A cubic Hermite interpolant and an interior bubble give
    each region its own field without separate interface penalties.
    """
    def __init__(self, physics=PlanarContactConfig(), config=ContinuityConfig(),
                 *, known_h, mode="known_contact"):
        super().__init__()
        if mode != "known_contact":
            raise ValueError("this forward-development field requires known contact")
        if not math.isfinite(known_h) or known_h <= 0:
            raise ValueError("known_h must be finite and positive")
        self.physics, self.config, self.known_h = physics, config, float(known_h)
        self.mode = mode
        self.regions = []
        if physics.contact_start > 0:
            self.regions.append((0., physics.contact_start, False))
        self.regions.append((physics.contact_start, physics.contact_end, True))
        if physics.contact_end < physics.width:
            self.regions.append((physics.contact_end, physics.width, False))
        n = len(self.regions)
        descriptors = [("A", j) for j in range(n+1)]
        descriptors += [("B", j) for j in range(1, n)]
        descriptors += [("C", j) for j in range(n)]
        self.corner_edges = ([edge for edge in (physics.contact_start, physics.contact_end)
                              if 0 < edge < physics.width] if config.corner_enrichment else [])
        descriptors += [("corner", j) for j in range(len(self.corner_edges))]
        self.descriptors = descriptors
        shared_inputs = 4 + int(config.bottom_log_feature)
        self.networks = nn.ModuleList([
            _trunk(3 if kind == "corner" else shared_inputs + int(kind == "C"), config)
            for kind, _ in descriptors])
        self.coefficients = nn.Parameter(torch.zeros(len(descriptors)*(config.width+1),
                                                     dtype=torch.float64), requires_grad=False)

    @property
    def contact(self):
        return torch.tensor(self.known_h, dtype=torch.float64)

    def _time_features(self, time):
        p, c = self.physics, self.config
        scales = (p.thickness**2/p.diffusivity, c.ramp_seconds,
                  p.width**2/(math.pi**2*p.diffusivity))
        return torch.cat([2*(-torch.expm1(-time*c.duration/scale))-1
                          for scale in scales], dim=1)

    def base_temperature(self, q):
        """Dimensionless known lifting for the imposed top heat flux."""
        p, c = self.physics, self.config
        ramp = -torch.expm1(-q[:, 2:3]*c.duration/c.ramp_seconds)
        return c.top_flux_watts_per_square_metre*p.thickness/(2*p.thermal_conductivity*c.temperature_scale)*ramp*q[:, 1:2]**2

    def _corner_feature(self, q, edge):
        p = self.physics
        xi = -torch.cos(math.pi*q[:, 0:1])
        edge_xi = -math.cos(math.pi*edge/p.width)
        s = xi-edge_xi
        eta = 1-(1-q[:, 1:2])**2
        factor = math.pi*p.thickness*math.sin(math.pi*edge/p.width)/(2*p.width)
        y = factor*eta
        # F is continuous at the corner but its derivatives are singular there.
        # Safe operands prevent the unused atan2/log branch from producing NaN
        # in AD; exact corner residuals are explicitly forbidden below.
        exact_corner = (q[:, 0:1] == edge/p.width) & (q[:, 1:2] == 0)
        return corner_function(torch.where(exact_corner, torch.zeros_like(s), s), y)

    def _basis_region(self, q, index):
        p, c = self.physics, self.config
        a, b, _ = self.regions[index]
        u = (q[:, 0:1]*p.width-a)/(b-a)
        h00, h10 = 2*u**3-3*u**2+1, u**3-2*u**2+u
        h01, h11 = -2*u**3+3*u**2, u**3-u**2
        zeta = 1-(1-q[:, 1:2])**2
        times = self._time_features(q[:, 2:3])
        inputs = [2*zeta-1, times]
        if c.bottom_log_feature:
            inputs.append(2*torch.log1p(zeta/c.bottom_log_scale)/math.log1p(1/c.bottom_log_scale)-1)
        shared_input = torch.cat(inputs, dim=1)
        ramp = -torch.expm1(-q[:, 2:3]*c.duration/c.ramp_seconds)
        blocks = []
        for network, (kind, j) in zip(self.networks, self.descriptors):
            multiplier = None
            if kind == "A" and j == index:
                multiplier = h00
            elif kind == "A" and j == index+1:
                multiplier = h01
            elif kind == "B" and j == index:
                multiplier = (b-a)/p.thickness*h10
            elif kind == "B" and j == index+1:
                multiplier = (b-a)/p.thickness*h11
            elif kind == "C" and j == index:
                multiplier = 16*u**2*(1-u)**2
            elif kind == "corner":
                multiplier = self._corner_feature(q, self.corner_edges[j])
            if multiplier is None:
                blocks.append(torch.zeros((len(q), c.width+1), dtype=q.dtype, device=q.device))
                continue
            network_input = times if kind == "corner" else shared_input
            if kind == "C":
                local_x = (q[:, 0:1]*p.width-(a+b)/2)/p.thickness
                network_input = torch.cat((local_x, shared_input), dim=1)
            features = network(network_input)
            features = torch.cat((features, torch.ones_like(q[:, 0:1])), dim=1)
            blocks.append(ramp*multiplier*features)
        return torch.cat(blocks, dim=1)

    def basis(self, q, *, region=None):
        if region is not None:
            return self._basis_region(q, region)
        result = torch.zeros((len(q), len(self.coefficients)), dtype=q.dtype, device=q.device)
        x = q[:, 0]*self.physics.width
        for index, (a, b, _) in enumerate(self.regions):
            mask = (x >= a) & ((x <= b) if index == len(self.regions)-1 else (x < b))
            if bool(mask.any()):
                result[mask] = self._basis_region(q[mask], index)
        return result

    linear_basis = basis

    def dimensionless_temperature(self, q, *, region=None):
        return self.base_temperature(q)+self.basis(q, region=region)@self.coefficients[:, None]

    def forward(self, coordinates):
        p, c = self.physics, self.config
        scales = torch.tensor([p.width, p.thickness, c.duration], dtype=coordinates.dtype,
                              device=coordinates.device)
        return c.initial_temperature+c.temperature_scale*self.dimensionless_temperature(coordinates/scales)[:, 0]


def _grad(value, q):
    return torch.autograd.grad(value, q, torch.ones_like(value), create_graph=True)[0]


def _check_corners(model, values):
    for edge in model.corner_edges:
        if np.any((values[:, 0] == edge/model.physics.width) & (values[:, 1] == 0)):
            raise ValueError("strong residuals exclude the singular bottom contact corners")


def residuals_continuity(model, points):
    """The v2 strong residuals and their unchanged physical scales."""
    p, c = model.physics, model.config
    result = {}
    for name, values in points.items():
        values = np.asarray(values, dtype=float)
        _check_corners(model, values)
        q = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        if name.startswith("interface_"):
            index = int(name.split("_")[1])
            left = model.dimensionless_temperature(q, region=index)
            right = model.dimensionless_temperature(q, region=index+1)
            result[name+"_temperature"] = (left-right)*c.temperature_scale/c.interface_temperature_scale
            result[name+"_flux"] = p.thermal_conductivity*c.temperature_scale/p.width*(_grad(left, q)[:, :1]-_grad(right, q)[:, :1])/c.top_flux_watts_per_square_metre
            continue
        theta = model.dimensionless_temperature(q)
        gradient = _grad(theta, q)
        if name == "interior":
            xx = _grad(gradient[:, :1], q)[:, :1]
            zz = _grad(gradient[:, 1:2], q)[:, 1:2]
            result[name] = gradient[:, 2:3]-c.duration*p.diffusivity*(xx/p.width**2+zz/p.thickness**2)
        elif name in ("left", "right"):
            result[name] = p.thermal_conductivity*c.temperature_scale/p.width*gradient[:, :1]/c.top_flux_watts_per_square_metre
        elif name in ("top", "bottom_contact", "bottom_insulated"):
            flux = p.thermal_conductivity*c.temperature_scale/p.thickness*gradient[:, 1:2]
            ramp = -torch.expm1(-q[:, 2:3]*c.duration/c.ramp_seconds)
            if name == "top":
                flux = flux-c.top_flux_watts_per_square_metre*ramp
            elif name == "bottom_contact":
                flux = flux-model.contact*(c.temperature_scale*theta+c.bath_drop_kelvin*ramp)
            result[name] = flux/c.top_flux_watts_per_square_metre
        else:
            raise ValueError("unknown residual family: "+name)
    return result


continuity_residuals = residuals_continuity


def training_points(physics, config, seed):
    """One declared shifted tensor grid, retained unchanged during fitting."""
    rng = np.random.default_rng(seed)
    n = config.interior_points
    nt, nz = ((16, 8) if n >= 1024 else (8, 4) if n >= 128 else (4, 2))
    nx = max(1, math.ceil(n/(nt*nz)))
    x = (np.arange(nx)+rng.uniform(.25, .75))/nx
    z = (np.arange(nz)+rng.uniform(.25, .75))/nz
    times = np.array([.2, .5, 1., 2., 3., 5., 8., 12., 18., 26., 36., 50., 68., 88., 105., 119.])
    time = times[np.linspace(0, len(times)-1, nt).round().astype(int)]/config.duration
    xx, zz, tt = np.meshgrid(x, z, time, indexing="ij")
    interior = np.column_stack((xx.ravel(), zz.ravel(), tt.ravel()))[:n]
    points = {"interior": interior}
    intervals = {"bottom_contact": [(physics.contact_start, physics.contact_end)],
                 "bottom_insulated": [(0., physics.contact_start), (physics.contact_end, physics.width)]}
    for name, ranges in intervals.items():
        ranges = [(a, b) for a, b in ranges if b > a]
        if not ranges:
            continue
        count = config.bottom_points
        lengths = np.array([b-a for a, b in ranges])
        choice = rng.choice(len(ranges), size=count, p=lengths/lengths.sum())
        bottom = np.zeros((count, 3))
        for index, (a, b) in enumerate(ranges):
            mask = choice == index
            bottom[mask, 0] = (a+(b-a)*rng.uniform(.0001, .9999, mask.sum()))/physics.width
        # Explicit equal early/late time allocations, independent of diagnostics.
        bottom[:, 2] = time[np.arange(count) % len(time)]
        points[name] = bottom
    return points


def _observation_tensors(model, observations):
    if not isinstance(observations, PlanarObservations):
        raise TypeError("only observed point temperatures may train the field")
    p, c = model.physics, model.config
    coordinates = observations.coordinates
    if (coordinates < 0).any() or (coordinates[:, 0] > p.width).any() or (coordinates[:, 1] > p.thickness).any() or (coordinates[:, 2] > c.duration).any():
        raise ValueError("observations lie outside the physical space-time domain")
    return (torch.tensor(coordinates.copy(), dtype=torch.float64),
            torch.tensor(observations.temperature.copy(), dtype=torch.float64))


def objective_continuity(model, points, observations):
    coordinates, temperatures = _observation_tensors(model, observations)
    observation = torch.mean(((model(coordinates)-temperatures)/observations.standard_deviation)**2)
    residuals = residuals_continuity(model, points)
    components = {name: torch.mean(values*values) for name, values in residuals.items()}
    physics = sum(components.values())
    total = model.config.observation_weight*observation+model.config.physics_weight*physics
    if not torch.isfinite(total):
        raise FloatingPointError("nonfinite continuity objective")
    return total, observation, physics, components


@torch.no_grad()
def _operator_columns(model, values, name, region):
    """Coordinate derivatives of [known lift, linear neural features]."""
    p, c = model.physics, model.config
    def features(q):
        row = q.unsqueeze(0)
        return torch.cat((model.base_temperature(row), model.basis(row, region=region)), dim=1)[0]
    derivative = jacfwd(features)
    q = torch.tensor(values, dtype=torch.float64)
    f = vmap(features)(q)
    gradient = vmap(derivative)(q)
    if name == "interior":
        hessian = vmap(jacfwd(derivative))(q)
        operator = gradient[:, :, 2]-c.duration*p.diffusivity*(hessian[:, :, 0, 0]/p.width**2+hessian[:, :, 1, 1]/p.thickness**2)
    elif name in ("left", "right"):
        operator = p.thermal_conductivity*c.temperature_scale/p.width/c.top_flux_watts_per_square_metre*gradient[:, :, 0]
    else:
        operator = p.thermal_conductivity*c.temperature_scale/p.thickness/c.top_flux_watts_per_square_metre*gradient[:, :, 1]
        ramp = -torch.expm1(-q[:, 2]*c.duration/c.ramp_seconds)
        if name == "top":
            operator[:, 0] = operator[:, 0]-ramp
        elif name == "bottom_contact":
            operator = operator-model.known_h*c.temperature_scale/c.top_flux_watts_per_square_metre*f
            operator[:, 0] = operator[:, 0]-model.known_h*c.bath_drop_kelvin/c.top_flux_watts_per_square_metre*ramp
    return operator.detach().numpy()


def linear_system(model, points, observations):
    """Return A,b such that ||A c-b||² is the exact declared objective."""
    coordinates, _ = _observation_tensors(model, observations)
    p, c = model.physics, model.config
    matrices, targets, families = [], [], []
    for name, values in points.items():
        if name.startswith("interface_"):
            raise ValueError("shared interfaces are identities, not projection penalties")
        values = np.asarray(values, dtype=float)
        _check_corners(model, values)
        columns = np.empty((len(values), len(model.coefficients)+1))
        x = values[:, 0]*p.width
        for index, (a, b, _) in enumerate(model.regions):
            indices = np.flatnonzero((x >= a) & ((x <= b) if index == len(model.regions)-1 else (x < b)))
            for start in range(0, len(indices), c.projection_chunk):
                subset = indices[start:start+c.projection_chunk]
                columns[subset] = _operator_columns(model, values[subset], name, index)
        weight = math.sqrt(c.physics_weight/len(values))
        matrices.append(columns[:, 1:]*weight)
        targets.append(-columns[:, 0]*weight)
        families.append({"name": name, "rows": len(values), "row_scale": weight})
    scales = torch.tensor([p.width, p.thickness, c.duration], dtype=torch.float64)
    q = coordinates/scales
    with torch.no_grad():
        basis = model.basis(q).numpy()
        base = model.base_temperature(q)[:, 0].numpy()
    weight = math.sqrt(c.observation_weight/len(q))/observations.standard_deviation
    matrices.append(basis*c.temperature_scale*weight)
    targets.append((observations.temperature-c.initial_temperature-c.temperature_scale*base)*weight)
    families.append({"name": "observations", "rows": len(q), "row_scale": float(weight)})
    return np.concatenate(matrices), np.concatenate(targets), {"families": families, "coefficient_count": len(model.coefficients)}


def project_output(model, points, observations):
    """Stable linear projection, accepted only if the same-point loss decreases."""
    started = perf_counter()
    before = float(objective_continuity(model, points, observations)[0].detach())
    original = model.coefficients.detach().numpy().copy()
    matrix, rhs, metadata = linear_system(model, points, observations)
    norms = np.linalg.norm(matrix, axis=0)
    scales = np.maximum(norms, 1.e-12)
    normalized = matrix/scales
    solution, _, rank, singular = np.linalg.lstsq(normalized, rhs, rcond=model.config.rank_rcond)
    proposed = solution/scales
    if not np.isfinite(proposed).all():
        raise FloatingPointError("nonfinite projected readout")
    with torch.no_grad():
        model.coefficients.copy_(torch.tensor(proposed, dtype=torch.float64))
    try:
        after = float(objective_continuity(model, points, observations)[0].detach())
    except FloatingPointError:
        after = math.inf
    accepted = math.isfinite(after) and after <= before
    if not accepted:
        with torch.no_grad():
            model.coefficients.copy_(torch.tensor(original, dtype=torch.float64))
    finite_singular = singular[singular > singular[0]*model.config.rank_rcond] if len(singular) else []
    with np.errstate(over="ignore", invalid="ignore"):
        proposed_norm = float(np.linalg.norm(proposed))
        algebraic_objective = float(np.sum((matrix@proposed-rhs)**2))
    return {"accepted": accepted, "objective_before": before,
            "objective_proposed": after if math.isfinite(after) else None,
            "proposed_objective_finite": math.isfinite(after),
            "objective_after": after if accepted else before, "rows": len(rhs), "columns": matrix.shape[1],
            "rank": int(rank), "rank_rcond": model.config.rank_rcond,
            "resolved_condition": float(singular[0]/finite_singular[-1]) if len(finite_singular) else None,
            "smallest_singular_value": float(singular[-1]) if len(singular) else None,
            "coefficient_norm_proposed": proposed_norm if math.isfinite(proposed_norm) else None,
            "coefficient_norm_returned": float(torch.linalg.vector_norm(model.coefficients)),
            "algebraic_objective_proposed": algebraic_objective if math.isfinite(algebraic_objective) else None,
            "wall_seconds": perf_counter()-started, "system": metadata,
            "solver": "float64 column-scaled SVD least squares; no normal equations"}


project_linear_readout = project_output


def fit_continuity(observations, *, physics=PlanarContactConfig(), config=ContinuityConfig(),
                   known_h, neural_seed, sampling_seed, progress=None):
    torch.set_num_threads(1)
    torch.manual_seed(neural_seed)
    started = perf_counter()
    model = ContinuityPINN(physics, config, known_h=known_h)
    _observation_tensors(model, observations)
    points = training_points(physics, config, sampling_seed)
    history, projections, update_history = [], [], []
    def project(stage, step):
        record = project_output(model, points, observations)
        record.update({"stage": stage, "step": step})
        projections.append(record)
        if progress:
            progress(record)
    project("initial_projection", 0)
    parameters = list(model.networks.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    optimizer_resets = 1
    for epoch in range(config.adam_epochs):
        optimizer.zero_grad()
        total, obs, physics_loss, _ = objective_continuity(model, points, observations)
        total.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip))
        if not math.isfinite(gradient_norm):
            raise FloatingPointError("nonfinite hidden gradient")
        previous = [parameter.detach().clone() for parameter in parameters]
        optimizer.step()
        proposal = [parameter.detach()-saved for parameter, saved in zip(parameters, previous)]
        before = float(total.detach())
        accepted, after, accepted_scale, backtracks = False, None, None, 0
        for backtracks in range(config.max_backtracks+1):
            step_scale = 2.**(-backtracks)
            if backtracks:
                with torch.no_grad():
                    for parameter, saved, difference in zip(parameters, previous, proposal):
                        parameter.copy_(saved+step_scale*difference)
            try:
                candidate = float(objective_continuity(model, points, observations)[0].detach())
            except FloatingPointError:
                candidate = math.inf
            if math.isfinite(candidate) and candidate <= before:
                accepted, after, accepted_scale = True, candidate, step_scale
                break
        if not accepted:
            with torch.no_grad():
                for parameter, saved in zip(parameters, previous):
                    parameter.copy_(saved)
            after = before
            optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
            optimizer_resets += 1
        update_history.append({"step": epoch+1, "objective_before": before,
                               "objective_after": after, "accepted": accepted,
                               "accepted_scale": accepted_scale, "backtracks": backtracks,
                               "gradient_norm_before_clip": gradient_norm})
        if (epoch+1) % config.projection_every == 0:
            project("periodic_projection", epoch+1)
            optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
            optimizer_resets += 1
        if epoch % 100 == 0 or epoch+1 == config.adam_epochs:
            record = {"stage": "hidden_adam", "step": epoch+1,
                      "objective_before_update": float(total.detach()),
                      "observation_before_update": float(obs.detach()),
                      "physics_before_update": float(physics_loss.detach())}
            history.append(record)
            if progress:
                progress(record)
    project("final_projection", config.adam_epochs)
    total, obs, physics_loss, components = objective_continuity(model, points, observations)
    report = {"architecture": "shared_hermite_continuity_mlp", "mode": "known_contact",
              "known_h_oracle": True, "known_h": float(known_h), "config": asdict(config),
              "physics": asdict(physics), "neural_seed": neural_seed, "sampling_seed": sampling_seed,
              "normalized_observation_mse": float(obs.detach()), "final_physics_loss": float(physics_loss.detach()),
              "final_objective": float(total.detach()),
              "training_family_mse": {key: float(value.detach()) for key, value in components.items()},
              "training_point_counts": {key: len(value) for key, value in points.items()},
              "training_points": {key: value.tolist() for key, value in points.items()},
              "projections": projections, "history": history, "hidden_update_history": update_history,
              "hidden_rejected_updates": sum(not row["accepted"] for row in update_history),
              "hidden_backtrack_count": sum(row["backtracks"] for row in update_history),
              "hidden_accepted_objective_increases": sum(row["objective_after"] > row["objective_before"] for row in update_history),
              "optimizer_resets": optimizer_resets, "wall_seconds": perf_counter()-started,
              "coefficient_count": len(model.coefficients),
              "checkpoint": "fixed-budget final optimizer and monotone readout projection state",
              "hidden_truth_labels_used": False,
              "independent_diagnostics_used_for_training": False,
              "exact_corner_policy": "field value continuous; undefined corner derivatives excluded",
              "identically_satisfied_training_families_omitted": ["left", "right", "top", "interfaces"]}
    return model, report


def predict_continuity(model, physical_coordinates):
    with torch.no_grad():
        return model(torch.tensor(np.asarray(physical_coordinates), dtype=torch.float64)).numpy()


def diagnose_continuity(model, normalizedpoints):
    """Score caller-supplied independent points without fitting or resampling."""
    sums, counts = {}, {}
    for name, values in normalizedpoints.items():
        for start in range(0, len(values), 128):
            for key, residual in residuals_continuity(model, {name: np.asarray(values)[start:start+128]}).items():
                sums[key] = sums.get(key, 0.)+float(torch.sum(residual*residual).detach())
                counts[key] = counts.get(key, 0)+residual.numel()
    normalized = {key: math.sqrt(sums[key]/counts[key]) for key in sums}
    c = model.config
    scales = {key: (c.temperature_scale/c.duration if key == "interior" else
                    c.interface_temperature_scale if key.endswith("_temperature") else
                    c.top_flux_watts_per_square_metre) for key in normalized}
    return {"normalized_rms": normalized, "physical_rms": {key: value*scales[key] for key, value in normalized.items()},
            "normalization_scales": scales, "point_counts": {key: len(value) for key, value in normalizedpoints.items()},
            "all_families_pass": all(math.isfinite(value) and value <= .05 for value in normalized.values()),
            "validation_scheme": "unchanged caller-supplied independent v2 strong residual points"}


def save_checkpoint(model, path):
    """Save only this final state; accepts a path or a binary file-like object."""
    torch.save({"state_dict": model.state_dict(), "config": asdict(model.config),
                "physics": asdict(model.physics), "known_h": model.known_h,
                "architecture": "shared_hermite_continuity_mlp"}, path)


def checkpoint_bytes(model):
    destination = io.BytesIO()
    save_checkpoint(model, destination)
    return destination.getvalue()
