"""Bounded 2D thermal reconstruction and scalar-contact PINN.

The network sees only known geometry/materials/boundary schedules and declared
point observations. It never receives a reference temperature field. The
known-contact mode is an explicitly labelled oracle diagnostic; data-only
training has no observation-to-contact link and emits no contact estimate.
This is a thermal plate benchmark, not a full electrothermal module model.
"""

from dataclasses import asdict, dataclass
import hashlib
import math
from time import perf_counter, process_time

import numpy as np
import torch
from torch import nn

from ..physics.planar_contact import PlanarContactConfig


@dataclass(frozen=True)
class PlanarPINNConfig:
    width: int = 32
    layers: int = 3
    adam_epochs: int = 1800
    lbfgs_iterations: int = 200
    interior_points: int = 384
    boundary_points_per_side: int = 64
    learning_rate: float = .001
    duration: float = 120.
    initial_temperature: float = 300.
    temperature_scale: float = 5.
    ramp_seconds: float = 10.
    bath_drop_kelvin: float = 5.
    top_flux_watts_per_square_metre: float = 500.
    h_bounds: tuple = (400., 2000.)
    initial_h: float = math.sqrt(400. * 2000.)

    def __post_init__(self):
        for key in ('width', 'layers', 'adam_epochs', 'interior_points', 'boundary_points_per_side'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        if type(self.lbfgs_iterations) is not int or self.lbfgs_iterations < 0:
            raise ValueError('LBFGS iterations must be nonnegative')
        for key in ('learning_rate', 'duration', 'initial_temperature', 'temperature_scale',
                    'ramp_seconds', 'bath_drop_kelvin', 'top_flux_watts_per_square_metre'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if (not isinstance(self.h_bounds, (tuple, list)) or len(self.h_bounds) != 2
            or any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in self.h_bounds)
            or not math.isfinite(self.initial_h)
            or not 0 < self.h_bounds[0] < self.initial_h < self.h_bounds[1]):
            raise ValueError('initial contact must lie strictly inside positive bounds')
        object.__setattr__(self, 'h_bounds', tuple(self.h_bounds))


@dataclass(frozen=True)
class PlanarObservations:
    coordinates: np.ndarray  # physical (x,z,t), exactly the observed points
    temperature: np.ndarray
    standard_deviation: float = .02

    def __post_init__(self):
        coordinates = np.asarray(self.coordinates, dtype=float).copy()
        values = np.asarray(self.temperature, dtype=float).copy()
        if (coordinates.ndim != 2 or coordinates.shape[1] != 3 or not len(coordinates)
            or values.shape != (len(coordinates),) or not np.isfinite(coordinates).all()
            or not np.isfinite(values).all()):
            raise ValueError('observations require finite point coordinates and one value per point')
        if not math.isfinite(self.standard_deviation) or self.standard_deviation <= 0:
            raise ValueError('observation noise scale must be positive')
        coordinates.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, 'coordinates', coordinates)
        object.__setattr__(self, 'temperature', values)

    def payload(self):
        return {'coordinates': self.coordinates.tolist(), 'temperature': self.temperature.tolist(),
                'standard_deviation': self.standard_deviation}


class PlanarContactPINN(nn.Module):
    def __init__(self, physics=PlanarContactConfig(), config=PlanarPINNConfig(),
                 *, mode='inverse', known_h=None):
        super().__init__()
        if mode not in ('inverse', 'known_contact', 'data_only'):
            raise ValueError('unknown reconstruction mode')
        if mode == 'known_contact' and (known_h is None or not config.h_bounds[0] <= known_h <= config.h_bounds[1]):
            raise ValueError('oracle diagnostic requires a declared positive known contact')
        if mode != 'known_contact' and known_h is not None:
            raise ValueError('truth contact is forbidden outside the oracle diagnostic')
        self.physics, self.config, self.mode = physics, config, mode
        self.known_h = known_h
        sizes = (3,) + (config.width,) * config.layers + (1,)
        layers = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            linear = nn.Linear(a, b, dtype=torch.float64)
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if i < len(sizes) - 2:
                layers.append(nn.Tanh())
        self.field = nn.Sequential(*layers)
        lo, hi = config.h_bounds
        probability = (config.initial_h - lo) / (hi - lo)
        self.raw_h = nn.Parameter(torch.tensor(math.log(probability / (1-probability)), dtype=torch.float64),
                                  requires_grad=mode == 'inverse')

    @property
    def contact(self):
        if self.mode == 'known_contact':
            return torch.tensor(float(self.known_h), dtype=torch.float64)
        lo, hi = self.config.h_bounds
        return lo + (hi-lo) * torch.sigmoid(self.raw_h)

    def dimensionless_temperature(self, coordinates):
        # Shared hard initial condition for all neural modes, including data-only.
        return coordinates[:, 2:3] * self.field(2*coordinates-1)

    def forward(self, physical_coordinates):
        scale = torch.tensor([self.physics.width, self.physics.thickness, self.config.duration],
                             dtype=torch.float64)
        theta = self.dimensionless_temperature(physical_coordinates / scale)
        return self.config.initial_temperature + self.config.temperature_scale * theta[:, 0]

    def initial_field_digest(self):
        digest = hashlib.sha256()
        for name, tensor in self.field.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().numpy().tobytes())
        return digest.hexdigest()


def collocation_points(config, seed, *, independent=False):
    rng = np.random.default_rng(seed)
    n = config.interior_points * (2 if independent else 1)
    interior = rng.uniform(0., 1., (n, 3))
    # The ramp is initial-condition compatible; exclude only the exact t=0 corner.
    interior[:, 2] = rng.uniform(.001, 1., n)
    boundaries = {}
    for label, axis, value in (('left', 0, 0.), ('right', 0, 1.), ('bottom', 1, 0.), ('top', 1, 1.)):
        points = rng.uniform(0., 1., (config.boundary_points_per_side * (2 if independent else 1), 3))
        points[:, axis] = value
        points[:, 2] = rng.uniform(.001, 1., len(points))
        boundaries[label] = points
    return {'interior': interior, **boundaries}


def _gradient(value, coordinates):
    return torch.autograd.grad(value, coordinates, torch.ones_like(value), create_graph=True)[0]


def residual_tensors(model, points):
    """Separate residual families; rate in K/s, boundary flux in W/m²."""
    p, c = model.physics, model.config
    result = {}
    for label, values in points.items():
        coordinates = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        theta = model.dimensionless_temperature(coordinates)
        gradient = _gradient(theta, coordinates)
        if label == 'interior':
            second_x = _gradient(gradient[:, 0:1], coordinates)[:, 0:1]
            second_z = _gradient(gradient[:, 1:2], coordinates)[:, 1:2]
            result[label] = c.temperature_scale * (
                gradient[:, 2:3] / c.duration
                - p.diffusivity * (second_x / p.width**2 + second_z / p.thickness**2))
        elif label in ('left', 'right'):
            result[label] = p.thermal_conductivity * c.temperature_scale / p.width * gradient[:, 0:1]
        else:
            ramp = 1-torch.exp(-coordinates[:, 2:3] * c.duration / c.ramp_seconds)
            conducted = p.thermal_conductivity * c.temperature_scale / p.thickness * gradient[:, 1:2]
            if label == 'top':
                result[label] = conducted - c.top_flux_watts_per_square_metre * ramp
            else:
                mask = ((coordinates[:, 0:1] * p.width >= p.contact_start)
                        & (coordinates[:, 0:1] * p.width <= p.contact_end)).to(torch.float64)
                excess = c.temperature_scale * theta + c.bath_drop_kelvin * ramp
                result[label] = conducted - mask * model.contact * excess
    return result


def residual_summary(model, points):
    residuals = residual_tensors(model, points)
    c = model.config
    rms = {name: float(torch.sqrt(torch.mean(value**2)).detach()) for name, value in residuals.items()}
    # Mixed boundary physics must not be hidden by a pooled bottom average.
    bottom_x = points['bottom'][:, 0] * model.physics.width
    contact = (bottom_x >= model.physics.contact_start) & (bottom_x <= model.physics.contact_end)
    counts = {}
    for name, subset in (('bottom_contact', contact), ('bottom_insulated', ~contact)):
        counts[name] = int(subset.sum())
        values = residuals['bottom'][torch.tensor(subset)]
        rms[name] = float(torch.sqrt(torch.mean(values**2)).detach()) if len(values) else None
    scales = {name: c.temperature_scale / c.duration if name == 'interior'
              else c.top_flux_watts_per_square_metre for name in rms}
    return {'physical_rms': rms, 'normalization_scales': scales,
            'normalized_rms': {name: None if rms[name] is None else rms[name]/scales[name] for name in rms},
            'bottom_region_point_counts': counts}


def train_planar_contact_pinn(observations, *, physics=PlanarContactConfig(), config=PlanarPINNConfig(),
                              mode='inverse', physics_weight=10., neural_seed=1,
                              collocation_seed=2, diagnostic_seed=3, known_h=None, progress=None):
    if not isinstance(observations, PlanarObservations):
        raise TypeError('training accepts only declared observed point temperatures')
    if not math.isfinite(physics_weight) or physics_weight <= 0:
        raise ValueError('physics weight must be finite and positive')
    if diagnostic_seed == collocation_seed:
        raise ValueError('diagnostic collocation must use a distinct predeclared seed')
    points = observations.coordinates
    if ((points < 0).any() or (points[:, 0] > physics.width).any()
        or (points[:, 1] > physics.thickness).any() or (points[:, 2] > config.duration).any()):
        raise ValueError('observed points must lie within the declared space-time domain')
    torch.set_num_threads(1)
    torch.manual_seed(neural_seed)
    model = PlanarContactPINN(physics, config, mode=mode, known_h=known_h)
    initial_digest = model.initial_field_digest()
    observation_coordinates = torch.tensor(points.copy(), dtype=torch.float64)
    temperatures = torch.tensor(observations.temperature.copy(), dtype=torch.float64)
    collocation = collocation_points(config, collocation_seed)
    started, cpu = perf_counter(), process_time()
    history = []

    def loss():
        observation = torch.mean(((model(observation_coordinates)-temperatures) / observations.standard_deviation)**2)
        if mode == 'data_only':
            physics_loss = torch.zeros((), dtype=torch.float64)
        else:
            residuals = residual_tensors(model, collocation)
            physics_loss = sum(torch.mean((value / (config.temperature_scale/config.duration
                if key == 'interior' else config.top_flux_watts_per_square_metre))**2)
                for key, value in residuals.items())
        total = observation + physics_weight * physics_loss
        if not torch.isfinite(total):
            raise FloatingPointError('nonfinite training objective; retain failed run')
        return total, observation, physics_loss

    parameters = [value for value in model.parameters() if value.requires_grad]
    adam = torch.optim.Adam(parameters, lr=config.learning_rate)
    for epoch in range(config.adam_epochs):
        adam.zero_grad(set_to_none=True)
        total, observation, physics_loss = loss()
        total.backward()
        adam.step()
        history.append({'stage': 'adam', 'step': epoch+1, 'objective_before_update': float(total.detach()),
                        'observation_before_update': float(observation.detach()),
                        'physics_before_update': float(physics_loss.detach())})
        if progress and ((epoch+1) % 100 == 0 or epoch+1 == config.adam_epochs):
            progress({'mode': mode, 'stage': 'adam', 'step': epoch+1,
                      'objective_before_update': float(total.detach())})
    evaluations = 0
    lbfgs_actual_iterations = 0
    if config.lbfgs_iterations:
        optimizer = torch.optim.LBFGS(parameters, lr=1., max_iter=config.lbfgs_iterations,
            max_eval=config.lbfgs_iterations*2, tolerance_grad=1e-7, tolerance_change=1e-10,
            history_size=50, line_search_fn='strong_wolfe')
        def closure():
            nonlocal evaluations
            optimizer.zero_grad(set_to_none=True)
            total, observation, physics_loss = loss()
            total.backward()
            evaluations += 1
            history.append({'stage': 'lbfgs_closure', 'step': evaluations,
                'objective_before_update': float(total.detach()), 'observation_before_update': float(observation.detach()),
                'physics_before_update': float(physics_loss.detach())})
            if progress and evaluations % 25 == 0:
                progress({'mode': mode, 'stage': 'lbfgs_closure', 'step': evaluations,
                          'objective_before_update': float(total.detach())})
            return total
        optimizer.step(closure)
        lbfgs_actual_iterations = int(optimizer.state[parameters[0]].get('n_iter', 0))
    final, observation, physics_loss = loss()
    diagnostic = residual_summary(model, collocation_points(config, diagnostic_seed, independent=True))
    if mode == 'data_only':
        diagnostic['physical_rms']['bottom'] = None
        diagnostic['normalized_rms']['bottom'] = None
        diagnostic['physical_rms']['bottom_contact'] = None
        diagnostic['normalized_rms']['bottom_contact'] = None
        diagnostic['bottom_contact_residual'] = 'not evaluated: data-only has no contact estimate'
    report = {'mode': mode, 'config': asdict(config), 'physics': asdict(physics),
              'physics_weight': physics_weight, 'neural_seed': neural_seed,
              'collocation_seed': collocation_seed, 'diagnostic_seed': diagnostic_seed,
              'initial_field_digest': initial_digest, 'dtype': 'float64', 'device': 'cpu', 'torch_threads': 1,
              'trainable_parameter_count': sum(value.numel() for value in parameters),
              'optimizer_caps_fixed_before_evaluation': True,
              'final_checkpoint': 'optimizer-returned final state; no truth-selected checkpoint',
              'contact_estimate': None if mode == 'data_only' else float(model.contact.detach()),
              'known_contact_oracle': mode == 'known_contact',
              'final_objective': float(final.detach()), 'normalized_observation_mse': float(observation.detach()),
              'final_training_physics_loss': float(physics_loss.detach()),
              'independent_residuals': diagnostic, 'lbfgs_closure_evaluations': evaluations,
              'lbfgs_actual_iterations': lbfgs_actual_iterations,
              'history': history, 'wall_seconds': perf_counter()-started, 'cpu_seconds': process_time()-cpu}
    return model, report


def predict_planar_temperature(model, coordinates):
    with torch.no_grad():
        return model(torch.tensor(np.asarray(coordinates), dtype=torch.float64)).cpu().numpy()
