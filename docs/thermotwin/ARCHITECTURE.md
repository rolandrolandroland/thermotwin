# ThermoTwin package architecture

## Purpose

ThermoTwin is organized by scientific responsibility and dependency level.
The structure keeps the equations and conventional solvers independent of
PyTorch and Matplotlib, while preserving all historical imports and commands.
This refactor changes code ownership, not physics or numerical behavior.

## Dependency direction

Dependencies should point downward through these layers:

```text
reports
  -> studies / pinn / inference / design
       -> observations / simulation
            -> physics / core / numerics
```

The allowed responsibilities are:

| Package | Responsibility | Optional dependency? |
| --- | --- | --- |
| `thermotwin.core` | Shared current-input structures with no scientific-layer dependencies | No |
| `thermotwin.physics` | Thermoelectric equations and two-/four-node thermal models | No |
| `thermotwin.numerics` | Small matrices, interpolation, integration, bracketing, quantiles | No |
| `thermotwin.simulation` | Diagnostics and reproducible reference simulations | No |
| `thermotwin.observations` | Sensor schemas, noise, bias, lag, missingness, quality, hardware CSV | No |
| `thermotwin.inference` | Conventional inverse problems, identifiability, experiment selection | No |
| `thermotwin.pinn` | Forward and inverse physics-informed neural networks | PyTorch |
| `thermotwin.design` | Operating maps, controls, PWM, materials, product co-design | No |
| `thermotwin.studies` | Frozen sensitivity and robustness campaigns | No |
| `thermotwin.reports` | Figures and interview-ready presentation artifacts | Matplotlib; some reports also use PyTorch |

Low-level packages must not import from reports, studies, or PINNs. Importing
`thermotwin` must not load PyTorch or Matplotlib.

## Stable public API and compatibility

`thermotwin.__init__` exposes the established dependency-light convenience
API through `_public_api.py`. Feature-specific work should prefer explicit
layered imports, for example:

```python
from thermotwin.physics import ThermoelectricParameters, cold_side_heat
from thermotwin.core.controls import PiecewiseConstantCurrent
from thermotwin.observations.noise import GaussianTemperatureNoise
from thermotwin.design.codesign import CodesignCampaignConfig
```

Historical paths such as `thermotwin.transient`,
`thermotwin.measurement_noise`, and `thermotwin.forward_pinn` are compatibility
facades. They re-export the same objects and keep existing notebooks, tests,
README commands, and external callers working. They should contain no new
implementation logic.

## Co-design decomposition

The former single `material_geometry_codesign.py` implementation is split into:

| Module | Ownership |
| --- | --- |
| `design/codesign/models.py` | Immutable configuration, application, design, and result records |
| `design/codesign/evaluation.py` | Material-to-module scaling, steady physics, constraints, current selection |
| `design/codesign/sampling.py` | Latin hypercube generation and feature encoding |
| `design/codesign/optimization.py` | Gaussian process, expected improvement, BO/random comparison |
| `design/codesign/robustness.py` | Fixed-current as-built perturbation study |
| `design/codesign/campaign.py` | End-to-end orchestration and text report |

This division prevents changes to the optimizer from silently changing module
physics, and makes the scientific assumptions easier to test independently.

## Documentation and generated artifacts

The concise and detailed READMEs, frozen experiment walkthroughs, and learning
notes remain at their established paths to preserve hundreds of working links.
They are documentation assets, not runtime dependencies. Generated figures
remain under `thermotwin/figures/`, are ignored by Git, and are never imported.

## Adding a feature

1. Put equations and state balances in `physics` or reusable algorithms in
   `numerics`.
2. Put reproducible execution scaffolding in `simulation`.
3. Add observation transforms only in `observations`.
4. Put estimation in `inference` or optional neural estimation in `pinn`.
5. Put frozen experiments in `studies` or product/control searches in `design`.
6. Put plotting and presentation in `reports`.
7. Add tests at the same responsibility level and at least one compatibility
   test when replacing an established public path.
8. Update both READMEs and the relevant public walkthrough.

## Migration policy

Compatibility facades can be removed only in a future major version after an
explicit deprecation period. Until then, new implementation must go into the
layered modules, not back into the flat facades.
