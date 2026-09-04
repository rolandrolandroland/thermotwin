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
| `thermotwin.physics` | Lumped thermoelectric equations plus the conservative 1-D leg model | No |
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
| `design/literature_materials.py` | Opt-in DOI-backed records that must not alter the indexed baseline catalog |
| `design/material_pair.py` | Explicit p/n pair evaluation shared by source-specific design studies |
| `design/contact_process_window.py` | Cost-free geometry/contact/application process window |
| `design/ag2se_substitution.py` | One-variable matched substitution over the frozen design pool |

## Lumped joint-inference decomposition

The completed lumped identifiability and experiment-selection workflow keeps
ranking, fitting, and presentation separate:

| Module | Ownership |
| --- | --- |
| `inference/experiment_selection.py` | Feasible local-information candidate ranking |
| `inference/joint_thermal_parameters.py` | Bounded multistart fit of contact resistance, face capacitance, and sensor lag with profiled biases |
| `studies/nonlinear_experiment_selection.py` | Paired nonlinear selected/naive/closest-energy trials, coverage, transfer, and profiles |
| `studies/imperfect_inverse_pinn.py` | Repeated inverse-PINN recovery under noise, missingness, and deliberate sensor-model mismatch |
| `pinn/energy_closure.py` | Post-training whole-system rate and switch-safe cumulative energy audit |
| `studies/forward_reconstruction_comparison.py` | Identically initialized sparse/missing physics-informed versus data-only comparison |
| `reports/nonlinear_experiment_selection.py` | Nonlinear-selection evidence and figure |
| `reports/imperfect_inverse_pinn.py` | Imperfect-data inverse-PINN evidence and figure |
| `reports/forward_reconstruction_comparison.py` | Matched reconstruction and energy-closure evidence |
| `reports/release_audit.py` | Recomputes principal public evidence and rejects stale headline values |

This division prevents changes to the optimizer from silently changing module
physics, and makes the scientific assumptions easier to test independently.

## Distributed constitutive-inference decomposition

The function-valued PDE extension follows the same dependency direction:

| Module | Ownership |
| --- | --- |
| `physics/distributed.py` | Property curves, local constitutive laws, conservative fluxes, face and cell balances |
| `simulation/distributed.py` | Transition-split RK4 and frozen distributed regimes |
| `simulation/distributed_independent.py` | Nodal/SSPRK3 synthetic truth that is numerically separate from the inference reference |
| `observations/distributed.py` | Sparse face, voltage, and heat-rate measurements |
| `inference/distributed_identifiability.py` | Noise-normalized sensitivities and singular spectrum |
| `inference/distributed_properties.py` | Conventional continuous property-curve fitting |
| `inference/distributed_regularization.py` | Explicit coefficient roughness shared by conventional and neural estimators |
| `inference/distributed_profile_likelihood.py` | Fixed-coefficient nonlinear profiles and local repeated-interval approximations |
| `inference/distributed_experiment_selection.py` | Local uncertainty and D-optimal candidates |
| `pinn/distributed_forward.py` | Forward PDE PINN with dynamic face boundaries |
| `pinn/distributed_inverse.py` | Shared-property single- and multi-experiment inverse PINNs |
| `studies/distributed_inverse_robustness.py` | Noisy seed trials, predeclared failure gate, and complete-trial summary |
| `studies/distributed_withheld_validation.py` | Whole-regime exclusion, frozen-curve transfer scoring, and predeclared prediction gate |
| `studies/distributed_independent_validation.py` | Independent truth, paired priors, model-mismatch holdouts, and complete-trial retention |
| `studies/distributed_observation_identifiability.py` | Sensor/current ablations, pre-fit rank decisions, multistart diagnostics, and fit rejection |
| `studies/distributed_profile_coverage.py` | Independent-truth nonlinear fits, repeated local intervals, and empirical coverage |
| `reports/distributed_properties.py` | Reproducible text and figure report |
| `reports/distributed_inverse_robustness.py` | Multi-seed robustness report and comparison figure |
| `reports/distributed_withheld_validation.py` | Trial-level transfer report, prediction metrics, and comparison figure |
| `reports/distributed_independent_validation.py` | Independent-truth and matched-regularization report and figure |
| `reports/distributed_observation_identifiability.py` | Observation-sufficiency decisions, diagnostic fits, and comparison figure |
| `reports/distributed_profile_coverage.py` | Nonlinear profile, repeated coverage, and PINN point-estimate report |

## Documentation and generated artifacts

The concise and detailed READMEs and frozen public experiment walkthroughs
remain at their established paths. Private learning notes under
`thermotwin/notes/` are intentionally ignored by Git. Generated figures
remain under `thermotwin/figures/`, are ignored by Git, and are never imported.
The Milestone 7 handoff consists of `TECHNICAL_SUMMARY.md`, `DEMO_SCRIPT.md`,
`PORTFOLIO_BULLETS.md`, and the source-noted five-slide deck under
`docs/thermotwin/`. The release audit recomputes the values quoted by those
artifacts; it does not replace the complete software test suite.

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
