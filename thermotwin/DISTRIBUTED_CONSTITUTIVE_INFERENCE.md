# Distributed constitutive inference

## Question

Can sparse device-level measurements recover temperature-dependent
thermoelectric transport properties while reconstructing the inaccessible
temperature field inside a leg? Which parts of those property curves are
actually distinguishable under the proposed experiments?

This extension exists because the earlier PINNs solve two- and four-node ODEs
that already have fast, accurate conventional solutions. A PINN becomes more
meaningful when the unknown is a spatial field or a function rather than one
scalar in a small lumped model.

The present study is a **synthetic method demonstration**. It is not calibrated
to hardware and does not claim that the declared sensor precision is available
on a particular test stand.

## Model boundary and sign convention

The first distributed topology is one homogeneous, oriented thermoelectric leg.
The spatial coordinate starts at the cold face and ends at the hot face:

```text
cold reservoir -> cold face -> internal finite-volume cells -> hot face -> hot reservoir
                       x = 0                              x = L
```

- Positive current density points from cold to hot.
- Positive heat flux points from cold to hot.
- Positive cold-side heat means heat is removed from the cold face.
- Positive hot-side heat means heat is delivered to the hot face.
- Positive terminal power `VI` enters the thermoelectric system.

The internal cells store energy. Therefore `Q_h - Q_c = VI` is generally a
steady-state relation, not an instantaneous transient identity. During a
transient,

```text
dU/dt = Q_c - Q_h + VI
```

before reservoir and external face-node terms are added to the whole-system
balance.

## Local equations

The electrical and thermal constitutive laws are

```text
E = rho_e(T) J + alpha(T) dT/dx
q = alpha(T) T J - kappa(T) dT/dx
```

where:

| Symbol | Meaning | Units |
| --- | --- | --- |
| `E` | electric field | V/m |
| `J` | signed current density | A/m² |
| `rho_e(T)` | electrical resistivity | Ω·m |
| `alpha(T)` | Seebeck coefficient | V/K |
| `q` | cold-to-hot heat flux | W/m² |
| `kappa(T)` | thermal conductivity | W/(m·K) |

Energy conservation is implemented in conservative form:

```text
rho_m c_p dT/dt = -dq/dx + J E
```

Expanding the fluxes gives

```text
rho_m c_p dT/dt = d/dx[kappa(T) dT/dx]
                  + rho_e(T) J²
                  - tau(T) J dT/dx
```

with the Kelvin relation

```text
tau(T) = T d(alpha)/dT.
```

The last term is Thomson heating or cooling. It vanishes when `alpha` is
constant. This relation makes the learned Seebeck curve and Thomson coefficient
thermodynamically consistent instead of learning them independently.

The face temperatures are dynamic states rather than prescribed labels:

```text
C_c dT_c/dt = G_c(T_c,inf - T_c) + q_c,ext - A q(0,t)
C_h dT_h/dt = G_h(T_h,inf - T_h) + q_h,ext + A q(L,t)
```

That choice makes face temperatures genuine outputs and allows terminal
measurements to inform the hidden material functions.

## Conventional finite-volume reference

The trusted reference discretizes the leg into cell-centred control volumes.
Each cell obeys a conservative energy balance using heat flux through its two
faces plus its electrical work. The face-node ODEs and all cell equations are
advanced with classical RK4. A step is ended exactly at every known current
transition.

The semidiscrete construction has an algebraic whole-system energy audit:

```text
stored-energy rate
  = cold-reservoir heat + hot-reservoir heat
  + external heat inputs + electrical power.
```

This closes to floating-point roundoff at every evaluated state. It does not
depend on numerically differentiating a plotted energy history.

The conventional checks include:

- zero-current equal-temperature equilibrium;
- passive hot-to-cold conduction;
- correct Peltier direction under positive current;
- current-reversal parity: Peltier terms change sign and Joule heating does not;
- exact terminal voltage for a constant-property steady profile;
- recovery of the lumped half-Joule face formulas after the steady parabolic
  temperature profile develops;
- RK4 time-step convergence;
- exact current-transition times; and
- positive-kelvin divergence errors with a time-step remedy.

An important transient distinction emerged from these tests. Immediately after
current is applied to a uniform leg, the boundary Peltier heat is the same at
both faces and the Joule heat initially accumulates internally. The familiar
half-Joule term at each face is a steady-profile result, not an instantaneous
assumption in the distributed model.

## Synthetic property representation

`alpha(T)`, `rho_e(T)`, and `kappa(T)` may be constant or piecewise linear. The
reference temperature-dependent material uses three knots at 285, 300, and
315 K. Mass density and specific heat remain constant in this first extension.

The generic values are deliberately Bi2Te3-like, but they are not a fit to a
specific material record:

| Property | 285 K | 300 K | 315 K |
| --- | ---: | ---: | ---: |
| `alpha` (µV/K) | 195 | 200 | 205 |
| `rho_e` (Ω·m) | 1.05e-5 | 1.00e-5 | 0.96e-5 |
| `kappa` (W/(m·K)) | 1.45 | 1.50 | 1.56 |

The leg is 1.5 mm long with a 2.25 mm² face. The reference uses small generic
face capacitances and reservoir conductances so useful transients appear in a
CPU-sized simulation. These are declared synthetic assumptions.

Piecewise-linear curves use constant endpoint extrapolation only to keep a
slight numerical excursion defined. No report may treat an extrapolated value
as material evidence.

## Observation model

The virtual instrumentation can expose any subset of:

- cold-face temperature;
- hot-face temperature;
- terminal voltage;
- cold-side heat rate; and
- hot-side heat rate.

The default identifiability study uses both face temperatures and voltage.
Heat-rate measurements are optional because they are generally harder to
obtain directly.

Temperatures are interpolated as continuous states. Voltage and heat flux are
then recomputed at the requested time using the right-continuous current. This
avoids the discontinuous-power integration defect that would result from
linearly interpolating voltage across a current switch.

The frozen local analysis assumes:

| Channel | Standard deviation used for sensitivity normalization |
| --- | ---: |
| Face temperature | 0.01 K |
| Voltage | 1e-5 V |
| Heat rate, when enabled | 5e-4 W |

These values define the result. Changing them changes the information matrix.

## Observation-first identifiability gate

Before an inverse PINN is trained, each spline coefficient is perturbed
continuously in log-magnitude coordinates. Central finite differences build a
noise-normalized Jacobian of all visible measurements. The code then evaluates
the singular values of that Jacobian and the corresponding information matrix.

The four frozen regimes are:

1. zero-current thermal relaxation;
2. a positive-current pulse;
3. a negative-current pulse; and
4. a constant-current experiment spanning a 20 K reservoir lift.

This combination separates even- and odd-in-current physics and explores the
property basis over 290–310 K.

### Result

| Curve fitted alone | Effective rank | Singular values | Condition number |
| --- | ---: | --- | ---: |
| `alpha(T)` | 3/3 | 1115, 109.9, 15.02 | 74.2 |
| `rho_e(T)` | 3/3 | 1531, 104.5, 4.812 | 318 |
| `kappa(T)` | 3/3 | 153.3, 20.14, 16.05 | 9.55 |
| All nine coefficients | 9/9 | see generated report | 392 |

Under these exact synthetic assumptions, the local Jacobian is full rank.
`rho_e(T)` is the most poorly conditioned individual curve: its weakest
coefficient combination is far less visible than its strongest. Full local
rank does not prove global uniqueness, robustness to model discrepancy, or
hardware identifiability.

## Conventional inverse baseline

The conventional estimator fits one property curve at a time. It begins with a
bounded coordinate search and finishes with a damped Gauss–Newton update using
continuous finite-difference sensitivities. There is no coefficient grid, so
truth cannot be recovered merely because it lies on a search node.

The frozen study now perturbs each curve separately at the 285, 300, and 315 K
knots:

| Released curve | Truth multipliers |
| --- | --- |
| `alpha(T)` | `(1.03, 1.06, 1.02)` |
| `rho_e(T)` | `(1.04, 1.07, 1.03)` |
| `kappa(T)` | `(0.96, 1.03, 1.08)` |

Four constant-current/temperature-lift experiments are fitted jointly in each
case. The other two constitutive curves remain fixed at baseline. The
noise-free same-model conventional estimator returns every truth to the six
decimal places printed by the report, using both terminal-only and
heat-assisted observation sets for conductivity. This checks the optimizer and
observation mapping. It is an inverse-crime result, not a realistic uncertainty
claim.

## Forward distributed PINN

The forward network maps `(x,t)` to temperature. Its output transformation
enforces the complete initial linear profile exactly. Automatic differentiation
supplies `dT/dt`, `dT/dx`, and the conductive-flux divergence. The loss contains:

- the interior thermoelectric PDE residual;
- the cold face-node energy residual; and
- the hot face-node energy residual.

It uses no finite-volume temperatures as labels. The finite-volume trajectory
is withheld until validation.

With the frozen 800-epoch CPU configuration:

| Metric | Result |
| --- | ---: |
| Physics loss | 5.976663 to 0.012332 |
| Cold-face RMSE | 0.007001 K |
| Hot-face RMSE | 0.007321 K |
| Hidden internal-field RMSE | 0.006420 K |
| Maximum absolute error | 0.015116 K |

This establishes that the network solves the declared PDE and dynamic boundary
conditions. It does not show that the PDE describes hardware.

## Shared-property inverse PINN

A single experiment does not span the property basis. The code therefore uses
one temperature network per experiment while sharing one trainable property
curve across all experiments. This prevents a single network from confusing
different initial and boundary conditions, while every regime updates the same
three spline coefficients.

Each 800-epoch CPU fit starts from multiplier `0.9` at every knot and uses the
same neural seed. No curve sees the finite-volume internal temperatures. The
results are:

| Released curve and observations | Inverse-PINN multipliers | Maximum absolute multiplier error |
| --- | --- | ---: |
| `alpha(T)`: face temperatures + voltage | `(1.046513, 1.058114, 1.014536)` | 0.016513 |
| `rho_e(T)`: face temperatures + voltage | `(1.051672, 1.066605, 1.048003)` | 0.018003 |
| `kappa(T)`: face temperatures + voltage | `(0.850175, 0.838626, 0.800058)` | 0.279942 |
| `kappa(T)`: same channels + both face heat rates | `(0.997626, 1.026726, 1.028525)` | 0.051475 |

The conductivity comparison is intentionally retained. Its terminal-only loss
falls from `9.556869e3` to `1.284848`, yet the recovered curve is badly wrong.
A falling loss is therefore not a recovery criterion. Adding idealized cold-
and hot-side heat-rate observations materially improves the result, but the hot
endpoint remains pulled toward the smoother interior solution. Direct heat
rates are also harder to measure than temperature or voltage, so the
heat-assisted case is a diagnostic about instrumentation, not a free solution.

The conventional solver remains markedly more accurate for all four small
ideal inverse problems. The PINN's value is its joint differentiable
representation of hidden fields, PDE constraints, dynamic boundaries, and a
function-valued unknown—not superior accuracy on an inverse crime. Noise,
multiple seeds, and independent truth were therefore required before any
stronger robustness claim; the follow-on sections add limited versions of all
three without reaching hardware validation.

## Noisy multi-seed follow-on

The next frozen study repeats the resistivity inverse with 0.01 K temperature
noise, 10 µV voltage noise, and five independent neural/noise seed blocks. The
inverse PINN passes all five predeclared coefficient-and-loss gates; the
unregularized conventional fit passes two. Because their regularization is not
matched and truth still uses the inference model, this is a repeatability result
rather than evidence of general PINN superiority. The full trial table, seed
allocation, caveats, and command are in
[`DISTRIBUTED_INVERSE_ROBUSTNESS.md`](DISTRIBUTED_INVERSE_ROBUSTNESS.md).

## Complete-regime transfer validation

The next validation withholds one entire experiment rather than merely holding
out scattered time rows. The fit sees the zero-current relaxation, positive
0.8 A/10 K, and negative 0.8 A/10 K regimes; it does not see the complete
positive 0.4 A/20 K regime. After fitting, the resistivity curve is frozen and
transferred through the conventional distributed solver. Face temperatures,
the hidden internal field, terminal voltage, maximum temperature error, and
energy closure are then compared with the noise-free held-out truth. No
withheld observations are used for refitting.

The frozen five-trial run uses 0.01 K temperature noise, 10 microvolt voltage
noise, 600 CPU inverse-PINN epochs, and the six prediction limits printed by
the report. The inverse PINN passes all six limits in 5/5 trials. The
unregularized conventional fit passes 2/5; its three failures are voltage
RMSE failures above 30 microvolts, not divergent temperature trajectories.
The inverse PINN's mean hidden-field RMSE is 0.000070 K and its worst
pointwise temperature error is 0.000155 K. The conventional mean voltage RMSE
is 34.24744 microvolts, above the fixed limit, despite small mean temperature
errors.

This experiment establishes transfer across one operating regime within the
same synthetic equations, finite-volume grid, time integrator, and three-knot
curve basis. It does not establish extrapolation to a new material law or an
independent discretization. The regularization is also not matched between the
two estimators, and only one curve family and one held-out regime are tested.
The complete setup, every trial, exact thresholds, and reproduction command
are in [`DISTRIBUTED_WITHHELD_VALIDATION.md`](DISTRIBUTED_WITHHELD_VALIDATION.md).

## Independent-truth and matched-regularization validation

The next benchmark removes the exact numerical and curve-basis inverse crime.
Synthetic truth uses 25 nodal temperatures, independently assembled voltage
and heat fluxes, transition-split SSPRK3, and a smooth cubic `rho_e(T)` that
matches the old truth at 285, 300, and 315 K but differs between the knots.
Inference retains the established cell-centred finite-volume/RK4 model and
three-knot curve.

Each estimator is run without an explicit property penalty and with the same
log-coefficient second-difference penalty of weight 25. Across three paired
noise trials, both PINN variants pass the predeclared in-support property and
transfer gate in 3/3 trials. Both conventional variants pass 1/3. Matching the
curvature term reduces conventional voltage error but does not constrain a
poorly observed linear log-slope, so it does not repair the endpoint failures.
The two PINN variants are nearly identical, showing that this explicit penalty
does not explain their stability. Implicit neural and field regularization
remain unmatched, and three trials are not a general estimator comparison.

The full equations, trial table, holdouts, gates, generated figure, and caveats
are in
[`DISTRIBUTED_INDEPENDENT_VALIDATION.md`](DISTRIBUTED_INDEPENDENT_VALIDATION.md).

## Next-experiment selection

Candidate pulse amplitude, duration, and reservoir lift are ranked by the local
posterior information gain for `rho_e(T)`. Candidates must also satisfy declared
temperature, voltage, and power limits.

Among the twelve frozen candidates, the selected experiment is:

```text
20 K lift, -0.8 A pulse, 0.5 s duration
information gain = 6.0338 nats.
```

This is the best point in a finite declared candidate set, not a global
optimum. Negative current is useful here because it reverses Peltier and
Thomson contributions without reversing Joule heating.

## Reproduce the study

Install all optional dependencies and run:

```bash
python3 -m pip install -e '.[all]'
python3 -m thermotwin.distributed_property_report \
  --train-pinn \
  --train-inverse-pinn \
  --pinn-epochs 800 \
  --inverse-pinn-epochs 800
```

or use the installed command:

```bash
thermotwin-distributed-properties \
  --train-pinn \
  --train-inverse-pinn
```

By default, inverse training runs `alpha(T)`, `rho_e(T)`, terminal-only
`kappa(T)`, and heat-assisted `kappa(T)`. To isolate one family, repeat only the
desired selector, for example:

```bash
thermotwin-distributed-properties \
  --train-inverse-pinn \
  --inverse-property thermal_conductivity
```

The figure is written to
`thermotwin/figures/DISTRIBUTED_CONSTITUTIVE_INFERENCE/distributed_property_study.png`
by default, with plotted data in the colocated `distributed_property_study.json`. Generated
figures are ignored by Git.

Run only the new tests with:

```bash
python3 -m unittest \
  tests.test_distributed_physics \
  tests.test_distributed_simulation \
  tests.test_distributed_observations \
  tests.test_distributed_identifiability \
  tests.test_distributed_property_inference \
  tests.test_distributed_experiment_selection \
  tests.test_distributed_forward_pinn \
  tests.test_distributed_inverse_pinn \
  tests.test_distributed_property_report \
  tests.test_distributed_inverse_robustness \
  tests.test_distributed_inverse_robustness_report \
  tests.test_distributed_withheld_validation \
  tests.test_distributed_withheld_validation_report \
  tests.test_distributed_independent \
  tests.test_distributed_independent_validation \
  tests.test_distributed_independent_validation_report \
  tests.test_distributed_profile_coverage \
  tests.test_distributed_profile_coverage_report
```

## Code ownership

| Responsibility | Module |
| --- | --- |
| Property laws, fluxes, PDE-consistent balances | `thermotwin.physics.distributed` |
| RK4 integration and frozen regimes | `thermotwin.simulation.distributed` |
| Sparse terminal observations and noise | `thermotwin.observations.distributed` |
| Sensitivity spectrum and local rank | `thermotwin.inference.distributed_identifiability` |
| Conventional property fitting | `thermotwin.inference.distributed_properties` |
| Uncertainty and next-experiment selection | `thermotwin.inference.distributed_experiment_selection` |
| Forward distributed PINN | `thermotwin.pinn.distributed_forward` |
| Single- and multi-experiment inverse PINNs | `thermotwin.pinn.distributed_inverse` |
| Text and figure report | `thermotwin.reports.distributed_properties` |
| Noisy multi-seed study | `thermotwin.studies.distributed_inverse_robustness` |
| Noisy multi-seed report | `thermotwin.reports.distributed_inverse_robustness` |
| Withheld-regime transfer study | `thermotwin.studies.distributed_withheld_validation` |
| Withheld-regime transfer report | `thermotwin.reports.distributed_withheld_validation` |
| Independent nodal/SSPRK3 truth | `thermotwin.simulation.distributed_independent` |
| Shared explicit coefficient roughness | `thermotwin.inference.distributed_regularization` |
| Fixed-coefficient profiles and repeated local intervals | `thermotwin.inference.distributed_profile_likelihood` |
| Independent-truth paired study | `thermotwin.studies.distributed_independent_validation` |
| Independent-truth report | `thermotwin.reports.distributed_independent_validation` |
| Nonlinear-profile and coverage study | `thermotwin.studies.distributed_profile_coverage` |
| Profile and coverage report | `thermotwin.reports.distributed_profile_coverage` |

## Limits and next steps

- The model is one homogeneous 1-D leg, not a p/n unicouple or full module.
- Current density is spatially uniform; electrical contact and spreading
  resistance are not part of this extension.
- Face capacitances and reservoir links are generic synthetic values.
- Density and heat capacity are constant.
- Radiation, lateral heat flow, ceramic plates, solder layers, and spatially
  varying defects are omitted.
- The identifiability result is local and uses optimistic declared sensor
  precision.
- The first family-comparison inverse data are noise-free; the robustness and
  withheld-transfer studies add same-model synthetic Gaussian noise. The
  independent-truth follow-on changes the grid, integrator, voltage quadrature,
  and property representation, but still assumes the same continuum equations.
- Smoothness regularization contributes to inverse-PINN endpoint estimates.
- Terminal-only conductivity recovery fails in the frozen neural run even
  though the local Jacobian is full rank; idealized heat-rate observations
  improve but do not eliminate the error.
- Multiple switched-current segments are supported by the finite-volume
  reference, but the first distributed PINNs use constant-current experiments;
  switched PINNs require domain decomposition at each discontinuity.
- Missing observations and broader model discrepancy remain before any robust
  inference claim. Nonlinear multistart recovery, representative profiles, and
  a 20-trial independent-truth local-interval audit are now implemented for
  `rho_e(T)`; the trial budget remains too small for precise coverage estimates.
- Hardware validation remains unstarted.

The next scientific step is not to release all three property curves at once.
It is to validate the current local D-optimal experiment recommendation with
complete nonlinear refits against naive candidates. The intentionally
underdetermined observation gate is documented in
[`DISTRIBUTED_OBSERVATION_IDENTIFIABILITY.md`](DISTRIBUTED_OBSERVATION_IDENTIFIABILITY.md),
and the completed one-property uncertainty follow-on is in
[`DISTRIBUTED_PROFILE_COVERAGE.md`](DISTRIBUTED_PROFILE_COVERAGE.md).
