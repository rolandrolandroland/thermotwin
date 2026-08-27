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

For the frozen inverse demonstration, the resistivity multipliers at 285, 300,
and 315 K are changed to

```text
truth = (1.04, 1.07, 1.03).
```

Four constant-current/temperature-lift experiments are fitted jointly. The
noise-free same-model conventional solution is

```text
(1.040000001, 1.070000000, 1.029999999).
```

That near-exact result checks the optimizer and observation mapping. It is an
inverse-crime result, not a realistic uncertainty claim.

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
`rho_e(T)` coefficients.

For the same noise-free resistivity truth above, 800 CPU epochs produce

```text
inverse PINN = (1.051672, 1.066605, 1.048003).
```

The absolute multiplier errors are approximately 0.0117, 0.0034, and 0.0180.
The central coefficient is best constrained. The endpoint estimates still
reflect both data and the declared smoothness regularization. The conventional
solver remains markedly more accurate for this small ideal inverse problem.
The PINN's value is its joint differentiable representation of hidden fields,
PDE constraints, dynamic boundaries, and a function-valued unknown—not superior
accuracy on an inverse crime.

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

The figure is written to
`thermotwin/figures/distributed_property_study.png` by default. Generated
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
  tests.test_distributed_property_report
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
- The reported inverse data are noise-free and generated by the same model.
- Smoothness regularization contributes to inverse-PINN endpoint estimates.
- Multiple switched-current segments are supported by the finite-volume
  reference, but the first distributed PINNs use constant-current experiments;
  switched PINNs require domain decomposition at each discontinuity.
- Nonlinear multi-start recovery, noisy coverage, missing observations, and
  model-mismatch studies remain before any robust inference claim.
- Hardware validation remains unstarted.

The next scientific step is not to release all three property curves at once.
It is to repeat the one-function recovery with realistic noise, withheld
temperature ranges, multiple neural seeds, and a genuinely independent
property representation, then test whether the recovered curve predicts a
complete experiment excluded from fitting.
