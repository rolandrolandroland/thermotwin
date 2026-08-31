# ThermoTwin: detailed reference

This is the technical companion to the root [`README.md`](../README.md). It documents the
equations, the numerics, the observation layer, the inference machinery, the
design workflows, and the assumptions each one depends on.

The organization mirrors the package layers, so where a topic lives here tells
you where its code lives — and where new work should go.

## Choose a path through this guide

| If you want to... | Read these sections first | Then run or open... |
| --- | --- | --- |
| Understand the thermoelectric physics | §1 and §2 | `python3 -m unittest tests.test_thermoelectric` |
| Run the conventional digital twin | §3 and §4 | `python3 -m unittest tests.test_transient tests.test_contact_transient` |
| Understand synthetic sensors and inverse problems | §5 and §6 | [`CONTACT_RESISTANCE_EXPERIMENT.md`](CONTACT_RESISTANCE_EXPERIMENT.md) |
| Understand what the PINNs contribute | §7 and §9.1 | [`PINN_SHOWCASE.md`](PINN_SHOWCASE.md) |
| Explore efficiency and controls | §8.1–§8.3 and §9.3 | [`COP_OPERATING_MAP_EXPERIMENT.md`](COP_OPERATING_MAP_EXPERIMENT.md) |
| Explore product-oriented co-design | §2, §8.4, and §9.4 | [`MATERIAL_GEOMETRY_BAYESIAN_CODESIGN.md`](MATERIAL_GEOMETRY_BAYESIAN_CODESIGN.md) |
| Translate material and contact measurements into process targets | §2, §8.5–§8.6, and §9.5 | [`ELECTRICAL_CONTACT_PROCESS_WINDOW.md`](ELECTRICAL_CONTACT_PROCESS_WINDOW.md) |
| Infer temperature-dependent properties and a hidden internal field | §1.9, §3, §6, and §7 | [`DISTRIBUTED_CONSTITUTIVE_INFERENCE.md`](DISTRIBUTED_CONSTITUTIVE_INFERENCE.md) |
| Test transfer to a complete regime withheld from fitting | §6, §7, and §9.6 | [`DISTRIBUTED_WITHHELD_VALIDATION.md`](DISTRIBUTED_WITHHELD_VALIDATION.md) |
| Extend the package | §13–§15 | [`docs/thermotwin/ARCHITECTURE.md`](../docs/thermotwin/ARCHITECTURE.md) |

Each technical chapter follows the same logic: physical question, equations and
signs, assumptions, code ownership, checks, demonstrated result, and remaining
limits. The walkthrough documents freeze experiment-specific settings and
outputs and the checks that support them.

**Navigation**

1. [The physical model](#1-the-physical-model) — `physics/`
2. [Module geometry and interfaces](#2-module-geometry-and-interfaces) — `design/codesign/`
3. [Numerics](#3-numerics) — `numerics/`
4. [Simulation and diagnostics](#4-simulation-and-diagnostics) — `simulation/`
5. [The observation layer](#5-the-observation-layer) — `observations/`
6. [Inference](#6-inference) — `inference/`, `studies/`
7. [Learned models](#7-learned-models) — `pinn/`
8. [Design and decision workflows](#8-design-and-decision-workflows) — `design/`
9. [Flagship evidence](#9-flagship-evidence)
10. [Validation levels](#10-validation-levels-and-what-they-mean)
11. [Assumptions and limits](#11-assumptions-and-limits)
12. [Tests](#12-what-the-tests-cover)
13. [API and extension](#13-api-and-extension)
14. [Reproducing the reports](#14-reproducing-the-reports)
15. [Roadmap and completion status](#15-roadmap-and-completion-status)
16. [Glossary](#16-glossary)

---

## 1. The physical model

### 1.1 The picture

The reduced model treats the heat pump as two lumped thermal nodes connected by
one thermoelectric module:

```text
cold reservoir                                  hot reservoir
 T_c,∞                                              T_h,∞
   │ G_c                                              │ G_h
   ▼                                                  ▼
[cold node T_c] ⇄ [thermoelectric module] ⇄ [hot node T_h]
        ▲                                          ▲
        │ q̇_c,ext                                 │ q̇_h,ext
```

The module is quasi-steady: it transports and generates heat but stores none.
The contact-aware model separates the thermoelectric faces from the heat
exchangers:

```text
cold reservoir → cold exchanger →[contact R_c]→ cold TE face
                                                     │ module
hot reservoir  ← hot exchanger  ←[contact R_h]← hot TE face
```

This adds two temperatures and makes both interface drops observable. The
shared *Q*<sub>c</sub> and *Q*<sub>h</sub> relations use the **face**
temperatures; reservoirs and external loads act on the **exchanger** nodes.

### 1.2 Symbols

| Symbol | Code name | Meaning | Units |
| --- | --- | --- | --- |
| *T*<sub>c</sub>, *T*<sub>h</sub> | `cold_temperature`, `hot_temperature` | Node or face temperatures | K |
| *T*<sub>c,∞</sub>, *T*<sub>h,∞</sub> | `*_reservoir_temperature` | Reservoir temperatures | K |
| *I* | `current` | Signed module current | A |
| α | `seebeck_coefficient` | Effective module Seebeck coefficient | V/K |
| *R* | `electrical_resistance` | Module electrical resistance | Ω |
| *K* | `thermal_conductance` | Internal parasitic thermal **conductance** | W/K |
| *C*<sub>c</sub>, *C*<sub>h</sub> | `*_thermal_capacitance` | Node thermal capacitances | J/K |
| *G*<sub>c</sub>, *G*<sub>h</sub> | `*_reservoir_conductance` | Node-to-reservoir conductances | W/K |
| *R*<sub>contact</sub> | `*_contact_resistance` | Interface thermal resistance | K/W |
| *Q*<sub>c</sub>, *Q*<sub>h</sub> | `cold_heat`, `hot_heat` | Heat **rates** at the faces | W |
| *V* | `voltage` | Terminal voltage | V |
| *VI* | `electrical_power` | Signed electrical power **into** the module | W |

*K* is a thermal conductance in W/K, not a material conductivity in W/(m·K). A
conductance already folds in geometry at the chosen model scale. *Q*<sub>c</sub>
and *Q*<sub>h</sub> carry no dots in the code notation but are rates in watts.

### 1.3 Sign conventions

1. Positive current is the chosen refrigeration polarity when α > 0.
2. *Q*<sub>c</sub> > 0 means heat is removed from the cold node.
3. *Q*<sub>h</sub> > 0 means heat is delivered to the hot node.
4. Positive external heat enters its node.
5. Positive *VI* means electrical power **enters** the module.
6. *T*<sub>h</sub> − *T*<sub>c</sub> > 0 means the hot face is warmer.

The model does not claim that positive current corresponds to one named
direction of charge flow for every manufactured module. Polarity is defined by
the effective coefficient and the desired cooling action.

### 1.4 Heat rates and voltage

Three mechanisms act at the faces. Peltier transport is linear in current and
uses **absolute** temperature:

$$\dot Q_{\mathrm{Peltier}} = \alpha I T.$$

Joule dissipation is quadratic in current and does not reverse with it; the
lumped model splits it equally between the faces:

$$\dot Q_{\mathrm{Joule}} = I^2 R, \qquad \tfrac{1}{2}I^2R \ \text{per face}.$$

Parasitic conduction returns heat from hot to cold, opposing refrigeration:

$$\dot Q_{\mathrm{leak}} = K(T_h - T_c).$$

Combining them:

$$Q_c = \alpha I T_c - \tfrac{1}{2}I^2R - K(T_h - T_c),$$

$$Q_h = \alpha I T_h + \tfrac{1}{2}I^2R - K(T_h - T_c),$$

$$V = \alpha(T_h - T_c) + IR.$$

The first voltage term is the Seebeck back-EMF and the second is the ohmic
drop. *V* is an **output** of the current-driven model, not an applied input:
reversing *I* reverses the ohmic term and can change the sign of *V*.

### 1.5 The energy identity

$$Q_h - Q_c = \alpha I(T_h - T_c) + I^2R = I\left[\alpha(T_h-T_c) + IR\right] = VI.$$

So *Q*<sub>h</sub> = *Q*<sub>c</sub> + *VI*: the module rejects what it removed
from the cold side plus the electrical power supplied to it. This identity is
checked directly at representative positive, zero, and negative currents. It is
an algebraic invariant of the implemented heat-rate and voltage definitions.

### 1.6 Limiting cases

**Zero current.** *Q*<sub>c</sub> = *Q*<sub>h</sub> = −*K*(*T*<sub>h</sub> −
*T*<sub>c</sub>). With *T*<sub>h</sub> > *T*<sub>c</sub> both are negative: heat
enters the cold node and leaves the hot node — ordinary hot-to-cold conduction.
The open-circuit voltage α(*T*<sub>h</sub> − *T*<sub>c</sub>) is nonzero, but
*VI* = 0.

**Excessive current.** At fixed temperatures the useful Peltier term grows
linearly while Joule grows quadratically, so *Q*<sub>c</sub> peaks at

$$I_{Q_c,\max} = \frac{\alpha T_c}{R},$$

beyond which more current removes less heat. Cooling COP begins declining before
maximum cooling is reached.

### 1.7 Coefficient of performance

$$\mathrm{COP}_{\mathrm{cool}} = \frac{Q_c}{VI}, \qquad
\mathrm{COP}_{\mathrm{heat}} = \frac{Q_h}{VI} = \mathrm{COP}_{\mathrm{cool}} + 1,$$

the second identity following directly from §1.5. Both are meaningful only when
the relevant heat rate and the electrical input are positive. At zero electrical
power the ratio is undefined: the direct functions raise `ZeroDivisionError`,
and trajectory diagnostics report `None`.

### 1.8 Transient energy balances

Two-node:

$$C_c\frac{dT_c}{dt} = G_c(T_{c,\infty}-T_c) + \dot q_{c,\mathrm{ext}} - Q_c,$$

$$C_h\frac{dT_h}{dt} = G_h(T_{h,\infty}-T_h) + \dot q_{h,\mathrm{ext}} + Q_h.$$

Adding them and using the energy identity removes the internal transport:

$$C_c\dot T_c + C_h\dot T_h = G_c(T_{c,\infty}-T_c) + G_h(T_{h,\infty}-T_h)
+ \dot q_{c,\mathrm{ext}} + \dot q_{h,\mathrm{ext}} + VI.$$

Stored energy changes only through reservoir heat, external heat, and electrical
power.

Four-node, with contact heat positive from cold exchanger toward cold face and
from hot face toward hot exchanger:

$$\dot q_{\mathrm{contact},c} = \frac{T_{cx}-T_{cf}}{R_{\mathrm{contact},c}},
\qquad
\dot q_{\mathrm{contact},h} = \frac{T_{hf}-T_{hx}}{R_{\mathrm{contact},h}},$$

$$C_{cf}\dot T_{cf} = \dot q_{\mathrm{contact},c} - Q_c, \qquad
C_{hf}\dot T_{hf} = Q_h - \dot q_{\mathrm{contact},h},$$

$$C_{cx}\dot T_{cx} = G_c(T_{c,\infty}-T_{cx}) + \dot q_{c,\mathrm{ext}}
- \dot q_{\mathrm{contact},c},$$

$$C_{hx}\dot T_{hx} = G_h(T_{h,\infty}-T_{hx}) + \dot q_{h,\mathrm{ext}}
+ \dot q_{\mathrm{contact},h}.$$

Zero contact resistance is **not** how you recover the two-node model — it is a
zero denominator. Use the two-node solver instead. As contact resistance shrinks
toward zero the tested four-node/two-node discrepancy decreases. That convergence
is a validation check, not a usage pattern or a claim of a fitted convergence
order.

### 1.9 Distributed thermoelectric leg

The lumped models remain the fast system-level default. A separate 1-D model
resolves temperature along one homogeneous leg from cold face to hot face. Its
local constitutive laws are

```text
E = rho_e(T) J + alpha(T) dT/dx
q = alpha(T) T J - kappa(T) dT/dx
rho_m c_p dT/dt = -dq/dx + J E.
```

Expanding the conservative energy equation produces conduction, Joule heating,
and Thomson heating with the Kelvin relation `tau(T) = T d(alpha)/dT`. The face
temperatures remain dynamic states coupled to reservoir-linked face masses, so
they can be observed rather than prescribed. During a transient the leg stores
energy and the correct balance is `dU_leg/dt = Q_c - Q_h + VI`; the lumped
identity `Q_h - Q_c = VI` reappears at steady state.

The complete derivation, frozen values, inverse setup, results, and limits are
in [`DISTRIBUTED_CONSTITUTIVE_INFERENCE.md`](DISTRIBUTED_CONSTITUTIVE_INFERENCE.md).

---

## 2. Module geometry and interfaces

The co-design layer maps material properties and geometry onto module α, *R*,
and *K*. For *N* p/n couples with leg length *L* and cross-section *A*, legs are
electrically in series and thermally in parallel within a couple, and couples
repeat in series:

$$\alpha = N(S_p - S_n), \qquad
R_{\mathrm{legs}} = N\frac{L}{A}(\rho_p + \rho_n), \qquad
K_{\mathrm{legs}} = N\frac{A}{L}(\kappa_p + \kappa_n).$$

Since *S*<sub>n</sub> < 0, the couple Seebeck is the sum of magnitudes. Note
that *R*<sub>legs</sub>·*K*<sub>legs</sub> = *N*²(ρ<sub>p</sub>+ρ<sub>n</sub>)(κ<sub>p</sub>+κ<sub>n</sub>)
is **independent of geometry** — material *ZT* cannot answer a geometry
question. Any geometric optimum comes from the parasitics.

Two parasitics are added, each with its own scaling:

$$R = R_{\mathrm{legs}} + \underbrace{4N\frac{\rho_c}{A}}_{R_{\mathrm{contact}}},
\qquad K = K_{\mathrm{legs}} + 0.04\ \mathrm{W/K}.$$

Each leg has two metal/thermoelectric interfaces, so a series p/n couple has
four, giving 4*N*ρ<sub>c</sub>/*A*. This term scales with *N*/*A* and is
**independent of leg length**, so it penalizes short legs as a growing fraction
of total resistance. The additive 0.04 W/K package parasitic conduction is
likewise geometry-independent and penalizes designs with low leg conductance.
Together they set where the geometric optimum sits.

The baseline uses ρ<sub>c</sub> = 2.0 × 10⁻¹⁰ Ω·m², anchored to a 298 K
Ti/Bi₂Te₃ transfer-length measurement of 1.94 × 10⁻¹⁰ Ω·m²
([AIP Advances 15, 035351](https://doi.org/10.1063/5.0253218)). It is one
literature-anchored baseline, not a claim about typical production quality. The
implemented [electrical-contact process window](ELECTRICAL_CONTACT_PROCESS_WINDOW.md)
now sweeps zero plus $10^{-11}$ through $5\times10^{-8}$ Ω·m² across 0.05–2.5 mm
legs. This deliberately includes interfaces from better than the baseline
through contact-dominated operation.

Material records come from a curated same-row extract of the fixed StarryData
snapshot (Figshare DOI `10.6084/m9.figshare.11340935.v1`, CC BY 4.0): twelve
300 K Bi/Te-family samples, six of each carrier type, spanning *ZT* from 0.15 to
1.39. Every triplet comes from one sample row — no cross-row property mixing.

---

## 3. Numerics

### 3.1 Integration

Both transient models advance with fixed-step classical RK4:

$$k_1=f(y_n),\quad k_2=f\!\left(y_n+\tfrac{\Delta t}{2}k_1\right),\quad
k_3=f\!\left(y_n+\tfrac{\Delta t}{2}k_2\right),\quad k_4=f(y_n+\Delta t\,k_3),$$

$$y_{n+1}=y_n+\frac{\Delta t}{6}(k_1+2k_2+2k_3+k_4).$$

All node temperatures advance together because each face heat rate depends on
both face temperatures. Trajectories always include the initial state and end
exactly at the requested duration, shortening the final step if needed.

**Current switches.** An RK4 interval never crosses a known piecewise-constant
transition: the integrator shortens the preceding step to land exactly on the
switch, then starts a new step with the new current. Within one interval all
four stages use one held current. Temperatures stay continuous across a switch
because finite capacitance cannot absorb finite energy instantaneously; current,
heat rates, voltage, power, COP, and temperature derivatives may all jump.

**Divergence.** Explicit RK4 is conditionally stable, and small contact
resistances make the system stiff (the contact time constant is
*R*<sub>contact</sub>·*C*). When a state leaves the positive-kelvin domain the
integrator raises `IntegrationDivergenceError` naming the time, the RK4 stage,
and the remedy, rather than returning non-finite temperatures.

### 3.2 Steady-state solvers

With constant inputs, setting all storage rates to zero gives a linear system —
2×2 for the two-node model, 4×4 for the contact model. Capacitances do not
appear. These solvers are independent of the integrators, so comparing a long
trajectory against them is a genuine cross-check rather than a restatement. The
tests verify agreement to their declared decimal tolerances after sufficiently
long integrations; they do not claim machine-precision transient agreement.

A unique algebraic solution does not guarantee the dynamics reach it. The
constant-property model has a positive-feedback branch when α*I* exceeds
*G*<sub>h</sub> + *K*, because Peltier heat delivered to the hot node grows with
its own temperature. Solutions outside the positive-kelvin domain raise
`ValueError` instead of being returned.

### 3.3 Linear algebra and statistics

`numerics/matrices.py` provides dependency-free Gauss–Jordan inversion with
partial pivoting (returning inverse and determinant), transpose, multiply, add,
and the Gram matrix *J*ᵀ*J*. Its tests cover known reference matrices, dimension
checks, singular cases, and the identities used by the inference workflows.
`numerics/statistics.py` provides interpolated quantiles and the integration
helpers described next.

### 3.4 Integrating discontinuous power

Electrical power *VI* jumps at every current switch. Trapezoidal integration
across such a jump is biased, and the bias depends on how switch times align
with the output grid. `piecewise_electrical_energy` therefore inserts transition
times as breakpoints and assigns each subinterval one midpoint-selected current,
so both endpoint powers use that interval's current and the left and right
limits are preserved. On an analytically integrable case it is exact to ~1e-14 J
in the tested case, including an output grid that does not contain every switch.

### 3.5 Conservative finite volumes

The distributed reference uses cell-centred finite volumes. Heat fluxes and
electrical voltage drops are assembled at cell faces, so summing all cell and
face-node equations cancels internal exchanges and closes the whole-system
energy balance to roundoff. A second-order one-sided boundary gradient exactly
reproduces constant-property quadratic steady profiles. The explicit step
recommendation scales with cell width squared, and the test suite checks both
that scaling and RK4 refinement.

---

## 4. Simulation and diagnostics

`simulation/` holds frozen experiment definitions and the derived histories that
make a trajectory interpretable.

The two-node reference experiment: α = 0.05 V/K, *R* = 2.0 Ω, *K* = 0.5 W/K,
*C*<sub>c</sub> = 100 J/K, *C*<sub>h</sub> = 200 J/K, *G*<sub>c</sub> = 2.0 W/K,
*G*<sub>h</sub> = 4.0 W/K, all temperatures 300 K, 1 A for 60 s at a 0.1 s step.
These are frozen pedagogical parameters, not a calibrated commercial module.
For this parameter set the fixed-temperature maximum-cooling current
α*T*<sub>c</sub>/*R* is 7.5 A.

At the initial equal-temperature state conduction vanishes, so
*Q*<sub>c</sub> = 14 W and *Q*<sub>h</sub> = 16 W, giving initial rates of
−0.14 K/s and +0.08 K/s. The identity gives 16 − 14 = *VI* = 2 W. After 60 s:

| Quantity | Value |
| --- | ---: |
| *T*<sub>c</sub> | 295.971976 K |
| *T*<sub>h</sub> | 302.404041 K |
| *Q*<sub>c</sub> | 10.582566 W |
| *V* | 2.321603 V |
| Cooling COP | 4.558301 |

The independent steady-state solver gives 295.107006 K and 303.045731 K, so the
60 s run is approaching but has not reached equilibrium. These are model
predictions, not measurements.

Diagnostics post-process every sample into aligned histories of current,
temperature difference, heat rates, voltage, power, COP, contact drops, and a
whole-system energy residual. That residual is computed from the assembled
right-hand side, so it verifies the balance assembly and the energy identity —
**not** the accuracy of the time integration. Time-step convergence is a
separate check.

The distributed simulator adds the complete internal cell profile, terminal
voltage, both boundary heat rates, stored internal energy, and an instantaneous
semidiscrete energy audit. It supports piecewise-constant current and ends RK4
steps exactly at switches. The first distributed PINNs intentionally use
constant-current regimes; extending them across switches requires the same
domain-decomposition logic used by the lumped piecewise PINNs.

---

## 5. The observation layer

`observations/` separates dense synthetic truth from what an inverse model is
allowed to see. Datasets are long-form records of time, sensor name, modeled
location, temperature, and aligned current. Dense trajectories are never
exposed.

The measurement pipeline applies, in order:

```text
truth → lag → sampling → bias → noise → remove unavailable readings
```

**Sampling.** Regular times from zero, always including the exact final time,
with linear interpolation between stored states and the same right-continuous
current convention as the integrator.

**Lag.** A first-order sensor model per channel. Between adjacent observations
the input is interpolated linearly and the sensor ODE is integrated **exactly**
for that piecewise-linear target — which is why a linear ramp is reproduced to
machine precision rather than acquiring the half-sample lead a zero-order hold
would introduce. Lag is applied to dense truth before output sampling, so
changing the output interval does not change the simulated sensor response. The
filter does not feed back into the thermal state.

**Bias.** Constant per-sensor offsets. Unlike zero-mean noise, bias does not
diminish under averaging — a fact the inference studies exploit.

**Noise.** Independent zero-mean Gaussian errors with a documented seed. The
generator draws for every record regardless of that sensor's standard deviation,
so the stream is independent of the configured magnitudes and a zero standard
deviation reproduces the ideal dataset exactly.

**Missingness.** Deterministic per-sensor outage intervals that omit records
rather than inserting zeros, `NaN`, or interpolated replacements. Applied after
noise, representing failed *reporting* while the sensor and the thermal model
keep evolving.

**Provenance.** Generated datasets carry the complete experiment definition,
ground-truth parameters, current schedule, split assignment, and the ordered
list of measurement transformations with their settings. `observations/quality.py`
audits record counts, completeness, ranges, provenance, and whole-regime split
integrity. `observations/hardware.py` defines the CSV contract
(`time_s,current_A,cold_exchanger_K,hot_exchanger_K,voltage_V`) for future
benchtop data; no hardware data exist in this repository.

For the distributed model, sparse observations may include both face
temperatures, voltage, and either boundary heat rate. Only continuous
temperatures are interpolated. Voltage and heat flux are recomputed using the
right-continuous current at the requested time, so a current switch is not
silently replaced by a linear ramp.

---

## 6. Inference

### 6.1 Conventional contact-resistance fitting

One positive scalar is unknown, so the baseline estimator needs no gradients: a
bounded golden-section search over 0.05–1.0 K/W minimizes temperature MSE at the
available cold-pair records, with whole current regimes assigned to separate
train/validation/test splits rather than random time rows. It recovers a hidden
0.25 K/W as 0.250000002 K/W in 42 evaluations. The near-floating-point accuracy
is expected — the same noise-free model generates and fits the data — and
validates the workflow, not hardware accuracy.

### 6.2 Identifiability and information

Parameter error alone is a poor measure in same-model studies, because exact
data recover the truth regardless of how informative the experiment was. The
studies therefore report a local information proxy: the curvature of the
training sum of squares with respect to the parameter, evaluated at the known
truth. Representative values for cold contact resistance:

| Available sensors | Information curvature |
| --- | ---: |
| Cold face + cold exchanger | 304.9 |
| Cold face only | 208.9 |
| Cold exchanger only | 96.0 |
| Hot pair only | 1.9 |
| All four | 306.8 |

The hot pair adds under 1% to the cold pair. Similarly, removing five
steady-state readings per sensor costs nothing, while removing the same count
around each current switch costs 29% — the information lives in the transients.

### 6.3 Sparse accessible sensors

The realistic problem exposes only the two exchanger temperatures, with both
face temperatures withheld. Contact resistance and a shared sensor lag are fit
jointly by coarse-to-fine grid search followed by a local pattern search, with
each sensor's constant bias profiled out analytically as the mean residual. The
local polish matters: without it the grid can lock onto a node that coincides
with the truth and report a spuriously exact recovery.

### 6.4 Next-experiment selection

With a Gaussian prior and Gaussian observation noise, the posterior information
for a candidate pulse is

$$\mathcal I_{\mathrm{post}} = \frac{J^\top J}{\sigma^2} + \mathcal I_{\mathrm{prior}},$$

where *J* is the sensitivity of predicted observations to the log-parameters,
computed by central differences in log space so that a resistance, a
capacitance, and a time constant are comparable. Per-sensor biases enter as
nuisance columns and are marginalized by taking the corresponding sub-block of
the **covariance** — not by inverting a sub-block of the information matrix. The
design objective is the Gaussian entropy reduction

$$\text{gain} = \tfrac{1}{2}\ln\frac{\det\Sigma_{\mathrm{prior}}}{\det\Sigma_{\mathrm{post}}}\ \text{nats},$$

maximized over feasible candidates subject to energy and temperature limits.

### 6.5 Robustness studies

`studies/` repeats the fit across many seeded trials under noise, bias, lag,
missingness, restricted sensors, and their combination. Two results are worth
carrying:

- Under 0.05 K noise alone, the estimator is nearly unbiased: mean 0.2498 K/W
  against a 0.25 K/W truth, sample standard deviation 0.0041.
- Under the combined pipeline, the mean shifts to 0.2016 K/W — a systematic bias
  of −0.048 that is roughly eight times the random spread. Repetition
  characterizes random variation; it does not correct a wrong measurement model.

### 6.6 Function-valued property inference

The distributed workflow perturbs piecewise-linear coefficients of `alpha(T)`,
`rho_e(T)`, or `kappa(T)` in continuous log-magnitude coordinates. A
noise-normalized finite-difference Jacobian is inspected before fitting; its
singular spectrum reports which coefficient combinations are locally visible.
The conventional baseline uses a bounded coordinate search followed by a
damped Gauss–Newton polish. This avoids truth-grid locking and, on the frozen
noise-free same-model one-function cases, recovers all declared Seebeck,
resistivity, and conductivity multipliers to the report's printed precision.
That exactness is an inverse-crime software check, not a hardware forecast.

The same information matrix supplies local log-coefficient intervals and ranks
candidate current/lift experiments by posterior entropy reduction. The frozen
three-coefficient resistivity selection chooses a 20 K, -0.8 A, 0.5 s pulse
from twelve declared candidates, with 6.0338 nats of local information gain.

`studies/distributed_inverse_robustness.py` then repeats the resistivity fit
under independent temperature/voltage noise and neural seeds. It stores every
trial, applies fixed coefficient-and-loss failure criteria, and reports search-
bound contact rather than retaining only favorable runs. The conventional and
PINN estimators see identical noisy observations, but their regularization is
not matched; the public walkthrough treats that as a limitation.

The next independent-truth study changes the spatial grid, time integrator,
voltage quadrature, and resistivity representation. It also applies the exact
same explicit log-curvature term to matched conventional and PINN variants.
Across three paired trials, the two PINN variants pass 3/3 and the two
conventional variants pass 1/3. The matched penalty does not explain the PINN's
stability: it barely changes either PINN curve, while the conventional prior
still permits an inaccurate nearly linear log-slope. This narrows—but does not
eliminate—the regularization caveat because implicit neural and hidden-field
bias remain unmatched.

---

## 7. Learned models

`pinn/` is optional and imports PyTorch. Every network maps time to
temperatures and is trained on ODE residuals; conventional trajectories are
withheld until validation.

**Exact initial conditions.** Outputs are transformed as
*T*(*t*) = *T*(0) + (*t*/*t*<sub>end</sub>)·*T*<sub>scale</sub>·*N*(*t*), so the
initial state holds for any weights and no soft penalty is needed. The transform
does not force the initial slope.

**Residuals.** Automatic differentiation supplies d*T*/d*t*; each residual is the
network rate minus the physical right-hand side, in K/s. A physically consistent
solution makes them zero throughout.

**Switched current.** A single smooth network cannot represent discontinuous
derivatives, so the piecewise models assign one subnetwork per constant-current
interval and chain each segment's start to the previous segment's endpoint.
Temperature continuity is therefore a construction, exactly zero up to floating
point, while one-sided rates remain free — which is what the physics permits.
Collocation points are duration-weighted interval midpoints and exclude the
switches, where no single classical derivative exists.

**Inverse parameters** are unconstrained scalars mapped through softplus to stay
positive, trained jointly with the network at a separate learning rate, with
residual and observation losses normalized to comparable magnitudes.

**What this establishes.** On these small, well-posed, same-model problems the
conventional estimators are *more* accurate — the golden-section fit reaches
0.250000 K/W where the piecewise inverse PINN reaches 0.250519 K/W. The PINN's
demonstrated value is the unified differentiable representation: physics alone
produces four continuous hidden states with zero temperature labels; partial
observations plus the same physics identify a positive physical parameter;
unobserved states are reconstructed without being used as labels; and the
inferred parameter transfers to unseen control schedules. Those are the
properties that matter when several states or parameters are unknown — not a
speed or accuracy win on this lumped model.

The distributed forward PINN maps `(x,t)` to `T` and enforces the full initial
profile exactly. Its loss contains the interior thermoelectric PDE plus two
dynamic face-node balances. In the frozen 800-epoch CPU case, withheld
finite-volume validation gives 0.006420 K internal-field RMSE and 0.015116 K
maximum error.

For inverse work, one temperature network is assigned to each constant-current
regime while all networks share one property curve. This is necessary because
one narrow-temperature experiment does not identify the endpoints of a
function basis. Separate frozen fits now release `alpha(T)`, `rho_e(T)`, or
`kappa(T)` while holding the other curves fixed. Temperature and voltage are
enough for the first two ideal demonstrations, but the terminal-only
conductivity fit is wrong despite its falling loss. Adding idealized face
heat-rate observations cuts the maximum conductivity multiplier error from
0.2799 to 0.0515. Smoothness regularization contributes to endpoint estimates,
and the conventional solver is more accurate on every ideal case. The result
demonstrates both a function-valued, field-constrained inverse representation
and why optimization loss is not a recovery metric—not PINN superiority.

---

## 8. Design and decision workflows

### 8.1 COP operating maps

Steady cooling and heating COP across current, external temperature lift, and
contact resistance, for both topologies. Comparisons are made at **equal
delivered load**, matching current by bracketing the first rising crossing and
then bisecting — which does not assume cooling stays monotonic to the current
ceiling. Explicit contacts cost 35.1% of cooling COP at zero lift and about 19%
at 20 K lift relative to the reduced topology.

### 8.2 Power electronics

A thermally averaged layer, not a converter circuit simulation. Because the
thermal time constants greatly exceed the switching period, the model averages
the current moments the equations actually use: **Peltier heat follows the mean
current, Joule heat follows the mean square**. So

$$P_{\mathrm{module}} = \alpha\langle I\rangle\Delta T + R\langle I^2\rangle.$$

For direct rectangular PWM at duty *D* with fixed mean current, ⟨*I*²⟩/⟨*I*⟩² =
1/*D* exactly — a 10× Joule penalty at 10% duty. For smoothed triangular ripple
of peak-to-peak amplitude Δ, ⟨*I*²⟩ = ⟨*I*⟩² + Δ²/12. Wall-plug COP adds a
converter efficiency and a fixed switching loss; the ideal-DC case carries no
converter loss and is a reference bound, not a realizable drive.

### 8.3 Control comparison

Continuous versus pulsed operation at equal delivered cooling after a periodic
warm-up, with storage-drift acceptance checks so an unfinished transient is not
credited to one schedule. The robust result is the **duty trend**, not a penalty
at an arbitrary duty ceiling. At the 5 W target the COP penalty falls from
24.23% at 75% duty to 0.90% at 99% duty, consistent with the 1/*D* Joule law
approaching the continuous limit.

### 8.4 Material and geometry co-design

Public material records feed the geometry mapping of §2, then a 24-design
Latin-hypercube screen, then cost-aware Gaussian-process Bayesian optimization
against equal-budget random search, then fixed-current robustness trials under
property, interface, and converter spread. Three application specifications with
different objectives are carried through.

The optimization result is reported honestly: it reaches the candidate-pool
optimum on the 25 K application (utility 6.43 versus a 3.90 random median) and
finds **nothing** on the two 10 K applications, where the initial screen already
contained the pool winner. The nominal 10 K efficiency winner then passes only
55.3% of as-built trials because its cooling rate barely clears the requirement
— a nominal optimizer can select a design that looks efficient and is difficult
to build reliably. The objective is not retroactively changed to hide this.

### 8.5 Electrical-contact process window

The process-window study removes the synthetic cost objective and asks for the
lowest rising-branch current that meets each application's cooling requirement.
It sweeps 61 logarithmic leg lengths from 0.05 to 2.5 mm, a zero-contact
reference plus 61 logarithmic contact-resistivity values through
$5\times10^{-8}$ Ω·m², with the existing 1 A/mm² campaign constraint and an
exploratory 3 A/mm² sensitivity reported side by side. Failed cooling targets
are separated into current-cap-limited and physics-limited cases by repeating
the reachability check without the selected current-density cap.

For $N$ couples with four interfaces per p/n couple,

$$
R_{\mathrm{contact}}=4N\frac{\rho_c}{A},
\qquad
\rho_{c,50}=\frac{L(\rho_p+\rho_n)}{4}.
$$

The crossover is independent of $N$ and $A$, while its position moves linearly
with leg length. Short legs are penalized twice: contact share grows relative
to bulk electrical resistance, and leg thermal conductance grows as $1/L$.

An electrical-only translation of a published 1.5 mm Ag₂Se unicouple gives a
50% crossover at $1.1069\times10^{-8}$ Ω·m². Its reported 7.4 mΩ per contact
and 2.25 mm² area imply $R_cA\approx1.6650\times10^{-8}$ Ω·m², corresponding
to 60.07% modeled contact share and 39.93% zero-contact device-$ZT$ retention.
That source device is a generator; the separate system panels use declared
$N=120$, 1.6 mm² cooling assumptions.

### 8.6 Matched Ag₂Se substitution

The optimized room-temperature Ag₂Se triplet—−153.3 µV/K, 117,400 S/m, and
0.85 W/(m·K)—is an opt-in DOI-backed record outside the indexed StarryData
catalog. The study takes all 204 frozen designs, holds the p material, geometry,
interfaces, exchangers, electronics, and application fixed, replaces only the
n material, and repeats the same current-grid optimization.

At the $2.0\times10^{-10}$ Ω·m² baseline, Ag₂Se improves scalar utility in
69.6–76.5% of matched designs. At the paper-derived contact landmark that range
falls to 50.0–54.9%. It produces no new best feasible design in any application
at either contact level. This null result is retained: a promising material can
improve much of the design space without beating the best existing complete
system.

---

## 9. Flagship evidence

All results in this section are generated by the implemented model. They are
useful for testing algorithms, exposing tradeoffs, and deciding what hardware
experiment to run. They are not measurements.

### 9.1 Forward and inverse PINNs

The physics-only switched-current contact PINN receives time and the known
current schedule. It receives no temperature labels. Against the withheld RK4
solution for the same equations:

| State | RMSE |
| --- | ---: |
| Cold thermoelectric face | 0.008862 K |
| Hot thermoelectric face | 0.001989 K |
| Cold exchanger | 0.009327 K |
| Hot exchanger | 0.004628 K |

Constructed temperature jumps at current switches are 0 K. That continuity is a
hard architectural property; it should not be mistaken for a learned result.

For inverse recovery, the contact-resistance truth is 0.25 K/W and the PINN
starts at 0.50 K/W:

| Quantity | Result |
| --- | ---: |
| Inverse PINN estimate | 0.250519 K/W |
| Parameter error | 0.208% |
| Conventional scalar estimate | 0.250000 K/W |
| Validation-schedule all-state RMSE after parameter transfer | 0.000322 K |
| Bipolar-test all-state RMSE after parameter transfer | 0.000534 K |

The conventional optimizer should win this one-dimensional, noise-free task.
The PINN demonstration is instead about combining continuous hidden states,
physical residuals, partial observations, positivity, and switched dynamics in
one differentiable model. Full conditions are in
[`PINN_SHOWCASE.md`](PINN_SHOWCASE.md).

### 9.2 Inference, imperfect measurements, and experiment choice

The ideal conventional contact fit recovers a 0.25 K/W hidden resistance as
0.250000002 K/W in 42 objective evaluations. Because the generator and fitter
share a model and the data are noise-free, this near-exact result validates the
implementation rather than the physical adequacy of the model.

The local information about the same parameter depends strongly on location:

| Visible temperature channels | Information curvature |
| --- | ---: |
| Cold face and cold exchanger | 304.9 |
| Cold face only | 208.9 |
| Cold exchanger only | 96.0 |
| Hot face and hot exchanger | 1.94 |
| All four temperatures | 306.8 |

The cold pair carries roughly 157 times the information of the hot pair. Losing
five readings around current switches costs 29% of the local information;
losing the same number at steady state costs effectively nothing in the frozen
study.

Measurement-model mismatch is visible rather than averaged away:

| Study | Contact-resistance estimate or distribution |
| --- | ---: |
| 0.05 K Gaussian noise only | mean 0.249783 K/W; standard deviation 0.004117 K/W |
| Unmodeled +0.10 K cold-face bias | 0.208885 K/W, 16.4% low |
| Asymmetric lag with the corrected lag-aware fit | 0.252630 K/W |
| Combined imperfect-data pipeline with incomplete correction | mean 0.201590 K/W; RMSE 0.048744 K/W |

The next-experiment planner ranks 25 feasible pulses while accounting for
contact resistance, face capacitance, shared sensor lag, and two nuisance
biases. It selects 0.8 A for 20 s beginning at 5 s, using 27.54 J. The predicted
joint information gain is 7.198 nats versus 2.889 nats for the naive 0.4 A,
5 s pulse. Across 250 linearized noise trials, the selected pulse reduces joint
log-parameter RMSE by 82.2%; approximate 95% coverage is 94.5% for the selected
pulse and 94.4% for the naive pulse.

This is encouraging but does not complete the nonlinear-validation claim: the
repeated comparison still uses linearized updates. See
[`NEXT_EXPERIMENT_WALKTHROUGH.md`](NEXT_EXPERIMENT_WALKTHROUGH.md) and §15.

### 9.3 COP, contacts, pulses, and PWM

For the baseline contact-aware assembly, the maximum useful cooling COP over the
tested current grid falls as the imposed reservoir lift rises:

| Reservoir lift | Best cooling COP | Current |
| ---: | ---: | ---: |
| 0 K | 23.423 | 0.15 A |
| 5 K | 4.122 | 0.35 A |
| 10 K | 1.835 | 0.70 A |
| 15 K | 1.076 | 1.10 A |
| 20 K | 0.697 | 1.45 A |
| 25 K | 0.437 | 1.50 A |
| 30 K | 0.195 | 1.50 A |

At equal 3 W delivered cooling, explicit baseline contacts lower COP relative to
the reduced two-node model by 35.14% at zero lift, 24.74% at 5 K, 20.73% at
10 K, 19.28% at 15 K, 19.30% at 20 K, and 20.44% at 25 K. The 30 K target is
infeasible under the declared 1.5 A bound.

At equal delivered cooling, continuous current outperforms the tested
seconds-scale rectangular pulses. At the 5 W target:

| Duty | COP penalty relative to continuous |
| ---: | ---: |
| 50% | 51.77% |
| 75% | 24.23% |
| 90% | 9.34% |
| 95% | 4.59% |
| 99% | 0.90% |

This trend—not the value at one arbitrary duty ceiling—is the transferable
result. At fixed mean current, direct rectangular chopping multiplies Joule heat
by `1 / duty`, and the penalty approaches zero continuously as duty approaches
one.

The averaged power-electronics study separates module COP from wall-plug COP.
At 0.60 A mean current and 10 K lift:

| Drive model | Cooling | Wall-plug COP |
| --- | ---: | ---: |
| Ideal DC reference | 1.950 W | 1.757 |
| Smoothed PWM-derived current | 1.950 W | 1.600 |
| Direct rectangular chopping | 1.459 W | 0.620 |

The averaging assumes temperatures change little during one electrical
switching period. More precisely, it neglects current-temperature covariance,
so `mean(I T)` is approximated by `mean(I) mean(T)`. It is not a converter
circuit simulation.

The full interpretations are in
[`COP_OPERATING_MAP_EXPERIMENT.md`](COP_OPERATING_MAP_EXPERIMENT.md),
[`CONTROL_COMPARISON_EXPERIMENT.md`](CONTROL_COMPARISON_EXPERIMENT.md), and
[`PWM_POWER_ELECTRONICS_EXPERIMENT.md`](PWM_POWER_ELECTRONICS_EXPERIMENT.md).

### 9.4 Material and geometry co-design

Each application begins with 24 space-filling designs, then receives 12
cost-aware Bayesian-optimization selections from a fixed pool of 180 candidates.
Twenty-five equal-budget random candidate orders provide the comparison.

| Application | Selected hardware and operating point | Outcome |
| --- | --- | --- |
| 10 K, efficiency-first | 83 couples, 1.079 mm legs, 0.845 mm² area, 0.494 A | 2.524 W cooling, COP 2.856, cost index 0.689 |
| 25 K, balanced | 98 couples, 1.179 mm legs, 2.216 mm² area, 2.111 A | 8.317 W cooling, COP 0.882, cost index 1.142 |
| 10 K, capacity-first | Same hardware as efficiency winner, 0.805 A | 4.662 W cooling, COP 2.207 |

The 25 K utility improves from 3.9015 to 6.4268 and reaches the tested-pool
winner after five Bayesian additions; the random-search median remains 3.9015.
For both 10 K applications, the initial screen already contains the tested-pool
winner, so no optimization improvement is claimed. The 25 K balanced and 10 K
capacity-first operating points are binding at the declared 1.0 A/mm² peak
current-density limit.

In 300 fixed-current robustness trials per design, requirement pass rates are
55.3%, 100.0%, and 100.0%, respectively. These distributions come from declared
synthetic property and interface spreads; they are not measured manufacturing
capability. See
[`MATERIAL_GEOMETRY_BAYESIAN_CODESIGN.md`](MATERIAL_GEOMETRY_BAYESIAN_CODESIGN.md).

### 9.5 Contact process target and matched material substitution

The electrical-contact experiment converts reported unicouple resistance into
a geometry-scaled process target, then passes contact quality through the
four-node cooling model. Near 1.5 mm, the maximum feasible $\rho_c$ across the
six Ag₂Se/p-type pairings is:

| Application | Maximum-feasible $\rho_c$ range |
| --- | ---: |
| 10 K efficiency-first | $9.1028\times10^{-9}$ to $1.6061\times10^{-8}$ Ω·m² |
| 25 K balanced | $3.8840\times10^{-9}$ to $9.1028\times10^{-9}$ Ω·m² |
| 10 K capacity-first | $6.8529\times10^{-9}$ to $1.3936\times10^{-8}$ Ω·m² |

The p-type envelope matters: Ag₂Se alone is not a module. The high-lift case is
the most restrictive under these assumptions. The full surface, failure
regions, and generator-versus-cooler boundary are in
[`ELECTRICAL_CONTACT_PROCESS_WINDOW.md`](ELECTRICAL_CONTACT_PROCESS_WINDOW.md).

The matched substitution audit then shows where the material record changes a
decision without changing the design pool. At the good-interface baseline, the
25 K median changes are +0.3401 W cooling and +0.1075 COP, and 26 designs gain
feasibility with none losing it. Even there, the best Ag₂Se utility is 5.9425
versus 6.4268 for the best original pair. At the paper-derived contact level,
median improvements nearly vanish and the best utilities fall. Full matched
counts and all six application/contact summaries are in
[`AG2SE_SUBSTITUTION_EXPERIMENT.md`](AG2SE_SUBSTITUTION_EXPERIMENT.md).

### 9.6 Distributed constitutive inference

The distributed extension is the first ThermoTwin PINN problem in which the
hidden state is a spatial field and the unknown is a temperature-dependent
function. A conservative finite-volume solver provides the reference, and a
four-regime sensitivity gate is evaluated before training.

Under the declared 0.01 K temperature and 10 µV voltage noise scales, each
three-knot property curve is locally full rank. Condition numbers are 74.2 for
`alpha(T)`, 318 for `rho_e(T)`, and 9.55 for `kappa(T)`; the joint nine-variable
condition number is 392. The forward PDE PINN reaches 0.006420 K hidden-field
RMSE. The inverse study then releases only one curve at a time while keeping the
other two at their baselines. With face temperatures and voltage, the
noise-free inverse PINN's maximum knot-multiplier errors are about 0.0165 for
`alpha(T)` and 0.0180 for `rho_e(T)`. The same terminal-only fit is not reliable
for `kappa(T)`. Adding idealized cold- and hot-side heat-rate observations
reduces the conductivity error to about 0.0515, which is improved but still
weaker than the other families. The conventional same-model estimator returns
all four declared cases to numerical precision.

Those results are deliberately ordered: observation model, local rank,
conventional baseline, then PINN. The full-rank statement is local to the
synthetic experiment and noise assumptions, and the conventional exactness is
same-model/noise-free. Falling PINN loss is not treated as successful property
recovery; the conductivity comparison is retained specifically to expose that
distinction. See
[`DISTRIBUTED_CONSTITUTIVE_INFERENCE.md`](DISTRIBUTED_CONSTITUTIVE_INFERENCE.md).

The follow-on noisy resistivity study varies both observation noise and neural
initialization across five collision-free seed blocks. Its predeclared gate
requires at most 0.10 maximum knot-multiplier error, at least 90% loss
reduction, and final normalized loss no greater than 5.0. The inverse PINN
passes 5/5 trials with 0.015370 coefficient RMSE and 0.025435 worst-trial error;
the unregularized conventional fit passes 2/5 with 0.102054 RMSE and 0.211269
worst-trial error. All trials are retained. Because the PINN has a smoothness
term and implicit neural regularization while the conventional fit does not,
this is a repeatability result for the implemented estimators—not a general
PINN-superiority result. See
[`DISTRIBUTED_INVERSE_ROBUSTNESS.md`](DISTRIBUTED_INVERSE_ROBUSTNESS.md).

The complete-regime transfer study withholds the entire
`positive_0.4A_20K_lift` experiment from every fit. The inferred resistivity
curve is frozen and inserted into the trusted solver, which predicts the
withheld face temperatures, hidden internal field, and terminal voltage. With
the same 0.01 K/10 microvolt observation noise and five fixed seed trials, the
inverse PINN passes all six prediction criteria in 5/5 trials. The conventional
fit passes 2/5; its three failures are voltage-gate failures even though its
temperature errors remain small. Mean inverse-PINN hidden-field RMSE is
0.000070 K and the worst pointwise temperature error is 0.000155 K. This is
within-model operating-regime transfer, not extrapolation to an independent
truth or hardware. See
[`DISTRIBUTED_WITHHELD_VALIDATION.md`](DISTRIBUTED_WITHHELD_VALIDATION.md).

The independent-truth follow-on replaces the truth grid, RK4 integrator, and
piecewise-linear truth curve with a 25-node nodal method, SSPRK3, and a smooth
cubic resistivity law. Three fits use noisy constant-current observations; the
frozen curves predict an excluded 20 K constant regime, a 20 K current pulse,
and a diagnostic 40 K case outside the fitted temperature support. Both PINN
variants pass 3/3 in-support gates with mean maximum property errors of about
0.0335--0.0338. Both conventional variants pass 1/3; applying the same explicit
curvature penalty improves voltage prediction but not continuous property
recovery. This is synthetic continuum-model transfer, not hardware validation,
and three trials are not a failure-rate estimate. See
[`DISTRIBUTED_INDEPENDENT_VALIDATION.md`](DISTRIBUTED_INDEPENDENT_VALIDATION.md).

The next study asks whether the observation set deserves a fitted curve at all.
A frozen local gate combines the noise-normalized singular spectrum with the
allowed +/-0.3 log-coefficient neighborhood and requires every direction to
create at least a one-standard-deviation signal. The bidirectional
temperature-plus-voltage suite supports 3/3 resistivity directions. Zero
current supports exactly 0/3 because both Joule heating and ohmic voltage lose
their resistivity dependence. Positive-current face temperatures support 0/3
at the declared 0.01 K precision; adding voltage supports 2/3, identifying an
average-resistance-like direction much more strongly than the final shape
direction. Multistart fits are retained for diagnosis but rejected whenever the
pre-fit gate fails. In particular, the one-direction inverse PINN returns a
stable, accurate-looking synthetic curve even though only 2/3 directions are
supported; implicit estimator bias is not treated as measured information. See
[`DISTRIBUTED_OBSERVATION_IDENTIFIABILITY.md`](DISTRIBUTED_OBSERVATION_IDENTIFIABILITY.md).

---

## 10. Validation levels and what they mean

These should not be conflated.

| Level | What it establishes | What it does not |
| --- | --- | --- |
| Algebra and units | Equations, signs, identities, dimensions are internally consistent | Anything about the physical world |
| Limiting cases | Known behavior at zero current, equilibrium, insulated nodes, absent identifiability | Behavior in the operating regime |
| Numerical checks | The implemented RK4 method passes known-rate cases, step-refinement checks, switch-splitting checks, long-time steady-state comparisons, and contact-reduction checks | That the equations describe a device, or that RK4 is appropriate at every step size |
| PINN versus conventional | The network approximates its own mathematical model | That the model matches hardware |
| Synthetic inverse recovery | The loss and optimizer can recover a parameter from data generated by the same equations | Performance under model mismatch |
| Observation-layer tests | Sensors sample the intended states at intended times; dense truth is not leaked | Realistic sensor behavior |
| Robustness studies | Sensitivity to controlled synthetic imperfections | Hardware uncertainty |
| Hardware validation | — | **Not implemented.** See [`HARDWARE_VALIDATION_PROTOCOL.md`](HARDWARE_VALIDATION_PROTOCOL.md) |

Two internal checks deserve explicit framing. The whole-system energy residual
is near machine precision **by construction** — it verifies balance assembly,
not integration accuracy. Likewise the piecewise PINN's boundary temperature
jump is exactly zero by construction, not by successful training. Neither is
evidence that a trajectory is correct.

---

## 11. Assumptions and limits

**Physical**

1. In the two- and four-node models, α, *R*, and *K* are constant with
   temperature and current.
2. In those lumped models, face heat-rate relations are quasi-steady and the
   module stores no energy.
3. Temperatures are uniform within each lumped node.
4. The lumped model divides Joule heat equally between the faces.
5. The lumped model neglects Thomson heating, consistently with constant α.
6. The separate distributed extension supports `alpha(T)`, `rho_e(T)`, and
   `kappa(T)`, internal energy storage, and Thomson heating, but represents only
   one homogeneous 1-D leg rather than a p/n unicouple or complete module.
7. Radiation and lateral heat spreading are not modeled.
8. Reservoir temperatures and external heat inputs are constant within a run.
9. Electrical contact resistance enters as a single symmetric areal term; there
   is no separate metallization stack, intermetallic growth, or thermal-cycling
   evolution.
10. Package parasitic conduction is one fixed 0.04 W/K path.

**Numerical**

11. Current is scalar or piecewise constant.
12. Explicit fixed-step RK4 is conditionally stable; small contact resistances
    require proportionally smaller steps.
13. Optimization over current, geometry, and pulse shape is performed on
    declared grids; reported optima are grid optima, and where a constraint is
    active the reports say so.

**Measurement**

14. Sensors are ideal apart from the explicitly modeled noise, bias, first-order
    lag, and deterministic dropout. Current is recorded without error. The
    lumped contact inverse does not use voltage; the distributed property
    inverse does.
15. Sensor models are output filters with no thermal feedback and no physical
    sensor geometry.
16. Outages are deterministic intervals, not random or value-dependent failures.

**Inference**

17. Inverse problems are same-model: the equations generating the data are the
    equations being fitted. This is an inverse crime by construction and is
    stated wherever a recovery result appears.
18. The established lumped studies release one parameter at a time except for
    the sparse-sensor study. The distributed extension releases one complete
    property curve at a time and uses local joint analysis before any proposed
    nine-coefficient fit.
19. Material records are literature values for the Bi/Te family at 300 K, not
    measurements of any device modeled here, and there is no
    processing-to-property model.

---

## 12. What the tests cover

The suite runs 469 tests with `python3 -m unittest discover -s tests`. Optional
learned-model and figure tests skip when PyTorch or Matplotlib are absent.

By category rather than by file, the tests check:

- **Algebraic** — term values and units, sign behavior under current reversal,
  the energy identity at positive, zero, and negative current, COP definitions
  and their undefined case.
- **Limiting-case** — passive conduction, equilibrium preservation, insulated
  energy conservation, and explicit non-identifiability limits where a parameter
  cannot be recovered because the experiment carries no information about it.
- **Numerical** — RK4 reproduction of constant rates, partial final steps, exact
  splitting at current transitions, steady-state agreement with long
  integrations, singular and out-of-domain rejection, and linear-algebra
  agreement with reference results.
- **Observation** — schema validity, interpolation, right-continuous current at
  switches, deterministic noise reproduction, the zero-noise and zero-bias
  identity limits, lag applied before sampling, and dropout as absent rows.
- **Inference** — parameter recovery, information metrics, split integrity,
  bound behavior, and uncertainty-interval coverage.
- **Learned models** — residual signs, exact initial conditions, exact segment
  continuity, short-training recovery, and agreement with withheld trajectories.
- **Reports** — aligned histories and figure generation.

---

## 13. API and extension

New code should import from the layered API:

```python
from thermotwin.physics import ThermoelectricParameters, cold_side_heat
from thermotwin.core.controls import PiecewiseConstantCurrent
from thermotwin.simulation.four_node_experiments import (
    constant_current_contact_reference_experiment,
    run_four_node_contact_experiment,
)
from thermotwin.design.codesign import CodesignCampaignConfig
```

For example, the frozen contact-aware reference case runs as:

```python
experiment = constant_current_contact_reference_experiment()
result = run_four_node_contact_experiment(experiment)

print(result.trajectory.cold_face[-1])
print(result.diagnostics.exchanger_cooling_cop[-1])
```

The `simulation` package intentionally has no eager top-level exports; import
the named experiment module as shown. Pre-restructure public modules remain as
thin compatibility facades where they are still documented. New implementation
code should use the layered modules rather than adding new logic to a facade.
Dependency rules between layers and the guide for adding a new experiment are in
[`docs/thermotwin/ARCHITECTURE.md`](../docs/thermotwin/ARCHITECTURE.md).

### 13.1 Ownership rules

- Put governing equations and state evolution in `physics/`.
- Put reusable numerical utilities with no physical ownership in `numerics/`.
- Put frozen experiment configurations and diagnostic assembly in `simulation/`.
- Transform dense truth into visible records only in `observations/`.
- Fit parameters and rank experiments in `inference/`; put repeated campaigns
  in `studies/`.
- Put application objectives, controls, electronics, and geometry selection in
  `design/`.
- Keep PyTorch in `pinn/` and Matplotlib in `reports/` so the conventional core
  stays lightweight.

### 13.2 Adding a new experiment

1. State the physical question and predeclare the metric.
2. Reuse a physics model or add the smallest new equation with sign and unit
   tests.
3. Create an immutable experiment configuration with every input needed to
   reproduce the run.
4. Keep dense numerical truth separate from the observation records visible to
   an estimator.
5. Add limiting-case and failure-mode tests before interpreting the central
   result.
6. Add a report entry point only after the underlying computation can run
   without plotting.
7. Freeze the conditions, output, caveats, and command in a walkthrough.

---

## 14. Reproducing the reports

### 14.1 Installation

From the repository root:

```bash
python3 -m pip install -e .
```

For figures and PINNs:

```bash
python3 -m pip install -e '.[all]'
```

Run all tests before comparing generated numbers:

```bash
python3 -m unittest discover -s tests
```

### 14.2 Commands

The installed console names and their exact historical module equivalents are:

| Workflow | Installed command | Module command |
| --- | --- | --- |
| Engineering showcase | `thermotwin-engineering-showcase` | `python3 -m thermotwin.engineering_showcase` |
| Material/geometry co-design | `thermotwin-codesign` | `python3 -m thermotwin.material_geometry_codesign_report` |
| COP map | `thermotwin-cop-map` | `python3 -m thermotwin.cop_operating_map_report` |
| PWM electronics | `thermotwin-pwm` | `python3 -m thermotwin.pwm_power_electronics_report` |
| Pulse map | `thermotwin-pulse-map` | `python3 -m thermotwin.pulse_operating_map_report` |
| Contact model | `thermotwin-contact-report` | `python3 -m thermotwin.contact_report` |
| PINN showcase | `thermotwin-pinn-showcase` | `python3 -m thermotwin.pinn_showcase` |
| Dataset audit | `thermotwin-dataset-quality` | `python3 -m thermotwin.dataset_quality` |
| Electrical-contact process window | `thermotwin-contact-process-window` | `python3 -m thermotwin.contact_process_window` |
| Matched Ag₂Se substitution | `thermotwin-ag2se-substitution` | `python3 -m thermotwin.ag2se_substitution` |
| Distributed constitutive inference | `thermotwin-distributed-properties` | `python3 -m thermotwin.distributed_property_report` |
| Distributed inverse robustness | `thermotwin-distributed-robustness` | `python3 -m thermotwin.distributed_inverse_robustness` |
| Distributed withheld-regime transfer | `thermotwin-distributed-withheld` | `python3 -m thermotwin.distributed_withheld_validation` |
| Distributed independent-truth validation | `thermotwin-distributed-independent` | `python3 -m thermotwin.distributed_independent_validation` |
| Distributed observation-sufficiency gate | `thermotwin-distributed-identifiability` | `python3 -m thermotwin.distributed_observation_identifiability` |

These module names are intentionally listed individually: replacing the command
name with a guessed `thermotwin.<name>` module is not reliable.

Figures are written to `thermotwin/figures/`, which is git-ignored because every
figure is reproducible from committed code. Pass `--output PATH` to override.

### 14.3 Reading a generated result

For every report, check the corresponding walkthrough before quoting a number.
It should identify:

- the topology and sign convention;
- fixed parameters, current schedule, and initial state;
- the compared metric and fairness constraint;
- whether the result is transient, steady, or thermally averaged;
- whether data are ideal, noisy, biased, lagged, incomplete, or restricted;
- whether an optimum is continuous or only the best tested grid point;
- whether uncertainty is assumed, locally linearized, or measured;
- and what evidence would still be required for a hardware claim.

---

## 15. Roadmap and completion status

The detailed exit criteria live in [`ROADMAP.md`](ROADMAP.md). This compact table
is a reading aid, not a replacement for that document.

| Milestone | Current status | What remains |
| --- | --- | --- |
| 0 — Scientific specification | Complete for the generic single-block scope | Revisit when physics assumptions change |
| 1 — Conventional reference physics | Complete for the two-node and four-node models | Higher-fidelity physics belongs in a future extension |
| 2 — Reproducible virtual test stand | Complete | Hardware-backed sensor behavior remains outside this milestone |
| 3 — Forward PINNs | Partial | Independent PINN energy-closure history and a matched observation-only baseline |
| 4 — Inverse parameter estimation | Partial | Selected imperfect-data inverse-PINN comparisons and neural-seed failure criteria |
| 5 — Identifiability and uncertainty | Partial | Broader nonlinear multi-start/coverage studies and profile-likelihood intervals |
| 6A — Control comparison | Complete for the current generic scope | Extend only with validated new physics or hardware conditions |
| 6B — Next-experiment selection | Partial | Repeated complete nonlinear refits of selected versus naive experiments |
| 6C — Material/product co-design | Complete for the public-data-seeded virtual method | Temperature-dependent properties, measured process/cost distributions, and hardware calibration |
| 7 — Research artifact | Partial and continuous | Final narrative, evidence audit, and reproducible presentation package |
| 8 — Hardware validation | Optional; not run | Safe hardware, calibrated sensors, uncertainty records, and protocol execution |
| 9 — Distributed constitutive inference | Partial with a validated reference, local and practical gates, forward/inverse PINNs, noisy repeatability, withheld-regime transfer, independent-numerics validation, and explicit observation ablations | Interval-scale nonlinear coverage, richer matched priors, profile likelihood, and broader model discrepancy |

The recommended scientific sequence is:

1. expand the distributed independent-truth comparison to interval-scale
   trials, nonlinear coverage, shrinkage-plus-curvature priors, and profile
   likelihood intervals;
2. test inverse PINNs on selected imperfect datasets with identical visible
   observations;
3. extend local identifiability results to representative nonlinear repeated
   fits;
4. validate the next-experiment recommendation with complete nonlinear refits;
5. add chance-constrained co-design only after measured process-capability data
   exist;
6. attempt hardware validation only when safe hardware and adequate calibration
   are available.

---

## 16. Glossary

| Term | Meaning in ThermoTwin |
| --- | --- |
| Cold-side heat `Q_c` | Heat-transfer rate removed from the cold thermoelectric face; positive supports cooling |
| Hot-side heat `Q_h` | Heat-transfer rate delivered by the module to the hot face |
| External heat | A heat source outside the modeled module that enters one thermal node |
| Parasitic thermal conductance `K` | Unwanted internal hot-to-cold heat-leak conductance of the module |
| Thermal conductivity | Material property in W/(m·K); geometry converts it into a conductance in W/K |
| Thermal resistance | Temperature drop divided by heat rate, in K/W |
| Thermal capacitance | Stored thermal energy per kelvin, in J/K |
| Contact-aware model | Four-node model with separate thermoelectric face and exchanger temperatures |
| Reduced model | Two-node model without explicit contact nodes |
| Delivered cooling | Heat removed at the cold reservoir/exchanger boundary, not merely module-face `Q_c` |
| Module COP | Useful heat divided by electrical power at the thermoelectric module |
| Wall-plug COP | Useful heat divided by modeled upstream electrical input including converter losses |
| Current schedule | A scalar current or a known piecewise-constant function of time |
| PWM | Fast electrical switching represented through current moments when thermal states are slow |
| Dense truth | Full conventional-solver trajectory, kept separate from visible observations |
| Virtual sensor | A declared transformation from dense truth to sampled, possibly imperfect records |
| Same-model recovery | Synthetic data are generated and fitted with the same equations |
| Identifiability | Whether the chosen sensors and excitation distinguish a parameter from alternatives |
| Information gain | Local Gaussian reduction in parameter uncertainty predicted for an experiment |
| PINN | Neural state representation trained partly or entirely through governing-equation residuals |
| Hidden state | A modeled temperature that is not supplied as an observation label |
| Constitutive function | A material law such as `alpha(T)`, `rho_e(T)`, or `kappa(T)`, represented over temperature rather than by one scalar |
| Thomson coefficient | `tau(T) = T d(alpha)/dT`; couples current and a temperature gradient when the Seebeck coefficient varies with temperature |
| Distributed model | One-dimensional leg model with internal energy storage and a spatial temperature field |
| Withheld regime | A complete current schedule excluded from fitting and used for transfer evaluation |
| Candidate-pool optimum | Best tested design in a finite declared pool, not a proof of a global optimum |
| Hardware validation | Comparison with calibrated physical measurements; not yet performed |

---

## 17. License and third-party material

ThermoTwin's original code and documentation are available under the
[MIT License](../LICENSE). The license does not replace the terms that apply to
third-party datasets, literature-derived material records, or publication
content. Their source-specific licenses, citations, and provenance remain in
the corresponding experiment documentation.
