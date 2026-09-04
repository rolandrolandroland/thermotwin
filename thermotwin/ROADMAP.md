# ThermoTwin project roadmap

This is the governing development roadmap for ThermoTwin. It replaces the
original draft roadmap while preserving its scientific purpose: build a
physics-informed digital twin that can recover hidden thermoelectric-system
parameters from sparse measurements, state when those parameters are not
identifiable, compare control strategies, and recommend informative
experiments.

The revisions reflect what the project has taught us. ThermoTwin is now an
explicitly modular two-node/four-node package, the forward PINNs are
per-experiment physics solvers rather than operating-condition surrogates, and
the most useful demonstration of physics-informed learning is inverse recovery
under sparse or incomplete observation—not merely fitting a temperature
curve.

## Status language

- **Complete:** every current exit criterion is implemented, tested, and
  documented.
- **Partial:** useful validated work exists, but at least one current exit
  criterion remains.
- **Not started:** supporting pieces may exist, but the milestone's scientific
  claim has not been demonstrated.
- **Optional:** valuable after the simulation project, but not required for the
  software MVP.

Passing unit tests establishes software behavior; it does not by itself prove
that a physical assumption matches hardware. Synthetic validation and hardware
validation are reported separately.

## Scope and modeling principles

ThermoTwin models a generic thermoelectric heat pump using public physics. It
does not reproduce any proprietary hardware.

The project follows these rules:

1. Keep conventional solvers as the numerical reference.
2. Use PINNs where physical constraints help with hidden states, sparse data,
   or parameter inference—not merely because a neural network can be used.
3. Preserve exact sign conventions, units, limiting cases, and energy checks.
4. Split complete operating regimes rather than random time rows.
5. Separate synthetic truth, sensor observations, and inferred quantities.
6. Report failure, bias, and non-identifiability rather than selecting only
   successful runs.
7. Keep standard workloads CPU-first for an M1 MacBook Pro with 16 GB memory;
   Apple MPS remains optional.

## Model hierarchy

The package deliberately retains two compatible physical topologies:

1. The **two-node model** combines each thermoelectric face and its attached
   thermal mass. It is the smallest model for sign, energy, integration, and
   first-PINN checks.
2. The **four-node contact model** separates cold face, hot face, cold
   exchanger, and hot exchanger. It is the main model for contact-resistance
   inference and virtual experiments.

Zero explicit contact resistance is represented by selecting the reduced
two-node topology, not by inserting a zero denominator into the four-node
equations.

---

## Milestone 0 — Scientific specification

**Status: Complete for the current generic single-block scope.**

### Goal

Define the modeled system, sign conventions, equations, known quantities,
unknown quantities, operating regimes, and success metrics before training a
network.

### Required work

- Define positive current and every heat-flow direction.
- Define the meanings and units of $Q_c$, $Q_h$, voltage, power, COP,
  temperature, conductance, resistance, and capacitance.
- Select generic two-node and four-node parameter sets.
- State every energy balance and modeling assumption.
- Separate known parameters from candidates for inference.
- Freeze representative constant-current, unipolar-pulse,
  lower-amplitude-pulse, and bipolar regimes.
- Define numerical, physical, inference, and transfer metrics.

### Exit criteria

- Every state and parameter is defined with a unit.
- Heating/cooling signs and COP definitions are unambiguous.
- The equations use no proprietary information.
- The selected synthetic baselines are explicitly labeled as generic rather
  than hardware-calibrated.

---

## Milestone 1 — Conventional reference physics

**Status: Complete for the implemented two-node and four-node models.**

### Goal

Provide trustworthy numerical reference solutions before asking a neural
network to solve or infer anything.

### Required work

- Implement Peltier heat, Joule heat, conductive leakage, voltage, and power.
- Implement two-node transient balances.
- Implement the modular four-node contact topology.
- Integrate with fixed-step RK4 while splitting steps at current transitions.
- Support constant, step, unipolar pulse, and bipolar current schedules.
- Calculate heat histories, contact drops, COP, and whole-system energy
  closure.
- Check units, signs, current reversal, zero current, steady state, time-step
  convergence, and algebraic energy identities.
- Generate reproducible reference reports.

### Exit criteria

- Temperatures remain finite and physically interpretable in frozen cases.
- Current reversal changes Peltier heat with the correct sign while Joule heat
  remains nonnegative.
- Energy closure satisfies the documented tolerance.
- Reducing the RK4 step produces convergent trajectories.

---

## Milestone 2 — Reproducible virtual test stand

**Status: Complete.**

### Goal

Create controlled synthetic sensor datasets that behave like measurements
without confusing observations with the dense numerical truth.

### Revised requirements

- Generate every regime with the conventional reference solver.
- Attach named sensors to explicit physical nodes.
- Sample observations independently of the integration time step.
- Support configurable Gaussian noise, fixed calibration bias, first-order
  sensor lag, and controlled missing-observation patterns.
- Preserve whole-regime train/validation/test separation.
- Record complete experiment provenance with each generated dataset:
  physical parameter truth, initial and reservoir temperatures, external heat
  inputs, integration step, duration, current schedule, regime name, and split.
- Record the ordered measurement-processing history, including random seeds
  and all applied sensor settings.
- Provide a compact automated quality summary covering schema, counts,
  completeness, ranges, provenance, ground-truth availability, and split
  integrity.

### Intentional changes from the original roadmap

- **Controlled missingness replaces mandatory random deletion.** Missing data
  around a current transition is more informative scientifically than
  arbitrary deletion. Random deletion can still be added as a future study.
- **A small designed regime suite replaces a broad parameter-space sampler.**
  The current training, validation, and bipolar test regimes are interpretable
  and sufficient for the single-parameter experiments. General candidate
  generation moves to experiment selection in Milestone 6B.
- **In-memory immutable datasets are acceptable.** Standardized disk export is
  required only when external tools, large batches, or a published dataset
  need it.
- **Dataset quality is an automated summary, not a separate large report.**

### Deliverables

- Ideal virtual test stand and immutable long-form observation schema.
- Noise, bias, lag, sampling, missingness, and restricted-sensor transforms.
- Complete dataset provenance and reproducible ground-truth configuration.
- Whole-regime training, validation, and test datasets.
- `python3 -m thermotwin.dataset_quality` quality audit.
- Unit and limiting-case tests for every observation transformation.

### Exit criteria

- Frozen datasets regenerate deterministically from their experiment and
  measurement configurations.
- Random transformations record their seeds.
- Validation and test regimes are not time-row fragments of training runs.
- Ground-truth physical parameters remain available for evaluation but dense
  RK4 trajectories are not exposed as observation columns.
- The quality audit passes provenance, truth, and split-integrity checks.

---

## Milestone 3 — Forward physics-informed models

**Status: Complete for the current synthetic lumped two-node/four-node and
known-switch scopes. Distributed PINN development remains in Milestone 9.**

### Goal

Show that neural temperature functions can satisfy the transient equations and
match conventional reference solutions, then demonstrate where physics helps
relative to an observation-only model.

### Required work

- Scale time, temperature outputs, and residual terms appropriately.
- Use automatic differentiation for temperature rates.
- Evaluate each node balance separately.
- Enforce initial temperatures exactly or verify an equivalent constraint.
- Validate the two-node constant-current PINN.
- Validate the four-node contact PINN.
- Validate piecewise subnetworks for known switched-current schedules while
  maintaining continuous temperatures and one-sided dynamics.
- Compare predictions, errors, residuals, and training histories against RK4.
- Add an explicit whole-system energy-closure diagnostic for PINN predictions.
- Build a matched observation-only baseline for a sparse/missing-data
  reconstruction problem and compare it fairly with the physics-informed
  model.

### Requirements removed or reinterpreted

- The physics-only forward PINN does **not** need a supervised temperature-data
  loss. Observation loss belongs in hybrid or inverse models.
- A second energy equation need not be added to the loss when it merely repeats
  the node balances. Energy closure must instead be calculated and reported as
  an independent diagnostic.
- One universal condition-aware surrogate is not required. The current PINNs
  take time as input and solve a specified experiment. A parametric surrogate
  taking current/conditions as inputs is a later optional acceleration project.
- A time-only network is not expected to generalize to an unseen current
  schedule. Generalization is tested by transferring inferred physical
  parameters through the trusted solver to withheld regimes.

### Completed pieces

- Two-node, contact-aware, and switched-current forward PINNs.
- Exact initial conditions and exact switch-temperature continuity.
- Automatic differentiation and separate residual histories.
- RK4-withheld same-experiment validation and report figures.
- A whole-system post-training energy diagnostic independently assembles
  storage rate, electrical power, reservoir heat, and external heat in physical
  units. Segment-wise integration retains both one-sided switch values and
  avoids interpolation across discontinuous power.
- A five-trial matched reconstruction compares bit-identically initialized
  1,116-parameter models using the same 56 sparse noisy exchanger rows, with
  six rows missing around turn-off. Both receive the exact initial state and
  known switch locations; only one receives the four node residuals.
- The data-only model fits retained noisy rows slightly better (0.017471 versus
  0.019433 K) and trains about 3.8 times faster.
- The physics-informed model reduces missing-exchanger RMSE by 87.86%, hidden-
  face RMSE by 99.68%, and whole-system rate-closure error by 99.19%. All three
  advantages and every documented regression gate hold in 5/5 trials.
- Mean physics-informed hidden-face RMSE is 0.007105 K; node-residual RMS is
  0.002655 K/s; energy-rate closure is 0.132833 W or 10.964% of the RMS net
  input; mean absolute final cumulative closure is 1.952169 J.
- The energy audit is independent as a post-training calculation, not as a new
  physical law: it is algebraically implied by the four node balances.
- Walkthrough: `FORWARD_RECONSTRUCTION_COMPARISON.md`.

### Exit criteria

- Frozen forward cases meet documented temperature-error thresholds.
- Node residuals and independent energy closure remain below documented
  thresholds.
- Exact initial and segment-interface constraints are verified.
- The data-only comparison uses the same visible observations and comparable
  model capacity/training budget.
- The project can state a defensible advantage or limitation rather than merely
  asserting that PINNs are better.

---

## Milestone 4 — Inverse physical-parameter estimation

**Status: Complete for the current synthetic one-parameter lumped scope.**

### Goal

Recover hidden device-level parameters from sparse observations and validate
the recovered physics on operating regimes excluded from fitting.

### Revised requirements

- Begin with one positive parameter: cold contact resistance.
- Infer it with both a conventional optimizer and an inverse PINN using the
  same observations and bounds.
- Validate hidden hot-side trajectories separately from observed cold-side
  histories.
- Transfer each estimate through the conventional solver to withheld
  validation and bipolar test regimes.
- Repeat inverse-PINN recovery for selected noise, bias, lag, missing-data, and
  restricted-sensor cases.
- Compare point estimates, trajectory errors, transfer errors, runtime, and
  failure modes.
- Document model mismatch explicitly when the inverse model does not include a
  measurement imperfection present in the data.

### Scope change

Jointly releasing a second parameter is no longer required merely to finish
Milestone 4. It belongs in Milestone 5, where correlation and identifiability
can be measured rather than obscured.

### Completed pieces

- Reduced-model inverse thermal-conductance PINN.
- Ideal constant-current and switched-current cold-contact inverse PINNs.
- Positivity constraints, sparse observation losses, conventional scalar
  comparison, hidden-state checks, and transfer to two unseen current regimes.
- Conventional robustness studies for all implemented measurement effects.

### Completed imperfect-data result

- Gaussian noise, structured turn-off missingness, and their combination are
  each tested with three independent observation/neural seed pairs and initial
  contact resistances of 0.15, 0.50, and 0.80 K/W.
- The PINN and conventional estimator receive identical transformed rows; both
  cold channels are absent in the missing interval.
- All nine expected-recovery PINN trials pass the predeclared parameter and
  complete-regime transfer gates.
- The conventional scalar estimator remains more accurate on this small
  problem, as it should.
- In an intentionally unmodeled +0.10 K cold-face bias case, every PINN run
  reduces loss by about 99.99%, but aggregate parameter RMSE is 0.035184 K/W
  and one run fails parameter and bipolar-transfer limits. The result is
  retained to show that optimization progress does not diagnose model
  adequacy.
- Walkthrough: `IMPERFECT_INVERSE_PINN.md`.

### Exit criteria

- Ideal recovery is accurate from more than one plausible initial guess.
- At least one noisy and one structured-missingness case are evaluated.
- Recovered parameters predict withheld regimes.
- Conventional and PINN methods receive identical visible observations.
- Failures and sensitivity to initialization are reported.

---

## Milestone 5 — Identifiability and uncertainty

**Status: Complete for the current synthetic lumped multi-parameter scope.
Distributed-function uncertainty remains in Milestone 9 and hardware-calibrated
uncertainty remains in Milestone 8.**

### Goal

Determine which parameter combinations can be recovered, with which sensors
and excitations, and with what uncertainty.

### Revised requirements

- Use sensitivity profiles or local information measures before expensive PINN
  ensembles.
- Vary sensor location/count, sampling interval, noise, lag, bias, missingness,
  and current excitation.
- Jointly release a scientifically motivated second parameter, likely cold
  contact resistance with one capacitance or effective reservoir conductance.
- Calculate parameter correlations and profile the loss surface.
- Use multi-start fitting and selected bootstrap/ensemble cases.
- Construct uncertainty intervals and check their coverage against synthetic
  truth.
- Identify cases where the data cannot separate contact dynamics, sensor lag,
  and thermal capacitance.

### Scope change

Large PINN ensembles are not required for every grid point. Conventional
sensitivities and optimizers can map the space cheaply; PINN ensembles should
be reserved for representative cases where they add evidence.

### Existing foundation

- Repeated Gaussian-noise trials.
- Isolated bias, lag, turn-off missingness, sensor-restriction, and combined
  imperfection studies.
- A local turn-off information metric.
- Joint exchanger-only recovery of cold contact resistance, shared sensor lag,
  and two sensor biases with structured missing readings.
- A local four-parameter covariance, resistance-lag correlation, approximate
  95% intervals, hidden-face reconstruction, and withheld-current transfer.
- Repeated linearized noise trials for selected versus naive experiment
  designs.
- Fixed-coefficient nonlinear resistivity profiles with bounded multistart
  anchoring.
- Twenty independent-truth conventional interval trials with explicit
  shrinkage-plus-curvature and unregularized variants.
- A bounded multistart nonlinear fit jointly releases cold contact resistance,
  cold-face capacitance, and sensor lag while profiling two nuisance biases.
- The selected experiment supports all 3/3 local physical directions; exactly
  zero current supports 0/3.
- Twenty paired nonlinear trials vary all three truths, both biases, and noise,
  and report correlations, representative re-optimized profiles, individual
  and simultaneous interval coverage, bound hits, and withheld-face transfer.
- Selected-pulse local 95% intervals cover 98.3% of individual parameters and
  95.0% of all three simultaneously in the frozen campaign. The strong mean
  absolute correlations of 0.9331 for contact/capacitance, 0.7435 for
  contact/lag, and 0.5611 for capacitance/lag remain visible.
- Walkthrough: `NONLINEAR_EXPERIMENT_SELECTION.md`.

### Exit criteria

- At least one identifiable and one underdetermined multi-parameter case are
  demonstrated.
- Correlations and failure regions are visible rather than inferred from one
  fit.
- Reported intervals have tested synthetic-truth coverage.
- Conclusions use repeated trials or multi-start fits.

---

## Milestone 6A — Control comparison

**Status: Complete for the current generic single-assembly, fixed-reservoir,
rectangular-pulse, and first averaged-electronics scope.**

### Goal

Compare continuous and pulsed operation objectively using the validated
physics model.

### Required work

- Define cooling/heating capacity, module COP, delivered COP, temperature
  constraints, current limits, and comparison horizon.
- Establish fair continuous-current baselines.
- Sweep pulse amplitude, duty cycle, and period.
- Compare equal electrical energy and equal delivered heat where appropriate.
- Produce COP-versus-capacity Pareto fronts and operating maps.
- Propagate representative parameter uncertainty through the comparison.

### Scope change

The conventional solver—not a PINN—is the default control-sweep engine because
it is fast and already validated. A neural surrogate is justified only if the
candidate count makes the conventional solver a demonstrated bottleneck.

Seconds-scale thermal pulses and high-frequency electrical PWM are now treated
as different experiments. The former is resolved by the transient solver. The
latter passes mean and mean-square current plus explicit converter losses into
the thermal equations instead of shrinking the thermal step to the switching
period.

### Exit criteria

- Comparisons use explicit fair constraints.
- Safety and temperature limits are enforced.
- Pulsed operation is allowed to win, lose, or tie according to the results.
- Conclusions remain stable across selected parameter uncertainty.

### Implemented result

- Useful cooling is heat extracted from the cold reservoir, not instantaneous
  module heat.
- Candidates warm for 360 s and are evaluated for 120 s with an explicit
  whole-system storage-drift check.
- Continuous current and every feasible pulse are matched at 2, 5, and 8 W.
- The pulse sweep now spans duty through 0.99 and reports the best period at
  every duty. The older 21.8--27.6% penalty is identified as the 0.75-duty
  slice, not a grid-independent optimized result.
- The grid-independent result is the expected direct-pulse law: at fixed mean
  current, Joule heating is multiplied by $1/D$, and the COP penalty approaches
  zero as duty approaches one. The highest-COP tested 0.99-duty points are
  0.86--1.04% below continuous COP and deliver 0.38--0.44% less cooling at
  equal electrical power.
- The negative result remains stable across the inferred cold-contact
  resistance interval.
- An exact 840-point steady map covers 0--30 K external lift, 0.05--1.50 A,
  reduced and explicit-contact topologies, three interface resistances, and
  both cooling and heating COP.
- At equal 3 W cooling, baseline 0.25 K/W contacts reduce COP by 19--35% over
  the feasible 0--25 K lift range; the target is infeasible at 30 K under the
  current bound.
- The warmed continuous control points agree with the algebraic steady COP
  envelope within 0.04%, and all tested pulses remain below it.
- The first averaged power-electronics comparison distinguishes direct current
  chopping from smoothed PWM-derived current through $\overline I$ and
  $\overline{I^2}$, and reports module versus wall-plug COP separately.
- At 0.6 A mean current, direct 1.5 A chopping produces 2.5 times the ideal-DC
  Joule heat, while the frozen 10% triangular-ripple smoothed case produces
  1.0008 times.
- The PWM closure explicitly names its neglected
  current-temperature-covariance term and its fast-electrical/slow-thermal
  validity condition.
- Walkthroughs are `CONTROL_COMPARISON_EXPERIMENT.md`,
  `COP_OPERATING_MAP_EXPERIMENT.md`, `PULSE_OPERATING_MAP_EXPERIMENT.md`, and
  `PWM_POWER_ELECTRONICS_EXPERIMENT.md`.

---

## Milestone 6B — Next-experiment selection

**Status: Complete for the current synthetic lumped candidate grid and
declared feasibility constraints. Distributed-property experiment selection
remains part of Milestone 9.**

### Goal

Recommend the next feasible experiment that is expected to reduce uncertainty
or separate correlated parameters.

### Required work

- Generate candidate current amplitudes, pulse timings, sampling intervals,
  and sensor configurations.
- Apply physical, duration, and safety constraints.
- Rank candidates using parameter sensitivity, expected information, or
  ensemble disagreement.
- Simulate the top recommendation with hidden truth.
- Refit the model using the added experiment.
- Measure the actual reduction in parameter error/correlation/interval width.
- Compare the recommended experiment with at least one naive choice.

Flow rate is not yet an independent input in the lumped model. It should enter
the candidate space only after a documented flow-to-heat-transfer model or
hardware data is added; until then, reservoir conductance is the effective
heat-transfer parameter.

### Exit criteria

- Candidate ranking is reproducible.
- The recommendation is feasible under stated constraints.
- The selected experiment improves a predeclared uncertainty metric in
  simulation.
- Improvement is measured against a baseline selection strategy.

### Implemented result

- Twenty-five amplitude/duration candidates are ranked after a 30 J energy
  limit and face-temperature constraints.
- The selected 0.8 A, 20 s pulse provides 7.198 nats of expected joint
  information about contact resistance, face capacitance, and sensor lag while
  including two nuisance biases.
- In 250 repeated linearized noise trials, it reduces joint log-parameter RMSE
  by 82.2% relative to the smallest feasible pulse.
- The selected pulse also drives a five-unit synthetic assembly fingerprint.
- A complete 20-trial bounded multistart nonlinear campaign reduces mean joint
  log-parameter RMSE by 81.46% relative to the naive 0.4 A, 5 s pulse.
- The feasible closest-energy grid control is 0.6 A for 30 s at 23.7720 J,
  compared with 27.5357 J for the selected pulse. The selected experiment
  still reduces mean log-parameter RMSE by 11.77% and local uncertainty volume
  by 21.93% relative to that control.
- Paired trials share each hidden truth and noise sequence across candidates;
  complete nonlinear refits, local coverage, parameter profiles, correlations,
  bound hits, and withheld transfer are all retained.
- Walkthrough: `NONLINEAR_EXPERIMENT_SELECTION.md`.

---

## Milestone 6C — Application-specific material and product co-design

**Status: Complete for the current public-data-seeded virtual method. Process,
cost, and hardware calibration remain future work.**

### Goal

Connect material properties, module geometry, thermal interfaces, heat
rejection, electrical drive, relative prototype burden, and application
requirements in one reproducible design-selection loop.

### Required work

- Preserve same-sample material-property relationships and fixed source-data
  provenance.
- Convert p/n Seebeck coefficient, electrical conductivity, thermal
  conductivity, couple count, leg length, and leg area into module $\alpha$,
  $R$, and $K$.
- Include explicit thermal contact resistance, length-independent areal
  electrical-interface resistance, exchanger conductance, current moments,
  converter loss, wall COP, current-density limits, and voltage limits.
- Define application-specific feasibility constraints and scalar objectives.
- Screen 20--30 initial space-filling designs.
- Compare Bayesian optimization with an equal-budget random baseline.
- Retain saturated or negative optimization results.
- Stress-test selected nominal designs against stated material/interface
  uncertainty without presenting assumed spread as measured manufacturing
  capability.
- Document every public-data, physics, and synthetic-assumption boundary.

### Implemented result

- Twelve 300 K same-row Bi/Te-family records are retained from fixed
  StarryData snapshot DOI `10.6084/m9.figshare.11340935.v1` with its checksum.
- Eight design coordinates cover p material, n material, couple count, leg
  length, leg area, contact resistance, and cold/hot exchanger conductance.
- A 24-design Latin-hypercube screen is followed by 12 cost-aware
  expected-improvement selections from 180 candidates for each of three
  application specifications.
- Twenty-five equal-budget random candidate orders provide a repeated
  comparison rather than a favorable single seed.
- The 25 K balanced case improves application utility from 3.9015 to 6.4268
  and reaches the tested pool optimum after five BO additions; the random
  median remains at 3.9015.
- Both 10 K initial screens already contain the tested pool winner, so no false
  improvement is claimed.
- The report identifies both the 25 K balanced and 10 K capacity-first
  selections as binding at the 1.0 A/mm2 peak current-density limit.
- Three 300-trial fixed-current robustness studies show 55.3%, 100.0%, and
  100.0% requirement pass rates. The fragile efficiency winner demonstrates
  why nominal COP optimization is not sufficient for commercialization.
- A separate cost-free process window now maps 0.05--2.5 mm leg length and
  electrical contact resistivity through $5\times10^{-8}$ ohm m2 under both
  the existing 1 A/mm2 campaign constraint and an exploratory 3 A/mm2
  sensitivity. Cooling-target failures are attributed separately to the
  selected current cap or to the finite modeled cooling maximum.
- An opt-in optimized Ag2Se record is tested by replacing only the n material
  in all 204 frozen-pool designs. It improves many matched designs but creates
  no new best feasible design, so the null system-level result is retained.
- Walkthrough: `MATERIAL_GEOMETRY_BAYESIAN_CODESIGN.md`.
- Extensions: `ELECTRICAL_CONTACT_PROCESS_WINDOW.md` and
  `AG2SE_SUBSTITUTION_EXPERIMENT.md`.

### Exit criteria

- Public material rows, synthetic assumptions, and hardware evidence are
  labeled separately.
- Geometry limiting cases and whole-system steady energy balances are tested.
- Candidate generation, GP acquisition, random comparison, and robustness are
  deterministic under recorded seeds.
- Optimization budgets and candidate pools are equal in the BO/random
  comparison.
- Selected designs are reported with requirements, current, COP, cooling,
  cost index, uncertainty, and limitations.

### Next scientific extension

The next co-design claim should be robust or chance-constrained selection using
measured process-capability distributions. Process variables, complex lattice
descriptors, actual material prices, and manufacturing cycle time should enter
only after paired process/property/cost data exist. Temperature-dependent
material curves and hardware-calibrated exchanger/converter models are also
needed before product recommendations.

---

## Milestone 7 — Interview-ready research artifact

**Status: Complete locally for the current synthetic research artifact. Hosted
CI passed on `dev` before the artifact additions and must be rechecked on the
next pushed commit.**

### Goal

Make the scientific result reproducible, understandable, and useful in an
interview without requiring the viewer to inspect the entire source tree.

### Required work

- Maintain concise and detailed READMEs, this roadmap, equations, assumptions,
  limitations, and learning notes.
- Keep automated unit, physics, regression, and report tests.
- Add a continuous-integration workflow for the CPU test suite.
- Maintain a one-command showcase and automatically generated figures.
- Create a final technical summary centered on engineering conclusions.
- Document negative, biased, and non-identifiable cases.
- Prepare a five-slide interview deck, 90-second demonstration, and concise
  resume bullets.

### Scope change

An interactive application is optional rather than mandatory. The existing
one-command static showcase can satisfy the MVP if it communicates the evidence
more clearly and reproducibly than a hurried interface.

### Completed pieces

- Dependency-layered package with installable metadata, compatibility-preserved
  historical imports, extensive tests, two levels of README documentation,
  experiment walkthroughs, report commands, and a focused
  PINN showcase.
- Separate physics, numerics, simulation, observation, inference, PINN,
  design, study, and report namespaces; importing the core does not load
  PyTorch or Matplotlib.
- A one-command engineering decision showcase covering sparse diagnosis,
  control comparison, experiment selection, and assembly screening.
- Separate walkthroughs that retain negative results and distinguish local
  synthetic uncertainty from hardware evidence.
- A concise technical summary centered on the engineering question, strongest
  matched PINN result, inverse/selection decisions, negative results, and
  evidence boundaries.
- A generic five-slide PowerPoint deck with source blocks in every slide's
  speaker notes, plus a 90-second spoken/screen demonstration.
- Concise portfolio bullets that preserve the synthetic-validation boundary.
- `thermotwin-release-audit`, which recomputes the engineering showcase,
  matched PINN comparison, nonlinear pulse validation, and co-design robustness
  value before accepting the public headline metrics.
- A hosted GitHub Actions run that passed all 522 predecessor tests on commit
  `59ba602`; the current local artifact adds three tests.

### Exit criteria

- A clean installation can reproduce principal numbers and figures.
- Continuous integration passes on the public repository.
- The main README communicates the project in roughly two minutes.
- The detailed report makes assumptions and limitations easy to locate.
- The presentation distinguishes synthetic validation from hardware evidence.

---

## Milestone 8 — Optional hardware validation

**Status: Optional; measurement schema and protocol implemented, physical
experiment not started.**

### Goal

Measure the synthetic-to-real gap with a safe benchtop Peltier experiment.

### Required work

- Define current, voltage, temperature, condensation, and handling limits.
- Calibrate temperature, current, and voltage sensors.
- Record ambient conditions and sensor placement.
- Collect constant-current and pulse experiments.
- Fit parameters using a subset of experiments.
- Predict an experiment withheld in its entirety.
- Compare inferred values with datasheet or independent estimates where those
  comparisons are physically meaningful.
- Document unmodeled losses and model discrepancy.

### Exit criteria

- Safety constraints and calibration procedures are documented.
- Raw data and processing provenance are retained.
- At least one genuinely withheld experiment is predicted.
- Disagreement is analyzed rather than hidden by refitting every case.

## Milestone 9 — Distributed constitutive inference

**Status: Partial. Conservative reference physics, practical identifiability,
continuous conventional inference, forward/inverse PINNs, independent-truth
transfer, nonlinear resistivity profiles, repeated local-interval coverage,
and candidate selection are implemented.**

### Goal

Recover a hidden one-dimensional temperature field and thermodynamically
consistent temperature-dependent transport functions from sparse terminal and
face measurements, while stating which coefficient combinations the
instrumentation cannot support.

### Implemented foundation

- One homogeneous leg with `alpha(T)`, `rho_e(T)`, and `kappa(T)` represented
  as constant or piecewise-linear functions.
- Conservative finite-volume heat and electrical fluxes coupled to dynamic
  cold and hot face nodes.
- Thomson heating through `tau(T) = T d(alpha)/dT` and a roundoff-level
  semidiscrete energy audit.
- Zero-current, passive-conduction, current-reversal, steady half-Joule,
  positive-kelvin, RK4-refinement, and switch-timing tests.
- Sparse face-temperature, voltage, and optional heat-rate observations with
  discontinuity-aware diagnostic evaluation.
- A four-regime, noise-normalized finite-difference Jacobian and local singular
  spectrum before neural training.
- Continuous conventional one-function fitting with damped Gauss–Newton polish.
- A forward `(x,t) -> T` PINN with exact initial profile and dynamic boundary
  residuals, validated against withheld finite-volume fields.
- A multi-experiment inverse PINN with one hidden temperature network per
  regime and one shared property curve.
- Independent noise-free inverse checks for `alpha(T)`, `rho_e(T)`, and
  `kappa(T)`, including a terminal-only versus heat-assisted conductivity
  comparison.
- A five-trial resistivity study with independent observation/neural seeds,
  fixed coefficient-and-loss failure criteria, complete trial retention, and
  an explicit unmatched-regularization caveat.
- A three-seed, three-budget inverse-PINN audit that reports observation loss,
  physical residual in K/s, average property level, curve amplitude, center
  contrast, and a truth-blind stopping gate separately.
- A complete-regime transfer study that withholds one entire constant-current
  experiment, freezes each recovered curve, and scores face temperatures,
  hidden internal temperatures, terminal voltage, pointwise error, and energy
  closure without refitting.
- An independent numerical truth generator with nodal temperatures, SSPRK3,
  independently assembled voltage/fluxes, and a smooth cubic resistivity law
  outside the fitted three-knot representation.
- A paired comparison of unregularized and identically curvature-regularized
  conventional/PINN estimators across constant, pulsed, and outside-support
  transfer regimes.
- An intentionally underdetermined observation study with a pre-fit practical
  singular-value gate, an exact zero-current structural rejection, weak-sensor
  ablations, multistart conventional/PINN diagnostics, and explicit refusal to
  promote unsupported optimizer output to a property estimate.
- Representative fixed-coefficient nonlinear resistivity profiles, including
  full bidirectional, explicitly regularized, and weak one-direction cases.
- A 20-trial independent-truth conventional coverage audit using fresh
  nonlinear multistart optima and local quadratic intervals, plus ten paired
  PINN point-estimate trials without a neural uncertainty claim.
- Local Gaussian coefficient uncertainty and D-optimal pulse/lift selection.
- Public walkthroughs, report commands, and reproducible generated figures.

### Current result

- The declared synthetic four-regime observation model is locally full rank for
  each three-knot curve and for all nine coefficients jointly; this is not a
  global or hardware-identifiability claim.
- The forward PINN reaches 0.006420 K internal-field RMSE and 0.015116 K maximum
  error after 800 CPU epochs.
- In noise-free same-model one-function cases, maximum inverse-PINN
  knot-multiplier error is 0.0165 for `alpha(T)` and 0.0180 for `rho_e(T)`.
  The terminal-only inverse-PINN `kappa(T)` fit fails with 0.2799 maximum error
  despite a falling loss; adding idealized face heat-rate observations reduces
  that error to 0.0515. The conventional estimator recovers each truth essentially
  exactly.
- The selected finite candidate is a 20 K, -0.8 A, 0.5 s pulse with 6.0338 nats
  of local information gain.
- Under 0.01 K and 10 µV independent Gaussian noise, the inverse PINN passes
  the predeclared recovery gate in 5/5 seed trials with 0.0254 worst-trial
  knot-multiplier error. The unregularized conventional fit passes 2/5 with
  0.2113 worst-trial error. The gate uses comparable observation loss after an
  audit found that the old report compared PINN total objective with
  conventional observation loss. This is a broad average-level check, not a
  failure-rate estimate or matched-prior superiority comparison.
- At the frozen 600 PINN epochs, mean curve amplitude is 38.0% of truth and the
  physics-residual RMS is 76.25% of the nominal temperature-rate RMS. At the
  unchanged 2,400-epoch budget, increasing physics-loss weight from 1 to 10
  lowers mean residual ratio from 49.35% to 21.17% and observation loss from
  1.113088 to 0.820109. The balanced protocol passes the truth-blind gate in
  3/3 trials and recovers curve shape in 2/3, so operational convergence has
  improved without establishing fully repeatable function recovery.
- With the complete `positive_0.4A_20K_lift` regime withheld, the inverse PINN
  passes all six predeclared transfer criteria in 5/5 trials. The conventional
  fit passes 2/5 because three transferred voltage errors exceed the fixed
  30 microvolt limit. Mean inverse-PINN hidden-field RMSE is 0.000070 K and
  worst pointwise temperature error is 0.000155 K. This is within-model
  operating-regime transfer: truth and prediction still share the equations,
  grid, and three-knot representation.
- Under independent nodal/SSPRK3/cubic truth, both unregularized and matched-
  curvature PINNs pass the in-support property-and-transfer gate in 3/3 paired
  noisy trials. The conventional estimator passes 1/3 both without and with
  the same explicit curvature weight. Mean maximum in-support property error
  is 0.0335--0.0338 for the PINNs versus 0.1561--0.1694 for the conventional
  fits. Matching curvature does not match implicit neural/field bias, and three
  trials do not support a general estimator-superiority or failure-rate claim.
- Under the frozen observation-sufficiency rule, the bidirectional
  temperature-plus-voltage set supports 3/3 local resistivity directions, zero
  current supports 0/3 exactly, positive-current temperatures support 0/3 at
  the declared noise scale, and adding one-direction voltage supports 2/3. A
  stable, accurate-looking PINN curve in the 2/3 case is retained as a warning,
  not reported as identified.
- Across 20 independent-truth noise trials, unregularized local intervals cover
  63.3% of individual resistivity coefficients at the nominal 68% level and
  98.3% at 95%. Matched shrinkage-plus-curvature intervals cover 78.3% and
  100%. Each fraction contains only 60 coefficient checks, penalized intervals
  are not classical confidence intervals, and the ten paired PINNs are point
  estimates rather than an uncertainty ensemble.

### Remaining exit criteria

- Confirm the loss-balanced protocol under fresh independent truth/noise and
  determine whether selected experiments improve the remaining curve-shape
  failure without tuning on the audited property truths.
- Increase repetitions if a precise coverage or failure-rate estimate is
  required; the current 20/10 budget has wide binomial and seed uncertainty.
- Build a calibrated PINN uncertainty method before making neural interval or
  coverage claims; matching visible coefficient penalties alone leaves implicit
  neural bias unmatched.
- Validate on additional complete temperature/current regimes and materially
  different continuum/boundary models. One constant, one pulsed, and one
  outside-support regime now use independent numerical/constitutive truth, but
  the truth is still synthetic and shares the continuum equations.
- Add a p/n unicouple and internal interface conditions only after the
  single-leg observation model is stable.
- Treat switched-current PINNs through explicit time-domain decomposition.
- Attempt hardware inference only after sensor precision and boundary
  conditions are measured.

---

## Recommended execution order from the current state

1. Review, commit, and push the Milestone 7 artifact, then verify hosted CI on
   that exact commit. Milestones 3 through 7 are complete locally for their
   current synthetic scopes.
2. Confirm Milestone 9's loss-balanced distributed protocol under fresh
   independent truth and test whether its selected experiment improves the
   remaining curve-shape failure.
3. Add switched-current distributed PINNs through explicit time-domain
   decomposition.
4. Extend Milestone 6C to chance-constrained optimization only after measured
   process-capability or hardware uncertainty data replace the current virtual
   spreads.
5. Refine Milestone 6A only when new validated physics—such as flowing-fluid
   states, a calibrated converter loss map, or multi-assembly staging—changes
   the control question.
6. Attempt Milestone 8 only if safe hardware and sufficient time are available.

## Final project claim

The intended final result is not simply that a PINN predicts temperature. It
is that ThermoTwin:

- enforces a transparent thermoelectric energy model;
- estimates hidden interface behavior from sparse, imperfect measurements;
- identifies when the measurements cannot support a unique conclusion;
- validates recovered physics on unseen current regimes;
- compares control schedules under explicit engineering objectives; and
- recommends the next experiment expected to reduce uncertainty.

That claim is complete only when each supporting result has a reproducible
configuration, conventional baseline, quantitative metric, and documented
limitation.
