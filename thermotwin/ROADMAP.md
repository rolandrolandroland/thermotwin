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

**Status: Partial. The core forward PINNs are validated; the comparison claim
is not finished.**

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

### Remaining pieces

- Independent whole-system energy-closure history for PINN trajectories.
- Matched data-only baseline under sparse or missing observations.
- A written comparison explaining when physics helps, when it does not, and
  the fairness limitations of the comparison.

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

**Status: Partial. Ideal single-parameter recovery is strong; imperfect-data
PINN recovery remains.**

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

### Remaining pieces

- Inverse PINN on selected imperfect datasets, starting with missing readings.
- Direct imperfect-data comparison with the conventional estimator.
- Failure/recovery criteria across several neural seeds.

### Exit criteria

- Ideal recovery is accurate from more than one plausible initial guess.
- At least one noisy and one structured-missingness case are evaluated.
- Recovered parameters predict withheld regimes.
- Conventional and PINN methods receive identical visible observations.
- Failures and sensitivity to initialization are reported.

---

## Milestone 5 — Identifiability and uncertainty

**Status: Partial. Joint accessible-sensor inference and local uncertainty are
implemented; nonlinear interval coverage and broader identifiability mapping
remain.**

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

**Status: Partial. Constrained local-information ranking and repeated
linearized validation are implemented; complete nonlinear refitting remains.**

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
- A complete nonlinear refit across repeated trials is still required before
  Milestone 6B is complete.

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

**Status: Partial and developed continuously.**

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

### Existing foundation

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

**Status: Partial. Conservative reference physics, local identifiability,
continuous conventional inference, first forward/inverse PINNs, local
uncertainty, and candidate selection are implemented.**

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
- Local Gaussian coefficient uncertainty and D-optimal pulse/lift selection.
- A public walkthrough, report command, generated figure, and private exercise
  sheet.

### Current result

- The declared synthetic four-regime observation model is locally full rank for
  each three-knot curve and for all nine coefficients jointly; this is not a
  global or hardware-identifiability claim.
- The forward PINN reaches 0.006420 K internal-field RMSE and 0.015116 K maximum
  error after 800 CPU epochs.
- In a noise-free same-model resistivity case, the conventional solver recovers
  truth multipliers `(1.04, 1.07, 1.03)` to numerical precision; the inverse
  PINN returns `(1.051672, 1.066605, 1.048003)`.
- The selected finite candidate is a 20 K, -0.8 A, 0.5 s pulse with 6.0338 nats
  of local information gain.

### Remaining exit criteria

- Repeat one-function recovery across realistic noise and multiple neural
  seeds; predeclare failure criteria and report all failures.
- Check nonlinear interval coverage rather than relying only on local Gaussian
  covariance.
- Validate a recovered curve on a complete temperature/current regime excluded
  from fitting.
- Generate synthetic truth with an independent discretization or constitutive
  representation to measure inverse crime and model discrepancy.
- Demonstrate an intentionally underdetermined observation set and show that
  the inference reports the loss of rank rather than inventing a curve.
- Add a p/n unicouple and internal interface conditions only after the
  single-leg observation model is stable.
- Treat switched-current PINNs through explicit time-domain decomposition.
- Attempt hardware inference only after sensor precision and boundary
  conditions are measured.

---

## Recommended execution order from the current state

1. Complete Milestone 9's noisy, multi-seed, withheld-regime one-function
   recovery before attempting a joint three-function inverse PINN.
2. Finish Milestone 3 with PINN energy closure and a matched data-only
   sparse/missing-data comparison.
3. Finish Milestone 4 by training the inverse PINN on selected imperfect
   datasets and comparing it fairly with the conventional estimator.
4. Extend Milestone 5's local joint inference to nonlinear repeated-fit
   coverage and an explicitly underdetermined multi-parameter case.
5. Finish Milestone 6B with complete nonlinear refits of selected and naive
   experiments over repeated synthetic trials.
6. Extend Milestone 6C to chance-constrained optimization only after measured
   process-capability or hardware uncertainty data replace the current virtual
   spreads.
7. Refine Milestone 6A only when new validated physics—such as flowing-fluid
   states, a calibrated converter loss map, or multi-assembly staging—changes
   the control question.
8. Finalize Milestone 7 deliverables throughout, rather than postponing all
   documentation until the end.
9. Attempt Milestone 8 only if safe hardware and sufficient time are available.

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
