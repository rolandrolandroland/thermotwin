# Budget-Constrained Bayesian Experiment Selection for Thermoelectric Parameter Estimation

**A planning criterion, and what happened when it was tested**

ThermoTwin technical report. All results are synthetic.

---

## Abstract

Hardware development is rate-limited by iteration: each physical test costs
stand time, energy, and calendar days. We implement and then audit a planner
that chooses the next test before any data exist. For a module-scale
thermoelectric assembly with three unknown thermal parameters observed through
two exchanger thermocouples, a local Bayesian D_s-optimality criterion scores 25
candidate current pulses in about two seconds under a 30 J electrical-energy
budget and face-temperature limits. Seventeen candidates are feasible, and the
planner selects 0.8 A for 20 s at 7.198 nats of expected information. We then
tested that recommendation against two comparators in 20 paired trials of
complete nonlinear refits, varying the true parameters, the sensor biases, and
the noise. Three findings follow. First, the planner's pre-experiment precision
forecast matched the errors the refits actually produced: all nine
realized-to-forecast ratios fall inside the 0.69–1.31 range that 20 trials can
resolve, which is what licenses ranking 25 candidates without running them.
Second, the selected pulse reduced the three parameter errors by approximately
6.55x, 6.60x, and 3.92x against the cheapest feasible pulse, a 3.9–6.6x range,
and won all 20 paired trials. Third, its advantage over
a similar-energy alternative was not established: it won 12 of 20 trials
(sign-test p = 0.50, paired t(19) = 1.31, p = 0.20). The dominant lever on the
recommendation is the energy budget itself, which changes the winner at nearly
every value tested and which the implementation does not justify. Truth and
inference share the same lumped equations throughout, so this validates the
selection methodology rather than hardware identifiability.

---

## 1. Introduction

For physical-science hardware, the binding constraint is usually not compute but
iteration count. Every characterization test occupies a rig, consumes energy,
perturbs the unit under test, and returns a finite quantity of information.
Choosing *which* test to run is therefore a design decision with the same
economics as choosing what to build.

Optimal experimental design (OED) formalizes that decision: rank candidate
excitations by a scalar functional of the information they are expected to
yield, subject to feasibility constraints, and run the winner. The standard
objection is that for a nonlinear model the ranking is computed from a
linearization at *assumed* parameter values, before the parameters are known,
with no guarantee that it predicts the behavior of a real nonlinear estimator at
parameter values displaced from those assumptions.

This report does two things:

1. Implements a constrained local Bayesian D_s-optimal planner for a
   module-scale thermoelectric model (Section 3.5), and
2. Audits its recommendation with paired nonlinear refits, reporting not only
   whether the selected experiment won, but whether the planner's own precision
   forecast was trustworthy and whether the measured advantage is
   statistically supportable (Sections 3.6 and 4).

The second half is the point of the exercise. A planner that cannot be checked
is a planner that cannot be deployed on hardware, where the truth is never
available.

## 2. Background

For a model predicting observations `y(θ)` with independent Gaussian noise of
standard deviation σ, the Fisher information matrix is `F = JᵀJ/σ²`, where `J`
holds the sensitivities `∂y/∂θ`. By the Cramér–Rao bound, `F⁻¹` is the best
achievable covariance for an unbiased estimator, so `sqrt(diag(F⁻¹))` is a
pre-experiment forecast of estimation precision (Rao 1945; Cramér 1946). In
heat-transfer parameter estimation this construction, with sensitivities scaled
by the parameter itself, is standard practice (Beck & Arnold 1977).

Design criteria are scalar summaries of that covariance. D-optimality maximizes
`det F`, equivalently minimizing the volume of the confidence ellipsoid (Kiefer
1959). Unlike A-optimality, which sums individual variances, the determinant
rewards designs that *separate* parameters rather than measuring one direction
well. When only a subset of parameters is of interest and the rest are nuisance,
the corresponding criterion is D_s-optimality (Atkinson, Donev & Tobias 2007).

With a Gaussian prior and a linear-Gaussian likelihood, the expected information
gain — the mutual information between parameters and data, and equivalently the
expected Kullback–Leibler divergence from prior to posterior (Lindley 1956) — has
the closed form

```
EIG = ½ · ln( det Σ_prior / det Σ_posterior )
```

measured in nats, the natural-logarithm unit of information (1 nat ≈ 1.443
bits). Because a 3x3 covariance determinant scales as the square of an ellipsoid
volume, this equals `ln(volume_prior / volume_posterior)`. Maximizing it is
Bayesian D-optimality (Chaloner & Verdinelli 1995). For nonlinear models the
fully Bayesian form averages the criterion over the prior; evaluating it at a
single nominal point, as here, is the classical *locally optimal* approximation
(Chernoff 1953).

## 3. Methods

### 3.1 Forward model

A four-node lumped thermal network represents the assembly: cold thermoelectric
face, hot face, cold heat exchanger, hot exchanger. Junction heat terms are

```
Q_c = α·I·T_c − ½·I²·R − K·(T_h − T_c)
Q_h = α·I·T_h + ½·I²·R − K·(T_h − T_c)
```

with Joule heat split evenly between faces and `K` the parasitic conduction
leak. Faces couple to their exchangers through finite thermal contacts, and
exchangers couple to fixed reservoirs:

```
C_cf·dT_cf/dt = (T_cx − T_cf)/R_c − Q_c
C_hf·dT_hf/dt = Q_h − (T_hf − T_hx)/R_h
C_cx·dT_cx/dt = G_c·(T_res − T_cx) − (T_cx − T_cf)/R_c
C_hx·dT_hx/dt = G_h·(T_res − T_hx) + (T_hf − T_hx)/R_h
```

Frozen reference values: α = 0.05 V/K, R = 2.0 Ω, K = 0.5 W/K; capacitances
(50, 100, 50, 100) J/K; both contacts 0.25 K/W; G = (2.0, 4.0) W/K. All nodes
and both reservoirs start at 300 K. Integration is classical fixed-step RK4 at
0.2 s over an 80 s horizon, with step boundaries aligned to current transitions.

### 3.2 Observation model

Only the two exchanger nodes are instrumented. The measurement chain applies, in
order: sampling of the trajectory, a shared first-order sensor lag
`τ_s·dT_sensor/dt = T_node − T_sensor` evaluated on the dense grid, a constant
per-sensor bias, independent Gaussian noise with σ = 0.02 K, and decimation to a
1 s interval. This yields **162 scalar observations** per experiment (81 times ×
2 sensors). The thermoelectric face temperatures are never observed.

### 3.3 Unknowns and parameterization

Three physical unknowns are estimated in log coordinates, which makes
perturbations dimensionless and enforces positivity:

| Symbol | Meaning | Nominal | Search bounds |
|---|---|---:|---|
| `R_c` | cold thermal contact resistance | 0.25 K/W | [0.08, 0.60] |
| `C_cf` | cold-face thermal capacitance | 50 J/K | [20, 120] |
| `τ_s` | shared sensor time constant | 1.5 s | [0.15, 5.0] |

Two constant sensor biases are nuisance parameters. The planner's prior is
diagonal with log standard deviations (0.20, 0.50, 0.30) and 0.10 K on each
bias.

All three parameters shape the same thermal rise: `R_c` throttles heat into the
face, `C_cf` sets the face's own time constant, and `τ_s` smears everything
downstream. Separating them is the identification problem.

### 3.4 Candidate space and constraints

Twenty-five single rectangular pulses: amplitude ∈ {0.4, 0.6, 0.8, 1.0, 1.2} A
crossed with duration ∈ {5, 10, 15, 20, 30} s, each beginning at t = 5 s.
Feasibility requires ≤ 30 J of modeled electrical energy and face temperatures
within 285–315 K. Energy is integrated per constant-current segment with pulse
edges as explicit quadrature boundaries, so the discontinuous `V·I` jumps do not
acquire a grid-dependent ramp.

### 3.5 Information criterion

Per candidate: build `J` (162 × 5) by central finite differences with a log step
of 0.01 for the three physical columns, appending exact indicator columns for the
two biases; form `F = JᵀJ/σ²`; add the diagonal prior precision; invert; take
the upper-left 3×3 block, which marginalizes over the biases rather than
conditioning on them; and score

```
½ · ln( det Σ_prior,phys / det Σ_post,phys )    [nats]
```

Infeasible candidates are scored and reported but excluded before ranking. The
whole sweep runs in about 2.3 s.

### 3.6 Validation campaign

Three arms were carried forward:

| Role | Rule | Pulse | Energy | Score |
|---|---|---|---:|---:|
| Selected | max information among feasible | 0.8 A / 20 s | 27.54 J | 7.198 nats |
| Naive | minimum energy among feasible | 0.4 A / 5 s | 1.64 J | 2.889 nats |
| Control | energy closest to the selected pulse | 0.6 A / 30 s | 23.77 J | 6.958 nats |

The control exists because the selected pulse uses 16.8x the naive pulse's
energy; beating the naive arm alone would not separate design quality from
budget spent. It is the nearest feasible grid point, 3.76 J away, not an exact
energy match.

**Paired design.** Twenty trials. Each draws one truth — each physical parameter
as `nominal · exp(N(0, 0.10))`, each bias as `N(0, 0.05 K)` — and one noise
seed, and **all three arms receive the identical truth and noise**. Errors across
trials are therefore correlated (0.85 between the selected and control arms),
and differencing within a trial removes that shared difficulty.

**Estimator.** Objective is the mean squared noise-normalized residual over the
162 observations. The two biases are profiled analytically at every evaluation
as the per-sensor mean residual, the exact conditional minimizer under iid
Gaussian noise, leaving a three-dimensional search. Optimization is damped
Gauss–Newton in log coordinates, 12 iterations, hard clipping to the bounds,
central-difference Jacobian, started from three deliberately off-truth points
and keeping the lowest-loss fit. The campaign takes 3 min 48 s.

**Metrics.** Per-parameter percentage RMSE across trials; joint log-parameter
RMSE per trial; the ratio of realized log RMSE to the planner's prior-free
forecast, compared against the χ²-based interval a calibrated forecast would
still produce over 20 trials; local interval coverage; the scale-free
uncertainty volume `sqrt(det Σ)` on the physical block; and predictive transfer
to a withheld schedule with a polarity reversal,
`I(t) = (0, 0.75, 0, −0.45, 0) A` switching at (8, 28, 42, 58) s.

The forecast used for the calibration ratio excludes the prior, because the
refits use no prior. Including it would compare mismatched quantities and
inflate the apparent miss for weakly informative pulses.

## 4. Results

### 4.1 The constraint binds, and it binds on energy

Seventeen of 25 candidates are feasible. Sorted by raw information, the selected
pulse ranks **eighth of 25**; the seven candidates above it are all rejected:

| Pulse | Energy | Score | Status |
|---|---:|---:|---|
| 1.2 A / 30 s | 94.99 J | 8.866 | rejected |
| 1.0 A / 30 s | 65.99 J | 8.366 | rejected |
| 1.2 A / 20 s | 61.92 J | 8.309 | rejected |
| 1.0 A / 20 s | 43.01 J | 7.810 | rejected |
| 1.2 A / 15 s | 45.82 J | 7.770 | rejected |
| 0.8 A / 30 s | 42.25 J | 7.750 | rejected |
| 1.0 A / 15 s | 31.82 J | 7.275 | rejected (misses by 1.8 J) |
| **0.8 A / 20 s** | **27.54 J** | **7.198** | **selected** |

Information increases monotonically in both amplitude and duration, so without
the constraint the problem is trivial. All eight rejections are on energy; the
declared 285–315 K face limits never bind, since the entire grid stays within
295.5–303.4 K. The information frontier also saturates: 0 → 27.5 J buys
2.9 → 7.2 nats, while 27.5 → 95 J buys only 1.7 more.

### 4.2 Parameter recovery under complete nonlinear refits

Percentage RMSE across the 20 trials:

| Pulse | Contact resistance | Heat storage | Sensor lag |
|---|---:|---:|---:|
| Selected | **4.96%** | **2.15%** | **8.71%** |
| Control | 6.21% | 2.57% | 9.23% |
| Naive | 32.47% | 14.21% | 34.16% |

Against the naive pulse, error is lower for all three parameters: approximately
6.55x for contact resistance, 6.60x for heat storage, and 3.92x for sensor lag.
Against the control it is small and non-uniform: the selected pulse's
advantage is concentrated in contact resistance and heat storage and nearly
absent in sensor lag.

Uncertainty volume falls by 99.65% versus naive and 21.93% versus the control.
Interval coverage is 98.3% individual / 95.0% simultaneous for the selected arm
and 100% / 100% for naive — the latter an artifact of intervals roughly 286x
larger in volume, not a virtue.

### 4.3 The pre-experiment forecast was calibrated

Realized log RMSE divided by the prior-free forecast, against the 0.69–1.31
range a perfectly calibrated forecast still yields over 20 trials:

| Pulse | Contact resistance | Heat storage | Sensor lag |
|---|---:|---:|---:|
| Selected | 0.85 | 0.77 | 1.25 |
| Control | 0.98 | 0.83 | 1.19 |
| Naive | 0.82 | 0.90 | 1.30 |

**All nine ratios fall inside the band**, for weakly and strongly informative
pulses alike. This is the result that licenses using a two-second calculation to
rank 25 candidates of which 22 were never run.

The sensor-lag ratio sits at the upper edge for all three arms, with a small
negative bias in the fitted lag. Because the three arms share the same 20 truths
and noise draws, this is one correlated observation rather than three
independent ones, and should be treated as a hint for a larger campaign.

### 4.4 The paired comparison

Against the naive pulse, the selected pulse wins **20 of 20** trials
(t(19) = 3.98, p = 0.001). Against the similar-energy control it wins **12 of 20**
(mean margin +0.0065; sign-test p = 0.50; paired t(19) = 1.31, p = 0.20). The
control's median error (0.04487) is in fact marginally better than the selected
arm's (0.04490); the selected arm wins on the mean and on tail behavior. Pairing
is what gives the test any power at all: treating the two sets of 20 errors as
independent samples drops the t statistic from 1.31 to 0.51.

### 4.5 Identifiability and residual correlation

For the selected pulse, the nuisance-profiled noise-normalized singular values
are (139.98, 27.25, 10.88), rank 3/3, condition number 12.86. At zero current
they are (0, 0, 0), rank 0/3: the system remains at its shared equilibrium and
the experiment is formally uninformative.

Optimality reduces volume but does not decorrelate. Under the *best* feasible
pulse the mean absolute correlation between log contact resistance and log heat
storage remains **0.9331**, essentially unchanged from the naive pulse's 0.9245.
Within this candidate class the degeneracy is not removable.

### 4.6 The budget is the dominant lever

Re-ranking the stored scores at other budgets (temperature limits never bind, so
energy alone decides):

| Budget | Feasible | Winner |
|---:|---:|---|
| 10 J | 7 | 0.4 A / 20 s |
| 20 J | 13 | 0.6 A / 20 s |
| 25 J | 16 | 0.6 A / 30 s *(the control)* |
| **30 J** | **17** | **0.8 A / 20 s** |
| 35 J | 19 | 1.0 A / 15 s |
| 100 J | 25 | 1.2 A / 30 s |

The winner changes at nearly every budget. At 25 J the pulse used here as a
control would have been the recommendation. The 30 J value is a configuration
default with no derivation anywhere in the implementation.

### 4.7 Predictive transfer

Simulating the fitted parameters on the withheld schedule, which contains a
polarity reversal absent from every training pulse, gives cold-face RMSE of
0.0207 K (selected), 0.0265 K (control), and 0.1367 K (naive), evaluated on
node temperatures that were never instrumented.

## 5. Discussion

**What this supports.** A local Bayesian D_s-optimal planner, costing seconds,
produced a recommendation that survived complete nonlinear refitting, and — more
usefully — its *precision forecast was calibrated*. On hardware the forecast is
all one has in advance, since running 20 known-truth replicates of each of 25
candidates is not possible. Demonstrating calibration in simulation, where truth
is available, is what earns the right to trust the forecast later.

**Where the value is concentrated.** The planner's payoff is in *excluding weak
experiments*: 3.9–6.6x lower parameter error against a cheap pulse, with lower
combined error in every trial. Among
comparable-cost experiments near the saturated part of the frontier, the fine
ranking is not resolvable at this sample size. The practical reading is to use
the criterion as a filter rather than as an oracle, and to set the resource
budget deliberately, because the budget — not the criterion — determines which
pulse is recommended.

**What optimality does not buy.** D-optimality shrinks the confidence
ellipsoid's volume; it does not rotate it toward the coordinate axes. The
0.93 correlation between contact resistance and heat storage persists under the
best available pulse, so breaking that degeneracy would require a different
*class* of excitation — two-pulse or ramped schedules, or an additional sensor —
rather than a different amplitude and duration.

**Consistency with the sequential study.** A companion campaign compares
adaptive selection, a precommitted greedy D-optimal batch, and a plausible
engineer heuristic under a cumulative budget. In one condition the heuristic
reaches the prediction gate on less energy with marginally lower parameter
error, and D-optimal policies achieve smaller uncertainty volume without a
corresponding predictive gain. That is the same conclusion as Section 4.4,
reached by a different route.

**Limitations.**

- Truth and inference share the same lumped equations; this is an
  estimator-and-design study, not a model-adequacy study.
- The criterion is evaluated at nominal parameters, and validation truths span
  only about ±10% in log around them. Behavior at larger displacement is
  untested.
- Only 3 of 25 candidates received full refits.
- Noise is treated as independent with known σ. Real thermocouple noise is
  typically autocorrelated, which would make `JᵀJ/σ²` overstate information.
- Prior widths are declared assumptions that enter the reported nats.
- Only single rectangular pulses and two exchanger sensors are candidates. Flow
  rate is excluded because the lumped model has no validated
  flow-to-conductance relationship.
- Twenty trials leave wide binomial uncertainty on every coverage figure.

## 6. Conclusion

A constrained local Bayesian D_s-optimal planner ranked 25 candidate current
pulses for a module-scale thermoelectric assembly in about two seconds, and its
pre-experiment precision forecast agreed with complete nonlinear refits to
within the resolution of a 20-trial paired campaign. The selected pulse improved
the three parameter errors by approximately 3.9–6.6x over the cheapest feasible
experiment, with lower combined error in all 20 trials. Its advantage over a
similar-energy alternative was not established
(12 of 20 trials, p = 0.50), and the recommendation itself is most sensitive to
the energy budget, which the implementation does not justify. The transferable
result is methodological: cheap information forecasts can be validated, and once
validated they are a defensible way to exclude uninformative physical tests
before committing rig time.

## Figures

- `figures/NONLINEAR_EXPERIMENT_SELECTION/next_experiment_story_corrected.png` —
  four-panel summary: constrained frontier, per-parameter recovery, forecast
  calibration with the 20-trial band, and the paired margin.
- `figures/NEXT_EXPERIMENT_WALKTHROUGH/experiment_selection.png` — planner
  detail: frontier, information heat map over the grid, forecast precision, and
  the linearized check.
- `figures/NONLINEAR_EXPERIMENT_SELECTION/nonlinear_experiment_selection.png` —
  validation detail: per-parameter error, interval widths, coverage, joint
  volume.

## Reproducibility

```bash
python3 -m thermotwin.experiment_selection                       # planner, ~2 s
python3 -m thermotwin.nonlinear_experiment_selection             # refits, ~4 min
python3 -m thermotwin.reports.next_experiment_story_corrected    # summary figure
```

Every run is seeded (planner seed 2030; campaign truth/noise seeds 52001 + 2i
and 52001 + 2i + 1) and writes a PNG with same-stem JSON and plain-text
sidecars. Detailed walkthroughs are `NEXT_EXPERIMENT_WALKTHROUGH.md` and
`NONLINEAR_EXPERIMENT_SELECTION.md`; scope and status are tracked in
`ROADMAP.md` under Milestones 5 and 6B. The implementation was developed with
AI coding assistance.

## References

- Rao, C. R. (1945). Information and the accuracy attainable in the estimation of statistical parameters. *Bulletin of the Calcutta Mathematical Society* 37.
- Cramér, H. (1946). *Mathematical Methods of Statistics*. Princeton University Press.
- Chernoff, H. (1953). Locally optimal designs for estimating parameters. *Annals of Mathematical Statistics* 24(4).
- Lindley, D. V. (1956). On a measure of the information provided by an experiment. *Annals of Mathematical Statistics* 27(4).
- Kiefer, J. (1959). Optimum experimental designs. *JRSS B* 21(2).
- Beck, J. V. & Arnold, K. J. (1977). *Parameter Estimation in Engineering and Science*. Wiley.
- Chaloner, K. & Verdinelli, I. (1995). Bayesian experimental design: a review. *Statistical Science* 10(3).
- Walter, E. & Pronzato, L. (1997). *Identification of Parametric Models from Experimental Data*. Springer.
- Atkinson, A. C., Donev, A. N. & Tobias, R. D. (2007). *Optimum Experimental Designs, with SAS*. Oxford University Press.
- Franceschini, G. & Macchietto, S. (2008). Model-based design of experiments for parameter precision: state of the art. *Chemical Engineering Science* 63(19).
