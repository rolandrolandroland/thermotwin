# Nonlinear validation of next-experiment selection

## Question

Does the current pulse selected by a local information calculation still
improve a complete nonlinear fit when the hidden parameters, nuisance sensor
biases, and measurement noise vary?

This is the completion experiment for the current lumped scopes of Roadmap
Milestones 5 and 6B. It tests the earlier recommendation rather than assuming
that local linearized information automatically predicts nonlinear recovery.

## Unknowns and observations

The fit releases three positive physical quantities in log coordinates:

1. cold thermal-contact resistance `R_c`, nominally 0.25 K/W;
2. cold-face thermal capacitance `C_cf`, nominally 50 J/K;
3. shared first-order sensor lag `tau_s`, nominally 1.5 s.

Two constant exchanger-sensor biases are nuisance terms. They are profiled at
every physical-parameter evaluation by taking the mean residual for each
sensor. The visible data are only the cold- and hot-exchanger temperature
histories at 1 s intervals with 0.02 K Gaussian noise.

The physical parameters enter the forward model differently:

```text
contact heat rate = (T_cx - T_cf) / R_c
C_cf dT_cf/dt = contact heat rate - module cold-side heat rate
tau_s dT_sensor/dt = T_target - T_sensor
```

That distinction is what a sufficiently rich transient can exploit. A weak or
zero-current experiment may not separate the effects.

## Candidate ranking

The existing planner evaluates 25 pulse combinations:

- amplitude: 0.4, 0.6, 0.8, 1.0, or 1.2 A;
- duration: 5, 10, 15, 20, or 30 s;
- pulse begins at 5 s;
- electrical energy must not exceed 30 J;
- cold and hot face temperatures must remain inside declared bounds.

The local ranking includes all three physical sensitivities and both nuisance
bias columns. The nonlinear validation compares three choices:

| Role | Pulse | Modeled energy | Local information |
| --- | --- | ---: | ---: |
| Selected | 0.8 A for 20 s | 27.5357 J | 7.1978 nats |
| Naive | 0.4 A for 5 s | 1.6395 J | 2.8893 nats |
| Closest-energy grid control | 0.6 A for 30 s | 23.7720 J | 6.9578 nats |

The third comparison matters because the selected pulse uses much more energy
than the naive pulse. It is the feasible point nearest the selected energy in
the frozen grid, not an exactly energy-matched design.

## Nonlinear estimator

For every candidate and trial, the estimator:

1. simulates the four-node model for a candidate parameter vector;
2. applies the declared first-order lag;
3. profiles both sensor biases;
4. forms noise-normalized residuals;
5. estimates finite-difference Jacobian columns;
6. performs bounded damped Gauss-Newton updates;
7. repeats from three off-truth starts and retains the lowest-loss fit.

Local covariance includes the three physical columns and both bias columns.
The reported uncertainty volume is

```text
sqrt(det(covariance of log R_c, log C_cf, log tau_s)).
```

This is a scale-independent three-parameter local-volume measure. It is not a
global confidence region.

## Paired repeated design

Twenty trials vary all three physical truths lognormally with a log standard
deviation of 0.10 and both sensor biases normally with a 0.05 K standard
deviation. Each trial gives the selected, naive, and closest-energy candidates
the same hidden truth and same random-noise sequence.

The predeclared primary metric is mean physical log-parameter RMSE. Additional
checks are:

- individual and simultaneous local 95% interval coverage;
- local uncertainty volume;
- contact-resistance/capacitance correlation;
- search-bound hits;
- cold- and hot-face RMSE on one common withheld current schedule.

One representative trial also fixes each log parameter at seven offsets and
nonlinearly re-optimizes the other two, producing selected-versus-naive profile
curves.

## Prefit identifiability

The selected pulse has noise-normalized nuisance-profiled singular values

```text
(139.98174, 27.24590, 10.88232)
```

and supports all three declared directions with condition number 12.8632.

At exactly zero current all three singular values are zero. The rank is 0/3:
the temperatures remain at their shared equilibrium, so this experiment
contains no information about contact resistance, face capacitance, or lag.
This is the explicit underdetermined limiting case.

## Nonlinear results

| Experiment | Mean log-RMSE | Worst log-RMSE | Individual 95% coverage | Simultaneous 95% coverage | Uncertainty volume | Withheld cold-face RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected | 0.049035 | 0.190960 | 98.3% | 95.0% | 2.524315e-5 | 0.020663 K |
| Naive | 0.264415 | 1.328424 | 100.0% | 100.0% | 7.209719e-3 | 0.136690 K |
| Closest-energy control | 0.055577 | 0.177147 | 95.0% | 90.0% | 3.233467e-5 | 0.026505 K |

The selected pulse reduces mean joint log-parameter RMSE by:

- 81.46% relative to the naive pulse;
- 11.77% relative to the closest-energy grid control.

It reduces the local uncertainty volume by:

- 99.65% relative to the naive pulse;
- 21.93% relative to the closest-energy grid control.

The selected and closest-energy experiments both avoid bound hits; the naive
experiment hits a search bound once. The selected pulse also gives the lowest
mean withheld cold-face error.

The selected fit retains mean absolute correlations of 0.9331 between log
contact resistance and log face capacitance, 0.7435 between log contact
resistance and log lag, and 0.5611 between log capacitance and log lag. The
pulse improves estimation but does not make the parameters independent. The
profiles and covariance are therefore part of the result, not optional
decoration.

The naive intervals are much wider and consequently over-cover all three
parameters in this small campaign. High coverage alone does not imply useful
precision.

## Decision

For the frozen candidate grid and synthetic lumped model, the local
recommendation survives complete nonlinear refitting. The closest-energy
comparison shows that the benefit is not solely the trivial consequence of
using 16.8 times the naive experiment's energy; the selected pulse still gives
lower mean parameter error and uncertainty than a nearby-resource alternative.

This closes the current synthetic lumped selection claim. It does not validate
the ranking on hardware or close the separate distributed-property selection
work in Milestone 9.

## Interpretation limits

- Truth and inference share the same lumped equations.
- Twenty trials give useful failure visibility but still have broad binomial
  uncertainty.
- The ranking is local at nominal parameters; validation truths vary only over
  the declared neighborhood.
- Local quadratic intervals are checked empirically here, but they are not
  guaranteed global confidence sets.
- The closest-energy control is selected from a discrete grid and differs by
  3.7637 J from the selected pulse.
- No hardware-calibrated prior, noise covariance, or sensor drift distribution
  is used.

## Reproduce

```bash
thermotwin-nonlinear-experiment
```

or

```bash
python3 -m thermotwin.nonlinear_experiment_selection
```

The report writes a PNG plus same-stem JSON and TXT sidecars under
`thermotwin/figures/NONLINEAR_EXPERIMENT_SELECTION/`.
