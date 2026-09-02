# Next-experiment selection walkthrough

## Question

Given a limited electrical-energy budget, which feasible single pulse is
expected to best separate cold contact resistance, cold-face thermal
capacitance, and exchanger-sensor lag?

This is the first implemented experiment-planning step in ThermoTwin. It ranks
experiments before collecting their observations.

## Candidate space and constraints

The candidate grid contains 25 pulses:

- current amplitudes: 0.4, 0.6, 0.8, 1.0, and 1.2 A;
- pulse durations: 5, 10, 15, 20, and 30 s;
- pulse start: 5 s;
- total observation horizon: 80 s;
- sensors: cold and hot exchanger temperatures at 1 s intervals.

A candidate must use at most 30 J of modeled electrical energy and keep the
face temperatures within 285--315 K. Seventeen of the 25 candidates are
feasible.

## Information calculation

The local unknown vector is

$$
\theta =
\left[
\log R_{c,\mathrm{contact}},
\log C_{c,\mathrm{face}},
\log \tau_s,
b_c,
b_h
\right].
$$

Log parameters make the first three perturbations dimensionless and preserve
positivity. For every candidate, centered finite differences calculate the
temperature Jacobian $J$. With assumed noise standard deviation $\sigma$, the
local data information is

$$
F_{\mathrm{data}}=\frac{J^T J}{\sigma^2}.
$$

The candidate score is the reduction in the log determinant of the
three-physical-parameter covariance after including nuisance sensor biases and
a broad prior. This is a joint-volume metric: it rewards experiments that
separate the parameters rather than measuring only one direction accurately.

## Selection result

The selected pulse is:

```text
0.8 A for 20 s, beginning at 5 s
```

Its modeled electrical energy is 27.54 J and its expected information gain is
7.198 nats. Its local log-parameter standard errors are:

| Parameter | Log-space standard error |
| --- | ---: |
| Cold contact resistance | 0.0541 |
| Cold-face capacitance | 0.0271 |
| Shared sensor lag | 0.0721 |

The deliberately naive comparison is the smallest feasible pulse, 0.4 A for
5 s. It provides 2.889 nats of expected information.

The highest-energy pulses are not selected automatically: candidates that
exceed the 30 J budget are displayed but rejected before ranking.

Electrical energy is integrated per constant-current segment. The pulse on
and off times are explicit boundaries, so the discontinuous $VI$ values are
not joined by a fictitious output-grid-dependent trapezoidal ramp.

## Repeated-noise validation

The selected and naive pulses are each evaluated with 250 reproducible
Gaussian-noise trials using the same local linear estimator. Relative to the
naive pulse, the selected experiment reduces joint log-parameter RMSE by
82.2%. Nominal 95% interval coverage is 94.5% for the selected pulse and 94.4%
for the naive pulse.

The similar coverage values mean both linearized uncertainty calculations are
internally calibrated in this same-model test. The selected pulse's benefit is
substantially narrower joint uncertainty, not artificially higher coverage.

## Reproduce

```bash
python3 -m thermotwin.experiment_selection
```

Generate the candidate-frontier figure, its JSON data, and its plain-text
explanation with:

```bash
python3 -m thermotwin.experiment_selection_report
```

The artifacts are written to `figures/NEXT_EXPERIMENT_WALKTHROUGH/`.

The implementation is in [`experiment_selection.py`](experiment_selection.py)
and the constraint/information regression tests are in
[`../tests/test_experiment_selection.py`](../tests/test_experiment_selection.py).

## Limitations

- Ranking is local around nominal parameter values.
- The validation uses the local linear response, not a complete nonlinear
  refit in every trial.
- The prior widths are explicit design assumptions.
- Only rectangular single pulses and two exchanger sensors are candidates.
- Flow rate is absent because the current lumped model has no validated
  flow-to-conductance relationship.
