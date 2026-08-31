# Noisy multi-seed distributed inverse recovery

## Question

Can sparse, noisy terminal measurements repeatedly recover a three-knot
electrical-resistivity curve when both measurement noise and inverse-PINN
initialization change?

This experiment is the first robustness step after the noise-free distributed
inverse checks. It tests repeatability. It does not yet test transfer to a
withheld regime, an independent truth model, or hardware.

## What is inferred

Only the electrical-resistivity function `rho_e(T)` is released. It is
piecewise linear at 285, 300, and 315 K. The Seebeck coefficient and thermal
conductivity remain fixed at their declared baselines.

The hidden truth multiplies the baseline resistivity values by

```text
(1.04, 1.07, 1.03).
```

Each trial uses the same four constant-current and temperature-lift regimes as
the noise-free shared-property inverse. One temperature network represents the
hidden field in each regime, and all four networks share one resistivity curve.

## Visible measurements

The inverse methods see only:

- cold-face temperature;
- hot-face temperature; and
- terminal voltage.

They do not see the finite-volume cell temperatures or truth coefficients.
Samples are spaced by 0.08 s. Independent Gaussian noise is added with declared
standard deviations

```text
temperature: 0.01 K
voltage:     1.0e-5 V
```

These are synthetic precision assumptions, not measured instrument
specifications.

## Independent seed allocation

Every trial receives a block of five unique seeds:

1. one PyTorch initialization seed; and
2. one Python noise seed for each of the four regimes.

Trial 0 uses 27001–27005, trial 1 uses 27006–27010, and so on. No noise stream
is reused across a regime or trial, and no neural seed collides with a noise
seed.

## Predeclared success criteria

The criteria were encoded before the frozen five-trial campaign ran. A
completed fit passes only if all three conditions hold:

| Criterion | Limit |
| --- | ---: |
| Maximum absolute knot-multiplier error | at most 0.10 |
| Loss reduction from the initial state | at least 90% |
| Final normalized loss | at most 5.0 |

The coefficient criterion prevents a low loss from being mistaken for correct
property recovery. The loss criteria reject a numerically stalled fit. Every
trial is retained whether it passes or fails.

The 0.10 coefficient limit is deliberately broad: it means a knot multiplier
must be within ten percentage points of its truth. Passing it is a minimum
recovery check, not a precision-material-characterization standard.

## Compared estimators

### Conventional estimator

The conventional baseline uses bounded coordinate search followed by damped
Gauss–Newton refinement. Its objective is the noise-normalized observation
error. It has no explicit coefficient-smoothness penalty.

### Inverse PINN

The inverse PINN jointly trains four hidden temperature fields and the shared
resistivity curve. It combines normalized observation loss, PDE and dynamic
boundary residuals, and a small second-difference smoothness term. Each trial
uses 600 CPU epochs.

Both estimators receive exactly the same noisy observations within a trial,
but they do not have matched regularization. Consequently, their comparison
shows the behavior of the implemented estimators, not an intrinsic theorem
that one estimator class is superior.

## Frozen results

| Trial | Conventional multipliers | Conventional max error | Pass? | Inverse-PINN multipliers | PINN max error | Pass? |
| ---: | --- | ---: | :---: | --- | ---: | :---: |
| 0 | `(1.218080, 1.074092, 0.818731)` | 0.211269 | no | `(1.048511, 1.066919, 1.053982)` | 0.023982 | yes |
| 1 | `(1.041083, 1.067091, 1.048710)` | 0.018710 | yes | `(1.057637, 1.064342, 1.055223)` | 0.025223 | yes |
| 2 | `(0.977199, 1.065985, 1.126568)` | 0.096568 | yes | `(1.056542, 1.064548, 1.055435)` | 0.025435 | yes |
| 3 | `(0.890551, 1.068883, 1.185697)` | 0.155697 | no | `(1.049106, 1.066006, 1.050019)` | 0.020019 | yes |
| 4 | `(0.937264, 1.071299, 1.125303)` | 0.102736 | no | `(1.049618, 1.067000, 1.048863)` | 0.018863 | yes |

Summary:

| Metric | Conventional | Inverse PINN |
| --- | ---: | ---: |
| Passing trials | 2/5 | 5/5 |
| Coefficient RMSE over all completed knots and trials | 0.102054 | 0.015370 |
| Worst trial maximum error | 0.211269 | 0.025435 |
| Mean recovered multipliers | `(1.012835, 1.069470, 1.061002)` | `(1.052283, 1.065763, 1.052705)` |

The conventional fit reaches its lower log-multiplier bound in trial 0. That
diagnostic is reported, but bound contact was not added retroactively to the
predeclared pass/fail gate; trial 0 already fails the coefficient-error limit.

## What the result means

The middle, 300 K coefficient is stable for both methods. The noisy
conventional endpoint coefficients vary much more strongly, consistent with
the earlier local analysis showing a weak resistivity coefficient combination.

The PINN estimates are tightly clustered and pass the declared recovery gate in
all five trials. They are also pulled toward a smoother curve: the mean 315 K
multiplier is 1.0527 rather than the 1.03 truth. The stability can therefore
come from useful physics regularization, explicit smoothness, implicit neural
bias, or a combination. This experiment does not separate those mechanisms.

The correct conclusion is:

> Under this same-model synthetic setup, declared noise, five seed blocks, and
> broad predeclared gate, the implemented inverse PINN is repeatable while the
> unregularized conventional endpoint estimates are noise-sensitive.

It is not correct to conclude that PINNs generally beat conventional inverse
methods.

## Important limits

- Five trials are too few to estimate a population failure probability.
- Synthetic truth and inference use the same finite-volume equations and
  three-knot property basis: inverse crime remains.
- Sensor errors are independent Gaussian draws; bias, lag, missingness, and
  correlated calibration error are absent.
- Boundary conditions and all non-resistivity properties are known exactly.
- The conventional and PINN estimators do not use matched priors or
  regularization.
- The follow-on complete-regime transfer study withholds one regime. A later
  independent-truth study changes the numerical grid, integrator, voltage
  quadrature, and property representation; broader transfer remains open.
- No hardware data are used.

## Reproduce

Install optional dependencies and run:

```bash
python3 -m pip install -e '.[all]'
thermotwin-distributed-robustness \
  --trials 5 \
  --epochs 600 \
  --first-seed 27001
```

Equivalent module command:

```bash
python3 -m thermotwin.distributed_inverse_robustness \
  --trials 5 \
  --epochs 600 \
  --first-seed 27001
```

The generated figure is written to
`thermotwin/figures/distributed_inverse_robustness.png` and is ignored by Git.

## Code ownership

| Responsibility | Module |
| --- | --- |
| Trial configuration, seed blocks, failure gate, fitting, summary | `thermotwin.studies.distributed_inverse_robustness` |
| Text report, two-panel figure, command line | `thermotwin.reports.distributed_inverse_robustness` |
| Historical module entry point | `thermotwin.distributed_inverse_robustness` |

## Next scientific step

The complete-regime transfer step is implemented in
[`DISTRIBUTED_WITHHELD_VALIDATION.md`](DISTRIBUTED_WITHHELD_VALIDATION.md), and
the independent-numerics/matched-curvature step is implemented in
[`DISTRIBUTED_INDEPENDENT_VALIDATION.md`](DISTRIBUTED_INDEPENDENT_VALIDATION.md).
The deliberate observation-removal step is implemented in
[`DISTRIBUTED_OBSERVATION_IDENTIFIABILITY.md`](DISTRIBUTED_OBSERVATION_IDENTIFIABILITY.md).
Next, expand the paired trial count and compare richer priors, multistart
profiles, and nonlinear interval coverage.
