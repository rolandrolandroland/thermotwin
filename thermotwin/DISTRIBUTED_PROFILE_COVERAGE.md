# Nonlinear profiles and repeated interval coverage for distributed resistivity

## 1. Question

When ThermoTwin reports a temperature-dependent electrical resistivity curve,
how wide is the range of curves supported by the observations, and how often do
nominal uncertainty intervals contain the synthetic truth?

This is the uncertainty follow-on to the
[observation-sufficiency gate](DISTRIBUTED_OBSERVATION_IDENTIFIABILITY.md). The
earlier gate asks whether the local sensitivity matrix contains three usable
directions before fitting. This study asks a harder question after fitting:
whether nonlinear estimates and their intervals behave reliably across new
noise realizations and independent numerical truth.

## 2. Why a best fit is not enough

A nonlinear optimizer can return one smooth, plausible curve even when many
nearby curves explain the observations almost as well. A local full-rank
Jacobian also does not guarantee that:

- the nonlinear loss has only one relevant basin;
- a quadratic approximation is accurate far from the optimum;
- regularization has not narrowed or shifted the result; or
- a nominal 95% interval contains truth about 95% of the time.

This study therefore separates three claims:

1. **Observation support:** the pre-fit local rank gate.
2. **Nonlinear support:** representative coefficient profiles in which the
   other coefficients are re-optimized.
3. **Repeated calibration:** empirical interval coverage under independent
   noise realizations.

## 3. Inferred curve and independent truth

The inference model writes the resistivity as a three-knot piecewise-linear
curve whose knot values are

$$
\rho_e(T_i;\boldsymbol\theta)
=\rho_{e,0}(T_i)\exp(\theta_i),
$$

with linear interpolation between 285, 300, and 315 K. The bounded log multipliers satisfy
$-0.3\leq\theta_i\leq0.3$.

The synthetic truth is deliberately outside that fitted basis: it is a smooth
cubic resistivity law evaluated by a 25-node nodal spatial discretization and
third-order SSPRK time integration with a 0.00025 s step. The truth multipliers
at the three inference knots are `(1.04, 1.07, 1.03)`, corresponding to log
multipliers `(0.039221, 0.067659, 0.029559)`.

This removes exact grid, integrator, voltage-quadrature, and basis agreement.
It does not remove every inverse crime: truth and inference still share the
same one-dimensional continuum equations, known boundary parameters, and all
non-resistivity properties.

## 4. Observation and transfer design

Each fit sees the frozen zero-current, +0.8 A, and -0.8 A constant-current
experiments. The observations are cold-face temperature, hot-face temperature,
and terminal voltage sampled every 0.08 s. Independent Gaussian noise has
declared standard deviations:

- temperature: 0.01 K per face channel;
- voltage: 10 microvolts.

After fitting, each curve is frozen and transferred without refitting to an
excluded +0.4 A, 20 K-lift experiment. The independent truth solver supplies
the hidden internal temperature field and voltage for that holdout.

The repeated study uses collision-free seed blocks. Twenty noise trials are
used for both conventional estimators. The first ten trials also train paired
PINNs from the same visible observations and neural seed.

## 5. Conventional estimators

Every conventional point estimate is selected from three slope-aware nonlinear
starts:

- low-to-high endpoint slope: `(ln 0.8, 0, ln 1.2)`;
- flat baseline: `(0, 0, 0)`;
- high-to-low endpoint slope: `(ln 1.2, 0, ln 0.8)`.

Those starts are important because resistivity endpoints can trade against one
another while preserving a similar integrated voltage.

Two estimators are compared:

### Unregularized

The objective is the noise-normalized observation error alone.

### Shrinkage plus curvature

The regularized objective adds two explicit terms,

$$
L_{reg}=L_{data}
+0.8\,R_{curve}(\boldsymbol\theta)
+0.9\,R_{zero}(\boldsymbol\theta),
$$

where `R_curve` is mean squared second-difference roughness and `R_zero` is
mean squared log-coefficient magnitude. The same numerical weights are used in
the regularized conventional and PINN estimators.

The shrinkage term encodes a real prior preference for the baseline curve. It
is not information obtained from the sensors.

## 6. Representative nonlinear profiles

For coefficient $i$, the nonlinear profile fixes $\theta_i=c$ and re-optimizes
the other two coefficients:

$$
\Delta S_i(c)=
\min_{\theta_{j\ne i}} S(\boldsymbol\theta)-S(\hat{\boldsymbol\theta}).
$$

The report evaluates this on a bounded five-point grid plus the exact fitted
anchor. Each point is a complete transient refit, not a slice through the loss
with the other coefficients frozen. Three representative cases are shown:

1. full bidirectional data without regularization;
2. full bidirectional data with shrinkage and curvature;
3. the weaker positive-current temperature-plus-voltage observation set.

The reported interval is the connected supported region containing that exact
anchor. If a second disconnected low-score basin exists, its points remain
visible in the profile and must not be silently merged into one interval.

For the unregularized profile, the horizontal thresholds 1.0 and 3.841459 are
the one-parameter 68% and 95% likelihood-ratio thresholds under the declared
independent Gaussian noise model. The regularized curves use the same
thresholds as a diagnostic convention; they are not classical confidence
intervals because the score includes prior penalties.

The profile search is deliberately bounded for CPU use. It uses a nonlinear
multistart anchor and warm-started re-optimization at each fixed value, but it
is not a proof that every profiled point is the global minimum over all
possible basins. The figure uses a symmetric-log vertical scale with a linear
region around zero so the 1.0 and 3.841459 thresholds remain visible beside
large score increases.

## 7. Repeated local intervals

Running every full nonlinear profile for every noise realization would require
hundreds of complete transient optimizations. The repeated audit therefore
uses a local quadratic profile approximation around each newly fitted
multistart optimum.

For the unregularized estimator, the local precision is the
noise-normalized information matrix $J^T J$. For the regularized estimator,
the analytically assembled shrinkage and curvature precision terms are added.
The covariance approximation is the inverse of that total precision. Marginal
68% and 95% log-coefficient intervals are clipped to the declared bounds.

This division is intentional:

- representative plots test the nonlinear loss shape;
- repeated local intervals test calibration at a practical CPU budget.

The repeated intervals must not be described as fully profiled intervals.

## 8. Frozen result

The corrected 20-trial study gives the following coefficient-level coverage:

| Conventional estimator | Empirical 68% coverage | Empirical 95% coverage |
| --- | ---: | ---: |
| Unregularized | 38/60 = 63.3% | 59/60 = 98.3% |
| Shrinkage + curvature | 47/60 = 78.3% | 60/60 = 100% |

The point-estimate and transfer summaries are:

| Estimator | Completed trials | Mean continuous-property relative RMSE | Mean per-trial maximum relative error | Mean holdout-voltage RMSE |
| --- | ---: | ---: | ---: | ---: |
| Conventional, unregularized | 20 | 8.4649% | 18.3888% | 28 microvolts |
| Conventional, shrinkage + curvature | 20 | 5.0747% | 10.5182% | 11 microvolts |
| PINN, unregularized | 10 | 1.7550% | 3.6619% | 9 microvolts |
| PINN, shrinkage + curvature | 10 | 1.7537% | 3.6589% | 9 microvolts |

The lower PINN point error is a result for these implemented estimators, not an
interval-calibration or general superiority result. The PINNs use a hidden
physics-constrained field and implicit neural bias, and only half as many trials
were run.

At 95%, all three coefficients are covered simultaneously in 19/20
unregularized conventional trials and 20/20 regularized trials. Only 35.0% of
unregularized and 63.3% of regularized individual 95% intervals avoid both
declared coefficient bounds; the mean log widths are 0.321579 and 0.284399.
Bound contact is part of the result, not a numerical interval failure.

There are 60 coefficient checks per estimator: three coefficients in each of
20 trials. With this small denominator, a difference of one covered coefficient
changes the reported fraction by 1.67 percentage points. The observed values
are therefore calibration diagnostics with wide binomial uncertainty, not
precise estimates of long-run coverage.

The main conclusion is not that one estimator has a universally superior
interval. The regularized estimator shifts the 68% behavior and narrows the
space of admissible curves using information supplied by the prior. At 95%,
both variants cover nearly all individual coefficients in this frozen study.
The full report also records simultaneous coverage, bound contact, interval
width, continuous property error, and holdout voltage and hidden-field errors.

## 9. PINN comparison

The paired PINNs release the same three resistivity coefficients and train one
hidden temperature network per constant-current experiment. Both variants see
the same noisy observations as their conventional counterparts; the
regularized variant receives the same explicit shrinkage and curvature weights.
The conventional profiles are explicitly bounded in log space, while the PINN
uses its established unconstrained log-multiplier parameterization. This is an
estimator difference, although it is inactive in this frozen run: all final
PINN log multipliers lie between 0.0556 and 0.0723, inside the conventional
`[-0.3, 0.3]` bounds.

PINNs are reported as point estimators only in this experiment. No uncertainty
interval is manufactured from neural-seed spread, and no PINN coverage claim is
made. A future PINN uncertainty study would need an explicit posterior or
calibrated ensemble construction and its own repeated coverage check.

This comparison also does not isolate every source of regularization. Even
after matching the visible coefficient penalties, the neural field
representation and optimization dynamics retain implicit biases that the
conventional estimator does not share.

## 10. What the study establishes

- A complete nonlinear profile implementation can fix one property coefficient
  while fitting the remaining curve.
- Every reported profile contains its own exact reported optimization anchor;
  this is protected by a regression test after a lower-basin anchoring defect
  was found during the audit.
- The repeated conventional study uses a new multistart optimum for every noise
  realization rather than perturbing one frozen fit.
- Explicit shrinkage and curvature are represented consistently in fitting and
  local precision assembly.
- Nominal interval labels are checked against independent synthetic truth rather
  than accepted from a Hessian alone.
- PINN point accuracy and interval calibration are kept as separate claims.

## 11. What it does not establish

- Twenty conventional and ten PINN trials do not provide precise failure-rate
  or coverage estimates.
- Repeated intervals are local quadratic approximations, not complete nonlinear
  profiles at every trial.
- Penalized intervals are diagnostics, not classical frequentist confidence
  intervals.
- The property law, noise, and boundary conditions remain synthetic.
- Noise is independent Gaussian noise without bias, lag, missingness, or
  correlated calibration error.
- Only `rho_e(T)` is inferred; `alpha(T)` and `kappa(T)` are held correctly.
- Conventional log coefficients are bounded; PINN log coefficients are not.
- The one-dimensional continuum model omits lateral heat flow, spatial defects,
  and internal material interfaces.
- No result is a hardware uncertainty claim.

## 12. Reproduce

Install all optional dependencies, then run the frozen campaign:

```bash
python3 -m pip install -e '.[all]'
thermotwin-distributed-profile-coverage \
  --trials 20 \
  --pinn-trials 10 \
  --epochs 400 \
  --profile-points 5 \
  --report-output thermotwin/figures/DISTRIBUTED_PROFILE_COVERAGE/distributed_profile_coverage.txt
```

The equivalent module command is:

```bash
python3 -m thermotwin.distributed_profile_coverage
```

The generated figure is written to
`thermotwin/figures/DISTRIBUTED_PROFILE_COVERAGE/distributed_profile_coverage.png`,
with plotted data in the colocated JSON sidecar; generated artifacts are ignored by Git.
The optional text path preserves the full trial-level report outside a terminal
scrollback. Use `--skip-profiles` when repeating only the coverage portion.

## 13. Code map

| Responsibility | Module |
| --- | --- |
| Fixed-coefficient fitting and multistart nonlinear profiles | `thermotwin.inference.distributed_profile_likelihood` |
| Bounded conventional curve fitting | `thermotwin.inference.distributed_properties` |
| Shared shrinkage and curvature terms | `thermotwin.inference.distributed_regularization` |
| Independent nodal/SSPRK3/cubic truth | `thermotwin.simulation.distributed_independent` |
| Shared-property inverse PINNs | `thermotwin.pinn.distributed_inverse` |
| Frozen trial design and coverage summaries | `thermotwin.studies.distributed_profile_coverage` |
| Text report, figure, and command line | `thermotwin.reports.distributed_profile_coverage` |

## 14. Next scientific step

The next high-value step is to validate experiment selection with complete
nonlinear refits: compare the locally D-optimal pulse/lift candidate against
naive alternatives over repeated independent-truth trials. That tests whether
the current local information criterion improves actual recovered curves and
holdout predictions, rather than merely improving a matrix at the nominal
parameter point.
