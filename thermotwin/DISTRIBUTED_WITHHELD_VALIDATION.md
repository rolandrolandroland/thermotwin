# Complete-regime distributed transfer validation

## The question

Can an inverse model recover a temperature-dependent electrical-resistivity
curve from noisy observations in three operating regimes and then predict a
fourth, complete operating regime that was never shown during fitting?

This is a stronger test than evaluating recovery on the same experiments used
for training. The inferred curve is frozen before the held-out trajectory is
simulated. The hidden internal temperatures are not used as observations and
are scored only after prediction.

The experiment is still synthetic. The truth and prediction use the same
distributed equations, finite-volume grid, time integrator, and three-knot
property basis. Passing therefore demonstrates transfer across an operating
regime within the declared model, not extrapolation to a new material law,
independent discretization, or hardware.

## Physical model and data split

The model is one homogeneous thermoelectric leg resolved along the cold-to-hot
coordinate. The leg has temperature-dependent Seebeck coefficient
`alpha(T)`, electrical resistivity `rho_e(T)`, and thermal conductivity
`kappa(T)`, with dynamic thermal nodes at both faces. A conservative finite-
volume reference simulator supplies the synthetic truth.

The unknown is only the three-knot multiplier curve for `rho_e(T)`:

```text
truth multipliers = (1.040000, 1.070000, 1.030000)
```

The four frozen constant-current experiments are:

```text
zero_current_20K_relaxation
positive_0.8A_10K_lift
negative_0.8A_10K_lift
positive_0.4A_20K_lift
```

The last experiment, `positive_0.4A_20K_lift`, is withheld in its entirety.
Each fit sees the other three regimes, sampled every 0.08 s with independent
Gaussian noise of 0.01 K on temperatures and 10 microvolts on voltage. The
held-out scoring trajectory is noise-free synthetic truth. Face temperatures
and terminal voltage are visible during fitting; internal cell temperatures
remain hidden.

## Two estimators

The conventional estimator uses bounded continuous optimization of the three
log-multipliers followed by a damped Gauss--Newton polish. It is intentionally
unregularized in this comparison.

The inverse PINN trains one hidden temperature network for each observed
constant-current regime and one shared positive resistivity curve. It combines
the distributed PDE residual, dynamic face-node residuals, sparse observation
losses, and a small smoothness penalty. After training, the shared curve is
inserted into the conventional reference solver for the withheld prediction;
the PINN does not refit on the withheld data.

## Predeclared prediction gate

Every trial must satisfy all six limits. Property-recovery error is reported,
but is deliberately not part of the transfer gate because the question is
whether the recovered model predicts the excluded operating regime.

| Metric | Limit |
| --- | ---: |
| Cold-face temperature RMSE | <= 0.030000 K |
| Hot-face temperature RMSE | <= 0.030000 K |
| Hidden internal-field RMSE | <= 0.030000 K |
| Terminal-voltage RMSE | <= 3.000000e-05 V |
| Maximum absolute temperature error | <= 0.080000 K |
| Maximum energy-balance residual | <= 1.000000e-10 W |

No trial is discarded. A trial fails if any metric is nonfinite or exceeds its
limit.

The energy residual is near roundoff by construction for trajectories produced
by the conservative solver. It is retained as a solver-integrity gate, not as
independent evidence that the inferred curve is accurate.

## Frozen five-trial run

The run used five neural seeds (`37001`, `37005`, `37009`, `37013`, `37017`)
and independent three-seed observation blocks. It used 600 inverse-PINN
epochs on CPU. The withheld regime and thresholds were fixed before running.

| Trial | Conventional | Inverse PINN | Conventional voltage RMSE | PINN voltage RMSE |
| ---: | :---: | :---: | ---: | ---: |
| 0 | FAIL | PASS | 4.066105e-05 V | 1.385876e-05 V |
| 1 | PASS | PASS | 1.959059e-05 V | 1.142947e-05 V |
| 2 | FAIL | PASS | 3.890826e-05 V | 1.411142e-05 V |
| 3 | FAIL | PASS | 4.235658e-05 V | 1.181689e-05 V |
| 4 | PASS | PASS | 2.972074e-05 V | 1.301147e-05 V |

The complete prediction counts are therefore:

```text
conventional: 2/5
inverse PINN: 5/5
```

The inverse-PINN mean prediction errors were 0.000008 K cold-face RMSE,
0.000011 K hot-face RMSE, 0.000070 K hidden internal-field RMSE,
1.284560e-05 V voltage RMSE, and 0.000117 K worst pointwise temperature
error. Its worst trial still remained below every prediction threshold. The
worst inverse-PINN property multiplier error was 0.037939, while the mean was
0.032305.

The conventional mean voltage RMSE was 3.424744e-05 V, above the fixed voltage
limit, even though its mean temperature errors were small. This is why its
three failures are voltage-gate failures, not trajectory blow-ups: the
unregularized fit sometimes shifts the resistivity curve enough to affect the
terminal voltage while barely changing the thermal trajectory.

The conventional fit reaches its upper log-multiplier search bound in trial 2.
This diagnostic is reported but was not added retroactively to the prediction
gate; trial 2 already fails the fixed voltage criterion.

The transferred inverse-PINN curves were:

```text
trial 0: (1.067555, 1.064396, 1.058736)
trial 1: (1.062935, 1.063383, 1.062237)
trial 2: (1.063003, 1.064498, 1.063751)
trial 3: (1.065993, 1.063622, 1.058863)
trial 4: (1.054774, 1.064938, 1.067939)
```

These are not exact truth recovery: all are smooth compromises around the
three-knot truth. The important result for this experiment is that those
curves transferred successfully to the excluded regime under the declared
gate.

That statement is deliberately narrower than property recovery. Only terminal
voltage binds in this frozen campaign. The worst cold- and hot-face RMSE values
remain about 385--390 times below their limits, the worst hidden-field RMSE is
about 57 times below its limit, and the energy residual is near roundoff by
construction. Consequently, the gate distinguishes effective curves that
predict this voltage regime; it does not strongly discriminate their thermal
fields or certify the recovered temperature dependence.

## Reproduce it

From the repository root, after installing the optional PINN dependencies:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/mpl-cache \
XDG_CACHE_HOME=/private/tmp/xdg-cache \
python3 -m thermotwin.distributed_withheld_validation \
  --trials 5 --epochs 600 --first-seed 37001
```

The report writes
`thermotwin/figures/DISTRIBUTED_WITHHELD_VALIDATION/distributed_withheld_validation.png`
by default, with plotted data beside it as JSON. Generated artifacts
are reproducible generated artifacts and are ignored by Git. The installed
entry point is:

```bash
thermotwin-distributed-withheld --trials 5 --epochs 600 --first-seed 37001
```

The implementation is split between
`thermotwin.studies.distributed_withheld_validation` and
`thermotwin.reports.distributed_withheld_validation`. The compatibility facade
is `thermotwin.distributed_withheld_validation`.

## Interpretation

This result supports three limited conclusions:

1. A shared inverse-PINN resistivity curve can be trained repeatedly from
   noisy sparse face/voltage data in this synthetic setup.
2. The transferred curves predict a complete unseen current/lift regime more
   reliably than the unregularized conventional fits under the fixed voltage
   gate; this is predictive equivalence, not proof of property uniqueness.
3. Hidden internal temperatures can be checked after the prediction even
   though they were never supplied as training labels.

It does **not** establish that PINNs generally outperform conventional
optimizers. The comparison has unequal regularization, uses the same model to
generate and fit truth, and tests only one held-out regime and one property
family. The conventional estimator's temperature predictions are also good;
its failures arise from the stricter voltage criterion. Independent truth
generation, matched priors, more repetitions, nonlinear uncertainty coverage,
and hardware data are still required for a stronger claim.

The numerical/property-basis part of that next step is now implemented in
[`DISTRIBUTED_INDEPENDENT_VALIDATION.md`](DISTRIBUTED_INDEPENDENT_VALIDATION.md).
It uses nodal/SSPRK3 cubic truth and applies the same explicit curvature term
to paired conventional and PINN variants. It remains synthetic, uses only
three trials, and does not match implicit neural regularization.

The later
[`DISTRIBUTED_PINN_TRAINING_AUDIT.md`](DISTRIBUTED_PINN_TRAINING_AUDIT.md)
also shows that the 600-epoch curves have not met a truth-blind PDE-residual
gate and often under-recover curve amplitude. That audit supersedes any reading
of this holdout as an independent property-recovery certificate.
