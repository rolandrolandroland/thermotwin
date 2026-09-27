# What physics-informed learning adds: two controlled studies

This four-panel figure connects the matched reconstruction study with the
later distributed-property comparison. The rows are different experiments and
have different trial counts; they are not one model trained and tested across
all four panels.

## Study 1: recover temperatures from incomplete measurements

The upper row uses `FORWARD_RECONSTRUCTION_COMPARISON.md`: five paired trials
with identical neural architectures, initial weights, and 56 noisy retained
sensor readings. Only one network receives the energy-balance equations.
Both receive exact initial temperatures and known current-switch times.

1. **Reconstruct the missing interval.** The gray band marks missing readings
   from both heat-exchanger sensors between 17 and 23 seconds. The trace shows
   the cold exchanger in the fixed representative trial. The annotation pools
   both exchangers across all five trials: mean gap RMSE falls from 0.07988 K
   for data-only to 0.00970 K with physics, an 87.86% reduction.
2. **Recover temperatures without direct labels.** The trace shows the cold
   module face. Neither module face supplies temperature labels after the
   initial state. Across both faces and all five trials, mean RMSE is
   0.00710 K with physics versus 2.19372 K for data-only. Temperature changes
   and errors are plotted in degrees Celsius; their numerical differences are
   identical to kelvin differences.

The data-only model has no hidden-state labels, regularizer, or learned prior.
The comparison demonstrates what the equations contribute in this setup; it
does not establish superiority over every alternative estimator. Physics-
informed training takes approximately 3.8 times longer in this benchmark.

## Study 2: infer a material property, then predict a new test

The lower row uses `DISTRIBUTED_PROFILE_COVERAGE.md`, which strengthens the
earlier independent-validation baseline with multiple conventional starting
guesses and shrinkage-plus-curvature priors. Both regularized methods receive
the same explicit prior weights. Only the ten trials that include both
estimators are compared; the additional ten conventional-only trials are
excluded from the plotted means.

Truth uses a separate 25-node spatial model, SSPRK3 time integration, and a
smooth cubic resistivity law. Inference uses the established finite-volume
model and three property knots. All other material properties and boundary
conditions are known. The methods see the same noisy temperatures and voltage
under zero, positive, and negative current.

3. **Estimate electrical resistivity over temperature.** Each connected pair
   shows errors in one shared trial. The property is the material's resistance
   to electrical current, represented over 285–315 K. Mean continuous-property
   relative RMSE is 4.6639% for the conventional fit and 1.7537% for the inverse
   PINN; PINN has lower error in 9/10 paired trials. Black horizontal marks
   indicate arithmetic means. The same ten paired trials supply both means; the
   separate conventional mean across all 20 conventional trials is 5.0747%.
   The curve metric covers 285–315 K, while fitted trajectories span only about
   294.83–305.21 K. Roughly two-thirds of the evaluation interval is therefore
   outside the training-temperature range. This does not assign two-thirds of
   the error to extrapolation or establish unique recovery of the full curve.
4. **Predict an operating condition excluded from fitting.** Each inferred
   curve is frozen and inserted into the same conventional forward solver for
   a withheld +0.4 A, 20 K-lift test. Mean terminal-voltage RMSE is 11.0188 µV
   for the conventional estimates versus 8.6405 µV for the PINN estimates.
   PINN has lower voltage error in only 5/10 individual trials. The lower
   average is therefore not a consistent per-trial win.

These ten-pair numbers differ from both the older three-trial independent
benchmark and the later report's twenty-trial conventional summaries because
the comparison here holds the set of noisy datasets fixed across methods.

## What the figure supports

Physics constraints improve reconstruction when measurements are incomplete.
The inverse PINN produces lower average property and held-out voltage errors
than the tested stronger conventional baseline in these paired synthetic
cases. Neither row demonstrates a forward-solver speedup or hardware accuracy.

The independent truth still shares the continuum equations. Matched explicit
priors do not match implicit neural regularization. The distributed PINNs use
short 400-epoch runs; small property error is not proof of full PDE convergence
or unique temperature-dependent curve-shape recovery. The separate
`DISTRIBUTED_PINN_TRAINING_AUDIT.md` explains why those claims must be checked
independently. Its three-trial adequacy summary includes development trial 0,
which selected physics weight 10, and two post-freeze trials; the ten-paired
inverse comparison instead used physics weight 1. The figure makes no PINN
uncertainty-calibration claim.

## Reproduce from saved evidence

```bash
python -m thermotwin.reports.pinn_evidence_summary
```

The command reads existing reconstruction and profile-coverage JSON sidecars
and performs no training or simulations. It writes `pinn_evidence_summary.png`,
plotted data with source hashes as JSON, and a TXT explanation under
`thermotwin/figures/PINN_EVIDENCE_SUMMARY/`.

The original figures are preserved. Optional `--reconstruction`, `--inverse`,
and `--output` arguments select alternate files.
