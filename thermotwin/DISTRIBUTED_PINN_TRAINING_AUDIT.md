# Distributed inverse-PINN training and curve-shape audit

## Question

Does a small error at three resistivity knots mean that the inverse PINN has
learned a physics-satisfying temperature-dependent curve, or can an
undertrained, nearly constant curve pass the earlier recovery check?

This study was added after an audit found that the frozen 600-epoch inverse
PINN still had a large physics residual and that its broad per-knot error gate
mostly tested average property level. The study does not hide that problem. It
separates four questions that the earlier headline combined:

1. Does the model fit the noisy terminal observations?
2. How closely does the learned field satisfy the PDE and face balances?
3. Is the average resistivity level correct?
4. Is the temperature dependence of `rho_e(T)` actually recovered?

## Synthetic problem

The benchmark uses the established four constant-current/lift regimes and
three-knot electrical-resistivity basis at 285, 300, and 315 K. The hidden
truth multipliers are

```text
(1.04, 1.07, 1.03).
```

Only cold-face temperature, hot-face temperature, and terminal voltage are
visible. Temperature noise has standard deviation 0.01 K and voltage noise has
standard deviation 10 microvolts. Each trial receives one unique neural seed
and four unique observation-noise seeds.

Truth and inference use the same finite-volume equations and three-knot
property representation. That same-model setup is intentional here: this is
an optimizer/training audit, not the independent-truth validation. The
independent nodal/SSPRK3/cubic benchmark remains documented in
[`DISTRIBUTED_INDEPENDENT_VALIDATION.md`](DISTRIBUTED_INDEPENDENT_VALIDATION.md).

## One training trajectory, three budgets

Three neural/noise seed blocks are trained on CPU. Each seed is trained once
for 2,400 Adam epochs. Metrics are retained at 600, 1,200, and 2,400 epochs
from that same uninterrupted deterministic trajectory.

This matters for both efficiency and scientific control:

- the 600-epoch checkpoint is exactly the state reached before the next 1,800
  updates, rather than a separately initialized run;
- no budget is selected after inspecting property truth; and
- one 2,400-epoch run costs less than three independent 600/1,200/2,400 runs.

## Comparable observation loss

The conventional estimator minimizes normalized observation error. The PINN
objective is different:

```text
total = physics_weight * physics + observation
        + smoothness_weight * smoothness
        + shrinkage_weight * shrinkage.
```

The earlier robustness gate applied the same final-loss limit to the
conventional observation loss and the PINN total objective. Those are not the
same quantity. The code now:

- applies the comparable gate to normalized observation loss for both methods;
- retains and reports the PINN total objective for history;
- reports the PINN physics loss separately; and
- prints failure reasons for both methods.

## Physics residual in physical units

For each experiment the inverse PINN physics loss is the sum of three
mean-square normalized residual families:

```text
interior PDE + cold-face balance + hot-face balance.
```

With `residual_rate_scale = 1 K/s`, the combined RMS residual is therefore

```text
RMS physics residual = sqrt(physics_loss / 3) K/s.
```

The released nominal finite-volume trajectories have an RMS temperature rate
of

```text
0.891389 K/s.
```

This nominal denominator is computed without the hidden resistivity truth, so
it can be used in a truth-blind checkpoint rule. Reporting the ratio makes a
value such as 0.45 K/s interpretable: it is not “small” merely because an
optimizer loss decreased.

## Truth-blind operational gate

A checkpoint is called operationally acceptable only if both conditions pass:

| Quantity | Limit |
| --- | ---: |
| Normalized observation loss | at most 2.0 |
| Physics-residual RMS / nominal-rate RMS | at most 25% |

The threshold is a declared engineering diagnostic, not a theorem. It can be
applied without looking at the hidden resistivity truth. The benchmark does
not select the checkpoint with the smallest known property error.

## Truth-known curve-shape diagnostics

A three-knot curve can have the right mean while being almost flat. The study
therefore reports:

```text
amplitude = max(m) - min(m)
center contrast = m_300 - (m_285 + m_315) / 2
```

Each is divided by its truth value. A shape check passes only if:

- the amplitude ratio lies from 0.75 to 1.50;
- the center-contrast ratio lies from 0.75 to 1.50; and
- coefficient RMSE beats the best constant three-knot curve.

These truth-known quantities evaluate a synthetic benchmark. They are not
available as stopping rules on hardware.

## Loss-balancing correction

The original run weighted normalized physics and observation losses equally.
Because those terms have different optimization geometry, equal numerical
weights did not produce equally satisfactory constraints. Three fixed
physics-weight candidates were compared on trial 0 at the unchanged 2,400
epoch budget:

| Physics weight | Observation loss | Residual / nominal rate | Operational? |
| ---: | ---: | ---: | :---: |
| 1 | 0.957834 | 50.84% | no |
| 3 | 0.687007 | 34.62% | no |
| 10 | 0.737299 | 20.79% | yes |

Weight 10 was selected using only the truth-blind observation and physics
criteria. The curve-shape metrics were not used to choose it. The weight was
then frozen before trials 1 and 2 were evaluated. No architecture, seed,
observation, learning rate, epoch budget, or acceptance threshold changed.

## Frozen results

### Mean checkpoint behavior

| Epoch | Mean observation loss | Mean residual / nominal rate | Mean coefficient RMSE | Mean amplitude ratio | Mean contrast ratio | Operational | Shape |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| 600 | 1.049335 | 50.68% | 0.015296 | 0.379 | 0.411 | 0/3 | 0/3 |
| 1,200 | 1.323320 | 33.19% | 0.007754 | 0.742 | 0.832 | 0/3 | 2/3 |
| 2,400 | 0.820109 | 21.17% | 0.008661 | 0.965 | 1.043 | 3/3 | 2/3 |

The 600-epoch checkpoint fits the observations but recovers only 38% of the
truth's curve amplitude on average. It is not a demonstrated
temperature-dependent recovery.

Loss balancing and longer training improve the physics and mean shape
substantially. At 2,400 epochs all three trials pass the truth-blind
observation-and-physics gate, with mean residual ratio 21.17%. The mean
amplitude and center contrast are close to truth, but one seed still
under-recovers both. Operational convergence is therefore demonstrated for
the declared gate; repeated curve-shape recovery is not yet 3/3.

### Trial-level 2,400-epoch results

| Trial | Multipliers | Observation loss | Residual / nominal rate | Amplitude ratio | Contrast ratio | Shape |
| ---: | --- | ---: | ---: | ---: | ---: | :---: |
| 0 | `(1.031582, 1.071691, 1.034245)` | 0.737299 | 20.79% | 1.003 | 1.108 | pass |
| 1 | `(1.044689, 1.068206, 1.048896)` | 0.843627 | 20.46% | 0.588 | 0.612 | fail |
| 2 | `(1.027160, 1.073627, 1.021398)` | 0.879402 | 22.25% | 1.306 | 1.410 | pass |

The observation loss is not monotonic with epoch or tightly coupled to shape.
For example, trial 1 remains too flat at 2,400 epochs despite passing both
operational criteria. This is why physics and observation convergence are
necessary but not sufficient evidence of unique function recovery.

## Correct conclusion

The audit changes the interpretation of the earlier five-seed result:

> The loss-balanced 2,400-epoch inverse PINN passes the declared observation
> and physics gate in 3/3 trials and recovers the temperature-dependent curve
> shape in 2/3. This demonstrates improved operational convergence, while one
> flat-curve result prevents a claim of fully repeatable function recovery.

The improvement comes from loss balancing, not a larger network or more
training epochs. Relative to weight 1 at the same 2,400-epoch budget, weight 10
reduces mean residual ratio from 49.35% to 21.17% while reducing mean
observation loss from 1.113088 to 0.820109. It does not remove the remaining
shape ambiguity.

## What this means for earlier transfer results

The complete-regime holdout remains a valid test of predictive equivalence
inside the declared same-model regime. It is not an independent certificate of
property recovery. In that study only the voltage limit distinguishes pass
from fail; the temperature and hidden-field limits are much looser, and the
energy residual is a conservative-solver integrity check by construction.

Thus the holdout result can support “this effective curve predicts that
operating regime,” but not “the true temperature-dependent resistivity was
identified.”

## Reproduce

Install the optional dependencies, then run:

```bash
python3 -m pip install -e '.[all]'
thermotwin-distributed-pinn-audit \
  --trials 3 \
  --first-seed 63001 \
  --physics-weight 10 \
  --checkpoints 600,1200,2400
```

Equivalent module command:

```bash
python3 -m thermotwin.distributed_pinn_training_audit \
  --trials 3 \
  --first-seed 63001 \
  --physics-weight 10 \
  --checkpoints 600,1200,2400
```

The generated figure is written to
`thermotwin/figures/DISTRIBUTED_PINN_TRAINING_AUDIT/distributed_pinn_training_audit.png`,
with plotted data beside it as JSON; both are ignored by Git.

## Code ownership

| Responsibility | Module |
| --- | --- |
| Frozen config, physical-rate scale, checkpoint extraction, shape gates, summaries | `thermotwin.studies.distributed_pinn_training_audit` |
| Text report, four-panel figure, command line | `thermotwin.reports.distributed_pinn_training_audit` |
| Historical-style module entry point | `thermotwin.distributed_pinn_training_audit` |

## Next implementation step

Do not tune further on these three property truths. The next step is to keep
the loss-balanced protocol fixed and test whether the locally D-optimal
experiment improves complete nonlinear recovery over naive alternatives under
fresh independent truth/noise trials. The remaining flat-curve trial should be
treated as a target for better experimental information, not automatically as
permission to tune another optimizer weight on the same seeds.
