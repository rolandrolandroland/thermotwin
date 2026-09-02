# Physics-informed versus data-only transient reconstruction

## Problem

When only sparse, noisy heat-exchanger temperatures are visible and both
sensors stop reporting around a current transition, does the known
thermoelectric physics materially improve reconstruction of:

- the visible exchanger temperatures inside the missing interval;
- the two thermoelectric-face temperatures that are never observed; and
- a whole-system first-law energy balance?

This experiment completes the current scope of Roadmap Milestone 3. It is a
matched comparison, not a claim that a PINN is always better than a
data-fitting network.

## The matched comparison

Each paired trial creates two copies of the same switch-aware neural
architecture:

| Item | Physics-informed model | Data-only model |
| --- | --- | --- |
| Trainable parameters | 1,116 | 1,116 |
| Hidden width and layers | 16 and 2 | 16 and 2 |
| Time subnetworks | three | three |
| Initial weights | bit-identical within a trial | bit-identical within a trial |
| Retained observations | identical | identical |
| Exact initial temperatures | supplied | supplied |
| Known switch locations | supplied | supplied |
| Adam updates | 5,000 at 0.001 | 5,000 at 0.001 |
| Observation objective | yes | yes |
| Four node-balance residuals | yes | no |

Both networks have exact temperature continuity at 5 and 20 s by construction.
The physics-informed model has more computation per update because automatic
differentiation evaluates four time derivatives. Equal architecture and epoch
count are therefore not equal wall-clock budgets.

The final physics weight of 100 was selected with a separate development seed
(`70001`). The reported five-trial evaluation uses untouched seeds beginning at
`72001`.

## Visible data

The hidden RK4 truth uses the established four-node contact model and the
frozen training pulse:

```text
0 A from 0 to 5 s
1 A from 5 to 20 s
0 A from 20 to 60 s
```

Only the cold- and hot-exchanger temperatures are measured. The cold and hot
module-face temperatures never appear in the observation loss.

- sampling interval: 2 s;
- Gaussian temperature-noise standard deviation: 0.02 K;
- complete rows: 62;
- both sensors absent from 17 through 23 s, including turn-off;
- removed rows: 6;
- retained rows: 56.

The RK4 temperature histories remain hidden until evaluation.

## Physics-informed objective

The physics-informed network minimizes the same observation error as the
data-only model plus the four normalized node-balance residuals. In schematic
form,

```text
loss = observation loss + 100 * mean(node residual / 0.10 K/s)^2
```

The observation differences are scaled by 0.10 K. This scale sets optimizer
conditioning; it is not the claimed sensor uncertainty or an acceptance
threshold.

## Independent energy-closure diagnostic

After training, ThermoTwin computes an additional whole-system diagnostic that
was not added as a fifth loss equation. The total stored thermal energy is

```text
U = C_cf T_cf + C_hf T_hf + C_cx T_cx + C_hx T_hx.
```

Summing the four node balances cancels internal contact heat flow and uses the
module identity `Q_h - Q_c = electrical power`. Therefore,

```text
dU/dt = electrical power
       + G_c (T_c,reservoir - T_cx)
       + G_h (T_h,reservoir - T_hx)
       + q_c,external + q_h,external.
```

The rate-closure error is the automatic-differentiation value of `dU/dt` minus
the independently assembled right side. Cumulative net input is trapezoid
integrated separately within every constant-current segment. Both one-sided
values are retained at a switch, so discontinuous electrical power is never
replaced by a fictitious linear ramp.

This is an independent post-training calculation, but it is not an independent
physical law: algebraically it follows from the same four node balances used
during PINN training. Its value is that it checks the trained prediction in
power and energy units without adding a redundant training constraint.

## Documented regression gate

Every physics-informed trial must satisfy:

- missing-exchanger RMSE no greater than 0.02 K;
- completely hidden-face RMSE no greater than 0.02 K;
- four-node residual RMS no greater than 0.005 K/s;
- whole-system rate-closure RMS no greater than 15% of RMS net input;
- maximum absolute cumulative closure error no greater than 4 J.

Exact initial states and switch continuity are also verified separately.
These thresholds are maintained regression criteria for the implemented
synthetic case, not preregistered statistical acceptance thresholds.

## Results

Five paired trials use independent observation and neural seeds. No trial is
discarded.

| Mean metric | Physics-informed | Data-only |
| --- | ---: | ---: |
| RMSE against retained noisy rows | 0.019433 K | **0.017471 K** |
| RMSE against noise-free truth at retained rows | **0.006793 K** | 0.013293 K |
| Missing-interval exchanger RMSE | **0.009696 K** | 0.079878 K |
| Completely hidden-face RMSE | **0.007105 K** | 2.193724 K |
| Hidden-face RMSE during missing interval | **0.007827 K** | 2.920145 K |
| All-state RMSE | **0.007202 K** | 1.551353 K |
| Four-node residual RMS | **0.002655 K/s** | 0.193915 K/s |
| Whole-system rate-closure RMS | **0.132833 W** | 16.493587 W |
| Rate-closure RMS / net-input RMS | **10.964%** | 1436.118% |
| Absolute final cumulative closure error | **1.952169 J** | 370.392719 J |
| Indicative training time on the evaluation machine | about 15.1 s | **about 3.9 s** |

The data-only network fits the noisy retained rows slightly better and trains
about 3.8 times faster. That result is important: the physics term does not win
by minimizing the only data it sees.

The physics-informed model instead improves the quantities not fixed by those
rows:

- missing-exchanger RMSE falls by 87.86%;
- hidden-face RMSE falls by 99.68%;
- hidden-face RMSE in the missing interval falls by 99.73%;
- whole-system rate-closure error falls by 99.19%.

All three primary advantages—missing exchanger, hidden faces, and energy
closure—hold in 5/5 trials. All 5/5 physics-informed trials also pass the
documented regression gate. Both models preserve switch continuity exactly.

## Interpretation

This is the current clearest lumped-model demonstration of what the physics
constraint adds. Sparse data alone can fit the visible points, but many hidden
face histories are compatible with those observations. The four energy
balances connect the unobserved faces to the exchangers, current, contacts,
reservoirs, and thermal storage, selecting a physically consistent trajectory.

The comparison also shows the cost: residual derivatives make training slower,
and the physics-informed network gives up a small amount of noisy-row fit to
obtain much better noise-free, missing-interval, hidden-state, and energy
behavior.

## Limitations

- Truth and the physics-informed model use the same four-node equations.
- This is synthetic validation, not hardware validation.
- The data-only baseline has no face labels and no regularizer or learned prior
  that could identify those states; its failure is not a claim about all
  observation-only methods.
- Both models receive true initial temperatures and known current-switch times.
- Equal epoch count does not equal equal runtime or floating-point work.
- Five trials expose seed sensitivity but do not estimate a general failure
  rate.
- The energy diagnostic is an independent calculation, not an independent law.

## Reproduce

```bash
thermotwin-forward-reconstruction
```

or

```bash
python3 -m thermotwin.forward_reconstruction_comparison
```

The command writes a figure, structured JSON data, and a plain-text explanation
under `thermotwin/figures/FORWARD_RECONSTRUCTION_COMPARISON/`.
