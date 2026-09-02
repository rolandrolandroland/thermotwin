# Inverse PINN under imperfect observations

## Question

Can a switched-current inverse physics-informed neural network recover a hidden
cold-side thermal contact resistance when the visible temperature records are
noisy or structurally incomplete, and will that recovered resistance predict
complete current regimes that were excluded from fitting?

This study finishes the current one-parameter scope of Roadmap Milestone 4. It
also retains an intentionally misspecified sensor-bias case to show why a
falling training loss is not a sufficient recovery criterion.

## Physical model

The four-node contact model has cold-face, hot-face, cold-exchanger, and
hot-exchanger temperatures. The thermoelectric module supplies face heat rates
`Q_c` and `Q_h`. Thermal contacts connect the faces to their exchangers. For the
cold pair, the balances have the form

```text
C_cf dT_cf/dt = contact heat entering the cold face - Q_c
C_cx dT_cx/dt = reservoir/external heat - contact heat leaving the exchanger
contact heat = (T_cx - T_cf) / R_c
```

The hidden quantity is the positive cold-contact resistance `R_c`; its
synthetic truth is 0.25 K/W. All other device parameters are fixed at the
documented contact-reference values.

## Training, validation, and test experiments

- Training: 0 A until 5 s, 1 A from 5 to 20 s, then 0 A.
- Validation: 0.6 A from 10 to 30 s, otherwise 0 A.
- Test: 1 A from 5 to 20 s and -1 A from 35 to 50 s.
- Training observations are sampled every 1 s.
- Validation and test regimes are withheld in their entirety.

The PINN receives the known time-dependent current, the sparse cold-face and
cold-exchanger temperature rows that remain after transformation, and the
four energy-balance residuals. It represents continuous hidden trajectories and
one positive shared `R_c`. Exact segment continuity prevents a current switch
from creating a nonphysical temperature jump.

The conventional comparison uses scalar bounded optimization. It and the PINN
receive exactly the same transformed observation rows.

## Frozen observation cases

| Case | Noise | Missing rows | Bias term in data | Recovery expectation |
| --- | ---: | --- | ---: | --- |
| Gaussian noise | 0.02 K | none | 0 K | expected |
| Turn-off missingness | none | both cold sensors within 2 s of turn-off | 0 K | expected |
| Noise plus turn-off missingness | 0.02 K | same structured gap | 0 K | expected |
| Unmodeled cold-face bias | 0.02 K | none | +0.10 K | not assumed |

The missingness is structurally important: both visible cold channels are
removed around the current turn-off, so the PINN cannot secretly use the dense
reference there. The bias case deliberately omits a bias parameter from the
inverse model.

## Predeclared gate

Every trial must satisfy all of these checks:

- at least 80% reduction in total training loss;
- absolute `R_c` error no greater than 0.03 K/W;
- all-sensor validation RMSE no greater than 0.05 K;
- all-sensor bipolar-test RMSE no greater than 0.05 K.

Three trials use independent observation and neural seeds and start from
0.15, 0.50, and 0.80 K/W. No failed trial is discarded.

## Results

| Case | PINN recovery passes | PINN parameter RMSE | Conventional parameter RMSE | Mean validation RMSE | Mean bipolar-test RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gaussian noise | 3/3 | 0.007554 K/W | 0.001936 K/W | 0.003895 K | 0.006437 K |
| Turn-off missingness | 3/3 | 0.002947 K/W | effectively zero | 0.001815 K | 0.003009 K |
| Noise plus missingness | 3/3 | 0.004133 K/W | 0.002044 K/W | 0.002188 K | 0.003624 K |
| Unmodeled cold-face bias | 2/3 | 0.035184 K/W | 0.041556 K/W | 0.021052 K | 0.035594 K |

All nine expected-recovery trials pass. Their inferred parameters also transfer
through the trusted conventional solver to the two complete withheld regimes.

The conventional estimator is more accurate in the cleanly specified small
inverse problem. That is the expected result: a scalar optimizer is the right
accuracy reference when only one parameter is unknown.

The bias case is the caution. All three runs reduce training loss by about
99.99%, yet one fails parameter and bipolar-transfer limits and the aggregate
parameter RMSE exceeds the declared parameter tolerance. Two happen to remain
inside the broad all-criteria gate, so this frozen gate does not reliably
diagnose every instance of model mismatch. Optimization progress alone is
therefore not evidence that the physical parameter is correct.

## What the experiment establishes

- The inverse PINN can use incomplete switched-current data without filling in
  missing targets from the numerical truth.
- Recovery is repeatable across three widely separated initial guesses for the
  selected noisy and missing-data cases.
- A recovered parameter can be tested through whole-regime transfer rather
  than only by fitting the training rows.
- The conventional estimator remains better for this one-dimensional search.
- An unmodeled calibration bias can be absorbed into the physical parameter,
  even while loss decreases and predictions appear reasonable.

## What it does not establish

- It is not hardware validation.
- Three trials per case do not estimate a population failure rate.
- The training data and inference model share the same four-node equations.
- Sensor lag and bias are not jointly learned by this PINN.
- It does not show that a PINN is superior to conventional optimization.

Joint contact resistance, face capacitance, lag, and nuisance-bias inference is
tested conventionally in
[`NONLINEAR_EXPERIMENT_SELECTION.md`](NONLINEAR_EXPERIMENT_SELECTION.md).

## Reproduce

```bash
thermotwin-imperfect-inverse-pinn
```

or

```bash
python3 -m thermotwin.imperfect_inverse_pinn
```

The report writes a PNG plus same-stem JSON and TXT sidecars under
`thermotwin/figures/IMPERFECT_INVERSE_PINN/`.
