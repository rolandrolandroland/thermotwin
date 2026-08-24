# Sparse accessible-sensor inference experiment

## Question

Can ThermoTwin recover an internal interface loss and reconstruct inaccessible
thermoelectric-face temperatures when it receives only the temperatures that
could plausibly be measured on the two heat exchangers?

This is a synthetic identifiability experiment. The conventional four-node
model generates the hidden truth and also supplies candidate predictions. It
does not claim that the assumed noise, lag, or interface model has been
calibrated to hardware.

## What is visible and hidden

Only these observations are supplied to the estimator:

- cold heat-exchanger temperature;
- hot heat-exchanger temperature;
- commanded current; and
- observation time.

The cold and hot thermoelectric-face temperatures remain hidden until final
validation. The estimator jointly releases four quantities:

- cold contact resistance $R_{c,\mathrm{contact}}$;
- one shared exchanger-sensor time constant $\tau_s$;
- the cold-sensor calibration offset $b_c$; and
- the hot-sensor calibration offset $b_h$.

Every other thermal and electrical parameter remains fixed at its synthetic
truth. This is therefore a controlled four-unknown problem, not a general
system-identification claim.

## Excitation and measurement process

The 80 s training current is

```text
0 A, 0--5 s
1.00 A, 5--20 s
0 A, 20--35 s
0.55 A, 35--50 s
0 A, 50--80 s
```

The hidden truth uses $R_{c,\mathrm{contact}}=0.25$ K/W. Both accessible
sensors use a 1.5 s first-order lag. Their fixed biases are +0.08 K on the
cold sensor and -0.04 K on the hot sensor. Independent 0.02 K Gaussian noise
is added with seed 2026. Samples are reported every 1 s.

Seven cold-exchanger readings from 18 through 24 s are removed. The hot sensor
continues reporting, so missingness is represented as unavailable records—not
zero temperatures or imputed values. The final dataset contains 155
temperature records.

## Estimation method

For each candidate resistance and lag, the four-node solver predicts both
exchanger-temperature histories. The best constant bias for sensor $s$ has an
analytic least-squares solution,

$$
b_s = \frac{1}{N_s}\sum_i\left(y_{s,i}-T_{s,i}^{\mathrm{model}}\right).
$$

Profiling the biases this way leaves a two-dimensional coarse-to-fine search
over resistance and lag. A bounded local pattern-search polish starts from the
best grid point and reduces its step sizes when none of the eight neighboring
points improves the loss. This final step prevents a hidden truth that happens
to lie on a grid node from producing an artificially exact-looking result.
The polish uses only the observation loss; it never sees the hidden truth.
Missing readings simply do not enter the sum.

After fitting, finite-difference sensitivities form a local information
matrix. Its inverse gives linearized standard errors, parameter correlations,
and approximate 95% intervals. These intervals assume the four-node model is
correct and the 0.02 K noise scale is known.

## Result

| Quantity | Hidden truth | Estimate | Local 95% interval |
| --- | ---: | ---: | ---: |
| Cold contact resistance | 0.25000 K/W | 0.25103 K/W | 0.23967--0.26238 K/W |
| Shared sensor lag | 1.5000 s | 1.5158 s | 1.33589--1.69563 s |
| Cold-sensor bias | +0.0800 K | +0.0792 K | +0.07208--+0.08629 K |
| Hot-sensor bias | -0.0400 K | -0.0421 K | -0.04652 to -0.03762 K |

All four synthetic truths fall inside their reported intervals. The training
observation RMSE is 0.02151 K, close to the imposed 0.02 K noise scale.

Resistance and lag have correlation -0.580. That is physically important:
both a slower sensor and a different interface resistance can change the
apparent transient response, so treating lag as known when it is not can bias
the inferred contact.

The polished estimate is 0.41% above the hidden resistance instead of being
locked exactly to its grid node. Consequently, the reconstructed training
face histories have small but nonzero RMSE: 0.00156 K on the cold face and
0.00017 K on the hot face. These are not face-temperature measurements—the
histories are consequences of the fitted physical model.

## Withheld-current validation

The fitted resistance, lag, and biases are transferred without refitting to a
different schedule containing +0.75 A and -0.45 A intervals. Against noiseless
synthetic truth, the two accessible sensor histories have 0.00187 K RMSE. The
hidden cold- and hot-face trajectories have 0.00112 K and 0.00010 K RMSE,
respectively. These nonzero errors are the honest consequence of estimating
the parameters from noisy sparse data.

This whole-regime transfer is stronger evidence than evaluating additional
time points from the training schedule, but it remains a same-equation
synthetic test.

## Reproduce

```bash
python3 -m thermotwin.sparse_sensor_inference
```

The implementation is in
[`sparse_sensor_inference.py`](sparse_sensor_inference.py), with regression and
limiting-case checks in
[`../tests/test_sparse_sensor_inference.py`](../tests/test_sparse_sensor_inference.py).

## What this experiment does not establish

- It does not validate the four-node model against a physical assembly.
- It assumes a shared first-order lag for the two sensors.
- It keeps all parameters except the four listed quantities exact.
- Its uncertainty is local and linearized, not a nonlinear bootstrap or a
  hardware-calibrated confidence statement.
- It uses the conventional solver for transparent CPU-first inference. The
  existing inverse PINNs remain separate ideal learned-model baselines.
- The local polish reduces grid artifacts but does not turn the local
  linearized interval into a globally valid uncertainty distribution.
