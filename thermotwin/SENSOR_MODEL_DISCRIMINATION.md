# Sensor model discrimination experiment

## Decision

If one new measurement can be added, measure the cold-face temperature during
the selected 0.8 A, 20 s pulse and during one withheld bipolar schedule.

In the frozen 20-pair synthetic campaign, that package selected the correct
four- versus five-state topology in **20/20 trials under each truth condition**
and passed the combined topology, physical-parameter, and hidden-state gate in
**40/40 trials**. It used one training pulse and 27.54 J of modeled training
energy. Four exchanger-temperature pulses used 64.41 J, selected the correct
topology in 37/40 trials, and passed the combined decision gate in only 32/40.

This is a software-only model-risk result. It does not establish that a real
cold-face sensor can be installed without changing the device.

## Why this experiment

The adaptive-campaign stress test found the important failure mode: a
four-state model could fit and predict the two exchanger temperatures while an
unmodeled cold-interface thermal mass corrupted the inferred physical
parameters and hidden cold-face temperature. More terminal tests did not fix
the omitted state.

This follow-on asks a hardware-team question rather than a solver question:

> Should the next unit of test effort buy more repetitions of the sensors we
> already have, or one measurement closer to the disputed physics?

No PINN is required. Both candidate topologies are fitted with the same bounded
nonlinear conventional estimator, and model choice is made on an excitation
excluded from parameter fitting.

## Candidate physics

The two candidates share thermoelectric physics, reservoir coupling, total
cold-contact resistance, and the three fitted physical quantities:

- cold-contact resistance `R_c`;
- cold-face thermal capacitance `C_cf`;
- shared temperature-sensor lag `tau_s`.

The **four-state candidate** contains cold face, hot face, cold exchanger, and
hot exchanger temperatures. The **five-state candidate** inserts a cold-side
interface temperature with its own fitted thermal mass. Its two half-contact
resistances preserve the same total steady contact resistance, so the
difference is transient storage rather than a different steady conductance.

The fifth-state mass has a 20 J/K nominal prior, an 8--80 J/K fit range, and is
varied independently around 20 J/K in the five-state truth trials. It is not
given its true value during fitting.

## Measurement packages

All packages include cold- and hot-exchanger temperatures. Temperature noise
is 0.02 K per observation and each run/channel receives an independent
constant bias with 0.05 K standard deviation. The added-channel assumptions
are 0.02 K for cold-face temperature, 0.05 W for cold-side heat rate, and
0.002 V for voltage; their constant biases scale by the same 2.5 ratio.

| Package | Training schedules | Added channel | Training energy |
| --- | ---: | --- | ---: |
| Baseline | 1 | none | 27.54 J |
| More exchanger tests | 4 | none | 64.41 J |
| Cold-face temperature | 1 | cold-face temperature | 27.54 J |
| Cold-side heat rate | 1 | exchanger-side cold-contact heat rate | 27.54 J |
| Voltage | 1 | terminal voltage | 27.54 J |

The single training schedule is the existing selected 0.8 A, 20 s pulse. The
four-test terminal-only package adds 0.6 A for 30 s, 0.6 A for 15 s, and 0.4 A
for 5 s. Every package then receives the same additional bipolar validation
schedule, measured with that package's installed channels. Validation energy
is common and is not included in the training-energy column.

This is deliberately not called an equal-cost hardware comparison. A heat-flux
sensor, embedded thermocouple, and voltage tap have different intrusion,
calibration, integration, and procurement costs that the simulation does not
know.

## Trial protocol

Twenty paired trials are run for each of two truth conditions:

1. **Matched four-state truth**, which checks that the extra-parameter model is
   not selected merely because it is more flexible.
2. **Independent five-state truth**, which contains the disputed interface
   storage and varies its mass between trials.

Each paired trial shares its physical truth and all common-channel noise across
packages. The three physical parameters vary lognormally with log standard
deviation 0.18; the interface mass varies with log standard deviation 0.25.
Both model candidates are fitted only to the training schedules. Per-run,
per-channel constant biases are profiled out. The candidate with lower
noise-normalized mean-square error on the complete withheld bipolar schedule is
selected.

The selected model is then audited against quantities available only because
this is synthetic truth:

- correct topology;
- log-RMSE of `R_c`, `C_cf`, and `tau_s`;
- cold-face RMSE over the withheld schedule;
- local uncertainty and fit-bound contact.

The predeclared cohort gate requires at least 90% correct topology selection in
both truth conditions and at most 10% false confidence. A trial passes the
physical decision gate only when topology is correct, physical log-parameter
RMSE is at most 0.10, and hidden cold-face RMSE is at most 0.05 K.

## Frozen result

Command:

```bash
python3 -m thermotwin.sensor_model_discrimination --trials 20 --no-figure
```

| Truth | Package | Correct model | Decision pass | Parameter log-RMSE | Hidden-face RMSE | Validation margin | False confidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Four-state | Baseline | 80% | 75% | 0.0664 | 0.0310 K | 0.069 | 15% |
| Four-state | Four exchanger tests | 90% | 90% | 0.0400 | 0.0199 K | 0.225 | 5% |
| Four-state | Cold-face temperature | **100%** | **100%** | **0.0149** | **0.0026 K** | 2.975 | **0%** |
| Four-state | Cold-side heat rate | **100%** | **100%** | 0.0175 | 0.0075 K | **5.877** | **0%** |
| Four-state | Voltage | **100%** | **100%** | 0.0212 | 0.0048 K | 0.584 | **0%** |
| Five-state | Baseline | 90% | 50% | 0.0859 | 0.0392 K | 0.051 | 15% |
| Five-state | Four exchanger tests | 95% | 70% | 0.0754 | 0.0326 K | 0.053 | 10% |
| Five-state | Cold-face temperature | **100%** | **100%** | 0.0300 | **0.0060 K** | **1.705** | **0%** |
| Five-state | Cold-side heat rate | **100%** | **100%** | **0.0273** | 0.0100 K | 1.363 | **0%** |
| Five-state | Voltage | **100%** | **100%** | 0.0417 | 0.0058 K | 0.927 | **0%** |

`Validation margin` is the correct model's advantage in held-out normalized
mean-square error, so positive and larger is better. It is not an absolute
probability or a calibrated Bayes factor.

## Interpretation

The terminal-only result is the useful negative control. Moving from one to
four exchanger-temperature training tests raises correct topology selection
from 80/90% to 90/95% under four-/five-state truth, but the five-state physical
decision pass rate reaches only 70%. More excitation helps, yet does not create
a measurement of the ambiguous internal dynamics.

All three added observables pass the frozen cohort gates. Cold-side heat rate
produces the largest separation under four-state truth, but a heat-flux sensor
is likely the most mechanically intrusive option. Voltage is attractive when
it is already logged by the drive electronics, though its topology margin is
smaller and its five-state physical-parameter error is highest among the added
channels.

Cold-face temperature is the best single recommendation for this demonstration
because it combines:

- perfect topology selection and physical-decision passes in the synthetic
  campaign;
- the lowest or nearly lowest physical-parameter error;
- the most direct audit of the hidden state that failed previously;
- much larger topology separation than exchanger temperatures alone;
- a measurement concept that is easier to explain and usually less intrusive
  than direct heat-flux sensing.

For an actual portfolio company, the final choice should be recomputed with
its accessible nodes, sensor dynamics, calibration data, installation cost,
and safety constraints. If voltage is already measured accurately, the next
hardware step may be to use that channel first rather than add a sensor.

## What is demonstrated now

ThermoTwin can now:

- express competing physical topologies behind one observation interface;
- generate paired noisy campaigns with shared truth and shared channel noise;
- fit both topologies with nuisance-bias profiling;
- choose a model on a fully withheld excitation;
- compare added sensing against additional test schedules with explicit
  experiment count and modeled energy;
- retain wrong choices, physical-recovery failures, and boundary hits.

This is reusable computational experiment machinery, not a result tied to a
PINN optimizer.

## Limitations

1. Both truth conditions come from one of the two candidate equation families.
   The withheld schedule prevents a training-fit contest, but it is not broad
   structural uncertainty.
2. Sensor errors are Gaussian noise plus constant bias. Drift, colored noise,
   quantization, synchronization error, and cross-channel calibration are
   omitted.
3. The simulated added sensor does not perturb thermal mass or contact
   resistance. A real embedded sensor can change exactly the transient being
   measured.
4. Heat rate is sampled ideally at the exchanger-side half contact. A practical
   heat-flux estimate may have a different location and bandwidth.
5. Voltage uses known thermoelectric constitutive parameters. Electrical
   contact resistance and converter measurement errors could confound it.
6. Twenty trials distinguish large effects, not small failure-rate differences.
7. Modeled pulse energy is not a prototype, labor, fixture, or calendar-time
   cost.
8. No hardware data have been used; none of the error rates are manufacturing
   predictions.

## Fifteen-minute walkthrough

1. **0:00--2:00 — Start with the failure.** Show that the previous four-state
   model fit two exchanger temperatures while missing a hidden interface mass.
2. **2:00--4:00 — Draw the two topologies.** Keep the same steady contact
   resistance and point only to the added transient-storage node.
3. **4:00--6:00 — State the decision.** Compare four more terminal tests with
   one pulse plus face temperature, heat rate, or voltage. Call out that the
   validation schedule is an additional common run.
4. **6:00--9:00 — Show the held-out rule.** Fit both candidates without the
   bipolar schedule, then let the withheld measurement choose. Avoid leading
   with optimizer or PINN details.
5. **9:00--12:00 — Show the two rows that matter.** Four exchanger tests reach
   only 70% decision passes under five-state truth; cold-face temperature
   reaches 100% with 0.0060 K hidden-face RMSE.
6. **12:00--14:00 — Explain the alternatives.** Heat rate has strong separation;
   voltage is compelling if already available; neither is declared universally
   cheaper by the simulation.
7. **14:00--15:00 — End on the boundary.** This run chooses what to instrument
   in a future test. It does not validate the model against hardware.

The line to leave with Ji Ke is: **when terminal data admit two plausible
physical stories, ThermoTwin can price the next measurement by whether it
changes the model decision—not by whether it makes the training fit prettier.**
