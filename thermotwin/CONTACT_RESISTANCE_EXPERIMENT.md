# Cold contact-resistance inference and noise study

## 1. Purpose

This document is a standalone walkthrough of ThermoTwin's first conventional
contact-parameter inference experiment. It explains the physical question,
frozen assumptions, current schedules, synthetic-data generation, regime-level
data split, loss function, scalar optimizer, validation procedure, numerical
results, interpretation, and limitations. It also documents the first
100-trial extension with controlled Gaussian temperature noise.

The experiment asks:

> Can ideal transient temperature observations recover one unknown cold-side
> thermal contact resistance when every other model quantity is known?

This is a controlled synthetic baseline. The same four-node mathematical
model generates and fits the observations. Success verifies the inference
workflow under ideal conditions; it does not validate the model against
hardware.

The implementation is in
[`contact_resistance_inference.py`](contact_resistance_inference.py). The
repeated-noise implementation is in
[`contact_resistance_noise_study.py`](contact_resistance_noise_study.py).

---

## 2. Result at a glance

The hidden synthetic cold contact resistance is 0.25 K/W. A bounded,
dependency-free golden-section search using only the unipolar training pulse
recovers:

| Quantity | Result |
| --- | ---: |
| True cold contact resistance | 0.250000000 K/W |
| Inferred cold contact resistance | 0.250000002 K/W |
| Absolute parameter error | approximately 1.52e-9 K/W |
| Relative parameter error | 6.078777e-7 % |
| Golden-section iterations | 39 |
| Loss evaluations | 42 |

The inferred resistance also reproduces two unseen current regimes:

| Split | Regime | Fitted cold-pair RMSE | All-sensor RMSE |
| --- | --- | ---: | ---: |
| Train | +1 A unipolar pulse | 1.698464e-9 K | 1.204815e-9 K |
| Validation | +0.6 A shifted pulse | 1.328620e-9 K | 9.420546e-10 K |
| Test | +1/−1 A bipolar pulse | 2.208849e-9 K | 1.563720e-9 K |

These errors are extremely small because the observations are noise-free and
the candidate simulator uses the same equations, numerical step, and fixed
parameters as the generator. This favorable situation is sometimes called an
inverse crime. It is useful as a software and identifiability baseline, but it
is much easier than real parameter inference.

The follow-on study adds independent 0.05 K Gaussian temperature noise and
repeats the fit for 100 saved trials:

| Quantity | Repeated-noise result |
| --- | ---: |
| Mean inferred resistance | 0.249782542 K/W |
| Sample standard deviation | 0.004116544 K/W |
| Mean parameter bias | -0.000217458 K/W |
| Parameter RMSE | 0.004101678 K/W |
| Empirical 5th--95th percentiles | 0.243722770--0.256246405 K/W |
| Search-bound hits | 0 |

This second result measures empirical variation under one isolated synthetic
noise model. It is not a hardware uncertainty interval. Section 19 derives
the statistics, traces the code, and explains the limits of the conclusion.

### 2.1 Dataset provenance and quality gate

Each of the three whole-regime datasets now carries its complete physical
configuration and synthetic parameter truth, including the current schedule,
integration step, initial and reservoir conditions, and split identity. The
observation table still excludes dense RK4 trajectories.

Run the compact quality gate with:

~~~bash
python3 -m thermotwin.dataset_quality
~~~

The frozen clean collection contains 732 of 732 expected sensor records. It
passes checks for recorded provenance, available ground truth, unique regime
names, and complete train/validation/test coverage. Noise seeds and applied
measurement settings are added to the ordered dataset provenance when those
transformations are used. This audit establishes reproducibility and split
integrity; it does not establish hardware realism or parameter
identifiability.

---

## 3. Physical topology

The contact-aware model has four dynamic temperatures:

| Symbol | Node |
| --- | --- |
| $T_{cf}$ | Cold thermoelectric face |
| $T_{hf}$ | Hot thermoelectric face |
| $T_{cx}$ | Cold heat exchanger |
| $T_{hx}$ | Hot heat exchanger |

The cold contact resistance $R_{contact,c}$ connects the cold exchanger to the
cold module face. The hot contact resistance $R_{contact,h}$ connects the hot
module face to the hot exchanger.

The cold contact heat rate is positive from the cold exchanger toward the
cold face:

$$
Q_{contact,c}=\frac{T_{cx}-T_{cf}}{R_{contact,c}}.
$$

The hot contact heat rate is positive from the hot face toward the hot
exchanger:

$$
Q_{contact,h}=\frac{T_{hf}-T_{hx}}{R_{contact,h}}.
$$

The four transient energy balances are

$$
C_{cf}\frac{dT_{cf}}{dt}=Q_{contact,c}-Q_c,
$$

$$
C_{hf}\frac{dT_{hf}}{dt}=Q_h-Q_{contact,h},
$$

$$
C_{cx}\frac{dT_{cx}}{dt}
=G_c(T_{c,\infty}-T_{cx})+\dot q_{c,ext}-Q_{contact,c},
$$

$$
C_{hx}\frac{dT_{hx}}{dt}
=G_h(T_{h,\infty}-T_{hx})+\dot q_{h,ext}+Q_{contact,h}.
$$

The thermoelectric face heat rates remain

$$
Q_c=\alpha I T_{cf}-\frac{1}{2}I^2R-K(T_{hf}-T_{cf}),
$$

$$
Q_h=\alpha I T_{hf}+\frac{1}{2}I^2R-K(T_{hf}-T_{cf}).
$$

Only $R_{contact,c}$ is inferred. Every other quantity in these equations is
fixed.

---

## 4. Why a pulse is informative

A constant current eventually approaches a steady state. Steady data can
contain information about contact resistance, but it may be difficult to
separate that information from module conductance, reservoir coupling, or
other thermal resistances.

A pulse adds two transitions:

1. current turns on and changes Peltier and Joule heat rates; and
2. current turns off and removes those active terms.

Temperatures remain continuous at each switch because every node has finite
thermal capacitance. Temperature derivatives can change immediately. The face
and exchanger then respond on different time scales, creating a transient
temperature difference across the contact.

A larger cold contact resistance weakens face-to-exchanger coupling. During
the driven pulse it generally produces:

- a colder, more isolated cold face;
- a warmer, more slowly responding cold exchanger relative to the face; and
- a larger value of $T_{cx}-T_{cf}$.

The recovery after turn-off provides passive relaxation information in
addition to the powered response.

---

## 5. Frozen physical and numerical parameters

All three regimes use the same generic reference values:

| Quantity | Value |
| --- | ---: |
| Seebeck coefficient $\alpha$ | 0.05 V/K |
| Electrical resistance $R$ | 2.0 ohm |
| Module thermal conductance $K$ | 0.5 W/K |
| Cold-face capacitance | 50 J/K |
| Cold-exchanger capacitance | 50 J/K |
| Hot-face capacitance | 100 J/K |
| Hot-exchanger capacitance | 100 J/K |
| Hidden cold contact resistance | 0.25 K/W |
| Fixed hot contact resistance | 0.25 K/W |
| Cold reservoir conductance | 2.0 W/K |
| Hot reservoir conductance | 4.0 W/K |
| All initial node temperatures | 300 K |
| Both reservoir temperatures | 300 K |
| Both external heat inputs | 0 W |
| Experiment duration | 60 s |
| RK4 time step | 0.1 s |
| Observation interval | 1.0 s |

The values are generic learning parameters, not measurements of a particular
thermoelectric assembly.

---

## 6. Current regimes and split

Whole experiments are assigned to splits. Individual time points are never
randomly divided between training and evaluation.

### 6.1 Training regime

| Time interval | Current |
| --- | ---: |
| 0 to 5 s | 0 A |
| 5 to 20 s | +1 A |
| 20 to 60 s | 0 A |

This unipolar pulse is the only regime used to select the resistance.

### 6.2 Validation regime

| Time interval | Current |
| --- | ---: |
| 0 to 10 s | 0 A |
| 10 to 30 s | +0.6 A |
| 30 to 60 s | 0 A |

The different amplitude and switching times check whether the inferred
resistance transfers beyond the training waveform.

### 6.3 Test regime

| Time interval | Current |
| --- | ---: |
| 0 to 5 s | 0 A |
| 5 to 20 s | +1 A |
| 20 to 35 s | 0 A |
| 35 to 50 s | −1 A |
| 50 to 60 s | 0 A |

The bipolar test is the most different held-out schedule. Current reversal
changes the sign of the Peltier term while Joule heating remains positive.

### 6.4 Right-continuous switch convention

The current schedule is right-continuous. At exactly 5 s in the training
regime, the recorded current is +1 A. The node temperatures are still 300 K at
that instant because they cannot jump. At exactly 20 s, the recorded current
is 0 A while the temperatures retain the values reached immediately before
turn-off.

---

## 7. Synthetic observation generation

For each regime, ThermoTwin performs these steps:

1. construct a four-node experiment using the hidden 0.25 K/W resistance;
2. integrate all four temperatures with RK4 at 0.1 s;
3. stop integration steps exactly at current transitions;
4. sample one ideal sensor at each of the four nodes every 1 s;
5. attach the right-continuous current to every long-form observation; and
6. return 61 times and 244 records without returning the dense trajectory.

The three regime datasets contain the current schedules and observations.
They do not contain a `true_contact_resistance` or hidden trajectory field.
The true value is used later only to score the completed synthetic recovery.

No measurement imperfection is active:

- no random noise;
- no fixed bias;
- no sensor lag; and
- no missing readings.

This isolates parameter recovery from observation-model complications.

---

## 8. Sensors used for fitting and checking

All four sensor histories are stored:

- cold face;
- cold exchanger;
- hot face; and
- hot exchanger.

Only the cold face and cold exchanger enter the fitting objective. They are
the two temperatures directly separated by the unknown cold contact.

The hot-side temperatures are withheld from the objective. After fitting,
they provide an independent consistency check within each regime. They are
not independent hardware data because the same synthetic model generated
them, but they can expose an implementation that matches the cold pair while
disturbing the rest of the coupled model.

---

## 9. Training loss

For candidate resistance $r$, the simulator generates cold-face and
cold-exchanger predictions at every training observation time. With
$s\in\{cf,cx\}$ and $N=61$ times, the loss is

$$
L(r)=\frac{1}{2N}
\sum_{s\in\{cf,cx\}}\sum_{k=1}^{N}
\left[T_{s,k}^{pred}(r)-T_{s,k}^{obs}\right]^2.
$$

The loss has units K squared. Both sensors and every time receive equal
weight. The initial 0 A interval is included even though it has little or no
sensitivity to contact resistance at exact equilibrium.

The training function accepts only datasets labeled `train`. Passing a
validation or test regime to the fitter raises an error.

---

## 10. Sensitivity before optimization

The frozen experiment explicitly evaluates three candidate resistances before
interpreting the optimizer:

| Candidate resistance | Training MSE |
| ---: | ---: |
| 0.10 K/W | 3.757467722442e-2 K² |
| 0.25 K/W | 0 K² |
| 0.50 K/W | 5.104280388841e-2 K² |

The corresponding fitted-pair RMSE values for the incorrect candidates are
approximately 0.19384 K and 0.22593 K. The exact zero at 0.25 K/W occurs
because the same deterministic simulator generated the observations.

At the training pulse turn-off, 20 s, the cold contact gap changes strongly
with resistance:

| Resistance | Cold face | Cold exchanger | $T_{cx}-T_{cf}$ |
| ---: | ---: | ---: | ---: |
| 0.10 K/W | 297.902608 K | 298.625977 K | 0.723369 K |
| 0.25 K/W | 297.448416 K | 298.990085 K | 1.541669 K |
| 0.50 K/W | 297.046956 K | 299.325978 K | 2.279022 K |

These distinguishable histories establish sensitivity in this controlled
problem. Sensitivity alone does not prove practical identifiability when
other parameters or sensor errors are unknown.

---

## 11. Scalar optimizer

Only one positive scalar is unknown, so the first conventional estimator does
not require PyTorch, SciPy, gradients, or an initial guess. It uses a bounded
golden-section search.

| Search choice | Value |
| --- | ---: |
| Lower resistance bound | 0.05 K/W |
| Upper resistance bound | 1.0 K/W |
| Resistance-interval tolerance | 1e-8 K/W |
| Maximum iterations | 96 |

Golden-section search maintains an interval containing the best region found.
At each iteration it compares two interior candidates, discards the worse
side, and reuses one previous evaluation. The frozen run converges in 39
iterations and 42 loss evaluations.

The search bounds enforce positivity and cover the generic truth. A result at
a bound would be a warning that the range, data, model, or identifiability
needs review.

---

## 12. Training transient results

Selected ideal training observations are:

| Time | Current | Cold face | Cold exchanger | Cold contact gap |
| ---: | ---: | ---: | ---: | ---: |
| 0 s | 0 A | 300.000000 K | 300.000000 K | 0 K |
| 5 s | +1 A | 300.000000 K | 300.000000 K | 0 K |
| 6 s | +1 A | 299.732848 K | 299.989572 K | 0.256724 K |
| 10 s | +1 A | 298.865345 K | 299.800846 K | 0.935501 K |
| 15 s | +1 A | 298.067299 K | 299.411668 K | 1.344368 K |
| 20 s | 0 A | 297.448416 K | 298.990085 K | 1.541669 K |
| 21 s | 0 A | 297.604115 K | 298.918046 K | 1.313930 K |
| 30 s | 0 A | 298.430921 K | 298.818330 K | 0.387409 K |
| 60 s | 0 A | 299.367531 K | 299.448204 K | 0.080673 K |

The maximum sampled cold contact gap is 1.541669 K at 20 s. The cold face also
reaches its minimum sampled temperature, 297.448416 K, at 20 s. After current
turn-off, the face warms and the contact gap decays toward zero.

---

## 13. Inference and held-out results

The fitted parameter is

$$
R_{contact,c}^{fit}=0.250000002\ \mathrm{K/W}.
$$

Per-sensor RMSE values are:

| Regime | Cold face | Cold exchanger | Hot face | Hot exchanger |
| --- | ---: | ---: | ---: | ---: |
| Training | 1.988106e-9 K | 1.347960e-9 K | 1.790815e-10 K | 6.847273e-11 K |
| Validation | 1.566795e-9 K | 1.037119e-9 K | 1.312767e-10 K | 4.660876e-11 K |
| Test | 2.329057e-9 K | 2.081711e-9 K | 1.400523e-10 K | 5.695359e-11 K |

The unseen-regime results show that the recovered scalar transfers across the
two frozen schedules. They do not demonstrate broad machine-learning
generalization: the same known differential equations are evaluated with one
recovered parameter.

---

## 14. Code path

The main code objects are:

| Object | Responsibility |
| --- | --- |
| `ContactResistanceRegime` | Names one current schedule and whole-data split |
| `ContactResistanceRegimeDataset` | Pairs a regime with its ideal observations |
| `ContactResistanceDatasetSplit` | Keeps train, validation, and test experiments separate |
| `contact_resistance_experiment` | Inserts one candidate resistance into fixed physics |
| `simulate_contact_resistance_observations` | Runs RK4 and returns ideal observations |
| `contact_resistance_training_loss` | Computes cold-pair MSE for training regimes |
| `ContactResistanceSearchConfig` | Defines positive bounds and stopping settings |
| `fit_cold_contact_resistance` | Performs golden-section scalar minimization |
| `evaluate_contact_resistance_regime` | Calculates all sensor RMSE values |
| `run_contact_resistance_inference_experiment` | Runs the complete frozen workflow |
| `format_contact_resistance_inference_report` | Produces the command-line text report |

The high-level workflow is:

~~~text
freeze current regimes
    -> generate hidden truth separately for each regime
    -> sample ideal four-sensor observations
    -> keep whole regimes in train/validation/test
    -> simulate candidate resistance on training regime
    -> minimize cold-face/cold-exchanger MSE
    -> freeze inferred resistance
    -> evaluate validation and test regimes
    -> compare with hidden truth only after fitting
~~~

---

## 15. Running the experiments

### 15.1 Ideal inference baseline

From the repository root, run:

~~~bash
python3 -m thermotwin.contact_resistance_inference
~~~

Expected report:

~~~text
cold contact resistance inference
true resistance: 0.250000000 K/W
inferred resistance: 0.250000002 K/W
relative parameter error: 6.078777e-07 %
search evaluations: 42
train unipolar_training_pulse: fitted-pair RMSE=1.698464e-09 K, all-sensor RMSE=1.204815e-09 K
validation lower_amplitude_validation_pulse: fitted-pair RMSE=1.328620e-09 K, all-sensor RMSE=9.420546e-10 K
test bipolar_test_pulse: fitted-pair RMSE=2.208849e-09 K, all-sensor RMSE=1.563720e-09 K
~~~

Run the focused tests with:

~~~bash
python3 -m unittest tests.test_contact_resistance_inference
~~~

### 15.2 Repeated-noise study

Run the frozen 100-trial study with:

~~~bash
python3 -m thermotwin.contact_resistance_noise_study
~~~

Use a smaller trial count while exploring:

~~~bash
python3 -m thermotwin.contact_resistance_noise_study --trials 5
~~~

The command also accepts `--first-seed` and
`--noise-standard-deviation`. Changing either produces a different controlled
study, so report both values whenever results are compared.

Run the focused robustness tests with:

~~~bash
python3 -m unittest tests.test_contact_resistance_noise_study
~~~

---

## 16. What the ideal-inference tests protect

The focused tests verify:

- exact train, validation, and test current schedules;
- whole-regime splitting and unique labels;
- positive candidate resistance and search bounds;
- preservation of all fixed physical parameters;
- ideal 61-time, 244-record datasets;
- hidden-truth exclusion from inference datasets;
- right-continuous current and continuous switch temperatures;
- the frozen 20 s contact-gap regression value;
- increasing driven contact gap with increasing resistance;
- an exact synthetic loss minimum at 0.25 K/W;
- exclusion of hot-side readings from the fitting loss;
- rejection of validation data by the fitter;
- scalar recovery and bounded search history;
- low error on unseen whole regimes; and
- reproducible report formatting.

---

## 17. What has been learned

Within the frozen mathematical model:

1. The unipolar pulse creates a clearly resistance-sensitive cold contact
   temperature gap.
2. Cold-face and cold-exchanger histories are sufficient to recover one
   resistance when every other quantity is known.
3. A dependency-free scalar optimizer is enough for this one-parameter
   baseline.
4. Whole-regime evaluation prevents adjacent-time leakage.
5. The recovered resistance reproduces unseen amplitudes, timings, and current
   reversal when the model is exact.
6. Hot-side histories provide a useful consistency check even though they do
   not enter the loss.
7. Under the isolated 0.05 K Gaussian-noise model, the 100-trial estimates
   remain centered near the hidden truth with a 0.00412 K/W sample standard
   deviation and no search-bound hits.
8. Fixed face and exchanger biases create different systematic parameter
   shifts, and common-mode bias does not cancel from an absolute-temperature
   loss.
9. Sensor lag can be partly misattributed to contact resistance but leaves
   dynamic residuals on held-out regimes.
10. Switch-adjacent readings carry more local resistance information than the
    same number of equilibrium readings.
11. Cold-side sensors dominate sensitivity to the cold contact in the frozen
    experiment; the hot pair contributes very little additional curvature.
12. Under combined imperfections, systematic bias can greatly exceed random
    trial spread.

---

## 18. What has not been learned

This experiment does not establish:

- the cold contact resistance of physical hardware;
- the correctness of the four-node lumped model;
- the accuracy of any fixed thermal parameter;
- robustness to bias, lag, missingness, or noise structures and magnitudes
  beyond the frozen synthetic cases;
- the realism of the selected bias, lag, noise, outage, or sensor-availability
  assumptions for physical instruments;
- identifiability when multiple parameters vary together;
- formal uncertainty bounds on the inferred resistance;
- correctness under temperature-dependent material properties;
- equivalence between contact paste, clamping pressure, geometry, and one
  constant lumped resistance; or
- safety of any current schedule on a real device.

None of the reported synthetic errors or empirical intervals should be
presented as physical measurement accuracy.

---

## 19. Repeated Gaussian-noise robustness study

### 19.1 Question

The first robustness extension asks:

> If independent zero-mean temperature errors with a 0.05 K standard
> deviation are added, how much does the inferred cold contact resistance vary
> across repeated synthetic experiments?

One noisy fit is not enough to answer that question. It can land unusually
close to or far from the truth by chance. The implementation therefore runs
100 reproducible trials and summarizes the distribution of fitted parameters.

### 19.2 What is held fixed

This stage changes only the temperature observations. It preserves:

- the 0.25 K/W hidden cold contact resistance;
- all other physical parameters;
- the three complete train, validation, and test current regimes;
- the 0.1 s RK4 step and 1 s observation interval;
- all four sensor locations;
- the cold-face and cold-exchanger fitting pair; and
- the equal-weight least-squares loss.

Bias, lag, missing readings, current error, correlated noise, parameter error,
and model discrepancy remain disabled. That isolation is essential: if the
fit changes, this experiment lets us attribute the change to the imposed
random temperature error rather than to several mechanisms at once.

### 19.3 Frozen trial design and seed mapping

All four temperature sensors receive independent Gaussian errors with mean
zero and standard deviation 0.05 K. Trial $i$, counted from zero, uses:

$$
s_{train}=2026+3i,
$$

$$
s_{validation}=2027+3i,
$$

$$
s_{test}=2028+3i.
$$

These are split-level base seeds. The first regime in each split retains its
base seed; any later regime is mapped into a disjoint Cantor-paired namespace.
Consequently, adding regimes cannot reuse another split or trial's random
stream. The mapping remains deterministic and preserves the established
one-regime results.

### 19.4 One-trial data path

Each trial follows this sequence:

~~~text
ideal four-node RK4 datasets
        |
        +--> independent noise on train, validation, and test observations
        |
        +--> fit R_contact,c using only the noisy cold training pair
        |
        +--> evaluate that same estimate on all three noisy regimes
        |
        +--> compare again with hidden ideal temperatures for analysis only
~~~

The optimizer searches from 0.05 to 1.0 K/W. The noise study uses a 1e-6 K/W
interval tolerance and at most 64 golden-section iterations. This tolerance is
far smaller than the parameter variation caused by 0.05 K noise and reduces
unnecessary repeated simulation. The frozen fits require 32 loss evaluations
per trial.

### 19.5 Why two temperature RMSEs are reported

Observation RMSE compares a prediction with the noisy readings:

$$
RMSE_{obs}=\sqrt{\frac{1}{N}\sum_{j=1}^{N}
\left(T_j^{pred}-T_j^{noisy}\right)^2}.
$$

This is the error an estimator can calculate from the available dataset.

Truth RMSE compares the same prediction with the hidden ideal temperatures:

$$
RMSE_{truth}=\sqrt{\frac{1}{N}\sum_{j=1}^{N}
\left(T_j^{pred}-T_j^{ideal}\right)^2}.
$$

This second quantity is available only because the experiment is synthetic.
It measures trajectory error without asking the model to reproduce individual
random errors. A physical experiment would not reveal exact hidden truth.

### 19.6 Parameter statistics

For estimates $r_1,\ldots,r_n$ and true resistance $r_{true}$, the report uses:

$$
bias=\frac{1}{n}\sum_{i=1}^{n}(r_i-r_{true}),
$$

$$
RMSE_r=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(r_i-r_{true})^2},
$$

and the sample standard deviation with denominator $n-1$. The empirical 5th
and 95th percentiles are linearly interpolated through the ordered estimates.
A bound-hit count checks whether the optimizer is being truncated by its
allowed interval.

### 19.7 Frozen 100-trial results

Run the study from the repository root:

~~~bash
python3 -m thermotwin.contact_resistance_noise_study
~~~

The saved configuration produces:

| Parameter metric | Result |
| --- | ---: |
| Trials | 100 |
| Mean inferred resistance | 0.249782542 K/W |
| Sample standard deviation | 0.004116544 K/W |
| Mean parameter bias | -0.000217458 K/W |
| Parameter RMSE | 0.004101678 K/W |
| Empirical 5th percentile | 0.243722770 K/W |
| Empirical 95th percentile | 0.256246405 K/W |
| Search-bound hits | 0 |

| Mean fitted-pair RMSE | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Against noisy observations | 0.049496 K | 0.050037 K | 0.049789 K |
| Against hidden ideal truth | 0.003580 K | 0.002800 K | 0.004655 K |

The mean estimate is 0.000217 K/W below the truth, while the trial-to-trial
standard deviation is 0.004117 K/W. Thus the observed bias is small compared
with the random spread in this finite study. The standard deviation is about
1.65 percent of the 0.25 K/W truth. No estimate reaches either search bound.

The observation errors remain close to the imposed 0.05 K scale. The smaller
truth errors show that the inferred physical trajectory remains much closer to
the ideal trajectory than to every individual noisy reading. The test truth
error is larger than the validation truth error because the bipolar test
schedule has a different sensitivity to a resistance error; this does not
mean its sensors received more noise.

### 19.8 Reproduction and exploration

A shorter development run is available without changing the frozen default:

~~~bash
python3 -m thermotwin.contact_resistance_noise_study --trials 5
~~~

`--first-seed` selects another reproducible set of trials, and
`--noise-standard-deviation` changes the isolated noise scale. The exact
zero-noise case is tested as the ideal-data limiting case.

The central objects are:

- `ContactResistanceNoiseStudyConfig`, which freezes the study controls;
- `ContactResistanceNoiseSeeds`, which records the three split seeds in one trial;
- `run_contact_resistance_noise_trial`, which performs one fit and evaluation;
- `run_contact_resistance_noise_study`, which repeats and summarizes trials;
  and
- `ContactResistanceNoiseStudySummary`, which stores parameter and
  temperature-error statistics.

The focused tests are in `tests/test_contact_resistance_noise_study.py`. They
check validation, seed uniqueness, exact reproducibility, schema preservation,
the zero-noise limit, frozen regression values, statistic calculations, and
the generated report.

### 19.9 Correct interpretation and limitations

The 5th--95th percentile range is an empirical interval across these 100
saved synthetic trials. It is not automatically a 90 percent confidence
interval for hardware, a guarantee of repeated-sample coverage, or a Bayesian
credible interval. The result assumes that the model and every non-noise
quantity are exactly correct.

The study has learned that the current one-parameter estimator is not strongly
destabilized by independent 0.05 K Gaussian temperature errors in the frozen
same-model problem. It has not learned whether a physical sensor has that
error distribution, whether its errors are independent, or how inference
behaves when systematic and physical uncertainties interact.

---

## 20. Fixed sensor bias

### 20.1 Physical question

A fixed bias adds the same temperature offset to every reading from one
sensor:

$$
T_s^{observed}(t)=T_s^{ideal}(t)+b_s.
$$

The error is systematic. Repeating the experiment or averaging more points
does not force it toward zero. The estimator does not know $b_s$ and can change
only $R_{contact,c}$, so part of the calibration error may be misattributed to
the contact.

### 20.2 Frozen cases

The implementation in `contact_resistance_bias_study.py` uses:

| Case | Cold-face bias | Cold-exchanger bias |
| --- | ---: | ---: |
| Zero limit | 0 K | 0 K |
| Face only | +0.10 K | 0 K |
| Exchanger only | 0 K | +0.10 K |
| Common mode | +0.10 K | +0.10 K |
| Differential | +0.05 K | -0.05 K |

Every bias pattern is applied independently to the complete train,
validation, and test regimes after sampling. The cold pair remains the fitting
pair, and all physical parameters remain exact.

### 20.3 Results

| Case | Inferred resistance | Test truth RMSE |
| --- | ---: | ---: |
| Zero limit | 0.249999776 K/W | approximately 0 K |
| Face only | 0.208885282 K/W | 0.063227 K |
| Exchanger only | 0.272817055 K/W | 0.032146 K |
| Common mode | 0.228450366 K/W | 0.032262 K |
| Differential | 0.218889695 K/W | 0.047190 K |

The face-only and exchanger-only cases shift the inferred parameter in
opposite directions. The common-mode result is especially important: equal
offsets preserve the measured contact temperature difference, but they shift
both absolute temperature histories relative to the reservoirs. Because the
loss fits absolute temperatures, common-mode error does not cancel.

~~~bash
python3 -m thermotwin.contact_resistance_bias_study
python3 -m unittest tests.test_contact_resistance_bias_study
~~~

The zero-bias case is the required ideal limiting case. This study does not
claim that a physical sensor has a constant +0.10 K offset; that value is a
controlled sensitivity input.

---

## 21. Sensor lag and confusion with contact dynamics

### 21.1 Sensor equation and ordering

Between dense truth samples, the ideal target is interpolated linearly. The
first-order sensor state then has the exact piecewise-linear-target update

$$
T_{s,k}^{lag}=a_kT_{s,k-1}^{lag}
 +(1-a_k)T_{s,k-1}^{ideal}
 +m_{s,k}\left[\Delta t_k-\tau_s(1-a_k)\right],
$$

$$
a_k=\exp\left(-\frac{\Delta t_k}{\tau_s}\right),\qquad
m_{s,k}=\frac{T_{s,k}^{ideal}-T_{s,k-1}^{ideal}}{\Delta t_k}.
$$

The former right-endpoint constant-target update was exact for that artificial
hold assumption but led a continuous ramp by approximately half a dense time
step. Linear interpolation removes that discretization artifact.

`contact_resistance_lag_study.py` simulates ideal observations every 0.1 s,
evolves this sensor state, and only then downsamples to the 1 s measurement
interval. Filtering only the already sparse readings would define a different
sensor model and make the result depend incorrectly on reporting frequency.

### 21.2 Frozen cases and results

| Lag case | Inferred resistance | Training observation RMSE | Test observation RMSE |
| --- | ---: | ---: | ---: |
| Zero lag | 0.249999776 K/W | approximately 0 K | approximately 0 K |
| Face 2 s | 0.246787415 K/W | 0.124839 K | 0.199408 K |
| Exchanger 2 s | 0.271277687 K/W | 0.049714 K | 0.079472 K |
| Both 2 s | 0.271434083 K/W | 0.134664 K | 0.216076 K |
| Face 2 s, exchanger 0.5 s | 0.252630129 K/W | 0.125666 K | 0.201088 K |

The estimator changes resistance because contact resistance also changes
transient temperature differences. That is confounding. However, the
nonzero residuals show that a static resistance cannot reproduce the entire
first-order sensor response. A later multi-parameter fit could also confuse
lag with thermal capacitance because both affect apparent response time. This
study identifies that risk but does not fit capacitance.

~~~bash
python3 -m thermotwin.contact_resistance_lag_study
python3 -m unittest tests.test_contact_resistance_lag_study
~~~

---

## 22. Missing readings around turn-off

### 22.1 Why turn-off is treated specially

The training contact gap reaches its largest sampled value at the 20 s pulse
turn-off. Validation turns off at 30 s, and the bipolar test turns off at 20
and 50 s. The implementation derives those nonzero-to-zero transitions from
each current schedule and centers the outages on them.

The frozen cases remove both cold-sensor readings at:

- no times;
- 0 through 4 s as an equilibrium control;
- only each turn-off instant;
- plus or minus 2 s around turn-off; or
- plus or minus 5 s around turn-off.

### 22.2 Why parameter error is insufficient

Every retained reading is still exact same-model data. Enough information
remains for all five cases to recover 0.249999776 K/W. That does not mean the
removed records were unimportant.

The implementation uses local curvature of the unnormalized sum of squared
errors $J$ as a sensitivity proxy:

$$
H_J\approx
\frac{J(r_0-\delta)-2J(r_0)+J(r_0+\delta)}{\delta^2}.
$$

Normalization is deliberately removed by multiplying MSE by the number of
available readings. Otherwise, deleting zero-sensitivity equilibrium records
could artificially increase an average loss.

| Case | Training records | SSE curvature | Fraction of complete curvature |
| --- | ---: | ---: | ---: |
| Complete | 122 | 304.8575 | 1.000 |
| Remove equilibrium control | 112 | 304.8575 | 1.000 |
| Remove turn-off instants | 120 | 286.4841 | 0.940 |
| Remove plus-or-minus 2 s | 112 | 216.4959 | 0.710 |
| Remove plus-or-minus 5 s | 100 | 133.4655 | 0.438 |

The control and narrow turn-off cases both retain 112 readings, yet their
curvatures differ substantially. Information depends on when measurements are
taken, not merely how many exist.

~~~bash
python3 -m thermotwin.contact_resistance_missingness_study
python3 -m unittest tests.test_contact_resistance_missingness_study
~~~

Curvature describes only local same-model sensitivity near the known truth.
It is not a confidence interval and does not capture all nonlinear or
multi-parameter ambiguities.

---

## 23. Restricted sensor sets

### 23.1 Dataset meaning

Restricted sensors are removed from both the dataset's sensor definitions and
its observation records. The fitter receives an explicit tuple of available
sensor names. It cannot use a sensor that is absent from the schema.

The generalized inference functions preserve their original default: the
cold face and cold exchanger are fitted unless another validated selection is
passed explicitly. Predictions are paired only to retained observed times,
which also makes partial outages valid.

### 23.2 Results

| Available sensors | Training records | SSE curvature | Exact inferred resistance |
| --- | ---: | ---: | ---: |
| Cold pair | 122 | 304.8575 | 0.249999776 K/W |
| Cold face only | 61 | 208.8583 | 0.249999776 K/W |
| Cold exchanger only | 61 | 95.9992 | 0.249999776 K/W |
| Hot pair only | 122 | 1.9426 | 0.249999776 K/W |
| All four | 244 | 306.8001 | 0.249999776 K/W |

Exact recovery proves that each noiseless loss has its minimum at the truth.
It does not mean the cases are equally robust. The hot pair responds only
weakly through the coupled model, and adding it to the cold pair contributes
less than 1 percent additional curvature. In this frozen experiment, a
cold-face sensor is substantially more informative than a cold-exchanger
sensor if only one can be retained.

~~~bash
python3 -m thermotwin.contact_resistance_sensor_study
python3 -m unittest tests.test_contact_resistance_sensor_study
~~~

Hardware sensor selection must additionally consider placement uncertainty,
calibration, cost, synchronization, thermal loading, and accessibility. Those
effects are not represented here.

---

## 24. Combined measurement imperfections

### 24.1 Frozen pipeline

The combined implementation preserves the agreed physical order:

~~~text
dense four-node truth
    -> first-order sensor lag at 0.1 s
    -> output sampling at 1 s
    -> fixed bias
    -> independent Gaussian noise
    -> regime-aligned turn-off outages
    -> restricted returned sensor schema
    -> scalar resistance fit
~~~

The frozen settings are:

| Mechanism | Setting |
| --- | --- |
| Lag | Cold face, 2 s |
| Bias | Cold face, +0.10 K |
| Noise | All four sensors, 0.05 K standard deviation |
| Noise seeds | Same 2026-based mapping as noise-only study |
| Missingness | Both cold sensors, plus or minus 2 s at turn-off |
| Available sensors | Cold face and cold exchanger only |
| Trials | 100 |

Noise is generated before unavailable records and sensors are removed. This
preserves the previously agreed measurement pipeline and keeps a fixed random
draw associated with each original sensor record.

### 24.2 Results

| Parameter metric | Combined result |
| --- | ---: |
| Mean inferred resistance | 0.201590126 K/W |
| Sample standard deviation | 0.005722516 K/W |
| Mean bias | -0.048409874 K/W |
| Parameter RMSE | 0.048743570 K/W |
| Empirical 5th percentile | 0.191932514 K/W |
| Empirical 95th percentile | 0.210875489 K/W |
| Search-bound hits | 0 |

| Mean RMSE | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Against imperfect observations | 0.148118 K | 0.118964 K | 0.218345 K |
| Against visible ideal truth | 0.049159 K | 0.039371 K | 0.065644 K |

The mean is nearly 0.05 K/W below the truth. That systematic error is much
larger than the 0.00568 K/W trial spread. The result illustrates why an
apparently precise empirical distribution can still be centered on the wrong
parameter.

### 24.3 Limiting case and reproduction

With noise, bias, and lag set to zero and outages disabled, the combined code
recovers 0.249999776 K/W and sub-microkelvin errors. This proves that the
composition machinery itself reduces to the ideal experiment.

~~~bash
python3 -m thermotwin.contact_resistance_combined_study
python3 -m thermotwin.contact_resistance_combined_study --trials 5
python3 -m unittest tests.test_contact_resistance_combined_study
~~~

The 5th--95th range is still an empirical interval over synthetic trials. It
does not account for unknown hardware bias, uncertain lag structure,
temperature-dependent properties, parameter mismatch, or the correctness of
the four-node topology.

---

## 25. Shared robustness implementation

`contact_resistance_robustness.py` provides the common safeguards used by the
isolated and combined studies:

- immutable whole-regime dataset transformations;
- dense-before-sparse lag processing;
- physical sensor-schema restriction;
- matching of hidden ideal truth only to visible record keys;
- selected-sensor record counting;
- local training-SSE curvature;
- common fit and train/validation/test scoring; and
- separate observation and hidden-truth RMSEs.

The fitter's original cold-pair behavior remains its default. New keyword
arguments select another sensor set deliberately. Missing readings are paired
by sensor and observed time rather than filled, interpolated, or treated as
zero.

Focused utility tests are in
`tests/test_contact_resistance_robustness.py`. Run every current ThermoTwin
test with:

~~~bash
python3 -m unittest discover -s tests
~~~

---

## 26. Ideal inverse-PINN comparison

The optional `inverse_contact_resistance.py` module now supplies a learned
one-parameter comparison. It reuses the validated four-output contact PINN,
makes only the cold contact resistance trainable through a positive softplus
transform, and adds sparse cold-face and cold-exchanger temperature mismatch
to the four physics residuals.

This first comparison uses a smooth constant 1 A experiment rather than the
training pulse above. Ideal cold-pair observations are retained every 5 s,
giving 13 paired times. The hidden resistance is 0.25 K/W, the neural initial
guess is 0.50 K/W, and every other physical parameter is held fixed. Dense RK4
temperatures and both hot-side histories remain withheld during optimization.

The conventional golden-section search is run on the same sparse
constant-current observations. With the frozen CPU settings:

| Estimator | Inferred cold contact resistance |
| --- | ---: |
| Inverse PINN | 0.250140756 K/W |
| Conventional scalar search | 0.250000002 K/W |
| Hidden truth | 0.250000000 K/W |

The PINN's relative parameter error is 0.056303 percent. Its dense
constant-current temperature RMSE remains between 0.000832 K and 0.001542 K
across the four states.

The learned resistance is then inserted into the conventional solver for the
existing lower-amplitude validation pulse and bipolar test pulse. All-sensor
RMSE is 0.000087 K and 0.000145 K, respectively. This transfers the physical
parameter, not the constant-current neural temperature function.

Run and plot the comparison with:

~~~bash
python3 -m thermotwin.inverse_contact_resistance_report
~~~

The ideal learned result does not use the noisy, biased, lagged, incomplete,
or restricted pulse datasets documented above. Section 27 adds the piecewise
forward representation and Section 28 extends it to ideal inverse pulse
training. The imperfect datasets remain the next neural comparisons.

---

## 27. Piecewise switched-current forward PINN

The optional `piecewise_contact_forward_pinn.py` workflow now represents the
established training pulse directly with a physics-only neural solver:

~~~text
0--5 s: 0 A  |  5--20 s: 1 A  |  20--60 s: 0 A
~~~

A separate four-output smooth subnetwork covers each constant-current
interval. The next segment starts from the previous segment's predicted final
four-temperature state. Therefore all face and exchanger temperatures are
exactly continuous at 5 s and 20 s, while their rates may change
discontinuously when the Peltier and Joule terms switch.

The PINN uses the conventional solver's right-continuous current convention:
the value at 5 s is 1 A and the value at 20 s is 0 A. Midpoint collocation
times exclude both switches because the two-sided classical derivative is not
defined there. RK4 reference temperatures remain withheld from training.

Run the frozen 5,000-epoch CPU comparison with:

~~~bash
python3 -m thermotwin.piecewise_contact_forward_pinn_report
~~~

The current result has a constructed maximum boundary-temperature jump of
exactly 0 K. Its RK4 comparison gives:

| State | RMSE |
| --- | ---: |
| Cold face | 0.008862 K |
| Hot face | 0.001989 K |
| Cold exchanger | 0.009327 K |
| Hot exchanger | 0.004628 K |

This is a fixed-parameter forward validation. It establishes the segmented
temperature and residual representation before the cold contact resistance is
made trainable on pulse observations in the next section.

---

## 28. Piecewise inverse contact-resistance PINN

The optional `piecewise_inverse_contact_resistance.py` workflow uses the same
whole 0--1--0 A training pulse and makes one cold contact resistance trainable.
The parameter is positive by construction and is shared by all three
temperature subnetworks. It therefore represents one constant physical
interface rather than allowing an artificial resistance change at current
switches.

The ideal virtual test stand provides cold-face and cold-exchanger readings
every 1 s, giving 61 paired observation times. All four physics residuals are
evaluated at 192 transition-free collocation points. Dense RK4 temperatures
and both hot-side histories remain withheld until validation. The conventional
golden-section estimator receives the identical long-form pulse dataset.

The normalized training objective is

$$
\mathcal L=\mathcal L_{physics}+20\mathcal L_{observations}.
$$

The observation weight is a numerical conditioning choice that reduces the
ability of flexible temperature subnetworks to retain a biased resistance. It
is not additional evidence and does not change the energy balances. Parameter
accuracy is checked directly against hidden truth and conventional search.

Run the frozen 8,000-epoch CPU workflow with:

~~~bash
python3 -m thermotwin.piecewise_inverse_contact_resistance_report
~~~

Starting from 0.50 K/W, the result is:

| Metric | Result |
| --- | ---: |
| Hidden resistance | 0.250000000 K/W |
| Piecewise inverse-PINN resistance | 0.250518948 K/W |
| Conventional fit, identical observations | 0.250000002 K/W |
| PINN relative parameter error | 0.207579 percent |
| Maximum boundary-temperature jump | 0 K |
| Validation-pulse all-sensor transfer RMSE | 0.000322 K |
| Bipolar-test all-sensor transfer RMSE | 0.000534 K |

The dense neural training-pulse RMSE values are 0.006704 K, 0.002868 K,
0.001797 K, and 0.002624 K for the cold face, hot face, cold exchanger, and hot
exchanger. The transfer results insert the PINN parameter into the conventional
solver; they transfer the physical resistance, not a neural trajectory with
different switch times.

This is still an ideal same-model, one-unknown result. Its exact limiting case
shows that resistance has zero loss gradient when no cold contact temperature
drop develops. The next comparisons will replace ideal records with the
already frozen missing, restricted-sensor, noisy, biased, lagged, and combined
datasets one mechanism at a time.

---

## 29. Planned progression

The next controlled extensions are:

1. compare the piecewise PINN and conventional estimator on identical missing,
   restricted-sensor, noisy, biased, lagged, and combined pulse observations;
2. infer contact resistance while perturbing other assumed-known parameters;
3. study simultaneous contact, capacitance, conductance, bias, and lag
   ambiguities one small set at a time;
4. quantify profile likelihood, bootstrap uncertainty, and practical
   identifiability;
5. use sensitivity to rank candidate current schedules and sensor layouts;
   and
6. design hardware trials only after safety limits and measurement definitions
   are agreed.

Every extension should preserve the ideal and zero-imperfection limits as
regression tests.
