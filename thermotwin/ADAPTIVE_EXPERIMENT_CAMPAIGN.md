# Adaptive experiment campaign under model mismatch

## Question

Does posterior-aware pulse selection reach a useful predictive result sooner
than a precommitted D-optimal batch or a plausible engineer-selected schedule?
More importantly, does collecting more informative data protect the inference
from physics that the fitted model does not contain?

This is a software-only experiment. It measures virtual test cycles, not
physical prototypes avoided.

## Campaign design

Every strategy receives at most four experiments and a 65 J cumulative modeled
electrical-energy budget. All experiments begin from the same 300 K equilibrium
so thermal carryover does not create a hidden advantage. Candidates come from
the existing 25-pulse amplitude/duration grid; 17 pass the existing per-run
energy and face-temperature limits. A candidate cannot be repeated within one
campaign.

The shared inferred quantities are:

1. cold thermal-contact resistance `R_c`;
2. cold-face thermal capacitance `C_cf`; and
3. shared sensor lag `tau_s`.

Each virtual run receives separate cold- and hot-sensor offsets, which are
profiled out. Physical parameters and lag are shared across runs. Observations
contain 0.02 K independent Gaussian noise and run-specific offsets drawn with a
0.05 K standard deviation. Physical truths vary lognormally with log standard
deviation 0.18. The original Gaussian prior is included once in every joint
fit; accumulated observations are refit together rather than folded back in as
a repeatedly counted prior.

### Compared policies

- **Adaptive:** jointly refit all accumulated runs, relinearize every feasible
  unused candidate at the current estimate, then select the largest expected
  log-determinant reduction that leaves enough energy for the remaining tests.
- **Static D-optimal:** construct the four-test campaign greedily at the nominal
  prior before observing any data.
- **Engineer heuristic:** precommit an amplitude/richness sweep without using
  posterior information.

The frozen plans are:

| Policy | Precommitted sequence |
| --- | --- |
| Static D-optimal | 0.8 A/20 s, 0.6 A/30 s, 0.6 A/15 s, 0.4 A/5 s |
| Engineer heuristic | 0.4 A/20 s, 0.6 A/20 s, 0.8 A/15 s, 1.0 A/10 s |

Adaptive choices are made separately in every paired trial. Each strategy sees
the identical virtual observation whenever it selects the same candidate for a
given truth condition and trial.

## Independent model-mismatch condition

The matched control generates and fits data with the same four-node equations.
The stress test instead generates observations with an independent five-state
truth model. It inserts an unobserved 20 J/K cold-interface thermal mass between
the thermoelectric face and exchanger and splits the declared cold-contact
resistance equally on its two sides. The total steady series resistance is
unchanged.

Planning and inference never receive the fifth state. They continue to use the
four-node model, so the campaign tests whether additional data reveal or merely
hide the missing dynamics.

## Predeclared gates

A campaign step passes prediction only when all three conditions hold on one
complete bipolar schedule excluded from fitting:

- accessible exchanger-temperature RMSE at most 0.040 K;
- hidden cold-face RMSE at most 0.050 K; and
- maximum mean residual in a three-second switch window at most 0.050 K.

Physical recovery separately requires joint log-parameter RMSE at most 0.100.
A fit is called confident when its largest local log-parameter standard error
is at most 0.100. `False confidence` means the local interval is tight while
either physical recovery or withheld prediction fails. The hidden-state and
physical checks are possible because this is a synthetic audit; they would
require independent instrumentation or intervention on hardware.

## Twenty-trial result

### Matched-model control

All policies ultimately pass every physical and prediction gate. Adaptive and
static selection choose the same first two tests in all 20 trials. Adaptation
changes only the third experiment: it chooses the static 0.6 A/15 s pulse in
14/20 trials and a 0.4 A/30 s pulse in 6/20. The fourth choice is always
0.4 A/5 s.

| Policy | Median first passing step | Median energy to pass | Final parameter log-RMSE | Final hidden-face RMSE | Final uncertainty volume |
| --- | ---: | ---: | ---: | ---: | ---: |
| Adaptive | 1 | 27.54 J | 0.02851 | 0.01499 K | 7.4151e-6 |
| Static D-optimal | 1 | 27.54 J | 0.02785 | 0.01576 K | 7.4219e-6 |
| Engineer heuristic | 1 | 6.89 J | 0.02717 | 0.01758 K | 8.3675e-6 |

There is no adaptive advantage in this condition. The heuristic reaches the
declared prediction gate using less modeled energy, and its final parameter
error is marginally lower. The D-optimal policies produce slightly smaller
local uncertainty volume, but that does not translate into a meaningful
predictive or physical-recovery improvement.

### Extra-interface-mass stress test

The adaptive policy changes its third choice to 0.4 A/30 s in 19/20 trials,
showing that the posterior does move the experiment ranking. That reaction does
not repair the omitted physics.

| Policy | Fit RMSE | Accessible withheld RMSE | Hidden-face withheld RMSE | Parameter log-RMSE | Prediction pass | False confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive | 0.01996 K | 0.00636 K | 0.09830 K | 0.19581 | 0/20 | 20/20 |
| Static D-optimal | 0.01989 K | 0.00650 K | 0.09489 K | 0.19739 | 0/20 | 20/20 |
| Engineer heuristic | 0.02035 K | 0.00643 K | 0.09749 K | 0.19499 | 0/20 | 20/20 |

All three policies fit their noisy observed data at approximately the declared
0.02 K noise scale. All also predict the two accessible held-out sensors to
within 0.0065 K on average. Nevertheless, every policy fails physical recovery
and the hidden-face transfer gate in every trial while reporting tight local
uncertainty. One trial in each final policy summary hits a parameter bound.

## Decision

Adaptive selection does **not** reduce virtual test cycles for this candidate
space. A well-designed static batch is effectively equivalent, and a simple
heuristic is adequate under matched equations.

The more valuable result is the failure under model mismatch: accumulating
high-information terminal measurements can make covariance shrink without
making the inferred physical quantities true. Accurate terminal prediction is
therefore insufficient evidence for contact-resistance identification or a
hidden-face claim. Before optimizing more pulses, the next modeled decision
should be whether to add an interface state, a face-proximal measurement, or an
independent heat-rate observable.

This is the software analogue of a hardware-team rule: do not spend additional
test cycles refining a parameter until a discrepancy-sensitive validation can
distinguish the parameter from missing physics.

## Reproduce

Run the numerical campaign without optional plotting dependencies:

```bash
python3 -m thermotwin.adaptive_experiment_campaign --trials 20 --no-figure
```

With the report dependency installed, generate the figure and JSON/TXT
sidecars:

```bash
python3 -m pip install -e '.[reports]'
thermotwin-adaptive-campaign
```

The default figure folder is
`figures/ADAPTIVE_EXPERIMENT_CAMPAIGN/`. The full 20-trial campaign takes
approximately seven minutes on the CPU used for the frozen result.

## Limitations

- Both truth models are synthetic and share the same bulk thermoelectric law,
  boundary conditions, and idealized reset-to-equilibrium assumption.
- Only one omitted-physics family and one interface-mass value are tested.
- Candidate energy is evaluated at nominal modeled parameters rather than from
  a physical power trace.
- The static policy is greedy rather than a globally optimized four-test batch.
- Candidate repetition is prohibited, so this does not evaluate the value of
  replicated measurements.
- Local Gaussian covariance is an audit target, not a calibrated global or
  neural uncertainty method.
- Twenty trials expose repeatable failure but do not estimate small differences
  between strategies precisely.
- Hidden-face error and physical-parameter error use synthetic truth unavailable
  on uninstrumented hardware.
- No result is hardware validation or evidence that four nodes are insufficient
  for a particular physical thermoelectric assembly.
