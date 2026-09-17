# Blinded operating-decision development pilot

Status: Stage 2 development pilot complete on 2026-09-10. This is synthetic
development evidence, not a frozen evaluation or hardware validation.

Audit note (2026-09-17): the September 11–12 audit reproduced every displayed
Stage 2 number, then found deterministic random-stream collisions in the
shared historical generator. The table below is retained unchanged as
descriptive development history; it has no independent-sample or inferential
interpretation. The later collision-free corrected replication and the
prospective four-action experiment supersede it for current conclusions. See
[`OPERATING_DECISION_AUDIT.md`](../docs/thermotwin/OPERATING_DECISION_AUDIT.md)
and
[`OPERATING_DECISION_PROJECT_STATUS.md`](../docs/thermotwin/OPERATING_DECISION_PROJECT_STATUS.md).

Stage 3 now reruns these unchanged policies with electrical-contact loss,
temporary face-probe loading, and an omitted temperature-dependent-contact
truth family. See [the realism stress test](OPERATING_DECISION_REALISM.md).

## Question

Can a fixed diagnostic package support an approve, reject, or
insufficient-evidence decision for a previously untried thermoelectric current
schedule when both parameter uncertainty and model-family ambiguity remain?

This stage tests the experimental plumbing needed for that question. It
implements a scalar operating margin, prospective measurement packages, a
strict acquisition/verification/final split, multiple candidate-model fits,
model adequacy checks, uncertainty envelopes, and postdecision scoring. It does
not yet test realistic voltage confounding, face-sensor loading, an omitted
physics truth family, or a measurement selector.

## Frozen development protocol

The final operating current is fixed for every policy:

```text
time transitions (s): 5, 25, 38, 58
current values (A):   0, +1.0, 0, -0.8, 0
run duration:         80 s
```

The acceptable cold-face band is 285.0 to 301.7 K. For a cold-face trajectory
`T(t)`, the operating margin is

```text
m = min_t min(T(t) - 285.0, 301.7 - T(t)).
```

A nonnegative margin passes. At the nominal parameters, this pulse gives a
-0.1579 K four-state margin and a +0.1173 K five-state margin. This deliberate
straddling makes the development problem sensitive to the hidden interface
state. It is a synthetic benchmark boundary, not a product temperature limit.

The cohort contains 10 fresh devices from each of the two established truth
families, starting at seed 191001. Four of ten four-state truths and eight of
ten extra-interface-mass truths pass the final band. Each policy sees the same
underlying device and the same noise on genuinely common channels.

Every policy begins with one 0.8 A, 20 s pulse observed at the two exchangers.
The fixed actions are:

| Policy | Additional acquisition data | Total diagnostic runs, including verification |
| --- | --- | ---: |
| Stop now | None | 2 |
| More thermal tests | 0.6 A for 30 s, 0.6 A for 15 s, and 0.4 A for 5 s | 5 |
| Add voltage | A new 0.8 A, 20 s run with voltage | 3 |
| Add face temperature | A new 0.8 A, 20 s run with cold-face temperature | 3 |

Added channels are prospective: neither voltage nor face temperature is
backfilled onto the initial pulse. All policies then receive the fixed bipolar
verification schedule from the earlier sensor-discrimination study.

Both four-state and five-state candidates are fitted from three declared
starting points. The best complete acquisition objective is retained for each
candidate. Verification never refits the physical parameters or a new
constant offset. Instead, residuals are scored under the declared Gaussian
noise plus run-offset covariance; candidates with normalized score at most 2.0
and no bound hit remain plausible.

For this development stage, each verified model receives an uncalibrated local
delta-method interval for the final scalar margin, using twice the propagated
standard error. The decision uses the envelope over every verified candidate:

- approve when the envelope lower bound is at least zero;
- reject when its upper bound is below zero;
- return insufficient evidence when it crosses zero or a required check fails.

The final response is absent from the blinded case type. A saved decision and a
revealed response share an immutable opaque case identity, and scoring
recomputes the true margin from the revealed trajectory. Automated tests reject
cross-case scoring, inconsistent revealed margins, final responses presented
as fitting runs, retrospective sensor channels, and verification data that is
misclassified as acquisition data.

## Development result

All 80 decisions completed: 2 truth families times 10 devices times 4 fixed
policies. There were no numerical failures or excluded devices.

| Truth family | Policy | Approve / reject / insufficient | Decision coverage | False approvals | False rejections | Interval coverage | Runs | Modeled net diagnostic energy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Four-state | Stop now | 2 / 6 / 2 | 8/10 | 0/2 | 0/6 | 9/10 | 2 | 57.99 J |
| Four-state | More thermal tests | 3 / 6 / 1 | 9/10 | 0/3 | 0/6 | 9/10 | 5 | 94.86 J |
| Four-state | Add voltage | 4 / 6 / 0 | 10/10 | 0/4 | 0/6 | 10/10 | 3 | 85.53 J |
| Four-state | Add face temperature | 4 / 6 / 0 | 10/10 | 0/4 | 0/6 | 10/10 | 3 | 85.53 J |
| Extra interface mass | Stop now | 4 / 0 / 6 | 4/10 | 0/4 | N/A (0 rejections) | 10/10 | 2 | 57.99 J |
| Extra interface mass | More thermal tests | 7 / 0 / 3 | 7/10 | 0/7 | N/A (0 rejections) | 10/10 | 5 | 94.86 J |
| Extra interface mass | Add voltage | 7 / 1 / 2 | 8/10 | 0/7 | 0/1 | 10/10 | 3 | 85.53 J |
| Extra interface mass | Add face temperature | 7 / 2 / 1 | 9/10 | 0/7 | 0/2 | 9/10 | 3 | 85.53 J |

Across both truth families, the descriptive decision coverages are 12/20 for
stop now, 16/20 for more thermal tests, 18/20 for voltage, and 19/20 for face
temperature. Face temperature is therefore the strongest fixed policy in this
idealized pilot, with voltage close behind and with the same run count and
modeled energy. The four-run acquisition package spends two more total runs
than either added-sensor policy while resolving fewer devices.

The absence of observed false approvals is not a zero-risk result. The
historical report calculated 35.4% as the upper endpoint of a two-sided 95%
Wilson interval for zero false approvals among seven approvals. That
descriptive endpoint is not a predeclared one-sided risk bound, and the audited
stream dependence removes an independent-sample interpretation from this
cohort. The cohort is small, both truths come from candidate model families, the
verification threshold is uncalibrated, and local covariance propagation can
understate uncertainty for a margin defined by a minimum over time. One of 20
face-policy intervals and one of 20 thermal-policy intervals miss the true
margin even here.

## Reproduction

From the repository root, run the dependency-free numerical report:

```bash
python3 -m thermotwin.operating_decision --no-figure
```

With the optional report dependencies installed, omit `--no-figure` to create
the two-panel policy-outcome figure and JSON/TXT data sidecars. The default run
uses 10 paired devices per truth family, seed 191001, three fit starts, and the
frozen schedules and thresholds above.

The focused checks are:

```bash
python3 -m unittest \
  tests.test_sensor_model_discrimination \
  tests.test_operating_decision
```

## What this changes

The result establishes a working blinded operating-decision benchmark. The
[Stage 3 realism stress test](OPERATING_DECISION_REALISM.md) performs the next
falsification step: it adds uncertain electrical contact resistance, response
lag and thermal mass for the temporary face sensor, and a
temperature-dependent-contact truth family that neither candidate model
matches. The fixed policies are rerun before any selector is built.
