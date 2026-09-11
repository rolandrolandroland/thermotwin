# Operating-decision calibration and selector

Status: Stage 4 calibration and the one-shot fresh-seed rehearsal completed on
2026-09-10. The calibrated artifact is frozen, and the reserved Stage 5 cohort
was not instantiated. This remains a software-only synthetic study, not a
final evaluation or hardware validation.

## Question

Can the Stage 3 operating-decision benchmark be turned into a defensible
measurement-selection procedure after realistic voltage loss, temporary-probe
loading, and a truth family absent from both fitted models exposed failures in
the original local uncertainty intervals?

Stage 4 does two things. It calibrates each complete fixed-policy procedure and
then calibrates a low-capacity decision-directed selector. It also runs one
fresh-seed development rehearsal after those choices are frozen. A separate
Stage 5 cohort remains reserved for the final comparison.

## Frozen partitions

The Stage 3 seed `191001` and its 10 devices per family remain development
data. Stage 4 uses disjoint namespaces:

| Partition | First seed | Paired device blocks per family | Use |
| --- | ---: | ---: | --- |
| Matched-pipeline gate development | `191001` | 10 | Set score retention using the existing Stage 3 cohort. |
| Calibration | `10191001` | 20 | Fit procedure-level interval paddings. |
| Fresh-seed rehearsal | `20191001` | 10 | One no-retuning development check. |
| Reserved Stage 5 evaluation | `30191001` | 50 | Never instantiated in Stage 4. |

One block is a common trial index across the matched four-state,
extra-interface-mass, and temperature-dependent-contact families. All four
fixed policies remain paired within each family row. The implementation
enumerates the observation and device-truth random-number namespaces and
rejects a configuration if any partition overlaps.

## Verification gate

Stage 3 used the arbitrary normalized score cutoff `Q <= 2.0`. Stage 4 derives
a policy-specific threshold from the complete fitted-model pipeline on the
existing Stage 3 development cohort. Each trial block contributes the larger
score from the correct four-state candidate under matched four-state truth and
the correct five-state candidate under extra-interface-mass truth. With 10
blocks and a 90% target, the finite-sample threshold uses the maximum block
score. Family C is excluded from this gate tuning.

The threshold is also floored at a conditional exact-prediction noise reference:
the 99th percentile of the chi-square score implied by the declared Gaussian
white noise and marginalized run offsets, using 50,000 deterministic draws per
policy. That reference alone would not control end-to-end rejection after
fitting; the matched-pipeline development scores provide that calibration. A
parameter-bound hit, nonfinite score, or numerical failure still excludes the
candidate.

This gate calibrates score retention for the two represented matched pipelines.
It is not evidence that a scalar residual score can identify the omitted
temperature-dependent contact law. Family C rejection is reported as a
diagnostic. The margin calibration below carries the missing-family protection
available within this benchmark.

## Blockwise split-conformal margin calibration

For a calibration row with uncalibrated candidate envelope `[L, U]` and hidden
unloaded-device margin `m`, define the outside miss

```text
r = max(L - m, m - U, 0).
```

For each procedure and trial block, Stage 4 takes the maximum `r` over all three
truth-family variants. With 20 blocks and a target of 90% block coverage, the
finite-sample split-conformal rank is

```text
ceil((20 + 1) * 0.90) = 19.
```

The corresponding additive padding `q` turns an emitted interval into
`[L - q, U + q]`. A fit or gate failure with no envelope is treated as an
unbounded set for the procedure-level coverage event and remains insufficient
evidence. A finite envelope that crosses zero is still emitted and scored for
coverage. Neither kind of abstention is dropped from the decision and resource
results. Each fixed policy gets its own `q` because it is a complete
predeclared procedure.

The selector also gets one procedure-level `q`, calibrated after applying its
frozen action rule to every calibration row. This protects the calibration
from the usual post-selection error: it calibrates the output of the rule that
will actually be used rather than borrowing the padding of whichever action
looks favorable later.

## Frozen selector

The selector receives only the common initial `0.8 A`, 20 s acquisition. It
fits both constant-contact candidates and forecasts the unloaded final margin
with their full local covariance.

- If both candidates fit without a numerical failure or bound hit and their
  joint interval envelope is wholly positive or wholly negative, stop.
- Otherwise, request the voltage package.

The rule never receives the truth family, device token, trial index,
verification observations, voltage observations before selecting voltage,
unchosen-policy data, or the final response. The action menu is deliberately
small. Stage 3 already showed that voltage matched face temperature in decision
coverage while transferring better after probe removal, and it beat the
three-run thermal package. Stage 4 therefore tests a transparent stop-or-voltage
rule rather than fitting a flexible policy to 60 calibration rows.

The final approve/reject/insufficient decision is made only after the chosen
acquisition and fixed verification run. The procedure-level conformal padding
is then applied before the final trajectory is scored.

## Declared expected-loss sensitivity

Raw decisions and resource counts remain primary. A dimensionless expected
loss makes the engineering trade explicit:

```text
100 * false approval
+ 20 * false rejection
+ 1 * insufficient evidence
+ w_run * added diagnostic runs
+ w_sensor * added sensors
+ w_energy * incremental energy relative to stop-now.
```

The three frozen scenarios use the same decision weights and vary the added
sensor weight from `0.02` to `0.10` to `0.50`; run and normalized energy weights
remain `0.10`. These values are sensitivity assumptions, not dollar costs.
Every report also exposes the error, abstention, run, energy, and sensor counts
so the scalar loss cannot hide an unsafe trade.

## Results

The full run completed the stages in order: it reran the deterministic Stage 3
gate-development cohort, froze the gates, generated the calibration cohort,
froze the complete artifact, and only then generated the rehearsal cohort. The
artifact records implementation revision
`86205dacbfecec694e4ace233aa974ca739041d1` and protocol digest
`0c87ccfdbec5b1047490a3e2408b2a599cfa71687f18ef55076724108e5b3b68`.
It is saved in
[`OPERATING_DECISION_CALIBRATION_ARTIFACT.json`](OPERATING_DECISION_CALIBRATION_ARTIFACT.json).

### Frozen gate and padding

All gate thresholds retained the correct fitted candidate in 10/10 matched
development blocks. The threshold is the larger of that finite-sample maximum
and the predeclared conditional-noise reference.

| Procedure | Gate threshold | Additive padding | Calibration finite-interval coverage, raw | Calibration finite-interval coverage, padded | Calibration block procedure-set coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stop now | 1.2760 | 0.0171 K | 54/56 (96.4%) | 55/56 (98.2%) | 19/20 (95%) |
| More thermal tests | 1.2768 | 0.0157 K | 54/58 (93.1%) | 57/58 (98.3%) | 19/20 (95%) |
| Add voltage | 1.2250 | 0.0669 K | 43/53 (81.1%) | 52/53 (98.1%) | 19/20 (95%) |
| Add face temperature | 1.2264 | 0.0521 K | 45/56 (80.4%) | 55/56 (98.2%) | 19/20 (95%) |
| Decision-directed selector | chosen action's gate | 0.0669 K | 44/53 (83.0%) | 52/53 (98.1%) | 19/20 (95%) |

These are calibration-cohort diagnostics, not independent performance
estimates. The 19/20 block result follows from the selected conformal order
statistic. The finite-interval columns condition on an interval being emitted
and are not the calibrated target.

### One-shot fresh-seed rehearsal

The table below pools 30 family rows from 10 paired blocks. The three family
rows inside a block share the trial index, so pooled row-level confidence
intervals are intentionally suppressed. Procedure-set coverage is calculated
at the 10-block level.

| Procedure | Approve / reject / insufficient | Decision coverage | Padded finite-interval coverage | Block procedure-set coverage | Mean runs | Mean energy | Added sensors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stop now | 3 / 7 / 20 | 10/30 (33.3%) | 25/26 (96.2%) | 9/10 (90%) | 2.00 | 60.72 J | 0.00 |
| More thermal tests | 3 / 9 / 18 | 12/30 (40.0%) | 27/27 (100%) | 10/10 (100%) | 5.00 | 99.29 J | 0.00 |
| Add voltage | 5 / 12 / 13 | 17/30 (56.7%) | 26/26 (100%) | 10/10 (100%) | 3.00 | 89.53 J | 1.00 |
| Add face temperature | 7 / 7 / 16 | 14/30 (46.7%) | 23/24 (95.8%) | 9/10 (90%) | 3.00 | 89.41 J | 1.00 |
| Decision-directed selector | 5 / 11 / 14 | 16/30 (53.3%) | 26/26 (100%) | 10/10 (100%) | 2.80 | 83.77 J | 0.80 |

No false approval or false rejection was observed among the determinate
rehearsal decisions. For the selector specifically, 0/5 approvals and 0/11
rejections were wrong, while 14/30 rows remained insufficient. The sample is
too small to interpret zero observed errors as a tight error-rate bound:
several family-specific Wilson intervals remain wide because they contain only
one to nine approvals or rejections.

The selector stopped after the common acquisition in 6/30 family rows and
requested voltage in 24/30. Relative to always adding voltage, it saved 0.20
runs, 5.76 J, and 0.20 added sensors per row, while producing one fewer
determinate decision. It tied voltage on padded finite-interval coverage and
block procedure-set coverage. Its empirical mean loss tied voltage under the
bench-time-dominant weights (0.601 each), was slightly lower under the balanced
weights (0.665 versus 0.681), and was lower under the
instrumentation-expensive weights (0.985 versus 1.081). Stop-now had the lowest
instrumentation-expensive loss at 0.667, so the preferred procedure changes
with the declared resource values.

The central selector hypothesis is therefore not established by Stage 4. The
selector achieved 53.3% decision coverage, below the experiment outline's 70%
development goal and slightly below fixed voltage at 56.7%. Its resource
savings are real within this rehearsal but small, and the balanced-loss
advantage over stop-now is only 0.002. On temperature-dependent-contact truth,
the selector approved 3/5 true passes, issued no rejection, and abstained on
all five violations plus two passes. The calibrated gate rejected 45/80 Family
C fixed-policy candidate fits, compared with none under the Stage 3 cutoff,
but this is only a misspecification diagnostic and does not establish that the
gate detects the omitted law.

Stage 4 supports a narrower conclusion: blockwise calibration converted the
local model intervals into a procedure whose observed rehearsal block coverage
met the 90% target while preserving abstentions in the decision metric. It did
not show that the initial-data selector materially improves the decision and
resource frontier. The reserved Stage 5 comparison is still required and must
remain untouched until the procedure to evaluate is frozen. The single
mismatch-guard revision and final protocol are specified in
[`OPERATING_DECISION_FINAL_EVALUATION.md`](OPERATING_DECISION_FINAL_EVALUATION.md).

## Reproduction

Run the dependency-free numerical report from the repository root:

```bash
python3 -m thermotwin.operating_decision_calibration \
  --workers 3 --no-figure \
  --source-revision 86205dacbfecec694e4ace233aa974ca739041d1 \
  --report-output /tmp/thermotwin-stage4-full-report.txt \
  --artifact-output thermotwin/OPERATING_DECISION_CALIBRATION_ARTIFACT.json
```

For a quick integration check, use a coverage target that has a finite
split-conformal order statistic at the reduced block count:

```bash
python3 -m thermotwin.operating_decision_calibration \
  --calibration-blocks 2 --rehearsal-blocks 1 \
  --gate-blocks 2 --gate-target-retention 0.66 \
  --target-coverage 0.66 --fit-iterations 1 \
  --gate-draws 1000 --workers 1 --no-figure
```

The focused checks are:

```bash
python3 -m unittest \
  tests.test_operating_decision_calibration \
  tests.test_operating_decision_realism
```

## Interpretation boundary

The split-conformal statement is finite-sample calibration under the declared
synthetic generator and block-exchangeability assumption. It is not a hardware
confidence guarantee, and it does not cover a new discrepancy family outside
the calibration distribution. The rehearsal is development evidence. Its
results cannot update the saved gate, paddings, or selector, and it is not a
substitute for the reserved Stage 5 evaluation. The 90% target applies to the
procedure-set block event, with only a missing envelope interpreted as an
unbounded set. A finite zero-crossing interval remains scored. Coverage
conditional on emitting a finite interval is reported separately as an
untargeted diagnostic.
