# Operating-decision calibration and selector

Status: Stage 4 implementation complete; the full calibration and fresh-seed
rehearsal are pending execution. This remains a software-only synthetic study,
not a frozen final evaluation or hardware validation.

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

The implementation completes the stages in order: rerun the deterministic
Stage 3 gate-development cohort, freeze the gates, generate the calibration
cohort, freeze the full artifact, and only then generate the rehearsal cohort.
The full result table will be inserted here after the frozen 20-block
calibration and 10-block rehearsal complete.

## Reproduction

Run the dependency-free numerical report from the repository root:

```bash
python3 -m thermotwin.operating_decision_calibration \
  --workers 3 --no-figure
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
