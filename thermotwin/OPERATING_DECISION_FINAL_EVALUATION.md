# Mismatch-guarded selector and reserved final evaluation

Status: completed on 2026-09-12 and superseded for inference by the subsequent
operating-decision audit. Guard development and procedure recalibration were
frozen before the reserved cohort was generated, and the saved arithmetic is
correct. However, deterministic random-stream reuse across runs and adjacent
device blocks invalidates the exchangeability and independent-bootstrap
interpretations. The Stage 5 counts and point losses below remain descriptive;
their confidence intervals do not support population claims. See
[`OPERATING_DECISION_AUDIT.md`](../docs/thermotwin/OPERATING_DECISION_AUDIT.md).

## Question

Can the common initial acquisition recognize when neither constant-contact
digital twin explains the device well enough to justify another voltage test?

Stage 4 showed that the stop-or-voltage selector often paid for voltage and
still abstained under temperature-dependent contact. This final cycle makes one
selector revision. It adds a generic acquisition lack-of-fit guard that can
abstain immediately. It leaves the two fitted model families, the unflagged
stop-or-voltage rule, verification gates, action packages, final operating
schedule, and loss weights unchanged.

The revision is deliberately conservative. It can remove a decision and save a
diagnostic run; it cannot introduce a decision that the Stage 4 selector would
not make. Its purpose is to detect futile measurement within the declared
synthetic benchmark, not to claim that software can identify arbitrary omitted
physics.

## Why the missing law is not added

The temperature-dependent-contact generator remains absent from the fitted
model set. Adding that exact law after seeing the Stage 4 failure would turn the
stress case into an in-model estimation exercise. Fresh device seeds would not
remove that family-level overfitting. A defensible model-repair experiment
would require a new thermal contrast, a wider constitutive class, and another
unseen discrepancy family; it is a separate experiment.

## Single frozen revision

Both constant-contact candidates are fit to the common `0.8 A` acquisition.
For each fitted candidate, the acquisition score removes the parameter-prior
terms from the fitted objective, retains the two profiled run-bias penalties,
and divides by the number of observed temperature samples:

```text
Q_j = (total fitted residual sum of squares - prior sum of squares)
      / observed temperature sample count

G = min(Q_four-state, Q_five-state).
```

The minimum asks whether at least one member of the declared model set explains
the acquisition. A missing fit, bound hit, or nonfinite score assigns infinity
to that candidate. The guard alarms when neither candidate retains a finite
score below the threshold.

The action rule is:

1. If `G` exceeds the frozen acquisition threshold, return insufficient
   evidence immediately after the common acquisition.
2. Otherwise, apply the unchanged Stage 4 rule: stop when the two acquisition
   intervals form a finite one-sided envelope; request voltage when they do
   not.
3. Apply the existing policy-specific verification gate to an unflagged chosen
   action, then apply the revised procedure padding before the final decision.

The selector input contains fit results from the common acquisition only. It
does not contain a truth-family label, device or trial identifier, later
observations, verification scores, the selected action's measurements, or the
hidden final response.

## Frozen partitions and chronology

| Partition | First seed | Paired blocks per family | Purpose |
| --- | ---: | ---: | --- |
| Guard development | `50191001` | 20 | Calibrate acquisition lack of fit on matched four-state and extra-interface-mass rows. |
| Procedure recalibration | `60191001` | 20 | Calibrate the complete revised procedure across all three families. |
| Reserved Stage 5 | `30191001` | 50 | One final paired evaluation across all three families. |

These namespaces are disjoint from Stages 1–4 and from each other, including
device-property, truth-law, observation-noise, and run-offset draws.

Chronology is enforced in separate commands:

1. Generate guard-development Families A and B.
2. Freeze the 90% matched-block guard threshold at rank 19/20. Each block score
   is the larger best-candidate score across those two families.
3. Generate Family C only after that threshold freezes. Its alarm rate is a
   diagnostic and cannot alter the rule.
4. Generate the new three-family calibration partition and fit one blockwise
   split-conformal padding at rank 19/20.
5. Floor that padding at the Stage 4 selector padding of `0.0669077 K`. This
   makes every revised definitive decision a decision the Stage 4 selector
   would also make with the same sign.
6. Serialize and commit the complete revised artifact.
7. In a new process, load that artifact and only then generate the reserved
   Stage 5 cohort.

The artifact also freezes the three loss scenarios and the paired bootstrap's
`20,000` draws and seed `70191001`. Its loader recomputes the parent Stage 4
digest, verifies both block order statistics and the Stage 4 padding floor, and
rejects any protocol-field mismatch before Stage 5 generation.

## Frozen calibration result

The matched-family rank 19/20 threshold is `G = 1.171965`. After it froze, the
guard alarmed on 1/20 matched four-state devices, 0/20 extra-interface-mass
devices, and 1/20 temperature-dependent-contact devices. Both alarms replaced
a Stage 4 voltage action, but neither voltage action would have ended in
insufficient evidence. This development diagnostic shows little separation of
the declared contact mismatch from matched devices and no evidence of the
intended futile-measurement triage mechanism.

The three-family recalibration rank 19/20 was `0.084003 K`, above the Stage 4
floor, and the frozen revised padding is therefore `0.084003 K`. It covered
19/20 procedure-set blocks. On this descriptive calibration cohort, the revised
selector reduced mean runs from `2.68` to `2.57` and energy from `80.41 J` to
`76.84 J`, while definitive-decision coverage fell from 58.3% to 46.7% and
balanced empirical loss rose from `0.586` to `0.690`.

These development results were unfavorable, but they did not alter the frozen
rule. The protocol did not declare a calibration-cohort futility stop, so the
reserved comparison proceeded without changing the threshold, padding, model
set, loss weights, or success criteria.

## Reserved Stage 5 result

The evaluator loaded the committed artifact before generating 50 paired blocks
per family from reserved seed `30191001`. It produced 900 procedure outcomes.
No threshold, selector branch, padding, model, loss weight, bootstrap setting,
or success criterion changed after reveal.

Across all 150 family rows, the Stage 4 selector made 96 definitive decisions
and the mismatch-guarded selector made 85. Neither produced a false approval or
false rejection. The revised procedure retained 49/50 blockwise procedure-set
coverage, but its small resource saving came with too much abstention:

| Metric | Stage 4 selector | Revised selector | Change |
| --- | ---: | ---: | ---: |
| Definitive-decision coverage | 64.0% (96/150) | 56.7% (85/150) | -7.3 percentage points |
| Blockwise procedure-set coverage | 98.0% (49/50) | 98.0% (49/50) | 0 points |
| Mean diagnostic runs | 2.61 | 2.55 | -0.06 |
| Mean diagnostic energy | 78.39 J | 76.56 J | -1.83 J |
| Mean added sensors | 0.61 | 0.59 | -0.02 |
| Balanced empirical loss | 0.512 | 0.579 | +0.067 |

The guard alarmed on 1/50 matched four-state devices, 0/50
extra-interface-mass devices, and 4/50 temperature-dependent-contact devices.
Only four of the five alarms replaced a voltage action, and only two of those
four voltage actions would have ended in insufficient evidence. Thus 2/5 of
all alarms, or 2/4 conditional on replacing voltage, represented the intended
futile-measurement triage. Both readings miss the 75% criterion.

The primary paired balanced-loss difference, revised minus Stage 4, was
`+0.0667`. Bench-time-dominant loss was `+0.0689`, and
instrumentation-expensive loss was `+0.0561`. The originally reported
block-bootstrap intervals reproduce arithmetically, but the audit found that
adjacent blocks share random draws. They therefore cannot support the stated
confidence interpretation. The point estimates descriptively favor the Stage
4 selector in this exposed cohort.

| Predeclared criterion | Stage 5 observation | Result |
| --- | --- | --- |
| No new wrong definitive decision | No false approvals or false rejections; monotonic audit passed | Pass |
| At least 90% block coverage | 98.0% (49/50) | Pass |
| At most 5-point decision-coverage loss | 7.3-point loss | Fail |
| Lower mean runs and energy | 0.06 fewer runs and 1.83 J less | Pass |
| At least 75% useful guard alarms | 40% of all alarms; 50% among voltage-replacing alarms | Fail |
| Balanced-loss interval wholly below zero | Descriptive point difference is positive; interval interpretation invalidated by the audit | Fail |

The acquisition residual score is therefore rejected as the selector revision
for this exposed benchmark. Stage 4 remains a benchmark with a favorable
descriptive tradeoff under some declared weights; it is not a scientifically
validated operating recommendation. The Stage 4 selector also missed the
outline's 70% decision-coverage target, and stop-now had the lowest descriptive
loss when instrumentation was expensive. The reserved cohort is final and must
not be used to tune a replacement.

A missing envelope remains an unbounded set for the conformal coverage event
and remains insufficient evidence for decision coverage. A finite interval
crossing zero is still emitted and scored.

## Final comparison

Stage 5 evaluates all four fixed policies, the Stage 4 selector, and the revised
selector on the same 50 device blocks and three truth variants. Primary raw
outputs are false approvals, false rejections, insufficient outcomes,
definitive-decision coverage, diagnostic runs, energy, added sensors, and
blockwise procedure-set coverage.

The primary scalar contrast is the paired block-level balanced-loss difference

```text
loss(revised selector) - loss(Stage 4 selector),
```

where the three family rows are averaged inside each block. A deterministic
20,000-draw block bootstrap reports its 95% interval. The bench-time and
instrumentation-expensive weights remain sensitivity analyses.

The simulator does not separately time acquisition fitting and later
verification work. For an early guard abstention, the saved computation time is
therefore the full stop-trial time as a conservative upper bound. Computation
time is excluded from the resource claim; runs and energy use the actual
one-acquisition path.

Evidence for useful triage requires all of the following:

- no new wrong definitive decision relative to the Stage 4 selector;
- at least 90% observed procedure-set block coverage;
- no more than a five-percentage-point loss of definitive-decision coverage;
- lower mean diagnostic runs and energy; and
- at least 75% of guard alarms replacing a voltage action that would have ended
  in insufficient evidence.

The result is statistically persuasive on the declared balanced loss only if
the paired bootstrap interval lies wholly below zero. Otherwise it is reported
as inconclusive or negative, regardless of the point estimate.

## Reproduction

Freeze the guard and revised calibration without touching Stage 5:

```bash
python3 -m thermotwin.operating_decision_final_evaluation freeze \
  --workers 3 \
  --source-revision IMPLEMENTATION_COMMIT \
  --artifact-output thermotwin/OPERATING_DECISION_FINAL_ARTIFACT.json \
  --report-output /tmp/thermotwin-stage5-freeze-report.txt
```

After the artifact is reviewed and committed, run the reserved evaluation in a
new process:

```bash
python3 -m thermotwin.operating_decision_final_evaluation evaluate \
  --workers 3 \
  --artifact thermotwin/OPERATING_DECISION_FINAL_ARTIFACT.json \
  --result-output thermotwin/OPERATING_DECISION_FINAL_RESULT.json \
  --report-output /tmp/thermotwin-stage5-final-report.txt
```

Both commands refuse to overwrite an existing artifact, result, or report, and
they reject output paths that alias an input artifact. The protocol designates
the reserved cohort for one final run; this protection prevents accidental
re-execution of the declared command without claiming an irreversible data
store.

These commands document the completed run. The saved Stage 5 result is the
inferential record; rerunning its seed must not be treated as new confirmation.

## Interpretation boundary

This remains synthetic evidence under the declared generator and
block-exchangeability assumption. Calibration does not supply a hardware
guarantee. The acquisition score is fit on the same data it scores, so its
matched-family threshold is empirical and specific to the complete fitting
pipeline. A successful Family C result would show useful detection only for the
declared temperature-dependent-contact distribution. It would not establish a
generic out-of-distribution detector or protection against a new physical law.
