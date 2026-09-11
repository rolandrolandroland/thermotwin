# Mismatch-guarded selector and reserved final evaluation

Status: protocol and software implemented on 2026-09-11. The new guard and
procedure calibration have not yet been run, and the long-reserved Stage 5
cohort has not been instantiated.

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

## Interpretation boundary

This remains synthetic evidence under the declared generator and
block-exchangeability assumption. Calibration does not supply a hardware
guarantee. The acquisition score is fit on the same data it scores, so its
matched-family threshold is empirical and specific to the complete fitting
pipeline. A successful Family C result would show useful detection only for the
declared temperature-dependent-contact distribution. It would not establish a
generic out-of-distribution detector or protection against a new physical law.
