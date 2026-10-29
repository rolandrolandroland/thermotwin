# Operating decision abstention diagnosis

Date: 2026-10-29. Status: completed exploratory analysis of the closed Phase F
cohort. The [frozen primary result](OPERATING_DECISION_PHASE_F_RESULT.md)
remains unchanged and did not meet its success criterion.

The main diagnostic finding is that **narrower uncertainty often failed to
resolve the operating decision**. Of 125 selected measurements, 116 narrowed
the development-padded interval relative to the common initial envelope;
85 of those cases still abstained after final calibration. This supports
testing a decision-resolution objective in a future experiment. It does not
establish that a redesigned selector would succeed.

The analysis uses the existing 100 paired blocks and all 300 cases. It performs
no refitting, physical simulation, new acquisition, threshold search, or cohort
extension. All comparisons below are post hoc and descriptive.

## Where the 137 abstentions arise

The layers below identify the first interval that crosses the zero-margin
decision boundary, keeping the originally selected action fixed. Development
offsets expand both endpoints before the separate final calibration correction.
Approval requires a lower endpoint at least zero; rejection requires an upper
endpoint strictly below zero.

| Selected action | Cases | Definitive | Raw interval already crosses zero | First crosses after development padding | First crosses after final calibration | Total abstentions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stop | 175 | 131 | 23 | 0 | 21 | 44 |
| Thermal | 74 | 9 | 60 | 0 | 5 | 65 |
| Voltage | 30 | 18 | 4 | 7 | 1 | 12 |
| Face temperature | 21 | 5 | 11 | 5 | 0 | 16 |
| All | 300 | 163 | 98 | 12 | 27 | 137 |

All 137 abstentions have finite, verified final intervals. There are no pipeline,
verification, or missing-interval failures. Of the 98 raw crossings, 95 include
at least one individual candidate interval that crosses zero. Only three arise
solely from combining opposing, individually definitive model intervals.
Seven abstentions lose an initially admissible model at the numerical-fit stage;
two recover a previously excluded model. These are descriptive candidate
transitions, not whole-case failures.

The two stopping paths have different meanings. In 152 cases the acquisition-only
envelope provisionally resolved the decision, so the selector stopped and
verified. Final calibration returned 21 of those cases to abstention. In the
other 23 stopping cases, the initial envelope was unresolved and no acquisition
met the frozen value threshold. All 23 remained unresolved.

| Truth family | Raw crossing | Development padding | Final calibration | Total abstentions / cases |
| --- | ---: | ---: | ---: | ---: |
| Matched four state | 25 | 5 | 11 | 41/100 |
| Extra interface mass | 36 | 3 | 6 | 45/100 |
| Temperature dependent contact | 37 | 4 | 10 | 51/100 |

Abstentions occur across all three families. Their median absolute true margin
is 0.106 K, compared with 0.331 K among definitive cases. Truth is available only
for this retrospective diagnosis; it was unavailable to the selector. The
one-candidate acquisition stratum abstains in 14/38 cases and the two-candidate
stratum in 123/262. These groups contain different cases and family mixtures.

## Predicted and realized measurement benefit

Both reductions use the same initial acquisition envelope, including the frozen
stop development offset. Predicted width is the saved N=16 forecast, including
action offsets, whole-draw failure imputation, candidate-loss penalties, and the
worst mean across source models. Observed width is the retained verified action
envelope plus its development offset. Neither includes final calibration q in
this width comparison; q still governs every reported final decision.

| Selected measurement | Cases | Mean predicted width reduction (K) | Mean observed width reduction (K) | Observed narrowing | Narrowed but final decision abstains |
| --- | ---: | ---: | ---: | ---: | ---: |
| Thermal | 74 | 0.0968 | 0.1300 | 68 | 59 |
| Voltage | 30 | 0.1215 | 0.1959 | 27 | 10 |
| Face temperature | 21 | 0.1662 | 0.2804 | 21 | 16 |
| All selected measurements | 125 | 0.1144 | 0.1711 | 116 | 85 |

Every selected measurement had positive predicted value. Nine produced no
observed narrowing, including negative gains, which remain in these averages.
Of 93 measured cases that ultimately abstained, 85 nevertheless narrowed the
interval. Thermal selection accounts for 65 of the 93 abstentions, including
59 with observed narrowing. Action groups differ in difficulty; their decision
rates do not estimate a causal advantage of one measurement over another.

Mean observed reduction exceeded the forecast by 0.0567 K; the mean absolute
width difference was 0.0864 K. This comparison does **not** show average
overprediction of width benefit. Forecasts precede independent verification,
whereas observed envelopes include its candidate filtering. Forecasts also
contain conservative candidate-loss floors and worst-source aggregation, and
each action has only one realized observation set per case. These differences
prevent interpreting the table as a calibrated forecast-error test.

The pattern persists in the 113 selected measurement cases with the same initial
and final verified model set: 107 narrow, 84 abstain, and mean predicted/observed
reductions are 0.1165/0.1726 K. The archived detail retains all 900 case/action
comparisons, including actions that were not selected; selection does not censor
the descriptive forecast comparison. Every predictive draw remains in its
source denominator. For the 125 selected measurements, the 3,792 retained draws
include two whole-draw failures and 422 draws with candidate loss; these are
different events and keep their original scoring conventions.

## What the other saved procedures could resolve

The four fixed procedures were already run on every paired case. Among the
137 selector abstentions, at least one fixed procedure was definitive in
51 cases; all four fixed procedures also abstained in the other 86.

| Fixed procedure | Selector abstains and fixed is definitive | Wrong fixed decisions among those cases | Selector is definitive and fixed abstains |
| --- | ---: | ---: | ---: |
| Stop | 0 | 0 | 45 |
| Thermal | 28 | 0 | 20 |
| Voltage | 41 | 0 | 22 |
| Face temperature | 31 | 0 | 29 |

These counts overlap across procedures and must not be added. Each fixed
procedure retains its own calibrated correction. Four thermal acquisitions
produce a definitive fixed-thermal decision but an abstaining selector decision
because their procedure-specific corrections differ. Conversely, the selector's
smaller correction makes 13 stopped acquisitions and one voltage acquisition
definitive where the corresponding fixed procedure abstains.

Of the 51 cases with a definitive fixed alternative, 24 follow a selected
measurement, nine follow stopping for insufficient predicted value, and 18
follow provisional resolution then stopping. This identifies several possible
selection and stopping questions for future development. Choosing the
successful alternative after seeing every outcome is not an executable policy;
the counts do not establish attainable gains or calibrated risk for a redesign.

## Why removing padding is not a demonstrated repair

The following diagnostic removes padding only from the final interval
classification, retaining the same frozen action choices and verification.
The first two rows have no final calibrated guarantee.

| Interval used to classify the same selected cases | Definitive / 300 | Approvals | False approvals |
| --- | ---: | ---: | ---: |
| Raw verified interval without either padding layer | 202 (67.3%) | 126 | 2 |
| Development padding only | 190 (63.3%) | 121 | 1 |
| Frozen complete calibrated interval | 163 (54.3%) | 104 | 1 |

All three have zero false rejections. Even the raw intervals fall below the
70% observed decision rate, before applying the required confidence lower
bound. Removing both padding layers also exposes another false approval.
This does not authorize changing the frozen corrections or choosing an easier
threshold from reserved outcomes.

## Implication for the next experiment

The [current figures](OPERATING_DECISION_FINAL_FIGURES.md) are complete.
Complete the final report and reproducibility package with the primary
non-success intact. The present diagnosis makes a
decision-aware acquisition objective a reasonable hypothesis for a new version:
value an action for its expected chance of producing a reliable final decision
per cost, accounting for the location of the interval relative to zero as well
as its width. A future development protocol should also examine provisional
stopping that becomes unresolved after conservative padding.

Any such comparison needs a new development protocol, fresh development and
calibration evidence, an explicit precision and compute feasibility review,
and a new unseen reserved cohort. The current exposed cases can motivate the
hypothesis and illustrate failure modes; they cannot confirm its performance.
The 86 cases unresolved by every fixed procedure also make a universal rescue
claim unsupported by the available action menu.

## Evidence and reproduction

The [compact diagnosis artifact](../../thermotwin/OPERATING_DECISION_PROSPECTIVE_ABSTENTION_DIAGNOSIS.json)
contains all summary tables, source/evidence hashes, and a digest of the detailed
300-case record. The detailed record remains ignored at
`.artifacts/operating-decision/phase-f-abstention-cases-20261029.json`.
The [analysis tool](../../tools/analyze_prospective_phase_f_abstentions.py)
uses only the standard library and the existing independent archive auditor.
It reconstructs all 6,000 saved outcomes, validates the 100 raw block hashes and
300 before-reveal records, verifies all 117 frozen numerical dependencies, and
independently aggregates all 900 N=16 action forecasts. Exact outward rounding
determines the interval-layer classification.

All 35 focused diagnosis, independent-audit and Phase F boundary checks passed.
A fresh extraction reproduced the summary and all detailed cases byte for byte;
a separate exact-fraction calculation reproduced the abstention-layer counts.

| Evidence identity | Value |
| --- | --- |
| Frozen numerical source | `7e3b9a3be22d991ec2e588cbd4a53a9d4d2d27d7` |
| Scientific result digest | `d565c0c6368b602eeb3d2721c128aab88221bce9a6b5abe68fcda4c727d4edf0` |
| Evaluation JSON SHA-256 | `299f75ad492e5bde0acc724ed4e72ae029c86f2c05538677c2a856c23734c564` |

With the retained raw evidence available, rerun from the repository root using
new output paths outside the raw archive:

```sh
python -m tools.analyze_prospective_phase_f_abstentions \
  --archive .artifacts/operating-decision/prospective-phase-f-reserved-7e3b9a3-v1/evaluation.json \
  --source-root . \
  --validation-record .artifacts/operating-decision/phase-f-durable-reserved-7e3b9a3-20261018/independent-validation.json \
  --recorded-date 2026-10-29 \
  --output /tmp/thermotwin-phase-f-abstention-summary.json \
  --case-output /tmp/thermotwin-phase-f-abstention-cases.json
```

The accepted Phase F result artifact remains the historical October 29 result
snapshot, including its then-pending closeout flags. The
[project status](OPERATING_DECISION_PROJECT_STATUS.md) records this subsequent
diagnosis and the work still outstanding.
The subsequent [bounded numerical replay](OPERATING_DECISION_PHASE_H_REPLAY.md)
has also passed; it does not change this diagnosis or establish a redesigned
selector's performance.
