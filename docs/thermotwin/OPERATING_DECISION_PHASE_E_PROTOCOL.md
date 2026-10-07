# Phase E: analysis freeze and independent calibration

Declared: 2026-10-06. Version: `operating_decision_prospective_phase_e_v1`.
Status: implementation and controlled tests complete; the committed-source CI,
disposable numerical replay, resource probe, and generation gate must pass
before calibration opens. No calibration or reserved case was generated to
design this protocol.

This adopts the Phase E recommendations in the
[completion plan](OPERATING_DECISION_COMPLETION_PLAN_2026_09_17.md) and the
[audit repair plan](THERMOTWIN_AUDIT_REPAIR_PLAN_2026_09_26.md). The
[final Phase D artifact](../../thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_D_FINAL_FREEZE.json)
is the predecessor. Its passing D6 internal check closes development; it does
not itself authorize another scientific partition.

## E1 — fixed analysis and precision

The primary question is whether the selector reduces realized diagnostic
energy while retaining acceptable decision quality against every qualifying
fixed alternative. Retain all four fixed comparators: stop, thermal, voltage,
and face temperature. Each uses the same paired devices, fitting protections,
verification standard, and independent calibration standard. Energy refers to
the simulated diagnostic acquisition and verification schedules; report the
software computation, schedule duration assumptions, instruments, and runs
separately. These are simulation results, not hardware validation or financial
savings.

The primary scenario remains nominal sensing and `balanced_face_equal` costs.
The frozen selector retains N=16, at most one unstable predictive draw per
source/action, zero stopping clearances, zero minimum reduction, minimum
utility 0.025, and action order stop/thermal/voltage/face. Development offsets
are 0.000/0.000/0.074/0.098 K. No Phase E correction enters selection, stopping,
predictive gains, fitting, verification, or sensor/cost choices. The D5 maps
remain development sensitivity evidence.

The adopted primary targets and project noninferiority margins are:

| Quantity | Requirement |
| --- | --- |
| False approvals conditional on approval | Upper confidence bound at most 10% |
| Definitive decisions among all cases | Lower confidence bound at least 70% |
| Decision coverage lost versus a fixed policy | Upper confidence bound at most 5 percentage points |
| False-approval risk increase versus a fixed policy | Upper confidence bound at most 2 percentage points |
| Realized diagnostic energy saved versus a fixed policy | Lower confidence bound strictly positive |

Interval coverage and conditional approval risk are different endpoints.
90% interval coverage alone does not establish the approval-risk target.

### Dependence, inference, and multiplicity

One block contains all three truth families. Blocks are assumed independent
and identically distributed under the frozen synthetic generator. Dependence
between families and procedures within a block is allowed. A family has one
case per independent block; 100 blocks are never treated as 300 independent
observations.

Adopt a deliberately conservative set of 73 one-sided primary bounds with
Bonferroni alpha `0.05/73 = 0.0006849315068493151` per bound:

- 30 exact Clopper–Pearson bounds: lower and upper conditional approval risk
  for five procedures and three families.
- 15 exact lower definitive-decision coverage bounds, one per procedure/family.
- 24 paired coverage bounds: upper fixed-only decision probability and lower
  selector-only decision probability, for four comparators and three families.
- Four lower energy-saving bounds, on paired block mean differences.

For pooled decision coverage, average the three family lower bounds. For
conditional approval risk in the equal-family population mixture, use the
maximum family risk upper bound and minimum family risk lower bound. Population
approval weights are unknown and may differ between procedures; empirical
approval weights are not substituted into a claimed exact bound. These extrema
are valid conservative mixture bounds. An empty family approval denominator
has N/A risk bounds and prevents qualification, rather than supplying a zero
error bound.

For paired coverage loss, average across families the upper fixed-only
probability minus the lower selector-only probability. For risk increase use
the selector mixture upper bound minus the fixed mixture lower bound. These
methods remain conservative when within-block outcomes are dependent. Zero
observed errors never produce a zero-width risk interval.

For energy use the mean of the three within-block fixed-minus-selector energy
differences, then a one-sided Student t lower bound across the 100 independent
block means. This has an approximate normal-sampling interpretation. The
Bonferroni allocation covers the declared primary directions; it does not
turn approximate energy inference into a finite-sample distribution-free
guarantee. Report that assumption with any positive energy claim. Other rates
and the secondary curves are descriptive and do not acquire this simultaneous
confidence interpretation.

Every comparator is reported individually, including voltage and face. A fixed
policy qualifies only if both its risk upper bound and coverage lower bound
pass. Report its observed energy/coverage/risk frontier among qualifying fixed
policies, but require primary comparisons against **every qualifying fixed
policy**, including dominated policies. This prevents choosing an easier
comparator after labels are seen.

Primary success requires the selector to qualify, at least one fixed policy
to qualify, and all three energy/coverage/risk comparison checks to pass
against every qualifying fixed policy. An empty qualifying set supplies no
strongest-fixed-strategy claim. Report named checks and narrower evidence when
only a subset passes; do not relabel it primary success.

### Denominators and secondary threshold curves

Retain every case and every failed procedure. Report false approvals/approvals;
false rejections/true passes; missed violations/true violations; decision
errors/definitive decisions; definitive decisions/all cases; abstentions/all
cases; pipeline and verification failures/all cases; and interval coverage/all
cases. A missed violation includes an abstention on a violating device, so it
is broader than false approval. A missing or invalid interval is uncovered.
Simultaneous interval block coverage requires all three family intervals to
cover. An empty rate denominator is null/N/A.

An incomplete realized-energy record makes the affected paired energy contrast
N/A and prevents an energy claim. No case is dropped and no unobserved energy
is replaced by a favorable value. Report recorded and required resource counts.

The secondary final-decision clearance grid is 0.00, 0.05, 0.10, and 0.20 K.
For each saved final interval, approve when its lower bound is at least the
clearance and reject when its upper bound is strictly below minus the
clearance; otherwise abstain. This reclassifies saved intervals only. It does
not change actions, fit outputs, corrections, verification, or energy. The
0.00 K rule is the sole headline threshold; the curves are descriptive, with
no threshold selection from reserved labels.

### Precision and compute decision

Retain **100 calibration blocks and 100 reserved blocks**. Each supplies 300
family cases and five paired procedures. No reactive extension is permitted.

The committed
[precision artifact](../../thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_E_PRECISION.json)
uses only the hash-authenticated closed D6 archive. It preserves all four
paired block energy contrasts, development approval and decision counts,
projected bounds, family decision-target power, and approximate energy power.
Only ten development blocks estimate these rates and variances. The
pre-correction projections are optimistic for decision coverage: independent
padding can turn definitive decisions into abstentions.

Under the adopted simultaneous bounds, even zero errors require 70 approvals
per family to put a 10% upper bound on risk, and 361 approvals per family to
put a 2% upper bound on risk when the fixed-policy lower bound is zero. The
selector's D6 rates project only 10/50/40 approvals in Families A/B/C at 100
blocks. Thus the intended 2-point risk comparison is explicitly underpowered;
the conservative absolute risk target is also unlikely to be established.
At the observed 10% Family A approval rate, those zero-error counts would
require approximately 700/3610 blocks. That exceeds the chosen compute scope.
The attainable report may establish calibration feasibility and describe
energy/coverage tradeoffs while remaining inconclusive on decision-quality
promotion. It must not advertise confirmed approval safety or strongest-fixed
superiority without the frozen bounds actually passing.

The development paired energy saving estimates are -8.446 J versus stop,
30.313 J versus thermal, 20.518 J versus voltage, and 20.397 J versus face.
Their projected 100-block lower bounds are -11.416/27.494/17.665/17.549 J under
the declared approximate sampling model. The energy projections do not settle
decision quality and do not guarantee reserved results.

D6 recorded 61,159.175 block CPU seconds for ten blocks. Linear scaling gives
about 170 CPU hours per 100-block partition and an ideal four-worker duration
of 42.5 hours. Plan roughly 43–60 active wall hours per partition, with
interruption and slower cases able to extend elapsed time. Raw block archives
project to about 367 MB per partition; the final calibration summary indexes
the raw files instead of copying all blocks into another large in-memory
archive. A 2 GiB disk budget includes the disposable storage probe. Keep the
4 GiB process-tree memory limit and the pinned Phase D runtime.

## E2 — independent calibration construction

For each procedure and case, retain its raw final interval `[L,U]`, chosen
action, verification status, failure status, and that action's frozen
development offset `d_a`. After the original procedure records and selector
choice are saved, reveal the true final margin `m`. Use:

`s = max(0, L - d_a - m, m - (U + d_a))`.

Evaluate the arithmetic exactly on the stored binary inputs, round a positive
score upward only when its float conversion would round down, and round final
interval endpoints outward only when necessary. This prevents cancellation at
the decision boundary from turning a score at `q` into an uncovered interval.
Exactly representable zero boundaries retain the existing approval/rejection
convention. No numeric tolerance is applied to truth scoring or decision errors.

A failed procedure, failed verification, missing/nonfinite/reversed interval,
or selector failure receives `+infinity`. Candidate exclusion alone is not a
whole-procedure failure. A failed or nonfinite truth computation is a material
incident that stops scientific interpretation; it is not a discarded case or
an ordinary missing-fit record.

Take the maximum score over the three families in each block, separately for
the selector and all four fixed policies. Each has exactly 100 block scores.
Choose its **96th smallest** score as its own additional correction `q`.
This is the smallest rank satisfying
`P[Binomial(100, 0.90) <= k-1] >= 0.95`.
The actual confidence is approximately 97.63% per procedure. It is not a
joint guarantee over the five procedures, nor a selected-action conditional
guarantee. This follows the tolerance-region interpretation described by
[Hulsman](https://arxiv.org/abs/2210.14735). The binomial limits use the
[exact Clopper–Pearson construction documented by SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html).

For finite `q`, emit `[L-d_a-q,U+d_a+q]`, and recompute the final decision using
the same verification/failure rules. This transformation has a truth-free
interface accepting only interval, offset, verification status, and failure
status. Future reserved execution must save the calibrated decision before
truth scoring; Phase E does not open that partition.

Four infinite block scores can still yield a finite 96th score; five force an
infinite correction. Never drop those blocks, replace their seeds, or decrease
the rank. If any required procedure correction is infinite, close calibration
as infeasible and do not generate reserved evidence. Preserve broad finite
corrections without retuning or narrowing them. A finite correction still
requires the separate Phase F freeze and replay.

## E3 — implementation, disposable gate, and execution

The new Phase E modules implement pure scoring/calibration, the predeclared
comparison, and checkpointed execution. They reuse the sealed D6 numerical
family worker with the new explicit partition argument. D1–D6 source bytes and
their evidence remain unchanged.

Before `p1_independent_calibration` opens:

1. Commit this protocol, precision artifact, implementation, and tests on dev.
2. Require passing full CI at that exact numerical-source commit.
3. In a clean clone at that commit, execute the one-block disposable namespace
   `p0_disposable_phase_e_roundtrip_v1` twice independently. Save/load/validate
   both exact raw archives. Require matching scientific digests, allowing
   timing and performance differences. The four-worker pool exists, with two
   workers computing the declared repeats; the prior D6 record covers four
   simultaneously active numerical workers using the same family worker.
4. Exercise a constructed 100-block-sized raw JSON archive, its full load and
   checksum path, plus a 1500-cell constructed calibration fixture. These
   repeated/constructed records are explicitly inadmissible as scientific
   calibration or reserved evidence.
5. Test process-tree sampling synchronously before computation and record the
   whole rehearsal's sampled peak, count, and errors. Require no sampling
   error, positive samples, and peak at most 4 GiB.
6. Commit a separate gate record binding the original numerical-source
   revision, manifest, protocol, CI success, rehearsal hashes, and resource
   acceptance. Run science from that original clean source clone. The later
   gate commit records acceptance without modifying or rebinding numerical
   bytes. Its gate file must match its committed bytes and descend from the
   tested numerical-source commit.

The CLI supports preflight, disposable rehearsal, gated calibration, and saved
calibration validation. It has no reserved execution mode. Completed blocks
are written atomically and resumed only after source, runtime, protocol,
semantic stream, content, and scientific seals validate. Preserve failed
cases. Save every observation, fit, candidate transition, predictive draw,
selection, verification, raw interval, resource value, and timing in indexed
block archives; the calibration summary binds all 100 checkpoint hashes and
recomputes all 1500 scores and five corrections from those saved records.

Keep any unresolved partial file, execution lock, source mismatch, or material
incident for explicit recovery review. Do not regenerate successful blocks or
pick replacement seeds. Ordinary fit or verification attrition remains in the
denominator. Material bugs require a documented version and a full affected
campaign replacement; exposed labels must not silently become tuning data.

## Remaining work

### CI dependency repair — 2026-10-07

The [CI run at `8e64cb9`](https://github.com/rolandrolandroland/thermotwin/actions/runs/37563413683)
failed with two Phase E test-module import errors because SciPy was absent.
The workflow installed `.[all]`, whose dependency declaration omitted SciPy,
although the pinned scientific runtime already requires SciPy 1.15.3. The
repair explicitly installs `scipy==1.15.3` in the workflow alongside `.[all]`.
The package metadata is part of the sealed source manifest, so it remains
byte-identical. The repair changes no numerical source, endpoint, partition,
or pinned runtime requirement.

Require a passing full CI run at the repaired source commit before executing
the disposable replay. The declared Phase E disposable namespace was never
opened at `8e64cb9`; calibration and reserved evidence also remain unopened.

E3's disposable acceptance and generation gate precede E4's full calibration.
After E4 validates, commit the finite correction artifact or the infeasibility
result. Phase F then freezes the complete calibrated procedure and reserved
analysis, passes its separate end-to-end replay, and opens the reserved cohort
once. Phase G/H interprets the frozen comparison, independently recomputes the
report, archives evidence, and prepares the presentation.
