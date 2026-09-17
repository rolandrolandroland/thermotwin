# Operating-decision project status

Date: 2026-09-17. Status: Phase A reconciliation and the Phase B acceptance
record are complete; the prospective scientific campaign remains unopened.

## Authoritative starting point

The reviewed prospective source starts at `9a21aa7`. The two September 17
independent audits found no implementation defect in prospective Steps 1–3 and
verified that the corrected-replication generator, parent artifact, guard
artifact, evidence manifests, chronology, and committed arithmetic still
reconstruct. The audits are retained as
[`THERMOTWIN_AUDIT_2026_09_17.md`](THERMOTWIN_AUDIT_2026_09_17.md) and
[`THERMOTWIN_AUDIT_2026_09_17_STEPS_2_3.md`](THERMOTWIN_AUDIT_2026_09_17_STEPS_2_3.md).

This status record does not freeze a scientific procedure. It records the
boundary between completed historical work and the unfinished prospective
experiment. No pilot, development, calibration, or reserved case in the new
campaign has been generated or opened.

## What is complete

### Corrected replication

The collision-free corrected `r2` replication is complete. Its source and
artifacts remain historical records and are not inputs for tuning the
prospective selector. The scalar mismatch guard failed its predeclared primary
criterion and is not promoted.

The post-completion audit adds a second conclusion. Across the 20-block parent
rehearsal, 30-block guard calibration, and 50-block reserved evaluation, the
frozen parent selector covered 80/100 fresh blocks against its 90% target. Its
95% Wilson interval was 71.1% to 86.7%, and the two-sided binomial p-value
against 90% was 0.002. Fixed voltage covered 92/100 of those blocks. The parent
selector's greater decision coverage and lower scalar loss are therefore not
evidence that it beat the fixed policies.

The concentration of 19/20 misses in Family C on the voltage branch is a post
hoc diagnosis. It supports guarding against heterogeneous branch padding, but
does not prove a universal mechanism or invalidate conformal calibration as a
method.

The replication's fixed-policy and nominal-versus-realized energy comparisons
are descriptive because selector-versus-fixed contrasts were not predeclared.
In the reserved cohort, all four fixed policies covered 45/50 blocks. Fixed
voltage made 89/150 definitive decisions at 89.73 J mean realized energy; stop
made 69/150 at 60.86 J; thermal made 87/150 at 99.49 J; and face temperature
made 83/150 at 89.62 J. Fixed-policy realized mean energy differed from its
nominal proxy by at most 0.24%. These values do not include reset energy,
sensor electronics, complete wall time, or hardware consumption.

The corrected results differ from the original Stage 3–5 campaign for more
than one reason. The corrected protocol repaired the random streams and also
changed iteration count, inference bounds, candidate-admissibility handling,
convergence testing, and realized energy accounting. Historical differences
cannot be attributed to the RNG repair alone.

### Prospective interfaces through Phase B

The prospective experiment has development interfaces for:

1. selecting among `stop_now`, `fixed_thermal`, `fixed_voltage`, and
   `fixed_face_temperature`;
2. estimating each measurement action's expected reduction in final-margin
   envelope width from authenticated acquisition-only evidence; and
3. applying explicit incremental energy, time, and typed-instrument costs over
   a predeclared 12-cell cost grid.

Those interfaces retain failed draws, prevent candidate attrition from looking
like information gain, bind semantic random streams, and keep truth and final
responses outside selection. The active Phase B rule is
`prospective_four_action_selector_v2`. It retains authenticated raw widths,
supports versioned development offsets in padded action value, records one-
versus two-candidate stopping strata, and provides a separate single-candidate
clearance. Its zero offsets and clearances remain unfitted development defaults.
It is not a calibrated or frozen operating procedure.

## What remains

| Phase | Status | Required result before advancing |
| --- | --- | --- |
| A — reconcile source and records | Complete in documentation | Commit this status, the audit corrections, and the partition ledger without generating a cohort. |
| B — harden scientific interfaces | Accepted for the disposable pilot | Selector v2 versions padded scoring and stop strata; focused physics, numerics, information-boundary, and provenance checks pass. Commit the revision before pilot generation. |
| C — disposable compute pilot | Unopened | Run the predeclared 12-case pilot and choose a draw count and worker budget from complete results. |
| D — selector development and maps | Unopened | Use only the development partitions to fit offsets and thresholds and produce cost and physical sensor-quality maps. |
| E — analysis freeze and calibration | Unopened | Freeze endpoints and the complete selector, then use the independent calibration partition only for the declared correction. |
| F — reserved evaluation | Unopened | Verify the committed chain and open the reserved partition once. |
| G/H — interpretation and closeout | Unstarted | Classify the result honestly, audit it, archive evidence, and produce the report and three figures. |

Phase B records the following boundaries for the pilot and later development:

- value must account for development-derived stop and action offsets while
  preserving failure and candidate-loss penalties;
- the stop gate must use the development-padded envelope, record whether one or
  two candidates remain, and examine the one-candidate stratum separately;
- the pilot reports the baseline 4/4, 8/8, and 15/16 eligibility behavior
  explicitly, and the later scientific rule freezes draw eligibility as a pair
  of draw count and maximum unstable draws before development;
- probe nuisance draws must be described as coming from the declared truth
  support with the prior spread; and
- the acceptance record must cover heat-flow signs, energy accounting, probe
  loading and removal, solver/time-grid convergence, information boundaries,
  saved-before-reveal decisions, and random-stream audits.

The next permitted data generation is the disposable pilot named in the
[prospective partition ledger](OPERATING_DECISION_PROSPECTIVE_PARTITION_LEDGER.md),
and only after Phase B's interfaces and focused checks are committed. Large
development, calibration, and reserved generation remains prohibited until its
preceding gate is satisfied.

## Evidence and runtime boundaries

The corrected replication's exact replay is bound to CPython 3.10.12 by its
stored runtime manifest. Ordinary package use still declares Python 3.10 or
newer. The prospective campaign must record and bind its own scientific
runtime before calibration.

The corrected full diagnostics are addressed by committed SHA-256 hashes and
round-trip-verified gzip archives. The GitHub releases that contain them were
reported as mutable on 2026-09-17. The hashes make replacement detectable but
do not prevent replacement or deletion. Future final evidence should use an
immutable release or archival deposit when available; otherwise it must be
described as hash-verified.
