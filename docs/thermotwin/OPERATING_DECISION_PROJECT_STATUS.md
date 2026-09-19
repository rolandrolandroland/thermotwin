# Operating-decision project status

Date: 2026-09-17. Last reconciled: 2026-09-19. Status: P1 is complete and
failed its engineering gate. P2 executed at source `9db5f3f`, but its saved
JSON failed the required load-and-validate round trip. P2 is preserved as an
invalid incident, its gate was not evaluated, and its N32 continuation never
opened. The representation repair and fresh P3 namespace are committed at
`8c232af`, passed local validation under CPython 3.10.12, and passed independent
clean-clone review. This audit closeout is recorded in this commit; the commit
must be pushed and its exact HEAD must pass CI. P3, development, calibration, and
reserved evidence remain unopened.

## Authoritative starting point

The reviewed prospective source starts at `9a21aa7`. At that audited boundary,
the two September 17 independent audits found no implementation defect in
prospective Steps 1–3 and
verified that the corrected-replication generator, parent artifact, guard
artifact, evidence manifests, chronology, and committed arithmetic still
reconstruct. The later disposable pilot exposed the eligibility defect
described below. The audits are retained as
[`THERMOTWIN_AUDIT_2026_09_17.md`](THERMOTWIN_AUDIT_2026_09_17.md) and
[`THERMOTWIN_AUDIT_2026_09_17_STEPS_2_3.md`](THERMOTWIN_AUDIT_2026_09_17_STEPS_2_3.md).

This status record does not freeze a scientific procedure. It records the
boundary between completed historical work and the unfinished prospective
experiment. The disposable P1 and P2 partitions have been generated; P2 is not
valid engineering evidence. No P3, development, calibration, or reserved case
in the new campaign has been opened.

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
responses outside selection. The first disposable pilot exposed an additional
eligibility defect: v2 also counted an ordinary candidate exclusion as
whole-draw instability, so the action could be rejected even though another
candidate supplied a valid conservatively scored interval. The bounded
prospective-uncertainty v3 repair keeps the no-gain floor but reserves
whole-draw instability for a true unusable draw. The selector algorithm remains
`prospective_four_action_selector_v2`; its complete pilot protocol now binds
the revised uncertainty identity. It retains authenticated raw widths,
supports versioned development offsets in padded action value, records one-
versus two-candidate stopping strata, and provides a separate single-candidate
clearance. Its zero offsets and clearances remain unfitted development defaults.
It is not a calibrated or frozen operating procedure.

Uncertainty v3 and the later validator hardening are ordinary defect repairs.
They do not consume the protocol's one planned scientific redesign and are not
evidence that the selector works. The pre-P2 validator checked deterministic
record structure, cross-record identities, exact stream inventory and RNG
offsets, fit invariants, and interval formulas in memory, but P2 exposed its
missing saved-JSON round trip. The current P3 repair adds JSON-native
configuration payloads and validation of the final serialized bytes; its final
local validation has passed. The exact non-additive validation runs are recorded
in the [Phase B acceptance record](OPERATING_DECISION_PHASE_B_ACCEPTANCE.md).
Archive validation still will not rerun acquisition or predictive simulations,
candidate refits, or post-reveal scoring; those calculations require a
source-bound replay from the retained raw record.

### Disposable pilot v1

The 4-block, 12-case `p1_disposable_draw_count_pilot` completed at committed
source `e32d091`. Its hashes, protocol binding, scientific digest, complete
case matrix, acceptance calculation, and archive size verify independently.
All 1,152 predictive draw records completed, but 422 candidate-level exclusions
were also labeled unstable. That made voltage and face temperature ineligible
in all 12 cases, thermal ineligible in 8/12, and left three cases with no
selectable acquisition action. The engineering gate therefore failed and no
draw count was accepted.

The run took 1.88 wall hours on four workers. Its measured projection for all
230 planned post-pilot blocks was 30.28 hours at N=4, 56.24 hours at N=8, and
108.18 hours at N=16, before extra sensor-quality simulations. The complete
diagnosis and content hashes are recorded in
[`OPERATING_DECISION_PROSPECTIVE_PILOT_V1_RESULT.md`](OPERATING_DECISION_PROSPECTIVE_PILOT_V1_RESULT.md).

### Disposable pilot v2 archive incident

The 4-block, 12-case `p2_disposable_candidate_exclusion_pilot` executed once at
source `9db5f3f`. Its detached hashes still identify the saved JSON and report,
but an independent `json.load` followed by archive validation raised
`ValueError: complete uncertainty protocol/config is invalid`. Tuple-valued
physical-configuration fields in memory had become lists in JSON, and the
validator compared those representations directly.

Scientific interpretation and the conditional N32 step stopped immediately.
P2's gate was not evaluated; no draw count or compute budget was accepted. Its
saved outcome and runtime fields are quarantined incident material and cannot
tune or trigger P3. The exact provenance, hashes, and response are in
[`OPERATING_DECISION_PROSPECTIVE_PILOT_V2_INCIDENT.md`](OPERATING_DECISION_PROSPECTIVE_PILOT_V2_INCIDENT.md).

## What remains

| Phase | Status | Required result before advancing |
| --- | --- | --- |
| A — reconcile source and records | Complete through the P2 incident record | Preserve P1 and invalid P2; version every replacement before opening it. |
| B — harden scientific interfaces | Repair committed at `8c232af`; local validation and independent clean-clone review passed | Push this audit closeout commit, then require green CI on exact HEAD before P3. |
| C — disposable compute pilot | P1 failed; P2 invalid with gate not evaluated; P3 unopened | Run only `p3_disposable_archive_roundtrip_replacement_pilot` after the repair gate. If valid P3 triggers its all-case N32 continuation, complete that before evaluating the combined gate. Freeze an accepted draw count and budget only in a later commit. |
| D — selector development and maps | Unopened | Use only the development partitions to fit offsets and thresholds and produce cost and physical sensor-quality maps. |
| E — analysis freeze and calibration | Unopened | Freeze endpoints and the complete selector, then use the independent calibration partition only for the declared correction. |
| F — reserved evaluation | Unopened | Verify the committed chain and open the reserved partition once. |
| G/H — interpretation and closeout | Unstarted | Classify the result honestly, audit it, archive evidence, and produce the report and three figures. |

Phase B records the following boundaries for the pilot and later development:

- value must account for development-derived stop and action offsets while
  preserving failure and candidate-loss penalties;
- the stop gate must use the development-padded envelope, record whether one or
  two candidates remain, and examine the one-candidate stratum separately;
- the replacement pilot reports candidate transitions separately from true
  whole-draw failures, compares authenticated N=4/8/16 prefixes, and freezes
  eligibility as a pair of draw count and maximum unusable draws before
  development;
- if N=16 is the smallest passing prefix and valid P3 has zero pipeline
  failures and zero N=16 selection failures, the predeclared P3 all-12-case
  N=32 follow-up must also pass
  the same 90% agreement and 5% regret limits with zero N=32 failures before a
  draw count can be frozen;
- probe nuisance draws must be described as coming from the declared truth
  support with the prior spread; and
- the acceptance record must cover heat-flow signs, energy accounting, probe
  loading and removal, solver/time-grid convergence, information boundaries,
  saved-before-reveal decisions, and random-stream audits.

The next permitted data generation is
`p3_disposable_archive_roundtrip_replacement_pilot` in the
[prospective partition ledger](OPERATING_DECISION_PROSPECTIVE_PARTITION_LEDGER.md),
and only after the reviewed JSON-native representation repair and final-byte
round-trip checks are recorded in the committed chain, pushed, and passing CI.
This audit closeout is recorded in this commit, which is not yet pushed, and CI
on its exact HEAD is not yet green. Large
development, calibration, and reserved generation remains
prohibited until the pilot gate passes and the selected draw count and measured
compute budget are frozen in a subsequent committed record.

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
