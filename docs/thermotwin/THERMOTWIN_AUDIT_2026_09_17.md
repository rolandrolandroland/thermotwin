# ThermoTwin audit: pushed `origin/dev` at `307a23a`

Date: 2026-09-17. Scope: everything pushed to `dev` since the previous audit (`ecb5030`):
`fa9eb89` (diagnostics archived outside Git), `9c51e59` (parent rehearsal), `f509329` (guard
freeze), `428a805` (reserved evaluation), and `307a23a` (prospective four-action selector,
Step 1). Project-wide integrity checks were also run.

## Method

- **Source:** a fresh clone of `origin/dev` from GitHub. The local working tree was not used.
- **No regeneration:** no scientific partition was generated or reopened.
- **Release assets not downloaded:** they sit outside the branch. Their GitHub-computed
  SHA-256 digests were compared with the committed manifests instead.
- **Older experiments:** their science was not re-audited; they received only the
  project-wide checks.

## Bottom line

- **Execution and records:** the corrected replication was executed as specified.
  Chronology, provenance, evidence manifests, and every committed number check out
  independently. The negative result for the scalar guard is reported accurately.
- **Main finding (P1):** the frozen stop-or-voltage selector missed its 90% block-coverage
  target on fresh data: 80/100 blocks, Wilson 95% [71.1%, 86.7%], binomial p = 0.002. The
  documents report the individual below-target numbers but never draw this conclusion.
  - Almost all misses come from one place: its voltage branch on the temperature-dependent
    contact family, where it applies less than half of fixed voltage's padding.
  - The selector's higher decision coverage and lower loss are therefore not evidence
    that it beats the fixed policies.
- **Completion claim:** the replication is labelled complete, but two of its three stated
  research questions are not answered.
- **Prospective selector:** Step 1 is carefully built, but the same padding-dilution
  mechanism will threaten it unless calibration is stratified by selected action.

## Verified

| Check | Result |
| --- | --- |
| Frozen source across the chain | No allowlisted numerical file changed from the generator freeze through `307a23a`, at any of the five steps. |
| Artifacts | Generator `0f2ca2d8`, parent `5a3262b2`, and guard `d96696f2` validate in the clean clone. The guard chain is byte-identical at HEAD (77 paths). |
| Chronology | Parent artifact committed (`ecb5030`), then rehearsal produced at `fa9eb89` and committed (`9c51e59`), then guard produced at `9c51e59` and committed (`f509329`), then reserved evaluation produced at `f509329` and committed (`428a805`). The reserved `source_commit` equals the guard-boundary commit. Each release was published after its evidence commit. |
| Evidence manifests | All three manifest digests recompute. Producer commits and commands match the documentation. Outcome provenance binds each partition's evidence digest. GitHub's SHA-256 for all four release assets equals the manifest archive hashes. |
| Archiver (`tools/archive_operating_decision_evidence.py`) | Recomputes the evidence digest; requires a complete partition, a clean stream audit, and HEAD equal to the source commit; uses deterministic gzip; verifies the round trip; refuses to overwrite. |
| Independent arithmetic | 1,740 committed outcome rows recomputed from raw intervals and paddings: rehearsal 300, guard calibration 540, reserved 900. Calibrated bounds, decisions, error flags, coverage, all summaries and losses, guard-trigger counts, and the monotone guard audit all match. The paired bootstrap, including SHA-256 stream seeding, reproduces bit for bit (see the P3 index quirk). |
| Tests | 686 tests OK on the clean clone (CPython 3.10.12, 276.5 s); tests leave no tracked-file changes. CI (`tests`, Python 3.11) is green on every `dev` push, including `307a23a`. |
| Documentation and packaging | 49 Markdown files have 0 broken relative links. All 33 console-script entry points resolve. |

## Reserved-cohort results (50 paired blocks × 3 families; recomputed)

| Procedure | Decisive | False approvals / rejections | Block coverage | Mean runs | Realized energy | Balanced loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stop now | 69/150 | 0 / 0 | 45/50 | 2.00 | 60.86 J | 0.540 |
| Fixed thermal | 87/150 | 0 / 0 | 45/50 | 5.00 | 99.49 J | 0.783 |
| Fixed voltage | 89/150 | 0 / 0 | 45/50 | 3.00 | 89.73 J | 0.654 |
| Fixed face temperature | 83/150 | 0 / 0 | 45/50 | 3.00 | 89.62 J | 0.694 |
| Stop-or-voltage selector | 98/150 | 0 / 0 | **38/50** | 2.65 | 79.54 J | 0.507 |
| Guarded selector | 93/150 | 0 / 0 | **40/50** | 2.49 | 74.86 J | 0.525 |

Primary contrast, guarded minus parent (balanced loss): +0.0185, 95% bootstrap interval
[−0.0032, +0.0452]. The success criterion (upper bound below zero) was not met, as reported.

## Findings

### [P1] The frozen selector under-covers, and the documents do not say so

Evidence:

- **Pooled coverage.** Across every fresh cohort evaluated after the parent freeze (rehearsal
  16/20, guard calibration 26/30, reserved 38/50), the frozen selector covered 80/100
  blocks.
  - These blocks are independent draws given the frozen artifact, so this is a direct
    estimate of the coverage delivered by this procedure: 80% (95% [71.1%, 86.7%]),
    binomial p = 0.002 against 90%.
  - On the same blocks, fixed voltage and fixed face temperature each covered 92/100
    (95% [85.0%, 95.9%]); fixed thermal and stop now each covered 94/100.
- **Mechanism (post hoc, descriptive).** 19 of the selector's 20 misses are rows where it
  chose voltage under temperature-dependent-contact truth.
  - It pads those raw voltage intervals by 0.030 K; fixed voltage pads the same intervals
    by 0.071 K.
  - In 14 of the 19, fixed voltage covered the same device.
  - Raw miss distances were 0.032–0.107 K.
- **Not a code defect.** Calibration nonconformity and evaluation coverage use identical
  definitions, and the data are exchangeable by construction. The split-conformal guarantee
  is marginal over calibration draws. With 30 calibration blocks, a selector padding this
  small had about a 7% chance (beta-binomial, 80/100).

Consequences:

- **Reading the results.** The selector's decision-coverage and loss advantages over the
  fixed policies were bought partly with under-coverage. They are not evidence of a better
  procedure. No decision errors occurred, but coverage is the stated safety property.
- **Documentation gap.** `OPERATING_DECISION_CORRECTED_REPLICATION.md` states only that
  coverage was "below the 90% target" (lines 211 and 277) and draws no conclusion. The
  README row still describes the replication in future tense.

Recommendations:

1. Add an explicit conclusion to the replication document: the frozen selector did not
   meet its coverage target on fresh data, and the likely mechanism is a pooled padding
   applied to heterogeneous branches.
2. For the prospective selector, calibrate paddings per selected action (group-conditional,
   or Mondrian, conformal calibration) and size calibration cohorts per action. At minimum,
   validate per-branch coverage in development before freezing.

### [P2] The replication is labelled complete, but questions 1 and 3 are unanswered

- **Question 1** (do fixed policies and the stop-or-voltage rule show the same practical
  tradeoffs?) has no answer. The fixed-policy reserved results are not discussed at all.
- **Question 3** (nominal versus realized energy) has no answer.
- **Post hoc only.** No fixed-policy contrast was predeclared before `evaluate-reserved`, so
  any selector-versus-fixed comparison must be labelled post hoc and descriptive.
- **Attribution.** Any comparison with Stages 3–5 should state that differences reflect the
  numerical-protocol changes (12 iterations, 2× fit bounds, exclusion of non-converged
  candidates) as well as the RNG fix.

### [P2] Prospective selector (Step 1): design risks to settle before any freeze

- **Padding dilution.** Four actions make the branches more heterogeneous than the
  stop-or-voltage rule, so the P1 mechanism is more likely with procedure-level
  calibration.
- **Less conservative stop gate.** `build_prospective_acquisition_snapshot` builds the
  provisional envelope from the candidates left after exclusion
  (`operating_decision_prospective.py:461`). The parent stop signal refused to stop when any
  candidate hit a bound (`operating_decision_calibration.py:534`). Under four-state truth,
  where a five-state bound hit is common, the new gate will stop on a single-model envelope.
  The default `stopping_clearance` is 0.0. Calibrate the clearance on development data, and
  check stop-branch coverage separately.
- **What is already right.** The mechanics are careful: frozen ranking and tie rules,
  quantized comparison, explicit selection failures, and one shared uncertainty baseline.
  Unreliable candidates are excluded individually, which also fixes the earlier latent
  inconsistency where the stop signal ignored convergence. The document is candid that fit
  provenance and action scores are not yet authenticated.

### [P3] "Immutable" releases are not immutable

The GitHub API reports `immutable: false` for `operating-decision-corrected-rehearsal-v1`,
`-guard-v1`, and `-final-v1`. The SHA-256 hashes in Git make tampering detectable, but
assets can still be deleted or replaced. Correct the wording, and enable immutable releases
for future evidence. For long-term availability, also deposit the archives in an archival
store.

### [P3] Bootstrap percentile index quirk

`alpha = 1.0 - 0.95` evaluates to 0.050000000000000044. The lower bound therefore uses sorted
index 500 instead of 499 (`operating_decision_replication_guard.py:1595`). The effect is at
most 6.7e-8 on the reported bounds, and no conclusion changes. Compute percentile indices
from integers in the next protocol.

### [P3] Documentation items carried over

- `thermotwin/OPERATING_DECISION_EXPERIMENT.md` (Stage 2) still has no audit note and keeps
  its 35.4% Wilson bound.
- The body of `thermotwin/OPERATING_DECISION_CALIBRATION.md` (line 253) still states the
  conformal guarantee as if it applied.
- The replication document does not state the CPython 3.10.12 requirement, and still says
  exceptions are fatal "for an otherwise eligible candidate" (line 137), which does not match
  the code.
- The selector stop signal's convergence gap is undocumented, and its occurrence count cannot
  be recovered from compact outcomes.
- Four tracked documents contain `/Users/rolandbennett` in plain text.

### Resolved since the previous audit

- Repository growth: later diagnostics are archived externally with content-addressed
  manifests.
- The end-to-end chain executed successfully on real partitions.
- CI and tests are green.

## Not examined

- The contents of the release assets (not downloaded). This can be done on request, and
  would allow the reserved decisions to be recomputed from full fits.
- The local working tree.
- Scientific re-audit of the non-operating-decision experiments.

Verification scripts were run from a temporary session directory and were not added to the
repository.
