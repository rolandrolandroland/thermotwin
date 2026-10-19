# Phase F: complete calibrated procedure and final evaluation

Declared: 2026-10-17. Version: `operating_decision_prospective_phase_f_v1`.
Status updated 2026-10-18: the complete rule/source freeze at `7e3b9a3`,
exact-source hosted CI, both disposable computations, independent reloads,
and resource acceptance passed. The [F3 acceptance](OPERATING_DECISION_PHASE_F3_ACCEPTANCE.md)
binds the separate reserved-generation gate. Reserved is unopened at gate
acceptance and may open once only after that gate is committed. The execution
journal records the subsequent launch and checkpoint progress.

The [accepted Phase E result](OPERATING_DECISION_PHASE_E_RESULT.md) supplies
five finite corrections. The [complete rule artifact](../../thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_F_RULE.json)
binds them to the final Phase D selector, original Phase E calibration and
analysis design, runtime, costs, sample sizes, and failure policy. The source
manifest includes all 109 original Phase E dependencies unchanged, the accepted
calibration artifact, complete-rule artifact, and new Phase F modules/CLI.
The exact source commit, manifest and protocol digests are bound in each
checkpoint and final archive. A later acceptance/gate commit must descend from
this source; numerical execution stays at the tested source commit.

## Preserved scientific choices

The primary question, comparators, 100 reserved blocks, nominal sensing,
`balanced_face_equal` costs, N=16, one allowed unusable predictive draw, and
development offsets are unchanged. Stop/thermal/voltage/face offsets remain
0.000/0.000/0.074/0.098 K. Candidate exclusions retain their existing semantics.
No final correction changes action selection, stopping, fitting, offsets,
predictive draws, or verification.

All five procedures retain their own Phase E correction. The selector's
correction is 0.025436425099518184 K regardless of the selected action.
Final intervals expand the raw interval by the selected action's development
offset plus the procedure's correction, using exact stored binary inputs and
outward endpoint rounding. Verification failure, missing interval, failed
procedure, or absent selection yields insufficient evidence; retain the case.
Unknown target truth or failure to calculate target energy stops interpretation
as a material incident.

The unchanged primary analysis uses 73 simultaneous one-sided bounds and the
predeclared 10% approval-risk ceiling, 70% decision-coverage floor, 5 percentage
point coverage noninferiority margin, and 2 percentage point risk margin.
Report all four fixed alternatives; primary success requires success against
every qualifying fixed alternative, including dominated alternatives, and at
least one must qualify. An empty qualifying set or inconclusive comparison
does not establish superiority. Preserve the development-only precision warning:
100 blocks are underpowered for the strict risk comparison. No extension or
rule change follows reserved outcomes.

## Durable before-reveal boundary

The new worker statically copies the frozen D6 orchestration and calls its
original numerical functions. A structural regression test verifies that
removing the added calibration-save insertion recovers the original function.
It runs the common acquisition, prospective predictions, frozen selection,
and four fixed-policy decision/verification procedures before calibration.

The calibration API accepts exactly four fields: raw interval, development
offset, verification success, and pipeline-failure status. No target margin,
truth-family label, device token, or target response enters that API. Case
identity lives in the archive envelope for pairing, outside the decision API.
All five procedures' decisions at clearances 0.00/0.05/0.10/0.20 K are written,
flushed, reloaded, and atomically published before the first target-scoring
call. Failure to persist stops before reveal. The selector uses only the
unchanged acquisition evidence and development rule.

Files live at `before-reveal/block-NNN/<family>.json`. After target reveal,
the archive validator reconstructs the four inputs from the original saved
policy decisions, authenticates every pre-reveal record, and checks that
post-reveal scoring reproduces the saved interval and decision exactly.
Secondary clearances change final decisions only, never acquisition. Their
analysis outputs are descriptive and do not support confirmatory claims or
threshold selection.

## F1 — source and environment freeze

Commit the accepted E artifact, complete F rule, F source and tests, protocol,
and ledger before generating the fresh disposable namespace. Run hosted CI
on that exact commit. Use a clean persistent source clone, pinned CPython
3.10.12 on Darwin arm64, and the unchanged package requirements. The complete
rule records the math environment used by Phase E: the listed 18 threading,
CPU-feature, and Python hash/optimization variables are unset. The CLI checks
those settings before importing numerical libraries. Preserve settings under
the native job environment; do not silently change thread configuration.

The Phase E archive must be revalidated in its original clean `23a076f` source
clone, in a separate interpreter, preserving its original import context.
The new runner never relabels that archive as a result of the Phase F commit.
The accepted raw summary hash must match before invoking its original validator.

## F2 — fresh disposable end-to-end rehearsal

Campaign: `operating_decision_prospective_v1_2026_09`.
Fresh engineering namespace: `p0_disposable_phase_f_roundtrip_v1`.
Compute one paired block twice independently in `run-1` and `run-2`, with the
same explicitly paired semantic keys. The four-worker executor is preserved;
each repetition has one computing task. Repetitions run sequentially so each
can pass the complete final-archive reload before the next begins. This costs
two computations and supplies three distinct disposable cases, not six
independent cases. Neither repetition contributes to calibration or inference.

Each computation must pass original acquisition/stream/case validation, new
pre-reveal persistence checks, all four clearance analyses, canonical JSON
round trips, and final archive reconstruction. Exact scientific result digests
must agree; elapsed time and memory measurements may differ. A separate
constructed storage fixture repeats a retained block 100 times and transports
1,500 synthetic analysis rows. It is visibly labeled resource/transport
evidence and is never counted as 100 independent blocks.

The whole workflow monitor includes original calibration validation, numerical
child processes, before-reveal files, archive reload, analysis, and the storage
fixture. Require successful sampling and a measured peak below 4 GiB. Retain
resource records for failed invocations as well. A persistent external journal
and a one-time native job protect operational provenance across app closure;
the job has no automatic restart. Mac sleep can pause computation.

A separate process must reload both completed disposable evaluation archives,
recompute their analysis, verify exact scientific replay, and hash the
constructed storage artifact and resource record before acceptance. CI/replay
success is engineering acceptance, not evidence that the selector beats a
fixed procedure.

## F3 — separate committed generation gate

After F2 and independent reload pass, commit an acceptance record and
`phase_f_generation_gate_v1`. Bind exact numerical source/manifest/protocol,
complete-rule digest, passing exact-source CI, original E validation, both
disposable computations, constructed storage, independent reload, and the
passing whole-workflow resource record. Include their evidence hashes.
The gate itself must match committed bytes at a descendant of the tested
source. The complete-rule artifact alone does not authorize reserved generation.

## F4 — execute and independently validate reserved evaluation once

Use the existing `p1_reserved_evaluation` namespace: 100 paired blocks,
300 cases, five procedures, 1,500 primary outcomes. The numerical source,
rule, sensor/cost scenario and analysis remain exactly those accepted in F3.
Retain every case and every fixed comparator, including ordinary failures.
Do not retune, reseed, replace completed blocks, or extend the sample size.

Completed block checkpoints and pre-reveal records have canonical content
seals bound to source, protocol, runtime, campaign, partition, block and family.
Recovery authenticates completed blocks and resumes only missing blocks under
the same seeds. An interrupted incomplete block may already have pre-reveal
files; deterministic recomputation must match their bytes before revealing
targets. An unexpected partial, changed decision, source mismatch or material
truth failure preserves evidence and stops for review. Locks prevent concurrent
coordinators; stale lock recovery verifies the old PID is absent and preserves
the lock. Finished reserved archives are closed against regeneration.

The summary indexes raw checkpoints, all calibrated outcomes, primary and
secondary analyses, scientific digest and invocation performance. Independent
validation must reconstruct outcomes and analysis from retained raw checkpoints
and before-reveal files. Reports and large raw archives remain gitignored.

## Execution interface and budget

The new CLI is `python -m thermotwin.reports.operating_decision_prospective_phase_f`.
All modes accept `--output-directory`, `--calibration-archive`, and
`--calibration-source-root`; the executing clone supplies its own source identity.

1. `--preflight` checks source, runtime, closed calibration and existing files;
   it generates no observations.
2. `--rehearsal --ci-evidence <saved exact-source CI JSON>` runs F2.
3. `--validate-evaluation <run-1/evaluation.json>` and the corresponding run-2
   command independently validate the two disposable archives.
4. `--execute-reserved --generation-gate <committed gate> --gate-repository-root
   <gate repository>` is permitted only after F3. It cannot run with just the
   complete-rule freeze.
5. `--validate-evaluation <reserved/evaluation.json>` reconstructs the final
   analysis after completion. `--recover-interrupted-lock` is restricted to
   resumed execution, with separate incident/resource review as needed.

Phase E actually used 255.26 completed-block CPU hours, above its original
development-based projection. A comparable 100-block reserved cohort should
be planned around that observed scale, plus additional persistence/validation
overhead: roughly 64 ideal four-worker active hours before overhead and sleep.
This is a planning estimate, not a completion promise. Disposable and original
archive checks have separate measured costs; preserve all interruption costs
and missing measurements. Final interpretation, three scientific figures,
report, presentation and reproducibility package follow successful archive
validation even if the scientific result is null or inconclusive.

## Local implementation checks before source freeze

The full local suite passed 944 tests in 1,093.943 seconds. After final source
formatting and a resource-report partition-label correction, all 20 focused
Phase F tests passed in 4.309 seconds, including the two added CLI checks.
These runs overlap; they are not 964 distinct tests. Hosted CI must test the
final committed source before numerical rehearsal.

The original D6 and E source manifests remain respectively
`79772d230eec28aaf41499f0b258083ca061b91fbec8d244515df44b4d89c7ad`
and `88e3f8187ab57683ec1c36a6bd24a96364e508d50bb2ae9aa709dfe354645a20`.
The complete calibrated rule digest is
`d8ae3d9ea3748d4de776a672cf19798b145b589d681211ba88b8789186dd6341`.
Focused checks cover the actual worker's persistence-before-reveal ordering,
write failure preventing reveal, original-worker structural equivalence, own
procedure corrections, truth exclusion, retained failures, canonical archives,
changed/raw/final decisions, stream namespaces, interruption files, completed
disposable resumption without recomputation, constructed storage/analysis,
closed partition entry points, exact-source CI rejection, and missing gate
rejection before any reserved worker. Reports and raw archives remain ignored.
