# Corrected operating-decision replication

Status: the corrected replication is complete. Corrected generator v4, the
parent rule, and the scalar mismatch guard are frozen. Parent protocol digest
`5a3262b2cfd3d1560d961a5dcd54bf5cd20f5cb8721795e1ea469fff2bc28362`
binds the gate-development and parent-calibration evidence; guard protocol
digest `d96696f265937b8d8f584aff6684c33cc1c279b9687bf864a4555e30c3ef63f2`
binds the guard-development and guard-calibration evidence. All six scientific
device partitions have clean random-stream audits, and the paired bootstrap is
complete. The scalar guard did not meet its primary reserved-cohort success
criterion and is not promoted. A post-completion audit also found that the
frozen parent selector covered only 80/100 fresh blocks across rehearsal,
guard-calibration, and reserved cohorts, below its 90% target. The selector is
therefore not promoted over the fixed policies. See
[`THERMOTWIN_AUDIT_2026_09_17.md`](THERMOTWIN_AUDIT_2026_09_17.md).

Generator freeze `c120e29` is superseded. An independent decision-rule audit
found that its corrected wrapper turned either candidate's bound hit or
nonconvergence into whole-case abstention and evaluated convergence with the
unprojected gradient. Source commit `a78030d` restores candidate exclusion,
uses the box-constrained projected KKT residual, exports that residual
explicitly, and rotates the scientific campaign. All 664 repository tests pass.

A disposable 12-block probe produced 42/48 decisive Family A outcomes, versus
15/48 under the superseded whole-case counterfactual on the exact same fits.
Fixed voltage produced 11/12 decisive outcomes versus 1/12. All 27 restored
decisions were correct, and the random-stream audit found no unintended reuse.
Generator v4 is frozen in this commit with artifact digest
`0f2ca2d80f53fe5df2ebd77e8046a2b6920026c9b034719577a1b0a6dfca43f7`
and source-manifest digest
`c5a52ab1c1261df4cb3cbacbdd9557557c6b2495ab1943ca1f366942a62d6968`.
The replacement campaign's gate-development and parent-calibration partitions
were generated only after that freeze was committed.

## Purpose

The September 2026 audit found deterministic reuse of random draws across
measurement channels, diagnostic runs, and adjacent device blocks. That defect
breaks the sampling assumptions behind the Stage 4 conformal calibration and
the Stage 5 block bootstrap. This replication repairs the generator and repeats
the existing decision procedures before a more capable selector is developed.

The old Stage 3–5 outputs remain preserved, exposed descriptive records. Their
device labels, residuals, thresholds, paddings, and final outcomes cannot enter
this replication.

## Questions

1. Do the fixed policies and the frozen stop-or-voltage rule show the same
   practical tradeoffs after every unintended random-stream collision is
   removed?
2. Does the rejected scalar mismatch guard remain unfavorable when its
   development, calibration, and final comparison are repeated on independent
   device blocks?
3. How much do realized per-device terminal energies differ from the nominal
   cost proxies used to choose an action?

This is a correction study. It does not yet test the original hypothesis that
software can choose among thermal, voltage, and face-temperature packages by
their expected decision value and cost.

## Supersession record

Commit `c120e29` froze generator v3 before this reliability audit. While its
parent freeze was running on 2026-09-14, `r2_gate_development` and
`r2_parent_calibration` in campaign
`operating_decision_audit_replication_2026_09` were instantiated once. The
resulting parent and diagnostic files were invalidated and deleted immediately
after the audit arrived; they were never committed, never used to revise the
rule, and never used to open rehearsal, guard, or reserved evaluation.

The v3 freeze is superseded for two reasons:

1. `decide_corrected_blinded_case` and `rebuild_with_corrected_gate` added a
   whole-case reliability gate that was inconsistent with the candidate-level
   exclusion used by Stages 3–5. In the stored Stage 5 result, 143/188 Family A
   decisions relied on excluding a bound-hit candidate, so the change was
   material rather than defensive bookkeeping.
2. The optimizer's convergence metric used the ordinary gradient at active box
   bounds. The correct first-order test is the projected KKT residual: retain
   the gradient in the interior, retain only negative components at a lower
   bound, and retain only positive components at an upper bound.

Changing only the generator version would reproduce the already opened v3
draws because random streams are keyed by campaign and partition. Generator v4
therefore uses a new campaign namespace. The abandoned v3 campaign is retained
only as an audit record and will not be resumed.

## Generator version

The corrected generator is `operating_decision_generator_v4`, using replication
protocol `operating_decision_audit_replication_v3` and random-stream protocol
`thermotwin-operating-decision-rng-v2`. Each random
stream is derived with SHA-256 from a typed semantic key containing:

```text
protocol version
campaign
partition
paired device block
purpose
truth family or explicit shared-family marker
run
channel or component
```

The key, rather than arithmetic addition, defines separation. The seed used by
Python's local random-number generator is only a deterministic rendering of
that key.

Physical device parameters are deliberately shared across the A/B/C variants
of one paired block. A common acquisition or verification observation is shared
across policies only when family, run, instrumentation, and channel are
identical. Different channels, distinct runs, family-specific noise, truth-law
parameters, and different blocks must have distinct keys. An exhaustive
registry check rejects any repeated seed unless the complete semantic key is
identical and the reuse is declared as pairing.

The corrected numerical protocol gives every nonlinear fit twelve iterations.
The synthetic truth support for series resistance and temporary-sensor
capacitance and response remains unchanged. Inference alone receives a
twofold buffer beyond each end of those truth ranges so ordinary observation
noise does not force an otherwise correct model onto the same boundary used
to generate truth. A disposable pre-freeze check found that the former
six-iteration limit caused four matched-model failures; the fifth was a probe
response estimate clipped at the 6 s truth ceiling. With twelve iterations and
the buffered inference bound, all three predeclared starts converged to the
same 6.37 s estimate.

The v4 convergence decision uses only the scaled projected KKT infinity norm.
Small steps and small objective changes are reported as stagnation diagnostics,
not accepted as stationarity. The serialized compatibility field
`scaled_gradient_infinity_norm` and the explicit
`scaled_projected_gradient_infinity_norm` field both contain the projected
value. A bound hit remains a separate candidate-admissibility flag even when
the KKT test passes.

At decision time, a bound-hit or finite nonconverged candidate is excluded and
the envelope is rebuilt from the remaining candidates. The case abstains when
none remain. An acquisition-fit failure for either candidate is case-fatal.
A verification exception or returned nonfinite verification score is also
case-fatal, including when that candidate is otherwise numerically
inadmissible. An uncertainty-propagation failure is removed only when it
belongs to a candidate already excluded for a bound hit or nonconvergence; a
failure for a retained candidate is case-fatal. Matched gate development also
continues to abort on an unreliable matched candidate rather than silently
changing its calibration sample.

## Disposable recovery probe

The recorded probe uses campaign
`operating_decision_disposable_candidate_exclusion_kkt_2026_09_14` and partition
`disposable_family_a_candidate_exclusion_probe`. The public partition runner
generated all three families so completeness and stream audits ran normally;
only the 48 requested Family A cases are retained in the result file.

The current rule made 42/48 cases decisive and 11/12 fixed-voltage cases
decisive, with zero false approvals, false rejections, or numerical failures.
Twenty-nine five-state fits placed interface mass at its 8 J/K lower bound. On
the same saved fits, whole-case abstention reduced coverage to 15/48 and fixed
voltage to 1/12, discarding 27 correct decisions and no wrong decisions. These
counts differ from the independent auditor's disposable stream because this
probe has its own semantic namespace; the coverage mechanism and direction
reproduce. The complete cases and audit are saved under
`docs/thermotwin/operating_decision_correction_2026_09_14/`.

## Five-command disposable rehearsal

On 2026-09-15, a throwaway clone starting from generator-v4 commit `2a0ee76`
ran all five executable commands in order under campaign
`operating_decision_disposable_five_command_dry_run_2026_09_15`. Each of the six
partitions contained ten paired blocks per family, and the bootstrap used 1,000
draws in `disposable_bootstrap`.

The clone committed the generator, parent, and guard artifacts at their required
boundaries. `rehearse-parent` verified the serialized parent in a new process;
`evaluate-reserved` verified the complete serialized chain and recorded the
guard-boundary commit before opening its partition. All five commands exited
successfully. Each partition retained 120 records, all six stream audits were
clean, every stream used only the disposable campaign, and every artifact
reconstructed from its serialized content. The compact audit, artifact digests,
boundary commits, and output hashes are recorded in
`operating_decision_correction_2026_09_14/five_command_dry_run.json`. Its final
comparison is an engineering dry-run result and is not scientific evidence.

## Scientific parent freeze

On 2026-09-15, the committed generator-v4 chain opened
`r2_gate_development` and `r2_parent_calibration` once. The gate partition
contains 240 complete records from 20 paired blocks per family; the calibration
partition contains 360 complete records from 30 paired blocks per family. Their
evidence digests are `e39ef2ba35892c72a08f61d68bbe9bbd60a35d27a13b9fb620b806c12dd1f53b`
and `1ada76adb1eb3abbc81de05fb67f32a5e17544bf013caa0b83fee77c7a2a722f`.
Both exhaustive stream audits report zero unintended reuse.

The verification thresholds are 1.276697 for stop now, 1.272123 for fixed
thermal, 1.223269 for fixed voltage, and 1.221628 for fixed face temperature.
The fitted margin paddings are 0 K for stop now and fixed thermal, 0.070730 K
for fixed voltage, 0.057124 K for fixed face temperature, and 0.030400 K for
the decision-directed selector. These calibration results were committed in
the parent artifact before the separate rehearsal partition was opened.

## Scientific parent rehearsal

On 2026-09-16, a new process at source commit
`fa9eb898e5bbd7e6d47a19020a5ea7885be0a566` verified the committed parent
artifact and source manifest before opening `r2_parent_rehearsal` once. The
partition contains 240 complete records from 20 paired blocks per family. Its
deterministic evidence digest is
`c9d4dd8d69c78ecbb27e228bed9282f0065ad1291819f51b668e3025cde50f20`.
The exhaustive audit found 2,560 unique keys and seeds across 4,040 uses, with
zero unintended reuse.

All five procedures had zero false approvals, zero false rejections, and zero
numerical failures. The decision-directed selector chose stop now in 17 cases
and fixed voltage in 43. It produced 45/60 definitive decisions (75.0%), used
81.45 J mean realized terminal energy, and had 0.427 empirical balanced loss.
Its simultaneous block interval coverage was 16/20 (80.0%), below the 90%
target. This unfavorable result is retained without a rule or protocol change;
at that checkpoint, the guard stages remained unopened and were restricted to
replacing a parent decision with abstention.

The tracked external-evidence manifest has canonical digest
`0ae40fd44828dc9c189bb094781419b152f0a603546872267d9e30a1f34cff03`.
Its content-addressed gzip archive has SHA-256
`df4bb3abbee2a5d2e0adc6298d71166c7d64309f4771478cc27d204aa936228e`
and expands to the exact 32,497,890-byte diagnostic JSON.

## Scientific guard freeze

On 2026-09-16, a new process at source commit
`9c51e59cb4dcedc72b06b75ab7918064c1021a28` verified the committed parent
artifact and source manifest before opening `r2_guard_development` and then
`r2_guard_calibration` once. Each partition contains 360 complete records from
30 paired blocks per family. Each audit found 3,840 unique keys and seeds
across 6,060 uses, with zero unintended reuse.

Matched guard development fixed the scalar alarm threshold at 1.178439, rank
28/30. It triggered in 2/30 matched four-state blocks, 0/30 extra interface
mass blocks, and 4/30 temperature-dependent-contact blocks. The complete guard
artifact has protocol digest
`d96696f265937b8d8f584aff6684c33cc1c279b9687bf864a4555e30c3ef63f2`.

In guard calibration, the parent selector made 56/90 decisions (62.2%) and the
guarded selector made 54/90 (60.0%). Mean realized terminal energy fell from
80.51 J to 78.80 J. Simultaneous block interval coverage rose from 26/30
(86.7%) to 28/30 (93.3%), while both procedures had zero false approvals, zero
false rejections, and zero numerical failures. The guarded padding remained at
the parent floor of 0.030400 K. These results are descriptive calibration
evidence; they do not use the reserved labels.

The development and calibration evidence digests are
`9701fa2368e1d3496dc16fd6fb1bcfdcc9923fd22e391a1b290ade28ed9b2799`
and `5f27e55ec1a186fa63cd5b9cbeea9750a8fb17f511c22eb8034676a2f53a5b2c`.
Their tracked external-evidence manifest has canonical digest
`31116f4f9558331acce8237c61501f8634e65fb63caae756c056cbcd7a1eeaad`.
The two deterministic gzip archives have SHA-256
`65d55550a2d9e0375129438957008e67a36b490f3ba44fd896a9e98b27b3b199`
and `68347d1970b7ad59139f8234dc7e30ff0c380922f495dc7d7a2618f923d4fac8`.

## Scientific reserved evaluation

On 2026-09-17, a new process at committed source and artifact state
`f509329dde422377145ddcc31368decb01f75fdb` verified the complete chain before
opening `r2_reserved_evaluation` once. The partition contains 600 complete
diagnostic records from 50 paired blocks per family and produced 900 compact
procedure outcomes. Its deterministic evidence digest is
`f1df0029d452c1b8fee68bf0ef622ad88e2f18a7148ee29779c59e101064295a`.
The exhaustive audit found 6,400 unique keys and seeds across 10,100 uses, with
zero unintended reuse.

The primary balanced-loss contrast, guarded minus parent, was +0.018487 with a
95% paired-block bootstrap interval of [-0.003231, +0.045153]. The predeclared
success criterion required the upper bound to be below zero, so it was not met.
The bench-time-dominant contrast was +0.023287 [-0.000030, +0.049954], and the
instrumentation-expensive contrast was -0.005513 [-0.029829, +0.019188]. None
of the three intervals excluded zero in the favorable direction.

The parent selector made 98/150 definitive decisions (65.3%), versus 93/150
(62.0%) for the guarded selector. Mean realized terminal energy fell from
79.54 J to 74.86 J. The guard triggered in 2/50 matched four-state blocks,
4/50 extra interface mass blocks, and 8/50 temperature-dependent-contact
blocks. Parent and guarded simultaneous block interval coverage were 38/50
(76.0%) and 40/50 (80.0%), both below the 90% target. All six procedures had
zero false approvals, zero false rejections, and zero numerical failures.

The complete 20,000-draw bootstrap arithmetic was reproduced independently
from the compact outcomes. The tracked external-evidence manifest has canonical
digest `c0126cef25434e8a054dcf55fc072ea71919c8345455f3b820b294f15f57d5a0`.
Its deterministic gzip archive has SHA-256
`d06ee288ebf149c5f46d59e2c4e1db274206380366e4c1c4cec231da9cadcbe8`
and expands to the exact 81,327,382-byte diagnostic JSON. The scalar guard is
therefore retained as a negative result and is not recommended over the parent
selector.

## Post-completion audit conclusion

The parent selector's simultaneous block interval covered 16/20 rehearsal
blocks, 26/30 guard-calibration blocks, and 38/50 reserved blocks. These are
100 fresh blocks drawn after the parent rule froze. Combined coverage was
80/100, with a 95% Wilson interval of 71.1% to 86.7% and a two-sided binomial
test p-value of 0.002 against the 90% target. This is evidence that this frozen
artifact under-covered in the declared synthetic generator. It does not prove
that a random-stream defect remains or that pooled conformal calibration is
intrinsically invalid.

The post hoc diagnosis localized 19/20 misses to the selector's voltage branch
under temperature-dependent-contact truth. The selector used 0.030400 K of
padding, compared with 0.070730 K for fixed voltage; fixed voltage covered 14
of those 19 missed devices. Small-sample calibration variation and pooling
heterogeneous action branches are both plausible contributors. This diagnosis
was not predeclared and is not a universal causal result.

The reserved fixed-policy results provide the following descriptive tradeoff.
No selector-versus-fixed contrast was predeclared before the reserved cohort,
so the table does not support a confirmatory superiority claim.

| Procedure | Definitive decisions | Block coverage | Mean runs | Mean realized energy | Balanced loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stop now | 69/150 | 45/50 | 2.00 | 60.86 J | 0.540 |
| Fixed thermal | 87/150 | 45/50 | 5.00 | 99.49 J | 0.783 |
| Fixed voltage | 89/150 | 45/50 | 3.00 | 89.73 J | 0.654 |
| Fixed face temperature | 83/150 | 45/50 | 3.00 | 89.62 J | 0.694 |
| Parent stop-or-voltage selector | 98/150 | 38/50 | 2.65 | 79.54 J | 0.507 |
| Guarded selector | 93/150 | 40/50 | 2.49 | 74.86 J | 0.525 |

The fixed policies all reached 45/50 block coverage, while the parent and
guarded selectors reached 38/50 and 40/50. The selectors' higher decision
coverage or lower scalar loss cannot be read as evidence that they beat the
fixed policies because those apparent advantages coincided with undercoverage.
All procedures observed zero decision errors, but the approval counts are too
small to establish a near-zero conditional risk.

Nominal selection energy and realized terminal energy were close in this
cohort. The fixed procedures differed by at most 0.24% in their cohort means;
the parent selector's nominal mean was 79.35 J versus 79.54 J realized, and the
guarded selector's nominal mean was 74.64 J versus 74.86 J realized. The
nominal proxy therefore preserved the energy ordering in this cohort. This is
a descriptive result for the frozen generator, not evidence about reset
energy, sensor electronics, wall-clock time, or hardware energy.

Differences from the original Stage 3–5 campaign cannot be attributed to the
random-stream repair alone. The corrected replication also used twelve fit
iterations, buffered inference bounds, projected-KKT convergence, candidate
exclusion for bound hits and nonconvergence, and realized terminal-energy
accounting. Historical comparisons must identify this complete numerical
protocol change.

## Resource accounting

The action rule may use a predeclared nominal energy proxy because hidden device
parameters are unavailable when the action is chosen. Final scoring separately
integrates terminal power for the actual simulated device and instrumentation
on every acquisition and verification run. Reports label the two quantities as
`nominal_selection_cost` and `realized_terminal_energy`.

Reset time, reset energy, sensor electronics, and complete wall-clock decision
time remain outside the simulator. Reports must not call the energized schedule
duration total bench time or use the incomplete timer as a resource claim.

## Recorded runtime

The corrected scientific chain was generated with CPython 3.10.12. That exact
implementation and version are stored in the source manifest, and artifact
verification rejects a different runtime. This requirement applies to exact
replay of the corrected replication; the package itself continues to declare
Python 3.10 or newer for ordinary use.

## Source and diagnostic freeze

Before a reserved cohort can be generated, the serialized calibration artifact
must contain a sorted manifest of the numerical source files that define truth
generation, observation generation, inference, decisions, calibration,
resource accounting, and random streams. The evaluator recomputes each SHA-256
from the repository and rejects missing, added, or changed numerical
dependencies. A revision label alone is insufficient. Documentation outside the
allowlist may change without invalidating the numerical freeze.

The verifier also resolves every named Python module and requires its executing
source path to match the repository whose bytes were hashed. Each calibration
artifact binds a deterministic evidence digest covering observations, truths,
fits, covariance, decisions, realized resources, and the full random-stream
manifest. Wall-clock timing is retained in diagnostics but excluded from that
digest because it is not reproducible.

Every generated row is saved independently of plotting. The diagnostic record
includes the case identity, semantic stream keys, observed samples, fitted
parameters, covariance, objectives, bound and convergence status, verification
scores and failures, raw and calibrated envelopes, final truth margin,
decision, error indicators, selected action, and both nominal and realized
resources.

## External evidence policy

The gate-development and parent-calibration diagnostics were already committed
as complete JSON records when the parent rule froze. They remain preserved in
the existing commits; rewriting those commits would weaken the reveal
chronology. Later complete diagnostic files are kept outside normal Git history
so rehearsal, guard, and final evidence do not add hundreds of megabytes to
every checkout.

Git continues to track each frozen rule artifact, compact outcome table,
human-readable report, and a content-addressed evidence manifest. The complete
diagnostic JSON is validated, compressed with deterministic gzip metadata, and
uploaded to the GitHub release named in that manifest. Those releases were
reported by the GitHub API as mutable on 2026-09-17: their committed hashes make
replacement detectable, but do not prevent deletion or replacement. Each
manifest records the generating commit and canonical command, scientific partition,
record and stream-audit counts, deterministic evidence digest, exact raw and
archive byte counts and SHA-256 hashes, and the release URL. Downloading and
decompressing the archive must reproduce the raw JSON hash before its evidence
is used.

The archiver is an unbound postprocessor at
`tools/archive_operating_decision_evidence.py`. It is intentionally outside the
generator-v4 numerical source manifest: changing the already frozen replication
CLI merely to package an output would invalidate the parent artifact. The
postprocessor cannot generate or reinterpret a cohort. It rejects incomplete
partitions, dirty stream audits, changed evidence digests, non-HTTPS locations,
local absolute command arguments, and a source commit different from `HEAD`.

The raw defaults and local archive staging directory are ignored explicitly;
there is no repository-wide JSON ignore rule. The compact manifests are:

| Command | Tracked manifest | External complete diagnostics |
| --- | --- | --- |
| `rehearse-parent` | `thermotwin/OPERATING_DECISION_CORRECTED_REHEARSAL_EVIDENCE_MANIFEST.json` | `r2_parent_rehearsal` |
| `freeze-guard` | `thermotwin/OPERATING_DECISION_CORRECTED_GUARD_EVIDENCE_MANIFEST.json` | `r2_guard_development`, `r2_guard_calibration` |
| `evaluate-reserved` | `thermotwin/OPERATING_DECISION_CORRECTED_FINAL_EVIDENCE_MANIFEST.json` | `r2_reserved_evaluation` |

For every command, capture the full `HEAD` immediately before invoking the
synchronous replication process and pass that literal SHA to the postprocessor.
Commit the compact outputs and manifest, create the manifest's release tag at
that evidence commit, upload the content-addressed archive, then download
and verify its SHA-256 and gzip round trip before proceeding to the next reveal
boundary.

## Fresh namespaces and chronology

The following semantic partitions are reserved under replacement campaign
`operating_decision_audit_replication_corrected_v2_2026_09`. Their namespaces
are disjoint from Stages 1–5 and the abandoned v3 campaign. Gate development
and parent calibration were each generated once for the parent freeze. The
parent rehearsal was generated once after that artifact was committed. Guard
development and guard calibration were then generated once for the guard
freeze. Finally, reserved evaluation and its bootstrap were generated once
after that artifact was committed. No scientific partition will be reopened.

| Partition | Paired blocks per family | Status | Use |
| --- | ---: | --- | --- |
| `r2_gate_development` | 20 | Frozen | Refit the matched A/B verification gates. |
| `r2_parent_calibration` | 30 | Frozen | Refit fixed-policy and stop-or-voltage conformal paddings. |
| `r2_parent_rehearsal` | 20 | Completed | One frozen-rule development rehearsal. |
| `r2_guard_development` | 30 | Frozen | Refit the scalar acquisition guard on matched A/B only. |
| `r2_guard_calibration` | 30 | Frozen | Refit the complete guarded-procedure padding. |
| `r2_reserved_evaluation` | 50 | Completed | One final A/B/C paired comparison after the artifact is committed. |
| `r2_bootstrap` | 20,000 draws | Completed | Paired block uncertainty for the final contrasts. |

Execution order is enforced:

1. Finish and test the corrected generator, realized energy, source manifest,
   convergence status, and complete export without generating a scientific
   cohort; commit that source.
2. Freeze the protocol and numerical source manifest, then commit the generator
   freeze. Parent calibration refuses an untracked artifact, a working-tree
   change to any bound source, or bytes that differ from `HEAD`.
3. Generate gate development and parent calibration, then freeze, serialize,
   and commit the corrected parent rule and its evidence.
4. Start a new process, verify the committed parent artifact, and run the
   parent rehearsal.
5. Generate guard development and guard calibration without consulting the
   old Stage 5 rows.
6. Serialize the complete artifact and commit it.
7. Start a new process, verify the numerical manifest, and instantiate
   `r2_reserved_evaluation` once.
8. Save every row, run the independent arithmetic audit, and report fixed
   policies, the corrected parent selector, and the guarded selector.

The reserved evaluation cannot run from an in-memory calibration result. The
evaluator requires the serialized guard artifact and every source file bound by
its manifest to be tracked and byte-identical to blobs in the same current Git
commit. The partition runner and its lower-level generators also reject the
reserved partition unless they receive authorization produced by complete
guard-artifact, source-manifest, partition, and physics validation. Every
worker repeats that validation against the initially authorized Git commit
before generating its block.

The executable workflow separates each reveal boundary:

```text
python -m thermotwin.operating_decision_replication freeze-generator
python -m thermotwin.operating_decision_replication freeze-parent --workers 4
python -m thermotwin.operating_decision_replication rehearse-parent --workers 4
python -m thermotwin.operating_decision_replication freeze-guard --workers 4
python -m thermotwin.operating_decision_replication evaluate-reserved --workers 4
```

The first command hashes the numerical source without generating devices. Commit
that artifact before the second command. The second verifies that generator
artifact and all bound source against `HEAD`, generates only gate-development
and parent-calibration partitions, and writes the parent artifact. Commit the
parent artifact before the third or fourth command. The third starts a new
process, verifies the committed serialized artifact, and only then opens the
rehearsal. The fourth calibrates the guard and complete revised procedure while
leaving the reserved partition unopened. After its serialized artifact is
committed, the fifth command verifies the complete Git, source, and artifact
chain before opening the reserved partition.

## Analysis

The primary unit is the paired device block. Family variants are summarized
separately and averaged within a block for paired comparisons; they are never
treated as independent rows. Missing envelopes count as unbounded sets for the
procedure-set coverage event and as abstentions for decision coverage.

Report, without selective omission:

- false approvals and false rejections;
- approvals, rejections, abstentions, and definitive-decision coverage;
- family-specific and simultaneous block interval coverage;
- selected actions and guard alarms;
- diagnostic runs, energized schedule duration, added sensors, nominal action
  cost, and realized terminal energy;
- numerical failures, optimizer convergence, and complete fit diagnostics;
- empirical loss under every predeclared cost scenario; and
- paired block-bootstrap intervals, conditional on the corrected independence
  audit passing.

The primary comparison is the paired block mean of guarded-selector loss minus
parent-selector loss under the predeclared `balanced` cost scenario. Success
requires the upper bound of its 95% percentile bootstrap interval to be below
zero. The guarded rule is structurally monotone: it may replace a parent
decision with abstention, but it cannot add or reverse a decision.

The replication will not promote a winner solely because it has the smallest
scalar loss. A recommendation must state the associated decision coverage,
error counts, instrumentation burden, and the cost scenarios under which it is
preferred.

## Following experiment

This replication is complete and does not promote the scalar mismatch guard.
The separate prospective protocol has implemented development interfaces for
four-action selection, uncertainty estimation, and resource cost. It has not
opened a pilot, development, calibration, or reserved partition. Its next work
is to harden padded scoring and stopping semantics before a small disposable
draw-count pilot; see
[`OPERATING_DECISION_PROJECT_STATUS.md`](OPERATING_DECISION_PROJECT_STATUS.md)
and the
[`prospective partition ledger`](OPERATING_DECISION_PROSPECTIVE_PARTITION_LEDGER.md).
