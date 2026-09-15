# Corrected operating-decision replication

Status: generator freeze `c120e29` is superseded. An independent decision-rule
audit found that its corrected wrapper turned either candidate's bound hit or
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
No named partition in the replacement campaign has been generated.

## Purpose

The September 2026 audit found deterministic reuse of random draws across
measurement channels, diagnostic runs, and adjacent device blocks. That defect
breaks the sampling assumptions behind the Stage 4 conformal calibration and
the Stage 5 block bootstrap. This replication repairs the generator and repeats
the existing decision procedures before a more capable selector is developed.

The old Stage 3–5 outputs remain immutable, exposed descriptive records. Their
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
none remain. Actual acquisition failures, verification exceptions, returned
nonfinite verification scores, and uncertainty failures for an otherwise
eligible candidate remain case-fatal. Matched gate development also continues
to abort on an unreliable matched candidate rather than silently changing its
calibration sample.

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

## Resource accounting

The action rule may use a predeclared nominal energy proxy because hidden device
parameters are unavailable when the action is chosen. Final scoring separately
integrates terminal power for the actual simulated device and instrumentation
on every acquisition and verification run. Reports label the two quantities as
`nominal_selection_cost` and `realized_terminal_energy`.

Reset time, reset energy, sensor electronics, and complete wall-clock decision
time remain outside the simulator. Reports must not call the energized schedule
duration total bench time or use the incomplete timer as a resource claim.

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

## Fresh namespaces and chronology

The following semantic partitions are reserved under replacement campaign
`operating_decision_audit_replication_corrected_v2_2026_09`. Their full
campaign-plus-partition keys have never been generated and are disjoint from
Stages 1–5 and the abandoned v3 campaign.

| Partition | Paired blocks per family | Use |
| --- | ---: | --- |
| `r2_gate_development` | 20 | Refit the matched A/B verification gates. |
| `r2_parent_calibration` | 30 | Refit fixed-policy and stop-or-voltage conformal paddings. |
| `r2_parent_rehearsal` | 20 | One frozen-rule development rehearsal. |
| `r2_guard_development` | 30 | Refit the scalar acquisition guard on matched A/B only. |
| `r2_guard_calibration` | 30 | Refit the complete guarded-procedure padding. |
| `r2_reserved_evaluation` | 50 | One final A/B/C paired comparison after the artifact is committed. |
| `r2_bootstrap` | 20,000 draws | Paired block uncertainty for the final contrasts. |

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

After this replication closes the generator defect, the next protocol will
implement the original prospective selector. From the common acquisition it
will estimate the expected change in final-margin uncertainty for stop,
thermal, voltage, and face-temperature actions, divide that change by frozen
costs, and map the selected action over sensor and test-cost ratios. That
selector will receive its own development partitions and an unseen final
cohort; the corrected replication's reserved labels will not be used to tune
it.
