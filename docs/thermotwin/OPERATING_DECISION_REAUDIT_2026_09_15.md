# Re-audit: operating-decision replication at `ecb5030`

Date: 2026-09-15. Scope: commits since the previous audit (`a78030d`, `2a0ee76`, `7df8fcc`,
`ecb5030`), the committed parent freeze and its evidence, and the status of every earlier
finding. No repository file was modified during the audit. No partition was generated: every
check reads committed JSON or calls a pure forecast function on serialized fits. Nothing else
was running on the machine during the checks.

## Bottom line

The P1 problem is fixed, and the fix is correct. The committed parent freeze is sound and
reproduces exactly from its own evidence. Nothing blocks `rehearse-parent`.

The open items are design and documentation choices, and the most important one should be
settled before `evaluate-reserved`: predeclare how the selector will be compared with the
fixed policies. Do **not** change any allowlisted source file until the reserved evaluation
is committed, because any change invalidates the frozen chain.

## What changed

| Commit | Content |
| --- | --- |
| `a78030d` | Candidate exclusion restored (bound-hit and finite non-converged candidates excluded; the case abstains only when none remain or on a genuine failure). Box-projected KKT convergence. Double-counted failures removed. Campaign rotated to `operating_decision_audit_replication_corrected_v2_2026_09`. Generator v4. Recovery-probe evidence. |
| `2a0ee76` | Generator v4 frozen; the replication document records the supersession of `c120e29`. |
| `7df8fcc` | Record of a disposable five-command dry run in a throwaway clone. |
| `ecb5030` | Scientific parent freeze: `r2_gate_development` (240 records) and `r2_parent_calibration` (360 records), the parent artifact, diagnostics, and outcomes. |

No numerical source changed after `a78030d`, and the physics digest equals v3. Across v3
to v4, no tuning constant changed (iterations, tolerances, bounds, gate and padding
settings). The abandoned v3 parent outputs were never committed on any ref, and v4 uses a
new campaign, so the already-opened v3 draws are not reused.

## Verified

| Check | Result |
| --- | --- |
| Fix logic | The projected gradient uses correct KKT signs. The wrapper and gated rebuild exclude inadmissible candidates and rebuild the envelope. No duplicate failure entries remain. |
| Generator v4 | The manifest verifies at HEAD, and the artifact digest rebuilds identically. |
| Parent artifact | Validates; committed at HEAD with all 77 bound paths byte-identical. |
| Evidence digests | Recomputed independently from the committed diagnostics JSON; both equal the artifact. |
| Stream audits | Gate: 4,040 uses, 0 unintended reuse. Calibration: 6,060 uses, 0. Only the v4 campaign and the correct partition names appear. |
| Allowlist | Equals the runtime import closure exactly (74 modules); pure Python. |
| Independent recomputation | All four gate thresholds match exactly, including the SHA-256 stream seeds, the 50,000-draw noise references, and observation counts of 162 and 243. All five paddings and block nonconformity vectors match. All 450 outcome rows match field by field. All summaries and loss scenarios match. 101 stored intervals were re-forecast from serialized fits exactly. |
| Recovery probe and dry run | The recorded JSON matches the document: 42/48 versus 15/48; 11/12 versus 1/12; 27 correct decisions restored; all five commands exited 0; six clean stream audits; the reserved commit matches the guard boundary. |
| Tests | 664 tests, OK, in 282.8 s (CPython 3.10.12). |

## What the calibration cohort shows (in-sample, descriptive)

Paddings were fitted on these rows, so they are not performance estimates. The rehearsal is
the first out-of-sample check.

| Procedure | Decisive (all) | Family A | Family B | Family C | Realized energy | Balanced loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stop now (padding 0 K) | 44/90 | 21/30 | 16/30 | 7/30 | 60.68 J | 0.511 |
| Fixed thermal (0 K) | 53/90 | 23/30 | 19/30 | 11/30 | 99.19 J | 0.775 |
| Fixed voltage (0.0707 K) | 63/90 | 20/30 | 24/30 | 19/30 | 89.46 J | 0.547 |
| Fixed face temperature (0.0571 K) | 65/90 | 22/30 | 23/30 | 20/30 | 89.37 J | 0.525 |
| Stop-or-voltage selector (0.0304 K) | 66/90 | 18/30 | 27/30 | 21/30 | 77.01 J | 0.407 |

- Zero false approvals and zero false rejections across all 450 procedure rows.
- Family A has recovered. Fixed voltage is decisive on 20/30 devices, even though its
  five-state candidate was excluded for a bound hit in 29/30.
- All four gate thresholds are set by the chi-square noise floor, not by matched
  development scores (matched 1.17–1.22, floor 1.22–1.28).
- Part of the selector's edge comes from padding, not only from the action it chooses. The
  selector chose voltage on 51/90 rows and stop on 39/90. On its voltage branch, with
  identical raw intervals, its smaller padding decided 6 devices that fixed voltage left
  undecided. On its stop branch, its larger padding abstained on 4 devices that stop-now
  decided. Read selector-versus-fixed comparisons in the rehearsal and reserved cohorts
  with that in mind.

## Open findings

### [P2] No predeclared comparison with the strongest fixed policy

The only inferential endpoint is still guarded selector minus parent selector
(`operating_decision_replication_guard.py:1543`). The calibration cohort already makes
selector versus fixed voltage and face temperature the obvious question. Any such
comparison made after the reserved reveal would be post hoc.

Changing the guard or report code now would break the chain. Instead:

1. Before `freeze-guard`, or at least before `evaluate-reserved`, predeclare secondary
   paired-block contrasts in the replication document, which sits outside the allowlist:
   selector minus fixed voltage, selector minus fixed face temperature, and selector minus
   stop now. Cover loss under each scenario, decision coverage, runs, and realized energy.
   Include a multiplicity statement.
2. Commit a standalone analysis script, outside the allowlist, that reads the reserved
   outcomes JSON and bootstraps with a new stream purpose.

### [P3] Selector signal ignores convergence

`initial_decision_signal` (`operating_decision_calibration.py:534`) checks acquisition
failures and bound hits, but not `converged`. In v3, non-convergence was an acquisition
failure. In v4 it is not, so a finite non-converged fit can contribute to a "stop" choice
while the stop-now decision itself excludes that fit. `acquisition_adequacy_signal`
(`operating_decision_replication_guard.py:297`) reuses the same envelope.

This occurred 0 times in the 90 calibration selector signals; 7 non-converged Family C
candidates were excluded elsewhere. Do not change code now. Document it, and report its
count in the rehearsal and reserved results.

### [P3] Single use of the reserved cohort is still procedural

The v3 episode is exactly this mechanism, and rotating the campaign was the right response.
Add a pre-run wrapper outside the allowlist that refuses `evaluate-reserved` if
`OPERATING_DECISION_CORRECTED_FINAL_OUTCOMES.json` appears anywhere in Git history. Commit
the reserved outputs immediately after the run.

### [P3] Reserved diagnostics are written only after summarization

`thermotwin/reports/operating_decision_replication.py:660–668` writes diagnostics only after
evaluation and summaries succeed. The dry run shows the path completes, so accept the risk
rather than change code. Ensure about 100 MB of free disk and an uninterrupted run.

### [P3] Documentation

- Replication question 1 still reads as if differences from Stages 3–5 come from the RNG
  fix alone. They also reflect 12 iterations, 2× fit bounds, and exclusion of non-converged
  candidates.
- The document says verification exceptions and nonfinite scores are fatal "for an
  otherwise eligible candidate" (lines 131–137). In code, a verification exception is fatal
  for any candidate, including a bound-hit one; this has occurred 0 times.
- Still carried over: no statement that CPython 3.10.12 is required;
  `OPERATING_DECISION_EXPERIMENT.md` (Stage 2) has no audit note and keeps its 35.4% Wilson
  bound; the body of `OPERATING_DECISION_CALIBRATION.md` (line 253) still states the
  conformal guarantee; no analysis for question 3 (nominal versus realized energy); the
  outline deliverables (risk–coverage sweep, decision figure, measurement map) are not
  mentioned.

### [P3] Engineering (carried over)

- The stream audit auto-derives pairing IDs (`operating_decision_replication.py:623`).
- The collision test uses non-production channel labels (`tests/test_operating_decision_replication.py:97`).
- There are no independent-solver or convergence-corner regression tests.
- The manifest records the Python version but not platform or architecture.

### [P3] Repository size

Committed diagnostics are 81 MB raw (`.git` is 27 MB compressed). The remaining partitions
add about 210 MB raw. The reserved diagnostics file will be about 81 MB (1.63 MB per block),
above GitHub's 50 MB warning and below its 100 MB limit. The chain verifies artifacts, not
diagnostics, so compressing diagnostics after writing, with recorded hashes, is safe.

## Resolved since the previous audit

- P1 whole-case abstention and unprojected convergence test.
- P2 end-to-end chronology never executed.
- Double-counted numerical failures.
- Stale status and inaccurate "still fail closed" wording.

## Uncommitted working tree

Unchanged since the previous audit: presentation-only edits and scripts, none allowlisted,
so `rehearse-parent` verification is unaffected. The duplicate `_old`, `_paired`, and
`_corrected` scripts should still be cleaned up.

## Evidence

The verification scripts (`verify_chain.py`, `recompute_parent.py`) and the full test-suite
log were written to a temporary session directory, which was cleared after the session
ended. The results above are as recorded when the checks ran. The scripts read only
committed JSON and can be recreated and re-run on request.
