# ThermoTwin project audit — September 26, 2026

Audited revision: `56953c7784a9d685a53ed87ca01b7f2d3ae58f06` (`dev`).
Comparison boundary: `9a21aa7`, the September 17 prospective audit.

## Assessment

The reviewed synthetic physics remains internally consistent, the P4 engineering pass is reproducible from the retained records, and the Phase D0 archive validates in its recorded environment. I found three issues to address before extending this machinery into the large development campaign: an incomplete infinite-offset fallback, an understated memory budget, and a timing-dependent scientific digest.

These findings do not establish a defect in the completed P4 action-agreement calculation. They also do not establish that the selector improves engineering decisions. The current evidence supports a small numerical-stability pilot and an archive rehearsal. The original four-action decision-quality question remains open until development, independent calibration, and reserved evaluation are completed.

## Scope and method

- Reviewed the current project status, outline, completion plan, P1–P4 history, Phase C freeze, Phase D protocol/plan, and D0 result.
- Scoped the changes since `9a21aa7` to 40 files with approximately 15,771 inserted lines, and concentrated detailed source review on selector scoring, candidate transitions, information boundaries, archive validation, development protocol, and resource estimates.
- Used an isolated clean clone for execution. Read the shared working-copy report diff separately; its modifications concern presentation. Seven relevant local presentation modules parse, but their untracked figures and manuscripts were not independently regenerated.
- Ran the full dependency-equipped test suite under CPython 3.10.12; final result is recorded below.
- Independently loaded and validated the P4 parent and N32 archives, checked their committed addresses, and validated the D0 archive under its actual CPython 3.13.3 runtime.
- Re-ran the existing independently implemented Dormand–Prince physics probe, which writes the node equations directly and does not call the production right-hand side or integrator.
- Checked source/history boundaries, Python syntax, console-script imports, and relative documentation links.

No new pilot, development, calibration, or reserved cohort was generated. This audit adds only audit records and makes no production-code changes.

## Findings

### 1. [P2] Resolve the infinite-offset fallback before opening tuning data

The [Phase D plan](OPERATING_DECISION_PHASE_D_PLAN.md), lines 99–113, defines infinite offsets when the 18th of 20 block scores is infinite. It then says an infinite stop offset disables early stopping and an infinite action offset disables that action.

The current [offset type](../../thermotwin/studies/operating_decision_prospective.py), lines 106–139, accepts finite offsets only. A direct construction with infinity in any of the four fields raises `ValueError`. There is no separate early-stop-disable flag in the selector rule. More fundamentally, the acquisition ranking uses `W_initial + 2*d_stop` as its common baseline. Merely disabling the stop gate leaves an infinite baseline and undefined/nonfinite utilities when `d_stop` is infinite.

This is an unresolved protocol/interface boundary, rather than evidence that D0 or P4 calculated a wrong decision: those runs used finite zero development offsets. The actual Phase D tuning executor has not been implemented yet. However, the protocol claims the fallback is fixed before data generation, and the needed mathematical rule is not specified.

**Required correction:** before opening tuning, either declare an infinite stop offset a Phase D feasibility failure, or define a finite utility baseline together with an explicit disabled-stop state. Define unavailable-action masks separately from offset magnitudes. Also state whether D4's “all measurement actions eligible” condition concerns numerical eligibility or includes actions deliberately disabled by the offset fallback. Add constructed finite/infinite boundary cases that exercise the entire selection path. Do not choose the fallback after inspecting which action receives infinity.

### 2. [P2] The recorded memory “cap” excludes the coordinating process and archive handling

The [budget calculation](../../thermotwin/studies/operating_decision_prospective_pilot.py), lines 5412–5418, multiplies the maximum worker RSS by the worker count. The [Phase D plan](OPERATING_DECISION_PHASE_D_PLAN.md), lines 262–267, calls the resulting 185,204,736 bytes a concurrent memory cap.

That calculation excludes the parent process, its retained block results, serialized byte buffers, parsed JSON, and validation copies. In a fresh CPython 3.10.12 process, simply loading the already retained 34,992,556-byte N32 JSON produced peak RSS of **192,626,688 bytes**, before running its validator and without any worker processes. Therefore 185 MB is not a conservative whole-workflow cap. It is, at best, a worker-only estimate for the measured pilot.

**Required correction:** rename the existing quantity to make its scope explicit; measure coordinating-process plus worker memory during generation, serialization, load, and validation; and budget memory by phase and archive size. Prefer per-block persistence or streaming where feasible. Measure the 20-block workflow before extrapolating to 100-block archives. The result does not prove the machine will run out of memory; it proves the recorded cap is unsupported.

### 3. [P2] The Phase D “scientific result” digest changes with runtime alone

The [Phase D rehearsal](../../thermotwin/studies/operating_decision_prospective_phase_d.py), lines 743–748, hashes the entire `block_result` and a replay summary whose block digest also includes timing. Unlike the earlier pilot's `prospective_pilot_scientific_payload`, it does not remove performance measurements.

I copied the saved block, added one second to `block_wall_seconds`, and recomputed the digest using the production functions. The timing-stripped physical/numerical evidence remained identical, but the scientific-result digest changed. This prevents that digest from serving as a stable identity for numerically identical reruns. It does not compromise the full-archive checksum, which appropriately binds every recorded byte.

**Required correction:** keep full timing in the archive seal, but construct a timing-independent scientific payload and block digest for reproducibility comparisons. Test that changes in timings leave the scientific digest unchanged while changes in observations, fits, decisions, or protocol alter it. Version this change; preserve the original D0 archive and its existing addresses.

The phrase “deterministic replay” should also be qualified: the present validator checks recorded structure, identities, streams, formulas, source, and hashes. It does not independently rerun all predictive simulations and nonlinear fits. The project status already describes this limit accurately; the D0 summary should use equally precise wording.

## Verified results

| Check | Independent result |
| --- | --- |
| Full repository test suite | **829/829 pass**, zero failures/errors/skips, CPython 3.10.12, 1155.461 seconds. |
| Python syntax | 354 tracked package/test files parse. |
| Console scripts | All 35 declared entry points resolve in the dependency-equipped environment. |
| Documentation | 55 Markdown files checked; no missing relative-link targets under the stated check. |
| Core source drift | No changes since `9a21aa7` in physics, simulation, inference, observations, or PINN modules. Corrected replication numerical algorithms and final outcome records are unchanged. |
| Historical replication | Generator source/runtime manifest and parent/guard artifact validation pass in a `9a21aa7` checkout with CPython 3.10.12. |
| P4 parent | Saved JSON validates; 15,395,653 bytes; SHA-256 matches `91f7d21c…5770bad`. |
| P4 N32 | Saved JSON validates; 34,992,556 bytes; SHA-256 matches `749e6896…2f66cf`. Final acceptance is true, with all 12 choices agreeing. |
| D0 | Saved JSON validates under recorded CPython 3.13.3, including the 84-file source manifest; SHA-256 matches `ace47b99…4e50ae`. Three cases, no pipeline failures, no N16 action ineligibility, no whole-draw failures. |

The P4 parent alone is correctly marked not yet accepted: it required the N32 continuation. The combined result passes. All 1,104 N16 predictive records are retained, including 378 candidate-loss transitions. Those transitions receive the no-gain floor and are distinguished from true whole-draw failures. P4 observed zero whole-draw failures, so its pass does not empirically establish performance under frequent numerical failure.

### Physics

The lumped model's Peltier, Joule, conductive, terminal-voltage, and power expressions agree with the standard single-stage equations in [Ferrotec's technical reference](https://thermal.ferrotec.com/technology/thermoelectric-reference-guide/thermalref11/). The identity `Q_h - Q_c = V*I` follows under the declared constant-property assumptions. The distributed model separately includes temperature-dependent properties and the Thomson term through the conservative energy equation.

The independent 38-case probe covered all three operating-decision truth families, loaded/unloaded nominal cases, ten earlier deterministic development draws, a fast parameter corner, and a reference-step refinement. Results:

- Largest final-margin difference: **8.882e-8 K**.
- Largest instantaneous energy-balance residual: **4.441e-15 W**.
- Minimum computed entropy production: **0 W/K**, with no negative values.
- Largest full-trajectory cold-face difference: **0.006360 K**, in the intentionally fast, loaded corner.
- No sampled-margin versus finer-reference minimum gap in these cases.

These are numerical/model consistency checks for the stated cases. They do not prove global continuous-time extrema accuracy over every possible parameter set, calibrate material properties to hardware, or validate all omitted physics. The larger transient corner error should remain visible alongside the much smaller final-margin error.

## Scientific interpretation and remaining design work

The corrected replication's negative guard result and parent undercoverage remain the appropriate conclusions. This audit did not reopen or retune that experiment. Its historical files require their historical source environment: checking the old generator manifest against today's HEAD correctly rejects the changed `pyproject.toml`, which now includes new prospective entry points. That is functioning provenance enforcement, not a reason to discard the historical result.

The new P4 result is a 12-case engineering heuristic. It is not a population guarantee of action stability, correct decisions, or interval coverage. The current documentation largely preserves this distinction.

Two additional decisions should be explicit before Phase D/E execution:

1. **Runtime identity.** Phase C's compute freeze records CPython 3.10.12, while the D0 artifact records CPython 3.13.3. D0 correctly fails source/runtime validation under 3.10.12 and passes under 3.13.3. State which environment will produce tuning evidence and record any migration and corresponding budget adjustment. A structural protocol freeze should not leave two different runtime assumptions implicit.
2. **Development loss and final intervals.** The Phase D grid specifies loss weights and action-score offsets, but should explicitly bind whether the definitive decisions used in its loss are recomputed from development-expanded final intervals. Reusing the fixed-policy raw decisions would optimize a different decision rule from one that later applies those offsets. Define the transformation, retain raw and adjusted decisions, and keep the independent Phase E correction outside tuning.

Preserve the independent calibration boundary and simultaneous three-family block target. Marginal conformal validity does not guarantee that every particular calibration realization achieves the target coverage; the [tolerance-region interpretation](https://arxiv.org/abs/2210.14735) remains relevant. Twenty development block scores and their 18th order statistic are an action-selection heuristic, not that final guarantee.

The strongest-fixed-policy comparison, risk/coverage non-inferiority rules, calibration correction, sample-size precision, final three figures, and reproducibility release remain unfinished. They are required work, not failed results. Sensor-quality maps must actually alter the corresponding simulations/likelihoods; cost-only rescoring cannot establish sensor-quality robustness.

There is also minor status drift: the end of the Phase D plan still calls D0 the next task although its opening status and the D0 result say it is complete. The project-wide `thermotwin/ROADMAP.md` execution order also predates this operating-decision campaign. Reconcile those pointers with the current project-status document so a future execution task starts at the intended boundary.

## Recommended next action

Resolve the fallback and loss semantics, separate scientific identity from timing, and correct the memory/runtime budget before authorizing the tuning executor. Review and test that executor on constructed boundary cases and a disposable transport check. Then open only the committed tuning partition. Keep the internal check, calibration, and reserved evaluation behind their existing separate freezes.

No restart of the corrected replication or P4 study is indicated by these findings. Preserve their artifacts and version the prospective corrections.

## Evidence and limits

Audit outputs are stored in [project_audit_2026_09_26](project_audit_2026_09_26/):

- [Check record](project_audit_2026_09_26/audit_checks.json).
- [Full test log](project_audit_2026_09_26/tests.log).
- [Independent physics results](project_audit_2026_09_26/physics.json).
- [P4 addresses and acceptance records](project_audit_2026_09_26/p4_evidence_summary.json).
- [D0 validation in its recorded runtime](project_audit_2026_09_26/d0_validation.json).

This was a project-wide test/integrity audit with deeper review of changes since September 17. It was not an independent rerun of every historical training campaign, manufacturing/material-data study, or distributed-PINN result. Tests and sampled independent calculations cannot certify all possible inputs or hardware behavior. Public release assets and GitHub CI were not re-downloaded or independently re-run in this audit.
