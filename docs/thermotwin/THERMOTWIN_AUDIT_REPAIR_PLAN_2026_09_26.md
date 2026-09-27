# ThermoTwin audit repair and completion plan

Date: September 26, 2026. Status: proposed implementation instructions, not a protocol freeze or authorization to open a scientific partition.

This plan combines the September 26 project audit and the additional research-report review. It specifies the repairs, evidence needed to close each finding, and conditions for proceeding with the unfinished operating-decision experiment. Creating this plan does not implement the repairs.

## 1. Preserve the evidence and establish the repair baseline

Start from the reviewed revision `56953c7784a9d685a53ed87ca01b7f2d3ae58f06`, checking for subsequent changes before implementation. Use an isolated checkout for implementation and deliberately incorporate relevant working-copy presentation files; several reports and generators are currently untracked. Preserve unrelated working-copy edits.

Record the starting source revision, relevant local-file hashes, runtime, existing archive identities, and which audit finding each change addresses. Preserve the corrected replication, P4, original D0, and research bundle v1 as historical evidence. New schemas, protocols, rehearsal results, and presentation bundles receive new versions. Validate historical artifacts against their historical source/runtime; do not rewrite their manifests to make them match current code.

The existing 829-test pass is the baseline, not evidence that a future revision passes. No audit finding currently requires restarting the completed corrected replication or P4 study.

## 2. Correct the research reports and presentation assets

Primary files:

- `thermotwin/PINN_EVIDENCE_REPORT_cd.md`
- `thermotwin/EXPERIMENT_SELECTION_REPORT_cd.md`
- `reports/hax_research_2026_09_17_v1/` as the preserved source for a revised presentation bundle
- Relevant summary documents and report generators under `thermotwin/reports/`
- `thermotwin/figures/NONLINEAR_EXPERIMENT_SELECTION/`

Make these corrections wherever the affected claims appear in the active deliverables:

1. Place the temperature-range caveat next to the PINN property-error headline and in methods: evaluation is over 285–315 K, whereas training temperatures span approximately 294.83–305.21 K. Roughly two-thirds of the evaluation temperature interval lies outside that range. Do not describe this as two-thirds of the error magnitude, and do not imply that fitting uniquely identifies the entire curve.
2. Disclose that physics weight 10 was selected on trial 0 and then frozen before trials 1 and 2. Label the three-trial training-adequacy summary as including a development trial. Keep this separate from the ten-paired-trial inverse-PINN comparison, whose physics weight was 1.
3. State the comparison as approximately 1.75% versus 4.66% across the same ten paired trials. Where the all-20 conventional result is useful, label its approximately 5.07% mean separately. Do not substitute it into the paired comparison.
4. Replace coarse improvement ranges with approximately 3.9–6.6 times, or the exact per-parameter factors derived from the retained data. Replace ambiguous claims of uniform improvement with explicit direction and magnitude. This is a precision correction, not a newly discovered arithmetic failure.
5. Archive or visibly mark `next_experiment_story.png` and its associated sidecars as superseded. The old figure displays 76.0% instead of approximately 34.2% for naive sensor-lag error. Remove it from active presentation paths. Mark its legacy generator accordingly so an ordinary report build cannot silently recreate it as the current figure. Keep the historical files recoverable and repair any affected references.
6. Build the revised deliverables through their active generators. Preserve bundle v1; put revised outputs and an accurate manifest in a new bundle version. Record which output files changed and which scientific input files are unchanged.

Acceptance: reconcile the affected values and trial subsets against retained data; check all active figure/document links; visually inspect regenerated figures and any exported documents; verify that the active deliverables contain the caveats and no longer select the obsolete figure. No retraining is required to correct these disclosures. Optional in-range/out-of-range diagnostics would be additional, explicitly labeled retrospective analysis, not a prerequisite for this repair.

## 3. Resolve the infinite-offset protocol before tuning

Primary files: `thermotwin/studies/operating_decision_prospective.py`, `thermotwin/studies/operating_decision_prospective_phase_d.py`, `docs/thermotwin/OPERATING_DECISION_PHASE_D_PLAN.md`, and their focused tests.

Recommended protocol amendment: if any action's prescribed development offset is infinite, declare the primary four-action Phase D design infeasible and stop before selecting a tuning winner or opening the internal check. Retain all scores, failures, and available diagnostic results. This conservative choice keeps the finite-offset interface and the Phase C requirement that every measurement action remain available consistent.

This replaces the existing incompletely specified disable-action fallback and must be versioned before tuning data is generated. It is a recommendation, not a claim about the current frozen protocol. Do not replace infinity with a large arbitrary number or discard failed cases.

If continuing with disabled actions is an explicit scientific requirement, implement that alternative completely before tuning instead: an independent disabled-stop flag, measurement-action masks, a finite common utility baseline, a defined no-useful-measurement path, verification behavior, and explicit separation of numerical eligibility from deliberate disabling. Amend the Phase C/D eligibility requirements consistently. Do not choose between these alternatives after seeing which action fails.

Acceptance: constructed cases must exercise finite offsets, an infinite stop offset, an infinite measurement offset, several infinite offsets, and the rank boundary where the 18th of 20 scores becomes infinite. The primary executor must either produce valid finite scores under the declared rule or produce the prescribed feasibility-stop record. No undefined utility, silently missing denominator, or accidental fallback is acceptable. Finite zero-offset behavior must remain consistent with the existing baseline.

## 4. Define the decisions used in development loss

Specify one transformation for every fixed policy and the selector. Starting from an emitted raw final interval `[L,U]`, apply the development offset for that policy to obtain `[L-d_a,U+d_a]`. Recompute the definitive approval/rejection/abstention using this interval and the existing verification and failure rules. Verification failure or a missing interval must not become a valid decision because an offset was applied.

Use these development-adjusted decisions in the declared 81-rule tuning objective and internal check. Preserve the raw interval/decision alongside the adjusted interval/decision, verification status, failure status, action, and block-level loss. Truth may enter the offline loss only after the required policy records are saved; it must remain absent from selection inputs.

Keep the later independent Phase E correction out of development tuning and online action selection. Specify its application to the already development-adjusted final interval separately.

Acceptance: controlled examples must cover an approval becoming abstention after expansion, a rejection becoming abstention, an unchanged decision, exact decision-boundary behavior, verification failure, and absent/nonfinite intervals. Check all 81 objective rows against an independently calculated small fixture, including block averaging and tie breakers. Existing no-gain treatment of predictive failure and candidate loss must remain intact.

## 5. Separate scientific identity from archive identity

Primary files: `thermotwin/studies/operating_decision_prospective_phase_d.py`, its validator/report code, and `tests/test_operating_decision_prospective_phase_d.py`.

Retain the full-file checksum, which must bind timings and every other archived byte. Add a versioned, canonically serialized scientific payload that includes the numerical observations, fitted results, decisions, failures, random-stream identities, protocol, and source/runtime identity, while excluding elapsed-time and resource-performance measurements. Apply this separation consistently to nested block digests and replay summaries.

Acceptance: changing only elapsed time or recorded resource usage changes the complete archive identity but leaves the scientific digest unchanged. Changing a numerical observation, fit, decision, failure state, stream identity, or protocol changes the scientific digest. Include archive round-trip and tamper-detection checks. Do not claim that this promises identical numerical results across different runtimes.

Describe the current validator as archive/provenance and recorded-consistency validation. Reserve claims of computational replay for checks that actually rerun the simulations and fitting. The replacement disposable rehearsal should include an actual repeated computation under the same frozen environment and stream identities.

## 6. Resolve runtime and measure the whole-workflow budget

Recommended starting runtime: CPython 3.10.12, aligning with the Phase C budget and the completed project-audit suite. Pin and record the dependency environment. Preserve the original D0 result as a CPython 3.13.3 artifact.

If the project deliberately adopts another runtime, record that migration before development, run the relevant checks and disposable rehearsal in it, and replace the prospective budget assumptions with measurements from that environment. Do not silently treat the two environments as interchangeable.

Rename the 185,204,736-byte estimate as a worker-only estimate. Measure aggregate concurrent memory for the coordinating process and workers, including retained results, serialization buffers, parsing, and validation. Record process-tree sampling and each workflow stage; distinguish empirical estimates from enforceable limits. Include elapsed time, CPU time, disk use, and intermediate copies in the budget.

Before opening tuning, exercise a representative 20-block-sized disposable archive and its write/load/validation path using retained or constructed evidence. This checks storage and memory without exposing tuning labels. Measure the actual 20-block campaign as it subsequently runs and update scaling evidence before planning larger calibration/reserved archives.

Acceptance: publish the chosen machine limit, measured peak, justified headroom, worker count, artifact-size assumptions, and stop/resume behavior. If the workflow does not fit, reduce concurrency and use per-block persistence or streaming where appropriate, then repeat the resource check. Preserve scientific sample counts and draw counts. If essential work remains unaffordable, stop with a feasibility result; secondary sensor-map work is the first scope to reconsider under the existing protocol.

## 7. Reconcile status and close the audit findings

Update the project status, Phase D plan, D0 wording, completion-plan entry point, original-outline status, and project roadmap. State that original D0 is complete, protocol repairs and their replacement rehearsal are next, and the primary scientific comparison is unfinished.

Add a closure table mapping each finding to changed files, its test or document check, and retained evidence. Record the `archive bytes: 1` and repeated-`b` hashes as deliberate CLI mocks. They do not require a hashing repair: real serialization/hash tests and real-archive checks already exist. Clarify fixture output only if useful.

On the complete repair revision, run the relevant focused checks, the full repository test suite in the selected environment, and the repository's required CI checks on that exact revision. Run the applicable existing physics and information-boundary checks; investigate any changed behavior before generating scientific data. A passing suite is an engineering gate, not evidence of scientific superiority.

Run the revised disposable end-to-end rehearsal, including save/load, validation, scientific-digest checks, source/runtime identities, and repeated computation. Confirm that it uses a disposable namespace and contributes no observations to tuning, calibration, or reserved evaluation. Commit the repair record and revised protocol before opening tuning.

## 8. Complete the scientific work after the repair gate

The items below are unfinished project work, rather than defects established by the audits. Follow the revised Phase D protocol and the completion plan, preserving their separate data-use boundaries.

| Step and likely result | Required next action |
| --- | --- |
| Repairs and disposable rehearsal pass | Open only the 20-block tuning partition. Retain all four fixed-policy outcomes and all failures. |
| A required offset is infinite | Apply the fallback chosen and versioned in step 3. Under the recommendation, stop the primary campaign as infeasible; retain diagnostic evidence. |
| Finite offsets and valid tuning evidence | Evaluate the full fixed grid, choose the rule by the declared loss/tie breakers, and build the prescribed development maps. Cost-only maps may reuse predictions; altered sensor quality/loading requires the corresponding simulations and fits. |
| N=16/N=32 draw-sensitivity check passes | Retain N=16. |
| N=16/N=32 check fails | Follow the existing bounded N=32/N=64 continuation, with its budget recorded first. If that gate also fails, stop for numerical infeasibility. Do not substitute favorable subset cases. |
| Provisional rule and maps are complete | Freeze the provisional design, then open the 10-block internal check once. |
| Internal check is valid but finds little advantage or low decision coverage | Report the result and follow the declared advancement rule. Do not force action diversity or tune until the result looks favorable. |
| A predeclared internal-check redesign trigger occurs | Use only the existing bounded revision procedure and a newly committed replacement namespace. Preserve the first attempt. A second failure closes Phase D as infeasible. |
| Phase D freezes successfully | Finalize the strongest-fixed-policy comparison, error and decision-coverage denominators, non-inferiority rules, multiplicity handling, and sample-size/precision analysis before independent calibration. Record which numerical choices from the older completion plan are actually adopted. |
| Planned precision is inadequate | Increase the planned size if affordable before opening outcomes, or narrow the attainable claims explicitly. Never extend the reserved sample because the observed result is almost favorable. |
| Independent calibration produces a finite valid correction | Freeze it; broad intervals and increased abstention are retained outcomes. Proceed to the reserved evaluation. |
| Required calibration correction is infinite or calibration is invalid | Report infeasibility, or repair a demonstrated implementation defect under a versioned replacement protocol. Do not tune on calibration outcomes to recover a favorable result. |
| Reserved evaluation is valid | Classify the outcome as adaptive advantage, fixed-policy advantage, inconclusive, or feasibility limitation under the frozen rules. Each can complete the project. |
| A material defect compromises evidence | Preserve the incident and affected artifacts; repair and regenerate the affected evidence under the appropriate new version/namespace. Do not delete only inconvenient rows. |

Before independent calibration, implement and verify the chosen simultaneous three-family block score and order-statistic rule, including failed cases. Give fixed comparators the same calibration standard. The earlier completion plan contains recommendations that still need to be adopted explicitly; do not present them as existing validated guarantees.

The final deliverables are the decision-quality/resource comparison, risk/decision-coverage figure, measurement-selection map, and a reproducibility package binding code, environment, protocols, all attempted variants, artifacts, and revised reports. Keep synthetic-model evidence distinct from hardware validation. A null or negative result completes the scientific question when the protocol and evidence are sound.

## Completion criteria

- Active research deliverables disclose extrapolation, development overlap, and paired denominators; obsolete figures cannot be mistaken for current outputs.
- Infinite-offset behavior and development-loss semantics are fully specified, versioned, and tested before tuning.
- Scientific identity excludes performance noise while full archive integrity remains enforced.
- Runtime, total memory, compute, and storage assumptions are measured and explicit.
- Documentation points to the same current phase; every audit finding has a closure record.
- The repaired revision passes its tests, CI, and disposable rehearsal before scientific partitions open.
- Development, independent calibration, and reserved evaluation follow their separate freezes, or a declared feasibility stop is documented.
- Final claims match the frozen comparisons and precision; all evidence and superseded attempts remain traceable.
