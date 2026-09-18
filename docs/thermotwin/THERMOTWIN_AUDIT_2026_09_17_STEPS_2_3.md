# ThermoTwin audit: pushed `origin/dev` at `9a21aa7`

> **Post-audit status, 2026-09-17:** this historical audit predates the
> disposable 12-case pilot. The pilot verified the no-gain attrition score but
> exposed a separate eligibility defect: Step 2 also counted ordinary
> candidate exclusion as whole-draw instability, which could invalidate an
> otherwise usable action. The finding, hashes, and bounded versioned repair
> are recorded in
> [`OPERATING_DECISION_PROSPECTIVE_PILOT_V1_RESULT.md`](OPERATING_DECISION_PROSPECTIVE_PILOT_V1_RESULT.md).
> Subsequent adversarial review also strengthened the P2/N32 archive
> validators while P2 remained unopened. Both changes are ordinary pre-P2
> defect repairs, not favorable scientific results. Their final validation is
> recorded in the current pre-P2 revision; P2 may open only from its committed
> and pushed form after CI passes.
> The original audit text below is preserved as issued.

Date: 2026-09-17 (second audit of the day). Scope: the two commits pushed since the previous
audit at `307a23a`:

- `d39f178` — prospective uncertainty estimator (Step 2) and prospective random streams.
- `9a21aa7` — prospective action cost model (Step 3).

About 6,700 lines including tests. Audited from a fresh clone of `origin/dev`; the local
working tree was not read. No scientific partition was generated or reopened. The clone was
deleted after the audit.

## Bottom line

The new work is clean. I found no defect in Steps 2 and 3, and the completed replication is
untouched and still verifies. Two things need attention before Step 5 freezes anything, and
one finding from the previous audit is still unaddressed in the documents.

## Verified

| Check | Result |
| --- | --- |
| Frozen replication | Generator, parent, and guard artifacts still validate; the guard chain is byte-identical at HEAD (77 paths). No allowlisted numerical file changed. The frozen CLI's import closure does not pull any prospective module. |
| No new cohorts | No scientific artifact, outcome file, or partition was added. Roadmap discipline holds: Steps 4–7 remain pending. |
| Information boundary (Step 2) | Synthetic future observations are simulated from *fitted* parameters drawn from the fit's own covariance, never from truth. The estimator takes no truth-family label, device token, verification result, true margin, or final response. The Step 2 evidence factory refits the exact common initial run internally, which closes the unauthenticated fit-provenance gap that Step 1 openly flagged. |
| Prospective random streams | Keys bind campaign, partition, block, acquisition-evidence digest, source model, draw index, purpose, action, run, and channel, and validate that a run and channel actually belong to the action. Sharing must be *explicitly declared*, and only for parameter draws, with one distinct action per use. Undeclared exact-key reuse and derived-seed collisions are both reported, and the estimator refuses a result whose audit is not clean. This is stronger than the `r2` registry, which inferred pairing from equal keys. |
| Estimator semantics | One shared pre-action baseline (the provisional envelope width); common random numbers for physical draws across the three actions; probe nuisance only for the face action; mean within a source model, worst case across source models; no model probabilities; negative reductions retained. |
| Failure and attrition policy | Every declared draw stays in the denominator; failures are imputed at the pre-action width; candidate attrition is scored as no gain. Matches the documentation. |
| Cost model | Independently reproduced from the frozen nominal proxies: incremental energies 38.5717 / 28.8143 / 28.6909 J and times 480 / 160 / 160 s; normalized costs 1.446211 (thermal), 1.0 (voltage), 0.998573 (face). The 12-cell grid contains the primary scenario exactly once, weights are convex, and the grid genuinely reorders actions (thermal is cheapest at 0.651 under instrumentation-dominant weights; face ranges 0.474–3.099). |
| Step 3 to Step 1 bridge | Eligible actions copy Step 2 values exactly; ineligible actions get no partial score; stop-now stays outside the cost ratio; the selector always uses the snapshot bound inside the same Step 2 result, so one case's scorecard cannot be paired with another's snapshot. |
| Tests and CI | 743 tests pass on the clean clone (up from 686; 74 new tests, including leakage, reuse, and collision cases). CI is green on both new commits. |

## Findings

### [P1, carried over and still unaddressed] The replication's coverage conclusion is missing

Nothing in these two commits touches the replication documents, so the finding from the
previous audit stands: the frozen stop-or-voltage selector covered 80/100 fresh blocks
(95% [71.1%, 86.7%], binomial p = 0.002 against its 90% target), while fixed voltage covered
92/100. The documents still report only "below the 90% target" without drawing the
conclusion, and the README row still describes the replication in future tense.

This matters more now, not less: Step 5 will calibrate a *four-action* selector, and the
mechanism that produced the shortfall is a pooled padding applied to heterogeneous branches.

### [P2] The value function optimizes raw width, but decisions will use padded width

Step 2 scores an action by the reduction in the **raw** candidate-envelope width. The final
approve/reject decision will instead use the **calibrated** envelope, raw width plus twice the
procedure's padding. In the completed replication those paddings differed substantially by
branch (selector 0.030 K versus fixed voltage 0.071 K), and that gap is exactly what caused
the coverage shortfall.

Before Step 5 freezes a calibration:

1. Plan per-action (group-conditional, or Mondrian) padding rather than one padding for the
   whole procedure, and size the calibration cohort per action.
2. Once paddings exist, either include the action's padding in the value function or
   demonstrate that the ranking is insensitive to it.
3. Report per-branch coverage in development, not only the pooled figure.

The prospective document does not yet describe any calibration design, so this is the right
moment to settle it.

### [P2] Measured runtime: plan the draw count and worker budget now

Measured on a disposable namespace, single core:

- Acquisition evidence (two models, three starts): 6.4 s.
- Step 2 with `draw_count = 1` on a device with **one** admissible source candidate: 61.3 s,
  about 20 s per action-draw.

Extrapolating, per device: about 4 minutes at N = 4 with one admissible candidate, roughly
8 minutes with two, and 16–33 minutes at N = 16. For a cohort the size of the `r2` campaign
(540 device-families), that is roughly 13–18 hours at four workers for N = 4, and on the
order of two days for N = 16, before the policy execution itself. The work is deterministic
and parallel by block. Scope the planned N = 4 / 8 / 16 stability study to a small disposable
cohort.

### [P3] The stability threshold confounds the planned draw-count study

Eligibility requires `ceil(0.90 * N)` stable draws per source candidate. That equals N for
both N = 4 and N = 8 (no unstable draw allowed at all), but permits one at N = 16. The
planned N = 4 / 8 / 16 comparison therefore varies Monte Carlo precision and eligibility
strictness together. Express the threshold as an explicit integer allowance, or report stable
counts alongside each N.

### [P3] Two small documentation points

- Probe nuisance parameters are drawn from the **truth support** (1–12 J/K, 0.5–6 s) with the
  prior spread, not from the wider inference bounds. That is legitimate, because the outline
  declares the probe calibration range as known to the policy, but the document should say so.
- Under four-state truth the five-state candidate is usually bound-excluded, so "worst case
  across source candidates" reduces to a single model in exactly those cases. This is benign
  (the remaining model is the true family), but worth stating where the worst-case claim is
  made.

### [P3] Carried-over documentation items

Unchanged from the previous audits: the Stage 2 document has no audit note and keeps its
35.4% Wilson bound; the Stage 4 body still states the conformal guarantee; the replication
document does not state the CPython 3.10.12 requirement and still says exceptions are fatal
"for an otherwise eligible candidate"; four tracked documents contain `/Users/rolandbennett`
in plain text.

## Not examined

- Release-asset contents (outside the branch; their GitHub-side hashes were checked in the
  previous audit).
- The local working tree.
- Scientific re-audit of the non-operating-decision experiments.
