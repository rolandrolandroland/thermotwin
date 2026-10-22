# Graded-grid log-coordinate comparison

This is the next bounded development experiment after the completed v3/v4
continuity repair. Both arms launched on October 22, 2026 at approximately
16:22 UTC, after the numerical plans, initialization proofs, grids, and source
closure were frozen. The declaration has SHA-256
`7d3847b0c1bf2aeb1233051e580dbd60c66958e232adc03db1aa2639fd7f5d0e`.
All eight fits completed without execution failure or retry. Log-off passed
2/4 complete forward checks; log-on passed 1/4. Neither arm passed all four,
so the complete forward gate remains blocked.

## Completed outcomes

All acceptance coordinates, scales, and thresholds were retained unchanged.
The table shows normalized heat-equation RMS and the saved overall gate;
the physics limit is 0.05 for every applicable family.

| Geometry and known h (W/(m² K)) | Log-on PDE RMS | Log-on overall | Log-off PDE RMS | Log-off overall |
| --- | ---: | --- | ---: | --- |
| Full contact, 800 | 0.035449 | pass | 0.026110 | pass |
| Partial contact, 800 | 0.100597 | fail | 0.087093 | fail |
| Full contact, 1400 | 0.050779 | fail | 0.032808 | pass |
| Partial contact, 1400 | 0.134466 | fail | 0.116271 | fail |

All eight fits passed the observation and temperature-field checks. Field RMS
errors ranged from 0.00632 to 0.01067 K, with maximum errors below 0.051 K.
Temperature continuity remained exact and normalized interface-flux jumps
remained at roundoff. Both partial-contact cases still failed the interior
heat equation and the bottom contact/insulation conditions in both arms.
For log-off, the partial-contact bottom-condition RMS values were
0.06378–0.09587, above the unchanged 0.05 limit. Accurate temperature values
therefore do not yet establish accurate local heat balance.

The thin-layer blow-up is substantially reduced. On the unchanged
near-interface audit, the preceding v4 values were 402.316 and 519.470;
log-off now records 0.26424 and 0.37181, while log-on records 0.29457 and
0.40393. This historical comparison combines changed sampling,
initialization, and budget; it does not isolate which change caused the
reduction. The new whole-plate bottom audit also remains elevated in the
partial-contact cases: 0.19057 and 0.25325 for log-off. Both audits remain
report only and do not replace the original acceptance gate.
Log-off's partial-contact bottom-audit values are slightly higher than
log-on's 0.18257 and 0.25232, despite lower primary PDE RMS. The measured
benefit is therefore not a uniform improvement in local physics.

The optimizer behavior limits interpretation. Log-off accepted all 80 hidden
updates in every case; log-on rejected all 80 in every case after its allowed
backtracking. Because its log-input column started at zero and every hidden
proposal was rolled back, log-on's returned hidden representation never
activated that input. Its linear readout was still fitted. The comparison
shows an optimization difficulty when that input is available under this
initialization and optimizer; it does not establish how a successfully
optimized log-input model would perform. Log-off objectives were still
decreasing at the budget endpoint, so convergence is not claimed for either
arm.

The parallel comparison took 4,769.19 seconds (79.49 minutes). Individual
campaign elapsed times were 4,755.05 seconds for log-on and 1,603.70 seconds
for log-off; peak process memory stayed below 1 GiB in each. Log-on's rejected
updates required three readout trial solves per update, versus one for
log-off's accepted updates. These are recorded execution costs, not a
controlled timing comparison against prior methods.

The complete software suite passed **1,097 tests**. Before/after checks
confirmed the prior numerical sources, v1/v2/v3/v4 evidence, old reports, and
separate operating-decision Phase F gate were preserved. No inverse fit,
unseen reserved cohort, or 3D study was opened.

The [complete report](../../reports/pinn-spatial-validation/graded-log-report/REPORT.md)
retains all eight outcomes, component gates, trajectories, conditioning, and
audits. The [compact result and provenance](../../thermotwin/PINN_SPATIAL_GRADED_LOG_RESULT.json)
are tracked in Git. Raw checkpoints, diagnostic arrays, figures, and sealed
reports remain in the ignored `reports/` directory.

The remaining development target is the partial-contact heat equation and
mixed bottom conditions. Log-off is a useful next development baseline, but
this comparison does not select or freeze a method for unseen evaluation.
Any further change needs a separately bounded plan; inverse fitting remains
blocked by the four-case gate.

## Question and rationale

V3 and V4 enforced temperature and normal-flux continuity but each passed
0/4 complete forward checks. V4 also repaired the top derivative restriction
and accepted its hidden updates. Those are useful implementation results;
they do not establish an accurate forward solution. The
[previous result](../../thermotwin/PINN_SPATIAL_CONTINUITY_REPAIR_RESULT.json)
and [continuity report](PINN_SPATIAL_CONTINUITY_REPAIR.md) remain preserved.

Subsequent checkpoint-only AD probes indicate that the large supplemental
PDE errors occur in a thin bottom layer, including points far from contact
edges and full-contact cases. They are not explained by failed interface
continuity. The logarithmic input `log(1 + zeta / 0.001)` has sharp second
derivatives in this layer. Bottom boundary equations constrain first
derivatives, while the first interior training row leaves a substantial gap
above the boundary. Large fitted coefficients and cancellation can amplify
unconstrained curvature there.

The next experiment asks whether retaining that learnable logarithmic input
helps or harms a model trained on the same explicit graded grid. It does not
assume that removing the input will pass the forward checks.

## Two arms and four paired datasets

The two arms are `log_on` and `log_off`. Both reuse the existing full-contact
and partial-contact datasets at known conductances of 800 and
1400 W/(m² K). This gives **eight declared method fits on four reused paired
datasets**, not eight independent datasets or noise replications.

Both arms retain shared Hermite temperature and normal-derivative traces,
the paired top-consistent readout, geometry-derived corner functions, the
same observed temperatures, and the same float64 truncated-SVD readout
solver. No hidden reference field enters training. Fields are used only for
the existing post-fit scoring operation.

Initialization matches common weights exactly. The extra logarithmic input
column starts at zero in `log_on`; `log_off` omits that column and clones all
common weights. Every bias and common input column is copied unchanged by the
matching step.
The extra column can subsequently learn in `log_on`. Saved proofs bind the
common state, initial field and coordinate derivatives, arm identity, and
probe coordinates. The absence of a column in `log_off` is recorded as absence,
not as a fabricated zero column.

This isolates the input-column choice under the new common sampling and
training budget. Comparison against the saved V4 uniform-grid results is
historical context: changed sampling and budget mean it is not a controlled
sampling-only comparison.

## Frozen collocation and objective

Each arm uses the same 8,192 interior PDE points for each paired case.

| Geometry | Bulk points | Bottom-band points | Contact-corner patch points |
| --- | ---: | ---: | ---: |
| Full contact | 6,144 | 2,048 | 0 |
| Partial contact | 4,096 | 2,048 | 2,048 |

The graded bands cover the gap between the bottom boundary and the previous
first interior row, while retaining bulk, top, and early-time coverage. The
uniform mean over all interior points is frozen. The implied interior-loss
weights are 0.75/0.25 for full-contact bulk/bottom and 0.5/0.25/0.25 for
partial-contact bulk/bottom/corner strata. These are collocation-emphasis
weights, not physical volume quadrature weights. Applicable bottom-condition
families retain the same 512 saved points as the preceding construction.

The plan and raw training reports preserve point identities, digests, stratum
bounds and counts, and objective weights. Neither arm adapts its points after
seeing independent diagnostics.

## Optimizer budget and acceptance

Every declared fit uses 80 fixed hidden variable-projection updates and at
most two halvings per proposal. Every proposed hidden state receives its own
linear readout solve before its complete training objective is compared with
the preceding state. A rejected proposal restores both the hidden weights
and readout; optimizer state is reset. The returned final state is the scored
checkpoint. There are no automatic retries, additional seeds, adaptive budget
extensions, or checkpoint selection by hidden-field error.

Eighty updates are a declared budget, not a convergence certificate. The
report retains accepted and rejected updates, objectives, backtracking, rank,
conditioning, coefficient norms, elapsed times, and process memory. Under
truncated SVD the fixed-readout gradient is a proposal direction; it is not
claimed to be an exact profiled gradient.

The original primary acceptance coordinates and limits remain unchanged:

- Normalized observation MSE at most 2.
- Every applicable independent normalized physics RMS at most 0.05.
- Hidden-field RMS error at most 0.05 K and maximum error at most 0.15 K.

Failed and unrun rows remain in the eight-fit denominator. Both arms finish
before the combined final result is opened. The finalizer does not select a
winning arm or turn a failed component into acceptance.

## Separate diagnostic audits

The existing near-interface audit remains report only. A new declared
whole-plate bottom-layer audit covers both geometries at heights
`z/H = 0.001, 0.002, 0.005, 0.01, 0.02, 0.05` and the same explicit early and
late audit times. Saved aggregate and per-height normalized and physical PDE
RMS values allow bottom-layer errors to be distinguished from edge-local
errors. These audits neither replace the primary gate nor select a model or
checkpoint.

## Records and next decision

The declaration is
[PINN_SPATIAL_GRADED_LOG_PLAN.json](../../thermotwin/PINN_SPATIAL_GRADED_LOG_PLAN.json).
The ignored raw campaign lives in
`reports/pinn-spatial-validation/graded-log-development/`, with separate
`log_on/` and `log_off/` sealed subcampaigns. The top plan binds both subplans,
all numerical sources, matched initialization and grid proofs, and the
protected v1/v2/v3/v4 and operating-decision evidence.

The additive [reporter](../../thermotwin/reports/pinn_spatial_graded.py)
validates saved sources and artifact indices, retains all eight rows, and
displays coordinator gates without recalculating scientific acceptance. It
writes into a separate report directory and preserves physical and normalized
metrics, pairing, training trajectories, stratum weights, cost, conditioning,
and supplemental audits.

Passing forward checks would justify the separately planned delayed or
alternating inverse development stage. Failure would require a new explicit
decision informed by the complete comparison; it would not authorize an
unseen evaluation cohort or a 3D extension. The completed 1D information-design
work and conventional 2D reference remain available while this gate is open.
