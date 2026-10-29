# Training-only optimizer probe and four-case 2D forward rerun

This development experiment follows the completed
[graded-grid log comparison](PINN_SPATIAL_GRADED_LOG_COMPARISON.md). The previous
log-off method passed both full-contact checks and failed both partial-contact
checks. All four temperature-field checks passed, while the partial-contact
heat-equation and bottom-boundary residuals remained above their original
limits. Inverse fitting remains blocked until a single frozen method passes
all four complete forward checks.

## Information boundary and unchanged problem

The optimization probes use copies of the already-open partial-contact
log-off checkpoints, their original training points, and the same noisy
temperature observations. They load only observed coordinates and observed
temperatures, with the frozen 0.02 K standard deviation. No clean temperature,
hidden reference field, independent residual diagnostic, acceptance outcome,
or supplemental audit enters optimizer selection.

Architecture, exact interface continuity, the log-off input, graded grid,
training-family weights, loss normalization, coefficient solver, and singular
value cutoff remain fixed. The probes assess the actual complete training
objective after re-solving the linear coefficients. Their gradients are
proposal directions; truncated SVD does not justify an exact profiled-gradient
claim.

## Bounded probe declaration

The directional-probe declaration was retained before its first trial. It
also declared the subsequent history comparison and the four-case start and
budget policies before directional outcomes were available.

1. On each copied partial-contact checkpoint, compare fresh Adam step
   multipliers 1, 4, 16, and 64 relative to the old learning rate 0.0001.
   Check central finite differences of the re-solved objective at maximum
   parameter displacements 0.00001 and 0.000001. The cap is nine readout solves
   per case, 900 seconds, and 2 GiB process RSS, with no retry.
2. Choose the common multiplier that minimizes the worse of the two objective
   ratios, requiring a finite strict decrease on both cases.
3. Compare eight updates with reset versus persistent Adam history, starting
   each arm from the same original checkpoint. Each proposal permits four
   halvings. The cap is 160 trial readout solves plus four baseline solves,
   1,800 seconds, and 2 GiB process RSS.
4. Select persistent history only if it is no worse on both cases within
   relative tolerance 0.00000001 and improves the worse terminal objective
   ratio by at least 0.0001. Otherwise retain reset history. Failed or
   nonfinite trajectories cannot justify selection.

The directional probe selected multiplier **64**, giving learning rate
**0.0064**. A single step reduced the complete training objective by 3.50%
and 5.12% for the two partial-contact cases. Directional finite differences
agreed with the fixed-readout proposal derivative to approximately 0.05–0.06%
at both tested displacements; retained rank was 508. This local check supports
the proposed direction at these states, rather than asserting differentiability
through every future rank transition. The directional probe took 97.19 seconds
and stayed below 1 GiB RSS.

The history comparison completed in 270.23 seconds, with peak process RSS
approximately 1.18 GiB. Persistent history was selected by the declared rule:

| Partial-contact case | Original training objective | Reset after eight updates | Persistent after eight updates |
| --- | ---: | ---: | ---: |
| h = 800 | 4.22588 | 3.80436 | 3.53621 |
| h = 1400 | 7.85061 | 6.96529 | 6.47939 |

All sixteen persistent updates were accepted at full scale. Each reset arm
accepted all eight updates, requiring fifteen trial solves. This is a short
training comparison from existing fitted states; it does not demonstrate a
converged fresh model or predict that the independent forward gate will pass.
The [compact probe record](../../thermotwin/PINN_SPATIAL_OPTIMIZATION_PROBE.json)
binds the declarations, results, source identities, and selection decision.

## Optimizer mechanics and four-case rule

The additive optimizer implementation snapshots hidden weights, readout,
Adam moments, counters, parameter groups, and gradient buffers. A rejected
or exceptional transaction restores that complete state. An accepted
backtracked displacement commits one Adam gradient-state update. The raw
report records accepted/rejected updates, trial objective values, ranks,
conditioning, and optimizer step counts. Final fit checkpoints retain the
optimizer state.

The four-case rerun starts **every case from its original fresh, matched
log-off initialization**. No probe checkpoint is used as an initial state.
Every case receives 320 fixed update attempts and at most four halvings per
proposal. The complete sequential campaign has a four-hour wall limit,
2 GiB process RSS limit, one thread, and no retry or budget extension. Failed
and unrun rows remain in the four-case denominator. The optimizer-returned
final state and raw training report are retained before independent scoring.
The selected learning rate is 0.0064 with persistent Adam history. The complete
[four-case declaration](../../thermotwin/PINN_SPATIAL_OPTIMIZATION_PLAN.json)
was retained before launch, with SHA-256
`dbd86371ac170aae688bd41dbecc99d6aced12036425dc42879439f0bde03b7b`.
The attempt ended at its four-hour resource limit. Two fits completed, the
third hit the limit during training, and the fourth did not start. The saved
overall gates retain **one pass in four declared cases**. The complete
[saved result](../../thermotwin/PINN_SPATIAL_OPTIMIZATION_RESULT.json) and
ignored `optimized-v6-report/REPORT.md` retain the failed and unrun rows.

The stop request was initially blocked by the execution sandbox; when the
approved stop command resumed, the resource guard had already ended the
attempt. No successful manual interruption is claimed. In the training logs,
t01 repeated 253 rejected transactions after its 67 accepted updates; t02
repeated 249 after its 71 accepted updates. The third case had 124 accepted
updates before its repeated rejections and resource interruption. These are
training-only observations, independent of scored physics or field errors.

The complete software suite for this revision passed 1,117 tests in 1,309.59
seconds. This proof covers the V6 source identities; subsequent additive
controller work has its own checks.

The separately frozen [bounded step-control follow-up](PINN_SPATIAL_STEP_CONTROL.md)
preserves this attempt and addresses the repeated-work failure. It does not
replace or retry this declaration.

Complete rollback with a fixed learning rate can repeat a rejected proposal.
That possibility is retained in the history; the campaign does not introduce
an undeclared learning-rate fallback or choose a checkpoint using acceptance
results.

Acceptance remains unchanged:

- Normalized observation MSE at most 2.
- Every applicable independent normalized physics RMS at most 0.05.
- Hidden-field RMSE at most 0.05 K and maximum error at most 0.15 K.

The existing near-interface and whole-plate bottom-layer audits remain report
only. Neither audit changes the acceptance gate. The two partial-contact probe
datasets and the four final forward datasets are reused development evidence,
not independent replications or an unseen evaluation cohort.

## Evidence locations

- Directional probe: `reports/pinn-spatial-validation/optimization-directional-probe/`.
- History probe: `reports/pinn-spatial-validation/optimization-history-probe/`.
- Four-case campaign: `reports/pinn-spatial-validation/optimized-v6-development/`.
- Execution and software validation: `reports/pinn-spatial-validation/optimization-validation/`.

Raw evidence remains in the ignored `reports/` directory. Compact declarations,
probe summaries, and final results are retained under `thermotwin/` for review.
Byte-identical archives of the executed preparation and probe scripts are
versioned in [tools/pinn_spatial](../../tools/pinn_spatial); their identities
are bound in the compact probe record. Raw declarations retain the actual
executed paths under `reports/`.
The prior v1/v2/v3/v4 and graded-grid evidence, and the separate operating-decision
Phase F experiment, remain protected by their original source and artifact seals.
