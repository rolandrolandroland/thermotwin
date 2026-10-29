# Bounded step control: final 2D forward development comparison

The preceding [optimizer attempt](PINN_SPATIAL_OPTIMIZATION_REPAIR.md) exhausted
its resource cap after repeatedly making identical rejected proposals. This
additive follow-up changes the optimizer controller and execution bounds,
while retaining the same model, training observations, graded grid, loss,
coefficient solver, singular-value cutoff, and acceptance thresholds.

## Completed outcome

All four fits and their forward checks completed on October 23, 2026, in
3,754.73 seconds (62.58 minutes). **Two of four complete gates passed**:

| Geometry and known h | Heat-equation RMS | Bottom contact RMS | Bottom insulation RMS | Complete gate |
| --- | ---: | ---: | ---: | --- |
| Full contact, 800 | 0.029360 | 0.019235 | not applicable | pass |
| Partial contact, 800 | 0.093487 | 0.046731 | 0.080000 | fail |
| Full contact, 1400 | 0.030766 | 0.015896 | not applicable | pass |
| Partial contact, 1400 | 0.253116 | 0.068747 | 0.115185 | fail |

Every listed physics limit remains 0.05. All four observation and
temperature-field checks passed. Both partial runs stopped by the declared
learning-rate-floor rule after 81 and 119 attempts. Both full-contact runs
completed 320 attempts. Execution is now bounded and all four cases were
scored, but the partial-contact physics did not improve over the original
log-off comparison (heat-equation RMS 0.087093 and 0.116271).

The [final saved result](../../thermotwin/PINN_SPATIAL_STEP_CONTROL_RESULT.json)
binds the [complete report](../../reports/pinn-spatial-validation/step-control-report/REPORT.md),
source identities, probes, all prior failures, and unchanged acceptance rules.
The numerical revision passed all 1,142 repository tests. A postprocessing
compatibility repair has a separate set of 20 passing targeted tests.

Reporting originally stopped because the completion helper expected test
status `passed`, whereas the test harness wrote `completed` with `passed=true`.
On October 29 the helper was repaired to validate that schema together with
test counts, terminal log, log hash, source manifest and source identities.
The report and compact result were then finalized without rerunning any fit
or changing scientific artifacts. Original failed helper/state evidence is
preserved under `step-control-validation/completion-repair-20261029/original/`.

## Follow-up diagnostic

The saved partial-contact trajectories stopped at the hard truncated-SVD
rank boundary. Final retained ranks were 510/550 and 514/550, with resolved
conditions 9,999,986.7 and 9,999,994.4, just below the 10,000,000 cutoff implied
by `rank_rcond=1e-7`. Their final rejected proposals each dropped one retained
direction and raised the projected objective. This identifies a numerical
boundary to investigate; it does not prove that correcting it will pass the
independent physics checks or establish an architectural accuracy floor.

The proposed next study is a separately declared training-only directional
audit on copies of the two terminal partial-contact checkpoints: one fixed
direction, three fixed positive/negative perturbation sizes, and two baseline
head solves, for 14 head solves total and no hidden optimization. It should
record rank, distance to the cutoff, the pure truncated-SVD objective, and
the actual returned projection-policy objective. Finite differences should
be compared with the fixed-head derivative only where rank is stable.
The existing projection can reject a newly solved head and retain its prior
head, so both objective definitions must be distinguished.

The separately frozen [fourteen-solve cutoff diagnostic](PINN_SPATIAL_CUTOFF_AUDIT.md)
completed on October 29 without neural training. Both cases changed rank at
every declared positive displacement, supporting a projection correction
before a corner-field revision. It does not establish improved independent
physics accuracy. A replacement method still needs its own declaration and
four-case check. Inverse fitting and 3D remain blocked by the current forward
gate.

## Frozen declaration and evidence

The final uniform four-case comparison is executed under its authoritative
[plan](../../thermotwin/PINN_SPATIAL_STEP_CONTROL_PLAN.json) and
[training-only probe record](../../thermotwin/PINN_SPATIAL_STEP_CONTROL_PROBE.json)
retained before any worker fit. Final saved outcomes are written to
`thermotwin/PINN_SPATIAL_STEP_CONTROL_RESULT.json` only after all completion,
report-seal and passing software-test checks. Until that record exists, the
execution and completion-pipeline state files under `step-control-validation/`
describe progress. No inverse, unseen evaluation cohort, or
3D study is opened. The prior V6 attempt, including failed and unrun cases,
remains protected and appears alongside the original log-off baseline in
the new saved-evidence report.

## Recovery evidence and prospective decision

The first controller probe, C, used copies of the saved stalled full-contact
checkpoint and the two original partial-contact checkpoints. It read only
training observations and points. With a three-failure stopping limit it
failed its declared strict recovery criterion: the full-contact copy stopped
after three unsuccessful smaller-rate searches. Both partial prefixes
accepted all sixteen updates, reducing training objectives from 4.22588 to
3.29018 and from 7.85061 to 6.00170. C remains explicitly recorded as failing
its original criterion.

A final, separately frozen mechanics check, D, changed only the consecutive
failure limit from three to eight. Its declaration committed to one uniform
all-four development comparison and no further optimizer search. It reported
mechanical validity and actual recovery separately; unsuccessful bounded
proposals would not have been called stationarity.

The stalled full-contact copy accepted its fourth recovery attempt. Its
carried learning rate became 0.00005, and the next four attempts also
decreased the complete training objective. The objective decreased from
0.485686193 to 0.485168480. Both original checkpoint and source identities
were preserved. D reused the two accepted partial prefixes from C with an
explicit execution-equivalence proof: the sole changed failure threshold
was never consulted in either all-accepted prefix. These reused prefixes
are not presented as new fits or independent evidence.

This supports the bounded recovery mechanism. It does not demonstrate
convergence, accuracy at independent points, or four successful forward gates.

## Frozen optimizer and resources

Each case starts from the same original fresh, matched log-off initialization.
The optimizer retains Adam history. Rejected transactions restore hidden
weights, readout, moments, counters, parameter groups, and gradient buffers
before the separate controller changes its state.

- Initial learning rate 0.0064; at most four backtracking halvings per proposal.
- After an exhausted search, halve the carried rate and restart moments.
- After an accepted search using at least two halvings, carry its accepted
  scale into the next rate, clamped at the 0.0000001 floor.
- After eight consecutive full-scale accepts, double the rate up to 0.0064.
- Stop after eight consecutive exhausted searches, a rejected floor-rate
  search, an inability to change the next proposal, 320 update attempts, or
  the 480-trial-solve budget. Initial/final coefficient solves add two calls.

Stopping by these rules is budget or stagnation termination, not a convergence
certificate. The terminal state and raw report are saved before scoring.

The four once-only workers run in declared waves: t01/t02, then t03/t04. Each
has one thread, 2 GiB process RSS and a 3,000-second wall limit. At most two
workers run concurrently, for a 4 GiB aggregate child-process RSS envelope.
The supervisor has a 7,200-second limit and records every observed child exit,
failure and untouched opportunity. No automatic retry or budget extension is
allowed. A failed worker cannot consume another worker's allocation.

## Unchanged scoring and interpretation

- Observation normalized MSE at most 2.
- Every applicable independent normalized physics RMS at most 0.05.
- Hidden-field RMSE at most 0.05 K and maximum error at most 0.15 K.

Near-interface and whole-plate bottom-layer audits retain their report-only
roles. Source, point, row-weight, initialization, controller-history and
budget proofs are checked before independent diagnostics. The final result
retains all four declared cases regardless of outcome. All data are previously
opened paired development evidence; no independent replication is claimed.

Raw campaign, report and software evidence remain in ignored directories
`reports/pinn-spatial-validation/step-control-development/`,
`step-control-report/` and `step-control-validation/`. Executed preparation
and probe scripts are versioned under [tools/pinn_spatial](../../tools/pinn_spatial).
A separately bounded local completion process waits for the scientific
supervisor and full software suite, then renders the saved-evidence report and
validates/writes the compact result automatically. It cannot start or retry
fits. A failed completion stage records its error and stops.
