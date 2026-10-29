# Continuous regularized readout: completed development attempt

The [cutoff audit](PINN_SPATIAL_CUTOFF_AUDIT.md) found a local rank boundary
in both terminal partial-contact fits. This additive attempt replaces the
abrupt singular-value discard with one continuous regularized readout. The
architecture, graded training points, noisy observations, original matched
initialization and independent forward acceptance thresholds stay fixed.
The training objective deliberately gains the explicitly reported penalty.

**Outcome: two of four cases pass the unchanged forward gates.** Both
full-contact cases pass. The h=800 partial-contact fit reaches its resource
limit without a scored final field; the h=1400 partial-contact fit completes
but severely regresses on independent physics and field accuracy. The method
checks pass, but this candidate does not clear partial-contact validation.

## Completed method verification

The frozen training-only probe completed on October 29, 2026, in 93.25 seconds:
two fresh-field calibration SVDs, fourteen readout solves and zero neural
updates. Both cases passed every predeclared method gate. All original
scientific evidence and numerical sources remained unchanged.

| Terminal partial-contact copy | Analytic directional derivative | Finest central finite difference | Largest relative head-stationarity residual |
| --- | ---: | ---: | ---: |
| h=800 | -6.651731800 | -6.651717796 | 3.27e-12 |
| h=1400 | -8.524842818 | -8.524867071 | 3.96e-12 |

The two finest finite differences also agreed under their frozen tolerance.
All solved heads retained all columns. Seventeen constructed model/probe
tests passed before the numerical verification. The
[prospective declaration](../../thermotwin/PINN_SPATIAL_REGULARIZED_METHOD_PLAN.json)
and [completed method result](../../thermotwin/PINN_SPATIAL_REGULARIZED_METHOD_RESULT.json)
bind this evidence. Its plan hash is
`204c115f893971aecafb276863d4d94552641b457ad164f877d86e6ec90ef82a`.

This verifies the declared local numerical method on the two training-only
copies. Independent forward physics accuracy remains a separate question.

## Completed forward comparison

The [four-case declaration](../../thermotwin/PINN_SPATIAL_REGULARIZED_PLAN.json)
was frozen after the method checks passed and four fresh initialization
calibrations completed. The once-only two-wave comparison ran on October 29,
from 19:35:01 to 22:23:36 UTC. Its authoritative plan hash is
`ec6d6b6bf1f6fe291e71647d3262559836a018b47dd94379188a608c0acdf4b3`.
The [completed result](../../thermotwin/PINN_SPATIAL_REGULARIZED_RESULT.json)
retains all four opportunities, including the unscored resource failure.
Clean supervisor completion is separate from scientific success: the saved
forward status is `blocked_forward_gate`.

| Case | Geometry, h (W/m²K) | Independent PDE RMS | Field RMS (K) | Field max (K) | Outcome |
| --- | --- | ---: | ---: | ---: | --- |
| t01 | Full, 800 | 0.0286703 | 0.00879469 | 0.0409573 | Pass |
| t02 | Partial, 800 | Unscored | Unscored | Unscored | Resource failure |
| t03 | Full, 1400 | 0.0322604 | 0.00801372 | 0.0404267 | Pass |
| t04 | Partial, 1400 | 34.5961 | 0.288071 | 1.95600 | Physics and field fail |

The unchanged limits are normalized PDE and each applicable boundary/interface
RMS at most 0.05, field RMS at most 0.05 K, maximum field error at most 0.15 K,
and observation normalized MSE at most 2. All three scored cases pass the
observation gate. In t04, bottom-contact RMS is 0.899581 and bottom-insulation
RMS is 0.535111; the enforced interface temperature and flux continuity remain
essentially exact.

t01 and t03 finish 320 attempts, using 412 and 431 trial head solves. t02 hits
the declared resource limit at 3,000.49 monotonic seconds; its last complete
progress event records 277 attempts and 423 trial solves. Those partial events
are retained, but they are not a completed fit or a scored field. t04 finishes
315 attempts and 477 trial solves before the declared projection-budget stop.
The controller reserves room for its maximum backtracking transaction, so it
can stop with fewer than 480 trials. Each completed fit also has its separately
counted initial and final head solves. No worker retries or supervisor
interventions occur.

The campaign records 6,018.80 monotonic seconds; the surrounding supervisor
records 6,022.67 seconds. Largest worker RSS is 1,093,189,632 bytes, below the
2 GiB limit. All four initialization calibrations occur once in preparation;
workers build no additional calibration matrices or singular-value
decompositions. Calendar waiting time is discussed below.

At its recorded snapshot, the full shared-workspace software suite passed all
1,239 tests in 25.19 minutes, with zero failures, errors or skips. All 495
tested source/proof hashes remain unchanged, including the complete frozen
PINN numerical closure. Seven unrelated operating-decision Python files were
added concurrently after that run; they are outside this QA proof and this
PINN commit. The test count describes the recorded snapshot, not that later
whole worktree. The finalizer authenticates the log, counts, tested source
manifest, campaign binding, saved outcomes and sealed report before writing
the compact result. Independent final verification confirms the exact saved
rows, report and finalization seals, calibrations and preserved earlier
evidence chains. This is local software verification, not a CI result.

The saved comparison is under
`reports/pinn-spatial-validation/regularized-v8-report/`; its JSON and figure
retain all twelve method/case combinations. The frozen generated Markdown
has a formatting limitation: the t02 failure paragraph interrupts its metric
table. The four-case table above supplies a clean summary while the original
sealed output remains unchanged. The actual comparison PNG was visually
inspected after rendering.

## Interpretation and next scientific question

The completed t04 heads are full-rank and stationary: the final relative
stationarity residual is 5.54e-12. All 315 hidden updates are accepted with
nonincreasing regularized objective, from 9.42184 after initial projection
to 3.85401. Its final original loss is 3.83690 and penalty is 0.0171090.
These are valid numerical-method checks; they do not certify forward accuracy
or nonlinear convergence.

t04 training PDE RMS is about 0.174583, versus 34.5961 on the independent
points. Even the training PDE residual exceeds the physics limit. Its
independent field error is also substantially worse than V7's 0.0161259 K RMS
and 0.0642284 K maximum, and V7's independent PDE RMS was 0.253116. Better
observation fitting and a lower training objective therefore do not establish
a better field. The penalty changes the optimized objective, so regularized
and unregularized total objectives must not be ranked directly.

The training-to-audit discrepancy is consistent with insufficient control of
variation away from collocation points. This experiment does not isolate
whether basis flexibility, collocation coverage, the fixed penalty or their
interaction is responsible. The h=800 resource failure separately establishes
a runtime bottleneck; it supplies no final independent-accuracy result.

The next useful work is a separately bounded, training-only stability and
runtime diagnostic before choosing another candidate. It should inspect
point coverage and variation between training locations, coefficient scales
and the cost of matrix construction versus the readout solve. It should
distinguish these issues from incomplete nonlinear optimization before
proposing stronger regularization, a different basis or a different point
rule. This completed attempt opens no additional search or numerical run.
Inverse fitting and 3D remain blocked by forward validation.

## Fixed numerical method

For each case, build the original training system `A0, b0` at the fresh common
initial field. Freeze `D_j = max(norm(A0[:, j]), 1e-12)` and
`lambda = 1e-7 * largest_singular_value(A0 / D)`. The factor 1e-7 links the
regularization transition to the previous cutoff. It is one declared
candidate, with no penalty sweep or claim that it is optimal.

At every hidden-parameter point minimize
`J = norm(A*c - b)^2 + lambda^2 * norm(D*c)^2` over the head `c`.
Solve the augmented system `[A/D; lambda*I] z = [b; 0]`, with `c = z/D`,
using float64 pivoted QR and a triangular solve. Do not truncate singular
directions or retain an older head as a fallback. Any invalid factorization,
nonfinite result or failed stationarity check is a method failure.

Both `D` and `lambda` remain constant throughout each fit. Therefore, for a
sufficiently stationary solved head, the derivative of the profiled **total
regularized objective** with respect to hidden parameters is the fixed-head
derivative of the original physical/observation loss. This does not assert
the same derivative identity for the unregularized profiled objective.
Report physical loss, penalty and total objective separately.

Outer Adam proposals accept only a nonincreasing regularized total objective.
They retain the preceding bounded controller, persistent moments and exact
rollback of hidden parameters, head and optimizer state on rejection.
No frozen previous numerical module is edited.

## Training-only method verification before a four-case run

Freeze the complete source, runtime, input identities and gates before the
first scientific numerical check. Use the two original terminal partial
checkpoint copies and their noisy training observations only. For each case:

1. Calibrate `D` and `lambda` at the **fresh original** common initialization,
   then transfer the terminal field parameters while preserving calibration.
2. Solve the baseline ridge head. Compute the un-clipped hidden gradient of
   total `J` there and normalize its negative by its largest absolute component.
3. Independently restore that baseline before positive and negative
   displacements of 0.0001, 0.00001 and 0.000001 along that fixed direction.
   Re-solve the unique ridge head at each point.

The exact budget is two calibration singular-value decompositions and fourteen
augmented readout solves, with zero neural optimizer updates. The probe has a
900-second wall limit, 2 GiB process memory limit and one thread for all
numerical libraries. Retain every failed or unrun opportunity. No alternative
direction, radius extension, retry or method search is allowed.

Every declared point must have finite physical, penalty and total losses,
full augmented column rank and the following stationarity measure at most
1e-8: `norm(C.T @ (C*z-d)) / max(1, norm(C.T @ d))`, where
`C=[A/D; lambda*I]` and `d=[b;0]`.
Directly evaluated and algebraic total objectives must agree within
`1e-8 + 1e-8*max(abs(direct), abs(algebraic))`.

The central finite difference at the **predeclared finest** radius must agree
with the un-clipped AD directional derivative within
`1e-4 + 0.005*max(abs(FD), abs(AD))`.
The two finest central finite differences must agree within
`2e-4 + 0.01*max(abs(FD_fine), abs(FD_middle))`.
Do not choose the best radius after observing the values. All method gates
must pass on both cases and all source/input preservation checks must pass
before a forward campaign can be prepared.

## Conditional four-case comparison

After method verification passes, freeze one uniform four-case rule. Use the
original fresh matched initialization in all four cases; terminal probe
states never seed the campaign. Calibrate each fresh case once during
preparation and bind that calibration for the worker to load and verify.

Retain the 320 update-attempt and 480 trial-solve budgets, learning rate
0.0064, four backtracking halvings, eight-failure controller limit and the
existing learning-rate floor and growth rules. Count calibration operations
separately from initial/trial/final readout solves.

Run the declared t01/t02 and t03/t04 waves with at most two isolated workers.
Each worker has one thread, 2 GiB process RSS and 3,000 seconds; the supervisor
has a 7,200-second cap. No retry or resource extension is permitted. Preserve
all four declared opportunities, including failures and untouched cases.
Save each terminal training checkpoint before independent scoring.

Recorded worker and supervisor durations use the predeclared monotonic timer,
not CPU time. Calendar waiting time was longer: at 21:33:05 UTC, approximately
118.1 minutes had passed since the supervisor's UTC start, while its monotonic
timer recorded 50.1 minutes. This discrepancy is consistent with host
suspension, but its cause was not independently established. No timing rule
or budget was changed, and the numerical duration is not a calendar deadline.

The original acceptance limits remain observation normalized MSE at most 2,
each applicable normalized independent physics RMS at most 0.05, field RMSE
at most 0.05 K and maximum field error at most 0.15 K. Existing near-edge and
bottom-layer audits remain report-only. Passing the method checks is distinct
from passing these four forward gates.

The earlier V7 comparison remains two of four passing. This separate candidate
also passes two of four, with a resource failure and a severe partial-contact
regression retained in its denominator. No candidate is promoted on the basis
of this attempt.

Raw evidence stays in ignored `reports/pinn-spatial-validation/` directories;
compact plans, decisions and final results belong under `thermotwin/`.
