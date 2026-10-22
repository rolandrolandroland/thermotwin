# Continuity and heat-equation repair

This development work follows the preserved v2 forward comparison. It reuses
the same four known-contact cases, observed temperatures, noise realizations,
reference fields, independent diagnostic coordinates, and acceptance limits.
It does not open inverse fitting or a reserved evaluation cohort.

## First bounded revision: v3

Shared cubic Hermite traces enforce temperature and normal heat-flux continuity
between regions. Exterior side insulation, the imposed top heat flux, and the
initial temperature remain exact. Bottom contact and insulation conditions are
residual equations; imposing both exactly at a smooth shared contact corner
would introduce an artificial edge-temperature constraint.

Geometry-derived corner features and a logarithmic bottom coordinate allow
edge behavior without hidden temperature labels. Linear output coefficients
are fitted to the PDE, bottom conditions, and sparse observations using
column-scaled float64 least squares. Hidden updates use the same training loss.

All four v3 fits completed in 352.22 seconds; none passed the overall gate.
Continuity was enforced, but heat-equation accuracy deteriorated. The sealed
[v3 report](../../reports/pinn-spatial-validation/continuity-v3-report/REPORT.md)
retains every case and the separately reported near-interface audit.

Two subsequent diagnostics explain why a corrective revision is needed:

- The top-flat coordinate made every learned feature even about the top.
  Together with the quadratic flux lift, this forced the third normal
  temperature derivative to zero. A changing top heat flux requires a nonzero
  third derivative through the heat equation. This is an unwanted restriction
  on the trial space, although its contribution to the observed error is not
  isolated by this diagnosis.
- Analytic hidden gradients agree with finite differences. With the output
  coefficients held fixed, a decreasing update required 15 halvings in the
  inspected full-contact case; the declared limit was 12. All 120 hidden
  updates were rejected. The existing rollback rule worked, but hidden features
  did not learn. The diagnostic restored all inspected weights exactly.

The original v3 source, plan, checkpoints, and results remain unchanged.

## One declared corrective revision: v4

Each existing feature is paired with an independent output coefficient for
`g(z/H) * feature`, where `g(s) = 3*s^2 - 2*s^3`. Because `g'(1)=0`, top heat
flux remains exact; `g'''(1)=-12` restores the missing third-derivative freedom.
Shared traces, exterior side insulation, and the initial condition remain
intact. The number of linear coefficients doubles; neural trunks are retained.

For each proposed hidden update, the output coefficients are solved again
before acceptance. Actual training loss must not increase. Rejection restores
both the hidden weights and readout. Rank, cutoff, conditioning, coefficients,
and trial acceptance are recorded; truncated least squares is not described as
an exact profile-gradient calculation.

The fixed corrective budget is 20 hidden steps, up to three proposals per
step, 24-unit/two-layer trunks, rank cutoff 1e-7, and learning rate 1e-4.
The full training grid has 8,192 interior points with 16 vertical levels;
bottom conditions retain 512 points per applicable family. The same four
cases share one configuration under a 3,600-second and 2-GiB process limit,
with no automatic retry. The complete plan and source hashes are saved before
these fits begin.

This is one corrective development comparison with several justified changes,
not an ablation identifying which change causes a numerical improvement.
V2's independent coordinates and limits remain the primary acceptance rule:
observation MSE at most 2, every normalized physics RMS at most 0.05, field
RMSE at most 0.05 K, and maximum field error at most 0.15 K. Supplemental
near-interface diagnostics remain separate and never select a checkpoint.

## Completed comparison and current status

All four v4 fits completed in 411.02 seconds, with peak process memory of
1,061.56 MiB. All 20 hidden updates were accepted in each case, so the revised
optimization did learn. Nevertheless, **v4 passed 0/4 overall forward checks**,
as did v3. No fit was retried or omitted. These are four reused paired datasets
and eight new method fits across v3 and v4, not eight independent replications.

Temperature jumps are zero at both interfaces, and normalized normal-flux
jumps are at floating-point roundoff. The continuity repair therefore works.
The remaining heat-equation errors exceed the unchanged normalized RMS limit
of 0.05 in every case:

| Case | V2 PDE RMS | V3 PDE RMS | V4 PDE RMS | V4 field RMS / maximum (K) |
| --- | ---: | ---: | ---: | ---: |
| Full contact, h = 800 | 0.047836 | 1.152243 | 0.437261 | 0.048628 / 0.095449 |
| Partial contact, h = 800 | 0.243176 | 1.216542 | 0.205231 | 0.065463 / 0.264583 |
| Full contact, h = 1400 | 0.057378 | 2.268972 | 0.309801 | 0.050459 / 0.122451 |
| Partial contact, h = 1400 | 0.361268 | 1.683544 | 0.346330 | 0.114498 / 0.381371 |

V4 improves heat-equation accuracy relative to v3 and modestly improves the
two partial-contact PDE residuals relative to v2. It performs worse than v2 on
both full-contact PDE residuals and on all four field RMS errors. Several
observation and bottom-condition checks also fail. This is not evidence of an
overall improvement over v2 or an accepted forward model.

The separately predeclared near-interface audit is substantially worse than
the primary-grid averages: v4 normalized heat-equation RMS is 402.316 and
519.470 for the two partial-contact cases, equivalent to 16.7632 and
21.6446 K/s. These points remain report only and do not replace the original
gate. They show that exact continuity has not resolved local PDE accuracy
near the contact edges.

The [complete comparison](../../reports/pinn-spatial-validation/continuity-repair-report/REPORT.md)
retains all saved gates, errors, computational costs, conditioning, and
supplemental diagnostics. The [tracked result and provenance](../../thermotwin/PINN_SPATIAL_CONTINUITY_REPAIR_RESULT.json)
preserves a compact record in Git; raw reports and checkpoints remain under
the ignored `reports/` directory. The separate
[v4 report](../../reports/pinn-spatial-validation/continuity-v4-report/REPORT.md)
includes the comparison figure.

Validation comprises 52 final focused checks and 38 preserved v2 revision
checks, all passing: 90 unique focused checks. This is not a new complete
repository test-suite result. The original numerical sources and protected
v1/v2 evidence, the v3 campaign, and the separate operating-decision Phase F
gate all passed preservation checks.

The bounded forward comparison is finished. No inverse fit, new reserved
cohort, or 3D study was opened. Further PINN work needs a new bounded
development plan addressing the remaining PDE and bottom-condition errors;
the established conventional reference remains the benchmark.
