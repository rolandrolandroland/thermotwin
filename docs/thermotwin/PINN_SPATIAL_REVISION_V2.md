# Spatial validation revision v2

This revision responds to the preserved v1 failures with two separate tasks:
improve the 2D neural representation and its verification, and design 1D
experiments that can resolve resistivity shape before fitting that shape.
The v1 numerical files, protocol, checkpoints, and campaign artifacts remain
unchanged. V2 develops a separate method and comparison.

Current state: the 1D `design-v2` information sweep is complete and selected;
its classical recovery benchmarks have not run. All four final declared 2D
forward cases completed; only one passes every gate. All pass field-value
checks, but three fail independent PDE or interface checks. There is no accepted
final 2D selection or frozen complete v2 protocol. The requested sequence
therefore remains incomplete. Named evaluation requires accepted development,
source freeze, and exact-source CI first.

| Requested step | Current state |
| --- | --- |
| 1. Repair and verify the known-contact 2D forward field | Implemented and tested; forward acceptance remains failed, 1/4 final arms passing. |
| 2. Delayed or alternating contact estimation | Implemented with stage checkpoints; no inverse development or evaluation fits run because Step 1 failed. |
| 3. Retain the scalar baseline and broaden 1D excitation | Scalar baseline retained and the 16-candidate design completed; fresh recovery comparison remains pending. |
| 4. Freeze the complete rule, then evaluate fresh data | Execution guards and reporting implemented; no complete protocol frozen or named v2 observations generated. |

The checked-in [development summary](../../thermotwin/PINN_SPATIAL_REVISION_V2_DEVELOPMENT.json)
records selections, source hashes, original-evidence checks, and the raw-artifact
inventory digest. It is a development checkpoint, not a frozen evaluation rule.

## Claims and method comparisons

The 2D problem remains a synthetic contacted thermal plate with known
geometry, material properties, and controls. Its scalar unknown is contact
conductance. Compare the selected revised neural inverse with a bounded
conventional forward-model fit and a nominal-parameter forward prediction.
Retain physical parameter, observation, full-field, independent residual,
held-out parameter-transfer, and computational-cost results.

If selected, the domain-decomposed MLP with boundary lifting changes the
architecture. It introduces interfaces whose temperature and flux conditions
must be independently checked. It must be described as a revised method;
successful results would not establish that v1 needed only more epochs.
Development may compare declared soft and decomposed alternatives, but cannot
select against fresh evaluation contact or hidden field errors.

The 1D task is information design followed by **classical scalar and
three-coefficient inverse fits**. It does not retrain a 1D PINN or establish
neural curve recovery. Matched constitutive truth and cubic mismatch are
separate benchmarks with different interpretations.

## Freeze and execution chronology

1. Record every development candidate, noise assumption, gate, optimizer cap,
   and selection rule before its development computation.
2. Complete development using only the declared development inputs. Retain
   failures and unsuccessful candidates. Choose by the predeclared rule;
   never choose a lucky seed or checkpoint.
3. Record the chosen configuration and numerical dependency hashes. Freeze
   source and machine-readable protocol, obtain CI evidence for that source,
   and verify the preserved v1 campaign.
4. Generate fresh named data only through the gated coordinator, using new
   declared namespaces. Execute all planned rows without retrying failed
   fits. A numerical-reference failure blocks the affected inverse stage.
5. Render scores and close the artifact index. Validation reads the saved
   evidence and its hashes; it does not regenerate figures, data, or fits.

Run with the existing pinned CPython 3.10 environment and one numerical CPU
thread. No installation or modification of a frozen dependency is needed.
The operating-decision evidence remains separate.

```sh
python -m thermotwin.reports.pinn_spatial_revision evaluate \
  --output-directory <new-output-directory> \
  --source-commit <frozen-commit> \
  --ci-evidence <ci-evidence.json> \
  --preserved-campaign <v1-campaign-directory>
```

Validate completed evidence without rerunning it:

```sh
python -m thermotwin.reports.pinn_spatial_revision validate \
  --output-directory <saved-output-directory>
```

The coordinator calls `render_revision_report` before sealing its artifact
index. The CLI does not render again afterward. Generated `REPORT.md` and
metric figures display supplied outcomes; they do not apply new acceptance
rules. Complete scores and histories stay in the machine-readable artifacts.

## 1D information design

The development grid contains 16 candidates: four initial/reservoir
temperature-window groups, current magnitudes 0.4 or 0.8 A, and observation
intervals 0.08 or 0.04 s. Each window is run at both current signs for 0.4 s.
The groups are a central window, a wide window, three overlapping windows,
and three separated bands. The declared physical domain is 285–315 K.

Use known nominal Seebeck coefficient, thermal conductivity, geometry, face
properties, initial states, and controls. Development evaluates only nominal
resistivity multipliers and the fixed generic curve `(0.97, 1.02, 1.04)`.
Neither the old cubic nor the future evaluation truths select the design.

Whiten the finite-difference observation Jacobian by standard deviations
0.01 K for face temperatures and 0.00001 V for voltage. Its parameters are
the three log multipliers of nominal resistivity knots. At both development
curves require:

- Minimum singular value at least 100.
- Signed center-contrast standard error at most 0.01.
- Every marginal log-parameter standard error at most 0.02.
- Temperatures within 285–315 K, with coverage below 288 K and above 312 K.
- Nonlinear forward displacement of at least three noise units for each
  declared ±0.035 contrast perturbation along the locally least-informative
  contrast direction, staying inside parameter bounds.

Among passing candidates choose by experiment count, observation count,
current magnitude, worst contrast standard error, minimum singular value,
and candidate identifier. Recheck the winner using a smaller finite-difference
step and a refined conventional spatial/time grid. Failure is unresolved;
it does not authorize selecting a different candidate retrospectively.

The source-consistent [selection record](../../reports/pinn-spatial-validation/revision-v2-development/one-d/design-v2/selection.json)
selected bands **286–290, 298–302, and 310–314 K**, each at **±0.4 A**,
sampled every **0.08 s**: six experiments and 108 observations. Eight of
16 candidates passed; no evaluation truth was used in selection.

| Completed development quantity | Recorded value |
|---|---:|
| Worst contrast standard error | 0.00179854 |
| Worst minimum singular value | 641.36465 |
| Worst marginal log standard error | 0.00145115 |
| Nonlinear ±0.035 contrast displacement | 19.4257–19.5003 noise units |
| Maximum Jacobian step-refinement discrepancy | 1.515e-7 relative |
| Maximum conventional-grid discrepancy | 0.008214 noise standard deviations |
| Total design-sweep wall time | 112.171 s |

CPU time and peak RSS were not recorded for this sweep. The development
manifest records unchanged sources; these results are design evidence, not
named recovery, 1D neural training, or a committed complete-protocol freeze.
The [numerical module](../../thermotwin/studies/pinn_spatial_revision_1d.py) and
[scientific tests](../../tests/test_pinn_spatial_revision_1d.py) implement the rule.
SHA-256 identities recorded for this selection are:

- Selection JSON: `aab1b595e07b2176bb18023f1896cb165062378a919660f40c15ef974fe08008`.
- Numerical module: `5387af1e6e1891e3b26b9e70b9921315ac3b4ef4d167a1b7970fbb08fb0dcaa0`.
- Specification: `512a7825febbab8c6cf22b306120890b618697c73bef21ebf30468629042f11b`.

## 1D fresh classical benchmarks

Use three paired independent noise blocks for each of two truth families:
six datasets and 12 method fits, not six independent noise replications.
Both methods see identical saved observations for a dataset. The scalar
method fits one common log multiplier; the three-knot method fits three
bounded log multipliers. Both use the same conventional forward model and
log bounds `[-0.3, 0.3]`, with fixed search budgets and no regularization.

The matched positive control uses knot multipliers `(1, 1.035, 1)` with
independent nodal numerical truth. The new cubic matches those knot values
but curves between them. Its normalized-temperature coefficients are
`(1.035e-5, -0.065e-5, -0.030e-5, 0.020e-5)` at reference 300 K and scale
15 K. It is distinct from the v1 cubic.

Before fitting either family, verify all regimes using 25 versus 49 truth
nodes and time steps 0.00025 versus 0.0000625 s. Maximum differences must
be at most 0.001 K and 0.000001 V, and both grids must remain in the physical
temperature domain. Numerical exceptions or failures retain all six planned
rows and stop fitting.

Matched three-knot recovery requires observation MSE at most 2, maximum
absolute multiplier error at most 0.02, and signed contrast error at most
0.01. Scalar fits report fit and curve diagnostics without a shape-recovery
claim. Cubic mismatch reports actual-curve errors, representation floors,
and coefficient diagnostics separately. Sampled cubic knot values and
function-space projections are not unique data-effective PWL truths.

Report local covariance with its parameterization: one common log parameter
for the scalar fit and three log coefficients for the full fit. Local Gaussian
precision does not prove global identifiability or account for model mismatch.

## 2D gates and evaluation

The final development uses three 48-unit, three-hidden-layer MLP subdomains
for partial contact, hard top/bottom and exterior boundary conditions, a ramp
temperature envelope, and a logarithmic time feature. It uses 6000 Adam epochs
and at most 4000 LBFGS iterations with refreshed training points. Independent
shifted grids include early times and every applicable PDE, boundary, and
interface family. Gates were retained throughout development.

| Geometry / known h (W/m²/K) | Field RMSE / maximum (K) | Normalized PDE RMS | Largest interface temperature RMS (K) | Overall |
| --- | --- | ---: | ---: | --- |
| Full / 800 | 0.00755 / 0.02311 | 0.04784 | N/A | Pass |
| Partial / 800 | 0.00769 / 0.02907 | 0.24318 | 0.01415 | Fail |
| Full / 1400 | 0.00680 / 0.01973 | 0.05738 | N/A | Fail |
| Partial / 1400 | 0.01003 / 0.04420 | 0.36127 | 0.01727 | Fail |

Limits are field RMSE 0.05 K, maximum field error 0.15 K, and each normalized
physics RMS 0.05. Interface temperature uses a 0.1 K normalization, so its
physical RMS limit is 0.005 K. All four observation checks pass, and all hard
exterior boundaries are satisfied to roundoff. Accurate values alone do not
verify the field's derivatives.

The [authoritative selection](../../reports/pinn-spatial-validation/revision-v2-development/two-d/resource-continuation/final-forward-selection.json)
declares `blocked_forward_gate` and no selected configuration. The complete
[forward development report](../../reports/pinn-spatial-validation/revision-v2-development/two-d/final-development-report.json)
and [four-case figure](../../reports/pinn-spatial-validation/revision-v2-development/phase-b-four-case-development.png)
preserve all outcomes.

An original 900 s operational limit interrupted the full-contact h=1400 arm.
A documented resource-only continuation retained both completed h=800 arms,
replayed the interrupted arm with identical observations, source, seeds,
points, and optimizer caps, and completed the previously unrun h=1400 partial
arm. Its first 25 logged events match the interrupted attempt bit for bit.
This is four statistical arms, with one resource-interrupted attempt. No
intermediate checkpoint was selected and no additional architecture was fitted.

A single common selected configuration must pass all four cases before inverse
development proceeds. `run_inverse_development(output_directory, forward_selection)`
validates that proof; a bare configuration does not establish the gate.
Independent PDE, boundary, and interface families remain separate.

Next development should share interface temperature and normal-flux traces
to enforce continuity away from corner endpoints, and separately improve
interior derivative accuracy. A post hoc finite-difference comparison on
saved fields finds curvature differences in the bulk away from contact edges;
it is exploratory and does not replace the independent AD gate. Full-contact
h=1400 also fails, so a corner-only remedy is insufficient. Any subsequent
architecture or training revision needs a new bounded development declaration.

The benchmark plate remains uniform along its 10 mm depth. Three-dimensional
validation remains deferred until geometry or boundary conditions vary in
that direction; no 3D model was added in this revision.

The planned denominator is six 2D evaluation trials. Keep failed and not-run
rows, and report the coordinator's parameter, field, residual, observation,
and transfer gates separately. A small observation loss alone is insufficient.
No source or gate may be changed after the named data are generated.

For a new prescribed schedule, transfer the frozen inferred conductance
through the verified conventional forward solver. A neural field trained for
one schedule is not an arbitrary-control predictor. Synthetic success does
not validate a hardware device or promise a PINN advantage over the
conventional estimator in accuracy, robustness, or cost.
