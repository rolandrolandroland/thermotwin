# Paired 2D continuity repair: development v3

This additive development study follows the failed v2 forward gate. It tests a
shared interface representation and a better conditioned heat-equation solve.
The original v1 and v2 numerical files and their sealed evidence remain fixed.
No inverse fit or reserved evaluation cohort is opened by this runner.

The neural model lives in `thermotwin/pinn/planar_contact_continuity.py`; the
bounded runner is `thermotwin/studies/pinn_spatial_continuity.py`. The model
shares interface temperature and normal derivative through a Hermite
representation. Its neural features remain trainable, while linear output
coefficients are periodically fitted to the same known physics and sparse
observations. The bottom Robin and insulated conditions remain explicit
physics equations; they must pass the same independent checks as before.

## Controlled comparison

One complete configuration is declared before fitting and used for all four
arms: full contact at h=800, partial contact at h=800, full contact at h=1400,
and partial contact at h=1400 W/(m² K). The runner reuses the byte-identical
saved v2 observation files for each arm. It draws no new observation noise
and reruns no reference solver. The fit receives only the observed
coordinates and temperatures, geometry, known conductance, physical
configuration, and declared neural/collocation seeds.

The original seeds remain neural 2026102202, sampling 2026102203, independent
diagnostic 2026102204, and the reused observations were generated with noise
seed 2026102201. The reference trace remains a 10 s ramp from a 300 K bath
toward 295 K, and a 500 W/m² top-flux ramp. The 36 observations are the three
sensors at x=4,20,36 mm and z=3 mm, sampled every 10 s through 120 s with
the center sensor omitted at 40,50,60 s.

The hidden reference coordinates and temperatures are loaded only after the
optimizer-returned final checkpoint and raw training report are atomically
retained. They cannot select epochs, feature weights, projection steps,
or checkpoints. Independent diagnostics also follow checkpoint retention.

## Acceptance

The independent coordinates are byte-identical to the v2 shifted spatial
grids and explicit times 0.25,0.75,1.5,3,6,12,22,38,60,88,115 s. Every
applicable family must be present: interior heat equation, exterior sides,
top flux, bottom contact, bottom insulation, and both temperature and
Fourier-flux conditions at each internal interface.

| Quantity | Gate |
| --- | --- |
| Normalized observation mean squared error | ≤2 |
| Each independent normalized physics RMS | ≤0.05 |
| Hidden field RMS error | ≤0.05 K |
| Hidden field maximum error | ≤0.15 K |

Normalization is unchanged: 5/120 K/s for the interior equation, 500 W/m²
for flux, and 0.1 K for interface temperature. Consequently an interface
temperature RMS of 0.005 K is the limit. Continuity by construction does
not replace the explicit independent interface checks or the interior gate.
Any supplemental near-interface audit is reported separately and does not
alter these acceptance coordinates or select the checkpoint.

The fixed supplemental audit samples normalized x offsets
−0.005,−0.001,−0.0002,+0.0002,+0.001,+0.005 around each contact edge,
z/H=0.002,0.01,0.05,0.2,0.5, and times 0.4,1.1,4.5,14,45,105 s.
It also checks both interface traces at the same noncorner heights and
times. Full-contact arms have no internal interface and are marked not
applicable. The plan stores the point digests; separate point and audit
files retain the results. A supplemental diagnostic failure is reported
without changing the primary acceptance decision.

All four arms must pass using the same declared configuration. A failed
arm remains a failed arm; field accuracy alone cannot pass the study.

## Execution and preservation

`prepare_development` writes the plan before fitting. The plan binds the
model configuration, four cases, random streams, acceptance gates, runtime,
source hashes, paired artifact hashes, and the sealed v2 evidence index.
The tracked v2 development record and its numerical source/test hashes,
v1 evidence, and Phase F source gate can be bound through
`v2_development_record` and `preserved_campaign`; the execution launcher
must supply both for the scientific run.

The campaign permits four fits, no automatic retry, one CPU thread,
3,600 seconds wall time, and a 2 GiB process RSS cap. The worker checks its
own elapsed time and peak RSS every 0.5 s. Its launcher sets BLAS/OpenMP
thread limits before starting Python; the runner also limits PyTorch to
one thread. A consumed plan cannot be resumed or rerun. Resource or
preservation failures retain the current failure and the remaining
predeclared arms as not run.

The runner saves per-arm observations, final checkpoint, raw training
report, independent diagnostics, hidden field scores, progress, and any
failure. A final artifact index covers its output. Old v2 evidence is
validated against its sealed file records and index hash; incidental
unindexed caches do not become part of that seal.

Runner tests use constructed temporary arrays and mocked fitting only.
They verify the truth boundary, raw-before-score ordering, unchanged
diagnostic coordinates and gates, source/evidence protection, failure
retention, resource stopping, and single-use execution. They establish
execution behavior, not scientific success of the neural repair.
