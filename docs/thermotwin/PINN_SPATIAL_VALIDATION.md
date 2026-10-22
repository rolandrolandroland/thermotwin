# Spatial PINN validation protocol

This study proceeds through fresh 1D validation, a verified 2D reference,
a bounded 2D inverse PINN, and then an assessment of whether a device-specific
3D study is justified. Each stage has a distinct claim. A failed 1D property
shape diagnostic does not decide whether a separate 2D scalar-contact problem
is identifiable. A failed numerical reference stops the affected inverse stage.

The additive implementations and focused checks are ready. This document
defines the design; the generated `manifest.json` and `state.json` record
whether the frozen named evaluation has started or finished.
Disposable smoke, timing, and solver-verification computations are development
checks, not named evaluation results. No PINN advantage has been established.

The v1 campaign completed on October 19, 2026. Its post hoc
[failure diagnosis](PINN_SPATIAL_FAILURE_DIAGNOSIS.md) separates weak 1D shape
information from the 2D field-training failure and defines the recommended
development sequence. The completed campaign and its scores are preserved.
The subsequent [v2 development record](PINN_SPATIAL_REVISION_V2.md) tracks
execution of that sequence. V1's completed evaluation and v2's development
checks have separate claims and evidence.

## Evidence and execution rules

Freeze the source, seed namespaces, optimizer caps, candidate schedules,
selection rule, gates, geometry, observations, and comparison methods before
generating named synthetic data. Record the source commit, source hashes,
CI evidence, runtime, and complete configuration in manifests.

Every planned evaluation remains in its declared denominator. Keep failures,
optimizer histories, final checkpoints, independent residuals, and observations.
Do not retry seeds, choose a lucky checkpoint, change truth after a failure,
or tune against evaluation property or field errors. An optimizer's declared
stopping rule may finish before its cap; its returned final state is the result.

Run with the existing CPython 3.10 environment, one Torch thread and one
BLAS/OpenMP thread, without multiprocessing or package installation. The
operating-decision Phase F run remains separate; its frozen numerical source
and existing artifacts must remain unchanged.

The entry point is:

```sh
python -m thermotwin.reports.pinn_spatial_validation run \
  --output-directory <newdir> \
  --source-commit <sha> \
  --ci-evidence <json>
```

Use a new directory beneath `reports/pinn-spatial-validation/`, which is
ignored by Git. Full machine-readable results and manifests belong there.
The checked-in [protocol JSON](../../thermotwin/PINN_SPATIAL_VALIDATION_PROTOCOL.json)
is checked against the implementation before execution and is authoritative
for numeric 2D gates. Generated evidence records what actually happened.

The one-shot supervisor entry point is
`python -m thermotwin.reports.pinn_spatial_supervisor`, with
`--source-commit`, `--output-directory`, and `--monitor-directory`.
It waits for the exact commit's tests workflow, enforces the frozen runtime,
limits computation to one thread, samples its own job's memory, and stops at
the declared three-hour compute or 2 GiB sampled-RSS limit. It never restarts.
Monitor `state.json` and `campaign.log`; a waiting CI state means no named
observations have been generated. On termination, retain partial evidence.
Validate final artifact integrity with
`python -m thermotwin.reports.pinn_spatial_validation validate --output-directory <newdir>`.
Hash validation does not independently recompute the scientific conclusions.

## Fresh 1D validation

Use three predetermined trials with SHA-256-derived neural and observation
seeds under `pinn_spatial_v1_2026_10_fresh_1d`. Each trial uses independent
noise on the existing three training regimes: zero-current relaxation and
positive/negative 0.8 A experiments. This is fresh observation noise and
neural initialization for a fixed material truth and fixed regimes. It is
not a fresh distribution of unknown material curves.

Synthetic truth uses the independent nodal solver, transition-split SSPRK3,
25 spatial nodes, a 0.00025 s time step, and the existing smooth cubic
resistivity. This differs from the original training-budget audit's
cell-centered FV/RK4 and three-knot piecewise-linear truth. Keep that change
explicit when comparing studies. Observe both face temperatures and voltage
every 0.08 s, with Gaussian standard deviations 0.01 K and 0.00001 V.

Train the existing shared-property PINN for 2400 uninterrupted Adam epochs:
three 20-unit hidden layers; seven interior spatial points; 18 time points;
16 voltage quadrature points; network and property learning rates 0.002;
initial resistivity multipliers 0.9; physics weight 10; smoothness weight
0.0001; and no shrinkage. The epoch count fixes the checkpoint.

Before any inverse training, compare all three training truths and all three
held-out truths against 49 nodes and a 0.0000625 s time step. Across 21
times and nine positions, require maximum temperature differences no larger
than 0.001 K and maximum voltage differences no larger than 0.000001 V.
These are one tenth of the observation noise scales. A failure records
`numerical_reference_passed=false` and
`blocked_reason=independent_truth_refinement_failed`, preserves the three
planned trials, and runs no inverse training. This combined refinement check
does not establish hardware fidelity.

Recompute losses on the frozen model after its final optimizer update.
The existing trainer logs losses before an update and properties after it;
those historical values must not be mixed to accept a checkpoint. The primary
truth-blind operational gate requires normalized observation MSE at most 2
and physics-rate RMS divided by nominal-model temperature-rate RMS at most
0.25. Compute the nominal normalization over the three training baselines.
Also report a separate gate on a denser 13-by-35 residual grid, including new
spatial points and time midpoints. Evaluate PDE and both face balances.

Report truth-known property diagnostics separately: coefficient error,
amplitude ratio, signed center-contrast ratio, and improvement over the best
constant knot multiplier. Shape recovery requires both ratios in
[0.75, 1.50] and improvement over that constant baseline. Report dense cubic
curve error and the best representable three-knot function-space projection
separately. The projection is a representation-error floor, not an identified
data-effective material truth. Record the actual training temperature range
and error within that range; the 285–315 K knot domain is broader.

Transfer the frozen inferred property through the conventional forward solver
to a positive 0.4 A experiment and a negative 0.8 A pulse. Both are within
the knot domain but extend beyond the training temperature range. Require
each face and interior temperature RMSE at most 0.03 K, voltage RMSE at most
0.00005 V, maximum temperature error at most 0.08 K, and energy-balance
residual at most 0.0000000001 W. A 280–320 K case is diagnostic only.
The combined synthetic gate also requires numerical refinement, both
operational gates, knot shape recovery, and the two transfer gates.

## Verified 2D reference

The benchmark is a synthetic thermal plate, 40 mm wide, 10 mm deep, and
4 mm thick. Known properties are thermal conductivity 15 W/(m K), density
7800 kg/m³, and specific heat 500 J/(kg K). The contact lies off center,
from x=8 to 24 mm, and spans the full 10 mm depth. Its only unknown is a
uniform contact conductance h, bounded between 400 and 2000 W/(m² K).
This is a thermal-contact benchmark, not a full thermoelectric module.

The conservative finite-volume reference includes partial-cell Robin contact,
insulated remaining boundaries, prescribed top heat flux and bath schedules,
and implicit time integration. Verify analytic limits, energy accounting,
spatial refinement, and time-step refinement before generating inverse data.
The declared verification compares 80×20, 160×40, and 320×80 meshes, with
separate fixed-mesh time refinement. Its sensor discrepancy limit is 0.002 K;
common-grid field RMS and maximum discrepancy limits are 0.005 K and 0.01 K.
The canonical truth uses the middle mesh; the scalar inverse baseline uses
the coarse mesh. Preserve physical and scaled energy residuals in evidence.

## Bounded 2D inverse and reconstruction

Use three embedded sensors at x/width=0.1, 0.5, and 0.9 and
z/thickness=0.75. The declared 36 temperature samples have independent
Gaussian noise with standard deviation 0.02 K. Give every estimator the
same observed values, known geometry, material constants, boundary schedules,
and parameter bounds. The neural estimator receives no reference field.

Compare bounded scalar fitting with the verified PDE solver, the inverse
PINN, a matched data-only neural reconstruction, and a known-h neural oracle
diagnostic. Include the conventional forward prediction at the frozen initial
h without fitting. The initial value is already only 10.56% from evaluation
truth: the broad 10% engineering tolerance alone is weak evidence of learning.
Meaningful recovery additionally requires reducing the initial absolute
parameter error by at least 50%. Match the neural architecture, initial field, observations, and
declared optimizer budgets. The data-only model reports no h estimate because
its objective has no observation-to-contact relation. The oracle receives
true h only in its explicitly labeled diagnostic mode; it cannot select the
inverse method or determine its acceptance.

Development uses h=800 W/(m² K) and the fixed candidate set: physics weights
10 and 100 crossed with 1800 and 5400 Adam epochs, followed by at most
200 L-BFGS iterations. Select using the declared truth-blind observation and
independent PDE/boundary-residual criteria; exclude parameter and reference
field errors from selection. Freeze the choice before three evaluation
trials at h=1000 W/(m² K). Keep every result even when no candidate passes.

Report final observation fit, h error, full-field reconstruction error,
independent PDE and boundary residuals, energy diagnostics, failures, and
runtime. Separate residuals on the contact and insulated bottom regions so
a pooled boundary average cannot hide a failure. Numerical reference gates,
truth-blind convergence gates, and truth-known recovery gates answer
different questions; all numeric thresholds are frozen in the protocol JSON.

For a new boundary schedule, transfer the frozen inferred scalar h through
the verified 2D forward solver. A network trained for one control schedule
does not thereby become a PINN predictor for arbitrary controls. Label neural
reconstruction and scalar-parameter forward transfer separately. Report the
baselines without promising PINN superiority in accuracy, robustness, or cost.

## When a 3D study would be warranted

The user selected the benchmark plate for this assessment. For this synthetic
geometry, exact extrusion and uniform boundary conditions
in the depth direction justify the 2D reduction. A 3D mesh would not add
independent physics under those assumptions.

Finite-depth contacts, localized heating, edge losses, and a module's separate
legs or interconnects can require 3D. Such a study needs actual dimensions,
contact topology, material definitions, out-of-plane boundary conditions, and
sensor locations, followed by a new numerical verification and frozen inverse
protocol. Do not invent a hardware geometry or interpret the synthetic plate
as a validated device. Device-specific 3D remains a future decision supported
by the 1D and 2D evidence and the geometry that is actually supplied.
