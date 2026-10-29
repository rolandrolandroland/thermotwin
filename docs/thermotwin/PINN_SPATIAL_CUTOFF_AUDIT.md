# Training-only readout cutoff diagnostic

The [completed step-controller comparison](PINN_SPATIAL_STEP_CONTROL.md)
passed two of four forward gates. Both partial-contact fits stopped close to
the hard singular-value cutoff. This diagnostic asks whether small changes
in the hidden network trigger discontinuities in the linear readout. It does
not train a replacement model or establish independent physics accuracy.

The completed optimizer experiments and their reporting repair were committed
on `dev` at `a9f0047` before preparing this diagnostic.

## Completed outcome

All fourteen declared solves completed on October 29, 2026, in 76.48 seconds,
with zero neural updates and a peak process memory of 1,036,173,312 bytes.
All original checkpoints, numerical sources and protected evidence remained
unchanged. The additive diagnostic and recorder passed 22 constructed tests.

Both cases showed the same pattern at all three radii: the negative
perturbation retained the baseline rank, while the positive perturbation
dropped one direction. At the smallest displacement, 0.000001:

| Case | Baseline rank | Negative / positive rank | Baseline training loss | Positive raw TSVD loss | Positive returned-policy loss |
| --- | ---: | --- | ---: | ---: | ---: |
| Partial contact, h=800 | 510/550 | 510 / 509 | 2.794009652 | 2.806849339 | 2.806849339 |
| Partial contact, h=1400 | 514/550 | 514 / 513 | 4.867755096 | 4.887137003 | 4.871766276 |

The h=1400 policy rejected the new head at that smallest positive displacement
and returned the original head at the perturbed hidden parameters. Its loss
still exceeded the original terminal loss. At other sampled points the new
head was accepted relative to the original head at the same perturbed point.
That head-level acceptance does not establish an acceptable optimizer step
relative to the unperturbed terminal state.

The algebraic and directly evaluated thermal objectives agreed to within
1.1e-10 at all sampled solved heads. Every central finite-difference pair
crossed a sampled rank boundary, so all six derivative comparisons remain
withheld. This audit does not validate an exact profiled gradient.

The [compact prospective declaration](../../thermotwin/PINN_SPATIAL_CUTOFF_AUDIT_PLAN.json)
and [completed result](../../thermotwin/PINN_SPATIAL_CUTOFF_AUDIT_RESULT.json)
bind the complete raw plan, source snapshots, runtime build, input identities,
preservation manifests and all fourteen opportunities. The authoritative
plan hash is `e0f97735386c7b7f5053fd4e49830e80e76dd2460a9d9d8151f0be3d395928c0`.

## Selected next correction

Address the numerical readout projection before revising the corner field.
The observed rank changes and associated loss increases support that order;
they do not show that a projection repair will remove every independent
heat-equation error.

The next separately declared method should replace the abrupt singular-value
discard with a continuous, regularized readout and use a hidden-parameter
derivative consistent with its declared solved objective. Its regularization,
scaling and acceptance rule must be fixed explicitly. Verify the solver and
directional derivatives on constructed problems and training-only checkpoint
copies before any new neural training. Simply lowering the hard cutoff or
extending the existing training budget would not address the identified
mechanism.

If the outer training objective includes the regularization penalty, a
fixed-head derivative requires a stationary solved head and all explicit
parameter dependencies in that penalty. If the outer objective retains only
the unregularized observation and physics loss, differentiate through the
regularized head solve. These are different methods and must not be combined
implicitly.

Only after those checks should a separately frozen four-case comparison start
from the original matched initialization with unchanged forward acceptance
thresholds. No replacement method or new neural campaign was launched by this
diagnostic. The completed forward result remains two of four passing, so
inverse fitting and 3D remain blocked.

## Fixed protocol

- Use copies of the two terminal partial-contact checkpoints, h=800 and
  h=1400, from the completed step-controller comparison.
- Retain their training observations, point coordinates, row weights,
  architecture, readout solver and `rank_rcond=1e-7` cutoff.
- At each original terminal state, compute the negative fixed-readout
  training-objective gradient with respect to the hidden parameters.
  Normalize that direction by its largest absolute component. Compute the
  direction before any baseline readout solve.
- Perform one baseline solve and six independent perturbation solves per
  case: positive and negative displacements of 0.0001, 0.00001 and 0.000001
  along that fixed direction. Restore both hidden parameters and readout to
  the original terminal state before each displacement.
- Count every attempted readout solve, including failed attempts: at most
  fourteen across both cases. No new neural training, alternative direction,
  radius search, retry or resource extension is allowed.
- Use one thread, a 900-second total wall limit and a 2 GiB process memory
  limit. Retain all declared opportunities and any failures.

For every point, retain the raw truncated-SVD objective and the objective
returned by the existing monotone readout policy. That policy compares the
new head against the original head at the same perturbed hidden parameters;
it may reject the newly solved head. Record the retained rank, the neighboring
singular values' distances from the cutoff and the fixed-head directional
derivative.

Central finite differences may be compared with that derivative only when
the baseline and both sampled perturbations have the same retained rank.
Even then, matching sampled ranks does not certify a smooth interval, and
truncation does not guarantee that the fixed-head derivative equals the
derivative through the solved readout. Report any discrepancy as evidence
to interpret, rather than automatically calling either derivative correct.

## Evidence boundary and decision

Freeze source, input and runtime identities before the first solve. Read only
the saved noisy training observations and training points. Do not use clean
observations, reference fields, independent residual audits, forward gates
or a reserved cohort to select the direction or perturbations. Preserve the
completed experiments unchanged and verify their identities afterward.

If the sampled evidence supports a cutoff discontinuity, the next proposed
method should address the numerical projection. If it does not, the audit
may be inconclusive; it cannot by itself establish that a corner-model
revision is required. Any replacement numerical method needs its own
declaration and validation before another four-case comparison. The existing
acceptance thresholds remain unchanged. Inverse fitting and 3D remain blocked.

Raw diagnostic evidence belongs under the ignored
`reports/pinn-spatial-validation/` directory. The compact frozen plan and
completed result are recorded under `thermotwin/`. Re-execution requires the
sealed local evidence bundle and frozen runtime; a fresh Git clone alone does
not contain the ignored checkpoints or raw evidence.
