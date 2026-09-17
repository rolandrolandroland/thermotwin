# Candidate-exclusion and projected-KKT recovery probe

`family_a_recovery_probe.py` runs the public corrected-partition pipeline in a
12-block namespace whose campaign and partition names both identify it as
disposable. It does not use any `r2_*` scientific partition or reserved
evaluation namespace.

Run it from the repository root with the project environment on `PYTHONPATH`:

```bash
PYTHONPATH=. python docs/thermotwin/operating_decision_correction_2026_09_14/family_a_recovery_probe.py \
  --workers 4 \
  --output /private/tmp/family_a_recovery_probe.json
```

The public runner generates all three truth families so its normal partition
completeness and random-stream audit execute end to end. The JSON retains only
Family A decision evidence. It reports all 48 Family A cases, decisive outcomes,
the 12 fixed-voltage outcomes, false approvals and rejections, numerical
failures, candidate bound hits, five-state interface-mass lower-bound hits, the
projected scaled-gradient norm, and bound hits that satisfy the projected KKT
convergence test. A bound fit may still be nonstationary when a free parameter
violates KKT; the projected norm distinguishes that case from convergence at an
active bound.

For every case, the script also applies the superseded strict rule to the exact
candidate fits stored with the current decision: if either fit is nonconverged
or reached a bound, the counterfactual abstains. This calculation performs zero
refits. The `additional_abstentions` field is therefore the number of current
decisive outcomes that the strict whole-case rule would discard on identical
fits and data.

No wall-clock timestamp, elapsed time, process identifier, or output path enters
the JSON. With the same source and constants, the serialized result is stable
across runs and worker counts.

`five_command_dry_run.json` records the separate 2026-09-15 chronology test.
Starting from generator-v4 commit `2a0ee76`, a throwaway clone ran
`freeze-generator`, `freeze-parent`, `rehearse-parent`, `freeze-guard`, and
`evaluate-reserved` with ten blocks per partition and 1,000 bootstrap draws.
The file retains the disposable campaign, boundary commits, linked artifact and
evidence digests, compact stream-audit counts, output hashes, and independent
post-run validation. The generated cohorts and descriptive result are
engineering evidence only.

`scientific_parent_rehearsal_report.md` records the one scientific rehearsal
opened on 2026-09-16 after the parent artifact was committed. Its compact
outcomes and external-evidence manifest are tracked at the repository root;
the manifest binds the complete diagnostics to a content-addressed immutable
release archive. The selector made 45/60 decisions without a decision error,
while simultaneous block coverage was 16/20 (80.0%), below the 90% target.
That result is preserved unchanged before guard development and calibration.

`scientific_guard_freeze_report.md` records the scientific guard development
and calibration opened once on 2026-09-16. The guard threshold is 1.178439 and
the complete artifact digest is
`d96696f265937b8d8f584aff6684c33cc1c279b9687bf864a4555e30c3ef63f2`.
On calibration, the guard reduced decisions from 56/90 to 54/90 and mean
realized terminal energy from 80.51 J to 78.80 J, while simultaneous block
coverage rose from 26/30 to 28/30. Both procedures made zero decision errors.
The compact outcomes and external-evidence manifest are tracked; the two full
diagnostic files remain in content-addressed release archives. That frozen
guard was then used unchanged for the reserved evaluation below.

`scientific_reserved_evaluation_report.md` records the final scientific
comparison completed on 2026-09-17. The balanced guarded-minus-parent loss was
+0.018487 with a 95% paired-block interval of [-0.003231, +0.045153], so the
predeclared success criterion was not met. The guard reduced mean energy from
79.54 J to 74.86 J and decisions from 98/150 to 93/150; simultaneous block
coverage rose from 38/50 to 40/50 but remained below the 90% target. All six
procedures had zero decision errors and zero numerical failures. The scalar
guard is retained as a negative result and is not promoted.
