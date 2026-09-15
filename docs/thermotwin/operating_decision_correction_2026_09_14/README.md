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
