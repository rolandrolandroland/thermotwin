# Phase E disposable v1: stream-validator incident

Date: 2026-10-07. Status: failed engineering attempt, closed against reuse.

The disposable namespace `p0_disposable_phase_e_roundtrip_v1` opened at clean
source `fde947df1b8f3ab0ecfbf2cb2fc44b174b5bde27`, after its
[full CI run](https://github.com/rolandrolandroland/thermotwin/actions/runs/37637288512)
passed all 920 tests. The scientific runtime matched the pinned CPython 3.10.12,
NumPy 2.2.6, SciPy 1.15.3, Matplotlib 3.10.8, and Torch 2.9.1 environment.

The coordinator started at 16:04:57 UTC. At 17:32 UTC both numerical workers
were still active, each with approximately 85 minutes of accumulated CPU
time. The stopped coordinator was inspected at 19:45 UTC. Its exact failure
time and final resource totals were not saved, so those observations are not
a measured total runtime or peak-memory result.

## Failure and evidence boundary

The first returned block reached `_save_block` but failed before any block
archive was written. The exception was:

```text
ValueError: corrected random-stream inventory label is unknown
```

The call chain was `seal_phase_e_block` → `validate_phase_e_block` →
`_validate_corrected_stream_records` → `_expected_corrected_stream_manifest`.
Phase E passed `label="Phase E"` to a sealed validator whose labels select an
inventory mode. It accepts `parent pilot` for all four fixed policies and
`N=32` for the stop-only continuation. Campaign and partition identities are
separate explicit arguments. The diagnostic label was mistakenly treated as
free-form text rather than an inventory selector.

No block, replay summary, constructed storage probe, or final resource record
was saved. The v1 coordinator did not preserve a rejected in-memory result or
write resource measurements on exceptions. Its worker outputs cannot now be
recovered. A retrospective incident record is retained in the ignored v1
output directory; it is diagnostic evidence, not a successful checkpoint.

Neither independent calibration nor reserved evaluation opened. No calibration
correction was estimated, no final comparison was inspected, and this incident
does not support a scientific performance or resource-acceptance claim. The
frozen D1–D6 source and artifacts remain unchanged.

## Scoped repair and replacement

Phase E protocol v2 maps its corrected stream check to the existing all-four-
policy inventory mode while preserving the actual Phase E campaign and
partition arguments. A synchronous metadata-only inventory check now runs in
preflight before numerical workers start. It generates no observations.

Regression tests exercise the real corrected-stream validator through block
save/load in both permitted namespaces, reject a clean but incomplete stop-
only inventory, and reject cross-partition stream identities. Other tests
check early preflight, retention of a rejected result and incident, and resource
recording on an execution exception. No selector, fit, action cost, development
offset, stopping threshold, calibration rule, endpoint, or sample size changes.

The new disposable namespace is `p0_disposable_phase_e_roundtrip_v2`. The v1
namespace is closed and the repaired entry point refuses it. Require a new
committed-source CI pass, both full independent computations, exact scientific
replay, constructed 100-block storage acceptance, and the full resource check.
Only then may a separate committed generation gate authorize the still-unopened
100-block calibration partition. The reserved partition remains closed.

Controlled validation passed all 108 prospective phase tests in the pinned
runtime. The D6 numerical source manifest still has digest
`79772d230eec28aaf41499f0b258083ca061b91fbec8d244515df44b4d89c7ad`.
The replacement's full committed-source CI and numerical replay remain
separate acceptance requirements.
