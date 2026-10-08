# Phase E3: disposable replay and calibration generation gate

Date accepted: 2026-10-08. Status: engineering acceptance passed. The separate
generation gate must be committed before independent calibration starts.
Reserved evaluation is not authorized.

The replacement namespace `p0_disposable_phase_e_roundtrip_v2` executed from
the clean numerical source `23a076f69ad4da294234b146e330b8615bed730e` after
[full CI](https://github.com/rolandrolandroland/thermotwin/actions/runs/37716142533)
passed all 926 tests. Both saved computations validated, their scientific
digests matched, and the constructed storage and resource checks passed. An
independent reload subsequently verified both sealed blocks, their source and
protocol bindings, all 100 constructed records, and the 1500-cell constructed
calibration calculation. No new observations were generated for that audit.

The [generation gate](../../thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_E_GENERATION_GATE.json)
binds the original source, full numerical manifest, protocol, pinned runtime,
CI result, evidence checksums, and resource measurements. Its artifact digest
is `99954923fd568437bbb28fe1d05e5090bd42222d4a6fc2a7cff4ec51251dbaf6`.
The later acceptance commit records this evidence; numerical execution must
continue from the original tested clone at `23a076f`.

## Engineering checks

| Check | Result |
| --- | --- |
| Controlled prospective phase tests | 108 passed in the pinned scientific runtime. |
| Full committed-source CI | 926 passed; 788.905 seconds for the tests. |
| Complete disposable computations | Two independent computations of one three-family block; both archives validated. |
| Exact scientific replay | Same scientific block digest; numerical records matched after the declared removal of timing fields. |
| Constructed raw archive | 100 repeated blocks, 356,125,490 bytes; full load and per-record checksum comparison passed. |
| Constructed calibration calculation | 1500 procedure/family/block cells; rank 96 and JSON roundtrip recomputation passed. |
| Process-tree sampling | 4996 samples, no sampling errors; synchronous preflight succeeded. |
| Sampled peak process-tree memory | 1,865,318,400 bytes, about 1.74 GiB; below the 4 GiB limit. |
| Whole successful replay wall time | 5139.237 seconds, about 85.65 minutes. |
| Numerical CPU time per repeat | 5092.427 and 5098.703 seconds. |

The pool is configured for four workers; two compute the declared independent
repeats. Evidence for four simultaneously active numerical workers remains the
prior D6 resource record using the same sealed family worker. The constructed
probe also records coordinator peak memory of 1,873,592,320 bytes, below the
same limit. Sampled process-tree memory is a measurement, not an absolute
continuous-memory guarantee.

There is one distinct disposable block and three distinct cases. Repeated and
constructed records are engineering evidence only. They are not 100 calibration
blocks and do not support scientific performance or coverage claims.

## Source and evidence identity

The source manifest digest is
`88e3f8187ab57683ec1c36a6bd24a96364e508d50bb2ae9aa709dfe354645a20`;
the Phase E protocol digest is
`9ef8795fc2bf2b7db536a3c72fd9593e538faa54acfea1af8cf2f78d7f96cfb6`.
Runtime: CPython 3.10.12, Darwin arm64, NumPy 2.2.6, SciPy 1.15.3,
Matplotlib 3.10.8, and Torch 2.9.1. The D6 numerical source manifest remains
`79772d230eec28aaf41499f0b258083ca061b91fbec8d244515df44b4d89c7ad`.

The scientific block digest is
`fa876e43659e0f81327d3136bb80fef6afa461187913cbdaf56e728cad424f5f`.
Evidence is retained in the ignored
`.artifacts/operating-decision/prospective-phase-e-rehearsal-23a076f-v2`
directory. The gate records every file's size and checksum. The original and
replay archive SHA-256 values are:

```text
original: 7286aef40900f4bb801a98c31cb4e7ae9d9f3c3d22af6a79d5a4d4788beed3f8
replay:   74b488e91607c5bdeb7702c5b80088cf840933d2175ab31ea150200a3a9b1b7a
```

Their full byte hashes differ because timing fields differ. The scientific
digest is identical. The failed v1 namespace remains closed as documented in
the [incident record](OPERATING_DECISION_PHASE_E_V1_INCIDENT.md); its missing
worker outputs and final resource record are not reconstructed or replaced.

## Authorized next execution

Once this acceptance and its gate are committed, E4 may open
`p1_independent_calibration` from the original clean `23a076f` source clone:
100 paired blocks, 300 cases, five procedures, four workers, rank 96. The gate
loader requires the committed gate bytes and an acceptance commit descending
from that tested source. Neither selector, development offset, stopping rule,
action cost, sample size, nor endpoint changes.

The frozen planning allowance remains 43–60 active wall hours for calibration.
Save complete blocks atomically; resume only authenticated checkpoints under
the same source, runtime, namespace, and protocol. Retain ordinary failures in
their denominators. Preserve material incidents and stop interpretation.
If any required correction is infinite, close calibration as infeasible and
keep reserved evaluation unopened.

At acceptance, both scientific partitions remain unopened. Finite calibration
still requires a separate Phase F freeze and end-to-end replay before reserved
evaluation. The adopted simultaneous risk comparison remains explicitly
underpowered; engineering acceptance does not change that limitation.
