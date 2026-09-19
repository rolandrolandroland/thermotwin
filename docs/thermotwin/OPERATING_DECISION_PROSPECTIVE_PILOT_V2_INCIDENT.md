# Prospective disposable pilot v2 archive incident

Date run: 2026-09-18. Scientific use: invalid incident evidence only.
Status: execution completed, archive preserved, required round-trip validation
failed, engineering gate not evaluated.

## Preserved provenance

```text
campaign: operating_decision_prospective_v1_2026_09
partition: p2_disposable_candidate_exclusion_pilot
source revision: 9db5f3f5b7a0fd92710ef5971091c16102877ac8
protocol digest: efbffb3e2ba0a9229f98c23c548bc6ae3957e492c6895bb7541ece8ed1eba265
stored scientific-result digest: b5b098b73050f460b350a32f0aeaaf9996d81a5179cdd2be3edc677798f7c84e
archive-content digest: 5b1739694fc13e7c71fc42e6fd9b2d262737d90ee2e6af0b22fcbfe16ab3e506
JSON SHA-256: 03f697b9b0013672c72164eba58353d230542dd065b3203f1a3508650fccfdf4
compact-report SHA-256: 2017dfa9e011171df7db5f94fc93b60a104c5366fc08dc5fd435b051a93ab334
JSON bytes: 15411937
```

The detached manifest still matches the preserved JSON and compact report.
Those hashes identify the exact bytes that were written; they do not establish
that the loaded archive satisfies its semantic validation contract.

## Validation failure

After the runner and artifact writer completed, an independent load of the
saved JSON followed by `validate_prospective_pilot_archive` stopped with:

```text
ValueError: complete uncertainty protocol/config is invalid
```

The failure occurs at the physical-configuration equality check. The in-memory
payload retained tuple-valued configuration fields, while the JSON round trip
loaded those arrays as lists. Pre-P2 checks exercised the in-memory validator
but did not require the final saved bytes to be loaded and validated. The
archive therefore did not meet the required reproducibility boundary.

## Interpretation boundary

P2 is not an accepted or failed draw-count pilot. Its engineering gate was not
evaluated. No saved action-agreement, regret, eligibility, failure, runtime,
projection, recommendation, or conditional-follow-up field may select a draw
count, trigger N=32, freeze a compute budget, or support a scientific claim.
Those fields remain only inside the hash-identified archive as quarantined
incident material.

Scientific interpretation stopped as soon as the round-trip failure was
confirmed. The P2 namespace is permanently closed and will not be overwritten
or rerun. Its conditional N32 artifact was never opened and is retired with its
invalid parent. Development, calibration, and reserved partitions remain
unopened.

## Required replacement

The representation defect is an ordinary software repair and does not consume
the protocol's planned scientific redesign. Repair commit `8c232af` implements
JSON-native configuration payloads, final-byte validation for P3
and conditional N32 archives, pilot protocol v4, and N32-follow-up protocol v2.
Local validation under CPython 3.10.12 passed 8 targeted protocol/archive tests
in 751.650 seconds, 137 prospective tests in 1330.082 seconds, 4 Phase B tests
in 0.195 seconds, and the full dependency-equipped 810-test suite in 1489.060
seconds. These runs overlap and are not a single summed test count. Independent
clean-clone review on 2026-09-19 of exact commit `8c232af` passed six focused
tests in 462.860 seconds, reproduced this incident, and found no scientific
drift or P3 artifact. This audit closeout is recorded in this commit; the
commit must be pushed and its exact HEAD must pass CI. The
implementation and local checks are not scientific evidence and do not
authorize P3.

Only after that closeout may the fresh four-block partition
`p3_disposable_archive_roundtrip_replacement_pilot` open. Its conditional
all-12-case continuation is
`p3_disposable_archive_roundtrip_replacement_pilot_n32_all_cases_v1` and may
open only if a valid P3 archive independently satisfies the unchanged N=16
trigger. P3 must retain the same N=4/8/16 design, acceptance rule, case count,
and scientific interfaces; P2's quarantined fields cannot tune it.
