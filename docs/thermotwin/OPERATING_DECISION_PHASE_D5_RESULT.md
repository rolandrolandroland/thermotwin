# Prospective operating-decision Phase D5 result

Date: 2026-10-02. Status: complete and passed its archive and execution
checks. The frozen measurement-selection rule was mapped over all 48 declared
sensor-and-cost cells. The primary nominal design remains unchanged, the
provisional design is now frozen, and only the 10-block Phase D6 internal check
is authorized.

## Authenticated execution

Phase D5 ran from clean source
`72e711bc29bded623f7f196f519c3a0d41b7622c` under the frozen CPython 3.10.12
environment. It reused the authenticated nominal D1 evidence and performed the
three declared 20-block stress replays. Each completed stress block was
validated and saved atomically. Interrupted coordinator sessions were resumed
only after every retained block passed the source, protocol, scenario,
configuration, block-index, scientific-digest, serialized-byte, and archive-
digest checks. No partial file was accepted.

| Record | Identity |
| --- | --- |
| D1 tuning JSON | SHA-256 `d83ae0e8ed52f328bf0a2accf5419dbd3a9ae10939552a14ccc0a445a1f30a2e` |
| D2 offsets JSON | SHA-256 `3b69f063de593a48af8430e94f802f43bf81dc02e0bec999b3b510ed143b0c2f` |
| D3 grid JSON | SHA-256 `8664fb237efba6854ed46ea600d9ad88d82117d795d108839c7a8deaeff69397` |
| D4 sensitivity JSON | SHA-256 `c63c56e654eef5ab3f7be4de5e92dff05e32c4baf13d976fb193446d226dc26b` |
| D5 source manifest | `c793021767ccf28cfa86a5f65f686e2d56b51270808c43f504c175a0edbd2845` |
| D5 protocol | `57d9cc8c4a5f737e297339092df224d4987493fa52c7ad289ddc58779bb0375b` |
| D5 scientific result | `b26352040b44c595f86d76003dd55bd7a2d8b43c0ccf9b825dda53c0a6edf4f3` |
| Complete D5 JSON | 1,196,903 bytes; SHA-256 `8ee535b760be9254bb9ddba8b82a909ee6fce6dd2f8c07b5790e47bdf4bf95c2` |
| Compact report | SHA-256 `e2d6079888fbcfdef28a294af2393d586ab6cf32dac1ac0454233125da81f640` |

The execution path reloaded and validated the final archive before it wrote
the resource record. All 60 stress blocks and all three scenario archives also
passed JSON reload, and the detached final-file hashes match.

On 2026-10-02, a fresh clean clone of `72e711b` restored the exact pinned
scientific runtime from the committed requirements and independently returned
`Phase D5 measurement maps: VALID` with the same scientific result digest.

## Primary result

The primary procedure remains nominal sensing under
`balanced_face_equal`. It is the same D3 selector, offsets, N=16 draw count,
and one-whole-draw allowance accepted before D5; the map did not retune any
parameter.

| Outcome | Count |
| --- | ---: |
| Stop now | 34/60 |
| Fixed thermal selected | 16/60 |
| Fixed voltage selected | 5/60 |
| Fixed face temperature selected | 5/60 |
| Definitive final decision | 32/60 |
| No useful measurement | 9/60 |
| Selection failures | 0 |
| Verification failures | 0 |
| Pipeline failures | 0 |

The primary result exactly reproduces the D3 development selection and outcome
counts. That agreement is expected because the nominal cell reuses the
authenticated D1 evidence; it is an identity check, not independent evidence.

## Sensor and cost map

The balanced/equal-cost cell changed under the three physical sensitivities as
follows:

| Sensor scenario | Stop | Thermal | Voltage | Face | Definitive | Ineligible actions |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Nominal | 34 | 16 | 5 | 5 | 32/60 | None |
| High voltage noise | 31 | 23 | 5 | 1 | 31/60 | None |
| High face noise | 33 | 20 | 3 | 4 | 30/60 | None |
| High probe loading | 32 | 20 | 3 | 5 | 30/60 | Face action ineligible in 1/60 cases |

Across all 48 map cells, stop-now selections ranged from 29 to 42, thermal
from 8 to 31, voltage from 0 to 10, and face temperature from 0 to 6.
Definitive decisions ranged from 28 to 32 of 60, while no-useful-measurement
counts ranged from 4 to 17. There were no selection, verification, or pipeline
failures in any cell. Thermal and voltage were always eligible. The one face-
action ineligibility under high probe loading remains in every affected cell's
denominator and is explicitly frozen in the map record.

The map therefore demonstrates that the selector responds to declared sensor
quality, probe loading, bench time, and instrumentation price. It does not
show that one cell transfers to new devices or that the selector beats a fixed
policy.

## Resource record

The three stress archives total 228,560,784 bytes, below the approximate
230,896,830-byte budget. Aggregate recorded block CPU time was 460,117.9
seconds (127.81 hours), above the approximate 109.84-hour stress budget. The
final monitored invocation recorded 24,992.9 wall seconds and peak aggregate
RSS of 1,630,978,048 bytes, below the frozen 4 GiB limit, across 23,893 valid
samples with no monitoring error.

Because the coordinator was restarted from authenticated checkpoints, the
final invocation's wall time is not total campaign wall time. The block CPU
sum and archive bytes cover all three complete stress scenarios; no total wall
time is claimed from the per-invocation records.

## Freeze and next action

The complete provisional procedure is frozen in
`thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_D_PROVISIONAL_FREEZE.json`.
It binds the D1-D5 evidence identities, selector and offsets, draw and failure
rules, primary scenario, 48-cell sensitivity summary, runtime, and partition
authorization. Its artifact digest is
`a8b4cc22f8bf1d868a743c2427c1225483f954a8aeb9f751e38bec60dd949ca7`.

The next permitted scientific action is to commit the Phase D6 execution and
analysis protocol and then open `p1_development_internal_check` once for 10
paired blocks and 30 cases. Independent calibration and reserved evaluation
remain closed. D5 is development evidence and does not establish calibrated
risk, transfer, hardware performance, or superiority over any fixed policy.
