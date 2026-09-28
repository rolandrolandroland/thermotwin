# Prospective operating-decision Phase D2 result

Date: 2026-09-28. Status: complete. The predeclared development offsets were
estimated from the preserved Phase D1 archive. All four required offsets are
finite, so the Phase D feasibility gate passed and Phase D3 grid rescoring is
authorized. No simulation, refitting, or new partition was used.

## Authenticated input and implementation

Phase D2 ran from clean source
`ed450411c6652a1e35a601ec8ecb9c426b761c51`. The analysis implementation
first replayed the complete Phase D1 validator against all 20 atomic block
files, then computed and retained every case and block score. Its source
manifest digest is
`2546f29d5420d23b3f45b105da5e2a78a78d826cdb81fd3c454e792bd6b79e3f`.

The input identities were:

| Phase D1 evidence | Identity |
| --- | --- |
| Complete JSON SHA-256 | `d83ae0e8ed52f328bf0a2accf5419dbd3a9ae10939552a14ccc0a445a1f30a2e` |
| Scientific result digest | `efe860298104352fd0b6b35fc685e2320e6cfebd6db31fe8efc9876c509b1b5a` |
| Source revision | `f848a9583e0b3a345971401e08047b4769edbb9c` |
| Blocks and cases | 20 paired blocks; 60 cases |

The exact CPython 3.10.12 Phase D environment was used. The focused Phase D
and D2 suite passed 33 tests before execution. A second complete replay from
the same clean commit reproduced the saved result and returned `VALID`.

## Frozen calculation

For each fixed policy and case, Phase D2 computed
`max(0, L - m, m - U)` from the saved raw interval `[L,U]` and the true margin
`m`. It assigned positive infinity to a missing, invalid, or nonfinite
interval or a fixed-policy pipeline failure. It then took the maximum across
the three truth families in each paired block. The offset is the 18th smallest
of the 20 block scores, rounded upward to 0.001 K.

| Fixed policy | Nonzero block scores | Infinite block scores | 18th order statistic (K) | Development offset (K) |
| --- | ---: | ---: | ---: | ---: |
| Stop now | 0/20 | 0/20 | 0.000000 | **0.000** |
| Thermal | 1/20 | 0/20 | 0.000000 | **0.000** |
| Voltage | 15/20 | 0/20 | 0.073056 | **0.074** |
| Face temperature | 11/20 | 0/20 | 0.097166 | **0.098** |

All 20 block scores and all 240 case-policy records remain in the
machine-readable result. No case or failed value was dropped. The zero stop
and thermal offsets are the direct predeclared order-statistic results; they
were not manually truncated or selected after comparison.

## Preserved result

| Evidence | Recorded identity |
| --- | --- |
| D2 JSON | 71,086 bytes; SHA-256 `3b69f063de593a48af8430e94f802f43bf81dc02e0bec999b3b510ed143b0c2f` |
| D2 scientific result | `242aa34e72cff260c7a16efc0ce8041928bebb63eeedc5dab88be86817d61b43` |

The ignored evidence directory is
`.artifacts/operating-decision/prospective-phase-d2-ed45041/`. Its detached
SHA-256 manifest verifies the complete JSON and compact report.

These offsets are development heuristics for Phase D rule selection. They are
not the independent Phase E calibration correction and do not establish a
coverage guarantee.

## Gate decision and next action

The saved status is `finite_offsets_ready`. Because none of the four 18th
order statistics is infinite, the predeclared infeasibility stop did not
trigger. Phase D3 may now rescore the same authenticated D1 evidence across
the fixed 81-rule grid without refitting. Phase D4 continuation, Phase D5
sensor-stress generation, the internal check, independent calibration, and
reserved evaluation remain closed.

## Subsequent status

Phase D3 completed at source `b0fd063` and selected the provisional grid values
`(0.00, 0.00, 0.00, 0.025)`. The full result and its development-only
interpretation are recorded in the
[Phase D3 result](OPERATING_DECISION_PHASE_D3_RESULT.md). Phase D4 is the next
authorized scientific action.
