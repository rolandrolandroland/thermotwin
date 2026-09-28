# Prospective operating-decision Phase D1 result

> Subsequent status: Phase D2 completed on this preserved archive and passed
> its finite-offset gate. See the
> [Phase D2 result](OPERATING_DECISION_PHASE_D2_RESULT.md).

Date: 2026-09-28. Status: the one authorized nominal development-tuning
partition was generated and independently validated. Phase D2 truth analysis,
offset estimation, rule selection, draw-count continuation, sensor-stress
scenarios, and the internal check have not started.

## Frozen source and authorization

Phase D1 ran from clean detached source
`f848a9583e0b3a345971401e08047b4769edbb9c` after the 848-test local
CPython 3.10.12 suite and
[exact-head CI run 36360855392](https://github.com/rolandrolandroland/thermotwin/actions/runs/36360855392)
passed. The non-generating preflight bound:

- campaign `operating_decision_prospective_v1_2026_09`;
- partition `p1_development_tuning`;
- 20 paired blocks and 60 cases;
- N=16 with at most one unusable draw per source/action;
- four workers;
- protocol digest
  `8b3bfcc4b274f0cbdf556a526c299d758aa11b8a4e4154f9fc80b4e14ef2f0c5`;
- source-manifest digest
  `9547c05bc833fc2d353b67fbaad2061027ac2f74099c8c1c724c8453e6a711d7`;
  and
- CPython 3.10.12 on Darwin arm64 with the pinned NumPy 2.2.6, SciPy 1.15.3,
  Matplotlib 3.10.8, and PyTorch 2.9.1 environment.

No other scientific partition was opened.

## Preserved evidence

All 20 paired blocks were written as separate canonical JSON checkpoints and
validated before the next completed block was accepted. The complete archive
then passed source-manifest, protocol, runtime, physical-configuration,
random-stream, fit-record, fixed-policy, chronology, block-inventory,
scientific-digest, content-seal, and canonical-size validation.

| Evidence | Recorded identity |
| --- | --- |
| Complete archive | 75,929,385 bytes; SHA-256 `d83ae0e8ed52f328bf0a2accf5419dbd3a9ae10939552a14ccc0a445a1f30a2e` |
| Scientific result | `efe860298104352fd0b6b35fc685e2320e6cfebd6db31fe8efc9876c509b1b5a` |
| Twenty-block checksum inventory | SHA-256 `745a8f7bfebcb41a86b7b5de8603e21c0afccc551aeb1b348652075004984800` |
| Post-run resource record | SHA-256 `b7948e007618ac4138c7665fdaead56a7e97ec641dc9806a122fa123d8bc8653` |

The ignored local evidence copy is
`.artifacts/operating-decision/prospective-phase-d1-f848a95/`. Its
`SHA256SUMS` file verifies the 20 block checkpoints, complete archive, compact
report, detached archive hashes, and resource record. The original execution
copy remains under `/private/tmp/thermotwin-phase-d1-f848a95-evidence/`.

The archive contains 20 blocks and 60 cases, with zero canonical block
failures and zero recorded case pipeline failures. These are execution and
archive-integrity observations. No selected-action counts, truth-family
comparisons, coverage values, errors, offsets, losses, or tuning winner were
computed or inspected for this record.

## Runtime and monitor record

The four-worker computation took 31,944.76 seconds of wall time (about 8 hours
52 minutes) and accumulated 122,224.20 seconds of block CPU time. The largest
worker peak retained in the block records was 59,228,160 bytes.

The complete archive and detached hashes were written before the command
returned a nonzero status. The final wrapper check reported
`PermissionError: [Errno 1] Operation not permitted: 'ps'` because the managed
macOS environment denied process-list access. Consequently this run has no
whole-process-tree RSS measurement: its resource record retains a null peak,
zero samples, and the exact error. This post-archive monitor failure did not
alter any block, scientific digest, archive byte, or validation result.

The pre-generation repair gate remains the applicable resource evidence: its
constructed 20-block archive probe measured a 581,386,240-byte process-tree
peak, and its repeated real replacement rehearsal measured at most 84,475,904
bytes. Phase D1 does not substitute its worker-only 59,228,160-byte maximum for
a process-tree measurement.

## Required stop and next action

Phase D1 stopped at the declared boundary. Truth analysis is recorded as
`not_performed_stop_before_phase_d2`.

The next permitted action is Phase D2 on this immutable archive: compute the
four predeclared block-max nonconformity sequences and their 18th-order,
upward-rounded offsets. If any required offset is infinite, retain the evidence
and close the primary Phase D campaign as infeasible. If all four offsets are
finite, Phase D3 may evaluate the fixed 81-rule grid. The internal-check,
calibration, and reserved partitions remain closed.
