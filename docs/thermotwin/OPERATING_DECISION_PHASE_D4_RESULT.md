# Prospective operating-decision Phase D4 result

Date: 2026-09-29. Status: complete and passed. The predeclared four-block
draw-count sensitivity check compared the authenticated N=16 Phase D1 prefix
with its N=32 continuation under the Phase D3 provisional rule. The gate
passed, N=16 remains frozen, and Phase D5 measurement-map construction is
authorized.

## Authenticated execution

Phase D4 ran from clean source
`028fbca2c4e7fb4099478e46cc2f7a1a645db67a` under the frozen CPython 3.10.12
environment. It used tuning blocks `0`, `5`, `10`, and `15`, with all three
truth families in every block. Each N=32 case reproduced the complete saved
N=16 uncertainty record as its authenticated prefix before both draw counts
were rescored with the same D2 offsets, D3 thresholds, eligibility rule, and
primary cost cell.

| Record | Identity |
| --- | --- |
| Scientific result digest | `1282a74f71073cffc1960d54a63a088c68f0d8e19838bb57b583c7617d4613b9` |
| Complete JSON | 19,220,851 bytes; SHA-256 `c63c56e654eef5ab3f7be4de5e92dff05e32c4baf13d976fb193446d226dc26b` |
| Compact report | SHA-256 `665dba6de4b2133a327307a55ba39b1652e8274bac52faa38f7dc1917a9b1a28` |
| Blocks and cases | 4 paired blocks; 12 cases |
| Draw counts | N=16 candidate; N=32 reference |

The final archive and all four atomic block files passed serialized reload and
source-bound validation. The process-tree monitor collected 13,475 samples
without error. Peak aggregate RSS was 1,196,572,672 bytes, below the frozen
4 GiB limit. The run used four workers and took 13,976.3 wall seconds
(3.88 hours); recorded block CPU time was 49,177.3 seconds.

## Gate result

| Criterion | Required | Observed |
| --- | ---: | ---: |
| Selected-action agreement | At least 11/12 | 12/12 |
| Maximum regret for a changed choice | At most 5% | 0%; no choices changed |
| Unevaluable changed choices | 0 | 0 |
| Measurement-action ineligibility records | 0 | 0 |
| Pipeline failures | 0 | 0 |
| Selection failures | 0 | 0 |
| Unavailable draw diagnostics | 0 | 0 |
| Whole-draw failures | 0 | 0 |

The complete acceptance gate passed. N=16 therefore remains the prospective
draw count. The all-tuning N=32 and four-block N=64 contingency is neither
triggered nor permitted by this result.

## Interpretation and next action

This result supports numerical stability of the selected action on the
predeclared development subset within the declared N=16 versus N=32 test. It
does not estimate decision accuracy, calibration, or performance on new
devices.

The next permitted scientific action is Phase D5, whose executable design is
frozen in the [Phase D5 protocol](OPERATING_DECISION_PHASE_D5_PROTOCOL.md).
Cost-only maps may rescore
the preserved nominal evidence without simulation. The three sensor-stress
maps must use their frozen physical scenarios on the same closed tuning
partition. The internal-check, calibration, and reserved partitions remain
closed until D5 is complete and the provisional design is committed.
