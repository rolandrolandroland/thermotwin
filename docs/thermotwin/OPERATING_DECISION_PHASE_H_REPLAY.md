# Phase H numerical replay

Date: 2026-10-29. Status: numerical replay, independent evidence verification
and test acceptance passed. This is a bounded implementation check of the completed
experiment. The [primary result](OPERATING_DECISION_PHASE_F_RESULT.md) remains
unchanged and did not meet its success criterion.

Fifteen archived cases reproduced the common acquisition fits, 15 fixed-policy
fits and verified intervals, six predictive draws, and 116 calibrated decisions
exactly under the original source and runtime. This supplies the computational
replay requested by the [completion plan](OPERATING_DECISION_COMPLETION_PLAN_2026_09_17.md).
It checks reproducibility of selected numerical paths; it does not establish
real-device accuracy or validate a redesigned selector.

## Sample fixed before computation

The [replay plan](../../thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_H_REPLAY_PLAN.json)
was saved at 19:06:04 UTC on October 29, before any replay fits. It binds the
runner file, accepted archive, original source/runtime, deterministic target
list and comparison rule. Selection uses the already exposed saved outcomes;
this is a post hoc audit sample, not a prospective efficacy experiment.

For each family/action combination, the rule selects the first block where the
selector chose that action. If no such case exists, it replays the saved fixed
procedure on that family's first block and records the fallback.

| Family | Stop block | Thermal block | Voltage block | Face temperature block |
| --- | ---: | ---: | ---: | ---: |
| Matched four state | 0 | 19 | 8 | 0, unselected fixed fallback |
| Extra interface mass | 0 | 5 | 26 | 2 |
| Temperature dependent contact | 0 | 1 | 3 | 24 |

The matched family selected face temperature zero times, so the fallback is
explicitly an input/fit/probe check of that fixed comparator. It is not presented
as a selected face-temperature example.

Three further saved selector cases minimize the absolute distance of either
final interval endpoint to zero within their decision label. Ties use block
then the frozen family order.

| Saved decision | Block | Family | Selected action | Nearest final endpoint (K) |
| --- | ---: | --- | --- | ---: |
| Approve | 9 | Matched four state | Stop | 0.0019944757129397583 |
| Reject | 25 | Matched four state | Stop | -0.00006671911812722309 |
| Abstain | 87 | Temperature dependent contact | Stop | -0.0012326274078289234 |

These labels, endpoints and exact decisions reproduced without a numerical
tolerance. The failure sample uses predictive draws because the 300 primary
cases contain no pipeline failures. All 20 archived whole-draw failures occur
in the face-temperature prediction branch; the first one is included.

| Predictive check | Block | Family | Source model | Action | Draw index |
| --- | ---: | --- | --- | --- | ---: |
| Stable | 0 | Matched four state | Four state | Thermal | 0 |
| Stable | 0 | Matched four state | Four state | Voltage | 0 |
| Stable | 0 | Matched four state | Four state | Face temperature | 0 |
| Usable candidate exclusion at a bound | 0 | Extra interface mass | Four state | Thermal | 1 |
| Usable non-converged candidate exclusion | 0 | Temperature dependent contact | Five state | Face temperature | 5 |
| Whole draw with no admissible candidate | 7 | Temperature dependent contact | Five state | Face temperature | 10 |

Tasks sharing a case reuse its reconstructed common acquisition within that
task. The complete list contains 15 distinct case tasks, including one with only
a predictive failure replay and one with two fixed-policy replays.

## Numerical checks

Observed acquisition and verification inputs are deserialized from the accepted
archive, including current schedules, channels and physical probe installation.
The replay does not call the reserved observation or target-truth generators.
Each common acquisition is refitted with the original multistart procedure.
Each listed fixed-policy acquisition is then refitted, independently verified,
and converted to its raw margin interval with the unchanged numerical code.

Comparison includes parameters, covariance, objective, convergence and bound
flags, iteration/evaluation counts, verification scores, candidate exclusions,
model intervals and the raw decision. The only excluded saved-decision field
is `decision_computation_seconds`. There is no numerical tolerance, replacement
case, extra optimizer iteration, reactive refit or automatic retry.

Each sampled predictive draw regenerates the same parameter/probe offsets and
synthetic observations using its original semantic stream keys. Its complete
candidate outcomes, synthetic-observation digest, failure/attrition state and
scored width must match the archive exactly. The whole-draw failure and both
usable candidate-exclusion paths reproduce, preserving their distinct handling.
All 72 reused stream-use records match the original retained stream inventory.

The four predeclared clearances are reapplied to each replayed fixed procedure
and to the selector when it chose the same acquisition. All 116 resulting
decisions and outward-rounded intervals equal their before-reveal saves.
Each procedure keeps its own correction. The complete N=16 simulation and
action-selection process is not rerun; the six predictive draws are the
predetermined computational sample.

All 117 frozen dependencies match in both the working repository and the
persistent source clone. Imported numerical modules must come from that clone
and appear in the frozen dependency manifest. Execution uses CPython 3.10.12,
Darwin arm64, and the original package versions and mathematical environment.

## Resources and acceptance

The numerical execution phase took 277.033 seconds (4.62 minutes) with two
workers. Their combined CPU time was 499.010 seconds; the coordinator used
1.250 CPU seconds in the measured phase. The process-tree monitor recorded
267 samples, with a peak of 1,260,650,496 bytes (1.17 GiB), no monitor error and
no observed crossing of the declared 4 GiB limit.

This sampled memory/time scope covers numerical execution, the coordinating
process retaining the archive and the concurrent workers. Initial archive
reconstruction/import preparation, the full test run and the later independent
reload are outside that measurement; it is not a whole-closeout resource peak.
The original Phase F benchmark budget remains recorded separately in the
[primary result](OPERATING_DECISION_PHASE_F_RESULT.md).

All 1,199 existing repository tests passed in 1,210.339 seconds. The 42 focused
checks also passed in 4.364 seconds, including seven new replay tests covering sample
determinism, absent-cell handling, failure retention, exact comparison, input
transport/probe loading, overwrite refusal and scientific/performance identity.
The repository run covers the 1,199 tests present when it started. The seven
new replay tests passed separately; overlapping focused checks are not counted
twice.

The independent reload passed. Three additional probes altered copies of a
fitted interval, a clearance's numeric type, and a failed-draw flag, then
recomputed their file and scientific seals. The verifier rejected all three
against the original evidence. Accepted files were preserved.

## Evidence and reproduction

The [compact result](../../thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_H_REPLAY_RESULT.json)
retains the complete acceptance and resource record. The
[independent verification](../../thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_H_REPLAY_VERIFICATION.json)
binds the reloaded evidence, source, test logs and verification code.
The [runner](../../tools/replay_prospective_phase_f_numerics.py) implements the
saved plan. The [independent verifier](../../tools/verify_prospective_phase_h_replay.py)
uses the standard library and reads completed outputs without another numerical
refit. It checks the complete planned matrix and exact numerical payloads
against the original checkpoints, before-reveal decisions and stream inventory.

With the retained evidence available, routine verification can be repeated from
the repository root with a fresh output:

```sh
python -m tools.verify_prospective_phase_h_replay \
  --archive .artifacts/operating-decision/prospective-phase-f-reserved-7e3b9a3-v1/evaluation.json \
  --plan thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_H_REPLAY_PLAN.json \
  --replay-directory .artifacts/operating-decision/phase-h-numerical-replay-20261029 \
  --repository-root . \
  --frozen-source-root .artifacts/operating-decision/source-7e3b9a3 \
  --full-test-log .artifacts/operating-decision/phase-h-full-suite-20261029.log \
  --focused-test-log .artifacts/operating-decision/phase-h-replay-focused-tests-20261029.log \
  --output /tmp/thermotwin-phase-h-replay-verification.json
```

Detailed fit outputs, logs and the execution start record remain ignored in
`.artifacts/operating-decision/phase-h-numerical-replay-20261029/`. Whole-file
hashes bind every retained byte, including performance. Separate case and
aggregate replay scientific digests exclude elapsed-time/resource fields. The
original scientific-result digest is a separate, unchanged identity.

| Identity | Value |
| --- | --- |
| Original numerical source | `7e3b9a3be22d991ec2e588cbd4a53a9d4d2d27d7` |
| Original scientific result | `d565c0c6368b602eeb3d2721c128aab88221bce9a6b5abe68fcda4c727d4edf0` |
| Replay plan | `273b40f7e126c7a722da4bcba0261dc375816acac3daac3d544e3b74d3ad2c90` |
| Replay scientific result | `5aac186cc9e17ef8fcaf3045e311582059c0276f9df316742990ecea77904b86` |
| Complete replay result | `9e08a8922451102f3a0c33359eaccbb35afff857b4f7c94cc544ff8524971a91` |

The tools expose separate `plan` and `run` commands. Use the retained plan,
original source clone and runtime, and a fresh output directory when an explicit
computational replay is wanted. For routine checking, use the independent
verifier against the retained outputs; it requires the archive, plan, source
roots and passing test logs. Neither operation changes the accepted source,
selector, corrections, sample size or primary conclusion.

The [three final scientific figures](OPERATING_DECISION_FINAL_FIGURES.md)
are now complete. The full report, presentation and reproducibility/deposit
package remain separate closeout work. The historical result artifact's
pending-closeout flags describe its creation time; the
[project status](OPERATING_DECISION_PROJECT_STATUS.md) records later completion.
