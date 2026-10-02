# Prospective operating-decision Phase D plan

Date: 2026-09-29. Last reconciled: 2026-10-02. Status: historical Phase D0 v1 and the audit repair gate are
complete. Phase D1 generated and validated the one authorized nominal
development-tuning archive, Phase D2 produced four finite development offsets,
Phase D3 selected the provisional rule, and the repaired Phase D4 execution
passed its frozen N=16/N=32 sensitivity gate at `028fbca`. N=16 remains frozen
and Phase D5 completed all 48 declared map cells at `72e711b`. The provisional
design is frozen and the Phase D6 internal check is authorized. This document fixed
the work sequence and numerical choices before
`p1_development_tuning` was generated. It is not itself a scientific result or
a calibration record. The Phase D1 evidence is recorded in the
[Phase D1 result](OPERATING_DECISION_PHASE_D1_RESULT.md), and its offset
analysis is recorded in the
[Phase D2 result](OPERATING_DECISION_PHASE_D2_RESULT.md). The grid result is
recorded in the [Phase D3 result](OPERATING_DECISION_PHASE_D3_RESULT.md). The historical implementation and rehearsal remain in the
[Phase D0 result](OPERATING_DECISION_PHASE_D0_RESULT.md); the superseding repair
gate is tracked in [the audit repair record](THERMOTWIN_AUDIT_REPAIR_RECORD_2026_09_27.md).

## Purpose and entry state

Phase D will turn the Phase B interfaces into one complete development rule
that can choose among:

1. stop now and run verification;
2. collect the fixed thermal package;
3. collect the fixed voltage package; or
4. install the temporary face-temperature probe.

The phase will also produce the measurement-selection maps promised in the
original experiment. It may use development truth offline to choose offsets
and thresholds, but the selector itself must continue to receive acquisition-
only evidence.

The entry state is the committed Phase C freeze at `dc29fbf`. It binds the P4
source at `c0518f5`, 16 prospective draws, at most one unusable draw per
source/action, four workers, complete draw denominators, the no-gain failure
score, and the requirement that all three measurement actions remain
eligible. The Phase C freeze payload digest is
`f9fd7d6760549f2b69ca4eac8fecabc3fc5783f3d2309e0d20fd527cd13830a8`.

All new Phase D evidence uses CPython 3.10.12 on Darwin arm64 with the exact
scientific packages pinned in
`thermotwin/requirements-prospective-phase-d.txt`. The executor rejects a
different Python, platform, architecture, NumPy, SciPy, Matplotlib, or PyTorch
identity. Historical D0 v1 remains a CPython 3.13.3 artifact and is not silently
reclassified as evidence from the new runtime.

Phase D uses 30 new paired blocks, with one case from each of the three truth
families in every block:

| Partition | Blocks | Cases | Role |
| --- | ---: | ---: | --- |
| `p1_development_tuning` | 20 | 60 | Estimate development offsets, select the rule, and build development maps. |
| `p1_development_internal_check` | 10 | 30 | Apply the locked provisional rule once and measure whether it transfers to new development cases. |

Calibration and reserved partitions remain closed throughout Phase D.

## Phase D0: implement and freeze the development protocol

Before generating a development case, add a versioned Phase D module, archive
schema, validator, command-line entry point, and focused tests. The protocol
must bind:

- the Phase C payload and source revision;
- both partition names and their exact block counts;
- the four-policy catalog, primary cost scenario, 12 cost-map cells, and the
  sensor-scenario catalog below;
- the offset estimator, rule grid, objective, tie breakers, support fallback,
  draw-sensitivity subset, and revision rule;
- the truth-reveal chronology and every required output;
- the source manifest, runtime manifest, semantic random-stream inventory,
  and artifact digest.

Run a one-block disposable rehearsal through save, `json.load`, validation,
and report generation. The archive validator checks provenance, structure,
formulas, streams, and recorded consistency; it does not independently rerun
the simulations and fits. Repeat the actual rehearsal computation under the
same frozen runtime and stream identities, and require identical scientific
digests. Full archive identities may differ because they include timings. The
rehearsal must use a disposable namespace and cannot contribute to any Phase D
numerical choice. Exact-head CI must pass before the tuning partition opens.

Historical Phase D0 v1 passed at source `a280d7f` on 2026-09-26 under
CPython 3.13.3. Its archive and hashes remain unchanged in the
[Phase D0 result](OPERATING_DECISION_PHASE_D0_RESULT.md). A September 26 audit
then found three pre-development protocol issues: undefined infinite-offset
behavior, timing in the scientific digest, and a worker-only memory estimate
presented as a concurrent cap. Protocol v2 supersedes those interfaces without
reinterpreting the v1 result. No development, calibration, or reserved case was
opened.

## Phase D1: generate the nominal tuning evidence

Generate all 20 tuning blocks under nominal sensing, keeping the cases paired
across the four policies. Every case must retain:

- the common initial acquisition and its candidate fits;
- all N=16 prospective draw records for thermal, voltage, and face-temperature
  actions;
- the scorecard and selected action saved before any future observation or
  truth is revealed;
- counterfactual outcomes for all four fixed policies, including verification
  and the unloaded final forecast;
- the final true margin, saved decision, raw interval, realized diagnostic
  energy, run count, elapsed schedule, instrument count, failures, candidate
  exclusions, and abstention reason; and
- a complete stream audit and source-bound provenance record.

Truth-family labels, true margins, future observations, verification results,
and target responses must remain absent from the online selection object.
They may enter only the offline development analysis after every policy record
for the case is saved.

## Phase D2: estimate development offsets

Use the raw final operating-margin interval `[L,U]` from each counterfactual
policy and its true margin `m`. The case score is:

`s = max(0, L - m, m - U)`.

A missing, invalid, or nonfinite interval has score `+infinity`; it is never
dropped. For each action, take the maximum score across the three families in
each paired block, giving 20 block scores. The action offset is the 18th
smallest score, the nearest-rank 90th percentile, rounded upward to the next
0.001 K. This produces `d_stop`, `d_thermal`, `d_voltage`, and `d_face`.

Each separate estimate therefore requires all 20 paired blocks. If the 18th
score is infinite for **any** of the four policies, stop the primary Phase D
campaign as infeasible before selecting a tuning winner or opening the internal
check. Retain all scores, failures, and diagnostic results. Do not disable an
action, substitute a large finite value, drop the failed case, or form an
infinite utility baseline. No family-specific or one-candidate-specific offset
is estimated; those strata are reported separately.

These offsets are development heuristics for action selection. They are not
the independent Phase E calibration correction and cannot be described as a
coverage guarantee.

Phase D2 completed at source `ed45041`. The four 18th-order statistics were
0.000000 K for stop, 0.000000 K for thermal, 0.073056 K for voltage, and
0.097166 K for face temperature. Upward rounding produced offsets of 0.000,
0.000, 0.074, and 0.098 K, respectively. All were finite, so Phase D3 is
authorized. See the [Phase D2 result](OPERATING_DECISION_PHASE_D2_RESULT.md).

## Phase D3: select one rule from a fixed grid

Rescore the authenticated raw prospective evidence without refitting. Evaluate
the Cartesian product below, 81 rules in total:

| Parameter | Frozen development grid |
| --- | --- |
| General stopping clearance | `0.00`, `0.05`, `0.10` K |
| Additional one-candidate stopping clearance | `0.00`, `0.05`, `0.10` K |
| Minimum expected width reduction | `0.00`, `0.01`, `0.025` K |
| Minimum utility per normalized cost | `0.00`, `0.01`, `0.025` K |

The action order, strict `>` threshold semantics, 12-decimal merit rounding,
cost formula, verification rule, candidate menu, and failure policy remain
unchanged.

For each fixed policy and the selector, retain the emitted raw interval
`[L,U]`, expand it with that policy's development offset to
`[L-d_a,U+d_a]`, and recompute approve/reject/abstain from the expanded
interval. Missing or nonfinite intervals, verification failures, and pipeline
failures remain abstentions. Preserve raw and adjusted intervals, decisions,
verification, failures, action, and block loss. Truth enters only the offline
loss after all policy records are saved; the independent Phase E correction is
not used here.

Choose the rule with the smallest mean paired-block loss from these
development-adjusted decisions under the already declared balanced
decision/resource loss: false approval 100, false rejection 20, abstention 1,
added run 0.10, added sensor 0.10, and normalized incremental energy 0.10. Average the three family losses within each block, then average
the 20 blocks. Break exact ties, in order, by:

1. fewer false approvals;
2. fewer false rejections;
3. more definitive decisions;
4. lower mean realized diagnostic energy; and
5. lexicographically smaller
   `(general clearance, one-candidate clearance, minimum reduction,
   minimum utility)`.

Report every grid row, not only the winner. Report approval, rejection,
abstention, failure, error, coverage, energy, time, and instrumentation counts
overall, by truth family, by selected action, and for the one- and two-
candidate acquisition strata. A rule that selects only one action is a valid
development outcome.

Phase D3 completed at source `b0fd063`. The selected grid values were
`(0.00, 0.00, 0.00, 0.025)` in the declared order. Nine rows were tied before
the lexicographic tie breaker. The selected rule made 32/60 definitive
development decisions with no false approvals or false rejections and mean
paired-block loss 0.6323. Stop-now made 25/60 definitive decisions with mean
loss 0.5833, so the selector's additional coverage did not offset its resource
cost under the development objective. See the
[Phase D3 result](OPERATING_DECISION_PHASE_D3_RESULT.md).

## Phase D4: recheck draw-count sensitivity

The sensitivity subset is fixed before tuning outcomes are seen: tuning block
indices `0`, `5`, `10`, and `15`, including all three families in each block.
Generate an authenticated N=32 continuation for those 12 cases after the
offsets and provisional rule are known. Compare N=16 and N=32 using the same
offsets, thresholds, costs, and eligibility rule.

N=16 remains accepted only if:

- at least 11 of 12 selected actions agree;
- every changed choice has evaluable normalized utility regret;
- the maximum changed-choice regret is at most 5%;
- every measurement action is eligible at both counts; and
- neither archive has a pipeline, selection, provenance, or diagnostic
  failure.

If this gate fails, generate N=32 for the remaining 16 tuning blocks and an
N=64 continuation on the same four-block subset. N=32 may replace N=16 only if
the N=32-to-N=64 comparison passes the same gate. Refit no physics and change
no thresholds while testing draw count. If N=32 also fails, stop Phase D and
report that the estimator is not stable within the declared compute bound.
There is no outcome-directed subset replacement and no N above 64.

The D4 implementation is frozen in the
[Phase D4 protocol](OPERATING_DECISION_PHASE_D4_PROTOCOL.md). Its monitor
preflight preserved the initial sandbox permission failure and the successful
approved retry. The exact-head execution is recorded in the
[Phase D4 result](OPERATING_DECISION_PHASE_D4_RESULT.md): all 12 actions agreed,
no failure occurred, and the gate passed.

## Phase D5: build the measurement maps

The executable design, exact map classifications, persistence rules, and
validation boundary are frozen in the
[Phase D5 protocol](OPERATING_DECISION_PHASE_D5_PROTOCOL.md). Exact-head CI
must pass before the three stress scenarios execute.

### Cost map

Use the existing authenticated nominal predictions to evaluate all 12 frozen
cost cells: balanced, energy-dominant, bench-time-dominant, and
instrumentation-dominant weights, each with face instrumentation priced at
one quarter, equal, and four times the reference instrument. The primary cell
is `balanced_face_equal`. Cost-only cells must not trigger simulation or
refitting.

### Sensor-quality and loading map

Use four exact physical scenarios on the 20-block tuning partition:

| Scenario | Change from nominal | Required recomputation |
| --- | --- | --- |
| `nominal` | Voltage noise `0.002 V`; face noise `0.02 K`; probe capacitance `5 J/K`; response time `2.5 s` | Existing nominal archive. |
| `high_voltage_noise` | Voltage noise `0.004 V`; all other fields nominal | Rerun and refit the affected voltage evidence and outcome. |
| `high_face_noise` | Face-temperature noise `0.04 K`; all other fields nominal | Rerun and refit the affected face-temperature evidence and outcome. |
| `high_probe_loading` | Probe capacitance `10 J/K`; response time `2.5 s`; all noise nominal | Rerun the loaded face trajectory and refit the affected face-temperature evidence and outcome. |

The run-bias/noise ratio remains 2.5 in every scenario. The altered noise
scale is known to the corresponding likelihood; the high-loading probe uses
the altered nominal in both generation and fitting. Unaffected common initial
measurements may be reused only when their authenticated evidence digest is
identical. For a conservative implementation and budget, a complete replay of
each stress scenario is allowed.

For every sensor scenario and cost cell, publish a measurement-selection map
with counts and rates for selected action and eventual definitive decision.
Also show action ineligibility, selection failure, verification failure, and
no-useful-measurement regions. A selected voltage or face action that later
abstains must remain visibly distinct from a successful definitive decision.

The sensor scenarios are development sensitivity analyses. Only nominal
sensing with `balanced_face_equal` is the primary procedure entering Phase E,
unless a later pre-calibration design freeze explicitly promotes another
scenario and supplies its own valid calibration plan.

## Phase D6: lock and run the internal check

After the offsets, rule, draw count, and maps are committed, authorize
`p1_development_internal_check`. Apply the primary nominal rule once to its 10
new blocks. Do not use internal-check labels to pick among the 81 grid rows.

The check must report the same complete metrics as tuning, plus the difference
from tuning for action shares, definitive-decision coverage, false approvals,
false rejections, failures, realized energy, and instrumentation. Family C and
one-candidate stopping receive explicit rows even when their denominators are
small. Empty denominators are `N/A`.

Poor benefit, low action diversity, or failure to hit the proposed 10% false-
approval and 70% decision-coverage targets does not by itself reopen tuning.
Those are reportable outcomes. A redesign is permitted only for one of these
predeclared conditions:

- a material information-boundary, physics, provenance, or implementation
  defect;
- at least two false approvals in distinct blocks within the same selected-
  action and candidate-count stratum, indicating repeated development
  overconfidence; or
- a measurement action is ineligible in more than 20% of the 30 cases.

If a redesign occurs, the first internal check becomes tuning evidence. Make
at most one change within the already declared offset/grid/fallback choices,
allocate and commit a new 10-block internal-check namespace, and rerun the
check once. A defect repair also requires a versioned replacement namespace;
the old artifact remains preserved. A second failure ends Phase D as a
feasibility limitation.

## Compute and storage budget

The Phase C conservative nominal estimates are:

| Work | Wall time at four workers | CPU time | Archive bytes |
| --- | ---: | ---: | ---: |
| 20-block tuning | 9.97 h | 36.61 h | 76,965,610 |
| 10-block internal check | 4.98 h | 18.31 h | 38,482,805 |

Three complete 20-block sensor-stress replays add at most 29.90 wall hours,
109.84 CPU hours, and 230,896,830 archive bytes. The planned Phase D nominal
plus sensor-map total is therefore capped at approximately **44.85 wall
hours**, **164.76 CPU hours**, and **346,345,245 bytes** before draw-count
continuations. The 185,204,736-byte Phase C value is a worker-only estimate, not a
whole-workflow cap. Before tuning, a constructed 20-block-sized disposable
archive must exercise retained results, canonical serialization, write, load,
and validation while aggregate coordinator-and-worker RSS is sampled. The
chosen machine has 16 GiB physical memory; the prospective process-tree limit
is 4 GiB with at least 50% measured headroom required. Record elapsed time, CPU
time, disk size, intermediate copies, worker count, and every measured stage.

The planned four-block N=32 sensitivity adds the measured P4 continuation
budget of approximately 4.06 wall hours, 14.27 CPU hours, and 34,992,556
bytes. N=16 failure, N=64 comparison, an all-tuning N=32 extension, or a
replacement internal check is contingency work and must receive a new
machine-readable budget record before it runs. Reduce or omit secondary
sensor-map work before weakening the primary nominal development and internal
check.

Before Phase D4 or any later long computation, synchronously test that the
process-tree memory monitor can observe the intended worker processes and
record its permission status. Abort before scientific computation if that
preflight fails. Do not rerun Phase D1 to replace its missing monitor sample;
its validated evidence remains closed against regeneration.

## Required Phase D outputs

Phase D is complete only when the repository contains:

1. a machine-readable protocol and validator that reproduce this plan;
2. immutable tuning and internal-check artifact identities with byte sizes and
   SHA-256 hashes;
3. the fitted development offsets and the complete 81-row tuning table;
4. one selected primary rule and its protocol digest;
5. the nominal N=16/N=32 sensitivity result and any declared continuation;
6. the 12-cell cost map and four-scenario sensor-quality/loading map;
7. selector and fixed-policy metrics for decisions, errors, abstention,
   interval coverage, failures, energy, time, instruments, and computation;
8. explicit Family C and one-candidate-stratum diagnostics;
9. a chronology of every attempted variant, including invalid or superseded
   development evidence; and
10. a Phase D freeze record that either authorizes Phase E or closes the
    experiment as a documented feasibility result.

## Execution order and stopping points

The complete protocol-v2 repair gate, Phase D1 archive capture, Phase D2
finite-offset gate, Phase D3 grid selection, and Phase D4 draw-count gate have passed,
as recorded in the [audit repair record](THERMOTWIN_AUDIT_REPAIR_RECORD_2026_09_27.md)
and [Phase D1 result](OPERATING_DECISION_PHASE_D1_RESULT.md). The
[Phase D2 result](OPERATING_DECISION_PHASE_D2_RESULT.md) records the four
offsets, and the [Phase D3 result](OPERATING_DECISION_PHASE_D3_RESULT.md)
records the selected provisional rule. The
[Phase D4 result](OPERATING_DECISION_PHASE_D4_RESULT.md) freezes N=16 after
12/12 agreement with N=32. The
[Phase D5 result](OPERATING_DECISION_PHASE_D5_RESULT.md) records all 48 map
cells and the committed provisional-design freeze. The tuning partition is
closed against regeneration. The next permitted scientific action is to
commit the Phase D6 execution and analysis protocol and then open the 10-block
internal check once. Phase E may start only after D6 and the final Phase D
freeze.
