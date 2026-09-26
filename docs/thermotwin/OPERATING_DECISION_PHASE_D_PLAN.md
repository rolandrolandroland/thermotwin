# Prospective operating-decision Phase D plan

Date: 2026-09-26. Status: execution plan; no development partition has been
opened. This document fixes the work sequence and the proposed numerical
choices that the machine-readable Phase D protocol must encode before
`p1_development_tuning` is generated. It is not a scientific result or a
calibration record.

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
report generation, and deterministic replay. The rehearsal must use a
disposable namespace and cannot contribute to any Phase D numerical choice.
Exact-head CI must pass before the tuning partition opens.

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

Each separate estimate therefore requires all 20 paired blocks. If its 18th
score is infinite, the fallback is deterministic:

- an infinite stop offset disables early resolved stopping; `stop_now` remains
  available only through the no-useful-measurement path and still requires
  verification;
- an infinite measurement-action offset makes that action unselectable in the
  primary rule while retaining it as a fixed comparator and as an explicitly
  failed region in the maps; and
- no family-specific or one-candidate-specific offset is estimated. Those
  strata are reported separately, avoiding an offset fitted from fewer than
  20 paired blocks.

These offsets are development heuristics for action selection. They are not
the independent Phase E calibration correction and cannot be described as a
coverage guarantee.

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

Choose the rule with the smallest mean paired-block loss under the already
declared balanced decision/resource loss: false approval 100, false rejection
20, abstention 1, added run 0.10, added sensor 0.10, and normalized incremental
energy 0.10. Average the three family losses within each block, then average
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

## Phase D5: build the measurement maps

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
continuations. Sequential scenario execution keeps the Phase C concurrent
memory cap at 185,204,736 bytes.

The planned four-block N=32 sensitivity adds the measured P4 continuation
budget of approximately 4.06 wall hours, 14.27 CPU hours, and 34,992,556
bytes. N=16 failure, N=64 comparison, an all-tuning N=32 extension, or a
replacement internal check is contingency work and must receive a new
machine-readable budget record before it runs. Reduce or omit secondary
sensor-map work before weakening the primary nominal development and internal
check.

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

The next concrete work is D0: implement the protocol, validator, commands, and
tests, then dry-run them in a disposable namespace. After exact-head CI passes,
generate only `p1_development_tuning`. Stop again after its archive validates
and before revealing/analysing its truth fields. Then perform D2-D5, commit the
locked provisional design, and only then open the internal check. Phase E may
start only after a separate Phase D freeze commit.
