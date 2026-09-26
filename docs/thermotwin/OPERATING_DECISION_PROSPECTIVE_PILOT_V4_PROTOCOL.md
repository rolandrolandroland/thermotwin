# Prospective disposable pilot v4 protocol

Date frozen for review: 2026-09-25. Result reconciled: 2026-09-26. Status: P4
and its required all-case N32 continuation completed and passed. See the
[`P4 result`](OPERATING_DECISION_PROSPECTIVE_PILOT_V4_RESULT.md).

## Purpose

P3 and its conditional N=32 continuation showed that a zero-failure
eligibility cutoff was too discontinuous for the prospective Monte Carlo
calculation. One face-temperature draw out of 32 had no admissible candidate.
The draw was correctly retained and scored at the no-gain baseline, but the
zero-failure cutoff then discarded the entire action and changed one of 12
choices. P3 is motivation for this bounded redesign; it is not evidence that
the redesigned rule will pass.

P4 is the single bounded scientific redesign permitted by the completion plan.
If its fresh engineering gate fails, the large prospective campaign stops and
the result is reported as a feasibility limitation.

## Fresh identity

```text
campaign: operating_decision_prospective_v1_2026_09
partition: p4_disposable_bounded_instability_pilot
parent protocol: operating_decision_prospective_draw_count_pilot_v5
conditional N32 protocol: operating_decision_prospective_n32_followup_v3
conditional N32 artifact: p4_disposable_bounded_instability_pilot_n32_all_cases_v1
```

The P1, P2, and P3 namespaces remain closed. P4 uses four fresh paired blocks,
one case from each of the three truth families per block. Development,
calibration, and reserved namespaces remain unopened.

## Eligibility rule

The maximum retained whole-draw failures per source model and measurement
action is frozen as:

| Predictive draws | Maximum whole-draw failures |
| ---: | ---: |
| 4 | 0 |
| 8 | 0 |
| 16 | 1 |
| 32, only if triggered | 1 |

Every declared draw stays in the denominator. A failed draw receives the
pre-action width, which is no gain. Candidate loss also keeps the no-gain
floor and remains separate from a whole-draw failure when another candidate
supplies a usable interval. The allowance changes eligibility only; it does
not delete failures, replace their scores, or count them as information gain.

All thermal, voltage, and face-temperature actions must be eligible in every
case at the reference draw count. A selection failure, an ineligible reference
action, or a case-level pipeline failure fails the engineering gate.

The same N=16 evidence is also rescored with zero allowed failures as a named
sensitivity analysis. This sensitivity cannot replace the primary rule or
choose the result after outcomes are seen.

## Parent pilot and conditional continuation

Each P4 case generates 16 predictive draws once. Authenticated prefixes at
N=4, N=8, and N=16 are compared against the N=16 reference. The pilot chooses
the smallest tested prefix with at least 90% action agreement and at most 5%
maximum normalized utility regret among changed choices. Stop, selection
failure, and eligibility changes remain part of the comparison.

If N=16 is the smallest passing prefix, the conditional N=32 continuation is
required only when the parent has:

- zero pipeline failures;
- zero N=16 selection failures; and
- zero ineligible N=16 measurement actions.

The continuation includes all 12 P4 cases. It generates 32 draws once per case
and compares its authenticated N=16 prefix with N=32. It passes only if the
same agreement and regret limits hold, there are no pipeline or selection
failures, every N=32 measurement action stays eligible within the one-failure
allowance, and every required diagnostic is available.

Neither a passing parent nor a passing continuation opens Phase D directly. A
separate commit must freeze the chosen draw count and measured compute budget.

## Chronology and reproducibility

The protocol digest binds the complete allowance table, zero-failure
sensitivity, source revision, partition plan, physical protocol, uncertainty
protocol, cost model, selector rule, acceptance rule, and conditional N=32
design. Saved JSON must pass load-and-validate replay before its gate is
interpreted.

Historical P3 artifacts remain bound to source `2f906a2`. Replaying those
archives requires that source revision; the P4 validator is intentionally a
new schema and protocol identity.

The protocol was committed before P4 opened. The later result record preserves
the chronology, hashes, acceptance replay, and separate Phase C freeze without
altering this predeclared rule.
