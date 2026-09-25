# Prospective operating-decision partition ledger

Ledger version: 4. Date declared: 2026-09-17. Last reconciled: 2026-09-25.
Status: P1 is complete and failed. P2 is closed as invalid incident evidence.
P3 and its conditional all-case N32 continuation executed at source `2f906a2`;
both archives validate, and the combined engineering gate failed. No draw count
or compute budget is frozen. Development, calibration, and reserved partitions
remain unopened. P4 is the one bounded eligibility redesign. Its source,
protocol, and fresh disposable namespace are predeclared here, but no P4 data
have been generated.

## Campaign identity

```text
campaign: operating_decision_prospective_v1_2026_09
procedure under development: prospective_four_action_selector_v2 with prospective uncertainty v3
next disposable protocol: draw-count pilot v5 with N32 follow-up v3
reviewed source baseline: 9a21aa7
```

The campaign name and partition name are both fields in every prospective
semantic random-stream key. Reusing a block number in another partition does
not reuse a stream because the full key includes both fields. These namespaces
are disjoint from the completed
`operating_decision_audit_replication_corrected_v2_2026_09` campaign, all
abandoned correction campaigns, and disposable namespaces used by unit tests.

One paired block contains one device from each of the three truth families, so
one block produces three cases. All procedures applied to a case share only the
observations explicitly declared common by the protocol.

## Allocated partitions

| Partition | Planned size | Permitted use | Prohibited use | Reveal gate |
| --- | ---: | --- | --- | --- |
| `p1_disposable_draw_count_pilot` | 4 paired blocks / 12 cases, complete | Runtime, memory, failure, eligibility, and prefix-matched `N=4/8/16` action-stability measurement under prospective uncertainty v2. | Offset or threshold fitting; calibration; confirmatory performance claims; reuse as acceptance evidence for the revised rule. | Opened once at source `e32d091`; closed after the engineering gate failed. |
| `p2_disposable_candidate_exclusion_pilot` | 4 paired blocks / 12 cases, execution complete, archive invalid | Diagnose the saved-JSON round-trip incident only. Preserve its exact bytes and hashes as identified in the [P2 incident record](OPERATING_DECISION_PROSPECTIVE_PILOT_V2_INCIDENT.md). | Draw-count or compute-budget selection; N32 triggering; action-stability, feasibility, calibration, development, or confirmatory claims; outcome-directed changes to P3; rerun or overwrite. | Opened once at source `9db5f3f`; closed after archive validation failed. Gate not evaluated. |
| `p3_disposable_archive_roundtrip_replacement_pilot` | 4 paired blocks / 12 cases, complete and valid | Fresh repeat of the unchanged runtime and prefix-matched `N=4/8/16` pilot after repairing final-byte JSON round-trip validation. | Offset or threshold fitting; calibration; confirmatory performance claims; rerunning or overwriting. | Opened once at source `2f906a2`; N=16 triggered the conditional continuation. Closed. |
| `p3_disposable_archive_roundtrip_replacement_pilot_n32_all_cases_v1` | 4 paired blocks / 12 cases at N=32, complete and valid | Compare the authenticated N=16 prefix with N=32 after the frozen trigger passed. | Choosing a subset; offset or threshold fitting; calibration; confirmatory claims; rerunning or overwriting. | Opened once at source `2f906a2`; combined gate failed because one eligibility-driven choice change had unevaluable regret. Closed. |
| `p4_disposable_bounded_instability_pilot` | 4 paired blocks / 12 cases, allocated and unopened | Test authenticated `N=4/8/16` prefixes under the frozen whole-draw allowances `0/0/1`, while retaining every failed draw at the no-gain baseline. | Offset or threshold fitting; calibration; confirmatory performance claims; using P3 outcomes as validation; rerunning or overwriting after P4 opens. | May open once only after the P4 source and protocol commit passes exact-HEAD CI. Every N=16 measurement action must remain eligible. |
| `p4_disposable_bounded_instability_pilot_n32_all_cases_v1` | Conditional 4 paired blocks / 12 cases at N=32, allocated and unopened | Compare the authenticated N=16 prefix with N=32 under one allowed whole-draw failure per source/action. | Opening unless N=16 is the smallest passing P4 prefix with zero pipeline failures, selection failures, and ineligible N=16 actions; choosing a subset; calibration or confirmatory claims. | Opens once only if the authenticated P4 parent trigger passes. Every N=32 measurement action must remain eligible. |
| `p1_development_tuning` | 20 paired blocks / 60 cases | Fit development action/stop offsets; choose stopping clearance, value thresholds, and exact sensor-quality scenarios from the declared grid under the P4-frozen eligibility rule. | Changing the draw count or instability allowance; independent checking, final calibration, or confirmatory claims. | Pilot accepts a draw count; its measured runtime, CPU, memory, and archive projections are reviewed and a compute budget is frozen in a separate commit. |
| `p1_development_internal_check` | 10 paired blocks / 30 cases | One internal check of the design selected on `p1_development_tuning`; draw-count sensitivity on a predetermined subset. | Final calibration or reserved claims. If its labels cause a redesign, it becomes tuning evidence and a new check namespace must be declared before generation. | Tuning design and check analysis committed before reveal. |
| `p1_independent_calibration` | 100 paired blocks / 300 cases | Compute only the frozen procedure-level interval correction and the same declared correction for each fixed comparator; retain failed or missing intervals as infinite scores. | Selector, offset, stop, threshold, sensor-scenario, endpoint, or sample-size tuning. | Complete design and analysis specification committed; size and calibration rank verified before generation. |
| `p1_reserved_evaluation` | 100 paired blocks / 300 cases | One final paired comparison of the frozen selector with stop, fixed thermal, fixed voltage, and fixed face temperature in the primary scenario. | Any tuning, recalibration, favorable-case selection, or reactive sample-size extension. | Finite calibration artifact, source/environment manifests, partition identities, comparison rules, and disposable end-to-end replay committed and verified. |

The 100-block calibration and reserved sizes are the initial compute/precision
plan, not evidence that the intended risk comparison is adequately powered.
Phase C and development approval/violation rates must check that question. A
size may change only through a reviewed, committed new ledger version before
that partition is generated. Once a partition is opened, its size cannot be
changed in response to its outcomes.

## Information permissions

- Pilot outcomes are engineering evidence and may choose draw count and compute
  strategy only.
- Development truth labels may be used offline for development offsets,
  thresholds, family diagnostics, and sensor-quality sensitivity. The selector
  itself receives acquisition-only fields.
- Calibration truth may be used only in the frozen calibration score. It cannot
  change action selection, stopping, fitted parameters, or development offsets.
- Reserved truth and the untouched final trajectory remain hidden until each
  procedure's decision is saved. They may be used only for the frozen analysis.
- The completed corrected-replication labels and all earlier Stage 3–5 labels
  are historical evidence and cannot tune this campaign.

## Chronology and incident rule

1. Commit Phase B source, protocol identities, tests, and acceptance record.
2. Generate `p1_disposable_draw_count_pilot`. Its all-or-action eligibility
   rule failed because ordinary candidate exclusions also invalidated the
   action. Preserve that result as engineering evidence.
3. Generate `p2_disposable_candidate_exclusion_pilot` once at source
   `9db5f3f`. Preserve its hashes, but stop interpretation when the saved JSON
   fails the required load-and-validate round trip. Record the incident, close
   P2 without evaluating its gate, and retire its unopened N32 continuation.
4. Repair commit `8c232af` changes only the representation boundary: it makes
   configuration payloads JSON-native, requires final P3 and N32 bytes to pass
   save, load, and archive validation, and versions pilot protocol v4,
   N32-follow-up protocol v2, the P3 namespace, and this ledger. Local validation
   passed under CPython 3.10.12, and independent clean-clone review passed.
   The audit closeout at `2f906a2` passed exact-HEAD CI.
5. P3 and its triggered all-case N32 continuation executed once at `2f906a2`.
   Both archives validate, and the combined gate failed. Preserve their hashes
   and result record without rerunning or overwriting either namespace.
6. Version the one permitted bounded eligibility redesign as P4. Freeze
   whole-draw allowances at 0 for N=4, 0 for N=8, 1 for N=16, and 1 for a
   triggered N=32 continuation. Failed draws remain in the denominator and are
   scored as no gain; every measurement action must remain eligible at the
   reference draw count. The complete design is in the
   [P4 protocol](OPERATING_DECISION_PROSPECTIVE_PILOT_V4_PROTOCOL.md).
7. After the P4 source/protocol commit passes exact-HEAD CI, open the P4 parent
   once. Open its all-case N32 continuation only if the authenticated parent
   trigger passes. Only after the complete P4 gate passes may a separate commit
   freeze a draw count and measured compute budget. If it fails, stop the large
   campaign and complete the feasibility report.
8. Commit the development grid, then generate tuning and internal-check
   partitions in that order.
9. Freeze selector, scenarios, endpoints, comparison rules, sizes, and runtime.
10. Generate independent calibration and commit a finite calibration artifact.
11. Verify a disposable end-to-end replay, then open reserved evaluation once.

The pre-P2 archive tests checked deterministic record structure, cross-record
identities, exact stream inventory and RNG offsets, fit invariants, and interval
formulas in memory. P2 exposed the missing serialized-byte round trip. P3
closed that boundary and both of its saved archives replay successfully.
Archive validation still will not rerun acquisition or predictive simulations,
candidate refits, or post-reveal scoring; a complete numerical reproduction
requires the bound source and retained raw record.

Ordinary fit failures, excluded candidates, failed verification, and
abstentions remain in their declared denominators. A material generator,
leakage, source, or truth-solver defect stops interpretation and requires a
versioned replacement campaign. Exposed reserved outcomes can never become a
tuning set for a rerun under the same reserved namespace.
