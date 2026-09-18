# Prospective operating-decision partition ledger

Ledger version: 2. Date declared: 2026-09-17. Last reconciled: 2026-09-18.
Status: the first disposable pilot is complete and closed; its versioned
replacement is allocated but has not been generated. The repair revision has
its final validation record; the replacement may open only from its committed
and pushed form after CI passes. Development, calibration, and reserved
partitions remain unopened.

## Campaign identity

```text
campaign: operating_decision_prospective_v1_2026_09
procedure under development: prospective_four_action_selector_v2 with prospective uncertainty v3
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
| `p2_disposable_candidate_exclusion_pilot` | 4 paired blocks / 12 cases | Fresh repeat of the runtime and prefix-matched `N=4/8/16` pilot after separating candidate-level exclusion from whole-draw failure. Generate 16 draws for every case. If N=16 is the smallest passing prefix and P2 has zero pipeline failures and zero N=16 selection failures, generate N=32 for all same 12 cases and compare the authenticated N=16 prefix with N=32. An infeasible P2 fails without an N=32 run. | Offset or threshold fitting; calibration; confirmatory performance claims; favorable comparison with the exposed `p1` cases; choosing an N=32 subset after seeing P2. | Revised uncertainty and eligibility interfaces, regression tests, `p1` diagnosis, final validation record, and this namespace committed and pushed with passing CI before generation. |
| `p1_development_tuning` | 20 paired blocks / 60 cases | Fit development action/stop offsets; choose stopping clearance, value thresholds, instability allowance, and exact sensor-quality scenarios from the declared grid. | Independent checking, final calibration, or confirmatory claims. | Pilot accepts a draw count; its measured runtime, CPU, memory, and archive projections are reviewed and a compute budget is frozen in a separate commit. |
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
3. Treat uncertainty v3 and the strengthened P2/N32 validators as ordinary
   pre-P2 defect repairs, not the planned scientific redesign. Record final
   validation, commit and push the repairs and this ledger, and require passing
   CI; only then generate `p2_disposable_candidate_exclusion_pilot` once.
   Freeze draw count and the
   compute budget only if that replacement passes its declared gate. If its
   tested-prefix recommendation is N=16 and P2 has zero pipeline failures and
   zero N=16 selection failures, the conditional artifact
   `p2_disposable_candidate_exclusion_pilot_n32_all_cases_v1` must generate 32
   draws for blocks 0--3 and all three truth families. N=16 must agree with N=32
   in at least 90% of all 12 cases and have at most 5% maximum normalized
   utility regret, with zero N=32 pipeline or selection failures. An infeasible
   P2 fails without N=32. After the pilot gate passes, commit the accepted draw
   count and measured compute budget in a separate commit before any
   development partition opens.
4. Commit the development grid, then generate tuning and internal-check
   partitions in that order.
5. Freeze selector, scenarios, endpoints, comparison rules, sizes, and runtime.
6. Generate independent calibration and commit a finite calibration artifact.
7. Verify a disposable end-to-end replay, then open reserved evaluation once.

The pilot archive validators check deterministic record structure,
cross-record identities, exact stream inventory and RNG offsets, fit
invariants, and interval formulas. They do not rerun acquisition or predictive
simulations, candidate refits, or post-reveal scoring. A complete numerical
reproduction therefore still requires the bound source and retained raw
record; archive validation alone is not an independent recomputation of the
saved scientific calculations.

Ordinary fit failures, excluded candidates, failed verification, and
abstentions remain in their declared denominators. A material generator,
leakage, source, or truth-solver defect stops interpretation and requires a
versioned replacement campaign. Exposed reserved outcomes can never become a
tuning set for a rerun under the same reserved namespace.
