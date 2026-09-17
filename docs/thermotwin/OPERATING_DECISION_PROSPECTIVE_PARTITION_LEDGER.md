# Prospective operating-decision partition ledger

Ledger version: 1. Date declared: 2026-09-17. Status: namespaces allocated;
no listed partition has been generated or opened.

## Campaign identity

```text
campaign: operating_decision_prospective_v1_2026_09
procedure under development: prospective_four_action_selector_v2
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
| `p1_disposable_draw_count_pilot` | 4 paired blocks / 12 cases | Runtime, memory, failure, eligibility, and prefix-matched `N=4/8/16` action-stability measurement. Generate 16 draws for every case; a predesignated `N=32` subset may be added only under the protocol's stated trigger. | Offset or threshold fitting; calibration; confirmatory performance claims. | Phase B interfaces and acceptance checks committed. |
| `p1_development_tuning` | 20 paired blocks / 60 cases | Fit development action/stop offsets; choose stopping clearance, value thresholds, instability allowance, and exact sensor-quality scenarios from the declared grid. | Independent checking, final calibration, or confirmatory claims. | Pilot accepts a draw count and compute budget. |
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
2. Generate the disposable pilot once and freeze draw count and compute budget.
3. Commit the development grid, then generate tuning and internal-check
   partitions in that order.
4. Freeze selector, scenarios, endpoints, comparison rules, sizes, and runtime.
5. Generate independent calibration and commit a finite calibration artifact.
6. Verify a disposable end-to-end replay, then open reserved evaluation once.

Ordinary fit failures, excluded candidates, failed verification, and
abstentions remain in their declared denominators. A material generator,
leakage, source, or truth-solver defect stops interpretation and requires a
versioned replacement campaign. Exposed reserved outcomes can never become a
tuning set for a rerun under the same reserved namespace.
