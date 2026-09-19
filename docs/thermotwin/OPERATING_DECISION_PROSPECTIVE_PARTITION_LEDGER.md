# Prospective operating-decision partition ledger

Ledger version: 3. Date declared: 2026-09-17. Last reconciled: 2026-09-19.
Status: P1 is complete and closed. P2 executed once, but its saved JSON failed
the required load-and-validate round trip; it is preserved as invalid incident
evidence and its gate was not evaluated. P2's N32 continuation was never opened
and is retired. P3 and its conditional N32 identity are allocated but unopened.
Their representation repair is committed at `8c232af`. Final local validation
passed under CPython 3.10.12 as recorded in the
[Phase B acceptance record](OPERATING_DECISION_PHASE_B_ACCEPTANCE.md).
Independent clean-clone review also passed. This audit closeout is recorded in
this commit; the commit must be pushed and its exact HEAD must pass CI.
Development, calibration, and reserved partitions remain unopened.

## Campaign identity

```text
campaign: operating_decision_prospective_v1_2026_09
procedure under development: prospective_four_action_selector_v2 with prospective uncertainty v3
next disposable protocol: draw-count pilot v4 with N32 follow-up v2
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
| `p3_disposable_archive_roundtrip_replacement_pilot` | 4 paired blocks / 12 cases | Fresh repeat of the unchanged runtime and prefix-matched `N=4/8/16` pilot after repairing final-byte JSON round-trip validation. Generate 16 draws for every case. | Offset or threshold fitting; calibration; confirmatory performance claims; use of quarantined P2 outcome fields; rerunning under P1 or P2 identities. | Local validation of pilot protocol v4, the JSON-native payload repair, and final-byte save/load validation is recorded. This ledger and incident response must be independently reviewed, committed, and pushed with green CI before generation. |
| `p3_disposable_archive_roundtrip_replacement_pilot_n32_all_cases_v1` | Conditional 4 paired blocks / 12 cases at N=32 | If and only if a valid P3 archive has zero pipeline and N=16 selection failures and N=16 is its smallest passing prefix, generate N=32 for the same four blocks and all three truth families and compare the authenticated N=16 prefix with N=32 under follow-up protocol v2. | Opening from P2; choosing a subset after seeing P3; offset or threshold fitting; calibration; confirmatory claims. | Valid complete P3 parent archive independently loaded and validated; frozen trigger satisfied exactly; P3 conditional identity and follow-up protocol v2 committed before P3 opens. |
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
3. Generate `p2_disposable_candidate_exclusion_pilot` once at source
   `9db5f3f`. Preserve its hashes, but stop interpretation when the saved JSON
   fails the required load-and-validate round trip. Record the incident, close
   P2 without evaluating its gate, and retire its unopened N32 continuation.
4. Repair commit `8c232af` changes only the representation boundary: it makes
   configuration payloads JSON-native, requires final P3 and N32 bytes to pass
   save, load, and archive validation, and versions pilot protocol v4,
   N32-follow-up protocol v2, the P3 namespace, and this ledger. Local validation
   passed under CPython 3.10.12, and independent clean-clone review passed.
   Push this audit closeout commit, then require green CI on exact HEAD. Those
   remaining gates are pending.
5. Generate `p3_disposable_archive_roundtrip_replacement_pilot` once. If a
   valid P3 archive triggers the conditional all-case N32 identity, complete
   that follow-up before evaluating the combined pilot gate. Only after a valid
   gate passes may a separate commit freeze the accepted draw count and measured
   compute budget before any development partition opens.
6. Commit the development grid, then generate tuning and internal-check
   partitions in that order.
7. Freeze selector, scenarios, endpoints, comparison rules, sizes, and runtime.
8. Generate independent calibration and commit a finite calibration artifact.
9. Verify a disposable end-to-end replay, then open reserved evaluation once.

The pre-P2 archive tests checked deterministic record structure, cross-record
identities, exact stream inventory and RNG offsets, fit invariants, and interval
formulas in memory. P2 exposed the missing serialized-byte round trip. P3 must
add that boundary before any new evidence opens. Archive validation still will
not rerun acquisition or predictive simulations, candidate refits, or
post-reveal scoring; a complete numerical reproduction requires the bound
source and retained raw record.

Ordinary fit failures, excluded candidates, failed verification, and
abstentions remain in their declared denominators. A material generator,
leakage, source, or truth-solver defect stops interpretation and requires a
versioned replacement campaign. Exposed reserved outcomes can never become a
tuning set for a rerun under the same reserved namespace.
