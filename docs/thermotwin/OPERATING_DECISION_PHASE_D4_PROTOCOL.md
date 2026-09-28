# Prospective operating-decision Phase D4 protocol

Date: 2026-09-28. Status: implementation complete; scientific execution not
yet opened. This protocol binds the one authorized draw-count sensitivity
continuation before its N=32 evidence is generated.

## Inputs and scope

Phase D4 uses the preserved Phase D1 tuning evidence, the validated Phase D3
result, and the tracked Phase D3 provisional-rule freeze. It generates only
tuning blocks `0`, `5`, `10`, and `15`, with all three truth families in each
block. No block may be replaced after seeing its outcome.

| Field | Frozen value |
| --- | --- |
| Campaign and partition | `operating_decision_prospective_v1_2026_09` / `p1_development_tuning` |
| Cases | 12: four paired blocks by three truth families |
| Candidate/reference draw count | N=16 / N=32 |
| Whole-draw allowance | At most one per source/action at both counts |
| Selector | Phase D3 values `(0.00, 0.00, 0.00, 0.025)` with D2 offsets `0.000/0.000/0.074/0.098 K` |
| Cost cell | `balanced_face_equal` |
| Parallel workers | Four |

Every generated N=32 case must reproduce the complete preserved Phase D1 N=16
uncertainty payload as its exact authenticated prefix. The validator also
recomputes both scorecards from their raw evidence under the Phase D3 rule and
checks the corrected and prospective random-stream audits.

## Acceptance gate

N=16 remains accepted only if all of these conditions hold:

1. at least 11 of the 12 selected actions agree with N=32;
2. every changed choice has evaluable normalized utility regret;
3. the maximum changed-choice regret is at most 5%;
4. thermal, voltage, and face-temperature actions are eligible at both draw
   counts in every case; and
5. there is no pipeline, selection, provenance, or draw-diagnostic failure.

Failed draws remain in the denominator and use the existing no-gain score.
Candidate exclusions remain candidate-local. Truth labels are used only to
ensure the predeclared family matrix is complete; no accuracy or coverage
claim is estimated in D4.

If the gate passes, N=16 stays frozen and Phase D5 map construction is
authorized. If it fails, execution stops. Before any contingency computation,
the project must commit a new budget for N=32 on the remaining 16 tuning
blocks and N=64 on the same four-block subset. Thresholds, offsets, costs,
physics, and subset membership cannot change during that comparison.

## Resource and archive controls

The required process-tree monitor preflight first failed inside the workspace
sandbox because access to `ps` was denied. No scientific work had started. The
same child-worker probe then passed with approved local process-table access:
14 monitor samples, 64,012,288-byte peak RSS, no sampling error, and a clean
child exit. Both attempts are retained in
`thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_D4_MONITOR_PREFLIGHT.json`.

Immediately before scientific execution, the command performs another
synchronous process-tree sample. Failure stops execution before simulation.
The monitor then remains active through generation, atomic block persistence,
final serialization, reload, and validation. Complete block files may be
resumed only when their source, protocol, input, stream, and content identities
validate. The final archive retains all 12 complete N=32 cases, both rescored
prefixes, every comparison, failures, block files, performance fields, and
detached hashes.

The implementation is in
`thermotwin/studies/operating_decision_prospective_phase_d4_sensitivity.py`,
with its command-line interface in
`thermotwin/reports/operating_decision_prospective_phase_d4.py`. Exact-head CI
must pass before the continuation opens.
