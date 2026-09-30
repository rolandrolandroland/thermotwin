# Prospective operating-decision Phase D5 protocol

Date: 2026-09-29. Status: implementation frozen before execution. Phase D4
passed and authorized the one Phase D5 measurement-map analysis. No
internal-check, calibration, or reserved case is opened by this protocol.

## Purpose and inputs

Phase D5 maps the action chosen by the Phase D3 provisional selector across
the 12 predeclared cost cells and four predeclared sensor scenarios. It binds
the immutable Phase D1 archive, finite Phase D2 offsets, Phase D3 result and
rule freeze, and passing Phase D4 result. Every input is validated through its
source-bound chain before computation begins.

The selector remains fixed at general clearance `0.000 K`, additional
one-candidate clearance `0.000 K`, minimum expected reduction `0.000 K`, and
minimum utility per normalized cost `0.025 K`. Its stop, thermal, voltage, and
face-temperature development offsets remain `0.000/0.000/0.074/0.098 K`.
N=16 and one allowed whole-draw failure per source/action remain frozen.

## Map design

The complete map has 48 cells: every combination of four sensor scenarios and
12 cost scenarios. The cost scenarios cross balanced, energy-dominant,
bench-time-dominant, and instrumentation-dominant weights with face-probe
instrumentation priced at one quarter, equal to, or four times the reference
instrument. `nominal` with `balanced_face_equal` remains the sole primary
cell.

The nominal sensor scenario reuses the authenticated Phase D1 N=16 predictions
and fixed-policy outcomes. Cost-only changes recompute action resource costs,
padded utility per cost, the selected action, and the routed saved outcome;
they do not simulate, refit, or open a partition.

The three stress scenarios replay all 20 paired tuning blocks:

| Scenario | Frozen change |
| --- | --- |
| `high_voltage_noise` | Voltage noise changes from `0.002 V` to `0.004 V`. |
| `high_face_noise` | Face-temperature noise changes from `0.02 K` to `0.04 K`. |
| `high_probe_loading` | Probe capacitance changes from `5 J/K` to `10 J/K`; response time remains `2.5 s`. |

Each stress scenario performs a complete replay, which is the conservative
option authorized by the Phase D plan. The altered observation noise is used
by both generation and likelihood evaluation. The high-loading capacitance is
the nominal value for both the generated probe truth and the fit. The
run-bias/noise ratio remains 2.5. All three scenarios reuse the same semantic
campaign, partition, block, and purpose streams as a paired sensitivity; their
scenario and physical-configuration identities remain explicit in every block
archive.

## Required map outputs

Each cell retains one record per case and reports:

- selected-action counts and rates for stop, thermal, voltage,
  face-temperature, and selection failure;
- eventual definitive-decision counts overall and within each selected action;
- action-ineligibility counts;
- selection, verification, and pipeline failures; and
- the no-useful-measurement count.

`no_useful_measurement` has one exact meaning: the selector returned
`no_valuable_acquisition_stop_then_verify`. A selected measurement that later
abstains stays under its selected action and is not merged with a successful
definitive decision. Truth labels and saved outcomes enter only this offline
development map after the selection inputs and all four fixed-policy outcomes
have been saved.

## Execution, persistence, and validation

The implementation is in
`thermotwin/studies/operating_decision_prospective_phase_d5_maps.py`; its
command-line entry point is
`thermotwin/reports/operating_decision_prospective_phase_d5.py`. Exact-head CI
must pass before execution.

Stress scenarios run sequentially, with four workers inside each scenario.
Every complete block is validated and written atomically before the next block
is accepted. A restart may reuse only a block whose source manifest, Phase D5
protocol digest, scenario, physical configuration, block index, scientific
digest, serialized bytes, and archive digest all validate. Failed blocks
remain as canonical pipeline failures and contribute three selection-failure
cases; they are never dropped.

Immediately before computation, the executor synchronously checks process-tree
visibility. It then monitors aggregate coordinator-and-worker RSS through
generation, block persistence, final serialization, reload, and validation.
The predeclared Phase D budget is approximately 29.90 wall hours, 109.84 CPU
hours, and 230,896,830 archive bytes for the three stress scenarios. The 4 GiB
process-tree memory limit remains in force.

The final archive must contain exactly 48 recomputable map cells, identities
for every input and stress archive, a complete source manifest, a scientific
result digest, detached file hashes, and a resource record. The internal check
can open only after this archive validates and a separate commit freezes the
complete provisional design.
