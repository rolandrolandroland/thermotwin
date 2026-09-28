# Prospective operating-decision Phase D3 result

Date: 2026-09-28. Status: complete. The fixed 81-rule grid was evaluated from
the authenticated Phase D1 evidence using the Phase D2 offsets. Phase D3
selected one provisional development rule and authorized the predeclared D4
draw-count sensitivity check. No simulation, refitting, or new partition was
used.

## Authenticated inputs and implementation

Phase D3 ran from clean source
`b0fd0632a8c94ac262029aa122ca2bd9fecd9103`. The analysis first replayed the
complete D1 and D2 validators, then reconstructed the padded action values,
selection, adjusted final decisions, resource costs, and loss for every rule
and case. The analysis source-manifest digest is
`8377b96b7c2468206821c4abf0e7040d2cda3fc18c3afbdd8a5b603093d47a1d`.

| Input | Recorded identity |
| --- | --- |
| Phase D1 JSON | SHA-256 `d83ae0e8ed52f328bf0a2accf5419dbd3a9ae10939552a14ccc0a445a1f30a2e`; scientific digest `efe860298104352fd0b6b35fc685e2320e6cfebd6db31fe8efc9876c509b1b5a` |
| Phase D2 JSON | SHA-256 `3b69f063de593a48af8430e94f802f43bf81dc02e0bec999b3b510ed143b0c2f`; scientific digest `242aa34e72cff260c7a16efc0ce8041928bebb63eeedc5dab88be86817d61b43` |
| Phase D3 JSON | 24,316,799 bytes; SHA-256 `8664fb237efba6854ed46ea600d9ad88d82117d795d108839c7a8deaeff69397` |
| Phase D3 scientific result | `dd349111b64722877a27536cbe5711fcffc8e91f0a2cb6c38f7c9337f1945c1b` |

The exact CPython 3.10.12 Phase D environment was used. The focused D1-D3
suite passed 37 tests. A strict clean-source replay returned `VALID`, the
detached JSON and report hashes passed, and an independent arithmetic script
reproduced all 4,860 case selections, the final decisions and losses, and the
winning rule.

The integration test also demonstrates that changing the minimum expected
reduction can change the selected action, that an action's development offset
can change its final decision, that the resulting loss changes, and that the
declared ranking selects the expected winner. It exercises the primary N=16
one-failure allowance rather than the zero-tolerance sensitivity rule.

## Selected provisional rule

The winning grid values, in the declared order, are:

| Parameter | Selected value |
| --- | ---: |
| General stopping clearance | 0.000 K |
| Additional one-candidate stopping clearance | 0.000 K |
| Minimum expected width reduction | 0.000 K |
| Minimum utility per normalized cost | 0.025 K |

Nine rows tied on mean paired-block loss, false approvals, false rejections,
definitive decisions, and mean realized diagnostic energy. They shared zero
general stopping clearance and 0.025 minimum utility. All combinations of
one-candidate clearance `0.00/0.05/0.10` and minimum reduction
`0.00/0.01/0.025` were tied. The predeclared lexicographic tie breaker selected
the values above. The tracked machine-readable rule is
`thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_D3_RULE.json`; its artifact
digest is `a5eaf1fa7f9f800e1b8938cac04bcb5bcd635f242d846d36c117c94f0ef38f66`.

## Development result

| Procedure | Definitive | Abstain | False approve | False reject | Adjusted interval coverage | Mean block loss | Mean energy (J) | Mean schedule (s) | Mean runs | Mean extra sensors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected rule | 32/60 (53.3%) | 28 | 0 | 0 | 60/60 | 0.6323 | 75.66 | 237.3 | 2.97 | 0.17 |
| Stop now | 25/60 (41.7%) | 35 | 0 | 0 | 60/60 | 0.5833 | 60.64 | 160.0 | 2.00 | 0.00 |
| Fixed voltage | 39/60 (65.0%) | 21 | 0 | 0 | 58/60 | 0.6500 | 89.40 | 240.0 | 3.00 | 1.00 |
| Fixed face temperature | 30/60 (50.0%) | 30 | 0 | 0 | 58/60 | 0.7996 | 89.29 | 240.0 | 3.00 | 1.00 |
| Fixed thermal | 27/60 (45.0%) | 33 | 0 | 0 | 59/60 | 0.9839 | 99.13 | 400.0 | 5.00 | 0.00 |

The selected rule chose stop now in 34 cases, thermal in 16, voltage in 5,
and face temperature in 5. It produced 17 approvals, 15 rejections, and 28
abstentions. Raw intervals covered 57/60 cases; development-adjusted intervals
covered 60/60. There were no selection, verification, pipeline, or
action-eligibility failures. The rule used 58 added diagnostic runs and 10
extra-sensor uses across the 60 cases.

| Development stratum | Definitive | False approvals | False rejections | Mean loss |
| --- | ---: | ---: | ---: | ---: |
| Matched four-state | 11/20 (55%) | 0 | 0 | 0.5017 |
| Extra interface mass | 10/20 (50%) | 0 | 0 | 0.7252 |
| Temperature-dependent contact | 11/20 (55%) | 0 | 0 | 0.6701 |
| One admissible candidate | 3/5 (60%) | 0 | 0 | 0.5200 |
| Two admissible candidates | 29/55 (52.7%) | 0 | 0 | 0.6425 |

## Interpretation and next gate

The selected rule increased development decision coverage over stop-now from
41.7% to 53.3%, while using more energy, time, runs, and instrumentation. Its
resource-weighted mean development loss, 0.6323, was therefore worse than
stop-now's 0.5833. It narrowly improved on fixed voltage's 0.6500 loss, while
fixed voltage made more definitive decisions. These are development results;
they do not establish calibration, transfer to new cases, or superiority over
a fixed policy.

Phase D4 is now authorized for tuning blocks `0`, `5`, `10`, and `15` at
N=32, using the frozen offsets and provisional rule. Before that computation,
the process-tree memory monitor must be tested synchronously and its permission
status recorded. A monitor failure stops the run before scientific work starts;
the completed D1 archive will not be regenerated. D5 sensor generation, the
internal check, independent calibration, and reserved evaluation remain
closed.
