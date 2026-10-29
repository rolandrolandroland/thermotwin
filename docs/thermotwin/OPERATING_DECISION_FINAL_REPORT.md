# ThermoTwin operating decision experiment

Final report by Roland Bennett. 29 October 2026. Synthetic four-action prospective selector study. Numerical evaluation completed on 21 October 2026. The accepted reserved cohort is closed.

### Conclusion

The selector reduced simulated diagnostic energy relative to each fixed measurement policy, but primary success was not established. It made 163 of 300 definitive decisions (54.3%), below the 70% target even before confidence adjustment. One false approval occurred among 104 approvals. Conservative simultaneous bounds did not establish the separate approval-risk target. No procedure qualified on both decision quality requirements, and no operating policy is promoted for deployment.

This experiment asks whether software can choose the next useful measurement before committing to an operating decision. A verified thermal model estimates a margin to a declared temperature band. The available actions are stop, three additional thermal runs, voltage sensing, or temporary face-temperature sensing. All decisions are saved before the final target truth is revealed.

| Procedure | Definitive / 300 | False approvals / approvals | Mean energy J |
| --- | --- | --- | --- |
| Selector | 163 | 1/104 | 75.03 |
| Stop | 118 | 0/67 | 60.65 |
| Thermal | 171 | 0/101 | 99.16 |
| Voltage | 182 | 0/129 | 89.43 |
| Face | 165 | 1/112 | 89.32 |

### Traceability to the seven original goals

| Goal | Completed evidence |
| --- | --- |
| 1 Four actions | Frozen prospective selector and authenticated acquisition boundary |
| 2 Expected uncertainty reduction | 16 local covariance predictive draws per surviving source model |
| 3 Value per frozen cost | Normalized energy, time and instrumentation costs |
| 4 Measurement map | 48 predeclared development sensor and cost cells |
| 5 New calibration partitions | 20 tuning blocks, 10 internal checks and 100 calibration blocks |
| 6 Freeze before reserved opening | Complete source and rule frozen before 100 reserved blocks |
| 7 Report outcomes and resources | Independent reconstruction, replay, figures and this closeout |

The practical result is a demonstrated software workflow and a measured usefulness limit. Hardware performance, financial savings, manufacturing capability and reliability outside the declared synthetic generator remain untested.

## Methods and information boundary

The physical benchmark models a thermoelectric assembly with thermal contacts, heat capacities, exchanger temperatures, sensor lag and electrical series resistance. Three paired truth variants share a device parameter draw: a matched four-state model, an extra interface mass, and temperature-dependent contact. Only the four-state and five-state candidates are fitted. The contact-law family is deliberately absent from the candidate menu.

The operating margin is the minimum distance of the complete cold-face target trajectory to either edge of the 285.0 to 301.7 K band. Negative margin means a violation. At primary clearance zero, approve when the final interval lower endpoint is at least zero, reject when the upper endpoint is strictly negative, and otherwise abstain. A positive margin estimate alone is insufficient for approval.

| Action | Added acquisition runs | Added channel | Nominal incremental J |
| --- | --- | --- | --- |
| Stop | 0 | None | 0 |
| Thermal | 3 | None | 38.5717 |
| Voltage | 1 | Terminal voltage | 28.8143 |
| Face | 1 | Temporary face temperature | 28.6909 |

Every procedure starts with the same initial temperature acquisition and receives the mandatory independent verification run. Stop therefore still uses two diagnostic runs. Thermal uses five runs in total. Voltage and face use three. The face probe changes the measured system through thermal loading and finite response, and the final unloaded operating forecast accounts for its removal.

### Prospective calculation

Initial fits and their local covariance generate hypothetical future observations, never the hidden device truth. Physical parameter draws are explicitly shared across the three actions as common random numbers. Probe nuisance is action specific. Each draw refits the candidate models and propagates uncertainty to the final margin. Means are taken within a source model, then the worst mean across surviving source models is used without model probabilities. This is a local covariance approximation, not a posterior guarantee.

The scoring baseline is the common pre-action envelope width. Development padding is included in expected post-action width. Negative reduction is retained. Failed draws stay in the denominator with no gain. Candidate attrition cannot create artificial gain. Bound hits, nonconvergence and candidate-specific failures exclude that candidate when another admissible candidate remains. Convergence uses a projected KKT gradient. Loss of every usable candidate remains a failure.

The frozen rule uses N=16 and permits one whole-draw failure per source and action. Minimum value per normalized cost is 0.025 K. Primary costs are 1.44621 for thermal, 1.00000 for voltage and 0.998573 for face, using equal weights. Stop stays outside the ratio. Nominal energy, bench-time assumptions and instrument burden are separate inputs, and the costs are dimensionless sensitivity units rather than prices in dollars.

The acquisition evidence factory internally fits the exact common initial run. Semantic random-stream keys bind campaign, partition, block, evidence, model, draw, purpose, action, run and channel. Only declared sharing is allowed. Target outcomes, truth-family labels, verification responses and realized action outcomes cannot enter prospective scoring.

## Development calibration and scientific freeze

Development used 20 fresh paired tuning blocks followed by 10 internal-check blocks. D2 produced development offsets, D3 selected one rule from 81 declared rows, D4 checked a predetermined subset against N=32, and D5 computed the full 48-cell map. D6 returned 24/30 definitive decisions with zero observed decision errors and no redesign trigger. Its development-padded intervals covered 27/30 cases and 7/10 complete blocks. These are development findings, not reserved validation.

The complete design was frozen before independent calibration. Each procedure was calibrated on 100 blocks, each containing all three families. A case score is the nonnegative amount by which the true margin lies outside the development-padded interval. The maximum score across the three families is the block score. Failures retain infinite scores. The final correction q is the 96th smallest block score. All five corrections were finite.

| Procedure | Development padding d K | Final correction q K |
| --- | --- | --- |
| Selector | Selected action d | 0.02543643 |
| Stop | 0.000 | 0.04965714 |
| Thermal | 0.000 | 0.01041381 |
| Voltage | 0.074 | 0.04106348 |
| Face | 0.098 | 0.02543643 |

Final endpoints are expanded by d plus q with outward rounding. For the selector, q belongs to the entire frozen procedure, including selection and verification. It does not change the action scores or stopping decision. Under the declared independent identically distributed block model, the rank targets 90% simultaneous-within-block coverage with at least 95% calibration confidence. The achieved discrete confidence is 97.6289% per procedure. This confidence is not joint across procedures and does not imply a bound on errors conditional on approval.

| Boundary | Recorded state |
| --- | --- |
| Development design | Closed through D6 at source 75f598a |
| Calibration source | 23a076f with pinned CPython 3.10.12 environment |
| Calibration completion | 100 blocks completed 16 October 2026 |
| Reserved numerical source | 7e3b9a3 with 117 frozen source dependencies |
| Reserved opening gate | Committed gate 1eaf547 preceded generation |
| Reserved completion | 100 blocks and independent validation on 21 October |
| Result and replay checks | Independent arithmetic and bounded replay on 29 October |

Calibration survived recorded process interruptions using authenticated completed checkpoints under the identical source and runtime. Successful cases were not replaced. Reserved execution completed once without resumed or replacement blocks. No exposed cohort was retuned, regenerated or enlarged. Historical disposable incidents and repairs remain documented separately in the project status and incident records.

## All primary outcomes and family diagnostics

The reserved experiment contains 100 independent paired blocks, 300 cases per procedure and 1,500 primary outcomes. There were 176 true passes and 124 true violations. Abstention is not a correct definitive decision. Missed violations include true violations that did not produce a rejection, including abstentions. All five procedures had zero false rejections, zero pipeline or verification failures and no missing final interval.

| Procedure | Decisions | Abstain | Approve | Reject | FA / appr. | Missed / 124 | Intervals / 300 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Selector | 163 | 137 | 104 | 59 | 1/104 | 65 | 293 |
| Stop | 118 | 182 | 67 | 51 | 0/67 | 73 | 299 |
| Thermal | 171 | 129 | 101 | 70 | 0/101 | 54 | 294 |
| Voltage | 182 | 118 | 129 | 53 | 0/129 | 71 | 300 |
| Face | 165 | 135 | 112 | 53 | 1/112 | 71 | 296 |

### Family results with frozen simultaneous bounds

| Procedure | Family | Decisions / 100 | FA / approvals | Coverage LB % | Risk UB % | Intervals / 100 |
| --- | --- | --- | --- | --- | --- | --- |
| Selector | Matched | 59 | 0/19 | 42.57 | 31.85 | 99 |
| Selector | Interface | 55 | 0/44 | 38.68 | 15.26 | 99 |
| Selector | Contact | 49 | 1/41 | 33.04 | 21.21 | 95 |
| Stop | Matched | 52 | 0/15 | 35.84 | 38.48 | 100 |
| Stop | Interface | 46 | 0/38 | 30.31 | 17.45 | 100 |
| Stop | Contact | 20 | 0/14 | 9.17 | 40.57 | 99 |
| Thermal | Matched | 73 | 0/26 | 57.00 | 24.44 | 97 |
| Thermal | Interface | 54 | 0/42 | 37.73 | 15.93 | 98 |
| Thermal | Contact | 44 | 0/33 | 28.51 | 19.81 | 99 |
| Voltage | Matched | 65 | 0/20 | 48.58 | 30.53 | 100 |
| Voltage | Interface | 59 | 0/52 | 42.57 | 13.07 | 100 |
| Voltage | Contact | 58 | 0/57 | 41.59 | 12.00 | 100 |
| Face | Matched | 62 | 0/18 | 45.55 | 33.29 | 100 |
| Face | Interface | 48 | 0/42 | 32.13 | 15.93 | 100 |
| Face | Contact | 55 | 1/52 | 38.68 | 17.09 | 96 |

LB and UB denote lower and upper confidence bounds. Equal-family decisiveness uses the mean of the three family lower bounds. Conditional approval risk uses the maximum family upper bound because population approval weights are unknown. The 300 dependent cases are not treated as 300 independent trials. Family C contact mismatch contains the selector false approval, while the widest selector risk bound arises from the matched family with only 19 approvals.

Observed complete-block interval coverage was 93/100 for the selector, 99/100 for stop, 94/100 for thermal, 100/100 for voltage and 96/100 for face. High interval coverage and sparse definitive decisions can coexist. Figure 2 also retains all 300 calibrated interval widths per procedure.

## Primary comparisons and secondary clearance results

Primary success required the selector and at least one fixed procedure to qualify on a risk upper bound no greater than 10% and a decision-coverage lower bound at least 70%. Energy savings and noninferiority then had to pass against every qualifying fixed procedure. Coverage loss was limited to 5 percentage points and approval-risk increase to 2 points. No procedure qualified. The selector bounds were 38.10% coverage and 31.85% risk, so primary success was not established.

| Comparator | Mean saving J | Saving LB J | Coverage loss UB pp | Risk increase UB pp | All checks |
| --- | --- | --- | --- | --- | --- |
| Stop | -14.38 | -17.85 | 0.45 | 31.85 | Fail |
| Thermal | 24.13 | 20.67 | 19.65 | 31.85 | Fail |
| Voltage | 14.39 | 10.94 | 25.62 | 31.85 | Fail |
| Face | 14.29 | 10.83 | 20.29 | 31.85 | Fail |

Positive saving means fixed-policy energy minus selector energy. Energy inference uses paired block means and an approximate Student t lower bound across 100 blocks. The 73 one-sided primary directions use Bonferroni alpha 0.05/73. Positive named energy results against thermal, voltage and face do not override failed quality checks. Confidence bounds are not observed errors or observed losses.

### All predeclared secondary clearances

| Clearance K | Procedure | Decisions | Abstain | Approvals | FA | Rejections | Missed / 124 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | Selector | 125 | 175 | 77 | 0 | 48 | 76 |
| 0.05 | Stop | 98 | 202 | 55 | 0 | 43 | 81 |
| 0.05 | Thermal | 137 | 163 | 77 | 0 | 60 | 64 |
| 0.05 | Voltage | 152 | 148 | 106 | 0 | 46 | 78 |
| 0.05 | Face | 134 | 166 | 94 | 0 | 40 | 84 |
| 0.1 | Selector | 98 | 202 | 60 | 0 | 38 | 86 |
| 0.1 | Stop | 76 | 224 | 43 | 0 | 33 | 91 |
| 0.1 | Thermal | 112 | 188 | 62 | 0 | 50 | 74 |
| 0.1 | Voltage | 119 | 181 | 83 | 0 | 36 | 88 |
| 0.1 | Face | 107 | 193 | 74 | 0 | 33 | 91 |
| 0.2 | Selector | 58 | 242 | 38 | 0 | 20 | 104 |
| 0.2 | Stop | 48 | 252 | 30 | 0 | 18 | 106 |
| 0.2 | Thermal | 71 | 229 | 39 | 0 | 32 | 92 |
| 0.2 | Voltage | 71 | 229 | 47 | 0 | 24 | 100 |
| 0.2 | Face | 60 | 240 | 37 | 0 | 23 | 101 |

These descriptive curves reclassify saved final intervals and leave acquisitions, verification, corrections, energy and interval coverage unchanged. False rejections are zero in every row. Zero false approvals at larger clearances does not establish conditional risk assurance. No headline threshold was selected from these outcomes.

## Exploratory failure analysis and supporting errors

The bounded post hoc diagnosis partitions the 137 selector abstentions into 98 whose raw interval already crossed zero, 12 first appearing after development padding, and 27 first appearing after final calibration. In 95 of the 98 raw-crossing cases an individual candidate interval crossed zero. Only three arose solely from opposing definitive candidate intervals. This limits an explanation based only on model disagreement.

Of 125 selected additional measurements, 116 narrowed the interval, but 85 of those still ended in abstention. Nine did not narrow it. Narrowing uncertainty and resolving a final decision are different objectives. Removing both padding layers while keeping the saved actions would yield 202/300 definitive decisions and two false approvals, still below 70% decisiveness before uncertainty bounds. That counterfactual is a diagnosis, not an executable validated replacement.

Among selector abstentions, 51 had at least one fixed procedure that was definitive and 86 remained unresolved under every fixed procedure. This post hoc oracle comparison cannot be used as an acquisition rule. Figure 1 illustrates the first eligible narrowing case, not a representative population benefit.

### Matched family parameter error

| Procedure | Passed / 100 | Excluded | Contact median % | Capacity median % | Lag median % | Series median % |
| --- | --- | --- | --- | --- | --- | --- |
| Selector | 100 | 0 | 3.55 | 1.79 | 3.99 | 26.32 |
| Stop | 100 | 0 | 4.04 | 2.05 | 4.09 | 28.66 |
| Thermal | 100 | 0 | 2.71 | 1.31 | 3.05 | 28.65 |
| Voltage | 100 | 0 | 0.95 | 0.53 | 2.47 | 0.68 |
| Face | 100 | 0 | 0.74 | 0.72 | 2.56 | 27.76 |

Errors are absolute relative errors, conditional on a passed four-state verification in the matched family. Excluded fits remain counted in their separate column. Original scalar device parameters were deterministically replayed from the authenticated stored seeds in the frozen CPython environment, without new draws, refitting or integration. These supporting estimates do not certify all-case identifiability. No correct-family recovery is required for the intentionally unmatched contact law.

### Illustrative target trajectory errors in Figure 1

| Forecast | Candidate | Cold face RMSE K | Maximum absolute error K |
| --- | --- | --- | --- |
| Initial | five_state | 0.06113 | 0.13568 |
| Initial | four_state | 0.10418 | 0.16997 |
| Verified | four_state | 0.10658 | 0.17527 |
| Verified | five_state | 0.05037 | 0.11459 |

These errors use the 321 saved target time points and already exported candidate curves for block 1 contact mismatch. They are illustrative trajectory diagnostics, not a full-cohort trajectory accuracy claim. Margin and interval results remain the decision endpoints.

## Measurement map resources and practical limits

Figure 3 contains all 48 predeclared development cells: four physical sensor scenarios crossed with 12 cost scenarios, retaining 20 blocks and 60 cases per cell. Physical scenarios double voltage noise from 0.002 to 0.004 V, face noise from 0.02 to 0.04 K, or probe capacity from 5 to 10 J/K. Probe response remains 2.5 s. The stress rows were actually recomputed. Changing price alone reuses authenticated predictions.

Cost weights for energy, time and instruments are balanced (one third each), energy weighted (0.70, 0.20, 0.10), time weighted (0.20, 0.70, 0.10), or instrument weighted (0.15, 0.15, 0.70). Each is crossed with face instrument prices one quarter, equal to, or four times the voltage price. Reset assumes 80 s per added run. The primary nominal balanced equal-price cell selected stop/thermal/voltage/face in 34/16/5/5 cases, with 32/60 definitive decisions.

Across cells, stop selections ranged 29 to 42, thermal 8 to 31, voltage 0 to 10 and face 0 to 6. Definitive decisions ranged 28 to 32 of 60. There were no selection, verification or pipeline failures. Face acquisition was ineligible in one of 60 cases in every high-loading cell. All cases remain counted. Cells share development devices, precede final q and have no reserved risk guarantee. Only nominal balanced equal-price sensing is confirmed by the reserved experiment.

| Procedure | Mean runs | Energized s | Additional sensors | Mean diagnostic J |
| --- | --- | --- | --- | --- |
| Selector | 2.91 | 232.8 | 0.17 | 75.03 |
| Stop | 2.00 | 160.0 | 0.00 | 60.65 |
| Thermal | 5.00 | 400.0 | 0.00 | 99.16 |
| Voltage | 3.00 | 240.0 | 1.00 | 89.43 |
| Face | 3.00 | 240.0 | 1.00 | 89.32 |

Reserved selector choices were 175 stop, 74 thermal, 30 voltage and 21 face. Simulated diagnostic joules cover acquisition and verification terminal energy. They exclude computer electricity, reset energy, sensor electronics and hardware validation. Energized schedule time is not full bench elapsed time. Costs are assumptions, not demonstrated financial savings.

The entire reserved software benchmark used 262.90 completed-block CPU hours, 68.66 execution wall hours and four workers. The sampled execution memory peak was 0.472 GiB. The broader monitored workflow including independent reload took 68.96 hours. These figures include all fixed comparators and predictive draws and do not measure deployed single-case selector latency. Candidate refitting inside many predictive draws dominates computation.

Limited approval counts and conservative simultaneous inference were known precision limitations before calibration opened. The 2-point risk comparison was explicitly underpowered. Unmodeled physics, distribution shift, local covariance approximation, probe priors and uniform reset assumptions restrict interpretation. The present evidence supports neither a hardware reliability claim nor adaptive superiority to a qualified fixed policy.

## Verification reproduction and recommended next work

Independent standard-library reconstruction checked all 6,000 saved decisions across four clearances, 100 raw checkpoints, 300 before-reveal files, scientific seals, final intervals, errors and resource comparisons. All 1,364 compared numerical fields agreed, with maximum absolute discrepancy 7.11e-15. Binomial inversion used 60-digit Decimal arithmetic. The Student t critical value used a separate integral recurrence. These checks are independent of the original SciPy scoring implementation.

The predetermined bounded numerical replay reproduced 15 policy refits, six predictive draws and 116 calibrated decisions exactly under the original source and runtime. It included families, actions, candidate exclusions, a whole-draw failure and near-boundary decisions. The preceding full run passed 1,199 existing tests, and focused runs passed the seven new replay tests and six figure tests. The figure-focused acceptance ran 48 tests. Tests check implementation behavior, not all physical assumptions.

The three figure exports reproduce byte for byte in the recorded Matplotlib 3.10.8 environment. The report and editable slide deck consume authenticated compact results. The reproduction package retains raw diagnostics and outcomes, protocol and calibration artifacts, source manifests, independent audit tools, frozen numerical source, environment requirements, figures and generated deliverables. The package verifier checks every archived byte against its manifest and a clean extracted archive repeats the result audit. It is a local hash-verified archive, not an immutable external deposit.

### Recommendation within the tested scope

Close this version with primary non-success intact. No procedure meets the declared operating qualification, so none is recommended as an assured operating policy. If a descriptive software benchmark comparator is needed, fixed voltage gave the highest observed decisiveness (182/300) and complete block interval coverage (100/100) at 89.43 J. This post hoc description does not qualify its conditional approval risk or establish it as an optimal deployable strategy.

A future software experiment can test action value based on the expected chance of resolving a reliable final decision per cost, including calibration and the distance to the decision boundary. That is a hypothesis motivated by this diagnosis. It needs a new protocol, fresh development and calibration partitions, an explicit precision and compute budget, and a new unseen reserved cohort. Do not revise the selector on this exposed cohort. Hardware sensing and probe-loading tests are a separate follow-on project.

### Evidence references

Repository records: OPERATING_DECISION_PHASE_E_PROTOCOL.md; OPERATING_DECISION_PHASE_F_RESULT.md; OPERATING_DECISION_ABSTENTION_DIAGNOSIS.md; OPERATING_DECISION_PHASE_H_REPLAY.md; OPERATING_DECISION_FINAL_FIGURES.md; OPERATING_DECISION_REPRODUCIBILITY.md. Machine-readable primary, secondary, calibration, diagnosis, replay and figure artifacts are retained under thermotwin/. Figures 1 to 3 follow as full vector pages.

| Identity | Recorded value |
| --- | --- |
| Numerical source | 7e3b9a3be22d991ec2e588cbd4a53a9d4d2d27d7 |
| Reserved science digest | d565c0c6368b602eeb3d2721c128aab88221bce9a6b5abe68fcda4c727d4edf0 |
| Protocol digest | 7f692093108b9e0c0765e5497475a89d1d27eea3d00bf13e4568d6f68fb4301b |

## Final figures

See [figures and captions](OPERATING_DECISION_FINAL_FIGURES.md) for all three exports and their interpretation.
