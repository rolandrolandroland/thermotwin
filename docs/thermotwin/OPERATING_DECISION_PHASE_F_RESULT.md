# Phase F: completed reserved operating-decision result

Result recorded: 2026-10-29. Numerical execution and independent archive
validation completed: **2026-10-21 at 19:24 Eastern**.
Status: valid completed experiment; **predeclared primary success not established**.
The reserved cohort is closed against regeneration, retuning, and extension.

The [machine-readable result](../../thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_F_RESULT.json)
records all primary metrics/comparisons, descriptive secondary results, source
and evidence identities, resource measurements, and the new independent audit.
It belongs to the prospective four-action experiment. Historical files named
`OPERATING_DECISION_FINAL_*` describe earlier studies and are not replaced.

## Result and scope

All 100 paired blocks completed once: 300 cases, five procedures, and 1,500
primary procedure/case outcomes. All 300 before-reveal files and 6,000 saved
decisions/outcomes across the four predeclared clearances are retained. No
blocks were resumed or replaced, and no extra observations were generated
for this result record. There were no pipeline, verification, or missing-final-
interval failures in the primary outcomes.

| Procedure | Definitive decisions / all cases | False approvals / approvals | Blocks with all three intervals covered | Mean diagnostic energy (J) |
| --- | ---: | ---: | ---: | ---: |
| Prospective selector | 163/300 (54.3%) | 1/104 | 93/100 | 75.03 |
| Stop now | 118/300 (39.3%) | 0/67 | 99/100 | 60.65 |
| Fixed thermal | 171/300 (57.0%) | 0/101 | 94/100 | 99.16 |
| Fixed voltage | 182/300 (60.7%) | 0/129 | 100/100 | 89.43 |
| Fixed face temperature | 165/300 (55.0%) | 1/112 | 96/100 | 89.32 |

All five procedures made zero false rejections. The selector abstained in
137/300 cases. Its intervals covered 293/300 individual cases and 93/100
complete blocks. Those are observed interval-coverage results; they are
different from definitive-decision coverage and from risk conditional on
approval. Neither a high interval-coverage rate nor zero observed errors
certifies the separate approval-risk target.

The selector selected stop/thermal/voltage/face in 175/74/30/21 cases. These
counts describe the primary nominal scenario, not a calibrated result for
every development-map sensor/cost scenario.

## Why primary success was not established

The [predeclared protocol](OPERATING_DECISION_PHASE_E_PROTOCOL.md) requires the
selector to qualify on absolute risk and decisiveness, at least one fixed
procedure to qualify, and all paired energy/coverage/risk checks to pass against
every qualifying fixed procedure. The primary clearance is 0.00 K.

| Absolute selector requirement | Observed rate | Required confidence bound | Obtained confidence bound |
| --- | ---: | ---: | ---: |
| Definitive-decision coverage | 54.33% | Lower bound at least 70% | Lower bound 38.10% |
| False approvals conditional on approval | 1/104 = 0.96% | Upper bound at most 10% | Upper bound 31.85% |

The observed decision rate already misses the 70% target. The 31.85% bound
is uncertainty about approval risk, **not** an observed 31.85% error rate.
Only 19/44/41 approvals occurred in the matched/interface/contact families;
family-specific exact bounds and the 73-bound simultaneous adjustment leave
substantial uncertainty. The precision limitation was documented before
calibration and reserved generation. The record supports an observed
decisiveness shortfall and inconclusive approval-risk assurance.

No fixed procedure qualified either: their decision rates range from 39.3% to
60.7%, all confidence lower bounds are below 70%, and all approval-risk upper
bounds exceed 10%. The required nonempty qualifying-comparator set is absent.
No strongest-fixed-strategy superiority or qualified operating policy is promoted.

All four named comparisons remain reported:

| Fixed comparator | Mean fixed-minus-selector energy (J) | Lower energy-saving bound (J) | Upper decision-coverage loss (percentage points) | Upper approval-risk increase (percentage points) |
| --- | ---: | ---: | ---: | ---: |
| Stop now | -14.38 | -17.85 | 0.45 | 31.85 |
| Thermal | 24.13 | 20.67 | 19.65 | 31.85 |
| Voltage | 14.39 | 10.94 | 25.62 | 31.85 |
| Face temperature | 14.29 | 10.83 | 20.29 | 31.85 |

The energy-saving check passes against the three measurement policies. It
fails against stop, which uses less energy. The maximum permitted coverage
loss is 5 percentage points and maximum permitted approval-risk increase is
2 percentage points. No comparator passes all three checks. These uncertainty
bounds are not observed losses or observed risk increases.

Energy inference uses a paired-block Student-t bound with its declared
approximate sampling interpretation. The 73 primary directions use the frozen
Bonferroni allocation, not 300 independent-case inference. Report these named
energy results alongside their decision-quality limits; do not relabel them
as success on the joint primary question.

## Secondary results

The predeclared clearance grid reclassifies saved final intervals without
changing acquisition, selection, fits, corrections, verification, or energy.
These results are descriptive; no threshold is chosen from reserved outcomes.

| Clearance (K) | Selector definitive decisions / 300 | Selector approvals | Selector false approvals |
| --- | ---: | ---: | ---: |
| 0.00 — primary | 163 | 104 | 1 |
| 0.05 | 125 | 77 | 0 |
| 0.10 | 98 | 60 | 0 |
| 0.20 | 58 | 38 | 0 |

Counts for every fixed comparator at every clearance are in the result
artifact. Zero observed false approvals at larger clearances does not establish
their risk guarantee; fewer approvals and decisions remain in the denominator.

## Independent result audit

The original independent archive validation passed on October 21. On October
29, a separate [standard-library audit tool](../../tools/verify_prospective_phase_f_result.py)
reconstructed all 6,000 outcomes from the 100 retained raw checkpoints and
300 before-reveal files. It verified canonical seals, checkpoint hashes,
scientific digests, exact outward-rounded intervals, all saved decisions,
paired truth identities, errors, resources, and complete matrices.

The tool imports no ThermoTwin, NumPy, or SciPy code. Its binomial bounds use
60-digit Decimal inversion of direct binomial sums, independently of the
original beta-quantile implementation. Its energy critical value comes from
an angular Student-t density integral recurrence and bisection, independently
of the original Student-t quantile implementation. It recomputed all primary
and secondary summaries, paired contrasts, confidence bounds, and the final
Boolean success result. The df=99 critical value is 3.2942299046144523.

All 1,364 compared floating-point fields agreed; the largest absolute difference
was **7.105427357601002e-15**. Count, Boolean, saved-row and decision checks are
exact. The separate numerical audit tolerance (2e-10 absolute / 2e-12 relative)
does not change any scientific threshold or decision rule. All 28 focused
audit and Phase F boundary tests passed in 4.297 seconds. No frozen numerical
dependency changed and no observation, physical simulation, or refit was run.
Resource means and paired energy contrasts are recomputed from retained
post-reveal scores; the underlying ODE and power integrations are not rerun.
This verifies arithmetic and retained evidence, not all real-world physics.
The predetermined small numerical-refit replay in the broader final-closeout
plan is still pending and is not claimed by this audit.

## Instruments, schedule assumptions, and software cost

| Procedure | Mean diagnostic runs | Energized schedule time (s) | Mean additional sensors |
| --- | ---: | ---: | ---: |
| Selector | 2.91 | 232.8 | 0.17 |
| Stop now | 2.00 | 160.0 | 0.00 |
| Thermal | 5.00 | 400.0 | 0.00 |
| Voltage | 3.00 | 240.0 | 1.00 |
| Face temperature | 3.00 | 240.0 | 1.00 |

Diagnostic joules are simulated acquisition/verification energy, not computer
electricity, complete bench wall time, reset energy, or hardware measurements.
The software benchmark consumed 946,449.80793 completed-block CPU seconds
(262.90 CPU hours). The execution CLI took 247,187.295 seconds (68.66 hours),
including its resource scope. The full external workflow including the later
independent reload took 248,271.973 calendar seconds (68.96 hours).

The execution monitor retained 217,773 samples with no sampling error and a
sampled peak of 506,757,120 bytes. The external workflow peak was 520,650,752
bytes, below 4 GiB, with no monitor error or recorded interruption. The idle-
sleep inhibitor lasted only for the job and was released afterward. These
are sampled peaks, not continuous-memory guarantees. Whole-benchmark CPU
includes all fixed comparators and predictive draws and is not a deployed
single-case selector timing benchmark.

## Provenance and reproduction

| Identity | Value |
| --- | --- |
| Numerical source | `7e3b9a3be22d991ec2e588cbd4a53a9d4d2d27d7` |
| Generation-gate commit | `1eaf547e045c63e936622324af921b1b4949c0b0` |
| Numerical source manifest | `002d8afd7641dfa2e78269d195a5375c37b3f8aa547eb86d574dc439e7858662` |
| Protocol digest | `7f692093108b9e0c0765e5497475a89d1d27eea3d00bf13e4568d6f68fb4301b` |
| Frozen scientific result | `d565c0c6368b602eeb3d2721c128aab88221bce9a6b5abe68fcda4c727d4edf0` |
| Raw evaluation JSON SHA-256 | `299f75ad492e5bde0acc724ed4e72ae029c86f2c05538677c2a856c23734c564` |
| Compact result artifact | `ed58ade9da509307ce8a88ae97b549b8dc7bc21907e5d5e1d5272a6c47c12210` |

Scientific execution used the frozen CPython 3.10.12 Darwin arm64 environment.
The 117 numerical dependencies still match their frozen source manifest.
Raw archives remain ignored under
`.artifacts/operating-decision/prospective-phase-f-reserved-7e3b9a3-v1/`.
The execution journal and original independent-validation record remain under
`.artifacts/operating-decision/phase-f-durable-reserved-7e3b9a3-20261018/`.
The compact result binds their hashes and the new audit, source, and test records.
It is hash-verifiable evidence; an external archival deposit is not yet complete.

The arithmetic audit can be repeated without generating a cohort. From the
repository root, with the retained raw files available, use a fresh output:

```sh
python tools/verify_prospective_phase_f_result.py \
  --archive .artifacts/operating-decision/prospective-phase-f-reserved-7e3b9a3-v1/evaluation.json \
  --source-root . \
  --validation-record .artifacts/operating-decision/phase-f-durable-reserved-7e3b9a3-20261018/independent-validation.json \
  --output /tmp/thermotwin-phase-f-audit.json
```

The [result builder](../../tools/record_prospective_phase_f_result.py) packages
that audit and the saved evidence; its CLI documents required inputs. It also
refuses to overwrite an accepted record. Fresh audit timestamps and evidence
hashes can change the packaging digest. The frozen scientific-result digest
and reported scientific numbers must remain unchanged. Neither tool imports
the original scoring implementation or regenerates reserved observations.

## Remaining work

This completes the validated Phase F result record and the independent
arithmetic/statistical check. The original campaign is not reopened. The
remaining Phase G/H work is the explicitly exploratory abstention diagnosis,
predetermined small numerical replay, three final figures, final report,
presentation, and reproducibility/deposit package. Any redesigned selector
requires a new version and fresh development, calibration and reserved
partitions; this exposed cohort cannot confirm its performance.
