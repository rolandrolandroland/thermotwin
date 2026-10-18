# Phase E: accepted independent calibration

Accepted: 2026-10-17. Numerical source: `23a076f69ad4da294234b146e330b8615bed730e`.
Status: complete, finite, independently validated, and closed against regeneration.
Reserved evaluation remains unopened and unauthorized.

All 100 paired blocks completed on October 16, supplying 300 cases and 1,500
procedure/case scores. The same cohort resumed across interruptions; completed
checkpoints were authenticated and retained. No replacement seeds, sample-size
extension, selector redesign, or refitting was used to obtain this result.

The accepted [calibration artifact](../../thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_E_CALIBRATION.json)
seals the following **additional final-interval corrections**. These corrections
do not enter acquisition, action selection, stopping, development offsets, fits,
or verification. The selector uses its own correction regardless of its action.

| Procedure | Correction (K) | Infinite block scores |
| --- | ---: | ---: |
| Prospective selector | 0.025436425099518184 | 0/100 |
| Stop now | 0.04965713743107725 | 0/100 |
| Fixed thermal | 0.010413810957399489 | 0/100 |
| Fixed voltage | 0.04106348498740038 | 0/100 |
| Fixed face temperature | 0.025436425099518184 | 0/100 |

Each correction is the frozen rank-96 order statistic of 100 block maxima,
including all three families and all failures in the denominator. The achieved
confidence over calibration samples is 0.9762889173365232 per procedure for the
90% block-coverage target. This is not a joint guarantee across the five
procedures, a guarantee conditional on the selected action, or a demonstration
of reserved-set performance or energy savings.

## Independent checks and retained evidence

A separate execution of the original frozen-source archive validator passed
after the complete campaign finished. A second, standard-library-only script
independently recomputed all 1,500 scores using exact rational arithmetic and
upward rounding, all 500 block maxima, the exact binomial rank/confidence, and
all five corrections. It checked the complete matrix, shared device identities,
all 100 checkpoint hashes, and summary/report checksums. Neither check generated
observations or opened reserved evidence.

Raw records remain ignored under
`.artifacts/operating-decision/prospective-phase-e-calibration-23a076f/`.
The acceptance artifact records sizes and hashes for the summary, report,
checksums, resource record, independent validation log, independent arithmetic
result, and its script. The independent arithmetic result retains its original
timestamped statement that the separate semantic validator was then in progress;
the subsequent successful validator result is recorded in this acceptance.

| Identity | Digest |
| --- | --- |
| Accepted calibration artifact | `124d97f19c1cecb8cfefa6ee5a2b78f4f78570af8337b649f04bbc61435f4f2d` |
| Raw calibration JSON SHA-256 | `d72b8bf030414a8a477545d12f80b3bd12001988d0c87efb68bed83f953e7995` |
| Scientific result | `5acc545652c186968fea327ced2f33727279c4b9e13419937f27bfe245586b38` |
| Original numerical source manifest | `88e3f8187ab57683ec1c36a6bd24a96364e508d50bb2ae9aa709dfe354645a20` |
| Original protocol | `9ef8795fc2bf2b7db536a3c72fd9593e538faa54acfea1af8cf2f78d7f96cfb6` |

## Measured computation and limitations

Completed blocks consumed **918,921.55567 CPU seconds (255.26 CPU hours)**.
That exceeds the original development-based projection; preserve that projection
as planning history. This total excludes lost partial work and independent
validation overhead. It is software compute cost, separate from simulated
diagnostic energy.

The final successful invocation resumed 85 blocks, computed the last 15, and
passed its resource gate: 48,923.35 seconds of invocation wall time, 43,069
process-tree samples, no sampling error, and a peak of 575,094,784 bytes under
the 4 GiB limit. Its external journal sampled a peak of 610,467,840 bytes.
These measurements apply to that invocation. They do not establish the peak
for the whole interrupted campaign.

Three earlier invocations were interrupted. The first two have no recovered
final elapsed-time or peak-memory summaries. The October 12–15 invocation
retains a partial external journal, including SIGTERM before the Mac restart,
17,922 samples, and a sampled peak of 523,665,408 bytes. Overnight sleep also
contributed to calendar delays. Complete campaign wall time and peak memory
cannot be reconstructed from completed-block counts; retain this limitation.

## Next gate

Phase F must commit the complete calibrated rule and its execution/analysis
source, pass exact-source CI, and independently compute and reload a fresh
disposable block twice. Calibrated decisions at all four declared clearances
must be durably saved before target reveal. Replay, archive, and resource
acceptance must be recorded in a separate committed generation gate before
the existing 100-block reserved namespace opens once. The predeclared analysis
and sample size remain unchanged even if the final comparison is inconclusive.
