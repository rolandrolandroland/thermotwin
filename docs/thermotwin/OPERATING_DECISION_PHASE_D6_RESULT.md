# Prospective operating-decision Phase D6 result

Date: 2026-10-06. Status: all 10 internal-check blocks and the complete saved
archive validated. No predeclared redesign trigger fired. The selected design
is unchanged and the final Phase D freeze authorizes Phase E protocol work.

## Authenticated execution

The internal check ran from clean source
`75f598ad5a6330207626df1a800bc3d2c4e7b3f7` after its exact-head
[CI passed](https://github.com/rolandrolandroland/thermotwin/actions/runs/37339036917).
It opened `p1_development_internal_check` for 10 paired blocks and 30 cases
under CPython 3.10.12 on Darwin arm64 with the pinned Phase D scientific
packages. The frozen rule selected each action before target reveal; all four
fixed-policy decisions were also saved before reveal. No offset, threshold,
physics model, draw count, or scenario was retuned.

Interrupted sessions resumed from validated checkpoints: first four, then
eight completed blocks were retained, and the last invocation computed only
blocks 8 and 9. The final archive was reloaded and validated by the execution
path. A separate clean-source validation subsequently returned
`Phase D6 internal-check archive: VALID` with the same scientific digest.
This validates retained records, formulas, streams, and provenance; it does
not independently rerun the simulations and numerical fits.

| Record | Identity |
| --- | --- |
| Source manifest | `79772d230eec28aaf41499f0b258083ca061b91fbec8d244515df44b4d89c7ad` |
| Protocol | `e1e10565df6c2d03c923a495de1f4ad772564c1e44306a6b4d565f79bc253579` |
| Scientific result | `19eeffd3d19e59556cd0f57073ad89c8aa70ef71bc839df38b491b785f2a9ebc` |
| Archive content digest | `2dca8ec5a273fcd7bc22ebd0283d61cdc99b7d11a5f8149bde730e16da8595bb` |
| Complete JSON | 36,663,022 bytes; SHA-256 `79e47dce4cd0666db1e7457917769992668186cc158aa1ba44869e1aa53b7beb` |
| Compact report | SHA-256 `74af182d3c6e7d05ff08c6e00771e1365b9fada90cf65364a4a5e7a92a5dab2d` |
| Resource record | SHA-256 `5d02e026b7c5a13371339e968c574e09bd8ffa7267f7b1995fac4eb21a0aefde` |

The full evidence remains in the ignored local directory
`.artifacts/operating-decision/prospective-phase-d6-75f598a/`. Its hashes,
compact metrics, and freeze record are tracked. A content-addressed archival
release remains part of the later project closeout.

## Primary result and transfer from tuning

| Outcome | Tuning | Internal check |
| --- | ---: | ---: |
| Stop selected | 34/60 (56.7%) | 22/30 (73.3%) |
| Thermal selected | 16/60 (26.7%) | 2/30 (6.7%) |
| Voltage selected | 5/60 (8.3%) | 4/30 (13.3%) |
| Face temperature selected | 5/60 (8.3%) | 2/30 (6.7%) |
| Definitive decisions | 32/60 (53.3%) | 24/30 (80.0%) |
| False approvals / false rejections | 0 / 0 | 0 / 0 |
| Selection / verification / pipeline failures | 0 / 0 / 0 | 0 / 0 / 0 |
| Mean realized diagnostic energy | 75.66 J | 69.51 J |
| Mean extra sensors | 0.167 | 0.200 |

The check produced 10 approvals, 14 rejections, and 6 abstentions. Two cases
stopped because no acquisition cleared the frozen value threshold. Decision
coverage increased by 26.7 percentage points from tuning, mean diagnostic
energy decreased by 6.15 J, and mean extra sensors increased by 0.033.
Mean recorded energized schedule time was 192 seconds and mean diagnostic
run count was 2.4. These are descriptive changes across two development
partitions, with no claim of statistically established improvement.

Development-adjusted intervals covered 27/30 cases and all three cases
simultaneously in 7/10 blocks. Raw intervals covered 25/30 cases. Interval
coverage and definitive-decision coverage are different endpoints. There is
no independent calibration guarantee at this stage.

## Family C, candidate count, and action diagnostics

| Group | Cases | Definitive | False approvals / rejections | Adjusted intervals covered |
| --- | ---: | ---: | ---: | ---: |
| Family A: matched four-state | 10 | 8 | 0 / 0 | 10/10 |
| Family B: extra interface mass | 10 | 9 | 0 / 0 | 9/10 |
| Family C: temperature-dependent contact | 10 | 7 | 0 / 0 | 8/10 |
| One admissible candidate, all selected stop | 6 | 6 | 0 / 0 | 4/6 |
| Two admissible candidates | 24 | 18 | 0 / 0 | 23/24 |
| No admissible candidate | 0 | N/A | N/A | N/A |

Within selected actions, stop produced 20/22 definitive decisions, thermal
1/2, voltage 3/4, and face temperature 0/2. Both face selections ended in
abstention. The one-candidate stopping stratum had no decision errors but
missed true margins in two of six intervals. Preserve this as a limitation
for the independent calibration stage; it did not meet a declared redesign
trigger. The small action and stratum counts do not justify conditional
coverage or risk claims.

## Fixed-policy frontier

All four fixed-policy outcomes were preserved during D6 execution. The table
below applies the same frozen development offsets and verification standard
to those saved outcomes using the existing D3 analysis helpers, without
refitting or opening any new partition. It completes descriptive development
reporting and does not change the D6 trigger decision.

| Procedure | Definitive | False approvals / rejections | Adjusted intervals covered | Mean energy | Mean extra sensors | Mean declared loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Selector | 24/30 | 0 / 0 | 27/30 | 69.51 J | 0.20 | 0.2889 |
| Stop | 20/30 | 0 / 0 | 28/30 | 61.07 J | 0 | 0.3333 |
| Thermal | 21/30 | 0 / 0 | 26/30 | 99.82 J | 0 | 0.7339 |
| Voltage | 21/30 | 0 / 0 | 26/30 | 90.03 J | 1 | 0.6000 |
| Face temperature | 22/30 | 0 / 0 | 28/30 | 89.91 J | 1 | 0.5662 |

The selector traded more energy than stop for more definitive decisions. It
used less energy than the measurement policies in this small check. These
comparisons require the separately frozen, powered Phase E/F analysis before
any confirmatory claim. The complete overall and stratified comparator
metrics are retained in the final Phase D artifact, and
`validate_phase_d_final_freeze_evidence` reconstructs them from the
hash-authenticated D6 archive.

## Redesign gate and resource record

No recorded material information-boundary, provenance, physics, or
implementation defect was detected by the execution and archive checks. There
were no false approvals, so no repeated-error stratum exists. Each measurement
action was eligible in all 30 cases. All three declared redesign triggers
were false; no revision or replacement namespace was used.

Aggregate recorded block CPU time was 61,159.2 seconds (16.99 hours), within
the approximate 18.31-hour internal-check budget. The final monitored
invocation took 6,479.5 wall seconds (1.80 hours), resumed eight completed
blocks, and recorded peak process-tree RSS of 547,028,992 bytes with 6,305
samples and no monitoring error, below the 4 GiB limit. The final archive is
below the approximate 38,482,805-byte budget. Earlier invocations are not
included in that final wall-time or process-tree record, so no total campaign
wall time or aggregate peak-memory bound is inferred from it.

## Final Phase D freeze and next action

The complete selected design and D1-D6 evidence identities are frozen in
`thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_D_FINAL_FREEZE.json`, artifact
digest `ecb490919a271357749101f7db7c30b98b445b522c2e39203508c3f716c137f6`.
The historical provisional freeze remains intact. The internal-check and
tuning partitions are closed against regeneration.

Next is Phase E: commit the endpoint and comparison specification, review
precision and compute for the planned 100 calibration and 100 reserved
blocks, implement and test the procedure-level calibration correction, and
pass a disposable rehearsal and exact-head CI before calibration opens.
Both independent calibration and reserved evaluation remain unopened. The
development design is frozen; analysis specification and calibration gates
still have to be completed.
