# Phase F3: accepted rehearsal and reserved-generation gate

Accepted: 2026-10-18. Status: engineering acceptance passed. The
[generation gate](../../thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_F_GENERATION_GATE.json)
must be committed before the reserved cohort opens. At acceptance the reserved
partition is unopened; this record authorizes its one execution after the gate
commit, under the already frozen procedure.

The complete calibrated rule and numerical source were committed and pushed
as `7e3b9a3be22d991ec2e588cbd4a53a9d4d2d27d7` before disposable generation.
[Exact-source CI](https://github.com/rolandrolandroland/thermotwin/actions/runs/39089299742)
passed at 22:36:43 Eastern on October 17. The disposable workflow started
afterward. Both independent computations and subsequent separate archive
validations finished by 01:08:23 Eastern on October 18, with no reported
execution or monitoring error.

## Acceptance evidence

| Check | Result |
| --- | --- |
| Committed complete rule and exact-source CI | Passed at `7e3b9a3`; the executing clone remains clean and unchanged. |
| Closed calibration | Original Phase E archive revalidated in its original `23a076f` import context; no calibration observation regenerated. |
| Disposable computation | One three-family block computed twice independently in the declared `p0_disposable_phase_f_roundtrip_v1` namespace. |
| Final archive round trips | Both complete saved evaluations reconstructed from their retained raw blocks and before-reveal files. |
| Exact scientific replay | The full scientific result digest, calibrated records, and primary/secondary analyses agree. |
| Independent archive reload | Two separate CLI processes validated the two completed evaluations. |
| Additional independent arithmetic check | Standard-library-only code verified both sealed blocks, six before-reveal files, all 120 calibrated decisions and their outcome labels using exact fractions and outward rounding. |
| Constructed storage/analysis | 100 repeated raw records, 305,867,067 bytes, and 1,500 constructed analysis rows passed transport checks. |
| Whole-rehearsal memory | 7,355 process-tree samples, no sampling errors; sampled peak 1,791,131,648 bytes (about 1.67 GiB), below 4 GiB. |
| External operational monitor | Sampled peak 1,853,587,456 bytes (about 1.73 GiB), including the wrapper and subsequent independent validations. |
| Rehearsal invocation wall time | 8,280.486 seconds, about 2 hours 18 minutes. |
| Per-repeat completed-block CPU | 3,924.088 and 3,912.660 seconds. |
| Gate rejection checks | Wrong source, missing independent-reload acceptance, and failed resources were each rejected before generation. |

The pool is configured for four workers. Each disposable repetition has only
one computing block, and the two repetitions run sequentially. This is not a
measurement of four simultaneous active fits. The 100-record storage fixture
and repeated computations are engineering evidence only: there is one distinct
disposable block and three distinct cases, not 100 scientific blocks or six
independent cases. These checks establish execution/reproducibility acceptance,
not a selector performance, coverage, energy, or hardware claim.

## Frozen identities and retained files

| Identity | Digest |
| --- | --- |
| Numerical source | `7e3b9a3be22d991ec2e588cbd4a53a9d4d2d27d7` |
| Numerical source manifest | `002d8afd7641dfa2e78269d195a5375c37b3f8aa547eb86d574dc439e7858662` |
| Phase F protocol | `7f692093108b9e0c0765e5497475a89d1d27eea3d00bf13e4568d6f68fb4301b` |
| Complete calibrated rule | `d8ae3d9ea3748d4de776a672cf19798b145b589d681211ba88b8789186dd6341` |
| Accepted calibration artifact | `124d97f19c1cecb8cfefa6ee5a2b78f4f78570af8337b649f04bbc61435f4f2d` |
| Matching disposable scientific result | `a60537f15b17343c6cf3cfa029a5ad888ab88d85b0d956841dc4aebe4bb5cfbd` |
| Generation-gate artifact | `b64c707b4337ba6b09f99c6346e26c7042c2947fdf80b9dbae32e67bcda94b24` |

Raw evidence remains ignored under
`.artifacts/operating-decision/prospective-phase-f-rehearsal-7e3b9a3-v1/`.
The external journal, saved exact-source CI response, and independent reload
record are under
`.artifacts/operating-decision/phase-f-durable-rehearsal-7e3b9a3-20261017/`.
The additional arithmetic checker and its result are
`phase-f-acceptance-check-20261018.py` and
`phase-f-acceptance-check-20261018.json` in the same artifact root. The committed
gate records 24 evidence file sizes/hashes, including both complete archives,
their raw blocks, all six before-reveal files, storage fixture, resource/CI
records, independent checks, and original local test logs. Full archive bytes
differ between repeats because their timing measurements differ.

All 109 original Phase E numerical dependencies remain unchanged. Runtime
remains CPython 3.10.12 on Darwin arm64 with the frozen package requirements
and 18 explicitly unset math/Python environment variables. The acceptance
commit adds the gate and records; numerical execution remains at the original
tested clean `7e3b9a3` source clone.

## Authorized execution and stopping rules

After this gate is committed, execute `p1_reserved_evaluation` once: 100 paired
blocks, 300 cases, all five procedures, 1,500 primary outcomes, four workers,
nominal sensing, and `balanced_face_equal` costs. Save all five procedures'
calibrated decisions at all four clearances before target reveal. Preserve
every ordinary failure and comparator. Stop interpretation on a material truth,
source, or archive incident; do not retune, reseed, replace completed blocks,
or extend the sample size after outcomes.

The unchanged analysis still uses 73 simultaneous bounds and the predeclared
qualification/noninferiority rules. The strict risk comparison remains
explicitly underpowered. A null or inconclusive result completes the stated
experiment and is not a reason to redesign this reserved cohort.

The launch will use a one-time native job with a persistent progress/memory
journal and no automatic restart. A temporary process-bound idle-sleep
inhibitor lasts only for this job; it does not change persistent power
preferences or prevent display sleep. The journal records launch/gate identity,
checkpoint counts, before-reveal files, process-tree memory, calendar elapsed
time and monotonic elapsed time. Progress will be retained under
`.artifacts/operating-decision/phase-f-durable-reserved-7e3b9a3-20261018/`;
the reserved archive will be under
`.artifacts/operating-decision/prospective-phase-f-reserved-7e3b9a3-v1/`.
The launcher will independently validate the completed reserved archive in a
separate process before recording completion. Launch and progress timestamps
come from that journal, not from a retrospective rewrite of this acceptance.

Retain the measured Phase E budget as the conservative planning reference:
255.26 completed-block CPU hours, about 64 ideal four-worker active hours
before overhead. Allow roughly 3–4 active days for reserved execution and
validation; the one disposable device block does not establish the whole
cohort's runtime. Final interpretation, figures, report, presentation, and
reproducibility packaging follow independent validation.
