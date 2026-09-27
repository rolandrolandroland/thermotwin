# Prospective operating-decision Phase D0 result

Date: 2026-09-26. Historical status: **PASS under protocol v1 and its recorded
CPython 3.13.3 runtime**. The Phase D development protocol was source-bound,
tested, and rehearsed through canonical save/load validation. A September 26
audit later required a versioned protocol-v2 repair before development. This
record and its artifacts remain unchanged historical evidence. No development,
calibration, or reserved partition was opened.

## Frozen implementation

The implementation is frozen at source revision
`a280d7f92d647847934ffcd75565e57189297558`. It adds the versioned Phase D
protocol, the strict rehearsal archive and validator, the command-line entry
point, and focused tests. The protocol binds the Phase C freeze, both Phase D
partition names and block counts, the four-policy catalog, the 12 cost cells,
four sensor scenarios, the 81-rule development grid, the offset estimator,
draw-count sensitivity and contingency rules, internal-check triggers,
truth-reveal chronology, output requirements, compute budget, runtime, and an
84-file numerical source manifest.

The numerical source-manifest digest is
`c429cb7026611a34dccdb73e74741a4946e4e8b332c1e04ceec1998fad8f1c0e`.
The exact source-bound protocol digest is
`96eba48d34c78b594366484c8304f86af20a9d8e32aa48f6e35be156596c969e`.
The command exposes protocol printing, disposable rehearsal execution, and
rehearsal validation. It does not yet expose a command that can open a
scientific Phase D partition.

## Verification before execution

- The full local suite passed: 829 tests in 811.708 seconds.
- Focused Phase C/D checks passed, including the pilot regression suite and
  prospective interfaces.
- A detached clean clone reproduced the 84-file manifest and the 81-rule
  protocol at the exact source revision.
- Exact-head GitHub Actions run
  [`36266342966`](https://github.com/rolandrolandroland/thermotwin/actions/runs/36266342966)
  passed at the exact source revision. The complete test step ran from
  19:32:46Z through 19:53:43Z.

## Disposable rehearsal

The one-block namespace was
`p0_disposable_phase_d_archive_roundtrip_v1`. It contained three cases, one
from each truth family. The command executed the full N=16 prospective path,
fixed-policy outcomes, saved-before-reveal chronology, content sealing,
canonical save/load, source-manifest verification, and recorded-consistency
validation. The validator did not independently rerun the simulations and
fits.

The saved JSON was loaded in a separate process from the detached clean clone
and passed the strict archive validator. The result was:

| Check | Result |
| --- | ---: |
| Cases | 3 |
| Pipeline failures | 0 |
| N=16 ineligible measurement actions | 0 |
| N=16 whole-draw failures | 0 |
| Wall time | 4,045.50 s |
| CPU time | 3,921.58 s |
| Peak RSS | 40,353,792 bytes |
| Archive size | 3,583,461 bytes |

The disposable N=16 choices were:

| Truth family | Choice |
| --- | --- |
| `matched_four_state` | `stop_now` |
| `extra_interface_mass` | `fixed_thermal` |
| `temperature_dependent_contact` | `fixed_face_temperature` |

These choices are transport and execution evidence only. They cannot tune an
offset, select a grid row, support a performance claim, or enter any later
scientific analysis.

## Artifact addresses

The local artifacts remain under the gitignored
`.artifacts/operating-decision/prospective-phase-d/` evidence directory.
Their committed addresses are:

- JSON SHA-256:
  `ace47b991c37a7e43a940a909dc63ae809d7839333969684190214ba634e50ae`;
- report SHA-256:
  `9b9e8ac17f7aba4ce2c37452ee84eb77689277a2b179dbb44e8f992ddf6dcef2`;
- archive-content digest:
  `1004a77e2cebf8dca67776851eea609cc51c55c1a9e01d42f217759203fccd2f`;
- scientific-result digest:
  `ccbd864b8866482435c5dc3884688b4fede0104c986f984a5e090220ae6aa191`;
  and
- block-evidence digest:
  `4d3428a22b794c5a8407a10d10be689358692e2824ec52bf91a055479e59cfdc`.

## Gate decision

Historical Phase D0 v1 passed. It was superseded before development by the
audit-repair protocol v2 described in
[`OPERATING_DECISION_PHASE_D_PLAN.md`](OPERATING_DECISION_PHASE_D_PLAN.md).
The replacement disposable rehearsal, repeated-computation check,
representative archive resource probe, full exact-head suite, and CI must pass
before a tuning command may open `p1_development_tuning`. The 10-block internal
check, independent calibration, and reserved evaluation remain prohibited.
