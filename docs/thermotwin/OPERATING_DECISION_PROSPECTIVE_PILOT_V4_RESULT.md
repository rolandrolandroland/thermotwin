# Prospective disposable pilot v4 result and Phase C freeze

Date completed: 2026-09-26. Scientific use: disposable engineering evidence
only. Status: the P4 parent and required all-case N32 continuation validate and
pass the frozen engineering gate. Phase C is complete. N=16 and its one-failure
eligibility allowance are frozen for Phase D development.

## Provenance and chronology

The P4 source and protocol commit is
`c0518f5885f8421a1f6f95d4f60c5dc5744cfb4a`. GitHub Actions run
[`36169289512`](https://github.com/rolandrolandroland/thermotwin/actions/runs/36169289512)
completed successfully on that exact commit before P4 opened.

The parent partition `p4_disposable_bounded_instability_pilot` generated 16
draws for all 12 cases and authenticated N=4, N=8, and N=16 prefixes. N=16 was
the smallest passing prefix. The parent had zero pipeline failures, selection
failures, ineligible N=16 actions, and whole-draw failures, so it triggered the
predeclared all-case continuation
`p4_disposable_bounded_instability_pilot_n32_all_cases_v1`.

The continuation retained all four blocks and three truth families. Its N=16
prefixes and case identities matched the parent exactly. Independent CPython
3.10.12 load-and-validate replay passed for both saved JSON archives.

## Preserved artifacts

The large artifacts remain ignored under
`.artifacts/operating-decision/prospective-pilot-v4/`. Their immutable record
is:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `pilot.json` | 15,395,653 | `91f7d21c9e3a72bb8f341efc78ff7ce0a1a570fef0ff8540ed998e1655770bad` |
| `pilot.txt` | 3,301 | `84b4b23232b31d3dc2ad5ddb6f20ddf70fbefa27fbe97ae216c41d68010fbac5` |
| `n32.json` | 34,992,556 | `749e6896bbd90503faf75c6deaffd24a5ede63baf5e6e15fa84e2d08c12f66cf` |
| `n32.txt` | 1,597 | `06b1afe892f199c6253f5be787f164e347c402438c9f14ed37d542307d7831a4` |

The parent protocol digest is
`5316dd5383001955c93b753748a8301f7f4a404709eec90b31bd6a6ca3f51864`,
and its scientific-result digest is
`ef07528b0fee6da440090f357b97845399b84aaddeaccaf0db30393204a97f00`.
The N32 protocol digest is
`8ec7b1c0bb0529ff5b02fd8bd98e24f57e9171880e7beea86aab2e8e016f1ec3`,
and its scientific-result digest is
`0845d230b3de5d4678bc2c23995a5d721f91d449b4eb1bdc643b326e5b9a3e7e`.

## Parent result

| Prefix | Agreement with primary N=16 | Maximum changed-choice regret | Gate |
| --- | ---: | ---: | --- |
| N=4, zero failures allowed | 10/12 (83.3%) | 100.0% | Fail |
| N=8, zero failures allowed | 11/12 (91.7%) | 100.0% | Fail |
| N=16, one failure allowed | 12/12 (100.0%) | 0.0% | Conditional pass; N32 required |

All 1,104 primary N=16 predictive draw records were retained. There were 378
candidate-loss transitions, including 377 bound hits and one nonconvergence,
but no whole-draw failures. Candidate transitions kept the no-gain floor and
did not invalidate an action when another candidate supplied an interval. The
zero-failure N=16 sensitivity selected the same action in all 12 cases.

## Required N32 result

The authenticated N=16-to-N=32 comparison agreed in 12/12 cases. There were no
changed choices, pipeline failures, selection failures, unavailable draw
diagnostics, whole-draw failures, or ineligible N=32 action records. The
maximum changed-choice regret was 0.0%. The complete engineering gate passed
and selected N=16.

| Block | Truth family | N=4 | N=8 | N=16 | Zero-failure N=16 | N=32 |
| ---: | --- | --- | --- | --- | --- | --- |
| 0 | Matched four-state | Thermal | Thermal | Thermal | Thermal | Thermal |
| 0 | Extra interface mass | Thermal | Thermal | Thermal | Thermal | Thermal |
| 0 | Temperature-dependent contact | Face temperature | Face temperature | Face temperature | Face temperature | Face temperature |
| 1 | Matched four-state | Stop | Stop | Stop | Stop | Stop |
| 1 | Extra interface mass | Stop | Stop | Stop | Stop | Stop |
| 1 | Temperature-dependent contact | Stop | Stop | Stop | Stop | Stop |
| 2 | Matched four-state | Stop | Stop | Stop | Stop | Stop |
| 2 | Extra interface mass | Stop | Stop | Stop | Stop | Stop |
| 2 | Temperature-dependent contact | Stop | Stop | Stop | Stop | Stop |
| 3 | Matched four-state | Stop | Stop | Thermal | Thermal | Thermal |
| 3 | Extra interface mass | Stop | Stop | Stop | Stop | Stop |
| 3 | Temperature-dependent contact | Voltage | Face temperature | Face temperature | Face temperature | Face temperature |

This distribution is engineering evidence about Monte Carlo stability. It is
not evidence that stopping, thermal testing, or face-temperature testing is
correct for these cases; truth-based decision quality remains unopened.

## Runtime and compute freeze

The parent used four workers and took 7,175.09 wall seconds, 26,360.46
aggregate block CPU seconds, and 46,301,184 bytes peak worker RSS. The N32
continuation took 14,612.75 wall seconds, 51,370.77 aggregate block CPU
seconds, and 54,968,320 bytes peak worker RSS.

Phase D freezes the primary prospective calculation at N=16 with at most one
whole-draw failure per source/action. Every failed draw remains in the
denominator and receives the pre-action no-gain width. Every reference-count
measurement action must remain eligible. This is the Phase D starting rule.
The completion plan's predeclared development sensitivity may increase N under
a committed decision rule before the internal check; no outcome-directed
decrease or unversioned eligibility change is permitted.

For the 230 currently planned post-pilot blocks, the conservative phase-by-
phase projection is 412,575.79 wall seconds (114.60 hours) at four workers,
1,515,726.27 CPU seconds (421.04 CPU hours), 185,204,736 bytes (176.63 MiB)
concurrent peak RSS, and 885,104,515 bytes (844.10 MiB) of primary archives.
This includes all fixed comparators and verification. Cost-only map cells reuse
authenticated predictions. Additional sensor-quality simulations are not
included and require a separately committed Phase D protocol and budget before
they run.

The machine-readable freeze is
`operating_decision_prospective_phase_c_freeze_v1`, with payload digest
`f9fd7d6760549f2b69ca4eac8fecabc3fc5783f3d2309e0d20fd527cd13830a8`.

## Conclusion and next permitted work

Phase C passes as an engineering stability and feasibility result. It does not
establish decision accuracy, interval coverage, selector benefit, or hardware
validity. The only newly authorized scientific partition is
`p1_development_tuning`. Before it opens, Phase D must commit its tuning grid,
support-count fallback, exact nominal and sensor-quality scenario catalog,
draw-sensitivity subset, measurement-map outputs, and incremental budget.
Internal check, calibration, and reserved partitions remain closed.
