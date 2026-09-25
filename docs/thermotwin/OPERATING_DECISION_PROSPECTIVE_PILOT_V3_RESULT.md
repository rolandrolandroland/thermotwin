# Prospective disposable pilot v3 result

Date run: 2026-09-19. Independently replayed: 2026-09-25. Scientific use:
disposable engineering evidence only. Status: the parent archive and required
all-case N=32 continuation are valid, and the frozen engineering gate failed.
Phase D is not authorized.

## Provenance and gate chronology

The source and audit closeout commit was
`2f906a22a9a79aa7b718cb5449005561e4c28080`. GitHub Actions run
[`35421074789`](https://github.com/rolandrolandroland/thermotwin/actions/runs/35421074789)
completed successfully on that exact commit before either P3 namespace was
interpreted.

The parent partition was
`p3_disposable_archive_roundtrip_replacement_pilot`. It generated 16 draws for
all 12 cases and authenticated the N=4, N=8, and N=16 prefixes. N=16 was the
smallest passing prefix under the frozen strict rule, with zero pipeline
failures and zero N=16 selection failures. That result triggered the
predeclared all-case continuation
`p3_disposable_archive_roundtrip_replacement_pilot_n32_all_cases_v1`.

The continuation retained the same four blocks and all three truth families.
Its N=16 prefixes and case identities matched the parent archive exactly.

## Preserved artifacts

The ignored local artifacts remain in
`.artifacts/operating-decision/prospective-pilot-v3/`. Their committed record is
the hashes and results below; the large JSON files are not added to Git.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `pilot.json` | 14,246,864 | `1d07f7a79b9716138088a883f5192229fe164f7f81e4b9c0bfca9115a37069aa` |
| `pilot.txt` | 3,288 | `2d3458055fa6f84cad274fde8d30a3927f8ffcb653e229ad21097c4740000213` |
| `n32.json` | 31,578,806 | `eb4c1a4155d3fc47f445bfab79109d8e4fecdd851021d310ce4f9a78f680566a` |
| `n32.txt` | 1,530 | `dfba12ba7cd3efacfe8c4fc83b04ff304827de00f822447f6db1eb7caa6521c8` |

On 2026-09-25, CPython 3.10.12 independently loaded the saved JSON and ran
`validate_prospective_pilot_archive` and
`validate_prospective_n32_followup_archive`. Both passed, including canonical
byte length, content digests, source and protocol bindings, case identities,
authenticated prefixes, stream inventories, fit invariants, and recomputed
acceptance records. This replay does not repeat the simulations or candidate
fits.

## Parent P3 result

| Prefix | Action agreement with strict N=16 | Maximum changed-choice regret | Gate |
| --- | ---: | ---: | --- |
| N=4 | 9/12 (75.0%) | 54.9% | Fail |
| N=8 | 11/12 (91.7%) | 39.7% | Fail |
| N=16 | 12/12 (100.0%) | 0.0% | Conditional pass; N=32 required |

All 960 strict N=16 predictive draws were retained. There were no pipeline
failures, whole-draw failures, selection failures, or ineligible N=16 action
records. Candidate-level changes remained common but conservatively scored:
296 losses and 70 recoveries. Allowing one whole-draw failure at N=16 changed
none of the 12 choices.

The run used four workers, took 7,190.28 wall seconds and 23,355.18 aggregate
block CPU seconds, and observed a 47,431,680-byte peak worker RSS. At N=16, the
measured projection for the 230 planned post-pilot blocks was 114.75 wall hours
and 372.73 CPU hours before additional sensor-quality simulations.

## Required N=32 result

The authenticated N=16-to-N=32 comparison agreed in 11/12 cases (91.7%). The
single changed case was block 0 under `extra_interface_mass`: N=16 selected
face temperature, while strict N=32 selected thermal. The change was caused by
an eligibility discontinuity, so its utility regret was not evaluable.

In that case, draw 19 for the five-state source and face-temperature action had
no admissible refit: both four- and five-state candidates reached a bound. The
failure remained in the denominator and received the frozen no-gain baseline
width. It was the only whole-draw failure in the N=32 evidence, but the strict
zero-failure eligibility rule made the entire face-temperature action
ineligible. There were no pipeline failures or selection failures.

The N=32 run used four workers, took 14,864.30 wall seconds and 49,467.00
aggregate block CPU seconds, and observed a 56,770,560-byte peak worker RSS.

## Conclusion and next permitted work

The final pilot engineering gate failed because the changed choice had an
eligibility difference and therefore an unevaluable regret. No draw count or
compute budget is frozen, and development, calibration, and reserved
partitions remain closed.

The failure is evidence that the zero-failure eligibility cutoff is too
discontinuous for the already-conservative failure score. It is not evidence
of a physics, leakage, random-stream, archive, or source-provenance defect.

The completion protocol permits one bounded scientific redesign. The next
permitted work is to version an explicit nonzero whole-draw allowance, retain
the no-gain imputation and complete denominator, and allocate a fresh
disposable namespace to test that rule. The P3 and N=32 outcomes may motivate
that engineering design, but they cannot serve as its acceptance evidence. If
the fresh bounded revision remains unstable, the prospective estimator must be
reported as not ready and the project should close with a feasibility result
instead of opening development.
