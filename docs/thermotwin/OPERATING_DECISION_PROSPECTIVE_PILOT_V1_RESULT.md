# Prospective disposable pilot v1 result

Date run: 2026-09-17. Scientific use: disposable engineering evidence only.
Status: complete, integrity-verified, engineering gate failed, superseded for
the next pilot by the candidate-exclusion eligibility repair.

## Provenance and integrity

```text
campaign: operating_decision_prospective_v1_2026_09
partition: p1_disposable_draw_count_pilot
source revision: e32d091a3e55387417ad5c04f76a3399d7a16727
protocol digest: 6e7db6eced2b0f67a9f5c3d99c5326d2c286309919da572bcf2410d9b88865a2
scientific result digest: a7995b18ebcb398869c590d1fe3e0847831473a7ee905e01bb18143acef98853
JSON SHA-256: af6760010c5a34e77f608b4b9a6210c3fbf45c4851f49ebae5c8a2589d997b57
JSON bytes: 15519179
```

The detached SHA-256 manifest validates the JSON and compact report. An
independent reconstruction also validates the campaign and partition, all 12
case identities, source revision, protocol digest, recomputed acceptance
record, scientific-result digest, archive-content digest, and exact archive
size. The source checkout remained clean at the recorded revision before and
after execution. All corrected and prospective random-stream audits report no
unintended key reuse or derived-seed collision.

The complete JSON is an ignored local engineering artifact under
`.artifacts/operating-decision/prospective-pilot-v1/`. It is identified here by
content hash and source revision; this document is not a substitute for the
raw archive in the final reproducibility deposit.

## Declared gate result

The runner generated 16 predictive draws for every action, source candidate,
and case, then evaluated authenticated `N=4`, `N=8`, and `N=16` prefixes. Under
the v2 eligibility rule:

| Prefix | Agreement with strict N=16 | Largest changed-choice regret | Result |
| ---: | ---: | ---: | --- |
| 4 | 11/12 (91.7%) | Not evaluable for one eligibility-driven change | Fail |
| 8 | 11/12 (91.7%) | Not evaluable for one eligibility-driven change | Fail |
| 16 | 12/12 (100%) | 0% | Stability-only pass |

The overall engineering gate failed. Strict N=16 had three selection failures,
32 ineligible action records, and no accepted draw count. Allowing one unstable
draw at N=16 did not change any selected action or recover the failures.

## Failure diagnosis

There was no predictive simulation, refit-execution, uncertainty, provenance,
or pipeline failure. All 1,152 declared predictive draw records completed and
retained at least one admissible candidate. The gate failed because uncertainty
v2 treated loss of any initially admissible candidate as both:

1. a conservatively scored transition, with its width floored at the pre-action
   baseline; and
2. a whole-draw instability that could invalidate the entire action.

There were 422 such completed candidate-transition draws: 59 thermal, 188
voltage, and 175 face-temperature. Of these, 420 were candidate-level bound
hits and two were candidate-level nonconvergence; another candidate remained
admissible in every draw. The dominant pattern was four-state-generated data
excluding the five-state candidate after the hypothetical added measurement
(406/422 transitions). Consequently voltage and face temperature were
ineligible in all 12 cases, thermal was ineligible in 8/12, and three unresolved
cases had no eligible acquisition action.

This is a rule defect exposed by the pilot. It recreates the earlier
whole-case exclusion problem at action level. Candidate attrition is already
prevented from creating apparent information gain by the baseline-width floor;
invalidating the action applies a second penalty. The bounded repair keeps
bound hits and nonconvergence as candidate-level exclusions, keeps transition
draws in every denominator with the conservative floor, and reserves
whole-draw instability for cases with no usable envelope or a true upstream
failure.

An artifact-only diagnostic under that revised definition is not acceptance
evidence, but it checks the repair's expected effect. It makes all 36 current
action records usable. N=8 then agrees with N=16 in 12/12 cases with zero
regret. N=4 agrees in 9/12, has mean normalized regret 12.7%, and has maximum
changed-choice regret 75.0%. A fresh partition must confirm the revised rule.

## Measured resource result

The four-worker run took 6,777.18 seconds wall time (1.88 hours), 26,398.70
aggregate CPU seconds, and at most 46.8 MB RSS per worker. The compact JSON was
15.5 MB. Predictive N=16 work accounted for nearly all runtime.

At the measured throughput, all 230 planned post-pilot blocks project to about
30.28 wall hours at N=4, 56.24 hours at N=8, or 108.18 hours at N=16 on four
workers, before any extra sensor-quality simulations. These estimates include
retained fixed-policy verification and final scoring plus measured artifact
serialization and writing. The replacement pilot must choose the smallest
stable draw count and the later design must reduce optional sensitivity work
before compromising the primary paired comparison.

## Next gate

Development remains closed. The only permitted data-generating step is the
fresh four-block `p2_disposable_candidate_exclusion_pilot`, after the versioned
repair, strengthened P2/N32 archive validators, tests, updated protocol records,
and new namespace are committed and pushed with passing CI. These are ordinary
pre-P2 defect repairs, not evidence that the replacement rule succeeds. If
that pilot does not establish a usable draw count within the declared budget,
the experiment must stop with the protocol's feasibility limitation rather
than opening development.

If N=16 is the smallest passing P2 prefix and P2 otherwise has zero pipeline
failures and zero N=16 selection failures, the predeclared follow-up uses all same 12 cases,
generates N=32 once, and authenticates its N=16 prefix against P2. It must meet
at least 90% agreement, at most 5% maximum normalized utility regret, and zero
N=32 pipeline or selection failures before a draw count can be frozen.
