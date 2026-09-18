# Prospective four-action selector

Status: the reviewed `9a21aa7` baseline implements roadmap Steps 1–3 as
development interfaces. Phase B versions the active selector as
`prospective_four_action_selector_v2` to add development-padded scoring and
explicit stop strata. The first disposable pilot is complete and exposed an
eligibility defect in prospective uncertainty v2. The bounded uncertainty v3
repair, its complete pre-P2 validation, and strengthened P2/N32 archive
validators are recorded in this source revision. These are ordinary pre-P2
defect repairs. A fresh disposable pilot may open only from the committed and
pushed revision after its CI passes. Development, independent calibration,
reserved evaluation, and the final report remain pending.
See the [current project status](OPERATING_DECISION_PROJECT_STATUS.md),
[Phase B acceptance record](OPERATING_DECISION_PHASE_B_ACCEPTANCE.md), and
[partition ledger](OPERATING_DECISION_PROSPECTIVE_PARTITION_LEDGER.md).

## Purpose

The prospective experiment will choose one action after the common initial
acquisition:

| Action | Added acquisition | Added sensor |
| --- | ---: | ---: |
| `stop_now` | 0 runs | 0 |
| `fixed_thermal` | 3 runs | 0 |
| `fixed_voltage` | 1 run | 1 |
| `fixed_face_temperature` | 1 run | 1 |

The implementation reuses the exact packages returned by
`default_fixed_policies()`. The active Phase B procedure has the distinct name
`prospective_four_action_selector_v2`; `prospective_four_action_selector_v1`
identifies the audited development baseline at `9a21aa7`, and the historical
`decision_directed_selector` continues to mean the completed stop-or-voltage
rule.

## Information boundary

`build_prospective_acquisition_snapshot` accepts only the acquisition fit set,
the known final current regime, and the frozen physical configuration. Its
saved snapshot contains candidate status and provisional margin intervals. It
has no truth-family label, device identity, verification result, selected
action outcome, true margin, or final response.

The Step 2 evidence factory closes the fit-provenance gap in the Step 1 data
type. It accepts one exact common-initial run, the known final current regime,
and the physical configuration. It verifies the initial regime, channels,
instrumentation, sample grid, and final regime; then it fits both candidates
internally with the declared three-start procedure. The resulting fit set,
snapshot, observations, final current, and physical-protocol digest are bound
by a canonical acquisition-evidence digest. A caller cannot substitute a fit
made from an added action, verification response, or final response.

Action scores remain separate from the acquisition snapshot. Step 2 calculates
authenticated, cost-free uncertainty scores. Step 3 now attaches the declared
energy, time, and instrumentation costs before the Step 1 selector ranks the
actions.

## Candidate reliability

Every frozen candidate model must be accounted for as fitted, numerically
failed, or excluded. A bound hit, nonconvergence, candidate-specific refit
exception, or candidate-specific forecast exception excludes only that
candidate when another candidate still supplies an admissible interval. A
failure before a usable envelope exists, or loss of every candidate, remains a
whole-draw or case-level failure as appropriate. Failed acquisition snapshots
cannot retain partial intervals or a provisional envelope. The evidence
factory still returns and serializes a failed acquisition record, so a later
campaign cannot omit it from its coverage denominator; uncertainty scoring
itself requires a usable baseline.

This preserves the corrected replication rule that recovered Family A
coverage: one unreliable candidate does not discard another reliable
candidate.

## Step 2 uncertainty calculation

The estimator starts from the full width of the Step 1 provisional
candidate-envelope interval. This is the single pre-action baseline for all
three measurement actions. For every candidate that is admissible after the
common acquisition, and for every predictive draw, it:

1. draws bounded, correlated log-parameters from that candidate's local fit
   covariance;
2. uses the same physical draw for the thermal, voltage, and face-temperature
   alternatives;
3. draws the temporary face probe's loading and response nuisance parameters
   independently for the face-temperature alternative, using the declared
   truth support of 1–12 J/K and 0.5–6 s with the prior spread rather than the
   wider inference bounds;
4. simulates the exact additional regimes in the existing fixed policy;
5. appends those synthetic observations to the real common-initial run;
6. refits both candidate models with the same frozen three-start multistart
   procedure; and
7. forecasts the unloaded final operating-margin intervals and records the
   full width of their conservative envelope.

The synthetic future data come only from acquisition-time fits. The estimator
does not accept a truth-family label, device token, trial index, verification
result, realized action outcome, true margin, or final response.

Within one source candidate, the expected post-action width is the arithmetic
mean over all declared draws. Across surviving source candidates, the estimator
takes the largest of those means and assigns no model probabilities. When only
one candidate survives the acquisition checks, this aggregation is necessarily
a single-model calculation; the result record and stop stratum must expose that
fact. The reported expected uncertainty reduction is the common baseline width
minus this worst-source expected width, so a harmful action can retain a
negative value.

## Numerical failures and candidate transitions

Every declared draw remains in the denominator. A sampling, simulation,
refit, or uncertainty-propagation failure receives the pre-action width rather
than being dropped. Bound hits and nonconvergence still exclude candidates
individually.

If a candidate that was initially admissible becomes inadmissible after a
hypothetical action, its scored width is the larger of its actual envelope
width and the pre-action width. Candidate loss therefore cannot look like
information gain. The transition remains explicit, but the draw remains usable
when another candidate supplies a valid envelope. Bound hits and
nonconvergence remain candidate-level exclusions; this rule does not accept the
excluded fit.

The first disposable pilot showed why the distinction matters. Uncertainty v2
also marked every candidate transition as whole-draw instability. All 1,152
draws completed, but 422 candidate exclusions made voltage and face temperature
ineligible in every case and caused three selection failures. Uncertainty v3
reserves an unstable draw for a true whole-draw failure, including sampling or
simulation failure, an exception that prevents a usable envelope, or no
admissible candidate. Candidate-transition counts are reported separately.

The development default is `N = 4` predictive draws. This is an implementation
and runtime setting, not the final scientific replicate count. The replacement
pilot must generate 16 draws for all 12 fresh cases and compare the
prefix-matched `N = 4`, `8`, and `16` results while reporting whole-draw
failures, candidate transitions, eligibility-driven changes, and regret. The
final scientific rule will freeze eligibility as the explicit pair `(N,
maximum unusable draws per source/action)` before a development partition is
opened. It will not infer that pair from an unlabeled percentage or inspect a
reserved cohort.

## Prospective random streams

Step 2 uses a new semantic random-stream namespace bound to campaign,
partition, block, source model, draw index, and acquisition-evidence digest.
Physical parameter draws are explicitly shared across the three alternative
actions as common random numbers. Probe parameters belong only to the face
action. Run-bias and white-noise streams are unique to one action, run, and
channel. The saved audit detects undeclared key reuse and derived-seed
collisions, and the estimator refuses a result when that audit is not clean.

## Step 3 resource accounting

The cost layer is incremental over the complete `stop_now` diagnostic plan.
The common initial acquisition cancels. The mandatory verification schedule
also cancels except when the face probe remains installed and changes its
nominal terminal energy. Every joule is calculated with the existing frozen
nominal selection proxy; hidden device truth and realized energy are not
accepted as inputs.

Raw resources remain visible beside the scalar cost:

| Action | Added runs | Incremental bench time | Added instrument | Incremental nominal energy |
| --- | ---: | ---: | --- | ---: |
| `stop_now` | 0 | 0 s | none | 0 J |
| `fixed_thermal` | 3 | 480 s | none | 38.5717 J |
| `fixed_voltage` | 1 | 160 s | terminal voltage | 28.8143 J |
| `fixed_face_temperature` | 1 | 160 s | cold-face temperature | 28.6909 J |

The development time model assigns one explicit 80-second reset assumption to
each added 80-second schedule. It is an assumption, not a simulated or measured
reset. Reset energy and sensor electronics are not in the terminal-energy
model. Voltage and face-temperature instruments remain distinct typed
resources even though each adds one channel. The raw-time sensitivity values
are predeclared as `0`, `80`, and `240` reset seconds per added run. Because
elapsed time and its reference use the same per-run reset, these values change
reported raw seconds and protocol identity but leave normalized time
`{0, 3, 1, 1}` and selection unchanged under this uniform-reset model.
Computation time is not captured by the Step 3 artifact; it must be reported
separately in Step 7.

The scalar cost uses dimensionless normalized resources:

```text
e = incremental nominal energy / voltage-action incremental energy
t = incremental bench time / one added run-and-reset slot
i = voltage instruments + m_face * face instruments
C = w_energy * e + w_time * t + w_instrumentation * i
```

The nonnegative weights sum to one and the energy and time weights cannot both
be zero. This makes the voltage action the unit-cost anchor and prevents raw
joules, seconds, and instrument counts from being added directly. The primary
development scenario is balanced: all three weights are `1/3` and `m_face =
1`. Its declared costs are `1.44621` for thermal, `1.0` for voltage, and
`0.998573` for face temperature. These are sensitivity units, not dollars.

Step 3 predeclares the Step 4 cost grid without applying it to a cohort. Four
resource mixes—balanced, energy-dominant, bench-time-dominant, and
instrumentation-dominant—are crossed with face-instrument multipliers `0.25`,
`1`, and `4`. The primary scenario appears exactly once in this 12-cell grid.
Changing these assumptions after seeing outcomes requires a new versioned
protocol.

An ineligible Step 2 action stays ineligible and receives no partial Step 1
score. Eligible actions copy the Step 2 uncertainty values and draw counts
exactly. Negative uncertainty reduction is retained, giving negative utility
per cost; it is never clipped into an apparent benefit. `stop_now` keeps zero
raw resources and remains outside the cost ratio.

## Phase B development padding

Selector v2 keeps the authenticated raw predictive widths and applies
versioned, nonnegative development offsets without rerunning the predictive
fits. For stop offset `d_stop` and action offset `d_a`, it uses:

```text
B = raw initial width + 2 * d_stop
P_a = worst-source mean of the per-draw padded future widths
value_a = (B - P_a) / declared cost_a
```

The offsets are development heuristics for selecting an action. They are not
the independent final calibration correction. Raw and padded values are both
serialized, and the selector rejects a scorecard whose offsets or raw baseline
do not match its rule.

For a stable draw, the padded future width is the raw future width plus
`2*d_a`. A failed draw receives exactly the padded baseline `B`. Candidate
loss receives the greater of `B` and the padded future width. These rules are
applied to each draw before the within-source mean and worst-source maximum;
adding an action offset therefore cannot turn failure or attrition into
apparent information gain. Cost-only or offset-only sensitivity can reuse the
expensive raw predictive evidence.

## Selection rule

The selector first pads the provisional margin envelope by `d_stop`, then
applies the general stopping clearance. A separate nonnegative clearance is
added when only one candidate remains. The output records the admissible model
names, one- versus two-candidate reliability stratum, padded envelope, and
effective clearance. If that gate is decisively positive or negative, it
selects `stop_now`. Stop means no added fitting run; the common verification
schedule is still required before an approve/reject decision.

For an unresolved envelope, eligible measurement actions are ranked by:

1. expected final-margin uncertainty reduction divided by declared cost;
2. expected uncertainty reduction;
3. lower declared cost; and
4. the frozen action order: thermal, voltage, then face temperature.

Utilities and reductions are rounded to the rule's declared decimal precision
before comparison. `tied_policies` records only a complete tie on quantized
utility, reduction, and cost, for which the frozen action order is decisive.
All eligible actions must use exactly the same uncertainty baseline from the
common acquisition. Reduction and value per cost are derived inside the
selector record rather than accepted as independent caller-supplied numbers.

If scored actions exist but none clears both minimum-value thresholds, the
selector chooses `stop_now` and still requires verification. If no acquisition
action has a usable score, or the acquisition snapshot itself failed, the
selector reports an explicit selection failure instead of silently counting it
as stop.

The output distinguishes an action that requires added acquisition from the
later mandatory verification phase. It also retains the full scorecard,
ranking, ties, provisional decision, and selection reason for audit.

## Reproducibility boundary

The selector-v2 rule has strict JSON serialization and a canonical SHA-256
protocol digest over the action packages, action order, thresholds, precision,
development offsets, general and single-candidate stopping clearances, and
procedure and algorithm identities. The Step 2 protocol digest additionally
binds the estimator, uncertainty metric, draw count, stability threshold,
within- and across-model aggregation, failure policy, candidate-attrition
policy, multistart refit procedure, prospective random-stream protocol,
physical protocol, and exact action packages. Its result digest binds every
draw, action summary, stream use, and stream audit to the authenticated common
acquisition. The JSON-ready record includes the normalized physical
configuration, common observations, acquisition fits, snapshot, final regime,
predictive draws, and random-stream audit. That record retains the inputs
needed for a deterministic source-bound replay. Archive validation itself
checks serialized structure, cross-record identities, the exact stream
inventory and RNG offsets, fit invariants, and interval formulas. It does not
rerun acquisition or predictive simulations, candidate refits, or post-reveal
scoring, so its digests bind the saved record rather than independently
authenticating those numerical calculations. The in-memory runner validates
the objects it produces before serialization; final scientific
reproducibility still requires replay from the bound source and retained raw
record.

The uncertainty-v3 behavior and strengthened P2/N32 validators are ordinary
pre-P2 corrections of demonstrated implementation and validation defects.
They do not constitute a scientific redesign or evidence from a new cohort.

The Step 3 protocol digest binds the nominal physical protocol, Step 2
protocol, energy convention, reset assumption, normalization references,
scenario weights, typed sensor burden, raw resource table, and scalar formula.
Its result digest binds one complete Step 2 result to every raw resource,
component cost, and selector-ready evaluation. The selection wrapper always
uses the acquisition snapshot contained in that same Step 2 result, preventing
a scorecard from one case being paired with another case's snapshot.
A later nondefault selector rule must be saved through the Step 1 rule payload
and bound by the Step 5 freeze; the Step 3 scorecard does not authenticate a
calibrated rule that has not yet been developed.

These are development protocol identities. A future scientific freeze must
also bind the final draw count, selected cost and sensor scenarios,
prospective generator, source manifest, development evidence, and calibration
evidence.

## Next roadmap step

The first four-block `p1_disposable_draw_count_pilot` is complete and failed its
engineering gate under uncertainty v2. After the bounded v3 repair, regression
checks, diagnosis, and new namespace are committed, the next data-generating
step, after final validation is recorded and the revision is committed,
pushed, and passes CI, is the fresh four-block, 12-case
`p2_disposable_candidate_exclusion_pilot`. It will measure complete-case
runtime, whole-draw failures, candidate transitions, action agreement, and
utility regret for prefix-matched `N = 4`, `8`, and `16` before fixing the
campaign draw count and worker budget.

If N=16 is the smallest passing prefix and P2 has zero pipeline failures and
zero N=16 selection failures, the predeclared follow-up generates N=32 for all
same 12 cases. Its authenticated N=16 prefix must agree with N=32 in at least 90% of
cases, have at most 5% maximum normalized utility regret, and have zero N=32
pipeline or selection failures. A passing pilot still supplies measured
planning evidence; the chosen draw count and compute budget must be frozen in a
separate committed record before development opens.

Only after the pilot passes may development produce the measurement map. The
map must reuse authenticated raw Step 2 evidence for cost-only changes, apply
the versioned Phase B development offsets without predictive refitting, retain
cells where no acquisition action is usable, and distinguish price changes
from sensor noise/loading changes that require new simulations. Its protocol
must bind the complete ordered cost, sensor-quality, and reset-sensitivity
catalogs rather than only the scenario used in one cell.
