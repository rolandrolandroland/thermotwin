# Prospective four-action selector

Status: roadmap Step 1 is implemented as a development interface. No new
scientific partition has been generated or opened. The completed corrected
`r2` replication, its artifacts, and its procedure names remain unchanged.
Steps 2–7 are still pending.

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
`default_fixed_policies()`. It gives the new procedure the distinct name
`prospective_four_action_selector_v1`; the historical
`decision_directed_selector` continues to mean the completed stop-or-voltage
rule.

## Information boundary

`build_prospective_acquisition_snapshot` accepts only the acquisition fit set,
the known final current regime, and the frozen physical configuration. Its
saved snapshot contains candidate status and provisional margin intervals. It
has no truth-family label, device identity, verification result, selected
action outcome, true margin, or final response.

The fit-set type does not itself prove which observations were used to produce
the fits. Step 1 therefore treats fit provenance as unauthenticated alongside
the action scores. Before any named partition is opened, later orchestration
must bind the fit record to the common initial acquisition data and reject fits
that used an added-action, verification, or final response.

Action scores are deliberately separate from this acquisition-only snapshot.
Step 1 accepts a complete scorecard so the selection mechanics can be tested,
but does not calculate or authenticate those scores. Step 2 must create them
inside an acquisition-only prospective calculation and bind the evidence to
that calculation's protocol before the selector can be used in a scientific
campaign.

## Candidate reliability

Every frozen candidate model must be accounted for as fitted, numerically
failed, or excluded. A bound hit or nonconvergence excludes only that
candidate. A fit exception remains a case-level selection failure. An
uncertainty-propagation exception for an otherwise admissible candidate also
remains case-fatal. Failed snapshots cannot retain partial intervals or a
provisional envelope.

This preserves the corrected replication rule that recovered Family A
coverage: one unreliable candidate does not discard another reliable
candidate.

## Selection rule

The selector first applies a stopping clearance to the provisional margin
envelope. If the envelope is decisively positive or negative, it selects
`stop_now`. Stop means no added fitting run; the common verification schedule
is still required before an approve/reject decision.

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

The Step 1 rule has strict JSON serialization and a canonical SHA-256 protocol
digest over the action packages, action order, thresholds, precision, and
procedure and algorithm identities. The algorithm identity commits to the stop
gate, strict value thresholds, and four-key ranking semantics. Semantically
identical numeric inputs such as `0`, `0.0`, and `-0.0` normalize to the same
digest.

This digest freezes the selection mechanics only. A future scientific freeze
must also bind the uncertainty estimator, prospective sampling plan, cost
scenario, generator, source manifest, development evidence, and calibration
evidence.

## Next roadmap step

Step 2 will estimate each action's expected reduction in final operating-margin
uncertainty from synthetic future observations generated under the admissible
fitted models. That calculation must use acquisition information only, use
independent prospective random streams, preserve both candidate models, and
produce one shared pre-action uncertainty baseline for all three measurement
actions. Frozen resource costs are introduced in Step 3.
