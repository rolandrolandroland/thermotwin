# Blinded measurement selection for an operating decision

Status: Stages 1–5 produced a baseline, realism stress test, calibrated
stop-or-voltage selector, mismatch-guard revision, and one reserved synthetic
evaluation. A September 2026 audit found deterministic random-stream reuse
across runs and adjacent device blocks. The stored counts remain descriptive,
but their conformal and bootstrap guarantees are superseded. The audit also
confirmed that the implemented selector is narrower than the cost-dependent
choice among thermal, voltage, and face-temperature packages specified below.
See [`OPERATING_DECISION_AUDIT.md`](OPERATING_DECISION_AUDIT.md). A versioned,
collision-free replication must be completed before the original selector
hypothesis receives a new final evaluation.

## 1. Problem and research question

A thermoelectric device can produce apparently well-predicted exchanger
temperatures while its inferred contact resistance and internal temperatures
are wrong. A team therefore needs to know whether another temperature-only
test, a voltage measurement, or a face-temperature measurement provides enough
evidence to approve a previously untried operating pulse.

**Can ThermoTwin select a useful measurement package and correctly approve,
reject, or withhold judgment on that pulse when its physical model may be
incomplete?**

Everything is simulated: the devices, sensors, diagnostic experiments, and
final operating experiment. The result concerns virtual test effort and
decision reliability under declared assumptions. It does not establish actual
prototype savings or a hardware safety certification.

## 2. Hypothesis and scope

The hypothesis is that selecting a measurement for its expected effect on the
operating decision can reduce incorrect approvals or diagnostic effort
relative to a fixed measurement package.

The hypothesis is allowed to fail. Always measuring voltage, always measuring
face temperature, or running a fixed thermal test sequence might perform as
well as or better than selection.

The first version makes one measurement-package selection after a common
initial experiment. This deliberately bounds the work. It does not claim a
globally minimal campaign or require a new general sequential optimizer.

Use conventional nonlinear inference for all primary comparisons. A PINN
comparison can follow if the measurement-selection result establishes a useful
baseline; it is not required to finish this experiment.

## 3. The decision being predicted

Choose one fixed final current schedule and a fixed acceptable cold-face
temperature band, `[T_min, T_max]`. Predict whether the entire final trajectory
stays within that band.

Define the operating margin in kelvins:

```text
m = min over time of min(T_cold_face(t) - T_min,
                        T_max - T_cold_face(t))
```

- `m >= 0`: the virtual device meets this temperature requirement.
- `m < 0`: it violates the requirement.

The requirement is synthetic and applies to the specified schedule. It is not
a complete cooling-product specification. The pulse is supplied to every
policy; a policy cannot make the problem trivial by selecting zero current or
changing the final schedule.

Use development devices to choose a schedule and band that produce both
passing and failing cases, including cases near the boundary. Freeze them
globally before final evaluation. Never set a different threshold from each
evaluation device's hidden truth. Report the resulting pass/fail prevalence.

## 4. Virtual devices and candidate models

Use three separately reported device families:

| Family | Data-generating device | Purpose |
| --- | --- | --- |
| A | Four-state device | Check that extra complexity is not automatically preferred. |
| B | Five-state device with interface thermal storage | Revisit the established contact/storage ambiguity. |
| C | Five-state device with temperature-dependent thermal contact resistance | Test cases that neither fitted model represents exactly. |

Fit the existing four-state and five-state models, with constant thermal
contact resistance in both. Share the inference protocol across policies.
Estimate contact resistance, face thermal capacitance, sensor lag, and the
five-state interface mass where applicable. Treat offsets and added-channel
calibration parameters as nuisance quantities with declared priors or bounds.

Begin development with the earlier synthetic parameter scales: nominal contact
resistance 0.25 K/W, cold-face capacitance 50 J/K, temperature-sensor lag 1.5 s,
and interface mass 20 J/K. The earlier study used log standard deviations of
0.18 for the first three quantities and 0.25 for interface mass. These are
starting assumptions, not measured manufacturing distributions.

Family C uses a positive temperature-dependent contact law. Choose and freeze
its strength range during development. Do not silently add that exact law to
the candidate models after observing evaluation failures.

The truth generator must have separately verified energy balances and
time-step convergence. Computational independence reduces shared numerical
errors; it does not turn synthetic truth into experimental evidence.

## 5. Measurements and sensor realism

Every policy initially receives known current and the two exchanger
temperature histories. Use the existing 1 s sampling and 0.02 K temperature
noise as starting assumptions, with run-specific constant offsets and sensor
lag. Retain shared noise for genuinely common channels in paired comparisons.

Add two focused realism changes:

1. **Voltage:** uncertain electrical series/contact resistance and voltage
   offset. Locate the associated Joule heating consistently in the energy
   model. The extra resistance must not exist only as an arbitrary voltage
   correction.
2. **Face temperature:** a separate sensor response time and uncertain added
   thermal mass. Installation changes the diagnostic device, so simulate and
   model that change explicitly.

The final operating target is the original device without the temporary face
sensor. A policy using that sensor must infer from the loaded diagnostic
device and forecast the unloaded target. It receives a calibration range for
sensor loading, not its true value. Assume removal has no permanent effect in
this first study and state that limitation.

This keeps the final decision comparable across packages. Final face
temperature remains unavailable to every policy until scoring.

## 6. Policies and available actions

All policies receive the same initial pulse and measurements. Use the existing
0.8 A, 20 s pulse as the initial development choice, subject to the frozen
virtual constraints.

| Policy | Action after the initial pulse |
| --- | --- |
| Stop-now baseline | Collect no additional fitting data. |
| Fixed thermal package | Run three additional preselected temperature-only pulses. |
| Fixed voltage package | Repeat a declared pulse with temperature and voltage observations. |
| Fixed face-temperature package | Install the temporary sensor and repeat a declared pulse. |
| Decision-directed selector | Choose one of the above actions from the initial observations. |

New channels cannot be obtained retroactively from the initial pulse. Adding a
sensor requires a new simulated run.

For the first selector, predict possible observations under the current
plausible models and parameter estimates. Estimate how each package would
reduce uncertainty in the final operating margin. Choose the best expected
reduction per declared diagnostic cost, or stop if the provisional decision
is already adequately resolved. Freeze the scoring rule, tie-breaks, and cost
weights before evaluation. Synthetic future observations used for this
calculation come from the fitted models, never from the hidden device.

All policies also receive one fixed verification schedule after acquisition.
Including verification, the menu uses two, three, or five diagnostic runs.
The final evaluation run is separate and disclosed as such.

Do not call these packages equal-cost. Record run count, simulated bench time
including resets, electrical energy, sensor additions, and computation time.
Repeat the selector across a small, predeclared range of sensor/test cost
ratios to expose when its recommendation changes. Financial costs remain
assumptions until actual instrumentation information exists.

## 7. Separate the data twice

### Within each virtual device

1. **Acquisition data:** the common pulse plus the selected package; used for
   parameter fitting and prospective measurement scoring.
2. **Verification schedule:** a different excitation, used once after
   acquisition to assess candidate adequacy and model ambiguity. Do not refit
   physical parameters or per-run offsets to improve its score. Predict new
   offsets from the declared observation model.
3. **Final operating schedule:** unseen responses, revealed only after the
   decision and forecast have been saved. Its known current command is an
   allowed input to the forecast.

If verification fails, return insufficient evidence. Do not keep acquiring
data until this same verification set passes. A later sequential extension
would need a separately specified validation protocol.

### Across devices

- **Development devices:** choose the schedule, band, truth ranges, fitting
  settings, and affordable uncertainty method.
- **Calibration devices:** set decision and discrepancy thresholds after the
  development choices are fixed.
- **Evaluation devices:** report the frozen comparison; no tuning or failure
  exclusions.

Calibration may use hidden margins on synthetic calibration devices. That is
an explicit simulation-derived calibration assumption and does not establish
calibration on hardware or on a new discrepancy family.

## 8. Uncertainty and decision rules

Use multiple starting points for each conventional fit. Propagate parameter
and nuisance uncertainty to the scalar operating margin. A bounded
parametric-bootstrap implementation is the initial candidate; choose its
replicate count through development runtime and stability checks.

Keep every candidate model that passes the fixed verification checks. Use the
envelope of their margin intervals rather than converting a small difference
in fit scores into an unjustified model probability. Calibrate the interval
construction on the separate calibration devices. Report its empirical
coverage on evaluation devices, separately by family.

For a saved margin interval `[L, U]`:

- **Approve:** verification passes and `L >= 0`.
- **Reject:** verification passes and `U < 0`.
- **Insufficient evidence:** the interval crosses zero, no candidate passes,
  or fitting/uncertainty checks fail.

Reject means a predicted violation of the declared temperature band.
Insufficient evidence means the software cannot support either conclusion.
Local parameter covariance alone must not be used as evidence that missing
physics has been excluded.

No finite set of visible-data checks is guaranteed to expose every omitted
effect. A well-fitting model that still makes a wrong operating decision is a
benchmark failure to retain, not an excuse to inspect hidden truth earlier.

## 9. Scoring and success criteria

Primary outputs:

| Metric | Definition |
| --- | --- |
| False approval fraction | Approvals with true `m < 0`, divided by all approvals. |
| Missed-violation fraction | Approvals with true `m < 0`, divided by all truly violating devices. |
| False rejection fraction | Rejections with true `m >= 0`, divided by all rejections. |
| Decision coverage | Approvals plus rejections, divided by all devices. |
| Abstention fraction | Insufficient-evidence outcomes divided by all devices. |
| Diagnostic effort | Runs, time, energy, instrumentation, and computation used before the decision. |

An empty denominator is reported as N/A with its count, never as zero error.
Always show approval and rejection counts separately so approving nothing
cannot masquerade as a strong approval policy.

Supporting outputs are operating-margin error, predictive-interval coverage,
cold-face trajectory error, parameter error, and model-family selection where
a correct family is available. Model-family accuracy is not a pass criterion
for Family C.

The main hypothesis is supported if the selector reduces false approvals at
comparable decision coverage and resource use, or reduces diagnostic effort
without a material loss of decision quality, relative to the fixed policies.
Compare against the strongest fixed alternative, not only stop-now.

A proposed development target is no more than 10% false approvals with at
least 70% decision coverage. These are benchmark targets, not hardware
requirements or established performance. Freeze the final thresholds and
non-inferiority tolerance before evaluation. Report counts and uncertainty
intervals; do not claim a reliable rate merely because a small sample happens
to clear a target.

Report every family separately. Also plot risk against decision coverage for
a small, frozen set of decision thresholds. Do not choose a favorable
threshold after examining evaluation labels.

## 10. Trial plan and implementation stages

1. **Reproduce the foundation.** Confirm the documented sensor-discrimination
   baseline and its report before extending it. At outline creation, the
   supporting adaptive and sensor studies are in the existing `dev` worktree;
   the current main checkout does not contain them. Do not silently switch,
   merge, or overwrite either checkout as part of this planning task.
2. **Implement the operating margin and splits.** Use fixed policies first;
   verify that final responses cannot enter fitting, selection, or calibration.
3. **Add the two sensor effects and Family C.** Check conservation, limiting
   cases, sensor-removal transfer, and numerical convergence.
4. **Run development devices.** Start with approximately 10 devices per family
   to find numerical failures, check runtime, and ensure a nontrivial decision
   boundary. These runs are not final evidence.
5. **Add the selector and calibrate.** Freeze the candidate menu, uncertainty
   construction, cost scenarios, and thresholds on separate devices. Reserve
   at least 20 calibration devices per family, expanding before evaluation if
   approvals or violations are too sparse to calibrate meaningfully.
6. **Freeze and evaluate.** Target 50 fresh devices per family: 150 paired
   evaluation devices. Every policy sees the same underlying device and common
   noise where applicable. Decide the final sample size from development
   runtime before revealing evaluation results. Keep failures and abstentions.
7. **Produce the report.** Include the frozen configuration, source revision,
   device-level outputs, all failure cases, and reproducible figures.

A fit crash is an insufficient-evidence outcome and a separately reported
numerical failure. Unexpected truth-solver failures require repair and a
versioned full rerun of the affected frozen campaign, not quiet deletion of
inconvenient devices.

## 11. Deliverables

- **Decision figure:** two plausible fits to initial observations; their
  different internal forecasts; the chosen measurement; and the untouched
  operating trajectory revealed after the saved decision.
- **Comparison figure:** false approvals versus decision coverage, accompanied
  by runs, energy, and instrumentation used by each policy.
- **Measurement map:** which package is selected as sensor quality, loading,
  or assumed instrumentation cost changes; include regions where no tested
  package supports a decision.
- **Written report and machine-readable results:** exact protocol, paired
  outcomes, uncertainty, null results, failures, and scope limits.

Choose the demonstration case by a frozen selection rule or label it clearly
as an illustration. Cohort results, rather than one attractive trajectory,
carry the comparison.

The strongest defensible outcome would be evidence that a computational
workflow chooses useful measurements and supports better operating decisions
within this declared virtual benchmark. If a fixed package wins, if sensor
loading erases an apparent advantage, or if all policies remain confidently
wrong for Family C, those are substantive outcomes to report.
