# Operating-decision Phase B physics and data-flow acceptance

Date: 2026-09-17. Last reconciled: 2026-09-18.

Source basis: audited parent `9a21aa77284fb88608467a39a7ada8dc811107d5`; pilot-v1 source `e32d091a3e55387417ad5c04f76a3399d7a16727`; invalid P2 source `9db5f3f5b7a0fd92710ef5971091c16102877ac8`; P3 archive-transport repair in the current uncommitted working tree

Status: the original acceptance authorized disposable pilot v1. That pilot is
complete and exposed a candidate-exclusion eligibility defect. The unchanged
physics and information-boundary findings remain accepted; the v2 eligibility
row is superseded by the bounded uncertainty-v3 repair. P2 later executed, but
its saved JSON failed the required load-and-validate round trip. P2's gate was
not evaluated. The P3 representation repair is implemented in the current
working tree and passed local validation under CPython 3.10.12. Independent
clean-clone review, commit, push, and green CI remain pending; P3 is unopened.

## Conclusion

The declared synthetic physics and the existing information boundaries are internally consistent for the next disposable pilot. No material physics, energy-accounting, leakage, random-stream, or provenance defect was found, so this review made no change to the numerical model. Four cross-layer regression tests were added in `tests/test_operating_decision_phase_b_acceptance.py` for the remaining acceptance gaps.

This acceptance says that the code implements the declared synthetic model consistently. It does not show selector benefit, establish hardware validity, or validate the later calibration and reserved-evaluation designs.

## Post-pilot supersession

`p1_disposable_draw_count_pilot` completed under source `e32d091`. All 1,152
predictive draws completed, but uncertainty v2 counted 422 ordinary
candidate-level exclusions as whole-draw instability in addition to applying
the conservative no-gain width floor. The result invalidated otherwise usable
actions and caused three selection failures. This was not a heat-flow,
energy-accounting, leakage, random-stream, source, or solver-execution defect.

The replacement rule keeps the original candidate-level exclusion and
baseline-width floor, records every transition, and treats a draw as unusable
only when no candidate supplies an interval or an upstream failure prevents a
usable envelope. The detailed evidence is in
[`OPERATING_DECISION_PROSPECTIVE_PILOT_V1_RESULT.md`](OPERATING_DECISION_PROSPECTIVE_PILOT_V1_RESULT.md).
No development, calibration, or reserved partition may open until that
versioned repair and its regression checks are committed and the fresh
disposable replacement pilot passes.

The subsequent validator hardening is also an ordinary repair made before P2,
after adversarial review found that a resealed archive could preserve valid
digests while changing some saved evidence. Neither uncertainty v3 nor this
validator hardening is a favorable experimental result or the protocol's one
planned scientific redesign. No P2 case was generated while either repair was
being made.

## P2 archive-transport incident

P2 executed at source `9db5f3f`, and its output hashes identify the preserved
files. Independent loading and validation failed with
`ValueError: complete uncertainty protocol/config is invalid` because tuple
fields in the in-memory physical configuration loaded from JSON as lists. The
pre-P2 validation record below therefore did not close the final serialized
transport boundary. See the
[`P2 incident record`](OPERATING_DECISION_PROSPECTIVE_PILOT_V2_INCIDENT.md).

This incident does not reverse the accepted physics or information-boundary
checks. It invalidates P2 as engineering evidence. Its gate was not evaluated,
its outcome fields are quarantined, and its N32 continuation did not open.

## Acceptance matrix

| Boundary | Result | Evidence |
| --- | --- | --- |
| Heat-flow signs and thermoelectric energy identity | Pass | `tests/test_thermoelectric.py` checks positive, zero, and reversed current and verifies `Q_h - Q_c = V I`. `tests/test_transient.py` and `tests/test_contact_transient.py` independently check node-rate signs and whole-system energy balance. |
| Positive parameters and limiting cases | Pass for the declared campaign | Four-node storage, contact resistance, probe mass, response time, interface mass, series resistance, and sampled truth values are positive and validated. Existing tests cover equilibrium, zero current, vanishing contact resistance, doubled storage, vanishing probe mass, and fast-probe limits. The new acceptance test checks the actual campaign reference and representative truth draws. |
| Temporary face-probe loading | Pass | The probe adds an equal-and-opposite energy-conserving thermal state. Only the face-temperature repeat and its verification are loaded. The final target forecast and reveal are explicitly unloaded. Existing tests also show that fitted probe nuisance parameters cannot change the unloaded target margin. |
| Terminal-energy sign | Pass | `piecewise_electrical_energy` is signed terminal energy: positive is electrical input and negative is generation. The new test checks both signs analytically. All declared diagnostic schedules consume positive energy, and campaign resource records reject negative consumption totals. |
| Nominal versus realized energy | Pass | Prospective selection resources accept only the physical configuration and declared cost scenario and use the frozen nominal proxy. Realized accounting requires a truth condition and device index or an explicitly bound truth draw. Existing tests confirm the values differ on a nonnominal device and that realized totals are immutable sums of run records. |
| Current switches and minimum-margin grid | Pass for the frozen schedule | All four current transitions are inserted in the integration grid. A fresh refinement check at 0.5, 0.25, 0.125, and 0.0625 seconds found the worst cold-face temperature at the 58-second switch for one deterministic device from each truth family. The production 0.25-second margin differed from the 0.0625-second reference by less than `1e-6 K`; classifications remained stable at `+/-1e-6 K` around a constructed zero-margin boundary. |
| Fit convergence boundary | Pass | Bound-constrained fit convergence uses the projected KKT residual together with step and relative-objective criteria. Bound hits and nonconvergence remain explicit candidate exclusions rather than silently usable fits or whole-case failures. |
| Acquisition-only selection | Pass | Selector and uncertainty-estimator interfaces cannot accept truth family, device truth, verification, final response, or true margin. The evidence factory validates the exact common initial grid, refits only that run, seals the fit provenance, and binds observations into the evidence digest. |
| Candidate failure and attrition | Implementation pass in pre-P2 regression checks; P2 evidence invalid | Every declared predictive draw remains in the denominator and candidate loss retains the conservative baseline floor. Uncertainty v3 separates a recorded candidate transition with a surviving interval from a true whole-draw failure. Candidate-specific fit and forecast exceptions remain usable when another admissible candidate supplies an interval; no-admissible-candidate and upstream failures remain whole-draw failures. Focused regression tests cover each boundary. |
| Verification is not a refit | Pass | Verification predictions use the saved acquisition fit. Only a regularized run-level mean residual is marginalized in the score; physical parameters and fitted probe parameters are not updated. Tests confirm arbitrarily changing verification observations cannot change acquisition fits. |
| Decision saved before reveal | Pass for existing and disposable-pilot orchestration | Runners call build, save, reveal, and score in that order. Scoring requires a saved decision with the same case identity. Regression tests record and assert this order for every fixed policy, including the new four-action disposable pilot path. |
| Semantic random streams | Pass | Keys bind campaign, partition, block, acquisition-evidence digest, source model, draw, purpose, action, run, and channel. Parameter sharing across actions must be explicit. Undeclared reuse, repeated use within one action, observation/probe sharing, and derived-seed collisions fail the audit. |
| Source and artifact provenance | Preflight pass; P2 saved-JSON round trip failed | Strict source manifests reject changed, missing, added, removed, symlinked, uncommitted, runtime-mismatched, or digest-tampered inputs. The validator checks deterministic structure, cross-record identities, exact stream inventory and RNG offsets, fit invariants, and interval formulas, but pre-P2 tests exercised an in-memory representation. P3 must additionally load and validate the final serialized bytes. Archive validation does not rerun acquisition or predictive simulations, candidate refits, or post-reveal scoring. |

## Numerical refinement record

The fresh switch/extremum test used trial 0 from each truth family and the unloaded final schedule. Values below are operating margins in kelvin.

| Truth family | 0.5 s | 0.25 s | 0.125 s | 0.0625 s | Worst-time location |
| --- | ---: | ---: | ---: | ---: | ---: |
| Matched four-state | -0.1440822571 | -0.1440822968 | -0.1440822991 | -0.1440822993 | 58.0 s |
| Extra interface mass | 0.1621561947 | 0.1621561545 | 0.1621561521 | 0.1621561520 | 58.0 s |
| Temperature-dependent contact | 0.1260134279 | 0.1260133917 | 0.1260133895 | 0.1260133894 | 58.0 s |

The refinement is an empirical acceptance check for this frozen schedule and declared parameter regime. `operating_margin` still takes the minimum over the switch-aligned numerical trajectory; it is not a general continuous-time event optimizer. The later pilot must retain the full trajectory and report any case whose margin lies within the frozen numerical tolerance of zero.

## Verification executed

The historical pilot-v1 preflight used the required CPython 3.10.12
environment. Its 104 focused prospective and Phase B tests passed, covering the selector,
uncertainty estimator, costs, semantic random streams, pilot runner, and the
four cross-layer physics/data-flow checks. The complete repository suite then
passed all 773 tests in 256.595 seconds. No scientific partition was generated
by either run.

The uncertainty-v3 replacement preflight in the same environment previously
passed 106 prospective tests and all four cross-layer Phase B checks. It
covers surviving bound hits, nonconvergence, candidate-specific fit
exceptions, candidate-specific forecast exceptions, and true no-envelope
failures. Later adversarial archive tests prompted the validator hardening
described above, so the final combined revision is governed by the pending
record below. None of these checks generated a scientific partition.

The pre-P2 combined revision was validated under CPython 3.10.12. All 136
prospective tests passed in 845.235 seconds, all four cross-layer Phase B checks
passed in 0.195 seconds, and the complete repository suite passed all 809 tests
in 1,375.251 seconds. An independent bounded adversarial review also passed its
archive-integrity, failure-prefix, candidate-attrition, and fit-semantics checks.
No scientific partition was generated by any of these validation runs.

The later P3 archive-transport repair was locally validated under CPython
3.10.12. Eight targeted protocol/archive tests passed in 751.650 seconds; 137
prospective tests passed in 1330.082 seconds; all 4 Phase B tests passed in
0.195 seconds; and the full dependency-equipped suite passed all 810 tests in
1489.060 seconds. These runs overlap, so their test counts must not be added.
No scientific partition was generated. Independent clean-clone review, commit,
push, and green CI remain pending.

## Recorded limits for the next phase

- Predictive draws are an approximate local-covariance calculation. Worst-case aggregation protects only across the surviving four- and five-state candidates.
- The low-level `ThermoelectricParameters` type is an algebraic container and permits idealized zero values for limiting tests. The campaign fixes its thermoelectric constants to checked positive values; a future experiment that varies those constants must validate them at its own boundary.
- Nonnegative campaign energy records are correct for the declared diagnostic schedules. A future regenerative diagnostic schedule would need an explicit gross-consumption/export policy rather than silently clipping or rejecting signed energy.
- The prospective runner may next open only the fresh four-block
  `p3_disposable_archive_roundtrip_replacement_pilot`. Its source-bound outputs and tests
  must preserve acquisition-only construction, save-before-reveal ordering,
  verification without refitting, complete failure records, and separate
  candidate-transition counts. P3 may open only after the locally validated
  JSON-native payload repair and final-byte round-trip checks complete
  independent clean-clone review, then are committed, pushed, and passing CI.
  Those remaining gates are not complete. Development,
  calibration, and reserved runners remain outside this acceptance record.
