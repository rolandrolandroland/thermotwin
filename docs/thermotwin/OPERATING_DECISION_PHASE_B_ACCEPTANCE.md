# Operating-decision Phase B physics and data-flow acceptance

Date: 2026-09-17

Source basis: audited parent `9a21aa77284fb88608467a39a7ada8dc811107d5`; Phase B and disposable Phase C source in the commit containing this record

Status: accepted for a disposable Phase C pilot; no scientific partition was generated or opened

## Conclusion

The declared synthetic physics and the existing information boundaries are internally consistent for the next disposable pilot. No material physics, energy-accounting, leakage, random-stream, or provenance defect was found, so this review made no change to the numerical model. Four cross-layer regression tests were added in `tests/test_operating_decision_phase_b_acceptance.py` for the remaining acceptance gaps.

This acceptance says that the code implements the declared synthetic model consistently. It does not show selector benefit, establish hardware validity, or validate the later calibration and reserved-evaluation designs.

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
| Candidate failure and attrition | Pass | Every declared predictive draw remains in the denominator. Failed draws receive the common baseline; loss of an initially admissible candidate cannot create positive information value. Initial fit failures, refit failures, candidate status, observation digests, and stable/unstable counts remain serializable. |
| Verification is not a refit | Pass | Verification predictions use the saved acquisition fit. Only a regularized run-level mean residual is marginalized in the score; physical parameters and fitted probe parameters are not updated. Tests confirm arbitrarily changing verification observations cannot change acquisition fits. |
| Decision saved before reveal | Pass for existing and disposable-pilot orchestration | Runners call build, save, reveal, and score in that order. Scoring requires a saved decision with the same case identity. Regression tests record and assert this order for every fixed policy, including the new four-action disposable pilot path. |
| Semantic random streams | Pass | Keys bind campaign, partition, block, acquisition-evidence digest, source model, draw, purpose, action, run, and channel. Parameter sharing across actions must be explicit. Undeclared reuse, repeated use within one action, observation/probe sharing, and derived-seed collisions fail the audit. |
| Source and artifact provenance | Pass | Strict source manifests reject changed, missing, added, removed, symlinked, uncommitted, runtime-mismatched, or digest-tampered inputs. Evidence archives reject row tampering, unclean stream audits, duplicate partitions, and overwrite attempts. |

## Numerical refinement record

The fresh switch/extremum test used trial 0 from each truth family and the unloaded final schedule. Values below are operating margins in kelvin.

| Truth family | 0.5 s | 0.25 s | 0.125 s | 0.0625 s | Worst-time location |
| --- | ---: | ---: | ---: | ---: | ---: |
| Matched four-state | -0.1440822571 | -0.1440822968 | -0.1440822991 | -0.1440822993 | 58.0 s |
| Extra interface mass | 0.1621561947 | 0.1621561545 | 0.1621561521 | 0.1621561520 | 58.0 s |
| Temperature-dependent contact | 0.1260134279 | 0.1260133917 | 0.1260133895 | 0.1260133894 | 58.0 s |

The refinement is an empirical acceptance check for this frozen schedule and declared parameter regime. `operating_margin` still takes the minimum over the switch-aligned numerical trajectory; it is not a general continuous-time event optimizer. The later pilot must retain the full trajectory and report any case whose margin lies within the frozen numerical tolerance of zero.

## Verification executed

Final pre-pilot validation used the required CPython 3.10.12 environment. The
104 focused prospective and Phase B tests passed, covering the selector,
uncertainty estimator, costs, semantic random streams, pilot runner, and the
four cross-layer physics/data-flow checks. The complete repository suite then
passed all 773 tests in 256.595 seconds. No scientific partition was generated
by either run.

## Recorded limits for the next phase

- Predictive draws are an approximate local-covariance calculation. Worst-case aggregation protects only across the surviving four- and five-state candidates.
- The low-level `ThermoelectricParameters` type is an algebraic container and permits idealized zero values for limiting tests. The campaign fixes its thermoelectric constants to checked positive values; a future experiment that varies those constants must validate them at its own boundary.
- Nonnegative campaign energy records are correct for the declared diagnostic schedules. A future regenerative diagnostic schedule would need an explicit gross-consumption/export policy rather than silently clipping or rejecting signed energy.
- The prospective runner currently opens only the four-block disposable pilot. Its source-bound outputs and tests preserve acquisition-only construction, save-before-reveal ordering, verification without refitting, and complete failure records. Development, calibration, and reserved runners remain outside this acceptance record.
