# Operating-decision experiment audit

Audit dates: September 11–12, 2026. Reviewed the outline, the available history of **ThermoTwin Development 03**, implementation, tests, and results through commit `3a030de` (final Stage 5 results). Numerical implementation was unchanged from `f95538a`; `93e401f` froze the revised calibration. The development task completed Stage 5 while this audit was underway. This audit did not generate or rerun its reserved devices.

**Assessment: the lumped physics is internally consistent and the tested code executes successfully, but the current experiment does not justify its stated statistical guarantees.** A confirmed random-stream collision creates undeclared dependence between measurement runs and between consecutive device blocks. This affects the foundation of the calibration and bootstrap uncertainty, despite all 602 existing tests passing. The implemented study also answers a narrower question than the original measurement-selection outline.

No production code, frozen artifact, or result was changed by this audit. Pre-existing unrelated report edits were left untouched. Reproduction logs, independent probes, and a source hash manifest are preserved in [the audit evidence directory](operating_decision_audit_2026_09_12).

**1. [P1] Random-number streams collide across runs and device blocks.**

[Run seeds](../../thermotwin/studies/operating_decision_realism.py#L953) advance between diagnostic runs by 100. The [observation generator](../../thermotwin/studies/sensor_model_discrimination.py#L510) also advances between channels by 100. Adding these offsets produces identical seeds for experiments intended to have independent noise and offsets. The earlier Stage 2 implementation uses the same arrangement at [operating_decision.py:648](../../thermotwin/studies/operating_decision.py#L648).

The development-only reproduction at first seed `191001` established:

| Streams intended to be separate | Reproduced collision |
| --- | --- |
| Initial hot-exchanger errors and first additional thermal-run cold-exchanger errors | Entire residual histories identical; maximum difference **0.0 K** |
| Device 0 Family A verification hot channel and device 1 physical-parameter generator | Both seed **201001** |
| Device 0 verification face channel and device 1 initial cold channel | Both seed **201101** |
| Device 0 verification voltage channel and device 1 second thermal-run cold channel | Both seed **201301** |

The standardized Gaussian driving the verification hot-channel offset is `−1.480106633520653`; the standardized log contact resistance of the next device is `−1.4801066335206532`. This is deterministic reuse, not an observed correlation that might disappear with a larger sample. Intentional pairing of genuinely common measurements across policies is appropriate; these additional collisions are not that pairing.

Consequences: the acquisition likelihood assumes independence it does not have, and consecutive device blocks share draws. The ordinary exchangeable-block conformal guarantee and independent-block bootstrap interpretation are therefore unsupported for this generator. The observed counts remain descriptive results of the implemented simulator. This finding does **not** establish the direction or magnitude of bias in a corrected campaign. Standard conformal theory requires exchangeability; adjacent dependence generally violates that assumption. [Angelopoulos and Bates, §5.3](https://arxiv.org/pdf/2107.07511).

The [seed-namespace check](../../thermotwin/studies/operating_decision_calibration.py#L381) enumerates run seeds but omits channel offsets and uses a set, so it does not detect these within-partition collisions. Checking only that large development/calibration/evaluation ranges do not overlap is insufficient.

Required repair: derive streams from structured identifiers that distinguish partition, block, truth/measurement purpose, family, run, and channel. Declare explicitly which identifiers should be shared for paired measurements. Test uniqueness of every actual consumed stream and deliberate reuse of common channels. Version the generator and regenerate affected development results, verification gates, calibrations, rehearsal, and final evidence. Preserve the present results as superseded records. Stage 5 has now been exposed; a corrected confirmation needs a newly frozen, fresh evaluation namespace and must not tune to the existing final labels.

Evidence: [rng_probe.py](operating_decision_audit_2026_09_12/rng_probe.py).

**2. [P2] Reported diagnostic energy is a nominal template, not each simulated device's energy.**

[nominal_realistic_schedule_energy](../../thermotwin/studies/operating_decision_realism.py#L1494) always uses the nominal four-state device and nominal series resistance. Its values are computed once and [assigned to every family and trial](../../thermotwin/studies/operating_decision_realism.py#L1854). Stage 3's generated text calls these nominal values, but later summaries and loss comparisons present them as mean diagnostic energy without consistently retaining that qualification.

An audit integrated terminal power using each of the 30 existing Stage 3 development devices and its actual diagnostic instrumentation. The voltage package is assigned **89.53209 J** everywhere; actual family means are **89.55396, 89.23099, and 89.17763 J** for A/B/C. The largest relative discrepancy across all 120 policy/device combinations was the Family A trial-0 face package: assigned **89.40873 J**, simulated **93.26668 J**, an underestimate of **4.14%**. Mean differences are small in this development cohort, but the current values cannot be called realized device energy, especially when energy contributes to the primary loss.

Either compute per-device energy during postdecision scoring or explicitly freeze and label these as nominal cost proxies. Separately, reset duration/energy and sensor electronics are unmodeled. The recorded 80 seconds per run is schedule duration, including zero-current portions; it is not elapsed bench time including resets.

Selector computation time also [copies the chosen fixed-policy timer](../../thermotwin/studies/operating_decision_calibration.py#L1006), omitting some initial fitting/forecasting used to choose that policy and subsequent calibration operations. Stage 5 correctly excludes computation time from its resource claim. Retain that exclusion until timing covers the complete procedure.

Evidence: [energy_results.json](operating_decision_audit_2026_09_12/energy_results.json), [energy_probe.py](operating_decision_audit_2026_09_12/energy_probe.py).

**3. [P2] The frozen artifact checks configuration integrity, but does not verify executing source code.**

[Artifact construction](../../thermotwin/studies/operating_decision_final_evaluation.py#L820) accepts any nonempty revision string. [Validation](../../thermotwin/studies/operating_decision_final_evaluation.py#L1273) reconstructs the digest using that same string without verifying the source files. A correctly constructed artifact claiming `NOT_A_REAL_REVISION` passed validation and reached a **mocked** generation boundary. No final device was generated by that probe.

The actual recorded revision exists, and I found no evidence that the recorded run secretly used different numerical code. The defect is that the promised freeze does not prevent a later code change from running under an old artifact. Bind the numerical source files and relevant runtime configuration to a manifest, verify it before generation, and record it in the result. A documentation-only commit should remain permissible when the numerical source hashes agree.

Evidence: [provenance_probe.py](operating_decision_audit_2026_09_12/provenance_probe.py).

**4. [P2] One published confidence bound treats clustered family rows as independent.**

[The Stage 3 report](../../thermotwin/OPERATING_DECISION_REALISM.md#L128) gives a 17.6% upper 95% Wilson bound for pooled 0/18 voltage approvals. The family variants share physical draws within ten device blocks, so an ordinary 18-independent-trial binomial interpretation is unjustified even after repairing the accidental RNG collisions. Use family-specific counts and an appropriate block analysis. Stages 4 and 5 correctly suppress ordinary pooled row-level Wilson intervals; retain that distinction.

**5. [P2] The preserved diagnostic record is incomplete.**

Stage 5 now has all **900 compact procedure outcomes**, which substantially improves reproducibility. However, the [compact schema](../../thermotwin/studies/operating_decision_calibration.py#L294) omits fitted parameters/covariances, detailed failure reasons, candidate trajectories, and supporting trajectory/parameter errors. These cannot be reconstructed from its saved interval endpoints and counts alone.

The documented Stage 4 `--no-figure` run preserves a calibration artifact and aggregate text, while the [complete result export is tied to the figure path](../../thermotwin/reports/operating_decision_calibration.py#L597). The guard-freeze report similarly does not retain its complete recalibration outcomes. Export numerical results independently of plotting and save enough information to investigate every failure without regenerating the campaign. The outline's decision illustration, measurement map, and frozen risk-versus-coverage sweep are also missing.

**Physics assessment.**

The thermoelectric equations use the expected constant-property convention:

```text
Qc = alpha I Tc - I²R/2 - K(Th - Tc)
Qh = alpha I Th + I²R/2 - K(Th - Tc)
V  = alpha(Th - Tc) + IR
Qh - Qc = IV
```

These agree with the standard module heat and voltage relations in [Ferrotec's mathematical reference](https://thermal.ferrotec.com/technology/thermoelectric-reference-guide/thermalref11/). Kelvin temperatures, current reversal, zero current, and Joule-heating symmetry are treated consistently. Constant Seebeck coefficient is a declared approximation; a temperature-dependent contact resistance alone does not require adding a Thomson term to that constant-property thermoelectric model.

The added electrical resistance contributes both voltage drop and heat. Splitting that added Joule heat equally between faces is a specific symmetric intrinsic-contact assumption, not a general description of arbitrary external wiring. The thermal contact links transfer equal and opposite heat between states. The temporary sensor receives heat from the face and removes the same amount from it; its capacitance and coupling determine both loading and response time. The unloaded final forecast removes that physical sensor while retaining its uncertainty in the fitted physical parameters. The positive exponential contact law preserves passive conduction, but its form and strength are synthetic stress assumptions, not measured material laws.

The resulting system balance is stored thermal-energy rate = terminal electrical power + net reservoir heat input. The independently written edge-balance equations also produced nonnegative entropy production in the audited trajectories. I found no sign, dimensional, or conservation error in this portion.

Production truth and candidate predictions do share [the same simulation helper](../../thermotwin/studies/operating_decision_realism.py#L1004), so their agreement is not independent numerical validation. To address that during the audit, a separate implementation wrote the heat balances independently and used a fifth-order Dormand–Prince scheme instead of the project's fourth-order Runge–Kutta solver:

| Independent check | Result |
| --- | --- |
| 30 existing development devices, A/B/C, unloaded final schedule | Maximum cold-face difference **1.70×10⁻⁶ K**; maximum margin difference **3.52×10⁻⁹ K**; no pass/fail changes |
| Six nominal A/B/C cases, loaded and unloaded | Maximum cold-face difference **6.87×10⁻⁷ K**; maximum temperature-observation difference **7.39×10⁻⁵ K** |
| Dense margin versus finer 0.025-second reference grid | No missed minimum affecting the margin in the checked cases; production uses the dense 0.25-second trajectory, not the 1-second sensor samples |
| Reference refinement to 0.0125 seconds | Nominal loaded-C margin changed about **3.4×10⁻¹³ K** |
| Fast, highly loaded parameter corner | Cold-face difference **0.00636 K**, temperature-observation differences below **0.000785 K**, margin difference **8.89×10⁻⁸ K** |

The corner case is a reason to add permanent convergence checks across parameter bounds; it did not produce a changed decision. These finite checks support numerical correctness in the tested domain and do not prove correctness for every possible input. Preserve the independent comparison as regression coverage rather than relying only on shared-model fits.

Evidence: [independent_physics_probe.py](operating_decision_audit_2026_09_12/independent_physics_probe.py), [independent_physics_results.json](operating_decision_audit_2026_09_12/independent_physics_results.json).

**Inference and outline traceability.**

No direct use of hidden truth, verification scores, or unchosen measurements was found in the action signals. Acquisition, verification, and final responses are separated; verification uses acquisition-fitted parameters and marginalizes fresh offsets without fitting them. Surviving model intervals form an envelope. Missing intervals produce abstentions, and zero denominators are N/A. These are sound design choices. They do not fix the RNG defect.

| Outline requirement | Audit status |
| --- | --- |
| Stage 1 foundation; fixed schedule, temperature band, margin, prospective acquisition runs | Implemented; full baseline and Stage 2 reproduced |
| A/B/C truth, uncertain electrical resistance, temporary sensor loading/removal | Implemented and physically checked under stated assumptions |
| Development/calibration/evaluation separation | Explicit namespaces and chronology exist; actual random streams have unintended cross-block collisions |
| Conventional multistart inference and adequacy envelopes | Implemented; no direct hidden-response leakage found |
| Develop a bounded parametric-bootstrap uncertainty method | Replaced by local covariance plus additive block conformal calibration; a disclosed methodological change |
| Predict possible new observations, compare expected decision-uncertainty reduction per cost across all packages | **Not implemented**; [the selector](../../thermotwin/studies/operating_decision_calibration.py#L567) is a stop-or-voltage heuristic, later supplemented by an early-abstention guard |
| Vary costs and sensor quality to map measurement selection | Costs score the same fixed rule afterward; no cost-dependent selection or measurement map |
| Compare against strongest fixed policy at comparable quality/resources | Fixed policies are present; the primary final statistical contrast is revised selector versus parent selector, a narrower triage question |
| 50 fresh devices per family, all failures retained, family reporting | Completed Stage 5: 50 blocks × 3 families × 6 procedures; uncertainty interpretation affected by finding 1 |
| Runs, bench time including resets, energy, instrumentation, computation | Partial: run/sensor counts available, energy nominal, resets absent, selector timing incomplete |
| Frozen risk/coverage sweep, decision illustration, complete machine-readable diagnostics | Incomplete |

The outline's opening status is stale. It should either be amended to explicitly describe the narrower calibration/triage experiment and its omissions, or the original prospective selector and remaining deliverables should be implemented in a new protocol. The current selector cannot support a claim that it chooses among thermal, voltage, and face measurements according to their expected value and cost.

Local covariance is a reasonable inexpensive starting interval, and conformal padding can calibrate an arbitrary starting interval under the required sampling assumptions. It does not show that missing physics was excluded, and 90% procedure-set coverage is not a guarantee of at most 10% false approvals among approvals. Here a missing interval counts as an unbounded set for coverage and an abstention for decision coverage; that is coherent, but high set coverage must always be read alongside abstention counts.

The optimizer has a fixed six-iteration budget without an explicit convergence status. Continuing ten candidate fits from five existing development cases for another 30 iterations changed margin estimates by at most about **4.4×10⁻⁵ K**, interval endpoints by about **6.9×10⁻⁵ K**, and no interval-sign classifications. This did not establish an optimizer bug; add convergence diagnostics before making broader numerical reliability claims.

**Execution and result verification.**

- Full repository suite: **602 tests passed, no skips**, in **287.341 seconds**, using `/Users/rolandbennett/.pyenv/versions/pinn-env/bin/python` (Python 3.10.12 with the optional scientific/reporting dependencies). [Full log](operating_decision_audit_2026_09_12/full_test_suite.log).
- The initial bundled Python 3.12 run had 10 errors and 106 skips due to absent optional Matplotlib/PyTorch dependencies. The full-environment run above resolved them; those were not failures of the operating-decision physics. [Initial log](operating_decision_audit_2026_09_12/minimal_runtime_tests.log).
- Full sensor-discrimination baseline: 20 devices per family, seed `91001`; all displayed values reproduce, including failed packages and false-confidence cases. [Reproduction](operating_decision_audit_2026_09_12/baseline_reproduction.txt).
- Full Stage 2: all eight policy/family table rows reproduce at documented precision, including margins, prevalence, interval coverage, energy proxies, and zero numerical failures. [Reproduction](operating_decision_audit_2026_09_12/stage2_reproduction.txt).
- Stage 4 and revised calibration: existing aggregate reports agree with the documented tables and saved artifact values. Both artifact validation paths were checked. These large calibration cohorts were not regenerated during this audit.
- Stage 5: independently recomputed all **900** unique outcome identities, pairing, error flags, interval decisions/padding, **24** summary cells, losses, guard counts, and all three deterministic **20,000-draw** bootstrap intervals from the stored rows. All agreed within `1e-12`; no simulations were run. [Stored-result check](operating_decision_audit_2026_09_12/final_stored_result_audit.txt).

The complete final comparison, using the recorded nominal-energy cost model, is:

| Procedure | Approve / reject / insufficient | Decision coverage | Mean runs | Balanced empirical loss |
| --- | ---: | ---: | ---: | ---: |
| Stop now | 33 / 34 / 83 | 44.7% | 2.00 | 0.5533 |
| Fixed thermal | 44 / 43 / 63 | 58.0% | 5.00 | 0.7835 |
| Fixed voltage | 57 / 46 / 47 | 68.7% | 3.00 | 0.5608 |
| Fixed face temperature | 55 / 36 / 59 | 60.7% | 3.00 | 0.6406 |
| Stage 4 selector | 54 / 42 / 54 | 64.0% | 2.61 | 0.5118 |
| Revised selector | 49 / 36 / 65 | 56.7% | 2.55 | 0.5785 |

All six procedures recorded zero incorrect determinate decisions. That is an observed count, not a proven small population error rate. The revised procedure lost 11 decisions relative to Stage 4: **three** through guard alarms and **eight** through its larger recalibrated padding. Its failed usefulness criteria describe the full revised procedure; attributing all deterioration to the acquisition score would be inaccurate.

The reported balanced-loss difference of `+0.0667` and bootstrap interval `[+0.0317, +0.1067]` reproduce. Because of finding 1, the confidence interpretation in [the final discussion](../../thermotwin/OPERATING_DECISION_FINAL_EVALUATION.md#L153) is not established by that arithmetic. Preserve the negative point estimate and failed descriptive criteria while qualifying the claimed statistical strength.

The statement that [Stage 4 remains the supported operating rule](../../thermotwin/OPERATING_DECISION_FINAL_EVALUATION.md#L169) also needs narrowing. It has the best empirical balanced loss, but stop-now wins under the instrumentation-expensive weights. Versus fixed voltage, Stage 4 saves **0.3867 runs**, **11.1415 nominal J**, and **0.3867 sensors** per device while losing **4.67 percentage points** of decision coverage. Its 64% coverage misses the outline's 70% development target, and the primary final contrast did not test superiority over fixed alternatives. It is a retained benchmark baseline with a favorable descriptive tradeoff under particular weights, not a scientifically validated recommendation.

The original main hypothesis remains unestablished. The priority is to repair and test random-stream construction, correct resource labeling/accounting and provenance, explicitly settle the narrower versus original scope, and only then freeze a corrected campaign. The existing physics and code structure are useful foundations; current passing tests and repeatable tables are insufficient to certify the experiment's scientific validity.
