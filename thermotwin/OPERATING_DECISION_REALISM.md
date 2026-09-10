# Operating-decision realism stress test

Status: Stage 3 development stress test complete on 2026-09-10. This is
synthetic development evidence, not a frozen evaluation or hardware
validation.

## Question

Does the strongest diagnostic package from the idealized Stage 2 pilot still
support approve, reject, or insufficient-evidence decisions when the extra
measurements have realistic nuisance physics and the virtual hardware contains
a truth family that neither fitted model can represent?

This is a software-only falsification experiment. It keeps the four Stage 2
policies, final schedule, operating band, acquisition/final split, verification
threshold, and local interval rule fixed. It then makes voltage and face
temperature harder to obtain honestly and adds an omitted constitutive law.

## Frozen realism model

Every virtual device has a positive terminal series resistance `R_s`. The
module's known 2.0 ohm resistance becomes `R_eff = 2.0 + R_s`, so the same
quantity enters terminal voltage and thermal dynamics:

```text
V = alpha * (T_hot - T_cold) + I * R_eff
Q_contact,cold = Q_contact,hot = 0.5 * I^2 * R_s.
```

The symmetric heat-deposition assumption gives exact whole-system energy
closure. The hidden truth uses a nominal `R_s` of 0.10 ohm with log standard
deviation 0.35. Both candidates fit it within 0.01 to 0.50 ohm under a log
prior standard deviation of 0.50. Voltage retains the Stage 2 white-noise
standard deviation of 0.002 V and a separate Gaussian run offset with standard
deviation 0.005 V. That measurement offset never enters Joule heat or energy
accounting.

The face-temperature policy now mounts a temporary probe with temperature
`T_s`, thermal capacitance `C_s`, and response time `tau_s`:

```text
G_s = C_s / tau_s
q_s = G_s * (T_cold_face - T_s)
dT_s/dt = q_s / C_s.
```

The same heat is removed from the cold-face balance. The probe therefore lags
and perturbs the diagnostic device. It is initialized at the 300 K reset
temperature, remains installed for the prospective face acquisition and
verification runs, and is explicitly removed before the untouched final run.
The hidden nominal values are 5 J/K and 2.5 s; inference sees bounds of 1 to 12
J/K and 0.5 to 6 s, with log prior standard deviation 0.35 for both. The probe
channel reads `T_s` directly rather than adding a second response filter.
On the nominal 0.8 A, 20 s diagnostic pulse, mounting this probe raises the
minimum true cold-face temperature by 0.121 K and creates up to 0.351 K of
probe-to-face separation at the one-second sample times. Its effect is therefore
material relative to the 0.02 K temperature-noise scale.

The two established truth families remain a matched four-state device and a
five-state device with hidden cold-interface mass. Family C retains the five
states and makes its total cold thermal-contact resistance depend on the hidden
interface temperature:

```text
R_c(T_interface) = R_300 * exp(beta * (T_interface - 300 K)),
beta ~ Uniform(0.08, 0.16) 1/K.
```

The existing 50/50 contact-resistance split is preserved. This law is positive,
reduces exactly to the established five-state truth at `beta = 0`, and cancels
internally in the energy balance. Neither fitted candidate contains `beta` or
this temperature dependence.

The fitter estimates device parameters, series resistance, and, for the face
policy, both probe parameters jointly. Acquisition offsets are analytically
profiled under their finite calibration variance, including their penalty;
they are not unrestricted free constants. Verification marginalizes over the
same declared offset distribution without refitting. Final uncertainty uses
the complete fitted covariance. The removed probe has zero direct derivative
in the final simulator, while its confounding with device parameters remains
in their marginal covariance.

## Conservation and limiting checks

The implementation tests the following before comparing policies:

- zero series resistance reproduces the Stage 2 trajectories and voltage;
- added voltage is `I * R_s`, added face heat sums to `I^2 * R_s`, and the
  combined thermal energy rate equals physical terminal power plus external
  heat;
- probe-to-face heat is equal and opposite in four- and five-state models;
- the small-probe-mass and fast-response limits approach the expected unloaded
  and lumped-capacitance behavior;
- `beta = 0` reproduces the established interface-mass solver;
- every new integrator lands on current switches and converges under time-step
  refinement; and
- final truth is identical across paired policies and never contains the
  temporary probe.

## Development result

The cohort contains 10 paired devices from each truth family, beginning at
seed 191001. Each of the 120 cases uses three fit starts and six Gauss-Newton
iterations per start. All decisions completed with no numerical failures.

| Truth family | Policy | Approve / reject / insufficient | Decision coverage | False approvals | False rejections | Interval coverage | Runs | Modeled terminal energy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Four-state | Stop now | 2 / 6 / 2 | 8/10 | 0/2 | 0/6 | 9/10 | 2 | 60.72 J |
| Four-state | More thermal tests | 3 / 6 / 1 | 9/10 | 0/3 | 0/6 | 9/10 | 5 | 99.29 J |
| Four-state | Add voltage | 4 / 6 / 0 | 10/10 | 0/4 | 0/6 | 10/10 | 3 | 89.53 J |
| Four-state | Add face temperature | 4 / 6 / 0 | 10/10 | 0/4 | 0/6 | 10/10 | 3 | 89.41 J |
| Extra interface mass | Stop now | 4 / 0 / 6 | 4/10 | 0/4 | N/A | 10/10 | 2 | 60.72 J |
| Extra interface mass | More thermal tests | 6 / 0 / 4 | 6/10 | 0/6 | N/A | 10/10 | 5 | 99.29 J |
| Extra interface mass | Add voltage | 7 / 0 / 3 | 7/10 | 0/7 | N/A | 10/10 | 3 | 89.53 J |
| Extra interface mass | Add face temperature | 7 / 0 / 3 | 7/10 | 0/7 | N/A | 9/10 | 3 | 89.41 J |
| Temperature-dependent contact | Stop now | 4 / 0 / 6 | 4/10 | 0/4 | N/A | 10/10 | 2 | 60.72 J |
| Temperature-dependent contact | More thermal tests | 4 / 0 / 6 | 4/10 | 0/4 | N/A | 10/10 | 5 | 99.29 J |
| Temperature-dependent contact | Add voltage | 7 / 0 / 3 | 7/10 | 0/7 | N/A | 7/10 | 3 | 89.53 J |
| Temperature-dependent contact | Add face temperature | 7 / 0 / 3 | 7/10 | 0/7 | N/A | 3/10 | 3 | 89.41 J |

Across all three truth families, decision coverage is 16/30 for stopping,
19/30 for more thermal runs, and 24/30 for both added-sensor policies. The
idealized Stage 2 face-temperature lead therefore does not survive this stress
test. Voltage matches its decision coverage with the same run count, avoids a
thermally loaded measurement transfer, and has stronger Family C interval
coverage: 7/10 versus 3/10.

Every determinate decision is correct in this small cohort, but that is not a
zero-risk result. For example, zero false approvals among the voltage policy's
18 approvals still has a 17.6% upper 95% Wilson bound. More seriously, the face
policy's Family C intervals miss the true final margin in seven of ten cases.
The decision labels happen to remain correct because those devices are not all
near the boundary. The interval result exposes the failure that this stage was
designed to find: a direct diagnostic measurement can produce a sharp but
miscalibrated unloaded-device forecast when constitutive physics is missing.

The fixed verification gate does not catch this failure. Across the 80 Family C
candidate-policy fits, 79 pass verification and one is excluded for a parameter
bound; none is rejected by the normalized-score threshold. The omitted contact
law can therefore hide inside the constant-resistance candidates over the
current verification schedule. The interval-coverage audit, rather than the
adequacy score, reveals the miss.

Modeled energy includes the thermoelectric module and terminal-contact power.
Probe readout electronics and reset energy are not modeled, so the nearly equal
voltage and face-policy joule totals are not a complete hardware cost
comparison.

## Reproduction

Run the dependency-free numerical report from the repository root:

```bash
python3 -m thermotwin.operating_decision_realism --no-figure
```

For a quick integration check:

```bash
python3 -m thermotwin.operating_decision_realism \
  --trials 1 --fit-iterations 1 --no-figure
```

The focused physics and decision checks are:

```bash
python3 -m unittest \
  tests.test_series_resistance_realism \
  tests.test_temporary_face_sensor \
  tests.test_temperature_dependent_contact \
  tests.test_operating_decision_realism
```

## Next stage

The next step is calibration rather than immediate policy optimization. Use
development resampling to redesign or calibrate the verification gate and
final-margin intervals under all three truth families, including the
sensor-removal transfer. Then build the decision-directed selector and compare
its expected value against these unchanged fixed policies on fresh seeds. The
selector should prefer voltage over face temperature unless the face policy can
demonstrate calibrated unloaded-device coverage after paying its physical
instrumentation cost.
