# Distributed resistivity observation-sufficiency study

## 1. Problem being answered

Can the selected current regimes and terminal sensors support a unique local
three-knot electrical-resistivity curve, or should ThermoTwin refuse to report
one?

This question must be answered before comparing optimizers. A conventional
fit or a PINN can return a smooth, repeatable curve even when the measurements
do not contain three independent directions of information. In that case the
curve is selected by bounds, initialization, regularization, or the estimator's
implicit bias—not by the experiment alone.

The study therefore separates two decisions:

1. **Pre-fit information decision:** inspect a noise-normalized sensitivity
   spectrum and decide how many curve directions the measurements support.
2. **Post-fit diagnostic:** run multiple initial curves only when resistivity is
   present in the observations, then check curve error, initialization spread,
   and transfer to a held-out regime.

An optimizer is never allowed to overrule a failed pre-fit information gate.

## 2. Unknown function

The fitted property is the electrical resistivity

$$
\rho_e(T),
$$

represented by three positive coefficients at 285, 300, and 315 K. The
coefficients are optimized as bounded log multipliers relative to the baseline
curve. The declared neighborhood is

$$
-0.3 \leq \log m_j \leq 0.3.
$$

This corresponds to multiplier bounds of approximately 0.741 to 1.350.

The independent synthetic truth is the same smooth cubic used by the
independent-validation benchmark. It matches truth multipliers

$$
(1.04,\ 1.07,\ 1.03)
$$

at the inference knots but is not piecewise linear between them.

## 3. Why the current and sensor choices matter

Electrical resistivity reaches the distributed model through two terms.

The volumetric Joule source scales as

$$
q'''_{\mathrm{Joule}} = J^2\rho_e(T),
$$

where $J=I/A$ is current density. The ohmic part of terminal voltage is

$$
V_{\mathrm{ohmic}} = \int_0^L J\rho_e(T)\,dx.
$$

These equations immediately predict the main limiting case. When $I=0$,
both terms vanish exactly. A passive temperature relaxation can reveal thermal
properties, but it contains no electrical-resistivity information.

At one nonzero current, face temperatures see resistivity only indirectly
through Joule heating. Voltage is much more sensitive, but an integral voltage
can strongly constrain an average resistance while remaining weak along one
curve-shape direction. Positive and negative currents help separate terms with
different current parity: Joule heating is even in current, while the ohmic
voltage is odd and thermoelectric terms respond differently under reversal.

## 4. Independent data generator

The observation data do not come from the solver used during fitting.

- Truth uses 25 nodal temperatures including both faces.
- Time integration uses transition-split SSPRK3 with a 0.00025 s step.
- Inference retains the established cell-centred finite-volume/RK4 model.
- Truth uses the smooth cubic resistivity law; inference uses three linear
  knots.
- Face temperatures receive independent 0.01 K Gaussian noise.
- Voltage receives independent 10 microvolt Gaussian noise.
- Samples are exposed every 0.08 s.

This removes exact grid, time-integrator, and property-basis inverse crime. It
does not create hardware validation: truth and inference still share the same
one-dimensional continuum equations, known non-resistivity properties, and
known boundary conditions.

## 5. Pre-fit gate

For each observation case, ThermoTwin perturbs each log coefficient and forms
the noise-normalized Jacobian

$$
J_{ij} = \frac{1}{\sigma_i}
\frac{\partial y_i}{\partial\log m_j}.
$$

Its singular values $s_1\geq s_2\geq s_3$ measure the visible strength of
orthogonal local coefficient combinations. Because the rows are divided by the
declared sensor standard deviations, a singular value has a direct local
signal-to-noise interpretation.

The frozen practical rule asks whether a log displacement no larger than 0.3
can create at least one noise-standard-deviation of change in every singular
direction:

$$
s_j(0.3) \geq 1.
$$

Equivalently, every singular value must be at least 3.3333. This is a declared
one-sigma local resolution rule. It is not a global uniqueness proof or a
confidence interval.

The possible decisions are:

- `structurally_non_identifiable`: every sensitivity is zero;
- `practically_non_identifiable`: some physics sensitivity exists, but fewer
  than three directions clear the declared noise-resolution threshold;
- `supported`: all three local directions clear the threshold, making a fit
  eligible for subsequent validation.

`supported` does not guarantee that an optimizer is accurate.

## 6. Observation cases

| Case | Current regimes | Visible channels | Purpose |
| --- | --- | --- | --- |
| Full bidirectional | 0, +0.8, and -0.8 A | Cold face, hot face, voltage | Reference experiment |
| Zero current only | 0 A | Cold face, hot face, voltage | Exact structural limiting case |
| Positive temperature only | +0.8 A | Cold and hot face temperatures | Test indirect Joule-heating information |
| Positive temperature plus voltage | +0.8 A | Cold face, hot face, voltage | Test average resistance versus curve shape |

The first three regimes start from the same 10 K face-temperature span used by
the distributed inverse benchmark, except for the zero-current relaxation's
declared reservoir forcing. Exact frozen definitions live in
`distributed_inverse_constant_experiments()`.

## 7. Singular-spectrum result

| Case | Singular values | Supported rank | Decision |
| --- | --- | ---: | --- |
| Full bidirectional | 1566.17, 9.50048, 4.72772 | 3/3 | Supported |
| Zero current only | 0, 0, 0 | 0/3 | Structurally non-identifiable |
| Positive temperature only | 1.72245, 0.241809, 0.0124026 | 0/3 | Practically non-identifiable |
| Positive temperature plus voltage | 1102.01, 4.24759, 0.559103 | 2/3 | Practically non-identifiable |

The result matches the physics:

- zero current has exactly zero resistivity sensitivity;
- face temperatures alone are too weak at the declared noise level;
- adding voltage creates one extremely strong average-resistance direction and
  a second supported direction, but the third shape direction remains below
  the gate;
- current reversal plus temperatures and voltage supports all three local
  directions.

The weakest one-sigma log displacements are 0.2115 for the full set, infinite
for zero current, 80.63 for temperature only, and 1.7886 for positive current
with voltage. Values larger than the allowed 0.3 neighborhood indicate that
the corresponding local direction cannot be resolved under this rule.

## 8. Multistart fits

Three property initializations are used for both estimators:

$$
(0.8,0.8,0.8),\quad (1.0,1.0,1.0),\quad (1.2,1.2,1.2).
$$

The conventional estimator uses the finite-volume model, coordinate search,
and damped Gauss-Newton polish. The inverse PINN uses one hidden temperature
network per regime and one shared resistivity curve for 500 CPU epochs. No
explicit curve-smoothness penalty is applied in this study.

The zero-current case is not fitted. Since its Jacobian is exactly zero, any
returned resistivity curve would be unrelated to the data.

### 8.1 Full bidirectional case

The pre-fit gate permits estimation. Across the three starts:

- conventional maximum multiplier spread is 0.00359;
- PINN maximum multiplier spread is 0.00624;
- PINN maximum continuous property error is 0.0363--0.0392;
- conventional maximum continuous property error is 0.2778--0.2813;
- PINN held-out voltage RMSE is 8.89--10.34 microvolts;
- conventional held-out voltage RMSE is 35.80--38.83 microvolts.

The full-rank gate does not rescue a biased conventional optimum under this
single noisy model-mismatch realization. It only says that all three local
directions are visible. Fit quality still requires multistart, truth-based
synthetic checks, withheld prediction, and eventually hardware evidence.

### 8.2 Positive-current temperatures only

The pre-fit gate rejects the curve. The conventional optimizer converges to
the same bounded curve from all three starts, with about 0.298 maximum property
error. The PINN retains strong initialization dependence: its maximum
multiplier spread is 0.207 and maximum property errors range from 0.401 to
0.591.

This case is important because it shows two different failure appearances.
One estimator is repeatable but wrong; the other visibly depends on its start.
Neither outcome overrides the information gate.

### 8.3 Positive-current temperatures plus voltage

The pre-fit gate supports two directions and rejects a three-coefficient curve.
The conventional maximum multiplier spread is 0.285, with maximum property
errors near 0.288--0.293. The PINN is much more stable and happens to be close
to the synthetic truth: spread is 0.00851 and maximum property error is
0.0308--0.0386.

That apparently strong PINN result is **not** evidence that the third
coefficient direction became measurable. The neural field and optimizer have
selected a stable curve using their implicit bias. A stable-looking inferred
function cannot manufacture rank that the sensor/current configuration lacks.
The PINN curves are therefore retained as diagnostics and explicitly rejected
as property estimates.

## 9. Held-out transfer

Every fitted curve is frozen and inserted into the conventional solver for a
separate +0.4 A, 20 K-lift regime. The independent nodal/cubic model supplies
the hidden truth. Internal-temperature RMSE is small for many wrong curves,
because the transient temperatures are not highly discriminating for
resistivity under these conditions. Voltage transfer is more revealing.

This reinforces a general rule: a low temperature RMSE does not prove that the
transport curve is correct. Property error, voltage, and experiment-level
information rank must be inspected separately.

## 10. What this study establishes

ThermoTwin now refuses the most misleading inverse cases before training:

- it detects the exact zero-current structural limit;
- it distinguishes weak temperature sensitivity from usable property
  information;
- it recognizes that one-direction voltage mostly constrains average
  resistance rather than all three curve directions;
- it prevents a stable PINN curve from being reported as identified when the
  experiment supports only two directions;
- it keeps diagnostic fits and held-out predictions visible instead of hiding
  estimator failures.

## 11. What it does not establish

- The singular spectrum is local to one baseline curve and finite experiment
  set.
- The one-sigma threshold and +/-0.3 log neighborhood are declared choices.
- One noise realization does not estimate failure probabilities.
- Multistart spread is not a profile-likelihood confidence interval.
- The independent truth still shares the same one-dimensional continuum
  equations and known boundary conditions.
- No result validates the material curve or sensor model against hardware.
- Only `rho_e(T)` is released; joint `alpha(T)`, `rho_e(T)`, and `kappa(T)`
  inference remains a later, harder problem.

## 12. Reproduce

Install the PINN and report dependencies, then run:

```bash
python3 -m pip install -e '.[all]'
thermotwin-distributed-identifiability
```

The equivalent module command is:

```bash
python3 -m thermotwin.distributed_observation_identifiability
```

The default figure is written to
`thermotwin/figures/distributed_observation_identifiability.png`. Change the
training budget or output path with:

```bash
thermotwin-distributed-identifiability --epochs 500 --output path/to/figure.png
```

The figure directory is ignored by Git because the artifact is reproducible.

## 13. Code map

| Responsibility | Module |
| --- | --- |
| Noise-normalized Jacobian and pre-fit gate | `thermotwin.inference.distributed_identifiability` |
| Independent nodal/SSPRK3 truth | `thermotwin.simulation.distributed_independent` |
| Conventional property fit | `thermotwin.inference.distributed_properties` |
| Multi-experiment inverse PINN | `thermotwin.pinn.distributed_inverse` |
| Frozen observation-ablation study | `thermotwin.studies.distributed_observation_identifiability` |
| Text and figure report | `thermotwin.reports.distributed_observation_identifiability` |

## 14. Follow-on uncertainty audit

The supported full-data case now has representative fixed-coefficient nonlinear
profiles and a 20-trial independent-truth local-interval coverage audit. The
weak one-direction case is also profiled as a contrast, but its failed pre-fit
gate still prevents promotion to a property estimate. See
[`DISTRIBUTED_PROFILE_COVERAGE.md`](DISTRIBUTED_PROFILE_COVERAGE.md).
