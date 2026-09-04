# ThermoTwin technical summary

## The problem

A thermoelectric material does not operate in isolation. Device behavior also
depends on leg geometry, electrical and thermal contacts, heat exchangers,
power electronics, sensor limitations, and the current schedule. ThermoTwin
asks a system-level question:

> Given sparse and imperfect measurements, what can be inferred about a
> thermoelectric heat pump, which conclusions are identifiable, and which
> experiment or operating decision should come next?

ThermoTwin is an open-source, CPU-first research package for answering that
question with transparent models. It is a synthetic modeling and experiment-
planning environment, not a hardware-calibrated product model.

## What is modeled

The smallest model contains a cold node and a hot node. The main lumped model
uses four thermal states: cold thermoelectric face, hot thermoelectric face,
cold heat exchanger, and hot heat exchanger. It includes:

- Peltier heat, Joule heat, and parasitic thermal conduction;
- finite face-to-exchanger thermal resistance;
- exchanger thermal mass and reservoir conductance;
- fixed electrical-contact resistivity distinct from bulk leg resistance;
- constant, stepped, pulsed, bipolar, and thermally averaged PWM current;
- module, delivered, and wall-plug coefficient of performance;
- temperature noise, calibration bias, sensor lag, missing readings, and
  restricted sensor sets.

Fixed-step RK4 solvers provide the conventional numerical reference. Steps are
split at current transitions, and discontinuous electrical power is integrated
within constant-current segments rather than across fictitious ramps.

A separate distributed model represents one temperature-dependent
thermoelectric leg with conservative finite-volume physics and a PDE PINN. It
supports exploratory inference of `alpha(T)`, electrical resistivity
`rho_e(T)`, and thermal conductivity `kappa(T)`.

## Why physics-informed learning is used

ThermoTwin does not use a PINN simply to replace an ODE solver. The conventional
solver is faster and more accurate when every parameter and boundary condition
is known. Physics-informed learning is reserved for problems with hidden
states, sparse observations, missing readings, or unknown physical parameters.

The forward PINNs represent temperature as a differentiable function of time
or space-time. Automatic differentiation supplies temperature rates and the
training objective penalizes violations of the governing energy balances.
Inverse PINNs share unknown positive physical parameters across experiments
while fitting only the declared sensor observations.

## Strongest matched PINN result

The clearest test gives identically initialized, equal-capacity networks the
same 56 noisy heat-exchanger temperature readings. Six readings are absent
around current turn-off, and both thermoelectric-face temperatures are never
observed. Both models know the initial temperatures and current-switch times;
only one model receives the four node-balance residuals.

Across five paired trials:

| Mean metric | Physics-informed | Data-only |
| --- | ---: | ---: |
| Retained noisy-row RMSE | 0.019433 K | **0.017471 K** |
| Missing-exchanger RMSE | **0.009696 K** | 0.079878 K |
| Completely hidden-face RMSE | **0.007105 K** | 2.193724 K |
| Whole-system rate-closure RMS | **0.132833 W** | 16.493587 W |
| Absolute final cumulative closure error | **1.952169 J** | 370.392719 J |

The data-only network fits the retained noisy values slightly better and trains
about 3.8 times faster. The physics-informed model instead reduces missing-
interval error by 87.86%, hidden-face error by 99.68%, and energy-rate
imbalance by 99.19%. Each advantage holds in all five trials.

The energy diagnostic is assembled after training from thermal storage,
electrical power, reservoir heat, and external heat. It is independent of the
training objective as a calculation, but it is not a new independent law: it
is algebraically implied by the four node balances.

## Inference and identifiability

The lumped inverse workflow recovers hidden cold-side thermal contact
resistance from sparse sensor histories and transfers that estimate to complete
current regimes excluded from fitting. The conventional scalar optimizer is
the appropriate accuracy baseline and outperforms the inverse PINN on the
smallest ideal problem.

The more realistic sparse-sensor study jointly estimates contact resistance,
face capacitance or sensor dynamics, lag, and calibration bias. Sensitivity
matrices, singular values, correlations, multistart fitting, profiles, and
repeated-noise trials are used before claiming a parameter is identified. An
intentionally unmodeled 0.10 K sensor bias demonstrates that a falling loss can
coexist with a biased physical estimate.

## Experiment selection

The planner ranks feasible current pulses by expected joint information under
energy and temperature constraints. The selected 0.8 A, 20 s pulse is compared
with a naive 0.4 A, 5 s pulse and a closer-energy grid control.

Twenty complete nonlinear refits show that the selected pulse reduces mean
joint log-parameter RMSE by 81.46% relative to the naive choice and by 11.77%
relative to the closest-energy control. The second comparison matters because
it separates waveform placement from the trivial benefit of spending much
more energy.

A sequential follow-on gives adaptive selection, a precommitted D-optimal
batch, and an engineer heuristic four tests under the same 65 J cap. In 20
matched-equation trials, adaptation provides no material advantage; the
heuristic reaches the prediction gate at lower modeled energy. When an
independent truth model inserts an unobserved interface thermal mass, all three
policies fit the visible data at the 0.02 K noise scale and predict accessible
held-out sensors within 0.0065 K, yet all fail physical-parameter and hidden-face
recovery in 20/20 trials while reporting tight local uncertainty. The negative
result shows that more informative experiments cannot repair omitted physics.

The sensor model-discrimination follow-on then asks whether more terminal tests
or one added observable best resolves that ambiguity. Four exchanger-only
training pulses use 64.41 J, choose the right four-/five-state topology in
37/40 combined trials, and pass the physical decision gate in 32/40. One
27.54 J training pulse augmented by cold-face temperature chooses correctly and
passes in 40/40, with zero false confidence and 0.0060 K mean hidden-face RMSE
under five-state truth. Added heat rate and voltage also pass all trials; the
cold-face channel is the recommended demonstration because it directly audits
the hidden state without assuming that idealized heat-flux sensing is cheap.

## Engineering decisions beyond inference

ThermoTwin also tests decisions that connect material and device physics:

- continuous current versus seconds-scale pulsing at matched delivered cooling;
- cooling and heating COP across current, temperature lift, and contact loss;
- electrical-contact process windows versus leg length;
- standardized pulse fingerprints for virtual assembly screening;
- material/geometry/application co-design with Bayesian optimization and an
  equal-budget random baseline.

The package keeps uncomfortable results. A nominal 10 K efficiency winner
passes only 55.3% of trials under the declared virtual manufacturing spread.
Two other application screens already contain their tested-pool winners, so no
Bayesian-optimization improvement is claimed. These are model results, not
manufacturing predictions.

## Evidence boundaries

What the current project establishes:

- the equations, signs, limiting cases, and numerical workflows are tested;
- the reported synthetic cases are reproducible;
- physics constraints materially improve hidden-state and missing-data
  reconstruction in the matched case;
- identifiability can be tested before accepting inverse estimates;
- selected experiments can be evaluated by complete nonlinear refitting;
- negative and biased cases are retained.

What it does not establish:

- agreement between the lumped or distributed equations and a physical device;
- superiority of PINNs over every data-only or conventional estimator;
- manufacturing process capability, actual cost, or product performance;
- global parameter identifiability from a local synthetic analysis;
- calibrated neural uncertainty intervals.

## Reproduce the evidence

Install all optional dependencies and run the tests:

```bash
python3 -m pip install -e '.[all]'
python3 -m unittest discover -s tests
```

Generate the main engineering and PINN reports:

```bash
thermotwin-engineering-showcase
thermotwin-forward-reconstruction
```

Recompute the headline values used by the public summary and presentation:

```bash
thermotwin-release-audit
```

The full report stores figures and machine-readable sidecars under the ignored
`thermotwin/figures/` directory. The detailed assumptions and complete project
status are in [`README_detailed.md`](README_detailed.md) and
[`ROADMAP.md`](ROADMAP.md).
