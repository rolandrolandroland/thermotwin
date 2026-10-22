# Spatial PINN failure diagnosis

Date: October 21, 2026 (America/New_York). Source examined: `0bfe84e`.

The 1D and 2D failures have different dominant explanations. **The 1D data
strongly constrain an effective resistivity level but weakly constrain curve
shape, with additional cubic-to-piecewise-linear model mismatch. The 2D data
identify contact conductance accurately with the conventional solver, but the
PINN does not learn a sufficiently accurate physical temperature field.**

Use the conventional inverse solver for current parameter estimates. Develop
the PINN first as a verified known-parameter forward reconstruction. Keep 3D
deferred for this extruded benchmark plate.

This is post hoc diagnosis, not a replacement validation. The original three
1D trials and three 2D trials remain scored exactly as run. All 123 indexed
artifacts validate, and all nine evaluation neural prediction arrays were
reproduced exactly from their saved checkpoints. No explanatory sign, unit,
coordinate, stale-checkpoint, or reference-output error was found. That is
evidence against these specific defects, not proof of hardware fidelity.

## 1D: the requested shape is more detailed than the data support

The present measurements contain 54 scalar observations: both face
temperatures and voltage at six times in three regimes. There are no internal
temperature sensors. Actual training temperatures span 294.83–305.21 K,
while the inferred property knots lie at 285, 300 and 315 K. The zero-current
regime provides exactly zero electrical-resistivity sensitivity in this model.

A conventional-forward finite-difference Jacobian, whitened by the declared
temperature/voltage noise, has singular values **1674.42, 9.87, 5.18** with
respect to log knot multipliers at the labeled cubic-at-knots reference.
It has full rank: this is weak practical precision, not structural rank loss.
The corresponding local linearized mode standard deviations are about
0.060%, 10.1% and 19.3% in log units. The true center-versus-endpoint contrast
is 0.035, versus a formal local standard deviation of 0.098. Similar spectra
occur at nominal, projected and inferred curves; changing the finite-difference
step changes the Jacobian by only about 1.5e-7 relatively.

These are optimistic local sensitivity diagnostics with other inputs known,
not calibrated nonlinear confidence intervals. Bounded nonlinear perturbations
also show that substantially different weak-direction shapes can produce
little observable change near the fitted curve.

The same conventional solver makes the ambiguity tangible. All observation
MSE values below are normalized by the declared channel noise variances:

| Frozen trial | Constant-scale fit observation MSE | Frozen PINN curve through conventional solver MSE | Total chi-square improvement of that curve |
| --- | ---: | ---: | ---: |
| 0 | 0.88878 | 0.88585 | 0.158 |
| 1 | 1.03424 | 1.03335 | 0.048 |
| 2 | 0.91415 | 0.91281 | 0.072 |

The constant-scale estimates are 1.06759–1.06804. This comparison concerns the
specific frozen curves, not the optimum of a three-parameter likelihood.
Nevertheless, the scalar model already fits at the declared noise level,
while the local information calculation shows poor shape precision.

There is also a real representation issue. Truth is cubic in temperature;
inference uses three piecewise-linear knots. Setting the inferred knots to
the actual cubic's values gives noise-normalized clean-observation MSE 0.77365. A diagnostic
function-space projection over the observed temperature interval gives
0.000462, with only 0.0297% relative curve RMS error in that interval—but a
knot-contrast ratio of 0.483, below the original 0.75 shape threshold.
Thus excellent local representation need not reproduce the cubic's actual
knot contrast. This truth-known projection is neither a training target nor
a uniquely identified data-effective material law.

Representation mismatch alone does not make the shape gate impossible:
the full-domain best representable projection satisfies its amplitude and
contrast limits. The difficulty is that the finite noisy data do not strongly
select that full-domain projection. Endpoint shape claims extrapolate beyond
the measured temperature range.

Optimization is an additional limitation: physics loss still falls 14–23%
over epochs 1800–2400. The denser residual grid gives 11–13% larger RMS, and
one trial changes from 0.2319 to 0.2578 of the nominal rate, crossing the
0.25 gate. More epochs are not an established shape remedy: coefficient
error worsens from epochs 1200–2400 in two trials despite falling physics loss.

**1D change:** report the effective scalar level and in-range predictions
first. If full curve recovery remains the claim, design broader temperature
excitation and verify shape precision at the intended tolerance before
training. Separate a matched piecewise-linear-truth positive control from
the cubic model-mismatch challenge; report in-range function error, its
representation floor, and endpoint extrapolation separately.

## 2D: the field and contact parameter compensate for one another

Conventional fitting recovers h with 0.053–0.209% error in all three trials;
the unmodified initial h has observation MSE about 56–58. There is useful
contact information in these measurements. In contrast, inverse PINNs end
at h=493–549 instead of 1000, with 45–51% error. The known-h neural oracle
also fails, with contact-flux RMS errors around 500–602 W/m² and interior
rate residuals around 0.016–0.017 K/s. Inadequate conditional field learning
therefore exists even without scalar inference.

The measured training mechanism is more specific than "wrong weights":

- At initialization, every inverse h gradient drives h downward before the
  random temperature field has been learned. The unconstrained Robin least-
  squares h optimum for these random fields is negative; its bounded optimum
  is the 400 lower bound. This is not evidence that physical h is small.
- At the final fields, the analytic conditional h optima are still only
  about 490–566 on training/dense contact points. Substituting true h into
  these biased fields makes their Robin residual worse. A scalar-only
  correction cannot repair the field.
- Final data and weighted-physics field gradients have nearly equal norms
  and cosine similarities from -0.971 to -0.987. At initialization, physics
  gradients dominate by 5.3–7.4 times. Uniform physics domination is therefore
  not an accurate explanation of the final state.
- Training residuals already fail; independent residuals usually fail by
  more. All 13 development/evaluation fits exhaust 200 L-BFGS iterations.
  Evaluation inverse objectives still drop 7.9–11.1% over their final 50
  closure evaluations.
- Collocation is fixed and sparse. One trial has no contact points before
  10 s, and none has joint early-time/contact-edge coverage. However, inverse
  maximum errors occur at 20 s inside the contact patch. Edge and early-time
  coverage are contributors to investigate, not a sufficient sole diagnosis.

The 10:1 plate aspect ratio also yields a 100:1 ratio between dimensionless
diffusion coefficients. This is correct physics and a possible conditioning
factor; it is not a units error or a demonstrated sole cause.

## Bounded disposable controls performed for this diagnosis

Three controls use a new diagnostic seed namespace, matched initial field,
1800 Adam epochs and at most 200 L-BFGS iterations. Validation uses different
spatial points and time midpoints from the dense supervision grid. A fourth
control restarts L-BFGS from the partial-contact known-h result with a fixed
additional 1000-iteration cap and the same data/collocation. No best checkpoint
was selected, and no original evaluation model was retrained.

| Control | Independent field RMS / max (K) | Important qualification |
| --- | --- | --- |
| Partial-contact, known h | 0.06217 / 0.23938 | Fails field gate; contact residual 542 W/m² |
| Full-contact, known h | 0.00631 / 0.01522 | Field RMS improves 9.85-fold; normalized PDE RMS 0.091 still exceeds 0.05 |
| Partial-contact, dense truth-supervised field | 0.02378 / 0.13296 | Passes value gates with privileged labels; normalized PDE RMS 1.198 fails badly |
| Partial-contact, extra 1000 L-BFGS iterations | 0.03693 / 0.17387 | Still fails maximum field and independent physics gates |

Simplifying the boundary geometry strongly helps this paired control. The
network can also approximate temperature **values** under dense supervision.
Neither result establishes adequate flux/derivative representation, and the
dense-label control is not an inverse competitor. Full contact changes both
boundary topology and lateral spreading; this pair does not isolate edge
regularity alone.

The continuation lowers the total training objective from 73.00 to 3.37 and
the summed normalized training physics MSE from 0.7077 to 0.03063. Yet the
independent normalized PDE RMS worsens from 0.346 to 0.410, and normalized contact/insulation RMS
remain 0.587/0.617 against the 0.05 limits. More optimization on the same
points improves the sampled objective without establishing a physical solution.

## Recommended next sequence

1. **Keep the conventional inverse solver as the current reference and
   working estimator.** Preserve the negative PINN findings and avoid a
   general PINN-advantage claim.
2. **Repair the 2D known-h forward problem first.** Require the existing
   field, observation, and independent PDE/boundary gates for all applicable
   families. Begin with full contact (insulated-bottom residual is N/A), then
   partial contact (both bottom regions must pass). Do not reopen inverse evaluation if this
   conditional problem still fails.
3. **Change sampling and constraint handling in a bounded development ablation.**
   Separate contact and insulated-bottom losses; stratify spatial regions
   and time, including the early transient; refresh Adam collocation while
   keeping deterministic independent checks and a fixed L-BFGS batch. Monitor
   each family's physical residual and gradient conflict. Compare a declared
   per-family constraint/balancing strategy with the original global weight.
   If derivative errors persist near the mixed boundary, test an integrated
   finite-volume/weak residual or a boundary-aware decomposition before
   increasing network size. These are candidates, not proven fixes.
4. **Then test delayed or alternating h updates.** Warm-start the field at a
   declared nominal h, not the unknown true h, before releasing h. Diagnose
   field/parameter compensation after every stage. An analytic Robin h update
   remains conditional on field accuracy and is not a substitute for it.
5. **For 1D shape, improve information before increasing model freedom.**
   Retain a scalar-level baseline. Choose wider thermal/current experiments
   using noise-whitened sensitivity and nonlinear checks of the intended
   shape tolerance. Add sensors only if they improve the weak modes. Use a
   matched-truth control before the cubic mismatch challenge.
6. **Freeze a revised rule and use entirely new untouched evaluation data.**
   The v1 results remain valid under their original protocol, but all inspected
   v1 trials and these controls are now development evidence for any revision.
   Retain failure denominators and independent residual gates. Keep the
   extruded plate in 2D; 3D does not repair these training or information failures.

The immediate implementation should therefore be a **development-only 2D
forward-training revision with h fixed**, alongside the 1D scalar baseline
and excitation-design diagnostic. A larger 3D PINN is not the next experiment.

Loss-gradient imbalance and optimization difficulty have primary-source
precedent, but those papers do not prove which remedy will work here:
[Wang, Teng and Perdikaris](https://arxiv.org/abs/2001.04536) analyze gradient
pathologies; [Krishnapriyan et al.](https://papers.nips.cc/paper/2021/hash/df438e5206f31600e6ae4af72f2725f1-Abstract.html)
demonstrate conditioning and curriculum/time-window approaches; the
[training guide](https://arxiv.org/abs/2308.08468) supplies reproducible ablations.
Our next choice should be decided by the bounded controls and independent
gates above.

## Local evidence and reproducibility

All diagnostic scripts and raw outputs are in the ignored directory
`reports/pinn-spatial-validation/diagnosis-20261021/`:

- `one-d/summary.json`, `diagnosis.json`, `diagnose.py`, and
  `history-analysis/history-analysis.json`; `one-d-diagnosis.png` shows the
  curves, singular values and nonlinear weak-direction perturbations.
- `two-d/findings.json`, `diagnostics.json`, `forensic.py`, and the spatial/
  conditional-h figures reproduce the saved-state diagnosis without fitting.
- `control_probe.py`, `controls-v1/plan-before-fitting.json` and
  `controls-v1/result.json` record the three new controls.
- `continuation_probe.py`, `continuation-v1/plan-before-fitting.json` and
  `continuation-v1/result.json` record the bounded restart control, including
  the audited equality of its starting objective to its parent checkpoint.

Use the pinned CPython 3.10.12 environment with one Torch/BLAS thread. Scripts
record source hashes and configurations; use fresh output directories for
reproduction and preserve the original runs. Projection benchmarks and dense
truth supervision are explicitly exploratory and must not enter an inverse
estimator's information set. Final diagnostic evidence hashes are recorded in
`evidence-index.json` under this diagnosis directory.
