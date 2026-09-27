# Experiment selection: a complete four-panel story

This presentation combines the candidate search in
`NEXT_EXPERIMENT_WALKTHROUGH.md` with the complete nonlinear checks and paired
comparisons in `NONLINEAR_EXPERIMENT_SELECTION.md`. It creates a new figure;
the earlier figures and their numerical evidence remain available.

## Read the panels in order

1. **Apply a pulse; read two sensors.** Change the electrical current and record
   the two heat-exchanger temperatures. The illustrated pulses all start at
   5 seconds, with different strengths and durations. Readings continue to
   80 seconds, beyond the first 40 seconds illustrated. The unknowns are how
   easily heat crosses an interface, how much heat the cold face stores, and
   how quickly the sensors respond.
2. **Choose a feasible pulse before collecting data.** The conventional physics
   model ranks 25 candidate pulses by their expected information about those
   three unknowns. Seventeen pass the 30 J energy limit and the 285–315 K
   face-temperature checks. All eight rejected candidates exceed the energy
   limit in this frozen grid; the temperature constraints are checked but do
   not bind. The selected pulse is 0.8 A for 20 seconds, using 27.54 J.
3. **Check parameter recovery with full fits.** Each of 20 trials varies the
   hidden properties, sensor biases, and noise. Every pulse sees the same truth
   and noise within a trial. The selected pulse's root-mean-square percentage
   errors are 4.96% for contact resistance, 2.15% for heat storage, and 8.71%
   for sensor response time. Against the naive pulse, the corresponding error
   reductions are approximately 6.55x, 6.60x, and 3.92x; all three directions
   improve, with materially different magnitudes. These errors are also lower
   than the control values in this campaign. The plotted metric is
   `100 * sqrt(mean((estimate/truth - 1)^2))`,
   computed separately for each physical parameter.
4. **Inspect the harder comparison, trial by trial.** Each dot subtracts the
   selected pulse's combined log-parameter error from the control's error for
   one paired trial. Positive values favor selected. The selected pulse wins
   20/20 comparisons against naive but only 12/20 against the closer-energy
   control. Its mean combined error is 81.46% lower than naive and 11.77% lower
   than control. The latter advantage is small and variable; these trials do
   not establish general superiority. The mean reduction is a ratio of mean
   errors, not the mean of per-trial percentage reductions.

## Why the comparison matters

The naive pulse uses only 1.64 J, versus 27.54 J for selected. The control uses
23.77 J, providing a more demanding comparison, but the energies are not equal.
The result supports choosing an informative next test under constraints. It
does not isolate timing from energy perfectly or measure a reduction in real
prototype count or development time.

Panel 2 is a local pre-experiment prediction. Panels 3–4 are complete nonlinear
refits. The earlier 250-trial linearized validation is deliberately excluded
from the headline to avoid confusing it with these 20 nonlinear trials.
This study uses conventional inference, not PINNs. Truth and fitting still
share the same synthetic lumped equations.

## Reproduce from saved evidence

```bash
python -m thermotwin.reports.experiment_selection_summary
```

Inputs are the existing `experiment_selection.json` and
`nonlinear_experiment_selection.json` sidecars. The command performs no new
simulation or fitting. It writes `experiment_selection_summary.png`, a JSON
containing plotted values and source hashes, and a TXT explanation under
`thermotwin/figures/EXPERIMENT_SELECTION_SUMMARY/`.

The original figures are preserved. Optional `--planner`, `--nonlinear`, and
`--output` arguments select alternate files.
