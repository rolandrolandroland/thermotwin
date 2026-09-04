# ThermoTwin 90-second demonstration

## Before presenting

From the repository root, generate the current evidence once:

```bash
thermotwin-engineering-showcase
thermotwin-forward-reconstruction
```

Open the generated engineering showcase first, followed by the matched forward-
reconstruction figure. Keep the detailed README available for questions.

## Spoken script

> ThermoTwin asks a device-level question: given sparse and imperfect
> measurements, what can we infer about a thermoelectric heat pump, and what
> experiment or operating decision should come next?
>
> The model connects Peltier, Joule, and conductive heat flow to finite thermal
> and electrical contacts, exchanger dynamics, current schedules, PWM, and
> wall-plug COP. A conventional RK4 solver remains the numerical reference.
> PINNs are used where they add something different: hidden states, missing
> data, and unknown parameters.
>
> Here, two matched networks receive the same 56 noisy exchanger readings, with
> six readings missing around current turn-off. Neither sees the module-face
> temperatures. The data-only network fits the noisy samples slightly better,
> but its hidden-face error is 2.19 kelvin. Adding the four energy balances
> lowers that to 0.0071 kelvin and cuts the independently calculated energy-rate
> imbalance by 99.19 percent. That advantage holds across all five trials.
>
> The same twin can then infer hidden contact behavior, compare controls, and
> choose a more informative pulse. The selected 0.8 amp, 20 second experiment
> reduces nonlinear joint parameter error by 81.46 percent versus a naive
> choice and 11.77 percent versus a similar-energy control.
>
> These are reproducible synthetic results, not hardware validation. The next
> decisive step is to collect a calibrated device dataset and measure the gap
> between the model and the physical system.

## Screen sequence

1. **0–20 s — problem and architecture:** show the engineering showcase title
   and identify the conventional solver as the reference.
2. **20–55 s — strongest PINN evidence:** switch to the matched reconstruction
   figure; point first to the visible fit, then the diverging hidden data-only
   faces, then the energy audit.
3. **55–78 s — decisions:** return to the engineering showcase and point to
   inference, constrained experiment selection, and control comparison.
4. **78–90 s — boundary and next proof:** end on the sentence “synthetic
   validation—not hardware validation” and state the proposed hardware step.

## If there is time for one question

Ask: “Which parameter or hidden state would be most valuable to recover from a
real thermoelectric test stand, and which sensors are actually available?”
