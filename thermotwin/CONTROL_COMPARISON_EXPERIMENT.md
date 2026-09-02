# Continuous-versus-pulsed control experiment

## Question

At the same delivered cooling rate, does periodic current improve cooling COP
over an optimized continuous-current baseline in the current four-node
ThermoTwin model?

The pulse is allowed to win, tie, or lose. The comparison is intentionally
structured to prevent a transient pulse from receiving credit for temporarily
draining energy stored in the faces or exchangers.

## Useful cooling and electrical input

The reported useful cooling is heat extracted from the fixed cold reservoir,

$$
\overline{\dot Q}_{\mathrm{useful}}
=\frac{1}{\Delta t}\int_{t_0}^{t_1}
G_c\left(T_{c,\infty}-T_{x,c}\right)\,dt.
$$

This differs from instantaneous module-side $Q_c$. It represents the heat the
cold reservoir supplies to the cooled exchanger over the evaluation window.

Average electrical input and delivered cooling COP are

$$
\overline P_e=\frac{1}{\Delta t}\int_{t_0}^{t_1}V(t)I(t)\,dt,
\qquad
\mathrm{COP}_{\mathrm{delivered}}
=\frac{\overline{\dot Q}_{\mathrm{useful}}}{\overline P_e}.
$$

Electrical power jumps when a rectangular current command switches because
$P=\alpha(T_h-T_c)I+RI^2$. The implementation therefore inserts every current
transition as an integration boundary. It evaluates both endpoint powers with
the constant current belonging to that interval, preserving the left and
right limits instead of drawing a fictitious trapezoidal ramp through the
jump. Face temperatures remain continuous and are linearly interpolated.

The code also evaluates the net change of energy stored in all four thermal
capacitances. A schedule is accepted only when its mean storage drift is below
0.05 W over the comparison window.

## Fair comparison procedure

1. Warm every candidate for 360 s.
2. Evaluate the next 120 s, which is divisible by all tested pulse periods.
3. For each target cooling rate, solve for the continuous current that meets
   the target.
4. Sweep pulse periods of 10, 20, 30, and 60 s and duty cycles of 0.25, 0.50,
   0.75, 0.90, 0.95, and 0.99.
5. For every pulse shape, solve for the pulse amplitude that meets the same
   target cooling rate.
6. Reject shapes that cannot reach the target below 1.5 A, violate the
   285--315 K face-temperature limits, or fail the storage-drift check.
7. Compare the highest-COP tested pulse with the optimized continuous case,
   and report the best tested period at every feasible duty.

The RK4 reference solver is used because it is faster and more accurate for a
large control sweep than retraining a per-experiment PINN.

## Result

| Delivered cooling | Continuous current | Continuous COP | Highest-COP tested pulse | Pulse COP | COP change |
| ---: | ---: | ---: | --- | ---: | ---: |
| 2 W | 0.2231 A | 15.6014 | 0.2255 A, 10 s, 99% duty | 15.4675 | -0.86% |
| 5 W | 0.5866 A | 5.6543 | 0.5928 A, 10 s, 99% duty | 5.6036 | -0.90% |
| 8 W | 0.9945 A | 3.1518 | 1.0059 A, 10 s, 99% duty | 3.1191 | -1.04% |

Four pulse shapes cannot reach 5 W under the current limit; eight cannot reach
8 W. All reported highest-COP tested points satisfy the temperature and
periodic-storage checks.

The highest-COP tested point is now at the 99% duty boundary. This is expected:
as $D\rightarrow1$, the rectangular pulse becomes continuous current and its
penalty must approach zero. It is therefore misleading to describe the older
21.8--27.6% values as an optimized-pulse penalty; those values came from a
search capped at 75% duty.

The stronger, grid-independent result is the duty law. If mean current is held
fixed, an ideal zero-to-peak rectangular waveform has

$$
I_{\mathrm{peak}}=\frac{\overline I}{D},
\qquad
\overline{I^2}=\frac{\overline I^2}{D}.
$$

Thus its Joule heating is multiplied by $1/D$ relative to DC, while its
mean-current Peltier drive is unchanged to first order. At the 5 W target, the
best tested period at each feasible duty gives:

| Duty | Delivered COP | Change from continuous |
| ---: | ---: | ---: |
| 0.50 | 2.7268 | -51.77% |
| 0.75 | 4.2845 | -24.23% |
| 0.90 | 5.1263 | -9.34% |
| 0.95 | 5.3945 | -4.59% |
| 0.99 | 5.6036 | -0.90% |

The 25% point cannot reach 5 W below the 1.5 A current limit. The matched-load
penalty is not exactly $1/D$ because amplitude matching and thermal dynamics
also change the temperatures, but it decreases continuously toward zero as
duty approaches one.

The conclusion is the same when the comparison direction is reversed. Holding
electrical power equal to each highest-COP tested pulse, optimized continuous current
delivers more cooling:

| Pulse target | Continuous cooling at the same power | Pulse cooling change |
| ---: | ---: | ---: |
| 2 W | 2.0076 W | -0.38% |
| 5 W | 5.0209 W | -0.42% |
| 8 W | 8.0362 W | -0.44% |

That conclusion is useful. It identifies which additional physics would need
evidence before claiming a pulse advantage, such as flow-dependent heat
transfer, spatial module dynamics, temperature-dependent properties,
multi-assembly staging, or a different building-side objective.

## Resistance-uncertainty stress test

The fixed 5 W schedules were reevaluated at cold contact resistances of 0.20,
0.25, and 0.30 K/W. The pulse COP changes were -0.90%, -0.90%, and -0.89%.
The qualitative conclusion is therefore stable over this representative
interface range.

This is not a claim that the conclusion is stable to every uncertain thermal
parameter.

## Connection to the steady map and electrical PWM

The continuous and pulsed markers are now overlaid on the exact steady
zero-lift COP envelope in
[`PULSE_OPERATING_MAP_EXPERIMENT.md`](PULSE_OPERATING_MAP_EXPERIMENT.md).
The warmed continuous results agree with that algebraic envelope within 0.04%,
while every tested pulse remains below it. The plot now shows the full
duty-dependent approach to the continuous limit rather than only the winner.

These 10--60 s schedules are thermal control pulses, not high-frequency
switch-mode PWM. The separate
[`PWM_POWER_ELECTRONICS_EXPERIMENT.md`](PWM_POWER_ELECTRONICS_EXPERIMENT.md)
models mean current, mean-square current, and converter loss without stepping
the thermal model at an electrical switching frequency.

## Reproduce

```bash
python3 -m thermotwin.control_comparison
```

Generate the dedicated four-panel result figure, its JSON data, and its
plain-text explanation with:

```bash
python3 -m thermotwin.control_comparison_report
```

The artifacts are written to `figures/CONTROL_COMPARISON_EXPERIMENT/`.

The implementation is in [`control_comparison.py`](control_comparison.py).
The tests verify clipped continuous integration, switch-aware power
integration, output-grid independence, nonmonotonic capacity bracketing,
equal-capacity matching, safety, and storage checks in
[`../tests/test_control_comparison.py`](../tests/test_control_comparison.py).

## Limitations

- Fixed reservoir temperatures replace flowing-fluid inlet/outlet states.
- Material properties are temperature independent.
- The model contains one assembly rather than a staged multi-assembly system.
- The comparison optimizes simple rectangular pulses, not arbitrary control
  waveforms or closed-loop comfort control.
- Because 99% duty is almost continuous, a meaningful search for a nontrivial
  pulse optimum would need an enforced off-time, modulation-depth requirement,
  or device-specific switching objective.
- Synthetic results should not be interpreted as contradicting or validating
  performance claims for a different physical device.
