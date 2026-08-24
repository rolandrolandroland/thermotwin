# Seconds-scale pulse connection to the COP operating map

## Engineering question

Where do the tested 10--60 s rectangular pulses lie relative to
the steady continuous-current cooling envelope?

This experiment does not rerun the pulse study under a new objective. It
connects the existing fair transient comparison to the new steady map and
checks that the continuous transient baseline has actually reached the same
operating curve.

## Two time scales that must not be confused

The pulses in this experiment have periods of 10, 20, 30, and 60 s. Face and
exchanger temperatures respond during each on/off cycle, so the four-node RK4
solver resolves those thermal transients.

Power-electronics PWM normally switches much faster. It is treated separately
in [`PWM_POWER_ELECTRONICS_EXPERIMENT.md`](PWM_POWER_ELECTRONICS_EXPERIMENT.md)
using averaged current moments rather than forcing the thermal solver to step
at the switching frequency.

## Fair transient comparison retained from the original study

For each 2, 5, and 8 W cooling target:

1. continuous current is solved to deliver the target;
2. pulse period and duty cycle are swept;
3. pulse amplitude is solved to deliver the same target;
4. the model warms for 360 s and is evaluated for 120 s;
5. current, face-temperature, feasibility, and stored-energy-drift limits are
   checked; and
6. the feasible pulse with the highest delivered COP is retained, while the
   best tested period at every feasible duty is also kept for the duty curve.

Useful heat is heat removed from the cold reservoir, not instantaneous module
$Q_c$. The mean stored-energy drift must be below 0.05 W so temporary cooling
of a thermal mass cannot masquerade as useful heat removal.

## Why mean and RMS current are shown

For an ideal rectangular pulse that alternates between zero and peak current,

$$
I_{\mathrm{mean}}=D I_{\mathrm{peak}},
$$

$$
I_{\mathrm{rms}}=\sqrt{D}\,I_{\mathrm{peak}}.
$$

The Peltier term is linear in current and therefore tracks mean current to
first order. Joule heating is proportional to $I^2R$ and therefore tracks
$I_{\mathrm{rms}}^2$. For $0<D<1$, RMS current is greater than mean current.

## Results

| Cooling target | Continuous current | Continuous COP | Highest-COP tested pulse peak / mean / RMS | Pulse setting | Pulse COP | COP change |
| ---: | ---: | ---: | --- | --- | ---: | ---: |
| 2 W | 0.2231 A | 15.6014 | 0.2255 / 0.2232 / 0.2244 A | 10 s, 99% | 15.4675 | -0.86% |
| 5 W | 0.5866 A | 5.6543 | 0.5928 / 0.5869 / 0.5898 A | 10 s, 99% | 5.6036 | -0.90% |
| 8 W | 0.9945 A | 3.1518 | 1.0059 / 0.9958 / 1.0008 A | 10 s, 99% | 3.1191 | -1.04% |

The continuous transient COP differs from the exact steady-map COP by only
0.021%, 0.029%, and 0.040%. The continuous markers can therefore be treated as
points on the steady envelope within the declared transient settling error.

Every tested pulse marker lies below that envelope. At the same electrical
power, the highest-COP tested pulses deliver 0.38%, 0.42%, and 0.44% less
cooling at the three targets.

All retained points satisfy the 0.05 W stored-energy-drift acceptance limit.
The negative pulse result is therefore not caused by crediting an unfinished
transient to one schedule and not the other.

The apparent optimum is the 99% duty boundary because it is almost the
continuous baseline. The previous 75%-limited grid produced 21.8--27.6%
penalties, but those percentages were properties of that duty ceiling, not a
grid-independent optimum. For fixed mean current, direct rectangular pulsing
multiplies mean-square current and Joule heat by $1/D$; as $D\rightarrow1$,
that multiplier and the matched-load COP approach the continuous limit. The
report therefore plots COP change against duty for every cooling target.

## Interpretation

In the current constant-property, fixed-reservoir model, a nontrivial
seconds-scale thermal pulse does not create a mechanism strong enough to
overcome its higher RMS-current Joule penalty. The robust result is the duty
trend, not the penalty at an arbitrarily selected duty ceiling. The conclusion
applies only to this tested model and objective.

It does not establish that every real pulse strategy loses. Flow-dependent
heat transfer, temperature-dependent coefficients, spatial module behavior,
staged assemblies, comfort constraints, and device-specific electronics could
change the comparison. Such mechanisms need equations or hardware evidence
before being added.

## Reproduce

Print the connected comparison:

```bash
python3 -m thermotwin.pulse_operating_map
```

Generate `thermotwin/figures/pulse_operating_map.png`:

```bash
python3 -m thermotwin.pulse_operating_map_report
```

Implementation: [`pulse_operating_map.py`](pulse_operating_map.py), which
reuses [`control_comparison.py`](control_comparison.py) and
[`cop_operating_map.py`](cop_operating_map.py).
