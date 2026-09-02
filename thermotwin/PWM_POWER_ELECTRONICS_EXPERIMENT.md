# Power-electronics-aware averaged PWM experiment

## Engineering question

At the same mean thermoelectric current, how do ideal DC current, a
well-smoothed PWM-derived current, and direct zero-to-peak current chopping
differ in Joule heat, delivered capacity, module COP, and wall-plug COP?

## Why this is not implemented with shorter thermal time steps

Electrical PWM may switch thousands of times during one meaningful thermal
time constant. Resolving every edge with the four-node thermal integrator would
be expensive and would still omit the electrical components that determine
current ripple.

The thermal equations need two waveform statistics:

- Peltier heat uses mean current, $\overline I$;
- Joule heat uses mean-square current, $\overline{I^2}$.

The implementation therefore averages the electrical waveform first and then
solves the contact-aware steady thermal balances. Scalar DC, averaged PWM, and
material/geometry co-design all delegate to one shared four-node steady kernel
that accepts $(\overline I,\overline{I^2})$. The scalar limit supplies
$(I,I^2)$, preventing duplicated matrices from drifting apart. This remains a
first power-electronics layer, not a switching-converter circuit simulation.

## Current models

### Ideal DC reference

$$
\overline{I^2}=\overline I^2.
$$

This reference has no converter loss. It separates the thermoelectric module's
best possible constant-current behavior from the two PWM-derived cases.

### Direct current PWM

Current alternates between zero and $I_{\mathrm{peak}}$ for duty cycle $D$:

$$
\overline I=D I_{\mathrm{peak}},
\qquad
\overline{I^2}=D I_{\mathrm{peak}}^2.
$$

At the same mean current, its Joule multiplier relative to DC is

$$
\frac{\overline{I^2}}{\overline I^2}=\frac{1}{D}.
$$

Low duty cycle is therefore especially costly when the module current itself
is chopped directly.

### Smoothed PWM-derived current

The simplified smoothed case has mean current $\overline I$ and triangular
peak-to-peak ripple $\Delta I_{pp}$. Its mean square is

$$
\overline{I^2}
=\overline I^2+\frac{\Delta I_{pp}^2}{12}.
$$

For ripple fraction $r=\Delta I_{pp}/\overline I$, the Joule multiplier is

$$
1+\frac{r^2}{12}.
$$

The frozen 10% peak-to-peak ripple therefore gives a multiplier of
1.000833, much closer to DC than direct PWM.

## Averaged thermoelectric equations

At face temperatures that are effectively constant over one electrical
switching cycle,

$$
\overline Q_c
=\alpha T_c\overline I
-\frac{1}{2}R\overline{I^2}
-K(T_h-T_c),
$$

$$
\overline Q_h
=\alpha T_h\overline I
+\frac{1}{2}R\overline{I^2}
-K(T_h-T_c),
$$

$$
\overline P_{\mathrm{module}}
=\alpha(T_h-T_c)\overline I
+R\overline{I^2}.
$$

This is a time-scale-separation closure, not an exact identity for arbitrary
thermal ripple. In general,

$$
\overline{I T_c}
=\overline I\,\overline T_c+\mathrm{Cov}(I,T_c),
$$

with analogous covariance terms for $T_h$ and $T_h-T_c$. The current-moment
model sets those covariance terms to zero. That approximation is appropriate
when the electrical switching period is much shorter than the thermal time
constants and face-temperature ripple within a switching cycle is negligible.
If switching and thermal time scales become comparable, the full coupled
waveform must be resolved or the covariance must be supplied by a faster
electrothermal model.

Within this zero-covariance moment closure, the energy check remains exact:

$$
\overline Q_h-\overline Q_c
=\overline P_{\mathrm{module}}.
$$

## First wall-plug correction

For both PWM-derived cases, the supply power is

$$
P_{\mathrm{supply}}
=\frac{P_{\mathrm{module}}}{\eta_{\mathrm{converter}}}
+P_{\mathrm{switching,fixed}}.
$$

The generic experiment uses 95% efficiency and 0.05 W fixed switching loss.
These are explicit study assumptions, not component measurements. Wall-plug
COP divides useful delivered heat by this supply power. Module COP remains
available separately.

## Experiment grid

| Setting | Value |
| --- | --- |
| External lifts | 0, 10, 20 K |
| Mean currents | 0.15 to 1.20 A in 0.15 A steps |
| Contacts | 0.25 K/W per side |
| Direct-PWM peak current | 1.50 A |
| Smoothed ripple | 10% peak-to-peak |
| Converter efficiency | 95% |
| Fixed switching loss | 0.05 W |

All three current cases are evaluated at the same mean current and external
temperature lift. This comparison isolates waveform and converter effects; it
does not claim equal supply power or equal delivered load.

## Representative results at 0.60 A mean current

| External lift | Current case | RMS current | Joule multiplier | Delivered cooling | Wall cooling COP | Wall heating COP |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 K | Ideal DC | 0.600 A | 1.000 | 5.104 W | 5.516 | 6.516 |
| 0 K | Smoothed PWM | 0.600 A | 1.001 | 5.104 W | 4.981 | 5.885 |
| 0 K | Direct PWM | 0.949 A | 2.500 | 4.613 W | 2.137 | 3.065 |
| 10 K | Ideal DC | 0.600 A | 1.000 | 1.950 W | 1.757 | 2.757 |
| 10 K | Smoothed PWM | 0.600 A | 1.001 | 1.950 W | 1.600 | 2.511 |
| 10 K | Direct PWM | 0.949 A | 2.500 | 1.459 W | 0.620 | 1.550 |
| 20 K | Ideal DC | 0.600 A | 1.000 | -1.204 W | not cooling | 0.070 |
| 20 K | Smoothed PWM | 0.600 A | 1.001 | -1.204 W | not cooling | 0.064 |
| 20 K | Direct PWM | 0.949 A | 2.500 | -1.695 W | not cooling | 0.266 |

At 10 K lift and 0.60 A mean current, the generic converter assumptions alone
reduce the smoothed case's wall cooling COP by about 8.9% from ideal DC, while
direct current chopping reduces it by about 64.7%. Direct PWM also removes
about 25% less useful heat than the smoothed case at this same mean current.

The 20 K, 0.60 A row is deliberately retained even though it is not cooling.
A positive current command does not guarantee useful cold-side heat removal at
every imposed lift.

Heating COP can respond differently to added Joule heat because electrical
resistance itself delivers heat. A higher heating number caused by extra Joule
heat should not be confused with more efficient heat pumping, especially when
the cold side is no longer being cooled.

## Interpretation

The phrase “PWM drive” is physically incomplete unless the module current
waveform is specified. A switch-mode converter that produces nearly continuous
TEC current behaves close to DC at the module, with a separate converter-loss
penalty. Directly chopping the TEC current at low duty cycle preserves mean
Peltier drive but greatly increases RMS-current Joule heat.

This explains why the seconds-scale pulse study and high-frequency PWM layer
must remain separate:

- seconds-scale pulses intentionally move the thermal states;
- high-frequency PWM is averaged while the thermal states are nearly fixed
  during each electrical period.

## Reproduce

Print the numerical experiment:

```bash
python3 -m thermotwin.pwm_power_electronics
```

Generate `thermotwin/figures/PWM_POWER_ELECTRONICS_EXPERIMENT/pwm_power_electronics.png`
and the colocated `pwm_power_electronics.json` source-data file:

```bash
python3 -m thermotwin.pwm_power_electronics_report
```

Implementation:
[`pwm_power_electronics.py`](pwm_power_electronics.py) and
[`pwm_power_electronics_report.py`](pwm_power_electronics_report.py).

## Main limitations and next electrical steps

- Converter efficiency and fixed loss are assumed constants.
- Ripple is prescribed; voltage, switching frequency, inductance, switching
  loss, dead time, and closed-loop current control are not solved.
- There is no RMS-current device limit or component thermal model yet.
- Electromagnetic interference and current/voltage sensor bandwidth are not
  modeled.
- The closure neglects current-temperature covariance within an electrical
  cycle; its validity requires negligible thermal ripple at the switching
  frequency.
- The comparison is synthetic and steady state.

A later hardware-calibrated electrical model should derive ripple and loss
from a specific converter topology and measured operating points. The current
moment interface is designed so those improved electrical outputs can feed the
thermal model without resolving every switching edge.
