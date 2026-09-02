# Cooling and heating COP operating-map experiment

## Engineering question

How do current, imposed temperature lift, and face-to-exchanger contact
resistance jointly change useful cooling, useful heating, and coefficient of
performance in the current generic ThermoTwin model?

This is an operating-envelope study, not a hardware performance claim. The
thermoelectric coefficients and thermal network remain the frozen generic
reference values.

## Three different temperature lifts

The experiment keeps three temperature differences separate:

1. external lift,
   $\Delta T_{\mathrm{ext}}=T_{h,\infty}-T_{c,\infty}$, between the fixed
   hot and cold reservoirs;
2. exchanger lift, $\Delta T_x=T_{x,h}-T_{x,c}$; and
3. module-face lift, $\Delta T_f=T_h-T_c$.

The reservoirs are placed symmetrically around 300 K. A 10 K external lift is
therefore represented by 295 K on the cold side and 305 K on the hot side.
Contact resistance normally makes the thermoelectric faces work across a
larger lift than the external reservoirs.

## Heat, power, and COP definitions

At steady state, useful cooling and heating are the reservoir-side rates

$$
\dot Q_{c,\mathrm{del}}
=G_c(T_{c,\infty}-T_{x,c}),
$$

$$
\dot Q_{h,\mathrm{del}}
=G_h(T_{x,h}-T_{h,\infty}).
$$

The module terminal power is

$$
P_{\mathrm{module}}
=VI
=\alpha I(T_h-T_c)+I^2R.
$$

The two delivered COPs are

$$
\mathrm{COP}_c
=\frac{\dot Q_{c,\mathrm{del}}}{P_{\mathrm{module}}},
\qquad
\mathrm{COP}_h
=\frac{\dot Q_{h,\mathrm{del}}}{P_{\mathrm{module}}}.
$$

At a true steady state, no face or exchanger is storing energy. Contact heat,
module heat, and delivered reservoir heat therefore agree on each side, and

$$
\dot Q_{h,\mathrm{del}}-\dot Q_{c,\mathrm{del}}
=P_{\mathrm{module}}.
$$

When both useful rates and power are positive, this gives the limiting check

$$
\mathrm{COP}_h=\mathrm{COP}_c+1.
$$

Cooling COP is not reported when the cold-side delivered heat is nonpositive;
that condition is not useful cooling. The same rule is applied independently
to heating COP.

## Why the map uses an algebraic steady solver

With constant material properties and fixed current, all four zero-storage
energy balances are linear in the four node temperatures. The implementation
solves

$$
\mathbf A
\begin{bmatrix}T_c&T_h&T_{x,c}&T_{x,h}\end{bmatrix}^{\mathsf T}
=\mathbf b
$$

directly for every operating point. Thermal capacitances do not appear because
they control approach time, not equilibrium. Tests independently verify all
four zero-rate balances and compare the algebraic solution with a long RK4
trajectory.

This reduces the full 840-point map to less than one second on the reference
machine instead of requiring hundreds of long transient simulations.

## Experiment grid

| Input | Values |
| --- | --- |
| Mean reservoir temperature | 300 K |
| External lift | 0, 5, 10, 15, 20, 25, 30 K |
| Current | 0.05 through 1.50 A in 0.05 A steps |
| Explicit symmetric contact resistance | 0.10, 0.25, 0.50 K/W per side |
| Reduced comparison | original two-node topology, no explicit contacts |
| Useful-rate threshold for a reported maximum COP | 1 W |
| Equal-load cooling comparison | 3 W |
| Equal-load heating comparison | 5 W |

The 1 W threshold prevents an extremely large ratio at nearly zero electrical
power and nearly zero capacity from being presented as a useful optimum. Raw
capacity remains available at every grid point.

The reduced model is a separate topology. Zero contact resistance is not
inserted into the four-node equations.

## Results: baseline 0.25 K/W contacts

The best cooling and heating COPs below are maxima only among points delivering
at least 1 W of useful heat.

| External lift | Best cooling COP | Current | Cooling at that point | Best heating COP | Current |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 K | 23.423 | 0.15 A | 1.36 W | 24.423 | 0.15 A |
| 5 K | 4.122 | 0.35 A | 1.52 W | 5.122 | 0.35 A |
| 10 K | 1.835 | 0.70 A | 2.70 W | 2.835 | 0.70 A |
| 15 K | 1.076 | 1.10 A | 3.89 W | 2.076 | 1.10 A |
| 20 K | 0.697 | 1.45 A | 4.38 W | 1.697 | 1.45 A |
| 25 K | 0.437 | 1.50 A | 3.02 W | 1.437 | 1.50 A |
| 30 K | 0.195 | 1.50 A | 1.39 W | 1.195 | 1.50 A |

The optimum moves toward higher current as lift increases. At high lift the
1.5 A current bound becomes active, and the maximum available cooling falls.

## Results: what the contacts cost

The fairest contact comparison asks both topologies to deliver the same useful
load. At the baseline 0.25 K/W resistance, the 3 W cooling result is:

| External lift | Reduced COP | Contact-aware COP | Contact COP change | Extra face lift |
| ---: | ---: | ---: | ---: | ---: |
| 0 K | 15.530 | 10.072 | -35.14% | 1.60 K |
| 5 K | 4.916 | 3.699 | -24.74% | 1.75 K |
| 10 K | 2.316 | 1.836 | -20.73% | 1.99 K |
| 15 K | 1.309 | 1.056 | -19.28% | 2.35 K |
| 20 K | 0.820 | 0.661 | -19.30% | 2.85 K |
| 25 K | 0.548 | 0.436 | -20.44% | 3.57 K |
| 30 K | infeasible | infeasible | — | — |

At 25 K, the 0.50 K/W case is already unable to deliver 3 W below 1.5 A. At
30 K, none of the explicit-contact cases reaches that target.

The equal-load matcher does not assume that cooling remains monotonic all the
way to the configured current ceiling. If the ceiling itself is below the
target, it scans the interval for the first below-to-above target crossing and
then bisects that rising branch. This still finds a feasible low-current
solution when excessive current has already pushed the endpoint past the
cooling maximum and back below the target.

For the 5 W heating comparison, the baseline contact COP penalty decreases
from 30.02% at 0 K to 4.68% at 30 K. This does not mean contacts become
unimportant. As useful heating becomes increasingly dominated by electrical
input, the reference COP approaches the behavior of a resistance heater, so a
fixed thermal-interface penalty becomes a smaller percentage of the ratio.

## Interpretation

The map exposes three engineering tradeoffs that a single 1 A simulation
cannot show:

- increasing current initially raises cooling capacity through the linear
  Peltier term, but its quadratic Joule cost eventually lowers COP;
- increasing external lift reduces cooling capacity and moves the useful COP
  optimum to higher current; and
- contact resistance forces a larger face lift and more current for the same
  delivered load, reducing delivered COP.

The numerical values are most useful as a transparent baseline and as targets
for later calibrated data. They are not a comparison with vapor-compression
hardware because this model has not been calibrated to a device and omits
pumps, fans, converters, fluid flow, ambient loss, and temperature-dependent
material properties.

## Reproduce

Print the numerical summary:

```bash
python3 -m thermotwin.cop_operating_map
```

Generate `thermotwin/figures/COP_OPERATING_MAP_EXPERIMENT/cop_operating_map.png`
and the colocated `cop_operating_map.json` source-data file:

```bash
python3 -m thermotwin.cop_operating_map_report
```

Implementation: [`cop_operating_map.py`](cop_operating_map.py) and
[`cop_operating_map_report.py`](cop_operating_map_report.py).

## Main limitations

- The parameters are generic constants rather than hardware calibration.
- Reservoir temperatures are fixed; fluid inlet/outlet and flow power are not
  modeled.
- The system contains one thermoelectric assembly.
- The current limit is an experiment setting, not a measured device rating.
- The map is steady state; startup speed and stored energy require the
  transient solver.
- Reported power is module terminal power. The PWM experiment adds a first
  wall-plug correction.
