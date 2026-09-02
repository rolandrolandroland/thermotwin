# Electrical-contact process window

## Decision this experiment supports

This experiment asks:

> What specific electrical contact resistivity must an interface process
> achieve before a chosen leg geometry and cooling application are worth
> building?

It converts a material/interface measurement into three device-level views:

1. the fraction of module electrical resistance consumed by contacts;
2. the corresponding loss of device $ZT$;
3. the cooling COP and application feasibility after thermal contacts,
   exchangers, converter loss, voltage, and current density are included.

The output is a **process window**, not a fitted property or a hardware claim.

## Headline result

For the published 1.5 mm unicouple used as the electrical landmark, the
analytical 50% contact-resistance crossover is

$$
\rho_{c,50}=1.1069\times10^{-8}\ \mathrm{\Omega\,m^2}.
$$

The paper reports approximately 7.4 mΩ per contact. Multiplying by its
2.25 mm² leg-face area gives the ideal full-area translation

$$
\rho_{c,\mathrm{paper}}\approx1.6650\times10^{-8}\ \mathrm{\Omega\,m^2}.
$$

At that value, ThermoTwin assigns **60.07%** of the idealized unicouple's
electrical resistance to the four interfaces. The directly reported estimate,
four contacts divided by the approximately 50 mΩ total device resistance, is
**59.20%**. The small difference is expected because the model reconstructs
bulk resistance from reported transport properties rather than forcing the
reported total resistance.

With material properties and thermal conductance held fixed, contact resistance
reduces device $ZT$ by the same fraction that it increases total resistance.
The paper-derived point therefore retains about **39.93%** of the zero-contact
device $ZT$ in this electrical-only translation.

These numbers do not claim that the published generator has a cooling COP.
They translate its reported electrical quantities. The cooling panels are a
separate, declared $N=120$ system study.

## Source-specific inputs

The opt-in Ag₂Se record is sourced from
[Bappy et al., *Materials Horizons* (2026), DOI 10.1039/D6MH00220J](https://doi.org/10.1039/D6MH00220J).
It is kept outside the fixed StarryData catalog so the original co-design
campaign remains byte-for-byte reproducible.

| Quantity | Value used | Role |
| --- | ---: | --- |
| Optimized Ag₂Se Seebeck coefficient | −153.3 µV/K | n-leg material input |
| Optimized Ag₂Se electrical conductivity | 117,400 S/m | n-leg material input |
| Optimized Ag₂Se thermal conductivity | 0.85 W/(m·K) | n-leg material input |
| Excess selenium | 9% | provenance only |
| Synthesis | 350 °C for 90 min | provenance only |
| Sintering | 375 °C for 60 min | provenance only |
| Published leg dimensions | 1.5 × 1.5 × 1.5 mm | electrical landmark |
| Published p-leg Seebeck coefficient | 210 µV/K | infer p conductivity |
| Published p-leg power factor | 2.1 mW/(m·K²) | infer p conductivity |
| Reported resistance per contact | about 7.4 mΩ | infer $\rho_c$ |
| Reported total device resistance | about 50 mΩ | independent share check |

The optimized material triplet is same-sample. It does not mix the optimized
Seebeck and conductivity with the baseline sample's thermal conductivity.

## Electrical equations

For $N$ p/n couples, equal leg length $L$, equal leg area $A$, and four
metal/thermoelectric interfaces per series couple,

$$
R_{\mathrm{bulk}}
=N\frac{L}{A}\left(\rho_p+\rho_n\right),
$$

$$
R_{\mathrm{contact}}=4N\frac{\rho_c}{A},
$$

$$
f_{\mathrm{contact}}
=\frac{R_{\mathrm{contact}}}
{R_{\mathrm{bulk}}+R_{\mathrm{contact}}}.
$$

At 50% contact share, the two resistance contributions are equal:

$$
\rho_{c,50}=\frac{L(\rho_p+\rho_n)}{4}.
$$

$N$ and $A$ cancel from the crossover. This is why leg length is the geometric
coordinate that exposes the electrical contact transition. Area still matters
to absolute current, resistance, heat rate, footprint, and current density.

For the paper landmark, the p-leg conductivity is inferred without inventing a
new material record:

$$
\sigma_p=\frac{PF_p}{S_p^2}=47{,}619\ \mathrm{S/m}.
$$

That gives 19.679 mΩ ideal bulk resistance for both legs and 29.6 mΩ for four
reported contacts, consistent with the two independent contact-share estimates
above.

The system-level device metric is

$$
ZT_{\mathrm{device}}=\frac{\alpha^2T}{RK}.
$$

The plotted device $ZT$ includes bulk electrical resistance, areal electrical
contacts, leg thermal conductance, and the declared 0.04 W/K package parasitic.

## Cooling process-window design

The system sweep deliberately excludes the synthetic co-design cost index.
Otherwise short legs would be rewarded by an assumed material-volume cost before
the contact and thermal physics could be read cleanly.

| Coordinate or assumption | Values |
| --- | --- |
| Leg length | 0.05–2.5 mm, 61 log-spaced values |
| Specific contact resistivity | zero reference plus 61 log-spaced values from $10^{-11}$ to $5\times10^{-8}$ Ω·m² |
| Couple count | 120 |
| Leg area | 1.6 mm² |
| Material pairs | reference 9107 + 10562, then optimized Ag₂Se with all six p-type catalog records |
| Applications | unchanged 10 K efficiency-first, 25 K balanced, and 10 K capacity-first specifications |
| Current-density limits | 1 A/mm² existing campaign constraint; 3 A/mm² exploratory sensitivity |
| Current rule | lowest rising-branch current that delivers the required cooling |
| Thermal contacts | symmetric 0.25 K/W |
| Reservoir conductances | 2.5 W/K cold, 5.0 W/K hot |
| Converter and ripple | existing declared assembly assumptions |
| Cost | excluded |

The contact-resistivity upper bound is extended to $5\times10^{-8}$ Ω·m² so
the paper-derived landmark lies inside the plotted domain.

## Why logarithmic leg sampling is necessary

The 0.05–2.5 mm range spans a factor of 50. Linear sampling would devote most
points to long legs and poorly resolve the intended thin-leg regime. A
logarithmic grid gives comparable resolution per multiplicative change in
length.

Short legs face two separate penalties:

- $R_{\mathrm{contact}}/R_{\mathrm{bulk}}$ grows as $1/L$ because areal
  contact resistance does not shrink with leg length;
- $K_{\mathrm{legs}}=NA(\kappa_p+\kappa_n)/L$ also grows as $1/L$, increasing
  hot-to-cold conduction.

For the Ag₂Se + p-9107 system assumptions, module $K$ changes from 7.155 W/K at
0.05 mm to 0.1823 W/K at 2.5 mm, a factor of about 39.2. The full four-node
solver uses the resulting face-temperature difference; multiplying $K$ by the
external reservoir lift would not be the actual conductive heat leak.

The one-dimensional cross-plane leg model remains a useful first idealization
for wide, thin printed legs, but it does not include contact spreading
resistance. That omission becomes more important as the aspect ratio flattens.

## Process-window results

The 61-point length grid contains 1.4839 mm as its nearest value to the
published 1.5 mm leg. At that geometry, the largest contact resistivity that
still satisfies each full application specification is:

| Material pair | Application | Maximum feasible $\rho_c$ (Ω·m²) |
| --- | --- | ---: |
| reference 9107 + 10562 | 10 K efficiency-first | $2.1334\times10^{-8}$ |
| reference 9107 + 10562 | 25 K balanced | $1.3936\times10^{-8}$ |
| reference 9107 + 10562 | 10 K capacity-first | $1.8511\times10^{-8}$ |
| Ag₂Se + p 9107 | 10 K efficiency-first | $1.6061\times10^{-8}$ |
| Ag₂Se + p 9107 | 25 K balanced | $9.1028\times10^{-9}$ |
| Ag₂Se + p 9107 | 10 K capacity-first | $1.3936\times10^{-8}$ |

The 1 and 3 A/mm² cases give the same ceilings at this near-1.5 mm section;
the target current is below the existing campaign constraint there. The two
limits do separate in the thin-leg part of the map. They are reported side by
side: 1 A/mm² preserves the frozen campaign assumption, while 3 A/mm² is an
exploratory sensitivity rather than a replacement baseline.

When the required cooling cannot be matched below a selected limit, the study
now removes only the current-density cap and searches through the finite
constant-property cooling maximum. A target reached in that diagnostic is
classified as **current-density-cap limited**. A target still not reached is
classified as **physics limited**, meaning the modeled cooling curve peaks
below the requirement because of the thermoelectric and thermal-network
balance. This avoids merging an assumed engineering constraint with conductive
backflow and other modeled physical limits.

The attribution percentage depends on the scope being counted. Across the
complete declared tensor, 93,813 points miss the cooling target: 18,759
(20.0%) are current-density-cap limited and 75,054 (80.0%) are physics limited.
In the displayed Ag₂Se + p-9107, 25 K, 3 A/mm² status panel, the corresponding
split is 212 of 2,006 (10.6%) and 1,794 of 2,006 (89.4%). Stating both the
numerator and scope prevents a subset-dependent percentage from being mistaken
for a universal property of the model.

Ag₂Se must be paired with a p-type leg, so the p-type envelope is more honest
than presenting p 9107 as universal:

| Application | Maximum-feasible $\rho_c$ range across six p records (Ω·m²) |
| --- | ---: |
| 10 K efficiency-first | $9.1028\times10^{-9}$ to $1.6061\times10^{-8}$ |
| 25 K balanced | $3.8840\times10^{-9}$ to $9.1028\times10^{-9}$ |
| 10 K capacity-first | $6.8529\times10^{-9}$ to $1.3936\times10^{-8}$ |

The high-lift application is the most contact-sensitive of the three. For the
Ag₂Se + p-9107 scenario, the paper-derived $1.6650\times10^{-8}$ Ω·m² landmark
lies above the high-lift feasibility ceiling under the declared cooling-system
assumptions. This is a process-target result for the simulated system, not a
judgment of the paper's generator performance.

## How to read the six panels

1. **Published electrical translation:** contact share and normalized $ZT$
   retention versus $\rho_c$, with the analytical crossover and paper-derived
   landmark.
2. **Contact-share map:** the geometry/interface decision surface for Ag₂Se +
   p 9107.
3. **Device-$ZT$ map:** the mode-agnostic device metric at 300 K.
4. **Application boundary:** the largest feasible $\rho_c$ versus leg length,
   with the existing 1 A/mm² constraint and exploratory 3 A/mm² sensitivity
   shown side by side.
5. **Cooling COP:** delivered wall COP only where all high-lift requirements
   pass.
6. **Limiting status:** separates current-cap-limited cooling, physics-limited
   cooling, COP, power, voltage, and other constraints. The heatmap displays
   the explicitly labeled exploratory 3 A/mm² sensitivity.

The white vertical line marks 1.5 mm. The paper area is 2.25 mm², while the
system maps use 1.6 mm²; this difference is stated in the figure rather than
hidden.

## Reproduce the experiment

Install report dependencies and run:

```bash
python3 -m pip install -e '.[reports]'
thermotwin-contact-process-window
```

Equivalent module command:

```bash
python3 -m thermotwin.contact_process_window
```

The default figure is
`thermotwin/figures/ELECTRICAL_CONTACT_PROCESS_WINDOW/electrical_contact_process_window.png`,
with source data in `electrical_contact_process_window.json`. Generated artifacts
are ignored by Git.

Relevant tests:

```bash
python3 -m unittest \
  tests.test_literature_materials \
  tests.test_material_pair \
  tests.test_contact_process_window \
  tests.test_contact_process_window_report
```

## Code ownership

| File | Responsibility |
| --- | --- |
| `design/literature_materials.py` | opt-in same-sample Ag₂Se record and published unicouple electrical quantities |
| `design/material_pair.py` | explicit p/n material evaluation, current optimization, and first-crossing target match |
| `design/contact_process_window.py` | frozen grids, process-window sweep, selection, and text summary |
| `reports/contact_process_window.py` | six-panel figure and command-line report |
| `contact_process_window.py` | compatibility facade and module entry point |

## What this experiment does not establish

- It does not fit the paper's contact resistivity; $R_cA$ is an ideal full-area
  translation of a reported lumped contact resistance.
- It does not model the paper's p-type material as a new record. The system
  sweep pairs Ag₂Se with the six existing same-row p-type records.
- It does not compare generator output with cooler COP. The electrical panel is
  mode-agnostic; cooling is a separate simulated operating mode.
- It does not include temperature-dependent properties, Thomson heat,
  distributed legs, contact spreading resistance, solder aging, diffusion, or
  measured converter/exchanger behavior.
- It does not identify a manufacturing route that will achieve a chosen
  $\rho_c$. It states the device-level target that such a route would need to
  meet.

The next hardware step is to measure specific contact resistivity on the same
processed leg/contact stack used in a device, then place that measurement on
this map with uncertainty bars.
