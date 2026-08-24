# Material, geometry, and Bayesian co-design campaign

This guide walks through ThermoTwin's first application-specific virtual
product-design campaign. It connects public room-temperature thermoelectric
material records to module geometry, thermal contacts, exchanger sizing,
smoothed PWM operation, relative prototype cost, Bayesian optimization, and
as-built robustness.

The most important qualification is this:

> This is a reproducible method demonstration using real literature-derived
> material properties and explicit synthetic product assumptions. It is not a
> validated commercial design, a manufacturing-process model, a dollar cost model,
> or evidence that the selected virtual module will match hardware.

## 1. Questions asked

The campaign contains three linked experiments.

1. **Initial design screen:** What can be learned from 24 space-filling
   material/geometry/interface prototypes?
2. **Cost-aware Bayesian optimization:** Under the same 12-prototype follow-up
   budget, does a Gaussian-process selector improve faster than random search?
3. **As-built robustness:** If the selected current is held fixed, how often
   does each nominal winner still meet its requirement when material
   properties, contacts, exchangers, and converter efficiency vary?

The three application specifications intentionally reward different behavior:

| Application | External lift | Minimum cooling | Minimum wall COP | Maximum supply power | Objective |
| --- | ---: | ---: | ---: | ---: | --- |
| 10 K efficiency-first | 10 K | 2.5 W | 0.75 | 25 W | wall COP divided by square root of cost index |
| 25 K balanced | 25 K | 0.75 W | 0.15 | 30 W | cooling times wall COP divided by cost index |
| 10 K capacity-first | 10 K | 4.0 W | 0.60 | 30 W | heat flux times square root of wall COP divided by square root of cost index |

The objective changes because a single universally optimal thermoelectric
module does not exist. A low-lift efficiency product, a large-lift product,
and a thin capacity-dense product place different values on COP, cooling rate,
area, cost, and heat rejection.

## 2. Public material data

### 2.1 Fixed source

The compact catalog in `material_catalog.py` comes from the
[official StarryData service](https://starrydata.nims.go.jp/starrydata2/) and
its fixed
[Figshare snapshot](https://figshare.com/articles/dataset/Starrydata_thermoelectric_data_snapshot_interpolated_data_/11340935):

- StarryData thermoelectric interpolated snapshot;
- Figshare DOI `10.6084/m9.figshare.11340935.v1`;
- file `Starrydata_interpolated_20190816.csv`;
- MD5 `5ae1d38f76fd872d40bff37c2bec29f6`;
- CC BY 4.0 license;
- fixed publication date in 2019.

The implementation does not silently download data when imported. Twelve
auditable rows are retained in source code so the experiment is reproducible
offline. The full snapshot is not committed.

StarryData digitized published experimental curves and interpolated this
snapshot at 100 K intervals. A record is therefore literature-derived data,
not a fresh ThermoTwin measurement. Original articles should be consulted
before treating any property as a design specification.

### 2.2 Curation rule

The selection required:

- exactly 300 K;
- canonical Seebeck coefficient, electrical conductivity, and thermal
  conductivity all present on the same sample row;
- a Bi/Te-family composition;
- $50\leq |S|\leq400$ microvolt per kelvin;
- $10\leq\sigma\leq500$ kilosiemens per metre;
- $0.3\leq k\leq4$ watt per metre-kelvin;
- six p-type and six n-type records spanning property trade-offs.

The same-row rule is essential. ThermoTwin does not take the largest Seebeck
coefficient from one sample, the largest electrical conductivity from a second,
and the smallest thermal conductivity from a third. Such a combination would
describe a material that was never measured.

### 2.3 Curated records

| Type | Sample ID | Short sample label | $S$ (microV/K) | $\sigma$ (kS/m) | $k$ (W/m K) | calculated $ZT$ |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| p | 9107 | nano-crystalline BixSb2-xTe3 | 198.59 | 117.42 | 1.003 | 1.385 |
| p | 10561 | p-type (Bi,Sb)2Te3 | 224.88 | 86.33 | 1.221 | 1.072 |
| p | 10879 | Bi0.5 Sb1.5 Te3 BST | 235.15 | 47.91 | 1.039 | 0.765 |
| p | 7986 | Sb1.52Bi0.48Te3, MS aligned | 180.40 | 80.83 | 1.164 | 0.678 |
| p | 5550 | Bi0.5Sb1.5Te3 NBH1 | 158.50 | 69.51 | 1.026 | 0.511 |
| p | 5553 | Bi0.5Sb1.5Te3 NBS2 | 173.99 | 15.89 | 0.726 | 0.199 |
| n | 10562 | Bi2(Te,Se)3 | -201.39 | 74.63 | 0.807 | 1.125 |
| n | 5606 | Bi2Te2.7Se0.3, parallel | -194.10 | 63.46 | 0.835 | 0.859 |
| n | 14771 | Bi2Te3, press-perpendicular | -168.85 | 71.66 | 0.922 | 0.665 |
| n | 5792 | Bi2Te2.25Se0.75 | -144.12 | 88.64 | 0.903 | 0.612 |
| n | 16848 | Bi2Te2.7Se0.3, SPS-250C | -262.99 | 16.70 | 0.602 | 0.576 |
| n | 16850 | Bi2Te2.7Se0.3, SPS-350C | -134.20 | 30.78 | 1.101 | 0.151 |

The calculated value in the last column is

$$
ZT=\frac{S^2\sigma T}{k}
$$

using the same-row property triplet. The optimizer does not optimize $ZT$
directly. A leg pair with good material $ZT$ can still produce a poor product
if geometry, contacts, heat rejection, electrical limits, or the application
objective are unfavorable.

## 3. From material records to a module

One couple contains a p leg and an n leg. The two legs conduct heat in parallel
and current in series. The module repeats $N$ couples electrically in series
and thermally in parallel.

For equal p- and n-leg length $L$ and area $A$, ThermoTwin calculates

$$
\alpha=N(S_p-S_n),
$$

$$
R_{\mathrm{legs}}
=N\frac{L}{A}\left(\frac{1}{\sigma_p}+\frac{1}{\sigma_n}\right),
$$

and

$$
K_{\mathrm{legs}}
=N\frac{A}{L}(k_p+k_n).
$$

The virtual assembly then applies two stated non-material assumptions. Bulk
leg resistance and electrical-interface resistance are separated:

$$
R_{\mathrm{contact}}
=4N\frac{\rho_c}{A},
$$

$$
R=R_{\mathrm{legs}}+R_{\mathrm{contact}},
\qquad
K=K_{\mathrm{legs}}+0.04\ \mathrm{W/K}.
$$

Each p and n leg has two metal/thermoelectric interfaces, giving four
interfaces per series p/n couple. The baseline uses one symmetric per-interface
specific contact resistivity
$\rho_c=2.0\times10^{-10}\ \mathrm{ohm\,m^2}$. That order of magnitude is
anchored to a 298 K Ti/Bi2Te3 measurement of
$1.94\times10^{-10}\ \mathrm{ohm\,m^2}$ in a transfer-length study
([AIP Advances 15, 035351](https://doi.org/10.1063/5.0253218)). ThermoTwin's
exact value is still a synthetic baseline, not a fitted property of the
curated StarryData samples or a manufactured module.

The contact term scales with $N/A$ and is independent of leg length. It
therefore penalizes short legs more strongly as a fraction of total electrical
resistance instead of hiding contact loss inside a constant multiplier. The
fixed 0.04 W/K term represents package parasitic conduction. Metal trace and
solder-bulk resistance are not separately modeled.

The equations make the geometry trade-off explicit:

- increasing $L$ increases only the bulk part of $R$ and decreases $K$;
- increasing $A$ decreases $R$ and increases $K$;
- increasing $N$ increases $\alpha$, $R$, and $K$ together;
- active volume is $2NAL$;
- estimated footprint is $2NA/f$, with packing fraction $f=0.65$.

A thin leg can move more heat by lowering electrical resistance, but it also
increases passive heat conduction. A larger leg area has the same opposing
effects. Geometry therefore cannot be optimized by looking at electrical
resistance alone.

## 4. Product-level thermal and electrical model

Every candidate uses the existing four-node contact topology:

1. cold thermoelectric face;
2. hot thermoelectric face;
3. cold exchanger;
4. hot exchanger.

The design variables include symmetric face-to-exchanger resistance and both
reservoir conductances. At steady state, storage terms are zero and the four
linear energy balances are solved algebraically.

The electrical layer assumes a smoothed PWM-derived current with 10% triangular
peak-to-peak ripple. The heat equations retain the current moments they need:

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
-K(T_h-T_c).
$$

For ripple fraction $r=0.10$,

$$
\overline{I^2}=\overline I^2\left(1+\frac{r^2}{12}\right).
$$

The converter assumptions are efficiency $\eta=0.95$ and fixed loss
$P_{0}=0.05$ W:

$$
P_{\mathrm{supply}}=\frac{P_{\mathrm{module}}}{\eta}+P_0,
\qquad
\mathrm{COP}_{\mathrm{wall}}=\frac{Q_{c,\mathrm{del}}}{P_{\mathrm{supply}}}.
$$

The current scan obeys a 1.0 A/mm2 peak current-density bound and a 12 V peak
module-voltage bound. Each application also has its own supply-power limit.
The report records current-density utilization explicitly and marks a
constraint as binding at 99.5% utilization or above.

## 5. Design variables and synthetic cost

Each Latin-hypercube row contains eight independent coordinates:

| Variable | Range or choices |
| --- | --- |
| p material | six curated records |
| n material | six curated records |
| couple count $N$ | 80--160 |
| leg length $L$ | 0.8--2.4 mm |
| leg area $A$ | 0.8--2.4 mm2 |
| symmetric contact resistance | 0.10--0.50 K/W |
| cold exchanger conductance | 1.5--5.0 W/K |
| hot exchanger conductance | 3.0--8.0 W/K |

There are no dollar cost data in the public material snapshot. ThermoTwin
therefore reports a relative prototype build-burden index:

$$
C_{\mathrm{index}}
=0.40\frac{V}{V_0}
+0.20\frac{N}{120}
+0.20\frac{G_c}{2.5}
+0.20\frac{G_h}{5.0},
$$

where $V_0=2(120)(1.6\ \mathrm{mm^2})(1.5\ \mathrm{mm})$.

This index says only that more active volume, more couples, and larger
exchangers are assumed to create more prototype burden. It does not include
material price, yield, tooling, labor, supply chain, lifetime, or economies of
scale. It must not be presented as dollars or levelized HVAC cost.

## 6. Experiment 1: 24-design space-filling screen

### Procedure

1. Generate a deterministic eight-dimensional Latin hypercube with seed
   `20260821`.
2. Map the unit coordinates to the bounds above.
3. For every design and application, scan 28 allowable mean currents.
4. Select the feasible current with the largest application utility. If no
   current is feasible, retain the least-violating point with negative utility.
5. Record material parameters, temperatures, delivered heat, terminal power,
   supply power, COP, heat flux, cost index, voltage, feasibility, and utility.

### Result

| Application | Feasible initial designs | Best initial utility |
| --- | ---: | ---: |
| 10 K efficiency-first | 17/24 | 3.4411 |
| 25 K balanced | 16/24 | 3.9015 |
| 10 K capacity-first | 16/24 | 3.8654 |

The screen is already strong enough to contain the retrospective pool winner
for both 10 K objectives. That is not a failed optimizer. It means the initial
space-filling budget covered those two tested objective landscapes unusually
well.

## 7. Experiment 2: cost-aware Bayesian optimization

### Surrogate

The optimizer encodes the six continuous coordinates as values between zero
and one and represents the p and n material choices with twelve one-hot
features. It fits a Gaussian process with radial-basis covariance

$$
k(\mathbf{x},\mathbf{x}')
=\exp\left(-\frac{\lVert\mathbf{x}-\mathbf{x}'\rVert^2}{2\ell^2}\right),
\qquad \ell=1.4.
$$

A $10^{-5}$ diagonal nugget stabilizes the covariance inverse. At each of 12
iterations, the selector maximizes expected improvement divided by the square
root of the prototype cost index. The candidate set contains 180 additional
space-filling virtual designs.

The comparison uses 25 random candidate orders. Every random run starts from
the same 24 observations and receives the same 12-prototype follow-up budget.
The full candidate-pool optimum is calculated afterward only as an audit; it
is not shown to the optimizer.

### Result

| Application | Initial best | BO final | Random median final | Pool optimum | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| 10 K efficiency-first | 3.4411 | 3.4411 | 3.4411 | 3.4411 | initial screen already contained pool winner |
| 25 K balanced | 3.9015 | 6.4268 | 3.9015 | 6.4268 | BO found pool winner after five additions |
| 10 K capacity-first | 3.8654 | 3.8654 | 3.8654 | 3.8654 | initial screen already contained pool winner |

For the high-lift application, BO improves utility by about 64.7% over the
initial best and reaches the pool optimum within the budget. The random median
does not improve on the initial best. The two flat 10 K curves are retained
because reporting only the successful high-lift case would overstate the value
of optimization.

## 8. Selected nominal designs

| Application | Design | p/n IDs | $N$ | $L$ (mm) | $A$ (mm2) | Mean current (A) | Cooling (W) | Wall COP | Cost index |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 K efficiency | initial-024 | 9107/10562 | 83 | 1.079 | 0.845 | 0.494 | 2.524 | 2.856 | 0.689 |
| 25 K balanced | candidate-116 | 10561/10562 | 98 | 1.179 | 2.216 | 2.111 | 8.317 | 0.882 | 1.142 |
| 10 K capacity | initial-024 | 9107/10562 | 83 | 1.079 | 0.845 | 0.805 | 4.662 | 2.207 | 0.689 |

The two 10 K objectives select the same hardware but different current. This is
an important control/design interaction: hardware selection does not by itself
define operation. The capacity objective drives the module harder, increasing
cooling from 2.524 W to 4.662 W while reducing wall COP from 2.856 to 2.207.

| Application | Bulk leg $R$ | Electrical-contact $R$ | Contact share | Peak current density | Binding? |
| --- | ---: | ---: | ---: | ---: | --- |
| 10 K efficiency | 2.3218 ohm | 0.0785 ohm | 3.3% | 0.6130 A/mm2 | no |
| 25 K balanced | 1.3030 ohm | 0.0354 ohm | 2.6% | 1.0000 A/mm2 | **yes** |
| 10 K capacity | 2.3218 ohm | 0.0785 ohm | 3.3% | 1.0000 A/mm2 | **yes** |

The selected mean-current caps computed from the unrounded areas are 2.11059 A
for the 25 K design and 0.80520 A for the capacity-first design. Their printed
currents are rounded to 2.111 A and 0.805 A. These points sit exactly at the
peak current-density constraint; they are not unconstrained interior optima.

The 25 K winner uses a much larger leg area and higher current. Its module
parameters are approximately $\alpha=0.0418$ V/K, $R=1.338$ ohm, and
$K=0.414$ W/K. That combination supports much higher cooling at the price of
greater electrical power and active/exchanger burden.

## 9. Experiment 3: fixed-current as-built robustness

### Perturbations

Each nominal winner is tested in 300 deterministic Monte Carlo trials. The
selected mean current is not re-optimized. That answers a commissioning
question: if the controller uses the nominal setpoint, does a built unit still
meet the product requirement?

| Quantity | Virtual distribution |
| --- | --- |
| p and n Seebeck coefficients | independent normal, 3% standard deviation |
| p and n electrical conductivities | independent unit-mean lognormal, log standard deviation 0.08 |
| p and n thermal conductivities | independent unit-mean lognormal, log standard deviation 0.08 |
| specific electrical contact resistivity | unit-mean lognormal, log standard deviation 0.20 |
| symmetric contact resistance | unit-mean lognormal, log standard deviation 0.15 |
| cold and hot exchanger conductances | independent unit-mean lognormal, log standard deviation 0.10 |
| converter efficiency | normal around 0.95 with 0.01 standard deviation, clipped to 0.85--0.99 |

These distributions are engineering stress assumptions. They are not fitted
process capability distributions.

### Result

| Application | Requirement pass rate | Cooling 5/50/95% (W) | Wall COP 5/50/95% |
| --- | ---: | --- | --- |
| 10 K efficiency-first | 55.3% | 2.326 / 2.515 / 2.699 | 2.507 / 2.849 / 3.132 |
| 25 K balanced | 100.0% | 7.056 / 8.347 / 9.536 | 0.738 / 0.878 / 1.019 |
| 10 K capacity-first | 100.0% | 4.388 / 4.643 / 4.945 | 1.985 / 2.189 / 2.472 |

The efficiency-first nominal design is fragile. Its nominal 2.524 W cooling is
only 0.024 W above the 2.5 W requirement, so ordinary property/interface spread
pushes many trials below the threshold even though COP remains high. A nominal
optimizer can therefore select a design that looks efficient but is difficult
to commercialize reliably.

This result motivates a future robust or chance-constrained acquisition rule,
for example requiring at least 95% predicted feasibility. ThermoTwin does not
retroactively change the objective in this report; doing so would obscure the
lesson learned from the predeclared nominal campaign.

## 10. How to run the campaign

From the repository root:

```bash
python3 -m thermotwin.material_geometry_codesign_report
```

The command prints the frozen result summary and writes
`thermotwin/figures/material_geometry_bayesian_codesign.png`.

The standard run performs:

- 24 initial designs;
- 180 hidden candidate designs;
- 12 BO additions for each of three applications;
- 25 equal-budget random baselines per application;
- 300 robustness trials per selected design;
- 28 current points per design/application evaluation.

It is CPU-first and uses dependency-free small-matrix algebra for the Gaussian
process. Matplotlib is needed only for the report figure.

## 11. Code map

| File | Responsibility |
| --- | --- |
| `design/materials.py` | public-data provenance, curated same-row records, material derived properties |
| `design/codesign/models.py` | immutable design, application, campaign, and result records |
| `design/codesign/evaluation.py` | module scaling, steady evaluation, constraints, and current selection |
| `design/codesign/sampling.py` | Latin-hypercube generation and feature encoding |
| `design/codesign/optimization.py` | Gaussian process, expected improvement, BO, and random baseline |
| `design/codesign/robustness.py` | fixed-current property and interface perturbations |
| `design/codesign/campaign.py` | experiment orchestration and text summary |
| `reports/material_codesign.py` | command-line runner and nine-panel evidence figure |
| `material_geometry_codesign.py` and `material_geometry_codesign_report.py` | historical compatibility facades |
| `test_material_catalog.py` | signs, provenance, $ZT$, and geometry limiting cases |
| `test_material_geometry_codesign.py` | space filling, energy-consistent evaluation, current scan, GP, acquisition, budgets, reproducibility, robustness |
| `test_material_geometry_codesign_report.py` | default figure location and PNG generation |

## 12. What the campaign establishes

Within its synthetic scope, the campaign establishes that ThermoTwin can:

- preserve same-sample material property relationships;
- translate material and geometry choices into module $\alpha$, $R$, and $K$;
- evaluate product-level contacts, exchangers, converter loss, COP, capacity,
  current density, voltage, and relative burden;
- change the preferred operating point with the application objective;
- use a reproducible Bayesian selector and compare it fairly with random
  search;
- expose cases where the initial design already wins;
- expose nominal designs that fail under as-built uncertainty.

## 13. What it does not establish

The current campaign does not establish:

- a causal relationship from powder processing, sintering, lattice geometry,
  or microstructure to $S$, $\sigma$, and $k$;
- temperature-dependent material curves away from the 300 K source row;
- Thomson heat, radiation, convection detail, fluid flow, spatial spreading,
  fatigue, aging, or condensation;
- calibrated converter losses or switching dynamics;
- material price, module manufacturing cost, yield, lifetime cost, or
  vapor-compression parity;
- model agreement with a physical prototype.

The most valuable next data are paired process/property/cost records and
hardware measurements of the same module across current and temperature lift.
Those data would turn the current virtual assumptions into inferable quantities
and allow process-aware, cost-calibrated, robust Bayesian optimization.
