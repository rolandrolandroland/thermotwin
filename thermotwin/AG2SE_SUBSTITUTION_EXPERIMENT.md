# Matched Ag₂Se material-substitution experiment

## Decision this experiment supports

This experiment asks:

> If only the n-type material is replaced by a published optimized Ag₂Se
> sample, how often does a fixed virtual module improve—and does it create a
> new best feasible design?

The matched design is the important part. A new global search could attribute
an improvement to a different geometry, p material, interface, exchanger, or
sampled candidate. This study changes one material record and re-optimizes only
the operating current.

## Headline result

At the good-interface baseline, optimized Ag₂Se improves the scalar application
utility of **69.6–76.5%** of the 204 matched designs, depending on application.
It improves cooling COP in **76.0–88.2%** of comparisons for which both COPs
exist.

Despite those broad design-level gains, Ag₂Se creates **no new best feasible
design** in any of the three applications. The strongest legacy combination in
the frozen pool remains better after the same current search.

At the paper-derived electrical contact level of
$1.6650\times10^{-8}$ Ω·m², the utility-improved fraction falls to
**50.0–54.9%**, the median changes approach zero, and Ag₂Se still creates no
new best. This is the central co-design lesson: a material advantage is
conditional on its mate, geometry, interfaces, and application.

## Material provenance

The replacement record is the optimized room-temperature sample from
[Bappy et al., *Materials Horizons* (2026), DOI 10.1039/D6MH00220J](https://doi.org/10.1039/D6MH00220J):

| Property | Value |
| --- | ---: |
| Seebeck coefficient | −153.3 µV/K |
| Electrical conductivity | 117,400 S/m |
| Thermal conductivity | 0.85 W/(m·K) |
| Calculated same-triplet $zT$ at 300 K | about 1.00 |
| Process provenance | 9% excess Se; 350 °C synthesis for 90 min; 375 °C sintering for 60 min |

The record is an opt-in literature extension with a ThermoTwin-local identifier,
not a StarryData row. It does not alter the six-record n-type baseline catalog.
That separation protects the original design indices and the frozen campaign.

## Matched comparison protocol

The original co-design pool contains 204 virtual prototypes:

- 24 seeded space-filling initial designs;
- 180 seeded candidate designs.

For every design, application, and contact-resistivity case, the study performs
two evaluations:

1. optimize current for the original p/n material pair;
2. replace only the n-type record with optimized Ag₂Se and repeat the same
   current-grid optimization.

The following stay exactly matched:

- p-type material;
- couple count;
- leg length and area;
- packing fraction;
- thermal contact resistance;
- cold and hot exchanger conductance;
- converter efficiency and fixed loss;
- PWM-ripple assumption;
- voltage and current-density limits;
- application definition;
- 28-point current grid.

The original 1 A/mm² current-density limit is used because this is a controlled
extension of the frozen co-design campaign, not the broader process-window
sensitivity.

Two electrical-interface cases are evaluated:

| Case | Specific contact resistivity | Interpretation |
| --- | ---: | --- |
| Good-interface baseline | $2.0\times10^{-10}$ Ω·m² | original literature-anchored assembly assumption |
| Paper-derived landmark | $1.6650\times10^{-8}$ Ω·m² | ideal $R_cA$ translation from about 7.4 mΩ and 2.25 mm² |

The second case is a sensitivity, not a claim that every printed interface has
that resistivity.

## Quantities compared

For each matched pair the study records:

- selected mean current;
- delivered cooling and heating;
- module and wall electrical power;
- cooling and heating COP;
- device $ZT$;
- current-density and voltage constraints;
- application feasibility;
- the original scalar utility function.

The reported fractions answer different questions. “COP improved” does not mean
“cooling improved,” and neither guarantees that the complete application
requirements pass.

## Full results

### Good-interface baseline: $\rho_c=2.0\times10^{-10}$ Ω·m²

| Application | Utility improved | Cooling improved | COP improved | Feasibility gained / lost | Median Δcooling | Median ΔCOP | Best original / Ag₂Se utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 K efficiency-first | 76.5% | 48.5% | 76.0% | 18 / 4 | −0.0029 W | +0.2586 | 3.4411 / 3.3332 |
| 25 K balanced | 69.6% | 63.2% | 77.6% | 26 / 0 | +0.3401 W | +0.1075 | 6.4268 / 5.9425 |
| 10 K capacity-first | 72.1% | 61.8% | 88.2% | 25 / 6 | +0.2815 W | +0.1684 | 3.8654 / 3.0234 |

At low lift, Ag₂Se often trades a small amount of cooling for a meaningful COP
gain. That is why the efficiency-first median cooling change is nearly zero and
slightly negative while utility and COP improve for roughly three quarters of
the pool.

For the 25 K balanced case, the median cooling and COP changes are both
positive, and 26 designs become feasible without any losing feasibility. Yet
the best Ag₂Se utility remains below the best legacy design. Broad improvement
and a new global winner are different claims.

### Paper-derived contact case: $\rho_c=1.6650\times10^{-8}$ Ω·m²

| Application | Utility improved | Cooling improved | COP improved | Feasibility gained / lost | Median Δcooling | Median ΔCOP | Best original / Ag₂Se utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 K efficiency-first | 50.0% | 44.6% | 50.5% | 18 / 11 | −0.0225 W | +0.0002 | 1.1189 / 0.9211 |
| 25 K balanced | 54.9% | 53.9% | 50.0% | 13 / 5 | +0.0654 W | +0.0018 | 0.6963 / 0.3386 |
| 10 K capacity-first | 51.0% | 48.0% | 48.0% | 9 / 8 | −0.0239 W | −0.0024 | 0.7314 / 0.5691 |

Once fixed areal contact resistance dominates a larger share of total
resistance, changing only the bulk n material has much less leverage. The
fractions cluster near 50%, median COP changes approach zero, and the best
feasible utilities fall for both original and replacement material sets.

## How to interpret the feasibility counts

A material substitution can gain and lose feasibility in the same application
because $S$, $\sigma$, and $\kappa$ move together:

- stronger Seebeck response can increase useful Peltier transport;
- higher conductivity can reduce bulk Joule loss;
- thermal conductivity changes hot-to-cold back-conduction;
- the re-optimized current can move voltage, power, COP, and cooling constraints
  in different directions.

The matched design therefore has to be re-evaluated as a system. Ranking the
n-type records by material $zT$ alone cannot predict every feasibility change.

## Relation to process optimization

The cited study uses Gaussian-process regression and Bayesian optimization to
map processing choices—composition, synthesis, and sintering—to material power
factor. ThermoTwin starts at the next layer: it maps measured material
properties to module and application performance.

Those workflows are complementary:

```text
process variables -> measured material properties -> device design -> application decision
```

A future closed loop could pass ThermoTwin's device-level value or feasibility
back to a process optimizer. This repository does not yet contain the paired
process/property/cost dataset needed to train that first arrow.

## Why this is not an index-remapping artifact

The six-entry baseline n-type catalog is unchanged. Adding Ag₂Se to that tuple
would change categorical indices and could silently alter every seeded design.
Instead, `MaterialPairDesign` carries explicit material objects. Each original
prototype is converted to an explicit pair only for this study, after its
geometry and p material have already been frozen.

Tests verify both catalog isolation and design matching.

## Reproduce the experiment

Install report dependencies and run:

```bash
python3 -m pip install -e '.[reports]'
thermotwin-ag2se-substitution
```

Equivalent module command:

```bash
python3 -m thermotwin.ag2se_substitution
```

The default figure is
`thermotwin/figures/ag2se_matched_substitution.png`. Generated figures are
ignored by Git.

Relevant tests:

```bash
python3 -m unittest \
  tests.test_literature_materials \
  tests.test_material_pair \
  tests.test_ag2se_substitution \
  tests.test_ag2se_substitution_report
```

## Code ownership

| File | Responsibility |
| --- | --- |
| `design/literature_materials.py` | opt-in Ag₂Se property triplet and provenance |
| `design/material_pair.py` | evaluation for explicit material pairs with parity to the frozen campaign |
| `design/ag2se_substitution.py` | matched pool, replacement, comparisons, and summaries |
| `reports/ag2se_substitution.py` | four-panel audit figure and command-line report |
| `ag2se_substitution.py` | compatibility facade and module entry point |

## What this experiment does not establish

- It does not validate Ag₂Se in a ThermoTwin cooling module.
- It does not claim the cited generator and the simulated cooler are directly
  comparable operating modes.
- It pairs Ag₂Se with public p-type records, not the paper's exact p leg.
- It uses constant 300 K properties rather than temperature-dependent curves.
- It does not include chemical compatibility, diffusion, solder selection,
  thermal cycling, aging, yield, or material price.
- It does not predict which process settings produce a target property triplet.
- It searches current on the frozen campaign grid; it is a controlled
  comparison, not a new globally exhaustive co-design.

The null result is still useful. It says the published n-type material is a
promising substitution across much of the virtual pool, but the most valuable
next experiment is interface and paired-leg validation—not a claim that a
single material record has already won at product level.
