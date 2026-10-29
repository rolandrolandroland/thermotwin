# Operating-decision final figures

Date: 2026-10-29. Status: complete; full report, presentation and deposit package
remain separate Phase H work.

The three figures present the accepted synthetic experiment, its limitations
and the development measurement map. Primary success was not established.
No selector, correction, scientific partition or accepted outcome was changed.

## Figures and captions

Exports are in `reports/operating-decision-final-figures-20261029/`, with PNG,
PDF and SVG versions of each figure. The generated directory remains
gitignored. These links work after rendering locally; a clean clone contains
the authenticated inputs and renderer rather than the generated exports.

| Figure | PNG preview | PDF | SVG |
| --- | --- | --- | --- |
| 1 — illustrative decision story | [PNG](../../reports/operating-decision-final-figures-20261029/figure_1_decision_story.png) | [PDF](../../reports/operating-decision-final-figures-20261029/figure_1_decision_story.pdf) | [SVG](../../reports/operating-decision-final-figures-20261029/figure_1_decision_story.svg) |
| 2 — reserved comparison | [PNG](../../reports/operating-decision-final-figures-20261029/figure_2_reserved_comparison.png) | [PDF](../../reports/operating-decision-final-figures-20261029/figure_2_reserved_comparison.pdf) | [SVG](../../reports/operating-decision-final-figures-20261029/figure_2_reserved_comparison.svg) |
| 3 — development measurement map | [PNG](../../reports/operating-decision-final-figures-20261029/figure_3_development_map.png) | [PDF](../../reports/operating-decision-final-figures-20261029/figure_3_development_map.pdf) | [SVG](../../reports/operating-decision-final-figures-20261029/figure_3_development_map.svg) |

### Figure 1. Uncertainty narrows without resolving the calibrated decision

An illustrative reserved case, block 1 with temperature-dependent contact,
shows two candidate models fitting the initial cold-exchanger observations,
their forecasts of the hidden cold face, the frozen action scores, the first
of three added thermal runs, verified target forecasts, and the resulting
operating-margin intervals. The selector chose thermal acquisition. Its raw
margin interval narrowed from 0.486 to 0.354 K and lay below zero, but the
final selector correction of 0.02544 K per endpoint returned it to an
abstention. The saved true margin was −0.277 K. The operating band was
285.0–301.7 K.

This is a post hoc illustration of one of the 27 calibration-induced
abstentions, not an estimate of overall efficacy or an explanation of all
137 abstentions. Selection used the first block, then frozen family order,
with two initially admissible candidates, an additional measurement selected,
a narrower development-padded interval and a final abstention. It did not
rank cases by true margin or improvement. Initial measured-channel fits
include the frozen Gaussian profiled run offsets; added-run forecasts use
zero-mean future offsets. Stop is outside the action value/cost ratio.

Eight forward integrations projected retained fitted candidates using the
original source and pinned runtime. The target reveal and all observations
were copied from the saved archive; no optimization, synthetic observation
generation or truth-model replay was performed. The initial and verified
forecast margins exactly matched their saved estimates.

### Figure 2. Lower simulated energy did not establish primary success

The five procedures are compared on the same 100 paired reserved blocks,
with three families and 300 cases per procedure. Panels show definitive
decisions with simultaneous lower bounds; false approvals conditional on
approval with simultaneous upper bounds; case and complete-block interval
coverage; mean simulated diagnostic energy; all 300 final interval widths per
procedure; and instrumentation, energized schedule time and software compute.
Boxplots show medians and interquartile ranges, with whiskers spanning the
complete minimum–maximum range.

The selector made 163/300 definitive decisions, with one false approval among
104 approvals, at 75.03 J mean simulated diagnostic energy. Its frozen
decision-coverage lower bound was 38.10%, below the required 70%; its
approval-risk upper bound was 31.85%, above the required 10%. The observed
false-approval fraction was 0.96%; the upper bound describes uncertainty,
not an observed error rate. No procedure qualified on both quality criteria,
so energy savings against the fixed measurement policies did not establish
the primary claim. Stop used the least energy. All procedures had zero
observed false rejections.

The 73-direction simultaneous allocation and block dependence are retained;
300 cases are not treated as independent replicates. Diagnostic joules
exclude computer electricity, resets and sensor electronics. Energized
schedule time is not full bench time. The displayed CPU time, execution wall
time and sampled memory peak describe the whole reserved benchmark,
including comparators and predictive draws; single-case deployed latency
was not measured. See the [accepted result](OPERATING_DECISION_PHASE_F_RESULT.md).

### Figure 3. Development choices respond to sensor quality and assumed costs

The complete predeclared 4 × 12 development map retains all 60 cases from
20 paired blocks in every cell. Colored segments show stop, thermal,
voltage and face-temperature selections; hatching shows abstentions within
each action. Counts retain ineligible actions and whole-case failures in the
denominator. Selection, verification and pipeline failures were zero in every
cell. Face acquisition was ineligible in one of 60 cases throughout the
higher-probe-loading row.

The physical rows use nominal noise and loading, doubled voltage noise
(0.002 → 0.004 V), doubled face noise (0.02 → 0.04 K), or doubled probe
capacity (5 → 10 J/K). Probe response remains 2.5 s and the run-bias/noise
ratio is 2.5. Cost columns cross energy/time/instrumentation weights of
(⅓, ⅓, ⅓), (0.70, 0.20, 0.10), (0.20, 0.70, 0.10) and
(0.15, 0.15, 0.70) with face-instrument prices of ¼, 1 and 4 times the
voltage-instrument price; reset cost assumes 80 s per added run.

The black outline identifies nominal sensors, balanced weights and equal
face/voltage instrument prices: 34 stop, 16 thermal, five voltage and five
face selections, with 32/60 definitive decisions. Nominal cells reuse D1
observations; stressed physical rows were actually recomputed in D5.
These correlated development cells precede final calibration q and provide
no reserved risk guarantee. Reserved confirmation covers only the primary
nominal/balanced/equal-price setting in Figure 2; the other 47 cells are
sensitivity evidence. See the [D5 record](OPERATING_DECISION_PHASE_D5_RESULT.md).

## Reproduction and provenance

The tracked [plot inputs](../../thermotwin/OPERATING_DECISION_PROSPECTIVE_FINAL_FIGURE_INPUTS.json)
contain all plotted curves, observations, intervals, comparison metrics and
48 cell summaries. The tracked [export manifest](../../thermotwin/OPERATING_DECISION_PROSPECTIVE_FINAL_FIGURES.json)
binds those inputs, the renderer and every exported file by SHA-256.

From the repository root, use the recorded Matplotlib 3.10.8 runtime and a
new output directory:

```sh
.artifacts/operating-decision/runtime-cpython31012-20261007/bin/python \
  -m tools.render_operating_decision_final_figures \
  --inputs thermotwin/OPERATING_DECISION_PROSPECTIVE_FINAL_FIGURE_INPUTS.json \
  --output-directory reports/operating-decision-final-figures-20261029
```

The local pinned executable is not distributed in Git. Another environment
with Matplotlib 3.10.8 can run the same command using its Python executable;
byte identity was checked in the recorded runtime. Rendering requires no
large scientific archive or numerical ThermoTwin imports. It performs no
simulation or fitting and refuses to overwrite an existing output directory.
The [input builder](../../tools/operating_decision_final_figure_data.py)
documents the additional retained archives and original source/runtime
needed to reconstruct the compact inputs. Routine rendering uses the
tracked inputs directly.

| Identity | Value |
| --- | --- |
| Reserved numerical source | `7e3b9a3be22d991ec2e588cbd4a53a9d4d2d27d7` |
| Reserved scientific result | `d565c0c6368b602eeb3d2721c128aab88221bce9a6b5abe68fcda4c727d4edf0` |
| D5 scientific result | `b26352040b44c595f86d76003dd55bd7a2d8b43c0ccf9b825dda53c0a6edf4f3` |
| Figure inputs | `84c6dd44c75a678e528f791b2bd8b0af4f89087e3fae1a6e9b4235add7bd47a0` |
| Export manifest | `121f2f0415e22b1c0ebdf7bbf2a98ae641afbcabfd99776e99620c03a3899381` |

All three PNGs were visually inspected for text, labels, legends and layout.
A fresh render reproduced all nine PNG/PDF/SVG exports and the manifest byte
for byte. All 48 focused tests passed, including six new tests covering
denominator integrity, ineligibility, failures, illustrative-case selection
and offset profiling. All 117 frozen numerical source files remained exact
in both the working repository and retained source clone. The accepted
Phase F, abstention diagnosis and numerical replay artifacts were unchanged.
The local verification record is retained at
`.artifacts/operating-decision/final-figure-verification-20261029.json`.

This completes the figure portion of Phase H. The full report, presentation
and reproducibility/deposit package remain. The study is software-only
evidence; a future decision-aware redesign requires a separate protocol and
fresh development, calibration and reserved partitions.
