# Engineering decision showcase

## Purpose

The engineering showcase combines four established ThermoTwin workflows into
one summary figure. It is a navigation and presentation report, not a fifth
independent experiment.

The panels answer four different questions:

1. Can sparse temperature sensors infer hidden contact resistance?
2. How do continuous and pulsed controls compare at matched delivered cooling?
3. Which feasible current pulse is locally most informative?
4. Can a standardized pulse rank virtual assemblies by hidden interface quality?

The underlying methods and limitations are documented separately in:

- [`SPARSE_SENSOR_EXPERIMENT.md`](SPARSE_SENSOR_EXPERIMENT.md);
- [`CONTROL_COMPARISON_EXPERIMENT.md`](CONTROL_COMPARISON_EXPERIMENT.md);
- [`NEXT_EXPERIMENT_WALKTHROUGH.md`](NEXT_EXPERIMENT_WALKTHROUGH.md); and
- [`ASSEMBLY_FINGERPRINT_EXPERIMENT.md`](ASSEMBLY_FINGERPRINT_EXPERIMENT.md).

## Generated artifacts

Run:

```bash
thermotwin-engineering-showcase
```

or:

```bash
python3 -m thermotwin.engineering_showcase
```

The default artifact folder is:

```text
thermotwin/figures/ENGINEERING_SHOWCASE/
```

It contains:

```text
engineering_decision_showcase.png
engineering_decision_showcase.json
engineering_decision_showcase.txt
```

The JSON sidecar contains the complete report result used by all four panels,
including the selected configurations, fitted values, uncertainties, control
metrics, and assembly classifications. The text sidecar explains how to read
the figure and states its interpretation boundary. All generated files are
ignored by Git and can be recreated from the committed code.

## Interpretation boundary

All four panels use synthetic models and declared uncertainty assumptions.
Combining them in one figure does not turn them into hardware validation. Read
the corresponding walkthrough before quoting a number from any panel.
