# Synthetic assembly thermal-fingerprint experiment

## Question

Can one standardized pulse turn accessible exchanger temperatures into a
repeatable assembly-level estimate of interface quality?

The experiment represents a possible quality-control use of ThermoTwin. It is
synthetic and does not define acceptance limits for real manufacturing.

## Standardized test

Every assembly receives the 0.8 A, 20 s pulse selected by the experiment
planner. Only cold- and hot-exchanger temperatures are recorded at 1 s
intervals. Both sensors use the known 1.5 s lag model, and independent 0.02 K
Gaussian noise is generated from an assembly-specific deterministic seed.

The estimator changes only cold contact resistance. Constant sensor offsets
are profiled out, so an absolute temperature calibration shift does not by
itself become an interface-resistance change. Local sensitivity gives an
approximate 95% interval.

## Synthetic batch and result

| Assembly | Hidden resistance | Inferred resistance | Local 95% interval | Classification |
| --- | ---: | ---: | ---: | --- |
| low_loss | 0.1500 K/W | 0.1532 K/W | 0.1480--0.1583 K/W | low interface loss |
| reference_a | 0.2400 K/W | 0.2407 K/W | 0.2342--0.2472 K/W | reference band |
| reference_b | 0.2600 K/W | 0.2571 K/W | 0.2504--0.2639 K/W | reference band |
| elevated_loss | 0.3500 K/W | 0.3489 K/W | 0.3404--0.3574 K/W | elevated interface loss |
| high_loss | 0.5000 K/W | 0.4994 K/W | 0.4874--0.5115 K/W | elevated interface loss |

The current illustrative bands are:

- below 0.20 K/W: low interface loss;
- 0.20 through 0.30 K/W: reference band;
- above 0.30 K/W: elevated interface loss.

These are software-demonstration thresholds, not engineering specifications.
All five hidden resistances fall within their local intervals, and the two
reference assemblies remain separated from both elevated-loss assemblies.

## What this could become

With real measurements and calibrated limits, a standardized fingerprint
could support:

- comparison of nominally identical assemblies;
- detection of interface-process drift;
- selection of units for destructive inspection;
- tracking inferred resistance against manufacturing metadata; and
- distinguishing a thermal-interface change from a sensor offset.

## Reproduce

```bash
python3 -m thermotwin.assembly_fingerprint
```

Generate the dedicated result figure, its JSON data, and its plain-text
explanation with:

```bash
python3 -m thermotwin.assembly_fingerprint_report
```

The artifacts are written to
`figures/ASSEMBLY_FINGERPRINT_EXPERIMENT/`.

The implementation is in
[`assembly_fingerprint.py`](assembly_fingerprint.py), and regression tests are
in
[`../tests/test_assembly_fingerprint.py`](../tests/test_assembly_fingerprint.py).

## Limitations

- Sensor lag and all non-contact physical parameters are treated as known.
- The batch varies only one hidden physical quantity.
- The intervals are local and use the assumed 0.02 K noise model.
- Real classification thresholds require repeatability, calibration, process
  capability, and independent inspection data.
