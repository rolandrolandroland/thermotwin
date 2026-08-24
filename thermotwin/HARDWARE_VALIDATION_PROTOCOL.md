# Hardware validation protocol and data contract

No hardware experiment has been run for this repository. This document
prepares the software boundary without fabricating physical evidence.

## Required decisions before energizing hardware

The experiment owner must obtain and record, from the actual module, power
supply, heat exchangers, and sensors:

- maximum continuous and pulse current;
- maximum voltage and electrical power;
- allowable hot- and cold-face temperatures;
- condensation and electrical-isolation controls;
- heat-sink or fluid-flow operating requirements;
- sensor mounting locations and attachment method;
- temperature, current, and voltage calibration results;
- emergency shutdown conditions; and
- whether polarity reversal is allowed.

ThermoTwin's generic synthetic limits are not hardware safety limits.

## Minimum measurement file

[`hardware_data.py`](hardware_data.py) accepts a CSV with these SI-unit column
names:

```text
time_s,current_A,cold_exchanger_K,hot_exchanger_K,voltage_V
```

`voltage_V` is optional. A blank exchanger-temperature cell represents one
missing reading. A row with both temperatures blank is retained in the input
history even though it creates no temperature observation. Time and current
are required on every row. Times must be strictly increasing, temperatures
must be positive kelvin values, and every numeric value must be finite.

Load and inspect a file with:

```python
from thermotwin import load_hardware_csv, summarize_hardware_dataset

dataset = load_hardware_csv("my_measurements.csv")
print(summarize_hardware_dataset(dataset))
```

The loader validates schema and units implied by the headers. It cannot verify
that a sensor was mounted at the stated location or calibrated correctly.

## Recommended experiment sequence

1. Record an unpowered equilibrium interval.
2. Run a conservative constant-current test well inside documented limits.
3. Return fully to equilibrium and repeat it to measure repeatability.
4. Run the selected pulse only after its amplitude, energy, and temperature
   limits have been checked for the actual hardware.
5. Reserve at least one complete schedule for validation rather than fitting
   every collected run.
6. Fit the accessible-sensor model to the training experiment.
7. Predict the withheld schedule without changing parameters.
8. Report temperature residuals, inferred parameters, energy closure, and
   systematic model discrepancy.

## Required result distinction

A future walkthrough must separately report:

- raw observations;
- calibration and missing-data handling;
- inferred quantities;
- within-experiment fit error;
- genuinely withheld-experiment prediction error; and
- discrepancies that the four-node model cannot explain.

Until those measurements exist, every numerical result in the other
walkthroughs remains explicitly synthetic.
