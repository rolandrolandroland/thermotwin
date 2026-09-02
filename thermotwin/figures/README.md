# Generated experiment artifacts

ThermoTwin groups generated figures and their source-data sidecars by the name
of the corresponding walkthrough Markdown file. The folder name is the
Markdown filename without `.md`.

For example:

```text
thermotwin/COP_OPERATING_MAP_EXPERIMENT.md
thermotwin/figures/COP_OPERATING_MAP_EXPERIMENT/
    cop_operating_map.png
    cop_operating_map.json
```

The PNG is the presentation artifact. The same-stem JSON contains the report
result used to construct the plot. The same-stem TXT explains what the figure
shows, how to use the JSON, and the interpretation boundary. All three are
generated, ignored by Git, and reproducible from committed code.

| Artifact folder | Walkthrough | Typical figures |
| --- | --- | --- |
| `AG2SE_SUBSTITUTION_EXPERIMENT/` | [`AG2SE_SUBSTITUTION_EXPERIMENT.md`](../AG2SE_SUBSTITUTION_EXPERIMENT.md) | `ag2se_matched_substitution.png` |
| `ASSEMBLY_FINGERPRINT_EXPERIMENT/` | [`ASSEMBLY_FINGERPRINT_EXPERIMENT.md`](../ASSEMBLY_FINGERPRINT_EXPERIMENT.md) | `assembly_fingerprint.png` |
| `CONTACT_RESISTANCE_EXPERIMENT/` | [`CONTACT_RESISTANCE_EXPERIMENT.md`](../CONTACT_RESISTANCE_EXPERIMENT.md) | contact model and four contact-PINN reports |
| `CONTROL_COMPARISON_EXPERIMENT/` | [`CONTROL_COMPARISON_EXPERIMENT.md`](../CONTROL_COMPARISON_EXPERIMENT.md) | `control_comparison.png` |
| `COP_OPERATING_MAP_EXPERIMENT/` | [`COP_OPERATING_MAP_EXPERIMENT.md`](../COP_OPERATING_MAP_EXPERIMENT.md) | `cop_operating_map.png` |
| `DISTRIBUTED_CONSTITUTIVE_INFERENCE/` | [`DISTRIBUTED_CONSTITUTIVE_INFERENCE.md`](../DISTRIBUTED_CONSTITUTIVE_INFERENCE.md) | `distributed_property_study.png` |
| `DISTRIBUTED_INDEPENDENT_VALIDATION/` | [`DISTRIBUTED_INDEPENDENT_VALIDATION.md`](../DISTRIBUTED_INDEPENDENT_VALIDATION.md) | `distributed_independent_validation.png` |
| `DISTRIBUTED_INVERSE_ROBUSTNESS/` | [`DISTRIBUTED_INVERSE_ROBUSTNESS.md`](../DISTRIBUTED_INVERSE_ROBUSTNESS.md) | `distributed_inverse_robustness.png` |
| `DISTRIBUTED_OBSERVATION_IDENTIFIABILITY/` | [`DISTRIBUTED_OBSERVATION_IDENTIFIABILITY.md`](../DISTRIBUTED_OBSERVATION_IDENTIFIABILITY.md) | `distributed_observation_identifiability.png` |
| `DISTRIBUTED_PINN_TRAINING_AUDIT/` | [`DISTRIBUTED_PINN_TRAINING_AUDIT.md`](../DISTRIBUTED_PINN_TRAINING_AUDIT.md) | `distributed_pinn_training_audit.png` |
| `DISTRIBUTED_PROFILE_COVERAGE/` | [`DISTRIBUTED_PROFILE_COVERAGE.md`](../DISTRIBUTED_PROFILE_COVERAGE.md) | coverage figure and optional text report |
| `DISTRIBUTED_WITHHELD_VALIDATION/` | [`DISTRIBUTED_WITHHELD_VALIDATION.md`](../DISTRIBUTED_WITHHELD_VALIDATION.md) | `distributed_withheld_validation.png` |
| `ELECTRICAL_CONTACT_PROCESS_WINDOW/` | [`ELECTRICAL_CONTACT_PROCESS_WINDOW.md`](../ELECTRICAL_CONTACT_PROCESS_WINDOW.md) | `electrical_contact_process_window.png` |
| `ENGINEERING_SHOWCASE/` | [`ENGINEERING_SHOWCASE.md`](../ENGINEERING_SHOWCASE.md) | `engineering_decision_showcase.png` |
| `MATERIAL_GEOMETRY_BAYESIAN_CODESIGN/` | [`MATERIAL_GEOMETRY_BAYESIAN_CODESIGN.md`](../MATERIAL_GEOMETRY_BAYESIAN_CODESIGN.md) | `material_geometry_bayesian_codesign.png` |
| `NEXT_EXPERIMENT_WALKTHROUGH/` | [`NEXT_EXPERIMENT_WALKTHROUGH.md`](../NEXT_EXPERIMENT_WALKTHROUGH.md) | `experiment_selection.png` |
| `PINN_SHOWCASE/` | [`PINN_SHOWCASE.md`](../PINN_SHOWCASE.md) | forward comparison and combined showcase |
| `PULSE_OPERATING_MAP_EXPERIMENT/` | [`PULSE_OPERATING_MAP_EXPERIMENT.md`](../PULSE_OPERATING_MAP_EXPERIMENT.md) | `pulse_operating_map.png` |
| `PWM_POWER_ELECTRONICS_EXPERIMENT/` | [`PWM_POWER_ELECTRONICS_EXPERIMENT.md`](../PWM_POWER_ELECTRONICS_EXPERIMENT.md) | `pwm_power_electronics.png` |
| `SPARSE_SENSOR_EXPERIMENT/` | [`SPARSE_SENSOR_EXPERIMENT.md`](../SPARSE_SENSOR_EXPERIMENT.md) | `sparse_sensor_inference.png` |

When `--output PATH` is supplied, the JSON and TXT sidecars are written beside
that custom figure using the same stem. A legacy PNG generated before sidecars
were introduced will not have reconstructable data or an explanation until its
report is rerun.

`HARDWARE_VALIDATION_PROTOCOL.md` intentionally has no result figure because
no hardware experiment has been run. Generating a synthetic chart for that
document would incorrectly imply experimental evidence.

Regenerate the complete catalog with:

```bash
python3 -m thermotwin.generate_all_figures
```

The command runs each report in an isolated process so optional plotting and
PINN state cannot leak from one experiment into another.
