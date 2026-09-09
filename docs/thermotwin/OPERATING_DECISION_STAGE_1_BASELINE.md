# Operating-decision experiment: Stage 1 baseline reproduction

Date: 2026-09-09

## Outcome

The sensor/model-discrimination baseline reproduced successfully. Every
displayed result matched the committed report at its reported precision,
including failed packages, false-confidence cases, and parameter-bound hits.

This completes Stage 1 of the
[operating-decision experiment outline](OPERATING_DECISION_EXPERIMENT_OUTLINE.md).
No operating-decision policy, new sensor model, or new virtual-device family
has been implemented or evaluated yet.

## Frozen source

The baseline is not present on `main`. It was executed from the existing clean
development worktree using:

```text
branch: dev
commit: fc7c1f34021be243adf4762dc25d647672e3a9a8
title:  Add model-mismatch experiment campaigns
remote: origin/dev at the same commit
```

At reproduction time, `dev` was one commit ahead of `main`. Neither branch was
switched, merged, rebased, or modified during the reproduction.

The relevant frozen artifacts at that revision are:

- `thermotwin/SENSOR_MODEL_DISCRIMINATION.md`;
- `thermotwin/studies/sensor_model_discrimination.py`;
- `thermotwin/reports/sensor_model_discrimination.py`;
- `thermotwin/simulation/interface_mass_mismatch.py`;
- `thermotwin/ADAPTIVE_EXPERIMENT_CAMPAIGN.md`; and
- `tests/test_sensor_model_discrimination.py` and
  `tests/test_adaptive_experiment_campaign.py`.

## Runtime environment

ThermoTwin declares Python 3.10 or newer. The default system interpreter was
Python 3.9.6 and failed during import when it encountered modern union-type
syntax. The reproduction therefore used the available bundled Python 3.12.14
interpreter:

```text
/Users/rolandbennett/.cache/codex-runtimes/
codex-primary-runtime/dependencies/python/bin/python3
```

This interpreter did not provide Matplotlib. The numerical report was therefore
reproduced with `--no-figure`; no PNG, JSON, or TXT sidecar was generated.

## Focused verification

Command:

```bash
/Users/rolandbennett/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m unittest \
  tests.test_adaptive_experiment_campaign \
  tests.test_sensor_model_discrimination \
  tests.test_generate_all_figures
```

Result: **13 tests passed in 14.4 seconds**.

These tests verify the independent fifth-state truth model's limiting cases,
paired campaign construction, model-choice mechanics, budget checks, report
scope statements, and report catalog. The study tests use reduced one-trial
configurations. They are smoke and invariant checks, not golden tests for the
complete numerical table.

## Frozen numerical reproduction

Command:

```bash
/Users/rolandbennett/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m thermotwin.sensor_model_discrimination \
  --trials 20 \
  --no-figure
```

The completed run used the configured default first seed, `91001`. Passing
`--first-seed 91001` explicitly is equivalent and is recommended for later
automation.

The command completed all 20 paired trials under four-state truth and all 20
paired trials under extra-interface-mass truth without a fit crash. Runtime was
approximately four minutes on the available CPU environment.

Aggregating the two truth conditions gives:

| Measurement package | Correct topology | Physical-decision passes | False confidence | Modeled training energy |
| --- | ---: | ---: | ---: | ---: |
| One pulse, exchanger temperatures | 34/40 | 25/40 | 6/40 | 27.54 J |
| Four pulses, exchanger temperatures | 37/40 | 32/40 | 3/40 | 64.41 J |
| One pulse plus cold-face temperature | 40/40 | 40/40 | 0/40 | 27.54 J |
| One pulse plus cold-side heat rate | 40/40 | 40/40 | 0/40 | 27.54 J |
| One pulse plus voltage | 40/40 | 40/40 | 0/40 | 27.54 J |

The common bipolar validation run is additional to the training runs and
training-energy totals above.

The predeclared gate passed for cold-face temperature, cold-side heat rate, and
voltage. It failed for both exchanger-temperature-only packages. These are
reproduced synthetic results, not new evidence about physical hardware.

## Detailed comparison with the frozen report

Under matched four-state truth:

| Package | Correct model | Decision pass | Parameter log-RMSE | Hidden-face RMSE | False confidence |
| --- | ---: | ---: | ---: | ---: | ---: |
| One pulse, exchanger temperatures | 80% | 75% | 0.0664 | 0.0310 K | 15% |
| Four pulses, exchanger temperatures | 90% | 90% | 0.0400 | 0.0199 K | 5% |
| Cold-face temperature | 100% | 100% | 0.0149 | 0.0026 K | 0% |
| Cold-side heat rate | 100% | 100% | 0.0175 | 0.0075 K | 0% |
| Voltage | 100% | 100% | 0.0212 | 0.0048 K | 0% |

Under extra-interface-mass truth:

| Package | Correct model | Decision pass | Parameter log-RMSE | Hidden-face RMSE | False confidence |
| --- | ---: | ---: | ---: | ---: | ---: |
| One pulse, exchanger temperatures | 90% | 50% | 0.0859 | 0.0392 K | 15% |
| Four pulses, exchanger temperatures | 95% | 70% | 0.0754 | 0.0326 K | 10% |
| Cold-face temperature | 100% | 100% | 0.0300 | 0.0060 K | 0% |
| Cold-side heat rate | 100% | 100% | 0.0273 | 0.0100 K | 0% |
| Voltage | 100% | 100% | 0.0417 | 0.0058 K | 0% |

## What is now frozen for Stage 2

Stage 2 should treat the following as baseline facts:

1. The existing deterministic command and seed reproduce the report.
2. The baseline comparison contains five measurement packages and two truth
   families.
3. The added-channel results are optimistic because truth always belongs to a
   candidate family, voltage has no uncertain electrical-series-resistance
   confounder, and a face sensor adds no thermal loading.
4. Correct topology and hidden-face RMSE are diagnostic outcomes. The extension's
   primary outcome will instead be a frozen operating decision on a third,
   untouched schedule.
5. All future changes must preserve a command capable of reproducing this
   baseline or explicitly version and explain any changed result.

Stage 2 can now implement the operating margin, fixed measurement policies, and
the acquisition/verification/final-evaluation split.
