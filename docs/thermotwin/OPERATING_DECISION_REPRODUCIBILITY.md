# Operating decision reproducibility package

Date: 2026-10-29. Scope: the completed prospective four-action synthetic
experiment. The local package is hash-verified. It has not been uploaded to
an immutable external repository or release.

## Deliverables

The generated files remain gitignored under
`reports/operating-decision-closeout-20261029/`:

- `operating-decision-final-report.pdf`: 11 pages, including all three vector figures.
- `operating-decision-final-presentation.pptx`: ten slides, with editable tables,
  editable charts, embedded chart-data workbooks and source notes.
- `operating-decision-reproducibility.zip`: source, scientific evidence,
  environment records, audit tools, figures and the report and presentation.

The [report source](OPERATING_DECISION_FINAL_REPORT.md),
[figure guide](OPERATING_DECISION_FINAL_FIGURES.md),
[package plan](../../thermotwin/OPERATING_DECISION_PROSPECTIVE_PACKAGE_PLAN.json),
[package manifest](../../thermotwin/OPERATING_DECISION_PROSPECTIVE_PACKAGE_MANIFEST.json)
and [closeout receipt](../../thermotwin/OPERATING_DECISION_PROSPECTIVE_CLOSEOUT.json)
are tracked. The receipt publishes the ZIP's SHA-256 and byte count separately
from the ZIP, avoiding a self-reference. Large raw evidence and generated
deliverables remain outside Git. A clean Git clone alone does not contain the
raw cohort evidence or the ZIP. Preserve the local package or deposit it
separately before moving to another machine.

## What is inside

The ZIP contains `PACKAGE_MANIFEST.json`, this guide as `README.md`,
`repository/` and `frozen-source/`. Every payload file has its own SHA-256 and
byte count. The repository snapshot comes from the recorded base commit plus
the explicit reviewed closeout overlays in the package plan. Unrelated
working-tree edits, Git metadata, virtual environments and credentials are
excluded.

The evidence includes the accepted P4 pilot, D1 acquisition/tuning archive,
D2 and D3 analyses, D4 sensitivity check, all D5 physical-stress map archives,
D6 internal check, 100-block independent calibration and 100-block reserved
evaluation. Original raw observations, candidate fits, predictive draws,
verification records, before-reveal decisions, target reveals and resource
records are retained. The bounded Phase H numerical replay, independent
audit records and relevant test logs are included. Older superseded and
invalid disposable campaigns remain outside this final package; their
incident history and scope are preserved in the repository documents.

`frozen-source/` contains exactly the 117 original numerical source files and
their source manifest. It is separate from the later analysis/reporting
snapshot. All 117 files also match the repository snapshot at the numerical
paths. The scientific source is
`7e3b9a3be22d991ec2e588cbd4a53a9d4d2d27d7`; the accepted reserved scientific
digest is
`d565c0c6368b602eeb3d2721c128aab88221bce9a6b5abe68fcda4c727d4edf0`.

## Verify and extract the package

Compare the ZIP SHA-256 and byte count with the tracked closeout receipt
before opening it. From this repository, run the standard-library verifier
with fresh output paths:

```sh
python -m tools.operating_decision_closeout_package verify \
  --archive reports/operating-decision-closeout-20261029/operating-decision-reproducibility.zip \
  --extract-to /tmp/thermotwin-operating-decision-verified \
  --output /tmp/thermotwin-operating-decision-package-check.json
```

The verifier checks the manifest seal, complete membership, every file length
and SHA-256, then every extracted file. Duplicate paths, traversal, absolute
paths, symbolic links, unlisted files and altered bytes are rejected. It
refuses to reuse an existing extraction directory. A local SHA-256 manifest
detects changes against the published identity; it is not a digital signature
or an independent trust authority.

## Reconstruct all reported decisions and comparisons

In the extracted `repository/` directory, the independent arithmetic audit
requires only Python's standard library. Use a fresh audit output:

```sh
python -m tools.verify_prospective_phase_f_result \
  --archive .artifacts/operating-decision/prospective-phase-f-reserved-7e3b9a3-v1/evaluation.json \
  --source-root . \
  --validation-record .artifacts/operating-decision/phase-f-durable-reserved-7e3b9a3-20261018/independent-validation.json \
  --output /tmp/thermotwin-extracted-result-audit.json
```

This reconstructs all 6,000 saved decisions, primary and secondary metrics,
family bounds and paired comparisons from original checkpoints and
before-reveal records. It performs no fitting, physical simulation or cohort
generation. Counts and decisions must match exactly; independently calculated
numerical fields use the separately declared audit tolerance. Fresh audit
timestamps may change an audit packaging hash without changing the scientific
result.

The retained numerical replay can also be checked without any new refits:

```sh
python -m tools.verify_prospective_phase_h_replay \
  --archive .artifacts/operating-decision/prospective-phase-f-reserved-7e3b9a3-v1/evaluation.json \
  --plan thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_H_REPLAY_PLAN.json \
  --replay-directory .artifacts/operating-decision/phase-h-numerical-replay-20261029 \
  --repository-root . \
  --frozen-source-root ../frozen-source \
  --full-test-log .artifacts/operating-decision/phase-h-full-suite-20261029.log \
  --focused-test-log .artifacts/operating-decision/phase-h-replay-focused-tests-20261029.log \
  --output /tmp/thermotwin-extracted-replay-check.json
```

Do not generate another reserved cohort to reproduce these results. The
retained evidence is sufficient for routine verification. A deliberate
computational replay of the bounded sample is described in
`docs/thermotwin/OPERATING_DECISION_PHASE_H_REPLAY.md`; it needs the frozen
runtime, retained sample plan and a new output directory.

## Recreate the figures and report

All plot inputs are in the tracked compact JSON, so no large scientific
archive or numerical ThermoTwin imports are needed to render the figures:

```sh
python -m tools.render_operating_decision_final_figures \
  --inputs thermotwin/OPERATING_DECISION_PROSPECTIVE_FINAL_FIGURE_INPUTS.json \
  --output-directory /tmp/thermotwin-operating-decision-new-figures
```

Use Matplotlib 3.10.8. Byte-for-byte figure reproduction was verified in the
recorded CPython 3.10.12 Darwin arm64 environment with NumPy 2.2.6 and
SciPy 1.15.3. Rendering on another platform may preserve scientific content
without producing identical font/raster bytes. The figure manifest binds all
nine original exports and the renderer. The input builder documents how to
reconstruct the compact plot data from retained evidence. It projects saved
candidate fits but does not optimize or regenerate target truth.

The final report builder reads compact accepted result, calibration, figure
and supporting-diagnostics artifacts. With ReportLab and pypdf installed,
use fresh output paths:

```sh
python -m tools.build_operating_decision_final_report \
  --repository-root . \
  --figures reports/operating-decision-final-figures-20261029 \
  --pdf /tmp/thermotwin-operating-decision-report.pdf \
  --markdown /tmp/thermotwin-operating-decision-report.md
```

The authored report used the Codex bundled document runtime. Its PDF has
eight report pages plus the three accepted full-size vector figures. Report
metadata may differ on regeneration. Scientific numbers must remain unchanged.
The supporting parameter diagnostic tool replays the already registered
scalar parameter seeds in the original environment. It does not generate
new draws, observations, ODE responses or fitted models. Matched-family
parameter summaries are conditional on passed four-state verifications and
explicitly retain excluded-fit counts.

The editable presentation and its source builder are included. Reauthoring
uses the Codex bundled Node runtime and `@oai/artifact-tool`, with its supplied
presentation finalizer. Run a copy of
`tools/build_operating_decision_final_presentation.mjs` from a private build
directory linked to the bundled `node_modules`. Its arguments are repository
root, private build directory, new final-output directory, presentation skill
directory and bundled Python executable. Set `RUNTIME_NODE_MODULES` and
`RUNTIME_PYTHON` to the resolved bundled paths. Ordinary slide editing does
not require that authoring runtime. The package preserves the already
validated PPTX and its embedded chart data.

## Verification boundary and interpretation

All 11 report pages and all ten slide previews were inspected. The presentation
passed structural, geometry, chart/workbook-reference and import checks.
It was not tested by opening it in native Microsoft PowerPoint. The focused
closeout tests cover archive integrity and extraction safety, original scalar
seed authentication, report denominator coverage and prior scientific audit
boundaries. The preceding full scientific suite and bounded replay acceptance
remain recorded with their original scope and timestamps.

The package was round-tripped into a fresh directory, every payload hash was
verified, and both the independent result audit and retained-replay verifier
passed against the extracted evidence. This is an auditable local closeout,
not an external archival deposit or a portability guarantee for numerical
fits on arbitrary platforms.

Primary success remains not established. Energy savings alone do not qualify
an operating policy. The 48 development map cells have no reserved risk
guarantee. Hardware validation and a future decision-aware selector require
separate work. No further cohort generation or retuning is part of this
closeout.
