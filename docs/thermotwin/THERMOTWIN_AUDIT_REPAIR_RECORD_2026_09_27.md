# ThermoTwin audit repair record

Date: 2026-09-27. Status: implementation complete; exact-head execution gates
pending. This record follows the
[September 26 repair plan](THERMOTWIN_AUDIT_REPAIR_PLAN_2026_09_26.md). It is
not a scientific outcome and does not authorize development data generation.

## Preserved baseline

Implementation began in an isolated checkout of reviewed revision
`56953c7784a9d685a53ed87ca01b7f2d3ae58f06`. The original working checkout and
its unrelated edits were not modified. The research bundle
`reports/hax_research_2026_09_17_v1/` is byte-for-byte preserved; the sorted
file-hash inventory has SHA-256
`6d46f068a653680d4b560f7b9a9a8c1102677dbd89bb1a29353cfe0a62d74443`.

Relevant original working-copy presentation sources were incorporated
deliberately. Their pre-repair SHA-256 values were:

| File | SHA-256 |
| --- | --- |
| `thermotwin/PINN_EVIDENCE_REPORT_cd.md` | `8065fdba9e369ba6e5a7911954a64705e2b151a8416994744989305e00cf193e` |
| `thermotwin/PINN_EVIDENCE_SUMMARY.md` | `f7347d1fa0b11d93c8b4062a4e5a1963539906a2485a28308297d7cafb8b24f4` |
| `thermotwin/EXPERIMENT_SELECTION_REPORT_cd.md` | `667442643cbda7d0c238bb92b217668c6299f7df1fc348c6387fdf5db2fd5e13` |
| `thermotwin/EXPERIMENT_SELECTION_SUMMARY.md` | `7bed91a708a77271db75ff5960fafba3b46f0122d437b46c4b5b98d4043932b3` |
| `thermotwin/reports/pinn_evidence_summary.py` | `3ad5a39943494f6a81c94e4b9a7a5b209e3997d143ba642230ec88d0067e4fc8` |
| `thermotwin/reports/next_experiment_story.py` | `9549aff34a166770533296ac34bcabe595591bd97bc94d5209948f7474358692` |
| `thermotwin/reports/next_experiment_story_corrected.py` | `01b346e5ba527ffeac4467513251423fdd0ca7f53435cfbbd1875771c0fa7db7` |

Historical scientific artifacts remain unchanged:

| Evidence | JSON SHA-256 | Status |
| --- | --- | --- |
| P4 parent | `91f7d21c9e3a72bb8f341efc78ff7ce0a1a570fef0ff8540ed998e1655770bad` | Valid frozen Phase C evidence. |
| P4 N32 | `749e6896bbd90503faf75c6deaffd24a5ede63baf5e6e15fa84e2d08c12f66cf` | Valid frozen continuation. |
| Historical D0 v1 | `ace47b991c37a7e43a940a909dc63ae809d7839333969684190214ba634e50ae` | Valid CPython 3.13.3 disposable artifact; preserved, not reinterpreted. |

## Versioned repairs

Protocol v2 chooses CPython 3.10.12 on Darwin arm64 and pins its scientific
package environment in `thermotwin/requirements-prospective-phase-d.txt`. It
uses a new disposable namespace and archive schema, so the historical D0
addresses remain meaningful.

| Audit finding | Changed evidence | Verification | Retained evidence |
| --- | --- | --- | --- |
| Infinite offsets had no complete implementation | `operating_decision_prospective_phase_d.py` and the Phase D plan stop the primary campaign if any required offset is infinite. | Constructed finite, stop, each measurement action, multiple-infinite, and 18th-order boundary tests. | All scores and failures remain reportable; no tuning winner or internal check is authorized after the stop. |
| Development loss did not bind adjusted final decisions | The v2 decision transform expands `[L,U]` to `[L-d_a,U+d_a]`, recomputes the decision, and retains raw and adjusted records. | Approval-to-abstention, rejection-to-abstention, unchanged, exact-boundary, verification, missing/nonfinite, failure, and all 81 grid-row tests. | The independent Phase E correction remains outside development. |
| Performance noise entered scientific identity | Versioned scientific block and rehearsal payloads remove timing/resource fields while the complete archive seal retains every byte. | Timing/resource mutations preserve scientific identity and change archive identity; observation, fit, decision, failure, stream, and protocol mutations change scientific identity; round-trip/tamper checks. | Historical D0 v1 identities are unchanged. |
| Memory budget described a worker estimate as a cap | The protocol now labels 185,204,736 bytes as worker-only and adds a constructed 20-block write/load/validation probe with process-tree sampling. | Exact-head probe result will be recorded below. | Phase C timing and worker measurements remain historical estimates. |
| Prospective runtime was ambiguous | New evidence is rejected unless it executes in the pinned CPython 3.10.12 environment. | Runtime identity test, exact-head rehearsal, and full suite in that environment. | Historical D0 v1 remains a CPython 3.13.3 artifact. |
| Report caveats and arithmetic were incomplete | Active reports disclose extrapolation, development overlap, paired denominators, the separate all-20 mean, and exact 3.92/6.55/6.60 factors. The incorrect legacy figure is visibly superseded. | Retained-data reconciliation, active-link tests, generated figure inspection, and page-by-page DOCX/PDF inspection. | Bundle v1 is byte-for-byte preserved; v2 records unchanged scientific inputs. |
| Status pointers drifted | Project status, Phase D plan, historical D0 result, original outline, completion-plan entry point, and roadmap now point to the same repair boundary. | Link and wording review. | Earlier dated records remain available. |

The CLI fixture strings `archive bytes: 1` and repeated `b` hashes are
deliberate mocks for command-output tests. They are not serialized archive
measurements. Real archive sealing, byte-size fixed points, round trips, and
tamper rejection are tested separately; no hashing repair was required.

## Revised presentation bundle

`reports/hax_research_2026_09_27_v2/` contains Markdown, editable Word, and PDF
reports plus regenerated active figures. Its manifest records that scientific
inputs did not change and binds every revised output. Both Word documents and
both PDFs were rendered and inspected page by page: 10 pages for experiment
selection and 12 pages for PINN evidence. No clipping, overlap, missing glyph,
or broken figure reference was observed.

## Exact-head repair gate

To be completed after the repair revision is committed:

| Gate | Result |
| --- | --- |
| Focused tests | 25 protocol, identity, resource-probe, and bundle tests passed before commit; exact-head rerun pending. |
| Full CPython 3.10.12 suite | 842 tests passed in 1,176.97 s before commit; exact-head rerun pending. |
| Constructed 20-block archive resource probe | Pending. |
| Replacement disposable computation 1 | Pending. |
| Replacement disposable computation 2 | Pending. |
| Scientific-digest repeatability and full-archive validation | Pending. |
| Exact-head CI | Pending. |

`p1_development_tuning`, the internal check, independent calibration, and the
reserved evaluation remain closed until every row above passes and this record
is updated with exact identities and measurements.
