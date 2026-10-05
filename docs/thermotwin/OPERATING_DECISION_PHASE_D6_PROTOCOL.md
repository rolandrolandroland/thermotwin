# Operating-decision Phase D6 internal-check protocol

Date: 2026-10-05. Status: implementation complete; exact-head CI and execution pending.

Phase D6 opens `p1_development_internal_check` exactly once for 10 paired blocks
and 30 cases. It applies the provisional design frozen after D5. It does not
refit physics, re-estimate offsets, revisit the 81-row grid, or open an
independent calibration or reserved-evaluation partition.

## Frozen design

The committed provisional-freeze artifact digest is
`a8b4cc22f8bf1d868a743c2427c1225483f954a8aeb9f751e38bec60dd949ca7`.
The primary procedure uses nominal sensing, the `balanced_face_equal` cost
scenario, 16 prospective draws, at most one unstable draw per source/action,
and four workers. The action order is stop, thermal, voltage, then face
temperature. Development offsets are 0.000, 0.000, 0.074, and 0.098 K. Both
stopping clearances and minimum expected reduction are zero; minimum utility
per normalized cost is 0.025.

For each case, the executor generates the common initial evidence and N=16
prospective records. It recomputes the development-padded action evaluations
and saves the frozen action selection before revealing any target response.
It also saves the four fixed-policy decisions before reveal. Truth enters only
the later transfer analysis. The archive validator reconstructs the frozen
selection from authenticated raw draws and rejects any mismatch.

Each completed block is written atomically and can be resumed only when its
partition, source manifest, runtime, protocol, stream audit, scientific digest,
and content seal validate. The final archive inventories every block by byte
size, SHA-256, content digest, and scientific digest. The pinned Phase D runtime
and process-tree monitor remain mandatory.

## Analysis and stopping rule

The analysis reports decisions, errors, abstention, interval coverage,
failures, realized energy, elapsed schedule, run count, instruments, action
shares, truth-family rows, selected-action rows, and candidate-count rows.
Family C and the one-candidate stratum remain explicit; empty rows use null,
which the report interprets as N/A. It also reports changes from the D3 tuning
baseline for action shares, definitive-decision coverage, false approvals,
false rejections, pipeline and verification failures, realized energy, and
instrumentation.

Poor benefit, low action diversity, a false-approval rate above 10% by itself,
or definitive coverage below 70% by itself does not reopen tuning. A redesign
is allowed only if at least one of these predeclared conditions occurs:

1. a material information-boundary, physics, provenance, or implementation
   defect;
2. at least two false approvals in distinct blocks within the same selected
   action and candidate-count stratum; or
3. any measurement action is ineligible in more than 6 of the 30 cases
   (strictly more than 20%).

If no trigger fires, Phase D is complete and a final Phase D freeze may
authorize Phase E protocol work. Phase E evidence is still unopened. If a
trigger fires, this archive becomes tuning evidence; only the already declared
single-revision contingency may be used, and a new namespace must be committed
before replacement evidence is generated.

## Execution gate

The protocol, runner, validator, command-line entry point, and tests must be
committed on `dev`, pushed, and pass exact-head CI before the first internal-
check stream is created. Execution then uses a clean clone at that exact commit
and writes only to the ignored `.artifacts/operating-decision/` tree. After all
10 blocks and the final archive validate, the result and either the final Phase
D freeze or the redesign decision are committed separately.
