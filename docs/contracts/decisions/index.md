# Architecture decision records

An ADR records one decision that shapes the library: the context, the choice, its consequences, the
alternatives, and which contract rules enforce it. Accepted ADRs are never edited except their Status line;
a change is a new ADR that supersedes the old one. When to write one: see [00-index](../00-index.md).

Template: `# NNNN Title` / Status (Proposed | Accepted | Superseded by NNNN) + Date / Context (3–6 lines) /
Decision ("We will …") / Consequences / Alternatives considered / Enforcement (rule IDs, ruff codes,
`tests/contract/*`). File name `NNNN-kebab-title.md`.

| ADR | Title | Status |
|-----|-------|--------|
| [0001](0001-torch-only.md) | Torch is the only implementation | Accepted |
| [0002](0002-immutable-fk-state.md) | A forward-kinematics query returns an immutable state | Accepted |
| [0003](0003-terms-plain-functions.md) | Terms are plain functions | Accepted |
| [0004](0004-viser-viewer.md) | One viewer, built on viser | Accepted |
| [0005](0005-nq-nv.md) | Configurations and velocities have different sizes | Accepted |
| [0006](0006-serialization.md) | Serialization and format versioning of Model and IR | Proposed |
