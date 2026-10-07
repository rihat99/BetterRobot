# 0006 Serialization and format versioning of Model and IR

**Status:** Proposed · **Date:** 2026-10-07

## Context
A `Model` is built from URDF, MJCF or the programmatic builder through the `io` intermediate representation
(IR). Nothing today writes a `Model` or an IR back to disk, and nothing stamps a format version on what is
written; `torch.save` of a `Model` ties the file to the current dataclass layout. Users who want to cache a
parsed model, ship a trained parameter set, or load a model in a process without the source file have no
supported path.

## Decision
Planned, not yet taken. The candidate decision is: `Model` and the IR get an explicit `save`/`load` pair in
`io`, the written file carries a `format_version` integer, and a loader refuses a newer version with a
`BetterRobotError` naming both versions. Until this ADR is Accepted there is no serialization contract:
`torch.save`/`pickle` of a `Model` is unsupported across releases.

## Consequences
Once accepted, a change to the layout of `ModelStructure`, `ModelValues` or the IR bumps `format_version` and
is listed in `docs/reference/changelog.md` (`C-EXT-6`). The loader becomes a public boundary (`C-ERR-1`).

## Alternatives considered
- Keep relying on re-parsing the source file: simplest, but loses programmatic models and learned values.
- Export to a third-party format (USD, MJCF): useful for interchange, not for round-tripping `ModelValues`.

## Enforcement
None until Accepted. Then a round-trip test under `tests/io/` and a `[review]` check on the version bump.
