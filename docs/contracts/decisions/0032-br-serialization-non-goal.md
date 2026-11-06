# 0032 BetterRobot writes no native model file

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
ADR 0006 (`format_version`, save and load of `Model` and IR) is Proposed and implemented nowhere. No `torch.save` of a
`Model` exists in BetterHuman, BetterRetarget or BetterVideoReconstruction. Today `Model.meta` keeps the parsed IR alive
instead of a file (0016 removes that). A native file would be a format with a promise for every later release (8.7)
and a migration chain (8.8), for no caller.

## Decision
We will state on the scope page that BetterRobot has no native model file (a model is rebuilt from its source, and
`torch.save` of a `Model` is unsupported across releases), and supersede ADR 0006 with this record.

## Example
```markdown
# Scope  (non-goals)
- A native model file. Rebuild a `Model` from its URDF, MJCF or builder script. `torch.save(model)` is unsupported
  across releases. Decision 0032; reopening needs a caller and a format spec (8.7).
```
```python { .no-run }
model = br.load("panda.urdf")        # the supported way to get the same model again
```

## Consequences
No format to version, migrate or fixture. A user who wants fast reloads re-parses; URDF parsing is not the bottleneck
(BetterHuman's 60 s load is an archive reader, BetterHuman 0013). The template of record statuses has no Rejected value, so ADR
0006's status line changes to `Superseded by` this record and the rejection reason lives here and on the scope page.

## Alternatives
`io.save_model` and `io.load_model` as an npz with `format="better_robot.model"`, `format_version`, a spec page and
migrations: a promise to maintain a layout that follows every `Model` change. Drake reads URDF, SDFormat and MJCF and
has no native model file; Pinocchio and MuJoCo can save a built model to a binary file. A format can be added later
without breaking anyone.

## Compatibility
Nothing breaks: no one writes the file. Adding the format later is compatible. Reversible: yes.

## Enforcement
Scope page (principle 8.1) with the non-goal line, and the commonly rejected list (8.3) linking this record; a review
question for any diff that adds `save` or `load` for a `Model`.

## Open questions
None.
