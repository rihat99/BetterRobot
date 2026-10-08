# 0004 One viewer, built on viser

**Status:** Accepted · **Date:** 2026-10-07 (recorded; decision taken earlier)

## Context
A viewer needs a browser scene, GUI widgets and a recording format, and must work on remote GPU servers
with no OpenGL context. CI has no browser client.

## Decision
We will ship one viewer, `better_robot.viewer.Viewer`, built on viser, with layers fed whole `(T, ...)` clips,
and an offline torch rasterizer for recordings. The viewer's time axis is called `step`.

## Consequences
One Python dependency gives scene, GUI and recordings; the torch rasterizer makes recording work where viser's
browser client cannot. The viewer sits at the top of the import DAG; nothing imports it, and viser is imported
only when a `Viewer` is created.

## Note 2026-10-08
viser stays the only backend. `viewer.server.gui` is the sanctioned API for studio GUIs (no separate GUI brick).
The viewer owns viser's single scene-pointer slot (`Viewer.on_pointer` / `clear_pointer`). An `.npz` export of
the layer data is planned so other front ends can replay a scene without viser.

## Alternatives considered
meshcat (no GUI widgets, weaker recording); rerun (own data model, duplicate state); a native OpenGL window
(needs a display on the compute machine).

## Enforcement
C-DEP-1, C-DEP-2, C-NAME-4; `tests/contract/test_import_dag.py`, `tests/contract/test_optional_deps.py`.
