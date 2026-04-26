# Design specs

These thirteen specs are the **authority on what the library does and
why**. Read them in order — early specs set vocabulary the later ones
rely on. They are denser than the {doc}`/concepts/index` pages; treat
them as the reference, not as introductory reading.

```{toctree}
:maxdepth: 1

00_VISION
01_ARCHITECTURE
02_DATA_MODEL
03_LIE_AND_SPATIAL
04_PARSERS
05_KINEMATICS
06_DYNAMICS
07_RESIDUALS_COSTS_SOLVERS
08_TASKS
09_COLLISION_GEOMETRY
10_BATCHING_AND_BACKENDS
11_SKELETON_AND_MIGRATION
12_VIEWER
```

## Suggested entry points

| Where to start | If you want to… |
|----------------|-----------------|
| 00 → 01 → 02 → 13 ({doc}`/conventions/13_NAMING`) → 05 | Get the lay of the land. |
| 15 ({doc}`/conventions/15_EXTENSION`) → the relevant core spec | Add a feature. |
| 14 ({doc}`/conventions/14_PERFORMANCE`) → 10 → the relevant core spec | Debug a perf issue. |
| 11 | Learn what the skeleton landing replaced. (Forensic record only.) |
