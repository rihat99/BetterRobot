# BetterRobot — Design Docs

This folder is the authority on *what* the library does and *why*. Each doc is
a focused, implementable proposal. Read them in order — the early docs set
vocabulary the later ones rely on.

## Core specs (00-12)

| # | File | Scope |
|---|------|-------|
| 00 | [VISION.md](00_VISION.md) | Goals, non-goals, unique contributions |
| 01 | [ARCHITECTURE.md](01_ARCHITECTURE.md) | Layered DAG, target directory layout |
| 02 | [DATA_MODEL.md](02_DATA_MODEL.md) | `Model` / `Data`, joints, bodies, frames, inertias |
| 03 | [LIE_AND_SPATIAL.md](03_LIE_AND_SPATIAL.md) | `SE3`/`SO3`/`Motion`/`Force`/`Inertia`, pypose backend, Warp path |
| 04 | [PARSERS.md](04_PARSERS.md) | URDF / MJCF → IR → `Model`, programmatic builders |
| 05 | [KINEMATICS.md](05_KINEMATICS.md) | FK, frame updates, unified analytic + autodiff Jacobians |
| 06 | [DYNAMICS.md](06_DYNAMICS.md) | RNEA/ABA/CRBA skeleton, three-layer action model, integrators |
| 07 | [RESIDUALS_COSTS_SOLVERS.md](07_RESIDUALS_COSTS_SOLVERS.md) | Residual registry, `CostStack`, `Protocol`-pluggable solvers |
| 08 | [TASKS.md](08_TASKS.md) | IK (two-stage), trajopt, retargeting, `Trajectory` type |
| 09 | [COLLISION_GEOMETRY.md](09_COLLISION_GEOMETRY.md) | Capsule-first collision, pairwise SDF dispatch, sparse Jacobians |
| 10 | [BATCHING_AND_BACKENDS.md](10_BATCHING_AND_BACKENDS.md) | `(B, [T,] ..., D)` convention, device/dtype, Warp bridge |
| 11 | [SKELETON_AND_MIGRATION.md](11_SKELETON_AND_MIGRATION.md) | File-by-file migration plan, phased milestones |
| 12 | [VIEWER.md](12_VIEWER.md) | Viser-backed visualiser, render modes, overlays |

## Cross-cutting specs (13-17)

These are **new** normative docs that previously existed only implicitly.
Everything the library expects at its boundaries lives here.

| # | File | Scope |
|---|------|-------|
| 13 | [NAMING.md](13_NAMING.md) | Naming convention, glossary, Pinocchio → BetterRobot rename table |
| 14 | [PERFORMANCE.md](14_PERFORMANCE.md) | Latency/memory budgets, compile strategy, profiling, CI guard |
| 15 | [EXTENSION.md](15_EXTENSION.md) | How to add residuals, joints, solvers, kernels, primitives, backends |
| 16 | [TESTING.md](16_TESTING.md) | Unit/integration/contract/regression/benchmark strategy |
| 17 | [CONTRACTS.md](17_CONTRACTS.md) | Input contracts, error taxonomy, numerical guarantees, SemVer |

## Status tracker

| # | File | Scope |
|---|------|-------|
| 18 | [ROADMAP.md](18_ROADMAP.md) | What is specified in the docs but still stub / missing in `src/` |

## Reading order

- **New to the codebase?** 00 → 01 → 02 → 13 → 05.
- **Adding a feature?** 15 → the relevant core spec (e.g. 07 for a new
  residual) → 16 → 17.
- **Debugging a perf issue?** 14 → 10 → the relevant core spec.
- **Writing new docs?** 13 sets the vocabulary; 17 sets the contract
  discipline.

## Rule of thumb

Every plan above is a specification for the **new** code. Existing code
stays only when explicitly listed in the migration plan (11). Everything
else is either rewritten or reshaped to fit the new data model and the
naming in 13.

When a newer doc contradicts an older one, the newer doc wins. Open an
issue to bring the older one in line.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for doc-level changes per phase.
