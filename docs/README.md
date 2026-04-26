# BetterRobot — Design Docs

This folder is the authority on *what* the library does and *why*. Each doc is
a focused, implementable proposal. Read them in order — the early docs set
vocabulary the later ones rely on.

## Layout

```
docs/
├── design/              — sequential design specs (00–12)
├── conventions/         — cross-cutting normative specs (13–17, 19, 20)
├── status/              — roadmap & known external-dep issues
├── style/               — ARCHIVED coding-style drafts (superseded by 19_STYLE)
├── claude_plan/         — strategic-review audit trail
│   └── accepted/        — the 17 accepted proposals (folded into design/conventions)
├── gpt_plan/            — colleague's strategic review (read alongside RECONCILIATION)
├── README.md            — this index
├── UPDATE_PHASES.md     — implementation phases for the code-change agent
└── CHANGELOG.md         — doc-level changes per phase
```

## Design specs — `design/` (00–12)

| # | File | Scope |
|---|------|-------|
| 00 | [VISION.md](design/00_VISION.md) | Goals, non-goals, unique contributions |
| 01 | [ARCHITECTURE.md](design/01_ARCHITECTURE.md) | Layered DAG, target directory layout, **26-symbol** public API |
| 02 | [DATA_MODEL.md](design/02_DATA_MODEL.md) | `Model` / `Data`, joints, bodies, frames, inertias, **cache invariants** |
| 03 | [LIE_AND_SPATIAL.md](design/03_LIE_AND_SPATIAL.md) | Typed `SE3`/`SO3`/`Pose` in `lie/types.py`; functional `lie.se3.*`; `Motion`/`Force`/`Inertia`; pure-PyTorch backend |
| 04 | [PARSERS.md](design/04_PARSERS.md) | URDF / MJCF → IR → `Model`; `IRModel.schema_version`; `AssetResolver` Protocol; `ModelBuilder` named helpers |
| 05 | [KINEMATICS.md](design/05_KINEMATICS.md) | FK, frame updates, unified analytic + autodiff Jacobians; `ReferenceFrame` enum |
| 06 | [DYNAMICS.md](design/06_DYNAMICS.md) | RNEA/ABA/CRBA skeleton, three-layer action model, integrators, **JointModel dynamics hooks** |
| 07 | [RESIDUALS_COSTS_SOLVERS.md](design/07_RESIDUALS_COSTS_SOLVERS.md) | Residual registry, `CostStack`, `Protocol`-pluggable solvers, **wired `OptimizerConfig` knobs**, `ResidualSpec`, matrix-free `gradient(x)`, `MultiStageOptimizer` |
| 08 | [TASKS.md](design/08_TASKS.md) | IK (two-stage), trajopt, retargeting, `Trajectory` (accepts `(T,nq)` and `(*B,T,nq)`), `TrajectoryParameterization` Protocol |
| 09 | [COLLISION_GEOMETRY.md](design/09_COLLISION_GEOMETRY.md) | Capsule-first collision, pairwise SDF dispatch, sparse Jacobians, **stable residual `dim`** |
| 10 | [BATCHING_AND_BACKENDS.md](design/10_BATCHING_AND_BACKENDS.md) | `(B, [T,] ..., D)` convention, device/dtype, **explicit `Backend` Protocol objects**, Warp bridge |
| 11 | [SKELETON_AND_MIGRATION.md](design/11_SKELETON_AND_MIGRATION.md) | Historical record of the skeleton landing — what was deleted, what survived. For forward work see `UPDATE_PHASES.md` |
| 12 | [VIEWER.md](design/12_VIEWER.md) | Viser-backed visualiser, render modes, overlays, `AssetResolver`-aware mesh loading |

## Cross-cutting specs — `conventions/`

These are **normative** docs that previously existed only implicitly.
Everything the library expects at its boundaries lives here.

| # | File | Scope |
|---|------|-------|
| 13 | [NAMING.md](conventions/13_NAMING.md) | Naming convention, glossary, Pinocchio → BetterRobot rename table; jaxtyping aliases; enum table |
| 14 | [PERFORMANCE.md](conventions/14_PERFORMANCE.md) | Latency/memory budgets, compile strategy, profiling, advisory-then-blocking CI guard, matrix-free trajopt memory wins |
| 15 | [EXTENSION.md](conventions/15_EXTENSION.md) | How to add residuals, joints (incl. dynamics hooks), solvers, kernels, primitives, backends, parameterizations, asset resolvers, actuators (muscles) |
| 16 | [TESTING.md](conventions/16_TESTING.md) | Unit/integration/contract/regression/benchmark strategy; gate-promotion ladder |
| 17 | [CONTRACTS.md](conventions/17_CONTRACTS.md) | Input contracts, error taxonomy (`IRSchemaVersionError`, `StaleCacheError`, …), numerical guarantees, SemVer |
| 19 | [STYLE.md](conventions/19_STYLE.md) | **Normative** coding style (NumPy docstrings, scalar-last quaternions, jaxtyping); supersedes the drafts in `style/` |
| 20 | [PACKAGING.md](conventions/20_PACKAGING.md) | Pyproject extras (`[urdf]`, `[mjcf]`, `[viewer]`, `[geometry]`, `[examples]`, `[warp]`, `[docs]`, `[bench]`, `[test]`, `[dev]`, `[all]`); SemVer pre/post 1.0; deprecation mechanism; release process |

## Status — `status/`

| File | Scope |
|------|-------|
| [18_ROADMAP.md](status/18_ROADMAP.md) | What is specified in the docs but still stub / missing in `src/` |
| [PYPOSE_ISSUES.md](status/PYPOSE_ISSUES.md) | Known PyPose correctness bugs and the **retirement plan** (P10 of UPDATE_PHASES) |

## Implementation guide

[UPDATE_PHASES.md](UPDATE_PHASES.md) is the operational guide for the
code-change agent. It sequences the canonical specs into independently
mergeable phases (P0 → P13) with acceptance criteria.

## Coding style — `style/`

ARCHIVED. The normative coding-style guide is
[conventions/19_STYLE.md](conventions/19_STYLE.md). The two drafts in
`style/` are kept for historical reference only and disagree with the
current codebase in several places (notably quaternion ordering).

| File | Status |
|------|--------|
| [style_by_claude.md](style/style_by_claude.md) | ARCHIVED — see 19_STYLE |
| [style_by_gpt.md](style/style_by_gpt.md) | ARCHIVED — see 19_STYLE |

## Strategic-review audit trail — `claude_plan/`

The 17 strategic proposals authored at the end of phase 1, all
**accepted** as of 2026-04-25. The proposal bodies live under
`claude_plan/accepted/`; their content has been folded into the
canonical specs above.

| File | Purpose |
|------|---------|
| [claude_plan/README.md](claude_plan/README.md) | Where each proposal landed |
| [claude_plan/RECONCILIATION.md](claude_plan/RECONCILIATION.md) | How the two strategic reviews were reconciled |
| [claude_plan/00_principles.md](claude_plan/00_principles.md) | The lens the proposals were drafted through |
| [claude_plan/accepted/](claude_plan/accepted/) | The 17 accepted proposals (historical record) |

## Reading order

- **New to the codebase?** 00 → 01 → 02 → 13 → 05.
- **Adding a feature?** 15 → the relevant core spec (e.g. 07 for a new
  residual) → 16 → 17 → 19.
- **Debugging a perf issue?** 14 → 10 → the relevant core spec.
- **Writing new code per the strategic plan?** Start at
  [UPDATE_PHASES.md](UPDATE_PHASES.md); follow the phases in order, or
  pick a parallel phase if its `Depends on` declaration is met.
- **Writing new docs?** 13 sets the vocabulary; 17 sets the contract
  discipline; 19 sets the style.

## Rule of thumb

The skeleton has landed: every directory in 01 exists, the layer DAG
is enforced, and ~530 tests are green. Forward work is feature work,
sequenced in [UPDATE_PHASES.md](UPDATE_PHASES.md). The design specs
above are the contract this work must satisfy; 11 is kept only as a
forensic record of what the landing deleted and replaced.

When a newer doc contradicts an older one, the newer doc wins. Open
an issue to bring the older one in line.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for doc-level changes per phase.
