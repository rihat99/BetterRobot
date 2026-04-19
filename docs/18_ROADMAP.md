# 18. Roadmap — Planned but Not Yet Implemented

This file tracks features that are specified in the docs but not yet in
the source tree. It is the single place to look when asking *"what is a
stub and what actually runs?"*

Status legend:

- **stub** — file exists; public functions raise `NotImplementedError`.
- **missing** — not present in `src/` at all.
- **partial** — something works, but the spec asks for more.

Last refreshed: 2026-04-19, after Phases 1-7 of the docs-refinement plan
landed (naming rename, exceptions, Protocols, `SolverState`, boundary
guards, `lm_then_lbfgs`, contract tests).

---

## 1. Dynamics bodies (docs [06](06_DYNAMICS.md))

Stubs in `src/better_robot/dynamics/`. The `Data` fields they are meant
to populate (`mass_matrix`, `coriolis_matrix`, `gravity_torque`,
`bias_forces`, `ddq`, `centroidal_momentum_matrix`, …) already exist on
`Data` — only the writers are missing.

| Symbol | Path | Status |
|--------|------|--------|
| `rnea` | `dynamics/rnea.py` | **partial** — `bias_forces` skeleton only |
| `aba`  | `dynamics/aba.py` | **stub** |
| `crba` | `dynamics/crba.py` | **stub** |
| `centroidal` (Ag, hg) | `dynamics/centroidal.py` | **stub** |
| `derivatives` (∂τ/∂q, ∂τ/∂v) | `dynamics/derivatives.py` | **stub** |
| `integrators` (semi-implicit Euler, RK4) | `dynamics/integrators.py` | **stub** |
| `state_manifold` (q ⊕ dv with SE3) | `dynamics/state_manifold.py` | **stub** |
| Three-layer action model (`dynamics/action/`) | folder exists | **stub** |

## 2. Trajectory optimisation + retargeting (docs [08](08_TASKS.md))

| Symbol | Path | Status |
|--------|------|--------|
| `Trajectory` type | `tasks/trajectory.py` | **stub** |
| `solve_trajopt` | `tasks/trajopt.py` | **stub** |
| `solve_retarget` | `tasks/retarget.py` | **stub** |
| B-spline parameterisation | — | **missing** |

`solve_ik` (single-frame, multi-target, fixed + floating base) *is*
implemented — including the `lm_then_lbfgs` two-stage composite from
docs 08 §1.

## 3. Performance hooks (docs [14](14_PERFORMANCE.md))

The spec sets latency/memory budgets and asks for compile boundaries;
these are deferred until after the `SolverState` refactor has been
benchmarked.

| Symbol | Path | Status |
|--------|------|--------|
| `@torch.compile(fullgraph=True)` on FK / Jacobian / CostStack | — | **missing** |
| `@graph_capture` context manager (docs 14 §5) | `backends/warp/graph_capture.py` | **stub** |
| `@cache_kernel` adaptive dispatch (docs 10/14) | — | **missing** |
| Hot-path anti-pattern AST linter (docs 14 §6) | — | **missing** |
| `BR_PROFILE` env hook | — | **missing** |

## 4. Warp backend (docs [10](10_BATCHING_AND_BACKENDS.md))

The backend bridge is stubbed — PyTorch remains the only runtime. A
working port requires the `dlpack` zero-copy transport plus Warp
kernels for FK, spatial algebra, and the collision primitives.

| Symbol | Path | Status |
|--------|------|--------|
| `WarpBridge.to_warp` / `to_torch` | `backends/warp/bridge.py` | **stub** |
| Warp FK / Jacobian kernels | — | **missing** |

## 5. Testing — regression oracle (docs [16](16_TESTING.md))

| Artifact | Path | Status |
|----------|------|--------|
| `fk_reference.npz` (frozen numerical oracle) | `tests/kinematics/` | **missing** |
| Pinocchio cross-check harness (docs 16 §4) | — | **missing** |

The AST-style contract tests (`test_naming`, `test_docstrings`,
`test_protocols`, `test_solver_state`, `test_public_api`) *are* in place.

## 6. Viewer (docs [12](12_VIEWER.md))

Skeleton + URDF-mesh render modes and the frame-axes overlay are
implemented. The rest of docs 12 (force-vector overlay, contact wrench
overlay, trajectory ghosting, panel presets, recorder export to video)
is still file-scaffolding only — see `viewer/overlays/`,
`viewer/recorder.py`, `viewer/trajectory_player.py`, `viewer/panels.py`.

---

## How to close an entry

1. Read the referenced core spec end-to-end.
2. Follow the extension guide in [15_EXTENSION.md](15_EXTENSION.md)
   for the relevant seam (Residual / Optimizer / JointModel / render
   mode / backend).
3. Cover the new code with the matching test tier from
   [16_TESTING.md](16_TESTING.md); add a regression entry to the
   contract suite if the new symbol is part of `__all__`.
4. Move the line from this file to
   [CHANGELOG.md](CHANGELOG.md) under the landing release.
