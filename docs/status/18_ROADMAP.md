# 18. Roadmap — Planned but Not Yet Implemented

This file tracks features that are specified in the docs but not yet in
the source tree. It is the single place to look when asking *"what is a
stub and what actually runs?"*

Status legend:

- **stub** — file exists; public functions raise `NotImplementedError`.
- **missing** — not present in `src/` at all.
- **partial** — something works, but the spec asks for more.

Last refreshed: 2026-04-26, after a wholesale audit against the source.
The closed work moved to [CHANGELOG.md](../CHANGELOG.md) under the
"Implementation phases P0–P11 partial close-out" entry. Implementation
phases continue to be tracked in
[UPDATE_PHASES.md](../UPDATE_PHASES.md).

---

## 1. Dynamics — remaining stubs (docs [06](../design/06_DYNAMICS.md))

The recursive Featherstone passes (`rnea`, `aba`, `crba`, `ccrba`,
`compute_centroidal_map/momentum`, `center_of_mass`) are implemented.
The `JointModel.joint_bias_acceleration` /
`joint_motion_subspace_derivative` hooks ship default-zero
implementations. The remaining gaps:

| Symbol | Path | Status |
|--------|------|--------|
| `semi_implicit_euler` | `dynamics/integrators.py` | **stub — D4+** |
| `symplectic_euler` | `dynamics/integrators.py` | **stub — D4+** |
| `rk4` | `dynamics/integrators.py` | **stub — D4+** |
| `compute_minverse(model, data, q)` | `dynamics/crba.py` | **stub — D4 (direct ABA factorisation)** |
| `compute_coriolis_matrix(model, data, q, v)` | `dynamics/rnea.py` | **stub — separate world-frame recursion** |
| `compute_centroidal_dynamics_derivatives` | `dynamics/derivatives.py` | **stub — analytic form deferred; autograd path documented** |
| Analytic Carpentier–Mansard `compute_*_derivatives` | `dynamics/derivatives.py` | **partial — uses `torch.autograd.functional.jacobian`; spec asks for analytic recursion** |

## 2. Trajectory optimisation + retargeting (docs [08](../design/08_TASKS.md))

`solve_ik`, `solve_trajopt` (with `KnotTrajectory` + `BSplineTrajectory`
parameterisations), and the full `Trajectory` shape API are all live.
The remaining gap is retargeting:

| Symbol | Path | Status |
|--------|------|--------|
| `solve_retarget` | `tasks/retarget.py` | **stub** |

## 3. Residuals — remaining stubs (docs [07](../design/07_RESIDUALS_COSTS_SOLVERS.md))

`PoseResidual`, `PositionResidual`, `OrientationResidual`,
`JointPositionLimit`, `RestResidual`, `ReferenceTrajectoryResidual`,
`ContactConsistencyResidual`, `TimeIndexedResidual`,
`VelocityResidual`, and `AccelerationResidual` (with analytic Jacobians
*and* `apply_jac_transpose` overrides) are live. The remaining stubs:

| Symbol | Path | Status |
|--------|------|--------|
| `JerkResidual` | `residuals/smoothness.py` | **stub — third-derivative smoothness; v1 spec marks it as not required** |
| `YoshikawaResidual` | `residuals/manipulability.py` | **stub** |
| `SelfCollisionResidual` | `residuals/collision.py` | **stub — depends on §5 below** |
| `WorldCollisionResidual` | `residuals/collision.py` | **stub — depends on §5** |
| `JointVelocityLimit.jacobian` | `residuals/limits.py` | **partial — `__call__` works, `.jacobian` stubbed** |
| `JointAccelLimit` | `residuals/limits.py` | **stub** |
| `NullspaceResidual` | `residuals/regularization.py` | **stub** |

## 4. Performance hooks (docs [14](../conventions/14_PERFORMANCE.md))

Spec sets latency/memory budgets and asks for compile boundaries; not
applied to source today. The `tests/contract/test_hot_path_lint.py`
linter that prevents `.item()` / in-loop allocs / rank-branching *is* in
place.

| Symbol | Path | Status |
|--------|------|--------|
| `@torch.compile(fullgraph=True)` on FK / Jacobian / CostStack | — | **missing** |
| `@graph_capture` context manager (docs 14 §5) | `backends/__init__.py` (single seam; no-op under torch_native, real CUDA-graph capture ships with the Warp backend) | **stub — body lands with the Warp backend** |
| `@cache_kernel` adaptive dispatch (docs 10/14) | — | **missing** |
| `BR_PROFILE` env hook | — | **missing** |

## 5. Collision geometry (docs [09](../design/09_COLLISION_GEOMETRY.md))

The capsule/sphere/box primitives, `RobotCollision`, and pair iteration
are partially in place; the residual side is stubbed.

| Symbol | Path | Status |
|--------|------|--------|
| `SelfCollisionResidual.__call__` / `.jacobian` | `residuals/collision.py` | **stub** |
| `WorldCollisionResidual.__call__` / `.jacobian` | `residuals/collision.py` | **stub** |
| Sparse pair-Jacobian assembly | `collision/closest_pts.py` | **partial** |

## 6. Warp backend (docs [10](../design/10_BATCHING_AND_BACKENDS.md))

Backend Protocol + torch_native default are live; Warp remains stubbed.

| Symbol | Path | Status |
|--------|------|--------|
| `WarpBridge.to_warp` / `.to_torch` | `backends/warp/bridge.py` | **stub** |
| `enable_warp_backend` / `disable_warp_backend` | `backends/warp/__init__.py` | **stub** |
| Warp FK / Jacobian / RNEA / ABA / CRBA kernels | `backends/warp/kernels/` | **missing** |
| `[warp]` extra in `pyproject.toml` | `pyproject.toml` | **missing — see §10 below** |

## 7. Testing — regression oracle (docs [16](../conventions/16_TESTING.md))

The contract suite + AST linters are in place. The bench baselines and
pinocchio oracle still need bodies:

| Artifact | Path | Status |
|----------|------|--------|
| `tests/bench/baseline_cpu.json` | `tests/bench/` | **placeholder — `{"_schema_version": 1}` only; populate from one CI run** |
| `tests/bench/baseline_cuda_l40.json` | `tests/bench/` | **placeholder — same** |
| `tests/kinematics/_pinocchio_oracle.npz` (frozen, opt-in) | — | **resolved 2026-04-26: superseded by the live `pytest.importorskip("pinocchio")` cross-check in `tests/test_pinocchio/`. Live coverage exercises every sampled q rather than one frozen set; the CI matrix runs `[test]` extras so Pinocchio is always present. The frozen-oracle entry in [UPDATE_PHASES P9](../UPDATE_PHASES.md) is retired.** |
| `tests/contract/test_semver_compat.py` (release tripwire) | `tests/contract/` | **missing — see [20_PACKAGING §3](../conventions/20_PACKAGING.md)** |

## 8. Viewer (docs [12](../design/12_VIEWER.md))

V1 ships `SkeletonMode`, `URDFMeshMode`, `CollisionMode` (stub),
`GridOverlay`, `FrameAxesOverlay`, `TargetsOverlay`,
`ForceVectorsOverlay`, `ViserBackend`, `MockBackend`,
`build_joint_panel`, and a minimal `TrajectoryPlayer.show_frame/.play`.
The remaining gaps:

| Symbol | Path | Status |
|--------|------|--------|
| `ComOverlay` | `viewer/overlays/com.py` | **stub** |
| `PathTraceOverlay` | `viewer/overlays/path_trace.py` | **stub** |
| `ResidualPlotOverlay` | `viewer/overlays/residual_plot.py` | **stub** |
| Contact-wrench overlay | — | **missing** |
| Trajectory ghosting / panel presets | `viewer/panels.py` | **missing — only `build_joint_panel` lives there today** |
| `VideoRecorder` / `render_trajectory` | `viewer/recorder.py` | **stub** |
| `OffscreenBackend` (real body) | `viewer/renderers/offscreen_backend.py` | **stub** |
| `TrajectoryPlayer.seek/.step/.pause/.set_speed/.set_loop/.set_ghost/.set_trace/.set_batch_index` | `viewer/trajectory_player.py` | **stub** |
| `URDFMeshMode` reads `model.meta["asset_resolver"]` for mesh paths | `viewer/render_modes/urdf_mesh.py` | **closed 2026-04-26: `_load_geom(geom, resolver=...)` routes mesh URIs through the active resolver; `URDFMeshMode(resolver=...)` takes an explicit override; falls back to the raw `geom.params["path"]` when the resolver fails** |

## 9. User docs

The Sphinx tree exists at `docs/site/` (4 tutorials, 4 guides, 3
concepts, reference/api index). The remaining bumps:

| Artifact | Status |
|----------|--------|
| `autodoc2`-generated API reference (currently uses `sphinx.ext.autodoc`) | **missing — `[docs]` extra also needs `autodoc2>=0.5`** |
| MyST-NB executable tutorials | **missing — `[docs]` extra needs `myst-nb>=1.1`** |

## 10. Packaging — extras taxonomy

(closed 2026-04-26: `pyproject.toml` now declares every extra the spec
in [20](../conventions/20_PACKAGING.md) requires — `[urdf]`, `[mjcf]`,
`[viewer]`, `[geometry]`, `[examples]`, `[warp]`, `[docs]` (with
`sphinx-autodoc2` and `myst-nb`), `[bench]`, `[test]` (with
`pytest-cov`, `pytest-xdist`, `hypothesis`), `[dev]` (with `mypy`,
`pre-commit`), and `[all]` (which now includes `bench` and `warp`).)

## 11. Test layout

(closed 2026-04-26: both files moved into `tests/contract/`.)

---

## How to close an entry

1. Read the referenced canonical doc end-to-end. Each entry above links
   either to a `docs/design/`, `docs/conventions/`, or
   `docs/UPDATE_PHASES.md` section.
2. Follow the extension guide in
   [15_EXTENSION.md](../conventions/15_EXTENSION.md) for the relevant
   seam.
3. Cover the new code with the matching test tier from
   [16_TESTING.md](../conventions/16_TESTING.md); add a regression entry
   to the contract suite if the new symbol is part of `__all__` or a
   documented Protocol.
4. Move the line from this file to
   [CHANGELOG.md](../CHANGELOG.md) under the landing release.
