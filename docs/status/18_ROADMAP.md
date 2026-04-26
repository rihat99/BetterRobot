# 18. Roadmap — Planned but Not Yet Implemented

This file tracks features that are specified in the docs but not yet in
the source tree. It is the single place to look when asking *"what is a
stub and what actually runs?"*

Status legend:

- **stub** — file exists; public functions raise `NotImplementedError`.
- **missing** — not present in `src/` at all.
- **partial** — something works, but the spec asks for more.

Last refreshed: 2026-04-25, after the strategic-plan fold-in (proposals
01–17 from `claude_plan/` accepted into the canonical specs;
implementation phases tracked in
[UPDATE_PHASES.md](../UPDATE_PHASES.md)).

---

## 1. Lie types and PyPose retirement (docs [03](../design/03_LIE_AND_SPATIAL.md))

| Symbol | Path | Status |
|--------|------|--------|
| `lie/types.py::SE3 / SO3 / Pose` typed dataclasses | — | **missing** |
| `lie/_torch_native_backend.py` (pure-PyTorch SE3/SO3) | — | **missing** |
| `BR_LIE_BACKEND=torch_native\|pypose` env var | — | **missing** |
| `JacobianStrategy.FINITE_DIFF` opt-in | — | **missing** |
| Default flip: `residual_jacobian` from FD to autodiff | — | **missing — gated on torch backend gradcheck** |
| `pypose` removal from `pyproject.toml` (v1.2) | `pyproject.toml` | **scheduled** |

Phase P10 of [UPDATE_PHASES.md](../UPDATE_PHASES.md) (sub-phases L-A
through L-E) governs the swap.

## 2. Backend abstraction (docs [10](../design/10_BATCHING_AND_BACKENDS.md))

| Symbol | Path | Status |
|--------|------|--------|
| `backends/protocol.py` — `Backend / LieOps / KinematicsOps / DynamicsOps` | — | **missing** |
| `backends/torch_native/` (lie_ops, kinematics_ops, dynamics_ops) | partial scaffold | **stub** |
| `default_backend()` / `get_backend(name)` / `set_backend(name)` | — | **missing** |
| Explicit `backend=` kwargs on `lie.se3.*`, `forward_kinematics`, etc. | — | **missing** |
| `tests/contract/test_backend_boundary.py` | — | **missing** |

Phase P1 of [UPDATE_PHASES.md](../UPDATE_PHASES.md) lands this; it is
the precondition for Phase P10 (PyPose retirement) and the Warp adapter.

## 3. Dynamics bodies (docs [06](../design/06_DYNAMICS.md))

Stubs in `src/better_robot/dynamics/`. The `Data` fields they are meant
to populate (`mass_matrix`, `coriolis_matrix`, `gravity_torque`,
`bias_forces`, `ddq`, `centroidal_momentum_matrix`, …) already exist on
`Data` — only the writers are missing.

| Symbol | Path | Status |
|--------|------|--------|
| `JointModel.joint_bias_acceleration` (default zero) | `data_model/joint_models/base.py` | **missing — required before D2** |
| `JointModel.joint_motion_subspace_derivative` (default zero) | same | **missing — required before D2** |
| `rnea` | `dynamics/rnea.py` | **partial** — `bias_forces` skeleton only |
| `aba`  | `dynamics/aba.py` | **stub** |
| `crba` | `dynamics/crba.py` | **stub** |
| `centroidal` (Ag, hg) | `dynamics/centroidal.py` | **stub** |
| `derivatives` (∂τ/∂q, ∂τ/∂v) | `dynamics/derivatives.py` | **stub** |
| `integrators` (semi-implicit Euler, RK4) | `dynamics/integrators.py` | **stub** |
| `state_manifold` (q ⊕ dv with SE3) | `dynamics/state_manifold.py` | **stub** |
| Three-layer action model (`dynamics/action/`) | folder exists | **stub** |

Sequencing per the dynamics milestone plan (
[06 §5](../design/06_DYNAMICS.md), [UPDATE_PHASES P11](../UPDATE_PHASES.md)):

```
        D2 ──→ D3
       ↗   ↘
 (D1) ↗     ↘
       ↘     D4 ──→ D6 ──→ D7
        ↘          ↗
         D5 ──────
```

D1 (centroidal map close-out) runs **in parallel** with D2 RNEA — RNEA
needs stable Lie/spatial conventions, not a finished centroidal pass.

## 4. Trajectory optimisation + retargeting (docs [08](../design/08_TASKS.md))

| Symbol | Path | Status |
|--------|------|--------|
| `Trajectory` type — accepts `(T, nq)` and `(*B, T, nq)` | `tasks/trajectory.py` | **stub** |
| `Trajectory.with_batch_dims`, `slice`, `resample(sclerp)` | — | **missing** |
| `solve_trajopt` | `tasks/trajopt.py` | **stub** |
| `solve_retarget` | `tasks/retarget.py` | **stub** |
| `TrajectoryParameterization` Protocol | — | **missing** |
| `KnotTrajectory` | — | **missing** |
| `BSplineTrajectory` | — | **missing** |

`solve_ik` (single-frame, multi-target, fixed + floating base) *is*
implemented — including the `lm_then_lbfgs` two-stage composite (which
is being generalised into `MultiStageOptimizer`; see §5).

## 5. Optimisation wiring (docs [07](../design/07_RESIDUALS_COSTS_SOLVERS.md))

| Symbol | Path | Status |
|--------|------|--------|
| `OptimizerConfig.linear_solver` / `.kernel` / `.damping` wired | `tasks/ik.py` | **partial — declared, not all wired** |
| `RobustKernel` Protocol; Identity/Huber/Cauchy/Tukey | `optim/kernels/` | **partial — exist, not used in LM/GN normal eqn** |
| `LinearSolver` Protocol; cholesky/qr/lsqr/cg/block_cholesky | `optim/linear_solvers/` | **missing as Protocol** |
| `DampingStrategy` Protocol | `optim/damping.py` | **missing** |
| `LeastSquaresProblem.gradient(x)` matrix-free path | — | **missing** |
| `Residual.spec()` returning full `ResidualSpec` | — | **missing** |
| `Residual.apply_jac_transpose()` (default + temporal overrides) | partial | **partial — temporal residuals only** |
| `MultiStageOptimizer` + `OptimizerStage` | — | **missing** |
| `LMThenLBFGS` rewritten as MultiStageOptimizer wrapper | exists | **partial** |

## 6. Performance hooks (docs [14](../conventions/14_PERFORMANCE.md))

The spec sets latency/memory budgets and asks for compile boundaries;
these are deferred until after the optim wiring (§5) has been
benchmarked.

| Symbol | Path | Status |
|--------|------|--------|
| `@torch.compile(fullgraph=True)` on FK / Jacobian / CostStack | — | **missing** |
| `@graph_capture` context manager (docs 14 §5) | `backends/warp/graph_capture.py` | **stub** |
| `@cache_kernel` adaptive dispatch (docs 10/14) | — | **missing** |
| Hot-path anti-pattern AST linter (docs 14 §6) | — | **missing — P8** |
| `BR_PROFILE` env hook | — | **missing** |

## 7. Warp backend (docs [10](../design/10_BATCHING_AND_BACKENDS.md))

The backend bridge is stubbed — PyTorch remains the only runtime. A
working port requires the `dlpack` zero-copy transport plus Warp
kernels for FK, spatial algebra, and the collision primitives.

| Symbol | Path | Status |
|--------|------|--------|
| `WarpBridge.to_warp` / `to_torch` | `backends/warp/bridge.py` | **stub** |
| Warp FK / Jacobian kernels | — | **missing** |
| Warp adapter as `Backend` Protocol implementation | — | **missing — gated on P1** |

## 8. Testing — regression oracle (docs [16](../conventions/16_TESTING.md))

| Artifact | Path | Status |
|----------|------|--------|
| `fk_reference.npz` (frozen numerical oracle, fp64) | `tests/kinematics/` | **missing** |
| `_generate_fk_reference.py` regenerator | `tests/kinematics/` | **missing** |
| Pinocchio cross-check `_pinocchio_oracle.npz` (opt-in) | — | **missing** |
| `tests/contract/test_cache_invariants.py` | — | **missing** |
| `tests/contract/test_backend_boundary.py` | — | **missing** |
| `tests/contract/test_optional_imports.py` | — | **missing** |
| `tests/contract/test_shape_annotations.py` (advisory) | — | **missing** |
| `tests/contract/test_no_legacy_strings.py` | — | **missing** |
| `tests/contract/test_submodule_public_imports.py` | — | **missing** |
| Bench baselines `baseline_cpu.json`, `baseline_cuda_l40.json` | `tests/bench/` | **missing** |

The AST-style contract tests (`test_naming`, `test_docstrings`,
`test_protocols`, `test_solver_state`, `test_public_api`) *are* in place.

## 9. Parsers — IO ergonomics (docs [04](../design/04_PARSERS.md))

| Symbol | Path | Status |
|--------|------|--------|
| `IRModel.schema_version` field + `IRSchemaVersionError` | — | **missing** |
| `ModelBuilder.add_revolute_z` etc. named helpers | partial | **stub** |
| `add_joint(kind=)` rejects strings, accepts `JointModel` instance | — | **missing** |
| `AssetResolver` Protocol | `io/assets.py` | **missing** |
| `FilesystemResolver`, `PackageResolver`, `CompositeResolver`, `CachedDownloadResolver` | — | **missing** |
| `parse_urdf(source, resolver=)` | — | **missing — currently hand-rolls path logic** |
| `Model.meta["asset_resolver"]` carrying the parse-time resolver | — | **missing** |

## 10. Public API + typing (docs [01](../design/01_ARCHITECTURE.md), [13](../conventions/13_NAMING.md))

| Symbol | Status |
|--------|--------|
| `__all__` matches **26-symbol** `EXPECTED` set (adds `SE3`, `ModelBuilder`) | **missing — currently 25** |
| Frozen `EXPECTED` test in `tests/contract/test_public_api.py` | **missing — currently asserts `len() == 25`** |
| `_typing.py` shape aliases (`SE3Tensor`, `JointPoseStack`, `ConfigTensor`, …) | partial — **missing the full table** |
| `ReferenceFrame` enum on `kinematics` | **missing — strings today** |
| `KinematicsLevel` enum on `Data` | **missing — int today** |
| `JacobianStrategy.FINITE_DIFF` member | **missing** |
| `Data.require()` / `.invalidate()` / `.joint_pose()` / `.frame_pose()` | **missing** |
| `Data.__setattr__` invalidation on `q` reassignment | **missing** |

## 11. Viewer (docs [12](../design/12_VIEWER.md))

Skeleton + URDF-mesh render modes and the frame-axes overlay are
implemented. The rest of docs 12 (force-vector overlay, contact wrench
overlay, trajectory ghosting, panel presets, recorder export to video)
is still file-scaffolding only — see `viewer/overlays/`,
`viewer/recorder.py`, `viewer/trajectory_player.py`, `viewer/panels.py`.
The URDFMeshMode → AssetResolver wiring lands with §9.

## 12. User docs (docs [10 in claude_plan/](../claude_plan/accepted/10_user_docs_diataxis.md))

| Artifact | Status |
|----------|--------|
| `docs/site/` Sphinx tree | **missing** |
| Diátaxis tutorials (4× in v1) | **missing** |
| Diátaxis how-to guides | **missing** |
| Diátaxis concepts pages | **missing** |
| Auto-generated API reference (autodoc2) | **missing** |
| `[docs]` extra in `pyproject.toml` | **missing — see [20_PACKAGING](../conventions/20_PACKAGING.md)** |

## 13. Style and packaging

| Artifact | Path | Status |
|----------|------|--------|
| Normative coding-style guide | [conventions/19_STYLE.md](../conventions/19_STYLE.md) | **landed (this batch)** |
| Operational packaging spec | [conventions/20_PACKAGING.md](../conventions/20_PACKAGING.md) | **landed (this batch)** |
| `.pre-commit-config.yaml` | repo root | **missing** |
| `.github/workflows/ci.yml` | repo root | **missing** |
| `pyproject.toml` extras taxonomy | repo root | **missing** |
| `RELEASING.md` | repo root | **missing** |
| `src/better_robot/_version.py` | — | **missing** |

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
