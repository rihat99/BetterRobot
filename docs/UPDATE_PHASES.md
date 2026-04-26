# Implementation Phases — guide for the code-change agent

> **Audience:** the agent that writes code. **Scope:** translate the
> accepted strategic plan (proposals 01–17, archived under
> `claude_plan/accepted/`) into ordered, verifiable code-change phases.
> **Authority:** non-normative. The canonical specs in `design/` and
> `conventions/` are normative; this file sequences their landing.

Each phase is **independently mergeable**. Phases declare their
dependencies; phases marked *parallel* can land alongside each other.
Within a phase, follow the file list in order; the acceptance criteria
are the merge gate.

If an acceptance check fails, the phase is not done. Do not proceed.
If the spec is unclear, file an issue against the canonical doc rather
than guessing.

---

## Phase ordering (top-level)

```
P0  Preparation — exceptions, enums, typing aliases   ──┐
P1  Backend Protocol + torch_native skeleton          ──┤
P2  Lie types (SE3/SO3/Pose) — additive             ────┤
P3  Cache invariants on Data — additive             ────┤   parallel after P0
P4  Public API audit — 26 symbols, frozen EXPECTED  ────┤
P5  IO ergonomics — schema_version, AssetResolver  ────┘
                  │
                  ▼
P6  Optim wiring (16.A–H) — depends on P0+P1+P3
P7  Trajectory shape lock-in — depends on P0
P8  Quality gates / CI scaffolding — depends on P0
P9  Regression oracle + bench baselines — depends on P1
P10 PyPose retirement (Phase L-A through L-E) — depends on P1+P2
P11 Dynamics milestones D1–D7 — D2 depends on P1+P3 + JointModel hooks
P12 User docs (Diátaxis Sphinx site) — depends on P4
P13 Packaging extras + release — depends on P8
```

The hot path is **P0 → P1 → (P2, P3, P4, P5, P8 in parallel) → P6 → P10 → P11**.
P9, P12, P13 land continuously and don't block the dynamics work.

---

## Phase P0 — Preparation: exceptions, enums, typing aliases

**Goal.** Add the new exception classes, enums, and `_typing.py` shape
aliases that the rest of the phases reference. Pure additive — nothing
breaks.

**References.** [13_NAMING.md §2.8 / §2.9](conventions/13_NAMING.md),
[17_CONTRACTS.md §2](conventions/17_CONTRACTS.md),
[04_typing_shapes_and_enums.md](claude_plan/accepted/04_typing_shapes_and_enums.md).

**Files.**

| File | Change |
|------|--------|
| `src/better_robot/exceptions.py` | Add `IRSchemaVersionError`, `StaleCacheError`, `BackendNotAvailableError` (if missing). Keep all subclassing `BetterRobotError`. |
| `src/better_robot/data_model/__init__.py` | Add `KinematicsLevel` enum (`NONE=0`, `PLACEMENTS=1`, `VELOCITIES=2`, `ACCELERATIONS=3`); subclass `int, Enum`. |
| `src/better_robot/kinematics/__init__.py` | Add `ReferenceFrame` enum (`WORLD`, `LOCAL`, `LOCAL_WORLD_ALIGNED`); subclass `str, Enum`. |
| `src/better_robot/kinematics/jacobian_strategy.py` | Add `FINITE_DIFF` member (the value `"finite_diff"`). |
| `src/better_robot/_typing.py` | Populate the alias table from [13 §2.9](conventions/13_NAMING.md): `SE3Tensor`, `SO3Tensor`, `Quaternion`, `TangentSE3`, `TangentSO3`, `JointPoseStack`, `FramePoseStack`, `ConfigTensor`, `VelocityTensor`, `JointJacobian`, `JointJacobianStack`. Wrap in `if TYPE_CHECKING:` so runtime is free of the `jaxtyping` dep. |
| `pyproject.toml` | Move under `[project.optional-dependencies]`: `jaxtyping>=0.2.30` to `dev`. |

**Acceptance.**

- `from better_robot.exceptions import IRSchemaVersionError, StaleCacheError, BackendNotAvailableError` works.
- `from better_robot.data_model import KinematicsLevel; KinematicsLevel.PLACEMENTS == 1` is True.
- `from better_robot.kinematics import ReferenceFrame; ReferenceFrame.WORLD == "world"` is True (str-subclass).
- `from better_robot.kinematics import JacobianStrategy; JacobianStrategy.FINITE_DIFF` exists.
- Import `better_robot._typing` at runtime does not pull `jaxtyping`.
- `pytest tests/` still green (no behavioural changes).

---

## Phase P1 — Backend Protocol + `torch_native` skeleton

**Goal.** Make the backend dispatch real **today**, while there is still
only one backend. Routes through explicit `Backend` objects;
`default_backend()` is convenience sugar.

**References.** [10_BATCHING_AND_BACKENDS.md §7](design/10_BATCHING_AND_BACKENDS.md),
[02_backend_abstraction.md](claude_plan/accepted/02_backend_abstraction.md).

**Files.**

| File | Change |
|------|--------|
| `src/better_robot/backends/protocol.py` | NEW. `LieOps`, `KinematicsOps`, `DynamicsOps`, `Backend` Protocols. |
| `src/better_robot/backends/__init__.py` | `default_backend()`, `current` (alias), `current_backend()` (returns name), `set_backend(name)`, `get_backend(name)`, `_load(name)`, `_ensure_warp_available()`. |
| `src/better_robot/backends/torch_native/__init__.py` | NEW. `TorchNativeBackend` + `BACKEND` instance. |
| `src/better_robot/backends/torch_native/lie_ops.py` | NEW. Forwards each `LieOps` method to `lie/_pypose_backend` (today). |
| `src/better_robot/backends/torch_native/kinematics_ops.py` | NEW. Forwards `forward_kinematics`, `compute_joint_jacobians` to existing `kinematics/forward.py`, `kinematics/jacobian.py`. |
| `src/better_robot/backends/torch_native/dynamics_ops.py` | NEW. Stubs for `rnea`/`aba`/`crba`/`center_of_mass` that route to existing implementations (or raise). |
| `src/better_robot/lie/se3.py` | Each function gains an optional `backend: Backend \| None = None` kwarg; routes through `(backend or default_backend()).lie.se3_*`. |
| `src/better_robot/lie/so3.py` | Same pattern. |
| `src/better_robot/kinematics/forward.py` | Read `backend or default_backend()`; route through `backend.kinematics.forward_kinematics`; the existing topo-walk in `forward_kinematics_raw` stays the body of the torch-native impl. |
| `src/better_robot/kinematics/jacobian.py` | Same pattern for Jacobian assembly. |
| `tests/contract/test_backend_boundary.py` | NEW. AST-walks `src/`; fails if any module outside `lie/`, `kinematics/`, `dynamics/`, or `backends/<name>/` imports a backend implementation module. Greps for `set_backend(` inside library code (the contract test for "library does not call set_backend internally"). |

**Acceptance.**

- `default_backend().lie.se3_compose(a, b)` returns the same tensor as
  `lie.se3.compose(a, b)` to bit-precision in fp32 and fp64.
- `lie.se3.compose(a, b, backend=get_backend("torch_native"))` works and
  bypasses the global default.
- `set_backend("warp")` raises `BackendNotAvailableError` if warp is not
  installed; `_DEFAULT_NAME` is unchanged on the failure path.
- `tests/contract/test_layer_dependencies.py` and the new
  `test_backend_boundary.py` pass.
- The bench `tests/bench/bench_forward_kinematics.py` does not regress
  beyond the 5% noise floor.

---

## Phase P2 — Lie types (SE3 / SO3 / Pose)

**Goal.** Add the user-facing typed Lie wrappers. Internal hot paths
are unchanged.

**References.** [03_LIE_AND_SPATIAL.md §7](design/03_LIE_AND_SPATIAL.md),
[01_lie_typed_value_classes.md](claude_plan/accepted/01_lie_typed_value_classes.md).

**Files.**

| File | Change |
|------|--------|
| `src/better_robot/lie/types.py` | NEW. Frozen `@dataclass`es `SE3`, `SO3`, `Pose` (Pose is alias of SE3). `.tensor` field; methods delegate to `lie.se3.*` / `lie.so3.*`. Single operator: `__matmul__`. |
| `src/better_robot/lie/__init__.py` | Re-export `SE3`, `SO3`, `Pose` from `lie.types`. |
| `src/better_robot/spatial/__init__.py` | Re-export `SE3`, `SO3`, `Pose` for the spatial-namespace fans. |
| `src/better_robot/data_model/data.py` | Add `joint_pose(joint_id) -> SE3` and `frame_pose(name_or_id) -> SE3` methods (typed accessors; raise `StaleCacheError` if `joint_pose_world is None`). |
| `src/better_robot/data_model/model.py` | Add `body_inertia(body_id) -> Inertia` accessor over `body_inertias`. |
| `src/better_robot/spatial/inertia.py` | `Inertia.se3_action(T)` accepts `SE3` *or* `(...,7)` tensor; returns `Inertia`. Add `from_mass_com_matrix` classmethod. |
| `src/better_robot/spatial/motion.py` | `Motion.se3_action(T)` accepts `SE3` or tensor. |
| `src/better_robot/spatial/force.py` | `Force.se3_action(T)` accepts `SE3` or tensor. **Keep** `cross_motion` raising `NotImplementedError`; upgrade message to point at `Motion.cross_force` and at [03 §7.X](design/03_LIE_AND_SPATIAL.md). |
| `tests/lie/test_types_se3.py` | NEW. Every `SE3` method matches the functional API to `1e-6` (fp32) / `1e-12` (fp64) on randomised batches. `T1 @ T2`, `T @ p` test. `T * 2.0` raises `TypeError`. |
| `tests/lie/test_types_so3.py` | NEW. Mirror. |

**Acceptance.**

- `from better_robot import SE3` works (top-level export).
- `from better_robot.lie import SE3, SO3, Pose` works.
- `from better_robot.spatial import SE3, SO3, Pose` works (re-export).
- `T1 @ T2`, `T @ p`, `T.inverse()`, `T.log()`, `SE3.exp(xi)` all match
  the functional API on random batches.
- `T * 2.0` raises `TypeError`.
- `Force.cross_motion(motion)` raises with the upgraded message.

---

## Phase P3 — Cache invariants on `Data`

**Goal.** Enforce `_kinematics_level`. Reassigning `q` invalidates
caches; functions that need a level call `data.require(level)` and raise
`StaleCacheError`.

**References.** [02_DATA_MODEL.md §3.1](design/02_DATA_MODEL.md),
[07_data_cache_invariants.md](claude_plan/accepted/07_data_cache_invariants.md).

**Files.**

| File | Change |
|------|--------|
| `src/better_robot/data_model/data.py` | Change `_kinematics_level: int` → `_kinematics_level: KinematicsLevel`. Add `require(level)`, `invalidate(level)`, `__setattr__` that fires on `q` / `v` / `a` reassignment. Reset dynamics caches on `q` change; reset velocity-level caches on `v` change; etc. |
| `src/better_robot/kinematics/forward.py` | After populating `joint_pose_world` (and optionally `frame_pose_world`), advance level via `object.__setattr__(data, "_kinematics_level", KinematicsLevel.PLACEMENTS)`. |
| `src/better_robot/kinematics/jacobian.py` | `compute_joint_jacobians`, `get_joint_jacobian`, `get_frame_jacobian` call `data.require(KinematicsLevel.PLACEMENTS)` on entry. |
| `src/better_robot/dynamics/centroidal.py` | `center_of_mass(... v=None)` requires `PLACEMENTS`; with non-None v requires `VELOCITIES`. |
| `tests/contract/test_cache_invariants.py` | NEW. Tests every transition in the matrix from [02 §3.1](design/02_DATA_MODEL.md). Includes the explicit "documented limitation" test: `data.q[..., 0] += 1.0` is *not* detected. |

**Acceptance.**

- `compute_joint_jacobians(model, fresh_data)` raises `StaleCacheError`.
- `forward_kinematics(model, q)` returns `Data` with
  `_kinematics_level == KinematicsLevel.PLACEMENTS`.
- After `data.q = new_q`: `data.joint_pose_world is None` and
  `data._kinematics_level == KinematicsLevel.NONE`.
- `data.joint_pose(0)` on stale data raises `StaleCacheError` with
  message naming `forward_kinematics`.
- `data.invalidate(KinematicsLevel.NONE)` works.
- The "documented limitation" test asserts in-place mutation is *not*
  detected — codifying the contract scope.

---

## Phase P4 — Public API audit (26 symbols, frozen `EXPECTED`)

**Goal.** Replace `assert len(__all__) == 25` with a frozen `EXPECTED`
set; add `SE3` and `ModelBuilder` to `__all__`. Confirm submodule
reachability for everything else documented as public.

**References.** [01_ARCHITECTURE.md §Public API contract](design/01_ARCHITECTURE.md),
[06_public_api_audit.md](claude_plan/accepted/06_public_api_audit.md).

**Files.**

| File | Change |
|------|--------|
| `src/better_robot/__init__.py` | Add `from .lie.types import SE3` and `from .io import ModelBuilder`. Update `__all__` to the 26-symbol set; remove `assert len(__all__) == 25`. |
| `src/better_robot/io/__init__.py` | Re-export `ModelBuilder` (it lives in `io/parsers/programmatic.py` today). |
| `tests/contract/test_public_api.py` | Rewrite: assert `set(better_robot.__all__) == EXPECTED` where `EXPECTED` is the frozen 26-symbol set. Each symbol has a docstring with at least one example. |
| `tests/contract/test_submodule_public_imports.py` | NEW. Asserts the documented submodule paths still resolve. |

**Acceptance.**

- `from better_robot import SE3, ModelBuilder` works.
- `from better_robot.lie import SO3, Pose` works.
- `from better_robot.spatial import Motion, Force, Inertia, Symmetric3` works.
- `from better_robot.tasks.ik import IKResult, IKCostConfig, OptimizerConfig` works.
- `from better_robot.kinematics import ReferenceFrame` works.
- `from better_robot import Symmetric3` raises `ImportError` (Symmetric3 is submodule-only).
- `pytest tests/contract/test_public_api.py` passes.

---

## Phase P5 — IO ergonomics: schema_version, AssetResolver, named helpers

**Goal.** Land `IRModel.schema_version`, the `AssetResolver` Protocol +
concrete resolvers, and the named-helper API on `ModelBuilder`. Fixes
the historical `kind="revolute_z"` confusion.

**References.** [04_PARSERS.md §2.1, §6, §11](design/04_PARSERS.md),
[17_io_ergonomics_and_assets.md](claude_plan/accepted/17_io_ergonomics_and_assets.md).

**Files.**

| File | Change |
|------|--------|
| `src/better_robot/io/ir.py` | Add `schema_version: int = 1` to `IRModel`. Module-level `IRModel.schema_version` (class-attribute) is the version this build expects. |
| `src/better_robot/io/build_model.py` | Validate `ir.schema_version == IRModel.schema_version`; raise `IRSchemaVersionError` otherwise. |
| `src/better_robot/io/assets.py` | NEW. `AssetResolver` Protocol; `FilesystemResolver`, `PackageResolver`, `CompositeResolver`, `CachedDownloadResolver`. |
| `src/better_robot/io/parsers/urdf.py` | Add `resolver: AssetResolver \| None = None` kwarg; default to `FilesystemResolver(base_path=Path(source).parent)`. Set `model.meta["asset_resolver"] = resolver` after parse. |
| `src/better_robot/io/parsers/mjcf.py` | Same kwarg. |
| `src/better_robot/io/parsers/programmatic.py` | Replace stringly-typed `kind` with named helpers: `add_revolute`, `add_revolute_x/y/z`, `add_prismatic`, `add_prismatic_x/y/z`, `add_spherical`, `add_planar`, `add_helical`, `add_free_flyer_root`, `add_fixed`. The catch-all `add_joint(kind=JointModel-instance)` accepts a `JointModel` only; passing a string raises `TypeError` pointing at the named helper. |
| `src/better_robot/viewer/render_modes/urdf_mesh.py` | Read `model.meta["asset_resolver"]` (or accept a `resolver=` kwarg in `URDFMeshMode`). |
| `examples/programmatic_panda.py` | NEW (or update). Use `add_revolute_z` etc. — no string `kind="..."` kwargs anywhere. |
| `tests/io/test_builder_helpers.py` | NEW. Each named helper round-trips through `build_model` and produces the expected joint kind. |
| `tests/io/test_ir_schema_version.py` | NEW. `build_model(IRModel(schema_version=99, ...))` raises `IRSchemaVersionError`. |
| `tests/io/test_asset_resolver.py` | NEW. Filesystem + package + composite cases. |

**Acceptance.**

- `add_joint(kind="revolute_z")` raises a typed error pointing at
  `add_revolute_z`.
- `IRModel.schema_version` exists; `build_model(IRModel(schema_version=99, ...))`
  raises `IRSchemaVersionError`.
- `parse_urdf(source, resolver=None)` defaults to a `FilesystemResolver`
  rooted at `Path(source).parent`.
- `Visualizer` reads `model.meta["asset_resolver"]` to find meshes.
- `examples/programmatic_panda.py` runs cleanly using only named
  helpers.

---

## Phase P6 — Optim wiring (16.A–H)

**Goal.** Wire every `OptimizerConfig` knob; add `ResidualSpec`,
`apply_jac_transpose`, `LeastSquaresProblem.gradient(x)`,
`MultiStageOptimizer`, `TrajectoryParameterization`. Stable collision
residual dim.

**Depends on.** P0 (enums, exceptions), P1 (backend), P3 (cache).

**References.** [07_RESIDUALS_COSTS_SOLVERS.md §4, §7, §8, §9, §10](design/07_RESIDUALS_COSTS_SOLVERS.md),
[16_optim_wiring_and_matrix_free.md](claude_plan/accepted/16_optim_wiring_and_matrix_free.md).

**Files (split into independent sub-PRs).**

### P6.1 — `OptimizerConfig` knob wiring

| File | Change |
|------|--------|
| `src/better_robot/tasks/ik.py` | Honour every documented field. `_make_optimizer`, `_make_linear_solver`, `_make_robust_kernel`, `_make_damping_strategy`. |
| `src/better_robot/optim/optimizers/levenberg_marquardt.py` | Accept `linear_solver`, `robust_kernel`, `damping` keyword args. |
| `src/better_robot/optim/optimizers/gauss_newton.py` | Same. |
| `src/better_robot/optim/optimizers/adam.py` | Read `problem.gradient(x)`; emit `UserWarning` if user set unused knobs (`linear_solver`, etc.). |
| `src/better_robot/optim/optimizers/lbfgs.py` | Same. |
| `tests/optim/test_config_wiring.py` | NEW. `OptimizerConfig(linear_solver="qr")` produces a *different* numerical trajectory than the default; `kernel="huber"` reweights normal eqn. |

### P6.2 — `ResidualSpec` + `apply_jac_transpose`

| File | Change |
|------|--------|
| `src/better_robot/optim/jacobian_spec.py` | Replace existing `ResidualSpec` with the full shape: `output_dim`, `tangent_dim`, `structure`, `time_coupling`, `affected_knots`, `affected_joints`, `affected_frames`, `dynamic_dim`. |
| `src/better_robot/residuals/base.py` | `Residual.spec(state)` default: dense + `output_dim=self.dim`. `Residual.apply_jac_transpose(state, vec)` default: build `J = self.jacobian(state)`; return `J.mT @ vec`. |
| `src/better_robot/residuals/temporal.py` | Override `apply_jac_transpose` for banded structure; `spec` returns `time_coupling="5-point"`. |
| `src/better_robot/residuals/collision.py` | `dim = number_of_candidate_pairs` (stable). `spec` returns `structure="block"`, `dynamic_dim=True`, `affected_joints=...`. |

### P6.3 — `LeastSquaresProblem.gradient(x)`

| File | Change |
|------|--------|
| `src/better_robot/optim/problem.py` | Add `gradient(x)` (matrix-free path, iterates active items, accumulates `apply_jac_transpose`). Add `jacobian_blocks(x)` (block-sparse for trajopt). |
| `tests/optim/test_matrix_free.py` | NEW. `gradient(x)` matches `(J(x).mT @ r(x))` to fp64 ulp on dense residuals. Adam and L-BFGS run on a non-trivial Panda IK without ever calling `problem.jacobian(x)`. |

### P6.4 — `MultiStageOptimizer`

| File | Change |
|------|--------|
| `src/better_robot/optim/optimizers/multi_stage.py` | NEW. `OptimizerStage`, `MultiStageOptimizer`. Snapshots `cost_stack` state; `try/finally` restores even on stage-raise. |
| `src/better_robot/optim/optimizers/lm_then_lbfgs.py` | Rewrite as a thin wrapper: returns `MultiStageOptimizer(stages=(LM, LBFGS))`. Backward-compatible. |
| `tests/optim/test_multi_stage.py` | NEW. Stage-wise weight overrides; restore-on-error. |

### P6.5 — `TrajectoryParameterization`

| File | Change |
|------|--------|
| `src/better_robot/tasks/parameterization.py` | NEW. `TrajectoryParameterization` Protocol; `KnotTrajectory` (identity); `BSplineTrajectory` (cubic, fixed basis). |
| `src/better_robot/tasks/trajopt.py` | Add `parameterization=` kwarg; default `KnotTrajectory()`. |
| `tests/tasks/test_trajopt_param.py` | NEW. Knot vs B-spline reach the same final cost on a small problem; B-spline tangent dim is smaller. |

### P6.6 — Linear solvers / damping / kernel as Protocols

| File | Change |
|------|--------|
| `src/better_robot/optim/linear_solvers/__init__.py` | NEW package: `cholesky.py`, `qr.py`, `lsqr.py`, `cg.py`, `block_cholesky.py`. Each implements `LinearSolver` Protocol. |
| `src/better_robot/optim/damping.py` | NEW. `DampingStrategy` Protocol; `Constant`, `Adaptive`, `TrustRegion`. |
| `src/better_robot/optim/kernels/*.py` | Conform to `RobustKernel` Protocol (`rho`, `weight`). |

**Acceptance.**

- A failing test at the start of the PR (`test_linear_solver_wired`) passes after.
- `kernel="huber"` reweights the LM normal eqn; unit test asserts.
- `weight=0` does **not** structurally remove a residual; `active=False` does.
- `MultiStageOptimizer` restores `CostStack` weights on stage-raise.
- `solve_trajopt(parameterization=BSplineTrajectory(...))` returns a
  `Trajectory` whose final IK-target residual is within tolerance.
- Collision residual dim is stable across an LM run.

---

## Phase P7 — Trajectory shape lock-in

**Goal.** Pin the `Trajectory` dataclass shape; accept both `(T, nq)`
and `(*B, T, nq)`.

**Depends on.** P0.

**References.** [08_TASKS.md §2](design/08_TASKS.md),
[08_trajectory_lock_in.md](claude_plan/accepted/08_trajectory_lock_in.md).

**Files.**

| File | Change |
|------|--------|
| `src/better_robot/tasks/trajectory.py` | Rewrite per [08 §2](design/08_TASKS.md): `t`, `q`, `v`, `a`, `tau`, `extras`, `metadata`. `__post_init__` validates. `batch_shape`, `num_knots`, `with_batch_dims`, `slice`, `resample(kind=)`, `downsample`, `to_data`. Accepts unbatched (T, nq) and batched (*B, T, nq). |
| `tests/tasks/test_trajectory.py` | NEW. Construction tests for both shapes; `batch_shape == ()` for unbatched; `with_batch_dims(1)`; sclerp resample matches `assert_close_manifold`. |

**Acceptance.**

- `Trajectory(t=..., q=...)` accepts `q.shape == (T, nq)` and
  `q.shape == (*B, T, nq)`; mismatched shapes raise `ShapeError`.
- `traj.batch_shape == ()` for unbatched.
- `traj.with_batch_dims(1)` returns a normalised view.
- `traj.resample(t_new, kind='sclerp')` is manifold-correct.

---

## Phase P8 — Quality gates / CI scaffolding

**Goal.** Land the contract bundle, CI workflow, pre-commit, and
advisory-then-blocking ladder.

**Depends on.** P0.

**References.** [16_TESTING.md](conventions/16_TESTING.md),
[11_quality_gates_ci.md](claude_plan/accepted/11_quality_gates_ci.md).

**Files.**

| File | Change |
|------|--------|
| `pyproject.toml` | `[tool.ruff]`, `[tool.mypy]`, `[tool.coverage]` config. |
| `.pre-commit-config.yaml` | NEW. ruff format + check + fast contract subset (`test_layer_dependencies`, `test_naming`, `test_public_api`). |
| `.github/workflows/ci.yml` | NEW. Jobs: `contract` (blocks all), `unit` (3.10/3.11/3.12), `cuda` (gpu-runner), `bench-cpu-advisory` (`continue-on-error: true`), `bench-baseline-nightly` (schedule), `docs`, `type-check`. |
| `tests/contract/test_hot_path_lint.py` | NEW. AST walks `kinematics/`, `dynamics/`, `optim/optimizers/`. Forbids `.item()`, `.cpu()`, `torch.zeros` in loops, branching on `tensor.dim()`. Suppress with `# bench-ok: <reason>`. |
| `tests/contract/test_optional_imports.py` | NEW. Forbidden imports per [20 §1.2](conventions/20_PACKAGING.md). |
| `tests/contract/test_no_legacy_strings.py` | NEW. Greps `src/` for `reference="(world\|local\|local_world_aligned)"`. |
| `tests/contract/test_shape_annotations.py` | NEW. AST walk over public symbols; advisory mode (prints coverage; does not fail). |
| `tests/contract/test_deprecations.py` | NEW. Stub for the version-gated removal tests. |
| `RELEASING.md` | NEW. The release process from [20 §5](conventions/20_PACKAGING.md). |

**Acceptance.**

- `tests/contract/` runs in < 60 seconds and includes every test in
  [16 §2 directory layout](conventions/16_TESTING.md).
- `.github/workflows/ci.yml` exists; `contract` blocks all other jobs.
- `pre-commit install` produces a working hook.
- `BR_STRICT=1` is set in the contract job; a deprecated alias call
  fails CI.
- The bench-cpu-advisory job records numbers but does not fail PRs.

---

## Phase P9 — Regression oracle + bench baselines

**Goal.** Land `fk_reference.npz` and the bench baselines.

**Depends on.** P1.

**References.** [16_TESTING.md §4.5](conventions/16_TESTING.md),
[12_regression_and_benchmarks.md](claude_plan/accepted/12_regression_and_benchmarks.md).

**Files.**

| File | Change |
|------|--------|
| `tests/kinematics/_generate_fk_reference.py` | NEW. Generates `fk_reference.npz` for Panda + G1 at fixed seed in fp64; metadata: `oracle_version`, `generation_seed`, `generated_with`, `fk_dtype`, `generated_at` (no SHA). |
| `tests/kinematics/fk_reference.npz` | NEW. The committed reference data. < 200 KB total. |
| `tests/kinematics/test_fk_regression.py` | NEW. Compares current FK vs the reference at `atol=1e-10` (fp64). |
| `tests/test_pinocchio/test_*.py` | NEW. **Live** Pinocchio cross-check (replaces the original `_pinocchio_oracle.npz` design — see acceptance note below). Each test calls `pytest.importorskip("pinocchio")` and compares BetterRobot output to a freshly-computed Pinocchio result. |
| `tests/bench/conftest.py` | Fixtures: `panda`, `g1`, `panda_data`. |
| `tests/bench/bench_lie.py`, `bench_forward_kinematics.py`, `bench_jacobian.py`, `bench_solve_ik.py` | NEW. |
| `tests/bench/baseline_cpu.json` | NEW. Initial baseline from one CI run. |
| `tests/bench/baseline_cuda_l40.json` | NEW. Same on the self-hosted GPU runner. |
| `tests/bench/README.md` | NEW. The bump procedure from [12 §12.D](claude_plan/accepted/12_regression_and_benchmarks.md). |
| `tests/bench/test_mem_watermark.py` | NEW. Nightly only. |

**Acceptance.**

- `fk_reference.npz` exists, < 200 KB.
- `test_fk_regression.py` passes.
- `_generate_fk_reference.py` reproduces the same file given the same
  seed and pinned versions.
- CI runs `pytest tests/bench/ --benchmark-compare` against the
  committed baseline (advisory).

> **Cross-check decision (2026-04-26).** The Pinocchio cross-check
> shipped as a *live* import-or-skip suite under `tests/test_pinocchio/`
> rather than a frozen `_pinocchio_oracle.npz`. The live form
> exercises whatever q's the test author samples (not just the seed
> baked into the npz) and the CI `[test]` extra ships `pin>=2.7`, so
> the runner always has Pinocchio. Drop the npz line from the
> roadmap; do not add the file.

---

## Phase P10 — PyPose retirement (Phase L-A through L-E)

**Goal.** Replace PyPose with a pure-PyTorch SE3/SO3 backend.

**Depends on.** P1 (backend Protocol), P2 (typed Lie wrappers — for
testing parity).

**References.** [03_LIE_AND_SPATIAL.md §10](design/03_LIE_AND_SPATIAL.md),
[03_replace_pypose.md](claude_plan/accepted/03_replace_pypose.md),
[status/PYPOSE_ISSUES.md](status/PYPOSE_ISSUES.md).

**Sub-phases (each is a release):**

### P10-A — write the torch backend

| File | Change |
|------|--------|
| `src/better_robot/lie/_torch_native_backend.py` | NEW. ~250 LOC of pure PyTorch. SE3/SO3 compose/inverse/log/exp/act/adjoint/normalize. Use the formulas in [03 §6](design/03_LIE_AND_SPATIAL.md) — `V(ω) = I + b·W + c·W²` with `b = (1−cos θ)/θ²`, `c = (θ−sin θ)/θ³`; Taylor `b ≈ 1/2 − θ²/24`, `c ≈ 1/6 − θ²/120`. **Do not transcribe; re-derive.** |
| `src/better_robot/backends/torch_native/lie_ops.py` | Add `BR_LIE_BACKEND` env-var dispatch: route to `_torch_native_backend` (default) or `_pypose_backend` (legacy). |
| `tests/lie/test_torch_backend_gradcheck.py` | NEW. fp64 gradcheck on `se3_log/exp/compose/inverse/act`. atol=1e-8, rtol=1e-6. |
| `tests/lie/test_torch_backend_fd_parity.py` | NEW. Central-FD parity. |
| `tests/lie/test_torch_backend_value_parity.py` | NEW. Forward-pass exact-equality with the legacy backend on randomised inputs. |
| `tests/lie/test_singularities.py` | NEW. `θ ∈ {0, π/2, π−1e-6}` for log/exp on both SE3 and SO3. |

### P10-B — CI runs both backends

| File | Change |
|------|--------|
| `.github/workflows/ci.yml` | Add a `unit-pypose` job that sets `BR_LIE_BACKEND=pypose`. Both must be green. |

### P10-C — flip the default

| File | Change |
|------|--------|
| `src/better_robot/backends/torch_native/lie_ops.py` | Default branch flips. |
| `src/better_robot/kinematics/jacobian.py::residual_jacobian` | Default flips from `JacobianStrategy.FINITE_DIFF` to `JacobianStrategy.AUTODIFF`. |
| `src/better_robot/optim/optimizers/adam.py` | Comment updated; both `loss.backward()` and `problem.gradient(x)` paths supported. |

### P10-D — drop PyPose

| File | Change |
|------|--------|
| `pyproject.toml` | Remove `pypose>=0.6,<0.8` from `dependencies`. |
| `src/better_robot/lie/_pypose_backend.py` | Deleted. |
| `tests/contract/test_layer_dependencies.py` | Remove the `_pypose_backend.py` exception. |

### P10-E — docs cleanup

Update CLAUDE.md notes; update `docs/status/PYPOSE_ISSUES.md` (which
becomes a historical document); remove the FD note from
`docs/conventions/14_PERFORMANCE.md §2.3` (or change it to "FD opt-in").

**Acceptance.**

- `lie/_torch_native_backend.py` passes fp64 gradcheck.
- Forward-pass parity within `1e-6` (fp32) / `1e-12` (fp64) on sizes 1, 32, 1024.
- The full test suite (`uv run pytest tests/ -v`) passes with both
  `BR_LIE_BACKEND=pypose` and `BR_LIE_BACKEND=torch_native` set.
- `tests/bench/bench_lie.py` shows the torch backend ≤ 1.5× PyPose on CPU,
  ≤ 1.2× on CUDA.
- After L-D: `import pypose` not present anywhere; `pyproject.toml`
  no longer lists `pypose`.

---

## Phase P11 — Dynamics milestones D1–D7

**Goal.** Body the dynamics layer per [06_DYNAMICS.md §5](design/06_DYNAMICS.md).

**Depends on.** P1 (backend), P3 (cache), P0 (enums). D2 specifically
depends on the `JointModel` dynamics hooks landing first.

**References.** [06_DYNAMICS.md](design/06_DYNAMICS.md),
[14_dynamics_milestone_plan.md](claude_plan/accepted/14_dynamics_milestone_plan.md).

### P11-pre — `JointModel` dynamics hooks

| File | Change |
|------|--------|
| `src/better_robot/data_model/joint_models/base.py` | Add `joint_bias_acceleration(q,v) -> Tensor[B...,6]` and `joint_motion_subspace_derivative(q,v) -> Tensor[B...,6,nv_j]` to the Protocol. Default impls return zeros. |
| Each concrete `JointModel` subclass | Inherits zero defaults (correct for revolute/prismatic/free-flyer). |

### P11-D1 — `compute_centroidal_map`, `compute_centroidal_momentum`

Wire centroidal map; close out D1. Runs **in parallel with D2**.

### P11-D2 — RNEA

`dynamics/rnea.py`. Two-pass Featherstone using
`spatial.Motion`/`Force`/`Inertia`. Acceptance: matches Pinocchio at fp64
on Panda + G1; satisfies `bias_forces == rnea(q,v,zeros)`.

### P11-D3 — CRBA

`dynamics/crba.py`. Backward pass over composite-rigid-body inertias.
Acceptance: `M(q) @ a + bias == rnea(q,v,a)` to fp64 ulp; SPD on neutral.

### P11-D4 — ABA

`dynamics/aba.py`. Articulated Body Algorithm. Acceptance:
`aba(q, v, rnea(q,v,a)) == a` to fp64 ulp.

### P11-D5 — centroidal (full)

Already partially landed in D1; complete `ccrba`.

### P11-D6 — derivatives

Carpentier-Mansard analytic derivatives wrapped in
`torch.autograd.Function.apply`. Acceptance: `gradcheck(rnea)` at fp64.

### P11-D7 — three-layer action model

`dynamics/action/{differential,integrated,action}.py` per
[06 §6](design/06_DYNAMICS.md). Acceptance: toy pendulum DDP/iLQR closes.

---

## Phase P12 — User docs (Diátaxis Sphinx site)

**Goal.** Stand up `docs/site/` with Sphinx + MyST + Diátaxis.

**Depends on.** P4 (public API frozen), P10 (Lie backend stable so
notebooks don't bit-rot).

**References.** [10_user_docs_diataxis.md](claude_plan/accepted/10_user_docs_diataxis.md).

**Files.**

| Path | Notes |
|------|-------|
| `docs/site/conf.py` | NEW. Sphinx config. Furo theme; MyST + MyST-NB; sphinx-design; autodoc2. |
| `docs/site/index.md` | Landing page: 5-line snippet + grid. |
| `docs/site/tutorials/{01-04}.md` | NEW. Install+FK / Basic IK / Floating-base IK / Custom residual. MyST-NB notebooks. |
| `docs/site/guides/*.md` | How-to guides. |
| `docs/site/concepts/*.md` | Concepts pages. |
| `docs/site/reference/api/index.md` | autodoc2 reference. |
| `docs/site/reference/changelog.md` | links to ../CHANGELOG.md. |
| `pyproject.toml` | Confirm `[docs]` extra. |
| `.github/workflows/ci.yml` | `docs` job runs `sphinx-build -W`. |

**Acceptance.**

- `sphinx-build -W -b html docs/site/ build/html/` succeeds without warnings.
- The four v1 tutorials run end-to-end in CI under MyST-NB.
- The auto-generated API reference contains every symbol in
  `better_robot.__all__` and links to its source.

---

## Phase P13 — Packaging extras + release

**Goal.** Implement the extras in [20_PACKAGING.md](conventions/20_PACKAGING.md);
ship v0.2.0.

**Depends on.** P8 (CI for `test_optional_imports.py`).

**Files.**

| File | Change |
|------|--------|
| `pyproject.toml` | Implement the full extras taxonomy. |
| `src/better_robot/_version.py` | NEW. `__version__ = "0.2.0"`. |
| `src/better_robot/__init__.py` | Re-export `__version__`. |
| `RELEASING.md` | The procedure from [20 §5](conventions/20_PACKAGING.md). |
| `tests/contract/test_optional_imports.py` | Already added in P8 — confirm passing. |

**Acceptance.**

- `pip install better-robot` installs only `torch`, `numpy`, `pypose`
  (until P10 lands), `rich`.
- `import better_robot; better_robot.load("foo.urdf")` without `[urdf]`
  raises `BackendNotAvailableError` whose message names
  `pip install better-robot[urdf]`.

---

## Continuous tasks (not phase-gated)

- **Naming sweep.** Whenever a file is touched, replace the legacy
  identifiers `oMi`, `oMf`, `liMi`, `nle`, `Ag`, `hg` per
  [13 §4](conventions/13_NAMING.md). The shims on `Data` keep working;
  internal code must use the new names. Removed in v1.1.
- **Rename the `Motion.data` / `Force.data` / `Inertia.data` fields to
  `.tensor`.** Low priority; not load-bearing. The new `SE3` / `SO3` /
  `Pose` classes already use `.tensor`.
- **Documentation hygiene.** Each PR that changes a public symbol
  updates the relevant doc in `docs/design/` or `docs/conventions/`. If
  the change is foundational, it lives in this doc as a phase update.
- **Roadmap maintenance.** Each closed work item moves from
  [status/18_ROADMAP.md](status/18_ROADMAP.md) to
  [CHANGELOG.md](CHANGELOG.md) under the landing release.

---

## Definition of done for v1.0

When the following are *all* true:

- Phases P0–P11 closed.
- `__all__` matches the frozen 26-symbol `EXPECTED` set.
- All contract tests green: layer DAG, public API, submodule
  reachability, naming, hot-path lint, backend boundary, optional
  imports, cache invariants, deprecations.
- Bench gates promoted to blocking (per the gate-promotion ladder in
  [16 §4.6](conventions/16_TESTING.md)).
- FK and dynamics regression oracles green at fp64.
- Pinocchio cross-check green.
- `examples/*.py` all run end-to-end and are imported by
  `tests/examples/test_examples.py`.
- `pip install better-robot` is a small core; extras bound the
  format/visualisation/Warp/docs/test deps.
- `pypose` is removed from `pyproject.toml`.
- The Sphinx site at `docs/site/` is shipped.
- `RELEASING.md` is followed for the v1.0 tag.

The release readme lists exactly which proposals
(`docs/claude_plan/accepted/`) the v1.0 cycle delivered.
