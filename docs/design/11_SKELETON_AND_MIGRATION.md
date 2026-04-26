# Skeleton landing — historical record

The skeleton has landed. Every directory in
[01_ARCHITECTURE.md §Target directory layout](01_ARCHITECTURE.md) exists
in `src/better_robot/`, every joint family in `data_model/joint_models/`
has a file, and the test suite (~530 tests) is green. Forward work —
backend Protocol, typed Lie value classes, matrix-free trajopt, dynamics
bodies, PyPose retirement — is sequenced in
[UPDATE_PHASES.md](../UPDATE_PHASES.md).

This document is kept as a historical record so that later readers can
understand what was deleted, what survived, and the shape of the
landing. If you are looking for *what to do next*, read
[UPDATE_PHASES.md](../UPDATE_PHASES.md), not this file.

## 1. What the pre-skeleton tree looked like

Before the skeleton landed, `src/better_robot/` had this layout:

```
src/better_robot/
├── math/               # SE3/SO3/spatial wrappers around pypose
├── models/             # RobotModel / RobotData / JointInfo / LinkInfo
│   └── parsers/        # URDF parser
├── algorithms/
│   ├── kinematics/     # _fk_impl, jacobian
│   └── geometry/       # collision primitives + distance
├── costs/              # pose, limits, rest, collision, manipulability
├── solvers/            # LM/GN/Adam/LBFGS + Problem + SOLVERS registry
├── tasks/
│   ├── ik/             # solve_ik, IKConfig, IKVariable
│   ├── trajopt/        # stub
│   └── retarget/       # stub
└── viewer/             # viser visualiser
```

This is gone. The new tree
([01_ARCHITECTURE.md](01_ARCHITECTURE.md)) replaced it in place — there
was no parallel `better_robot_v2/` tree; the rewrite landed directly.

## 2. What was deleted

Symbols and modules that were removed in the landing:

- `RobotModel._frozen` custom immutability guard → `@dataclass(frozen=True)`.
- `RobotData` → `data_model/data.py`.
- `IKVariable` (flat-packing helper) → `q` already contains the free-flyer DOFs.
- `solvers/registry.py` (`SOLVERS` global) → optimisers are plain classes
  matching the `Optimizer` Protocol.
- `solvers/levenberg_marquardt_pypose.py` → benchmark-only; removed when
  PyPose became a backend rather than a peer dependency in the solver path.
- `tasks/ik/variable.py`, `tasks/ik/config.py` (the original files) →
  replaced by `tasks/ik.py` + `IKCostConfig` + `OptimizerConfig`.
- `_solve_floating_autodiff` / `_solve_floating_analytic` /
  `_analytic_collision_jacobian` triplet → folded into one `solve_ik`
  driven by `JointFreeFlyer` topology.
- `algorithms/kinematics/jacobian.py::compute_jacobian`, `limit_jacobian`,
  `rest_jacobian` → preserved as `.jacobian()` methods on the respective
  residual classes.

## 3. What survived (with a new home)

- Forward-kinematics traversal logic — moved into
  `kinematics/forward.py` (`forward_kinematics_raw` /
  `forward_kinematics`), now driven by `model.topo_order` and `JointModel`.
- SE3 and spatial functions in `math/` — moved to `lie/se3.py`, `lie/so3.py`,
  `spatial/`.
- Capsule-based collision primitives and SDF functions — moved to
  `collision/`, registered against the dispatch table.
- The viser visualiser — moved to `viewer/`.
- LM / GN / Adam / LBFGS solver loops (the mechanics, not the interfaces)
  — moved to `optim/optimizers/`.
- Tests that didn't bind to private attributes (e.g.
  higher-level tests in `test_ik.py`, `test_jacobian.py`).

## 4. Naming-rename alongside the landing

The Pinocchio-cryptic field names (`oMi`, `oMf`, `liMi`, `nle`, `Ag`)
were renamed at the same time. Sequence:

1. New names landed on `Data` (`joint_pose_world`, `frame_pose_world`,
   `joint_pose_local`, `bias_forces`, `centroidal_momentum_matrix`).
2. All internal readers were migrated to the new names.
3. The old names live on `Data` as `@property` shims that emit
   `DeprecationWarning` (see
   [02_DATA_MODEL.md §11](02_DATA_MODEL.md)).
4. `tests/contract/test_naming.py` enforces that no `src/` file uses
   the cryptic names internally.
5. The shims are scheduled for removal in v1.1.

## 5. Acceptance criteria for v1 release

These criteria translate the skeleton-era goals into the v1 release
checklist. Items that have already landed are marked **[done]**; items
that remain are tracked in [UPDATE_PHASES.md](../UPDATE_PHASES.md) and
[18_ROADMAP.md](../status/18_ROADMAP.md).

- [done] Skeleton tree present at `src/better_robot/`; layer DAG enforced.
- [done] `JointFreeFlyer` works end-to-end; G1 floating-base IK runs
  through the same `solve_ik` path as Panda.
- [ ] `better_robot.__all__` matches the frozen `EXPECTED` set
      (currently 25; targets 26 after `SE3` and `ModelBuilder` ship —
      see [01_ARCHITECTURE.md](01_ARCHITECTURE.md)).
- [ ] Full contract test bundle green: layer DAG, public API, submodule
      reachability, naming, hot-path lint, backend boundary, optional
      imports, docstrings, cache invariants, shape annotations
      advisory. See [16_TESTING.md §5](../conventions/16_TESTING.md).
- [ ] CPU benchmark budgets met (advisory until two release cycles of
      stable runner variance below 5% noise floor); CUDA budgets met
      nightly. See
      [14_PERFORMANCE.md §1](../conventions/14_PERFORMANCE.md) and
      [16_TESTING.md gate-promotion ladder](../conventions/16_TESTING.md).
- [ ] `tests/tasks/test_ik_regression.py` — Panda + G1 IK outputs
      within `1e-4` of the pinned regression oracle on a 50-sample
      benchmark.
- [ ] `tests/kinematics/fk_reference.npz` exists; FK regression test
      catches numerical drift at fp64 ulp.
- [ ] All `examples/*.py` run end-to-end and are imported by
      `tests/examples/test_examples.py`.
- [ ] `forward_kinematics` on CUDA matches CPU to `1e-5`.
- [ ] `OptimizerConfig.linear_solver` / `.kernel` / `.damping` are
      honoured and exercise different numerical trajectories — wired
      knob test passes (P6).
- [ ] `LeastSquaresProblem.gradient(x)` matches `J(x).T @ r(x)` to fp64
      ulp on dense residuals; Adam and L-BFGS run via the matrix-free
      path (P6.4).
- [ ] Every dynamics symbol exists and either implements its body or
      raises `NotImplementedError` with a pointer at `06_DYNAMICS.md`
      (P11).
- [done] No pinocchio-cryptic identifiers in new code (enforced by
      `tests/contract/test_naming.py`); deprecated shims present only
      on `Data` and carry `DeprecationWarning`.
- [ ] `pip install better-robot` does not pull `yourdfpy`, `mujoco`,
      `viser`, `warp`, `sphinx`, `robot_descriptions`, or `pinocchio`
      (P13; `tests/contract/test_optional_imports.py`).

## 6. Out-of-scope for v1

Same as before the landing — the boundaries didn't move:

- Optimal control solvers (DDP/iLQR) — skeleton of `ActionModel`
  family only.
- Contact dynamics — placeholder; interop with mjwarp deferred.
- OSIM / SMPL parsers — `ModelBuilder` proves the shape is expressive;
  real parsers are future work in `better_robot_human` (see
  [15_EXTENSION.md §HUMAN extension](../conventions/15_EXTENSION.md)).
- Muscle actuators — future work, strictly behind the `dynamics/` layer.
- ROS / real-robot drivers — out of scope forever.

## 7. How to use this document going forward

If you are about to write code, this document is **not** your guide.
Read instead:

- [UPDATE_PHASES.md](../UPDATE_PHASES.md) — the operational sequence
  from the current state through v1.0.
- [01_ARCHITECTURE.md](01_ARCHITECTURE.md) — the target shape of the
  package.
- [16_TESTING.md](../conventions/16_TESTING.md) — what "green CI" means.
- [18_ROADMAP.md](../status/18_ROADMAP.md) — what comes after v1.

This document exists so that future readers can answer two questions
quickly:

1. *What was here before, and why isn't it here anymore?* (sections 1–4)
2. *What did the v1 release commit to, and how much of it is done?*
   (section 5)

Both are forensic, not aspirational.
