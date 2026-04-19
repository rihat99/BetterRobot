# 11 · Skeleton and Migration Plan

This is the execution plan: how we get from the current BetterRobot layout
to the design in 00–10 without throwing away the working pieces (FK, basic
IK, collision distance math, solver scaffolding). The headline rule:
**skeleton lands complete before any existing behaviour is migrated.**

## 1. Inventory — what exists today

```
src/better_robot/
├── math/               # SE3/SO3/spatial wrappers around pypose    (keep, reshape)
├── models/             # RobotModel / JointInfo / LinkInfo          (rewrite shape)
│   └── parsers/        # URDF parser                                (move, reshape)
├── algorithms/
│   ├── kinematics/     # FK / Jacobian / chain                      (split)
│   └── geometry/       # collision primitives + distance            (move)
├── costs/              # pose, limits, rest, collision, manipul.    (rename → residuals/)
├── solvers/            # LM/GN/Adam/LBFGS + Problem + SOLVERS reg.  (rewrite)
├── tasks/
│   ├── ik/             # solve_ik, IKConfig, IKVariable             (shrink to facade)
│   ├── trajopt/        # stub                                       (reshape)
│   └── retarget/       # stub                                       (reshape)
└── viewer/             # viser visualiser                           (keep)
```

Classes / files that survive the rewrite (with a new name or new home):

- Forward kinematics math — the traversal code in
  `algorithms/kinematics/forward.py::_fk_impl` is close to the new FK; it
  just needs to be rewired to use `model.topo_order` and `JointModel`.
- SE3 functions in `math/se3.py` — become `lie/se3.py` verbatim; the body
  is already the right idea.
- `math/spatial.py::adjoint_se3` — moves to `lie/se3.py::adjoint`.
- The capsule-based collision primitives and SDF functions in
  `algorithms/geometry/` — rename and re-register against the dispatch
  table.
- The viser visualiser and helpers.
- The LM / GN / Adam / LBFGS solver loops (the mechanics, not the
  interfaces).
- Tests that don't bind to private attributes (e.g. `test_imports.py`,
  higher-level tests in `test_ik.py`, `test_jacobian.py`).

Classes / files that are **deleted outright**:

- `RobotModel._frozen` custom immutability guard — replaced by
  `@dataclass(frozen=True)`.
- `RobotData` — replaced by `data_model/data.py`.
- `IKVariable` — replaced by the fact that `q` already contains the
  free-flyer DOFs.
- `solvers/registry.py` (`SOLVERS`) — there is no global solver registry
  in the new layout. Optimisers are just classes.
- `solvers/levenberg_marquardt_pypose.py` — benchmark-only, adds pypose
  surface area.
- `tasks/ik/variable.py` / `tasks/ik/config.py` current files — replaced
  by `tasks/ik.py` + `IKCostConfig` + `OptimizerConfig`.
- The `_solve_floating_autodiff` / `_solve_floating_analytic` /
  `_analytic_collision_jacobian` triplet — the whole point of the rewrite.
- `algorithms/kinematics/jacobian.py::compute_jacobian`, `limit_jacobian`,
  `rest_jacobian` — the logic is preserved as `.jacobian()` methods on
  the respective residual classes.

## 2. Target layout (recap from 01 ARCHITECTURE)

```
src/better_robot/
├── __init__.py
├── _typing.py
├── backends/
├── lie/              se3, so3, tangents, _pypose_backend
├── spatial/          motion, force, inertia, symmetric3, ops
├── data_model/       model, data, joint_models/, frame, body, topology, indexing
├── kinematics/       forward, jacobian, jacobian_strategy, chain
├── dynamics/         rnea, aba, crba, centroidal, derivatives, integrators, action/
├── residuals/        pose, limits, regularization, smoothness, manipulability, collision
├── costs/            stack
├── optim/            problem, optimizers/, solvers/, kernels/, strategies/, jacobian_spec
├── tasks/            ik, trajopt, retarget, trajectory
├── collision/        geometry, pairs, closest_pts, robot_collision
├── io/               ir, build_model, parsers/, builders/
├── viewer/           visualizer, helpers
└── utils/            batching, broadcasting, logging, testing
```

## 3. Phase plan

Every phase lands a green test suite, a green lint suite, and a
runnable example. Nothing merges in a broken state.

### Phase 0 — Docs land (this batch)

- [x] Write `docs/00_VISION.md` through `docs/12_VIEWER.md`.
- [x] Write cross-cutting docs: `13_NAMING.md`, `14_PERFORMANCE.md`,
      `15_EXTENSION.md`, `16_TESTING.md`, `17_CONTRACTS.md`.
- [x] Add `docs/README.md` as an index.

### Phase 1 — Skeleton only

Create the new directory tree with every file in place. Each file
contains:

- Correct imports.
- Correct public class / function signatures, matching this doc set.
- One-line docstrings referencing the design doc that owns the feature.
- A body that either implements the function (when trivial) or raises
  `NotImplementedError("see docs/<name>.md")`.

Touch **no** behaviour yet. The existing `src/better_robot/*` keeps
working in parallel. The new tree lives next to it as
`src/better_robot_v2/` for the duration of phases 1–4; once phase 5
lands, `v2` becomes `src/better_robot`.

Pass criteria:

- `python -c "import better_robot_v2 as br; print(br.__all__)"` lists
  all **25** public symbols (see [01_ARCHITECTURE.md](01_ARCHITECTURE.md)).
- `pytest tests_v2/contract/` green (skeleton signatures, layer DAG,
  public API, naming, hot-path lint; see
  [16_TESTING.md §5](16_TESTING.md)).

### Phase 2 — Lie / Spatial / Data model

Land real implementations for the bottom layers:

- `lie/se3.py`, `lie/so3.py`, `lie/_pypose_backend.py` — port
  `math/se3.py` + `math/so3.py` verbatim (same bodies, new module paths).
- `lie/tangents.py` — new: right/left Jacobians, hat/vee. Tested against
  finite differences.
- `spatial/motion.py`, `spatial/force.py`, `spatial/inertia.py`,
  `spatial/symmetric3.py`, `spatial/ops.py` — new value types. These are
  plain dataclasses around tensors with method bodies that all route
  through `lie/`.
- `data_model/model.py`, `data_model/data.py`, `data_model/frame.py`,
  `data_model/body.py`, `data_model/topology.py`, `data_model/indexing.py`.
- `data_model/joint_models/` — one file per joint family:
  `fixed.py`, `revolute.py`, `prismatic.py`, `spherical.py`,
  `free_flyer.py`, `planar.py`, `translation.py`, `helical.py`,
  `composite.py`, `mimic.py`.

Every `JointModel` gets a **unit test** that exercises
`joint_transform`, `joint_motion_subspace`, `integrate`, `difference`,
`neutral` on a random batch and compares against a reference (pypose for
revolute/prismatic/spherical, hand-derived for free-flyer / planar).

Pass criteria:

- `tests_v2/lie/*` green.
- `tests_v2/spatial/*` green.
- `tests_v2/data_model/*` green.

### Phase 3 — IO / parsers / build_model

Port URDF parsing from `models/parsers/_urdf_impl.py` to
`io/parsers/urdf.py` via the new `IRModel`. Implement `build_model()`.
Add a minimal MJCF parser (joints + bodies + frames; meshes later) if
`mujoco` is importable.

Programmatic builder lands as `io/parsers/programmatic.py::ModelBuilder`.

Pass criteria:

- `br.load("panda.urdf")` returns a `Model` equivalent (joint count,
  body count, limits) to the current `RobotModel`.
- `br.load("g1.urdf", free_flyer=True)` returns a `Model` whose
  `joint_models[1]` is `JointFreeFlyer` and whose `nq` is 7 + 29 = 36.
- `br.load(make_smpl_like_body)` returns a `Model` with a free-flyer root
  and 23 spherical joints.
- `tests_v2/io/*` green.

### Phase 4 — Kinematics / residuals / optim / tasks

Wire up the compute stack:

1. `kinematics/forward.py` — the topological scan against `JointModel`.
2. `kinematics/jacobian.py` — `compute_joint_jacobians`,
   `get_joint_jacobian`, `get_frame_jacobian`, `residual_jacobian`,
   `JacobianStrategy`.
3. `residuals/` — port the five current costs to the new `Residual`
   class layout, each with an analytic `.jacobian()` where feasible.
4. `costs/stack.py` — `CostStack` with the pre-allocated buffer
   implementation.
5. `optim/optimizers/levenberg_marquardt.py` — port the existing LM
   loop, but now it calls `problem.residual(x)` /
   `problem.jacobian(x)` and is agnostic to the underlying structure.
6. `optim/optimizers/gauss_newton.py`, `adam.py`, `lbfgs.py`.
7. `optim/kernels/`, `optim/solvers/`, `optim/strategies/`.
8. `tasks/ik.py` — the ≤120-line facade.

Pass criteria:

- `examples/01_panda_ik.py` and `examples/02_g1_floating_ik.py` run,
  with the *same* `solve_ik` code path.
- `tests_v2/kinematics/test_fk_equivalence.py` — new FK matches old FK
  on random configs within `1e-6`.
- `tests_v2/kinematics/test_analytic_vs_autodiff.py` — for every residual
  with a `.jacobian()`, analytic and autodiff match within `1e-5`.
- `tests_v2/tasks/test_ik_regression.py` — `solve_ik(panda, …)` reaches
  the same target pose as the legacy code on the IK benchmark set.

### Phase 5 — Cutover

When phase 4 is green:

1. Move `src/better_robot_v2/` → `src/better_robot/`, deleting the old
   tree in the same commit.
2. Move `tests_v2/` → `tests/`, dropping the old tests that tested
   removed internals.
3. Update `pyproject.toml`, `CLAUDE.md`, and every `CLAUDE.md` under
   `src/` to reflect the new shapes. The current `CLAUDE.md` sections
   on PyPose conventions and IK dispatch are largely preserved; the
   floating-base sections are rewritten to point at `JointFreeFlyer`.
4. Add a one-shot `docs/CHANGELOG.md` entry describing what changed
   for any downstream user.

### Phase 6 — Dynamics skeleton → dynamics bodies

See 06 DYNAMICS §5 for the dynamics milestone breakdown (D0 already
landed in phase 1; D1 center-of-mass is the first numerical milestone).

### Phase 7 — Warp backend

See 10 BATCHING_AND_BACKENDS §7. Starts after dynamics D4 lands.

### Rename sprint (continuous, per 13_NAMING)

Parallel to Phase 2 onwards, the rename to readable storage names runs
as described in [13_NAMING.md §5](13_NAMING.md):

1. Add new field names on `Data` alongside the old.
2. Migrate all internal readers to the new names.
3. Convert old names to `@property` shims that emit `DeprecationWarning`.
4. Remove shims in v1.1.

`tests/contract/test_naming.py` is the enforcement mechanism.

## 4. Acceptance criteria for v1 release

- [ ] `better_robot.__all__` has exactly the **25** symbols from
      [01_ARCHITECTURE.md](01_ARCHITECTURE.md).
- [ ] All contract tests green (DAG, public API, naming, hot-path lint,
      docstrings) — see [16_TESTING.md §5](16_TESTING.md).
- [ ] All benchmark budgets met on the reference hardware (see
      [14_PERFORMANCE.md §1](14_PERFORMANCE.md)).
- [ ] `tests/tasks/test_ik_regression.py` — Panda + G1 IK outputs within
      `1e-4` of the pinned regression oracle on a 50-sample benchmark.
- [ ] All `examples/*.py` run end-to-end and are imported by
      `tests/examples/test_examples.py`.
- [ ] `forward_kinematics` on CUDA matches CPU to `1e-5`.
- [ ] Every dynamics symbol exists and raises `NotImplementedError`
      (except `center_of_mass` and `integrate_q`, which are implemented).
- [ ] No pinocchio-cryptic identifiers in new code (enforced by
      `tests/contract/test_naming.py`); deprecated shims present only on
      `Data` and carry `DeprecationWarning`.

## 5. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Free-flyer retraction differs subtly from the current `_solve_floating_analytic` adjoint trick | Phase 4 regression test freezes the legacy outputs on a fixed seed before cutover |
| `JointModel` protocol introduces per-joint Python dispatch overhead | Dispatch is resolved at `build_model` time into a tuple; the hot loop is a fixed Python for-loop over `model.topo_order` that `torch.compile` can flatten |
| PyPose backward passes change shape under the new wrapper | `lie/_pypose_backend.py` tests use `torch.autograd.gradcheck` on every primitive |
| Analytic pose Jacobian via `right_jacobian_inv_se3` regresses on near-singular configurations | Add specific regression test at `log_err ≈ π` (rotation close to 180°) — this is why the current code uses `ori_weight=0.1`; with proper `Jr_inv` we should be able to drop that hack |
| Tests that bind to removed private attributes (`_fk_joint_order`, `_frozen`) break | Phase 1 runs the new tests in `tests_v2/`; phase 5 deletes legacy tests in the same commit as the cutover |
| Dynamics skeleton gets abandoned as "placeholder land" | Every dynamics function's docstring ends with a pointer to its owning docs section and a `TODO(milestone D2)` tag; CI greps for orphan TODOs |

## 6. Out-of-scope for v1

- Optimal control solvers (DDP/iLQR) — skeleton of `ActionModel` family only.
- Contact dynamics — placeholder; interop with mjwarp deferred.
- OSIM / SMPL parsers — the builder DSL proves the shape is expressive; real
  parsers are future work.
- Muscle actuators — future work, strictly behind the `dynamics/` layer.
- ROS / real-robot drivers — out of scope forever.

## 7. Definition of done

A contributor should be able to read 00–11, follow the phase order, and
produce the library without improvising the data model, the Jacobian
interface, or the solver shape. If a phase cannot be executed from the
docs alone, the docs are wrong and must be fixed before any code is
written.
