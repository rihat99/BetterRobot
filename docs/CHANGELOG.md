# Changelog

## Unreleased — Docs refinement 2026-04-19

Comprehensive doc-layer refinement ahead of the next implementation
sprint. **No code changes** in this batch — all work is in `docs/`.

### New cross-cutting specs

| # | File | Purpose |
|---|------|---------|
| 13 | [NAMING.md](13_NAMING.md) | Readable rename of pinocchio cryptic storage names (`oMi` → `joint_pose_world`, `nle` → `bias_forces`, `Ag` → `centroidal_momentum_matrix`, …) with a one-release deprecation shim on `Data`. |
| 14 | [PERFORMANCE.md](14_PERFORMANCE.md) | Latency / memory budgets, kernel fusion, adaptive dispatch (cuRobo), CUDA graph capture, anti-pattern lint. |
| 15 | [EXTENSION.md](15_EXTENSION.md) | `Protocol`-shaped seams for residuals, joints, optimisers, kernels, strategies, linear solvers, collision primitives, render modes, parsers, backends. |
| 16 | [TESTING.md](16_TESTING.md) | Unit / integration / contract / regression / benchmark strategy; coverage budgets per layer. |
| 17 | [CONTRACTS.md](17_CONTRACTS.md) | Input contracts, error taxonomy, numerical guarantees, autograd rules, SemVer policy. |

### Updated specs

- **00_VISION** — added performance targets section; clarified the name
  stance ("steal algorithms, not jargon"); added reference to
  [13_NAMING.md](13_NAMING.md).
- **01_ARCHITECTURE** — added an **Extension seams** section pointing
  to [15_EXTENSION.md](15_EXTENSION.md); replaced the ad-hoc testing
  story with pointers to [16_TESTING.md](16_TESTING.md) and
  [14_PERFORMANCE.md](14_PERFORMANCE.md).
- **02_DATA_MODEL** — renamed every `Data` field to its readable form
  (`joint_pose_world`, `frame_pose_world`, `joint_pose_local`,
  `joint_velocity_world`, `mass_matrix`, `bias_forces`,
  `centroidal_momentum_matrix`, `com_position`, `joint_jacobians`, …).
  Added a §11 describing the migration window and the deprecated-name
  shims.
- **03_LIE_AND_SPATIAL** — added pointers to
  [13_NAMING.md](13_NAMING.md) (keep `Jr` / `hat` / `vee` as Lie
  notation, use verbose function names at the boundary) and
  [17_CONTRACTS.md](17_CONTRACTS.md) for numerical stability.
- **05_KINEMATICS** — declared as the single source of truth for
  `JacobianStrategy` and the `Residual` protocol; updated every
  `oMi`/`oMf`/`liMi` reference to `joint_pose_world` /
  `frame_pose_world` / `joint_pose_local` in the example code.
- **06_DYNAMICS** — renamed `nle` → `bias_forces`, updated `Data`
  storage references (`data.M` → `data.mass_matrix`, `data.Ag` →
  `data.centroidal_momentum_matrix`, …). Added explicit note that
  RNEA/ABA/CRBA own their **analytic backward kernels**, not autograd.
- **07_RESIDUALS_COSTS_SOLVERS** — added `@runtime_checkable` to the
  `Optimizer`/`LinearSolver`/`RobustKernel`/`DampingStrategy` protocols;
  introduced the shared `SolverState` dataclass (cuRobo pattern); added
  a sparsity-aware assembly note for self-collision residuals.
- **08_TASKS** — added the **two-stage LM-then-LBFGS** solver
  (`optimizer="lm_then_lbfgs"`) for high-DoF humanoids; added
  **B-spline** as the default trajectory parameterisation, with
  implicit smoothness and fewer optimisation variables; fixed `oMf`
  reference.
- **10_BATCHING_AND_BACKENDS** — added the **`WarpBridge` pattern**
  (mjlab) for torch↔warp tensor shims with per-shape caching; added
  **adaptive kernel dispatch** (cuRobo) for collision SDF kernels;
  renamed `oMi` / `oMf` in example code.
- **11_SKELETON_AND_MIGRATION** — corrected the public-API count from
  23 to 25; added a **rename sprint** tied to
  [13_NAMING.md](13_NAMING.md); rewired the acceptance criteria to
  reference the new contract and benchmark suites.
- **README** — refreshed the doc index into *Core specs* (00–12) and
  *Cross-cutting specs* (13–17) with a reading-order guide.

### Rationale

The previous doc set locked down *structure* but left four dimensions
implicit: **naming**, **performance**, **extensibility**, **testing /
contract discipline**. The new cross-cutting docs make each of these
normative so that the next implementation sprint can regress against
specific bars (perf budgets, contract tests, naming lint) rather than
informal agreement.

No core architectural decisions were reversed — the DAG, the frozen
`Model` / mutable `Data` split, the `JacobianStrategy` flag, the 25-symbol
public API, the `JointFreeFlyer`-as-root floating-base story all stand
exactly as specified in v0.2.

## v0.2.0 — Phase 5 cutover (2026-04-11)

Complete rewrite of `better_robot`. The old `src/better_robot/` tree is
replaced by the pinocchio-inspired v2 architecture. Old tests that bound
to removed internals are dropped; all 227 new tests pass.

### Breaking changes

The public API is entirely new. Nothing from `better_robot` v0.1 is
forward-compatible.

| Old | New |
|-----|-----|
| `br.load_urdf(yourdfpy_obj)` | `br.load(path_or_urdf_obj)` |
| `model.joints.num_actuated_joints` | `model.nq` |
| `model.links.num_links` | `model.nbodies` |
| `model.q_default` | `model.q_neutral` |
| `br.IKConfig(...)` | `IKCostConfig(...) + OptimizerConfig(...)` |
| `br.solve_ik(...) → tensor` | `br.solve_ik(...) → IKResult` |
| `result.q` (was returned directly) | `result.q` (on `IKResult`) |
| `br.compute_jacobian(model, q, link_idx, ...)` | `br.get_frame_jacobian(model, data, frame_id)` |
| `better_robot.algorithms.geometry.RobotCollision` | `better_robot.collision.RobotCollision` |
| `better_robot.math.se3_*` | `better_robot.lie.se3.*` |
| Fixed-base vs floating-base split | Single code path — use `load(..., free_flyer=True)` |

### New features

- **Pinocchio-style Model/Data separation** — `Model` is frozen; `Data` is the per-query workspace.
- **Universal joint system** — revolute (R{X,Y,Z}, unaligned, unbounded), prismatic, spherical, free-flyer, fixed, helical, planar, mimic, composite.
- **Batched FK** — `forward_kinematics(model, q: (B, nq))` runs over any batch shape with no Python loops.
- **Analytic Jacobians** — all built-in residuals have a `.jacobian()` method; `JacobianStrategy.AUTO` picks analytic, falls back to finite differences.
- **Single IK code path** — fixed and floating-base IK through the same `solve_ik` facade; the difference is the model, not the solver.
- **URDF + MJCF parsers** — `br.load(path)` dispatches by suffix; `free_flyer=True` adds a free-flyer root.
- **25-symbol public API** — see `better_robot.__all__`.

### Architecture

```
src/better_robot/
  lie/          SE3/SO3/tangents/_pypose_backend
  spatial/      Motion, Force, Inertia
  data_model/   Model, Data, Frame, Body, Joint, joint_models/
  kinematics/   forward_kinematics, compute_joint_jacobians, get_frame_jacobian
  dynamics/     rnea, aba, crba stubs (raise NotImplementedError)
  residuals/    PoseResidual, PositionResidual, OrientationResidual, limits, regularization
  costs/        CostStack
  optim/        LeastSquaresProblem, LevenbergMarquardt, GaussNewton, Adam, LBFGS
  tasks/        solve_ik, IKCostConfig, OptimizerConfig, IKResult
  collision/    Sphere, Capsule, Box, HalfSpace, RobotCollision
  io/           load, build_model, IRModel, parsers/
  viewer/       Visualizer (stub — pending port)
  utils/        batching, logging
```

### Known limitations (Phase 6+)

- `viewer.Visualizer` raises `NotImplementedError` — viser port pending.
- `rnea`, `aba`, `crba`, `center_of_mass`, `compute_centroidal_map` raise
  `NotImplementedError` — dynamics bodies land in Phase 6.
- No trajectory optimization or retargeting — stubs only.
