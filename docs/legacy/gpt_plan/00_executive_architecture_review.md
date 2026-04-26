# Executive Architecture Review

## Context

BetterRobot is still early enough that the right answer can be a breaking change. The current docs already describe a much cleaner package than the code has fully implemented: a small public API, strict layer direction, Pinocchio-style `Model` and `Data`, PyTorch-first tensors, and a future Warp backend that never leaks into user code. The codebase also has useful guardrails already in place: dependency tests, public API tests, naming contract tests, Pinocchio comparison tests, and an explicit PyPose isolation rule.

The biggest architectural tension is the Lie layer. The current design doc says `lie/` should stay as plain functions over tensors and explicitly forbids `SO3` and `SE3` classes. The user direction is different: BetterRobot should grow durable types such as `SO3` with overloaded operators. I agree with changing this now, before more modules depend on raw 4-vector and 7-vector pose values everywhere.

## Main Recommendation

Keep tensor kernels as the performance and backend substrate, but make typed value objects the public and internal ergonomic API for manifold values.

Concretely:

- Add `SO3` and `SE3` thin dataclasses, not `torch.Tensor` subclasses.
- Keep `lie.so3` and `lie.se3` functional APIs as stable low-level kernels used by hot loops and tests.
- Replace direct PyPose dependence with pure Torch kernels for SO3 and SE3 before serious Warp work begins.
- Add a backend protocol around kernels, not around high-level robotics concepts.
- Split runtime state from cache-heavy algorithm workspaces before trajectory optimization and dynamics expand.

This gives BetterRobot both nice Python semantics and a clean future for Torch compile, custom autograd, and Warp kernels.

## What To Keep

- The layer DAG is right: `backends -> lie -> spatial -> data_model -> kinematics/dynamics/collision -> residuals -> costs -> optim -> tasks -> viewer`.
- `Model` as immutable topology and `Data` as mutable per-query workspace is the right base.
- Per-joint `JointModel` protocols are the right abstraction for universal joints.
- Residuals as callable objects with optional analytic Jacobians are the right extension point.
- Cost stacks with named, weighted, activatable residuals are the right optimization composition model.
- The public top-level API ceiling is valuable, even if the exact 25 names can change before v1.
- Contract tests are worth preserving and expanding.

## What To Change First

1. Introduce `SO3` and `SE3` value types.

   They should wrap a tensor, validate trailing shape and convention, expose `.tensor`, `.batch_shape`, `.device`, `.dtype`, `.to(...)`, `.detach()`, and named constructors. Operators should be limited and predictable: `@` for group composition and group action, `inverse()` for inverse, `log()` and `exp()` for maps. Consider `*` only as an optional same-type composition alias after tests prove it is not confusing.

2. Replace PyPose as the default Lie implementation.

   PyPose is useful as an oracle and historical bootstrap, but the docs already record incorrect ambient gradients. A robotics package that will be used for optimization cannot leave exp/log/compose gradients behind a known-bad backend. Keep PyPose tests or comparison utilities, but make BetterRobot's own Torch implementation canonical.

3. Redesign the backend boundary.

   The current global `set_backend("warp")` skeleton is too coarse for Torch compile, mixed execution, per-kernel capability selection, and testing. Add explicit backend objects or execution configs, then default them quietly. Backend selection should happen at kernel dispatch boundaries, not through global mutable package state.

4. Split state, cache, and output concepts.

   `Data` is useful, but it is becoming a catch-all. Add smaller `RobotState`, `JointState`, and trajectory state types so optimizers and tasks can talk about variables without depending on every cache slot in `Data`.

5. Make trajectory optimization a first-class architecture, not "flatten T times q".

   Dense `(dim, T * nv)` Jacobians will not survive long horizons. The residual stack needs sparse/block specs, matrix-free `J^T r` paths, and temporal batching policies from the start.

## Code-Specific Observations

- `src/better_robot/lie/_pypose_backend.py` is the only PyPose import, which is good, but every functional Lie operation still routes there.
- `src/better_robot/lie/so3.py` and `src/better_robot/lie/se3.py` are concise and easy to keep as low-level kernels.
- `src/better_robot/spatial` already proves that thin dataclasses around tensors can work well in this codebase.
- `src/better_robot/backends/__init__.py` is currently a global selector and no-op graph capture hook, not a real backend architecture.
- `src/better_robot/kinematics/forward.py` uses list accumulation for autograd safety in FK, but `update_frame_placements` and Jacobian routines use in-place tensor writes. That may be fine for non-gradient outputs, but it should be documented per function.
- `src/better_robot/tasks/ik.py` has config fields for linear solver, robust kernel, and damping, but they are not wired through to the optimizer yet.
- `ModelBuilder` docs show `kind="revolute_z"`, while `build_model` expects `kind="revolute"` with an axis. That mismatch should be fixed early because examples define user habits.
- Optional packages such as `viser`, `trimesh`, `yourdfpy`, and `robot_descriptions` are currently hard dependencies. That conflicts with cheap imports and optional integration guidance.

## Priority Order

P0: decisions that affect every future feature:

- Lie value classes and pure Torch Lie kernels.
- Backend protocol and capability matrix.
- State/cache split.
- One canonical docstring, shape, dtype, and exception policy.

P1: feature architecture that should be shaped before implementation expands:

- Sparse and matrix-free optimization.
- Dynamics algorithm and derivative plan.
- Collision pair representation and map-reduce kernels.
- Programmatic builder and parser IR cleanup.

P2: later but worth designing now:

- Warp graph capture.
- OpenSim and SMPL-like human model import.
- Viewer and recording backends.
- High-level robot builder automation.

## Proposed Plan Files

- `01_lie_types_and_backend_boundary.md`
- `02_backend_and_kernel_architecture.md`
- `03_model_data_state_and_trajectory.md`
- `04_optimization_tasks_and_trajopt.md`
- `05_dynamics_and_spatial_refactor.md`
- `06_io_collision_viewer_and_human_models.md`
- `07_quality_testing_docs_public_api.md`
- `08_migration_sequence.md`
- `09_claude_plan_review.md`
