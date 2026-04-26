# Migration Sequence

## Goal

Make the big architectural changes while BetterRobot is still early, without creating an unreviewable rewrite. The order below keeps the code runnable and lets tests protect each step.

## Phase 0: Decisions And Contract Updates

Deliverables:

- Update design docs to allow `SO3` and `SE3` value classes.
- Decide dtype policy: float32 and float64 only for v1.
- Decide docstring style.
- Decide whether top-level `better_robot.__all__` can change before v1.
- Move optional dependency intent into packaging docs.

Exit criteria:

- Docs no longer contradict the intended Lie class direction.
- Known contradictions are tracked in status docs.

## Phase 1: Pure Torch Lie Kernels

Deliverables:

- Implement pure Torch SO3 kernels.
- Implement pure Torch SE3 kernels.
- Keep existing `lie.so3` and `lie.se3` function names.
- Move PyPose bridge into optional oracle/tests.
- Add central finite-difference gradient tests.

Exit criteria:

- Existing Lie tests pass.
- New value and gradient tests pass.
- PyPose is no longer a runtime dependency of core Lie operations.

Risk:

- SE3 log/exp near singularities are easy to get wrong.

Mitigation:

- Use Pinocchio and PyPose as comparison oracles.
- Start with strict small-angle tests and sampled moderate rotations.

## Phase 2: `SO3` And `SE3` Value Types

Deliverables:

- Add `lie.types.SO3` and `lie.types.SE3`.
- Add operator policy with `@`.
- Add typed constructor and method docs.
- Add class/function equivalence tests.
- Update `_typing` aliases to avoid name conflict.

Exit criteria:

- High-level examples can use typed poses.
- Existing tensor API still works.
- No TensorSubclass behavior exists.

Risk:

- Object wrappers could sneak into hot loops.

Mitigation:

- Keep packed tensors inside `Model` and kernel functions.
- Benchmark before replacing hot-loop internals.

## Phase 3: Backend Protocol

Deliverables:

- Add a small `Backend` protocol and default Torch backend object.
- Replace global backend dependence with explicit default dispatch helpers.
- Add capability matrix.
- Add optional Warp import checks and errors.
- Add `ModelTensors` view.

Exit criteria:

- Torch backend is the only implemented backend but follows the final shape.
- Tests can instantiate backend objects without mutating global state.
- No public API exposes backend-specific arrays.

Risk:

- Overdesign before Warp exists.

Mitigation:

- Keep the protocol small and only cover existing kernels.

## Phase 4: State And Data Cleanup

Deliverables:

- Add `RobotState`.
- Add or finish `StateMultibody`.
- Add cache validity policy for `Data`.
- Add `Trajectory.to`, validation, slicing, and basic manifold resampling.

Exit criteria:

- Tasks and optimizers can pass state without depending on every `Data` cache field.
- Stale data behavior is tested.

Risk:

- Too much churn in kinematics and residuals.

Mitigation:

- Accept both old and new inputs during one migration window.

## Phase 5: Optimization Wiring And Matrix-Free Paths

Deliverables:

- Wire IK optimizer config fields.
- Make robust kernels affect solver equations.
- Add `LeastSquaresProblem.gradient`.
- Use matrix-free gradient in Adam and LBFGS.
- Add stable collision residual dimension policy.
- Add B-spline trajectory parameterization.

Exit criteria:

- `solve_ik` config knobs are real.
- Dense and matrix-free gradients agree.
- `solve_trajopt` supports sample and B-spline parameterizations.

Risk:

- Solver behavior changes may affect existing regression tests.

Mitigation:

- Keep old defaults until new paths match current tests.

## Phase 6: Dynamics Core

Deliverables:

- Harden RNEA.
- Implement CRBA.
- Implement ABA.
- Add dynamics derivative plan and first derivative tests.
- Add required `JointModel` dynamics hooks.

Exit criteria:

- Dynamics algorithms match Pinocchio on reference models.
- RNEA/CRBA/ABA consistency tests pass.
- Batch tests pass.

Risk:

- Floating-base and spherical joint conventions can drift.

Mitigation:

- Use `StateMultibody`, Lie classes, and explicit tangent ordering everywhere.

## Phase 7: Collision, IO, And Optional Dependency Cleanup

Deliverables:

- Fix builder kind helpers and examples.
- Move optional dependencies to extras.
- Implement primitive SDFs.
- Implement stable self-collision residual.
- Add asset resolver.

Exit criteria:

- Core package imports without viewer/parser extras.
- Builder examples execute.
- Collision residuals are usable in IK.

## Phase 8: Warp Pilot

Deliverables:

- Implement one small Warp kernel family, preferably collision spheres or batched FK.
- Wrap with `torch.autograd.Function` if differentiable.
- Add stream and graph-capture tests.
- Add fallback tests.

Exit criteria:

- Warp acceleration exists without changing public APIs.
- Torch fallback remains the reference implementation.

Risk:

- Warp work can distort public architecture.

Mitigation:

- Keep Warp behind backend protocol and packed tensor views.

## Recommended First Pull Requests

After reading `docs/claude_plan`, the first PR queue should be slightly sharper:

1. Docs update: accept Lie value classes, resolve dtype/docstring contradictions, and add a single normative style guide.
2. Add `ReferenceFrame`, `KinematicsLevel`, and `StaleCacheError`; enforce `Data` cache invalidation on `q`/`v`/`a` assignment.
3. Pure Torch SO3 implementation and tests.
4. Pure Torch SE3 implementation and tests. Do not copy formula sketches without independent derivation and gradcheck.
5. Add `SO3` and `SE3` classes as wrappers over existing functions.
6. Rename `_typing.SO3` and `_typing.SE3` tensor aliases to `SO3Tensor` and `SE3Tensor`.
7. Wire IK config components.
8. Move optional dependencies into extras and add optional-import contract tests.
9. Add regression oracle scaffolding for FK and Lie values; keep perf benchmarks advisory until runners are stable.
10. Add `RobotState` and finish `StateMultibody`.

See `09_claude_plan_review.md` for the adopted/adapted/rejected Claude ideas.

## Release Gate For v1

Before calling the package v1:

- No known-bad PyPose gradients in runtime paths.
- Public pose APIs use typed `SO3`/`SE3` or clearly documented tensor aliases.
- Backend boundary exists and has Torch implementation.
- Core import is cheap and optional integrations are lazy.
- FK, Jacobian, IK, RNEA, CRBA, ABA have oracle tests.
- Trajectory optimization has a scalable gradient path.
- Public APIs document shapes, frames, units, dtype, and differentiability.
