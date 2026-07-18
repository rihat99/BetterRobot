# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

- **Lean optimization core.** `better_robot.optim` is now one flat package
  with `Problem.add_variable` / `add_residual`, one named-context residual
  protocol, recursive provider memoization, dense or block-banded LM/GN, and a
  small adapter for ordinary `torch.optim` optimizers. The scalar-objective
  subsystem, custom Adam and phase engine, matrix-free normal-CG route, LSTSQ
  option, shadow prevalidated methods, and Jacobian-strategy enum were removed.
- **CUDA-validated opt-in Warp FK.** The fused FK lane now covers fp32/fp64,
  fixed/free bases, branched and deep models, value batching, q/placement
  VJPs, current-stream ordering, and forward graph replay on RTX 6000 Ada.
  Forward-only SMPL measurements beat compiled Torch in the four committed
  batches; the Torch-recompute backward was not timed, so no default changed.
- **Definition-first performance evidence.** The M6 harness defines a
  144-selector Panda/free-Panda/SMPL matrix with isolated cold starts, raw
  samples, memory, and provenance. Only the hardware-named Warp FK cases and a
  filtered SMPL B=1 Torch result are measured; the complete matrix and
  external competitor measurements remain open.
- **Manual-only CI.** The GitHub Actions workflow remains
  `workflow_dispatch` only by owner request. It has no hosted CUDA, automatic
  pull-request/nightly, coverage, or blocking benchmark gate.
- **Opt-in implicit LM/GN differentiation.** Generic named-block solvers add
  `solve(..., differentiate="implicit")` for first-order gradients to declared
  external context tensors. The backward recomputes the exact robust
  tangent/KKT system, supports product manifolds and stable active bounds, and
  strictly rejects invalid batches, Huber kinks, terminal quaternion
  representatives at absolute pi, tensor-role identity collisions, and singular systems. Dense size is
  capped; true structured backward and direct ModelValues/weight rebinding
  remain explicit gaps.
- **Structured trajectory optimization.** Named-block variables may declare
  `time_axis=0`; temporal residuals expose `TemporalPattern` plus exact local
  Jacobian blocks. LM routes between dense Cholesky and block-banded
  `BandedCholesky`, with stable requested/used/reason/detail diagnostics and
  dense automatic fallback.
- **Named-block `solve_trajopt`.** An explicit sequence of `ResidualItem`
  values is adapted to one temporal `RobotConfig` block with arbitrary leading
  batches, sanitized optional bounds, per-element state diagnostics, and route
  fields in `TrajOptResult`.
- **Explicit deferrals.** The component-space `BSplineTrajectory` remains a
  numerical utility rather than a robot-manifold parameterization. Schur
  elimination for temporal plus shared variables also remains deferred; M5
  does not claim either feature.

## v0.2.0 — 2026-04-11

The first stable release of the PyTorch-native BetterRobot stack.

### Highlights

- **Pinocchio-style `Model` / `Data` architecture.** Frozen `Model`,
  mutable `Data`, polymorphic `.to()`.
- **Universal joint system.** Revolute (R{X,Y,Z}, unaligned, unbounded),
  prismatic, spherical, free-flyer, fixed, helical, planar, mimic,
  composite. Single code path for fixed and floating base — a
  floating-base robot is one whose root joint is `JointFreeFlyer`.
- **Batched-by-default FK.** `forward_kinematics(model, q: (B..., nq))`
  runs over any batch shape with no Python loops.
- **Analytic Jacobians.** All built-in residuals have a `.jacobian()`
  method; `JacobianStrategy.AUTO` prefers analytic, falls back to
  central finite differences.
- **Unified solver stack.** `LeastSquaresProblem` + `CostStack` + `Optimizer`
  serves IK and trajectory optimisation through the same substrate.
  Pluggable optimisers (LM / GN / Adam / L-BFGS / multi-stage), linear
  selectable solvers (Cholesky / LSTSQ), robust kernels
  (L2 / Huber / Cauchy / Tukey), and damping strategies (Constant /
  Adaptive). Structured and iterative linear solves were not part of v0.2.0.
- **Featherstone dynamics.** RNEA, ABA, CRBA, CCRBA, centroidal momentum,
  centre of mass, and autograd-derived RNEA, ABA, and CRBA derivative helpers. Three-layer
  Crocoddyl-style action models for future optimal-control work.
- **Trajectory optimisation.** `solve_trajopt` with knot parameterisation;
  manifold-aware `Trajectory.resample`. The Euclidean B-spline basis remains a
  numerical utility and is gated from robot trajopt pending a separately
  reviewed manifold mapping.
- **URDF + MJCF parsers.** `br.load(path)` dispatches by suffix;
  `free_flyer=True` adds a free-flyer root. Programmatic `ModelBuilder`
  for robots not described by a file.
- **Asset resolution.** `AssetResolver` Protocol with `Filesystem`,
  `Package`, `Composite`, and `CachedDownload` resolvers; mesh path
  logic lives in one place.
- **Viewer.** `Visualizer` with viser backend, `SkeletonMode`,
  `URDFMeshMode`, draggable IK target gizmos, frame-axes / grid /
  force-vector overlays, minimal `TrajectoryPlayer`.
- **Public API.** 26 frozen symbols at `better_robot.__init__`,
  enforced by `tests/contract/test_public_api.py`.

### Public API

```
Model, Data, Frame, Joint, Body,
load, ModelBuilder,
SE3,
forward_kinematics, update_frame_placements,
compute_joint_jacobians, get_joint_jacobian, get_frame_jacobian,
JacobianStrategy,
rnea, aba, crba, center_of_mass, compute_centroidal_map,
register_residual,
CostStack,
LeastSquaresProblem,
solve_ik, solve_trajopt, retarget, Trajectory,
```

### Known limitations

The named symbols are public and stable; their bodies are stubbed and
listed on the project roadmap:

- Dynamics: `compute_minverse`, `compute_coriolis_matrix`, the
  higher-order integrators, analytic Carpentier–Mansard derivatives.
- Residuals: `JerkResidual`, `YoshikawaResidual`, `NullspaceResidual`,
  `SelfCollisionResidual`, `WorldCollisionResidual`,
  `JointAccelLimit`.
- Tasks: `retarget`.
- Backends: Warp kernels.
