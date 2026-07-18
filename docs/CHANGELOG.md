# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

- `Problem.add_variable` and `Problem.add_residual` now build least-squares
  problems directly from named variables and residual callables. `VarSpec`
  and `ResidualItem` remain available when explicit records are useful.
- `solve_trajopt` now takes a sequence of `ResidualItem` objects through its
  `residuals=` argument. Callers choose active terms by passing only those
  items; robust kernels, weights, and grouping remain attached to each item.
- `LevenbergMarquardt` and `GaussNewton` now share one dense or block-banded
  path. `run_first_order` adapts the same `Problem` to a standard
  `torch.optim.Optimizer`, and eligible converged solves can request guarded
  implicit differentiation.
- Temporal variables and residual patterns can route trajectory problems
  through block-banded Cholesky. Problems without complete temporal structure
  continue to use the dense path and report why.
- Tensor-only FK, frame-placement, Jacobian, RNEA, ABA, CRBA, and CCRBA passes
  are public and return named result objects. `ModelStructure` and
  `ModelValues` are public, dynamics workspaces are optional, and gradients
  can flow through configurations and model values.
- `solve_ik` exposes implicit differentiation for eligible solves, and
  `solve_contact_forces` no longer detaches the force and dynamics tensors it
  returns. Ordinary task solves remain detached where documented.
- Public functions validate tensor shape, dtype, and device once. Model values
  are checked when attached to a model instead of during every kinematics or
  dynamics call, and public input errors now use consistent wording.
- `ccrba` now returns `CCRBAResult`, which names the centroidal map and
  momentum while retaining tuple unpacking.
- The former optimization object hierarchy and import paths were removed,
  including `CostStack`, `CostItem`, `CostKind`, `LeastSquaresProblem`,
  `Optimizer`, `OptimizationResult`, `SolverState`, `SolverStatus`, the custom
  `Adam` state machine, L-BFGS and multi-stage wrappers, damping strategies,
  phase records, and the `better_robot.costs` package. Every removed symbol
  and its supported replacement is listed in the
  [migration details](https://github.com/rihat99/BetterRobot/blob/dev/plan/migration_ledger.md).
- Scalar objectives, `ResidualState`, residual-owned Jacobian hooks,
  `ResidualSpec`, provider `inputs` declarations, and the root/kinematics
  `JacobianStrategy` export were removed or replaced by the single callable
  residual protocol. Exact replacements are in the
  [migration details](https://github.com/rihat99/BetterRobot/blob/dev/plan/migration_ledger.md).
- `LSTSQ`, the rank-deficient Cholesky fallback, `NormalCG`, `NormalOperator`,
  matrix-free routing, pure linear-solve diagnostics, and nested
  `optim.blocks`, `optim.kernels.*`, `optim.solvers.*`, and `optim.structure`
  import paths were removed. Dense and declared temporal routes replace them;
  the [migration details](https://github.com/rihat99/BetterRobot/blob/dev/plan/migration_ledger.md)
  cover each surface.
- Unimplemented public dynamics exports (`compute_minverse`,
  `compute_coriolis_matrix`, centroidal dynamics derivatives, three dynamics
  integrators, and `nle`) and residual exports (`YoshikawaResidual`, collision
  residuals, and `JointAccelLimit`) were removed. Their supported alternatives
  are listed in the
  [migration details](https://github.com/rihat99/BetterRobot/blob/dev/plan/migration_ledger.md).
- The unused `ReferenceFrame` enum was replaced by the literal frame strings
  accepted by the Jacobian API; see the
  [migration details](https://github.com/rihat99/BetterRobot/blob/dev/plan/migration_ledger.md).
- Fused Warp forward kinematics remains an explicit GPU alternative with
  forward and gradient parity coverage; PyTorch remains the default path.
- Tutorials now introduce the robotics concepts they use, guides show complete
  tasks, concept chapters explain design trade-offs, and the generated
  reference matches the current public API.

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
