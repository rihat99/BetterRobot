# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
  autodiff.
- **Unified solver stack.** `LeastSquaresProblem` + `CostStack` + `Optimizer`
  serves IK and trajectory optimisation through the same substrate.
  Pluggable optimisers (LM / GN / Adam / L-BFGS / multi-stage), linear
  solvers (Cholesky / LSTSQ / CG / sparse Cholesky), robust kernels
  (L2 / Huber / Cauchy / Tukey), and damping strategies (Constant /
  Adaptive / TrustRegion).
- **Featherstone dynamics.** RNEA, ABA, CRBA, CCRBA, centroidal momentum,
  centre of mass, autograd-derived `compute_*_derivatives`. Three-layer
  Crocoddyl-style action models for future optimal-control work.
- **Trajectory optimisation.** `solve_trajopt` with knot and B-spline
  parameterisations; manifold-aware `Trajectory.resample`.
- **URDF + MJCF parsers.** `br.load(path)` dispatches by suffix;
  `free_flyer=True` adds a free-flyer root. Programmatic `ModelBuilder`
  for robots not described by a file.
- **Asset resolution.** `AssetResolver` Protocol with `Filesystem`,
  `Package`, `Composite`, and `CachedDownload` resolvers; mesh path
  logic lives in one place.
- **Viewer V1.** `Visualizer` with viser backend, `SkeletonMode`,
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
- Viewer: `CollisionMode`, `ComOverlay`, `PathTraceOverlay`,
  `ResidualPlotOverlay`, `VideoRecorder`, `OffscreenBackend`,
  transport controls, camera paths, multi-robot sessions.
- Backends: Warp kernels.
