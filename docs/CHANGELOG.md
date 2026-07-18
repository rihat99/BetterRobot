# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

- **One optimization API.** `Problem.add_variable` and `add_residual` build
  least-squares problems over named variable blocks. LM and Gauss–Newton share
  the same dense or block-banded evaluation path, while `run_first_order`
  adapts a problem to ordinary `torch.optim` optimizers. Unused parallel
  problem, objective, and first-order implementations were removed.
- **An open differentiable core.** Tensor-only kinematics and dynamics passes
  are public and return named results. Dynamics workspaces are optional.
  Gradients can flow to configurations and model values, and IK exposes an
  implicit differentiation option for eligible solves.
- **Clearer public boundaries.** Public functions validate shapes, dtypes,
  and devices once. Model values are checked when attached to a model instead
  of during every kinematics or dynamics call. Error messages now follow one
  consistent pattern.
- **A truthful surface.** Importable placeholders, unused aliases, and modules
  without working behavior were removed. The roadmap now lists only explicit
  runtime guards that remain in source.
- **Structured trajectories.** Temporal variables and residual patterns can
  route LM through block-banded Cholesky. Problems without complete temporal
  structure continue to use the dense correctness path.
- **Opt-in Warp forward kinematics.** The fused GPU pass remains an explicit
  alternative to the PyTorch reference pass. It has forward and gradient
  parity tests, but PyTorch remains the default and the source of truth.
- **Documentation for readers.** Tutorials now define the robotics ideas they
  use, guides show complete tasks, concept chapters explain the trade-offs,
  and the reference matches the current public API.

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
