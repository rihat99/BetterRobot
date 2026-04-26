# Vision

BetterRobot is a PyTorch-native library for robot kinematics,
dynamics, and trajectory optimisation. It owes Pinocchio its
mathematical discipline (the `Model` / `Data` split, the spatial
algebra primitives, the algorithm names everyone in the field
already knows), it owes PyRoki its taste in residuals and registries,
and it owes its temperament to a pile of lessons about what *not* to
do — most of which were collected the hard way, by other libraries,
and recorded in reading rather than reimplemented in code.

The first three commitments are the ones that decided everything else.

**PyTorch on the hot path, no exceptions.** Forward kinematics,
Jacobians, residuals, costs, and the solver iterates all participate
in autograd. There is no `AutoDiffXd` scalar to switch into, no JAX
mode flag, no C extension that breaks the gradient graph. When you
write a loss in PyTorch and ask for `.backward()`, the gradient flows
through `solve_ik` cleanly, because everything below it is a tensor
operation. The cost of the commitment is that we cannot lean on
hand-tuned C++ kernels the way Pinocchio does; the benefit is that
users write robot code the same way they write neural-network code,
and the two compose without seams.

**Batched by default.** Every public function accepts a leading batch
dimension `(B..., feature)`. A "single pose" is `(1, 7)`, not `(7,)`.
This sounds like a style choice and is actually the whole reason GPU
support is free: when the loop is over `model.topo_order` (a fixed
Python tuple) instead of over the batch axis, there is no Python work
to amortise across batch entries, and `torch.compile` unrolls the
topology cleanly. A library that is batch-aware for some functions
and scalar-aware for others ends up with two of every algorithm; we
refused that fork before it could happen.

**One code path for fixed and floating base.** A floating-base robot
is one whose root joint is `JointFreeFlyer` (`nq=7`, `nv=6`). That is
the only difference. There is no `setFloatingBase()` flag, no
`base_pose` argument that some functions take and some do not, no
`_solve_floating_*` family of solvers parallel to the fixed-base one.
The first 7 entries of `q` are the base pose for a free-flyer model;
they are not the base pose for a Panda. Either way, `solve_ik` walks
the same code path. A library that splits "static" from "floating"
into separate machinery quickly finds itself maintaining each of them
for every operation; the unified path is dramatically less code and
makes G1 humanoid IK use the same call as Panda IK without ceremony.

The fourth and fifth commitments fall out of those:

**One residual / cost / solver stack.** IK, trajectory optimisation,
filtering, and (when they land) optimal-control problems all share
the same substrate — a `Residual` Protocol, a `CostStack`, a
`LeastSquaresProblem`, and an `Optimizer`. There is one place to
implement a Cauchy robust kernel, one place to swap LM for Adam, one
place to register a sparse linear solver, and that place is shared
across every higher-level task.

**A backend boundary that does not leak.** The math layer (`lie/`)
routes through a `Backend` Protocol with a torch-native default. A
future Warp backend can land kernel-by-kernel without touching call
sites, because users see `torch.Tensor` in and `torch.Tensor` out at
every public surface. The discipline is enforced by a contract test
that walks the import graph: only `lie/`, `kinematics/`, `dynamics/`,
and the `backends/<name>/` packages are allowed to cross the
backend boundary. Every other module composes through their public
APIs.

## What "looks like a story" means in code

A minimal example — load a Panda URDF, solve IK to a target pose,
read back the joint solution:

```python
import better_robot as br

model  = br.load("panda.urdf")
result = br.solve_ik(model, {"panda_hand": target_pose})

result.q                          # (nq,) joint solution
result.frame_pose("panda_hand")   # (7,) SE(3) pose at the solution
```

Three function calls. No solver to instantiate, no problem object to
build, no fixed/floating mode flag. The same three lines work for the
G1 humanoid if you load it with `free_flyer=True`. The first 7 DOFs
of `result.q` are then the base pose; the remaining 29 are the joint
configuration. `solve_ik` does not need to be told.

## What ships today

Implemented and tested: forward kinematics; analytic and autograd
Jacobians (with a finite-diff fallback for diagnostics); the full
residual library (pose / position / orientation, joint position
limits, rest, contact consistency, reference trajectories, velocity
and acceleration smoothness, time-indexed residuals); `CostStack`;
LM, GN, Adam, L-BFGS, and multi-stage optimisers; pluggable linear
solvers (Cholesky, LSTSQ, CG, sparse Cholesky); pluggable robust
kernels (L2, Huber, Cauchy, Tukey) and damping strategies (Constant,
Adaptive, TrustRegion); IK on fixed and floating-base robots;
trajectory optimisation with knot and B-spline parameterisations;
Featherstone dynamics (RNEA, ABA, CRBA, CCRBA), centroidal momentum,
and autograd-derived `compute_*_derivatives`; a three-layer
Crocoddyl-style action model; URDF and MJCF parsers; a programmatic
`ModelBuilder`; a viewer with skeleton / URDF-mesh / collision render
modes, draggable IK target gizmos, and trajectory playback.

A handful of named symbols are deliberately stubbed — they have the
correct signatures, raise `NotImplementedError`, and are listed in
{doc}`/reference/roadmap`. They exist as targets so user code can
already reach for them and tests can already assume them.

## What we deliberately do not do

- We do not ship a physics engine. Contact solvers and integrators
  beyond `integrate_q` and `RK4` come from interop with `mjwarp` and
  similar; we are not in the business of competing with them.
- We do not ship optimal control. There is room reserved in the
  architecture (the three-layer action model is in place) but no
  DDP / iLQR solver. That is intentionally future work.
- We do not ship anatomical joints, muscles, or SMPL parsing in core.
  The data model is expressive enough to host an SMPL-like body
  through the programmatic builder, and the SMPL-and-muscles
  extensions live in a sibling package (`better_robot_human`) under
  the `[human]` extra. Core BetterRobot does not import `chumpy`,
  SMPL, or OpenSim.
- We do not subclass `torch.Tensor`. Typed Lie wrappers (`SE3`,
  `SO3`, `Pose`) are frozen dataclasses *around* a tensor, never
  subclasses of one. `__torch_function__` overrides are a deep
  cautionary tale — see {doc}`lie_and_spatial`.
- We do not invent control / observer / filter frameworks à la Drake.
  Those layers can be built on top of the optimisation substrate
  without adding new abstractions to the core.

## How to read the rest of the book

Each chapter that follows is one layer of the architecture, written so
that a reader who has not seen the source can follow the design
choices:

- {doc}`architecture` lays out the layered DAG in one diagram.
- {doc}`model_and_data` is the Pinocchio-style split between frozen
  topology and per-query workspace.
- {doc}`joints_bodies_frames` describes the universal joint taxonomy
  that lets fixed and floating base share the same code.
- {doc}`lie_and_spatial` is the math layer — SE(3), SO(3), and the
  spatial-algebra value types.
- {doc}`parsers_and_ir` is how URDF / MJCF / programmatic builders
  all converge on a single intermediate representation before
  becoming a `Model`.
- {doc}`kinematics` covers forward kinematics and the unified
  Jacobian dispatch.
- {doc}`dynamics` covers RNEA, ABA, CRBA, the centroidal map, and
  the action-model framework.
- {doc}`residuals_and_costs` is the residual library plus
  `CostStack`.
- {doc}`solver_stack` is `LeastSquaresProblem` and the four
  pluggable axes (Optimizer, LinearSolver, RobustKernel,
  DampingStrategy).
- {doc}`tasks` is `solve_ik`, `solve_trajopt`, and `Trajectory`.
- {doc}`collision_and_geometry` is the collision subsystem.
- {doc}`batching_and_backends` pins the tensor and device
  conventions and explains the backend Protocol.
- {doc}`viewer` is the viser-backed visualisation layer.

The {doc}`/conventions/index` chapters complement the book: they are
the rules contributors must follow when extending the library.
