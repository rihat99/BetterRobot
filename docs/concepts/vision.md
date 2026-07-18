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

**A PyTorch tensor surface on the hot path.** Forward kinematics,
Jacobians, residuals, costs, and solver updates are expressed through PyTorch
tensors. Eager Torch Lie maps, kinematics, dynamics, and direct problem
evaluation have path-specific autograd coverage; that does not make every task
facade differentiable. Ordinary named-block `run`, `solve_ik`, and
`solve_trajopt` return detached results. Small generic LM/GN problems can opt
into the separately documented dense first-order implicit `solve` mode. The
optional fused Warp FK operation keeps Torch tensors at its boundary and
recomputes its VJP through the Torch oracle.

**Batched by default.** Supported tensor routines use leading batch axes
`(B..., feature)`. A single pose is `(7,)`; a batch of one is `(1, 7)`.
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

**One forward direction for optimisation.**
IK and knot trajectory optimisation construct named-block `Problem` instances
with `VarSpec` variables, structural residuals, providers, and batched solver
state. Direct callers use that same construction and solver lifecycle; new
multi-block work belongs on this contract rather than a parallel abstraction.

**A whole-pass compute seam that does not leak runtime types.** The direct
Torch math and raw algorithm passes are the default. `ModelStructure` and
`ModelValues` give an opt-in kernel the same topology and tensor inputs. The
current public selector is the pass-specific
`forward_kinematics(..., use_warp=True)` flag; there is no process-wide registry
or per-Lie-operation dispatch. Users still see `torch.Tensor` in and out.

## What "looks like a story" means in code

Once a model and target pose are available, solving and inspecting IK stays
at the task API:

```python
result = br.solve_ik(model, {"body_panda_hand": target_pose})

result.q  # (nq,) joint solution
result.frame_pose("body_panda_hand")  # (7,) SE(3) pose at the solution
```

The {doc}`../index` front page contains the complete Panda example and is
mechanically executed by the test suite. No solver or problem object needs
to be instantiated. The same task call accepts a G1 loaded with
`free_flyer=True`; its first seven configuration entries are the base pose
and the remaining 29 are articulated coordinates. `solve_ik` itself does
not need a separate fixed/floating mode flag.

## What ships today

Implemented and tested: forward kinematics; analytic Jacobians with an
unbatched central finite-difference fallback; the
residual library (pose / position / orientation, joint position
limits, rest, contact consistency, reference trajectories, velocity
and acceleration smoothness, time-indexed residuals); named-block LM and GN
plus a first-order `torch.optim` adapter; dense and block-banded linear
solvers (Cholesky and BandedCholesky); pluggable robust
kernels (L2, Huber, Cauchy, Tukey, Geman–McClure); batched IK on fixed and floating-base robots;
trajectory optimisation with knot parameterisation and automatic banded/dense
routing (the Euclidean B-spline basis remains a numerical utility, not a
robot-manifold map);
Featherstone dynamics (RNEA, ABA, CRBA, CCRBA), centroidal momentum,
and the autograd-derived `compute_rnea_derivatives`,
`compute_aba_derivatives`, and `compute_crba_derivatives` helpers; URDF and MJCF parsers; a programmatic
`ModelBuilder`; a CUDA-validated, explicitly selected fused Warp FK lane
whose backward recomputes the Torch oracle; and a viewer with skeleton and
URDF-mesh render modes, draggable IK target gizmos, and trajectory playback.
The public solver loop remains eager and has no end-to-end IK capture
benchmark.

A handful of named symbols are deliberately stubbed — they have the
correct signatures, raise `NotImplementedError`, and are listed in
{doc}`/reference/roadmap`. They exist as targets so user code can
already reach for them and tests can already assume them.

## What we deliberately do not do

- We do not ship a physics engine. `integrate_q` is live, while
  `semi_implicit_euler`, `symplectic_euler`, and `rk4` are explicit stubs.
  Applications that need simulation must supply that layer separately.
- We do not ship optimal control or action-model skeletons. A future DDP/iLQR
  milestone should introduce that boundary together with executable behavior.
- We do not ship anatomical joints, muscles, or SMPL parsing in core.
  The programmatic builder can construct the repository's SMPL-like test model,
  but no `[human]` runtime extra is declared by this package. Core BetterRobot
  does not import `chumpy`, SMPL, or OpenSim.
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
  rigid-body state manifolds; removed action-model placeholders are documented
  there as future work.
- {doc}`residuals_and_costs` covers structural residuals and explicit problem
  composition.
- {doc}`solver_stack` covers the named-block `Problem` and solver lifecycle.
- {doc}`tasks` is `solve_ik`, `solve_trajopt`, and `Trajectory`.
- {doc}`collision_and_geometry` records the reserved, currently stubbed
  collision surface.
- {doc}`batching_and_backends` pins the tensor/device conventions and
  explains the structure/value seam and whole-pass lane choice.
- {doc}`viewer` is the viser-backed visualisation layer.

The {doc}`/conventions/index` chapters complement the book: they are
the rules contributors must follow when extending the library.
