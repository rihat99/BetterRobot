# BetterRobot — Claude Code Guide

## Project Overview

PyTorch-native, GPU-ready library for robot kinematics and optimization. Pinocchio-style Model/Data architecture with path-specific PyTorch autograd coverage. Single code path for fixed-base and floating-base (free-flyer) robots.

**Implemented:** forward kinematics, Jacobians (analytic + central finite-difference fallback), pose/position/orientation/limits/rest/smoothness/contact-consistency/reference-trajectory residuals, CostStack, legacy LM/GN/Adam/LBFGS/MultiStage plus named-block LM/GN/Adam/phases, batched IK (fixed + floating base), knot-based trajectory optimisation with dense/banded/operator routing (`solve_trajopt`; the Euclidean B-spline basis remains a numerical utility only), contact-force fitting, dynamics (RNEA/ABA/CRBA/CCRBA, centroidal map + momentum, and the autograd-derived `compute_rnea_derivatives`, `compute_aba_derivatives`, and `compute_crba_derivatives` helpers), viewer V1 (Skeleton, URDFMesh, Grid, FrameAxes, Targets, ForceVectors, ViserBackend, build_joint_panel, minimal TrajectoryPlayer).
**Open work:** dynamic integrators (`semi_implicit_euler` / `symplectic_euler` / `rk4`), `compute_minverse`, `compute_coriolis_matrix`, analytic Carpentier–Mansard derivatives, jerk / Yoshikawa / collision / nullspace residuals, unshipped viewer COM/PathTrace/ResidualPlot and recording surfaces, and Warp whole-pass kernels beyond the CUDA-validated opt-in FK lane. See `docs/reference/roadmap.md`.

The named-block optimization layer is also implemented: `VarSpec`/`Values`/
`Problem`, `Euclidean`/`SO3Manifold`/`SE3Manifold`/`RobotConfig` manifolds,
state-space feasible retraction, eliminated tangent masks, evaluation-local
provider DAGs, tangent gradients, and analytic/`jacrev`/`jacfwd` Jacobian
blocks. A `time_axis=0` block plus residual `TemporalPattern` declarations can
assemble block-banded normals or an explicit normal operator. Named-block LM
routes between dense `Cholesky`, `BandedCholesky`, and `NormalCG`; matrix-free
batched Adam and functional phases share the same problem surface. Solver
state is tensor-only per element with robust groups, per-block step caps, and
projected active-set bounds. `solve_ik` and knot `solve_trajopt` use this
stack; the flat optimizer contract remains for direct compatibility callers.

## Commands

```bash
uv run pytest tests/ -v                        # run all tests
uv run python examples/01_basic_ik.py          # Panda IK demo
uv run python examples/02_g1_ik.py             # G1 humanoid floating-base IK
```

## Architecture

```
src/better_robot/
  lie/              — direct Torch SE3/SO3 ops, tangent algebra, hat/vee, right Jacobians, typed `SE3`/`SO3`/`Pose`
  spatial/          — 6D spatial algebra value types (Motion, Force, Inertia)
  data_model/       — Model/Data plus ModelStructure, ModelValues, ExecutionBatch, Frame, Body, Joint, joint_models/
  kinematics/       — Torch raw FK + public wrappers; whole-pass kernel lanes live beside their Torch counterparts
  dynamics/         — Torch raw rigid-body passes + public wrappers; optional whole-pass kernels stay local
  residuals/        — Residual classes (Pose / Position / Orientation / JointPositionLimit / Rest /
                      Velocity / Acceleration / TimeIndexed / ContactConsistency /
                      ReferenceTrajectory; analytic `.jacobian()` plus legacy
                      `apply_jac_transpose` overrides retained for test coverage)
  costs/            — forwarding compatibility imports for optim.cost_stack
  optim/            — legacy CostStack/LeastSquaresProblem + direct-use LM/GN/Adam/LBFGS/MultiStage;
                      named-block Problem + dense/banded/operator batched Adam/LM/GN/phases
  tasks/            — solve_ik(), solve_trajopt(), solve_contact_forces(), Trajectory, KnotTrajectory, BSplineTrajectory
  collision/        — reserved geometry/pair/RobotCollision surface; current implementations are stubbed
  io/               — load(), internal IRModel, parsers (URDF/MJCF), ModelBuilder, AssetResolver + concrete resolvers
  viewer/           — Visualizer, Scene, SkeletonMode, URDFMeshMode, ForceVectorsOverlay, …
```

**Dependency rule (never violate):** `lie → spatial → data_model → kinematics → dynamics → residuals → optim → tasks → viewer`, with kinematics and dynamics assigned the same contract rank so their current one-way reuse (`dynamics` imports raw FK helpers) is permitted. The legacy `costs` import path is part of the `optim` layer. `io` reads from `data_model` only; `collision` is parallel to `kinematics`. Enforced by `tests/contract/test_layer_dependencies.py`.

The compute seam is whole-pass: `ModelStructure` provides validated static/device topology, `ModelValues` provides the tensor pytree, and raw Torch passes are the default correctness lane. An optional Warp kernel is selected explicitly at the FK/RNEA-style integration point only after eligibility and parity checks. Warp is not a library layer and does not replace individual Lie operations.

## SE3 / Lie Algebra Convention (critical — never deviate)

| Object | Format | Notes |
|--------|--------|-------|
| SE3 pose | `[tx, ty, tz, qx, qy, qz, qw]` | 7-vector, scalar `qw` last |
| se3 tangent | `[tx, ty, tz, rx, ry, rz]` | 6-vector, linear first |
| Quaternion | `[qx, qy, qz, qw]` | scalar last |
| Spatial Jacobian rows | `[v_lin (3), omega (3)]` | linear block first |

SE3/SO3 ops live in `lie/_impl.py` as direct pure-Torch formulas. Other modules use the `lie/se3.py` / `lie/so3.py` functional facades, which call that implementation directly. PyPose is no longer a dependency, and there is no runtime compute registry.

## Jacobian Conventions

`get_frame_jacobian` defaults to the **LOCAL_WORLD_ALIGNED** Jacobian; callers
can also request `WORLD` or `LOCAL`:
- Linear rows: velocity of the frame origin expressed in world frame
- Angular rows: angular velocity in world frame

To get the body-frame Jacobian from LOCAL_WORLD_ALIGNED:
```python
# Correct (only rotate, don't apply full adjoint):
R_ee = so3.to_matrix(T_ee[..., 3:])   # (B..., 3, 3)
J_local = cat([R_ee.mT @ J_world[:3, :], R_ee.mT @ J_world[3:, :]])

# Wrong (adds spurious cross-term when J is LWA, not WORLD):
J_local = se3.adjoint_inv(T_ee) @ J_world
```

`compute_joint_jacobians` returns the WORLD-frame Jacobian (velocity at world origin).

## Autodiff / Finite-Diff Note

`residual_jacobian` uses central finite differences as the AUTO fallback when no analytic Jacobian is registered. FD is retained as the independent legacy fallback across joint kinds. Its epsilons are `1e-3` for float32 and `1e-7` for float64.

That paragraph describes the **legacy** residual path. The named-block
`Problem` uses analytic blocks or real `torch.func.jacrev`/`jacfwd`; finite
differences are an explicit debug strategy only. Its masks remove fixed
tangent columns rather than leaving zeros, and its providers cache graph-bearing
work only for one evaluation.

## Public API

```python
import better_robot as br
from robot_descriptions import panda_description

# Load robot (URDF path, yourdfpy.URDF object, or callable builder)
model = br.load(panda_description.URDF_PATH)
model = br.load(urdf_obj)                          # yourdfpy.URDF
model = br.load("g1.urdf", free_flyer=True)        # floating-base

# Forward kinematics
data = br.forward_kinematics(model, q)             # (nq,) or (B, nq)
data = br.forward_kinematics(model, q, compute_frames=True)  # also fills frame_pose_world

# Jacobians
br.compute_joint_jacobians(model, data)            # fills data.joint_jacobians
J = br.get_frame_jacobian(model, data, frame_id)   # (B..., 6, nv)
J = br.get_joint_jacobian(model, data, joint_id)   # (B..., 6, nv)

# IK
from better_robot.tasks.ik import IKCostConfig, OptimizerConfig, solve_ik
result = br.solve_ik(model, {"body_panda_hand": target_pose})
result.q        # (B..., nq) solution
result.fk()     # Data with FK at solution
result.frame_pose("body_panda_hand")  # (7,) pose
```

## IK API

```python
result = solve_ik(
    model,
    targets={"frame_name": T_target},   # frame name → (B..., 7) SE3 pose
    initial_q=q,                         # optional; defaults to model.q_neutral
    cost_cfg=IKCostConfig(
        pos_weight=1.0,
        ori_weight=1.0,
        pose_weight=1.0,
        limit_weight=0.1,
        rest_weight=0.01,
    ),
    optimizer_cfg=OptimizerConfig(
        optimizer="lm",                  # "lm" | "gn" | "adam" | "lm_then_adam"
        max_iter=100,
        jacobian_strategy=JacobianStrategy.AUTO,
    ),
)
```

**Floating-base robots:** load with `free_flyer=True`. The first 7 DOF of `q` are `[tx, ty, tz, qx, qy, qz, qw]` for the base pose. `solve_ik` handles this transparently — no `initial_base_pose` argument.

**Joint limits:** Panda joint 4 upper limit is −0.07 rad (range `[-3.07, -0.07]`). `q_neutral` has joint 4 = 0, so the IK preset projects its seed before strict named-block validation. Direct block callers must supply feasible `Values` themselves.

## Model Attributes

```python
model.nq          # configuration space dimension
model.nv          # tangent-space dimension (a free-flyer contributes nq=7, nv=6)
model.njoints     # number of joints (including universe joint 0)
model.nbodies     # number of bodies
model.nframes     # number of frames
model.q_neutral   # neutral configuration (nq,)
model.lower_pos_limit  # (nq,) lower joint limits
model.upper_pos_limit  # (nq,) upper joint limits
model.frame_id("name")  # → int
model.integrate(q, dv)  # SE3-aware retraction: q ⊕ dv
```

## Legacy LM Solver Notes

- Adaptive damping: starts at `1e-4`, doubles on reject, halves on accept.
- Every LM trial point is clamped to `[lower, upper]` before residual evaluation.
- Bounds have no active-set, projected-gradient, or KKT treatment; an active-bound run can exit `maxiter` with error remaining.
- Initial `x0` is **not** clamped — caller must provide feasible `x0` if limits matter.

These notes describe `optim.optimizers.LevenbergMarquardt`, not the public
named-block `optim.LevenbergMarquardt`. The named-block solver uses
`init_state`/pure `update`/`finalize` or detached `run`, keeps one tensor status
and damping value per batch element, applies grouped robust kernels, and uses
projected-gradient KKT termination for supported state-space bounds. Fixed
groups of named-block LM updates are capture-certified by the private M6
`GraphExecutor` CUDA harness. That evidence does not certify public `run`,
Adam, end-to-end IK, or an arbitrary custom residual.

## Batching Rules

Tensor math such as FK, residuals, and analytic Jacobians accepts arbitrary
leading batch dimensions. Named-block `Problem` also supports leading-batch
residual/objective evaluation, tangent gradients, Jacobian blocks, dense
assembly, temporal bands, and normal operators. Named-block LM/GN preserves
those axes in per-element damping, accept/reject, factorization, linear-solve
status, and convergence tensors.

The legacy optimizer stack remains a single-problem path invoked through an
optimizer's `minimize` method, while named-block consumers use `run`.
`solve_ik` and `solve_trajopt` use named blocks and accept arbitrary common
leading batch axes with per-element diagnostics. `solve_trajopt` also reports
the requested/used linearization plus its stable reason/detail. Other
named-block consumers call `optim.LevenbergMarquardt().run(values, problem)`
directly.
`ResidualItem.kernel`/`group_size` are active in that solver, scalar objectives
are for consumer-owned first-order/manual loops and are rejected at GN/LM
boundaries, state-space `Bounds` are not tangent-space step bounds, and provider
caches never survive an evaluation.

## torch.compile Friendliness

- Loops over `model.topo_order` are static (unroll cleanly)
- Watched hot paths keep tensor decisions tensorized
- No unexempted `.item()` calls in the contract-linted hot paths
- Joint-type dispatch at compile time (tuple lookup, not tensor operation)
- Named-block LM `update` is fixed-shape and sync-free; `init_state` owns static
  validation and eager `run` may sync once per iteration for early exit

## Tests

```bash
uv run pytest tests/ -v   # all tests must pass
```

Tests use real Panda URDF via `robot_descriptions`. No mocking of FK or URDF parsing.
`tests/contract/test_layer_dependencies.py` enforces the dependency DAG via AST parsing.
`tests/contract/test_public_api.py` enforces the required top-level core and duplicate-free `__all__`
(25 required symbols, including `SE3` and `ModelBuilder`). `tests/contract/` carries the rest of
the AST + structural contract suite (cache invariants, optional
imports, no-legacy-strings, hot-path lint,
pluggable Protocols, solver state, naming, docstrings, submodule reachability).
