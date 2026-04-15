# BetterRobot — Claude Code Guide

## Project Overview

PyTorch-native, GPU-ready library for robot kinematics and optimization. Pinocchio-style Model/Data architecture, PyTorch autograd throughout. Single code path for fixed-base and floating-base (free-flyer) robots.

**Implemented:** forward kinematics, Jacobians (analytic + finite-diff), pose/position/orientation/limits/rest residuals, CostStack, LM optimizer, IK (fixed + floating base).
**Stubs:** dynamics (RNEA/ABA/CRBA/centroidal), trajectory optimization, retargeting, viewer.

## Commands

```bash
uv run pytest tests/ -v                        # run all tests (297 pass)
uv run python examples/01_basic_ik.py          # Panda IK demo
uv run python examples/02_g1_ik.py             # G1 humanoid floating-base IK
```

## Architecture

```
src/better_robot/
  lie/              — SE3/SO3 group operations, tangent algebra, hat/vee, right Jacobians
  spatial/          — 6D spatial algebra value types (Motion, Force, Inertia)
  data_model/       — Model (frozen), Data (workspace), Frame, Body, Joint, joint_models/
  kinematics/       — forward_kinematics(), compute_joint_jacobians(), get_frame_jacobian()
  dynamics/         — rnea, aba, crba stubs (raise NotImplementedError)
  residuals/        — Residual classes: PoseResidual, PositionResidual, OrientationResidual,
                      JointPositionLimit, RestResidual (all with analytic .jacobian())
  costs/            — CostStack
  optim/            — LeastSquaresProblem, LevenbergMarquardt, GaussNewton, Adam, LBFGS
  tasks/            — solve_ik(), IKCostConfig, OptimizerConfig, IKResult
  collision/        — geometry, pairs, RobotCollision (port of old capsule mode)
  io/               — load(), IRModel, parsers (URDF/MJCF), ModelBuilder
  viewer/           — Visualizer stub
  utils/            — batching, logging
```

**Dependency rule (never violate):** `lie → spatial → data_model → (kinematics, dynamics) → residuals → costs → optim → tasks → viewer`. `io` reads from `data_model` only; `collision` is parallel to `kinematics`.

## SE3 / Lie Algebra Convention (critical — never deviate)

| Object | Format | Notes |
|--------|--------|-------|
| SE3 pose | `[tx, ty, tz, qx, qy, qz, qw]` | 7-vector, scalar `qw` last |
| se3 tangent | `[tx, ty, tz, rx, ry, rz]` | 6-vector, linear first |
| Quaternion | `[qx, qy, qz, qw]` | scalar last |
| Spatial Jacobian rows | `[v_lin (3), omega (3)]` | linear block first |

PyPose is confined to `lie/_pypose_backend.py` only. All other modules use `lie/se3.py` and `lie/so3.py` wrapper functions.

## Jacobian Conventions

`get_frame_jacobian` returns the **LOCAL_WORLD_ALIGNED** Jacobian:
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

`residual_jacobian` uses central finite differences (NOT `torch.autograd.functional.jacobian`) because PyPose's `SE3.Log().backward()` has an incorrect factor-of-2 in the quaternion gradient. FD eps: `1e-3` for float32, `1e-7` for float64.

## Public API

```python
import better_robot as br

# Load robot (URDF path, yourdfpy.URDF object, or callable builder)
model = br.load("panda.urdf")
model = br.load(urdf_obj)                          # yourdfpy.URDF
model = br.load("g1.urdf", free_flyer=True)        # floating-base

# Forward kinematics
data = br.forward_kinematics(model, q)             # (nq,) or (B, nq)
data = br.forward_kinematics(model, q, compute_frames=True)  # also fills oMf

# Jacobians
br.compute_joint_jacobians(model, data)            # fills data.J
J = br.get_frame_jacobian(model, data, frame_id)   # (B..., 6, nv)
J = br.get_joint_jacobian(model, data, joint_id)   # (B..., 6, nv)

# IK
from better_robot.tasks.ik import IKCostConfig, OptimizerConfig, solve_ik
result = br.solve_ik(model, {"panda_hand": target_pose})
result.q        # (nq,) solution
result.fk()     # Data with FK at solution
result.frame_pose("panda_hand")  # (7,) pose
```

## IK API

```python
result = solve_ik(
    model,
    targets={"frame_name": T_target},   # frame name → (7,) SE3 pose
    initial_q=q,                         # optional; defaults to model.q_neutral
    cost_cfg=IKCostConfig(
        pos_weight=1.0,
        ori_weight=1.0,
        pose_weight=1.0,
        limit_weight=0.1,
        rest_weight=0.01,
    ),
    optimizer_cfg=OptimizerConfig(
        optimizer="lm",                  # "lm" | "gn" | "adam" | "lbfgs"
        max_iter=100,
        jacobian_strategy=JacobianStrategy.AUTO,
    ),
)
```

**Floating-base robots:** load with `free_flyer=True`. The first 7 DOF of `q` are `[tx, ty, tz, qx, qy, qz, qw]` for the base pose. `solve_ik` handles this transparently — no `initial_base_pose` argument.

**Joint limits:** Panda joint 4 upper limit is −0.07 rad (range `[-3.07, -0.07]`). `q_neutral` has joint 4 = 0 which is outside bounds. Use `q_neutral.clamp(model.lower_pos_limit, model.upper_pos_limit)` as the starting point.

## Model Attributes

```python
model.nq          # configuration space dimension
model.nv          # tangent space dimension (= nq for revolute; 6 for free-flyer)
model.njoints     # number of joints (including universe joint 0)
model.nbodies     # number of bodies
model.nframes     # number of frames
model.q_neutral   # neutral configuration (nq,)
model.lower_pos_limit  # (nv,) lower joint limits
model.upper_pos_limit  # (nv,) upper joint limits
model.frame_id("name")  # → int
model.integrate(q, dv)  # SE3-aware retraction: q ⊕ dv
```

## LM Solver Notes

- Adaptive damping: starts at `1e-4`, doubles on reject, halves on accept.
- After each accepted step: clamps `x_new` to `[lower, upper]`.
- Initial `x0` is **not** clamped — caller must provide feasible `x0` if limits matter.

## Batching Rules

All tensors carry a leading batch dimension. No "unbatched mode" — single poses are `(1, nq)`. No `if x.dim() == 1:` branches in hot paths. Shape convention: `(B..., feature)` for configs, `(B..., njoints, 7)` for FK output, `(B..., 6, nv)` for Jacobians.

## torch.compile Friendliness

- Loops over `model.topo_order` are static (unroll cleanly)
- No Python branching on tensor values
- No `.item()` calls in hot paths
- Joint-type dispatch at compile time (tuple lookup, not tensor operation)

## Tests

```bash
uv run pytest tests/ -v    # 297 tests, all must pass
```

Tests use real Panda URDF via `robot_descriptions`. No mocking of FK or URDF parsing.
`test_layer_dependencies.py` enforces the dependency DAG via AST parsing.
`test_public_api.py` enforces exactly 25 symbols in `__all__`.
