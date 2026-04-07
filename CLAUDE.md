# BetterRobot — Claude Code Guide

## Project Overview

PyTorch-native library for robot kinematics and optimization. Layered, modular architecture inspired by Pinocchio (Model/Data), IsaacLab (registry/config), and PyRoki (functional residuals).

**Implemented:** forward kinematics, IK (fixed + floating base, autodiff + analytic Jacobian), LM solver, GN/Adam/LBFGS solvers, all cost terms (pose, limits, rest, collision, manipulability), RobotCollision (capsule mode from URDF + sphere mode legacy), collision distance functions.
**Stubs:** trajectory optimization, retargeting.

## Commands

```bash
uv run pytest tests/ -v                        # run all tests
uv run python examples/01_basic_ik.py          # Panda IK demo (http://localhost:8080)
uv run python examples/02_g1_ik.py             # G1 humanoid floating-base IK
```

## Architecture

```
src/better_robot/
  math/           — SE3 ops, SO3 ops, adjoint, quaternion utils  (no internal deps)
  models/         — RobotModel, JointInfo, LinkInfo, parsers/load_urdf()
  algorithms/
    kinematics/   — forward_kinematics(), compute_jacobian(), get_chain()
    geometry/     — CollGeom, Sphere, Capsule, Box, HalfSpace, distance functions, RobotCollision
  costs/          — CostTerm, residual + factory functions (pose, limits, rest, collision, manipulability)
  solvers/        — Registry, Problem, Solver ABC, LM, PyPose LM, GN, Adam, LBFGS
  tasks/
    ik/           — solve_ik(), IKConfig, IKVariable (fixed + floating base unified)
    trajopt/      — stub with TrajOptConfig
    retarget/     — stub with RetargetConfig
  viewer/         — Visualizer (viser wrapper), helpers
```

**Dependency rule (never violate):** `math → models → algorithms → costs → solvers → tasks → viewer`.
No layer may import from a layer above it.

All Lie group operations go through `math/se3.py`, `math/spatial.py`, and `math/so3.py`. PyPose is now confined to `math/se3.py` and `math/so3.py` only — all other modules use `math/so3` functions instead of importing pypose directly.

## SE3 / Lie Algebra Convention (critical — never deviate)

| Object | Format | Notes |
|--------|--------|-------|
| SE3 pose | `[tx, ty, tz, qx, qy, qz, qw]` | PyPose native, scalar qw last |
| se3 tangent | `[tx, ty, tz, rx, ry, rz]` | PyPose native |
| Quaternion from PyPose | `so3.tensor()` → `[qx, qy, qz, qw]` | |
| Viser wxyz | `(w, x, y, z)` | scalar-first, convert via `viewer/helpers.py` |

Use `.tensor()` (not `.data`) when extracting values from PyPose LieTensors — `.data` detaches from the autograd graph.

## PyPose Usage

```python
import pypose as pp
pp.SE3(t)                 # t: (..., 7) SE3 LieTensor
pp.se3(v)                 # v: (..., 6) se3 LieTensor
pp.mat2SO3(R)             # R: (3,3) rotation matrix → SO3
a @ b                     # SE3 composition
t.Inv()                   # SE3 inverse
t.Log().tensor()          # SE3 → se3 tangent, grad-preserving
pp.se3(v).Exp().tensor()  # se3 tangent → SE3, grad-preserving
```

PyPose is only imported in `math/se3.py`, `math/so3.py`, and the benchmarking solver `solvers/levenberg_marquardt_pypose.py`. All other modules use `math/so3` and `math/se3` wrapper functions.

## Public API

```python
import better_robot as br

# Load robot
model = br.load_urdf(urdf)          # yourdfpy.URDF → RobotModel

# Forward kinematics
fk = br.forward_kinematics(model, cfg)          # free function (canonical implementation)
fk = model.forward_kinematics(cfg)              # convenience method (thin delegate)

# IK
cfg = br.solve_ik(model, targets={"panda_hand": pose})
base_pose, cfg = br.solve_ik(model, targets={...}, initial_base_pose=base)

# Jacobian
J = br.compute_jacobian(model, cfg, link_idx, target_pose, pos_w, ori_w)
```

## Solver Notes

**Our LM** (`solvers/levenberg_marquardt.py`) — registered as `"lm"` (default):
- `jacobian_fn=None` → uses `torch.func.jacrev` (autodiff)
- `jacobian_fn=fn` → calls `fn(x)` directly (analytic)
- Adaptive damping: `damping=1e-4`, `factor=2.0`, `reject=16`
- Clamps to `problem.lower_bounds / upper_bounds` after each accepted step

**PyPose LM** (`solvers/levenberg_marquardt_pypose.py`) — registered as `"lm_pypose"`:
- Always uses PyPose autograd for J; ignores `jacobian_fn`
- Kept for benchmarking only

**GaussNewton** (`solvers/gauss_newton.py`) — registered as `"gn"`:
- Same structure as LM with `lam=0`; tiny regularization `1e-8 * I` for rank-deficiency
- Supports `jacobian_fn` (analytic or autodiff via `jacrev`)
- Bounds clamped after each step

**AdamSolver** (`solvers/adam.py`) — registered as `"adam"`:
- `torch.optim.Adam([x], lr=lr)` with scalar loss `0.5 * ‖r‖²`
- Default `lr=1e-2`; bounds clamped via `x.data.clamp_()` after each step

**LBFGSSolver** (`solvers/lbfgs.py`) — registered as `"lbfgs"`:
- `torch.optim.LBFGS` with `strong_wolfe` line search, `max_iter=20` per outer step
- Default `lr=1.0`; bounds clamped after each outer step

**Registry usage:**
```python
from better_robot.solvers import SOLVERS
solver = SOLVERS.get("lm")(damping=1e-3)
result = solver.solve(problem, max_iter=50)
solver_gn = SOLVERS.get("gn")()
solver_adam = SOLVERS.get("adam")(lr=5e-3)
solver_lbfgs = SOLVERS.get("lbfgs")(lr=0.5)
```

**Jacobian mode** (`IKConfig.jacobian`):
- `"autodiff"` (default): `jacrev` — works for all cost terms including custom ones
- `"analytic"`: geometric body-frame Jacobian — faster; supported for both fixed and floating base

**Floating-base base Jacobian** (the "smart trick"):
```python
T_ee_local = se3_compose(se3_inverse(base), fk_world[link_idx])
J_base = diag([pos_w]*3 + [ori_w]*3) * adjoint_se3(se3_inverse(T_ee_local)) * pose_weight
```
No extra FK call needed; reuses the FK already computed by `_fb_residual`.

**Limit cost** (`costs/limits.py`): uses `torch.clamp(..., min=0)` — zero within limits, penalizes only violations. **Do NOT remove the clamp** — un-clamped residuals act as a centering force that overwhelms the pose cost (18 limit dims vs 6 pose dims).

**Orientation weight** (`IKConfig.ori_weight=0.1`): intentionally low. The Panda's default config has ~173° orientation from identity, which puts the SO3 log map near its singularity. Higher `ori_weight` causes oscillation.

## IK Usage Pattern

```python
model = br.load_urdf(urdf)

# Capsule self-collision (recommended) — auto-generated from URDF
from better_robot.algorithms.geometry.robot_collision import RobotCollision
robot_coll = RobotCollision.from_urdf(urdf, model)   # capsule mode

# Fixed-base IK with collision
cfg = br.solve_ik(
    model,
    targets={"panda_hand": target_pose},
    config=br.IKConfig(rest_weight=0.001),
    initial_q=cfg,
    max_iter=20,
    robot_coll=robot_coll,  # optional; omit to disable collision avoidance
)

# Fixed-base IK (no collision)
cfg = br.solve_ik(
    model,
    targets={"panda_hand": target_pose},
    cfg=br.IKConfig(rest_weight=0.001),
    initial_cfg=cfg,
    max_iter=20,
)

# Floating-base IK (humanoid)
base_pose, cfg = br.solve_ik(
    model,
    targets={
        "left_rubber_hand":     p_lh,
        "right_rubber_hand":    p_rh,
        "left_ankle_roll_link": p_lf,
        "right_ankle_roll_link": p_rf,
    },
    initial_base_pose=base_pose,
    initial_cfg=cfg,
    max_iter=20,
)
```

**Dispatch**: `initial_base_pose is None` → fixed-base; not None → floating-base via `_solve_floating()`.
**Returns**: fixed base `(n,)` tensor; floating base `(base_pose (7,), cfg (n,))`.

**IKVariable**: floating-base IK uses `IKVariable(cfg, base_pose)` to bundle the joint config and base pose. Flat representation is `[cfg (n,), base_tangent (6,)]`. Base updated via SE3 retraction: `new_base = se3_exp(delta_base) @ base`.

**Initial orientation tip**: use robot's FK orientation at default config, not identity quaternion:
```python
fk0 = model.forward_kinematics(model._default_cfg)
nat_q = fk0[model.link_index("panda_hand"), 3:7]
target = torch.cat([position, nat_q])
```

## Tests

```bash
uv run pytest tests/ -v    # all tests must pass
```

Tests use real Panda URDF via `robot_descriptions`. No mocking of FK or URDF parsing.
