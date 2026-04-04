# BetterRobot — Claude Code Guide

## Project Overview

PyTorch-native library for robot kinematics and inverse kinematics. Currently implemented: forward kinematics, IK via Levenberg-Marquardt. Stubs exist for trajectory optimization, retargeting, other solvers (GN/Adam/LBFGS), and collision.

## Commands

```bash
uv run pytest tests/ -v          # run all tests
uv run python examples/01_basic_ik.py  # interactive IK demo (open http://localhost:8080)
```

## Architecture

```
src/better_robot/
  core/        — Robot, FK, URDF parsing, Lie ops (foundation — change with care)
  costs/       — Residual functions (pose, limits, rest, …)
  solvers/     — LM (working), GN/Adam/LBFGS (stubs)
  tasks/       — solve_ik (unified, fixed+floating base), IKConfig; solve_trajopt/retarget (stubs)
  collision/   — stubs
  viewer/      — viser wrapper
```

All Lie group operations go through `core/_lie_ops.py`. To swap backends, only that file needs changing.

## SE3 / Lie Algebra Convention (critical — never deviate)

| Object | Format | Notes |
|--------|--------|-------|
| SE3 pose | `[tx, ty, tz, qx, qy, qz, qw]` | PyPose native, scalar qw last |
| se3 tangent | `[tx, ty, tz, rx, ry, rz]` | PyPose native |
| Quaternion from PyPose | `so3.tensor()` → `[qx, qy, qz, qw]` | |
| Viser wxyz | `(w, x, y, z)` | scalar-first, convert before passing to SE3 |

Use `.tensor()` (not `.data`) when extracting values from PyPose LieTensors — `.data` detaches from the autograd graph.

## PyPose Usage

```python
import pypose as pp
pp.SE3(t)            # t: (..., 7) SE3 LieTensor
pp.se3(v)            # v: (..., 6) se3 LieTensor
pp.mat2SO3(R)        # R: (3,3) rotation matrix → SO3
a @ b                # SE3 composition
t.Inv()              # SE3 inverse
t.Log().tensor()     # SE3 → se3 tangent, grad-preserving
pp.se3(v).Exp().tensor()  # se3 tangent → SE3, grad-preserving
```

## Solver Notes

**Our LM** (`solvers/_lm.py`) — registered as `"lm"` (default):
- Uses `torch.func.jacrev` for autodiff mode (when `problem.jacobian_fn is None`)
- Uses `problem.jacobian_fn(x)` for analytic mode (set by `IKConfig(jacobian="analytic")`)
- Adaptive damping: `damping=1e-4`, `factor=2.0`, `reject=16`
- Clamps to `problem.lower_bounds / upper_bounds` after each accepted step

**PyPose LM** (`solvers/_levenberg_marquardt.py`) — registered as `"lm_pypose"`:
- Always uses PyPose autograd for J; ignores `jacobian_fn`
- Kept for benchmarking; no longer the default

**Jacobian mode** (`IKConfig.jacobian`):
- `"autodiff"` (default): `jacrev` — works for all cost terms including custom ones
- `"analytic"`: geometric body-frame Jacobian from `costs/_jacobian.py` — faster; supported for both fixed and floating base

**Floating-base base Jacobian** (the "smart trick"):
```python
T_ee_local = se3_compose(se3_inverse(base), fk_world[link_idx])
J_base = diag([pos_w]*3 + [ori_w]*3) * adjoint_se3(se3_inverse(T_ee_local)) * pose_weight
```
No extra FK call; `adjoint_se3` is in `core/_lie_ops.py`.

**Limit cost** (`costs/_limits.py`): uses `torch.clamp(..., min=0)` — zero within limits, penalizes only violations. Do NOT remove the clamp; un-clamped limit residuals act as a centering force that overwhelms the pose cost.

**Orientation weight** (`tasks/_ik.py`): `ori_weight=0.1` by default. The Panda's default config has a ~173° orientation relative to identity, which causes the SO3 log map to be near-singular. Reducing ori_weight keeps position as the dominant objective.

## IK Usage Pattern

```python
robot = br.Robot.from_urdf(urdf)

# Fixed-base IK
cfg = br.solve_ik(
    robot,
    targets={"panda_hand": target_pose},
    cfg=br.IKConfig(rest_weight=0.001),   # optional tuning
    initial_cfg=cfg,                       # warm start
    max_iter=20,
)

# Floating-base IK (humanoid)
base_pose, cfg = br.solve_ik(
    robot,
    targets={
        "left_rubber_hand":     p_lh,
        "right_rubber_hand":    p_rh,
        "left_ankle_roll_link": p_lf,
        "right_ankle_roll_link": p_rf,
    },
    initial_base_pose=base_pose,  # non-None triggers floating-base path
    initial_cfg=cfg,
    max_iter=20,
)
```

**Pose format**: `[tx, ty, tz, qx, qy, qz, qw]` (same SE3 convention for all poses).

**Dispatch**: `initial_base_pose is None` → fixed-base path; not None → floating-base path.

**Returns**: Fixed base returns `(n,)` cfg tensor. Floating base returns `(base_pose (7,), cfg (n,))`.

**Initial orientation**: initialize IK targets with the robot's FK orientation at the default config, not identity. Identity orientation causes ~173° error from the default config, which oscillates the solver.

```python
fk0 = robot.forward_kinematics(robot._default_cfg)
nat_q = fk0[robot.get_link_index("panda_hand"), 3:7]  # [qx,qy,qz,qw]
target = torch.cat([position, nat_q])
```

**Floating-base setup**:
```python
base_pose = torch.tensor([0., 0., 0.78, 0., 0., 0., 1.])  # G1 standing height
cfg = robot._default_cfg.clone()
fk0 = robot.forward_kinematics(cfg, base_pose=base_pose)  # get natural EE orientations
```

**Viser update**: `base_frame.position = base_pose[:3]` and
`base_frame.wxyz = (qw, qx, qy, qz)`. URDF meshes are parented to `/base`.

## Tests

Run with `uv run pytest tests/ -v`. All 34 tests must pass before any commit. Tests load the real Panda URDF via `robot_descriptions`.
