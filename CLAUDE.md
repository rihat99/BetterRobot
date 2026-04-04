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
  tasks/       — solve_ik (working), solve_trajopt/retarget (stubs)
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

**LM solver** (`solvers/_levenberg_marquardt.py`):
- Uses `pypose.optim.LevenbergMarquardt` with `vectorize=True` and `Adaptive` damping strategy
- `vectorize=True` works with our FK and is ~3.4x faster than `vectorize=False`
- Projects variables onto `problem.lower_bounds / upper_bounds` after each step (hard joint limit enforcement)
- `reject=16` (default) enables adaptive damping with up to 16 retries per step

**Limit cost** (`costs/_limits.py`): uses `torch.clamp(..., min=0)` — zero within limits, penalizes only violations. Do NOT remove the clamp; un-clamped limit residuals act as a centering force that overwhelms the pose cost.

**Orientation weight** (`tasks/_ik.py`): `ori_weight=0.1` by default. The Panda's default config has a ~173° orientation relative to identity, which causes the SO3 log map to be near-singular. Reducing ori_weight keeps position as the dominant objective.

## IK Usage Pattern

```python
robot = br.Robot.from_urdf(urdf)

# Cold start
cfg = br.solve_ik(robot, target_link="panda_hand", target_pose=pose, max_iter=50)

# Warm-started loop (interactive, ~10 iters is enough)
cfg = br.solve_ik(robot, "panda_hand", pose, initial_cfg=cfg, max_iter=10)
```

**target_pose format**: `[tx, ty, tz, qx, qy, qz, qw]`

**Initial orientation**: initialize the IK target with the robot's FK orientation at the default config, not identity. Identity orientation causes ~173° orientation error from the default config, which oscillates the solver.

```python
fk0 = robot.forward_kinematics(robot._default_cfg)
nat_q = fk0[robot.get_link_index("panda_hand"), 3:7]  # [qx,qy,qz,qw]
target = torch.cat([position, nat_q])
```

## Tests

Run with `uv run pytest tests/ -v`. All 24 tests must pass before any commit. Tests load the real Panda URDF via `robot_descriptions`.
