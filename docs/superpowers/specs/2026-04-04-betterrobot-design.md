# BetterRobot Design Spec
**Date:** 2026-04-04
**Status:** Approved

---

## Overview

BetterRobot is a PyTorch-native Python library for robot kinematics, inverse kinematics, trajectory optimization, and motion retargeting. It is designed as a research tool accessible to students, inspired by PyRoki but built on PyTorch + PyPose instead of JAX. The core design goal is a layered, extensible architecture so future modules (dynamics, learned policies, richer collision, sampling-based planners) can be added without touching existing layers.

---

## Target Users

- Robotics researchers and PhD students
- Users already familiar with PyTorch
- Anyone needing a clean, hackable alternative to PyRoki without JAX friction

---

## Key Improvements Over PyRoki

| | PyRoki | BetterRobot |
|---|---|---|
| Backend | JAX | PyTorch + PyPose |
| Solvers | LM only | LM, GN, Adam, LBFGS — swappable |
| DL integration | Requires JAX↔PyTorch bridge | Native PyTorch |
| GPU setup | Fragile (jaxlib version pinning) | PyTorch CUDA (stable) |
| Install | jaxls from git (not PyPI) | 100% PyPI |
| Public API | `pks.solve_ik` (separate snippets package) | `br.solve_ik` (built-in flat API) |
| Extensibility | Flat costs.py | Layered — new modules don't touch core |

---

## Architecture: Layered Module Design

Three explicit layers with strict dependency direction: Tasks → Solvers → Core. Future modules extend Tasks and Costs only.

```
Core layer      — Robot, FK, Lie group ops, URDF parsing (no solver dependency)
Solver layer    — Solver ABC + LM, GN, Adam, LBFGS implementations
Costs layer     — Pure differentiable residual functions
Tasks layer     — High-level solve_ik, solve_trajopt, retarget APIs
```

---

## Project Structure

```
BetterRobot/
├── src/
│   └── better_robot/
│       ├── __init__.py               # Flat public API re-exports
│       ├── core/
│       │   ├── __init__.py
│       │   ├── _robot.py             # Robot class, FK
│       │   ├── _lie_ops.py           # PyPose SE3/SO3 wrapper (swap point)
│       │   └── _urdf_parser.py       # JointInfo, LinkInfo from yourdfpy
│       ├── solvers/
│       │   ├── __init__.py
│       │   ├── _base.py              # Problem, CostTerm, Solver ABC
│       │   ├── _levenberg_marquardt.py  # pypose.optim.LevenbergMarquardt
│       │   ├── _gauss_newton.py         # pypose.optim.GaussNewton
│       │   ├── _adam.py                 # torch.optim.Adam wrapper
│       │   └── _lbfgs.py               # torch.optim.LBFGS wrapper
│       ├── costs/
│       │   ├── __init__.py
│       │   ├── _pose.py              # Pose residuals (end-effector targets)
│       │   ├── _limits.py            # Joint limit, velocity, acceleration
│       │   ├── _regularization.py    # Rest pose, smoothness
│       │   ├── _collision.py         # Self + world collision residuals
│       │   └── _manipulability.py    # Yoshikawa manipulability
│       ├── tasks/
│       │   ├── __init__.py
│       │   ├── _ik.py                # Single, bimanual, mobile IK
│       │   ├── _trajopt.py           # Trajectory optimization
│       │   └── _retarget.py          # Motion retargeting
│       ├── collision/
│       │   ├── __init__.py
│       │   ├── _geometry.py          # Sphere, Capsule, Box, HalfSpace, Heightmap
│       │   └── _robot_collision.py   # RobotCollision (sphere decomposition)
│       └── viewer/
│           ├── __init__.py
│           └── _visualizer.py        # Thin viser wrapper
├── examples/
│   ├── 01_basic_ik.py
│   ├── 02_bimanual_ik.py
│   ├── 03_trajopt.py
│   └── 04_retargeting.py
├── tests/
│   ├── test_robot.py
│   ├── test_solvers.py
│   └── test_costs.py
├── docs/
├── pyproject.toml
├── README.md
└── .gitignore
```

---

## Core Layer

### `_robot.py` — Robot class

- `Robot.from_urdf(urdf: yourdfpy.URDF) -> Robot` — loads kinematic tree
- `robot.forward_kinematics(cfg: torch.Tensor) -> torch.Tensor` — returns `(*batch, link_count, 7)` SE3 poses as wxyz+xyz
- `robot.joints` — JointInfo (names, limits, velocity limits, parent indices)
- `robot.links` — LinkInfo (names, parent joint indices)
- Internally uses `_lie_ops.py` for all SE3 composition

### `_lie_ops.py` — Lie group abstraction

Thin wrapper over PyPose SE3/SO3. All Lie group operations in the codebase go through this file. If a future switch to pure PyTorch is needed, only this file changes.

```python
# All ops go through here — never import pypose directly elsewhere
def se3_exp(tangent: torch.Tensor) -> torch.Tensor: ...
def se3_compose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...
def se3_inverse(t: torch.Tensor) -> torch.Tensor: ...
def se3_log(t: torch.Tensor) -> torch.Tensor: ...
```

---

## Solver Layer

### `_base.py` — Abstractions

```python
class CostTerm:
    residual_fn: Callable[[torch.Tensor], torch.Tensor]
    weight: float
    kind: Literal["soft", "constraint_leq_zero"]  # soft or hard constraint

class Problem(pp.module.System):    # pypose System subclass
    variables: torch.Tensor
    costs: list[CostTerm]
    def forward(self) -> torch.Tensor   # returns full residual vector

class Solver(ABC):
    def solve(self, problem: Problem, max_iter: int, **kwargs) -> torch.Tensor
```

### Concrete Solvers

- `LevenbergMarquardt` — wraps `pypose.optim.LevenbergMarquardt`. Default solver. Handles Lie group manifold updates natively via PyPose.
- `GaussNewton` — wraps `pypose.optim.GaussNewton`. Faster on well-conditioned problems.
- `AdamSolver` — wraps `torch.optim.Adam`. For learning-integrated pipelines and noisy objectives.
- `LBFGSSolver` — wraps `torch.optim.LBFGS`. For smooth trajectory smoothing objectives.

All four implement the same `Solver` ABC, so they are swappable via a string argument.

---

## Costs Layer

Pure differentiable functions. No solver dependency. Each returns a residual vector.

| Cost | File | Description |
|---|---|---|
| `pose_residual` | `_pose.py` | SE3 log-space error between actual and target link pose |
| `limit_residual` | `_limits.py` | Joint limit violation |
| `velocity_residual` | `_limits.py` | Joint velocity limit violation |
| `acceleration_residual` | `_limits.py` | 5-point stencil acceleration |
| `jerk_residual` | `_limits.py` | 7-point stencil jerk |
| `rest_residual` | `_regularization.py` | Bias toward rest pose |
| `smoothness_residual` | `_regularization.py` | Penalize config differences |
| `self_collision_residual` | `_collision.py` | Sphere-sphere self collision |
| `world_collision_residual` | `_collision.py` | Robot vs world geometry |
| `manipulability_residual` | `_manipulability.py` | Yoshikawa measure (inverse, to maximize) |

Costs can be used as **soft penalties** (minimized via least squares) or **hard constraints** (augmented Lagrangian, `kind="constraint_leq_zero"`).

---

## Public API

Everything re-exported flat from `src/better_robot/__init__.py`:

```python
import better_robot as br

# Load robot
robot = br.Robot.from_urdf(urdf)

# Solve IK — solver swappable
solution = br.solve_ik(
    robot=robot,
    target_link="panda_hand",
    target_pose=pose,          # pypose SE3 or (7,) tensor
    solver="lm",               # "lm" | "gn" | "adam" | "lbfgs"
    weights={"pose": 1.0, "limits": 0.1, "rest": 0.01},
)

# Trajectory optimization
traj = br.solve_trajopt(
    robot=robot,
    robot_coll=robot_coll,
    world_coll=world_coll,
    target_link="panda_hand",
    start_pose=start_pose,
    end_pose=end_pose,
    timesteps=50,
    dt=0.02,
    solver="lm",
)

# Motion retargeting
retargeted = br.retarget(
    source_robot=source,
    target_robot=target,
    source_motion=motion,      # (T, joints) tensor
    solver="lm",
)

# Collision
robot_coll = br.collision.RobotCollision.from_sphere_decomposition(...)
world_geom = br.collision.Box.from_extent(...)
```

---

## Dependencies

```toml
[project]
name = "better_robot"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0.0",
    "pypose>=0.6.0",
    "yourdfpy",
    "trimesh",
    "viser",
    "robot_descriptions",
    "numpy",
]
```

100% PyPI — no git-installed packages.

---

## What's Left for Future Modules

The layered architecture means the following can be added without touching Core or Solvers:

- **Rigid body dynamics** — add `costs/_dynamics.py` + `tasks/_dynamics.py`
- **Deep learning integration** — learned IK policies, neural residuals as cost terms
- **Richer collision** — mesh SDF, convex decomposition; swap `collision/` internals
- **Sampling-based planners** — RRT/PRM as separate `tasks/_plan.py`
- **Task hierarchy** — priority-weighted cost stacking in `_base.py`
- **Pure PyTorch backend** — replace `_lie_ops.py` only
