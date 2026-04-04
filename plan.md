# BetterRobot — Project Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold the BetterRobot Python package using uv — correct src layout, all module files as stubs, dependencies installed and importable.

**Architecture:** Layered src-layout package (`src/better_robot/`) with four layers: core → solvers → costs → tasks. Each module file is a placeholder stub (docstring + `__all__ = []`). Public API re-exported flat from top-level `__init__.py`.

**Tech Stack:** Python ≥3.10, uv (package manager), PyTorch ≥2.0, PyPose ≥0.6, yourdfpy, trimesh, viser, robot_descriptions, numpy.

---

## File Map

Files to create (all stubs unless noted as real content):

```
BetterRobot/
├── pyproject.toml                          ← real content (uv manages)
├── .python-version                         ← real content (uv manages)
├── .gitignore                              ← real content
├── README.md                               ← real content (one-liner)
├── plan.md                                 ← symlink / copy of this plan
├── src/
│   └── better_robot/
│       ├── __init__.py                     ← real re-exports (stubs imported)
│       ├── core/
│       │   ├── __init__.py
│       │   ├── _robot.py
│       │   ├── _lie_ops.py
│       │   └── _urdf_parser.py
│       ├── solvers/
│       │   ├── __init__.py
│       │   ├── _base.py
│       │   ├── _levenberg_marquardt.py
│       │   ├── _gauss_newton.py
│       │   ├── _adam.py
│       │   └── _lbfgs.py
│       ├── costs/
│       │   ├── __init__.py
│       │   ├── _pose.py
│       │   ├── _limits.py
│       │   ├── _regularization.py
│       │   ├── _collision.py
│       │   └── _manipulability.py
│       ├── tasks/
│       │   ├── __init__.py
│       │   ├── _ik.py
│       │   ├── _trajopt.py
│       │   └── _retarget.py
│       ├── collision/
│       │   ├── __init__.py
│       │   ├── _geometry.py
│       │   └── _robot_collision.py
│       └── viewer/
│           ├── __init__.py
│           └── _visualizer.py
├── examples/
│   ├── 01_basic_ik.py
│   ├── 02_bimanual_ik.py
│   ├── 03_trajopt.py
│   └── 04_retargeting.py
└── tests/
    ├── __init__.py
    ├── test_imports.py                     ← verifies all stubs importable
    ├── test_robot.py
    ├── test_solvers.py
    └── test_costs.py
```

---

## Task 1: Initialize uv project

**Files:**
- Create: `pyproject.toml` (managed by uv)
- Create: `.python-version`
- Create: `.gitignore`

- [ ] **Step 1: Initialize uv library project**

Run from the `BetterRobot/` directory (it already exists, so use `--no-workspace` and init in-place):

```bash
cd /Users/rikhat.akizhanov/Desktop/cources/PhD/ROB803/assignment_3/BetterRobot
uv init --lib --name better-robot --python 3.11
```

Expected output:
```
Initialized project `better-robot`
```

This creates `pyproject.toml`, `.python-version`, and `src/better_robot/__init__.py`.

- [ ] **Step 2: Verify uv created the src layout**

```bash
ls src/better_robot/
```

Expected: `__init__.py` (uv lib default)

- [ ] **Step 3: Replace pyproject.toml with correct content**

uv's default pyproject.toml needs our dependencies. Replace it:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "better-robot"
version = "0.1.0"
description = "PyTorch-native robot kinematics, IK, trajectory optimization, and retargeting"
readme = "README.md"
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

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff",
    "pyright",
]

[tool.hatch.build.targets.wheel]
packages = ["src/better_robot"]

[tool.ruff.lint]
select = ["E", "F", "PLC", "PLE", "PLR", "PLW"]
ignore = ["E501", "PLR0913", "PLR2004"]
```

- [ ] **Step 4: Create .gitignore**

```
__pycache__/
*.py[cod]
*.egg-info/
.venv/
dist/
build/
.pytest_cache/
.ruff_cache/
*.pyc
.DS_Store
```

- [ ] **Step 5: Create README.md**

```markdown
# BetterRobot

PyTorch-native library for robot kinematics, inverse kinematics, trajectory optimization, and motion retargeting.

```bash
pip install better-robot
```
```

- [ ] **Step 6: Install dependencies**

```bash
uv sync --extra dev
```

Expected: uv creates `.venv/`, downloads all dependencies. Should end with:
```
Installed N packages in Xs
```

- [ ] **Step 7: Verify PyTorch and PyPose are installed**

```bash
uv run python -c "import torch; import pypose; print(torch.__version__, pypose.__version__)"
```

Expected: prints version strings without errors.

---

## Task 2: Create core/ stubs

**Files:**
- Create: `src/better_robot/core/__init__.py`
- Create: `src/better_robot/core/_robot.py`
- Create: `src/better_robot/core/_lie_ops.py`
- Create: `src/better_robot/core/_urdf_parser.py`

- [ ] **Step 1: Create core/__init__.py**

```python
"""Core layer: Robot, forward kinematics, Lie group ops, URDF parsing."""

from ._robot import Robot as Robot
from ._urdf_parser import JointInfo as JointInfo, LinkInfo as LinkInfo

__all__ = ["Robot", "JointInfo", "LinkInfo"]
```

- [ ] **Step 2: Create _urdf_parser.py stub**

```python
"""URDF parsing utilities. Converts yourdfpy.URDF into JointInfo and LinkInfo dataclasses."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class JointInfo:
    """Stores joint metadata for the kinematic tree."""

    names: tuple[str, ...]
    """Ordered joint names (topological order)."""

    num_joints: int
    """Total number of joints (including fixed)."""

    num_actuated_joints: int
    """Number of actuated (non-fixed) joints."""

    lower_limits: torch.Tensor
    """Shape: (num_actuated_joints,). Joint lower limits in radians/meters."""

    upper_limits: torch.Tensor
    """Shape: (num_actuated_joints,). Joint upper limits in radians/meters."""

    velocity_limits: torch.Tensor
    """Shape: (num_actuated_joints,). Max joint velocity limits."""

    parent_indices: tuple[int, ...]
    """Parent joint index for each joint (-1 for root joints)."""

    twists: torch.Tensor
    """Shape: (num_joints, 6). Screw axis (twist) for each joint in parent frame."""

    parent_transforms: torch.Tensor
    """Shape: (num_joints, 7). SE3 transform from parent to joint frame (wxyz+xyz)."""


@dataclass
class LinkInfo:
    """Stores link metadata."""

    names: tuple[str, ...]
    """Ordered link names."""

    num_links: int
    """Total number of links."""

    parent_joint_indices: tuple[int, ...]
    """Index of the parent joint for each link (-1 for base link)."""


class RobotURDFParser:
    """Parses a yourdfpy.URDF into JointInfo and LinkInfo."""

    @staticmethod
    def parse(urdf: object) -> tuple[JointInfo, LinkInfo]:
        """Parse a yourdfpy.URDF object.

        Args:
            urdf: A loaded yourdfpy.URDF instance.

        Returns:
            Tuple of (JointInfo, LinkInfo).
        """
        raise NotImplementedError
```

- [ ] **Step 3: Create _lie_ops.py stub**

```python
"""Lie group operations wrapping PyPose SE3/SO3.

All Lie group operations in BetterRobot go through this module.
To switch to a pure-PyTorch backend, only this file needs changing.
"""

from __future__ import annotations

import torch


def se3_exp(tangent: torch.Tensor) -> torch.Tensor:
    """Map se(3) tangent vector to SE3 transform.

    Args:
        tangent: Shape (..., 6). Lie algebra element [translation, rotation].

    Returns:
        Shape (..., 7). SE3 transform as wxyz+xyz quaternion+translation.
    """
    raise NotImplementedError


def se3_compose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compose two SE3 transforms: result = a @ b.

    Args:
        a: Shape (..., 7). Left SE3 transform.
        b: Shape (..., 7). Right SE3 transform.

    Returns:
        Shape (..., 7). Composed SE3 transform.
    """
    raise NotImplementedError


def se3_inverse(t: torch.Tensor) -> torch.Tensor:
    """Invert an SE3 transform.

    Args:
        t: Shape (..., 7). SE3 transform.

    Returns:
        Shape (..., 7). Inverse SE3 transform.
    """
    raise NotImplementedError


def se3_log(t: torch.Tensor) -> torch.Tensor:
    """Map SE3 transform to se(3) tangent vector.

    Args:
        t: Shape (..., 7). SE3 transform.

    Returns:
        Shape (..., 6). Lie algebra element.
    """
    raise NotImplementedError


def se3_identity() -> torch.Tensor:
    """Return the SE3 identity transform.

    Returns:
        Shape (7,). Identity: wxyz=[1,0,0,0], xyz=[0,0,0].
    """
    raise NotImplementedError
```

- [ ] **Step 4: Create _robot.py stub**

```python
"""Robot class: loads URDF and runs forward kinematics."""

from __future__ import annotations

import torch
import yourdfpy

from ._urdf_parser import JointInfo, LinkInfo, RobotURDFParser


class Robot:
    """Differentiable robot kinematic tree backed by PyTorch."""

    joints: JointInfo
    links: LinkInfo

    def __init__(self, joints: JointInfo, links: LinkInfo) -> None:
        self.joints = joints
        self.links = links

    @staticmethod
    def from_urdf(
        urdf: yourdfpy.URDF,
        default_joint_cfg: torch.Tensor | None = None,
    ) -> "Robot":
        """Load a robot kinematic tree from a yourdfpy URDF.

        Args:
            urdf: Loaded yourdfpy.URDF instance.
            default_joint_cfg: Shape (num_actuated_joints,). Initial joint config.
                Defaults to midpoint of joint limits.

        Returns:
            Robot instance.
        """
        raise NotImplementedError

    def forward_kinematics(
        self,
        cfg: torch.Tensor,
    ) -> torch.Tensor:
        """Compute world poses of all links.

        Args:
            cfg: Shape (*batch, num_actuated_joints). Joint configuration.

        Returns:
            Shape (*batch, num_links, 7). SE3 poses (wxyz+xyz) for each link.
        """
        raise NotImplementedError

    def get_link_index(self, link_name: str) -> int:
        """Return the index of a link by name.

        Args:
            link_name: Name of the link as it appears in the URDF.

        Returns:
            Integer index into the link dimension of forward_kinematics output.

        Raises:
            ValueError: If link_name not found.
        """
        raise NotImplementedError
```

---

## Task 3: Create solvers/ stubs

**Files:**
- Create: `src/better_robot/solvers/__init__.py`
- Create: `src/better_robot/solvers/_base.py`
- Create: `src/better_robot/solvers/_levenberg_marquardt.py`
- Create: `src/better_robot/solvers/_gauss_newton.py`
- Create: `src/better_robot/solvers/_adam.py`
- Create: `src/better_robot/solvers/_lbfgs.py`

- [ ] **Step 1: Create solvers/__init__.py**

```python
"""Solver layer: swappable optimization backends."""

from ._base import CostTerm as CostTerm, Problem as Problem, Solver as Solver
from ._levenberg_marquardt import LevenbergMarquardt as LevenbergMarquardt
from ._gauss_newton import GaussNewton as GaussNewton
from ._adam import AdamSolver as AdamSolver
from ._lbfgs import LBFGSSolver as LBFGSSolver

SOLVER_REGISTRY: dict[str, type[Solver]] = {
    "lm": LevenbergMarquardt,
    "gn": GaussNewton,
    "adam": AdamSolver,
    "lbfgs": LBFGSSolver,
}

__all__ = [
    "CostTerm", "Problem", "Solver",
    "LevenbergMarquardt", "GaussNewton", "AdamSolver", "LBFGSSolver",
    "SOLVER_REGISTRY",
]
```

- [ ] **Step 2: Create _base.py stub**

```python
"""Base abstractions: CostTerm, Problem, Solver ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Literal

import torch


@dataclass
class CostTerm:
    """A single differentiable cost/constraint term."""

    residual_fn: Callable[[torch.Tensor], torch.Tensor]
    """Function mapping joint config to residual vector."""

    weight: float = 1.0
    """Scalar weight applied to the residual."""

    kind: Literal["soft", "constraint_leq_zero"] = "soft"
    """'soft': minimized via least squares.
    'constraint_leq_zero': enforced via augmented Lagrangian (residual <= 0)."""


@dataclass
class Problem:
    """Optimization problem: variables + list of cost terms."""

    variables: torch.Tensor
    """Initial values for the optimization variable (joint config or trajectory)."""

    costs: list[CostTerm] = field(default_factory=list)
    """List of cost/constraint terms."""

    def total_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate all weighted soft residuals into a single vector.

        Args:
            x: Current variable values.

        Returns:
            1D residual vector.
        """
        raise NotImplementedError


class Solver(ABC):
    """Abstract base class for all solvers."""

    @abstractmethod
    def solve(
        self,
        problem: Problem,
        max_iter: int = 100,
        **kwargs: object,
    ) -> torch.Tensor:
        """Run optimization and return the solution.

        Args:
            problem: Problem instance with variables and costs.
            max_iter: Maximum number of iterations.
            **kwargs: Solver-specific hyperparameters.

        Returns:
            Optimized variable tensor, same shape as problem.variables.
        """
        raise NotImplementedError
```

- [ ] **Step 3: Create _levenberg_marquardt.py stub**

```python
"""Levenberg-Marquardt solver via pypose.optim.LevenbergMarquardt."""

from __future__ import annotations

import torch

from ._base import Problem, Solver


class LevenbergMarquardt(Solver):
    """LM solver wrapping pypose.optim.LevenbergMarquardt.

    Default solver. Handles Lie group manifold updates natively.
    Suitable for IK and trajectory optimization.
    """

    def solve(
        self,
        problem: Problem,
        max_iter: int = 100,
        damping: float = 1e-4,
        **kwargs: object,
    ) -> torch.Tensor:
        """Run Levenberg-Marquardt optimization.

        Args:
            problem: Problem instance.
            max_iter: Maximum iterations.
            damping: Initial LM damping factor.

        Returns:
            Optimized variable tensor.
        """
        raise NotImplementedError
```

- [ ] **Step 4: Create _gauss_newton.py stub**

```python
"""Gauss-Newton solver via pypose.optim.GaussNewton."""

from __future__ import annotations

import torch

from ._base import Problem, Solver


class GaussNewton(Solver):
    """GN solver wrapping pypose.optim.GaussNewton.

    Faster than LM on well-conditioned problems (no damping).
    """

    def solve(
        self,
        problem: Problem,
        max_iter: int = 100,
        **kwargs: object,
    ) -> torch.Tensor:
        """Run Gauss-Newton optimization.

        Args:
            problem: Problem instance.
            max_iter: Maximum iterations.

        Returns:
            Optimized variable tensor.
        """
        raise NotImplementedError
```

- [ ] **Step 5: Create _adam.py stub**

```python
"""Adam solver wrapping torch.optim.Adam."""

from __future__ import annotations

import torch

from ._base import Problem, Solver


class AdamSolver(Solver):
    """Gradient descent solver using torch.optim.Adam.

    Best for noisy objectives or learning-integrated pipelines.
    """

    def solve(
        self,
        problem: Problem,
        max_iter: int = 500,
        lr: float = 1e-3,
        **kwargs: object,
    ) -> torch.Tensor:
        """Run Adam optimization.

        Args:
            problem: Problem instance.
            max_iter: Number of gradient steps.
            lr: Learning rate.

        Returns:
            Optimized variable tensor.
        """
        raise NotImplementedError
```

- [ ] **Step 6: Create _lbfgs.py stub**

```python
"""L-BFGS solver wrapping torch.optim.LBFGS."""

from __future__ import annotations

import torch

from ._base import Problem, Solver


class LBFGSSolver(Solver):
    """Quasi-Newton solver using torch.optim.LBFGS.

    Best for smooth objectives, e.g. trajectory smoothing.
    """

    def solve(
        self,
        problem: Problem,
        max_iter: int = 100,
        lr: float = 1.0,
        **kwargs: object,
    ) -> torch.Tensor:
        """Run L-BFGS optimization.

        Args:
            problem: Problem instance.
            max_iter: Maximum iterations.
            lr: Step size.

        Returns:
            Optimized variable tensor.
        """
        raise NotImplementedError
```

---

## Task 4: Create costs/ stubs

**Files:**
- Create: `src/better_robot/costs/__init__.py`
- Create: `src/better_robot/costs/_pose.py`
- Create: `src/better_robot/costs/_limits.py`
- Create: `src/better_robot/costs/_regularization.py`
- Create: `src/better_robot/costs/_collision.py`
- Create: `src/better_robot/costs/_manipulability.py`

- [ ] **Step 1: Create costs/__init__.py**

```python
"""Costs layer: differentiable residual functions.

Each function takes a joint config tensor and returns a residual vector.
No solver dependency — pure PyTorch functions.
"""

from ._pose import pose_residual as pose_residual
from ._limits import (
    limit_residual as limit_residual,
    velocity_residual as velocity_residual,
    acceleration_residual as acceleration_residual,
    jerk_residual as jerk_residual,
)
from ._regularization import (
    rest_residual as rest_residual,
    smoothness_residual as smoothness_residual,
)
from ._collision import (
    self_collision_residual as self_collision_residual,
    world_collision_residual as world_collision_residual,
)
from ._manipulability import manipulability_residual as manipulability_residual

__all__ = [
    "pose_residual",
    "limit_residual",
    "velocity_residual",
    "acceleration_residual",
    "jerk_residual",
    "rest_residual",
    "smoothness_residual",
    "self_collision_residual",
    "world_collision_residual",
    "manipulability_residual",
]
```

- [ ] **Step 2: Create _pose.py stub**

```python
"""Pose residuals for end-effector target matching."""

from __future__ import annotations

import torch

from ..core._robot import Robot


def pose_residual(
    cfg: torch.Tensor,
    robot: Robot,
    target_link_index: int,
    target_pose: torch.Tensor,
    pos_weight: float = 1.0,
    ori_weight: float = 1.0,
) -> torch.Tensor:
    """Compute SE3 log-space error between actual and target link pose.

    Args:
        cfg: Shape (num_actuated_joints,). Current joint configuration.
        robot: Robot instance.
        target_link_index: Index into robot.links.names for the target link.
        target_pose: Shape (7,). Target SE3 pose as wxyz+xyz.
        pos_weight: Weight on position error (first 3 dims of log).
        ori_weight: Weight on orientation error (last 3 dims of log).

    Returns:
        Shape (6,). Weighted SE3 log error [pos*w, ori*w].
    """
    raise NotImplementedError
```

- [ ] **Step 3: Create _limits.py stub**

```python
"""Joint limit, velocity, acceleration, and jerk residuals."""

from __future__ import annotations

import torch

from ..core._robot import Robot


def limit_residual(
    cfg: torch.Tensor,
    robot: Robot,
    weight: float = 1.0,
) -> torch.Tensor:
    """Compute joint limit violation residual.

    Positive values indicate violation (for use as constraint_leq_zero).

    Args:
        cfg: Shape (num_actuated_joints,). Current joint configuration.
        robot: Robot instance.
        weight: Scalar weight.

    Returns:
        Shape (2 * num_actuated_joints,). Upper and lower violations concatenated.
    """
    raise NotImplementedError


def velocity_residual(
    cfg: torch.Tensor,
    cfg_prev: torch.Tensor,
    robot: Robot,
    dt: float,
    weight: float = 1.0,
) -> torch.Tensor:
    """Compute joint velocity limit violation residual.

    Args:
        cfg: Shape (num_actuated_joints,). Current config.
        cfg_prev: Shape (num_actuated_joints,). Previous config.
        robot: Robot instance.
        dt: Timestep in seconds.
        weight: Scalar weight.

    Returns:
        Shape (num_actuated_joints,). Velocity violations.
    """
    raise NotImplementedError


def acceleration_residual(
    cfg: torch.Tensor,
    cfg_tp2: torch.Tensor,
    cfg_tp1: torch.Tensor,
    cfg_tm1: torch.Tensor,
    cfg_tm2: torch.Tensor,
    dt: float,
    weight: float = 1.0,
) -> torch.Tensor:
    """Compute joint acceleration using 5-point stencil.

    Args:
        cfg: Shape (num_actuated_joints,). Config at time t.
        cfg_tp2, cfg_tp1: Configs at t+2, t+1.
        cfg_tm1, cfg_tm2: Configs at t-1, t-2.
        dt: Timestep in seconds.
        weight: Scalar weight.

    Returns:
        Shape (num_actuated_joints,). Weighted acceleration magnitude.
    """
    raise NotImplementedError


def jerk_residual(
    cfg_tp3: torch.Tensor,
    cfg_tp2: torch.Tensor,
    cfg_tp1: torch.Tensor,
    cfg_tm1: torch.Tensor,
    cfg_tm2: torch.Tensor,
    cfg_tm3: torch.Tensor,
    dt: float,
    weight: float = 1.0,
) -> torch.Tensor:
    """Compute joint jerk using 7-point stencil.

    Args:
        cfg_tp3, cfg_tp2, cfg_tp1: Configs at t+3, t+2, t+1.
        cfg_tm1, cfg_tm2, cfg_tm3: Configs at t-1, t-2, t-3.
        dt: Timestep in seconds.
        weight: Scalar weight.

    Returns:
        Shape (num_actuated_joints,). Weighted jerk magnitude.
    """
    raise NotImplementedError
```

- [ ] **Step 4: Create _regularization.py stub**

```python
"""Rest pose and smoothness regularization residuals."""

from __future__ import annotations

import torch


def rest_residual(
    cfg: torch.Tensor,
    rest_pose: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    """Penalize deviation from a rest/default pose.

    Args:
        cfg: Shape (num_actuated_joints,). Current config.
        rest_pose: Shape (num_actuated_joints,). Target rest configuration.
        weight: Scalar weight.

    Returns:
        Shape (num_actuated_joints,). Weighted deviation from rest.
    """
    raise NotImplementedError


def smoothness_residual(
    cfg: torch.Tensor,
    cfg_prev: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    """Penalize large configuration changes between timesteps.

    Args:
        cfg: Shape (num_actuated_joints,). Current config.
        cfg_prev: Shape (num_actuated_joints,). Previous config.
        weight: Scalar weight.

    Returns:
        Shape (num_actuated_joints,). Weighted config difference.
    """
    raise NotImplementedError
```

- [ ] **Step 5: Create _collision.py stub**

```python
"""Self-collision and world-collision residuals."""

from __future__ import annotations

import torch

from ..core._robot import Robot
from ..collision._robot_collision import RobotCollision
from ..collision._geometry import CollGeom


def self_collision_residual(
    cfg: torch.Tensor,
    robot: Robot,
    robot_coll: RobotCollision,
    margin: float = 0.0,
    weight: float = 1.0,
) -> torch.Tensor:
    """Compute self-collision violation residual.

    Positive values indicate collision (for constraint_leq_zero).

    Args:
        cfg: Shape (num_actuated_joints,). Current config.
        robot: Robot instance.
        robot_coll: RobotCollision with sphere decomposition.
        margin: Safety margin in meters.
        weight: Scalar weight.

    Returns:
        Shape (num_collision_pairs,). Violation per pair.
    """
    raise NotImplementedError


def world_collision_residual(
    cfg: torch.Tensor,
    robot: Robot,
    robot_coll: RobotCollision,
    world_geom: CollGeom,
    margin: float = 0.0,
    weight: float = 1.0,
) -> torch.Tensor:
    """Compute world-collision violation residual.

    Positive values indicate collision (for constraint_leq_zero).

    Args:
        cfg: Shape (num_actuated_joints,). Current config.
        robot: Robot instance.
        robot_coll: RobotCollision with sphere decomposition.
        world_geom: World collision geometry.
        margin: Safety margin in meters.
        weight: Scalar weight.

    Returns:
        Shape (num_robot_spheres,). Violation per sphere.
    """
    raise NotImplementedError
```

- [ ] **Step 6: Create _manipulability.py stub**

```python
"""Yoshikawa manipulability residual."""

from __future__ import annotations

import torch

from ..core._robot import Robot


def manipulability_residual(
    cfg: torch.Tensor,
    robot: Robot,
    target_link_index: int,
    weight: float = 1.0,
) -> torch.Tensor:
    """Penalize low manipulability (inverse Yoshikawa measure).

    Minimizing this residual maximizes manipulability.
    Manipulability = sqrt(det(J @ J^T)) where J is the translation Jacobian.

    Args:
        cfg: Shape (num_actuated_joints,). Current config.
        robot: Robot instance.
        target_link_index: Index of the link to measure manipulability for.
        weight: Scalar weight.

    Returns:
        Shape (1,). Weighted inverse manipulability.
    """
    raise NotImplementedError
```

---

## Task 5: Create collision/ stubs

**Files:**
- Create: `src/better_robot/collision/__init__.py`
- Create: `src/better_robot/collision/_geometry.py`
- Create: `src/better_robot/collision/_robot_collision.py`

- [ ] **Step 1: Create collision/__init__.py**

```python
"""Collision geometry and robot collision model."""

from ._geometry import (
    CollGeom as CollGeom,
    Sphere as Sphere,
    Capsule as Capsule,
    Box as Box,
    HalfSpace as HalfSpace,
    Heightmap as Heightmap,
)
from ._robot_collision import RobotCollision as RobotCollision

__all__ = [
    "CollGeom", "Sphere", "Capsule", "Box", "HalfSpace", "Heightmap",
    "RobotCollision",
]
```

- [ ] **Step 2: Create _geometry.py stub**

```python
"""Collision geometry primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch


class CollGeom:
    """Base class for all collision geometry."""


@dataclass
class Sphere(CollGeom):
    """Sphere collision geometry."""

    center: torch.Tensor
    """Shape (3,). Center position in world frame."""

    radius: float
    """Sphere radius in meters."""

    @staticmethod
    def from_center_and_radius(center: torch.Tensor, radius: float) -> "Sphere":
        """Create a Sphere from center and radius."""
        raise NotImplementedError


@dataclass
class Capsule(CollGeom):
    """Capsule collision geometry (cylinder with hemispherical caps)."""

    point_a: torch.Tensor
    """Shape (3,). First endpoint."""

    point_b: torch.Tensor
    """Shape (3,). Second endpoint."""

    radius: float
    """Capsule radius in meters."""


@dataclass
class Box(CollGeom):
    """Axis-aligned box collision geometry."""

    position: torch.Tensor
    """Shape (3,). Box center in world frame."""

    extent: torch.Tensor
    """Shape (3,). Full extents [width, depth, height] in meters."""

    @staticmethod
    def from_extent(extent: torch.Tensor, position: torch.Tensor) -> "Box":
        """Create a Box from extent and center position."""
        raise NotImplementedError


@dataclass
class HalfSpace(CollGeom):
    """Half-space (infinite plane) collision geometry."""

    point: torch.Tensor
    """Shape (3,). Any point on the plane."""

    normal: torch.Tensor
    """Shape (3,). Outward normal (unit vector)."""

    @staticmethod
    def from_point_and_normal(point: torch.Tensor, normal: torch.Tensor) -> "HalfSpace":
        """Create a HalfSpace from a point on the plane and outward normal."""
        raise NotImplementedError


@dataclass
class Heightmap(CollGeom):
    """Heightmap collision geometry."""

    heights: torch.Tensor
    """Shape (H, W). Height values on a regular grid."""

    origin: torch.Tensor
    """Shape (3,). World-space position of the grid origin (corner)."""

    resolution: float
    """Grid cell size in meters."""
```

- [ ] **Step 3: Create _robot_collision.py stub**

```python
"""RobotCollision: sphere decomposition of robot links."""

from __future__ import annotations

import torch

from ..core._robot import Robot
from ._geometry import CollGeom


class RobotCollision:
    """Robot collision model using sphere decomposition.

    Each robot link is approximated as a set of spheres.
    Sphere centers are transformed by FK at query time.
    """

    @staticmethod
    def from_sphere_decomposition(
        sphere_decomposition: dict,
        urdf: object,
    ) -> "RobotCollision":
        """Create a RobotCollision from a sphere decomposition dict.

        Args:
            sphere_decomposition: Dict mapping link name to list of
                {'center': [x,y,z], 'radius': r} dicts.
            urdf: yourdfpy.URDF instance (used to resolve link frame transforms).

        Returns:
            RobotCollision instance.
        """
        raise NotImplementedError

    def compute_self_collision_distance(
        self,
        robot: Robot,
        cfg: torch.Tensor,
    ) -> torch.Tensor:
        """Compute signed distances for all active self-collision pairs.

        Args:
            robot: Robot instance.
            cfg: Shape (num_actuated_joints,). Current config.

        Returns:
            Shape (num_active_pairs,). Signed distance per pair
            (negative = penetrating).
        """
        raise NotImplementedError

    def compute_world_collision_distance(
        self,
        robot: Robot,
        cfg: torch.Tensor,
        world_geom: CollGeom,
    ) -> torch.Tensor:
        """Compute signed distances from each robot sphere to world geometry.

        Args:
            robot: Robot instance.
            cfg: Shape (num_actuated_joints,). Current config.
            world_geom: World collision geometry.

        Returns:
            Shape (num_robot_spheres,). Signed distance per sphere.
        """
        raise NotImplementedError
```

---

## Task 6: Create tasks/ and viewer/ stubs

**Files:**
- Create: `src/better_robot/tasks/__init__.py`
- Create: `src/better_robot/tasks/_ik.py`
- Create: `src/better_robot/tasks/_trajopt.py`
- Create: `src/better_robot/tasks/_retarget.py`
- Create: `src/better_robot/viewer/__init__.py`
- Create: `src/better_robot/viewer/_visualizer.py`

- [ ] **Step 1: Create tasks/__init__.py**

```python
"""Tasks layer: high-level solve_ik, solve_trajopt, retarget APIs."""

from ._ik import solve_ik as solve_ik
from ._trajopt import solve_trajopt as solve_trajopt
from ._retarget import retarget as retarget

__all__ = ["solve_ik", "solve_trajopt", "retarget"]
```

- [ ] **Step 2: Create _ik.py stub**

```python
"""Inverse kinematics task."""

from __future__ import annotations

from typing import Literal

import torch

from ..core._robot import Robot
from ..collision._robot_collision import RobotCollision
from ..collision._geometry import CollGeom


def solve_ik(
    robot: Robot,
    target_link: str,
    target_pose: torch.Tensor,
    solver: Literal["lm", "gn", "adam", "lbfgs"] = "lm",
    robot_coll: RobotCollision | None = None,
    world_coll: list[CollGeom] | None = None,
    weights: dict[str, float] | None = None,
    max_iter: int = 100,
    initial_cfg: torch.Tensor | None = None,
) -> torch.Tensor:
    """Solve inverse kinematics for a single end-effector target.

    Args:
        robot: Robot instance.
        target_link: Name of the target link (e.g. 'panda_hand').
        target_pose: Shape (7,). Target SE3 pose as wxyz+xyz.
        solver: Which solver to use. Default 'lm'.
        robot_coll: Optional robot collision model for collision avoidance.
        world_coll: Optional list of world collision geometries.
        weights: Cost weights. Keys: 'pose', 'limits', 'rest', 'collision'.
            Defaults: {'pose': 1.0, 'limits': 0.1, 'rest': 0.01}.
        max_iter: Maximum solver iterations.
        initial_cfg: Shape (num_actuated_joints,). Starting config.
            Defaults to robot's default joint config.

    Returns:
        Shape (num_actuated_joints,). Optimized joint configuration.
    """
    raise NotImplementedError
```

- [ ] **Step 3: Create _trajopt.py stub**

```python
"""Trajectory optimization task."""

from __future__ import annotations

from typing import Literal

import torch

from ..core._robot import Robot
from ..collision._robot_collision import RobotCollision
from ..collision._geometry import CollGeom


def solve_trajopt(
    robot: Robot,
    target_link: str,
    start_pose: torch.Tensor,
    end_pose: torch.Tensor,
    timesteps: int = 50,
    dt: float = 0.02,
    solver: Literal["lm", "gn", "adam", "lbfgs"] = "lm",
    robot_coll: RobotCollision | None = None,
    world_coll: list[CollGeom] | None = None,
    weights: dict[str, float] | None = None,
    max_iter: int = 200,
) -> torch.Tensor:
    """Solve trajectory optimization from start to end pose.

    Args:
        robot: Robot instance.
        target_link: Name of the end-effector link.
        start_pose: Shape (7,). Start SE3 pose as wxyz+xyz.
        end_pose: Shape (7,). End SE3 pose as wxyz+xyz.
        timesteps: Number of trajectory waypoints.
        dt: Timestep duration in seconds.
        solver: Which solver to use. Default 'lm'.
        robot_coll: Optional robot collision model.
        world_coll: Optional world collision geometries.
        weights: Cost weights. Keys: 'pose', 'limits', 'smoothness',
            'velocity', 'acceleration', 'collision'.
        max_iter: Maximum solver iterations.

    Returns:
        Shape (timesteps, num_actuated_joints). Optimized trajectory.
    """
    raise NotImplementedError
```

- [ ] **Step 4: Create _retarget.py stub**

```python
"""Motion retargeting task."""

from __future__ import annotations

from typing import Literal

import torch

from ..core._robot import Robot


def retarget(
    source_robot: Robot,
    target_robot: Robot,
    source_motion: torch.Tensor,
    solver: Literal["lm", "gn", "adam", "lbfgs"] = "lm",
    weights: dict[str, float] | None = None,
    max_iter: int = 100,
) -> torch.Tensor:
    """Retarget a motion sequence from source robot to target robot.

    Matches end-effector and key link poses from source to target,
    while respecting target robot's joint limits.

    Args:
        source_robot: Robot whose motion is being retargeted.
        target_robot: Robot to retarget the motion onto.
        source_motion: Shape (T, source_num_actuated_joints). Source trajectory.
        solver: Which solver to use per frame. Default 'lm'.
        weights: Cost weights. Keys: 'pose', 'limits', 'smoothness'.
        max_iter: Maximum solver iterations per frame.

    Returns:
        Shape (T, target_num_actuated_joints). Retargeted trajectory.
    """
    raise NotImplementedError
```

- [ ] **Step 5: Create viewer/__init__.py and _visualizer.py stubs**

`src/better_robot/viewer/__init__.py`:
```python
"""Visualization utilities (thin viser wrapper)."""

from ._visualizer import Visualizer as Visualizer

__all__ = ["Visualizer"]
```

`src/better_robot/viewer/_visualizer.py`:
```python
"""Thin viser wrapper for robot visualization."""

from __future__ import annotations

import torch
import yourdfpy


class Visualizer:
    """Wraps viser for interactive robot visualization."""

    def __init__(self, urdf: yourdfpy.URDF, port: int = 8080) -> None:
        """Initialize viser server and load URDF mesh.

        Args:
            urdf: Loaded yourdfpy.URDF.
            port: Port for the viser web server.
        """
        raise NotImplementedError

    def update_cfg(self, cfg: torch.Tensor) -> None:
        """Update the robot visualization to a new joint configuration.

        Args:
            cfg: Shape (num_actuated_joints,). Joint configuration.
        """
        raise NotImplementedError
```

---

## Task 7: Wire up the top-level __init__.py

**Files:**
- Modify: `src/better_robot/__init__.py`

- [ ] **Step 1: Write the flat public API**

```python
"""BetterRobot: PyTorch-native robot kinematics and optimization.

Quick start:
    import better_robot as br
    robot = br.Robot.from_urdf(urdf)
    solution = br.solve_ik(robot, target_link="panda_hand", target_pose=pose)
"""

from .core._robot import Robot as Robot
from .tasks._ik import solve_ik as solve_ik
from .tasks._trajopt import solve_trajopt as solve_trajopt
from .tasks._retarget import retarget as retarget
from . import collision as collision
from . import solvers as solvers
from . import costs as costs
from . import viewer as viewer

__version__ = "0.1.0"

__all__ = [
    "Robot",
    "solve_ik",
    "solve_trajopt",
    "retarget",
    "collision",
    "solvers",
    "costs",
    "viewer",
]
```

---

## Task 8: Create example and test stubs + verify imports

**Files:**
- Create: `examples/01_basic_ik.py`
- Create: `tests/__init__.py`
- Create: `tests/test_imports.py`
- Create: `tests/test_robot.py`
- Create: `tests/test_solvers.py`
- Create: `tests/test_costs.py`

- [ ] **Step 1: Create examples/01_basic_ik.py stub**

```python
"""Basic IK example (stub).

Usage:
    uv run python examples/01_basic_ik.py
"""

import better_robot as br
from robot_descriptions.loaders.yourdfpy import load_robot_description


def main() -> None:
    urdf = load_robot_description("panda_description")
    robot = br.Robot.from_urdf(urdf)
    print(f"Loaded robot with {robot.joints.num_actuated_joints} actuated joints")
    print("IK solve: not yet implemented")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create tests/__init__.py**

```python
```

(empty file)

- [ ] **Step 3: Create tests/test_imports.py**

```python
"""Smoke test: verify all public modules and symbols are importable."""


def test_top_level_imports() -> None:
    import better_robot as br

    assert hasattr(br, "Robot")
    assert hasattr(br, "solve_ik")
    assert hasattr(br, "solve_trajopt")
    assert hasattr(br, "retarget")
    assert hasattr(br, "collision")
    assert hasattr(br, "solvers")
    assert hasattr(br, "costs")
    assert hasattr(br, "viewer")


def test_collision_imports() -> None:
    from better_robot import collision

    assert hasattr(collision, "RobotCollision")
    assert hasattr(collision, "Sphere")
    assert hasattr(collision, "Capsule")
    assert hasattr(collision, "Box")
    assert hasattr(collision, "HalfSpace")
    assert hasattr(collision, "Heightmap")


def test_solver_imports() -> None:
    from better_robot import solvers

    assert hasattr(solvers, "LevenbergMarquardt")
    assert hasattr(solvers, "GaussNewton")
    assert hasattr(solvers, "AdamSolver")
    assert hasattr(solvers, "LBFGSSolver")
    assert hasattr(solvers, "SOLVER_REGISTRY")
    assert set(solvers.SOLVER_REGISTRY.keys()) == {"lm", "gn", "adam", "lbfgs"}


def test_costs_imports() -> None:
    from better_robot import costs

    for name in [
        "pose_residual", "limit_residual", "velocity_residual",
        "acceleration_residual", "jerk_residual", "rest_residual",
        "smoothness_residual", "self_collision_residual",
        "world_collision_residual", "manipulability_residual",
    ]:
        assert hasattr(costs, name), f"Missing: {name}"
```

- [ ] **Step 4: Create tests/test_robot.py stub**

```python
"""Tests for Robot and forward kinematics (stubs — fail until implemented)."""

import pytest


def test_robot_from_urdf_placeholder() -> None:
    """Placeholder: verify Robot.from_urdf raises NotImplementedError until implemented."""
    from better_robot import Robot
    import yourdfpy
    from robot_descriptions.loaders.yourdfpy import load_robot_description

    urdf = load_robot_description("panda_description")
    with pytest.raises(NotImplementedError):
        Robot.from_urdf(urdf)
```

- [ ] **Step 5: Create tests/test_solvers.py stub**

```python
"""Tests for solvers (stubs — fail until implemented)."""

import pytest
import torch


def test_lm_solver_placeholder() -> None:
    """Placeholder: verify LM solve raises NotImplementedError until implemented."""
    from better_robot.solvers import LevenbergMarquardt, Problem

    problem = Problem(variables=torch.zeros(7))
    solver = LevenbergMarquardt()
    with pytest.raises(NotImplementedError):
        solver.solve(problem)
```

- [ ] **Step 6: Create tests/test_costs.py stub**

```python
"""Tests for cost residuals (stubs — fail until implemented)."""

import pytest
import torch


def test_pose_residual_placeholder() -> None:
    """Placeholder: verify pose_residual raises NotImplementedError."""
    from better_robot.costs import pose_residual

    with pytest.raises(NotImplementedError):
        pose_residual(
            cfg=torch.zeros(7),
            robot=None,  # type: ignore
            target_link_index=0,
            target_pose=torch.tensor([1.0, 0, 0, 0, 0, 0, 0]),
        )
```

- [ ] **Step 7: Run the import tests to verify scaffold is correct**

```bash
uv run pytest tests/test_imports.py -v
```

Expected output:
```
tests/test_imports.py::test_top_level_imports PASSED
tests/test_imports.py::test_collision_imports PASSED
tests/test_imports.py::test_solver_imports PASSED
tests/test_imports.py::test_costs_imports PASSED

4 passed in Xs
```

- [ ] **Step 8: Commit the scaffold**

```bash
git init
git add .
git commit -m "feat: scaffold BetterRobot package with uv and placeholder stubs"
```

---

## Task 9: Copy plan to plan.md at project root

- [ ] **Step 1: Copy this plan to plan.md**

```bash
cp docs/superpowers/plans/2026-04-04-betterrobot-scaffold.md plan.md
```

- [ ] **Step 2: Verify**

```bash
ls plan.md
```

Expected: `plan.md`
