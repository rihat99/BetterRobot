# API Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace three IK functions (`solve_ik`, `solve_ik_multi`, `solve_ik_floating_base`) with a single `solve_ik`, expose all tuning knobs in an `IKConfig` dataclass, and extract shared viser utilities into `viewer/_helpers.py`.

**Architecture:** `IKConfig` dataclass holds all weights; `solve_ik` dispatches on `initial_base_pose` to a fixed-base path (existing `Problem`/LM) or floating-base path (`_FloatingBaseIKModule`). Viewer helpers live in `viewer/_helpers.py` and are imported by both examples.

**Tech Stack:** PyTorch, PyPose, viser, pytest

---

## Files

| File | Action | Purpose |
|------|--------|---------|
| `src/better_robot/tasks/_config.py` | Create | `IKConfig` dataclass |
| `src/better_robot/tasks/_ik.py` | Rewrite | Unified `solve_ik` + `_solve_fixed` + `_solve_floating` |
| `src/better_robot/tasks/_floating_base_ik.py` | Modify | Use `IKConfig`; remove `solve_ik_floating_base`; add `_run_floating_base_lm` |
| `src/better_robot/tasks/__init__.py` | Modify | Export `IKConfig`, `solve_ik` only |
| `src/better_robot/__init__.py` | Modify | Same; update module docstring |
| `src/better_robot/viewer/_helpers.py` | Create | `wxyz_pos_to_se3`, `qxyzw_to_wxyz`, `build_cfg_dict` |
| `src/better_robot/viewer/__init__.py` | Modify | Export helpers |
| `examples/01_basic_ik.py` | Rewrite | New API + viewer helpers |
| `examples/02_g1_ik.py` | Rewrite | New API + viewer helpers + `--profile` flag |
| `tests/test_ik.py` | Rewrite | All tests use new API |
| `tests/test_imports.py` | Modify | Check `IKConfig`; remove old function checks |
| `src/better_robot/tasks/CLAUDE.md` | Modify | Document unified API |
| `src/better_robot/viewer/CLAUDE.md` | Create | Document helpers |
| `CLAUDE.md` | Modify | Update IK usage pattern |

---

### Task 1: `IKConfig` dataclass

**Files:**
- Create: `src/better_robot/tasks/_config.py`

- [ ] **Step 1: Write the failing smoke test**

In `tests/test_imports.py`, add to `test_top_level_imports`:

```python
from better_robot import IKConfig
cfg = IKConfig()
assert cfg.pos_weight == 1.0
assert cfg.ori_weight == 0.1
assert cfg.pose_weight == 1.0
assert cfg.limit_weight == 0.1
assert cfg.rest_weight == 0.01
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_imports.py::test_top_level_imports -v
```

Expected: `ImportError: cannot import name 'IKConfig'`

- [ ] **Step 3: Create `src/better_robot/tasks/_config.py`**

```python
"""IK solver configuration."""

from dataclasses import dataclass


@dataclass
class IKConfig:
    """Configuration for all IK solvers.

    All weights are non-negative floats. Larger values increase
    the influence of that cost term relative to others.
    """

    pos_weight: float = 1.0    # position component of pose error
    ori_weight: float = 0.1    # orientation component of pose error
    pose_weight: float = 1.0   # overall pose cost scale
    limit_weight: float = 0.1  # soft joint limit penalty
    rest_weight: float = 0.01  # pull toward robot._default_cfg
```

- [ ] **Step 4: Export from `src/better_robot/tasks/__init__.py`**

Replace the entire file:

```python
"""Tasks layer: high-level IK API."""

from ._config import IKConfig as IKConfig
from ._ik import solve_ik as solve_ik
from ._trajopt import solve_trajopt as solve_trajopt
from ._retarget import retarget as retarget

__all__ = ["IKConfig", "solve_ik", "solve_trajopt", "retarget"]
```

- [ ] **Step 5: Export from `src/better_robot/__init__.py`**

Replace the entire file:

```python
"""BetterRobot: PyTorch-native robot kinematics and optimization.

Quick start:
    import better_robot as br
    robot = br.Robot.from_urdf(urdf)

    # Fixed base — single or multiple targets
    cfg = br.solve_ik(robot, targets={"panda_hand": pose})

    # Floating base (humanoid whole-body IK)
    base_pose, cfg = br.solve_ik(
        robot,
        targets={"left_rubber_hand": p_lh, "right_rubber_hand": p_rh},
        initial_base_pose=torch.tensor([0., 0., 0.78, 0., 0., 0., 1.]),
    )
"""

from .core._robot import Robot as Robot
from .tasks._config import IKConfig as IKConfig
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
    "IKConfig",
    "solve_ik",
    "solve_trajopt",
    "retarget",
    "collision",
    "solvers",
    "costs",
    "viewer",
]
```

Note: `solve_ik` is already exported from `tasks/_ik.py` (existing file, unchanged in this task).
The import will work once `_ik.py` still has `solve_ik` defined (it does).

- [ ] **Step 6: Run smoke test to verify it passes**

```bash
uv run pytest tests/test_imports.py::test_top_level_imports -v
```

Expected: PASS

- [ ] **Step 7: Run all tests to check nothing is broken**

```bash
uv run pytest tests/ -v
```

Expected: all 33 tests pass (existing `solve_ik` still works, new export is additive).

- [ ] **Step 8: Commit**

```bash
git add src/better_robot/tasks/_config.py src/better_robot/tasks/__init__.py src/better_robot/__init__.py tests/test_imports.py
git commit -m "feat: add IKConfig dataclass and wire into public API"
```

---

### Task 2: Viewer helpers

**Files:**
- Create: `src/better_robot/viewer/_helpers.py`
- Modify: `src/better_robot/viewer/__init__.py`

- [ ] **Step 1: Write the failing test**

Add a new test to `tests/test_imports.py`:

```python
def test_viewer_helpers() -> None:
    import torch
    from better_robot.viewer import wxyz_pos_to_se3, qxyzw_to_wxyz, build_cfg_dict
    from better_robot import Robot
    from robot_descriptions.loaders.yourdfpy import load_robot_description

    # wxyz_pos_to_se3
    se3 = wxyz_pos_to_se3((1.0, 0.0, 0.0, 0.0), (0.1, 0.2, 0.3))
    assert se3.shape == (7,)
    assert abs(se3[0].item() - 0.1) < 1e-6  # tx
    assert abs(se3[6].item() - 1.0) < 1e-6  # qw

    # qxyzw_to_wxyz
    q = torch.tensor([0.0, 0.0, 0.0, 1.0])  # identity [qx,qy,qz,qw]
    wxyz = qxyzw_to_wxyz(q)
    assert wxyz == (1.0, 0.0, 0.0, 0.0)

    # build_cfg_dict
    urdf = load_robot_description("panda_description")
    robot = Robot.from_urdf(urdf)
    cfg = robot._default_cfg.clone()
    d = build_cfg_dict(robot, cfg)
    assert isinstance(d, dict)
    assert len(d) == robot.joints.num_actuated_joints
    assert all(isinstance(v, float) for v in d.values())
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_imports.py::test_viewer_helpers -v
```

Expected: `ImportError: cannot import name 'wxyz_pos_to_se3'`

- [ ] **Step 3: Create `src/better_robot/viewer/_helpers.py`**

```python
"""Shared viser conversion utilities for robot examples."""

from __future__ import annotations

import torch

from ..core._robot import Robot


def wxyz_pos_to_se3(wxyz: tuple, pos: tuple | list) -> torch.Tensor:
    """Convert viser wxyz + position to SE3 [tx, ty, tz, qx, qy, qz, qw].

    Args:
        wxyz: Viser scalar-first quaternion (w, x, y, z).
        pos: Position (x, y, z).

    Returns:
        Shape (7,) SE3 tensor [tx, ty, tz, qx, qy, qz, qw].
    """
    w, x, y, z = wxyz
    return torch.tensor([pos[0], pos[1], pos[2], x, y, z, w], dtype=torch.float32)


def qxyzw_to_wxyz(q: torch.Tensor) -> tuple:
    """Convert [qx, qy, qz, qw] tensor to viser (w, x, y, z) tuple.

    Args:
        q: Shape (4,) quaternion tensor [qx, qy, qz, qw].

    Returns:
        Viser scalar-first quaternion (w, x, y, z).
    """
    return (q[3].item(), q[0].item(), q[1].item(), q[2].item())


def build_cfg_dict(robot: Robot, cfg: torch.Tensor) -> dict[str, float]:
    """Build joint_name→angle dict for ViserUrdf.update_cfg.

    Filters to actuated joints (revolute, continuous, prismatic) in
    the same BFS order used by forward_kinematics.

    Args:
        robot: Robot instance.
        cfg: Shape (num_actuated_joints,) joint configuration tensor.

    Returns:
        Dict mapping joint name to float angle value.
    """
    names = [
        name
        for name, jtype in zip(robot.joints.names, robot._fk_joint_types)
        if jtype in ("revolute", "continuous", "prismatic")
    ]
    return {name: float(v) for name, v in zip(names, cfg.detach().cpu())}
```

- [ ] **Step 4: Update `src/better_robot/viewer/__init__.py`**

```python
"""Visualization utilities (thin viser wrapper)."""

from ._visualizer import Visualizer as Visualizer
from ._helpers import wxyz_pos_to_se3 as wxyz_pos_to_se3
from ._helpers import qxyzw_to_wxyz as qxyzw_to_wxyz
from ._helpers import build_cfg_dict as build_cfg_dict

__all__ = ["Visualizer", "wxyz_pos_to_se3", "qxyzw_to_wxyz", "build_cfg_dict"]
```

- [ ] **Step 5: Run test to verify it passes**

```bash
uv run pytest tests/test_imports.py::test_viewer_helpers -v
```

Expected: PASS

- [ ] **Step 6: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/better_robot/viewer/_helpers.py src/better_robot/viewer/__init__.py tests/test_imports.py
git commit -m "feat: add viewer helpers wxyz_pos_to_se3, qxyzw_to_wxyz, build_cfg_dict"
```

---

### Task 3: Unified `solve_ik` — rewrite `_ik.py`

This task rewrites all IK tests and the `_ik.py` file. By the end, fixed-base tests pass;
floating-base tests are also updated here but may still fail until Task 4.

**Files:**
- Modify: `tests/test_ik.py`
- Rewrite: `src/better_robot/tasks/_ik.py`

- [ ] **Step 1: Rewrite `tests/test_ik.py` to use new API**

Replace the entire file:

```python
"""Tests for the unified solve_ik API."""

import torch
import pytest
from robot_descriptions.loaders.yourdfpy import load_robot_description
from better_robot import Robot, solve_ik, IKConfig


@pytest.fixture(scope="module")
def panda():
    urdf = load_robot_description("panda_description")
    return Robot.from_urdf(urdf)


# --- Fixed-base tests ---

def test_solve_ik_returns_correct_shape(panda):
    target = torch.tensor([0.3, 0.0, 0.5, 0., 0., 0., 1.])
    result = solve_ik(panda, targets={"panda_hand": target}, max_iter=30)
    assert result.shape == (panda.joints.num_actuated_joints,)


def test_solve_ik_respects_joint_limits(panda):
    target = torch.tensor([0.3, 0.0, 0.5, 0., 0., 0., 1.])
    result = solve_ik(panda, targets={"panda_hand": target}, max_iter=50)
    lo = panda.joints.lower_limits
    hi = panda.joints.upper_limits
    assert (result >= lo - 0.1).all(), f"Below lower limits: {result}"
    assert (result <= hi + 0.1).all(), f"Above upper limits: {result}"


def test_solve_ik_converges_to_reachable_pose(panda):
    """Target is FK of default config — guaranteed reachable, should converge in few iters."""
    cfg_default = panda._default_cfg
    fk = panda.forward_kinematics(cfg_default)
    hand_idx = panda.get_link_index("panda_hand")
    target = fk[hand_idx].detach()

    result = solve_ik(
        panda,
        targets={"panda_hand": target},
        initial_cfg=cfg_default.clone(),
        max_iter=5,
    )
    fk_result = panda.forward_kinematics(result)
    pos_error = (fk_result[hand_idx, :3] - target[:3]).norm().item()
    assert pos_error < 0.05, f"Position error too large: {pos_error}"


def test_solve_ik_with_custom_config(panda):
    target = torch.tensor([0.4, 0.0, 0.4, 0., 0., 0., 1.])
    result = solve_ik(
        panda,
        targets={"panda_hand": target},
        cfg=IKConfig(pose_weight=2.0, limit_weight=0.5, rest_weight=0.001),
        max_iter=30,
    )
    assert result.shape == (panda.joints.num_actuated_joints,)


def test_solve_ik_multi_target_shape(panda):
    """Multi-target is just solve_ik with multiple keys in targets dict."""
    cfg = panda._default_cfg
    fk = panda.forward_kinematics(cfg)
    targets = {
        "panda_link6": fk[panda.get_link_index("panda_link6")].detach(),
        "panda_hand": fk[panda.get_link_index("panda_hand")].detach(),
    }
    result = solve_ik(panda, targets=targets, max_iter=5)
    assert result.shape == (panda.joints.num_actuated_joints,)


def test_solve_ik_multi_target_converges(panda):
    cfg = panda._default_cfg
    fk = panda.forward_kinematics(cfg)
    hand_idx = panda.get_link_index("panda_hand")
    link6_idx = panda.get_link_index("panda_link6")
    targets = {
        "panda_link6": fk[link6_idx].detach(),
        "panda_hand": fk[hand_idx].detach(),
    }
    result = solve_ik(panda, targets=targets, initial_cfg=cfg.clone(), max_iter=5)
    fk_result = panda.forward_kinematics(result)
    assert (fk_result[hand_idx, :3] - fk[hand_idx, :3]).norm().item() < 0.05
    assert (fk_result[link6_idx, :3] - fk[link6_idx, :3]).norm().item() < 0.05


# --- Floating-base tests ---

def test_solve_ik_floating_base_return_shapes(panda):
    """Floating-base solve_ik returns (base_pose(7), cfg(n)) tuple."""
    identity_base = torch.tensor([0., 0., 0., 0., 0., 0., 1.])
    hand_idx = panda.get_link_index("panda_hand")
    fk = panda.forward_kinematics(panda._default_cfg)
    targets = {"panda_hand": fk[hand_idx].detach()}

    base_pose, cfg = solve_ik(
        panda, targets=targets, initial_base_pose=identity_base, max_iter=3
    )
    assert base_pose.shape == (7,)
    assert cfg.shape == (panda.joints.num_actuated_joints,)


def test_solve_ik_floating_base_converges(panda):
    """Target is FK of default config at identity base — should converge quickly."""
    cfg0 = panda._default_cfg
    hand_idx = panda.get_link_index("panda_hand")
    fk0 = panda.forward_kinematics(cfg0)
    target = fk0[hand_idx].detach()
    identity_base = torch.tensor([0., 0., 0., 0., 0., 0., 1.])

    base_pose, cfg = solve_ik(
        panda,
        targets={"panda_hand": target},
        initial_base_pose=identity_base,
        initial_cfg=cfg0.clone(),
        max_iter=5,
    )
    fk_result = panda.forward_kinematics(cfg, base_pose=base_pose)
    pos_err = (fk_result[hand_idx, :3] - target[:3]).norm().item()
    assert pos_err < 0.05, f"Position error: {pos_err}"


def test_solve_ik_floating_base_base_moves(panda):
    """Target 1 m in X from default FK — only reachable by translating the base."""
    cfg0 = panda._default_cfg
    fk0 = panda.forward_kinematics(cfg0)
    hand_idx = panda.get_link_index("panda_hand")
    target = fk0[hand_idx].detach().clone()
    target[0] += 1.0
    identity_base = torch.tensor([0., 0., 0., 0., 0., 0., 1.])

    base_pose, _ = solve_ik(
        panda, targets={"panda_hand": target},
        initial_base_pose=identity_base,
        max_iter=30,
    )
    assert base_pose[0].abs().item() > 0.1, f"Base did not translate: {base_pose}"
```

- [ ] **Step 2: Run to verify tests fail**

```bash
uv run pytest tests/test_ik.py -v
```

Expected: all tests FAIL — `solve_ik` still has the old signature (`target_link`, `target_pose`).

- [ ] **Step 3: Rewrite `src/better_robot/tasks/_ik.py`**

```python
"""Inverse kinematics — unified fixed-base and floating-base solver."""

from __future__ import annotations

import functools

import torch

from ..core._robot import Robot
from ..costs._pose import pose_residual
from ..costs._limits import limit_residual
from ..costs._regularization import rest_residual
from ..solvers._base import CostTerm, Problem
from ..solvers import SOLVER_REGISTRY
from ._config import IKConfig


def solve_ik(
    robot: Robot,
    targets: dict[str, torch.Tensor],
    cfg: IKConfig = IKConfig(),
    initial_cfg: torch.Tensor | None = None,
    initial_base_pose: torch.Tensor | None = None,
    max_iter: int = 100,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Solve inverse kinematics for one or more end-effector targets.

    Args:
        robot: Robot instance.
        targets: {link_name: (7,) SE3 target [tx, ty, tz, qx, qy, qz, qw]}.
        cfg: IK configuration (weights, position/orientation balance).
        initial_cfg: (n,) starting joint config. Defaults to robot._default_cfg.
        initial_base_pose: (7,) SE3 initial base pose. When provided, the base
            is also optimized (floating-base mode). Default None = fixed base.
        max_iter: Solver iterations.

    Returns:
        Fixed base: (num_actuated_joints,) optimized joint config tensor.
        Floating base: tuple (base_pose (7,), cfg (num_actuated_joints,)).
    """
    if initial_base_pose is not None:
        from ._floating_base_ik import _run_floating_base_lm
        return _run_floating_base_lm(robot, targets, cfg, initial_cfg, initial_base_pose, max_iter)
    return _solve_fixed(robot, targets, cfg, initial_cfg, max_iter)


def _solve_fixed(
    robot: Robot,
    targets: dict[str, torch.Tensor],
    ik_cfg: IKConfig,
    initial_cfg: torch.Tensor | None,
    max_iter: int,
) -> torch.Tensor:
    initial = initial_cfg.clone() if initial_cfg is not None else robot._default_cfg.clone()
    rest = robot._default_cfg.clone()

    costs = []
    for link_name, target_pose in targets.items():
        link_idx = robot.get_link_index(link_name)
        costs.append(CostTerm(
            residual_fn=functools.partial(
                pose_residual,
                robot=robot,
                target_link_index=link_idx,
                target_pose=target_pose,
                pos_weight=ik_cfg.pos_weight,
                ori_weight=ik_cfg.ori_weight,
            ),
            weight=ik_cfg.pose_weight,
        ))

    costs += [
        CostTerm(
            residual_fn=functools.partial(limit_residual, robot=robot),
            weight=ik_cfg.limit_weight,
        ),
        CostTerm(
            residual_fn=functools.partial(rest_residual, rest_pose=rest),
            weight=ik_cfg.rest_weight,
        ),
    ]

    problem = Problem(
        variables=initial,
        costs=costs,
        lower_bounds=robot.joints.lower_limits.clone(),
        upper_bounds=robot.joints.upper_limits.clone(),
    )
    return SOLVER_REGISTRY["lm"]().solve(problem, max_iter=max_iter)
```

- [ ] **Step 4: Run tests to verify fixed-base tests pass**

```bash
uv run pytest tests/test_ik.py -v -k "not floating"
```

Expected: 6 tests PASS (`test_solve_ik_returns_correct_shape`, `test_solve_ik_respects_joint_limits`, `test_solve_ik_converges_to_reachable_pose`, `test_solve_ik_with_custom_config`, `test_solve_ik_multi_target_shape`, `test_solve_ik_multi_target_converges`).

Floating-base tests will still fail — `_run_floating_base_lm` not yet defined. That's expected.

- [ ] **Step 5: Commit**

```bash
git add tests/test_ik.py src/better_robot/tasks/_ik.py
git commit -m "refactor: unified solve_ik replaces solve_ik_multi (fixed-base path done)"
```

---

### Task 4: Floating-base dispatch

**Files:**
- Modify: `src/better_robot/tasks/_floating_base_ik.py`

- [ ] **Step 1: Run floating-base tests to confirm they currently fail**

```bash
uv run pytest tests/test_ik.py -v -k "floating"
```

Expected: FAIL — `ImportError: cannot import name '_run_floating_base_lm'`

- [ ] **Step 2: Rewrite `src/better_robot/tasks/_floating_base_ik.py`**

Remove `solve_ik_floating_base`. Update `_FloatingBaseIKModule` to accept `IKConfig`.
Add `_run_floating_base_lm` as the entry point called by `solve_ik`.

```python
"""Floating-base whole-body IK using PyPose LM — internal implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import pypose as pp
import pypose.optim as ppo
import pypose.optim.strategy as ppo_strategy

from ..core._robot import Robot
from ..core._lie_ops import se3_log, se3_compose, se3_inverse
from ..costs._limits import limit_residual
from ..costs._regularization import rest_residual
from ._config import IKConfig


class _FloatingBaseIKModule(nn.Module):
    """LM-compatible module for whole-body IK with a floating base.

    Optimizes two parameters simultaneously:
    - self.base: pp.Parameter(pp.SE3) — base frame SE3 pose in world
    - self.cfg:  nn.Parameter          — joint angles

    Residual: [pose_errors..., limit_violations, rest_deviation].
    """

    def __init__(
        self,
        robot: Robot,
        target_link_indices: list[int],
        target_poses: list[torch.Tensor],
        initial_base: torch.Tensor,
        initial_cfg: torch.Tensor,
        rest_cfg: torch.Tensor,
        ik_cfg: IKConfig,
    ) -> None:
        super().__init__()
        self.base = pp.Parameter(pp.SE3(initial_base.float()))
        self.cfg = nn.Parameter(initial_cfg.float())
        self._robot = robot
        self._target_link_indices = target_link_indices
        self._target_poses = [tp.float() for tp in target_poses]
        self._rest_cfg = rest_cfg.float()
        self._ik_cfg = ik_cfg

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        fk = self._robot.forward_kinematics(self.cfg, base_pose=self.base.tensor())

        residuals = []
        for link_idx, target_pose in zip(self._target_link_indices, self._target_poses):
            actual_pose = fk[..., link_idx, :]
            T_err = se3_compose(se3_inverse(target_pose), actual_pose)
            log_err = se3_log(T_err)  # (6,) [tx, ty, tz, rx, ry, rz]
            weighted = torch.cat([
                log_err[:3] * self._ik_cfg.pos_weight,
                log_err[3:] * self._ik_cfg.ori_weight,
            ]) * self._ik_cfg.pose_weight
            residuals.append(weighted)

        residuals.append(limit_residual(self.cfg, self._robot) * self._ik_cfg.limit_weight)
        residuals.append(rest_residual(self.cfg, self._rest_cfg) * self._ik_cfg.rest_weight)

        return torch.cat(residuals)


def _run_floating_base_lm(
    robot: Robot,
    targets: dict[str, torch.Tensor],
    ik_cfg: IKConfig,
    initial_cfg: torch.Tensor | None,
    initial_base_pose: torch.Tensor,
    max_iter: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run floating-base LM. Called by solve_ik when initial_base_pose is provided."""
    initial_cfg_t = (
        initial_cfg.clone().float() if initial_cfg is not None
        else robot._default_cfg.clone().float()
    )
    rest_cfg = robot._default_cfg.clone().float()

    target_link_indices = [robot.get_link_index(name) for name in targets]
    target_poses = list(targets.values())

    lo = robot.joints.lower_limits.float()
    hi = robot.joints.upper_limits.float()

    module = _FloatingBaseIKModule(
        robot=robot,
        target_link_indices=target_link_indices,
        target_poses=target_poses,
        initial_base=initial_base_pose.clone().float(),
        initial_cfg=initial_cfg_t,
        rest_cfg=rest_cfg,
        ik_cfg=ik_cfg,
    )

    strategy = ppo_strategy.Adaptive(damping=1e-4)
    optimizer = ppo.LevenbergMarquardt(module, strategy=strategy, vectorize=True)
    dummy = torch.zeros(1)

    for _ in range(max_iter):
        optimizer.step(input=dummy)
        with torch.no_grad():
            # Hard joint limit enforcement
            module.cfg.data.clamp_(
                lo.to(device=module.cfg.device),
                hi.to(device=module.cfg.device),
            )
            # Renormalize base quaternion to unit length.
            # SE3 layout: [tx, ty, tz, qx, qy, qz, qw] — quaternion at indices 3:7.
            # Use copy_ via .tensor() to stay within PyPose's expected access patterns.
            raw = module.base.tensor().clone()
            raw[3:7] = raw[3:7] / raw[3:7].norm()
            module.base.data.copy_(raw)

    return module.base.tensor().detach(), module.cfg.detach()
```

- [ ] **Step 3: Run all IK tests**

```bash
uv run pytest tests/test_ik.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 4: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass. (Some tests that imported from `_floating_base_ik.solve_ik_floating_base`
were already updated in Task 3's test rewrite — no remaining references.)

- [ ] **Step 5: Commit**

```bash
git add src/better_robot/tasks/_floating_base_ik.py
git commit -m "refactor: _FloatingBaseIKModule uses IKConfig; floating-base dispatch via _run_floating_base_lm"
```

---

### Task 5: Update public API exports and smoke tests

**Files:**
- Modify: `tests/test_imports.py`

The `__init__.py` files were already updated in Task 1. This task cleans up the smoke tests
to match: remove `solve_ik_multi` / `solve_ik_floating_base`, add `IKConfig`.

- [ ] **Step 1: Update `test_top_level_imports` in `tests/test_imports.py`**

Replace the existing `test_top_level_imports` function:

```python
def test_top_level_imports() -> None:
    import better_robot as br
    from better_robot import IKConfig

    assert hasattr(br, "Robot")
    assert hasattr(br, "IKConfig")
    assert hasattr(br, "solve_ik")
    assert hasattr(br, "solve_trajopt")
    assert hasattr(br, "retarget")
    assert hasattr(br, "collision")
    assert hasattr(br, "solvers")
    assert hasattr(br, "costs")
    assert hasattr(br, "viewer")

    # IKConfig field defaults
    cfg = IKConfig()
    assert cfg.pos_weight == 1.0
    assert cfg.ori_weight == 0.1
    assert cfg.pose_weight == 1.0
    assert cfg.limit_weight == 0.1
    assert cfg.rest_weight == 0.01
```

- [ ] **Step 2: Run smoke tests**

```bash
uv run pytest tests/test_imports.py -v
```

Expected: all 5 tests PASS (including `test_viewer_helpers` added in Task 2).

- [ ] **Step 3: Run full suite**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_imports.py
git commit -m "test: update smoke tests to match unified API (IKConfig, remove old exports)"
```

---

### Task 6: Update examples

**Files:**
- Rewrite: `examples/01_basic_ik.py`
- Rewrite: `examples/02_g1_ik.py`

No automated tests for examples. After rewriting, run each manually (or just do a syntax check).

- [ ] **Step 1: Rewrite `examples/01_basic_ik.py`**

```python
"""Interactive IK example for Franka Panda.

Usage:
    uv run python examples/01_basic_ik.py
Then open http://localhost:8080 in your browser.
Drag the red transform handle to move the target end-effector pose.
"""
import time
import torch
import viser
import viser.extras
from robot_descriptions.loaders.yourdfpy import load_robot_description
import better_robot as br
from better_robot.viewer import wxyz_pos_to_se3, qxyzw_to_wxyz, build_cfg_dict


def main() -> None:
    urdf = load_robot_description("panda_description")
    robot = br.Robot.from_urdf(urdf)
    print(f"Loaded robot: {robot.links.num_links} links, {robot.joints.num_actuated_joints} actuated joints")

    server = viser.ViserServer(port=8080)
    urdf_vis = viser.extras.ViserUrdf(server, urdf)

    # Natural orientation at default config — avoids near-180° LM oscillation
    cfg = robot._default_cfg.clone()
    fk0 = robot.forward_kinematics(cfg)
    hand_idx = robot.get_link_index("panda_hand")
    target_handle = server.scene.add_transform_controls(
        "/target_ee",
        position=tuple(fk0[hand_idx, :3].detach().tolist()),
        wxyz=qxyzw_to_wxyz(fk0[hand_idx, 3:7].detach()),
        scale=0.15,
    )

    print("Open http://localhost:8080 in your browser")
    print("Drag the transform handle to set the IK target. Press Ctrl+C to quit.")

    while True:
        target_pose = wxyz_pos_to_se3(target_handle.wxyz, target_handle.position)
        cfg = br.solve_ik(
            robot=robot,
            targets={"panda_hand": target_pose},
            cfg=br.IKConfig(rest_weight=0.001),
            initial_cfg=cfg,
            max_iter=20,
        )
        urdf_vis.update_cfg(build_cfg_dict(robot, cfg))
        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax-check `01_basic_ik.py`**

```bash
uv run python -c "import ast; ast.parse(open('examples/01_basic_ik.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Rewrite `examples/02_g1_ik.py`**

```python
"""Interactive whole-body IK for the Unitree G1 humanoid.

Controls both hands and both feet simultaneously with a floating base.

Usage:
    uv run python examples/02_g1_ik.py
Then open http://localhost:8080 in your browser.
Drag the coloured transform handles to move the targets.

Pass --profile to run 10 IK iterations, print a CPU profiling breakdown, then exit.
"""
import argparse
import time
import torch
import viser
import viser.extras
from robot_descriptions.loaders.yourdfpy import load_robot_description
import better_robot as br
from better_robot.viewer import wxyz_pos_to_se3, qxyzw_to_wxyz, build_cfg_dict


# End-effector handle names and their robot link names
TARGET_SPECS = [
    ("left_hand",  "left_rubber_hand"),
    ("right_hand", "right_rubber_hand"),
    ("left_foot",  "left_ankle_roll_link"),
    ("right_foot", "right_ankle_roll_link"),
]

# G1 standing height: pelvis ~0.78 m above ground
INITIAL_BASE_POSE = torch.tensor([0.0, 0.0, 0.78, 0.0, 0.0, 0.0, 1.0])


def main(profile: bool = False) -> None:
    urdf = load_robot_description("g1_description")
    robot = br.Robot.from_urdf(urdf)
    print(f"Loaded G1: {robot.links.num_links} links, {robot.joints.num_actuated_joints} actuated joints")

    server = viser.ViserServer(port=8080)
    server.scene.add_grid("/ground", width=4, height=4)
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = viser.extras.ViserUrdf(server, urdf, root_node_name="/base")

    # Zero config = natural standing pose for G1
    base_pose = INITIAL_BASE_POSE.clone()
    cfg = torch.zeros(robot.joints.num_actuated_joints)

    # Init handles at actual FK positions (avoids 180° orientation singularity)
    fk0 = robot.forward_kinematics(cfg, base_pose=base_pose)

    target_controls: list[tuple[str, viser.TransformControlsHandle]] = []
    for name, link_name in TARGET_SPECS:
        link_idx = robot.get_link_index(link_name)
        handle = server.scene.add_transform_controls(
            f"/targets/{name}",
            position=tuple(fk0[link_idx, :3].detach().tolist()),
            wxyz=qxyzw_to_wxyz(fk0[link_idx, 3:7].detach()),
            scale=0.12,
        )
        target_controls.append((link_name, handle))

    with server.gui.add_folder("IK"):
        timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
        reset_button = server.gui.add_button("Reset Targets")

    @reset_button.on_click
    def _(_) -> None:
        fk_curr = robot.forward_kinematics(cfg, base_pose=base_pose)
        for link_name, handle in target_controls:
            link_idx = robot.get_link_index(link_name)
            handle.position = fk_curr[link_idx, :3].detach().numpy()
            handle.wxyz = qxyzw_to_wxyz(fk_curr[link_idx, 3:7].detach())

    print("Open http://localhost:8080 in your browser")
    print("Drag transform handles to set IK targets. Press Ctrl+C to quit.")

    if profile:
        import torch.profiler
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
        )
        prof.__enter__()
        profile_steps = 0

    while True:
        start = time.time()

        targets = {
            link_name: wxyz_pos_to_se3(handle.wxyz, handle.position)
            for link_name, handle in target_controls
        }

        base_pose, cfg = br.solve_ik(
            robot=robot,
            targets=targets,
            initial_base_pose=base_pose,
            initial_cfg=cfg,
            max_iter=20,
        )

        elapsed_ms = (time.time() - start) * 1000
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * elapsed_ms

        base_frame.position = base_pose[:3].detach().numpy()
        base_frame.wxyz = qxyzw_to_wxyz(base_pose[3:7].detach())
        urdf_vis.update_cfg(build_cfg_dict(robot, cfg))

        if profile:
            profile_steps += 1
            if profile_steps >= 10:
                prof.__exit__(None, None, None)
                print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
                return

        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", action="store_true",
        help="Profile 10 IK solve calls and print CPU breakdown, then exit",
    )
    args = parser.parse_args()
    main(profile=args.profile)
```

- [ ] **Step 4: Syntax-check `02_g1_ik.py`**

```bash
uv run python -c "import ast; ast.parse(open('examples/02_g1_ik.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Run full test suite one more time**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add examples/01_basic_ik.py examples/02_g1_ik.py
git commit -m "refactor: update examples to unified solve_ik API and viewer helpers"
```

---

### Task 7: Update CLAUDE.md files

**Files:**
- Rewrite: `src/better_robot/tasks/CLAUDE.md`
- Create: `src/better_robot/viewer/CLAUDE.md`
- Modify: `CLAUDE.md` (root)

No tests. Run `uv run pytest tests/ -v` at the end to confirm nothing regressed.

- [ ] **Step 1: Rewrite `src/better_robot/tasks/CLAUDE.md`**

```markdown
# tasks/ — High-Level IK API

## `solve_ik` (`_ik.py`) — the only public IK function

```python
solve_ik(
    robot: Robot,
    targets: dict[str, Tensor],        # {link_name: (7,) SE3}
    cfg: IKConfig = IKConfig(),
    initial_cfg: Tensor | None = None,
    initial_base_pose: Tensor | None = None,  # None = fixed base
    max_iter: int = 100,
) -> Tensor | tuple[Tensor, Tensor]
```

Returns `Tensor` (fixed base) or `(base_pose, cfg)` tuple (floating base).

**Fixed base** (`initial_base_pose=None`): one `CostTerm(pose_residual)` per target,
plus shared `limit_residual` and `rest_residual`. Uses `Problem`/`SOLVER_REGISTRY["lm"]`.

**Floating base** (`initial_base_pose` provided): delegates to `_run_floating_base_lm`
in `_floating_base_ik.py`, which uses `_FloatingBaseIKModule` (a PyPose LM nn.Module
with `pp.Parameter` base and `nn.Parameter` joints).

## `IKConfig` (`_config.py`)

```python
@dataclass
class IKConfig:
    pos_weight: float = 1.0    # position component of pose error
    ori_weight: float = 0.1    # orientation component of pose error
    pose_weight: float = 1.0   # overall pose cost scale
    limit_weight: float = 0.1  # soft joint limit penalty
    rest_weight: float = 0.01  # pull toward robot._default_cfg
```

`ori_weight=0.1` is intentional — high orientation weight causes LM to oscillate near
the SO3 log singularity at 180° rotation (common for humanoids and Panda in default config).

## Usage

```python
# Fixed base, single target
cfg = solve_ik(robot, targets={"panda_hand": pose})

# Fixed base, multi-target
cfg = solve_ik(robot, targets={"panda_link6": p1, "panda_hand": p2})

# Floating base (humanoid whole-body IK), warm-started
base_pose, cfg = solve_ik(
    robot,
    targets={"left_rubber_hand": p_lh, "right_rubber_hand": p_rh,
             "left_ankle_roll_link": p_lf, "right_ankle_roll_link": p_rf},
    cfg=IKConfig(),
    initial_base_pose=base_pose,
    initial_cfg=cfg,
    max_iter=20,
)
```

## Internal structure

- `_config.py` — `IKConfig` dataclass
- `_ik.py` — `solve_ik` + `_solve_fixed` (private)
- `_floating_base_ik.py` — `_FloatingBaseIKModule` + `_run_floating_base_lm` (private)
- `_trajopt.py`, `_retarget.py` — stubs
```

- [ ] **Step 2: Create `src/better_robot/viewer/CLAUDE.md`**

```markdown
# viewer/ — Visualization Utilities

## Helpers (`_helpers.py`)

Shared conversion utilities for viser-based robot examples.
Import via `from better_robot.viewer import ...`.

| Function | Input | Output | Purpose |
|----------|-------|--------|---------|
| `wxyz_pos_to_se3(wxyz, pos)` | viser `(w,x,y,z)` + `(x,y,z)` | `(7,)` SE3 tensor | Handle → SE3 target |
| `qxyzw_to_wxyz(q)` | `(4,)` tensor `[qx,qy,qz,qw]` | `(w,x,y,z)` tuple | FK quat → viser |
| `build_cfg_dict(robot, cfg)` | Robot + `(n,)` tensor | `dict[str, float]` | For `ViserUrdf.update_cfg` |

`build_cfg_dict` filters to actuated joints (revolute, continuous, prismatic) in BFS order,
matching the order of `forward_kinematics` cfg dimension.

## `Visualizer` (`_visualizer.py`)

Thin wrapper combining `ViserServer` + `ViserUrdf` for fixed-base robots.
Accepts a numpy array in `update_cfg`. Use `ViserUrdf` directly for floating-base robots
(need separate base frame and URDF parenting).
```

- [ ] **Step 3: Update root `CLAUDE.md` — IK usage pattern section**

Find the `## IK Usage Pattern` section and replace it:

```markdown
## IK Usage Pattern

```python
robot = br.Robot.from_urdf(urdf)

# Fixed base — single or multiple targets
cfg = br.solve_ik(robot, targets={"panda_hand": pose}, max_iter=50)

# Warm-started loop (~10 iters is enough per frame)
cfg = br.solve_ik(robot, targets={"panda_hand": pose}, initial_cfg=cfg, max_iter=10)

# Custom weights via IKConfig
cfg = br.solve_ik(robot, targets={"panda_hand": pose},
                  cfg=br.IKConfig(rest_weight=0.001), max_iter=20)
```

**target_pose format**: `[tx, ty, tz, qx, qy, qz, qw]`

**Initial orientation**: initialize handles at the robot's FK orientation at default config,
not identity. Identity causes ~173° orientation error → oscillation.

```python
fk0 = robot.forward_kinematics(robot._default_cfg)
nat_q = fk0[robot.get_link_index("panda_hand"), 3:7]
target = torch.cat([position, nat_q])
```
```

Also replace the `## Floating-Base IK` section:

```markdown
## Floating-Base IK (Humanoid Whole-Body)

Pass `initial_base_pose` to `solve_ik` to enable floating-base mode.
Returns `(base_pose, cfg)` tuple instead of just `cfg`.

```python
base_pose = torch.tensor([0., 0., 0.78, 0., 0., 0., 1.])  # G1 standing height
cfg = torch.zeros(robot.joints.num_actuated_joints)          # zero = natural standing

# Warm-started loop
base_pose, cfg = br.solve_ik(
    robot=robot,
    targets={
        "left_rubber_hand":      pose_lh,
        "right_rubber_hand":     pose_rh,
        "left_ankle_roll_link":  pose_lf,
        "right_ankle_roll_link": pose_rf,
    },
    initial_base_pose=base_pose,
    initial_cfg=cfg,
    max_iter=20,
)
```

**Viser update**: `base_frame.position = base_pose[:3]` and
`base_frame.wxyz = qxyzw_to_wxyz(base_pose[3:7])`. URDF meshes are parented to `/base`.
```

- [ ] **Step 4: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/better_robot/tasks/CLAUDE.md src/better_robot/viewer/CLAUDE.md CLAUDE.md
git commit -m "docs: update CLAUDE.md files for unified solve_ik API and viewer helpers"
```

---

## Self-Review

**Spec coverage:**
- ✅ `IKConfig` dataclass — Task 1
- ✅ Unified `solve_ik` fixed-base — Task 3
- ✅ Floating-base dispatch in `solve_ik` — Task 4
- ✅ `_FloatingBaseIKModule` uses `IKConfig` — Task 4
- ✅ Viewer helpers — Task 2
- ✅ Examples updated — Task 6
- ✅ `--profile` in `02_g1_ik.py` — Task 6
- ✅ Smoke tests updated — Task 5
- ✅ CLAUDE.md updated — Task 7

**Type consistency check:**
- `IKConfig` defined in Task 1, used in Tasks 3, 4, 6, 7 — consistent field names throughout
- `_run_floating_base_lm` defined in Task 4, imported lazily in `_ik.py` from Task 3 — consistent
- `build_cfg_dict(robot, cfg)` defined in Task 2, used in Task 6 — consistent signature
- `qxyzw_to_wxyz(q)` defined in Task 2, used in Task 6 — consistent
- `wxyz_pos_to_se3(wxyz, pos)` defined in Task 2, used in Task 6 — consistent
