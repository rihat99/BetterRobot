# Analytical Jacobian & Custom LM Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an analytical (geometric) Jacobian and a custom Levenberg-Marquardt solver that works for both fixed-base and floating-base IK, switchable via `IKConfig.jacobian`.

**Architecture:** Add `Robot.get_chain` + `adjoint_se3` as Lie-algebra helpers; implement `costs/_jacobian.py` for analytical J; build `solvers/_lm.py` (our LM using `torch.func.jacrev` for autodiff mode and `problem.jacobian_fn` for analytic mode); extend `Problem` with an optional `jacobian_fn` field; add `IKConfig.jacobian` to dispatch fixed-base and floating-base paths to the analytical loop. Keep the PyPose LM registered as `"lm_pypose"` for comparison.

**Tech Stack:** PyTorch (`torch.func.jacrev`, `torch.linalg.solve`), PyPose (SE3/SO3 ops only), existing BetterRobot infrastructure.

---

## File Map

| File | Change |
|------|--------|
| `src/better_robot/core/_robot.py` | Add `Robot.get_chain(link_idx) -> list[int]` |
| `src/better_robot/core/_lie_ops.py` | Add `adjoint_se3(T) -> Tensor` |
| `src/better_robot/costs/_jacobian.py` | NEW: `pose_jacobian`, `limit_jacobian`, `rest_jacobian` |
| `src/better_robot/solvers/_base.py` | Add `jacobian_fn` field to `Problem` |
| `src/better_robot/solvers/_lm.py` | NEW: our `LevenbergMarquardt` |
| `src/better_robot/solvers/__init__.py` | Register `"lm"` → ours, `"lm_pypose"` → PyPose |
| `src/better_robot/tasks/_config.py` | Add `jacobian` field to `IKConfig` |
| `src/better_robot/tasks/_ik.py` | Build `jacobian_fn` and attach to `Problem` when analytic |
| `src/better_robot/tasks/_floating_base_ik.py` | Add analytic loop alongside PyPose path |
| `tests/test_jacobian.py` | NEW: shape + finite-diff tests |
| `tests/test_lm_benchmark.py` | NEW: correctness + timing benchmark |
| `src/better_robot/solvers/CLAUDE.md` | Update solver docs |
| `src/better_robot/costs/CLAUDE.md` | Document `_jacobian.py` |
| `src/better_robot/core/CLAUDE.md` | Document `get_chain`, `adjoint_se3` |
| `CLAUDE.md` | Update solver notes |

---

### Task 1: `Robot.get_chain`

**Files:**
- Modify: `src/better_robot/core/_robot.py`
- Test: `tests/test_robot.py`

- [ ] **Step 1: Write the failing test**

Open `tests/test_robot.py` and add at the end:

```python
def test_get_chain_panda_hand(panda):
    """Chain from root to panda_hand has exactly 7 actuated joints."""
    hand_idx = panda.get_link_index("panda_hand")
    chain = panda.get_chain(hand_idx)
    assert len(chain) == 7
    # All returned indices are actuated joints
    assert all(panda._fk_cfg_indices[j] >= 0 for j in chain)


def test_get_chain_root_link(panda):
    """Chain to root link is empty."""
    chain = panda.get_chain(panda._root_link_idx)
    assert chain == []


def test_get_chain_ordering(panda):
    """Chain is in root→EE topological order (parent joint before child joint)."""
    hand_idx = panda.get_link_index("panda_hand")
    chain = panda.get_chain(hand_idx)
    # cfg indices should be 0,1,2,... (BFS order for Panda arm)
    cfg_indices = [panda._fk_cfg_indices[j] for j in chain]
    assert cfg_indices == sorted(cfg_indices)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_robot.py::test_get_chain_panda_hand -v
```
Expected: `FAILED` — `AttributeError: 'Robot' object has no attribute 'get_chain'`

- [ ] **Step 3: Implement `get_chain`**

In `src/better_robot/core/_robot.py`, add after `get_link_index`:

```python
def get_chain(self, link_idx: int) -> list[int]:
    """Joint indices (actuated only) on the path from root to link_idx.

    Args:
        link_idx: Target link index.

    Returns:
        List of joint indices in root→EE order (only actuated joints).
        Empty list if link_idx is the root.
    """
    # Map child_link → joint_index for fast lookup
    child_to_joint: dict[int, int] = {
        child: j for j, child in enumerate(self._fk_joint_child_link)
    }

    chain: list[int] = []
    current = link_idx
    while current != self._root_link_idx:
        if current not in child_to_joint:
            break
        j = child_to_joint[current]
        if self._fk_cfg_indices[j] >= 0:   # actuated joints only
            chain.append(j)
        current = self._fk_joint_parent_link[j]

    chain.reverse()   # was built EE→root, reverse to root→EE
    return chain
```

- [ ] **Step 4: Run to verify passing**

```bash
uv run pytest tests/test_robot.py -v
```
Expected: all robot tests pass, including the three new ones.

- [ ] **Step 5: Commit**

```bash
git add src/better_robot/core/_robot.py tests/test_robot.py
git commit -m "feat: add Robot.get_chain for kinematic chain traversal"
```

---

### Task 2: `adjoint_se3`

**Files:**
- Modify: `src/better_robot/core/_lie_ops.py`
- Test: `tests/test_robot.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_robot.py`:

```python
import torch
from better_robot.core._lie_ops import adjoint_se3, se3_identity, se3_inverse


def test_adjoint_se3_identity():
    """Ad(identity) = I_6."""
    T = se3_identity()
    Ad = adjoint_se3(T)
    assert Ad.shape == (6, 6)
    assert torch.allclose(Ad, torch.eye(6), atol=1e-6)


def test_adjoint_se3_pure_translation():
    """Ad([1,0,0, 0,0,0,1]) has skew(p)@R in top-right block."""
    T = torch.tensor([1.0, 0.0, 0.0,  0.0, 0.0, 0.0, 1.0])  # x=1, identity rotation
    Ad = adjoint_se3(T)
    # Top-left: R = I
    assert torch.allclose(Ad[:3, :3], torch.eye(3), atol=1e-6)
    # Bottom-right: R = I
    assert torch.allclose(Ad[3:, 3:], torch.eye(3), atol=1e-6)
    # Bottom-left: zeros
    assert torch.allclose(Ad[3:, :3], torch.zeros(3, 3), atol=1e-6)
    # Top-right: skew([1,0,0]) @ I = [[0,0,0],[0,0,-1],[0,1,0]]
    expected_top_right = torch.tensor([
        [ 0.,  0.,  0.],
        [ 0.,  0., -1.],
        [ 0.,  1.,  0.],
    ])
    assert torch.allclose(Ad[:3, 3:], expected_top_right, atol=1e-6)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_robot.py::test_adjoint_se3_identity -v
```
Expected: `FAILED` — `ImportError: cannot import name 'adjoint_se3'`

- [ ] **Step 3: Implement `adjoint_se3`**

Add to `src/better_robot/core/_lie_ops.py`:

```python
def adjoint_se3(T: torch.Tensor) -> torch.Tensor:
    """6×6 Adjoint matrix of SE3 element T.

    For PyPose se3 tangent convention [tx, ty, tz, rx, ry, rz]:

        Ad(T) = [[R,          skew(p) @ R],
                  [zeros(3,3), R          ]]

    where p = T[:3], R = rotation_matrix(T[3:7]).

    Args:
        T: (7,) SE3 pose [tx, ty, tz, qx, qy, qz, qw].

    Returns:
        (6, 6) Adjoint matrix.
    """
    p = T[:3]
    R = pp.SO3(T[3:7]).matrix()   # (3, 3)

    skew = torch.zeros(3, 3, dtype=T.dtype, device=T.device)
    skew[0, 1] = -p[2];  skew[0, 2] =  p[1]
    skew[1, 0] =  p[2];  skew[1, 2] = -p[0]
    skew[2, 0] = -p[1];  skew[2, 1] =  p[0]

    Ad = torch.zeros(6, 6, dtype=T.dtype, device=T.device)
    Ad[:3, :3] = R
    Ad[:3, 3:] = skew @ R
    Ad[3:, 3:] = R
    return Ad
```

- [ ] **Step 4: Run to verify passing**

```bash
uv run pytest tests/test_robot.py -v
```
Expected: all robot tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/better_robot/core/_lie_ops.py tests/test_robot.py
git commit -m "feat: add adjoint_se3 to lie_ops"
```

---

### Task 3: Analytical Jacobian Functions

**Files:**
- Create: `src/better_robot/costs/_jacobian.py`
- Modify: `src/better_robot/costs/__init__.py`
- Create: `tests/test_jacobian.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_jacobian.py`:

```python
"""Tests for analytical Jacobian functions."""

import torch
import pytest
from robot_descriptions.loaders.yourdfpy import load_robot_description
import better_robot as br
from better_robot.costs._jacobian import pose_jacobian, limit_jacobian, rest_jacobian
from better_robot.costs._pose import pose_residual


@pytest.fixture(scope="module")
def panda():
    urdf = load_robot_description("panda_description")
    return br.Robot.from_urdf(urdf)


def test_pose_jacobian_shape(panda):
    cfg = panda._default_cfg
    link_idx = panda.get_link_index("panda_hand")
    target = panda.forward_kinematics(cfg)[link_idx].detach()
    J = pose_jacobian(cfg, panda, link_idx, target, 1.0, 0.1)
    assert J.shape == (6, panda.joints.num_actuated_joints)


def test_pose_jacobian_finite_diff(panda):
    """Analytical Jacobian matches central finite differences."""
    eps = 1e-5
    cfg = panda._default_cfg.clone()
    link_idx = panda.get_link_index("panda_hand")
    target = panda.forward_kinematics(cfg)[link_idx].detach()

    J_analytic = pose_jacobian(cfg, panda, link_idx, target, 1.0, 0.1)

    n = len(cfg)
    J_fd = torch.zeros(6, n)
    for i in range(n):
        cp = cfg.clone(); cp[i] += eps
        cm = cfg.clone(); cm[i] -= eps
        rp = pose_residual(cp, panda, link_idx, target, 1.0, 0.1)
        rm = pose_residual(cm, panda, link_idx, target, 1.0, 0.1)
        J_fd[:, i] = (rp - rm) / (2 * eps)

    assert torch.allclose(J_analytic, J_fd, atol=1e-3), \
        f"Max diff: {(J_analytic - J_fd).abs().max():.5f}"


def test_limit_jacobian_shape(panda):
    cfg = panda._default_cfg
    J = limit_jacobian(cfg, panda)
    n = panda.joints.num_actuated_joints
    assert J.shape == (2 * n, n)


def test_limit_jacobian_inside_limits_is_zero(panda):
    """Within limits, limit_jacobian rows are all zero."""
    cfg = panda._default_cfg  # midpoint — guaranteed inside limits
    J = limit_jacobian(cfg, panda)
    assert torch.all(J == 0.0)


def test_limit_jacobian_violated_lower(panda):
    """Below lower limit: lower violation row has -1 on diagonal."""
    cfg = panda.joints.lower_limits.clone() - 0.1  # below lower limit
    J = limit_jacobian(cfg, panda)
    n = panda.joints.num_actuated_joints
    # Lower violation rows (top n): diagonal should be -1
    assert torch.allclose(J[:n], torch.diag(torch.full((n,), -1.0)), atol=1e-6)


def test_rest_jacobian_is_identity(panda):
    cfg = panda._default_cfg
    rest = panda._default_cfg.clone()
    J = rest_jacobian(cfg, rest)
    n = panda.joints.num_actuated_joints
    assert J.shape == (n, n)
    assert torch.allclose(J, torch.eye(n), atol=1e-6)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_jacobian.py -v
```
Expected: `FAILED` — `ModuleNotFoundError: No module named 'better_robot.costs._jacobian'`

- [ ] **Step 3: Implement `costs/_jacobian.py`**

Create `src/better_robot/costs/_jacobian.py`:

```python
"""Analytical Jacobian functions for IK cost terms.

All functions return the Jacobian dR/dcfg where R is the corresponding
residual vector (from _pose.py, _limits.py, _regularization.py).

Convention
----------
- pose_jacobian applies pos_weight and ori_weight internally (matching pose_residual).
- jlog is approximated as identity (valid for errors < ~30°); noted in CLAUDE.md.
"""

from __future__ import annotations

import pypose as pp
import torch

from ..core._robot import Robot
from ..core._lie_ops import se3_compose, adjoint_se3


def pose_jacobian(
    cfg: torch.Tensor,
    robot: Robot,
    target_link_index: int,
    target_pose: torch.Tensor,
    pos_weight: float,
    ori_weight: float,
    base_pose: torch.Tensor | None = None,
) -> torch.Tensor:
    """Geometric Jacobian of pose_residual wrt cfg.

    Args:
        cfg: (num_actuated_joints,) current joint config.
        robot: Robot instance.
        target_link_index: Index of the target link.
        target_pose: (7,) SE3 target [tx, ty, tz, qx, qy, qz, qw]. Unused here
            (Jacobian at current cfg, not at target); kept for API symmetry.
        pos_weight: Weight on position rows (matches pose_residual).
        ori_weight: Weight on orientation rows (matches pose_residual).
        base_pose: (7,) optional floating base SE3. Passed to forward_kinematics.

    Returns:
        (6, num_actuated_joints) Jacobian matrix.
    """
    device, dtype = cfg.device, cfg.dtype
    n = robot.joints.num_actuated_joints

    fk = robot.forward_kinematics(cfg, base_pose=base_pose)   # (num_links, 7)
    p_ee = fk[target_link_index, :3]

    chain = robot.get_chain(target_link_index)

    J = torch.zeros(6, n, dtype=dtype, device=device)

    for j in chain:
        cfg_idx = robot._fk_cfg_indices[j]
        parent_link = robot._fk_joint_parent_link[j]

        T_parent = fk[parent_link]                                        # (7,) world
        T_origin = robot._fk_joint_origins[j].to(device=device, dtype=dtype)  # (7,)
        T_j = se3_compose(T_parent, T_origin)                            # joint frame in world

        p_j = T_j[:3]                                                    # joint origin (world)
        local_axis = robot._fk_joint_axes[j].to(device=device, dtype=dtype)  # (3,)
        axis_world = pp.SO3(T_j[3:7]).Act(local_axis)                   # axis in world frame

        jtype = robot._fk_joint_types[j]
        if jtype in ("revolute", "continuous"):
            lin = torch.linalg.cross(axis_world, p_ee - p_j)
            J[:, cfg_idx] = torch.cat([lin * pos_weight, axis_world * ori_weight])
        else:  # prismatic
            J[:, cfg_idx] = torch.cat([
                axis_world * pos_weight,
                torch.zeros(3, dtype=dtype, device=device),
            ])

    return J


def limit_jacobian(cfg: torch.Tensor, robot: Robot) -> torch.Tensor:
    """Jacobian of limit_residual wrt cfg.

    limit_residual returns [clamp(lo-cfg, min=0), clamp(cfg-hi, min=0)].
    Derivative: -1 where cfg < lo (lower block), +1 where cfg > hi (upper block).

    Returns:
        (2 * num_actuated_joints, num_actuated_joints) diagonal Jacobian.
    """
    n = robot.joints.num_actuated_joints
    lo = robot.joints.lower_limits.to(device=cfg.device, dtype=cfg.dtype)
    hi = robot.joints.upper_limits.to(device=cfg.device, dtype=cfg.dtype)

    lower_grad = torch.where(cfg < lo, torch.full_like(cfg, -1.0), torch.zeros_like(cfg))
    upper_grad = torch.where(cfg > hi, torch.ones_like(cfg), torch.zeros_like(cfg))

    J = torch.zeros(2 * n, n, dtype=cfg.dtype, device=cfg.device)
    J[:n] = torch.diag(lower_grad)
    J[n:] = torch.diag(upper_grad)
    return J


def rest_jacobian(cfg: torch.Tensor, rest_pose: torch.Tensor) -> torch.Tensor:
    """Jacobian of rest_residual wrt cfg.

    rest_residual = cfg - rest_pose, so J = I.

    Returns:
        (num_actuated_joints, num_actuated_joints) identity matrix.
    """
    n = len(cfg)
    return torch.eye(n, dtype=cfg.dtype, device=cfg.device)
```

- [ ] **Step 4: Export from `costs/__init__.py`**

Open `src/better_robot/costs/__init__.py` and add:

```python
from ._jacobian import pose_jacobian as pose_jacobian
from ._jacobian import limit_jacobian as limit_jacobian
from ._jacobian import rest_jacobian as rest_jacobian
```

- [ ] **Step 5: Run to verify passing**

```bash
uv run pytest tests/test_jacobian.py -v
```
Expected: all 7 tests pass.

- [ ] **Step 6: Run full suite to check no regressions**

```bash
uv run pytest tests/ -v
```
Expected: all existing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add src/better_robot/costs/_jacobian.py src/better_robot/costs/__init__.py tests/test_jacobian.py
git commit -m "feat: analytical Jacobian functions (pose, limit, rest)"
```

---

### Task 4: `Problem.jacobian_fn` Field

**Files:**
- Modify: `src/better_robot/solvers/_base.py`
- Test: `tests/test_solvers.py`

- [ ] **Step 1: Write the failing test**

Open `tests/test_solvers.py` and add:

```python
def test_problem_jacobian_fn_is_called():
    """When jacobian_fn is set on Problem, it is called by our LM."""
    from better_robot.solvers._lm import LevenbergMarquardt
    calls = []

    def residual(x):
        return x  # identity residual

    def jac_fn(x):
        calls.append(1)
        return torch.eye(len(x))

    problem = Problem(
        variables=torch.zeros(3),
        costs=[CostTerm(residual_fn=residual, weight=1.0)],
        lower_bounds=None,
        upper_bounds=None,
        jacobian_fn=jac_fn,
    )
    LevenbergMarquardt().solve(problem, max_iter=1)
    assert len(calls) >= 1, "jacobian_fn was never called"
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_solvers.py::test_problem_jacobian_fn_is_called -v
```
Expected: `FAILED` — `TypeError: Problem.__init__() got an unexpected keyword argument 'jacobian_fn'`

- [ ] **Step 3: Add `jacobian_fn` to `Problem`**

In `src/better_robot/solvers/_base.py`, add the import and field:

```python
from typing import Callable, Literal, Optional
```

Add after `upper_bounds` in the `Problem` dataclass:

```python
    jacobian_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    """Optional analytical Jacobian function.

    If provided, our LM calls jacobian_fn(x) -> (m, n) Tensor instead of
    computing J via torch.func.jacrev. The PyPose LM ignores this field.
    """
```

- [ ] **Step 4: Run to verify still failing (our LM not created yet)**

```bash
uv run pytest tests/test_solvers.py::test_problem_jacobian_fn_is_called -v
```
Expected: `FAILED` — `ImportError: cannot import name 'LevenbergMarquardt' from 'better_robot.solvers._lm'`

- [ ] **Step 5: Run full suite to check no regressions**

```bash
uv run pytest tests/ -v
```
Expected: all existing tests still pass (`jacobian_fn` defaults to `None`).

- [ ] **Step 6: Commit**

```bash
git add src/better_robot/solvers/_base.py
git commit -m "feat: add optional jacobian_fn field to Problem"
```

---

### Task 5: Our LM Solver

**Files:**
- Create: `src/better_robot/solvers/_lm.py`
- Modify: `src/better_robot/solvers/__init__.py`
- Test: `tests/test_solvers.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_solvers.py`:

```python
from better_robot.solvers._lm import LevenbergMarquardt as OurLM


def test_our_lm_converges_quadratic():
    """Our LM converges on a simple quadratic (same as existing LM test)."""
    # Minimize ||Ax - b||^2 where A=I, b=[1,2,3] → solution x=[1,2,3]
    b = torch.tensor([1.0, 2.0, 3.0])

    def residual(x):
        return x - b

    problem = Problem(
        variables=torch.zeros(3),
        costs=[CostTerm(residual_fn=residual, weight=1.0)],
        lower_bounds=None,
        upper_bounds=None,
    )
    result = OurLM().solve(problem, max_iter=10)
    assert torch.allclose(result, b, atol=1e-4), f"Expected {b}, got {result}"


def test_our_lm_autodiff_matches_pypose_lm(panda):
    """Our LM (autodiff mode) gives same IK result as PyPose LM (within 1 cm)."""
    from better_robot.solvers._levenberg_marquardt import LevenbergMarquardt as PyposeLM
    import functools
    from better_robot.costs._pose import pose_residual
    from better_robot.costs._limits import limit_residual
    from better_robot.costs._regularization import rest_residual

    cfg0 = panda._default_cfg
    hand_idx = panda.get_link_index("panda_hand")
    target = panda.forward_kinematics(cfg0)[hand_idx].detach()
    target[0] += 0.05  # small offset so there's something to solve

    def make_problem():
        costs = [
            CostTerm(functools.partial(pose_residual, robot=panda, target_link_index=hand_idx,
                                       target_pose=target, pos_weight=1.0, ori_weight=0.1), weight=1.0),
            CostTerm(functools.partial(limit_residual, robot=panda), weight=0.1),
            CostTerm(functools.partial(rest_residual, rest_pose=panda._default_cfg), weight=0.01),
        ]
        return Problem(variables=cfg0.clone(), costs=costs,
                       lower_bounds=panda.joints.lower_limits.clone(),
                       upper_bounds=panda.joints.upper_limits.clone())

    result_ours = OurLM().solve(make_problem(), max_iter=20)
    result_pypose = PyposeLM().solve(make_problem(), max_iter=20)

    fk_ours = panda.forward_kinematics(result_ours)
    fk_pypose = panda.forward_kinematics(result_pypose)
    err_ours = (fk_ours[hand_idx, :3] - target[:3]).norm().item()
    err_pypose = (fk_pypose[hand_idx, :3] - target[:3]).norm().item()
    assert abs(err_ours - err_pypose) < 0.01, \
        f"Our LM err={err_ours:.4f}, PyPose err={err_pypose:.4f}"


def test_our_lm_uses_jacobian_fn_when_provided(panda):
    """Our LM calls jacobian_fn when problem.jacobian_fn is set."""
    import functools
    from better_robot.costs._pose import pose_residual
    from better_robot.costs._jacobian import pose_jacobian, limit_jacobian, rest_jacobian

    cfg0 = panda._default_cfg
    hand_idx = panda.get_link_index("panda_hand")
    target = panda.forward_kinematics(cfg0)[hand_idx].detach()
    rest = panda._default_cfg.clone()

    costs = [
        CostTerm(functools.partial(pose_residual, robot=panda, target_link_index=hand_idx,
                                   target_pose=target, pos_weight=1.0, ori_weight=0.1), weight=1.0),
        CostTerm(functools.partial(limit_residual, robot=panda), weight=0.1),
        CostTerm(functools.partial(rest_residual, rest_pose=rest), weight=0.01),
    ]

    def jac_fn(cfg):
        rows = [
            pose_jacobian(cfg, panda, hand_idx, target, 1.0, 0.1) * 1.0,
            limit_jacobian(cfg, panda) * 0.1,
            rest_jacobian(cfg, rest) * 0.01,
        ]
        return torch.cat(rows, dim=0)

    problem = Problem(
        variables=cfg0.clone(), costs=costs,
        lower_bounds=panda.joints.lower_limits.clone(),
        upper_bounds=panda.joints.upper_limits.clone(),
        jacobian_fn=jac_fn,
    )
    result = OurLM().solve(problem, max_iter=5)
    fk = panda.forward_kinematics(result)
    assert (fk[hand_idx, :3] - target[:3]).norm().item() < 0.05
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_solvers.py::test_our_lm_converges_quadratic -v
```
Expected: `FAILED` — `ImportError`

- [ ] **Step 3: Create `solvers/_lm.py`**

Create `src/better_robot/solvers/_lm.py`:

```python
"""Custom Levenberg-Marquardt solver.

Supports two Jacobian modes (set via problem.jacobian_fn):
  - autodiff:  J = torch.func.jacrev(problem.total_residual)(x)
  - analytic:  J = problem.jacobian_fn(x)   [user-provided]

Parameters mirror PyPose's LevenbergMarquardt:
  damping  — initial lambda (same as Adaptive(damping=...))
  factor   — multiplicative scale for adaptive damping
  reject   — max retries per step before accepting increased lambda
"""

from __future__ import annotations

import torch

from ._base import Problem, Solver


class LevenbergMarquardt(Solver):
    """LM solver. Uses torch.func.jacrev when problem.jacobian_fn is None."""

    def __init__(
        self,
        damping: float = 1e-4,
        factor: float = 2.0,
        reject: int = 16,
    ) -> None:
        self.damping = damping
        self.factor = factor
        self.reject = reject

    def solve(self, problem: Problem, max_iter: int = 100) -> torch.Tensor:
        x = problem.variables.clone().float()
        lam = float(self.damping)
        lo = problem.lower_bounds
        hi = problem.upper_bounds

        for _ in range(max_iter):
            r = problem.total_residual(x)                    # (m,)
            J = self._jacobian(problem, x)                   # (m, n)
            JtJ = J.T @ J                                    # (n, n)
            Jtr = J.T @ r                                    # (n,)
            n_vars = x.shape[0]

            for _ in range(self.reject):
                A = JtJ + lam * torch.eye(n_vars, dtype=x.dtype, device=x.device)
                delta = torch.linalg.solve(A, -Jtr)
                x_new = x + delta
                if lo is not None and hi is not None:
                    x_new = x_new.clamp(
                        lo.to(dtype=x.dtype, device=x.device),
                        hi.to(dtype=x.dtype, device=x.device),
                    )
                if problem.total_residual(x_new).norm() <= r.norm():
                    x = x_new
                    lam = max(lam / self.factor, 1e-7)
                    break
                lam = min(lam * self.factor, 1e7)

        return x

    def _jacobian(self, problem: Problem, x: torch.Tensor) -> torch.Tensor:
        if problem.jacobian_fn is not None:
            return problem.jacobian_fn(x)
        return torch.func.jacrev(problem.total_residual)(x)
```

- [ ] **Step 4: Update `solvers/__init__.py`**

Replace the content of `src/better_robot/solvers/__init__.py`:

```python
"""Solver layer: swappable optimization backends."""

from ._base import CostTerm as CostTerm, Problem as Problem, Solver as Solver
from ._lm import LevenbergMarquardt as LevenbergMarquardt
from ._levenberg_marquardt import LevenbergMarquardt as PyposeLevenbergMarquardt
from ._gauss_newton import GaussNewton as GaussNewton
from ._adam import AdamSolver as AdamSolver
from ._lbfgs import LBFGSSolver as LBFGSSolver

SOLVER_REGISTRY: dict[str, type[Solver]] = {
    "lm": LevenbergMarquardt,           # our LM (autodiff or analytic via jacobian_fn)
    "lm_pypose": PyposeLevenbergMarquardt,  # PyPose LM (kept for comparison)
    "gn": GaussNewton,
    "adam": AdamSolver,
    "lbfgs": LBFGSSolver,
}

__all__ = [
    "CostTerm", "Problem", "Solver",
    "LevenbergMarquardt", "PyposeLevenbergMarquardt",
    "GaussNewton", "AdamSolver", "LBFGSSolver",
    "SOLVER_REGISTRY",
]
```

- [ ] **Step 5: Run to verify passing**

```bash
uv run pytest tests/test_solvers.py -v
```
Expected: all solver tests pass, including the three new ones.

- [ ] **Step 6: Run full suite**

```bash
uv run pytest tests/ -v
```
Expected: all 34+ tests pass. The existing IK tests now run through our LM (registered as `"lm"`).

- [ ] **Step 7: Commit**

```bash
git add src/better_robot/solvers/_lm.py src/better_robot/solvers/__init__.py tests/test_solvers.py
git commit -m "feat: custom LM solver (autodiff + analytical via jacobian_fn)"
```

---

### Task 6: `IKConfig.jacobian` + Fixed-Base Analytic Dispatch

**Files:**
- Modify: `src/better_robot/tasks/_config.py`
- Modify: `src/better_robot/tasks/_ik.py`
- Test: `tests/test_ik.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ik.py`:

```python
def test_solve_ik_analytic_converges(panda):
    """IKConfig(jacobian='analytic') gives same convergence as autodiff."""
    cfg0 = panda._default_cfg
    hand_idx = panda.get_link_index("panda_hand")
    fk0 = panda.forward_kinematics(cfg0)
    target = fk0[hand_idx].detach()

    cfg_analytic = solve_ik(
        panda,
        targets={"panda_hand": target},
        cfg=IKConfig(jacobian="analytic"),
        initial_cfg=cfg0.clone(),
        max_iter=5,
    )
    fk = panda.forward_kinematics(cfg_analytic)
    pos_err = (fk[hand_idx, :3] - target[:3]).norm().item()
    assert pos_err < 0.05, f"Analytic IK pos_err={pos_err:.4f}"


def test_solve_ik_analytic_matches_autodiff(panda):
    """Analytic and autodiff IK give solutions within 1 cm of each other."""
    cfg0 = panda._default_cfg
    hand_idx = panda.get_link_index("panda_hand")
    target = panda.forward_kinematics(cfg0)[hand_idx].detach().clone()
    target[0] += 0.05

    cfg_autodiff = solve_ik(
        panda, targets={"panda_hand": target},
        cfg=IKConfig(jacobian="autodiff"), initial_cfg=cfg0.clone(), max_iter=20,
    )
    cfg_analytic = solve_ik(
        panda, targets={"panda_hand": target},
        cfg=IKConfig(jacobian="analytic"), initial_cfg=cfg0.clone(), max_iter=20,
    )

    fk_ad = panda.forward_kinematics(cfg_autodiff)
    fk_an = panda.forward_kinematics(cfg_analytic)
    err_diff = (fk_ad[hand_idx, :3] - fk_an[hand_idx, :3]).norm().item()
    assert err_diff < 0.01, f"Solutions differ by {err_diff:.4f} m"
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_ik.py::test_solve_ik_analytic_converges -v
```
Expected: `FAILED` — `IKConfig() got an unexpected keyword argument 'jacobian'`

- [ ] **Step 3: Add `jacobian` to `IKConfig`**

Replace the content of `src/better_robot/tasks/_config.py`:

```python
"""IK solver configuration."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class IKConfig:
    """Configuration for all IK solvers."""

    pos_weight: float = 1.0    # position component of pose error
    ori_weight: float = 0.1    # orientation component of pose error
    pose_weight: float = 1.0   # overall pose cost scale
    limit_weight: float = 0.1  # soft joint limit penalty
    rest_weight: float = 0.01  # pull toward robot._default_cfg
    jacobian: Literal["autodiff", "analytic"] = "autodiff"
    """Jacobian computation mode.

    "autodiff": torch.func.jacrev (default, works for all cost terms).
    "analytic": geometric Jacobian (faster; fixed-base and floating-base supported).
    """
```

- [ ] **Step 4: Update `_solve_fixed` in `tasks/_ik.py`**

Open `src/better_robot/tasks/_ik.py`. Add imports at the top:

```python
import functools

import torch

from ..core._robot import Robot
from ..costs._pose import pose_residual
from ..costs._limits import limit_residual
from ..costs._regularization import rest_residual
from ..costs._jacobian import pose_jacobian, limit_jacobian, rest_jacobian
from ..solvers._base import CostTerm, Problem
from ..solvers import SOLVER_REGISTRY
from ._config import IKConfig
```

Replace the `_solve_fixed` function:

```python
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

    jac_fn = None
    if ik_cfg.jacobian == "analytic":
        jac_fn = _build_fixed_jacobian_fn(robot, targets, ik_cfg, rest)

    problem = Problem(
        variables=initial,
        costs=costs,
        lower_bounds=robot.joints.lower_limits.clone(),
        upper_bounds=robot.joints.upper_limits.clone(),
        jacobian_fn=jac_fn,
    )
    return SOLVER_REGISTRY["lm"]().solve(problem, max_iter=max_iter)


def _build_fixed_jacobian_fn(
    robot: Robot,
    targets: dict[str, torch.Tensor],
    ik_cfg: IKConfig,
    rest: torch.Tensor,
):
    """Build analytical Jacobian for the full IK problem (fixed base)."""
    target_specs = [
        (robot.get_link_index(name), pose) for name, pose in targets.items()
    ]

    def jacobian_fn(cfg: torch.Tensor) -> torch.Tensor:
        rows = []
        for link_idx, target_pose in target_specs:
            J = pose_jacobian(
                cfg, robot, link_idx, target_pose,
                ik_cfg.pos_weight, ik_cfg.ori_weight,
            )
            rows.append(J * ik_cfg.pose_weight)
        rows.append(limit_jacobian(cfg, robot) * ik_cfg.limit_weight)
        rows.append(rest_jacobian(cfg, rest) * ik_cfg.rest_weight)
        return torch.cat(rows, dim=0)

    return jacobian_fn
```

- [ ] **Step 5: Run to verify passing**

```bash
uv run pytest tests/test_ik.py -v
```
Expected: all IK tests pass, including the two new analytic ones.

- [ ] **Step 6: Commit**

```bash
git add src/better_robot/tasks/_config.py src/better_robot/tasks/_ik.py tests/test_ik.py
git commit -m "feat: IKConfig.jacobian field + analytic dispatch for fixed-base IK"
```

---

### Task 7: Floating-Base Analytic Path

**Files:**
- Modify: `src/better_robot/tasks/_floating_base_ik.py`
- Test: `tests/test_ik.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ik.py`:

```python
def test_solve_ik_floating_analytic_converges(panda):
    """Floating-base IK with jacobian='analytic' converges to reachable target."""
    cfg0 = panda._default_cfg
    hand_idx = panda.get_link_index("panda_hand")
    identity_base = torch.tensor([0., 0., 0., 0., 0., 0., 1.])
    fk0 = panda.forward_kinematics(cfg0)
    target = fk0[hand_idx].detach()

    base_pose, cfg = solve_ik(
        panda,
        targets={"panda_hand": target},
        cfg=IKConfig(jacobian="analytic"),
        initial_base_pose=identity_base,
        initial_cfg=cfg0.clone(),
        max_iter=5,
    )
    fk = panda.forward_kinematics(cfg, base_pose=base_pose)
    pos_err = (fk[hand_idx, :3] - target[:3]).norm().item()
    assert pos_err < 0.05, f"Floating analytic pos_err={pos_err:.4f}"


def test_solve_ik_floating_analytic_matches_pypose(panda):
    """Floating analytic and PyPose LM give solutions within 2 cm."""
    cfg0 = panda._default_cfg
    hand_idx = panda.get_link_index("panda_hand")
    identity_base = torch.tensor([0., 0., 0., 0., 0., 0., 1.])
    target = panda.forward_kinematics(cfg0)[hand_idx].detach().clone()
    target[0] += 0.05

    base_pypose, cfg_pypose = solve_ik(
        panda, targets={"panda_hand": target},
        cfg=IKConfig(jacobian="autodiff"),
        initial_base_pose=identity_base.clone(), initial_cfg=cfg0.clone(), max_iter=20,
    )
    base_analytic, cfg_analytic = solve_ik(
        panda, targets={"panda_hand": target},
        cfg=IKConfig(jacobian="analytic"),
        initial_base_pose=identity_base.clone(), initial_cfg=cfg0.clone(), max_iter=20,
    )

    fk_pp = panda.forward_kinematics(cfg_pypose, base_pose=base_pypose)
    fk_an = panda.forward_kinematics(cfg_analytic, base_pose=base_analytic)
    diff = (fk_pp[hand_idx, :3] - fk_an[hand_idx, :3]).norm().item()
    assert diff < 0.02, f"Floating solutions differ by {diff:.4f} m"
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_ik.py::test_solve_ik_floating_analytic_converges -v
```
Expected: `FAILED` — analytic path not yet dispatched for floating base.

- [ ] **Step 3: Implement floating-base analytic path**

Open `src/better_robot/tasks/_floating_base_ik.py`. Add imports:

```python
from ..costs._jacobian import pose_jacobian, limit_jacobian, rest_jacobian
from ..core._lie_ops import se3_exp, adjoint_se3
```

Add the helper and the new entry function. Insert before `_run_floating_base_lm`:

```python
def _fb_residual(
    cfg: torch.Tensor,
    base: torch.Tensor,
    robot: Robot,
    target_link_indices: list[int],
    target_poses: list[torch.Tensor],
    ik_cfg: IKConfig,
    rest: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (fk, residual_vector) for floating-base IK."""
    fk = robot.forward_kinematics(cfg, base_pose=base)
    parts = []
    for link_idx, tp in zip(target_link_indices, target_poses):
        T_err = se3_compose(se3_inverse(tp), fk[link_idx])
        log_e = se3_log(T_err)
        parts.append(
            torch.cat([log_e[:3] * ik_cfg.pos_weight, log_e[3:] * ik_cfg.ori_weight])
            * ik_cfg.pose_weight
        )
    parts.append(limit_residual(cfg, robot) * ik_cfg.limit_weight)
    parts.append(rest_residual(cfg, rest) * ik_cfg.rest_weight)
    return fk, torch.cat(parts)


def _run_floating_base_lm_analytic(
    robot: Robot,
    targets: dict[str, torch.Tensor],
    ik_cfg: IKConfig,
    initial_cfg: torch.Tensor | None,
    initial_base_pose: torch.Tensor,
    max_iter: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Floating-base LM using the analytical Jacobian.

    Variables: [cfg (n,) | base_tangent (6,)].
    Base updated via SE3 retraction: base_new = se3_exp(delta_base) ⊕ base.
    """
    base = initial_base_pose.clone().float()
    cfg = (
        initial_cfg.clone().float()
        if initial_cfg is not None
        else robot._default_cfg.clone().float()
    )
    rest = robot._default_cfg.clone().float()
    lo = robot.joints.lower_limits.float()
    hi = robot.joints.upper_limits.float()

    target_link_indices = [robot.get_link_index(name) for name in targets]
    target_poses = [tp.float() for tp in targets.values()]

    n = robot.joints.num_actuated_joints
    lam = 1e-4
    factor = 2.0
    reject = 16
    w_vec = cfg.new_tensor([ik_cfg.pos_weight] * 3 + [ik_cfg.ori_weight] * 3)

    for _ in range(max_iter):
        fk, r = _fb_residual(cfg, base, robot, target_link_indices, target_poses, ik_cfg, rest)

        # Build Jacobian: (m, n+6) — joint cols first, then 6 base cols
        J_rows = []
        for link_idx, target_pose in zip(target_link_indices, target_poses):
            J_joints = pose_jacobian(
                cfg, robot, link_idx, target_pose,
                ik_cfg.pos_weight, ik_cfg.ori_weight, base_pose=base,
            ) * ik_cfg.pose_weight                                             # (6, n)
            T_ee_local = se3_compose(se3_inverse(base), fk[link_idx])
            Ad = adjoint_se3(se3_inverse(T_ee_local))                          # (6, 6)
            J_base = w_vec.unsqueeze(1) * Ad * ik_cfg.pose_weight              # (6, 6)
            J_rows.append(torch.cat([J_joints, J_base], dim=1))                # (6, n+6)

        J_lim = torch.cat([
            limit_jacobian(cfg, robot) * ik_cfg.limit_weight,
            torch.zeros(2 * n, 6, dtype=cfg.dtype, device=cfg.device),
        ], dim=1)
        J_rest = torch.cat([
            rest_jacobian(cfg, rest) * ik_cfg.rest_weight,
            torch.zeros(n, 6, dtype=cfg.dtype, device=cfg.device),
        ], dim=1)
        J = torch.cat(J_rows + [J_lim, J_rest], dim=0)                        # (m, n+6)

        JtJ = J.T @ J
        Jtr = J.T @ r

        for _ in range(reject):
            A = JtJ + lam * torch.eye(n + 6, dtype=cfg.dtype, device=cfg.device)
            delta = torch.linalg.solve(A, -Jtr)

            cfg_new = (cfg + delta[:n]).clamp(lo, hi)
            base_new = se3_compose(se3_exp(delta[n:]), base)
            base_new = torch.cat([base_new[:3], base_new[3:7] / base_new[3:7].norm()])

            _, r_new = _fb_residual(
                cfg_new, base_new, robot, target_link_indices, target_poses, ik_cfg, rest
            )
            if r_new.norm() <= r.norm():
                cfg, base = cfg_new, base_new
                lam = max(lam / factor, 1e-7)
                break
            lam = min(lam * factor, 1e7)

    return base.detach(), cfg.detach()
```

- [ ] **Step 4: Dispatch to analytic path in `_run_floating_base_lm`**

In `src/better_robot/tasks/_floating_base_ik.py`, update `_run_floating_base_lm` to dispatch:

```python
def _run_floating_base_lm(
    robot: Robot,
    targets: dict[str, torch.Tensor],
    ik_cfg: IKConfig,
    initial_cfg: torch.Tensor | None,
    initial_base_pose: torch.Tensor,
    max_iter: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Entry point called by solve_ik when initial_base_pose is not None."""
    if ik_cfg.jacobian == "analytic":
        return _run_floating_base_lm_analytic(
            robot, targets, ik_cfg, initial_cfg, initial_base_pose, max_iter
        )

    # --- PyPose LM path (autodiff) ---
    initial_base = initial_base_pose.clone().float()
    initial_cfg_t = (
        initial_cfg.clone().float()
        if initial_cfg is not None
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
        initial_base=initial_base,
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
            module.cfg.data.clamp_(
                lo.to(device=module.cfg.device),
                hi.to(device=module.cfg.device),
            )
            raw = module.base.tensor().clone()
            raw[3:7] = raw[3:7] / raw[3:7].norm()
            module.base.data.copy_(raw)

    return module.base.tensor().detach(), module.cfg.detach()
```

- [ ] **Step 5: Run to verify passing**

```bash
uv run pytest tests/test_ik.py -v
```
Expected: all IK tests pass, including the two new floating-base analytic tests.

- [ ] **Step 6: Run full suite**

```bash
uv run pytest tests/ -v
```
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/better_robot/tasks/_floating_base_ik.py tests/test_ik.py
git commit -m "feat: floating-base analytic LM path with adjoint base Jacobian"
```

---

### Task 8: Benchmark Test

**Files:**
- Create: `tests/test_lm_benchmark.py`

- [ ] **Step 1: Create benchmark**

Create `tests/test_lm_benchmark.py`:

```python
"""Benchmark: PyPose LM vs our LM (autodiff) vs our LM (analytic).

Run with:
    uv run pytest tests/test_lm_benchmark.py -v -s

Correctness: all modes must reach pos_err < 5 cm.
Speed: printed per-call timings; analytic expected fastest.
"""

import time
import torch
import pytest
from robot_descriptions.loaders.yourdfpy import load_robot_description
import better_robot as br
from better_robot import solve_ik, IKConfig


@pytest.fixture(scope="module")
def panda():
    urdf = load_robot_description("panda_description")
    return br.Robot.from_urdf(urdf)


def _run_fixed_base(robot, target, link_idx, jacobian_mode, n_iter=20, n_runs=10):
    cfg0 = robot._default_cfg.clone()
    elapsed = 0.0
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = solve_ik(
            robot,
            targets={"panda_hand": target},
            cfg=IKConfig(jacobian=jacobian_mode),
            initial_cfg=cfg0.clone(),
            max_iter=n_iter,
        )
        elapsed += time.perf_counter() - t0
    fk = robot.forward_kinematics(result)
    pos_err = (fk[link_idx, :3] - target[:3]).norm().item()
    return elapsed / n_runs * 1000, pos_err


def test_benchmark_fixed_base(panda, capsys):
    """All three modes converge; analytic should not be slower than autodiff."""
    hand_idx = panda.get_link_index("panda_hand")
    target = panda.forward_kinematics(panda._default_cfg)[hand_idx].detach().clone()
    target[0] += 0.1  # offset to give solver something to do

    results = {}
    for mode in ("lm_pypose", "autodiff", "analytic"):
        if mode == "lm_pypose":
            # Run PyPose LM via solve_ik with autodiff (it's always autodiff internally)
            cfg0 = panda._default_cfg.clone()
            elapsed = 0.0
            N = 10
            for _ in range(N):
                import functools
                from better_robot.costs._pose import pose_residual
                from better_robot.costs._limits import limit_residual
                from better_robot.costs._regularization import rest_residual
                from better_robot.solvers._base import CostTerm, Problem
                from better_robot.solvers._levenberg_marquardt import LevenbergMarquardt as PyposeLM
                costs = [
                    CostTerm(functools.partial(pose_residual, robot=panda, target_link_index=hand_idx,
                                               target_pose=target, pos_weight=1.0, ori_weight=0.1), weight=1.0),
                    CostTerm(functools.partial(limit_residual, robot=panda), weight=0.1),
                    CostTerm(functools.partial(rest_residual, rest_pose=panda._default_cfg), weight=0.01),
                ]
                prob = Problem(variables=cfg0.clone(), costs=costs,
                               lower_bounds=panda.joints.lower_limits.clone(),
                               upper_bounds=panda.joints.upper_limits.clone())
                t0 = time.perf_counter()
                result = PyposeLM().solve(prob, max_iter=20)
                elapsed += time.perf_counter() - t0
            elapsed /= N
            fk = panda.forward_kinematics(result)
            pos_err = (fk[hand_idx, :3] - target[:3]).norm().item()
            results[mode] = (elapsed * 1000, pos_err)
        else:
            ms, pos_err = _run_fixed_base(panda, target, hand_idx, mode)
            results[mode] = (ms, pos_err)

    with capsys.disabled():
        print("\n--- Fixed-Base IK Benchmark (20 iters, 10 runs avg) ---")
        for mode, (ms, err) in results.items():
            print(f"  {mode:20s}: {ms:7.2f} ms/call   pos_err={err:.4f} m")

    # Correctness: all modes must converge
    for mode, (_, err) in results.items():
        assert err < 0.05, f"{mode} pos_err={err:.4f} > 0.05"

    # Speed: analytic must not be more than 20% slower than autodiff
    assert results["analytic"][0] <= results["autodiff"][0] * 1.2, (
        f"Analytic ({results['analytic'][0]:.1f} ms) is >20% slower than "
        f"autodiff ({results['autodiff'][0]:.1f} ms)"
    )


def test_benchmark_floating_base(panda, capsys):
    """Floating-base: analytic and PyPose modes converge."""
    hand_idx = panda.get_link_index("panda_hand")
    identity_base = torch.tensor([0., 0., 0., 0., 0., 0., 1.])
    target = panda.forward_kinematics(panda._default_cfg)[hand_idx].detach().clone()
    target[0] += 0.1

    results = {}
    for mode in ("autodiff", "analytic"):
        cfg0 = panda._default_cfg.clone()
        elapsed = 0.0
        N = 5
        for _ in range(N):
            t0 = time.perf_counter()
            base_r, cfg_r = solve_ik(
                panda, targets={"panda_hand": target},
                cfg=IKConfig(jacobian=mode),
                initial_base_pose=identity_base.clone(),
                initial_cfg=cfg0.clone(), max_iter=20,
            )
            elapsed += time.perf_counter() - t0
        fk = panda.forward_kinematics(cfg_r, base_pose=base_r)
        pos_err = (fk[hand_idx, :3] - target[:3]).norm().item()
        results[mode] = (elapsed / N * 1000, pos_err)

    with capsys.disabled():
        print("\n--- Floating-Base IK Benchmark (20 iters, 5 runs avg) ---")
        for mode, (ms, err) in results.items():
            print(f"  {mode:20s}: {ms:7.2f} ms/call   pos_err={err:.4f} m")

    for mode, (_, err) in results.items():
        assert err < 0.05, f"Floating {mode} pos_err={err:.4f} > 0.05"
```

- [ ] **Step 2: Run the benchmark**

```bash
uv run pytest tests/test_lm_benchmark.py -v -s
```
Expected: both tests pass and timing output is printed. Analytic should be faster than autodiff.

- [ ] **Step 3: Commit**

```bash
git add tests/test_lm_benchmark.py
git commit -m "test: LM benchmark — correctness + timing for all three modes"
```

---

### Task 9: CLAUDE.md Updates

**Files:**
- Modify: `src/better_robot/solvers/CLAUDE.md`
- Modify: `src/better_robot/costs/CLAUDE.md`
- Modify: `src/better_robot/core/CLAUDE.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update `solvers/CLAUDE.md`**

Replace the LM section and add the new content. The full updated `src/better_robot/solvers/CLAUDE.md`:

```markdown
# solvers/ — Optimization Backends

## Abstractions (`_base.py`)

```python
@dataclass
class CostTerm:
    residual_fn: Callable[[Tensor], Tensor]
    weight: float = 1.0
    kind: Literal["soft", "constraint_leq_zero"] = "soft"

@dataclass
class Problem:
    variables: Tensor
    costs: list[CostTerm]
    lower_bounds: Tensor | None
    upper_bounds: Tensor | None
    jacobian_fn: Callable[[Tensor], Tensor] | None = None
    # None → LM uses torch.func.jacrev
    # not None → LM calls jacobian_fn(x) directly → (m, n) Tensor
```

`Problem.total_residual(x)` concatenates all soft residuals.

## LM Solver — OUR IMPLEMENTATION (`_lm.py`) — DEFAULT

`SOLVER_REGISTRY["lm"]` → `LevenbergMarquardt` from `_lm.py`.

```python
LevenbergMarquardt(damping=1e-4, factor=2.0, reject=16).solve(problem, max_iter=100)
```

- `jacobian_fn=None` → uses `torch.func.jacrev(problem.total_residual)(x)` (autodiff)
- `jacobian_fn=fn` → calls `fn(x)` directly (analytic or custom)
- Adaptive damping: accept step if `||r_new|| <= ||r||`, else multiply λ × factor (up to `reject` retries)
- Bounds enforced via `.clamp()` after each accepted step

## LM Solver — PYPOSE (`_levenberg_marquardt.py`) — COMPARISON

`SOLVER_REGISTRY["lm_pypose"]` → PyPose-based LM. Always uses autograd for J. Ignores `jacobian_fn`.

Use this only for comparison/benchmarking. The new default is `_lm.py`.

## Registry

```python
SOLVER_REGISTRY = {
    "lm":        LevenbergMarquardt,           # our LM (default)
    "lm_pypose": PyposeLevenbergMarquardt,     # PyPose LM (comparison)
    "gn":        GaussNewton,                  # stub
    "adam":      AdamSolver,                   # stub
    "lbfgs":     LBFGSSolver,                  # stub
}
```
```

- [ ] **Step 2: Update `costs/CLAUDE.md`**

Add a new section to `src/better_robot/costs/CLAUDE.md` after the existing content:

```markdown
### `_jacobian.py` — Analytical Jacobian Functions

**`pose_jacobian(cfg, robot, target_link_index, target_pose, pos_weight, ori_weight, base_pose=None) → (6, n)`**

Geometric (analytical) Jacobian of `pose_residual` wrt `cfg`. Uses cross-product formula:
- Revolute joint j: `J[:,j] = [ω_j × (p_ee - p_j) * pos_weight ; ω_j * ori_weight]`
- Prismatic joint j: `J[:,j] = [d_j * pos_weight ; 0, 0, 0]`

`robot.get_chain(target_link_index)` determines which joints contribute.

**jlog approximation**: `jlog(T_err) ≈ I` (identity). Valid for errors < ~30°. For large errors the analytical J is an approximation; the solver still converges but may take more iterations.

**`limit_jacobian(cfg, robot) → (2n, n)`**
Diagonal: `-1` where `cfg < lo`, `+1` where `cfg > hi`, `0` inside limits. Matches `limit_residual`.

**`rest_jacobian(cfg, rest_pose) → (n, n)`**
Identity matrix. Matches `rest_residual = cfg - rest_pose`.
```

- [ ] **Step 3: Update `core/CLAUDE.md`**

Add to `src/better_robot/core/CLAUDE.md` under the Files table:

```markdown
## New Functions

### `Robot.get_chain(link_idx) -> list[int]`

Returns joint indices (actuated only) on the path from root to `link_idx`, in root→EE topological order. Uses `_fk_joint_child_link` / `_fk_joint_parent_link` for traversal. Returns `[]` for the root link.

Used by `pose_jacobian` to determine which joints affect a target EE.

### `adjoint_se3(T: Tensor) -> Tensor`  (`_lie_ops.py`)

6×6 Adjoint matrix for SE3 pose T = `[tx, ty, tz, qx, qy, qz, qw]`:

```
Ad(T) = [[R,          skew(p) @ R],
          [zeros(3,3), R          ]]
```

Convention: PyPose se3 tangent `[tx, ty, tz, rx, ry, rz]` (translation first).

Used for the floating-base Jacobian: `J_base = diag_w * adjoint_se3(se3_inverse(T_ee_local)) * pose_weight`.
The "smart trick": `T_ee_local = se3_compose(se3_inverse(base), fk_world[link_idx])` — no extra FK call needed.
```

- [ ] **Step 4: Update root `CLAUDE.md`**

In `CLAUDE.md`, find the "Solver Notes" section and replace it with:

```markdown
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
- `"analytic"`: geometric Jacobian from `costs/_jacobian.py` — faster; supported for both fixed and floating base

**Floating-base base Jacobian** (the "smart trick"):
```python
T_ee_local = se3_compose(se3_inverse(base), fk_world[link_idx])
J_base = diag([pos_w]*3 + [ori_w]*3) * adjoint_se3(se3_inverse(T_ee_local)) * pose_weight
```
No extra FK call; `adjoint_se3` is a 15-line 6×6 matrix computation.
```

- [ ] **Step 5: Run full test suite to confirm all docs are consistent**

```bash
uv run pytest tests/ -v
```
Expected: all tests pass (docs don't affect tests).

- [ ] **Step 6: Commit**

```bash
git add src/better_robot/solvers/CLAUDE.md src/better_robot/costs/CLAUDE.md \
        src/better_robot/core/CLAUDE.md CLAUDE.md
git commit -m "docs: update CLAUDE.md files for analytical Jacobian and custom LM"
```

---

## Self-Review

**Spec coverage:**
- ✅ `Robot.get_chain` → Task 1
- ✅ `adjoint_se3` → Task 2
- ✅ `pose_jacobian`, `limit_jacobian`, `rest_jacobian` → Task 3
- ✅ `Problem.jacobian_fn` → Task 4
- ✅ Our LM (`_lm.py`) + registry → Task 5
- ✅ `IKConfig.jacobian` → Task 6
- ✅ Fixed-base analytic dispatch → Task 6
- ✅ Floating-base analytic path + base Jacobian trick → Task 7
- ✅ Benchmark (correctness + timing, fixed + floating) → Task 8
- ✅ CLAUDE.md updates (all 4 files) → Task 9
- ✅ PyPose LM kept unchanged, registered as `"lm_pypose"`

**Placeholder scan:** No TBDs found. All code blocks are complete.

**Type consistency:**
- `pose_jacobian` signature used consistently across Tasks 3, 6, 7
- `adjoint_se3` imported from `_lie_ops` in Tasks 2 and 7
- `jacobian_fn: Callable[[Tensor], Tensor] | None` consistent across Tasks 4, 5, 6
- `IKConfig.jacobian: Literal["autodiff", "analytic"]` consistent across Tasks 6 and 7
