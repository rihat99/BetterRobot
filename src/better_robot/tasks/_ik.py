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


# Keep solve_ik_multi as a compatibility shim (Task 5 will remove the export)
def solve_ik_multi(
    robot: Robot,
    targets: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
    max_iter: int = 100,
    initial_cfg: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compatibility shim — delegates to solve_ik (fixed-base)."""
    ik_cfg = IKConfig()
    if weights:
        if "pose" in weights:
            ik_cfg.pose_weight = weights["pose"]
        if "limits" in weights:
            ik_cfg.limit_weight = weights["limits"]
        if "rest" in weights:
            ik_cfg.rest_weight = weights["rest"]
    return solve_ik(robot, targets=targets, cfg=ik_cfg, initial_cfg=initial_cfg, max_iter=max_iter)
