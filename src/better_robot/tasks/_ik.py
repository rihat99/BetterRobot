"""Inverse kinematics task."""

from __future__ import annotations

import functools
from typing import Literal

import torch

from ..core._robot import Robot
from ..collision._robot_collision import RobotCollision
from ..collision._geometry import CollGeom
from ..costs._pose import pose_residual
from ..costs._limits import limit_residual
from ..costs._regularization import rest_residual
from ..solvers._base import CostTerm, Problem
from ..solvers import SOLVER_REGISTRY


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
    w = {"pose": 1.0, "limits": 0.1, "rest": 0.01}
    if weights:
        w.update(weights)

    target_link_index = robot.get_link_index(target_link)
    initial = initial_cfg.clone() if initial_cfg is not None else robot._default_cfg.clone()
    rest = robot._default_cfg.clone()

    costs = [
        CostTerm(
            residual_fn=functools.partial(
                pose_residual,
                robot=robot,
                target_link_index=target_link_index,
                target_pose=target_pose,
                pos_weight=1.0,
                ori_weight=0.1,
            ),
            weight=w["pose"],
        ),
        CostTerm(
            residual_fn=functools.partial(limit_residual, robot=robot),
            weight=w["limits"],
        ),
        CostTerm(
            residual_fn=functools.partial(rest_residual, rest_pose=rest),
            weight=w["rest"],
        ),
    ]

    problem = Problem(
        variables=initial,
        costs=costs,
        lower_bounds=robot.joints.lower_limits.clone(),
        upper_bounds=robot.joints.upper_limits.clone(),
    )
    solver_cls = SOLVER_REGISTRY[solver]
    return solver_cls().solve(problem, max_iter=max_iter)
