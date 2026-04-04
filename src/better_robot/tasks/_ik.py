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
