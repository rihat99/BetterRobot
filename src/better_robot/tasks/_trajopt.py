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
