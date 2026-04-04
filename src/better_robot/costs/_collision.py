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
    world_geom: list[CollGeom],
    margin: float = 0.0,
    weight: float = 1.0,
) -> torch.Tensor:
    """Compute world-collision violation residual.

    Positive values indicate collision (for constraint_leq_zero).

    Args:
        cfg: Shape (num_actuated_joints,). Current config.
        robot: Robot instance.
        robot_coll: RobotCollision with sphere decomposition.
        world_geom: List of world collision geometry objects to check against.
        margin: Safety margin in meters.
        weight: Scalar weight.

    Returns:
        Shape (num_robot_spheres * len(world_geom),). Violation per sphere-geometry pair.
    """
    raise NotImplementedError
