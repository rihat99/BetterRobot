"""Self-collision and world-collision residuals."""
from __future__ import annotations

import torch

from ..models.robot_model import RobotModel
from ..algorithms.geometry.robot_collision import RobotCollision
from ..algorithms.geometry.primitives import CollGeom
from .cost_term import CostTerm


def self_collision_residual(
    cfg: torch.Tensor,
    robot: RobotModel,
    robot_coll: RobotCollision,
    margin: float = 0.0,
    weight: float = 1.0,
) -> torch.Tensor:
    """Compute self-collision violation residual.

    Positive values indicate collision (for constraint_leq_zero).

    Returns:
        Shape (num_collision_pairs,). Violation per pair.
    """
    raise NotImplementedError


def world_collision_residual(
    cfg: torch.Tensor,
    robot: RobotModel,
    robot_coll: RobotCollision,
    world_geom: list[CollGeom],
    margin: float = 0.0,
    weight: float = 1.0,
) -> torch.Tensor:
    """Compute world-collision violation residual.

    Returns:
        Shape (num_robot_spheres * len(world_geom),). Violation per pair.
    """
    raise NotImplementedError
