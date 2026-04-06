"""Self-collision and world-collision residuals."""
from __future__ import annotations
import functools
import torch
from ..models.robot_model import RobotModel
from ..algorithms.geometry.robot_collision import RobotCollision
from ..algorithms.geometry.primitives import CollGeom
from ..algorithms.geometry.distance import colldist_from_sdf
from .cost_term import CostTerm


def self_collision_residual(
    cfg: torch.Tensor,
    robot: RobotModel,
    robot_coll: RobotCollision,
    margin: float = 0.02,
    weight: float = 1.0,
) -> torch.Tensor:
    """Compute self-collision violation residual.

    Returns:
        Shape (num_active_pairs,). Violation per pair (higher = worse).
    """
    dists = robot_coll.compute_self_collision_distance(robot, cfg)
    # Convert to cost: negative colldist_from_sdf means collision
    cost = -colldist_from_sdf(dists, activation_dist=margin)
    return cost * weight


def world_collision_residual(
    cfg: torch.Tensor,
    robot: RobotModel,
    robot_coll: RobotCollision,
    world_geom: list[CollGeom],
    margin: float = 0.02,
    weight: float = 1.0,
) -> torch.Tensor:
    """Compute world-collision violation residual.

    Returns:
        Shape (num_robot_spheres * len(world_geom),). Violation per pair.
    """
    dists = robot_coll.compute_world_collision_distance(robot, cfg, world_geom)
    cost = -colldist_from_sdf(dists, activation_dist=margin)
    return cost * weight


def self_collision_cost(
    robot: RobotModel,
    robot_coll: RobotCollision,
    margin: float = 0.02,
    weight: float = 1.0,
) -> CostTerm:
    """Factory: CostTerm for self-collision avoidance."""
    return CostTerm(
        residual_fn=functools.partial(
            self_collision_residual,
            robot=robot,
            robot_coll=robot_coll,
            margin=margin,
            weight=weight,
        ),
        weight=1.0,
    )


def world_collision_cost(
    robot: RobotModel,
    robot_coll: RobotCollision,
    world_geom: list[CollGeom],
    margin: float = 0.02,
    weight: float = 1.0,
) -> CostTerm:
    """Factory: CostTerm for world-collision avoidance."""
    return CostTerm(
        residual_fn=functools.partial(
            world_collision_residual,
            robot=robot,
            robot_coll=robot_coll,
            world_geom=world_geom,
            margin=margin,
            weight=weight,
        ),
        weight=1.0,
    )
