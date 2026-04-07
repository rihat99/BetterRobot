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
    q: torch.Tensor,
    model: RobotModel,
    robot_coll: RobotCollision,
    margin: float = 0.02,
    weight: float = 1.0,
    base_pose: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute self-collision violation residual.

    Args:
        q: Joint configuration ``(num_actuated,)``.
        model: RobotModel.
        robot_coll: Collision model (capsule or sphere mode).
        margin: Activation distance in metres.
        weight: Scalar weight applied to the residual.
        base_pose: Optional base pose ``[tx,ty,tz,qx,qy,qz,qw]`` for
            floating-base robots.

    Returns:
        Shape (num_active_pairs,). Violation per pair (higher = worse).
    """
    dists = robot_coll.compute_self_collision_distance(model, q, base_pose=base_pose)
    cost = -colldist_from_sdf(dists, activation_dist=margin)
    return cost * weight


def world_collision_residual(
    q: torch.Tensor,
    model: RobotModel,
    robot_coll: RobotCollision,
    world_geom: list[CollGeom],
    margin: float = 0.02,
    weight: float = 1.0,
    base_pose: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute world-collision violation residual.

    Args:
        q: Joint configuration.
        model: RobotModel.
        robot_coll: Collision model.
        world_geom: List of world collision primitives.
        margin: Activation distance in metres.
        weight: Scalar weight.
        base_pose: Optional base pose for floating-base robots.

    Returns:
        Shape (num_robot_geoms * len(world_geom),). Violation per pair.
    """
    dists = robot_coll.compute_world_collision_distance(model, q, world_geom, base_pose=base_pose)
    cost = -colldist_from_sdf(dists, activation_dist=margin)
    return cost * weight


def self_collision_cost(
    model: RobotModel,
    robot_coll: RobotCollision,
    margin: float = 0.02,
    weight: float = 1.0,
) -> CostTerm:
    """Factory: CostTerm for self-collision avoidance."""
    return CostTerm(
        residual_fn=functools.partial(
            self_collision_residual,
            model=model,
            robot_coll=robot_coll,
            margin=margin,
            weight=weight,
        ),
        weight=1.0,
    )


def world_collision_cost(
    model: RobotModel,
    robot_coll: RobotCollision,
    world_geom: list[CollGeom],
    margin: float = 0.02,
    weight: float = 1.0,
) -> CostTerm:
    """Factory: CostTerm for world-collision avoidance."""
    return CostTerm(
        residual_fn=functools.partial(
            world_collision_residual,
            model=model,
            robot_coll=robot_coll,
            world_geom=world_geom,
            margin=margin,
            weight=weight,
        ),
        weight=1.0,
    )
