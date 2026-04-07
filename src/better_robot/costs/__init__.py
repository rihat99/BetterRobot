"""Costs layer: differentiable cost and residual functions.

Each module provides both:
- raw residual functions: ``(x: Tensor, ...) -> Tensor``
- factory functions returning ``CostTerm`` objects ready for use in ``Problem``

Residuals are pure PyTorch functions with no solver dependencies.
"""
from .cost_term import CostTerm
from .pose import pose_residual, pose_cost
from .limits import (
    limit_residual,
    limit_cost,
    velocity_residual,
    acceleration_residual,
    jerk_residual,
)
from .regularization import rest_residual, rest_cost, smoothness_residual
from .collision import (
    self_collision_residual, world_collision_residual,
    self_collision_cost, world_collision_cost,
)
from .manipulability import manipulability_residual, manipulability_cost

__all__ = [
    "CostTerm",
    "pose_residual", "pose_cost",
    "limit_residual", "limit_cost",
    "velocity_residual", "acceleration_residual", "jerk_residual",
    "rest_residual", "rest_cost", "smoothness_residual",
    "self_collision_residual", "world_collision_residual",
    "self_collision_cost", "world_collision_cost",
    "manipulability_residual", "manipulability_cost",
]
