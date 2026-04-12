"""Self / world collision residuals.

Consumes a ``RobotCollision`` decomposition and produces a variable-length
residual: one entry per *active* pair (pair within ``margin``).

See ``docs/09_COLLISION_GEOMETRY.md §6``.
"""

from __future__ import annotations

from typing import Sequence

import torch

from ..collision.geometry import Box, Capsule, Sphere
from ..collision.robot_collision import RobotCollision
from ..data_model.model import Model
from .base import Residual, ResidualState
from .registry import register_residual


@register_residual("self_collision")
class SelfCollisionResidual(Residual):
    """One residual value per self-pair currently within ``margin``.

    Residual = ``-colldist_from_sdf(d_p, margin) * weight``: zero outside
    the margin, quadratic inside it, linear on penetration.

    See docs/09_COLLISION_GEOMETRY.md §6.
    """

    def __init__(
        self,
        model: Model,
        robot_collision: RobotCollision,
        *,
        margin: float = 0.02,
        weight: float = 1.0,
    ) -> None:
        self.model = model
        self.robot_collision = robot_collision
        self.margin = margin
        self.weight = weight
        self.dim = int(robot_collision.self_pairs.shape[0])

    def __call__(self, state: ResidualState) -> torch.Tensor:
        raise NotImplementedError("see docs/09_COLLISION_GEOMETRY.md §6")

    def jacobian(self, state: ResidualState) -> torch.Tensor | None:
        """Sparse analytic Jacobian — rewrite of ``_analytic_collision_jacobian``.

        See docs/09_COLLISION_GEOMETRY.md §6.
        """
        raise NotImplementedError("see docs/09_COLLISION_GEOMETRY.md §6")


@register_residual("world_collision")
class WorldCollisionResidual(Residual):
    """Collision residual against an external geometry set (obstacles, ground)."""

    def __init__(
        self,
        model: Model,
        robot_collision: RobotCollision,
        world: Sequence[Sphere | Capsule | Box],
        *,
        margin: float = 0.02,
        weight: float = 1.0,
    ) -> None:
        self.model = model
        self.robot_collision = robot_collision
        self.world = tuple(world)
        self.margin = margin
        self.weight = weight
        self.dim = 0  # variable, recomputed per call

    def __call__(self, state: ResidualState) -> torch.Tensor:
        raise NotImplementedError("see docs/09_COLLISION_GEOMETRY.md §8")

    def jacobian(self, state: ResidualState) -> torch.Tensor | None:
        raise NotImplementedError("see docs/09_COLLISION_GEOMETRY.md §8")
