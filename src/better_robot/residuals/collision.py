"""Self / world collision residuals.

Consumes a ``RobotCollision`` decomposition and produces a variable-length
residual: one entry per *active* pair (pair within ``margin``).

See ``docs/design/09_COLLISION_GEOMETRY.md §6``.
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

    See docs/design/09_COLLISION_GEOMETRY.md §6.
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
        # ``dim`` is the *number of candidate pairs*, not the live count —
        # keeping it stable across LM iterations is what the spec requires
        # (docs/design/07_RESIDUALS_COSTS_SOLVERS.md §10).
        self.dim = int(robot_collision.self_pairs.shape[0])

    def __call__(self, state: ResidualState) -> torch.Tensor:
        raise NotImplementedError("see docs/design/09_COLLISION_GEOMETRY.md §6")

    def jacobian(self, state: ResidualState) -> torch.Tensor | None:
        """Sparse analytic Jacobian — rewrite of ``_analytic_collision_jacobian``.

        See docs/design/09_COLLISION_GEOMETRY.md §6.
        """
        raise NotImplementedError("see docs/design/09_COLLISION_GEOMETRY.md §6")

    @property
    def spec(self):
        from ..optim.jacobian_spec import ResidualSpec

        return ResidualSpec(
            dim=self.dim,
            structure="block",
            dynamic_dim=True,
            affected_joints=tuple(int(j) for j in self.robot_collision.self_pairs.flatten().unique().tolist()),
        )


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
        # Stable across iterations: one residual per (link, world-shape) pair.
        n_links = int(robot_collision.link_indices.shape[0]) if hasattr(robot_collision, "link_indices") else 0
        self.dim = n_links * len(self.world)

    def __call__(self, state: ResidualState) -> torch.Tensor:
        raise NotImplementedError("see docs/design/09_COLLISION_GEOMETRY.md §8")

    def jacobian(self, state: ResidualState) -> torch.Tensor | None:
        raise NotImplementedError("see docs/design/09_COLLISION_GEOMETRY.md §8")
