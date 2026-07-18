"""Self / world collision residuals.

Consumes a ``RobotCollision`` decomposition and produces a variable-length
residual: one entry per *active* pair (pair within ``margin``).

See ``docs/concepts/collision_and_geometry.md §6``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Sequence
from typing import Any

import torch

from ..collision.geometry import Box, Capsule, Sphere
from ..collision.robot_collision import RobotCollision
from ..data_model.model import Model
from .base import Residual


class SelfCollisionResidual(Residual):
    """One residual value per self-pair currently within ``margin``.

    Residual = ``-colldist_from_sdf(d_p, margin) * weight``: zero outside
    the margin, quadratic inside it, linear on penetration.

    See docs/concepts/collision_and_geometry.md §6.
    """

    name: str = "self_collision"
    reads = ("q", "data")

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
        # keeping it stable across LM iterations is part of the residual's
        # public shape contract (docs/concepts/residuals_and_costs.md §10).
        self.dim = int(robot_collision.self_pairs.shape[0])

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        del ctx
        raise NotImplementedError("see docs/concepts/collision_and_geometry.md §6")


class WorldCollisionResidual(Residual):
    """Collision residual against an external geometry set (obstacles, ground)."""

    name: str = "world_collision"
    reads = ("q", "data")

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

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        del ctx
        raise NotImplementedError("see docs/concepts/collision_and_geometry.md §8")
