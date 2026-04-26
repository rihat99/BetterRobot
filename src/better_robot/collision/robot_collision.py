"""``RobotCollision`` — frame-indexed capsule decomposition of a robot.

Each collision capsule is attached to a *frame* (not a joint) via a local
``(a, b, radius)`` triple. At query time, ``RobotCollision`` uses
``data.frame_pose_world`` to transform the capsules into the world frame.

See ``docs/concepts/collision_and_geometry.md §5``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch

from ..data_model.data import Data
from ..data_model.model import Model
from .geometry import Box, Capsule, Sphere


@dataclass
class RobotCollision:
    """Capsule/sphere decomposition of a robot, attached to frames.

    See docs/concepts/collision_and_geometry.md §5.
    """

    frame_ids: tuple[int, ...]
    local_a: torch.Tensor  # (n_caps, 3)
    local_b: torch.Tensor  # (n_caps, 3)
    radii: torch.Tensor  # (n_caps,)
    self_pairs: torch.Tensor  # (n_pairs, 2)
    allowed_pairs_mask: torch.Tensor  # (n_caps, n_caps) bool

    @classmethod
    def from_model(
        cls,
        model: Model,
        *,
        mode: Literal["capsule", "sphere"] = "capsule",
        allow_adjacent: bool = False,
    ) -> "RobotCollision":
        """Build a capsule (or sphere) decomposition from a ``Model``.

        See docs/concepts/collision_and_geometry.md §5.
        """
        raise NotImplementedError("see docs/concepts/collision_and_geometry.md §5")

    def world_capsules(self, data: Data) -> Capsule:
        """Return a ``(B..., n_caps)`` ``Capsule`` with world-frame endpoints.

        See docs/concepts/collision_and_geometry.md §5.
        """
        raise NotImplementedError("see docs/concepts/collision_and_geometry.md §5")

    def self_distances(self, data: Data) -> torch.Tensor:
        """``(B..., n_pairs)`` signed distances for every self-pair.

        See docs/concepts/collision_and_geometry.md §5.
        """
        raise NotImplementedError("see docs/concepts/collision_and_geometry.md §5")

    def world_distances(
        self,
        data: Data,
        world: Sequence[Sphere | Capsule | Box],
    ) -> torch.Tensor:
        """``(B..., n_caps, len(world))`` signed distances against external geometry.

        See docs/concepts/collision_and_geometry.md §5.
        """
        raise NotImplementedError("see docs/concepts/collision_and_geometry.md §5")
