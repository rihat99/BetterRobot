"""``render_modes/collision.py`` — collision capsule render mode (stub).

See ``docs/design/12_VIEWER.md §4.4``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import RenderContext

if TYPE_CHECKING:
    from ...data_model.data import Data
    from ...data_model.model import Model


class CollisionMode:
    """Render the capsule decomposition from a ``RobotCollision``.

    Uses ``RobotCollision.world_capsules(data)`` to pull the active capsules
    into the world frame and draws each as a capsule primitive.
    """

    name = "Collision"
    description = "Capsule-based collision decomposition"

    def __init__(self, robot_collision: object, *, alpha: float = 0.35) -> None:
        self._robot_collision = robot_collision
        self.alpha = alpha

    @classmethod
    def is_available(cls, model: "Model", data: "Data", robot_collision: object = None) -> bool:
        return robot_collision is not None

    def attach(self, context: RenderContext, model: "Model", data: "Data") -> None:
        raise NotImplementedError("see docs/design/12_VIEWER.md §4.4")

    def update(self, data: "Data") -> None:
        raise NotImplementedError("see docs/design/12_VIEWER.md §4.4")

    def set_visible(self, visible: bool) -> None:
        raise NotImplementedError("see docs/design/12_VIEWER.md §4.4")

    def detach(self) -> None:
        raise NotImplementedError("see docs/design/12_VIEWER.md §4.4")
