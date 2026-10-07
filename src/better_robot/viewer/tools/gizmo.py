"""``PoseGizmo``: a draggable SE(3) handle that reports poses in BetterRobot order."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .._math import xyzw_to_wxyz


class PoseGizmo:
    """Transform controls at ``pose`` ``[x, y, z, qx, qy, qz, qw]``; ``on_change(pose)`` fires on drags.

    Tools are not layers: they hold the viewer and callbacks and sit outside the frame loop.
    ``scale`` is the handle size in metres (viser's default of 1 dwarfs a robot). Extra
    ``options`` go to viser's ``add_transform_controls`` (``disable_sliders=True`` and
    ``disable_axes=True`` make a rotation-only handle).
    """

    def __init__(
        self,
        viewer: Any,
        name: str,
        pose: np.ndarray,
        *,
        on_change: Callable[[np.ndarray], None] | None = None,
        scale: float = 0.15,
        **options: Any,
    ) -> None:
        pose = np.asarray(pose, dtype=np.float32)
        self.handle = viewer.server.scene.add_transform_controls(
            f"/tools/{name}", scale=scale, position=pose[:3], wxyz=xyzw_to_wxyz(pose[3:]), **options
        )
        self._callbacks: list[Callable[[np.ndarray], None]] = [on_change] if on_change else []
        self.handle.on_update(lambda _: self._notify())

    @property
    def pose(self) -> np.ndarray:
        """Current pose ``(7,)`` in ``[x, y, z, qx, qy, qz, qw]`` order."""
        wxyz = np.asarray(self.handle.wxyz, dtype=np.float32)
        return np.concatenate([np.asarray(self.handle.position, dtype=np.float32), wxyz[[1, 2, 3, 0]]])

    @pose.setter
    def pose(self, pose: np.ndarray) -> None:
        pose = np.asarray(pose, dtype=np.float32)
        self.handle.position = pose[:3]
        self.handle.wxyz = xyzw_to_wxyz(pose[3:])

    def on_change(self, callback: Callable[[np.ndarray], None]) -> None:
        self._callbacks.append(callback)

    def remove(self) -> None:
        self.handle.remove()

    def _notify(self) -> None:
        pose = self.pose
        for callback in self._callbacks:
            callback(pose)
