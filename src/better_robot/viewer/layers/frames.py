"""``FramesLayer``: coordinate axes for many frames, one batched node."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._math import as_frames, xyzw_to_wxyz
from ..layer import Layer


class FramesLayer(Layer):
    """Poses ``(T, F, 7)`` as ``[x, y, z, qx, qy, qz, qw]`` (BetterRobot order) drawn as axes."""

    def __init__(self, name: str, poses: np.ndarray, *, axes_length: float = 0.1, axes_radius: float = 0.004) -> None:
        self.poses = as_frames(poses, 2)
        if self.poses.shape[-1] != 7:
            raise ValueError("poses must be (T, F, 7)")
        super().__init__(name, self.poses.shape[0])
        self.axes_length = axes_length
        self.axes_radius = axes_radius
        self._axes: Any = None

    def build(self, scene: Any, root: str) -> None:
        self._axes = self._track(
            scene.add_batched_axes(
                f"{root}/axes",
                batched_wxyzs=xyzw_to_wxyz(self.poses[0, :, 3:]),
                batched_positions=self.poses[0, :, :3],
                axes_length=self.axes_length,
                axes_radius=self.axes_radius,
            )
        )

    def world_points(self, step: int) -> np.ndarray | None:
        """World positions ``(N, 3)`` drawn at ``step``, for camera fitting."""
        return self.poses[step, :, :3]

    def set_step(self, step: int) -> None:
        self._axes.batched_positions = self.poses[step, :, :3]
        self._axes.batched_wxyzs = xyzw_to_wxyz(self.poses[step, :, 3:])
