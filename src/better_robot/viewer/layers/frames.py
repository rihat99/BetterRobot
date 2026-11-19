"""``FramesLayer``: coordinate axes for many frames, one batched node."""

from __future__ import annotations

from typing import Any, final, override

import numpy as np

from ...exceptions import ShapeError
from ..layer import Bounds, Layer, as_frames, box


@final
class FramesLayer(Layer):
    """Poses ``(T, F, 7)`` as ``[x, y, z, qx, qy, qz, qw]`` (BetterRobot order) drawn as axes, ``axes_length`` long and
    ``axes_radius`` thick, each frame's times its ``scales`` entry ``(F,)`` (``None``: 1 for all)."""

    def __init__(
        self,
        name: str,
        poses: np.ndarray,
        *,
        axes_length: float = 0.1,
        axes_radius: float = 0.004,
        scales: np.ndarray | None = None,
    ) -> None:
        self.poses = as_frames(poses, 2)
        if self.poses.shape[-1] != 7:
            raise ShapeError(f"poses must be (F, 7) or (T, F, 7), got {self.poses.shape}")
        super().__init__(name, self.poses.shape[0])
        frames = self.poses.shape[1]
        self.scales = np.ones(frames, np.float32) if scales is None else np.ascontiguousarray(scales, dtype=np.float32)
        if self.scales.shape != (frames,):
            raise ShapeError(f"scales must be (F,) = ({frames},), one per frame, got {self.scales.shape}")
        self._scales = self.scales.copy()  # the scales the node holds: a push writes them only when they change
        self.axes_length = axes_length
        self.axes_radius = axes_radius
        self._axes: Any = None

    @override
    def build(self, scene: Any, root: str) -> None:
        # viser's batched-axes node crashes the client when it holds zero instances; an empty layer draws nothing.
        if self.poses.shape[1] == 0:
            self._axes = None
            return
        self._axes = self._track(
            scene.add_batched_axes(
                f"{root}/axes",
                batched_wxyzs=self.poses[0][:, [6, 3, 4, 5]],  # viser's scalar-first quaternions
                batched_positions=self.poses[0, :, :3],
                batched_scales=self.scales,
                axes_length=self.axes_length,
                axes_radius=self.axes_radius,
            )
        )

    @override
    def bounds(self, step: int) -> Bounds | None:
        return box(self.poses[step, :, :3])

    @override
    def set_step(self, step: int) -> None:
        if self._axes is None:
            return
        self._axes.batched_positions = self.poses[step, :, :3]
        self._axes.batched_wxyzs = self.poses[step][:, [6, 3, 4, 5]]
        if not np.array_equal(self.scales, self._scales):  # an edit of scales
            self._axes.batched_scales = self.scales
            self._scales = self.scales.copy()
