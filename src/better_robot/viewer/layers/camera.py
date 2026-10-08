"""``CameraLayer``: camera frustums with optional per-step images."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .._math import RGB, as_frames, rgb_uint8, xyzw_to_wxyz
from ..layer import Layer


class CameraLayer(Layer):
    """Camera frustums at ``poses (T, 7)`` (camera-to-world, OpenCV ``+z`` forward), optional images.

    ``image`` is ``(H, W, 3)`` uint8, ``(T, H, W, 3)``, or a callable ``step -> image`` for lazy
    loading; the image is shown inside the frustum.
    """

    def __init__(
        self,
        name: str,
        poses: np.ndarray,
        *,
        fov: float,
        aspect: float,
        scale: float = 0.3,
        color: RGB = (0.15, 0.15, 0.15),
        image: np.ndarray | Callable[[int], np.ndarray] | None = None,
    ) -> None:
        self.poses = as_frames(poses, 1)
        super().__init__(name, self.poses.shape[0])
        self.fov, self.aspect, self.scale, self.color = fov, aspect, scale, color
        self._image = image
        self._frustum: Any = None

    def image_at(self, step: int) -> np.ndarray | None:
        """Image shown in the frustum at ``step``: ``(H, W, 3)``, or ``None`` when the layer has no image."""
        if self._image is None:
            return None
        if callable(self._image):
            return self._image(step)
        return self._image if self._image.ndim == 3 else self._image[step]

    def build(self, scene: Any, root: str) -> None:
        self._frustum = self._track(
            scene.add_camera_frustum(
                f"{root}/frustum",
                fov=self.fov,
                aspect=self.aspect,
                scale=self.scale,
                color=rgb_uint8(self.color),
                image=self.image_at(0),
                wxyz=xyzw_to_wxyz(self.poses[0, 3:]),
                position=self.poses[0, :3],
            )
        )

    def set_step(self, step: int) -> None:
        self._frustum.position = self.poses[step, :3]
        self._frustum.wxyz = xyzw_to_wxyz(self.poses[step, 3:])
        if self._image is not None and (callable(self._image) or self._image.ndim == 4):
            self._frustum.image = self.image_at(step)
