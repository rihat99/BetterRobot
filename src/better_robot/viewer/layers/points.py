"""``PointsLayer``: a point cloud, static or per step."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._math import RGB, as_frames, instance_mesh, rgb_uint8
from ..layer import Geometry, Layer

_OCTAHEDRON = (
    np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float32) * 0.5,
    np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]]),
)


class PointsLayer(Layer):
    """Points ``(T, N, 3)`` or static ``(N, 3)`` with one RGB, ``(N, 3)`` or ``(T, N, 3)`` colours.

    N is fixed per layer; pad with duplicates of a visible point to vary the count.
    """

    def __init__(
        self,
        name: str,
        points: np.ndarray,
        *,
        colors: np.ndarray | RGB = (0.25, 0.25, 0.25),
        point_size: float = 0.01,
        point_shape: str = "circle",
    ) -> None:
        self.points = as_frames(points, 2)
        super().__init__(name, self.points.shape[0])
        colors = np.asarray(colors, dtype=np.float32)
        self._colors = rgb_uint8(colors, (self.points.shape[1],)) if colors.ndim <= 2 else rgb_uint8(colors)
        self.point_size = point_size
        self.point_shape = point_shape
        self._cloud: Any = None

    def build(self, scene: Any, root: str) -> None:
        colors = self._colors if self._colors.ndim == 2 else self._colors[0]
        self._cloud = self._track(
            scene.add_point_cloud(
                f"{root}/points",
                self.points[0],
                colors=colors,
                point_size=self.point_size,
                point_shape=self.point_shape,
                precision="float32",
            )
        )

    def geometry(self, step: int) -> list[Geometry]:
        n = self.points.shape[1]
        wxyz = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1))
        vertices, faces = instance_mesh(*_OCTAHEDRON, self.points[step], wxyz, self.point_size)
        colors = (self._colors if self._colors.ndim == 2 else self._colors[step]).astype(np.float32) / 255.0
        return [Geometry(vertices, faces, np.repeat(colors, 8, axis=0))]

    def set_step(self, step: int) -> None:
        self._cloud.points = self.points[step]
        if self._colors.ndim == 3:
            self._cloud.colors = self._colors[step]
