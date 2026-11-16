"""``PointsLayer``: a point cloud, static or per step, of a fixed or a changing size."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, final, override

import numpy as np

from ...exceptions import ShapeError
from ..layer import Bounds, Layer, as_frames, box
from ..theme import RGB, rgb_uint8


@final
class PointsLayer(Layer):
    """Points ``(T, N, 3)`` or static ``(N, 3)`` with one RGB, ``(N, 3)`` or ``(T, N, 3)`` colours.

    A list of ``(N_t, 3)`` arrays, one per step, is a cloud whose size changes (ragged), with one RGB or a list of
    ``(N_t, 3)`` colours. It is padded to the largest step with copies of the step's first point, which draw nothing
    new, and a step without points hides the cloud.
    """

    def __init__(
        self,
        name: str,
        points: np.ndarray | Sequence[np.ndarray],
        *,
        colors: np.ndarray | RGB | Sequence[np.ndarray] = (0.25, 0.25, 0.25),
        point_size: float = 0.01,
        point_shape: str = "circle",
    ) -> None:
        #: ``(T,)`` the number of points of each step of a ragged cloud; ``None`` for an array.
        self.counts: np.ndarray | None = None
        if isinstance(points, (list, tuple)) and all(np.ndim(step) == 2 for step in points):
            points, colors, self.counts = _pad(points, colors)
        self.points = as_frames(points, 2, name="points", width=3)
        super().__init__(name, self.points.shape[0])
        colors = np.asarray(colors, dtype=np.float32)
        self._colors = rgb_uint8(colors, (self.points.shape[1],)) if colors.ndim <= 2 else rgb_uint8(colors)
        self.point_size = point_size
        self.point_shape = point_shape
        self._cloud: Any = None

    @override
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

    @override
    def bounds(self, step: int) -> Bounds | None:
        return box(self.points[step] if self.counts is None else self.points[step, : self.counts[step]])

    @override
    def set_step(self, step: int) -> None:
        self._cloud.points = self.points[step]
        if self._colors.ndim == 3:
            self._cloud.colors = self._colors[step]
        if self.counts is not None:
            self._cloud.visible = self.visible and bool(self.counts[step] > 0)


def _pad(
    steps: Sequence[np.ndarray], colors: np.ndarray | RGB | Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A ragged cloud as ``points (T, N, 3)``, colours (one RGB, or ``(T, N, 3)``) and ``counts (T,)``: each step is
    padded to the largest ``N`` by repeating its first point."""
    counts = np.array([len(step) for step in steps])
    ragged_colors = isinstance(colors, (list, tuple)) and np.ndim(colors[0]) == 2
    if ragged_colors and [len(step) for step in colors] != [len(step) for step in steps]:
        raise ShapeError("ragged colors must hold one (N_t, 3) array per step, as many rows as that step's points")
    columns = np.arange(max(1, int(counts.max(initial=0))))
    rows = np.where(columns < counts[:, None], columns, 0)  # (T, N): the row of the step each padded point copies

    def padded(arrays: Sequence[np.ndarray]) -> np.ndarray:
        out = np.zeros((len(arrays), len(columns), 3), dtype=np.float32)
        for step, array in enumerate(arrays):
            if len(array):
                out[step] = np.asarray(array, dtype=np.float32).reshape(-1, 3)[rows[step]]
        return out

    return padded(steps), padded(colors) if ragged_colors else np.asarray(colors, np.float32), counts
