"""``GroundLayer``: a static reference grid."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._math import RGB, rgb_uint8
from ..layer import Geometry, Layer


class GroundLayer(Layer):
    """Grid in the ``plane`` at ``height`` along its normal.

    ``plane=None`` (the default) follows the viewer's theme: ``"xy"`` for a z-up theme, ``"xz"`` for y-up,
    ``"yz"`` for x-up. Pass it explicitly to override, or when the layer is rendered without a ``Viewer``.
    """

    fits_camera = False

    def __init__(
        self,
        name: str = "ground",
        *,
        size: float = 10.0,
        height: float = 0.0,
        plane: str | None = None,
        cell_color: RGB = (0.78, 0.78, 0.78),
        section_color: RGB = (0.55, 0.55, 0.55),
    ) -> None:
        super().__init__(name, 1)
        self.size = size
        self.height = height
        self.plane = plane
        self.cell_color = cell_color
        self.section_color = section_color

    def build(self, scene: Any, root: str) -> None:
        plane = self.plane or "xy"
        normal = {"xy": (0.0, 0.0, 1.0), "xz": (0.0, 1.0, 0.0), "yz": (1.0, 0.0, 0.0)}[plane]
        self._track(
            scene.add_grid(
                f"{root}/grid",
                width=self.size,
                height=self.size,
                plane=plane,
                cell_color=tuple(int(c) for c in rgb_uint8(self.cell_color)),
                section_color=tuple(int(c) for c in rgb_uint8(self.section_color)),
                position=tuple(self.height * n for n in normal),
            )
        )

    def geometry(self, frame: int) -> list[Geometry]:
        """A plane quad plus grid lines (1 m sections) as thin quads, in the layer's plane."""
        half, step, width = self.size / 2.0, 1.0, 0.006
        ticks = np.arange(-half, half + 1e-6, step, dtype=np.float32)
        quads = [np.array([[-half, -half], [half, -half], [half, half], [-half, half]], dtype=np.float32)]
        for t in ticks:
            quads.append(
                np.array(
                    [[t - width, -half], [t + width, -half], [t + width, half], [t - width, half]], dtype=np.float32
                )
            )
            quads.append(
                np.array(
                    [[-half, t - width], [half, t - width], [half, t + width], [-half, t + width]], dtype=np.float32
                )
            )
        corners = np.stack(quads)  # (Q, 4, 2)
        lift = np.concatenate([[0.0], np.full(len(quads) - 1, 1e-3)]).astype(np.float32)
        axes = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}[self.plane or "xy"]
        vertices = np.zeros((len(quads), 4, 3), dtype=np.float32)
        vertices[..., axes[0]] = corners[..., 0]
        vertices[..., axes[1]] = corners[..., 1]
        vertices[..., axes[2]] = self.height + lift[:, None]
        faces = np.array([[0, 1, 2], [0, 2, 3]])[None] + (np.arange(len(quads)) * 4)[:, None, None]
        colors = np.repeat(
            np.concatenate([[[0.97, 0.97, 0.97]], np.tile(self.cell_color, (len(quads) - 1, 1))]), 2, axis=0
        )
        return [Geometry(vertices.reshape(-1, 3), faces.reshape(-1, 3), colors.astype(np.float32))]

    def set_frame(self, frame: int) -> None:
        pass
