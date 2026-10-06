"""``GroundLayer``: a static reference grid."""

from __future__ import annotations

from typing import Any

from .._math import RGB, rgb_uint8
from ..layer import Layer


class GroundLayer(Layer):
    """Grid in the ``plane`` (``"xy"`` for z-up) at ``height`` along the normal."""

    def __init__(
        self,
        name: str = "ground",
        *,
        size: float = 10.0,
        height: float = 0.0,
        plane: str = "xy",
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
        normal = {"xy": (0.0, 0.0, 1.0), "xz": (0.0, 1.0, 0.0), "yz": (1.0, 0.0, 0.0)}[self.plane]
        self._track(
            scene.add_grid(
                f"{root}/grid",
                width=self.size,
                height=self.size,
                plane=self.plane,
                cell_color=tuple(int(c) for c in rgb_uint8(self.cell_color)),
                section_color=tuple(int(c) for c in rgb_uint8(self.section_color)),
                position=tuple(self.height * n for n in normal),
            )
        )

    def set_frame(self, frame: int) -> None:
        pass
