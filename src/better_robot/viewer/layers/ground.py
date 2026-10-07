"""``GroundLayer``: a static reference grid."""

from __future__ import annotations

from typing import Any


from .._math import RGB, rgb_uint8
from ..layer import Layer


class GroundLayer(Layer):
    """Grid ``size`` wide in the ``plane`` at ``height`` along its normal, ``cell_size`` cells, a line every ``section_size``.

    ``plane=None`` (the default) follows the viewer's theme: ``"xy"`` for a z-up theme, ``"xz"`` for y-up,
    ``"yz"`` for x-up. Pass it explicitly to override, or when the layer is rendered without a ``Viewer``.
    """

    listed = False

    def __init__(
        self,
        name: str = "ground",
        *,
        size: float = 10.0,
        height: float = 0.0,
        cell_size: float = 0.25,
        section_size: float = 1.0,
        plane: str | None = None,
        cell_color: RGB = (0.88, 0.88, 0.89),
        section_color: RGB = (0.74, 0.75, 0.77),
    ) -> None:
        super().__init__(name, 1)
        self.size = size
        self.height = height
        self.cell_size = cell_size
        self.section_size = section_size
        self.plane = plane
        self.cell_color = cell_color
        self.section_color = section_color

    def apply_theme(self, theme: Any) -> None:
        if self.plane is None:
            self.plane = {"z": "xy", "y": "xz", "x": "yz"}[theme.up[-1]]

    def build(self, scene: Any, root: str) -> None:
        plane = self.plane or "xy"
        normal = {"xy": (0.0, 0.0, 1.0), "xz": (0.0, 1.0, 0.0), "yz": (1.0, 0.0, 0.0)}[plane]
        self._track(
            scene.add_grid(
                f"{root}/grid",
                width=self.size,
                height=self.size,
                plane=plane,
                cell_size=self.cell_size,
                section_size=self.section_size,
                cell_color=tuple(int(c) for c in rgb_uint8(self.cell_color)),
                section_color=tuple(int(c) for c in rgb_uint8(self.section_color)),
                position=tuple(self.height * n for n in normal),
            )
        )

    def set_step(self, step: int) -> None:
        pass
