"""``GroundLayer``: a static reference grid."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .._math import RGB, quat_z_to, rgb_uint8
from ..layer import Layer


class GroundLayer(Layer):
    """Grid ``size`` wide in the ``plane`` at ``height`` along its normal, ``cell_size`` cells, a line every ``section_size``.

    ``plane=None`` (the default) follows the viewer's theme: ``"xy"`` for a z-up theme, ``"xz"`` for y-up,
    ``"yz"`` for x-up. Pass it explicitly to override, or when the layer is rendered without a ``Viewer``.
    ``normal`` ``(3,)`` tilts the grid to the plane with that normal (a fitted floor), replacing ``plane``;
    ``origin`` ``(3,)`` is the point the grid is centred on before ``height`` moves it along the normal.
    ``fill`` is the opacity of a solid ``fill_color`` plane under the lines; 0 (the default) draws lines only.
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
        origin: Sequence[float] | None = None,
        normal: Sequence[float] | None = None,
        fill: float = 0.0,
        cell_color: RGB = (0.88, 0.88, 0.89),
        section_color: RGB = (0.74, 0.75, 0.77),
        fill_color: RGB = (0.80, 0.80, 0.80),
    ) -> None:
        super().__init__(name, 1)
        if normal is not None and not np.linalg.norm(np.asarray(normal, dtype=np.float64).reshape(3)) > 0.0:
            raise ValueError("normal must be a non-zero 3-vector")
        self.size = size
        self.height = height
        self.cell_size = cell_size
        self.section_size = section_size
        self.plane = plane
        self.origin = origin
        self.normal = normal
        self.fill = fill
        self.cell_color = cell_color
        self.section_color = section_color
        self.fill_color = fill_color

    def apply_theme(self, theme: Any) -> None:
        if self.plane is None:
            self.plane = {"z": "xy", "y": "xz", "x": "yz"}[theme.up[-1]]

    def build(self, scene: Any, root: str) -> None:
        if self.normal is None:
            plane = self.plane or "xy"
            axis = np.array({"xy": (0.0, 0.0, 1.0), "xz": (0.0, 1.0, 0.0), "yz": (1.0, 0.0, 0.0)}[plane])
            wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            plane = "xy"  # the grid's own normal is +z; rotate it onto ``normal``
            axis = np.asarray(self.normal, dtype=np.float64).reshape(3)
            axis = axis / np.linalg.norm(axis)
            wxyz = quat_z_to(axis)
        origin = np.zeros(3) if self.origin is None else np.asarray(self.origin, dtype=np.float64).reshape(3)
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
                plane_color=tuple(int(c) for c in rgb_uint8(self.fill_color)),
                plane_opacity=self.fill,
                wxyz=tuple(float(c) for c in wxyz),
                position=tuple(float(c) for c in origin + self.height * axis),
            )
        )

    def set_step(self, step: int) -> None:
        pass
