"""``GroundLayer``: a static reference grid."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING, final, override

import numpy as np
import torch

from ...exceptions import ArgumentError
from ...lie import so3
from ..theme import LIGHT, RGB, rgb_uint8
from ..layer import Layer

if TYPE_CHECKING:
    from ..theme import Theme

#: The grid without edges fades out over this distance from the camera, in metres, at this strength.
_FADE_DISTANCE, _FADE_STRENGTH = 15.0, 1.5


@final
class GroundLayer(Layer):
    """Grid in the ``plane`` at ``height`` along its normal, ``cell_size`` cells, a line every ``section_size``.

    ``size=None`` (the default) draws a grid without edges that fades out with the distance from the camera; a
    number draws a square ``size`` metres wide. Shadows fall on it at the theme's ``shadow`` opacity.

    ``plane=None`` (the default) follows the viewer's theme: ``"xy"`` for a z-up theme, ``"xz"`` for y-up,
    ``"yz"`` for x-up. Pass it explicitly to override, or when the layer is rendered without a ``Viewer``.
    ``normal`` ``(3,)`` tilts the grid to the plane with that normal (a fitted floor), replacing ``plane``;
    ``origin`` ``(3,)`` is the point the grid is centred on before ``height`` moves it along the normal.
    ``fill`` is the opacity of a solid ``fill_color`` plane under the lines; 0 (the default) draws lines only.
    ``cell_color`` and ``section_color`` default to the theme's ``grid`` and follow ``Viewer.set_theme``.
    """

    def __init__(
        self,
        name: str = "ground",
        *,
        size: float | None = None,
        height: float = 0.0,
        cell_size: float = 0.25,
        section_size: float = 1.0,
        plane: str | None = None,
        origin: Sequence[float] | None = None,
        normal: Sequence[float] | None = None,
        fill: float = 0.0,
        cell_color: RGB | None = None,
        section_color: RGB | None = None,
        fill_color: RGB = (0.80, 0.80, 0.80),
    ) -> None:
        super().__init__(name, 1)
        if plane not in (None, "xy", "xz", "yz"):
            raise ArgumentError(f"plane must be 'xy', 'xz', 'yz' or None (follow the theme's up axis), got {plane!r}")
        if normal is not None and not np.linalg.norm(np.asarray(normal, dtype=np.float64).reshape(3)) > 0.0:
            raise ArgumentError(f"normal must be a non-zero 3-vector, got {normal!r}")
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
        self._colors = (cell_color or LIGHT.grid[0], section_color or LIGHT.grid[1])  # (cell, section) drawn
        self._shadow = LIGHT.shadow
        self._grid: Any = None

    @override
    def apply_theme(self, theme: Theme) -> None:
        if self.plane is None:
            self.plane = {"z": "xy", "y": "xz", "x": "yz"}[theme.up[-1]]
        self._colors = (self.cell_color or theme.grid[0], self.section_color or theme.grid[1])
        self._shadow = theme.shadow
        if self._grid is not None:
            self._grid.cell_color = _uint8(self._colors[0])
            self._grid.section_color = _uint8(self._colors[1])
            self._grid.shadow_opacity = theme.shadow

    @override
    def build(self, scene: Any, root: str) -> None:
        if self.normal is None:
            plane = self.plane or "xy"
            axis = np.array({"xy": (0.0, 0.0, 1.0), "xz": (0.0, 1.0, 0.0), "yz": (1.0, 0.0, 0.0)}[plane])
            wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            plane = "xy"  # the grid's own normal is +z; rotate it onto ``normal``
            axis = np.asarray(self.normal, dtype=np.float64).reshape(3)
            axis = axis / np.linalg.norm(axis)
            turn = np.cross((0.0, 0.0, 1.0), axis)  # the shortest turn of +z onto the normal; a half turn about x
            sine = float(np.linalg.norm(turn))
            unit = turn / sine if sine > 1e-9 else np.array([1.0, 0.0, 0.0])
            xyzw = so3.from_axis_angle(
                torch.from_numpy(unit), torch.tensor(math.atan2(sine, axis[2]), dtype=torch.float64)
            ).numpy()
            wxyz = xyzw[[3, 0, 1, 2]]
        origin = np.zeros(3) if self.origin is None else np.asarray(self.origin, dtype=np.float64).reshape(3)
        self._grid = self._track(
            scene.add_grid(
                f"{root}/grid",
                # viser scales the plane of a grid without edges by ``1 + fade_distance``: it needs a real size too
                width=self.size or _FADE_DISTANCE,
                height=self.size or _FADE_DISTANCE,
                infinite_grid=self.size is None,
                fade_distance=_FADE_DISTANCE,
                fade_strength=_FADE_STRENGTH,
                shadow_opacity=self._shadow,
                plane=plane,
                cell_size=self.cell_size,
                section_size=self.section_size,
                cell_color=_uint8(self._colors[0]),
                section_color=_uint8(self._colors[1]),
                plane_color=_uint8(self.fill_color),
                plane_opacity=self.fill,
                wxyz=tuple(float(c) for c in wxyz),
                position=tuple(float(c) for c in origin + self.height * axis),
            )
        )

    @override
    def set_step(self, step: int) -> None:
        pass


def _uint8(color: RGB) -> tuple[int, int, int]:
    red, green, blue = (int(c) for c in rgb_uint8(color))
    return red, green, blue
