"""``Theme``: the palette and GUI settings shared by every layer and app built on the viewer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, final, override

import numpy as np

from ..exceptions import ArgumentError, ShapeError

RGB = tuple[float, float, float]
_UP_AXES = ("+x", "-x", "+y", "-y", "+z", "-z")  # what viser's set_up_direction takes as a word


def rgb_uint8(color: np.ndarray | RGB, shape: tuple[int, ...] = ()) -> np.ndarray:
    """Convert colours in ``[0, 1]`` to uint8 ``(*shape, 3)``, broadcasting a single colour; a colour that is not an
    RGB triple, or a count other than ``shape``, raises ``ShapeError``."""
    out = np.asarray(color, dtype=np.float32)
    if out.shape == (3,) and shape:
        out = np.broadcast_to(out, (*shape, 3))
    if out.shape[-1:] != (3,) or (shape and out.shape != (*shape, 3)):
        expected = f"one RGB or {(*shape, 3)}, one per element" if shape else "RGB triples (..., 3)"
        raise ShapeError(f"colors must be {expected}, got shape {np.shape(color)}")
    return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def colormap(values: np.ndarray, *stops: RGB) -> np.ndarray:
    """Piecewise-linear colours ``(..., 3)`` through evenly spaced ``stops`` (at least two) for ``values`` in
    ``[0, 1]`` (clamped); two stops are a plain blend from the first to the second."""
    stops_array = np.asarray(stops, dtype=np.float32)
    position = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0) * (len(stops_array) - 1)
    segment = np.minimum(position.astype(np.int64), len(stops_array) - 2)
    t = (position - segment)[..., None]
    return (1.0 - t) * stops_array[segment] + t * stops_array[segment + 1]


@final
@dataclass(frozen=True, kw_only=True)
class Theme:
    """Colours in ``[0, 1]`` RGB plus the few scene settings the viewer applies at start-up."""

    dark: bool = False
    brand_color: RGB = (0.122, 0.227, 0.408)  # navy #1f3a68: GUI accent and skeleton joints
    control_width: str = "medium"  # viser sidebar width: "small", "medium" or "large"
    joint: RGB = (0.122, 0.227, 0.408)
    link: RGB = (0.62, 0.64, 0.68)
    robot: RGB = (0.62, 0.63, 0.65)
    human: RGB = (0.85, 0.72, 0.62)
    point: RGB = (0.30, 0.32, 0.36)
    force: RGB = (0.80, 0.10, 0.10)  # red: contact-force arrows
    torque: RGB = (0.82, 0.60, 0.30)
    # Joint load |tau| / limit from 0 to 1 (dark navy, teal, green, yellow; saturated enough to survive viser's
    # lights), and past the limit (magenta).
    torque_ramp: tuple[RGB, ...] = ((0.03, 0.07, 0.20), (0.0, 0.38, 0.42), (0.25, 0.55, 0.10), (0.95, 0.75, 0.0))
    over_limit: RGB = (0.80, 0.0, 0.70)
    contact: RGB = (0.10, 0.75, 0.30)  # green: a contact point or frame in contact
    target: RGB = (0.27, 0.58, 0.47)
    grid_cell: RGB = (0.86, 0.87, 0.89)
    grid_section: RGB = (0.70, 0.72, 0.76)
    up: str = "+z"

    @override
    def __repr__(self) -> str:
        return f"Theme(dark={self.dark}, up={self.up!r})"

    def __post_init__(self) -> None:
        """Raise ``ArgumentError`` unless ``up`` names a signed axis, such as ``"+z"``."""
        if self.up not in _UP_AXES:
            raise ArgumentError(f"up must be one of {list(_UP_AXES)}, got {self.up!r}")

    @property
    def up_vector(self) -> tuple[float, float, float]:
        """``up`` as a unit vector, e.g. ``"+y"`` -> ``(0, 1, 0)``."""
        sign = -1.0 if self.up.startswith("-") else 1.0
        axis = "xyz".index(self.up[-1])
        x, y, z = (sign if index == axis else 0.0 for index in range(3))
        return x, y, z

    def apply(self, server: Any) -> None:
        """Configure the GUI theme, up direction and viser's default lights on a viser server."""
        server.gui.configure_theme(
            dark_mode=self.dark,
            brand_color=tuple(int(c) for c in rgb_uint8(self.brand_color)),
            control_layout="collapsible",
            control_width=self.control_width,
            show_logo=False,
            show_share_button=False,
        )
        server.scene.set_up_direction(self.up)
        server.scene.configure_default_lights(enabled=True, cast_shadow=False)


LIGHT = Theme()
DARK = Theme(
    dark=True,
    joint=(0.45, 0.60, 0.85),
    link=(0.55, 0.57, 0.60),
    grid_cell=(0.30, 0.30, 0.30),
    grid_section=(0.45, 0.45, 0.45),
    point=(0.80, 0.80, 0.80),
    torque_ramp=((0.18, 0.28, 0.62), *LIGHT.torque_ramp[1:]),  # a brighter navy at zero load: visible on dark
)
