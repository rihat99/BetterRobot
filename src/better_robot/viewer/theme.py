"""``Theme``: the palette and GUI settings shared by every layer and app built on the viewer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


from ._math import RGB, rgb_uint8


@dataclass(frozen=True)
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
    force: RGB = (0.74, 0.32, 0.30)
    torque: RGB = (0.82, 0.60, 0.30)
    target: RGB = (0.27, 0.58, 0.47)
    grid_cell: RGB = (0.86, 0.87, 0.89)
    grid_section: RGB = (0.70, 0.72, 0.76)
    up: str = "+z"

    @property
    def up_vector(self) -> tuple[float, float, float]:
        """``up`` as a unit vector, e.g. ``"+y"`` -> ``(0, 1, 0)``."""
        sign = -1.0 if self.up.startswith("-") else 1.0
        axis = "xyz".index(self.up[-1])
        return tuple(sign if index == axis else 0.0 for index in range(3))

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
)
