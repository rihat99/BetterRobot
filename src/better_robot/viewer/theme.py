"""``Theme``: the palette and lighting shared by every layer and app built on the viewer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._math import RGB, quat_z_to, rgb_uint8


@dataclass(frozen=True)
class Theme:
    """Colours in ``[0, 1]`` RGB plus the few scene settings the viewer applies at start-up."""

    dark: bool = False
    brand: RGB = (0.16, 0.45, 0.85)
    joint: RGB = (0.20, 0.60, 1.00)
    link: RGB = (0.72, 0.72, 0.72)
    robot: RGB = (0.80, 0.80, 0.80)
    human: RGB = (0.85, 0.72, 0.62)
    point: RGB = (0.25, 0.25, 0.25)
    force: RGB = (0.90, 0.25, 0.20)
    torque: RGB = (0.95, 0.60, 0.10)
    target: RGB = (0.10, 0.75, 0.35)
    grid_cell: RGB = (0.78, 0.78, 0.78)
    grid_section: RGB = (0.55, 0.55, 0.55)
    key_light_direction: tuple[float, float, float] = (-0.4, -0.6, -1.0)
    up: str = "+z"

    @property
    def up_vector(self) -> tuple[float, float, float]:
        """``up`` as a unit vector, e.g. ``"+y"`` -> ``(0, 1, 0)``."""
        sign = -1.0 if self.up.startswith("-") else 1.0
        axis = "xyz".index(self.up[-1])
        return tuple(sign if index == axis else 0.0 for index in range(3))

    def apply(self, server: Any) -> None:
        """Configure the GUI theme, up direction and lights of a viser server."""
        server.gui.configure_theme(
            dark_mode=self.dark,
            control_layout="collapsible",
            brand_color=tuple(int(c) for c in rgb_uint8(self.brand)),
            show_logo=False,
        )
        server.scene.set_up_direction(self.up)
        server.scene.configure_default_lights(enabled=False)
        # Directional lights shine along their local -z axis.
        direction = -np.asarray(self.key_light_direction, dtype=np.float32)
        server.scene.add_light_directional("/lights/key", intensity=2.5, cast_shadow=True, wxyz=quat_z_to(direction))
        server.scene.add_light_hemisphere(
            "/lights/fill", sky_color=(230, 235, 255), ground_color=(120, 110, 100), intensity=1.2
        )


LIGHT = Theme()
DARK = Theme(dark=True, grid_cell=(0.30, 0.30, 0.30), grid_section=(0.45, 0.45, 0.45), point=(0.80, 0.80, 0.80))
