"""``Theme``: the one source of the viewer's colours, lights and backdrop, shared by every layer and app on it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, final, override

import numpy as np

from ..exceptions import ArgumentError, ShapeError

RGB = tuple[float, float, float]
_UP_AXES = ("+x", "-x", "+y", "-y", "+z", "-z")  # what viser's set_up_direction takes as a word
#: How far ``shade`` mixes a hue towards black (negative) or white (positive) per role.
_SHADE_MIX = {"joint": -0.22, "face": -0.58, "body": 0.0, "stick": 0.25}
#: Directions the key and fill lights shine from, for a z-up scene; ``apply`` turns them to the theme's ``up``.
_KEY_LIGHT = (-2.0, -3.0, 4.0)
_FILL_LIGHT = (3.0, -1.5, 1.0)
_HEMISPHERE_GROUND = (150, 150, 160)


def _srgb(red: int, green: int, blue: int) -> RGB:
    """An sRGB colour given in 0-255 as the ``[0, 1]`` triple every colour of the viewer takes."""
    return red / 255.0, green / 255.0, blue / 255.0


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


def _uint8(color: RGB) -> tuple[int, int, int]:
    red, green, blue = (int(c) for c in rgb_uint8(color))
    return red, green, blue


@final
@dataclass(frozen=True, kw_only=True)
class Theme:
    """Colours (RGB in ``[0, 1]``, as every layer takes them), lights and the up axis of a viewer.

    ``backdrop`` is the page's vertical gradient (top, bottom), ``grid`` the ground's cell and section lines,
    ``shadow`` the opacity of shadows on the ground, ``lights`` the intensities (ambient, hemisphere, key, fill).
    ``hues`` are the robots' colours in the order robots are added (``hue(k)``), ``shade`` derives a part's colour
    from a hue, ``accent`` colours the GUI and the tools, ``collision`` and ``collision_opacity`` the collision
    overlay, ``text`` labels.
    ``apply(server)`` sends all of it to a viser server; ``Viewer.set_theme`` re-applies it with the layers.
    """

    dark: bool = False
    backdrop: tuple[RGB, RGB] = (_srgb(246, 246, 248), _srgb(226, 227, 232))  # a white robot needs a grey ground
    grid: tuple[RGB, RGB] = (_srgb(218, 219, 224), _srgb(204, 206, 212))
    shadow: float = 0.22
    up: str = "+z"
    lights: tuple[float, float, float, float] = (0.25, 0.5, 2.6, 0.3)  # little ambient: white meshes keep their form
    hues: tuple[RGB, ...] = (
        _srgb(82, 160, 104),  # green
        _srgb(238, 130, 50),  # orange; (221, 132, 82) turned brown under the rig
        _srgb(72, 124, 200),  # blue
        _srgb(142, 112, 192),  # purple
        _srgb(60, 160, 170),  # teal
        _srgb(205, 95, 120),  # rose
        _srgb(110, 115, 125),  # graphite
    )
    accent: RGB = _srgb(72, 124, 200)
    collision: RGB = _srgb(140, 150, 165)  # a quiet grey: the body under the overlay stays the subject
    collision_opacity: float = 0.18
    text: RGB = _srgb(40, 42, 48)

    @override
    def __repr__(self) -> str:
        return f"Theme(dark={self.dark}, up={self.up!r})"

    def __post_init__(self) -> None:
        """Raise ``ArgumentError`` unless ``up`` names a signed axis, such as ``"+z"``, and ``hues`` is not empty."""
        if self.up not in _UP_AXES:
            raise ArgumentError(f"up must be one of {list(_UP_AXES)}, got {self.up!r}")
        if not self.hues:
            raise ArgumentError("hues must hold at least one colour")

    @property
    def up_vector(self) -> tuple[float, float, float]:
        """``up`` as a unit vector, e.g. ``"+y"`` -> ``(0, 1, 0)``."""
        sign = -1.0 if self.up.startswith("-") else 1.0
        axis = "xyz".index(self.up[-1])
        x, y, z = (sign if index == axis else 0.0 for index in range(3))
        return x, y, z

    def hue(self, index: int) -> RGB:
        """The colour of the ``index``-th robot: ``hues[index % len(hues)]``."""
        return self.hues[index % len(self.hues)]

    def shade(self, hue: RGB, role: Literal["joint", "face", "body", "stick"]) -> RGB:
        """``hue`` as a part of ``role`` draws it: ``joint`` 22 % towards black, ``face`` 58 % towards black (drum
        faces), ``body`` the hue itself (bones, hulls), ``stick`` 25 % towards white (thin bones)."""
        if role not in _SHADE_MIX:
            raise ArgumentError(f"role must be one of {list(_SHADE_MIX)}, got {role!r}")
        mix = _SHADE_MIX[role]
        target = 1.0 if mix > 0.0 else 0.0
        red, green, blue = (channel + abs(mix) * (target - channel) for channel in hue)
        return red, green, blue

    def apply(self, server: Any) -> None:
        """Send the theme to a viser server: GUI colours, up axis, the backdrop gradient and the light rig.

        viser's default lights and its environment map are switched off: meshes and solid colours share one rig, an
        ambient, a hemisphere, a key light that casts shadows and a fill light, as scene nodes under ``/theme``.
        Applying another theme replaces them.
        """
        server.gui.configure_theme(
            dark_mode=self.dark,
            brand_color=_uint8(self.accent),
            control_layout="collapsible",
            control_width="medium",
            show_logo=False,
            show_share_button=False,
        )
        scene = server.scene
        scene.set_up_direction(self.up)
        ramp = np.linspace(0.0, 1.0, 256, dtype=np.float32)[:, None, None]
        top, bottom = np.asarray(self.backdrop[0], np.float32), np.asarray(self.backdrop[1], np.float32)
        scene.set_background_image(np.repeat(rgb_uint8((1.0 - ramp) * top + ramp * bottom), 2, axis=1), format="png")
        scene.configure_default_lights(enabled=False, cast_shadow=False)
        scene.configure_environment_map(None)  # viser's default city map washes solid colours out to pastel
        ambient, hemisphere, key, fill = self.lights
        scene.add_light_ambient("/theme/ambient", intensity=ambient)
        scene.add_light_hemisphere(
            "/theme/hemisphere", sky_color=(255, 255, 255), ground_color=_HEMISPHERE_GROUND, intensity=hemisphere
        )
        scene.add_light_directional("/theme/key", intensity=key, cast_shadow=True, position=self._from_z_up(_KEY_LIGHT))
        scene.add_light_directional("/theme/fill", intensity=fill, position=self._from_z_up(_FILL_LIGHT))

    def _from_z_up(self, vector: tuple[float, float, float]) -> tuple[float, float, float]:
        """``vector`` given for a z-up scene, turned so that its z component lies along ``up``."""
        x, y, z = vector
        axis, sign = "xyz".index(self.up[-1]), -1.0 if self.up.startswith("-") else 1.0
        turned = {0: (sign * z, x, y), 1: (y, sign * z, x), 2: (x, y, sign * z)}[axis]
        return turned


LIGHT = Theme()
DARK = Theme(
    dark=True,
    backdrop=(_srgb(44, 45, 50), _srgb(22, 22, 25)),
    grid=(_srgb(58, 59, 64), _srgb(86, 88, 95)),
    shadow=0.45,
    text=_srgb(228, 229, 233),
)
