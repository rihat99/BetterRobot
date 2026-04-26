"""``themes.py`` — colour/style configuration for the viewer.

See ``docs/design/12_VIEWER.md §6``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Theme:
    """Visual theme for the robot viewer."""

    background_colour: tuple[float, float, float] = (0.15, 0.15, 0.15)
    joint_colour: tuple[float, float, float, float] = (0.2, 0.6, 1.0, 1.0)
    link_colour: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)
    grid_colour: tuple[float, float, float, float] = (0.3, 0.3, 0.3, 0.5)
    grid_size: float = 2.0
    grid_divisions: int = 10
    axes_length: float = 0.1


DEFAULT_THEME = Theme()
