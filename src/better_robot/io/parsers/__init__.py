"""``better_robot.io.parsers`` — URDF / MJCF / programmatic parsers.

See ``docs/design/04_PARSERS.md``.
"""

from __future__ import annotations

from .mjcf import parse_mjcf
from .programmatic import ModelBuilder
from .urdf import parse_urdf

__all__ = ["parse_urdf", "parse_mjcf", "ModelBuilder"]
