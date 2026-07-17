"""``overlays`` — composable scene overlays.

See ``docs/concepts/viewer.md §5``.
"""

from __future__ import annotations

from .frame_axes import FrameAxesOverlay
from .force_vectors import ForceVectorsOverlay
from .grid import GridOverlay
from .targets import TargetsOverlay

__all__ = [
    "ForceVectorsOverlay",
    "FrameAxesOverlay",
    "GridOverlay",
    "TargetsOverlay",
]
