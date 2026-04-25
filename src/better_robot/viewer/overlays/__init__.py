"""``overlays`` — composable scene overlays.

See ``docs/12_VIEWER.md §5``.
"""

from __future__ import annotations

from .frame_axes import FrameAxesOverlay
from .force_vectors import ForceVectorsOverlay
from .grid import GridOverlay
from .com import ComOverlay
from .path_trace import PathTraceOverlay
from .targets import TargetsOverlay
from .residual_plot import ResidualPlotOverlay

__all__ = [
    "ComOverlay",
    "ForceVectorsOverlay",
    "FrameAxesOverlay",
    "GridOverlay",
    "PathTraceOverlay",
    "ResidualPlotOverlay",
    "TargetsOverlay",
]
