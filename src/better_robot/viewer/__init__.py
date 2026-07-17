"""``better_robot.viewer`` — viser-backed visualisation helpers.

Topmost in the DAG — nothing imports from ``viewer``.

Scope: interactive viser-backed rendering of one robot via
``SkeletonMode`` / ``URDFMeshMode`` plus ``GridOverlay`` /
``FrameAxesOverlay``, with straight-through-the-frames trajectory
playback, target gizmos, force vectors, and public primitive styling.

All heavy rendering dependencies (viser, trimesh, and later pyrender /
imageio-ffmpeg) are imported lazily inside their respective submodules.
This module can be imported on machines with none of those packages
installed; the relevant error fires only when a code path that needs a
missing library is first taken.

See ``docs/concepts/viewer.md §13``.
"""

from __future__ import annotations

from . import helpers
from .primitive import PrimitiveHandle
from .render_modes.base import RenderContext, RenderMode
from .render_modes.skeleton import SkeletonMode
from .render_modes.urdf_mesh import URDFMeshMode
from .renderers.base import RendererBackend
from .renderers.viser_backend import ViserBackend
from .scene import Scene
from .trajectory_player import TrajectoryPlayer
from .visualizer import Visualizer

__all__ = [
    # Interactive facade (V1)
    "Visualizer",
    "Scene",
    # Render modes
    "RenderMode",
    "RenderContext",
    "SkeletonMode",
    "URDFMeshMode",
    # Renderer backends
    "RendererBackend",
    "ViserBackend",
    # Playback and styling
    "TrajectoryPlayer",
    "PrimitiveHandle",
    # misc
    "helpers",
]
