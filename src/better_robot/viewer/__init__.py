"""``better_robot.viewer`` — viser-backed visualisation helpers.

Topmost in the DAG — nothing imports from ``viewer``.

V1 scope: interactive viser-backed rendering of one robot via
``SkeletonMode`` / ``URDFMeshMode`` plus ``GridOverlay`` /
``FrameAxesOverlay``, with straight-through-the-frames trajectory
playback. Video recording, offscreen rendering, collision capsules,
target gizmos, COM marker, path trace, residual plot, camera paths,
and multi-robot sessions all live as placeholders — see
``docs/design/12_VIEWER.md §10``.

All heavy rendering dependencies (viser, trimesh, and later pyrender /
imageio-ffmpeg) are imported lazily inside their respective submodules.
This module can be imported on machines with none of those packages
installed; the relevant error fires only when a code path that needs a
missing library is first taken.

See ``docs/design/12_VIEWER.md §13``.
"""

from __future__ import annotations

from . import helpers
from .camera import Camera, CameraPath
from .recorder import VideoRecorder, render_trajectory
from .render_modes.base import RenderContext, RenderMode
from .render_modes.collision import CollisionMode
from .render_modes.skeleton import SkeletonMode
from .render_modes.urdf_mesh import URDFMeshMode
from .renderers.base import RendererBackend
from .renderers.offscreen_backend import OffscreenBackend
from .renderers.viser_backend import ViserBackend
from .scene import Scene
from .trajectory_player import TrajectoryPlayer
from .visualizer import Visualizer

__all__ = [
    # Interactive facade (V1)
    "Visualizer",
    "Scene",
    # Render modes (V1: Skeleton, URDFMesh; stubs: Collision)
    "RenderMode",
    "RenderContext",
    "SkeletonMode",
    "URDFMeshMode",
    "CollisionMode",
    # Renderer backends (V1: Viser; stubs: Offscreen)
    "RendererBackend",
    "ViserBackend",
    "OffscreenBackend",
    # Playback (V1 minimal)
    "TrajectoryPlayer",
    # Camera (dataclass V1; path variants are §10.7 stubs)
    "Camera",
    "CameraPath",
    # Video recording — §10.1 stubs
    "VideoRecorder",
    "render_trajectory",
    # misc
    "helpers",
]
