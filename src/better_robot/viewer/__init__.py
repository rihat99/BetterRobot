"""Interactive and offline viewing of robots, bodies and data clips, built on viser.

A ``Viewer`` runs a viser server (``http://127.0.0.1:8080`` by default) and holds ``Layer``s under tags; a tag
is toggled, offset and removed as one unit. Every layer is fed preloaded numpy arrays with a leading time
axis ``(T, ...)`` (static input counts as ``T = 1``) and keeps them by reference. One ``Timeline`` drives all
layers: ``show(k)`` pushes frame ``k``, ``layer.redraw()`` re-pushes one layer after in-place edits, ``serve()``
plays by wall clock until interrupted, ``save(path)`` writes a ``.viser`` recording for viser's static web client,
and ``record(path)`` draws the live scene in a headless Chromium into a video (``.mp4`` or ``.webm``) or a still
(``.png``). The ``Sidebar`` holds the controls in ordered sections; one ``Theme`` sets colours, lights and the
backdrop. ``PointsLayer`` (a list of arrays is a cloud whose size changes), ``LinesLayer``, ``ArrowsLayer``,
``MeshLayer`` (one colour or one per vertex) and ``CameraLayer`` (a frustum from the intrinsics ``K``) draw data.

A robot is one layer, ``RobotVisual(model, q)``: it runs forward kinematics over ``q (nq,)`` or a clip ``(T, nq)``
and shows the robot as its ``"meshes"`` (a ``Geometry`` from ``io.load_geometry``, or a ``Skin``) or as a
``"minimal"`` or ``"maximal"`` skeleton laid out by a ``SkeletonRecipe`` (``MINIMAL``, ``MAXIMAL``), with a collision
overlay and frame triads; the style switches while the viewer runs and ``update(q)`` shows new configurations.
``viewer.tools`` adds a draggable ``PoseGizmo`` and numpy picking (``ray_hit``, ``vertices_in_sphere``,
``vertices_in_rect``). ``LIGHT`` and ``DARK`` are the built-in ``Theme``s; ``Theme(up="+y")`` sets the up axis of
the browser scene and of the ``record`` camera. viser is imported only when a ``Viewer`` is created.

```python
import numpy as np
from better_robot.viewer import PointsLayer, Viewer

viewer = Viewer()
viewer.add(PointsLayer("cloud", np.random.rand(100, 50, 3)))  # T = 100 frames of 50 points
viewer.serve()  # plays and blocks until Ctrl-C
```
"""

from __future__ import annotations

from .layer import Layer
from .layers import ArrowsLayer, FramesLayer, GroundLayer, LinesLayer, PointsLayer
from .layers.image import ImagePanel
from .layers.mesh import MeshLayer, MeshSpec, RigidMeshLayer, SpheresLayer
from .layers.plot import PlotPanel
from .layers.camera import CameraLayer
from .layers.skinned import SkinnedMeshLayer
from .robot import RobotVisual, Skin
from .sidebar import Sidebar
from .skeleton import MAXIMAL, MINIMAL, SkeletonRecipe
from .theme import DARK, LIGHT, Theme
from .timeline import Timeline
from .viewer import Viewer

__all__ = [
    "DARK",
    "LIGHT",
    "MAXIMAL",
    "MINIMAL",
    "ArrowsLayer",
    "CameraLayer",
    "FramesLayer",
    "GroundLayer",
    "ImagePanel",
    "Layer",
    "LinesLayer",
    "MeshLayer",
    "MeshSpec",
    "PlotPanel",
    "PointsLayer",
    "RigidMeshLayer",
    "RobotVisual",
    "Sidebar",
    "SkeletonRecipe",
    "Skin",
    "SkinnedMeshLayer",
    "SpheresLayer",
    "Theme",
    "Timeline",
    "Viewer",
]
