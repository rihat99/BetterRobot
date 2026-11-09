"""Interactive and offline viewing of robots, bodies and data clips, built on viser.

A ``Viewer`` runs a viser server (``http://127.0.0.1:8080`` by default) and holds named groups of ``Layer``s; a
group is toggled, offset and removed as one unit. Every layer is fed preloaded numpy arrays with a leading time
axis ``(T, ...)`` (static input counts as ``T = 1``) and keeps them by reference. One ``Timeline`` drives all
layers: ``show(k)`` pushes frame ``k``, ``show()`` re-pushes after in-place edits, ``play()`` plays by wall clock,
``save(path)`` writes a ``.viser`` recording for viser's static web client, and ``record(path)`` screenshots the
live scene in a headless Chromium into WebM, GIF or PNG frames (``viewer.capture``).

A robot is drawn from one forward-kinematics pass over a clip ``q (T, nq)``: a ``RigidMeshLayer`` takes the joint
poses and the meshes of ``robot_meshes(geometry, root=)`` (a ``Geometry`` from ``io.load_geometry``),
``robot_skeleton(model, joint_poses)`` draws joints (coloured and thickened by ``|tau| / effort_limit``) and bones,
and ``FramesLayer`` and ``SpheresLayer`` take frame poses and sphere centres. ``viewer.tools`` adds a draggable
``PoseGizmo`` and numpy picking (``ray_hit``, ``vertices_in_sphere``, ``vertices_in_rect``). ``LIGHT``
and ``DARK`` are the built-in ``Theme``s; ``Theme(up="+y")`` sets the up axis of the browser scene and of the
``record`` camera. viser is imported only when a ``Viewer`` is created.

```python
import numpy as np
from better_robot.viewer import PointsLayer, Viewer

viewer = Viewer()
viewer.add([PointsLayer("cloud", np.random.rand(100, 50, 3))])  # T = 100 frames of 50 points
viewer.play()  # blocks until Ctrl-C
```
"""

from __future__ import annotations

from .layer import Layer
from .layers import ArrowsLayer, FramesLayer, GroundLayer, PointsLayer, SkeletonLayer
from .layers.image import ImagePanel
from .layers.mesh import MeshLayer, MeshSpec, RigidMeshLayer, SpheresLayer
from .layers.plot import PlotPanel
from .layers.camera import CameraLayer
from .layers.skinned import SkinnedMeshLayer
from .robot import robot_meshes, robot_skeleton
from .theme import DARK, LIGHT, Theme
from .timeline import Timeline
from .viewer import Viewer

__all__ = [
    "DARK",
    "LIGHT",
    "ArrowsLayer",
    "CameraLayer",
    "FramesLayer",
    "GroundLayer",
    "ImagePanel",
    "Layer",
    "MeshLayer",
    "MeshSpec",
    "PlotPanel",
    "PointsLayer",
    "RigidMeshLayer",
    "SkeletonLayer",
    "SkinnedMeshLayer",
    "SpheresLayer",
    "Theme",
    "Timeline",
    "Viewer",
    "robot_meshes",
    "robot_skeleton",
]
