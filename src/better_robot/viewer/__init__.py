"""``better_robot.viewer``: one modular viewer on viser.

``Viewer`` owns a server, groups of ``Layer``s and a ``Timeline``; layers are fed preloaded
``(T, ...)`` numpy arrays and pushed one frame at a time. ``robot_layers`` / ``RobotView`` turn a
``Model`` and a clip ``q (T, nq)`` into layers. Tools (``viewer.tools``) and offline rendering
(``viewer.render``) sit on top. viser is imported only when a ``Viewer`` is created.
"""

from __future__ import annotations

from .layer import Layer
from .layers import ArrowsLayer, FramesLayer, GroundLayer, PointsLayer, SkeletonLayer
from .layers.mesh import MeshLayer, MeshSpec, RigidMeshLayer, SpheresLayer
from .layers.plot import PlotPanel
from .layers.skinned import CameraLayer, SkinnedMeshLayer
from .robot import RobotView, robot_layers, robot_meshes
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
    "Layer",
    "MeshLayer",
    "MeshSpec",
    "PlotPanel",
    "PointsLayer",
    "RigidMeshLayer",
    "RobotView",
    "SkeletonLayer",
    "SkinnedMeshLayer",
    "SpheresLayer",
    "Theme",
    "Timeline",
    "Viewer",
    "robot_layers",
    "robot_meshes",
]
