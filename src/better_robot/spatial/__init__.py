"""6D spatial algebra value types: twists, wrenches, rigid-body inertia, plus mesh-inertia helpers.

- ``Motion`` ``(..., 6)`` = ``[vx, vy, vz, wx, wy, wz]`` and ``Force`` ``(..., 6)`` = ``[fx, fy, fz, tx, ty, tz]``:
  linear first, angular second, the same order as the ``lie`` tangents.
- ``Inertia`` ``(..., 10)`` = ``[mass, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]``; ``c`` is the centre of mass and
  ``I`` the rotational inertia about the centre of mass, both in the body frame.
- ``Symmetric3`` ``(..., 6)`` = ``[xx, yy, zz, xy, xz, yz]``.
- Each is a frozen dataclass around one tensor (``.data``); methods return new values. ``SE3``, ``SO3`` and
  ``Pose`` are re-exported from ``better_robot.lie``. Mesh helpers: ``validate_closed_manifold``,
  ``orient_faces_consistently``, ``orient_faces_by_component`` (inputs for ``Inertia.from_mesh``).
"""

from __future__ import annotations

from ..lie.types import SE3, SO3, Pose
from .force import Force
from .inertia import Inertia
from .mesh import orient_faces_by_component, orient_faces_consistently, validate_closed_manifold
from .motion import Motion
from .symmetric3 import Symmetric3

__all__ = [
    "Motion",
    "Force",
    "Inertia",
    "Symmetric3",
    "SE3",
    "SO3",
    "Pose",
    "orient_faces_by_component",
    "orient_faces_consistently",
    "validate_closed_manifold",
]
