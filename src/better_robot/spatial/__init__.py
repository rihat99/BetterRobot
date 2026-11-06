"""Spatial algebra on tensors: twists, wrenches and rigid-body inertias, plus mesh face-table helpers.

- Twist (motion) and wrench (force) ``(..., 6)``: linear first, angular second, the order of the ``lie`` tangents.
  `cross_motion` and `cross_force` are the spatial cross products; `transform_motion`, `transform_force` and
  `transform_inertia` re-express a quantity in another frame, taking the pose first like ``se3.act(pose, point)``.
- Inertia ``(..., 10)`` = ``[mass, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]``: the centre of mass ``c`` and the
  rotational inertia about it, both in the body frame, in MuJoCo's ``fullinertia`` order (not Pinocchio's
  ``toDynamicParameters``, which is about the frame origin). `inertia_matrix` gives the ``(..., 6, 6)`` matrix
  about the frame origin and `rotational_inertia` the ``(..., 3, 3)`` matrix about the centre of mass;
  `inertia_from_mass_com`, `inertia_of_sphere`, `inertia_of_box`, `inertia_of_capsule`, `inertia_of_ellipsoid` and
  `inertia_of_mesh` build one.
- Mesh helpers on the integer face table: `validate_closed_manifold`, `orient_faces_consistently`,
  `orient_faces_by_component` (inputs for `inertia_of_mesh`).
"""

from __future__ import annotations

from ..lie.types import SE3, SO3, Pose
from .algebra import cross_force, cross_motion, transform_force, transform_inertia, transform_motion
from .force import Force
from .inertia import (
    Inertia,
    inertia_from_mass_com,
    inertia_matrix,
    inertia_of_box,
    inertia_of_capsule,
    inertia_of_ellipsoid,
    inertia_of_mesh,
    inertia_of_sphere,
    rotational_inertia,
)
from .mesh import orient_faces_by_component, orient_faces_consistently, validate_closed_manifold
from .motion import Motion
from .symmetric3 import Symmetric3

__all__ = [
    "cross_motion",
    "cross_force",
    "transform_motion",
    "transform_force",
    "transform_inertia",
    "inertia_matrix",
    "rotational_inertia",
    "inertia_from_mass_com",
    "inertia_of_sphere",
    "inertia_of_box",
    "inertia_of_capsule",
    "inertia_of_ellipsoid",
    "inertia_of_mesh",
    "orient_faces_by_component",
    "orient_faces_consistently",
    "validate_closed_manifold",
    "Motion",
    "Force",
    "Inertia",
    "Symmetric3",
    "SE3",
    "SO3",
    "Pose",
]
