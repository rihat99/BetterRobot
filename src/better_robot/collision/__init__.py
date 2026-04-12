"""``better_robot.collision`` — geometry primitives, pair dispatch, robot
collision decomposition.

Collision lives in its own layer parallel to ``kinematics``/``dynamics`` and
is **optional**: a user who only cares about IK pays nothing for the
collision machinery.

See ``docs/09_COLLISION_GEOMETRY.md``.
"""

from __future__ import annotations

from .geometry import Box, Capsule, HalfSpace, Plane, Sphere, colldist_from_sdf
from .pairs import distance, register_pair
from .robot_collision import RobotCollision

__all__ = [
    "Sphere",
    "Capsule",
    "Box",
    "HalfSpace",
    "Plane",
    "colldist_from_sdf",
    "distance",
    "register_pair",
    "RobotCollision",
]
