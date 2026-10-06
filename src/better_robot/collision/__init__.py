"""``better_robot.collision`` — signed distances between spheres and simple shapes, and a robot as spheres.

Functional, batched and differentiable; torch only. Negative distance means penetration.
Collision lives in its own layer parallel to ``kinematics``/``dynamics``.
"""

from __future__ import annotations

from .distance import sphere_box, sphere_capsule, sphere_halfspace, sphere_sdf, sphere_sphere
from .robot_spheres import RobotSpheres, self_collision_pairs, self_distances

__all__ = [
    "RobotSpheres",
    "self_collision_pairs",
    "self_distances",
    "sphere_box",
    "sphere_capsule",
    "sphere_halfspace",
    "sphere_sdf",
    "sphere_sphere",
]
