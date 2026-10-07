"""Signed distances between spheres and simple shapes, and a robot approximated by spheres.

- Geometry is attached to the robot as ``RobotSpheres``: sphere ``s`` is rigidly fixed to frame ``frame_ids[s]`` at
  ``offsets[s]`` (in that frame's coordinates) with radius ``radii[s]``. ``RobotSpheres.centers(state)`` places them
  in ``WORLD`` from a ``KinematicState``.
- Every distance function returns a signed scalar per query, ``(...)``: positive is the gap between surfaces,
  zero is touching, negative is penetration depth. Gradients stay finite at coincident points.
- ``self_collision_pairs`` lists sphere pairs that can collide; ``self_distances`` returns ``(B..., P)`` for them.
- Functional, batched over leading axes, differentiable, torch only. Robot terms built on this live in
  ``better_robot.terms``.
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
