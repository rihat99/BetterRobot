"""Signed distances between spheres, capsules and simple shapes, and a robot approximated by spheres or capsules.

- Geometry is attached to the robot as ``RobotSpheres``: sphere ``s`` is rigidly fixed to frame ``frame_ids[s]`` at
  ``offsets[s]`` (in that frame's coordinates) with radius ``radii[s]``. ``RobotSpheres.centers(state)`` places them
  in ``WORLD`` from a ``KinematicState``. ``RobotCapsules`` does the same for segments ``starts[c]``-``ends[c]``
  with radius ``radii[c]`` (``segments(state)``).
- Every distance function returns a signed scalar per query, ``(...)``: positive is the gap between surfaces,
  zero is touching, negative is penetration depth. Gradients stay finite at coincident points.
- ``self_collision_pairs`` lists sphere (or capsule) pairs that can collide; ``self_distances`` and
  ``self_capsule_distances`` return ``(B..., P)`` for them.
- Functional, batched over leading axes, differentiable, torch only. Robot terms built on this live in
  ``better_robot.terms``.
"""

from __future__ import annotations

from .distance import capsule_capsule, sphere_box, sphere_capsule, sphere_halfspace, sphere_sdf, sphere_sphere
from .robot_capsules import RobotCapsules, self_capsule_distances
from .robot_spheres import RobotSpheres, self_collision_pairs, self_distances

__all__ = [
    "RobotCapsules",
    "RobotSpheres",
    "capsule_capsule",
    "self_capsule_distances",
    "self_collision_pairs",
    "self_distances",
    "sphere_box",
    "sphere_capsule",
    "sphere_halfspace",
    "sphere_sdf",
    "sphere_sphere",
]
