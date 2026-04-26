"""``Body`` — rigid body attached to a joint.

Bodies carry inertial properties; ``Model.body_inertias`` stores them as a
packed ``(nbodies, 10)`` tensor. This struct is the metadata (name, parent
joint, optional geometry index).

See ``docs/concepts/joints_bodies_frames.md §9``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Body:
    """A rigid body attached to a joint via ``model.joint_placements[parent_joint]``."""

    name: str
    parent_joint: int
    visual_geom: int | None = None
