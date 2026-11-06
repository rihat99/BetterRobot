"""``Geometry``: the shapes attached to a robot's bodies, a value beside ``Model`` (decision record 0016).

A shape record carries its own placement and colour; ``AttachedShape`` fixes it to a body of the model and says
whether it is collision geometry; ``Geometry`` is the tuple of those. ``io.build_geometry`` and ``io.load_geometry``
build one, ``viewer.robot_meshes`` reads one. Nothing here holds a tensor: a placement is a plain 7-tuple, so every
record is a hashable value and a consumer packs the tensors it needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import final

#: ``[tx, ty, tz, qx, qy, qz, qw]`` of a shape in its body's joint frame.
Pose = tuple[float, float, float, float, float, float, float]
#: ``(red, green, blue, alpha)`` in ``[0, 1]``.
Color = tuple[float, float, float, float]


@final
@dataclass(frozen=True)
class Sphere:
    """A sphere of ``radius`` centred at ``placement``."""

    radius: float
    placement: Pose
    color: Color | None = None


@final
@dataclass(frozen=True)
class Box:
    """A box of full extents ``size`` ``(x, y, z)`` centred at ``placement``."""

    size: tuple[float, float, float]
    placement: Pose
    color: Color | None = None


@final
@dataclass(frozen=True)
class Cylinder:
    """A cylinder of ``radius`` and ``length`` along the z axis of ``placement``, centred there."""

    radius: float
    length: float
    placement: Pose
    color: Color | None = None


@final
@dataclass(frozen=True)
class Capsule:
    """A cylinder of ``radius`` and ``length`` along the z axis of ``placement`` with a hemisphere on each end."""

    radius: float
    length: float
    placement: Pose
    color: Color | None = None


@final
@dataclass(frozen=True)
class Mesh:
    """A mesh file named by ``uri`` as the description wrote it (a path or ``package://...``), scaled per axis."""

    uri: str
    scale: tuple[float, float, float]
    placement: Pose
    color: Color | None = None


#: The shape records a ``Geometry`` holds.
Shape = Sphere | Box | Cylinder | Capsule | Mesh


@final
@dataclass(frozen=True)
class AttachedShape:
    """``shape`` fixed to the body with id ``body``: collision geometry, or with ``collision=False`` display only."""

    body: int
    shape: Shape
    collision: bool


@final
@dataclass(frozen=True)
class Geometry:
    """The shapes attached to a robot's bodies, in description order."""

    shapes: tuple[AttachedShape, ...]
