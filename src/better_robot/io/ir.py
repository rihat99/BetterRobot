"""The intermediate representation: what a parser or ``ModelBuilder`` emits and ``build_model`` reads (record 0015).

The records are plain, editable and unordered: parse or build several, merge or edit their lists, then build one
``Model``. Names tie the records together (a joint names its two bodies, a frame and a shape their body);
``build_model`` orders the joints and assigns every index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import final, override

import torch

from ..lie import se3
from ..model.geometry import Shape
from ..model.joints import JointModel


@final
@dataclass(kw_only=True)
class IRJoint:
    """A joint of the description: ``joint`` is its kind, ``origin (7,)`` its placement in the parent body's frame.

    The scalar limits apply to every coordinate of the joint; ``damping`` and ``armature`` to every velocity
    coordinate (URDF ``<dynamics>``, MJCF ``damping`` and ``armature``). A joint with ``mimic_source`` follows that
    joint as ``mimic_multiplier * q_source + mimic_offset`` and has no coordinates of its own. ``neutral (nq,)`` is
    the configuration the model starts from (MJCF ``qpos0``), the kind's own neutral when ``None``; ``build_model``
    clamps it into the limits.
    """

    name: str
    parent_body: str
    child_body: str
    joint: JointModel
    origin: torch.Tensor = field(default_factory=se3.identity)
    lower: float | None = None
    upper: float | None = None
    velocity_limit: float | None = None
    effort_limit: float | None = None
    mimic_source: str | None = None
    mimic_multiplier: float = 1.0
    mimic_offset: float = 0.0
    damping: float = 0.0
    armature: float = 0.0
    neutral: torch.Tensor | None = None

    @override
    def __repr__(self) -> str:
        return f"IRJoint({self.name!r}, {self.parent_body!r} -> {self.child_body!r}, {self.joint!r})"


@final
@dataclass(frozen=True, kw_only=True)
class IRShape:
    """A shape of the description fixed to the body named ``body``: collision geometry, or display only."""

    body: str
    shape: Shape
    collision: bool


@final
@dataclass(kw_only=True)
class IRBody:
    """A rigid body: ``mass`` in kg, ``com (3,)`` and ``inertia (3, 3)`` about the centre of mass, in the body frame."""

    name: str
    mass: float = 0.0
    com: torch.Tensor = field(default_factory=lambda: torch.zeros(3), repr=False)
    inertia: torch.Tensor = field(default_factory=lambda: torch.zeros(3, 3), repr=False)


@final
@dataclass(kw_only=True)
class IRFrame:
    """A named frame placed by ``placement (7,)`` on the body ``parent_body``."""

    name: str
    parent_body: str
    placement: torch.Tensor


@final
@dataclass(kw_only=True)
class IRModel:
    """Flat, unordered intermediate representation of a robot; ``gravity (6,)`` is the world's spatial acceleration."""

    name: str
    bodies: list[IRBody]
    joints: list[IRJoint]
    frames: list[IRFrame] = field(default_factory=list)
    shapes: list[IRShape] = field(default_factory=list)
    root_body: str = ""
    gravity: torch.Tensor = field(
        default_factory=lambda: torch.tensor([0.0, 0.0, -9.81, 0.0, 0.0, 0.0], dtype=torch.float64)
    )

    @override
    def __repr__(self) -> str:
        sizes = f"bodies={len(self.bodies)}, joints={len(self.joints)}, frames={len(self.frames)}, shapes={len(self.shapes)}"
        return f"IRModel({self.name!r}, {sizes})"
