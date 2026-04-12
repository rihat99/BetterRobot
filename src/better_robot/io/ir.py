"""Intermediate representation — parser output, ``build_model()`` input.

All parsers (URDF, MJCF, programmatic builder) emit an ``IRModel``. A single
``build_model()`` factory consumes the IR and produces a frozen ``Model``.
The IR is flat and ordering-unconstrained; topology/idx_q/idx_v assignment
is ``build_model``'s job.

See ``docs/04_PARSERS.md §2``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class IRJoint:
    """A joint in the intermediate representation."""

    name: str
    parent_body: str
    child_body: str
    kind: str
    axis: Optional[torch.Tensor] = None
    origin: torch.Tensor = field(default_factory=lambda: torch.zeros(7))
    lower: Optional[float] = None
    upper: Optional[float] = None
    velocity_limit: Optional[float] = None
    effort_limit: Optional[float] = None
    mimic_source: Optional[str] = None
    mimic_multiplier: float = 1.0
    mimic_offset: float = 0.0


@dataclass
class IRGeom:
    """A visual or collision primitive attached to a body."""

    kind: str
    params: dict
    origin: torch.Tensor
    rgba: Optional[tuple[float, float, float, float]] = None


@dataclass
class IRBody:
    """A rigid body in the intermediate representation."""

    name: str
    mass: float = 0.0
    com: torch.Tensor = field(default_factory=lambda: torch.zeros(3))
    inertia: torch.Tensor = field(default_factory=lambda: torch.zeros(3, 3))
    visual_geoms: list[IRGeom] = field(default_factory=list)
    collision_geoms: list[IRGeom] = field(default_factory=list)


@dataclass
class IRFrame:
    """A named operational / sensor frame attached to a body."""

    name: str
    parent_body: str
    placement: torch.Tensor
    frame_type: str = "op"


@dataclass
class IRModel:
    """Flat, unordered intermediate representation of a robot."""

    name: str
    bodies: list[IRBody]
    joints: list[IRJoint]
    frames: list[IRFrame] = field(default_factory=list)
    root_body: str = ""
    gravity: torch.Tensor = field(
        default_factory=lambda: torch.tensor([0.0, 0.0, -9.81, 0.0, 0.0, 0.0])
    )


class IRError(ValueError):
    """Raised by ``build_model`` when the IR is structurally invalid."""
