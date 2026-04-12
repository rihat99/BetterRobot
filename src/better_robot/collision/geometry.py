"""Collision primitives and the ``colldist_from_sdf`` smoothing function.

Primitives are tensor-valued from day one: a ``Sphere`` whose ``center`` has
shape ``(B, K, 3)`` represents ``B*K`` spheres, not one. This is how the
same types express "one sphere" and "a robot's worth of self-collision
spheres" without a second dataclass.

See ``docs/09_COLLISION_GEOMETRY.md §3, §7``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Sphere:
    """Sphere in body-local frame. ``center`` ``(..., 3)``, ``radius`` ``(...,)``."""

    center: torch.Tensor
    radius: torch.Tensor


@dataclass(frozen=True)
class Capsule:
    """Capsule defined by two endpoints + radius, body-local frame."""

    a: torch.Tensor  # (..., 3)
    b: torch.Tensor  # (..., 3)
    radius: torch.Tensor  # (...,)


@dataclass(frozen=True)
class Box:
    """Box with local rotation, half-extents, and center."""

    center: torch.Tensor  # (..., 3)
    half_extents: torch.Tensor  # (..., 3)
    rotation: torch.Tensor  # (..., 4) SO3 quaternion, body-local


@dataclass(frozen=True)
class HalfSpace:
    """``n · x + d >= 0`` half-space."""

    normal: torch.Tensor  # (..., 3)
    offset: torch.Tensor  # (...,)


@dataclass(frozen=True)
class Plane(HalfSpace):
    """Alias for ``HalfSpace`` used for world-ground checks."""


def colldist_from_sdf(d: torch.Tensor, margin: float) -> torch.Tensor:
    """Smooth one-sided penalty from a signed distance.

    ``d >= margin   → 0``
    ``0 <= d < m    → -0.5/m * (d - m)^2``
    ``d < 0         → d - 0.5 * m``

    See docs/09_COLLISION_GEOMETRY.md §7.
    """
    raise NotImplementedError("see docs/09_COLLISION_GEOMETRY.md §7")
