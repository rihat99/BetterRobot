"""Configuration manifolds: how a joint's coordinates step along a tangent and how two configurations differ.

Each `JointManifold` is the pair ``integrate`` (the retraction ``q ⊕ v``) and ``difference`` (its inverse ``q1 ⊖ q0``)
on coordinates of fixed width ``nq`` and tangents of width ``nv``, written once here and shared by every joint kind
on that manifold (record 0013). Every map is the true group exponential or logarithm of its group, so the tangent is
the twist of the joint's motion subspace: ``d/dt placement(integrate(q, t v))`` at ``t = 0`` equals ``S(q) v``.

- ``FIXED``: no coordinate at all (fixed joints); ``integrate`` and ``difference`` return the empty slice.
- ``EUCLIDEAN``: one real coordinate, addition (revolute, prismatic, helical and translation joints).
- ``CIRCLE``: ``[cos θ, sin θ]`` with the angle step ``dθ`` (unbounded revolute joints); the pair is renormalised.
- ``ROTATION``: a unit quaternion ``[qx, qy, qz, qw]`` with an SO(3) tangent (spherical joints).
- ``PLANE``: ``[x, y, cos θ, sin θ]`` with the SE(2) body-frame twist ``[vx, vy, dθ]`` (planar joints).
- ``RIGID``: a pose ``[tx, ty, tz, qx, qy, qz, qw]`` with the SE(3) body-frame twist (free-flyer joints).

The functions broadcast over leading axes and keep the dtype and device of their inputs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import final

import torch

from ..lie import se3, so3

Map = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@final
@dataclass(frozen=True, kw_only=True, repr=False)
class JointManifold:
    """``integrate(q, v)`` steps ``q: (..., nq)`` along ``v: (..., nv)``; ``difference(q0, q1)`` is the tangent
    ``(..., nv)`` with ``integrate(q0, difference(q0, q1)) == q1``."""

    name: str
    nq: int
    nv: int
    integrate: Map
    difference: Map

    def __repr__(self) -> str:
        return f"JointManifold({self.name!r}, nq={self.nq}, nv={self.nv})"


def _euclidean_integrate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return q + v


def _euclidean_difference(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    return q1 - q0


def _so2_compose(cos_t: torch.Tensor, sin_t: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """``[cos, sin]`` of ``θ + angle`` from ``[cos θ, sin θ]``, renormalised so rounding never leaves the circle."""
    cos_d, sin_d = torch.cos(angle), torch.sin(angle)
    pair = torch.stack((cos_t * cos_d - sin_t * sin_d, sin_t * cos_d + cos_t * sin_d), dim=-1)
    return pair / pair.norm(dim=-1, keepdim=True)


def _so2_angle(cos_0: torch.Tensor, sin_0: torch.Tensor, cos_1: torch.Tensor, sin_1: torch.Tensor) -> torch.Tensor:
    """The wrapped angle from ``[cos θ0, sin θ0]`` to ``[cos θ1, sin θ1]``, in ``(-π, π]``."""
    return torch.atan2(cos_0 * sin_1 - sin_0 * cos_1, cos_0 * cos_1 + sin_0 * sin_1)


def _circle_integrate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _so2_compose(q[..., 0], q[..., 1], v[..., 0])


def _circle_difference(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    return _so2_angle(q0[..., 0], q0[..., 1], q1[..., 0], q1[..., 1]).unsqueeze(-1)


def _rotation_integrate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return so3.normalize_raw(so3.compose_raw(q, so3.exp_raw(v)))


def _rotation_difference(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    return so3.log_raw(so3.normalize_raw(so3.compose_raw(so3.inverse_raw(q0), q1)))


def _se2_coefficients(angle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``a = sin θ / θ`` and ``b = (1 - cos θ) / θ``, the entries of the SE(2) left Jacobian ``[[a, -b], [b, a]]``.

    Below the threshold the Taylor series keeps the derivative finite at ``θ = 0``.
    """
    small = angle.abs() < torch.finfo(angle.dtype).eps ** 0.25
    safe = torch.where(small, torch.ones_like(angle), angle)
    a = torch.where(small, 1.0 - angle * angle / 6.0, torch.sin(safe) / safe)
    b = torch.where(small, angle / 2.0 - angle * angle * angle / 24.0, (1.0 - torch.cos(safe)) / safe)
    return a, b


def _plane_integrate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    x, y, cos_t, sin_t = q.unbind(-1)
    vx, vy, angle = v.unbind(-1)
    a, b = _se2_coefficients(angle)
    body_x, body_y = a * vx - b * vy, b * vx + a * vy  # the exponential's translation, in the joint frame
    translation = torch.stack((x + cos_t * body_x - sin_t * body_y, y + sin_t * body_x + cos_t * body_y), dim=-1)
    return torch.cat((translation, _so2_compose(cos_t, sin_t, angle)), dim=-1)


def _plane_difference(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    x0, y0, cos_0, sin_0 = q0.unbind(-1)
    x1, y1, cos_1, sin_1 = q1.unbind(-1)
    angle = _so2_angle(cos_0, sin_0, cos_1, sin_1)
    dx, dy = x1 - x0, y1 - y0
    body_x, body_y = cos_0 * dx + sin_0 * dy, cos_0 * dy - sin_0 * dx  # the displacement in the frame of q0
    a, b = _se2_coefficients(angle)
    det = a * a + b * b
    return torch.stack(((a * body_x + b * body_y) / det, (a * body_y - b * body_x) / det, angle), dim=-1)


def _rigid_integrate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return se3.normalize_raw(se3.compose_raw(q, se3.exp_raw(v)))


def _rigid_difference(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    return se3.log_raw(se3.compose_raw(se3.inverse_raw(q0), q1))


FIXED = JointManifold(name="fixed", nq=0, nv=0, integrate=_euclidean_integrate, difference=_euclidean_difference)
EUCLIDEAN = JointManifold(
    name="euclidean", nq=1, nv=1, integrate=_euclidean_integrate, difference=_euclidean_difference
)
CIRCLE = JointManifold(name="circle", nq=2, nv=1, integrate=_circle_integrate, difference=_circle_difference)
ROTATION = JointManifold(name="rotation", nq=4, nv=3, integrate=_rotation_integrate, difference=_rotation_difference)
PLANE = JointManifold(name="plane", nq=4, nv=3, integrate=_plane_integrate, difference=_plane_difference)
RIGID = JointManifold(name="rigid", nq=7, nv=6, integrate=_rigid_integrate, difference=_rigid_difference)

__all__ = ["CIRCLE", "EUCLIDEAN", "FIXED", "PLANE", "RIGID", "ROTATION", "JointManifold"]
