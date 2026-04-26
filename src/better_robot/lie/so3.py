"""SO3 group operations — pure functional facade over the active ``Backend``.

Storage convention: ``(..., 4)`` unit quaternion ``[qx, qy, qz, qw]``
(scalar-last). Tangent vectors are ``(..., 3)``.

See ``docs/concepts/lie_and_spatial.md §4``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..backends import default_backend
from . import _torch_native_backend as _be

if TYPE_CHECKING:
    from ..backends.protocol import Backend


def _lie(backend: "Backend | None"):
    return (backend or default_backend()).lie


def identity(
    *,
    batch_shape: tuple[int, ...] = (),
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return an SO3 identity quaternion with the given batch shape."""
    return _be.so3_identity(batch_shape, device, dtype)


def compose(
    a: torch.Tensor, b: torch.Tensor, *, backend: "Backend | None" = None,
) -> torch.Tensor:
    """SO3 composition. ``(..., 4), (..., 4) → (..., 4)``."""
    return _lie(backend).so3_compose(a, b)


def inverse(
    q: torch.Tensor, *, backend: "Backend | None" = None,
) -> torch.Tensor:
    """SO3 inverse (quaternion conjugate for unit quats)."""
    return _lie(backend).so3_inverse(q)


def log(
    q: torch.Tensor, *, backend: "Backend | None" = None,
) -> torch.Tensor:
    """SO3 → so3 tangent. ``(..., 4) → (..., 3)``."""
    return _lie(backend).so3_log(q)


def exp(
    w: torch.Tensor, *, backend: "Backend | None" = None,
) -> torch.Tensor:
    """so3 tangent → SO3. ``(..., 3) → (..., 4)``."""
    return _lie(backend).so3_exp(w)


def act(
    q: torch.Tensor, p: torch.Tensor, *, backend: "Backend | None" = None,
) -> torch.Tensor:
    """Rotate a point by a quaternion. ``q: (..., 4), p: (..., 3) → (..., 3)``."""
    return _lie(backend).so3_act(q, p)


def adjoint(
    q: torch.Tensor, *, backend: "Backend | None" = None,
) -> torch.Tensor:
    """3×3 adjoint of SO3 — equals the rotation matrix. ``(..., 4) → (..., 3, 3)``."""
    return _lie(backend).so3_to_matrix(q)


def from_matrix(
    R: torch.Tensor, *, backend: "Backend | None" = None,
) -> torch.Tensor:
    """Rotation matrix → unit quaternion. ``(..., 3, 3) → (..., 4)``."""
    return _lie(backend).so3_from_matrix(R)


def to_matrix(
    q: torch.Tensor, *, backend: "Backend | None" = None,
) -> torch.Tensor:
    """Unit quaternion → rotation matrix. ``(..., 4) → (..., 3, 3)``."""
    return _lie(backend).so3_to_matrix(q)


def from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Unit quaternion from axis-angle. axis: (...,3) unit, angle: (...,) → (...,4)."""
    return _be.so3_from_axis_angle(axis, angle)


def normalize(
    q: torch.Tensor, *, backend: "Backend | None" = None,
) -> torch.Tensor:
    """Re-normalize the quaternion to unit length."""
    return _lie(backend).so3_normalize(q)


def slerp(
    q1: torch.Tensor, q2: torch.Tensor, t: torch.Tensor | float,
) -> torch.Tensor:
    """SO3 spherical linear interpolation. ``(..., 4), (..., 4), (...) → (..., 4)``.

    Classical quaternion SLERP along the shortest arc. Falls back to
    normalized LERP when the two quaternions are nearly parallel to avoid
    a ``0/sin(0)`` division. ``t`` is broadcast against the leading batch
    of ``q1``/``q2`` (scalar or tensor accepted; ``t`` outside ``[0, 1]``
    extrapolates along the geodesic).
    """
    t = torch.as_tensor(t, dtype=q1.dtype, device=q1.device)
    while t.dim() < q1.dim():
        t = t.unsqueeze(-1)

    dot = (q1 * q2).sum(dim=-1, keepdim=True)
    q2 = torch.where(dot < 0, -q2, q2)
    dot = dot.abs()

    # Clamp before acos so the backward through the unselected branch stays finite.
    dot_safe = dot.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(dot_safe)
    sin_theta = torch.sin(theta).clamp(min=1e-10)

    w1_slerp = torch.sin((1.0 - t) * theta) / sin_theta
    w2_slerp = torch.sin(t * theta) / sin_theta
    w1_lerp = 1.0 - t
    w2_lerp = t

    near_parallel = dot > 1.0 - 1e-6
    w1 = torch.where(near_parallel, w1_lerp, w1_slerp)
    w2 = torch.where(near_parallel, w2_lerp, w2_slerp)

    out = w1 * q1 + w2 * q2
    return normalize(out)
