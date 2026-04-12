"""SO3 group operations — pure functional facade over the backend.

Storage convention: ``(..., 4)`` unit quaternion ``[qx, qy, qz, qw]``
(scalar-last). Tangent vectors are ``(..., 3)``.

See ``docs/03_LIE_AND_SPATIAL.md §4``.
"""

from __future__ import annotations

import torch

from . import _pypose_backend as _pp


def identity(
    *,
    batch_shape: tuple[int, ...] = (),
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return an SO3 identity quaternion with the given batch shape."""
    return _pp.so3_identity(batch_shape, device, dtype)


def compose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SO3 composition. ``(..., 4), (..., 4) → (..., 4)``."""
    return _pp.so3_compose(a, b)


def inverse(q: torch.Tensor) -> torch.Tensor:
    """SO3 inverse (quaternion conjugate for unit quats)."""
    return _pp.so3_inverse(q)


def log(q: torch.Tensor) -> torch.Tensor:
    """SO3 → so3 tangent. ``(..., 4) → (..., 3)``."""
    return _pp.so3_log(q)


def exp(w: torch.Tensor) -> torch.Tensor:
    """so3 tangent → SO3. ``(..., 3) → (..., 4)``."""
    return _pp.so3_exp(w)


def act(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Rotate a point by a quaternion. ``q: (..., 4), p: (..., 3) → (..., 3)``."""
    return _pp.so3_act(q, p)


def adjoint(q: torch.Tensor) -> torch.Tensor:
    """3×3 adjoint of SO3 — equals the rotation matrix. ``(..., 4) → (..., 3, 3)``."""
    return _pp.so3_adjoint(q)


def from_matrix(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix → unit quaternion. ``(..., 3, 3) → (..., 4)``."""
    return _pp.so3_from_matrix(R)


def to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion → rotation matrix. ``(..., 4) → (..., 3, 3)``."""
    return _pp.so3_to_matrix(q)


def from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Unit quaternion from axis-angle. axis: (...,3) unit, angle: (...,) → (...,4)."""
    return _pp.so3_from_axis_angle(axis, angle)


def normalize(q: torch.Tensor) -> torch.Tensor:
    """Re-normalize the quaternion to unit length."""
    return _pp.so3_normalize(q)
