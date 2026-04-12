"""SE3 group operations — pure functional facade over the backend.

Storage convention: ``(..., 7)`` tensor ``[tx, ty, tz, qx, qy, qz, qw]``
(scalar-last quaternion, PyPose-native). Tangent vectors are ``(..., 6)``
``[vx, vy, vz, wx, wy, wz]``.

Everything is batched — any leading shape ``B...`` is supported by
broadcasting.

See ``docs/03_LIE_AND_SPATIAL.md §3`` for the full rationale.
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
    """Return an SE3 identity with the given leading batch shape."""
    return _pp.se3_identity(batch_shape, device, dtype)


def compose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SE3 composition. ``a: (..., 7), b: (..., 7) → (..., 7)``."""
    return _pp.se3_compose(a, b)


def inverse(t: torch.Tensor) -> torch.Tensor:
    """SE3 inverse. ``(..., 7) → (..., 7)``."""
    return _pp.se3_inverse(t)


def log(t: torch.Tensor) -> torch.Tensor:
    """SE3 → se3 tangent. ``(..., 7) → (..., 6)``."""
    return _pp.se3_log(t)


def exp(v: torch.Tensor) -> torch.Tensor:
    """se3 tangent → SE3. ``(..., 6) → (..., 7)``."""
    return _pp.se3_exp(v)


def act(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Apply SE3 to a point. ``t: (..., 7), p: (..., 3) → (..., 3)``."""
    return _pp.se3_act(t, p)


def adjoint(t: torch.Tensor) -> torch.Tensor:
    """6x6 adjoint matrix. ``(..., 7) → (..., 6, 6)``.

    Ad(T) = [[R, hat(p)@R], [0, R]] where p=translation, R=rotation.
    """
    return _pp.se3_adjoint(t)


def adjoint_inv(t: torch.Tensor) -> torch.Tensor:
    """Inverse adjoint ``Ad(T^-1)`` — faster than inverting the adjoint.

    Ad(T^{-1}) = [[R^T, -(R^T @ hat(p))], [0, R^T]].
    """
    return _pp.se3_adjoint_inv(t)


def from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Pure-rotation SE3 from axis-angle. axis: (...,3), angle: (...,) → (...,7)."""
    return _pp.se3_from_axis_angle(axis, angle)


def from_translation(axis: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
    """Pure-translation SE3 along ``axis`` with scalar displacement ``disp``.

    axis: (3,), disp: (...,) → (..., 7).
    """
    return _pp.se3_from_translation_axis(axis, disp)


def normalize(t: torch.Tensor) -> torch.Tensor:
    """Re-normalize the quaternion part to project back onto SE3."""
    return _pp.se3_normalize(t)


def apply_base(base: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
    """Compose a base transform with ``(..., N, 7)`` link poses."""
    return _pp.se3_apply_base(base, poses)
