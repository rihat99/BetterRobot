"""SO3 group operations — pure functional facade over the torch implementation.

Storage convention: ``(..., 4)`` unit quaternion ``[qx, qy, qz, qw]``
(scalar-last). Tangent vectors are ``(..., 3)``.

See ``docs/concepts/lie_and_spatial.md §4``.
"""

from __future__ import annotations

import torch

from . import _impl


def identity(
    *,
    batch_shape: tuple[int, ...] = (),
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return an SO3 identity quaternion with the given batch shape."""
    return _impl.so3_identity(batch_shape, device, dtype)


def compose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SO3 composition. ``(..., 4), (..., 4) → (..., 4)``."""
    return _impl.so3_compose(a, b)


def inverse(q: torch.Tensor) -> torch.Tensor:
    """SO3 inverse (quaternion conjugate for unit quats)."""
    return _impl.so3_inverse(q)


def log(q: torch.Tensor) -> torch.Tensor:
    """SO3 → so3 tangent. ``(..., 4) → (..., 3)``."""
    return _impl.so3_log(q)


def exp(w: torch.Tensor) -> torch.Tensor:
    """so3 tangent → SO3. ``(..., 3) → (..., 4)``."""
    return _impl.so3_exp(w)


def act(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Rotate a point by a quaternion. ``q: (..., 4), p: (..., 3) → (..., 3)``."""
    return _impl.so3_act(q, p)


def adjoint(q: torch.Tensor) -> torch.Tensor:
    """3×3 adjoint of SO3 — equals the rotation matrix. ``(..., 4) → (..., 3, 3)``."""
    return _impl.so3_to_matrix(q)


def from_matrix(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix → unit quaternion. ``(..., 3, 3) → (..., 4)``."""
    return _impl.so3_from_matrix(R)


def to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion → rotation matrix. ``(..., 4) → (..., 3, 3)``."""
    return _impl.so3_to_matrix(q)


def from_euler(euler: torch.Tensor) -> torch.Tensor:
    """Convert extrinsic XYZ Euler angles to a unit quaternion.

    ``euler[..., :]`` is ``[roll, pitch, yaw]`` in radians.  The rotations
    are active and extrinsic XYZ (equivalently intrinsic ZYX), so the
    resulting matrix is ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.  The returned
    quaternion is scalar-last ``[qx, qy, qz, qw]``.

    This is the fixed library convention; the function deliberately does
    not accept other axis orders.  See :func:`to_euler` for the inverse and
    its gimbal-lock behaviour.
    """
    if euler.shape[-1:] != (3,):
        raise ValueError(f"euler must have shape (..., 3); got {tuple(euler.shape)}")

    half = euler * torch.full_like(euler, 0.5)
    roll, pitch, yaw = half.unbind(dim=-1)
    sr, cr = torch.sin(roll), torch.cos(roll)
    sp, cp = torch.sin(pitch), torch.cos(pitch)
    sy, cy = torch.sin(yaw), torch.cos(yaw)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return torch.stack((qx, qy, qz, qw), dim=-1)


def to_euler(q: torch.Tensor) -> torch.Tensor:
    """Convert a unit quaternion to extrinsic XYZ Euler angles.

    Returns ``[..., (roll, pitch, yaw)]`` in radians for the same active
    convention as :func:`from_euler`: ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
    The principal branch has roll/yaw in ``[-pi, pi]`` and pitch in
    ``[-pi/2, pi/2]``.  As with every Euler representation, roll and yaw
    are not individually identifiable at pitch ``+/- pi/2``.
    """
    if q.shape[-1:] != (4,):
        raise ValueError(f"q must have shape (..., 4); got {tuple(q.shape)}")

    q = normalize(q)
    qx, qy, qz, qw = q.unbind(dim=-1)
    roll = torch.atan2(
        2.0 * (qw * qx + qy * qz),
        1.0 - 2.0 * (qx.square() + qy.square()),
    )
    sin_pitch = 2.0 * (qw * qy - qz * qx)
    pitch = torch.asin(sin_pitch.clamp(-1.0, 1.0))
    yaw = torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy.square() + qz.square()),
    )
    return torch.stack((roll, pitch, yaw), dim=-1)


def from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Unit quaternion from axis-angle. axis: (...,3) unit, angle: (...,) → (...,4)."""
    return _impl.so3_from_axis_angle(axis, angle)


def normalize(q: torch.Tensor) -> torch.Tensor:
    """Re-normalize the quaternion to unit length."""
    return _impl.so3_normalize(q)


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
