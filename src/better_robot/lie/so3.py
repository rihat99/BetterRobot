"""SO(3) group operations on ``(..., 4)`` unit quaternions ``[qx, qy, qz, qw]`` (scalar last).

Tangents are ``(..., 3)`` rotation vectors ``[wx, wy, wz]`` (axis times angle, radians). ``compose(a, b)`` equals
the matrix product ``R_a @ R_b``; ``inverse(q)`` is the conjugate. ``exp`` maps a rotation vector to a quaternion
and ``log`` returns the rotation vector with angle in ``[0, pi]``; ``q`` and ``-q`` give the same ``log``. Euler
helpers use ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` with input ``[roll, pitch, yaw]``. Leading batch axes broadcast;
dtype and device are preserved.
"""

from __future__ import annotations

import math

import torch


# ``1 - cos(theta)`` loses significance much earlier in fp32 than fp64.
# These cutoffs are on theta squared, not theta.
_TAYLOR_THETA2_FP32 = 1e-5
_TAYLOR_THETA2_FP64 = 1e-8


def _taylor_theta2(dtype: torch.dtype) -> float:
    """Return the small-angle ``theta²`` cutoff for ``dtype``.

    The dtype branch is static under ``torch.compile``. Lower-precision
    floating dtypes use the fp32 cutoff rather than the fp64 one.
    """
    return _TAYLOR_THETA2_FP64 if dtype == torch.float64 else _TAYLOR_THETA2_FP32


def _hat3(v: torch.Tensor) -> torch.Tensor:
    """Return the skew-symmetric ``3×3`` matrix for an ``(..., 3)`` vector."""
    z = torch.zeros_like(v[..., 0])
    return torch.stack(
        [
            torch.stack([z, -v[..., 2], v[..., 1]], dim=-1),
            torch.stack([v[..., 2], z, -v[..., 0]], dim=-1),
            torch.stack([-v[..., 1], v[..., 0], z], dim=-1),
        ],
        dim=-2,
    )


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return the Hamilton product of scalar-last quaternions ``a`` and ``b``."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    qx = aw * bx + ax * bw + ay * bz - az * by
    qy = aw * by - ax * bz + ay * bw + az * bx
    qz = aw * bz + ax * by - ay * bx + az * bw
    qw = aw * bw - ax * bx - ay * by - az * bz
    return torch.stack([qx, qy, qz, qw], dim=-1)


def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert scalar-last quaternions ``(..., 4)`` to rotation matrices."""
    qx, qy, qz, qw = q.unbind(-1)
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return torch.stack(
        [
            torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
            torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
            torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1),
        ],
        dim=-2,
    )


def _matrix_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to stable scalar-last unit quaternions.

    The Shepperd 1978 variant picks the largest diagonal component to avoid
    cancellation as ``qw → 0`` and keeps every branch differentiable.
    """
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    trace = m00 + m11 + m22

    s1 = (trace + 1.0).clamp(min=1e-30).sqrt() * 2.0
    qw1 = 0.25 * s1
    qx1 = (m21 - m12) / s1
    qy1 = (m02 - m20) / s1
    qz1 = (m10 - m01) / s1

    s2 = (1.0 + m00 - m11 - m22).clamp(min=1e-30).sqrt() * 2.0
    qw2 = (m21 - m12) / s2
    qx2 = 0.25 * s2
    qy2 = (m01 + m10) / s2
    qz2 = (m02 + m20) / s2

    s3 = (1.0 - m00 + m11 - m22).clamp(min=1e-30).sqrt() * 2.0
    qw3 = (m02 - m20) / s3
    qx3 = (m01 + m10) / s3
    qy3 = 0.25 * s3
    qz3 = (m12 + m21) / s3

    s4 = (1.0 - m00 - m11 + m22).clamp(min=1e-30).sqrt() * 2.0
    qw4 = (m10 - m01) / s4
    qx4 = (m02 + m20) / s4
    qy4 = (m12 + m21) / s4
    qz4 = 0.25 * s4

    cond1 = trace > 0
    cond2 = (m00 > m11) & (m00 > m22)
    cond3 = m11 > m22
    qw = torch.where(cond1, qw1, torch.where(cond2, qw2, torch.where(cond3, qw3, qw4)))
    qx = torch.where(cond1, qx1, torch.where(cond2, qx2, torch.where(cond3, qx3, qx4)))
    qy = torch.where(cond1, qy1, torch.where(cond2, qy2, torch.where(cond3, qy3, qy4)))
    qz = torch.where(cond1, qz1, torch.where(cond2, qz2, torch.where(cond3, qz3, qz4)))
    q = torch.stack([qx, qy, qz, qw], dim=-1)
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def identity(
    *,
    batch_shape: tuple[int, ...] = (),
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return an SO3 identity quaternion with the given batch shape."""
    out = torch.zeros((*batch_shape, 4), device=device, dtype=dtype)
    out[..., 3] = 1.0
    return out


def compose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SO3 composition. ``(..., 4), (..., 4) → (..., 4)``."""
    return _quat_mul(a, b)


def inverse(q: torch.Tensor) -> torch.Tensor:
    """SO3 inverse (quaternion conjugate for unit quats)."""
    return torch.cat([-q[..., :3], q[..., 3:]], dim=-1)


def log(q: torch.Tensor) -> torch.Tensor:
    """SO3 → so3 tangent. ``(..., 4) → (..., 3)``."""
    q = torch.where(q[..., 3:4] < 0, -q, q)
    qxyz = q[..., :3]
    qw = q[..., 3:4]
    sin_half2 = (qxyz * qxyz).sum(dim=-1, keepdim=True)
    use_taylor = sin_half2 < _taylor_theta2(q.dtype) / 4.0
    sin_half2_safe = torch.where(use_taylor, torch.ones_like(sin_half2), sin_half2)
    sin_half = sin_half2_safe.sqrt()

    theta = 2.0 * torch.atan2(sin_half, qw.clamp(min=-1.0, max=1.0))
    factor_full = theta / sin_half.clamp(min=1e-30)
    factor_taylor = 2.0 + sin_half2 * (2.0 / 3.0)
    factor = torch.where(use_taylor, factor_taylor, factor_full)
    return factor * qxyz


def exp(w: torch.Tensor) -> torch.Tensor:
    """so3 tangent → SO3. ``(..., 3) → (..., 4)``."""
    theta2 = (w * w).sum(dim=-1, keepdim=True)
    use_taylor = theta2 < _taylor_theta2(w.dtype)
    theta2_safe = torch.where(use_taylor, torch.ones_like(theta2), theta2)
    theta = theta2_safe.sqrt()
    half = theta / 2.0
    sin_half_over_theta_full = torch.sin(half) / theta.clamp(min=1e-30)
    sin_half_over_theta_taylor = 0.5 - theta2 / 48.0
    sin_half_over_theta = torch.where(use_taylor, sin_half_over_theta_taylor, sin_half_over_theta_full)
    qxyz = sin_half_over_theta * w
    qw = torch.where(use_taylor, 1.0 - theta2 / 8.0, torch.cos(half))
    return torch.cat([qxyz, qw], dim=-1)


def act(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Rotate a point by a quaternion. ``q: (..., 4), p: (..., 3) → (..., 3)``."""
    qxyz = q[..., :3]
    qw = q[..., 3:4]
    qxyz_b, p_b = torch.broadcast_tensors(qxyz, p)
    qw_b = qw.expand(p_b.shape[:-1] + (1,))
    cross1 = torch.linalg.cross(qxyz_b, p_b, dim=-1) + qw_b * p_b
    return p_b + 2.0 * torch.linalg.cross(qxyz_b, cross1, dim=-1)


def adjoint(q: torch.Tensor) -> torch.Tensor:
    """3×3 adjoint of SO3 — equals the rotation matrix. ``(..., 4) → (..., 3, 3)``."""
    return _quat_to_matrix(q)


def from_matrix(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix → unit quaternion. ``(..., 3, 3) → (..., 4)``."""
    return _matrix_to_quat(R)


def to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion → rotation matrix. ``(..., 4) → (..., 3, 3)``."""
    return _quat_to_matrix(q)


def from_euler(euler: torch.Tensor) -> torch.Tensor:
    """Convert extrinsic XYZ Euler angles to a unit quaternion.

    ``euler[..., :]`` is ``[roll, pitch, yaw]`` in radians.  The rotations
    are active and extrinsic XYZ (equivalently intrinsic ZYX), so the
    resulting matrix is ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.  The returned
    quaternion is scalar-last ``[qx, qy, qz, qw]``.

    This is the fixed library convention; the function deliberately does
    not accept other axis orders.  See ``to_euler`` for the inverse and
    its gimbal-lock behaviour.
    """
    if not isinstance(euler, torch.Tensor) or not euler.is_floating_point():
        raise TypeError("euler must be a floating torch.Tensor")
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
    convention as ``from_euler``: ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
    The principal branch has roll/yaw in ``[-pi, pi]`` and pitch in
    ``[-pi/2, pi/2]``.  At pitch ``+/- pi/2`` (gimbal lock) roll and yaw are
    not individually identifiable; there the result has yaw ``0``, pitch
    exactly ``+/- pi/2`` and the combined angle in roll, so the round trip
    through ``from_euler`` is exact.
    """
    if not isinstance(q, torch.Tensor) or not q.is_floating_point():
        raise TypeError("q must be a floating torch.Tensor")
    if q.shape[-1:] != (4,):
        raise ValueError(f"q must have shape (..., 4); got {tuple(q.shape)}")

    q = normalize(q)
    qx, qy, qz, qw = q.unbind(dim=-1)
    sin_pitch = 2.0 * (qw * qy - qz * qx)
    roll_sin, roll_cos = 2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx.square() + qy.square())
    # The regular formulas read atan2(~0, ~0) when cos(pitch) = hypot(roll_sin, roll_cos) is
    # rounding-sized; the band is set in angle (sqrt(eps) rad) because a sine band of a few
    # ulps is ~1e-3 rad in float32. Masked inputs keep both branches' gradients finite
    # (C-DIFF-3) and leave every other input bit-identical.
    lock = torch.hypot(roll_sin, roll_cos) < math.sqrt(torch.finfo(q.dtype).eps)
    zero, one = torch.zeros_like(qx), torch.ones_like(qx)
    roll = torch.atan2(torch.where(lock, zero, roll_sin), torch.where(lock, one, roll_cos))
    pitch = torch.asin(torch.where(lock, zero, sin_pitch).clamp(-1.0, 1.0))
    yaw = torch.atan2(
        torch.where(lock, zero, 2.0 * (qw * qz + qx * qy)),
        torch.where(lock, one, 1.0 - 2.0 * (qy.square() + qz.square())),
    )
    # At the lock q = +/-(qy(+/-pi/2) * qx(roll - +/-yaw)), so tan(combined / 2) = qx / qw for both signs.
    lock_qx, lock_qw = torch.where(lock, qx, zero), torch.where(lock, qw, one)
    lock_roll = torch.atan2(2.0 * lock_qx * lock_qw, lock_qw.square() - lock_qx.square())
    roll = torch.where(lock, lock_roll, roll)
    pitch = torch.where(lock, torch.copysign(torch.full_like(pitch, math.pi / 2), sin_pitch), pitch)
    return torch.stack((roll, pitch, yaw), dim=-1)


def from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Unit quaternion from axis-angle. axis: (...,3) unit, angle: (...,) → (...,4)."""
    # Python ``2.0`` promotes a forward-mode fp32 tangent to fp64 on Torch
    # 2.13, producing mixed dual tensors in FK under ``jacfwd``.
    half = angle * torch.full_like(angle, 0.5)
    sin_h = torch.sin(half)
    cos_h = torch.cos(half)
    qxyz = sin_h.unsqueeze(-1) * axis
    qw = cos_h.unsqueeze(-1)
    return torch.cat([qxyz, qw], dim=-1)


def normalize(q: torch.Tensor) -> torch.Tensor:
    """Re-normalize the quaternion to unit length."""
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def slerp(
    q1: torch.Tensor,
    q2: torch.Tensor,
    t: torch.Tensor | float,
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
