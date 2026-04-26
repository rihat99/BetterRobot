"""Pure-PyTorch SE3/SO3 backend — the only Lie implementation.

The legacy PyPose backend was removed in P10-D; this module is now the
sole owner of the SE3/SO3 surface used by ``lie/se3.py`` and ``lie/so3.py``.
The implementation follows ``docs/concepts/lie_and_spatial.md §6``:

* Quaternion convention ``[qx, qy, qz, qw]`` (scalar last).
* SE3 storage ``[tx, ty, tz, qx, qy, qz, qw]`` (7-vector).
* se3 tangent ``[vx, vy, vz, wx, wy, wz]`` (6-vector, linear first).

Singularity handling uses the Taylor expansions

  ``b = (1 − cos θ)/θ²    ≈ 1/2 − θ²/24    + O(θ⁴)``
  ``c = (θ − sin θ)/θ³    ≈ 1/6 − θ²/120  + O(θ⁴)``

stitched in via ``torch.where`` against a ``θ²`` cutoff so the autograd
graph stays smooth across ``θ = 0``.
"""

from __future__ import annotations

import torch

# Cutoffs below which we use the Taylor expansion. Both cover the dtype
# precision range (`fp32 ≈ 1e-7`, `fp64 ≈ 1e-14`).
_TAYLOR_THETA2 = 1e-8


# ─────────────────────────────── helpers ───────────────────────────────

def _hat3(v: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric ``3×3`` from ``(..., 3)`` axis vector."""
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
    """Hamilton product ``a ⊗ b``; both ``[qx, qy, qz, qw]``."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    qx = aw * bx + ax * bw + ay * bz - az * by
    qy = aw * by - ax * bz + ay * bw + az * bx
    qz = aw * bz + ax * by - ay * bx + az * bw
    qw = aw * bw - ax * bx - ay * by - az * bz
    return torch.stack([qx, qy, qz, qw], dim=-1)


def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """``(..., 4)`` ``[qx, qy, qz, qw]`` → ``(..., 3, 3)`` rotation matrix."""
    qx, qy, qz, qw = q.unbind(-1)
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    R = torch.stack(
        [
            torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
            torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
            torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1),
        ],
        dim=-2,
    )
    return R


def _matrix_to_quat(R: torch.Tensor) -> torch.Tensor:
    """``(..., 3, 3)`` rotation matrix → unit quaternion ``[qx, qy, qz, qw]``.

    Numerically stable variant (Shepperd 1978) that picks the largest
    diagonal component to avoid the ``qw → 0`` cancellation. Fully
    differentiable via ``torch.where`` branches.
    """
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    trace = m00 + m11 + m22

    # Branch 1: trace > 0.
    s1 = (trace + 1.0).clamp(min=1e-30).sqrt() * 2.0  # 4 qw
    qw1 = 0.25 * s1
    qx1 = (m21 - m12) / s1
    qy1 = (m02 - m20) / s1
    qz1 = (m10 - m01) / s1

    # Branch 2: m00 largest.
    s2 = (1.0 + m00 - m11 - m22).clamp(min=1e-30).sqrt() * 2.0
    qw2 = (m21 - m12) / s2
    qx2 = 0.25 * s2
    qy2 = (m01 + m10) / s2
    qz2 = (m02 + m20) / s2

    # Branch 3: m11 largest.
    s3 = (1.0 - m00 + m11 - m22).clamp(min=1e-30).sqrt() * 2.0
    qw3 = (m02 - m20) / s3
    qx3 = (m01 + m10) / s3
    qy3 = 0.25 * s3
    qz3 = (m12 + m21) / s3

    # Branch 4: m22 largest.
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


# ──────────────────────────────── SO3 ────────────────────────────────


def so3_identity(batch_shape: tuple[int, ...], device, dtype) -> torch.Tensor:
    out = torch.zeros((*batch_shape, 4), device=device, dtype=dtype)
    out[..., 3] = 1.0
    return out


def so3_compose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _quat_mul(a, b)


def so3_inverse(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a unit quaternion."""
    sign = torch.tensor([-1.0, -1.0, -1.0, 1.0], dtype=q.dtype, device=q.device)
    return q * sign


def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """``(..., 3)`` axis-angle → ``(..., 4)`` quaternion ``[qx, qy, qz, qw]``."""
    theta2 = (omega * omega).sum(dim=-1, keepdim=True)
    theta = theta2.clamp(min=0.0).sqrt()
    half = theta / 2.0
    # ``sin(θ/2)/θ`` factor with a Taylor-stitch at small θ.
    use_taylor = theta2 < _TAYLOR_THETA2
    sin_half_over_theta_full = torch.sin(half) / theta.clamp(min=1e-30)
    # Taylor: sin(θ/2)/θ = 1/2 − θ²/48 + O(θ⁴).
    sin_half_over_theta_taylor = 0.5 - theta2 / 48.0
    sin_half_over_theta = torch.where(
        use_taylor, sin_half_over_theta_taylor, sin_half_over_theta_full
    )
    qxyz = sin_half_over_theta * omega
    qw = torch.cos(half)
    return torch.cat([qxyz, qw], dim=-1)


def so3_log(q: torch.Tensor) -> torch.Tensor:
    """``(..., 4)`` quaternion → ``(..., 3)`` axis-angle ``ω`` such that ``exp(ω) = q``.

    Folds ``q`` and ``-q`` (double-cover) by flipping the sign when ``qw < 0``.
    """
    qw_neg = q[..., 3:4] < 0
    q = torch.where(qw_neg, -q, q)

    qxyz = q[..., :3]
    qw = q[..., 3:4]
    sin_half2 = (qxyz * qxyz).sum(dim=-1, keepdim=True)
    sin_half = sin_half2.clamp(min=0.0).sqrt()

    # θ = 2·atan2(|qxyz|, qw); then ω = (θ/sin(θ/2))·qxyz.
    theta = 2.0 * torch.atan2(sin_half, qw.clamp(min=-1.0, max=1.0))
    use_taylor = sin_half2 < _TAYLOR_THETA2 / 4.0  # since sin(θ/2)² ~ θ²/4
    factor_full = theta / sin_half.clamp(min=1e-30)
    # Taylor: θ/sin(θ/2) ≈ 2 + θ²/12 + O(θ⁴), and ω = qxyz · (θ/sin(θ/2));
    # for small θ, qxyz ≈ ω/2, so ω ≈ 2·qxyz · (1 + sin_half²·…).
    factor_taylor = 2.0 + sin_half2 * (2.0 / 3.0)
    factor = torch.where(use_taylor, factor_taylor, factor_full)
    return factor * qxyz


def so3_act(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Rotate ``p`` by quaternion ``q``: ``p' = q ⊗ [p, 0] ⊗ q*``.

    Broadcasts ``q`` against ``p`` to a common shape because
    ``torch.linalg.cross`` requires equal ``ndim`` (does not broadcast).
    """
    qxyz = q[..., :3]
    qw = q[..., 3:4]
    qxyz_b, p_b = torch.broadcast_tensors(qxyz, p)
    qw_b = qw.expand(p_b.shape[:-1] + (1,))
    # Standard quaternion-vector rotation, vectorised:
    #   p' = p + 2·qxyz × (qxyz × p + qw·p)
    cross1 = torch.linalg.cross(qxyz_b, p_b, dim=-1) + qw_b * p_b
    return p_b + 2.0 * torch.linalg.cross(qxyz_b, cross1, dim=-1)


def so3_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def so3_from_matrix(R: torch.Tensor) -> torch.Tensor:
    return _matrix_to_quat(R)


def so3_to_matrix(q: torch.Tensor) -> torch.Tensor:
    return _quat_to_matrix(q)


def so3_adjoint(q: torch.Tensor) -> torch.Tensor:
    """SO3 adjoint = rotation matrix."""
    return _quat_to_matrix(q)


def so3_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    half = angle / 2.0
    sin_h = torch.sin(half)
    cos_h = torch.cos(half)
    qxyz = sin_h.unsqueeze(-1) * axis
    qw = cos_h.unsqueeze(-1)
    return torch.cat([qxyz, qw], dim=-1)


# ──────────────────────────────── SE3 ────────────────────────────────


def se3_identity(batch_shape: tuple[int, ...], device, dtype) -> torch.Tensor:
    out = torch.zeros((*batch_shape, 7), device=device, dtype=dtype)
    out[..., 6] = 1.0
    return out


def se3_compose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``c = a · b``: ``c.t = a.t + R(a.q) · b.t``, ``c.q = a.q ⊗ b.q``."""
    a_t = a[..., :3]
    a_q = a[..., 3:7]
    b_t = b[..., :3]
    b_q = b[..., 3:7]
    rotated = so3_act(a_q, b_t)
    c_t = a_t + rotated
    c_q = _quat_mul(a_q, b_q)
    return torch.cat([c_t, c_q], dim=-1)


def se3_inverse(t: torch.Tensor) -> torch.Tensor:
    """``T⁻¹``: ``q⁻¹ = q*``, ``t⁻¹ = -R(q*) · t``."""
    q = t[..., 3:7]
    t_lin = t[..., :3]
    q_inv = so3_inverse(q)
    t_inv = -so3_act(q_inv, t_lin)
    return torch.cat([t_inv, q_inv], dim=-1)


def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    """``ξ = [v, ω]`` (linear first) → ``T ∈ SE3``.

    Uses ``T = (V(ω)·v, exp(ω))`` with
    ``V(ω) = I + b·hat(ω) + c·hat(ω)²``,
    ``b = (1 − cos θ)/θ²``, ``c = (θ − sin θ)/θ³`` and Taylor stitches
    at ``θ → 0``.
    """
    v = xi[..., :3]
    omega = xi[..., 3:6]
    theta2 = (omega * omega).sum(dim=-1, keepdim=True)
    theta = theta2.clamp(min=0.0).sqrt()
    use_taylor = theta2 < _TAYLOR_THETA2

    b_full = (1.0 - torch.cos(theta)) / theta2.clamp(min=1e-30)
    b_taylor = 0.5 - theta2 / 24.0
    b = torch.where(use_taylor, b_taylor, b_full)

    c_full = (theta - torch.sin(theta)) / (theta * theta2).clamp(min=1e-30)
    c_taylor = (1.0 / 6.0) - theta2 / 120.0
    c = torch.where(use_taylor, c_taylor, c_full)

    W = _hat3(omega)
    W2 = W @ W
    eye3 = torch.eye(3, dtype=xi.dtype, device=xi.device).expand_as(W)
    V = eye3 + b.unsqueeze(-1) * W + c.unsqueeze(-1) * W2  # (..., 3, 3)
    t_lin = (V @ v.unsqueeze(-1)).squeeze(-1)
    q = so3_exp(omega)
    return torch.cat([t_lin, q], dim=-1)


def se3_log(t: torch.Tensor) -> torch.Tensor:
    """SE3 → ``ξ = [v, ω]``. ``v = V(ω)⁻¹ · t_lin``."""
    t_lin = t[..., :3]
    q = t[..., 3:7]
    omega = so3_log(q)  # (..., 3)
    theta2 = (omega * omega).sum(dim=-1, keepdim=True)
    theta = theta2.clamp(min=0.0).sqrt()
    use_taylor = theta2 < _TAYLOR_THETA2

    # V(ω)⁻¹ = I − ½·W + (1/θ² − (1+cos θ)/(2θ·sin θ)) · W²
    half_theta = theta / 2.0
    cot_half = torch.cos(half_theta) / torch.sin(half_theta).clamp(min=1e-30)
    coeff_full = (1.0 / theta2.clamp(min=1e-30)) - cot_half / (2.0 * theta.clamp(min=1e-30))
    coeff_taylor = (1.0 / 12.0) + theta2 / 720.0
    coeff = torch.where(use_taylor, coeff_taylor, coeff_full)

    W = _hat3(omega)
    W2 = W @ W
    eye3 = torch.eye(3, dtype=t.dtype, device=t.device).expand_as(W)
    V_inv = eye3 - 0.5 * W + coeff.unsqueeze(-1) * W2
    v = (V_inv @ t_lin.unsqueeze(-1)).squeeze(-1)
    return torch.cat([v, omega], dim=-1)


def se3_act(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Apply ``T`` to a 3-point: ``p' = R(q)·p + t_lin``."""
    return so3_act(t[..., 3:7], p) + t[..., :3]


def se3_adjoint(t: torch.Tensor) -> torch.Tensor:
    """``Ad(T) = [[R, hat(p)·R]; [0, R]]`` — ``(..., 6, 6)``."""
    p = t[..., :3]
    R = _quat_to_matrix(t[..., 3:7])
    skew = _hat3(p)
    pR = skew @ R
    *batch, _, _ = R.shape
    zeros33 = torch.zeros(*batch, 3, 3, dtype=t.dtype, device=t.device)
    top = torch.cat([R, pR], dim=-1)
    bottom = torch.cat([zeros33, R], dim=-1)
    return torch.cat([top, bottom], dim=-2)


def se3_adjoint_inv(t: torch.Tensor) -> torch.Tensor:
    """``Ad(T⁻¹) = [[R^T, -R^T·hat(p)]; [0, R^T]]``."""
    p = t[..., :3]
    R = _quat_to_matrix(t[..., 3:7])
    RT = R.transpose(-1, -2)
    skew = _hat3(p)
    neg_RT_skew = -(RT @ skew)
    *batch, _, _ = R.shape
    zeros33 = torch.zeros(*batch, 3, 3, dtype=t.dtype, device=t.device)
    top = torch.cat([RT, neg_RT_skew], dim=-1)
    bottom = torch.cat([zeros33, RT], dim=-1)
    return torch.cat([top, bottom], dim=-2)


def se3_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    q = so3_from_axis_angle(axis, angle)
    zeros = torch.zeros(*angle.shape, 3, device=angle.device, dtype=angle.dtype)
    return torch.cat([zeros, q], dim=-1)


def se3_from_translation_axis(axis: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
    trans = disp.unsqueeze(-1) * axis.to(dtype=disp.dtype, device=disp.device)
    qxyz = torch.zeros(*disp.shape, 3, device=disp.device, dtype=disp.dtype)
    qw = torch.ones(*disp.shape, 1, device=disp.device, dtype=disp.dtype)
    return torch.cat([trans, qxyz, qw], dim=-1)


def se3_normalize(t: torch.Tensor) -> torch.Tensor:
    q = t[..., 3:7]
    q_normed = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return torch.cat([t[..., :3], q_normed], dim=-1)


def se3_apply_base(base: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
    base_expanded = base.unsqueeze(-2)  # (..., 1, 7)
    return se3_compose(base_expanded.expand(*poses.shape[:-1], 7), poses)
