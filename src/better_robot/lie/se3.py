"""SE3 group operations implemented directly with PyTorch tensors.

Storage convention: ``(..., 7)`` tensor ``[tx, ty, tz, qx, qy, qz, qw]``
(scalar-last quaternion). Tangent vectors are ``(..., 6)``
``[vx, vy, vz, wx, wy, wz]``.

See ``docs/concepts/lie_and_spatial.md §3``.
"""

from __future__ import annotations

import torch

from . import so3
from .so3 import _hat3, _quat_mul, _quat_to_matrix, _taylor_theta2


def identity(
    *,
    batch_shape: tuple[int, ...] = (),
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return an SE3 identity with the given leading batch shape."""
    out = torch.zeros((*batch_shape, 7), device=device, dtype=dtype)
    out[..., 6] = 1.0
    return out


def compose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SE3 composition. ``a: (..., 7), b: (..., 7) → (..., 7)``."""
    a_t = a[..., :3]
    a_q = a[..., 3:7]
    b_t = b[..., :3]
    b_q = b[..., 3:7]
    c_t = a_t + so3.act(a_q, b_t)
    c_q = _quat_mul(a_q, b_q)
    return torch.cat([c_t, c_q], dim=-1)


def inverse(t: torch.Tensor) -> torch.Tensor:
    """SE3 inverse. ``(..., 7) → (..., 7)``."""
    q = t[..., 3:7]
    q_inv = so3.inverse(q)
    t_inv = -so3.act(q_inv, t[..., :3])
    return torch.cat([t_inv, q_inv], dim=-1)


def log(t: torch.Tensor) -> torch.Tensor:
    """SE3 → se3 tangent. ``(..., 7) → (..., 6)``."""
    t_lin = t[..., :3]
    omega = so3.log(t[..., 3:7])
    theta2 = (omega * omega).sum(dim=-1, keepdim=True)
    use_taylor = theta2 < _taylor_theta2(t.dtype)
    theta2_safe = torch.where(use_taylor, torch.ones_like(theta2), theta2)
    theta = theta2_safe.sqrt()

    half_theta = theta / 2.0
    cot_half = torch.cos(half_theta) / torch.sin(half_theta).clamp(min=1e-30)
    coeff_full = (1.0 / theta2_safe.clamp(min=1e-30)) - cot_half / (2.0 * theta.clamp(min=1e-30))
    coeff_taylor = (1.0 / 12.0) + theta2 / 720.0
    coeff = torch.where(use_taylor, coeff_taylor, coeff_full)

    W = _hat3(omega)
    W2 = W @ W
    eye3 = torch.eye(3, dtype=t.dtype, device=t.device).expand_as(W)
    V_inv = eye3 - 0.5 * W + coeff.unsqueeze(-1) * W2
    v = (V_inv @ t_lin.unsqueeze(-1)).squeeze(-1)
    return torch.cat([v, omega], dim=-1)


def exp(v: torch.Tensor) -> torch.Tensor:
    """se3 tangent → SE3. ``(..., 6) → (..., 7)``."""
    linear = v[..., :3]
    omega = v[..., 3:6]
    theta2 = (omega * omega).sum(dim=-1, keepdim=True)
    use_taylor = theta2 < _taylor_theta2(v.dtype)
    theta2_safe = torch.where(use_taylor, torch.ones_like(theta2), theta2)
    theta = theta2_safe.sqrt()

    b_full = (1.0 - torch.cos(theta)) / theta2_safe.clamp(min=1e-30)
    b_taylor = 0.5 - theta2 / 24.0
    b = torch.where(use_taylor, b_taylor, b_full)

    c_full = (theta - torch.sin(theta)) / (theta * theta2_safe).clamp(min=1e-30)
    c_taylor = (1.0 / 6.0) - theta2 / 120.0
    c = torch.where(use_taylor, c_taylor, c_full)

    W = _hat3(omega)
    W2 = W @ W
    eye3 = torch.eye(3, dtype=v.dtype, device=v.device).expand_as(W)
    V = eye3 + b.unsqueeze(-1) * W + c.unsqueeze(-1) * W2
    t_lin = (V @ linear.unsqueeze(-1)).squeeze(-1)
    return torch.cat([t_lin, so3.exp(omega)], dim=-1)


def act(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Apply SE3 to a point. ``t: (..., 7), p: (..., 3) → (..., 3)``."""
    return so3.act(t[..., 3:7], p) + t[..., :3]


def adjoint(t: torch.Tensor) -> torch.Tensor:
    """6x6 adjoint matrix. ``(..., 7) → (..., 6, 6)``.

    Ad(T) = [[R, hat(p)@R], [0, R]] where p=translation, R=rotation.
    """
    p = t[..., :3]
    R = _quat_to_matrix(t[..., 3:7])
    pR = _hat3(p) @ R
    *batch, _, _ = R.shape
    zeros33 = torch.zeros(*batch, 3, 3, dtype=t.dtype, device=t.device)
    top = torch.cat([R, pR], dim=-1)
    bottom = torch.cat([zeros33, R], dim=-1)
    return torch.cat([top, bottom], dim=-2)


def adjoint_inv(t: torch.Tensor) -> torch.Tensor:
    """Inverse adjoint ``Ad(T^-1)`` — faster than inverting the adjoint.

    Ad(T^{-1}) = [[R^T, -(R^T @ hat(p))], [0, R^T]].
    """
    p = t[..., :3]
    R = _quat_to_matrix(t[..., 3:7])
    RT = R.transpose(-1, -2)
    neg_RT_skew = -(RT @ _hat3(p))
    *batch, _, _ = R.shape
    zeros33 = torch.zeros(*batch, 3, 3, dtype=t.dtype, device=t.device)
    top = torch.cat([RT, neg_RT_skew], dim=-1)
    bottom = torch.cat([zeros33, RT], dim=-1)
    return torch.cat([top, bottom], dim=-2)


def from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Convert a homogeneous matrix to the library's 7-vector SE3 layout.

    ``matrix`` has shape ``(..., 4, 4)`` with rotation in the upper-left
    block and translation in the last column.  The result has shape
    ``(..., 7)`` and layout ``[tx, ty, tz, qx, qy, qz, qw]``.
    """
    if not isinstance(matrix, torch.Tensor) or not matrix.is_floating_point():
        raise TypeError("matrix must be a floating torch.Tensor")
    if matrix.shape[-2:] != (4, 4):
        raise ValueError(f"matrix must have shape (..., 4, 4); got {tuple(matrix.shape)}")
    translation = matrix[..., :3, 3]
    quaternion = so3.from_matrix(matrix[..., :3, :3])
    return torch.cat((translation, quaternion), dim=-1)


def to_matrix(t: torch.Tensor) -> torch.Tensor:
    """Convert a 7-vector SE3 pose to a homogeneous matrix.

    ``t`` has shape ``(..., 7)`` and layout
    ``[tx, ty, tz, qx, qy, qz, qw]``.  The result has shape
    ``(..., 4, 4)`` and bottom row ``[0, 0, 0, 1]``.
    """
    if not isinstance(t, torch.Tensor) or not t.is_floating_point():
        raise TypeError("t must be a floating torch.Tensor")
    if t.shape[-1:] != (7,):
        raise ValueError(f"t must have shape (..., 7); got {tuple(t.shape)}")
    rotation = so3.to_matrix(t[..., 3:7])
    upper = torch.cat((rotation, t[..., :3].unsqueeze(-1)), dim=-1)
    bottom = t.new_zeros((*t.shape[:-1], 1, 4))
    bottom[..., 0, 3] = 1.0
    return torch.cat((upper, bottom), dim=-2)


def from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Pure-rotation SE3 from axis-angle. axis: (...,3), angle: (...,) → (...,7)."""
    q = so3.from_axis_angle(axis, angle)
    zeros = torch.zeros(*angle.shape, 3, device=angle.device, dtype=angle.dtype)
    return torch.cat([zeros, q], dim=-1)


def from_translation(axis: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
    """Pure-translation SE3 along ``axis`` with scalar displacement ``disp``.

    axis: (3,), disp: (...,) → (..., 7).
    """
    trans = disp.unsqueeze(-1) * axis.to(dtype=disp.dtype, device=disp.device)
    qxyz = torch.zeros(*disp.shape, 3, device=disp.device, dtype=disp.dtype)
    qw = torch.ones(*disp.shape, 1, device=disp.device, dtype=disp.dtype)
    return torch.cat([trans, qxyz, qw], dim=-1)


def normalize(t: torch.Tensor) -> torch.Tensor:
    """Re-normalize the quaternion part to project back onto SE3."""
    q = t[..., 3:7]
    q_normed = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return torch.cat([t[..., :3], q_normed], dim=-1)


def apply_base(base: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
    """Compose a base transform with ``(..., N, 7)`` link poses."""
    base_expanded = base.unsqueeze(-2)
    return compose(base_expanded.expand(*poses.shape[:-1], 7), poses)


def sclerp(
    T1: torch.Tensor,
    T2: torch.Tensor,
    t: torch.Tensor | float,
) -> torch.Tensor:
    """SE3 screw-linear interpolation. ``(..., 7), (..., 7), (...) → (..., 7)``.

    Geodesic on SE3: ``T1 · exp(t · log(T1⁻¹ · T2))``. Equivalent to the
    screw-axis (Chasles) motion between the two poses. ``t`` broadcasts
    against the leading batch of the inputs. ``t`` outside ``[0, 1]``
    extrapolates along the screw axis.

    """
    t = torch.as_tensor(t, dtype=T1.dtype, device=T1.device)
    while t.dim() < T1.dim():
        t = t.unsqueeze(-1)

    xi = log(compose(inverse(T1), T2))
    return compose(T1, exp(t * xi))
