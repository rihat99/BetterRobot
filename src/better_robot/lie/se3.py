"""SE(3) group operations on ``(..., 7)`` pose tensors ``[tx, ty, tz, qx, qy, qz, qw]`` (quaternion scalar last).

Tangents are ``(..., 6)`` twists ``[vx, vy, vz, wx, wy, wz]``, linear first. ``compose(a, b)`` equals the
homogeneous product ``a @ b``; ``inverse(t)`` is the rigid inverse; ``act(t, p)`` maps a point ``p (..., 3)`` by
``R p + t``. ``exp`` is the full SE(3) exponential (translation ``V(w) v``) and ``log`` is its inverse, so
``log(exp(xi)) == xi`` for ``|w| < pi``. ``adjoint(t) (..., 6, 6)`` satisfies
``adjoint(T) @ xi == log(T * exp(xi) * T^-1)``. Leading batch axes broadcast; dtype and device are preserved.
"""

from __future__ import annotations

import torch

from . import so3
from ._checks import check_tensor, check_tensors
from .so3 import _hat3, _taylor_theta2


def identity(
    *,
    batch_shape: tuple[int, ...] = (),
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """SE3 identity pose ``(*batch_shape, 7)``."""
    out = torch.zeros((*batch_shape, 7), device=device, dtype=dtype)
    out[..., 6] = 1.0
    return out


def compose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SE3 composition. ``a: (..., 7), b: (..., 7) → (..., 7)``."""
    check_tensors(a=(a, (7,)), b=(b, (7,)))
    return compose_raw(a, b)


def compose_raw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`compose` without input checks, for checked inputs: ``a: (..., 7), b: (..., 7) → (..., 7)``."""
    a_t = a[..., :3]
    a_q = a[..., 3:7]
    b_t = b[..., :3]
    b_q = b[..., 3:7]
    c_t = a_t + so3.act_raw(a_q, b_t)
    c_q = so3.compose_raw(a_q, b_q)
    return torch.cat([c_t, c_q], dim=-1)


def inverse(t: torch.Tensor) -> torch.Tensor:
    """SE3 inverse. ``(..., 7) → (..., 7)``."""
    check_tensor("t", t, shape=(7,))
    return inverse_raw(t)


def inverse_raw(t: torch.Tensor) -> torch.Tensor:
    """`inverse` without input checks, for checked inputs: ``(..., 7) → (..., 7)``."""
    q = t[..., 3:7]
    q_inv = so3.inverse_raw(q)
    t_inv = -so3.act_raw(q_inv, t[..., :3])
    return torch.cat([t_inv, q_inv], dim=-1)


def log(t: torch.Tensor) -> torch.Tensor:
    """SE3 → se3 tangent. ``(..., 7) → (..., 6)``."""
    check_tensor("t", t, shape=(7,))
    return log_raw(t)


def log_raw(t: torch.Tensor) -> torch.Tensor:
    """`log` without input checks, for checked inputs: ``(..., 7) → (..., 6)``."""
    t_lin = t[..., :3]
    omega = so3.log_raw(t[..., 3:7])
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
    check_tensor("v", v, shape=(6,))
    return exp_raw(v)


def exp_raw(v: torch.Tensor) -> torch.Tensor:
    """`exp` without input checks, for checked inputs: ``(..., 6) → (..., 7)``."""
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
    return torch.cat([t_lin, so3.exp_raw(omega)], dim=-1)


def act(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Apply SE3 to a point. ``t: (..., 7), p: (..., 3) → (..., 3)``."""
    check_tensors(t=(t, (7,)), p=(p, (3,)))
    return so3.act_raw(t[..., 3:7], p) + t[..., :3]


def adjoint(t: torch.Tensor) -> torch.Tensor:
    """6x6 adjoint matrix. ``(..., 7) → (..., 6, 6)``.

    Ad(T) = [[R, hat(p)@R], [0, R]] where p=translation, R=rotation.
    """
    check_tensor("t", t, shape=(7,))
    p = t[..., :3]
    R = so3.to_matrix_raw(t[..., 3:7])
    pR = _hat3(p) @ R
    *batch, _, _ = R.shape
    zeros33 = torch.zeros(*batch, 3, 3, dtype=t.dtype, device=t.device)
    top = torch.cat([R, pR], dim=-1)
    bottom = torch.cat([zeros33, R], dim=-1)
    return torch.cat([top, bottom], dim=-2)


def adjoint_inverse(t: torch.Tensor) -> torch.Tensor:
    """Inverse adjoint ``Ad(T)^-1 = Ad(T^-1)``. ``(..., 7) → (..., 6, 6)``.

    Built from the blocks ``[[R^T, -R^T hat(p)], [0, R^T]]``, so no matrix is inverted.
    """
    check_tensor("t", t, shape=(7,))
    p = t[..., :3]
    R = so3.to_matrix_raw(t[..., 3:7])
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
    check_tensor("matrix", matrix, shape=(4, 4))
    translation = matrix[..., :3, 3]
    quaternion = so3.from_matrix_raw(matrix[..., :3, :3])
    return torch.cat((translation, quaternion), dim=-1)


def to_matrix(t: torch.Tensor) -> torch.Tensor:
    """Convert a 7-vector SE3 pose to a homogeneous matrix.

    ``t`` has shape ``(..., 7)`` and layout
    ``[tx, ty, tz, qx, qy, qz, qw]``.  The result has shape
    ``(..., 4, 4)`` and bottom row ``[0, 0, 0, 1]``.
    """
    check_tensor("t", t, shape=(7,))
    return to_matrix_raw(t)


def to_matrix_raw(t: torch.Tensor) -> torch.Tensor:
    """`to_matrix` without input checks, for checked inputs."""
    rotation = so3.to_matrix_raw(t[..., 3:7])
    upper = torch.cat((rotation, t[..., :3].unsqueeze(-1)), dim=-1)
    bottom = t.new_zeros((*t.shape[:-1], 1, 4))
    bottom[..., 0, 3] = 1.0
    return torch.cat((upper, bottom), dim=-2)


def from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Pure-rotation pose from axis-angle. ``axis: (..., 3)`` unit, ``angle: (...) → (..., 7)``."""
    check_tensors(axis=(axis, (3,)), angle=(angle, ()))
    return from_axis_angle_raw(axis, angle)


def from_axis_angle_raw(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """`from_axis_angle` without input checks, for checked inputs: ``angle: (...) → (..., 7)``."""
    q = so3.from_axis_angle_raw(axis, angle)
    zeros = torch.zeros(*angle.shape, 3, device=angle.device, dtype=angle.dtype)
    return torch.cat([zeros, q], dim=-1)


def from_translation(translation: torch.Tensor) -> torch.Tensor:
    """Pure-translation pose with the identity rotation. ``(..., 3) → (..., 7)``."""
    check_tensor("translation", translation, shape=(3,))
    return from_translation_raw(translation)


def from_translation_raw(translation: torch.Tensor) -> torch.Tensor:
    """`from_translation` without input checks, for checked inputs: ``(..., 3) → (..., 7)``."""
    batch = translation.shape[:-1]
    return torch.cat((translation, translation.new_zeros(*batch, 3), translation.new_ones(*batch, 1)), dim=-1)


def normalize(t: torch.Tensor) -> torch.Tensor:
    """Pose with its quaternion re-normalized to unit length. ``(..., 7) → (..., 7)``."""
    check_tensor("t", t, shape=(7,))
    return normalize_raw(t)


def normalize_raw(t: torch.Tensor) -> torch.Tensor:
    """`normalize` without input checks, for checked inputs: ``(..., 7) → (..., 7)``."""
    q = t[..., 3:7]
    q_normed = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return torch.cat([t[..., :3], q_normed], dim=-1)


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
    check_tensors(T1=(T1, (7,)), T2=(T2, (7,)))
    if isinstance(t, torch.Tensor):
        check_tensor("t", t, shape=(), dtype=T1.dtype, device=T1.device, owner="T1")
    t = torch.as_tensor(t, dtype=T1.dtype, device=T1.device)
    while t.dim() < T1.dim():
        t = t.unsqueeze(-1)

    xi = log_raw(compose_raw(inverse_raw(T1), T2))
    return compose_raw(T1, exp_raw(t * xi))


__all__ = [
    "act",
    "adjoint",
    "adjoint_inverse",
    "compose",
    "exp",
    "from_axis_angle",
    "from_matrix",
    "from_translation",
    "identity",
    "inverse",
    "log",
    "normalize",
    "sclerp",
    "to_matrix",
]
