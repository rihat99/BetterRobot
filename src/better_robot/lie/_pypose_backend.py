"""Private PyPose bridge — the ONLY module in BetterRobot that imports pypose.

Every function in ``lie/se3.py`` and ``lie/so3.py`` forwards here. To swap
pypose for Warp, replace this file with a same-shaped backend and nothing
else changes.

See ``docs/03_LIE_AND_SPATIAL.md §6`` for the full rationale.
"""

from __future__ import annotations

import pypose as _pp
import torch

# ──────────────────────────────── SE3 ────────────────────────────────


def se3_identity(batch_shape: tuple[int, ...], device, dtype) -> torch.Tensor:
    """Return an SE3 identity tensor of shape ``(*batch_shape, 7)``."""
    out = torch.zeros((*batch_shape, 7), device=device, dtype=dtype)
    out[..., 6] = 1.0  # qw = 1
    return out


def se3_compose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (_pp.SE3(a) @ _pp.SE3(b)).tensor()


def se3_inverse(t: torch.Tensor) -> torch.Tensor:
    return _pp.SE3(t).Inv().tensor()


def se3_log(t: torch.Tensor) -> torch.Tensor:
    return _pp.SE3(t).Log().tensor()


def se3_exp(v: torch.Tensor) -> torch.Tensor:
    return _pp.se3(v).Exp().tensor()


def se3_act(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return _pp.SE3(t).Act(p)


# ──────────────────────────────── SO3 ────────────────────────────────


def so3_identity(batch_shape: tuple[int, ...], device, dtype) -> torch.Tensor:
    out = torch.zeros((*batch_shape, 4), device=device, dtype=dtype)
    out[..., 3] = 1.0
    return out


def so3_compose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (_pp.SO3(a) @ _pp.SO3(b)).tensor()


def so3_inverse(q: torch.Tensor) -> torch.Tensor:
    return _pp.SO3(q).Inv().tensor()


def so3_log(q: torch.Tensor) -> torch.Tensor:
    return _pp.SO3(q).Log().tensor()


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    return _pp.so3(w).Exp().tensor()


def so3_from_matrix(R: torch.Tensor) -> torch.Tensor:
    return _pp.mat2SO3(R).tensor()


def so3_to_matrix(q: torch.Tensor) -> torch.Tensor:
    return _pp.SO3(q).matrix()


def so3_act(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return _pp.SO3(q).Act(p)


def so3_adjoint(q: torch.Tensor) -> torch.Tensor:
    """Adjoint of SO3 is the rotation matrix itself. (..., 4) → (..., 3, 3)."""
    return _pp.SO3(q).matrix()


def so3_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """(..., 3) unit axis + (...) angle → (..., 4) quaternion [qx,qy,qz,qw]."""
    half = angle / 2.0
    sin_h = torch.sin(half)
    cos_h = torch.cos(half)
    # axis: (..., 3), sin_h: (...)
    qxyz = sin_h.unsqueeze(-1) * axis
    qw = cos_h.unsqueeze(-1)
    return torch.cat([qxyz, qw], dim=-1)


def so3_normalize(q: torch.Tensor) -> torch.Tensor:
    """Re-normalize quaternion to unit length."""
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def se3_adjoint(t: torch.Tensor) -> torch.Tensor:
    """6×6 batch adjoint. (..., 7) → (..., 6, 6).

    Ad(T) = [[R,          hat(p) @ R],
              [0_{3×3},   R         ]]
    where p = T[..., :3], R = rotation from T[..., 3:7].
    """
    p = t[..., :3]   # (..., 3)
    R = _pp.SO3(t[..., 3:7]).matrix()  # (..., 3, 3)

    # skew-symmetric matrix of p — batched
    z = torch.zeros_like(p[..., 0])
    skew = torch.stack([
        torch.stack([z,        -p[..., 2],  p[..., 1]], dim=-1),
        torch.stack([p[..., 2],  z,         -p[..., 0]], dim=-1),
        torch.stack([-p[..., 1], p[..., 0],  z       ], dim=-1),
    ], dim=-2)  # (..., 3, 3)

    pR = skew @ R  # (..., 3, 3)
    *batch, _, _ = R.shape
    zeros33 = torch.zeros(*batch, 3, 3, dtype=t.dtype, device=t.device)

    top    = torch.cat([R,       pR     ], dim=-1)  # (..., 3, 6)
    bottom = torch.cat([zeros33, R      ], dim=-1)  # (..., 3, 6)
    return torch.cat([top, bottom], dim=-2)  # (..., 6, 6)


def se3_adjoint_inv(t: torch.Tensor) -> torch.Tensor:
    """Ad(T^{-1}) = Ad(T)^{-1}. Faster than inverting the 6×6 matrix."""
    # T_inv = Inv(T); Ad(T_inv) directly:
    p = t[..., :3]   # (..., 3)
    R = _pp.SO3(t[..., 3:7]).matrix()  # (..., 3, 3)
    RT = R.transpose(-1, -2)           # (..., 3, 3)

    # skew of p
    z = torch.zeros_like(p[..., 0])
    skew = torch.stack([
        torch.stack([z,        -p[..., 2],  p[..., 1]], dim=-1),
        torch.stack([p[..., 2],  z,         -p[..., 0]], dim=-1),
        torch.stack([-p[..., 1], p[..., 0],  z       ], dim=-1),
    ], dim=-2)  # (..., 3, 3)

    # Ad(T^{-1}) = [[R^T,  -(R^T @ hat(p))],
    #               [0,     R^T           ]]
    neg_RT_skew = -(RT @ skew)  # (..., 3, 3)
    *batch, _, _ = R.shape
    zeros33 = torch.zeros(*batch, 3, 3, dtype=t.dtype, device=t.device)

    top    = torch.cat([RT,      neg_RT_skew], dim=-1)
    bottom = torch.cat([zeros33, RT         ], dim=-1)
    return torch.cat([top, bottom], dim=-2)


def se3_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Pure-rotation SE3 from axis + angle. (...) → (..., 7)."""
    q = so3_from_axis_angle(axis, angle)   # (..., 4)
    zeros = torch.zeros(*angle.shape, 3, device=angle.device, dtype=angle.dtype)
    return torch.cat([zeros, q], dim=-1)


def se3_from_translation_axis(axis: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
    """Pure-translation SE3 along ``axis`` with scalar displacement ``disp``.

    axis: (3,), disp: (...,) → (..., 7).
    """
    trans = disp.unsqueeze(-1) * axis.to(dtype=disp.dtype, device=disp.device)
    qxyz = torch.zeros(*disp.shape, 3, device=disp.device, dtype=disp.dtype)
    qw   = torch.ones(*disp.shape, 1, device=disp.device, dtype=disp.dtype)
    return torch.cat([trans, qxyz, qw], dim=-1)


def se3_normalize(t: torch.Tensor) -> torch.Tensor:
    """Re-normalize the quaternion part of an SE3 tensor."""
    q = t[..., 3:7]
    q_normed = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return torch.cat([t[..., :3], q_normed], dim=-1)


def se3_apply_base(base: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
    """Compose base transform with (..., N, 7) link poses."""
    base_expanded = _pp.SE3(base.unsqueeze(-2))  # (..., 1, 7)
    return (base_expanded @ _pp.SE3(poses)).tensor()
