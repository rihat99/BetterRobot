"""SE3 group operations — pure functional facade over the torch implementation.

Storage convention: ``(..., 7)`` tensor ``[tx, ty, tz, qx, qy, qz, qw]``
(scalar-last quaternion). Tangent vectors are ``(..., 6)``
``[vx, vy, vz, wx, wy, wz]``.

See ``docs/concepts/lie_and_spatial.md §3``.
"""

from __future__ import annotations

import torch

from . import _impl
from . import so3


def identity(
    *,
    batch_shape: tuple[int, ...] = (),
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return an SE3 identity with the given leading batch shape."""
    return _impl.se3_identity(batch_shape, device, dtype)


def compose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SE3 composition. ``a: (..., 7), b: (..., 7) → (..., 7)``."""
    return _impl.se3_compose(a, b)


def inverse(t: torch.Tensor) -> torch.Tensor:
    """SE3 inverse. ``(..., 7) → (..., 7)``."""
    return _impl.se3_inverse(t)


def log(t: torch.Tensor) -> torch.Tensor:
    """SE3 → se3 tangent. ``(..., 7) → (..., 6)``."""
    return _impl.se3_log(t)


def exp(v: torch.Tensor) -> torch.Tensor:
    """se3 tangent → SE3. ``(..., 6) → (..., 7)``."""
    return _impl.se3_exp(v)


def act(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Apply SE3 to a point. ``t: (..., 7), p: (..., 3) → (..., 3)``."""
    return _impl.se3_act(t, p)


def adjoint(t: torch.Tensor) -> torch.Tensor:
    """6x6 adjoint matrix. ``(..., 7) → (..., 6, 6)``.

    Ad(T) = [[R, hat(p)@R], [0, R]] where p=translation, R=rotation.
    """
    return _impl.se3_adjoint(t)


def adjoint_inv(t: torch.Tensor) -> torch.Tensor:
    """Inverse adjoint ``Ad(T^-1)`` — faster than inverting the adjoint.

    Ad(T^{-1}) = [[R^T, -(R^T @ hat(p))], [0, R^T]].
    """
    return _impl.se3_adjoint_inv(t)


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
    return _impl.se3_from_axis_angle(axis, angle)


def from_translation(axis: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
    """Pure-translation SE3 along ``axis`` with scalar displacement ``disp``.

    axis: (3,), disp: (...,) → (..., 7).
    """
    return _impl.se3_from_translation_axis(axis, disp)


def normalize(t: torch.Tensor) -> torch.Tensor:
    """Re-normalize the quaternion part to project back onto SE3."""
    return _impl.se3_normalize(t)


def apply_base(base: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
    """Compose a base transform with ``(..., N, 7)`` link poses."""
    return _impl.se3_apply_base(base, poses)


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
