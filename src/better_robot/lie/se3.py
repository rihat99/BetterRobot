"""SE3 group operations — pure functional facade over the active ``Backend``.

Storage convention: ``(..., 7)`` tensor ``[tx, ty, tz, qx, qy, qz, qw]``
(scalar-last quaternion). Tangent vectors are ``(..., 6)``
``[vx, vy, vz, wx, wy, wz]``.

Each function takes an optional ``backend`` keyword argument. When
``None``, the active default (see :func:`better_robot.backends.default_backend`)
is used. The ``identity`` / ``from_*`` constructors do not need a backend
and ignore the kwarg.

See ``docs/concepts/lie_and_spatial.md §3``.
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
    """Return an SE3 identity with the given leading batch shape."""
    return _be.se3_identity(batch_shape, device, dtype)


def compose(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    backend: "Backend | None" = None,
) -> torch.Tensor:
    """SE3 composition. ``a: (..., 7), b: (..., 7) → (..., 7)``."""
    return _lie(backend).se3_compose(a, b)


def inverse(
    t: torch.Tensor, *, backend: "Backend | None" = None,
) -> torch.Tensor:
    """SE3 inverse. ``(..., 7) → (..., 7)``."""
    return _lie(backend).se3_inverse(t)


def log(
    t: torch.Tensor, *, backend: "Backend | None" = None,
) -> torch.Tensor:
    """SE3 → se3 tangent. ``(..., 7) → (..., 6)``."""
    return _lie(backend).se3_log(t)


def exp(
    v: torch.Tensor, *, backend: "Backend | None" = None,
) -> torch.Tensor:
    """se3 tangent → SE3. ``(..., 6) → (..., 7)``."""
    return _lie(backend).se3_exp(v)


def act(
    t: torch.Tensor, p: torch.Tensor, *, backend: "Backend | None" = None,
) -> torch.Tensor:
    """Apply SE3 to a point. ``t: (..., 7), p: (..., 3) → (..., 3)``."""
    return _lie(backend).se3_act(t, p)


def adjoint(
    t: torch.Tensor, *, backend: "Backend | None" = None,
) -> torch.Tensor:
    """6x6 adjoint matrix. ``(..., 7) → (..., 6, 6)``.

    Ad(T) = [[R, hat(p)@R], [0, R]] where p=translation, R=rotation.
    """
    return _lie(backend).se3_adjoint(t)


def adjoint_inv(
    t: torch.Tensor, *, backend: "Backend | None" = None,
) -> torch.Tensor:
    """Inverse adjoint ``Ad(T^-1)`` — faster than inverting the adjoint.

    Ad(T^{-1}) = [[R^T, -(R^T @ hat(p))], [0, R^T]].
    """
    return _lie(backend).se3_adjoint_inv(t)


def from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Pure-rotation SE3 from axis-angle. axis: (...,3), angle: (...,) → (...,7)."""
    return _be.se3_from_axis_angle(axis, angle)


def from_translation(axis: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
    """Pure-translation SE3 along ``axis`` with scalar displacement ``disp``.

    axis: (3,), disp: (...,) → (..., 7).
    """
    return _be.se3_from_translation_axis(axis, disp)


def normalize(
    t: torch.Tensor, *, backend: "Backend | None" = None,
) -> torch.Tensor:
    """Re-normalize the quaternion part to project back onto SE3."""
    return _lie(backend).se3_normalize(t)


def apply_base(base: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
    """Compose a base transform with ``(..., N, 7)`` link poses."""
    return _be.se3_apply_base(base, poses)


def sclerp(
    T1: torch.Tensor, T2: torch.Tensor, t: torch.Tensor | float,
    *,
    backend: "Backend | None" = None,
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

    xi = log(compose(inverse(T1, backend=backend), T2, backend=backend), backend=backend)
    return compose(T1, exp(t * xi, backend=backend), backend=backend)
