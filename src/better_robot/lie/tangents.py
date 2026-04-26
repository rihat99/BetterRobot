"""Right/left Jacobians, ``hat``/``vee``, small-angle-safe tangent algebra.

These are what ``kinematics/jacobian.py`` uses to drop the legacy
``Jlog ≈ I`` approximation in the pose residual Jacobian.

Conventions
-----------
- SO3 tangent: ``(..., 3)`` = ``[wx, wy, wz]``
- SE3 tangent: ``(..., 6)`` = ``[vx, vy, vz, wx, wy, wz]`` (translation first,
  rotation second)

See ``docs/concepts/lie_and_spatial.md §5``.
"""

from __future__ import annotations

import math

import torch

# ────────────────────────── hat / vee ──────────────────────────────────────


def hat_so3(w: torch.Tensor) -> torch.Tensor:
    """``(..., 3) → (..., 3, 3)`` skew-symmetric matrix.

    hat([wx, wy, wz]) = [[ 0,  -wz,  wy],
                          [ wz,  0,  -wx],
                          [-wy,  wx,   0]]
    """
    z = torch.zeros_like(w[..., 0])
    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    return torch.stack([
        torch.stack([ z,  -wz,  wy], dim=-1),
        torch.stack([ wz,  z,  -wx], dim=-1),
        torch.stack([-wy,  wx,   z ], dim=-1),
    ], dim=-2)


def vee_so3(W: torch.Tensor) -> torch.Tensor:
    """``(..., 3, 3) → (..., 3)`` — extract 3-vector from skew-symmetric matrix."""
    return torch.stack([W[..., 2, 1], W[..., 0, 2], W[..., 1, 0]], dim=-1)


def hat_se3(xi: torch.Tensor) -> torch.Tensor:
    """``(..., 6) → (..., 4, 4)`` se3 tangent to 4×4 homogeneous matrix.

    hat([vx, vy, vz, wx, wy, wz]) = [[hat_so3(w), v],
                                       [0,          0]]
    """
    v = xi[..., :3]
    w = xi[..., 3:]
    W = hat_so3(w)        # (..., 3, 3)
    *batch, _, _ = W.shape
    zero_row = torch.zeros(*batch, 1, 3, dtype=xi.dtype, device=xi.device)
    zero_col = torch.zeros(*batch, 4, 1, dtype=xi.dtype, device=xi.device)
    # Build 4×4: [[W, v.unsqueeze(-1)], [0, 0]]
    top = torch.cat([W, v.unsqueeze(-1)], dim=-1)       # (..., 3, 4)
    bot = torch.cat([zero_row, torch.zeros(*batch, 1, 1,
                                            dtype=xi.dtype, device=xi.device)], dim=-1)  # (..., 1, 4)
    mat = torch.cat([top, bot], dim=-2)                 # (..., 4, 4)
    # Attach the zero column for homogeneous form (already done above)
    return mat


def vee_se3(X: torch.Tensor) -> torch.Tensor:
    """``(..., 4, 4) → (..., 6)`` — 4×4 homogeneous matrix to se3 tangent."""
    v = X[..., :3, 3]                              # (..., 3)  translation column
    w = vee_so3(X[..., :3, :3])                    # (..., 3)  rotation part
    return torch.cat([v, w], dim=-1)


# ──────────────────────── SO3 Jacobians ───────────────────────────────────


def _so3_jac_coefficients(theta2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (A, B) for the SO3 right Jacobian formula.

    Jr(phi) = I - A * hat(phi) + B * hat(phi)^2

    A = (1 - cos(theta)) / theta^2
    B = (theta - sin(theta)) / theta^3

    Uses Taylor expansion near theta=0 for numerical stability.
    """
    theta = torch.sqrt(theta2.clamp(min=0.0))
    # Taylor threshold: below this, use the small-angle series
    safe = theta2 > 1e-10

    # Full expressions
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    A_full = (1.0 - cos_t) / theta2.clamp(min=1e-30)
    B_full = (theta - sin_t) / (theta * theta2).clamp(min=1e-30)

    # Taylor: A ≈ 1/2 - theta^2/24, B ≈ 1/6 - theta^2/120
    A_taylor = 0.5 - theta2 / 24.0
    B_taylor = 1.0 / 6.0 - theta2 / 120.0

    A = torch.where(safe, A_full, A_taylor)
    B = torch.where(safe, B_full, B_taylor)
    return A, B


def right_jacobian_so3(omega: torch.Tensor) -> torch.Tensor:
    """``Jr(ω)`` — right Jacobian of the SO3 exp. ``(..., 3) → (..., 3, 3)``.

    Jr(phi) = I - A * hat(phi) + B * hat(phi)^2
    where A = (1-cosθ)/θ², B = (θ-sinθ)/θ³, θ = ‖phi‖.
    """
    theta2 = (omega * omega).sum(dim=-1)          # (...)
    A, B = _so3_jac_coefficients(theta2)
    H = hat_so3(omega)                             # (..., 3, 3)
    H2 = H @ H                                    # (..., 3, 3)
    *batch, _, _ = H.shape
    I3 = torch.eye(3, dtype=omega.dtype, device=omega.device).expand(*batch, 3, 3)
    A = A[..., None, None]
    B = B[..., None, None]
    return I3 - A * H + B * H2


def right_jacobian_inv_so3(omega: torch.Tensor) -> torch.Tensor:
    """``Jr^{-1}(ω)`` — inverse of the right Jacobian of SO3.

    Jr_inv(phi) = I + (1/2) * hat(phi) + C * hat(phi)^2
    where C = 1/θ² - (1+cosθ)/(2θ sinθ)  →  1/12 for θ→0.
    """
    theta2 = (omega * omega).sum(dim=-1)
    theta = torch.sqrt(theta2.clamp(min=0.0))
    safe = theta2 > 1e-10

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    # C_full = 1/theta2 - (1+cos_t) / (2 * theta * sin_t)
    C_full = 1.0 / theta2.clamp(min=1e-30) - (1.0 + cos_t) / (2.0 * theta * sin_t).clamp(min=1e-30)
    C_taylor = 1.0 / 12.0 + theta2 / 720.0

    C = torch.where(safe, C_full, C_taylor)

    H = hat_so3(omega)
    H2 = H @ H
    *batch, _, _ = H.shape
    I3 = torch.eye(3, dtype=omega.dtype, device=omega.device).expand(*batch, 3, 3)
    C = C[..., None, None]
    return I3 + 0.5 * H + C * H2


def left_jacobian_so3(omega: torch.Tensor) -> torch.Tensor:
    """``Jl(ω)`` — left Jacobian of the SO3 exp.

    Jl(phi) = Jr(-phi).
    """
    return right_jacobian_so3(-omega)


def left_jacobian_inv_so3(omega: torch.Tensor) -> torch.Tensor:
    """``Jl^{-1}(ω)``."""
    return right_jacobian_inv_so3(-omega)


# ──────────────────────── SE3 adjoint matrix ─────────────────────────────


def _ad_matrix(xi: torch.Tensor) -> torch.Tensor:
    """Adjoint representation of se3 tangent. ``(..., 6) → (..., 6, 6)``.

    ad(xi) = ad([v, w]) = [[hat(w), hat(v)],
                             [0,      hat(w)]]
    """
    v = xi[..., :3]
    w = xi[..., 3:]
    Vhat = hat_so3(v)   # (..., 3, 3)
    What = hat_so3(w)   # (..., 3, 3)
    *batch, _, _ = What.shape
    zeros33 = torch.zeros(*batch, 3, 3, dtype=xi.dtype, device=xi.device)
    top    = torch.cat([What, Vhat  ], dim=-1)   # (..., 3, 6)
    bottom = torch.cat([zeros33, What], dim=-1)  # (..., 3, 6)
    return torch.cat([top, bottom], dim=-2)       # (..., 6, 6)


# ──────────────────────── SE3 Jacobians ──────────────────────────────────


def right_jacobian_se3(xi: torch.Tensor) -> torch.Tensor:
    """``Jr(ξ)`` — right Jacobian of the SE3 exp. ``(..., 6) → (..., 6, 6)``.

    Computed via the matrix series:
        Jr = sum_{k=0}^{N} (-1)^k / (k+1)! * ad(xi)^k

    Truncated at N=9 (error < machine epsilon for ‖xi‖ < 6).
    """
    *batch, _ = xi.shape
    ad = _ad_matrix(xi)                            # (..., 6, 6)
    I6 = torch.eye(6, dtype=xi.dtype, device=xi.device).expand(*batch, 6, 6)
    result = I6.clone()
    ad_power = I6.clone()
    for k in range(1, 10):
        ad_power = ad_power @ ad
        coeff = ((-1.0) ** k) / math.factorial(k + 1)
        result = result + coeff * ad_power
    return result


def right_jacobian_inv_se3(xi: torch.Tensor) -> torch.Tensor:
    """``Jr^{-1}(ξ)``. This is the kernel used by ``PoseResidual.jacobian``."""
    Jr = right_jacobian_se3(xi)
    return torch.linalg.inv(Jr)


def left_jacobian_se3(xi: torch.Tensor) -> torch.Tensor:
    """``Jl(ξ)``."""
    return right_jacobian_se3(-xi)


def left_jacobian_inv_se3(xi: torch.Tensor) -> torch.Tensor:
    """``Jl^{-1}(ξ)``."""
    return right_jacobian_inv_se3(-xi)
