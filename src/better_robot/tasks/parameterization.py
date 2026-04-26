"""Trajectory parameterizations — knot vs B-spline.

A ``TrajectoryParameterization`` maps a low-dimensional optimisation
variable ``z`` to a full ``(T, nq)`` trajectory and back. ``KnotTrajectory``
is the identity (``z`` *is* the trajectory). ``BSplineTrajectory`` uses a
fixed cubic B-spline basis whose control points are the optimisation
variables — fewer parameters, smoother trajectories, faster IK in the
trajopt loop.

See ``docs/design/08_TASKS.md §3`` and
``docs/claude_plan/accepted/16_optim_wiring_and_matrix_free.md``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class TrajectoryParameterization(Protocol):
    """Map between optimisation variables and a full ``(T, nq)`` trajectory."""

    def init(self, q_traj_seed: torch.Tensor) -> torch.Tensor:
        """Seed an optimisation variable from a desired ``(T, nq)`` trajectory."""
        ...

    def expand(self, z: torch.Tensor, *, T: int, nq: int) -> torch.Tensor:
        """Decode ``z`` into the full ``(T, nq)`` trajectory."""
        ...

    def tangent_dim_per_step(self) -> int:
        """Tangent-space dimension *per knot* of the optimisation variable."""
        ...


class KnotTrajectory:
    """Identity parameterisation — ``z`` is the trajectory itself."""

    def __init__(self) -> None:
        self._nq: int | None = None

    def init(self, q_traj_seed: torch.Tensor) -> torch.Tensor:
        if q_traj_seed.dim() != 2:
            raise ValueError(
                f"KnotTrajectory.init expects (T, nq); got {tuple(q_traj_seed.shape)}"
            )
        self._nq = int(q_traj_seed.shape[1])
        return q_traj_seed.detach().clone()

    def expand(self, z: torch.Tensor, *, T: int, nq: int) -> torch.Tensor:
        if z.shape != (T, nq):
            raise ValueError(
                f"KnotTrajectory.expand: z.shape {tuple(z.shape)} != ({T}, {nq})"
            )
        return z

    def tangent_dim_per_step(self) -> int:
        if self._nq is None:
            raise RuntimeError("KnotTrajectory.tangent_dim_per_step() requires init() first")
        return self._nq


class BSplineTrajectory:
    """Cubic B-spline trajectory with ``num_control_points`` control points.

    The control points (size ``(C, nq)``) are the optimisation variable.
    The basis matrix ``B ∈ R^{T×C}`` is fixed, sampled uniformly between
    the first and last control point. ``expand(z) = B @ z`` (batched
    over the second dim). For ``C == T`` this reduces to identity (modulo
    boundary effects). For ``C ≪ T`` the parameterisation enforces
    smoothness implicitly and shrinks the optimisation variable.

    See ``docs/design/08_TASKS.md §3``.
    """

    def __init__(self, *, num_control_points: int, degree: int = 3) -> None:
        if num_control_points < degree + 1:
            raise ValueError(
                f"num_control_points={num_control_points} must be ≥ degree+1={degree+1}"
            )
        self.num_control_points = int(num_control_points)
        self.degree = int(degree)
        self._basis: torch.Tensor | None = None
        self._cached_T: int | None = None
        self._nq: int | None = None

    def _build_basis(self, T: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Cubic uniform open-clamped B-spline basis, shape ``(T, C)``."""
        C = self.num_control_points
        deg = self.degree

        # Open-clamped uniform knot vector of length C + deg + 1.
        n_knots = C + deg + 1
        n_inner = n_knots - 2 * (deg + 1) + 2  # interior knots incl. endpoints
        if n_inner < 2:
            n_inner = 2
        inner = torch.linspace(0.0, 1.0, n_inner, dtype=dtype, device=device)
        knots = torch.cat([
            torch.zeros(deg, dtype=dtype, device=device),
            inner,
            torch.ones(deg, dtype=dtype, device=device),
        ])
        # Sample at T uniformly spaced parameters in [0, 1].
        u = torch.linspace(0.0, 1.0, T, dtype=dtype, device=device)

        # Cox–de Boor recursion.
        # N[i, p, t] = basis function i of degree p evaluated at u[t].
        # We only need degree=deg, so accumulate.
        N = torch.zeros(C + deg, T, dtype=dtype, device=device)
        # Degree 0 basis: indicator on knot interval.
        for i in range(C + deg):
            mask = (u >= knots[i]) & (u < knots[i + 1])
            N[i] = mask.to(dtype)
        # Cox-de Boor up to degree `deg`.
        for p in range(1, deg + 1):
            for i in range(C + deg - p):
                d1 = knots[i + p] - knots[i]
                d2 = knots[i + p + 1] - knots[i + 1]
                term1 = ((u - knots[i]) / d1) * N[i] if d1 > 0 else torch.zeros_like(u)
                term2 = ((knots[i + p + 1] - u) / d2) * N[i + 1] if d2 > 0 else torch.zeros_like(u)
                N[i] = term1 + term2
        # Force the last sample to clamp at the final basis (open-clamp).
        N[:, -1] = 0.0
        N[C - 1, -1] = 1.0
        return N[:C].T  # (T, C)

    def init(self, q_traj_seed: torch.Tensor) -> torch.Tensor:
        """Project the seed trajectory onto the spline basis (least squares)."""
        if q_traj_seed.dim() != 2:
            raise ValueError(
                f"BSplineTrajectory.init expects (T, nq); got {tuple(q_traj_seed.shape)}"
            )
        T, nq = q_traj_seed.shape
        self._nq = int(nq)
        self._cached_T = int(T)
        B = self._build_basis(T, q_traj_seed.dtype, q_traj_seed.device)
        self._basis = B
        # z = argmin ||B @ z - q_traj||² → z = (BᵀB)⁻¹Bᵀ q_traj
        z = torch.linalg.lstsq(B, q_traj_seed).solution  # (C, nq)
        return z.contiguous()

    def expand(self, z: torch.Tensor, *, T: int, nq: int) -> torch.Tensor:
        if z.shape != (self.num_control_points, nq):
            raise ValueError(
                f"BSplineTrajectory.expand: z.shape {tuple(z.shape)} != "
                f"({self.num_control_points}, {nq})"
            )
        if self._basis is None or self._cached_T != T:
            self._basis = self._build_basis(T, z.dtype, z.device)
            self._cached_T = T
        return self._basis @ z  # (T, nq)

    def tangent_dim_per_step(self) -> int:
        if self._nq is None:
            raise RuntimeError("BSplineTrajectory.tangent_dim_per_step() requires init() first")
        return self._nq
