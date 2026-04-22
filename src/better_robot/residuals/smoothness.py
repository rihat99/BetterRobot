"""Manifold-aware smoothness residuals on trajectory-shaped variables.

``state.variables`` is expected to be shape ``(T, nq)``; residuals produce
flat vectors and analytic Jacobians of shape ``(dim, T*nv)`` — one dense
block per timestep. Velocity / acceleration are computed in the tangent
space via ``Model.difference`` — so SE(3) floating bases, spherical joints,
and revolute joints all contribute the right number of DOFs without
special-casing.

See ``docs/07_RESIDUALS_COSTS_SOLVERS.md §2``.
"""

from __future__ import annotations

import torch

from ..data_model.model import Model
from .base import Residual, ResidualState
from .registry import register_residual


def _require_traj(q: torch.Tensor, name: str) -> int:
    if q.dim() != 2:
        raise ValueError(
            f"{name}: expected state.variables with shape (T, nq); got {tuple(q.shape)}"
        )
    T = int(q.shape[0])
    if T < 3:
        raise ValueError(f"{name}: need at least 3 timesteps, got T={T}")
    return T


@register_residual("velocity")
class VelocityResidual:
    """3-point central-difference velocity in tangent space.

    For ``t ∈ [1, T-1)``::

        v_t = model.difference(q_{t-1}, q_{t+1}) / (2 * dt)

    Output dim: ``nv * (T - 2)``. Flattened row-major: outer axis is
    timestep ``t``, inner axis is the ``nv`` tangent components.

    Analytic Jacobian uses the identity-right-Jacobian approximation
    ``J_r(difference) ≈ I`` — valid in the small-step regime that motion
    optimization operates in.
    """

    name: str = "velocity"

    def __init__(self, model: Model, *, dt: float, weight: float = 1.0) -> None:
        self.model = model
        self.dt = float(dt)
        self.weight = float(weight)
        # dim is determined by T at first call
        self.dim: int = 0

    def __call__(self, state: ResidualState) -> torch.Tensor:
        q = state.variables
        T = _require_traj(q, "VelocityResidual")
        q_prev = q[:-2]
        q_next = q[2:]
        v = self.model.difference(q_prev, q_next) / (2.0 * self.dt)  # (T-2, nv)
        self.dim = v.numel()
        return (v * self.weight).reshape(-1)

    def jacobian(self, state: ResidualState) -> torch.Tensor | None:
        q = state.variables
        T = _require_traj(q, "VelocityResidual")
        nv = self.model.nv
        self.dim = nv * (T - 2)

        device, dtype = q.device, q.dtype
        J = torch.zeros(self.dim, T * nv, device=device, dtype=dtype)
        eye = torch.eye(nv, device=device, dtype=dtype)
        scale = self.weight / (2.0 * self.dt)
        for s in range(T - 2):
            r0, r1 = s * nv, (s + 1) * nv
            J[r0:r1, s * nv:(s + 1) * nv] = -eye * scale
            J[r0:r1, (s + 2) * nv:(s + 3) * nv] = eye * scale
        return J

    def apply_jac_transpose(self, state: ResidualState, r: torch.Tensor) -> torch.Tensor:
        """``J^T @ r`` without materialising the dense Jacobian — O(T·nv)."""
        q = state.variables
        T = _require_traj(q, "VelocityResidual")
        nv = self.model.nv
        r_mat = r.reshape(T - 2, nv)
        scale = self.weight / (2.0 * self.dt)

        g = torch.zeros(T, nv, dtype=q.dtype, device=q.device)
        g[:T - 2] += -scale * r_mat
        g[2:T] += scale * r_mat
        return g.reshape(-1)


@register_residual("acceleration")
class AccelerationResidual:
    """3-point tangent-space acceleration.

    For ``t ∈ [1, T-1)``::

        a_t = (model.difference(q_t, q_{t+1}) - model.difference(q_{t-1}, q_t)) / dt²

    Output dim: ``nv * (T - 2)``. Analytic Jacobian is tridiagonal over
    timesteps with blocks ``[+I, −2I, +I] / dt²`` (identity-right-Jacobian
    approximation — see ``VelocityResidual`` docstring).
    """

    name: str = "acceleration"

    def __init__(self, model: Model, *, dt: float, weight: float = 1.0) -> None:
        self.model = model
        self.dt = float(dt)
        self.weight = float(weight)
        self.dim: int = 0

    def __call__(self, state: ResidualState) -> torch.Tensor:
        q = state.variables
        T = _require_traj(q, "AccelerationResidual")
        diff_fwd = self.model.difference(q[1:-1], q[2:])    # v^+_t for t ∈ [1, T-1)
        diff_back = self.model.difference(q[:-2], q[1:-1])  # v^-_t for t ∈ [1, T-1)
        a = (diff_fwd - diff_back) / (self.dt ** 2)          # (T-2, nv)
        self.dim = a.numel()
        return (a * self.weight).reshape(-1)

    def jacobian(self, state: ResidualState) -> torch.Tensor | None:
        q = state.variables
        T = _require_traj(q, "AccelerationResidual")
        nv = self.model.nv
        self.dim = nv * (T - 2)

        device, dtype = q.device, q.dtype
        J = torch.zeros(self.dim, T * nv, device=device, dtype=dtype)
        eye = torch.eye(nv, device=device, dtype=dtype)
        scale = self.weight / (self.dt ** 2)
        for s in range(T - 2):
            r0, r1 = s * nv, (s + 1) * nv
            J[r0:r1, s * nv:(s + 1) * nv] = eye * scale
            J[r0:r1, (s + 1) * nv:(s + 2) * nv] = -2.0 * eye * scale
            J[r0:r1, (s + 2) * nv:(s + 3) * nv] = eye * scale
        return J

    def apply_jac_transpose(self, state: ResidualState, r: torch.Tensor) -> torch.Tensor:
        """``J^T @ r`` without materialising the dense Jacobian — O(T·nv).

        The full Jacobian is block-tridiagonal over time with blocks
        ``[+I, −2I, +I] / dt²``; the transpose has the same structure, and
        ``J^T r`` collapses into three aligned accumulations.
        """
        q = state.variables
        T = _require_traj(q, "AccelerationResidual")
        nv = self.model.nv
        r_mat = r.reshape(T - 2, nv)
        scale = self.weight / (self.dt ** 2)

        g = torch.zeros(T, nv, dtype=q.dtype, device=q.device)
        g[:T - 2] += scale * r_mat
        g[1:T - 1] += -2.0 * scale * r_mat
        g[2:T] += scale * r_mat
        return g.reshape(-1)


@register_residual("jerk")
class JerkResidual:
    """Placeholder — jerk (third-derivative) smoothness on trajectory.

    Not implemented in v1: acceleration regularization is sufficient for
    the human motion / manipulator scenarios in ``docs/08_TASKS.md §3``.
    """

    name: str = "jerk"
    dim: int = 0

    def __init__(self, model: Model, *, dt: float, weight: float = 1.0) -> None:
        self.model = model
        self.dt = dt
        self.weight = weight

    def __call__(self, state: ResidualState) -> torch.Tensor:
        raise NotImplementedError("jerk residual not implemented — use AccelerationResidual instead")

    def jacobian(self, state: ResidualState) -> torch.Tensor | None:
        raise NotImplementedError("jerk residual not implemented")
