"""Joint position / velocity / acceleration limit residuals.

Uses a clamped penalty — zero inside limits, positive outside. This is
what replaces the current ``costs/limits.py`` ``torch.clamp(min=0)`` pattern.

See ``docs/design/07_RESIDUALS_COSTS_SOLVERS.md §2``.
"""

from __future__ import annotations

import torch

from ..data_model.model import Model
from .base import Residual, ResidualState
from .registry import register_residual


@register_residual("joint_position_limit")
class JointPositionLimit:
    """One-sided clamped penalty on joint position limits.

    ``r = [clamp(lower - q, min=0); clamp(q - upper, min=0)] * weight``

    ``dim = 2 * nq`` (lower violations concatenated with upper violations).
    The analytic Jacobian is taken in **tangent space** ``nv`` via a
    precomputed ``(nq, nv)`` ``dq/dv`` projection. For single-DOF joints
    ``nq == nv`` and the projection is identity-per-joint. For joints with
    ``nq != nv`` (free-flyer, spherical, planar, unbounded) the projection
    rows are zero — their position limits are always ±inf (or ±1 on unit
    components that never actually violate) so the gradient there is zero
    anyway, and this keeps the tangent-space contract consistent for
    floating-base robots.
    """

    name: str = "joint_position_limit"

    def __init__(self, model: Model, *, weight: float = 1.0) -> None:
        self.model = model
        self.weight = weight
        self.dim = 2 * model.nq

        # Precompute dq/dv projection: (nq, nv) with identity blocks where
        # a joint has nq == nv, zeros elsewhere. Built on CPU once; the
        # jacobian() method moves it to the caller's device/dtype lazily.
        dq_dv = torch.zeros(model.nq, model.nv, dtype=torch.float32)
        for j in range(model.njoints):
            jm = model.joint_models[j]
            if jm.nq == 0 or jm.nq != jm.nv:
                continue
            iq = model.idx_qs[j]
            iv = model.idx_vs[j]
            for k in range(jm.nq):
                dq_dv[iq + k, iv + k] = 1.0
        self._dq_dv = dq_dv  # (nq, nv)

    def __call__(self, state: ResidualState) -> torch.Tensor:
        q = state.variables  # (B..., nq)
        lo = self.model.lower_pos_limit.to(q.device, q.dtype)  # (nq,)
        hi = self.model.upper_pos_limit.to(q.device, q.dtype)  # (nq,)
        lower_viol = torch.clamp(lo - q, min=0.0) * self.weight  # (B..., nq)
        upper_viol = torch.clamp(q - hi, min=0.0) * self.weight  # (B..., nq)
        return torch.cat([lower_viol, upper_viol], dim=-1)       # (B..., 2*nq)

    def jacobian(self, state: ResidualState) -> torch.Tensor | None:
        """Analytic Jacobian in tangent space. Shape ``(B..., 2*nq, nv)``."""
        q = state.variables
        lo = self.model.lower_pos_limit.to(q.device, q.dtype)
        hi = self.model.upper_pos_limit.to(q.device, q.dtype)

        # Per-q indicator of active lower/upper violation, scaled by weight.
        lower_diag = torch.where(
            q < lo,
            torch.full_like(q, -self.weight),
            torch.zeros_like(q),
        )  # (B..., nq)
        upper_diag = torch.where(
            q > hi,
            torch.full_like(q, self.weight),
            torch.zeros_like(q),
        )  # (B..., nq)

        # Project to nv columns via the precomputed (nq, nv) mapping.
        dq_dv = self._dq_dv.to(q.device, q.dtype)            # (nq, nv)
        J_lower = lower_diag.unsqueeze(-1) * dq_dv           # (B..., nq, nv)
        J_upper = upper_diag.unsqueeze(-1) * dq_dv           # (B..., nq, nv)
        return torch.cat([J_lower, J_upper], dim=-2)         # (B..., 2*nq, nv)


@register_residual("joint_velocity_limit")
class JointVelocityLimit:
    """One-sided clamped penalty on joint velocity limits. ``dim = 2 * nv``."""

    name: str = "joint_velocity_limit"

    def __init__(self, model: Model, *, weight: float = 1.0) -> None:
        self.model = model
        self.weight = weight
        self.dim = 2 * model.nv

    def __call__(self, state: ResidualState) -> torch.Tensor:
        # For velocity limits we check state.data.v if available,
        # otherwise fall back to a zero residual.
        v = state.data.v if state.data.v is not None else torch.zeros_like(state.variables)
        lim = self.model.velocity_limit.to(v.device, v.dtype)  # (nv,)
        lower_viol = torch.clamp(-lim - v, min=0.0) * self.weight
        upper_viol = torch.clamp(v - lim, min=0.0) * self.weight
        return torch.cat([lower_viol, upper_viol], dim=-1)

    def jacobian(self, state: ResidualState) -> torch.Tensor | None:
        raise NotImplementedError("see docs/design/07_RESIDUALS_COSTS_SOLVERS.md §2")


@register_residual("joint_accel_limit")
class JointAccelLimit:
    """One-sided clamped penalty on joint acceleration limits. ``dim = 2 * nv``."""

    name: str = "joint_accel_limit"

    def __init__(self, model: Model, *, weight: float = 1.0) -> None:
        self.model = model
        self.weight = weight
        self.dim = 2 * model.nv

    def __call__(self, state: ResidualState) -> torch.Tensor:
        raise NotImplementedError("see docs/design/07_RESIDUALS_COSTS_SOLVERS.md §2")

    def jacobian(self, state: ResidualState) -> torch.Tensor | None:
        raise NotImplementedError("see docs/design/07_RESIDUALS_COSTS_SOLVERS.md §2")
