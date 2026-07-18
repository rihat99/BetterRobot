"""Joint position / velocity / acceleration limit residuals.

Uses a clamped penalty — zero inside limits, positive outside. This is
what replaces the current ``costs/limits.py`` ``torch.clamp(min=0)`` pattern.

See ``docs/concepts/residuals_costs_and_solvers.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ..data_model.model import Model
from .base import _configuration


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
    reads = ("q",)

    def __init__(
        self,
        model: Model,
        *,
        weight: float = 1.0,
        name: str = "joint_position_limit",
    ) -> None:
        self.model = model
        self.name = name
        self.weight = weight
        self.dim = 2 * model.nq

        # Precompute dq/dv projection: (nq, nv) with identity blocks where
        # a joint has nq == nv, zeros elsewhere. Built on CPU once; the
        # analytic block helper moves it to the caller's device/dtype lazily.
        dq_dv = torch.zeros(model.nq, model.nv, dtype=torch.float32)
        for j in range(model.njoints):
            nq_j = model.nqs[j]
            nv_j = model.nvs[j]
            if nq_j == 0 or nq_j != nv_j:
                continue
            iq = model.idx_qs[j]
            iv = model.idx_vs[j]
            for k in range(nq_j):
                dq_dv[iq + k, iv + k] = 1.0
        self._dq_dv = dq_dv  # (nq, nv)

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        q = _configuration(ctx)
        lo = self.model.lower_pos_limit.to(q.device, q.dtype)  # (nq,)
        hi = self.model.upper_pos_limit.to(q.device, q.dtype)  # (nq,)
        lower_viol = torch.clamp(lo - q, min=0.0) * self.weight  # (B..., nq)
        upper_viol = torch.clamp(q - hi, min=0.0) * self.weight  # (B..., nq)
        return torch.cat([lower_viol, upper_viol], dim=-1)  # (B..., 2*nq)

    def _analytic_jacobian(
        self,
        ctx: Mapping[str, Any],
    ) -> torch.Tensor:
        """Analytic Jacobian in tangent space. Shape ``(B..., 2*nq, nv)``."""
        q = _configuration(ctx)
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
        dq_dv = self._dq_dv.to(q.device, q.dtype)  # (nq, nv)
        J_lower = lower_diag.unsqueeze(-1) * dq_dv  # (B..., nq, nv)
        J_upper = upper_diag.unsqueeze(-1) * dq_dv  # (B..., nq, nv)
        return torch.cat([J_lower, J_upper], dim=-2)  # (B..., 2*nq, nv)

    def jacobian_blocks(
        self,
        ctx: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Return the mask-reduced analytic ``q`` block."""
        full = self._analytic_jacobian(ctx)
        indices = ctx.free_indices("q").to(device=full.device)
        return {"q": full.index_select(-1, indices)}


class JointVelocityLimit:
    """One-sided clamped penalty on joint velocity limits. ``dim = 2 * nv``."""

    name: str = "joint_velocity_limit"
    reads = ("q", "data")

    def __init__(self, model: Model, *, weight: float = 1.0) -> None:
        self.model = model
        self.weight = weight
        self.dim = 2 * model.nv

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        q = _configuration(ctx)
        data = ctx["data"]
        # For velocity limits we check data.v if available,
        # otherwise fall back to a zero residual.
        v = data.v if data.v is not None else torch.zeros_like(q)
        lim = self.model.velocity_limit.to(v.device, v.dtype)  # (nv,)
        lower_viol = torch.clamp(-lim - v, min=0.0) * self.weight
        upper_viol = torch.clamp(v - lim, min=0.0) * self.weight
        return torch.cat([lower_viol, upper_viol], dim=-1)
