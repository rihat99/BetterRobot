"""Rest / nullspace regularization residuals.

Keep the configuration near a user-provided reference. Residuals live in
**tangent space** (``nv``) via ``model.difference``, so free-flyer and
spherical joints contribute the right number of DOFs instead of the raw
``nq`` slices.

See ``docs/07_RESIDUALS_COSTS_SOLVERS.md §2``.
"""

from __future__ import annotations

import torch

from ..data_model.model import Model
from .base import Residual, ResidualState
from .registry import register_residual


@register_residual("rest")
class RestResidual:
    """``model.difference(q_rest, q) * weight``. ``dim = model.nv``.

    The residual is the tangent-space displacement from ``q_rest`` to the
    current configuration.  For fixed-base robots (``nq == nv``) this
    reduces to ``(q - q_rest) * weight``; for free-flyer robots it does
    SE3 log on the base slice and scalar difference on the joint slices.
    Analytic Jacobian: ``weight * I`` of shape ``(nv, nv)`` — the exact
    right-Jacobian correction ``Jr_inv`` is dropped as a small-angle
    approximation, consistent with the treatment in
    ``docs/05_KINEMATICS.md §5``.
    """

    name: str = "rest"

    def __init__(
        self,
        model: Model,
        q_rest: torch.Tensor,
        *,
        weight: float = 1.0,
    ) -> None:
        self.model = model
        self.q_rest = q_rest
        self.weight = weight
        self.dim = model.nv

    def __call__(self, state: ResidualState) -> torch.Tensor:
        q = state.variables  # (B..., nq)
        q_rest = self.q_rest.to(device=q.device, dtype=q.dtype)
        # Broadcast q_rest across any leading batch dims.
        if q.dim() > 1 and q_rest.dim() == 1:
            q_rest = q_rest.expand_as(q)
        return state.model.difference(q_rest, q) * self.weight  # (B..., nv)

    def jacobian(self, state: ResidualState) -> torch.Tensor | None:
        """Identity Jacobian (scaled by ``weight``). Shape ``(B..., nv, nv)``."""
        q = state.variables
        nv = state.model.nv
        *batch, _ = q.shape
        I = torch.eye(nv, dtype=q.dtype, device=q.device)
        if batch:
            I = I.expand(*batch, nv, nv)
        return I * self.weight


@register_residual("nullspace")
class NullspaceResidual:
    """Nullspace projection of ``(q - q_rest)`` onto the unconstrained subspace.

    ``dim = len(q_rest)``.
    """

    name: str = "nullspace"

    def __init__(self, q_rest: torch.Tensor, *, weight: float = 1.0) -> None:
        self.q_rest = q_rest
        self.weight = weight
        self.dim = int(q_rest.shape[-1])

    def __call__(self, state: ResidualState) -> torch.Tensor:
        raise NotImplementedError("see docs/07_RESIDUALS_COSTS_SOLVERS.md §2")

    def jacobian(self, state: ResidualState) -> torch.Tensor | None:
        raise NotImplementedError("see docs/07_RESIDUALS_COSTS_SOLVERS.md §2")
