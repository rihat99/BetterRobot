"""Rest / nullspace regularization residuals.

Keep the configuration near a user-provided reference. Residuals live in
**tangent space** (``nv``) via ``model.difference``, so free-flyer and
spherical joints contribute the right number of DOFs instead of the raw
``nq`` slices.

See ``docs/concepts/residuals_and_costs.md §2``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ..data_model.model import Model
from .base import ResidualState, _residual_model_q


class RestResidual:
    """``model.difference(q_rest, q) * weight``. ``dim = model.nv``.

    The residual is the tangent-space displacement from ``q_rest`` to the
    current configuration.  For fixed-base robots (``nq == nv``) this
    reduces to ``(q - q_rest) * weight``; for free-flyer robots it does
    SE3 log on the base slice and scalar difference on the joint slices.
    Analytic Jacobian: ``weight * I`` of shape ``(nv, nv)`` — the exact
    right-Jacobian correction ``Jr_inv`` is dropped as a small-angle
    approximation, consistent with the treatment in
    ``docs/concepts/kinematics.md §5``.
    """

    name: str = "rest"
    reads = ("q",)

    def __init__(
        self,
        model: Model,
        q_rest: torch.Tensor,
        *,
        weight: float = 1.0,
        name: str = "rest",
        target_name: str | None = None,
    ) -> None:
        if target_name is not None and (not isinstance(target_name, str) or not target_name):
            raise TypeError("target_name must be a non-empty string or None")
        self.model = model
        self.name = name
        self.q_rest = q_rest
        self.target_name = target_name
        self.reads = ("q", target_name) if target_name is not None else ("q",)
        self.weight = weight
        self.dim = model.nv

    def __call__(self, value: ResidualState | Mapping[str, Any]) -> torch.Tensor:
        model, q = _residual_model_q(value, model=self.model)
        q_rest = self.q_rest
        if self.target_name is not None and not isinstance(value, ResidualState):
            q_rest = value[self.target_name]
            if not isinstance(q_rest, torch.Tensor):
                raise TypeError(f"named-block context entry {self.target_name!r} must be a tensor")
        q_rest = q_rest.to(device=q.device, dtype=q.dtype)
        # Broadcast q_rest across any leading batch dims.
        if q.dim() > 1 and q_rest.dim() == 1:
            q_rest = q_rest.expand_as(q)
        return model.difference(q_rest, q) * self.weight  # (B..., nv)

    def jacobian(
        self,
        value: ResidualState | Mapping[str, Any],
    ) -> torch.Tensor | None:
        """Identity Jacobian (scaled by ``weight``). Shape ``(B..., nv, nv)``."""
        model, q = _residual_model_q(value, model=self.model)
        nv = model.nv
        *batch, _ = q.shape
        identity = torch.eye(nv, dtype=q.dtype, device=q.device)
        if batch:
            identity = identity.expand(*batch, nv, nv)
        return identity * self.weight

    def jacobian_blocks(
        self,
        ctx: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Return the mask-reduced analytic ``q`` block."""
        full = self.jacobian(ctx)
        assert full is not None
        indices = ctx.free_indices("q").to(device=full.device)
        return {"q": full.index_select(-1, indices)}


class ReferenceTrajectoryResidual:
    """Penalize tangent-space deviation of a trajectory from a reference.

    For a trajectory ``q: (T, nq)`` and reference ``q_ref: (T, nq)``::

        r = weight * model.difference(q_ref, q)    # (T, nv), flattened

    Analytic Jacobian is the scaled identity ``weight * I_{T*nv}`` under
    the same small-step approximation used by ``RestResidual`` —
    ``Jr_inv ≈ I`` since motion optim operates near the reference.

    Use ``weight_per_frame`` (shape ``(T,)``) to soften or sharpen the
    reference term at specific frames — e.g. to lock in known start/end
    poses or relax during contact transitions.
    """

    name: str = "reference_trajectory"

    def __init__(
        self,
        model: Model,
        q_ref: torch.Tensor,
        *,
        weight: float = 1.0,
        weight_per_frame: torch.Tensor | None = None,
    ) -> None:
        if q_ref.dim() != 2:  # bench-ok: constructor shape validation runs once
            raise ValueError(f"q_ref must be (T, nq); got {tuple(q_ref.shape)}")
        self.model = model
        self.q_ref = q_ref
        self.weight = float(weight)
        self.weight_per_frame = weight_per_frame  # (T,) or None
        self.dim = int(q_ref.shape[0] * model.nv)

    def _per_frame_scale(self, T: int, device, dtype) -> torch.Tensor:
        if self.weight_per_frame is None:
            return torch.full((T,), self.weight, device=device, dtype=dtype)
        w = self.weight_per_frame.to(device=device, dtype=dtype)
        if w.shape != (T,):
            raise ValueError(f"weight_per_frame must be ({T},); got {tuple(w.shape)}")
        return w * self.weight

    def __call__(self, state: ResidualState) -> torch.Tensor:
        q = state.variables
        if q.dim() != 2:  # bench-ok: trajectory-shape contract validation
            raise ValueError(f"ReferenceTrajectoryResidual expects (T, nq); got {tuple(q.shape)}")
        T, nq = q.shape
        if T != self.q_ref.shape[0]:
            raise ValueError(f"trajectory length {T} != q_ref length {self.q_ref.shape[0]}")
        q_ref = self.q_ref.to(device=q.device, dtype=q.dtype)
        r = state.model.difference(q_ref, q)  # (T, nv)
        w = self._per_frame_scale(T, q.device, q.dtype).unsqueeze(-1)  # (T, 1)
        return (r * w).reshape(-1)

    def jacobian(self, state: ResidualState) -> torch.Tensor | None:
        q = state.variables
        T, _ = q.shape
        nv = state.model.nv
        device, dtype = q.device, q.dtype

        w = self._per_frame_scale(T, device, dtype)  # (T,)
        # Block-diagonal of scaled identities; build explicitly to stay clear
        # (trajectory sizes are small relative to the fullJacobian of the CostStack).
        J = torch.zeros(T * nv, T * nv, device=device, dtype=dtype)
        eye = torch.eye(nv, device=device, dtype=dtype)
        for t in range(T):
            J[t * nv : (t + 1) * nv, t * nv : (t + 1) * nv] = eye * w[t]
        return J

    def apply_jac_transpose(self, state: ResidualState, r: torch.Tensor) -> torch.Tensor:
        """``J^T @ r`` without materialising the dense Jacobian — O(T·nv).

        Jacobian is diagonal (scaled identity per timestep), so the
        transpose-product is just per-frame scaling.
        """
        q = state.variables
        T, _ = q.shape
        nv = state.model.nv
        w = self._per_frame_scale(T, q.device, q.dtype)  # (T,)
        r_mat = r.reshape(T, nv)
        return (w.unsqueeze(-1) * r_mat).reshape(-1)


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
        raise NotImplementedError("see docs/concepts/residuals_and_costs.md §2")

    def jacobian(self, state: ResidualState) -> torch.Tensor | None:
        raise NotImplementedError("see docs/concepts/residuals_and_costs.md §2")
