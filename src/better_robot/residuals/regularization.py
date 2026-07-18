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

from .._validation import check_tensor
from ..data_model.model import Model
from ._temporal_jacobian import dense_temporal_jacobian, temporal_free_indices
from .base import _configuration
from .structure import TemporalPattern


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

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        q = _configuration(ctx)
        q_rest = self.q_rest
        if self.target_name is not None:
            q_rest = ctx[self.target_name]
            if not isinstance(q_rest, torch.Tensor):
                raise TypeError(f"{self.target_name!r} must be a tensor, got {type(q_rest).__name__}")
        q_rest = q_rest.to(device=q.device, dtype=q.dtype)
        # Broadcast q_rest across any leading batch dims.
        if q.dim() > 1 and q_rest.dim() == 1:
            q_rest = q_rest.expand_as(q)
        return self.model.difference(q_rest, q) * self.weight  # (B..., nv)

    def _analytic_jacobian(
        self,
        ctx: Mapping[str, Any],
    ) -> torch.Tensor:
        """Identity Jacobian (scaled by ``weight``). Shape ``(B..., nv, nv)``."""
        q = _configuration(ctx)
        nv = self.model.nv
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
        full = self._analytic_jacobian(ctx)
        indices = ctx.free_indices("q").to(device=full.device)
        return {"q": full.index_select(-1, indices)}


class JointRotationPrior:
    """Per-joint weighted tangent deviation from a mean configuration.

    ``r = per_joint_weight * model.difference(q_mean, q)`` with
    ``dim = model.nv``. A weight table shaped ``(model.njoints,)`` is expanded
    across each joint's tangent slice; an explicit ``(model.nv,)`` table may
    weight individual tangent coordinates. Fixed joints naturally contribute
    no rows.

    Unlike :class:`RestResidual`, this class does not label the identity as an
    analytic Jacobian. The exact derivative away from the mean includes Lie
    right-Jacobian factors, so callers use tangent AD or the explicit
    finite-difference debug strategy.
    """

    reads = ("q",)

    def __init__(
        self,
        model: Model,
        q_mean: torch.Tensor,
        per_joint_weight: torch.Tensor,
        *,
        name: str = "joint_rotation_prior",
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        q_mean = check_tensor("q_mean", q_mean, floating=True)
        if tuple(q_mean.shape) != (model.nq,):
            raise ValueError(f"q_mean must have shape ({model.nq},), got {tuple(q_mean.shape)}")
        per_joint_weight = check_tensor("per_joint_weight", per_joint_weight, floating=True)
        if tuple(per_joint_weight.shape) == (model.njoints,):
            repeats = torch.tensor(model.nvs, dtype=torch.long, device=per_joint_weight.device)
            tangent_weight = torch.repeat_interleave(per_joint_weight, repeats)
        elif tuple(per_joint_weight.shape) == (model.nv,):
            tangent_weight = per_joint_weight
        else:
            raise ValueError(
                f"per_joint_weight must have shape ({model.njoints},) or ({model.nv},), "
                f"got {tuple(per_joint_weight.shape)}"
            )
        if bool(torch.any(tangent_weight < 0.0)):
            raise ValueError("per_joint_weight must be non-negative")

        self.model = model
        self.name = name
        self.q_mean = q_mean
        self.per_joint_weight = tangent_weight
        self.dim = model.nv

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        q = _configuration(ctx)
        q_mean = self.q_mean.to(device=q.device, dtype=q.dtype)
        weight = self.per_joint_weight.to(device=q.device, dtype=q.dtype)
        return self.model.difference(q_mean, q) * weight


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
    reads = ("q",)

    def __init__(
        self,
        model: Model,
        q_ref: torch.Tensor,
        *,
        weight: float = 1.0,
        weight_per_frame: torch.Tensor | None = None,
        name: str = "reference_trajectory",
    ) -> None:
        if q_ref.dim() != 2:  # bench-ok: constructor shape validation runs once
            raise ValueError(f"q_ref must be (T, nq); got {tuple(q_ref.shape)}")
        if q_ref.shape[0] < 1:
            raise ValueError("ReferenceTrajectoryResidual requires at least one timestep")
        if q_ref.shape[1] != model.nq:
            raise ValueError(f"q_ref trailing size must equal model.nq={model.nq}; got {q_ref.shape[1]}")
        self.model = model
        self.name = name
        self.q_ref = q_ref
        self.weight = float(weight)
        self.weight_per_frame = weight_per_frame  # (T,) or None
        self.horizon = int(q_ref.shape[0])
        self.dim = self.horizon * model.nv

    def _per_frame_scale(self, T: int, device, dtype) -> torch.Tensor:
        if self.weight_per_frame is None:
            return torch.full((T,), self.weight, device=device, dtype=dtype)
        w = self.weight_per_frame.to(device=device, dtype=dtype)
        if w.shape != (T,):
            raise ValueError(f"weight_per_frame must be ({T},); got {tuple(w.shape)}")
        return w * self.weight

    def _trajectory(
        self,
        ctx: Mapping[str, Any],
    ) -> torch.Tensor:
        q = _configuration(ctx)
        if q.dim() < 2:  # bench-ok: trajectory-shape contract validation
            raise ValueError(f"ReferenceTrajectoryResidual expects (B..., T, nq); got {tuple(q.shape)}")
        if q.shape[-2] != self.horizon:
            raise ValueError(f"trajectory length {q.shape[-2]} != q_ref length {self.horizon}")
        return q

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        q = self._trajectory(ctx)
        T = self.horizon
        q_ref = self.q_ref.to(device=q.device, dtype=q.dtype)
        r = self.model.difference(q_ref, q)  # (B..., T, nv)
        w = self._per_frame_scale(T, q.device, q.dtype).unsqueeze(-1)  # (T, 1)
        return (r * w).reshape(*q.shape[:-2], self.dim)

    def temporal_structure(self, variable_name: str) -> TemporalPattern | None:
        if variable_name != "q":
            return None
        return TemporalPattern(
            rows=self.horizon,
            row_width=self.model.nv,
            row_origin=0,
            offsets=(0,),
        )

    def _temporal_blocks(
        self,
        q: torch.Tensor,
        indices: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        nv = self.model.nv
        identity = torch.eye(nv, dtype=q.dtype, device=q.device).index_select(-1, indices)
        weights = self._per_frame_scale(self.horizon, q.device, q.dtype)
        base = (weights[:, None, None] * identity).expand(
            *q.shape[:-2],
            self.horizon,
            nv,
            indices.numel(),
        )
        anchor = q.sum(dim=(-2, -1)) * 0.0
        return {0: base + anchor[..., None, None, None]}

    def temporal_jacobian_blocks(
        self,
        ctx: Mapping[str, Any],
        variable_name: str,
    ) -> Mapping[int, torch.Tensor]:
        if variable_name != "q":
            return {}
        q = self._trajectory(ctx)
        indices = temporal_free_indices(ctx, "q", device=q.device)
        return self._temporal_blocks(q, indices)

    def jacobian_blocks(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        pattern = self.temporal_structure("q")
        assert pattern is not None
        return {
            "q": dense_temporal_jacobian(
                pattern,
                self.temporal_jacobian_blocks(ctx, "q"),
                horizon=self.horizon,
            )
        }


class NullspaceResidual:
    """Nullspace projection of ``(q - q_rest)`` onto the unconstrained subspace.

    ``dim = len(q_rest)``.
    """

    name: str = "nullspace"
    reads = ("q",)

    def __init__(self, q_rest: torch.Tensor, *, weight: float = 1.0) -> None:
        self.q_rest = q_rest
        self.weight = weight
        self.dim = int(q_rest.shape[-1])

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        del ctx
        raise NotImplementedError("see docs/concepts/residuals_and_costs.md §2")
