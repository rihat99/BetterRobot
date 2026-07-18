"""Contact-consistency residual: penalise cartesian motion of contact frames.

Given a contact mask ``c ∈ (T, K)`` over ``K`` target frames, the residual
penalises the per-frame linear velocity of each contact frame — measured
as the world-frame displacement of the frame origin between consecutive
timesteps. This is exactly the linear part of the frame's
``LOCAL_WORLD_ALIGNED`` spatial velocity (see
``docs/concepts/kinematics_and_jacobians.md``):

    r_{t,k,:3} = c_{t,k} * (p_k(q_{t+1}) - p_k(q_t)) / dt

where ``p_k(q) = data.frame_pose_world[t, frame_ids[k], :3]``.

Output dim: ``3 * K * (T - 1)`` — linear only in v1. The ``angular`` flag
is reserved as an expansion hook.

See ``docs/concepts/residuals_costs_and_solvers.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ..data_model.data import Data
from ..data_model.model import Model
from ._temporal_jacobian import dense_temporal_jacobian, temporal_free_indices
from .base import _configuration
from .structure import TemporalPattern


class ContactConsistencyResidual:
    """Linear cartesian velocity penalty on tagged contact frames."""

    name: str = "contact_consistency"
    reads = ("q", "data")

    def __init__(
        self,
        model: Model,
        frame_ids: tuple[int, ...],
        contact_weights: torch.Tensor,
        *,
        dt: float,
        weight: float = 1.0,
        angular: bool = False,
        name: str = "contact_consistency",
    ) -> None:
        if angular:
            raise NotImplementedError("angular contact-consistency is an expansion hook; not implemented in v1")
        if contact_weights.dim() != 2 or contact_weights.shape[1] != len(frame_ids):
            raise ValueError(f"contact_weights must be (T, {len(frame_ids)}); got {tuple(contact_weights.shape)}")
        if not frame_ids:
            raise ValueError("ContactConsistencyResidual requires at least one contact frame")
        if contact_weights.shape[0] < 2:
            raise ValueError("ContactConsistencyResidual requires at least two timesteps")
        self.model = model
        self.name = name
        self.frame_ids = tuple(int(i) for i in frame_ids)
        self.contact_weights = contact_weights
        self.dt = float(dt)
        self.weight = float(weight)
        T = int(contact_weights.shape[0])
        self.horizon = T
        K = len(self.frame_ids)
        self.dim = 3 * K * (T - 1)

    def _context(
        self,
        ctx: Mapping[str, Any],
    ) -> tuple[torch.Tensor, Data]:
        q = _configuration(ctx)
        data = ctx["data"]
        if not isinstance(data, Data):
            raise TypeError(f"data must be Data, got {type(data).__name__}")
        if q.ndim < 2:  # bench-ok: trajectory-shape contract validation
            raise ValueError(f"ContactConsistencyResidual expects (B..., T, nq); got {tuple(q.shape)}")
        if q.shape[-2] != self.horizon:
            raise ValueError(f"trajectory length {q.shape[-2]} != contact_weights length {self.horizon}")
        return q, data

    def _frame_positions(self, data: Data) -> torch.Tensor:
        """Return ``(B..., T, K, 3)`` frame-origin positions."""
        if data.frame_pose_world is None:
            raise RuntimeError(
                "ContactConsistencyResidual: data.frame_pose_world is None; "
                "RobotStateProvider must compute frame placements"
            )
        frame_idx = torch.as_tensor(self.frame_ids, device=data.frame_pose_world.device)
        p = data.frame_pose_world[..., :3]
        return p.index_select(-2, frame_idx)

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        q, data = self._context(ctx)
        p = self._frame_positions(data)  # (T, K, 3)
        dp = (p[..., 1:, :, :] - p[..., :-1, :, :]) / self.dt

        w = self.contact_weights.to(device=q.device, dtype=q.dtype)
        # Use the average of the endpoint masks — so transitions do not
        # weight the displacement asymmetrically.
        w_pair = 0.5 * (w[:-1] + w[1:]).unsqueeze(-1)  # (T-1, K, 1)
        r = self.weight * w_pair * dp  # (T-1, K, 3)
        return r.reshape(*q.shape[:-2], self.dim)

    def temporal_structure(self, variable_name: str) -> TemporalPattern | None:
        if variable_name != "q":
            return None
        return TemporalPattern(
            rows=self.horizon - 1,
            row_width=3 * len(self.frame_ids),
            row_origin=0,
            offsets=(0, 1),
        )

    def _frame_jacobians(self, data: Data) -> torch.Tensor:
        """Return LWA linear frame Jacobians ``(B..., T, K, 3, nv)``."""
        from ..kinematics.jacobian import get_frame_jacobian  # noqa: PLC0415

        return torch.stack(
            [
                get_frame_jacobian(
                    self.model,
                    data,
                    frame_id,
                )[..., :3, :]
                for frame_id in self.frame_ids
            ],
            dim=-3,
        )

    def _temporal_blocks(
        self,
        q: torch.Tensor,
        data: Data,
        indices: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        K = len(self.frame_ids)
        frame_jacobians = self._frame_jacobians(data).index_select(-1, indices)
        weights = self.contact_weights.to(device=q.device, dtype=q.dtype)
        pair_weights = 0.5 * (weights[:-1] + weights[1:])
        scale_shape = (*((1,) * len(q.shape[:-2])), self.horizon - 1, K, 1, 1)
        scales = (self.weight * pair_weights / self.dt).reshape(scale_shape)
        left = -scales * frame_jacobians[..., :-1, :, :, :]
        right = scales * frame_jacobians[..., 1:, :, :, :]
        output_shape = (
            *q.shape[:-2],
            self.horizon - 1,
            3 * K,
            indices.numel(),
        )
        return {0: left.reshape(output_shape), 1: right.reshape(output_shape)}

    def temporal_jacobian_blocks(
        self,
        ctx: Mapping[str, Any],
        variable_name: str,
    ) -> Mapping[int, torch.Tensor]:
        if variable_name != "q":
            return {}
        q, data = self._context(ctx)
        indices = temporal_free_indices(ctx, "q", device=q.device)
        return self._temporal_blocks(q, data, indices)

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
