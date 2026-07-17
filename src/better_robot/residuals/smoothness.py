"""Manifold-aware smoothness residuals on trajectory-shaped variables.

``state.variables`` is expected to be shape ``(T, nq)``; residuals produce
flat vectors and analytic Jacobians of shape ``(dim, T*nv)`` — one dense
block per timestep. Velocity / acceleration are computed in the tangent
space via ``Model.difference`` — so SE(3) floating bases, spherical joints,
and revolute joints all contribute the right number of DOFs without
special-casing.

See ``docs/concepts/residuals_and_costs.md §2``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ..data_model.model import Model
from ._temporal_jacobian import dense_temporal_jacobian, temporal_free_indices
from .base import ResidualState, _residual_model_q
from .structure import TemporalPattern


def _require_traj(q: torch.Tensor, name: str) -> int:
    if q.dim() < 2:  # bench-ok: trajectory-shape contract validation
        raise ValueError(f"{name}: expected trajectory shape (B..., T, nq); got {tuple(q.shape)}")
    T = int(q.shape[-2])
    if T < 3:
        raise ValueError(f"{name}: need at least 3 timesteps, got T={T}")
    return T


def _static_horizon(horizon: int | None, name: str) -> int | None:
    if horizon is None:
        return None
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise TypeError(f"{name}: horizon must be an int or None")
    if horizon < 3:
        raise ValueError(f"{name}: need at least 3 timesteps, got T={horizon}")
    return horizon


def _anchor_constant_block(block: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Keep a mathematically constant analytic block graph-connected."""
    anchor = q.sum(dim=(-2, -1)) * 0.0
    return block + anchor[..., None, None, None]


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
    reads = ("q",)

    def __init__(
        self,
        model: Model,
        *,
        dt: float,
        weight: float = 1.0,
        horizon: int | None = None,
        name: str = "velocity",
    ) -> None:
        self.model = model
        self.name = name
        self.dt = float(dt)
        self.weight = float(weight)
        self.horizon = _static_horizon(horizon, "VelocityResidual")
        # Legacy callers may omit a horizon and retain first-call sizing.
        self.dim = 0 if self.horizon is None else (self.horizon - 2) * model.nv

    def _trajectory(
        self,
        value: ResidualState | Mapping[str, Any],
    ) -> tuple[torch.Tensor, int]:
        _model, q = _residual_model_q(value, model=self.model)
        T = _require_traj(q, "VelocityResidual")
        if self.horizon is not None and T != self.horizon:
            raise ValueError(f"VelocityResidual: trajectory horizon {T} != declared horizon {self.horizon}")
        if not isinstance(value, ResidualState) and self.horizon is None:
            raise ValueError("VelocityResidual requires horizon=... for named-block use")
        self.dim = (T - 2) * self.model.nv
        return q, T

    def __call__(self, value: ResidualState | Mapping[str, Any]) -> torch.Tensor:
        q, T = self._trajectory(value)
        q_prev = q[..., :-2, :]
        q_next = q[..., 2:, :]
        v = self.model.difference(q_prev, q_next) / (2.0 * self.dt)  # (T-2, nv)
        return (v * self.weight).reshape(*q.shape[:-2], self.dim)

    @staticmethod
    def _pattern(T: int, nv: int) -> TemporalPattern:
        return TemporalPattern(
            rows=T - 2,
            row_width=nv,
            row_origin=1,
            offsets=(-1, 1),
        )

    def temporal_structure(self, variable_name: str) -> TemporalPattern | None:
        if variable_name != "q" or self.horizon is None:
            return None
        return self._pattern(self.horizon, self.model.nv)

    def _temporal_blocks(
        self,
        q: torch.Tensor,
        T: int,
        indices: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        nv = self.model.nv
        rows = T - 2
        reduced_identity = torch.eye(nv, dtype=q.dtype, device=q.device).index_select(-1, indices)
        base = reduced_identity.expand(*q.shape[:-2], rows, nv, indices.numel())
        scale = self.weight / (2.0 * self.dt)
        return {
            -1: _anchor_constant_block(-scale * base, q),
            1: _anchor_constant_block(scale * base, q),
        }

    def temporal_jacobian_blocks(
        self,
        ctx: Mapping[str, Any],
        variable_name: str,
    ) -> Mapping[int, torch.Tensor]:
        if variable_name != "q":
            return {}
        q, T = self._trajectory(ctx)
        indices = temporal_free_indices(ctx, "q", device=q.device)
        return self._temporal_blocks(q, T, indices)

    def jacobian_blocks(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        pattern = self.temporal_structure("q")
        if pattern is None:
            raise ValueError("VelocityResidual requires horizon=... for named-block use")
        blocks = self.temporal_jacobian_blocks(ctx, "q")
        return {"q": dense_temporal_jacobian(pattern, blocks, horizon=self.horizon)}

    def jacobian(
        self,
        value: ResidualState | Mapping[str, Any],
    ) -> torch.Tensor | None:
        q, T = self._trajectory(value)
        indices = torch.arange(self.model.nv, device=q.device)
        return dense_temporal_jacobian(
            self._pattern(T, self.model.nv),
            self._temporal_blocks(q, T, indices),
            horizon=T,
        )

    def apply_jac_transpose(self, state: ResidualState, r: torch.Tensor) -> torch.Tensor:
        """``J^T @ r`` without materialising the dense Jacobian — O(T·nv)."""
        q, T = self._trajectory(state)
        nv = self.model.nv
        r_mat = r.reshape(*q.shape[:-2], T - 2, nv)
        scale = self.weight / (2.0 * self.dt)

        g = torch.zeros(*q.shape[:-2], T, nv, dtype=q.dtype, device=q.device)
        g[..., : T - 2, :] += -scale * r_mat
        g[..., 2:T, :] += scale * r_mat
        return g.reshape(*q.shape[:-2], T * nv)


class AccelerationResidual:
    """3-point tangent-space acceleration.

    For ``t ∈ [1, T-1)``::

        a_t = (model.difference(q_t, q_{t+1}) - model.difference(q_{t-1}, q_t)) / dt²

    Output dim: ``nv * (T - 2)``. Analytic Jacobian is tridiagonal over
    timesteps with blocks ``[+I, −2I, +I] / dt²`` (identity-right-Jacobian
    approximation — see ``VelocityResidual`` docstring).
    """

    name: str = "acceleration"
    reads = ("q",)

    def __init__(
        self,
        model: Model,
        *,
        dt: float,
        weight: float = 1.0,
        horizon: int | None = None,
        name: str = "acceleration",
    ) -> None:
        self.model = model
        self.name = name
        self.dt = float(dt)
        self.weight = float(weight)
        self.horizon = _static_horizon(horizon, "AccelerationResidual")
        self.dim = 0 if self.horizon is None else (self.horizon - 2) * model.nv

    def _trajectory(
        self,
        value: ResidualState | Mapping[str, Any],
    ) -> tuple[torch.Tensor, int]:
        _model, q = _residual_model_q(value, model=self.model)
        T = _require_traj(q, "AccelerationResidual")
        if self.horizon is not None and T != self.horizon:
            raise ValueError(f"AccelerationResidual: trajectory horizon {T} != declared horizon {self.horizon}")
        if not isinstance(value, ResidualState) and self.horizon is None:
            raise ValueError("AccelerationResidual requires horizon=... for named-block use")
        self.dim = (T - 2) * self.model.nv
        return q, T

    def __call__(self, value: ResidualState | Mapping[str, Any]) -> torch.Tensor:
        q, T = self._trajectory(value)
        diff_fwd = self.model.difference(q[..., 1:-1, :], q[..., 2:, :])
        diff_back = self.model.difference(q[..., :-2, :], q[..., 1:-1, :])
        a = (diff_fwd - diff_back) / (self.dt**2)  # (T-2, nv)
        return (a * self.weight).reshape(*q.shape[:-2], self.dim)

    @staticmethod
    def _pattern(T: int, nv: int) -> TemporalPattern:
        return TemporalPattern(
            rows=T - 2,
            row_width=nv,
            row_origin=1,
            offsets=(-1, 0, 1),
        )

    def temporal_structure(self, variable_name: str) -> TemporalPattern | None:
        if variable_name != "q" or self.horizon is None:
            return None
        return self._pattern(self.horizon, self.model.nv)

    def _temporal_blocks(
        self,
        q: torch.Tensor,
        T: int,
        indices: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        nv = self.model.nv
        rows = T - 2
        reduced_identity = torch.eye(nv, dtype=q.dtype, device=q.device).index_select(-1, indices)
        base = reduced_identity.expand(*q.shape[:-2], rows, nv, indices.numel())
        scale = self.weight / (self.dt**2)
        return {
            -1: _anchor_constant_block(scale * base, q),
            0: _anchor_constant_block(-2.0 * scale * base, q),
            1: _anchor_constant_block(scale * base, q),
        }

    def temporal_jacobian_blocks(
        self,
        ctx: Mapping[str, Any],
        variable_name: str,
    ) -> Mapping[int, torch.Tensor]:
        if variable_name != "q":
            return {}
        q, T = self._trajectory(ctx)
        indices = temporal_free_indices(ctx, "q", device=q.device)
        return self._temporal_blocks(q, T, indices)

    def jacobian_blocks(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        pattern = self.temporal_structure("q")
        if pattern is None:
            raise ValueError("AccelerationResidual requires horizon=... for named-block use")
        blocks = self.temporal_jacobian_blocks(ctx, "q")
        return {"q": dense_temporal_jacobian(pattern, blocks, horizon=self.horizon)}

    def jacobian(
        self,
        value: ResidualState | Mapping[str, Any],
    ) -> torch.Tensor | None:
        q, T = self._trajectory(value)
        indices = torch.arange(self.model.nv, device=q.device)
        return dense_temporal_jacobian(
            self._pattern(T, self.model.nv),
            self._temporal_blocks(q, T, indices),
            horizon=T,
        )

    def apply_jac_transpose(self, state: ResidualState, r: torch.Tensor) -> torch.Tensor:
        """``J^T @ r`` without materialising the dense Jacobian — O(T·nv).

        The full Jacobian is block-tridiagonal over time with blocks
        ``[+I, −2I, +I] / dt²``; the transpose has the same structure, and
        ``J^T r`` collapses into three aligned accumulations.
        """
        q, T = self._trajectory(state)
        nv = self.model.nv
        r_mat = r.reshape(*q.shape[:-2], T - 2, nv)
        scale = self.weight / (self.dt**2)

        g = torch.zeros(*q.shape[:-2], T, nv, dtype=q.dtype, device=q.device)
        g[..., : T - 2, :] += scale * r_mat
        g[..., 1 : T - 1, :] += -2.0 * scale * r_mat
        g[..., 2:T, :] += scale * r_mat
        return g.reshape(*q.shape[:-2], T * nv)


class JerkResidual:
    """Placeholder — jerk (third-derivative) smoothness on trajectory.

    Not implemented in v1: acceleration regularization is sufficient for
    the human motion / manipulator scenarios in ``docs/concepts/tasks.md §3``.
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
