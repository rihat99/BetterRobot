"""``TimeIndexedResidual`` — wrap a point-in-time residual for trajectory state.

Most residuals (``PoseResidual``, ``JointPositionLimit``) operate on a
single configuration. When the optimization variable is a whole trajectory
``(T, nq)``, we need to slice out a single timestep, call the inner
residual, and — for the analytic Jacobian — scatter the resulting block
into the correct columns of a ``(dim_inner, T*nv)`` matrix.

See ``docs/concepts/residuals_costs_and_solvers.md``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import fields
from typing import Any

import torch

from ..data_model.data import Data
from ._temporal_jacobian import dense_temporal_jacobian, temporal_free_indices
from .base import _configuration
from .structure import TemporalPattern


def _slice_data(data: Data, t_idx: int) -> Data:
    """Slice one trajectory knot while preserving populated FK caches."""
    time_axis = len(data.batch_shape) - 1
    if time_axis < 0:
        raise ValueError("TimeIndexedResidual Data must carry a trajectory batch axis")
    values: dict[str, Any] = {}
    for descriptor in fields(Data):
        if descriptor.name == "_kinematics_level":
            continue
        value = getattr(data, descriptor.name)
        values[descriptor.name] = value.select(time_axis, t_idx) if isinstance(value, torch.Tensor) else value
    sliced = Data(**values)
    object.__setattr__(sliced, "_kinematics_level", data._kinematics_level)
    return sliced


class _TimeSliceContext(Mapping[str, Any]):
    """Evaluation-context view with trajectory ``q``/``data`` sliced."""

    def __init__(self, ctx: Mapping[str, Any], t_idx: int) -> None:
        self._ctx = ctx
        self._t_idx = t_idx
        self._data: Data | None = None

    def __getitem__(self, key: str) -> Any:
        if key == "q":
            return self._ctx[key][..., self._t_idx, :]
        if key == "data":
            if self._data is None:
                data = self._ctx[key]
                if not isinstance(data, Data):
                    raise TypeError(f"data must be Data, got {type(data).__name__}")
                self._data = _slice_data(data, self._t_idx)
            return self._data
        return self._ctx[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._ctx)

    def __len__(self) -> int:
        return len(self._ctx)

    def free_indices(self, variable_name: str) -> torch.Tensor:
        q = self._ctx[variable_name]
        return temporal_free_indices(self._ctx, variable_name, device=q.device)


class TimeIndexedResidual:
    """Evaluate an inner residual at a single timestep of a trajectory state.

    Parameters
    ----------
    inner : Residual
        Single-configuration residual (e.g. ``PoseResidual``,
        ``JointPositionLimit``).
    t_idx : int
        Timestep to evaluate at.
    name : str | None
        Override the default name ``f"{inner.name}_t{t_idx}"``.
    """

    def __init__(
        self,
        inner,
        t_idx: int,
        *,
        horizon: int | None = None,
        name: str | None = None,
    ) -> None:
        self.inner = inner
        self.t_idx = int(t_idx)
        self.name = name if name is not None else f"{inner.name}_t{t_idx}"
        self.dim = int(inner.dim)
        inner_reads = getattr(inner, "reads", None)
        self.reads = inner_reads if isinstance(inner_reads, tuple) else ("q", "data")
        if horizon is not None:
            if isinstance(horizon, bool) or not isinstance(horizon, int):
                raise TypeError("TimeIndexedResidual horizon must be an int or None")
            if horizon < 1:
                raise ValueError("TimeIndexedResidual horizon must be positive")
            if not 0 <= self.t_idx < horizon:
                raise ValueError(f"TimeIndexedResidual t_idx={self.t_idx} out of range for T={horizon}")
        self.horizon = horizon

    def _trajectory_q(
        self,
        ctx: Mapping[str, Any],
    ) -> tuple[torch.Tensor, int]:
        q = _configuration(ctx)
        if self.horizon is None:
            raise ValueError("TimeIndexedResidual requires horizon=... for problem use")
        if q.ndim < 2:  # bench-ok: trajectory-shape contract validation
            raise ValueError(f"TimeIndexedResidual expects (B..., T, nq); got {tuple(q.shape)}")
        T = int(q.shape[-2])
        if T != self.horizon:
            raise ValueError(f"trajectory horizon {T} != declared horizon {self.horizon}")
        return q, T

    def _slice_input(
        self,
        ctx: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._trajectory_q(ctx)
        return _TimeSliceContext(ctx, self.t_idx)

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        r = self.inner(self._slice_input(ctx))
        if not isinstance(r, torch.Tensor) or r.ndim < 1 or r.shape[-1] != self.dim:
            actual = tuple(r.shape) if isinstance(r, torch.Tensor) else type(r).__name__
            raise ValueError(f"inner residual must return (..., {self.dim}), got {actual}")
        return r

    def temporal_structure(self, variable_name: str) -> TemporalPattern | None:
        if variable_name != "q" or self.horizon is None:
            return None
        return TemporalPattern(
            rows=1,
            row_width=self.dim,
            row_origin=self.t_idx,
            offsets=(0,),
        )

    def _inner_reduced_block(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        sliced = _TimeSliceContext(ctx, self.t_idx)
        analytic = getattr(self.inner, "jacobian_blocks", None)
        if not callable(analytic):
            raise ValueError("TimeIndexedResidual inner residual has no analytic q block")
        blocks = analytic(sliced)
        if set(blocks) != {"q"}:
            raise ValueError("TimeIndexedResidual inner analytic blocks must contain exactly 'q'")
        block = blocks["q"]
        q = ctx["q"]
        return block + (q.sum(dim=(-2, -1)) * 0.0)[..., None, None]

    def temporal_jacobian_blocks(
        self,
        ctx: Mapping[str, Any],
        variable_name: str,
    ) -> Mapping[int, torch.Tensor]:
        if variable_name != "q":
            return {}
        self._trajectory_q(ctx)
        return {0: self._inner_reduced_block(ctx).unsqueeze(-3)}

    def jacobian_blocks(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        if not callable(getattr(self.inner, "jacobian_blocks", None)):
            # Returning no blocks lets Problem's auto/jacrev/jacfwd dispatch
            # differentiate the wrapper itself.
            return {}
        pattern = self.temporal_structure("q")
        if pattern is None:
            raise ValueError("TimeIndexedResidual requires horizon=... for problem use")
        return {
            "q": dense_temporal_jacobian(
                pattern,
                self.temporal_jacobian_blocks(ctx, "q"),
                horizon=self.horizon,
            )
        }
